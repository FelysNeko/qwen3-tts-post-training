"""GRPO training loop (Phase 4): one group per optimizer step, end to end.

Pipeline per step (B=1, update after each group):
    prompts → rollout (sample → decode → wav) → scorer → reward_v3 →
    needs_resample? (skip) → compute_ref/compute_policy → grpo_loss →
    backward → grad clip → optimizer step → monitor line → ckpt.

The reference policy is the same weights with LoRA adapters disabled
(TrainerModel.set_adapter), so only one model lives in VRAM. Ckpts carry the
LoRA deltas + semantic head + optimizer state so `--resume` continues cleanly.
"""

from __future__ import annotations

import json
import random
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from qwen3_tts_post_training.reward.reward import RewardConfig, reward_v3
from qwen3_tts_post_training.scorers.client import ScoreItem, ScorerClient
from qwen3_tts_post_training.scorers.protocol import ScorerError
from qwen3_tts_post_training.train.grpo import GRPOConfig, grpo_loss, needs_resample
from trainer.decoder import Decoder
from trainer.logprob import LogProbComputer
from trainer.model import TrainerModel
from trainer.rollout import rollout_group
from trainer.sampler import Sampler

# v1 placeholder text pool (domain: narrative / dialogue / rhythmic narration).
TEXT_POOL = (
    "风从山谷那边吹来，把整片麦田都推成了金色的波浪。",
    "他说：“我们回家吧。”声音很轻，却像是商量了一辈子。",
    "雨落在屋顶上，嗒嗒嗒，像一只不肯睡的小猫在敲门。",
    "她翻开那本旧书，第一页上写着：给你的，永远晚一步。",
    "深夜的便利店里，热水壶咕噜噜地响，他望着窗外发呆。",
    "“别担心，”母亲说，“路再长，也有走到头的一天。”",
    "钟声敲了十二下，院子里那棵老槐树的影子慢慢转了个方向。",
    "我们沿着河堤走了很远，直到路灯一盏盏亮起来。",
)


@dataclass
class TrainConfig:
    model_path: str = "/mnt/d/Repository/models/PhiLia093-TTS/"
    device: str = "cuda:1"
    dtype: str = "bfloat16"
    lora_r: int = 16
    lora_alpha: float = 64
    speaker: str = "cyrene"

    text_pool: tuple[str, ...] = TEXT_POOL
    text_pool_path: str | None = None  # if set, overrides text_pool (one line each)
    num_prompts: int = 8  # distinct prompts per step (Fish Audio S2 layout)
    group_size: int = 8  # rollouts per prompt (the GRPO group)
    num_steps: int = 1
    seed: int = 0
    max_new_tokens: int = 4096
    runaway_t_max: int = 400  # per-group no-EOS guard (also OOM guard)

    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float | None = None
    sampler_impl: str = "hf"  # hf | fast | compiled (PROJECT_STATUS §9)

    variant: str = "dr"
    kl_beta: float = 0.01
    lr: float = 1e-5
    warmup_steps: int = 10  # linear LR ramp: tames the Adam first-step sign
    # jolt (all params move ±lr at once; observed as KL 586 on step 1, fish3)
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    scorer_device: str = "cuda:0"
    out_dir: str = "runs/grpo_v1"
    ckpt_every: int = 1
    resume: bool = False
    monitor: bool = True


def _load_text_pool(cfg: TrainConfig) -> list[str]:
    """Text pool from file (one prompt per line) or the built-in placeholder."""
    if cfg.text_pool_path is None:
        return list(cfg.text_pool)
    return [line.strip() for line in Path(cfg.text_pool_path).read_text().splitlines()]


def _current_rss_mb() -> int:
    """Current resident set size of this process (MB), from /proc."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * 4096 // 2**20
    except OSError:
        return -1


def _pick_prompts(pool: list[str], cfg: TrainConfig, step: int) -> list[str]:
    """Fish-Audio-style batch: `num_prompts` DISTINCT prompts per step, each
    rolled out `group_size` times (8×8 = 64 rollouts per update). Distinct
    prompts average out per-group reward noise; within-group variance stays
    pure sampling noise of the same text (the GRPO baseline)."""
    rng = random.Random(cfg.seed * 1000003 + step)
    k = min(cfg.num_prompts, len(pool))
    return rng.sample(pool, k)


def _scores_to_tensor(results: list[dict], key: str, device: str) -> torch.Tensor:
    """Extract one score column; `error`/None anywhere → raises so the step skips."""
    values = [r[key] for r in results]
    if any(v is None for v in values):
        raise RuntimeError(f"scorer column {key!r} is None (scorer failure?)")
    return torch.tensor(values, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# ckpt (LoRA deltas + semantic head + optimizer) with resume support
# ---------------------------------------------------------------------------


def _ckpt_path(out_dir: Path, step: int) -> Path:
    return out_dir / f"step_{step:05d}.pt"


def _save_ckpt(cfg: TrainConfig, ttm: TrainerModel, optimizer, step: int) -> None:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "lora": [
            {"a": m.lora_a.detach().cpu(), "b": m.lora_b.detach().cpu()}
            for m in ttm.lora_modules
        ],
        "codec_head": ttm.codec_head.weight.detach().cpu(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, _ckpt_path(out, step))
    (out / "latest").write_text(str(step))


def _load_ckpt(cfg: TrainConfig, ttm: TrainerModel, optimizer) -> int:
    """Restore last ckpt in out_dir. Returns the next step to run."""
    out = Path(cfg.out_dir)
    marker = out / "latest"
    if not marker.exists():
        return 0
    step = int(marker.read_text().strip())
    state = torch.load(
        _ckpt_path(out, step), map_location=cfg.device, weights_only=False
    )
    for m, sd in zip(ttm.lora_modules, state["lora"]):
        m.lora_a.data.copy_(sd["a"])
        m.lora_b.data.copy_(sd["b"])
    ttm.codec_head.weight.data.copy_(state["codec_head"])
    optimizer.load_state_dict(state["optimizer"])
    print(f"[resume] restored step {step} from {_ckpt_path(out, step)}")
    return step + 1


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------


def run_grpo(cfg: TrainConfig | None = None) -> None:
    cfg = cfg or TrainConfig()
    dtype = getattr(torch, cfg.dtype)

    if not Path(cfg.model_path).exists():
        raise FileNotFoundError(
            f"TTS model ckpt not found at {cfg.model_path!r}. "
            "Pass --model-path (or set TrainConfig.model_path) to your "
            "PhiLia093-TTS ckpt once it is downloaded."
        )

    ttm = TrainerModel(
        cfg.model_path,
        device=cfg.device,
        dtype=dtype,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
    )
    sampler = Sampler(ttm, speaker=cfg.speaker, impl=cfg.sampler_impl)
    decoder = Decoder(ttm)
    lpc = LogProbComputer(ttm, speaker=cfg.speaker)
    scorer = ScorerClient(device=cfg.scorer_device)
    scorer.start()
    try:
        scorer.ping()
        _train_loop(cfg, ttm, sampler, decoder, lpc, scorer)
    finally:
        scorer.stop()


def _train_loop(
    cfg: TrainConfig,
    ttm: TrainerModel,
    sampler: Sampler,
    decoder: Decoder,
    lpc: LogProbComputer,
    scorer: ScorerClient,
) -> None:
    optimizer = torch.optim.AdamW(
        ttm.trainable_parameters, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    start_step = _load_ckpt(cfg, ttm, optimizer) if cfg.resume else 0

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    monitor_f = (out / "monitor.jsonl").open("a") if cfg.monitor else None

    algo = GRPOConfig(variant=cfg.variant, kl_beta=cfg.kl_beta)
    reward_cfg = RewardConfig()
    pool = _load_text_pool(cfg)
    group_ids = torch.zeros(cfg.group_size, dtype=torch.long, device=cfg.device)

    for step in range(start_step, start_step + cfg.num_steps):
        lr_t = cfg.lr * min(1.0, (step + 1) / max(1, cfg.warmup_steps))
        for pg in optimizer.param_groups:
            pg["lr"] = lr_t
        t0 = time.monotonic()
        prompts = _pick_prompts(pool, cfg, step)
        gs = cfg.group_size
        skips: dict[str, int] = {}

        # ---- phase 1: rollout — one group per prompt (graphed batch = gs) ----
        t_roll0 = time.monotonic()
        groups: list[dict] = []
        for gi, prompt in enumerate(prompts):
            rollout = rollout_group(
                sampler,
                decoder,
                [prompt] * gs,
                seed=cfg.seed * 1000003 + step * 1009 + gi,
                tag=f"step{step}g{gi}",
            )
            t_max = max(c.shape[0] for c in rollout.codes)
            if t_max > cfg.runaway_t_max:
                # runaway no-EOS group (degenerate policy state): scoring it
                # wastes time and teacher-forcing [gs, >400] with grads OOMs
                # the 16 GB trainer GPU
                skips["runaway"] = skips.get("runaway", 0) + 1
                continue
            groups.append(
                {
                    "gi": gi,
                    "prompt": prompt,
                    "codes": rollout.codes,
                    "wavs": rollout.wav_paths,
                    "t_max": t_max,
                }
            )
        t_rollout = time.monotonic() - t_roll0

        # ---- phase 2: score — ONE request for all surviving groups ----
        t_score = 0.0
        if groups:
            t_s0 = time.monotonic()
            try:
                items = [
                    ScoreItem(wav=str(g["wavs"][j]), text=g["prompt"])
                    for g in groups
                    for j in range(gs)
                ]
                results = scorer.score(items)
            except (RuntimeError, ScorerError) as e:
                print(
                    json.dumps(
                        {"step": step, "skip": "scorer_failure", "reason": str(e)},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            t_score = time.monotonic() - t_s0

            scored: list[dict] = []
            pos = 0
            for g in groups:
                rs = results[pos : pos + gs]
                pos += gs
                if any(r.get("error") is not None or r.get("cer") is None for r in rs):
                    skips["scorer_item"] = skips.get("scorer_item", 0) + 1
                    continue
                for key in ("sim", "cer", "mos"):
                    g[key] = torch.tensor(
                        [r[key] for r in rs], dtype=torch.float32, device=cfg.device
                    )
                scored.append(g)
            groups = scored

        # per-group zero-signal filter (DAPO dynamic sampling): only groups
        # whose WER actually spreads carry a learnable within-group signal
        trainable: list[dict] = []
        for g in groups:
            if needs_resample(g["sim"], g["cer"]):
                skips["flat_group"] = skips.get("flat_group", 0) + 1
                continue
            g["R"], g["bd"] = reward_v3(g["sim"], g["cer"], g["mos"], reward_cfg)
            trainable.append(g)

        if not trainable:
            print(
                json.dumps(
                    {"step": step, "skip": "no_trainable_group", "skips": skips}
                ),
                flush=True,
            )
            continue

        # ---- phase 3: train — gradient accumulation, one group per pass ----
        # (peak activation memory stays at single-group level; each group
        # contributes 1/len(trainable) to the update, equal group weighting)
        t_train0 = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        trained: list[tuple[torch.Tensor, object, dict]] = []
        for g in trainable:
            ref = lpc.compute_ref(
                [g["prompt"]] * gs, g["codes"], cfg.temperature, cfg.top_k
            )
            pol = lpc.compute_policy(
                [g["prompt"]] * gs, g["codes"], cfg.temperature, cfg.top_k
            )
            loss, metrics = grpo_loss(
                pol.log_probs, ref.log_probs, g["R"], pol.mask, group_ids, algo
            )
            if not torch.isfinite(loss):
                # non-finite loss (e.g. a sampled token falling out of the
                # teacher-forcing top-k) backprops NaN grads — drop the group
                skips["nonfinite_loss"] = skips.get("nonfinite_loss", 0) + 1
                continue
            (loss / len(trainable)).backward()
            trained.append((loss, metrics, g))
        t_train = time.monotonic() - t_train0

        if not trained:
            print(
                json.dumps(
                    {"step": step, "skip": "all_losses_nonfinite", "skips": skips}
                ),
                flush=True,
            )
            continue

        # ---- phase 4: update ----
        t_opt0 = time.monotonic()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            ttm.trainable_parameters, cfg.grad_clip
        )
        optimizer.step()
        t_opt = time.monotonic() - t_opt0

        losses = torch.stack([l for l, _, _ in trained])
        pol_losses = torch.stack([m.policy_loss for _, m, _ in trained])
        kls = [m.kl for _, m, _ in trained if m.kl is not None]
        advs = torch.cat([m.advantage for _, m, _ in trained])
        R_all = torch.cat([g["R"] for _, _, g in trained])
        bds = [g["bd"] for _, _, g in trained]

        def _col(key: str, bds=bds) -> float:
            return torch.cat([getattr(b, key) for b in bds]).float().mean().item()

        def _col_g(key: str, trained=trained) -> float:
            return torch.cat([g[key] for _, _, g in trained]).mean().item()

        monitor = {
            "step": step,
            "groups_trained": len(trained),
            "groups_skipped": sum(skips.values()),
            "skips": skips,
            "loss": round(losses.mean().item(), 4),
            "policy_loss": round(pol_losses.mean().item(), 4),
            "kl": round(torch.stack(kls).mean().item(), 5) if kls else None,
            "grad_norm": round(grad_norm.item(), 4),
            "lr": f"{lr_t:.2e}",
            "mean_R": round(R_all.mean().item(), 4),
            "t_max": max(g["t_max"] for _, _, g in trained),
            "adv_std": round(advs.std(unbiased=False).item(), 4),
            "r_sv_mean": round(_col("r_sv"), 4),
            "r_wer_mean": round(_col("r_wer"), 4),
            "r_mos_mean": round(_col("r_mos"), 4),
            "std_sv": round(torch.cat([b.std_sv for b in bds]).mean().item(), 5),
            "std_wer": round(torch.cat([b.std_wer for b in bds]).mean().item(), 5),
            "mos_dead_frac": round(
                torch.stack(
                    [(b.std_mos < reward_cfg.mos_flameout_eps).float() for b in bds]
                )
                .mean()
                .item(),
                3,
            ),
            "sim_mean": round(_col_g("sim"), 4),
            "cer_mean": round(_col_g("cer"), 4),
            "mos_mean": round(_col_g("mos"), 4),
            "t_rollout": round(t_rollout, 2),
            "t_score": round(t_score, 2),
            "t_train": round(t_train, 2),
            "t_opt": round(t_opt, 2),
            "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
            "rss_cur_mb": _current_rss_mb(),
            "gpu_alloc_mb": round(torch.cuda.memory_allocated(cfg.device) / 2**20, 1),
            "gpu_reserved_mb": round(torch.cuda.memory_reserved(cfg.device) / 2**20, 1),
            "dur_s": round(time.monotonic() - t0, 2),
        }
        line = json.dumps(monitor, ensure_ascii=False)
        print(line)
        if monitor_f is not None:
            monitor_f.write(line + "\n")
            monitor_f.flush()

        if (step + 1) % cfg.ckpt_every == 0:
            _save_ckpt(cfg, ttm, optimizer, step)

    if monitor_f is not None:
        monitor_f.close()
