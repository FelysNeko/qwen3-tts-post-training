"""GRPO training loop: `num_prompts` distinct prompts × `group_size` rollouts
per step (Fish-Audio S2 layout), one optimizer update per step.

Pipeline per step:
    prompts → rollout (sample → decode → wav) → scorer (ZMQ PUSH/PULL) →
    reward_v3 → needs_resample? (skip) → compute_ref/compute_policy →
    grpo_loss → backward (per-group accumulation, equal group weighting) →
    grad clip → optimizer step → monitor line → ckpt.

The reference policy is the same weights with LoRA adapters disabled
(LoraTrainerModel.set_adapter), so only one model lives in VRAM. Ckpts carry
the LoRA deltas + codec head + optimizer state so `--resume` continues
cleanly.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from qwen3_tts_post_training.cache import CacheLayout
from qwen3_tts_post_training.client.protocol import ScoreField, ScoreItem
from qwen3_tts_post_training.client.trainer import Client
from qwen3_tts_post_training.paths import repo_root
from qwen3_tts_post_training.reward.reward import reward_v3
from qwen3_tts_post_training.system import (
    current_rss_mb,
    gpu_allocated_mb,
    gpu_reserved_mb,
    peak_rss_mb,
)
from trainer.grpo.grpo import GRPOConfig, grpo_loss, needs_resample
from trainer.grpo.logprob import LogProbComputer
from trainer.grpo.rollout import rollout_group
from trainer.grpo.samplers.base import Sampler
from trainer.lora import LoraTrainerModel

logger = logging.getLogger(__name__)

# mirrors rollout_group's pinned subtalker sampling trio (do_sample@T=0.9/top_k=50)
SUBTALKER_TEMPERATURE = 0.9

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
    # The pool selector — REQUIRED, no default: the pool dir is
    # {cache_dir}/{namespace}. The CLI enforces --namespace; a programmatic
    # caller must name its pool explicitly.
    namespace: str

    # §16: the preprocess cache is the SINGLE source for calibration AND
    # data — metrics.json (sim stats → RewardConfig) + centroid.npy MUST
    # come from the same cache dir the training belongs to; an
    # external-metrics path would allow cross-pool calibration mismatch, so
    # there is no such flag. cache_dir = cache ROOT location (default
    # <repo>/.cache).
    cache_dir: str = str(repo_root() / ".cache")

    model_path: str = "/mnt/d/Repository/models/PhiLia093-TTS/"
    device: str = "cuda:1"
    lora_r: int = 16
    lora_alpha: float = 64
    speaker: str = "cyrene"

    text_pool: tuple[str, ...] = TEXT_POOL
    text_pool_path: str | None = None  # if set, overrides text_pool (one line each)
    num_prompts: int = 8  # distinct prompts per step (Fish Audio S2 layout)
    group_size: int = 8  # rollouts per prompt (the GRPO group)
    num_steps: int = 1
    seed: int = 0
    token_budget: int = 512  # total tokens budget (prompt + new + 8 overhead) for training: OOM guard (B8 T500 14.4G) and runaway guard; rollout max_new = budget - cur_len
    token_budget_infer: int = (
        1024  # total tokens budget for inference (graphed lmax); 1024 ≈60s audio
    )

    temperature: float = 0.9
    top_k: int = 50
    sampler_impl: str = "hf"  # hf | fast | compiled | graphed (PROJECT_STATUS §9)

    variant: str = "dr"
    kl_beta: float = 0.001
    lr: float = 1e-5
    warmup_steps: int = 10  # linear LR ramp: tames the Adam first-step sign
    # jolt (all params move ±lr at once; observed as KL 586 on step 1, fish3)
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    scorer_push_endpoint: str = "tcp://127.0.0.1:5555"
    scorer_pull_endpoint: str = "tcp://127.0.0.1:5556"
    scorer_timeout: float = 600.0
    out_dir: str = "runs/grpo_v1"
    ckpt_every: int = 1
    resume: bool = False
    monitor: bool = True


def _load_text_pool(cfg: TrainConfig) -> list[str]:
    """Text pool from file (one prompt per line) or the built-in placeholder."""
    if cfg.text_pool_path is None:
        return list(cfg.text_pool)
    return [line.strip() for line in Path(cfg.text_pool_path).read_text().splitlines()]


def _pick_prompts(pool: list[str], cfg: TrainConfig, step: int) -> list[str]:
    """Fish-Audio-style batch: `num_prompts` DISTINCT prompts per step, each
    rolled out `group_size` times (8×8 = 64 rollouts per update). Distinct
    prompts average out per-group reward noise; within-group variance stays
    pure sampling noise of the same text (the GRPO baseline)."""
    rng = random.Random(cfg.seed * 1000003 + step)
    k = min(cfg.num_prompts, len(pool))
    return rng.sample(pool, k)


def _cleanup_wavs(wav_paths: list[Path]) -> None:
    for p in wav_paths:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
    # try remove parent dir if empty
    if wav_paths:
        try:
            wav_paths[0].parent.rmdir()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# ckpt (LoRA deltas + codec head + optimizer) with resume support
# ---------------------------------------------------------------------------


def _ckpt_path(out_dir: Path, step: int) -> Path:
    return out_dir / f"step_{step:05d}.pt"


def _save_ckpt(cfg: TrainConfig, ttm: LoraTrainerModel, optimizer, step: int) -> None:
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


def _load_ckpt(cfg: TrainConfig, ttm: LoraTrainerModel, optimizer) -> int:
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
    logger.info(f"resumed at step {step} from {_ckpt_path(out, step)}")
    return step + 1


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------


def run_grpo(cfg: TrainConfig) -> None:

    assert Path(cfg.model_path).exists(), (
        f"TTS ckpt not found at {cfg.model_path!r} — pass --model-path"
    )

    ttm = LoraTrainerModel(
        cfg.model_path,
        device=cfg.device,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
    )
    sampler = Sampler.build(
        ttm,
        impl=cfg.sampler_impl,
        speaker=cfg.speaker,
        batch_size=cfg.group_size,
        lmax=cfg.token_budget_infer,
    )
    lpc = LogProbComputer(ttm, speaker=cfg.speaker)
    scorer = Client(
        push_endpoint=cfg.scorer_push_endpoint,
        pull_endpoint=cfg.scorer_pull_endpoint,
        timeout_s=cfg.scorer_timeout,
    )
    try:
        _train_loop(cfg, ttm, sampler, lpc, scorer)
    finally:
        scorer.close()


def _train_loop(
    cfg: TrainConfig,
    ttm: LoraTrainerModel,
    sampler: Sampler,
    lpc: LogProbComputer,
    scorer: Client,
) -> None:
    optimizer = torch.optim.AdamW(
        ttm.trainable_parameters, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    start_step = _load_ckpt(cfg, ttm, optimizer) if cfg.resume else 0

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    monitor_f = (out / "monitor.jsonl").open("a") if cfg.monitor else None

    algo = GRPOConfig(
        variant=cfg.variant,
        kl_beta=cfg.kl_beta,
        num_code_groups=ttm.talker.config.num_code_groups,
    )
    layout = CacheLayout(Path(cfg.cache_dir) / cfg.namespace)
    reward_cfg = layout.reward_config()
    sv_centroid = torch.as_tensor(
        layout.load_centroid(), dtype=torch.float32, device=cfg.device
    )
    sv_centroid /= sv_centroid.norm()  # mirrors the old scorer set_ref recipe
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

        # ---- phase 1: rollout + push (zero-thread ZMQ batch pipeline) ----
        # Trainer rolls out group by group and PUSHes each group's wavs
        # (non-blocking, HWM 1000); the scorer drains and scores in parallel
        # while the trainer continues. Phase 2 drains the PULL side — wall
        # time ≈ n_rollouts × rollout + last score.
        t_roll0 = time.monotonic()
        pending: list[tuple[dict, int]] = []  # (group, req_id)
        for gi, prompt in enumerate(prompts):
            rollout = rollout_group(
                sampler,
                ttm,
                prompt,
                seed=cfg.seed * 1000003 + step * 1009 + gi,
                tag=f"step{step}g{gi}",
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                token_budget=cfg.token_budget,
            )
            t_max = max(c.shape[0] for c in rollout.codes)
            cur_len = rollout.cur_len
            if t_max + cur_len >= cfg.token_budget:
                skips["runaway"] = skips.get("runaway", 0) + 1
                _cleanup_wavs(rollout.wav_paths)
                continue
            g = {
                "gi": gi,
                "prompt": prompt,
                "codes": rollout.codes,
                "wavs": rollout.wav_paths,
                "t_max": t_max,
            }
            try:
                rid = scorer.send_score(
                    [ScoreItem(wav_path=str(p), text=prompt) for p in g["wavs"]],
                    fields={ScoreField.EMBEDDING, ScoreField.CER, ScoreField.MOS},
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"step {step}: scorer send failed ({e}) — group skipped")
                _cleanup_wavs(g["wavs"])
                skips["scorer_item"] = skips.get("scorer_item", 0) + 1
                continue
            pending.append((g, rid))
        t_rollout = time.monotonic() - t_roll0

        # ---- phase 2: score — drain PULL responses, trainer owns unlink ----
        t_score = 0.0
        groups: list[dict] = []
        if pending:
            t_s0 = time.monotonic()
            for g, rid in pending:
                try:
                    results = scorer.recv_score(rid, timeout=scorer.timeout_s)
                except (TimeoutError, RuntimeError) as e:
                    logger.warning(f"step {step}: scorer failed ({e}) — group skipped")
                    _cleanup_wavs(g["wavs"])
                    skips["scorer_item"] = skips.get("scorer_item", 0) + 1
                    continue
                # trainer owns lifecycle: delete tmpfs wavs after scoring
                _cleanup_wavs(g["wavs"])
                # sims are local now: one batched matmul against the centroid
                # (float32-exact transport; ~1e-7 accumulation difference vs
                # the old scorer-side dot — see STATUS §16.9)
                g["sim"] = (
                    torch.tensor(
                        [r.get_embedding_unwrap() for r in results],
                        dtype=torch.float32,
                        device=cfg.device,
                    )
                    @ sv_centroid
                )
                g["cer"] = torch.tensor(
                    [r.get_cer_unwrap() for r in results],
                    dtype=torch.float32,
                    device=cfg.device,
                )
                g["mos"] = torch.tensor(
                    [r.get_mos_unwrap() for r in results],
                    dtype=torch.float32,
                    device=cfg.device,
                )
                groups.append(g)
            t_score = time.monotonic() - t_s0

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
            logger.warning(f"step {step}: no trainable group — skipped")
            continue

        # ---- phase 3: train — gradient accumulation, one group per pass ----
        # (peak activation memory stays at single-group level; each group
        # contributes 1/len(trainable) to the update, equal group weighting)
        t_train0 = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        trained: list[tuple[torch.Tensor, object, dict]] = []
        for g in trainable:
            ref = lpc.compute_ref(
                [g["prompt"]] * gs,
                g["codes"],
                cfg.temperature,
                subtalker_temperature=SUBTALKER_TEMPERATURE,
            )
            pol = lpc.compute_policy(
                [g["prompt"]] * gs,
                g["codes"],
                cfg.temperature,
                subtalker_temperature=SUBTALKER_TEMPERATURE,
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
            logger.warning(f"step {step}: all losses non-finite — skipped")
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
            "rss_mb": peak_rss_mb(),
            "rss_cur_mb": current_rss_mb(),
            "gpu_alloc_mb": gpu_allocated_mb(cfg.device),
            "gpu_reserved_mb": gpu_reserved_mb(cfg.device),
            "dur_s": round(time.monotonic() - t0, 2),
        }
        line = json.dumps(monitor, ensure_ascii=False)
        logger.info(
            f"step {step} | {len(trained)} groups, {sum(skips.values())} skipped"
            f" | loss {monitor['loss']} (policy {monitor['policy_loss']},"
            f" kl {monitor['kl']}) | grad {monitor['grad_norm']} lr {monitor['lr']}"
            f" | R {monitor['mean_R']} adv_std {monitor['adv_std']}"
            f" t_max {monitor['t_max']} | r_sv {monitor['r_sv_mean']}"
            f" r_wer {monitor['r_wer_mean']} r_mos {monitor['r_mos_mean']}"
            f" | sim {monitor['sim_mean']} cer {monitor['cer_mean']}"
            f" mos {monitor['mos_mean']} | {monitor['dur_s']}s"
        )
        if monitor_f is not None:
            monitor_f.write(line + "\n")
            monitor_f.flush()

        if (step + 1) % cfg.ckpt_every == 0:
            _save_ckpt(cfg, ttm, optimizer, step)

    if monitor_f is not None:
        monitor_f.close()
