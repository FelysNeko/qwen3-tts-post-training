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
    group_size: int = 8
    num_steps: int = 1
    seed: int = 0
    max_new_tokens: int = 4096

    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float | None = None
    sampler_impl: str = "hf"  # hf | fast | compiled (PROJECT_STATUS §9)

    variant: str = "dr"
    kl_beta: float = 0.001
    lr: float = 1e-5
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
    """Standard GRPO group: ONE prompt per step, rolled out `group_size` times
    (the within-group variance is pure sampling noise of the same text; the
    group-advantage std then measures rollout randomness, not prompt spread)."""
    rng = random.Random(cfg.seed * 1000003 + step)
    return [rng.choice(pool)] * cfg.group_size


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
        t0 = time.monotonic()
        prompts = _pick_prompts(pool, cfg, step)
        rollout = rollout_group(
            sampler, decoder, prompts, seed=cfg.seed + step, tag=f"step{step}"
        )
        t_rollout = time.monotonic() - t0

        t_max = max(c.shape[0] for c in rollout.codes)
        if t_max > 400:
            # runaway no-EOS group (degenerate policy state): scoring it is a
            # waste and teacher-forcing [8, >400] with grads OOMs the 16 GB
            # trainer GPU (graphed smoke C1v8)
            print(json.dumps({"step": step, "skip": "runaway_rollout",
                              "t_max": t_max}), flush=True)
            continue

        t_score = 0.0
        try:
            results = scorer.score(
                [
                    ScoreItem(wav=str(w), text=p)
                    for w, p in zip(rollout.wav_paths, prompts)
                ]
            )
            sim = _scores_to_tensor(results, "sim", cfg.device)
            cer = _scores_to_tensor(results, "cer", cfg.device)
            mos = _scores_to_tensor(results, "mos", cfg.device)
            t_score = time.monotonic() - t0 - t_rollout
        except RuntimeError as e:
            print(
                json.dumps(
                    {"step": step, "skip": "scorer_failure", "reason": str(e)},
                    ensure_ascii=False,
                )
            )
            continue

        if needs_resample(sim, cer):
            print(
                json.dumps(
                    {
                        "step": step,
                        "skip": "needs_resample",
                        "sim_std": f"{sim.float().std().item():.5f}",
                        "cer_std": f"{cer.float().std().item():.5f}",
                    },
                    ensure_ascii=False,
                )
            )
            continue

        R, bd = reward_v3(sim, cer, mos, reward_cfg)
        t_ref0 = time.monotonic()
        ref = lpc.compute_ref(prompts, rollout.codes, cfg.temperature, cfg.top_k)
        t_ref = time.monotonic() - t_ref0
        pol = lpc.compute_policy(prompts, rollout.codes, cfg.temperature, cfg.top_k)
        loss, metrics = grpo_loss(
            pol.log_probs, ref.log_probs, R, pol.mask, group_ids, algo
        )

        t_opt0 = time.monotonic()
        optimizer.zero_grad()
        loss.backward()
        if not torch.isfinite(loss):
            # non-finite loss (e.g. a sampled token falling out of the
            # teacher-forcing top-k after a bad update) backprops NaN grads;
            # stepping on them writes NaN into the weights permanently
            print(
                json.dumps({"step": step, "skip": "non_finite_loss"}),
                flush=True,
            )
            optimizer.zero_grad(set_to_none=True)
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_(
            ttm.trainable_parameters, cfg.grad_clip
        )
        optimizer.step()
        t_opt = time.monotonic() - t_opt0

        monitor = {
            "step": step,
            "loss": round(loss.item(), 4),
            "policy_loss": round(metrics.policy_loss.item(), 4),
            "kl": round(metrics.kl.item(), 5) if metrics.kl is not None else None,
            "grad_norm": round(grad_norm.item(), 4),
            "mean_R": round(R.mean().item(), 4),
            "t_max": t_max,
            "adv_std": round(metrics.advantage.std(unbiased=False).item(), 4),
            "r_sv_mean": round(bd.r_sv.mean().item(), 4),
            "r_wer_mean": round(bd.r_wer.mean().item(), 4),
            "r_mos_mean": round(bd.r_mos.mean().item(), 4),
            "std_sv": round(bd.std_sv.item(), 5),
            "std_wer": round(bd.std_wer.item(), 5),
            "mos_dead": bool((bd.std_mos < reward_cfg.mos_flameout_eps).item()),
            "sim_mean": round(sim.mean().item(), 4),
            "cer_mean": round(cer.mean().item(), 4),
            "mos_mean": round(mos.mean().item(), 4),
            "t_rollout": round(t_rollout, 2),
            "t_score": round(t_score, 2),
            "t_ref": round(t_ref, 2),
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
