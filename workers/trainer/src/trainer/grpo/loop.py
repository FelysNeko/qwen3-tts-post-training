"""GRPO training loop: `num_prompts` distinct prompts × `group_size` rollouts
per step (Fish-Audio S2 layout), one optimizer update per step.

Pipeline per step:
    Prompt construction (pool item + its pool calibration: centroid
    embedding + RewardConfig) → rollout.pipelined_rollout (sample → decode →
    wav → eager scorer submit → poll → advantage; list[Sample] is the sole
    carrier) → compute_ref/compute_policy → grpo_loss → backward (per-group
    accumulation, equal group weighting) → grad clip → optimizer step →
    monitor line → ckpt.

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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from qwen3_tts_post_training.cache import CacheLayout
from qwen3_tts_post_training.client.trainer import Client
from qwen3_tts_post_training.paths import repo_root
from qwen3_tts_post_training.system import (
    current_rss_mb,
    gpu_allocated_mb,
    gpu_reserved_mb,
    peak_rss_mb,
)
from trainer.grpo.grpo import GRPOConfig, GRPOMetrics, column_weights, grpo_loss
from trainer.grpo.logprob import LogProbComputer
from trainer.grpo.reward import RewardBreakdown, RewardConfig
from trainer.grpo.rollout import Prompt, RolloutParams, Sample, pipelined_rollout
from trainer.grpo.samplers.base import Sampler
from trainer.lora import LoraTrainerModel

logger = logging.getLogger(__name__)

# mirrors rollout_group's pinned subtalker sampling trio (do_sample@T=0.9/top_k=50)
SUBTALKER_TEMPERATURE = 0.9


@dataclass
class TrainConfig:
    # The pool selector(s) — pool dirs are {cache_dir}/{namespace},
    # namespace mirroring the corpus hierarchy. The namespace string IS the
    # speaker name (SFT export spk_id) — `speaker` left None falls back to
    # it. Multi-speaker GRPO passes all pool namespaces: the text pool's
    # speaker keys resolve against them (case-insensitively — pool keys are
    # lowercase export spk_ids like "cyrene/chinese(prc)", on-disk dirs are
    # "cyrene/Chinese(PRC)").
    namespaces: list[str]

    # .jsonl = {"speaker", "text"} rows (multi-speaker pool); otherwise one
    # prompt per line, bound to the single --namespaces entry. Required.
    text_pool_path: str

    # §16: the preprocess cache is the SINGLE source for calibration AND
    # data — metrics.json (sim stats → RewardConfig) + centroid.npy MUST
    # come from the same cache dir the training belongs to; an
    # external-metrics path would allow cross-pool calibration mismatch, so
    # there is no such flag. cache_dir = cache ROOT location (default
    # <repo>/.cache).
    cache_dir: str = str(repo_root() / ".cache")

    # §44: GRPO continues from the pipeline's custom_voice export — the
    # multi-speaker d_ep1 base (bit-verified copy of the 5e-6/B8 SFT arm).
    model_path: str = "runs/d_ep1"
    device: str = "cuda:1"
    lora_r: int = 16
    lora_alpha: float = 64

    num_prompts: int = 8  # distinct prompts per step (Fish Audio S2 layout)
    group_size: int = 8  # rollouts per prompt (the GRPO group)
    num_steps: int = 1
    seed: int = 0
    # §47 VRAM ladder (d_ep1 1.7B, B8, total 821 tokens worst case): rollout
    # 5.1G, ref full-B8 9.4G, policy+backward micro=2 14.4G (micro=4 OOM).
    # 896 > max observed cur_len+t_max 864 with runaway-guard margin.
    token_budget: int = 896  # total tokens (prompt + new) for training: OOM guard and runaway guard; rollout max_new = budget - cur_len
    token_budget_infer: int = 896  # graphed KV pool (lmax) — sized to the budget

    temperature: float = 0.9
    top_k: int = 50
    sampler_impl: str = "graphed"  # hf | fast | compiled | graphed (PROJECT_STATUS §9)

    variant: str = "fish"  # fish (default, A·log π + β·KL) | dr | gspo; vanilla = legacy std-norm, unreachable from the pipelined rollout
    kl_beta: float = 0.001
    logprob_micro: int = 2  # policy micro-chunk (ref stays full-B8: inference-mode fits at 9.4G); micro=4 OOMs in backward at this budget (§47 ladder)
    lr: float = 1e-6
    warmup_steps: int = 20  # linear LR ramp: tames the Adam first-step sign
    # jolt (all params move ±lr at once; observed as KL 586 on step 1, fish3)
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # MOS ablation (2026-09): reward MOS-term shape — lam_p835 > 0 also arms
    # the scorer request (p835=True) and fills g.p835 / the monitor rows.
    lam_mos: float = 0.2  # 0 silences the UTMOS term (ablation arms 1/2)
    lam_p835: float = 0.0  # P.835 OVRL driver weight (arm 2: 0.4; default off)
    mos_role: str = "floor"  # floor | raw — raw drives with raw UTMOS (arm 3)

    scorer_url: str = "http://127.0.0.1:8000"  # FastAPI scorer (§50); client buffers/re-sends — downtime costs latency, never groups
    out_dir: str = "runs/grpo_v1"
    ckpt_every: int = 1
    resume: bool = False
    monitor: bool = True


PoolItem = tuple[str, str]  # (speaker key, prompt text)


def _load_text_pool(cfg: TrainConfig) -> list[PoolItem]:
    """Prompt pool as (speaker, text) pairs — `--text-pool-path` is required.

    `.jsonl` rows carry `{"speaker", "text"}` — the multi-speaker pool; the
    speaker key is the export spk_id (e.g. "cyrene/chinese(prc)"). Legacy
    one-prompt-per-line files bind every prompt to the single `--namespaces`
    entry."""
    assert cfg.text_pool_path, (
        'grpo requires --text-pool-path (.jsonl with {"speaker", "text"}'
        " rows, or a one-prompt-per-line .txt bound to one namespace)"
    )
    if cfg.text_pool_path.endswith(".jsonl"):
        rows = [
            json.loads(line)
            for line in Path(cfg.text_pool_path).read_text().splitlines()
        ]
        return [(row["speaker"], row["text"]) for row in rows]
    assert len(cfg.namespaces) == 1, (
        "single-speaker text pools bind to exactly one --namespaces entry; "
        'multi-speaker pools are .jsonl files with {"speaker", "text"} rows'
    )
    spk = cfg.namespaces[0]
    return [
        (spk, line.strip())
        for line in Path(cfg.text_pool_path).read_text().splitlines()
    ]


def _pick_prompts(pool: list[PoolItem], cfg: TrainConfig, step: int) -> list[PoolItem]:
    """Fish-Audio-style batch: `num_prompts` DISTINCT pool items per step,
    each rolled out `group_size` times (8×8 = 64 rollouts per update).
    Sampling is uniform over the pool rows, so per-step speaker composition
    mirrors the pool's; distinct prompts average out per-group reward noise,
    while within-group variance stays pure sampling noise of the same
    (speaker, text) pair (the GRPO baseline)."""
    rng = random.Random(cfg.seed * 1000003 + step)
    k = min(cfg.num_prompts, len(pool))
    return rng.sample(pool, k)


def _resolve_namespace(root: Path, key: str) -> Path:
    """Case-insensitively resolve a '{voice}/{lang}' pool key under the cache
    root — pool keys are lowercase export spk_ids while on-disk pool dirs
    mirror the corpus hierarchy (e.g. 'cyrene/chinese(prc)' ->
    'cyrene/Chinese(PRC)')."""
    cur = root
    for part in key.split("/"):
        cur = next(
            (p for p in cur.iterdir() if p.is_dir() and p.name.lower() == part.lower()),
            None,
        )
        if cur is None:
            raise FileNotFoundError(f"no cache pool under {root} resolves {key!r}")
    return cur


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


def run_grpo(cfg: TrainConfig) -> None:

    assert cfg.namespaces and all(cfg.namespaces), (
        "grpo takes at least one --namespaces entry (the calibration pool "
        "selector(s); multi-speaker pools resolve their speaker keys "
        "against them)"
    )
    assert Path(cfg.model_path).exists(), (
        f"TTS ckpt not found at {cfg.model_path!r} — pass --model-path"
    )
    pool = _load_text_pool(cfg)

    ttm = LoraTrainerModel(
        cfg.model_path,
        device=cfg.device,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
    )
    sampler = Sampler.build(
        ttm,
        impl=cfg.sampler_impl,
        batch_size=cfg.group_size,
        lmax=cfg.token_budget_infer,
    )
    lpc = LogProbComputer(ttm)
    scorer = Client(url=cfg.scorer_url)
    try:
        _train_loop(cfg, pool, ttm, sampler, lpc, scorer)
    finally:
        scorer.close()


def _train_group_one(
    cfg: TrainConfig,
    g: Sample,
    gs: int,
    micro: int | None,
    n_tr: int,
    lpc: LogProbComputer,
    algo: GRPOConfig,
    group_ids: torch.Tensor,
    skips: Counter,
) -> tuple[torch.Tensor, GRPOMetrics, Sample] | None:
    """One group's loss pass (backward included). Returns the detached step
    loss + metrics, or None when every chunk was non-finite.

    micro=None: full-batch policy graph, one backward.
    micro=k: the policy graph is backwarded PER CHUNK so at most `k`
    sequences' activations ever coexist. Exact recombination: per-token terms
    are row-independent and the loss is a weighted token mean, so
    full = Σ_c loss_c·(W_c/W_total); the advantage arrives precomputed on
    the FULL group (compute_advantage in rollout.py) — never per chunk
    (that would change the Dr.GRPO baseline)."""
    ref = lpc.compute_ref(
        [g.prompt.text] * gs,
        g.codes,
        cfg.temperature,
        subtalker_temperature=SUBTALKER_TEMPERATURE,
        micro=micro,
        speaker=g.prompt.speaker,
    )
    if micro is None:
        pol = lpc.compute_policy(
            [g.prompt.text] * gs,
            g.codes,
            cfg.temperature,
            subtalker_temperature=SUBTALKER_TEMPERATURE,
            speaker=g.prompt.speaker,
        )
        loss, metrics = grpo_loss(
            pol.log_probs, ref.log_probs, g.R, pol.mask, group_ids, algo,
            advantage=g.advantage,
        )
        if not torch.isfinite(loss):
            # non-finite loss (e.g. a sampled token falling out of the
            # teacher-forcing top-k) backprops NaN grads — drop the group
            skips["nonfinite_loss"] += 1
            return None
        (loss / n_tr).backward()
        return loss.detach(), metrics, g

    A = g.advantage
    gmean = A.new_zeros(())
    gstd = A.new_zeros(())
    W_total = (ref.mask * column_weights(ref.mask, algo)).sum().clamp_min(1e-12)
    W_used = ref.log_probs.new_zeros(())
    loss_acc = ref.log_probs.new_zeros(())
    pol_acc = ref.log_probs.new_zeros(())
    kl_acc: list[torch.Tensor] = []
    for i in range(0, gs, micro):
        sl = slice(i, i + micro)
        pol_c = lpc.compute_policy(
            [g.prompt.text] * micro,
            g.codes[sl],
            cfg.temperature,
            subtalker_temperature=SUBTALKER_TEMPERATURE,
            speaker=g.prompt.speaker,
        )
        width = ref.log_probs.shape[1]
        pad = width - pol_c.log_probs.shape[1]
        lp_c = F.pad(pol_c.log_probs, (0, pad)) if pad else pol_c.log_probs
        mk_c = F.pad(pol_c.mask, (0, pad)) if pad else pol_c.mask
        loss_c, met_c = grpo_loss(
            lp_c,
            ref.log_probs[sl],
            A[sl],
            mk_c,
            group_ids[sl],
            algo,
            advantage=A[sl],
        )
        if not torch.isfinite(loss_c):
            # one non-finite chunk drops only that chunk (earlier
            # chunks are already backwarded and cannot be undone);
            # remaining chunks renormalize over W_used below
            skips["nonfinite_loss"] += 1
            continue
        W_used = W_used + met_c.weight_mass
        loss_acc = loss_acc + loss_c.detach() * met_c.weight_mass
        pol_acc = pol_acc + met_c.policy_loss.detach() * met_c.weight_mass
        if met_c.kl is not None:
            kl_acc.append(met_c.kl * met_c.weight_mass)
        # equal group weighting across the step: each chunk carries
        # its share of the group's 1/n_tr contribution
        (loss_c * met_c.weight_mass / W_total / n_tr).backward()
    if W_used.item() == 0.0:
        return None  # every chunk non-finite — the group contributed no grads
    metrics = GRPOMetrics(
        loss_acc / W_used,
        pol_acc / W_used,
        torch.stack(kl_acc).sum() / W_used if kl_acc else None,
        A,
        gmean,
        gstd,
    )
    return loss_acc.detach() / W_used, metrics, g


def _train_groups(
    cfg: TrainConfig,
    trainable: list[Sample],
    lpc: LogProbComputer,
    algo: GRPOConfig,
    group_ids: torch.Tensor,
    skips: Counter,
) -> tuple[list[tuple[torch.Tensor, GRPOMetrics, Sample]], float]:
    """Phase 3: one optimizer pass worth of gradient accumulation — one group
    per iteration, equal group weighting (each group contributes
    1/len(trainable) to the update)."""
    t0 = time.monotonic()
    micro = cfg.logprob_micro or None
    gs = cfg.group_size
    n_tr = len(trainable)
    trained: list[tuple[torch.Tensor, GRPOMetrics, Sample]] = []
    for g in trainable:
        out = _train_group_one(cfg, g, gs, micro, n_tr, lpc, algo, group_ids, skips)
        if out is not None:
            trained.append(out)
    return trained, time.monotonic() - t0


def _bundle_mean(key: str, bds: list[RewardBreakdown]) -> float:
    return torch.cat([getattr(b, key) for b in bds]).float().mean().item()


def _group_mean(key: str, groups: list[Sample]) -> float:
    return torch.cat([getattr(g, key) for g in groups]).mean().item()


def _build_monitor(
    step: int,
    device: str,
    trained: list[tuple[torch.Tensor, GRPOMetrics, Sample]],
    skips: Counter,
    grad_norm: torch.Tensor,
    lr_t: float,
    reward_cfgs: dict[str, RewardConfig],
    t_rollout: float,
    t_score: float,
    t_train: float,
    t_opt: float,
    t0: float,
) -> dict:
    losses = torch.stack([l for l, _, _ in trained])
    pol_losses = torch.stack([m.policy_loss for _, m, _ in trained])
    kls = [m.kl for _, m, _ in trained if m.kl is not None]
    advs = torch.cat([m.advantage for _, m, _ in trained])
    R_all = torch.cat([g.R for _, _, g in trained])
    bds = [g.bd for _, _, g in trained]
    p835_groups = [g for _, _, g in trained if g.p835 is not None]

    per_speaker: dict[str, dict[str, float]] = {}
    for _, _, g in trained:
        d = per_speaker.setdefault(g.prompt.speaker, {"n": 0, "sim": 0.0, "cer": 0.0})
        d["n"] += 1
        d["sim"] += float(g.sim.mean())
        d["cer"] += float(g.cer.mean())

    return {
        "step": step,
        "groups_trained": len(trained),
        "groups_skipped": sum(skips.values()),
        "skips": dict(skips),
        "loss": round(losses.mean().item(), 4),
        "policy_loss": round(pol_losses.mean().item(), 4),
        "kl": round(torch.stack(kls).mean().item(), 5) if kls else None,
        "grad_norm": round(grad_norm.item(), 4),
        "lr": f"{lr_t:.2e}",
        "mean_R": round(R_all.mean().item(), 4),
        "t_max": max(g.t_max for _, _, g in trained),
        "adv_std": round(advs.std(unbiased=False).item(), 4),
        "r_sv_mean": round(_bundle_mean("r_sv", bds), 4),
        "r_wer_mean": round(_bundle_mean("r_wer", bds), 4),
        "r_mos_mean": round(_bundle_mean("r_mos", bds), 4),
        "r_p835_mean": round(_bundle_mean("r_p835", bds), 4),
        "std_sv": round(torch.cat([b.std_sv for b in bds]).mean().item(), 5),
        "std_wer": round(torch.cat([b.std_wer for b in bds]).mean().item(), 5),
        "mos_dead_frac": round(
            torch.stack(
                [
                    (b.std_mos < reward_cfgs[g.speaker].mos_flameout_eps).float()
                    for b, (_, _, g) in zip(bds, trained)
                ]
            )
            .mean()
            .item(),
            3,
        ),
        "p835_dead_frac": round(
            torch.stack(
                [
                    (b.std_p835 < reward_cfgs[g.speaker].p835_flameout_eps).float()
                    for b, (_, _, g) in zip(bds, trained)
                ]
            )
            .mean()
            .item(),
            3,
        ),
        "sim_mean": round(_group_mean("sim", [g for _, _, g in trained]), 4),
        "cer_mean": round(_group_mean("cer", [g for _, _, g in trained]), 4),
        "mos_mean": round(_group_mean("mos", [g for _, _, g in trained]), 4),
        # p835 travels only when armed (cfg.lam_p835 > 0) — None otherwise
        "p835_mean": (
            round(_group_mean("p835", p835_groups), 4) if p835_groups else None
        ),
        "per_speaker": {
            k: {
                "n": v["n"],
                "sim": round(v["sim"] / v["n"], 4),
                "cer": round(v["cer"] / v["n"], 4),
            }
            for k, v in sorted(per_speaker.items())
        },
        "t_rollout": round(t_rollout, 2),
        "t_score": round(t_score, 2),
        "t_train": round(t_train, 2),
        "t_opt": round(t_opt, 2),
        "rss_mb": peak_rss_mb(),
        "rss_cur_mb": current_rss_mb(),
        "gpu_alloc_mb": gpu_allocated_mb(device),
        "gpu_reserved_mb": gpu_reserved_mb(device),
        "dur_s": round(time.monotonic() - t0, 2),
    }


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------


def _train_loop(
    cfg: TrainConfig,
    pool: list[PoolItem],
    ttm: LoraTrainerModel,
    sampler: Sampler,
    lpc: LogProbComputer,
    scorer: Client,
) -> None:
    # fused like SFT (math identical, single kernel per param, no foreach
    # transient) — trainable set is small (LoRA + codec_head), but every MB
    # counts at the §47 14.4G teacher-forcing peak
    optimizer = torch.optim.AdamW(
        ttm.trainable_parameters,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        fused=True,
    )
    start_step = _load_ckpt(cfg, ttm, optimizer) if cfg.resume else 0

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    monitor_f = (out / "monitor.jsonl").open("a") if cfg.monitor else None
    groups_f = (out / "groups.jsonl").open("a")

    algo = GRPOConfig(
        variant=cfg.variant,
        kl_beta=cfg.kl_beta,
        num_code_groups=ttm.talker.config.num_code_groups,
    )
    # per-speaker calibration: one RewardConfig (sv stats from that pool's
    # metrics.json) and one unit-norm centroid row per pool speaker; sims are
    # a row-select off the stacked centroid matrix (one batched matmul).
    speakers = sorted({spk for spk, _ in pool})
    reward_cfgs: dict[str, RewardConfig] = {}
    centroid_rows: list[torch.Tensor] = []
    for spk in speakers:
        layout = CacheLayout(_resolve_namespace(Path(cfg.cache_dir), spk))
        sim_stats = layout.load_metrics()["sim"]
        reward_cfgs[spk] = RewardConfig(
            sv_center=sim_stats["mean"],
            sv_scale=sim_stats["std"],
            lam_mos=cfg.lam_mos,
            lam_p835=cfg.lam_p835,
            mos_role=cfg.mos_role,
        )
        centroid = torch.as_tensor(
            layout.load_centroid(), dtype=torch.float32, device=cfg.device
        )
        centroid_rows.append(
            centroid / centroid.norm()
        )  # mirrors the old scorer set_ref recipe
    centroids = torch.stack(centroid_rows)  # [n_spk, hidden]
    spk_row = {spk: i for i, spk in enumerate(speakers)}
    group_ids = torch.zeros(cfg.group_size, dtype=torch.long, device=cfg.device)

    for step in range(start_step, start_step + cfg.num_steps):
        lr_t = cfg.lr * min(1.0, (step + 1) / max(1, cfg.warmup_steps))
        for pg in optimizer.param_groups:
            pg["lr"] = lr_t
        t0 = time.monotonic()
        prompts = [
            Prompt(
                loc=gi,
                speaker=spk,
                text=text,
                embedding=centroids[spk_row[spk]],
                reward_cfg=reward_cfgs[spk],
            )
            for gi, (spk, text) in enumerate(_pick_prompts(pool, cfg, step))
        ]
        skips: Counter = Counter()

        # pipeline: rollout → eager submit (overlap) → poll → advantage;
        # list[Sample] is the sole carrier into the update
        samples, t_rollout, t_score = pipelined_rollout(
            sampler,
            scorer,
            prompts,
            RolloutParams(
                step=step,
                seed_base=cfg.seed,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                token_budget=cfg.token_budget,
            ),
            groups_f=groups_f,
            skips=skips,
        )
        if not samples:
            logger.warning(f"step {step}: no trainable group — skipped")
            continue

        # gradient accumulation, one group per pass; then update
        trained, t_train = _train_groups(cfg, samples, lpc, algo, group_ids, skips)
        if not trained:
            logger.warning(f"step {step}: all losses non-finite — skipped")
            continue
        t_opt0 = time.monotonic()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            ttm.trainable_parameters, cfg.grad_clip
        )
        optimizer.step()
        t_opt = time.monotonic() - t_opt0

        monitor = _build_monitor(
            step,
            cfg.device,
            trained,
            skips,
            grad_norm,
            lr_t,
            reward_cfgs,
            t_rollout,
            t_score,
            t_train,
            t_opt,
            t0,
        )
        logger.info(
            f"step {step} | {monitor['groups_trained']} groups,"
            f" {monitor['groups_skipped']} skipped"
            f" | loss {monitor['loss']} (policy {monitor['policy_loss']},"
            f" kl {monitor['kl']}) | grad {monitor['grad_norm']} lr {monitor['lr']}"
            f" | R {monitor['mean_R']} adv_std {monitor['adv_std']}"
            f" t_max {monitor['t_max']} | r_sv {monitor['r_sv_mean']}"
            f" r_wer {monitor['r_wer_mean']} r_mos {monitor['r_mos_mean']}"
            f" | sim {monitor['sim_mean']} cer {monitor['cer_mean']}"
            f" mos {monitor['mos_mean']} | {monitor['dur_s']}s"
        )
        if monitor_f is not None:
            monitor_f.write(json.dumps(monitor, ensure_ascii=False) + "\n")
            monitor_f.flush()

        if (step + 1) % cfg.ckpt_every == 0:
            _save_ckpt(cfg, ttm, optimizer, step)

    if monitor_f is not None:
        monitor_f.close()
    groups_f.close()
