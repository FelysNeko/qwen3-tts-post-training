"""GRPO family training losses — pure torch, no model, switchable variants.

The three algorithms (design truth source: MD §七 + §4.3 of Fish S2 report,
arXiv:2603.08823):

- "vanilla"  A = (R − mean) / (std + eps); per-token clipped ratio loss.
             DeepSeekMath / FlowTTS-GRPO.
- "dr"       A = R − mean  (NO std division — low-variance groups don't get
             amplified); same per-token clipped ratio loss. Fish S2 / MD §7.
- "gspo"     No per-token clip; sequence-level IS ratio
             ρ_seq = Π_t ρ_t = exp(Σ_t log-ratio_t) weights the advantage
             (Qwen3-TTS official route, arXiv:2601.15621).

DAPO Clip-Higher is layered on the clipped variants (GLM validated): decoupled
ε_low/ε_high with ε_high > ε_low. KL via the Schulman estimator
kl_t = ρ_t − 1 − log ρ_t (per token, non-negative, ≈ x²/2 + x³/6 for small
log-ratio x).

Group resampling criterion uses SV/WER variance ONLY (MOS excluded — GLM EMO
lesson: a bimodal r_mos makes "all-1" groups look zero-variance while being
good samples).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

ADV_VANILLA = "vanilla"
ADV_DR = "dr"
VARIANT_GSPO = "gspo"


@dataclass
class GRPOConfig:
    variant: str = ADV_DR  # "vanilla" | "dr" | "gspo"
    dapo_clip: bool = True  # decoupled ε_low/ε_high (clip-higher)
    eps_low: float = 0.2
    eps_high: float = 0.3
    std_eps: float = 1e-4  # vanilla advantage denominator guard
    kl_beta: float = 0.001
    kl_estimator: str = "schulman_k3"  # "schulman_k3" | "k2"
    use_kl: bool = True
    subtalker_weight: float = 1.0  # MTP γ: loss/KL weight on codebooks 1..15
    subtalker_time_norm: bool = True  # divide packed columns by codebook count
    num_code_groups: int = 16  # packing stride (set from model config)

    def __post_init__(self) -> None:
        if self.subtalker_weight < 0:
            raise ValueError(
                f"subtalker_weight must be >= 0, got {self.subtalker_weight}"
            )


def _column_weights(
    mask: torch.Tensor,
    cfg: GRPOConfig,
) -> torch.Tensor:
    """Per-column loss/KL weights [B, T] for the packed layout.

    Packing contract (logprob.py): column `t*Q + j` = codebook j of time step
    t; block 0 of each step is the semantic codebook. γ weights codebooks
    1..Q-1; time normalization divides all columns by Q so a full 16-codebook
    step contributes the same total weight as a semantic-only (Q=1) step.
    Returns ones when Q==1 (no packing) — legacy shape-agnostic behavior."""
    if cfg.num_code_groups <= 1:
        return torch.ones_like(mask)
    w = torch.full_like(mask, cfg.subtalker_weight)
    is_sem = (
        torch.arange(mask.shape[1], device=mask.device) % cfg.num_code_groups == 0
    ).unsqueeze(0)
    w = torch.where(is_sem, torch.ones_like(w), w)
    if cfg.subtalker_time_norm:
        w = w / cfg.num_code_groups
    return w


class GRPOMetrics(NamedTuple):
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl: torch.Tensor | None
    advantage: torch.Tensor
    group_mean: torch.Tensor
    group_std: torch.Tensor
    rho: torch.Tensor | None = None
    clamped: torch.Tensor | None = None


def _segment_mean_std(
    x: torch.Tensor, group_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-group mean/std of x grouped by group_ids (both length B)."""
    n_groups = int(group_ids.max().item()) + 1
    w = torch.zeros(n_groups, device=x.device, dtype=x.dtype)
    w.scatter_add_(0, group_ids, torch.ones_like(x))
    s = torch.zeros(n_groups, device=x.device, dtype=x.dtype)
    s.scatter_add_(0, group_ids, x)
    mean = s / w.clamp_min(1)
    s2 = torch.zeros(n_groups, device=x.device, dtype=x.dtype)
    s2.scatter_add_(0, group_ids, x * x)
    var = (s2 / w.clamp_min(1) - mean * mean).clamp_min(0)
    std = var.sqrt()
    return mean[group_ids], std[group_ids]


def group_advantage(
    rewards: torch.Tensor,
    variant: str = ADV_DR,
    group_ids: torch.Tensor | None = None,
    std_eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dr.GRPO / vanilla advantage. Returns (A, group_mean, group_std).

    Dr.GRPO: A = R − mean. Vanilla: A = (R − mean)/(std + eps). GSPO uses the
    same mean-subtracted rewards as the advantage (control variate).
    """
    if group_ids is None:
        gmean = rewards.mean()
        gstd = rewards.std(unbiased=False)
    else:
        gmean, gstd = _segment_mean_std(rewards, group_ids)
    A = rewards - gmean
    if variant == ADV_VANILLA:
        A = A / (gstd + std_eps)
    return A, gmean, gstd


def kl_divergence(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
    estimator: str = "schulman_k3",
    col_w: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean per-token KL over masked positions (optionally column-weighted).

    Column-weighted reduction: both the numerator and the effective weight
    mass divide by Σ(mask·col_w), so a γ-weighted column set stays a proper
    mean over what it actually includes."""
    log_ratio = log_probs - ref_log_probs
    if estimator == "schulman_k3":
        kl = torch.exp(log_ratio) - 1.0 - log_ratio
    elif estimator == "k2":
        kl = 0.5 * log_ratio * log_ratio
    else:
        raise ValueError(f"unknown kl_estimator {estimator!r}")
    if col_w is None:
        return (kl * mask).sum() / mask.sum().clamp_min(1)
    # guard inf*0=NaN: exp overflow in a zero-weighted column (e.g. a bogus
    # probe value) must not leak NaN into the sum — drop it before the mul
    w = mask * col_w
    kl = torch.where(w > 0, kl, torch.zeros_like(kl))
    return (kl * w).sum() / w.sum().clamp_min(1e-12)


def _clipped_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantage: torch.Tensor,
    mask: torch.Tensor,
    cfg: GRPOConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    col_w = _column_weights(mask, cfg)
    w = mask * col_w
    log_ratio = log_probs - ref_log_probs
    # guard inf*0=NaN (see kl_divergence): zero-weighted columns whose exp
    # overflowed must not leak NaN into the sum
    safe_ratio = torch.where(w > 0, log_ratio, torch.zeros_like(log_ratio))
    rho = torch.exp(safe_ratio)
    low, high = 1.0 - cfg.eps_low, 1.0 + cfg.eps_high
    if not cfg.dapo_clip:
        high = 1.0 + cfg.eps_low
    clamped = rho.clamp(low, high)
    adv = advantage.unsqueeze(-1)
    loss_t = -(torch.minimum(rho * adv, clamped * adv))  # [B, T]
    loss = (loss_t * w).sum() / w.sum().clamp_min(1e-12)
    return loss, {"rho": torch.exp(log_ratio), "clamped": clamped}


def _gspo_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantage: torch.Tensor,
    mask: torch.Tensor,
    cfg: GRPOConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    log_ratio = log_probs - ref_log_probs
    seq_ratio = torch.exp((log_ratio * mask).sum(-1))  # [B]
    loss = -(seq_ratio * advantage).mean()
    return loss, {"seq_ratio": seq_ratio}


def grpo_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    mask: torch.Tensor,
    group_ids: torch.Tensor | None = None,
    cfg: GRPOConfig | None = None,
) -> tuple[torch.Tensor, GRPOMetrics]:
    """One optimizer-step GRPO loss.

    Args:
        log_probs: [B, T] per-token log-probs of the sampled sequence under the
            policy (adapter on), already aligned to the sampling distribution.
            Packed layout (logprob.py): column `t*Q + j` = codebook j of time
            step t — the semantic head lives in columns where t*Q % Q == 0.
        ref_log_probs: [B, T] same, under the reference (adapter off).
        rewards: [B] composed reward R (see reward.reward_v3), one per sample.
        mask: [B, T] 1 on valid code slots, 0 on pad/prompt.
        group_ids: optional [B] group index per sample (default: one group).

    MTP γ (`cfg.subtalker_weight`): weight on packed columns of codebooks
    1..Q-1. History: MD §四 pinned γ=0 ("v1 不提取 15 码本 logprob") when only
    the semantic head was reconstructed; the all-codebook kernel made γ=1 the
    effective default, and Fish S2 §4.3 validates per-component PG with a
    shared sequence advantage (their Fast AR term). Now explicit — 0.0
    reproduces the original semantic-only behavior.

    Returns (loss, metrics) where loss = policy_loss + β·KL.
    """
    cfg = cfg or GRPOConfig()

    A, gmean, gstd = group_advantage(rewards, cfg.variant, group_ids, cfg.std_eps)
    col_w = _column_weights(mask, cfg)

    if cfg.variant == VARIANT_GSPO:
        policy_loss, info = _gspo_loss(log_probs, ref_log_probs, A, mask, cfg)
    elif cfg.variant in (ADV_DR, ADV_VANILLA):
        policy_loss, info = _clipped_loss(log_probs, ref_log_probs, A, mask, cfg)
    else:
        raise ValueError(f"unknown GRPO variant {cfg.variant!r}")

    kl = (
        kl_divergence(log_probs, ref_log_probs, mask, cfg.kl_estimator, col_w=col_w)
        if cfg.use_kl
        else None
    )
    loss = policy_loss + (
        cfg.kl_beta * kl if kl is not None else torch.zeros_like(policy_loss)
    )
    return loss, GRPOMetrics(
        loss,
        policy_loss,
        kl,
        A,
        gmean,
        gstd,
        rho=info.get("rho"),
        clamped=info.get("clamped"),
    )


def needs_resample(
    sim: torch.Tensor,
    cer: torch.Tensor,
    sv_eps: float = 1e-3,
    wer_eps: float = 1e-3,
) -> bool:
    """Zero-signal group → skip the update (DAPO-style dynamic sampling).

    The only reliable within-group signal is WER spread: MOS is dead by
    design (hinge floor, std≡0 in healthy groups) and SV rank on
    same-policy takes is measurement noise (SV_REWARD_FINDINGS §四). So a
    group trains ONLY when r_wer actually spreads — this covers both
    all-perfect groups (r_wer ≡ 1) and flat-CER groups (8 takes with
    identical cer, observed on the graphed smoke C1v8: std_wer = 0 with
    cer = 0.14, where a single Adam step along the leftover SV-noise
    ranking collapsed the policy into 12-79 s babbling rollouts).
    """
    r_wer = 1.0 - cer
    return bool(r_wer.std(unbiased=False) < wer_eps)
