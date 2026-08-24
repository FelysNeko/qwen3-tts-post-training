"""Reward v3 (design truth source: playground/SV_REWARD_FINDINGS.md §四/§七).

    R = λ_sv·r_sv/std(r_sv) + λ_wer·r_wer/std(r_wer) + λ_mos·r_mos/max(std(r_mos), eps)

- r_sv  = sigmoid((sim_e2v2 − 0.8585)/0.0966)      (E2V2 speaker sim, unit-normalized)
- r_wer = 1 − CER_qwen3asr                          (normalize() + edit-distance CER)
- r_mos = max(0, 2.5 − mos_utmosv2fold0)            (hinge 护栏, 线性地板: 只挡不驱动)
- std is the *within-group* std of each component (batch-std layer; per MD the
  std≈0 in all-above-τ groups). Every term 熄火 (zeroed) when its group std
  drops below its flameout eps (MD: 组内 std<eps 熄火) — MOS by construction in
  healthy groups, SV/WER in degenerate groups (e.g. all-perfect transcripts);
  otherwise a dead group would divide by eps and inject a 1e6 sentinel.
- λ = (1.0, 1.0, 0.2) — v3 定稿.

r_mos floor (2026-08-23, UTMOS-replacement A/B conclusion): a sigmoid is still
sloped inside the healthy zone (mos 2.7-3.3 → r_mos 0.73-0.98), so healthy
groups leak a within-group std ≈ 0.09 into the advantage — pure ranking noise
(empirically UTMOS invents ~1.3σ within-group spread on indistinguishable good
takes). max(0, τ−mos) is exactly 0 above τ, so a healthy group has r_mos ≡ 0 →
std = 0 *by construction* and the flameout is guaranteed, not a threshold
gamble; the continuous linear penalty lives only where it is reliable (below τ).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch


@dataclass
class RewardConfig:
    sv_center: float = 0.8585
    sv_scale: float = 0.0966
    mos_tau: float = 2.5
    mos_scale: float = 0.2
    lam_sv: float = 1.0
    lam_wer: float = 1.0
    lam_mos: float = 0.2
    std_eps: float = 1e-6
    flameout_eps: float = 1e-3
    mos_flameout_eps: float = 1e-4
    mos_flameout: bool = True


class RewardBreakdown(NamedTuple):
    r_sv: torch.Tensor
    r_wer: torch.Tensor
    r_mos: torch.Tensor
    std_sv: torch.Tensor
    std_wer: torch.Tensor
    std_mos: torch.Tensor
    R: torch.Tensor


def r_sv_fn(sim: torch.Tensor, cfg: RewardConfig) -> torch.Tensor:
    return torch.sigmoid((sim - cfg.sv_center) / cfg.sv_scale)


def r_wer_fn(cer: torch.Tensor, cfg: RewardConfig) -> torch.Tensor:
    return 1.0 - cer


def r_mos_fn(mos: torch.Tensor, cfg: RewardConfig) -> torch.Tensor:
    """Hinge floor: 0 above τ (healthy, strictly silent), linear penalty below.

    Healthy groups get r_mos ≡ 0, so their within-group std is 0 by
    construction and the flameout fires deterministically — no threshold gamble
    on a noisy std estimate. Penalty slope is 1 per MOS unit below τ.
    """
    return torch.clamp(cfg.mos_tau - mos, min=0.0)


def reward_v3(
    sim: torch.Tensor,
    cer: torch.Tensor,
    mos: torch.Tensor,
    cfg: RewardConfig | None = None,
    group_dim: int = -1,
) -> tuple[torch.Tensor, RewardBreakdown]:
    """v3.1 composite reward: RAW component magnitudes, no within-group std
    division. Dr.GRPO subtracts the group mean in the advantage, so flat
    groups already produce near-zero A on their own; the old r/std(r)
    standardization (a) amplified pure ranking noise to full scale on flat
    groups (one Adam step along it collapsed the policy — smoke C1v8/C1v9)
    and (b) was philosophically at odds with Dr.GRPO's "keep magnitude
    information" design. The std flameout survives only as a signal-health
    indicator (breakdown + needs_resample); a degenerate component (std <
    flameout_eps) is zeroed so it cannot leak a constant offset either.

    Component scales (per take): r_sv ∈ (0,1) sigmoid; r_wer ∈ [0,1];
    r_mos = max(0, τ−mos), linear penalty in MOS units, λ_mos=0.2.
    """
    cfg = cfg or RewardConfig()

    r_sv = r_sv_fn(sim, cfg)
    r_wer = r_wer_fn(cer, cfg)
    r_mos = r_mos_fn(mos, cfg)

    std_sv = r_sv.std(dim=group_dim, unbiased=False, keepdim=True)
    std_wer = r_wer.std(dim=group_dim, unbiased=False, keepdim=True)
    std_mos = r_mos.std(dim=group_dim, unbiased=False, keepdim=True)

    # dead components contribute nothing (constant offsets are erased by the
    # group-mean subtraction anyway; this keeps R itself interpretable)
    zeros = torch.zeros_like(r_sv)
    term_sv = torch.where(std_sv < cfg.flameout_eps, zeros, cfg.lam_sv * r_sv)
    term_wer = torch.where(
        std_wer < cfg.flameout_eps, zeros, cfg.lam_wer * r_wer
    )
    if cfg.mos_flameout:
        term_mos = torch.where(
            std_mos < cfg.mos_flameout_eps, zeros, cfg.lam_mos * r_mos
        )
    else:
        term_mos = cfg.lam_mos * r_mos

    R = term_sv + term_wer + term_mos
    return R, RewardBreakdown(r_sv, r_wer, r_mos, std_sv, std_wer, std_mos, R)
