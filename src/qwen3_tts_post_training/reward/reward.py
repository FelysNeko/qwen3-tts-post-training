"""Reward v3 (design truth source: playground/SV_REWARD_FINDINGS.md §四/§七).

    R = λ_sv·r_sv/std(r_sv) + λ_wer·r_wer/std(r_wer) + λ_mos·r_mos/max(std(r_mos), eps)

- r_sv  = sigmoid((sim_e2v2 − 0.8585)/0.0966)      (E2V2 speaker sim, unit-normalized)
- r_wer = 1 − CER_qwen3asr                          (normalize() + edit-distance CER)
- r_mos = max(0, 2.5 − mos_utmosv2fold0)            (hinge 护栏, 线性地板: 只挡不驱动)
- std is the *within-group* std of each component (batch-std layer; per MD the
  std≈0 in all-above-τ groups). The MOS term 熄火 (zeroed) when its group std
  drops below mos_flameout_eps (MD: 组内 std<eps 熄火). SV/WER keep the
  max(std, eps) guard so they never divide by zero.
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
    """v3 composite reward. Each component is normalized by its within-group
    std; the MOS term 熄火 (zeroed) when its group std < mos_flameout_eps.

    Returns (R, breakdown) with R of the same shape as the inputs (grouped
    along group_dim).
    """
    cfg = cfg or RewardConfig()

    r_sv = r_sv_fn(sim, cfg)
    r_wer = r_wer_fn(cer, cfg)
    r_mos = r_mos_fn(mos, cfg)

    std_sv = r_sv.std(dim=group_dim, unbiased=False, keepdim=True)
    std_wer = r_wer.std(dim=group_dim, unbiased=False, keepdim=True)
    std_mos = r_mos.std(dim=group_dim, unbiased=False, keepdim=True)

    term_sv = cfg.lam_sv * r_sv / std_sv.clamp_min(cfg.std_eps)
    term_wer = cfg.lam_wer * r_wer / std_wer.clamp_min(cfg.std_eps)

    if cfg.mos_flameout:
        dead = std_mos < cfg.mos_flameout_eps
        term_mos = torch.where(
            dead, torch.zeros_like(r_mos), r_mos
        ) / std_mos.clamp_min(cfg.std_eps)
    else:
        term_mos = r_mos / std_mos.clamp_min(cfg.std_eps)
    term_mos = cfg.lam_mos * term_mos

    R = term_sv + term_wer + term_mos
    return R, RewardBreakdown(r_sv, r_wer, r_mos, std_sv, std_wer, std_mos, R)
