"""metrics.json (preprocess output, §16) → reward-side calibration.

The preprocess worker computes the SV centroid and the sim distribution of
the actual training corpus; these two functions are the ONLY consumers the
trainer/scorer need, replacing the playground npy path and the hardcoded
0.8585/0.0966 pair.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qwen3_tts_post_training.reward.reward import RewardConfig


def reward_config_from_metrics(path: str | Path) -> RewardConfig:
    """sv_center/sv_scale from the corpus sim distribution (mean/std of the
    per-clip cosine to the centroid). mos_tau stays at the current 2.5: the
    utmosv2 stats in metrics.json are informational until a gate rule is
    decided (STATUS.md §16)."""
    with open(path, encoding="utf-8") as f:
        metrics = json.load(f)
    return RewardConfig(
        sv_center=metrics["sim"]["mean"],
        sv_scale=metrics["sim"]["std"],
    )


def load_centroid(metrics_path: str | Path) -> np.ndarray:
    """Unit-norm ERes2NetV2 centroid — what the scorer's `--metrics` mode
    installs as the SV reference. Stored as centroid.npy (np.save, float64)
    beside metrics.json — the only artifact that predates the per-stage npy
    convention was a 192-float JSON list; same layout now as
    codes/embedding."""
    return np.load(Path(metrics_path).parent / "centroid.npy")
