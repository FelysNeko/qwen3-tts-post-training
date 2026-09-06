"""UTMOSv2 scoring: vendored fusion_stage3 fold0, deterministic mode.

Randomness trap (MD §4 bake-off): the dataset does random 1.4s crops ×2 +
same-file mixup per prediction — single calls drift up to Δ0.65 MOS.
Fix (validated, this repo): np.random.seed(fixed) once before the repetition
loop + num_repetitions=8 averaged + sequential item processing (no forked
workers). num_workers=8/4 broke reproducibility regardless of seeding;
num_workers=0 is bit-identical. Bit-stability holds WITHIN one scorer process
only — a restart may shift MOS by ~O(0.1) (§16.9).

The vendored scorer (vendor/utmos/) is bit-exact with upstream UTMOSv2 @
cc2700db: identical weights, inputs, forward, and float16 weak-scalar
accumulation. First-load weight fetch delegates to huggingface_hub from the
official `sarulab-speech/UTMOSv2` repo: `ensure_utmos` (was scorer/fetch.py,
absorbed 2026-09-05).
"""

from __future__ import annotations

import logging
from pathlib import Path

from scorer.vendor.utmos.utmos import UTMOS

logger = logging.getLogger(__name__)

UTMOS_HF_REPO = "sarulab-speech/UTMOSv2"


def ensure_utmos(fold: int, seed: int) -> Path:
    """Resolve (downloading once via HF cache if needed) the UTMOSv2 fold
    weights. Returns the HF-managed cache path."""
    from huggingface_hub import hf_hub_download

    logger.info(f"fetching UTMOSv2 fold{fold} via HF ({UTMOS_HF_REPO})")
    local = hf_hub_download(
        repo_id=UTMOS_HF_REPO,
        filename=f"fold{fold}_s{seed}_best_model.pth",
    )
    logger.info(f"UTMOSv2 fold{fold} ready: {local}")
    return Path(local)


class UTMOSv2Scorer:
    def __init__(
        self,
        fold: int = 0,
        seed: int = 42,
        num_repetitions: int = 8,
        device: str = "cuda:0",
        gpu_mel: bool = True,
    ):
        self.reps = num_repetitions
        self.seed = seed
        self.model = UTMOS(
            ckpt=ensure_utmos(fold, seed), device=device, gpu_mel=gpu_mel
        )

    def score(self, wavs: list[str], chunk: int = 32) -> list[float]:
        """MOS per wav, input order preserved."""
        scores = [0.0] * len(wavs)
        for i in range(0, len(wavs), chunk):
            scores[i : i + chunk] = self.model.predict(
                wavs[i : i + chunk],
                num_repetitions=self.reps,
                batch_size=8,
                seed=self.seed,
            )
        return scores
