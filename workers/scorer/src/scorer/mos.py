"""MOS scoring: vendored UTMOSv2 fusion_stage3 fold0, deterministic mode.

Randomness trap (MD §4 bake-off): the dataset does random 1.4s crops ×2 +
same-file mixup per prediction — single calls drift up to Δ0.65 MOS.
Fix (validated, this repo): np.random.seed(fixed) once before the repetition
loop + num_repetitions=8 averaged + sequential item processing (no forked
workers). num_workers=8/4 broke reproducibility regardless of seeding;
num_workers=0 is bit-identical.

The vendored scorer (utmos/) is bit-exact with upstream UTMOSv2 @ cc2700db:
identical weights, inputs, forward, and float16 weak-scalar accumulation.
"""

from __future__ import annotations

from scorer.utmos import UTMOS


class MOSScorer:
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
        self.model = UTMOS(fold=fold, seed=seed, device=device, gpu_mel=gpu_mel)

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
