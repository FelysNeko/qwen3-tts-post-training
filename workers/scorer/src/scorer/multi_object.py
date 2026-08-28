"""Multi-objective scorer: SV embedding + ASR/CER + MOS, dispatched by the
request's ScoreField set — only the model groups a caller actually needs are
run (and lazy-loaded). Calibration-free: similarities are the caller's job."""

from __future__ import annotations

import time
from pathlib import Path

from qwen3_tts_post_training.client.protocol import (
    ScoreField,
    ScoreItem,
    ScoreResult,
    Timing,
)


class Scorers:
    def __init__(self, args):
        self.args = args
        self._sv = None
        self._asr = None
        self._mos = None

    @property
    def sv(self):
        if self._sv is None:
            from scorer.sv import SVScorer

            t0 = time.time()
            self._sv = SVScorer(Path(self.args.sv_dir), self.args.device)
            print(f"[load] sv {time.time() - t0:.1f}s")
        return self._sv

    @property
    def asr(self):
        if self._asr is None:
            from scorer.asr import ASRScorer

            t0 = time.time()
            self._asr = ASRScorer(
                self.args.asr_model, self.args.device, self.args.asr_batch
            )
            print(f"[load] asr {time.time() - t0:.1f}s")
        return self._asr

    @property
    def mos(self):
        if self._mos is None:
            from scorer.mos import MOSScorer

            t0 = time.time()
            self._mos = MOSScorer(
                fold=self.args.mos_fold,
                seed=self.args.mos_seed,
                num_repetitions=self.args.mos_reps,
                device=self.args.device,
                gpu_mel=self.args.gpu_mel,
            )
            print(f"[load] mos {time.time() - t0:.1f}s")
        return self._mos

    def score(
        self, items: list[ScoreItem], fields: frozenset[ScoreField]
    ) -> tuple[list[ScoreResult], Timing]:
        results = [ScoreResult(wav_path=item.wav_path) for item in items]
        t_sv = t_asr = t_mos = 0.0

        if ScoreField.VECTOR in fields:
            t0 = time.time()
            embeddings = [
                self.sv.embed_wav(item.wav_path, "eres2netv2") for item in items
            ]
            t_sv = time.time() - t0
            for result, embedding in zip(results, embeddings):
                result.vector = embedding.tolist()

        if fields & {ScoreField.TRANSCRIPT, ScoreField.CER}:
            t0 = time.time()
            got = self.asr.score(
                [item.wav_path for item in items], [item.text for item in items]
            )
            t_asr = time.time() - t0
            for result, row in zip(results, got):
                result.transcript = row["transcript"]
                result.cer = row["cer"]

        if ScoreField.MOS in fields:
            t0 = time.time()
            mos_scores = self.mos.score([item.wav_path for item in items])
            t_mos = time.time() - t0
            for result, mos in zip(results, mos_scores):
                result.mos = mos

        timing = Timing(sv=round(t_sv, 2), asr=round(t_asr, 2), mos=round(t_mos, 2))
        return results, timing
