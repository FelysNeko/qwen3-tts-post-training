"""Multi-objective scorer: SV embedding + ASR transcript + two MOS stages
(UTMOSv2, P.835 DNSMOS), dispatched by the request's four service bools —
only the model groups a caller actually needs are run (and lazy-loaded).
Calibration-free: similarities and CERs are the caller's job."""

from __future__ import annotations

import logging
import time
from functools import cached_property

from qwen3_tts_post_training.client.protocol import ScoreItem, ScoreResult, Timing

logger = logging.getLogger(__name__)


class Scorers:
    def __init__(self, args):
        self.args = args

    @cached_property
    def sv(self):
        from scorer.models.sv import SVScorer

        t0 = time.time()
        scorer = SVScorer(self.args.device)
        logger.info(f"sv loaded in {time.time() - t0:.1f}s")
        return scorer

    @cached_property
    def asr(self):
        from scorer.models.asr import ASRScorer

        t0 = time.time()
        scorer = ASRScorer(self.args.asr_model, self.args.device, self.args.asr_batch)
        logger.info(f"asr loaded in {time.time() - t0:.1f}s")
        return scorer

    @cached_property
    def utmosv2(self):
        from scorer.models.utmosv2 import UTMOSv2Scorer

        t0 = time.time()
        scorer = UTMOSv2Scorer(
            fold=self.args.utmosv2_fold,
            seed=self.args.utmosv2_seed,
            num_repetitions=self.args.utmosv2_reps,
            device=self.args.device,
            gpu_mel=self.args.gpu_mel,
        )
        logger.info(f"utmosv2 loaded in {time.time() - t0:.1f}s")
        return scorer

    @cached_property
    def p835(self):
        from scorer.models.p835 import P835Scorer

        t0 = time.time()
        scorer = P835Scorer(workers=self.args.p835_workers)
        logger.info(f"p835 loaded in {time.time() - t0:.1f}s")
        return scorer

    def score(
        self,
        items: list[ScoreItem],
        asr: bool = False,
        utmosv2: bool = False,
        p835: bool = False,
        sv: bool = False,
    ) -> tuple[list[ScoreResult], Timing]:
        results = [ScoreResult(wav_path=item.wav_path) for item in items]
        t_sv = t_asr = t_utmosv2 = t_p835 = 0.0

        if sv:
            t0 = time.time()
            embeddings = [
                self.sv.embed_wav(item.wav_path, "eres2netv2") for item in items
            ]
            t_sv = time.time() - t0
            for result, embedding in zip(results, embeddings):
                result.embedding = embedding.tolist()

        if asr:
            t0 = time.time()
            transcripts = self.asr.transcribe([item.wav_path for item in items])
            t_asr = time.time() - t0
            for result, item in zip(results, items):
                result.transcript = transcripts[item.wav_path]

        if utmosv2:
            t0 = time.time()
            utmosv2_scores = self.utmosv2.score([item.wav_path for item in items])
            t_utmosv2 = time.time() - t0
            for result, score in zip(results, utmosv2_scores):
                result.utmosv2 = score

        if p835:
            t0 = time.time()
            p835_scores = self.p835.score([item.wav_path for item in items])
            t_p835 = time.time() - t0
            for result, score in zip(results, p835_scores):
                result.p835 = score

        timing = Timing(
            sv=round(t_sv, 2),
            asr=round(t_asr, 2),
            utmosv2=round(t_utmosv2, 2),
            p835=round(t_p835, 2),
        )
        return results, timing
