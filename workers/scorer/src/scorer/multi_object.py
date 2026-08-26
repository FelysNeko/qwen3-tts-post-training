"""Multi-objective scorer: SV + ASR/CER + MOS batch scoring."""

from __future__ import annotations

import time
from pathlib import Path

from qwen3_tts_post_training.client.protocol import ScoreItem, ScoreResult, Timing


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
            s = SVScorer(Path(self.args.sv_dir), self.args.device)
            if self.args.sv_ref:
                s.set_ref("eres2netv2", Path(self.args.sv_ref))
            if self.args.sv_ref_camp:
                s.set_ref("campplus", Path(self.args.sv_ref_camp))
            self._sv = s
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

    def score(self, items: list[ScoreItem]) -> tuple[list[ScoreResult], Timing]:
        t0 = time.time()
        sims: list[float] = []
        sim_camps: list[float] = []
        for it in items:
            sim = self.sv.score(it.wav_path, "eres2netv2")
            sims.append(sim)
            if self.args.sv_ref_camp:
                sim_camps.append(self.sv.score(it.wav_path, "campplus"))
            else:
                sim_camps.append(sim)
        t_sv = time.time() - t0

        t0 = time.time()
        got = self.asr.score([it.wav_path for it in items], [it.text for it in items])
        t_asr = time.time() - t0

        t0 = time.time()
        mos_scores = self.mos.score([it.wav_path for it in items])
        t_mos = time.time() - t0

        results = [
            ScoreResult(
                wav_path=it.wav_path,
                sim=sims[i],
                sim_camp=sim_camps[i],
                transcript=got[i]["transcript"],
                cer=got[i]["cer"],
                mos=mos_scores[i],
            )
            for i, it in enumerate(items)
        ]
        timing = Timing(sv=round(t_sv, 2), asr=round(t_asr, 2), mos=round(t_mos, 2))
        return results, timing
