"""UTMOSv2 inference glue — replaces upstream create_model + UTMOSv2Model.predict.

Determinism contract (validated): np.random.seed(fixed) before each
repetition loop + sequential in-order item processing (no fork workers).
Weights are fetched (once, via HF cache) from the official
sarulab-speech/UTMOSv2 repo by scorer.fetch.ensure_utmos."""

from __future__ import annotations

import numpy as np
import torch

from scorer.fetch import ensure_utmos
from scorer.utmos.config import DATASET_MAP, build_cfg
from scorer.utmos.dataset import GPUSpecBuilder, UTMOSSample, load_audio
from scorer.utmos.model import SSLMultiSpecExtModelV2


class UTMOS:
    def __init__(
        self,
        fold: int = 0,
        seed: int = 42,
        device: str = "cuda:0",
        gpu_mel: bool = True,
    ):
        self.cfg = build_cfg()
        self.device = device
        self.model = SSLMultiSpecExtModelV2(self.cfg)
        ckpt = ensure_utmos(fold, seed)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.eval().to(device)
        # domain one-hot index: upstream predict() default predict_dataset="sarulab"
        self.dataset_idx = DATASET_MAP["sarulab"]
        self.gpu_builder = GPUSpecBuilder(self.cfg, device) if gpu_mel else None

    def _forward_batch(
        self,
        feats: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> np.ndarray:
        x1 = torch.stack([f[0] for f in feats]).to(self.device, non_blocking=True)
        x2 = torch.stack([f[1] for f in feats]).to(self.device, non_blocking=True)
        d = torch.stack([f[2] for f in feats]).to(self.device, non_blocking=True)
        with torch.no_grad(), torch.cuda.amp.autocast():
            out = self.model(x1, x2, d).squeeze(1)
        return out.cpu().numpy()

    def predict(
        self,
        wavs: list[str],
        num_repetitions: int = 8,
        batch_size: int = 8,
        seed: int = 42,
    ) -> list[float]:
        """MOS per wav, input order preserved."""
        sample = UTMOSSample(self.cfg, self.dataset_idx, gpu_builder=self.gpu_builder)
        audios = [load_audio(self.cfg.sr, w) for w in wavs]
        # res = 0.0 (python float) is deliberate: numpy's weak-scalar promotion
        # keeps the accumulation in float16, matching upstream _predict_impl's
        # `res += np.concatenate(pred) / num_repetitions` bit-for-bit.
        res = 0.0
        np.random.seed(seed)  # seed ONCE; reps consume the stream continuously
        for rep in range(num_repetitions):
            preds = []
            for i in range(0, len(wavs), batch_size):
                feats = [
                    sample.build(audios[j])
                    for j in range(i, min(i + batch_size, len(wavs)))
                ]
                preds.append(self._forward_batch(feats))
            res = res + np.concatenate(preds) / num_repetitions
        return [float(v) for v in res]
