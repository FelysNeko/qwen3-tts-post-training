"""Semantic code groups → 24 kHz audio (speech_tokenizer, the frozen
environment renderer) + PCM wav writer for the scorer to read by path."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import torch

from trainer.model import TrainerModel


def write_wav(path: str | Path, audio: np.ndarray, sr: int) -> Path:
    path = Path(path)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return path


class Decoder:
    def __init__(self, ttm: TrainerModel):
        self.ttm = ttm

    def decode(self, codes: list[torch.Tensor]) -> tuple[list[np.ndarray], int]:
        """codes: list of [T, num_code_groups]. Returns (wavs, sample_rate)."""
        wavs, fs = self.ttm.model.speech_tokenizer.decode(
            [{"audio_codes": c} for c in codes]
        )
        return wavs, fs
