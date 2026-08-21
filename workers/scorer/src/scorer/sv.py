"""SV scoring: ERes2NetV2 (reward) + CAM++ (cross-monitor), vendored 3D-Speaker
model defs + FBank frontend. Matches playground/compare_tts_sv.py exactly
(same ckpt args, same 16k fbank mean-nor, same unit-norm cosine)."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
import torchaudio.compliance.kaldi as Kaldi

TARGET_SR = 16000

MODELS = {
    "eres2netv2": {
        "obj": "scorer.speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2",
        "args": {
            "feat_dim": 80,
            "embedding_size": 192,
            "baseWidth": 24,
            "scale": 4,
            "expansion": 4,
        },
        "ckpt": "pretrained/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common/pretrained_eres2netv2w24s4ep4.ckpt",
    },
    "campplus": {
        "obj": "scorer.speakerlab.models.campplus.DTDNN.CAMPPlus",
        "args": {"feat_dim": 80, "embedding_size": 192},
        "ckpt": "pretrained/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin",
    },
}


class FBank:
    def __init__(self, n_mels: int, sample_rate: int, mean_nor: bool = False):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

    def __call__(self, wav: torch.Tensor, dither: int = 0) -> torch.Tensor:
        sr = 16000
        assert sr == self.sample_rate
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[0] > 1:
            wav = wav[0, :].unsqueeze(0)
        assert len(wav.shape) == 2 and wav.shape[0] == 1
        feat = Kaldi.fbank(
            wav, num_mel_bins=self.n_mels, sample_frequency=sr, dither=dither
        )
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat


class SVScorer:
    def __init__(self, sv_dir: Path, device: str):
        self.sv_dir = Path(sv_dir)
        self.device = device
        self.fe = FBank(80, sample_rate=TARGET_SR, mean_nor=True)
        self._models: dict[str, tuple[torch.nn.Module, str]] = {}
        self._refs: dict[str, np.ndarray | None] = {}

    def _load(self, name: str) -> tuple[torch.nn.Module, str]:
        if name not in self._models:
            conf = MODELS[name]
            module, cls = conf["obj"].rsplit(".", 1)
            model = getattr(importlib.import_module(module), cls)(**conf["args"])
            model.load_state_dict(
                torch.load(self.sv_dir / conf["ckpt"], map_location="cpu")
            )
            model.to(self.device).eval()
            self._models[name] = (model, self.device)
        return self._models[name]

    def set_ref(self, name: str, ref_npy: Path) -> None:
        ref = np.load(ref_npy).astype(np.float32)
        ref = ref / np.linalg.norm(ref)
        self._refs[name] = ref

    @torch.no_grad()
    def embed(self, audio: np.ndarray, sr: int, name: str = "eres2netv2") -> np.ndarray:
        model, device = self._load(name)
        t = torch.from_numpy(audio)
        if sr != TARGET_SR:
            t = AF.resample(t, sr, TARGET_SR)
        feat = self.fe(t.to(device))
        emb = model(feat.unsqueeze(0)).squeeze(0).cpu().numpy().astype(np.float32)
        return emb / np.linalg.norm(emb)

    def score(self, wav_path: str, name: str = "eres2netv2") -> float:
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        emb = self.embed(audio.mean(axis=1), sr, name)
        ref = self._refs.get(name)
        if ref is None:
            raise RuntimeError(
                f"reference embedding for '{name}' not set (set_ref first)"
            )
        return float(emb @ ref)
