"""Dataset side of UTMOSv2 inference — verbatim numeric/RNG paths from
UTMOSv2/utmosv2/dataset/{_utils,multi_spec,ssl,ssl_multispec,_base}.py @ cc2700db.

RNG call order is the determinism anchor (np.random.seed(42) + num_workers=0):
per item per repetition — 1 ssl crop, then per spec frame: 1 crop, then per
spec config: mel, 1 mixup crop, 1 beta draw. DO NOT reorder.
"""

from __future__ import annotations

import librosa
import numpy as np
import torch

from scorer.utmos.config import DATASET_MAP


def load_audio(sr_target: int, file) -> np.ndarray:
    try:
        y, sr = librosa.load(file, sr=None)
        y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
    except Exception:  # noqa: BLE001 — audio loader fallback (npy / odd files)
        y = np.load(file)
    return y


def remove_silent_section(audio: np.ndarray, min_length: int = 4800) -> np.ndarray:
    mask = audio < 0.1
    mask = np.pad(mask, (1, 0)) ^ np.pad(mask, (0, 1))
    indices = np.where(mask)[0]
    length = indices[1::2] - indices[::2]
    indices_mask = np.repeat(length > min_length, 2)
    indices = indices[indices_mask]
    mask2 = np.zeros(audio.shape[0] + 1, dtype=int)
    mask2[indices] = np.where(np.arange(indices.shape[0]) % 2, -1, 1)
    mask2 = np.cumsum(mask2).astype(bool)[:-1]
    return audio[~mask2]


def extend_audio(y: np.ndarray, length: int, method: str) -> np.ndarray:
    if y.shape[0] > length:
        return y
    elif method == "tile":
        n = length // y.shape[0] + 1
        return np.tile(y, n)
    else:
        raise NotImplementedError


def select_random_start(y: np.ndarray, length: int) -> np.ndarray:
    start = np.random.randint(0, y.shape[0] - length)
    return y[start : start + length]


def _make_melspec(cfg, spec_cfg, y: np.ndarray) -> np.ndarray:
    spec = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sr,
        n_fft=spec_cfg.n_fft,
        hop_length=spec_cfg.hop_length,
        n_mels=spec_cfg.n_mels,
        win_length=spec_cfg.win_length,
    )
    spec = librosa.power_to_db(spec, ref=np.max)
    if spec_cfg.norm is not None:
        spec = (spec + spec_cfg.norm) / spec_cfg.norm
    return spec


class UTMOSSample:
    """Per-item feature builder: (ssl_wave, spec_stack, domain_onehot)."""

    def __init__(self, cfg, dataset_idx: int):
        self.cfg = cfg
        self.dataset_idx = dataset_idx

    def build(self, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        if cfg.dataset.remove_silent_section:
            y = remove_silent_section(y)

        # --- ssl side (wav2vec2 input): single 3s random crop ---
        length = int(cfg.dataset.ssl.duration * cfg.sr)
        y_ssl = extend_audio(y, length, method="tile")
        y_ssl = select_random_start(y_ssl, length)
        x1 = torch.from_numpy(y_ssl)

        # --- spec side: num_frames crops × specs, inner mixup ---
        specs = []
        length = int(cfg.dataset.spec_frames.frame_sec * cfg.sr)
        y = extend_audio(y, length, method=cfg.dataset.spec_frames.extend)
        for _ in range(cfg.dataset.spec_frames.num_frames):
            y1 = select_random_start(y, length)
            for spec_cfg in cfg.dataset.specs:
                spec = _make_melspec(cfg, spec_cfg, y1)
                if cfg.dataset.spec_frames.mixup_inner:
                    y2 = select_random_start(y, length)
                    spec2 = _make_melspec(cfg, spec_cfg, y2)
                    lmd = np.random.beta(
                        cfg.dataset.spec_frames.mixup_alpha,
                        cfg.dataset.spec_frames.mixup_alpha,
                    )
                    spec = lmd * spec + (1 - lmd) * spec2
                spec = np.stack([spec, spec, spec], axis=0)
                spec_tensor = torch.tensor(spec, dtype=torch.float32)
                spec_tensor = cfg.transform["valid"](spec_tensor)
                specs.append(spec_tensor)
        x2 = torch.stack(specs).float()

        d = torch.zeros(len(DATASET_MAP), dtype=torch.float32)
        d[self.dataset_idx] = 1.0
        return x1, x2, d
