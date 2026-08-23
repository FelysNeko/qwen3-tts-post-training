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
    """Per-item feature builder: (ssl_wave, spec_stack, domain_onehot).

    Pass `gpu_builder` (a GPUSpecBuilder) to run the melspectrogram stack on
    GPU. The RNG call sequence is identical either way — only the mel compute
    moves off librosa/CPU — so the determinism contract is preserved.
    """

    def __init__(self, cfg, dataset_idx: int, gpu_builder: GPUSpecBuilder | None = None):
        self.cfg = cfg
        self.dataset_idx = dataset_idx
        self.gpu_builder = gpu_builder

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
            for sc_idx, spec_cfg in enumerate(cfg.dataset.specs):
                if self.gpu_builder is not None:
                    spec = self.gpu_builder.melspec(sc_idx, y1)
                else:
                    spec = _make_melspec(cfg, spec_cfg, y1)
                    spec = torch.from_numpy(np.asarray(spec, dtype=np.float32))
                if cfg.dataset.spec_frames.mixup_inner:
                    y2 = select_random_start(y, length)
                    if self.gpu_builder is not None:
                        spec2 = self.gpu_builder.melspec(sc_idx, y2)
                    else:
                        spec2 = _make_melspec(cfg, spec_cfg, y2)
                        spec2 = torch.from_numpy(np.asarray(spec2, dtype=np.float32))
                    lmd = float(
                        np.random.beta(
                            cfg.dataset.spec_frames.mixup_alpha,
                            cfg.dataset.spec_frames.mixup_alpha,
                        )
                    )
                    spec = lmd * spec + (1 - lmd) * spec2
                spec = torch.stack([spec, spec, spec], dim=0)
                spec_tensor = cfg.transform["valid"](spec)
                specs.append(spec_tensor)
        x2 = torch.stack(specs).float()

        d = torch.zeros(len(DATASET_MAP), dtype=torch.float32)
        d[self.dataset_idx] = 1.0
        return x1, x2, d


class GPUSpecBuilder:
    """librosa-mel-equivalent spectrogram builder running on torch (GPU).

    The mel filterbank matrix (librosa.filters.mel, default norm='slaney') and
    the padded hann windows are precomputed once on CPU then moved to device;
    the per-crop work is torch.stft + filterbank matmul + power_to_db + norm.
    Crop selection stays in numpy (identical RNG order), so determinism holds
    (torch.stft/cufft and torchvision Resize are deterministic).
    """

    def __init__(self, cfg, device: str):
        self.cfg = cfg
        self.device = torch.device(device)
        self.mel_fbanks: list[torch.Tensor] = []
        self.windows: list[torch.Tensor] = []
        for sc in cfg.dataset.specs:
            fb = librosa.filters.mel(sr=cfg.sr, n_fft=sc.n_fft, n_mels=sc.n_mels)
            self.mel_fbanks.append(
                torch.from_numpy(np.asarray(fb, dtype=np.float32)).to(self.device)
            )
            # torch.stft expects a win_length-sized window and pads it to n_fft
            # internally (same centering as librosa's pad_center).
            self.windows.append(
                torch.hann_window(sc.win_length, periodic=True, device=self.device)
            )

    def melspec(self, sc_idx: int, y: np.ndarray) -> torch.Tensor:
        sc = self.cfg.dataset.specs[sc_idx]
        t = torch.from_numpy(y).to(self.device, non_blocking=True)
        spec = torch.stft(
            t,
            n_fft=sc.n_fft,
            hop_length=sc.hop_length,
            win_length=sc.win_length,
            window=self.windows[sc_idx],
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        mag = spec.abs().pow(2)
        mel = self.mel_fbanks[sc_idx] @ mag
        ref = mel.max().clamp_min(1e-10)
        db = 10.0 * (torch.log10(mel.clamp_min(1e-10)) - torch.log10(ref))
        # librosa.power_to_db default top_db=80.0: floor at max - top_db
        db = torch.maximum(db, db.max() - 80.0)
        if sc.norm is not None:
            db = (db + sc.norm) / sc.norm
        return db
