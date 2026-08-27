"""Decode driver for the vendored MossFormer2_SE_48K path.

`_stft/_istft/_compute_fbank` are ported verbatim from
clearvoice/clearvoice/utils/misc.py (args -> cfg renamed); the decode loop
follows clearvoice/clearvoice/utils/decode_batch.py::
decode_one_audio_mossformer2_se_48k's batched path, which is the ONLY path
the pip package can ever take for tensor-mode input: its `one_time_decode_
length` guard compares `inputs.shape[0]` — the BATCH dimension — against
20s*48kHz, so for [1, T] mono input the sliding-window branch is unreachable
(1 > 960000 is always False; the branch itself also crashes on a 2-D tensor
via a 3-index write). We reproduce the effective behavior: every clip is
decoded in ONE batched pass (fbank+Δ+ΔΔ -> model -> mask*STFT -> iSTFT),
whatever its length. Corpus max is 26s — attention over ~2600 fbank frames
is trivial; feeds much longer than that would need the (dead upstream)
windowing reimplemented.

Numerical note: `_compute_fbank` uses kaldi `dither=1.0` — the fbank (hence
the output) is seeded-RNG-dependent by design; seed torch before calling for
reproducibility. Given the same seed and weights the vendored path is
BIT-IDENTICAL to the pip package (probes/probe_clearvoice_ab.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torchaudio

MAX_WAV_VALUE = 32768.0


@dataclass(frozen=True)
class MossFormer2SE48KConfig:
    """Mirrors clearvoice/clearvoice/config/inference/MossFormer2_SE_48K.yaml
    (only the fields the SE decode path reads; one_time_decode_length /
    decode_window belong to the unreachable windowed branch and are kept for
    documentation)."""

    sampling_rate: int = 48000
    win_type: str = "hamming"
    win_len: int = 1920
    win_inc: int = 384
    fft_len: int = 1920
    num_mels: int = 60
    one_time_decode_length: float = 20.0  # dead upstream (batch-dim bug)
    decode_window: float = 4.0  # dead upstream


def _window(
    cfg: MossFormer2SE48KConfig, length: int, periodic: bool, device
) -> torch.Tensor:
    if cfg.win_type == "hamming":
        return torch.hamming_window(length, periodic=periodic).to(device)
    if cfg.win_type == "hanning":
        return torch.hann_window(length, periodic=periodic).to(device)
    raise ValueError(f"unsupported win_type {cfg.win_type!r}")


def _stft(x: torch.Tensor, cfg: MossFormer2SE48KConfig) -> torch.Tensor:
    win = _window(cfg, cfg.win_len, periodic=False, device=x.device)
    return torch.stft(
        x, cfg.fft_len, cfg.win_inc, cfg.win_len, center=False, window=win,
        onesided=None, return_complex=False,
    )


def _istft(
    x: torch.Tensor, cfg: MossFormer2SE48KConfig, slen: int | None = None
) -> torch.Tensor:
    win = _window(cfg, cfg.win_len, periodic=False, device=x.device)
    try:
        return torch.istft(
            x, n_fft=cfg.fft_len, hop_length=cfg.win_inc, win_length=cfg.win_len,
            window=win, center=False, normalized=False, onesided=None,
            length=slen, return_complex=False,
        )
    except RuntimeError:
        return torch.istft(
            torch.view_as_complex(x), n_fft=cfg.fft_len, hop_length=cfg.win_inc,
            win_length=cfg.win_len, window=win, center=False, normalized=False,
            onesided=None, length=slen, return_complex=False,
        )


def _compute_fbank(
    audio_in: torch.Tensor, cfg: MossFormer2SE48KConfig
) -> torch.Tensor:
    frame_length = cfg.win_len / cfg.sampling_rate * 1000
    frame_shift = cfg.win_inc / cfg.sampling_rate * 1000
    return torchaudio.compliance.kaldi.fbank(
        audio_in,
        dither=1.0,
        frame_length=frame_length,
        frame_shift=frame_shift,
        num_mel_bins=cfg.num_mels,
        sample_frequency=cfg.sampling_rate,
        window_type=cfg.win_type,
    )


def _fbank_180(model_input: torch.Tensor, cfg: MossFormer2SE48KConfig) -> torch.Tensor:
    """[1, T] audio -> [1, T', 180] fbank + delta + delta-delta (the model's
    180 input channels)."""
    fbanks = _compute_fbank(model_input, cfg)
    fbank_tr = torch.transpose(fbanks, 0, 1)
    fbank_delta = torchaudio.functional.compute_deltas(fbank_tr)
    fbank_delta_delta = torchaudio.functional.compute_deltas(fbank_delta)
    fbank_delta = torch.transpose(fbank_delta, 0, 1)
    fbank_delta_delta = torch.transpose(fbank_delta_delta, 0, 1)
    return torch.cat([fbanks, fbank_delta, fbank_delta_delta], dim=1).unsqueeze(0)


def _masked_istft(
    segment: torch.Tensor, pred_mask: torch.Tensor, cfg: MossFormer2SE48KConfig
) -> torch.Tensor:
    """[1, T] segment + [1, F, T'] mask row -> [T] reconstructed audio."""
    spectrum = _stft(segment[0, :], cfg)
    pred_mask = pred_mask.permute(2, 1, 0)
    masked_spec = spectrum * pred_mask.detach().cpu()
    masked_spec_complex = masked_spec[:, :, 0] + 1j * masked_spec[:, :, 1]
    return _istft(masked_spec_complex, cfg, len(segment[0, :]))


def decode_one_audio_mossformer2_se_48k(
    model, device, inputs: np.ndarray, cfg: MossFormer2SE48KConfig
) -> np.ndarray:
    """inputs [B, T] float (numpy, [-1, 1]) -> enhanced [B, T] numpy."""
    inputs = inputs * MAX_WAV_VALUE
    audio = torch.from_numpy(inputs).type(torch.FloatTensor)
    b = audio.shape[0]

    fbanks_batch = torch.cat(
        [_fbank_180(audio[i : i + 1, :], cfg) for i in range(b)], dim=0
    ).to(device)
    pred_mask_b = model(fbanks_batch)[-1]
    outputs = torch.stack(
        [
            _masked_istft(
                audio[i : i + 1, :], pred_mask_b[i : i + 1, :, :], cfg
            )
            for i in range(b)
        ],
        dim=0,
    )
    return outputs.numpy() / MAX_WAV_VALUE


def enhance(
    model,
    cfg: MossFormer2SE48KConfig,
    audio: np.ndarray,
    device,
) -> np.ndarray:
    """[T] float mono in [-1, 1] -> enhanced [T]. The pipeline entry."""
    return decode_one_audio_mossformer2_se_48k(model, device, audio[None, :], cfg)[0]
