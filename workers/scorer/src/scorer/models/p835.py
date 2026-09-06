"""P.835 DNSMOS scoring — the DNSMOS ONNX pair, vendored byte-faithful.

Ports the inference core of microsoft/DNS-Challenge's `dnsmos_local.py`
(ComputeScore, non-personalized path only). The reward-facing number is the
polyfit-calibrated **OVRL** (FlowTTS-GRPO uses the same; SIG/BAK stay
diagnostic via `score_one`). Per §53 the scorer-side batch baseline
(`probes/tmp/dnsmos_d_ep1.csv`) was produced by exactly these weights under
exactly these session options — determinism cross-checks must stay bit-equal:

- sig_bak_ovr.onnx  sha256 269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd
- model_v8.onnx     sha256 9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91

Invariants: 16 kHz resample (librosa ≥0.10 keyword-only API); clips shorter
than INPUT_LENGTH are self-tiled by the upstream while-append protocol before
1 s hopping; ONNX CPU sessions with fixed intra_op=2/inter_op=1 (thread count
is part of the pinned setup — do not "tune" it).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf

_SAMPLING_RATE = 16000
_INPUT_LENGTH = 9.01

_MODELS_DIR = Path(__file__).resolve().parents[1] / "vendor" / "dnsmos"

_SO = ort.SessionOptions()
_SO.intra_op_num_threads = 2
_SO.inter_op_num_threads = 1

# non-personalized polyfit (dnsmos_local.py verbatim) — maps raw model
# outputs onto the human MOS scale
_P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
_P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
_P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])


class P835Scorer:
    """Batch P.835 scorer: `score(wavs) -> list[OVRL]`, input order kept.

    `workers` = clip-level scoring concurrency (default 4). The hot path is
    native code that releases the GIL (numpy/onnxruntime/soxr/libsndfile), so
    threads scale — measured 0.76→0.48→0.41s/clip at 1/3/4 workers (sublinear:
    short clips self-tile into MULTIPLE 9.01s onnx hops — ~4.2 hops/clip avg —
    and workers × intra_op=2 spinning threads contend for cores past ~4
    workers). The per-session `intra_op=2` stays FIXED (part of the pinned
    determinism setup); concurrent `session.run` calls are ORT-thread-safe and
    each clip is independent, so results are worker-count invariant."""

    def __init__(self, workers: int = 4) -> None:
        self.workers = workers
        self._sess = ort.InferenceSession(
            str(_MODELS_DIR / "sig_bak_ovr.onnx"), sess_options=_SO
        )
        self._p808_sess = ort.InferenceSession(
            str(_MODELS_DIR / "model_v8.onnx"), sess_options=_SO
        )

    @staticmethod
    def _audio_melspec(audio: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=audio, sr=_SAMPLING_RATE, n_fft=321, hop_length=160, n_mels=120
        )
        return ((librosa.power_to_db(mel, ref=np.max) + 40) / 40).T

    def score_one(self, wav_path: str) -> dict[str, float]:
        """Full per-clip readout: OVRL/SIG/BAK (calibrated), *_raw, P808_MOS,
        len_in_sec — the diagnostic shape of the §53 CSVs."""
        aud, input_fs = sf.read(wav_path)
        fs = _SAMPLING_RATE
        audio = (
            librosa.resample(aud, orig_sr=input_fs, target_sr=fs)
            if input_fs != fs
            else aud
        )
        actual_audio_len = len(audio)
        len_samples = int(_INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / fs) - _INPUT_LENGTH) + 1
        hop_len_samples = fs
        segs = {
            "SIG": [],
            "BAK": [],
            "OVR": [],
            "SIG_raw": [],
            "BAK_raw": [],
            "OVR_raw": [],
            "p808": [],
        }
        for idx in range(num_hops):
            audio_seg = audio[
                int(idx * hop_len_samples) : int(
                    (idx + _INPUT_LENGTH) * hop_len_samples
                )
            ]
            if len(audio_seg) < len_samples:
                continue
            feats = np.array(audio_seg).astype("float32")[np.newaxis, :]
            p808_feats = np.array(self._audio_melspec(audio_seg[:-160])).astype(
                "float32"
            )[np.newaxis, :, :]
            p808_mos = self._p808_sess.run(None, {"input_1": p808_feats})[0][0][0]
            sig_raw, bak_raw, ovr_raw = self._sess.run(None, {"input_1": feats})[0][0]
            segs["SIG"].append(_P_SIG(sig_raw))
            segs["BAK"].append(_P_BAK(bak_raw))
            segs["OVR"].append(_P_OVR(ovr_raw))
            segs["SIG_raw"].append(sig_raw)
            segs["BAK_raw"].append(bak_raw)
            segs["OVR_raw"].append(ovr_raw)
            segs["p808"].append(p808_mos)

        return {
            "len_in_sec": actual_audio_len / fs,
            "num_hops": num_hops,
            "OVRL_raw": float(np.mean(segs["OVR_raw"])),
            "SIG_raw": float(np.mean(segs["SIG_raw"])),
            "BAK_raw": float(np.mean(segs["BAK_raw"])),
            "OVRL": float(np.mean(segs["OVR"])),
            "SIG": float(np.mean(segs["SIG"])),
            "BAK": float(np.mean(segs["BAK"])),
            "P808_MOS": float(np.mean(segs["p808"])),
        }

    def score(self, wavs: list[str]) -> list[float]:
        """Calibrated P.835 OVRL per wav, input order preserved."""
        if self.workers <= 1:
            return [self.score_one(w)["OVRL"] for w in wavs]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return [d["OVRL"] for d in pool.map(self.score_one, wavs)]
