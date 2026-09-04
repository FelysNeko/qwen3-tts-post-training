"""Probe: vendored MossFormer2_SE_48K decode vs the pip `clearvoice` package
(same weights, same clips, same torch seed — kaldi fbank dither=1.0 consumes
the global torch RNG, so identical seeds must give identical fbanks).

Checks:
1. Short (<20s) and long (26s) clips decode BIT-IDENTICALLY to the pip
   package (same seed → same dithered fbank; pip's sliding-window branch is
   unreachable for tensor-mode input — batch-dim bug — so every clip takes
   the batched path on both sides).

Run in the preprocess venv while the `clearvoice` pip package is still
installed (the A side needs it):
    workers/preprocess/.venv/bin/python probes/probe_clearvoice_ab.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path("/home/felys/workspace/qwen3-tts-post-training")
sys.path.insert(0, str(REPO / "workers/preprocess/src"))

import numpy as np
import soundfile as sf
import torch

CORPUS = Path("/home/felys/workspace/delta-me13/corpora/tts/cyrene/Chinese(PRC)")
SHORT = [
    "vo_ambient_w4_v340_greeting_cyrene_117",
    "chapter4_77_cyrene_234",
    "vo_ambient_w4_v340_greeting_cyrene_120",
]
LONG = "archive_cyrene_14"
DEVICE = "cuda:1"


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), sr


def main() -> None:
    try:
        import clearvoice  # noqa: F401
    except ImportError:
        print(
            "pip `clearvoice` not installed — the vendored path "
            "(preprocess.clearvoice) is now the only implementation. This probe "
            "is archival: its 2026-08-27 run recorded BIT-IDENTICAL outputs "
            "(max_abs=0.0, 4 clips incl. one 26s) vs pip clearvoice 0.1.2; "
            "reinstall the pip package temporarily to re-verify."
        )
        return

    from preprocess.clearvoice.decode import MossFormer2SE48KConfig, enhance
    from preprocess.clearvoice.load import ensure_clearvoice, load_mossformer2_se_48k

    snapshot = ensure_clearvoice()
    v_model = load_mossformer2_se_48k(snapshot, DEVICE)
    cfg = MossFormer2SE48KConfig()

    # pip path: construct inside a chdir sandbox over a local_dir pre-populated
    # via snapshot_download (upstream checkpoint_dir is CWD-relative)
    import os

    import clearvoice as _cv_unused  # noqa: F401  (A side: pip package)
    from clearvoice import ClearVoice

    pip_root = REPO / ".cache" / "_clearvoice_ab"
    target = pip_root / "checkpoints" / "MossFormer2_SE_48K"
    if not (target / "last_best_checkpoint").is_file():
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id="alibabasglab/MossFormer2_SE_48K", local_dir=str(target))
    cwd = os.getcwd()
    os.chdir(pip_root)
    try:
        pip_cv = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
        pip_model = pip_cv.models[0]
        pip_model.device = torch.device(DEVICE)
        pip_model.model.to(DEVICE)
    finally:
        os.chdir(cwd)

    def pip_enhance(audio: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return np.asarray(pip_cv(audio[None, :]).squeeze(), dtype=np.float32)

    print("=== all clips (pip always takes the batched path) ===")
    for name in [*SHORT, LONG]:
        audio, sr = load_mono(CORPUS / f"{name}.wav")
        assert sr == 48000

        torch.manual_seed(0)
        out_v = enhance(v_model, cfg, audio, DEVICE)

        torch.manual_seed(0)
        out_p = pip_enhance(audio)

        max_abs = float(np.max(np.abs(out_v - out_p)))
        print(
            f"{name[:44]:44s} ({len(audio) / sr:5.1f}s) "
            f"max_abs={max_abs:.2e} identical={np.array_equal(out_v, out_p)}"
        )


if __name__ == "__main__":
    main()
