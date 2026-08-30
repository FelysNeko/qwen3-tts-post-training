"""MossFormer2_SE_48K checkpoint loading — ported from
clearvoice/clearvoice/networks.py::SpeechModel.load_model/_load_model (the
non-ModuleList branch, model_key='model'), with the fetch moved to a plain
huggingface_hub snapshot_download (canonical HF cache, no CWD-relative
checkpoint_dir, no chdir sandbox, no symlink bridge).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

CLEARVOICE_HF_REPO = "alibabasglab/MossFormer2_SE_48K"

logger = logging.getLogger(__name__)


def ensure_clearvoice() -> Path:
    """Resolve (downloading once via HF if needed) the snapshot dir containing
    `last_best_checkpoint` + the ckpt file."""
    from huggingface_hub import snapshot_download

    logger.info(f"fetching ClearVoice ckpt via HF ({CLEARVOICE_HF_REPO})")
    snapshot = Path(snapshot_download(repo_id=CLEARVOICE_HF_REPO))
    logger.info(f"ClearVoice ckpt ready: {snapshot}")
    return snapshot


def load_mossformer2_se_48k(snapshot_dir: Path, device) -> torch.nn.Module:
    """Build the model and load the published ckpt (fp32, eval mode — same as
    the pip package's CLS_MossFormer2_SE_48K).

    Key mapping ported from networks.py::SpeechModel._load_model: checkpoints
    saved from `torch.nn.DataParallel` runs carry a `module.` prefix on every
    key, so upstream tries exact -> stripped -> prefixed, each guarded by a
    shape check (name collision with a wrong shape loads nothing). The
    published MossFormer2_SE_48K ckpt has NO prefix — 929/929 keys hit the
    exact branch, the fallbacks are parity-only. One deliberate deviation
    from upstream: misses RAISE instead of silently keeping random init.
    """
    from preprocess.clearvoice.mossformer2_se.mossformer2_se_wrapper import (
        MossFormer2_SE_48K,
    )

    model = MossFormer2_SE_48K(None).model
    snapshot_dir = Path(snapshot_dir)
    best_name = snapshot_dir / "last_best_checkpoint"
    with open(best_name) as f:
        model_name = f.readline().strip()
    checkpoint = torch.load(snapshot_dir / model_name, map_location="cpu")
    state = checkpoint.get("model", checkpoint)

    sd = model.state_dict()
    missed = []
    for key in sd:
        if key in state and sd[key].shape == state[key].shape:
            sd[key] = state[key]
        elif (
            key.replace("module.", "") in state
            and sd[key].shape == state[key.replace("module.", "")].shape
        ):
            sd[key] = state[key.replace("module.", "")]
        elif (
            "module." + key in state
            and sd[key].shape == state["module." + key].shape
        ):
            sd[key] = state["module." + key]
        else:
            missed.append(key)
    if missed:
        raise RuntimeError(
            f"{len(missed)}/{len(sd)} keys did not match the ClearVoice ckpt "
            f"(e.g. {missed[:3]}) — wrong/vendored-model drift, refusing a "
            "silent partial load"
        )
    model.load_state_dict(sd)
    model.to(device).eval()
    return model
