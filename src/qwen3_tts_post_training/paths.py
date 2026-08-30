"""Path helpers shared by worker/probe code (`from qwen3_tts_post_training.paths
import repo_root`)."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Repository root — this package lives at `<root>/src/…`, and every
    worker/probe venv installs the core lib EDITABLE (§18.3), so `__file__`
    always resolves inside the repo tree."""
    return Path(__file__).resolve().parents[2]
