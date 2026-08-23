"""First-load weight fetch — delegates cache/download-location management to
the upstream tools (HF / ModelScope), no manual paths here.

Sources:
- SV ckpts (ModelScope — canonical 3D-Speaker host, NOT on HF):
    iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common
    iic/speech_campplus_sv_zh-cn_16k-common
- UTMOSv2 folds (official HF repo, public):
    sarulab-speech/UTMOSv2
- Qwen3-ASR-1.7B-hf and facebook/wav2vec2-base already auto-download via
  transformers / huggingface_hub on first load.

`sv_dir` (CLI `--sv-dir`) remains an override: if a ckpt already sits under
sv_dir/pretrained/... (e.g. a cloned 3D-Speaker checkout) it is used verbatim;
otherwise the ModelScope snapshot is fetched through the modelscope client.
"""

from __future__ import annotations

import sys
from pathlib import Path

UTMOS_HF_REPO = "sarulab-speech/UTMOSv2"

# model name -> (modelscope model_id, file in repo, file rel to sv_dir)
SV_SOURCES = {
    "eres2netv2": {
        "model_id": "iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common",
        "file": "pretrained_eres2netv2w24s4ep4.ckpt",
        "rel": (
            "pretrained/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common/"
            "pretrained_eres2netv2w24s4ep4.ckpt"
        ),
    },
    "campplus": {
        "model_id": "iic/speech_campplus_sv_zh-cn_16k-common",
        "file": "campplus_cn_common.bin",
        "rel": (
            "pretrained/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
        ),
    },
}


def _log(msg: str) -> None:
    print(f"[fetch] {msg}", file=sys.stderr, flush=True)


def ensure_sv_ckpt(sv_dir: str | Path | None, name: str) -> Path:
    """Resolve (downloading once via ModelScope if needed) the SV ckpt for
    `name`. A file already present under sv_dir/pretrained/... wins."""
    conf = SV_SOURCES[name]
    manual = Path(sv_dir) / conf["rel"] if sv_dir else None
    if manual is not None and manual.exists() and manual.stat().st_size > 0:
        return manual
    from modelscope.hub.file_download import model_file_download

    _log(f"fetching SV ckpt {name} via ModelScope ({conf['model_id']})")
    local = model_file_download(
        model_id=conf["model_id"], file_path=conf["file"], revision="master"
    )
    _log(f"SV ckpt {name} ready: {local}")
    return Path(local)


def ensure_utmos(fold: int, seed: int) -> Path:
    """Resolve (downloading once via HF cache if needed) the UTMOSv2 fold
    weights. Returns the HF-managed cache path."""
    from huggingface_hub import hf_hub_download

    _log(f"fetching UTMOSv2 fold{fold} via HF ({UTMOS_HF_REPO})")
    local = hf_hub_download(
        repo_id=UTMOS_HF_REPO,
        filename=f"fold{fold}_s{seed}_best_model.pth",
    )
    _log(f"UTMOSv2 fold{fold} ready: {local}")
    return Path(local)
