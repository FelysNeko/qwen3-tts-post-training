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
"""

from __future__ import annotations

import logging
from pathlib import Path

UTMOS_HF_REPO = "sarulab-speech/UTMOSv2"

# model name -> (modelscope model_id, file in repo)
SV_SOURCES = {
    "eres2netv2": {
        "model_id": "iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common",
        "file": "pretrained_eres2netv2w24s4ep4.ckpt",
    },
    "cam++": {
        "model_id": "iic/speech_campplus_sv_zh-cn_16k-common",
        "file": "campplus_cn_common.bin",
    },
}

logger = logging.getLogger(__name__)


def ensure_sv_ckpt(name: str) -> Path:
    """Resolve (downloading once via ModelScope if needed) the SV ckpt for
    `name`. Returns the ModelScope-managed cache path."""
    from modelscope.hub.file_download import model_file_download

    conf = SV_SOURCES[name]
    logger.info(f"fetching SV ckpt {name} via ModelScope ({conf['model_id']})")
    local = model_file_download(
        model_id=conf["model_id"], file_path=conf["file"], revision="master"
    )
    logger.info(f"SV ckpt {name} ready: {local}")
    return Path(local)


def ensure_utmos(fold: int, seed: int) -> Path:
    """Resolve (downloading once via HF cache if needed) the UTMOSv2 fold
    weights. Returns the HF-managed cache path."""
    from huggingface_hub import hf_hub_download

    logger.info(f"fetching UTMOSv2 fold{fold} via HF ({UTMOS_HF_REPO})")
    local = hf_hub_download(
        repo_id=UTMOS_HF_REPO,
        filename=f"fold{fold}_s{seed}_best_model.pth",
    )
    logger.info(f"UTMOSv2 fold{fold} ready: {local}")
    return Path(local)
