"""SFT trainer model: full fine-tune of the generative stack.

`SftTrainerModel` is the `ModelWrapper` subclass the shared-kernel docstring
anticipates: every generative parameter stays unfrozen (the whole talker —
layers, embeddings, text_projection, codec_head AND the code predictor /
MTP heads, unlike GRPO which freezes the predictor), while the environment
(`speech_tokenizer`, never in the teacher-forcing graph) and the conditioning
extractor (`speaker_encoder`, whose output enters training detached) stay
frozen.

Speaker conditioning: ONE user-specified reference audio →
`extract_speaker_vec` → [hidden] vector broadcast into embedding slot 6 for
every batch item (the `ModelWrapper.teacher_forcing` speaker-id-lookup path;
same 24kHz mel recipe as the official voice-clone prompt). CustomVoice ckpts
ship without the speaker encoder — `load_base_speaker_encoder` borrows it
from the base ckpt (local dir or HF repo id) the official SFT flow starts
from.
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import soundfile as sf
import torch

from trainer.model import ModelWrapper

logger = logging.getLogger(__name__)


class SftTrainerModel(ModelWrapper):
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:1",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(model_path, device=device, dtype=dtype)
        # speech_tokenizer is a plain wrapper (not an nn.Module) — its weights
        # never enter model.parameters(); only speaker_encoder needs freezing
        if self.model.speaker_encoder is not None:
            for p in self.model.speaker_encoder.parameters():
                p.requires_grad_(False)


def load_base_speaker_encoder(model: ModelWrapper, base_model_path: str) -> None:
    """Borrow the speaker encoder from the base ckpt (official SFT's
    `--init_model_path`).

    CustomVoice ckpts construct `speaker_encoder` only for
    `tts_model_type == "base"` — a finetuned model carries None, so the
    audio→embedding path is impossible without this. `base_model_path` is a
    local ckpt dir or an HF repo id (the single `model.safetensors` is fetched
    through the canonical HF cache); ONLY the `speaker_encoder.*` tensors are
    read out of it (lazy safe_open, no full-model load). The encoder is built
    from OUR config's speaker_encoder_config (identical ECAPA-TDNN recipe,
    enc_dim 2048 == talker hidden_size), attached to the model and frozen.
    """
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder
    from safetensors import safe_open

    path = Path(base_model_path)
    if path.is_dir():
        weights_path = path / "model.safetensors"
        assert weights_path.exists(), f"{weights_path} missing in base ckpt dir"
    else:
        from huggingface_hub import hf_hub_download

        weights_path = Path(hf_hub_download(base_model_path, "model.safetensors"))
        logger.info(f"fetched base speaker-encoder weights: {weights_path}")

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(weights_path, framework="pt", device="cpu") as file:
        all_keys: list[str] = list(file.keys())
    keys = [k for k in all_keys if k.startswith("speaker_encoder.")]
    assert keys, f"no speaker_encoder.* keys in {weights_path}"
    with safe_open(weights_path, framework="pt", device="cpu") as file:
        for key in keys:
            tensors[key.removeprefix("speaker_encoder.")] = file.get_tensor(key)
    encoder = Qwen3TTSSpeakerEncoder(model.model.config.speaker_encoder_config)
    encoder.load_state_dict(tensors, strict=True)
    encoder.to(device=model.device, dtype=model.dtype)
    for p in encoder.parameters():
        p.requires_grad_(False)
    model.model.speaker_encoder = encoder
    logger.info(
        f"speaker encoder: {len(tensors)} tensors loaded from {base_model_path!r}"
    )


def extract_speaker_vec(model: ModelWrapper, audio_path: str | Path) -> torch.Tensor:
    """Reference audio → speaker embedding [hidden] (model device/dtype).

    soundfile read (float32, stereo→mono) → resample to the speaker encoder's
    rate (24k) → the model's own `extract_speaker_embedding` (mel(128) →
    speaker_encoder). Called ONCE at startup; the vector is a constant input
    (inference_mode output, same detach semantics as the official
    `speaker_encoder(ref_mels).detach()`).
    """
    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    target_sr = model.model.speaker_encoder_sample_rate
    if sr != target_sr:
        audio = librosa.resample(y=audio, orig_sr=int(sr), target_sr=int(target_sr))
    return model.model.extract_speaker_embedding(audio=audio, sr=int(target_sr))
