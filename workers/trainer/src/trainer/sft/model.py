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
same 24kHz mel recipe as the official voice-clone prompt). SFT starts ONLY
from a base ckpt — base models ship the speaker encoder in-model (custom_voice
ckpts carry none); `run_sft` asserts this fail-loudly.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import soundfile as sf
import torch

from trainer.model import ModelWrapper


class SftTrainerModel(ModelWrapper):
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:1",
    ):
        super().__init__(model_path, device=device)
        # speech_tokenizer is a plain wrapper (not an nn.Module) — its weights
        # never enter model.parameters(); only speaker_encoder needs freezing
        if self.model.speaker_encoder is not None:
            for p in self.model.speaker_encoder.parameters():
                p.requires_grad_(False)


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
