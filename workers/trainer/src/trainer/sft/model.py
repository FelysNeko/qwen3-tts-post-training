"""SFT trainer model: full fine-tune of the generative stack.

`SftTrainerModel` is the `ModelWrapper` subclass the shared-kernel docstring
anticipates: every generative parameter stays unfrozen (the whole talker —
layers, the codec embedding tables, text_projection, codec_head AND the code
predictor / MTP heads, unlike GRPO which freezes the predictor), while the
environment (`speech_tokenizer`, never in the teacher-forcing graph) and the
conditioning extractor (`speaker_encoder`, whose output enters training
detached) stay frozen. The ONE deliberate exception is
`talker.model.text_embedding` (311M, Qwen2-lineage text vocab): frozen
UNCONDITIONALLY, mirroring the official voice-SFT recipe — official
0.6B-Base → official CustomVoice moves it by 0.005% (bf16 noise) while every
other block drifts 0.8-1.6%.

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
    """Component-freeze ablation: `freeze` is a composable set of frozen
    components — `{"subtalker"}` (code predictor weight-frozen, its CE still
    backprops through past_hidden into the talker) and `{"talker"}` (talker
    backbone + codec_head frozen) form the symmetric pair: text_projection,
    the shared ingress feeding BOTH stacks, trains in both arms so the pair
    differs by exactly one variable — which generative stack learns.
    `{"text"}` composes onto either arm (e.g. `{"talker", "text"}` = the
    strict predictor-only variant). `{"embedding"}` freezes BOTH codec-table
    groups — the talker's main table AND the predictor's 15 MTP tables
    (`{"subtalker"}` already takes the latter with its weights) — leaving the
    transformer stacks + heads as the only trainees (the tables-vs-blocks
    attribution probe). `{"blocks"}` is its complement: both transformer
    stacks (layers + final norm) and codec_head frozen, so ONLY the two audio
    table groups + text_projection train (40.9M) — the "adaptation lives in
    embedding space, not transformer weights" probe.

    Unconditional (not part of `freeze`): `talker.model.text_embedding` is
    always frozen — the official recipe never trains it (official
    Base→CustomVoice drift 0.005% vs 0.8-1.6% everywhere else), it is the
    largest sparse-update victim under Adam (rare-token rows sign-step as
    fast as common ones), and freezing it saves 311M×3 bf16 optimizer bytes.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:1",
        freeze: list[str] | None = None,
    ):
        super().__init__(model_path, device=device)
        # speech_tokenizer is a plain wrapper (not an nn.Module) — its weights
        # never enter model.parameters(); only speaker_encoder needs freezing
        if self.model.speaker_encoder is not None:
            for p in self.model.speaker_encoder.parameters():
                p.requires_grad_(False)

        # official-aligned invariant (no CLI switch): voice SFT never touches
        # the text vocab embedding
        for p in self.talker.model.text_embedding.parameters():
            p.requires_grad_(False)

        if freeze is not None:
            if "subtalker" in freeze:
                for p in self.talker.code_predictor.parameters():
                    p.requires_grad_(False)
            if "talker" in freeze:
                for p in self.talker.model.parameters():
                    p.requires_grad_(False)
                for p in self.talker.codec_head.parameters():
                    p.requires_grad_(False)

    def enable_grad_checkpoint(self) -> None:
        """Recompute-activations mode for memory-tight full-FT: 1.7B bf16
        AdamW states (w+g+m+v ≈ 13.5GB) leave ~2GB of a 16GB card for
        activations, which B1 long clips exceed. Checkpointing both
        transformer stacks cuts activation memory several-fold at a
        ~30% step-time cost. Training-only — sampling runs on a separate
        ModelWrapper and never enables this."""
        for sub in (
            self.talker.model,
            self.talker.code_predictor.model,
        ):
            sub.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
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
