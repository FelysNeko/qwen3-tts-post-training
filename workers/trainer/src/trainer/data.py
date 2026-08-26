"""Shared SFT-collate layout — port of official `finetuning/dataset.py::
TTSDataset.collate_fn`, consumed by BOTH backends:

- SFT (future `trainer/sft.py`): items carry REAL audio_codes from data; the
  trailing codec-EOS entry written into `codec_0_labels` participates in the
  CE loss. Reference audio shares ONE fixed clip for new-speaker runs, so mel
  extraction lives in the future dataset wrapper — this function never sees
  audio, tensors in / tensors out.
- GRPO logprob reconstruction (trainer/logprob.py): items carry SAMPLED codes
  (EOS truncated by generation); the reserved EOS label slot falls OUTSIDE
  `codec_mask` and is ignored downstream (verified: `labels[:, 1:]` selected
  by `codec_mask[:, 1:]` covers exactly the T sampled semantic tokens).

Batch tensors are built on CPU; use `CollateBatch.to(device)` before embedding
lookup. Consumers differ in EMBEDDING construction only (speaker vector
source): shared here is the single source of truth for token LAYOUT.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig


@dataclass
class CollateBatch:
    """Dense teacher-forcing batch (official collate_fn field names/shapes).

    input_ids [b, t, 2]: channel 0 = text stream (`input_text_ids`), channel
    1 = codec stream (`input_codec_ids`); masks unsqueezed to [b, t, 1];
    codec_0_labels [b, t] with -100 outside the codec span; codec_ids
    [b, t, num_code_groups]; codec_mask [b, t] True on the T code-group
    positions (False on the reserved EOS label slot).
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    text_embedding_mask: torch.Tensor
    codec_embedding_mask: torch.Tensor
    codec_0_labels: torch.Tensor
    codec_ids: torch.Tensor
    codec_mask: torch.Tensor

    @property
    def input_text_ids(self) -> torch.Tensor:
        """Text channel ids, `input_ids[:, :, 0]` ([b, t])."""
        return self.input_ids[:, :, 0]

    @property
    def input_codec_ids(self) -> torch.Tensor:
        """Codec channel ids, `input_ids[:, :, 1]` ([b, t])."""
        return self.input_ids[:, :, 1]

    def to(self, device: str | torch.device) -> CollateBatch:
        return CollateBatch(
            **{k: v.to(device) for k, v in self.__dict__.items()}
        )


def collate(
    items: list[tuple[torch.Tensor, torch.Tensor]],
    config: Qwen3TTSConfig,
    num_code_groups: int,
) -> CollateBatch:
    """Build the dense teacher-forcing batch (official collate_fn layout).

    Args:
        items: one ``(text_ids, audio_codes)`` pair per sample.
            ``text_ids`` [n] or [1, n] — official `_build_assistant_text`
            tokenization WITHOUT the trailing 5 special tokens (official
            `[:, :-5]`); layout 3 role + L text tokens, n = 3 + L.
            ``audio_codes`` [T, num_code_groups] (per-position codebook stack).
        config: top-level model config providing ``tts_{pad,bos,eos}_token_id``
            and ``talker_config.codec_{nothink,think_bos,think_eos,pad}`
            `_id`` / ``codec_eos_token_id`` (same object the official collate
            reads).
        num_code_groups: codebook count (Qwen3-TTS 12Hz: 16) — threaded
            through from the model config by the caller, no default.
    """
    b = len(items)
    t = max(ti.shape[-1] + ac.shape[0] for ti, ac in items) + 8

    input_ids = torch.zeros((b, t, 2), dtype=torch.long)
    codec_ids = torch.zeros((b, t, num_code_groups), dtype=torch.long)
    text_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
    codec_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
    codec_mask = torch.zeros((b, t), dtype=torch.bool)
    attention_mask = torch.zeros((b, t), dtype=torch.long)
    codec_0_labels = torch.full((b, t), -100, dtype=torch.long)

    tc = config.talker_config
    codec_prefill = torch.tensor(
        [
            tc.codec_nothink_id,
            tc.codec_think_bos_id,
            tc.codec_think_eos_id,
            0,  # speaker-embedding slot (patched by consumers)
            tc.codec_pad_id,
        ],
        dtype=torch.long,
    )

    for i, (text_ids, audio_codes) in enumerate(items):
        if text_ids.dim() == 2:  # [1, n] raw processor output
            text_ids = text_ids[0]
        n = text_ids.shape[0]  # 3 role + L
        q = audio_codes.shape[0]
        assert audio_codes.shape[1] == num_code_groups

        # --- text channel ---
        input_ids[i, :3, 0] = text_ids[:3]
        input_ids[i, 3:7, 0] = config.tts_pad_token_id
        input_ids[i, 7, 0] = config.tts_bos_token_id
        input_ids[i, 8 : 8 + n - 3, 0] = text_ids[3:]
        input_ids[i, 8 + n - 3, 0] = config.tts_eos_token_id
        input_ids[i, 8 + n - 2 : 8 + n + q, 0] = config.tts_pad_token_id
        text_embedding_mask[i, : 8 + n + q] = True

        # --- codec channel ---
        input_ids[i, 3:8, 1] = codec_prefill
        input_ids[i, 8 : 8 + n - 3, 1] = tc.codec_pad_id
        input_ids[i, 8 + n - 3, 1] = tc.codec_pad_id
        input_ids[i, 8 + n - 2, 1] = tc.codec_bos_id
        input_ids[i, 8 + n - 1 : 8 + n - 1 + q, 1] = audio_codes[:, 0]
        input_ids[i, 8 + n - 1 + q, 1] = tc.codec_eos_token_id

        codec_0_labels[i, 8 + n - 1 : 8 + n - 1 + q] = audio_codes[:, 0]
        codec_0_labels[i, 8 + n - 1 + q] = tc.codec_eos_token_id

        codec_ids[i, 8 + n - 1 : 8 + n - 1 + q, :] = audio_codes

        codec_embedding_mask[i, 3 : 8 + n + q] = True
        codec_embedding_mask[i, 6] = False  # speaker-embedding slot

        codec_mask[i, 8 + n - 1 : 8 + n - 1 + q] = True
        attention_mask[i, : 8 + n + q] = True

    return CollateBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        text_embedding_mask=text_embedding_mask.unsqueeze(-1),
        codec_embedding_mask=codec_embedding_mask.unsqueeze(-1),
        codec_0_labels=codec_0_labels,
        codec_ids=codec_ids,
        codec_mask=codec_mask,
    )
