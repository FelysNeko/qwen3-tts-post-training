"""Trainer model wrapper: the SHARED input pipeline + teacher-forcing kernel.

`ModelWrapper` owns ckpt loading, the backend-agnostic forward path
(`teacher_forcing`) and the optimizer interface (`trainable_parameters`);
parameter assembly is left to subclasses — `LoraTrainerModel` (GRPO, adapter
on/off on one weight set) and a future SFT full-FT variant (every param
unfrozen). Both backends share the full (texts, codes) → loss chain:

    tokenize_assistant / collate / teacher_forcing

CollateBatch + collate are a port of the official SFT collate
(finetuning/dataset.py::TTSDataset.collate_fn), verified byte-equal against
the upstream reference.

The trainable unit for GRPO is the talker (text → semantic code groups); the
speech tokenizer is the environment renderer (always frozen).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


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
        return CollateBatch(**{k: v.to(device) for k, v in self.__dict__.items()})


@dataclass
class TeacherForcing:
    """Selected outputs of `ModelWrapper.teacher_forcing` (shared SFT/GRPO).

    talker_logits [b, L-1, V]: semantic-head logits at position p-1 — the dense
    view `logits[:, :-1]`, so the shifted target `batch.codec_0_labels[:, 1:]`
    (SFT, includes EOS) or `talker_targets[:, 1:]` (GRPO) gathers directly.
    predict_mask [b, L-1]: True where a code group sits at p.
    talker_codec_ids [N, Q]: flattened code groups (`codec_ids[codec_mask]`,
    row-major = sample order).
    sub_talker_logits [N, Q-1, V]: predictor-head logits; head j at embedded
    slot j+1 predicts c_{j+1}, conditioned on slots up to j (+ talker hidden).
    """

    talker_logits: torch.Tensor
    predict_mask: torch.Tensor
    talker_codec_ids: torch.Tensor
    sub_talker_logits: torch.Tensor


class ModelWrapper:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:1",
        dtype: torch.dtype = torch.bfloat16,
    ):
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

        wrapper = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
            attn_implementation="flash_attention_2",
        )
        self.model = wrapper.model  # Qwen3TTSForConditionalGeneration
        self.processor = wrapper.processor
        self.device = device
        self.dtype = dtype
        self.talker = self.model.talker

    def tokenize_assistant(self, text: str) -> torch.Tensor:
        """Official `_build_assistant_text` + `_tokenize_texts` — tokenize the
        full assistant-formatted prompt. Returns [1, len] input ids on CPU;
        callers drop the trailing 5 special tokens, mirroring the official
        `[:, :-5]`."""
        prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        ids = self.processor(
            text=prompt,
            return_tensors="pt",
            padding=True,
        )["input_ids"]
        return ids.unsqueeze(0) if ids.dim() == 1 else ids

    def collate(
        self,
        texts: list[str],
        codes: list[torch.Tensor],
    ) -> CollateBatch:
        """(texts, codes) → official-layout batch on the model device.

        Shared by BOTH backends (GRPO logprob reconstruction and the future
        SFT on-the-fly path). Codes are cloned so an inference-mode rollout
        tensor can enter the policy backward graph; text goes through the
        official trailing-special drop (`[:-5]`).

        Layout is a port of the official SFT collate (TTSDataset.collate_fn):
        token ids / codebook count read off `self.model.config` — the same
        object the upstream collate reads."""
        config = self.model.config
        tc = config.talker_config  # Qwen3TTSTalkerConfig (num_code_groups lives here)

        items = [
            (self.tokenize_assistant(t)[0][:-5], cc.clone())
            for t, cc in zip(texts, codes)
        ]
        b = len(items)
        t = max(ti.shape[-1] + ac.shape[0] for ti, ac in items) + 8

        input_ids = torch.zeros((b, t, 2), dtype=torch.long)
        codec_ids = torch.zeros((b, t, tc.num_code_groups), dtype=torch.long)
        text_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_mask = torch.zeros((b, t), dtype=torch.bool)
        attention_mask = torch.zeros((b, t), dtype=torch.long)
        codec_0_labels = torch.full((b, t), -100, dtype=torch.long)

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
            assert audio_codes.shape[1] == tc.num_code_groups

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
        ).to(self.model.device)

    def teacher_forcing(
        self, batch: CollateBatch, speaker_vec: torch.Tensor
    ) -> TeacherForcing:
        """Shared teacher-forcing kernel: embeddings → ONE talker forward →
        the sub-talker pass. Identical for SFT and GRPO (official
        `teacher_forcing` / `sft_12hz.py` structure, with the upstream
        target-leak bug fixed via the p-1 hidden selection). Consumers diverge
        only downstream:

        - SFT: CE on `talker_logits` vs `batch.codec_0_labels[:, 1:]`
          (ignore_index=-100; INCLUDES the EOS label slot — it sits outside
          `codec_mask`, hence dense logits rather than a masked select) plus
          CE on `sub_talker_logits` vs `talker_codec_ids[:, 1:]` at weight 0.3;
        - GRPO: temperature-scaled full-softmax log-probs (grpo/logprob.py).

        Embedding construction (inlined): official SFT input assembly with ONE
        deliberate correction vs the upstream script — the text channel goes
        through `text_projection`, exactly as generation does (upstream omits
        it; verified against captured generation inputs). `speaker_vec` is
        [hidden] (speaker-id lookup, broadcast) or [b, hidden]
        (`speaker_encoder(ref_mels)` output) — both assign into slot 6.

        No labels are passed to the talker: the internal loss would hardcode
        temperature 1.0 and duplicate what consumers do explicitly. Adapter
        state, grad mode and inference_mode stay caller-owned.
        """
        text_embedding = (
            self.talker.text_projection(
                self.talker.model.text_embedding(batch.input_text_ids)
            )
            * batch.text_embedding_mask
        )
        codec_embedding = (
            self.talker.model.codec_embedding(batch.input_codec_ids)
            * batch.codec_embedding_mask
        )
        codec_embedding[:, 6, :] = speaker_vec

        input_embeddings = text_embedding + codec_embedding

        for k in range(1, self.model.config.talker_config.num_code_groups):
            codec_k_embedding = self.talker.code_predictor.get_input_embeddings()[
                k - 1
            ](batch.codec_ids[:, :, k])
            codec_k_embedding = codec_k_embedding * batch.codec_mask.unsqueeze(-1)
            input_embeddings = input_embeddings + codec_k_embedding

        outputs = self.talker(
            inputs_embeds=input_embeddings,
            attention_mask=batch.attention_mask,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[0][-1][:, :-1, :]
        predict_mask = batch.codec_mask[:, 1:]  # [b, L-1]: code at p ↔ hidden p-1
        talker_hidden_states = hidden_states[predict_mask]
        talker_codec_ids = batch.codec_ids[batch.codec_mask]  # [N, Q] row-major
        sub_talker_logits, _ = self.talker.forward_sub_talker_finetune(
            talker_codec_ids, talker_hidden_states
        )
        return TeacherForcing(
            talker_logits=outputs.logits[:, :-1],
            predict_mask=predict_mask,
            talker_codec_ids=talker_codec_ids,
            sub_talker_logits=sub_talker_logits,
        )

    @torch.inference_mode()
    def decode(self, codes: list[torch.Tensor]) -> tuple[list[np.ndarray], int]:
        """codes: list of [T, num_code_groups]. Returns (wavs, sample_rate)."""
        wavs, fs = self.model.speech_tokenizer.decode(
            [{"audio_codes": c} for c in codes]
        )
        return wavs, fs

    @property
    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]
