"""Teacher-forcing log-prob extraction (MD §7 缺口 #4).

The talker's generation does not expose per-step logits, so we rebuild the
exact sampling-time input from the prompt text + sampled code groups, using the
official SFT collate layout (TTSDataset.collate_fn) with ONE correction: the
text channel goes through `text_projection`, exactly as generation does.

This reconstruction is only valid because rollout uses language="Auto": the
generation prefill `[nothink, think_bos, think_eos]` then matches the collate
layout position-for-position (verified: max abs diff vs captured generation
logits ≈ 0.31 bf16 noise, sampled-token ranks identical).

The sampled codes exclude the EOS stop token (generation truncates it), so the
log-prob sum covers exactly the returned semantic tokens.

Sampling-consistency (MD §7 缺口 #3, revised after C1v10): logits are divided
by the same temperature used for sampling and evaluated on the FULL fp32
softmax. The sampling-time top_k/suppress masks are deliberately NOT re-applied
(a hard truncation is discontinuous in the weights — see
`logprobs_from_logits`); repetition_penalty is not part of the RL sampling
contract at all (sampler signatures exclude it), so the reconstruction stays
stateless: one teacher-forcing forward, temperature scaling, done.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from trainer.model import TrainerModel
from trainer.samplers.base import tokenize_assistant

# SFT-collate layout (official TTSDataset.collate_fn), per sample:
#   0-2     role tokens (text channel)
#   3-7     codec prefill [nothink, think_bos, think_eos, speaker, pad]
#   8..8+L  text tokens (codec channel = pad)
#   8+L     text eos (codec pad)
#   9+L     codec bos
#   10+L..  codec semantic groups (code_len positions, 16 codebooks)
#   10+L+T  codec eos
# where L = number of text tokens after the 3-token role.


@dataclass
class LogProbResult:
    log_probs: torch.Tensor  # [B, T] per-token log-probs of sampled semantic tokens
    mask: torch.Tensor  # [B, T] 1 on valid (non-pad) positions
    lengths: torch.Tensor  # [B] valid sequence lengths


def logprobs_from_logits(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Per-position log-prob of `tokens` under the temperature-scaled model
    distribution, evaluated on the FULL softmax.

    The sampling-time top_k / suppress masks are deliberately NOT re-applied:
    a hard truncation makes the log-prob DISCONTINUOUS — a 1e-5 weight nudge
    (one clipped Adam step) flips borderline tokens across the k-th boundary,
    producing ±inf log-ratios, KL = inf and exploding gradients (diagnosed
    C1v10: post-update per-token |δ| = inf, grad_norm 2976; the same root
    cause behind every earlier NaN / runaway-policy smoke failure). Sampled
    tokens always live inside the sampling support, so their unmasked
    log-probs are finite and continuous; top-50 mass at T=0.9 ≈ 1, so the
    behavior-policy bias is negligible. Temperature is kept (continuous, no
    boundary). Softmax runs in fp32 — bf16 log_softmax quantization
    (~0.02/token) would otherwise dominate the small deltas the ratio and
    KL are made of."""
    logits = logits.float() / temperature
    return F.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


class LogProbComputer:
    """Builds the SFT-collate input from (texts, codes) and reads semantic-head
    log-probs via one teacher-forcing forward (adapter on/off = policy/ref)."""

    def __init__(self, ttm: TrainerModel, speaker: str = "cyrene"):
        self.ttm = ttm
        self.model = ttm.model
        self.processor = ttm.processor
        self.talker = ttm.model.talker

        talker_config = self.talker.config
        self.num_code_groups = talker_config.num_code_groups
        self.speaker_vec = self.talker.model.codec_embedding.weight[
            talker_config.spk_id[speaker.lower()]
        ]

    def _tokenize(self, text: str) -> torch.Tensor:
        return tokenize_assistant(self.processor, text)[0]

    def _build_input(
        self,
        texts: list[str],
        codes: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
        """Returns (input_embeddings, attention_mask, meta) where
        meta[i] = {code_len, pred_start, semantic_tokens}."""
        codes = [
            cc.clone() for cc in codes
        ]  # drop inference-mode so policy backward works
        batch = len(codes)
        device = self.model.device
        config = self.model.config
        talker_config = self.talker.config

        text_ids = [
            self._tokenize(text)[:-5] for text in texts
        ]  # 3 role + L text tokens
        code_lengths = [cc.shape[0] for cc in codes]
        text_lengths = [ti.shape[0] for ti in text_ids]  # 3 + L
        max_len = max(tl + cl for tl, cl in zip(text_lengths, code_lengths)) + 8

        input_ids = torch.zeros((batch, max_len, 2), dtype=torch.long, device=device)
        codec_ids = torch.zeros(
            (batch, max_len, self.num_code_groups), dtype=torch.long, device=device
        )
        text_embedding_mask = torch.zeros(
            (batch, max_len, 1), dtype=torch.bool, device=device
        )
        codec_embedding_mask = torch.zeros(
            (batch, max_len, 1), dtype=torch.bool, device=device
        )
        codec_mask = torch.zeros((batch, max_len), dtype=torch.bool, device=device)
        attention_mask = torch.zeros((batch, max_len), dtype=torch.long, device=device)
        meta: list[dict] = []

        for i in range(batch):
            text_ids_i = text_ids[i]
            codes_i = codes[i]
            text_len = text_lengths[i]  # 3 + L
            code_len = code_lengths[i]  # T
            l = text_len - 3  # actual text token count

            # --- text channel ---
            input_ids[i, :3, 0] = text_ids_i[:3]
            input_ids[i, 3:7, 0] = config.tts_pad_token_id
            input_ids[i, 7, 0] = config.tts_bos_token_id
            input_ids[i, 8 : 8 + l, 0] = text_ids_i[3:]
            input_ids[i, 8 + l, 0] = config.tts_eos_token_id
            input_ids[i, 9 + l : 11 + l + code_len, 0] = config.tts_pad_token_id
            text_embedding_mask[i, : 11 + l + code_len] = True

            # --- codec channel ---
            input_ids[i, 3:8, 1] = torch.tensor(
                [
                    talker_config.codec_nothink_id,
                    talker_config.codec_think_bos_id,
                    talker_config.codec_think_eos_id,
                    0,  # speaker slot (overwritten below)
                    talker_config.codec_pad_id,
                ],
                device=device,
            )
            input_ids[i, 8 : 8 + l, 1] = talker_config.codec_pad_id
            input_ids[i, 8 + l, 1] = talker_config.codec_pad_id
            input_ids[i, 8 + l + 1, 1] = talker_config.codec_bos_id
            input_ids[i, 10 + l : 10 + l + code_len, 1] = codes_i[:, 0]
            input_ids[i, 10 + l + code_len, 1] = talker_config.codec_eos_token_id
            codec_ids[i, 10 + l : 10 + l + code_len, :] = codes_i
            codec_embedding_mask[i, 3 : 11 + l + code_len] = True
            codec_embedding_mask[i, 6] = False  # speaker slot
            codec_mask[i, 10 + l : 10 + l + code_len] = True
            attention_mask[i, : 11 + l + code_len] = True

            meta.append(
                {
                    "code_len": code_len,
                    "pred_start": 8
                    + l
                    + 1,  # codec bos position → predicts first semantic
                    "semantic_tokens": codes_i[:, 0],
                }
            )

        # --- input embeddings (official construction + text_projection fix) ---
        input_text_ids = input_ids[:, :, 0]
        input_codec_ids = input_ids[:, :, 1]

        input_text_embedding = (
            self.talker.text_projection(
                self.talker.model.text_embedding(input_text_ids)
            )
            * text_embedding_mask
        )
        input_codec_embedding = (
            self.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
        )
        input_codec_embedding[:, 6, :] = self.speaker_vec

        input_embeddings = input_text_embedding + input_codec_embedding

        codec_predictor_embeddings = self.talker.code_predictor.model.codec_embedding
        for k in range(1, self.num_code_groups):
            input_embeddings = input_embeddings + codec_predictor_embeddings[k - 1](
                codec_ids[:, :, k]
            ) * codec_mask.unsqueeze(-1)

        return input_embeddings, attention_mask, meta

    @torch.inference_mode()
    def compute_ref(
        self,
        texts: list[str],
        codes: list[torch.Tensor],
        temperature: float = 0.9,
    ) -> LogProbResult:
        """Reference (adapter OFF) log-probs — frozen base forward, no grads."""
        self.ttm.set_adapter(False)
        try:
            return self._forward(texts, codes, temperature)
        finally:
            self.ttm.set_adapter(True)

    def compute_policy(
        self,
        texts: list[str],
        codes: list[torch.Tensor],
        temperature: float = 0.9,
    ) -> LogProbResult:
        """Policy (adapter ON) log-probs — differentiable, part of the train graph."""
        self.ttm.set_adapter(True)
        return self._forward(texts, codes, temperature)

    def _forward(
        self,
        texts: list[str],
        codes: list[torch.Tensor],
        temperature: float,
    ) -> LogProbResult:
        input_embeddings, attention_mask, meta = self._build_input(texts, codes)
        logits = self.talker(
            inputs_embeds=input_embeddings, attention_mask=attention_mask
        ).logits

        max_code_len = max(m["code_len"] for m in meta)
        log_prob_cols, lengths = [], []
        for i, m in enumerate(meta):
            code_len = m["code_len"]
            pred_positions = m["pred_start"] + torch.arange(
                code_len, device=logits.device
            )
            token_log_probs = logprobs_from_logits(
                logits[i, pred_positions],
                m["semantic_tokens"],
                temperature,
            )
            log_prob_cols.append(F.pad(token_log_probs, (0, max_code_len - code_len)))
            lengths.append(code_len)

        log_probs = torch.stack(log_prob_cols)
        lengths = torch.tensor(lengths, device=log_probs.device)
        mask = (
            torch.arange(max_code_len, device=log_probs.device)[None, :]
            < lengths[:, None]
        ).float()
        # zero padded positions so `log_probs * mask` can't produce -inf*0 = NaN
        log_probs = torch.where(mask.bool(), log_probs, torch.zeros_like(log_probs))
        return LogProbResult(log_probs, mask, lengths)
