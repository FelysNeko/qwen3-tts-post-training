"""Teacher-forcing log-prob extraction over ALL 16 codebooks (MD §7 缺口 #4).

The talker's generation does not expose per-step logits, so we rebuild the
exact sampling-time input from the prompt text + sampled code groups through
the SHARED teacher-forcing kernel `ModelWrapper.teacher_forcing` (via
`collate`, one talker forward, one sub-talker forward) — a
port of the official SFT input pipeline verified byte-equal against the
upstream reference, with ONE correction at the embedding stage: the text
channel goes through `text_projection`, exactly as generation does. This file
only adds what RL needs on top of the kernel: temperature-scaled full-softmax
log-probs and packed masking.

This reconstruction is only valid because rollout uses language="Auto": the
generation prefill `[nothink, think_bos, think_eos]` then matches the collate
layout position-for-position (verified: max abs diff vs captured generation
logits ≈ 0.31 bf16 noise, sampled-token ranks identical).

The sampled codes exclude the EOS stop token (generation truncates it). With
collate reserving exactly one label slot beyond `codec_mask`, gathering
targets via `codec_ids[codec_mask]` covers precisely the T returned code
groups; the EOS decision is NOT part of the ratio (its label slot falls
outside `codec_mask`), matching the rollout contract.

Flat teacher-forcing structure mirrors the official finetuning teacher-forcing
script with one deliberate correction of an upstream bug: the official version
slices inputs `[:, :-1]` and selects `hidden_states[codec_mask[:, :-1]]` — the
hidden AT position p, whose input embedding already contains codec_ids[p]'s
own layer 1..15 embeddings. That leaks the prediction targets into the
conditioning and diverges from generation, where the code predictor conditions
on the PREVIOUS step's `past_hidden`. The correct pairing is
    talker_hidden = hidden[:, :-1][codec_mask[:, 1:]]   # position p-1
    targets       = codec_ids[codec_mask]               # position p
Per group, the semantic head (from logit at p-1) and each predictor head
(autoregressive over the embedded slots `[h | c_0 .. c_14]`) all evaluate at
time p — no extra shift between them.

Why all 16 codebooks instead of only the semantic head: the reward is produced
by the full rendered audio, driven by every sampled code. A policy update
moves the talker's hidden states (LoRA + codec_head live there) while the
frozen code predictor conditions on them — so even frozen-weight predictor
likelihoods shift between policy and reference. An IS ratio / KL limited to
codebook 0 under-corrects the policy movement and assigns no credit through
codebooks 1..15. Predictor WEIGHTS stay frozen at any γ; only the
training signal coverage changes.

Sampling-consistency (MD §7 缺口 #3, revised after C1v10): logits are divided
by the SAME temperatures used for sampling (`temperature` for codebook 0,
`subtalker_temperature` for codebooks 1..15) and evaluated on the FULL fp32
softmax; sampling-time top_k/suppress masks are deliberately NOT re-applied
(see `token_log_probs`). repetition_penalty is not part of the RL sampling
contract at all (sampler signatures exclude it), so the reconstruction stays
stateless: one talker forward + one sub-talker forward, temperature scaling,
done.

Output packing: log_probs/mask are [B, (L-1) * Q] where L = padded width;
column (j * Q + jb) holds codebook jb's log-prob FOR the code at global
position j+1 (P_added columns included — under `mask`=0 they are zeros).
grpo_loss consumes log_probs * mask, so no -inf * 0 NaN can occur.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from trainer.lora import LoraTrainerModel


@dataclass
class LogProbResult:
    log_probs: (
        torch.Tensor
    )  # [B, (L-1)*Q] per-code log-probs, packed (module docstring)
    mask: torch.Tensor  # [B, (L-1)*Q] 1 on valid code slots
    lengths: torch.Tensor  # [B] valid code-group counts (pre-packing)


def token_log_probs(
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
    log-probs are finite and continuous. Softmax runs in fp32 — bf16
    log_softmax quantization (~0.02/token) would otherwise dominate the small
    deltas the ratio and KL are made of."""
    logits = logits.float() / temperature
    return F.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


class LogProbComputer:
    """Rebuilds the collate batch from (texts, codes) and reads ALL codebook
    log-probs via two teacher-forcing passes (adapter on/off = policy/ref):
    one talker forward (semantic head + last-layer hiddens) feeding
    `forward_sub_talker_finetune` for the predictor heads (codebooks 1..15).
    """

    def __init__(self, ttm: LoraTrainerModel, speaker: str = "cyrene"):
        self.ttm = ttm
        self.model = ttm.model
        self.talker = ttm.model.talker

        talker_config = self.talker.config
        self.num_code_groups = talker_config.num_code_groups
        self.speaker_vec = self.talker.model.codec_embedding.weight[
            talker_config.spk_id[speaker.lower()]
        ]

    @torch.inference_mode()
    def compute_ref(
        self,
        texts: list[str],
        codes: list[torch.Tensor],
        temperature: float = 0.9,
        subtalker_temperature: float = 0.9,
        micro: int | None = None,
    ) -> LogProbResult:
        """Reference (adapter OFF) log-probs — frozen base forward, no grads.

        `micro` splits the batch into inference chunks (transient activation
        peak = one chunk) and re-pads the packed rows to a common width.
        Exact per-sequence equality in exact arithmetic: right padding +
        causal attention make valid-position logits independent of the batch
        shape; only bf16 kernel reassociation differs (~1e-3)."""
        self.ttm.set_adapter(False)
        try:
            if micro is None or micro >= len(texts):
                return self._forward(texts, codes, temperature, subtalker_temperature)
            results = [
                self._forward(texts[i : i + micro], codes[i : i + micro],
                              temperature, subtalker_temperature)
                for i in range(0, len(texts), micro)
            ]
            width = max(r.log_probs.shape[1] for r in results)
            log_probs = torch.cat(
                [F.pad(r.log_probs, (0, width - r.log_probs.shape[1])) for r in results]
            )
            mask = torch.cat(
                [F.pad(r.mask, (0, width - r.mask.shape[1])) for r in results]
            )
            lengths = torch.cat([r.lengths for r in results])
            return LogProbResult(log_probs, mask, lengths)
        finally:
            self.ttm.set_adapter(True)

    def compute_policy(
        self,
        texts: list[str],
        codes: list[torch.Tensor],
        temperature: float = 0.9,
        subtalker_temperature: float = 0.9,
    ) -> LogProbResult:
        """Policy (adapter ON) log-probs — differentiable, part of the train graph."""
        self.ttm.set_adapter(True)
        return self._forward(texts, codes, temperature, subtalker_temperature)

    def _forward(
        self,
        texts: list[str],
        codes: list[torch.Tensor],
        temperature: float,
        subtalker_temperature: float,
    ) -> LogProbResult:
        batch = self.ttm.collate(texts, codes)
        tf = self.ttm.teacher_forcing(batch, self.speaker_vec)

        b = len(codes)
        q = self.num_code_groups
        lengths = torch.tensor(
            [c.shape[0] for c in codes], device=batch.codec_ids.device
        )
        predict_mask = tf.predict_mask  # [B, L-1]: a sampled code sits at j+1

        # --- codebook 0 (semantic head): logits at j predict the code at j+1 ---
        sem_targets = torch.zeros_like(batch.codec_ids[:, :, 0])
        sem_targets[batch.codec_mask] = tf.talker_codec_ids[:, 0]
        sem_log_probs = token_log_probs(
            tf.talker_logits, sem_targets[:, 1:], temperature
        )  # [B, L-1], garbage-but-finite off the code span

        # --- codebooks 1..15 (predictor heads), conditioned on hidden at p-1 ---
        sub_log_probs = token_log_probs(
            tf.sub_talker_logits, tf.talker_codec_ids[:, 1:], subtalker_temperature
        )  # [N, Q-1]

        # --- pack [B, L-1, Q] -> [B, (L-1)*Q]; zeros off the code span ---
        n_idx, j_idx = predict_mask.nonzero(as_tuple=True)  # row-major == N order
        max_j = predict_mask.shape[1]
        sub_full = torch.zeros(b, max_j, q - 1, device=sub_log_probs.device)
        sub_full[n_idx, j_idx] = sub_log_probs

        log_probs = torch.where(
            predict_mask, sem_log_probs, torch.zeros_like(sem_log_probs)
        )
        log_probs = torch.cat(
            [
                log_probs.unsqueeze(-1),
                sub_full.masked_fill(~predict_mask.unsqueeze(-1), 0.0),
            ],
            dim=-1,
        ).view(b, -1)
        mask = predict_mask.unsqueeze(-1).expand(b, -1, q).reshape(b, -1).float()
        return LogProbResult(log_probs, mask, lengths)
