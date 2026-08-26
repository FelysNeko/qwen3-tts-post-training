"""Trainer-side Qwen3TTS model wrapper: load ckpt, attach rsLoRA to the talker
MLP, make the semantic head trainable, and switch the adapters on/off.

Design (MD §7 决策 3): rsLoRA r=16, α=64, MLP-only; reference policy = same
weights with adapters disabled (no second model in VRAM). MTP γ=0 → the code
predictor (sub-talker) + small_to_mtp_projection stay frozen; only the semantic
head (talker.codec_head) + LoRA are trained.

The trainable unit for GRPO is the talker (text → semantic code groups); the
speech tokenizer is the environment renderer (always frozen).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from trainer.batch import CollateBatch


@dataclass
class TeacherForcing:
    """Selected outputs of `TrainerModel.teacher_forcing` (shared SFT/GRPO).

    sem_logits [b, L-1, V]: semantic-head logits at position p-1 — the dense
    view `logits[:, :-1]`, so the shifted target `batch.codec_0_labels[:, 1:]`
    (SFT, includes EOS) or `sem_targets[:, 1:]` (GRPO) gathers directly.
    predict_mask [b, L-1]: True where a code group sits at p.
    codes_flat [N, Q]: flattened code groups (`codec_ids[codec_mask]`,
    row-major = sample order).
    sub_logits [N, Q-1, V]: predictor-head logits; head j at embedded slot
    j+1 predicts c_{j+1}, conditioned on slots up to j (+ talker hidden).
    """

    sem_logits: torch.Tensor
    predict_mask: torch.Tensor
    codes_flat: torch.Tensor
    sub_logits: torch.Tensor


class LoRALinear(nn.Module):
    """Frozen base Linear + rank-stabilized LoRA delta (rsLoRA, α/√r scaling).

    `enabled=False` reproduces the reference (base-only) forward for KL —
    adapter on/off on the SAME weights, no second model.
    """

    def __init__(
        self, base: nn.Linear, r: int = 16, alpha: float = 64, rsloRA: bool = True
    ):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / (r**0.5 if rsloRA else r)
        self.enabled = True
        dtype = base.weight.dtype
        device = base.weight.device
        self.lora_a = nn.Parameter(
            torch.empty(base.in_features, r, device=device, dtype=dtype)
        )
        self.lora_b = nn.Parameter(
            torch.empty(r, base.out_features, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)  # delta = 0 at start → identical to base
        for p in base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.enabled:
            # parameters are stored pre-transposed ([in, r] / [r, out]) so both
            # GEMMs take contiguous weight operands: a strided `.t()` view here
            # makes cuBLAS materialize the weight during CUDA-graph capture,
            # baking the value into the graph (probe C1v8: in-place optimizer
            # updates stopped reaching graphed rollouts)
            out = out + self.scaling * (x @ self.lora_a) @ self.lora_b
        return out


class TrainerModel:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:1",
        dtype: torch.dtype = torch.bfloat16,
        lora_r: int = 16,
        lora_alpha: float = 64,
        rsloRA: bool = True,
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

        # rsLoRA on the talker MLP (gate/up/down), semantic head stays trainable
        self.lora_modules = self._attach_lora(r=lora_r, alpha=lora_alpha, rsloRA=rsloRA)
        self.codec_head = self.model.talker.codec_head
        self._freeze_non_trainable()

    def _attach_lora(self, r: int, alpha: float, rsloRA: bool) -> list[LoRALinear]:
        modules: list[LoRALinear] = []
        talker = self.model.talker
        for layer in talker.model.layers:
            for name in ("gate_proj", "up_proj", "down_proj"):
                lora = LoRALinear(
                    getattr(layer.mlp, name), r=r, alpha=alpha, rsloRA=rsloRA
                )
                setattr(layer.mlp, name, lora)
                modules.append(lora)
        return modules

    def _freeze_non_trainable(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)
        for lora in self.lora_modules:
            lora.lora_a.requires_grad_(True)
            lora.lora_b.requires_grad_(True)
        self.codec_head.weight.requires_grad_(True)

    def set_adapter(self, enabled: bool) -> None:
        """Switch LoRA on/off for policy (on) vs reference (off) forwards."""
        for lora in self.lora_modules:
            lora.enabled = bool(enabled)

    def teacher_forcing(
        self, batch: CollateBatch, speaker_vec: torch.Tensor
    ) -> TeacherForcing:
        """Shared teacher-forcing kernel: embeddings → ONE talker forward →
        the sub-talker pass. Identical for SFT and GRPO (official
        `teacher_forcing` / `sft_12hz.py` structure, with the upstream
        target-leak bug fixed via the p-1 hidden selection). Consumers diverge
        only downstream:

        - SFT: CE on `sem_logits` vs `batch.codec_0_labels[:, 1:]`
          (ignore_index=-100; INCLUDES the EOS label slot — it sits outside
          `codec_mask`, hence dense logits rather than a masked select) plus
          CE on `sub_logits` vs `codes_flat[:, 1:]` at weight 0.3;
        - GRPO: temperature-scaled full-softmax log-probs (logprob.py).

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

        for k in range(1, self.talker.config.num_code_groups):
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

        predict_mask = batch.codec_mask[:, 1:]  # [b, L-1]: code at p ↔ hidden p-1
        codes_flat = batch.codec_ids[batch.codec_mask]  # [N, Q] row-major
        talker_hidden = outputs.hidden_states[0][-1][:, :-1][predict_mask]
        sub_logits, _ = self.talker.forward_sub_talker_finetune(
            codes_flat, talker_hidden
        )
        return TeacherForcing(
            sem_logits=outputs.logits[:, :-1],
            predict_mask=predict_mask,
            codes_flat=codes_flat,
            sub_logits=sub_logits,
        )

    @property
    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]
