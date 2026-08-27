"""LoRA trainer model (GRPO backend): rsLoRA on the talker MLP + trainable
semantic head; the reference policy is the SAME weights with adapters disabled
(no second model in VRAM).

Design (MD §7 决策 3): rsLoRA r=16, α=64, MLP-only. MTP γ=0 → the code
predictor (sub-talker) + small_to_mtp_projection stay frozen; only
`talker.codec_head` + the LoRA deltas train.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from trainer.model import ModelWrapper


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


class LoraTrainerModel(ModelWrapper):
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:1",
        dtype: torch.dtype = torch.bfloat16,
        lora_r: int = 16,
        lora_alpha: float = 64,
        rsloRA: bool = True,
    ):
        super().__init__(model_path, device=device, dtype=dtype)

        # rsLoRA on the talker MLP (gate/up/down), semantic head stays trainable
        self.lora_modules = self._attach_lora(r=lora_r, alpha=lora_alpha, rsloRA=rsloRA)
        self.codec_head = self.talker.codec_head
        self._freeze_non_trainable()

    def _attach_lora(self, r: int, alpha: float, rsloRA: bool) -> list[LoRALinear]:
        modules: list[LoRALinear] = []
        talker = self.talker
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
