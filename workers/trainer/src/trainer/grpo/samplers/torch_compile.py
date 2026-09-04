"""torch.compile variant of the eager decode loop (Phase 2, `impl="compiled"`).

Wraps the two backbone forwards of the eager sampler path (main talker + code
predictor) with ``torch.compile(dynamic=None, options={"epilogue_fusion":
False})`` — ~2.4x over eager; epilogue fusion off because RMSNorm/RoPE
epilogue fusions lose fp32 precision, compounding over AR steps (vllm-omni's
finding, matching AGENTS.md's torch.compile notes).

Determinism contract, from the Step-0 probes: the first call walks a
static->dynamic graph promotion and drifts from later runs, so __init__ warms
up with two short dummy generations of different lengths; after warmup
same-seed runs are bitwise self-reproducible and no per-length recompiles
occur. Compiled kernels legitimately drift vs eager (inductor float
reassociation, greedy argmax flips at step 0) — bit-equality vs the HF path
only holds for the plain eager ``EagerSampler``.
"""

from __future__ import annotations

import torch

from trainer.grpo.samplers.eager import EagerSampler
from trainer.lora import LoraTrainerModel


def enable_compile(ttm: LoraTrainerModel) -> bool:
    """Wrap the two backbone forwards with torch.compile (idempotent).

    Returns True if this call installed the wrappers. ``dynamic=None``:
    first trace is static, later shapes promote to dynamic graphs (verified:
    no per-length recompile storm). Dynamo cache limit is raised because the
    LoRA on/off guard variants alone account for ~200 graphs.
    """
    talker = ttm.model.talker
    if getattr(talker.model, "_q3tts_compiled", False):
        return False
    torch._dynamo.config.cache_size_limit = 256
    talker.model.forward = torch.compile(
        talker.model.forward, dynamic=None, options={"epilogue_fusion": False}
    )
    talker.code_predictor.model.forward = torch.compile(
        talker.code_predictor.model.forward,
        dynamic=None,
        options={"epilogue_fusion": False},
    )
    talker.model._q3tts_compiled = True
    return True


class TorchCompileSampler(EagerSampler):
    """``EagerSampler`` with the two backbone forwards behind torch.compile.

    Fixed batch size (GRPO group), mirroring ``CudaGraphSampler``; mismatched
    batch sizes assert — use ``EagerSampler`` for arbitrary batches."""

    def __init__(
        self,
        ttm: LoraTrainerModel,
        language: str = "Auto",
        batch_size: int = 8,
    ):
        super().__init__(ttm, language=language, batch_size=batch_size)
        enable_compile(ttm)
        self._warmup()

    def _warmup(self) -> None:
        """Two short dummy generations of different lengths at the fixed
        batch size.

        Absorbs dynamo's static->dynamic graph promotions for the sequence
        AND the batch dimension (Step-0 probes) so real ``sample`` calls
        always run on the settled (bitwise self-reproducible) path. Cold
        cost is ~2 min (compile), then seconds.
        """
        self.warmup_sample("你好。", token_budget=64)
        self.warmup_sample(
            "这是一段用于预热的稍长文本，用来触发动态形状图的编译与稳定。",
            token_budget=96,
        )

    def sample(
        self,
        text: str,
        *,
        seed: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        token_budget: int,
        subtalker_temperature: float,
        subtalker_top_k: int,
        speaker: str,
    ) -> tuple[list[torch.Tensor], int]:
        """Same contract as ``EagerSampler.sample``."""
        return super().sample(
            text,
            seed=seed,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            token_budget=token_budget,
            subtalker_temperature=subtalker_temperature,
            subtalker_top_k=subtalker_top_k,
            speaker=speaker,
        )
