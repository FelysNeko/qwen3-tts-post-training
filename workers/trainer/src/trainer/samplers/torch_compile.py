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

from trainer.model import TrainerModel
from trainer.samplers.eager import EagerSampler


def enable_compile(ttm: TrainerModel) -> bool:
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
        ttm: TrainerModel,
        speaker: str = "cyrene",
        language: str = "Auto",
        batch_size: int = 8,
    ):
        super().__init__(ttm, speaker=speaker, language=language)
        enable_compile(ttm)
        self.batch = batch_size
        self._warmup()

    def _warmup(self) -> None:
        """Two short dummy generations of different lengths at the fixed
        batch size.

        Absorbs dynamo's static->dynamic graph promotions for the sequence
        AND the batch dimension (Step-0 probes) so real ``sample`` calls
        always run on the settled (bitwise self-reproducible) path. Cold
        cost is ~2 min (compile), then seconds.
        """
        self.sample(["你好。"] * self.batch, seed=0, max_new_tokens=32)
        self.sample(
            ["这是一段用于预热的稍长文本，用来触发动态形状图的编译与稳定。"] * self.batch,
            seed=0,
            max_new_tokens=32,
        )

    def sample(
        self,
        texts: list[str],
        seed: int | None = None,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        max_new_tokens: int = 4096,
        subtalker_do_sample: bool | None = None,
        subtalker_temperature: float = 0.9,
        subtalker_top_k: int = 50,
    ) -> list[torch.Tensor]:
        """Same contract as ``EagerSampler.sample``. Batch size must equal
        ``batch_size`` (the warmed compile shape); use ``EagerSampler``
        directly for anything else."""
        assert len(texts) == self.batch, f"compiled sampler is fixed batch={self.batch}; use --sampler-impl fast"
        return super().sample(
            texts,
            seed=seed,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            subtalker_do_sample=subtalker_do_sample,
            subtalker_temperature=subtalker_temperature,
            subtalker_top_k=subtalker_top_k,
        )
