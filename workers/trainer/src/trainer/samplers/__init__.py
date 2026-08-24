"""Rollout sampler implementations, one per `--sampler-impl` choice
(PROJECT_STATUS §9): `hf` (HF GenerationMixin reference), `fast` (eager
hand-rolled loop), `compiled` (fast + torch.compile backbones), `graphed`
(CUDA-graph decode). `build_sampler` is the factory used by the training
loop; impl modules are imported lazily so an hf-only run never loads
eager/torch_compile/cuda_graph."""

from __future__ import annotations

from typing import TYPE_CHECKING

from trainer.samplers.base import Sampler, tokenize_assistant

if TYPE_CHECKING:
    from trainer.model import TrainerModel

__all__ = [
    "CudaGraphSampler",
    "EagerSampler",
    "HFSampler",
    "Sampler",
    "TorchCompileSampler",
    "build_sampler",
    "tokenize_assistant",
]


def build_sampler(
    ttm: TrainerModel,
    impl: str = "hf",
    speaker: str = "cyrene",
    language: str = "Auto",
) -> Sampler:
    """Instantiate the sampler named by ``impl`` (see module docstring)."""
    if impl == "hf":
        from trainer.samplers.hf import HFSampler

        return HFSampler(ttm, speaker=speaker, language=language)
    if impl == "fast":
        from trainer.samplers.eager import EagerSampler

        return EagerSampler(ttm, speaker=speaker, language=language)
    if impl == "compiled":
        from trainer.samplers.torch_compile import TorchCompileSampler

        return TorchCompileSampler(ttm, speaker=speaker, language=language)
    if impl == "graphed":
        from trainer.samplers.cuda_graph import CudaGraphSampler

        return CudaGraphSampler(ttm, speaker=speaker, language=language)
    raise ValueError(f"unknown sampler impl: {impl}")


def __getattr__(name: str):
    """Lazy class re-exports (keep hf-only runs free of the fast-path deps)."""
    if name == "HFSampler":
        from trainer.samplers.hf import HFSampler

        return HFSampler
    if name == "EagerSampler":
        from trainer.samplers.eager import EagerSampler

        return EagerSampler
    if name == "TorchCompileSampler":
        from trainer.samplers.torch_compile import TorchCompileSampler

        return TorchCompileSampler
    if name == "CudaGraphSampler":
        from trainer.samplers.cuda_graph import CudaGraphSampler

        return CudaGraphSampler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
