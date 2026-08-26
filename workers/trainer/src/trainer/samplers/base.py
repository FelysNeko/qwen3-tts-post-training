"""Rollout sampler ABC + shared tokenization (Auto language path).

`language="Auto"` uses the nothink generation prefill, which matches the SFT
training layout — so the teacher-forcing logprob reconstruction (logprob.py)
can be rebuilt purely from (text, codes) without intercepting generation
internals (verified bit-consistent vs captured generation logits).

Sampling uses torch's global RNG — seed via torch.manual_seed(seed) before
each group for reproducible rollout. repetition_penalty is not part of the
RL sampling contract at all (MD §7 缺口 3: stateless logprob reconstruction
requires a history-free sampling distribution; the official inference
default 1.05 is a serving-only setting).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from trainer.model import TrainerModel


def tokenize_assistant(processor, text: str) -> torch.Tensor:
    """Official `_build_assistant_text` + `_tokenize_texts` — tokenize the full
    assistant-formatted prompt. Returns [1, len] input ids (no `[:-5]` drop)."""
    prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    ids = processor(
        text=prompt,
        return_tensors="pt",
        padding=True,
    )["input_ids"]
    return ids.unsqueeze(0) if ids.dim() == 1 else ids


def prefill_cur_len(processor, text: str) -> int:
    """Prefill length `cur_len` for `token_budget` accounting.

    Single-text — `batch_size` copies are homogeneous, so no `max` needed
    (old `list[str]` variant with `max` is removed). Exact
    `eager._build_prefill` logic up to the `mask` (no forward), so `hf`
    no longer depends on `eager.py`. `cur_len` is the padded prefill length
    (text + cie/role overhead), and `max_new = token_budget - cur_len`.
    """
    # head 8 + tail n+1 + last 1 = n+10 where n = ids.shape[1]-8
    ids = tokenize_assistant(processor, text)
    return (ids.shape[1] - 8) + 10


class Sampler(ABC):
    """Abstract rollout sampler: text prompts → semantic code groups.

    Implementations (one per `--sampler-impl`, PROJECT_STATUS §9): HF
    GenerationMixin reference (hf.py), eager hand-rolled loop (eager.py),
    torch.compile variant (torch_compile.py), CUDA-graph decode
    (cuda_graph.py). Shared contract:

    - `sample` returns one [T, num_code_groups] tensor per text (first
      column = semantic tokens; the EOS stop token is truncated by the
      generation path);
    - Auto + non-streaming prefill layout (SFT parity — the validity
      precondition for logprob.py, see module docstring);
    - torch global RNG: identical (text, seed) reproduce identical codes
      within an impl; cross-impl bit-equality holds only for `fast` vs `hf`.
    - Batching is internal: the single `text` is repeated `batch_size` times
      to form the GRPO group (no heterogeneous prompts, no assert needed).
    """

    def __init__(
        self,
        ttm: TrainerModel,
        speaker: str = "cyrene",
        language: str = "Auto",
        batch_size: int = 8,
    ):
        # Generation layout is pinned: language="Auto" + non-streaming (SFT
        # collate parity — the validity precondition for logprob.py).
        self.ttm = ttm
        self.speaker = speaker
        self.language = language
        self.batch_size = batch_size

    def warmup_sample(
        self, text: str, token_budget: int
    ) -> list[torch.Tensor]:
        """One dummy generation at the RL contract config (seed 0, T=0.9,
        top_k=50, subtalker trio at upstream defaults); returns its codes.
        Uses the sampler's fixed `batch_size` (the GRPO group size).
        ``token_budget`` is total tokens (prefill cur_len + new) budget."""
        codes, _ = self.sample(
            text,
            seed=0,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            token_budget=token_budget,
            subtalker_temperature=0.9,
            subtalker_top_k=50,
        )
        return codes

    @staticmethod
    def build(
        ttm: TrainerModel,
        impl: str = "hf",
        speaker: str = "cyrene",
        language: str = "Auto",
        batch_size: int = 8,
        lmax: int = 1024,
    ) -> Sampler:
        """Factory — `impl` selects the sampler (lazy imports, no eager deps)."""
        match impl:
            case "hf":
                from trainer.samplers.hf import HFSampler

                return HFSampler(
                    ttm, speaker=speaker, language=language, batch_size=batch_size
                )
            case "fast":
                from trainer.samplers.eager import EagerSampler

                return EagerSampler(
                    ttm, speaker=speaker, language=language, batch_size=batch_size
                )
            case "compiled":
                from trainer.samplers.torch_compile import TorchCompileSampler

                return TorchCompileSampler(
                    ttm, speaker=speaker, language=language, batch_size=batch_size
                )
            case "graphed":
                from trainer.samplers.cuda_graph import CudaGraphSampler

                return CudaGraphSampler(
                    ttm,
                    speaker=speaker,
                    language=language,
                    batch_size=batch_size,
                    lmax=lmax,
                )
            case _:
                raise ValueError(f"unknown sampler impl: {impl}")

    @abstractmethod
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
    ) -> tuple[list[torch.Tensor], int]:
        """Generate one code-group sequence per group item. Returns
        ``(codes, cur_len)`` where ``codes`` is list of [T, num_code_groups]
        tensors (first column = semantic tokens; EOS truncated) and ``cur_len``
        is prefill length (mask.shape[1]) for token_budget accounting.

        `text` is a single prompt, internally repeated `batch_size` times to
        form the GRPO group — no heterogeneous list, no assert needed.

        All sampling params are keyword-only WITHOUT defaults: callers state
        the full config explicitly (the RL contract lives at call sites;
        probes vary one knob at a time). ``seed`` is REQUIRED — RL rollouts
        are always seeded (reproducibility is part of the contract; unseeded
        sampling would poison the same-(text, seed) replay guarantee).
        Params mirror upstream ``generate`` with top_p pinned to the official
        1.0, repetition_penalty pinned to None (the official 1.05 default is
        serving-only; RL rollout must stay stateless so the teacher-forcing
        logprob reconstruction holds), and ONE ``do_sample`` governing both
        the outer (semantic token) loop and the code-predictor loop — the
        split variant was never used; greedy verification wants both greedy,
        sampling wants both sampling. The loops keep separate
        temperature/top_k (a real TTS codec research dimension).
        ``token_budget`` is total tokens (prefill cur_len + new) budget;
        effective ``max_new = token_budget - cur_len`` (replaces
        max_new_tokens/lmax/runaway_t_max, ``AGENTS.md`` token_budget)."""
