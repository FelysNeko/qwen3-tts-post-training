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
    - torch global RNG: identical (texts, seed) reproduce identical codes
      within an impl; cross-impl bit-equality holds only for `fast` vs `hf`.
    """

    def __init__(
        self,
        ttm: TrainerModel,
        speaker: str = "cyrene",
        language: str = "Auto",
    ):
        # Generation layout is pinned: language="Auto" + non-streaming (SFT
        # collate parity — the validity precondition for logprob.py).
        self.ttm = ttm
        self.speaker = speaker
        self.language = language

    def warmup_sample(
        self, text: str, batch: int, max_new_tokens: int
    ) -> list[torch.Tensor]:
        """One dummy generation at the RL contract config (seed 0, T=0.9,
        top_k=50, subtalker trio at upstream defaults); returns its codes.
        Used by the compiled/graphed impls' warmup paths — dummy generations
        must consume the same RNG stream shape as real rollouts."""
        return self.sample(
            [text] * batch,
            seed=0,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            max_new_tokens=max_new_tokens,
            subtalker_temperature=0.9,
            subtalker_top_k=50,
        )

    @abstractmethod
    def sample(
        self,
        texts: list[str],
        *,
        seed: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        max_new_tokens: int,
        subtalker_temperature: float,
        subtalker_top_k: int,
    ) -> list[torch.Tensor]:
        """Generate one code-group sequence per text. Returns list of
        [T, num_code_groups] tensors (first column = semantic tokens; the EOS
        stop token is truncated by the generation path).

        All sampling params are keyword-only WITHOUT defaults: callers state
        the full config explicitly (the RL contract lives at call sites;
        probes vary one knob at a time). ``seed`` is REQUIRED — RL rollouts
        are always seeded (reproducibility is part of the contract; unseeded
        sampling would poison the same-(texts, seed) replay guarantee).
        Params mirror upstream ``generate`` with top_p pinned to the official
        1.0, repetition_penalty pinned to None (the official 1.05 default is
        serving-only; RL rollout must stay stateless so the teacher-forcing
        logprob reconstruction holds), and ONE ``do_sample`` governing both
        the outer (semantic token) loop and the code-predictor loop — the
        split variant was never used; greedy verification wants both greedy,
        sampling wants both sampling. The loops keep separate
        temperature/top_k (a real TTS codec research dimension)."""
