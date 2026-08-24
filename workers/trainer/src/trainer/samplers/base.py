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

ASSISTANT_PREFIX = "<|im_start|>assistant\n"
ASSISTANT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


def tokenize_assistant(processor, text: str) -> torch.Tensor:
    """Official `_build_assistant_text` + `_tokenize_texts` — tokenize the full
    assistant-formatted prompt. Returns [1, len] input ids (no `[:-5]` drop)."""
    ids = processor(
        text=ASSISTANT_PREFIX + text + ASSISTANT_SUFFIX,
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

    @abstractmethod
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
        """Generate one code-group sequence per text. Returns list of
        [T, num_code_groups] tensors (first column = semantic tokens; the EOS
        stop token is truncated by the generation path).

        Sampling params mirror upstream ``generate`` with top_p pinned to the
        official 1.0 and repetition_penalty pinned to None (the official
        1.05 default is serving-only; RL rollout must stay stateless so the
        teacher-forcing logprob reconstruction holds): the outer (semantic
        token) loop uses temperature/top_k; the code-predictor loop uses its
        own subtalker_* trio (defaults = upstream defaults 0.9/50;
        ``subtalker_do_sample=None`` inherits the outer ``do_sample``)."""
