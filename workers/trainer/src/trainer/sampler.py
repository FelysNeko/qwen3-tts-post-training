"""Rollout sampling: text prompt → semantic code groups (Auto language path).

`language="Auto"` uses the nothink generation prefill, which matches the SFT
training layout — so the teacher-forcing logprob reconstruction (logprob.py)
can be rebuilt purely from (text, codes) without intercepting generation
internals (verified bit-consistent vs captured generation logits).

Sampling uses torch's global RNG — seed via torch.manual_seed(seed) before
each group for reproducible rollout. repetition_penalty defaults to None
(MD §7 缺口 3: RL sampling drops rep penalty so logprob needs no history-
dependent logit rewriting).
"""

from __future__ import annotations

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


class Sampler:
    def __init__(
        self,
        ttm: TrainerModel,
        speaker: str = "cyrene",
        language: str = "Auto",
        non_streaming_mode: bool = True,
        impl: str = "hf",
    ):
        self.ttm = ttm
        self.speaker = speaker
        self.language = language
        self.non_streaming_mode = non_streaming_mode
        self.impl = impl
        if impl in ("fast", "compiled", "graphed"):
            from trainer.fastgen import FastSampler

            if impl == "graphed":
                from trainer.fastgraph import GraphFastSampler

                self._fast = GraphFastSampler(
                    ttm,
                    speaker=speaker,
                    language=language,
                    non_streaming_mode=non_streaming_mode,
                )
            else:
                self._fast = FastSampler(
                    ttm,
                    speaker=speaker,
                    language=language,
                    non_streaming_mode=non_streaming_mode,
                    compile=(impl == "compiled"),
                )
        elif impl != "hf":
            raise ValueError(f"unknown sampler impl: {impl}")

    def _tokenize(self, text: str) -> torch.Tensor:
        return tokenize_assistant(self.ttm.processor, text).to(self.ttm.device)

    @torch.inference_mode()
    def sample(
        self,
        texts: list[str],
        seed: int | None = None,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float | None = None,
        max_new_tokens: int = 4096,
    ) -> list[torch.Tensor]:
        """Generate one code-group sequence per text. Returns list of
        [T, num_code_groups] tensors (first column = semantic tokens; the EOS
        stop token is truncated by the generation path)."""
        if self.impl in ("fast", "compiled", "graphed"):
            return self._fast.sample(
                texts,
                seed=seed,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
            )
        if seed is not None:
            torch.manual_seed(seed)
        input_ids = [self._tokenize(t) for t in texts]
        n = len(texts)
        codes, _ = self.ttm.model.generate(
            input_ids=input_ids,
            instruct_ids=[None] * n,
            languages=[self.language] * n,
            speakers=[self.speaker] * n,
            non_streaming_mode=self.non_streaming_mode,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )
        return codes
