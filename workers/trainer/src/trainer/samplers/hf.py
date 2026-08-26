"""Reference sampler (`impl="hf"`): the stock HF GenerationMixin path."""

from __future__ import annotations

import torch

from trainer.samplers.base import Sampler, prefill_cur_len, tokenize_assistant


class HFSampler(Sampler):
    """The reference rollout: outer ``talker.generate`` + an inner
    ``code_predictor.generate`` per step, exactly as upstream ships it. The
    faster impls are validated against this path — bit-equality for ``fast``,
    distribution-level equivalence for ``compiled``/``graphed``."""

    def _tokenize(self, text: str) -> torch.Tensor:
        return tokenize_assistant(self.ttm.processor, text).to(self.ttm.device)

    @torch.inference_mode()
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
        """Same contract as ``Sampler.sample`` (see base class)."""
        torch.manual_seed(seed)
        cur_len = prefill_cur_len(self.ttm.processor, text)
        max_new_tokens = max(0, token_budget - cur_len)
        input_ids = [self._tokenize(text) for _ in range(self.batch_size)]
        codes, _ = self.ttm.model.generate(
            input_ids=input_ids,
            instruct_ids=[None] * self.batch_size,
            languages=[self.language] * self.batch_size,
            speakers=[self.speaker] * self.batch_size,
            non_streaming_mode=True,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            # pinned: upstream's default is 1.05 (serving config); RL rollout
            # must sample from a stateless distribution (logprob parity).
            repetition_penalty=None,
            max_new_tokens=max_new_tokens,
            subtalker_dosample=do_sample,
            subtalker_temperature=subtalker_temperature,
            subtalker_top_k=subtalker_top_k,
        )
        return codes, cur_len
