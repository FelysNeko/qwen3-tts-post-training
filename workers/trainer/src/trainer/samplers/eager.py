"""Hand-rolled eager decode loop for the Qwen3-TTS talker (Auto, non-streaming).

Replaces the nested HF GenerationMixin machinery (outer ``talker.generate`` +
an inner ``code_predictor.generate`` per step) with direct module calls and
DynamicCache. The arithmetic is kept identical to the reference path so the
two are interchangeable:

- same forward call sequence and shapes: main backbone prefill [B, L] then
  [B, 1] steps; code predictor prefill [B, 2] then 14 [B, 1] steps per main
  step (DynamicCache both levels, same cache_position / position_ids math);
- same logits pipeline: cast to float32, then min_new_tokens EOS ban ->
  suppress band -> temperature -> top_k (HF processor order; greedy skips the
  warpers), sampled with multinomial on an fp32 softmax — the RNG draw order
  (15 inner + 1 outer per step) matches the reference stream, so identical
  seeds reproduce identical rollouts across both implementations;
- same mrope handling (``get_rope_index`` at prefill, arange+delta after)
  and trailing-text mixing.

Phase-0 profile (PROJECT_STATUS §9): B=8 generation spends ~61% in code
predictor forwards, ~24% main forwards, ~13% HF loop machinery, all
CPU-launch bound (GPU busy < 1 ms per forward). This module strips the
machinery layer and pins down the exact per-step op sequence as groundwork
for the torch.compile / CUDA-graph phases (see torch_compile.py /
cuda_graph.py).
"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

from trainer.model import TrainerModel
from trainer.samplers.base import Sampler, prefill_cur_len, tokenize_assistant


class EagerSampler(Sampler):
    """Drop-in replacement for ``HFSampler`` with the HF loops stripped out."""

    def __init__(
        self,
        ttm: TrainerModel,
        speaker: str = "cyrene",
        language: str = "Auto",
        batch_size: int = 8,
    ):
        assert language.lower() == "auto", "eager sampler supports the Auto (non-streaming) layout only"
        super().__init__(ttm, speaker=speaker, language=language, batch_size=batch_size)
        model = ttm.model
        self.talker = model.talker
        tc = model.config.talker_config
        self.tc = tc
        self.q = tc.num_code_groups
        self.eos = tc.codec_eos_token_id
        self.min_new = 2
        self.suppress = torch.tensor(
            [i for i in range(tc.vocab_size - 1024, tc.vocab_size) if i != self.eos],
            dtype=torch.long,
            device=ttm.device,
        )
        self.codec_emb = self.talker.get_input_embeddings()
        self.text_emb = self.talker.get_text_embeddings()
        self.text_proj = self.talker.text_projection
        self.codec_head = self.talker.codec_head
        self.cp = self.talker.code_predictor
        self.cp_model = self.cp.model
        self.cp_proj = self.cp.small_to_mtp_projection
        self.cp_emb = self.cp.get_input_embeddings()
        self.cp_heads = self.cp.lm_head

    # ------------------------------------------------------------------
    # prefill
    # ------------------------------------------------------------------

    def _build_prefill(self, texts: list[str]) -> tuple:
        """Port of ``Qwen3TTSForConditionalGeneration.generate`` lines that
        assemble talker input embeds for the Auto + speaker + non-streaming
        case (no instruct, no voice clone)."""
        dev = self.ttm.device
        cfg = self.ttm.model.config
        tc = self.tc
        dtype_ids = torch.long

        ids3 = torch.tensor(
            [[cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]],
            device=dev,
            dtype=dtype_ids,
        )
        bos_e, eos_e, pad_e = self.text_proj(self.text_emb(ids3)).chunk(
            3, dim=1
        )  # [1,1,d] each

        pre0 = self.codec_emb(
            torch.tensor(
                [[tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id]],
                device=dev,
                dtype=dtype_ids,
            )
        )
        spk_e = self.codec_emb(
            torch.tensor(
                [[tc.spk_id[self.speaker.lower()]]], device=dev, dtype=dtype_ids
            )
        )
        pre1 = self.codec_emb(
            torch.tensor(
                [[tc.codec_pad_id, tc.codec_bos_id]], device=dev, dtype=dtype_ids
            )
        )
        cie = torch.cat([pre0, spk_e, pre1], dim=1)  # [1,6,d]

        embeds, trailing = [], []
        for text in texts:
            ids = tokenize_assistant(self.ttm.processor, text).to(dev)
            role = self.text_proj(self.text_emb(ids[:, :3]))
            base = (
                torch.cat([pad_e.expand(-1, cie.shape[1] - 2, -1), bos_e], dim=1)
                + cie[:, :-1]
            )
            # `first` (text[0] + codec_bos) is computed then truncated by the
            # [:, :-1] below — faithful to upstream qwen-tts modeling_qwen3_tts.py
            # (generate, tts_text_first_token): streaming mode consumes it as real
            # input, non_streaming_mode slices it off (the full text re-enters via
            # `tail`). Dead in our Auto+non-streaming path only; kept for parity.
            first = self.text_proj(self.text_emb(ids[:, 3:4])) + cie[:, -1:]
            head = torch.cat([role, base, first], dim=1)[:, :-1]
            n = ids.shape[1] - 8  # text tokens = ids[:, 3:-5]
            tail = torch.cat(
                [self.text_proj(self.text_emb(ids[:, 3:-5])), eos_e], dim=1
            ) + self.codec_emb(
                torch.full((1, n + 1), tc.codec_pad_id, device=dev, dtype=dtype_ids)
            )
            last = pad_e + self.codec_emb(
                torch.tensor([[tc.codec_bos_id]], device=dev, dtype=dtype_ids)
            )
            embeds.append(torch.cat([head, tail, last], dim=1))
            trailing.append(pad_e)

        b = len(embeds)
        lengths = [e.shape[1] for e in embeds]
        max_len = max(lengths)
        flipped = [e.squeeze(0).flip(0) for e in embeds]
        padded = torch.nn.utils.rnn.pad_sequence(
            flipped, batch_first=True, padding_value=0.0
        )
        embeds_b = padded.flip(1)  # left padding, [B, Lmax, d]
        indices = torch.arange(max_len).expand(b, -1)
        mask = (
            (indices >= (max_len - torch.tensor(lengths)).unsqueeze(1)).long().to(dev)
        )

        pad_vec = pad_e.squeeze()
        padded_t = torch.nn.utils.rnn.pad_sequence(
            [t.squeeze(0) for t in trailing], batch_first=True, padding_value=0.0
        )
        ar = torch.arange(padded_t.shape[1], device=dev).expand(b, -1)
        lens = torch.tensor([t.shape[1] for t in trailing], device=dev).unsqueeze(1)
        padded_t[ar >= lens] = pad_vec
        trailing_b = padded_t
        return embeds_b, mask, trailing_b, pad_e

    def _prefill(self, texts: list[str]) -> tuple:
        embeds_b, mask, trailing_b, pad_e = self._build_prefill(texts)
        dev = self.ttm.device
        cache = DynamicCache()
        position_ids, rope_deltas = self.talker.get_rope_index(mask)
        rope_deltas = rope_deltas - (1 - mask).sum(-1, keepdim=True)
        out = self.talker.model(
            inputs_embeds=embeds_b,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(mask.shape[1], device=dev),
            output_hidden_states=False,
        )
        past_hidden = out.last_hidden_state[:, -1:, :]
        # full-length head for kernel-shape parity with the reference prefill
        logits = self.codec_head(out.last_hidden_state).float()[:, -1]
        return cache, mask, trailing_b, pad_e, rope_deltas, past_hidden, logits

    # ------------------------------------------------------------------
    # logits pipeline (HF parity)
    # ------------------------------------------------------------------

    def _process_outer(
        self,
        logits: torch.Tensor,
        step: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        if step < self.min_new:
            logits[:, self.eos] = float("-inf")
        logits[:, self.suppress] = float("-inf")
        if do_sample:
            logits = logits / temperature
            if top_k and 0 < top_k < logits.shape[-1]:
                kth = logits.topk(top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
        return logits

    def _process_inner(
        self, logits: torch.Tensor, do_sample: bool, temperature: float, top_k: int
    ) -> torch.Tensor:
        if do_sample:
            logits = logits / temperature
            if top_k and 0 < top_k < logits.shape[-1]:
                kth = logits.topk(top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
        return logits

    @staticmethod
    def _choose(logits: torch.Tensor, do_sample: bool) -> torch.Tensor:
        if do_sample:
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return torch.argmax(logits, dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------

    def _predictor_pass(
        self,
        past_hidden: torch.Tensor,
        last_id_hidden: torch.Tensor,
        do_sample: bool,
        st: float,
        sk: int,
    ) -> torch.Tensor:
        """Inner code predictor: prefill [B,2] + 14 cached steps -> [B, Q-1]."""
        dev = past_hidden.device
        cp_cache = DynamicCache()
        pe = self.cp_proj(torch.cat([past_hidden, last_id_hidden], dim=1))
        out = self.cp_model(
            inputs_embeds=pe,
            past_key_values=cp_cache,
            use_cache=True,
            cache_position=torch.arange(2, device=dev),
        )
        logits = self.cp_heads[0](out.last_hidden_state).float()[:, -1]
        code = self._choose(
            self._process_inner(logits, do_sample, st, sk), do_sample
        )
        codes = [code]
        for gs in range(1, self.q - 1):
            emb = self.cp_proj(self.cp_emb[gs - 1](code))
            out = self.cp_model(
                inputs_embeds=emb,
                past_key_values=cp_cache,
                use_cache=True,
                cache_position=torch.arange(gs + 1, gs + 2, device=dev),
            )
            logits = self.cp_heads[gs](out.last_hidden_state).float()[:, -1]
            code = self._choose(
                self._process_inner(logits, do_sample, st, sk), do_sample
            )
            codes.append(code)
        return torch.cat(codes, dim=1)

    def _decode_step(
        self,
        tok: torch.Tensor,
        step: int,
        cur_len: int,
        past_hidden: torch.Tensor,
        cache: DynamicCache,
        mask: torch.Tensor,
        trailing: torch.Tensor,
        pad_e: torch.Tensor,
        rope_deltas: torch.Tensor,
        do_sample: bool,
        st: float,
        sk: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = tok.device
        b = tok.shape[0]
        last_id_hidden = self.codec_emb(tok)
        seqs = self._predictor_pass(past_hidden, last_id_hidden, do_sample, st, sk)

        hiddens = [last_id_hidden] + [
            self.cp_emb[i](seqs[:, i : i + 1]) for i in range(self.q - 1)
        ]
        inputs_embeds = torch.cat(hiddens, dim=1).sum(dim=1, keepdim=True)
        if step < trailing.shape[1]:
            inputs_embeds = inputs_embeds + trailing[:, step : step + 1]
        else:
            inputs_embeds = inputs_embeds + pad_e

        delta = rope_deltas + cur_len
        pos = (
            torch.arange(1, device=dev)
            .view(1, -1)
            .expand(b, -1)
            .add(delta)
            .unsqueeze(0)
            .expand(3, -1, -1)
        )
        mask = torch.cat([mask, torch.ones(b, 1, device=dev, dtype=mask.dtype)], dim=1)
        out = self.talker.model(
            inputs_embeds=inputs_embeds,
            attention_mask=mask,
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(cur_len, cur_len + 1, device=dev),
            output_hidden_states=False,
        )
        logits = self.codec_head(out.last_hidden_state).float()[:, -1]
        return seqs, out.last_hidden_state[:, -1:, :], logits, mask

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
        """Same contract as ``Sampler.sample``: returns (codes, cur_len)."""
        torch.manual_seed(seed)
        texts = [text] * self.batch_size
        cache, mask, trailing, pad_e, rope_deltas, past_hidden, logits = self._prefill(
            texts
        )
        cur_len = prefill_cur_len(self.ttm.processor, text)
        init_cur_len = cur_len
        max_new_tokens = max(0, token_budget - cur_len)
        tok = self._choose(
            self._process_outer(logits, 0, do_sample, temperature, top_k), do_sample
        )
        rows: list[torch.Tensor] = []
        while True:
            seqs, past_hidden, logits, mask = self._decode_step(
                tok,
                len(rows),
                cur_len,
                past_hidden,
                cache,
                mask,
                trailing,
                pad_e,
                rope_deltas,
                do_sample,
                subtalker_temperature,
                subtalker_top_k,
            )
            rows.append(torch.cat([tok, seqs], dim=1))
            cur_len += 1
            step = len(rows)
            ntok = self._choose(
                self._process_outer(logits, step, do_sample, temperature, top_k),
                do_sample,
            )
            has_eos = (torch.stack([r[:, 0] for r in rows]) == self.eos).any(0)
            has_eos = has_eos | (ntok[:, 0] == self.eos)
            if bool(has_eos.all()) or step + 1 >= max_new_tokens:
                break
            tok = ntok

        all_rows = torch.stack(rows, dim=0)  # [T, B, Q]
        results: list[torch.Tensor] = []
        for i in range(all_rows.shape[1]):
            col = all_rows[:, i, 0]
            hit = (col == self.eos).nonzero()
            length = int(hit[0]) if hit.numel() else all_rows.shape[0]
            results.append(all_rows[:length, i, :])
        return results, init_cur_len
