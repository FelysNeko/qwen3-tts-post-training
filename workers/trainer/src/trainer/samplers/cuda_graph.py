"""CUDA-graph decode path for the Qwen3-TTS talker (Phase 3, `impl="graphed"`).

Moves the hot decode loop off Python-driven per-kernel launches and onto one
captured CUDA graph per model (main backbone + code predictor). Attention uses
``flash_attn_with_kvcache`` — the same FA2-native decode kernel vLLM-omni's
PagedAttention backend is built on — with a pre-allocated contiguous KV pool
and a content-variable ``cache_seqlens`` tensor, so no padding ever enters the
kernel. The pooling trick is valid because a GRPO group rolls out the SAME text
8 times: every row advances ``cache_seqlens`` together each step.

- Main backbone: prefill (eager, writes KV into the pool at seqlens 0..L-1)
  then one captured 1-token decode step per generation step.
- Code predictor: prefill [B,2] (eager) + 14 captured 1-token steps per main
  step, its own small pool (LMAX = num_code_groups).
- Sampling stays eager outside the graphs (same multinomial draw order as
  the eager path), EOS via a tiny D2H check, ``cache_seqlens`` bounds every row so
  stale pool bytes are never read. LoRA adapter is ON for capture+replay;
  optimizer in-place weight updates are picked up by replay (same addresses).

Numerics: ``flash_attn_with_kvcache`` agrees with ``flash_attn_func`` within
~0.004 (bf16) — a different kernel family than the eager path
(DynamicCache + HF FA2), so this path is NOT bit-equal to `fast`/`compiled`;
its contract is same-seed self-reproducibility + distribution-level
equivalence, verified by the probe C1v4 (graph capture + replay + multi-step
in-place growth all match the reference).
"""

from __future__ import annotations

import torch
from flash_attn import flash_attn_with_kvcache
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSAttention,
    Qwen3TTSTalkerAttention,
    apply_multimodal_rotary_pos_emb,
    apply_rotary_pos_emb,
)
from transformers.cache_utils import Cache

from trainer.model import TrainerModel
from trainer.samplers.eager import EagerSampler


class StaticKVCache(Cache):
    """Fixed-address contiguous KV pool; attention writes/reads it in-kernel.

    ``k_pool`` / ``v_pool`` are ``[n_layers, batch, lmax, kv_heads, head_dim]``;
    ``seqlens`` is the content-variable ``[batch]`` int32 tensor the graph
    updates each replay. Subclasses transformers ``Cache`` so the model's
    ``isinstance(past_key_values, Cache)`` guard passes.
    """

    def __init__(
        self,
        n_layers: int,
        batch: int,
        lmax: int,
        kv_heads: int,
        head_dim: int,
        device,
        dtype,
    ):
        shape = (batch, lmax, kv_heads, head_dim)
        self.k_pool = [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)
        ]
        self.v_pool = [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)
        ]
        self.seqlens = torch.zeros(batch, dtype=torch.int32, device=device)
        self.lmax = lmax

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return key_states, value_states

    def get_mask_sizes(self, cache_position=None, layer_idx=0):
        return self.lmax, 0

    def get_seq_length(self, layer_idx=0):
        return int(self.seqlens.max().item()) if self.seqlens.numel() else 0

    def get_max_cache_shape(self, layer_idx=0):
        return self.lmax

    def reset(self):
        self.seqlens.zero_()

    def reorder_cache(self, beam_idx):
        pass


def _static_attention(self, hidden_states, position_embeddings, cache):
    """Faithful to ``Qwen3TTSTalkerAttention.forward`` up to the cache write,
    then delegates to ``flash_attn_with_kvcache`` (write + attention, one
    kernel). KV is cached post-RoPE exactly like the DynamicCache path."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(
        1, 2
    )
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(
        1, 2
    )
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        self.rope_scaling["mrope_section"],
        self.rope_scaling["interleaved"],
    )
    out = flash_attn_with_kvcache(
        query_states.transpose(1, 2).contiguous(),
        cache.k_pool[self.layer_idx],
        cache.v_pool[self.layer_idx],
        key_states.transpose(1, 2).contiguous(),
        value_states.transpose(1, 2).contiguous(),
        cache_seqlens=cache.seqlens,
        causal=True,
        softmax_scale=self.scaling,
    )  # [B, L, nheads, hd] — FA layout is already seq-major: reshape
    # directly (a .transpose(1, 2) here scrambles heads into seq for L > 1;
    # it is invisible for L == 1, so decode replays self-reproduce while
    # multi-token prefills corrupt — probe C1v8)
    attn_output = out.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), None


def _static_attention_cp(self, hidden_states, position_embeddings, cache):
    """Code-predictor variant of ``_static_attention``.

    The code predictor uses a DIFFERENT attention class (``Qwen3TTSAttention``,
    plain RoPE) than the talker; without this second patch its forwards fell
    through to the original code, where ``StaticKVCache.update`` is a no-op —
    every 1-token decode step attended to itself only and the pool was never
    written (probe C1v8: step-2 hidden diverged by 12.4)."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(
        1, 2
    )
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(
        1, 2
    )
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    out = flash_attn_with_kvcache(
        query_states.transpose(1, 2).contiguous(),
        cache.k_pool[self.layer_idx],
        cache.v_pool[self.layer_idx],
        key_states.transpose(1, 2).contiguous(),
        value_states.transpose(1, 2).contiguous(),
        cache_seqlens=cache.seqlens,
        causal=True,
        softmax_scale=self.scaling,
    )
    attn_output = out.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), None


def install_attention_patch() -> None:
    """Install the graph-mode attention dispatch once per process (idempotent).

    Both attention classes must be patched: the talker backbone uses
    ``Qwen3TTSTalkerAttention`` while the code predictor uses
    ``Qwen3TTSAttention`` — a patch on only one leaves the other
    context-free (no cache reads/writes, probe C1v8).

    Idempotence via a class sentinel (same pattern as ``enable_compile``):
    re-install would capture the patched function as "original" and the
    fallback path would recurse into itself. The original forward is bound
    as a default arg of the closure, not a module global."""
    for cls, graph_impl in (
        (Qwen3TTSTalkerAttention, _static_attention),
        (Qwen3TTSAttention, _static_attention_cp),
    ):
        if getattr(cls, "_q3tts_graph_attn", False):
            continue
        orig = cls.forward

        def patched_forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values=None,
            cache_position=None,
            _orig=orig,
            _graph=graph_impl,
            **kwargs,
        ):
            if isinstance(past_key_values, StaticKVCache):
                return _graph(self, hidden_states, position_embeddings, past_key_values)
            return _orig(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                cache_position,
                **kwargs,
            )

        cls.forward = patched_forward
        cls._q3tts_graph_attn = True


def capture_graph(fn, out_buf: torch.Tensor, probe_fn) -> torch.cuda.CUDAGraph:
    """Warm the workload, capture it into a CUDA graph, and verify the replay
    reproduces the eager output (catches corrupt captures) and responds to
    input-content changes (catches empty captures — WSL2 secondary-GPU
    default-stream bug, probe C1v7).

    MUST capture on an explicit stream: torch.cuda.graph() with the default
    stream records an EMPTY graph on the secondary GPU (cuda:1 / RTX 5070 Ti,
    WSL2). vLLM sidesteps the same bug by always capturing on a dedicated
    stream. Capture failures here are deterministic environment/code bugs,
    not transient states — they surface immediately instead of retrying."""
    dev = out_buf.device
    capture_stream = torch.cuda.Stream(device=dev)
    with torch.inference_mode(), torch.cuda.stream(capture_stream):
        for _ in range(3):
            fn()
        torch.cuda.synchronize(dev)

        eager = fn().clone()
        torch.cuda.synchronize(dev)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=capture_stream):
            out_buf.copy_(fn())
        torch.cuda.synchronize(dev)

    out_buf.zero_()
    g.replay()
    torch.cuda.synchronize(dev)
    assert torch.equal(out_buf, eager), "graph replay diverged from eager output"

    # Safety net: perturb an input buffer and check the replay output
    # changes — catches empty/corrupt captures.
    probe_baseline = out_buf.clone()
    probe_fn()
    torch.cuda.synchronize(dev)
    g.replay()
    torch.cuda.synchronize(dev)
    assert not torch.equal(out_buf, probe_baseline), "empty graph (capture recorded nothing); use --sampler-impl compiled"
    return g


class CudaGraphSampler(EagerSampler):
    """CUDA-graph rollout sampler. Fixed batch size (GRPO group); mismatched
    batch sizes assert — switch to ``EagerSampler`` for arbitrary batches."""

    def __init__(
        self,
        ttm: TrainerModel,
        speaker: str = "cyrene",
        language: str = "Auto",
        batch_size: int = 8,
        lmax: int = 1024,
    ):
        super().__init__(ttm, speaker=speaker, language=language)
        install_attention_patch()
        self.batch = batch_size
        self.lmax = lmax
        dev = ttm.device
        self.main_cache = StaticKVCache(
            self.tc.num_hidden_layers,
            batch_size,
            lmax,
            self.tc.num_key_value_heads,
            self.tc.head_dim,
            dev,
            ttm.dtype,
        )
        # Capture main BEFORE allocating the cp pool: fresh CUDA memory around
        # capture can shift cuBLAS workspace allocation into the capture window
        # ("operation not permitted when stream is capturing"). Empirically
        # (probe C1v5) main capture succeeds only when no new device allocation
        # happens between warmup and capture.
        self._capture_main()
        self.cp_cache = StaticKVCache(
            self.cp_model.config.num_hidden_layers,
            batch_size,
            self.q,
            self.cp_model.config.num_key_value_heads,
            self.cp_model.config.head_dim,
            dev,
            ttm.dtype,
        )
        self._capture_cp()
        self._warmup_graph()

    # ------------------------------------------------------------------
    # graph capture
    # ------------------------------------------------------------------

    def _capture_main(self) -> None:
        dev = self.ttm.device
        h = self.tc.hidden_size
        self.main_emb_buf = torch.zeros(
            self.batch, 1, h, dtype=self.ttm.dtype, device=dev
        )
        self.main_pos_buf = torch.zeros(3, self.batch, 1, dtype=torch.long, device=dev)
        self.main_cp_buf = torch.zeros(1, dtype=torch.long, device=dev)
        self.main_out_buf = torch.zeros(
            self.batch, 1, h, dtype=self.ttm.dtype, device=dev
        )

        def run_main():
            return self.talker.model(
                inputs_embeds=self.main_emb_buf,
                attention_mask=None,
                position_ids=self.main_pos_buf,
                past_key_values=self.main_cache,
                use_cache=True,
                cache_position=self.main_cp_buf,
                output_hidden_states=False,
            ).last_hidden_state

        self.main_graph = capture_graph(
            run_main, self.main_out_buf, lambda: self.main_emb_buf.normal_()
        )

    def _capture_cp(self) -> None:
        dev = self.ttm.device
        cp_h = self.cp_model.config.hidden_size
        self.cp_emb_buf = torch.zeros(
            self.batch, 1, cp_h, dtype=self.ttm.dtype, device=dev
        )
        self.cp_cp_buf = torch.zeros(1, dtype=torch.long, device=dev)
        self.cp_out_buf = torch.zeros(
            self.batch, 1, cp_h, dtype=self.ttm.dtype, device=dev
        )

        def run_cp():
            return self.cp_model(
                inputs_embeds=self.cp_emb_buf,
                past_key_values=self.cp_cache,
                use_cache=True,
                cache_position=self.cp_cp_buf,
                output_hidden_states=False,
            ).last_hidden_state

        self.cp_graph = capture_graph(
            run_cp, self.cp_out_buf, lambda: self.cp_emb_buf.normal_()
        )

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------

    def _graph_prefill(self, texts: list[str]) -> tuple:
        """Prefill the main StaticKVCache with ONLY the valid text tokens.

        ``_build_prefill`` left-pads to a fixed length; padding must never
        reach the pool, because ``flash_attn_with_kvcache`` (causal) attends
        over every cached position and cannot mask padding. Since GRPO rolls
        the same text out group_size times, text_len is uniform, so we drop
        the left pad and prefill [B, text_len] directly into pool positions
        0..text_len-1."""
        embeds_b, mask, trailing_b, pad_e = self._build_prefill(texts)
        dev = self.ttm.device
        text_len = int(mask.sum(-1).max().item())
        # FA2 kvcache writes past the pool unchecked — an oversized prefill is
        # an out-of-bounds write, not a Python error.
        assert text_len < self.lmax, f"prefill {text_len} >= lmax {self.lmax}"
        position_ids, rope_deltas = self.talker.get_rope_index(mask)
        rope_deltas = rope_deltas - (1 - mask).sum(-1, keepdim=True)
        valid = embeds_b[:, -text_len:, :]
        self.main_cache.seqlens.zero_()
        out = self.talker.model(
            inputs_embeds=valid,
            attention_mask=None,
            position_ids=position_ids[:, :, -text_len:],
            past_key_values=self.main_cache,
            use_cache=True,
            cache_position=torch.arange(text_len, device=dev),
            output_hidden_states=False,
        )
        self.main_cache.seqlens.fill_(text_len)
        past_hidden = out.last_hidden_state[:, -1:, :]
        logits = self.codec_head(out.last_hidden_state).float()[:, -1]
        return mask, trailing_b, pad_e, rope_deltas, past_hidden, logits, text_len

    def _predictor_pass_graph(
        self,
        past_hidden: torch.Tensor,
        last_id_hidden: torch.Tensor,
        do_sample: bool,
        st: float,
        sk: int,
    ) -> torch.Tensor:
        dev = past_hidden.device
        self.cp_cache.seqlens.zero_()
        pe = self.cp_proj(torch.cat([past_hidden, last_id_hidden], dim=1))
        out = self.cp_model(
            inputs_embeds=pe,
            past_key_values=self.cp_cache,
            use_cache=True,
            cache_position=torch.arange(2, device=dev),
            output_hidden_states=False,
        )
        logits = self.cp_heads[0](out.last_hidden_state).float()[:, -1]
        code = self._choose(self._process_inner(logits, do_sample, st, sk), do_sample)
        codes = [code]
        self.cp_cache.seqlens.fill_(2)
        for gs in range(1, self.q - 1):
            self.cp_emb_buf.copy_(self.cp_proj(self.cp_emb[gs - 1](code)))
            self.cp_cp_buf.fill_(gs + 1)
            self.cp_cache.seqlens.fill_(gs + 1)
            self.cp_graph.replay()
            logits = self.cp_heads[gs](self.cp_out_buf).float()[:, -1]
            code = self._choose(self._process_inner(logits, do_sample, st, sk), do_sample)
            codes.append(code)
        return torch.cat(codes, dim=1)

    def _main_step_graph(
        self,
        tok: torch.Tensor,
        step: int,
        cur_len: int,
        past_hidden: torch.Tensor,
        mask: torch.Tensor,
        trailing: torch.Tensor,
        pad_e: torch.Tensor,
        rope_deltas: torch.Tensor,
        do_sample: bool,
        temperature: float,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = tok.device
        b = tok.shape[0]
        last_id_hidden = self.codec_emb(tok)
        seqs = self._predictor_pass_graph(
            past_hidden, last_id_hidden, do_sample, temperature, top_k
        )
        hiddens = [last_id_hidden] + [
            self.cp_emb[i](seqs[:, i : i + 1]) for i in range(self.q - 1)
        ]
        inputs_embeds = torch.cat(hiddens, dim=1).sum(dim=1, keepdim=True)
        if step < trailing.shape[1]:
            inputs_embeds = inputs_embeds + trailing[:, step : step + 1]
        else:
            inputs_embeds = inputs_embeds + pad_e

        self.main_emb_buf.copy_(inputs_embeds)
        delta = rope_deltas + cur_len
        pos = (
            torch.arange(1, device=dev)
            .view(1, -1)
            .expand(b, -1)
            .add(delta)
            .unsqueeze(0)
            .expand(3, -1, -1)
        )
        self.main_pos_buf.copy_(pos)
        self.main_cp_buf.fill_(cur_len)
        self.main_cache.seqlens.fill_(cur_len)
        self.main_graph.replay()
        past_hidden = self.main_out_buf
        logits = self.codec_head(self.main_out_buf).float()[:, 0]
        return seqs, past_hidden, logits, mask

    @torch.inference_mode()
    def sample(
        self,
        texts: list[str],
        *,
        seed: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        token_budget: int,
        subtalker_temperature: float,
        subtalker_top_k: int,
    ) -> tuple[list[torch.Tensor], int]:
        """Same contract as ``EagerSampler.sample``. Batch size must equal
        ``batch_size`` (the captured graph shape); use ``EagerSampler``
        directly for anything else.
        ``token_budget`` is total tokens (prefill cur_len + new) budget;
        effective ``max_new = token_budget - cur_len`` clamped by ``lmax``."""
        assert len(texts) == self.batch, f"graphed sampler is fixed batch={self.batch}; use --sampler-impl fast"
        torch.manual_seed(seed)

        (
            mask,
            trailing,
            pad_e,
            rope_deltas,
            past_hidden,
            logits,
            text_len,
        ) = self._graph_prefill(texts)
        cur_len = text_len
        init_cur_len = cur_len
        max_new_tokens = max(0, token_budget - cur_len)
        max_new_tokens = min(max_new_tokens, self.lmax - cur_len)
        tok = self._choose(
            self._process_outer(logits, 0, do_sample, temperature, top_k), do_sample
        )
        rows: list[torch.Tensor] = []
        while True:
            seqs, past_hidden, logits, _ = self._main_step_graph(
                tok,
                len(rows),
                cur_len,
                past_hidden,
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

    def _warmup_graph(self) -> None:
        """One tiny graph-path generation + self-repro sanity check."""
        a = self.warmup_sample("你好。", self.batch, token_budget=64)
        b = self.warmup_sample("你好。", self.batch, token_budget=64)
        assert all(torch.equal(x, y) for x, y in zip(a, b)), "graph sampler failed self-reproducibility check"
