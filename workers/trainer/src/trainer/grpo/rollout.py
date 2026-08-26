"""GRPO rollout: text prompts → code groups → 24 kHz wavs for the scorer.

Rollouts run under the current policy (LoRA adapters enabled). Each group is
seeded via torch.manual_seed so identical (prompts, seed) reproduce the same
codes. Wavs go to a per-run tmpfs dir under /dev/shm — the scorer reads them
by path, and the OS reclaims them when the process exits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from trainer.grpo.decoder import Decoder, write_wav
from trainer.grpo.samplers.base import Sampler


@dataclass
class RolloutResult:
    prompt: str
    codes: list[torch.Tensor]
    wav_paths: list[Path]
    fs: int
    cur_len: int  # prefill length for token_budget accounting (no default: must be explicit)


def rollout_group(
    sampler: Sampler,
    decoder: Decoder,
    prompt: str,
    seed: int,
    tag: str,
    *,
    temperature: float,
    top_k: int,
    token_budget: int,
    work_dir: Path | None = None,
) -> RolloutResult:
    """Sample one group and render it to wav. Returns codes + wav paths.

    ``prompt`` is a single text, internally repeated `sampler.batch_size`
    times to form the GRPO group (homogeneous, no assert needed).
    ``token_budget`` is total tokens (prefill cur_len + new) budget;
    ``max_new = token_budget - cur_len`` is derived inside the sampler.
    The non-varying part of the RL sampling contract is pinned here
    (do_sample=True governs both loops, subtalker trio at 0.9/50); probes
    that need other values call the sampler directly."""
    codes, cur_len = sampler.sample(
        prompt,
        seed=seed,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        token_budget=token_budget,
        subtalker_temperature=0.9,
        subtalker_top_k=50,
    )
    wavs, fs = decoder.decode(codes)
    if work_dir is None:
        work_dir = Path("/dev/shm") / f"grpo_{tag}"
    work_dir.mkdir(parents=True, exist_ok=True)
    wav_paths = [
        write_wav(work_dir / f"{tag}_{i}.wav", wav, fs) for i, wav in enumerate(wavs)
    ]
    return RolloutResult(prompt, codes, wav_paths, fs, cur_len=cur_len)
