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

from trainer.decoder import Decoder, write_wav
from trainer.samplers import Sampler


@dataclass
class RolloutResult:
    prompts: list[str]
    codes: list[torch.Tensor]
    wav_paths: list[Path]
    fs: int


def rollout_group(
    sampler: Sampler,
    decoder: Decoder,
    prompts: list[str],
    seed: int,
    tag: str,
    *,
    temperature: float,
    top_k: int,
    max_new_tokens: int,
    work_dir: Path | None = None,
) -> RolloutResult:
    """Sample one group and render it to wav. Returns codes + wav paths.

    The non-varying part of the RL sampling contract is pinned here
    (do_sample=True governs both loops, subtalker trio at 0.9/50); probes
    that need other values call the sampler directly."""
    codes = sampler.sample(
        prompts,
        seed=seed,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
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
    return RolloutResult(prompts, codes, wav_paths, fs)
