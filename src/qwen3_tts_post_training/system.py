"""Process- and device-level system metrics shared by the worker loops —
one place so every monitor.jsonl / ScoreResponse reports identical semantics
(peak/current RSS from the OS, CUDA allocator views for GPU memory)."""

from __future__ import annotations

import resource

import torch


def peak_rss_mb() -> int:
    """Peak resident set size of this process (MB) — getrusage ru_maxrss
    (KB units on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def current_rss_mb() -> int:
    """Current resident set size of this process (MB), from /proc."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * 4096 // 2**20
    except OSError:
        return -1


def gpu_allocated_mb(device: str) -> float:
    """Memory allocated by the CUDA caching allocator on `device` (MB)."""
    return round(torch.cuda.memory_allocated(device) / 2**20, 1)


def gpu_reserved_mb(device: str) -> float:
    """Memory reserved by the CUDA caching allocator on `device` (MB)."""
    return round(torch.cuda.memory_reserved(device) / 2**20, 1)
