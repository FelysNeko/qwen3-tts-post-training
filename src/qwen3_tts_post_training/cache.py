"""The preprocess cache layout (`.cache/{lang}/` — AGENTS metrics.json
contract) as a single source of path knowledge: preprocess stages, GRPO
calibration, the SFT speaker reference and dataset loader, and probes all
resolve the directory through `CacheLayout` instead of hand-rolling
sibling-path logic. Layout is properties; derived artifacts (metrics.json
values, centroid, medoid ref, text/codes pairs) load through methods."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from qwen3_tts_post_training.reward.reward import RewardConfig


@dataclass(frozen=True)
class CacheLayout:
    """One preprocessed pool directory (e.g. `.cache/Chinese(PRC)`)."""

    cache_namespace_dir: Path

    # ------------------------------------------------------------ layout
    @property
    def enhanced_dir(self) -> Path:
        return self.cache_namespace_dir / "enhanced"

    @property
    def codes_dir(self) -> Path:
        return self.cache_namespace_dir / "codes"

    @property
    def embedding_dir(self) -> Path:
        return self.cache_namespace_dir / "embedding"

    @property
    def centroid_npy(self) -> Path:
        return self.cache_namespace_dir / "centroid.npy"

    @property
    def asset_jsonl(self) -> Path:
        return self.cache_namespace_dir / "asset.jsonl"

    @property
    def metrics_json(self) -> Path:
        return self.cache_namespace_dir / "metrics.json"

    # ------------------------------------------------- derived artifacts
    def load_metrics(self) -> dict:
        with open(self.metrics_json, encoding="utf-8") as f:
            return json.load(f)

    def load_centroid(self) -> np.ndarray:
        """Unit-norm ERes2NetV2 corpus centroid (np.save, float64) — the
        trainer loads it and computes rollout sims locally as
        `vectors @ centroid`; preprocess writes it in `finalize`, same np.save
        layout as the codes/embedding artifacts."""
        return np.load(self.centroid_npy)

    def reward_config(self) -> RewardConfig:
        """sv_center/sv_scale from the corpus sim distribution (mean/std of
        the per-clip cosine to the centroid). mos_tau stays at the current
        2.5: the mos stats in metrics.json are informational until a gate
        rule is decided (STATUS.md §16)."""
        metrics = self.load_metrics()
        return RewardConfig(
            sv_center=metrics["sim"]["mean"],
            sv_scale=metrics["sim"]["std"],
        )

    def speaker_ref(self) -> Path:
        """Enhanced wav of the pool's ERes2NetV2 medoid (the `medoid` key
        finalize writes = max mean-pairwise cosine) — the DEFAULT SFT
        conditioning reference: the clip is SELECTED in E2V2 (the space that
        hears channel/quality differences) but EMBEDDED with the model's own
        speaker encoder (STATUS §19.4). An explicit --speaker-audio
        overrides."""
        name = self.load_metrics().get("medoid")
        assert name, (
            f"{self.metrics_json} has no 'medoid' — regenerate the cache (idle "
            "rerun of workers/preprocess/main.py) with the current finalize"
        )
        path = self.enhanced_dir / f"{name}.wav"
        assert path.exists(), f"medoid clip missing on disk: {path}"
        return path

    def load_sft_dataset(
        self, limit: int | None = None
    ) -> list[tuple[str, torch.Tensor]]:
        """(text, codes[T, 16] long) pairs — asset.jsonl rows joined with
        their sibling codes/*.npy. `limit` slices the head (debug only:
        a partial pool's metrics/medoid are meaningless — see the removed
        preprocess --limit)."""
        assert self.cache_namespace_dir.is_dir(), (
            f"cache dir not found: {self.cache_namespace_dir}"
        )
        rows = [
            json.loads(line)
            for line in self.asset_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        data: list[tuple[str, torch.Tensor]] = []
        for row in rows:
            codes_path = self.codes_dir / f"{row['name']}.npy"
            assert codes_path.exists(), (
                f"missing codes for {row['name']!r} — run the preprocess "
                "pipeline first"
            )
            codes = torch.from_numpy(np.load(codes_path)).long()
            data.append((row["text"], codes))
        if limit is not None:
            data = data[:limit]
        return data


def load_multi_sft_dataset(
    layouts: list[CacheLayout], per_pool_cap: int | None = None
) -> list[tuple[str, torch.Tensor, int]]:
    """Multi-speaker SFT dataset: `(text, codes, speaker_tag)` triples — each
    pool's `load_sft_dataset` rows tagged with the pool's index into
    `layouts` (the tag indexes the per-pool speaker vectors the trainer
    extracts from each pool's medoid, `CacheLayout.speaker_ref`). Speaker
    identity travels ONLY through that vector (slot 6 at train time, the
    baked export row at inference) — the model cannot bake one voice into
    shared weights when batches mix speakers.

    `per_pool_cap` head-slices every pool to equal counts (balanced batches:
    an unbalanced pool would let the majority speaker dominate the shared
    weights). Train-side only — the pools' metrics/medoid stay computed over
    the FULL pools; the head slice is deterministic (asset.jsonl order)."""
    data: list[tuple[str, torch.Tensor, int]] = []
    for tag, layout in enumerate(layouts):
        rows = layout.load_sft_dataset()
        if per_pool_cap is not None:
            rows = rows[:per_pool_cap]
        data.extend((text, codes, tag) for text, codes in rows)
    return data
