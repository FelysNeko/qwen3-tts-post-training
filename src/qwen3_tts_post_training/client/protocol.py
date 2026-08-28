"""ZMQ protocol between trainer (PUSH/PULL bind) and stateless scorer worker.

Audio crosses as absolute tmpfs paths (/dev/shm); scores come back raw —
sigmoid/std/lambda composition lives in qwen3_tts_post_training.reward. The
scorer is calibration-free: it returns raw embeddings, and the caller derives
similarities against its own centroid (`reward.metrics.load_centroid`).
Validated with pydantic — no manual json building.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class ScoreField(StrEnum):
    """What a caller wants back per wav. The scorer derives which model
    groups to run ({VECTOR} → SV embed, {TRANSCRIPT, CER} → ASR, {MOS} →
    MOS), lazy-loads only those, and None-fills everything unrequested."""

    VECTOR = "vector"
    TRANSCRIPT = "transcript"
    CER = "cer"
    MOS = "mos"


ALL_FIELDS: frozenset[ScoreField] = frozenset(ScoreField)


class ScoreItem(BaseModel):
    wav_path: str
    text: str


class ScoreResult(BaseModel):
    """Unrequested fields come back None; requested ones are always filled.
    `vector` is the raw unit-norm ERes2NetV2 embedding — float32-exact
    through the JSON round-trip (float32 ⊂ float64) — and the similarity to
    the corpus centroid is the caller's job (one batched matmul)."""

    wav_path: str
    vector: list[float] | None = None
    transcript: str | None = None
    cer: float | None = None
    mos: float | None = None


class ScoreRequest(BaseModel):
    """Bottom-level ZMQ message: trainer -> scorer."""

    id: int
    items: list[ScoreItem]
    fields: frozenset[ScoreField] = ALL_FIELDS


class Timing(BaseModel):
    sv: float
    asr: float
    mos: float


class ScoreResponse(BaseModel):
    """Bottom-level ZMQ message: scorer -> trainer."""

    id: int
    results: list[ScoreResult]
    timing: Timing
    rss_mb: int
