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
    groups to run ({EMBEDDING} → SV embed, {TRANSCRIPT, CER} → ASR, {MOS} →
    MOS), lazy-loads only those, and None-fills everything unrequested."""

    EMBEDDING = "embedding"
    TRANSCRIPT = "transcript"
    CER = "cer"
    MOS = "mos"


class ScoreItem(BaseModel):
    wav_path: str
    text: str


class ScoreResult(BaseModel):
    """Unrequested fields come back None; requested ones are always filled.
    `embedding` is the raw unit-norm ERes2NetV2 embedding — float32-exact
    through the JSON round-trip (float32 ⊂ float64) — and the similarity to
    the corpus centroid is the caller's job (one batched matmul).

    Two read paths: plain attribute access is the honest one (typed
    `| None`); the `get_*_unwrap` methods are for call sites that know what
    they requested — the assert strips the None and crashes on a field the
    request did not ask for instead of leaking None downstream (防呆)."""

    wav_path: str
    embedding: list[float] | None = None
    transcript: str | None = None
    cer: float | None = None
    mos: float | None = None

    def get_embedding_unwrap(self) -> list[float]:
        embedding = self.embedding
        assert embedding is not None, "embedding was not requested from the scorer"
        return embedding

    def get_transcript_unwrap(self) -> str:
        transcript = self.transcript
        assert transcript is not None, "transcript was not requested from the scorer"
        return transcript

    def get_cer_unwrap(self) -> float:
        cer = self.cer
        assert cer is not None, "cer was not requested from the scorer"
        return cer

    def get_mos_unwrap(self) -> float:
        mos = self.mos
        assert mos is not None, "mos was not requested from the scorer"
        return mos


class ScoreRequest(BaseModel):
    """Bottom-level ZMQ message: trainer -> scorer. `fields` is REQUIRED —
    every caller states exactly what it needs, there is no implicit
    score-everything default."""

    id: int
    items: list[ScoreItem]
    fields: frozenset[ScoreField]


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
