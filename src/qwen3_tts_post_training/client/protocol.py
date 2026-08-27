"""ZMQ protocol between trainer (PUSH/PULL bind) and stateless scorer worker.

Audio crosses as absolute tmpfs paths (/dev/shm); scores come back raw —
sigmoid/std/lambda composition lives in qwen3_tts_post_training.reward.
Validated with pydantic — no manual json building.
"""

from __future__ import annotations

from pydantic import BaseModel


class ScoreItem(BaseModel):
    wav_path: str
    text: str


class ScoreResult(BaseModel):
    """`sim`/`sim_camp` are None when the scorer runs without a reference
    (preprocess mode: the centroid only exists after every clip is embedded);
    `vector` carries the raw unit-norm ERes2NetV2 embedding (reward source of
    truth). Trainer mode always has refs set, so sim fields are floats there."""

    wav_path: str
    sim: float | None
    sim_camp: float | None
    vector: list[float] | None = None
    transcript: str
    cer: float
    mos: float


class ScoreRequest(BaseModel):
    """Bottom-level ZMQ message: trainer -> scorer."""

    id: int
    items: list[ScoreItem]


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
