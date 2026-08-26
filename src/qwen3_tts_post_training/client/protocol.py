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
    wav_path: str
    sim: float
    sim_camp: float
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
