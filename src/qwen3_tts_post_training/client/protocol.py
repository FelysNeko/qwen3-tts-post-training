"""HTTP protocol between trainer/preprocess clients and the FastAPI scorer.

Audio crosses as absolute tmpfs paths (/dev/shm); scores come back raw —
sigmoid/std/lambda composition lives in the trainer's grpo.reward module. The
scorer is calibration-free: it returns raw embeddings and transcripts, and
the caller derives similarities against its own centroid
(`cache.CacheLayout.load_centroid`) and CERs against its own reference texts
(`qwen3_tts_post_training.text.cer`). Validated with pydantic — no manual
json building.

Wire (async request/lookup, crash-tolerant by construction):
- POST /request  {items, asr, utmosv2, p835, sv} -> {req_id}   (queued server-side)
- POST /lookup   {req_id} -> 200 ready (CONSUMED, once) / 202 in flight /
  404 unknown id (never existed, already consumed, or scorer restarted) /
  500 scored-with-error (consumed; all-None results + `error` text)
- DELETE /pool   clears the ready pool
Clients keep their own handle -> req_id mapping and re-send automatically on
404/500, so a scorer restart loses nothing the caller still holds.
"""

from __future__ import annotations

from pydantic import BaseModel


class ScoreItem(BaseModel):
    wav_path: str


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
    utmosv2: float | None = None
    p835: float | None = None

    def get_embedding_unwrap(self) -> list[float]:
        embedding = self.embedding
        assert embedding is not None, "embedding was not requested from the scorer"
        return embedding

    def get_transcript_unwrap(self) -> str:
        transcript = self.transcript
        assert transcript is not None, "transcript was not requested from the scorer"
        return transcript

    def get_utmosv2_unwrap(self) -> float:
        utmosv2 = self.utmosv2
        assert utmosv2 is not None, "utmosv2 was not requested from the scorer"
        return utmosv2

    def get_p835_unwrap(self) -> float:
        p835 = self.p835
        assert p835 is not None, "p835 was not requested from the scorer"
        return p835


class ScoreRequest(BaseModel):
    """POST /request body: what to score and which services to run (a
    service left False is simply not run — an all-False request scores
    nothing and returns all-None results). `utmosv2` is the UTMOSv2 MOS,
    `p835` the P.835 DNSMOS calibrated OVRL — independent stages."""

    items: list[ScoreItem]
    asr: bool = False
    utmosv2: bool = False
    p835: bool = False
    sv: bool = False


class LookupRequest(BaseModel):
    """POST /lookup body."""

    req_id: int


class Timing(BaseModel):
    sv: float
    asr: float
    utmosv2: float = 0.0
    p835: float = 0.0


class ScoreResponse(BaseModel):
    """POST /lookup 200/500 body (consume-once: the id leaves the ready pool
    on either). `timing`/`rss_mb` are diagnostics; on a scoring failure the
    results are all-None, `error` carries the message, and the transport
    status is 500."""

    req_id: int
    results: list[ScoreResult]
    timing: Timing | None = None
    rss_mb: int | None = None
    error: str | None = None
