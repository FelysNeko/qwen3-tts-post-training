"""Scorer HTTP service — FastAPI request/lookup over an internal GPU queue.

All state lives in `Scoreboard`: an unbounded request queue, a working set
(req_ids queued or being scored), a ready pool (req_id -> ScoreResponse,
consume-once via /lookup), one auto-incrementing req_id counter, and a single
GPU thread that pops the queue, runs the (unchanged) scoring core and lands
results in the pool — an exception lands an all-None error entry instead,
surfaced as HTTP 500 on lookup. Endpoints are thin HTTP translations of the
Scoreboard's domain outcomes; everything is lock-guarded (sync endpoints run
in uvicorn's threadpool).

Robustness contract: the server holds no expectations about callers — any
client may connect at any time, and a restart wipes state (clients detect
the unknown req_id as 404 and re-send).
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import queue
import threading
from collections.abc import Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from qwen3_tts_post_training.client.protocol import (
    LookupRequest,
    ScoreRequest,
    ScoreResponse,
    ScoreResult,
)
from qwen3_tts_post_training.system import peak_rss_mb

logger = logging.getLogger(__name__)


class Scoreboard:
    """The scorer's in-process state machine: queue → working set → ready
    pool. Single process, one lock, thread-safe."""

    def __init__(self) -> None:
        self._q: queue.Queue[tuple[int, ScoreRequest]] = queue.Queue()
        self._pool: dict[int, ScoreResponse] = {}
        self._working: set[int] = set()
        self._next_id = itertools.count(1)
        self._lock = threading.Lock()

    def enqueue(self, req: ScoreRequest) -> int:
        """Register a request; returns its req_id (reserved immediately, so
        /lookup can say 202 while it is still queued)."""
        with self._lock:
            rid = next(self._next_id)
            self._working.add(rid)
        self._q.put((rid, req))
        return rid

    def lookup(self, req_id: int) -> tuple[str, ScoreResponse | None]:
        """Consume-once read: ("ready", entry) — popped from the pool (may
        carry an error); ("pending", None) — queued or being scored;
        ("unknown", None) — never existed, already consumed, or the server
        restarted."""
        with self._lock:
            entry = self._pool.pop(req_id, None)
            if entry is not None:
                return "ready", entry
            if req_id in self._working:
                return "pending", None
        return "unknown", None

    def clear_pool(self) -> int:
        """Drop every ready entry (manual maintenance op); returns the count."""
        with self._lock:
            n = len(self._pool)
            self._pool.clear()
        return n

    def _finalize(self, rid: int, resp: ScoreResponse) -> None:
        with self._lock:
            self._working.discard(rid)
            self._pool[rid] = resp
        logger.info(
            f"req {rid}: n={len(resp.results)} timing={resp.timing} "
            f"rss={resp.rss_mb}MB error={resp.error}"
        )

    def worker_loop(self, score_fn: Callable) -> None:
        """Drain the queue into the pool, one request at a time. A scoring
        exception lands an all-None error entry (clients see 500 and
        re-send) — never a stuck 202."""
        import torch as _torch

        while True:
            rid, req = self._q.get()
            try:
                results, timing = score_fn(
                    req.items, asr=req.asr, mos=req.mos, sv=req.sv
                )
                error = None
            except Exception as e:
                logger.exception(f"req {rid}: scoring failed")
                results = [ScoreResult(wav_path=item.wav_path) for item in req.items]
                timing, error = None, f"{type(e).__name__}: {e}"
            if _torch.cuda.is_available():
                # release cached blocks so the allocator doesn't hoard VRAM
                # (GPU-PV shares one Windows video memory manager — scorer
                # bloat starves the trainer on the other GPU)
                _torch.cuda.empty_cache()
            self._finalize(
                rid,
                ScoreResponse(
                    req_id=rid,
                    results=results,
                    timing=timing,
                    rss_mb=peak_rss_mb(),
                    error=error,
                ),
            )

    def start_worker(self, score_fn: Callable) -> None:
        threading.Thread(
            target=self.worker_loop, args=(score_fn,), daemon=True, name="scorer-gpu"
        ).start()


def create_app(score_fn: Callable, start_thread: bool = True) -> tuple[FastAPI, Scoreboard]:
    """Build the app around a `score_fn(items, asr, mos, sv) ->
    (results, timing)`. Returns the board too, so probes can drive it
    directly (start_thread=False) instead of loading real models."""
    board = Scoreboard()
    if start_thread:
        board.start_worker(score_fn)

    app = FastAPI(title="qwen3-tts scorer", version="2")

    @app.post("/request")
    def request_score(req: ScoreRequest) -> dict:
        return {"req_id": board.enqueue(req)}

    @app.post("/lookup")
    def lookup(req: LookupRequest):
        state, entry = board.lookup(req.req_id)
        if state == "ready":
            if entry.error is not None:
                return JSONResponse(status_code=500, content=entry.model_dump())
            return entry
        if state == "pending":
            raise HTTPException(status_code=202, detail="scoring in progress")
        raise HTTPException(status_code=404, detail=f"unknown req_id {req.req_id}")

    @app.delete("/pool")
    def clear_pool() -> dict:
        return {"cleared": board.clear_pool()}

    return app, board


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--asr-model", default="Qwen/Qwen3-ASR-1.7B-hf")
    parser.add_argument("--asr-batch", type=int, default=8)
    parser.add_argument("--mos-fold", type=int, default=0)
    parser.add_argument("--mos-seed", type=int, default=42)
    parser.add_argument("--mos-reps", type=int, default=8)
    parser.add_argument(
        "--gpu-mel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="build UTMOS mel spectrograms on GPU (default on; ~20x faster, MOS within ~0.03 of the librosa path)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logger.info(f"pid={os.getpid()} device={args.device} port={args.port}")

    from scorer.multi_objective import Scorers

    scorers = Scorers(args)
    app, _ = create_app(scorers.score)
    logger.info("models loaded, serving")
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
