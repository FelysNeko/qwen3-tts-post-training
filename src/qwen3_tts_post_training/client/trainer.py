"""HTTP client for the FastAPI scorer (workers/scorer).

Crash-tolerant by construction, per the async request/lookup wire protocol:

- `submit` NEVER raises on scorer downtime — it POSTs eagerly on a
  short-timeout client (a dead scorer costs ~1s per submit, never an
  exception) and keeps the payload buffered on failure; `poll` retries the
  send each round until the scorer is reachable. Startup order is therefore
  irrelevant, and the POST lands *during* the caller's next rollout instead
  of at the first poll (the rollout∥score overlap lives exactly here).
- `poll` translates every lookup outcome: 200 → results (handle consumed);
  202 → None (keep waiting); 404 (unknown/already consumed — e.g. the scorer
  restarted) and 500 (scored-with-error) → transparent re-send under a fresh
  server req_id, None for this round. Callers hold the handle forever and the
  loop just keeps polling — scorer death loses nothing the caller still owns.
- `score` is the blocking convenience (submit + poll loop) used by
  preprocess; same infinite-patience semantics.

Trainer-side lifecycle stays: tmpfs wavs are deleted by the caller only after
results come back.
"""

from __future__ import annotations

import logging
import threading
import time
from itertools import count

import httpx

from qwen3_tts_post_training.client.protocol import (
    ScoreItem,
    ScoreRequest,
    ScoreResponse,
    ScoreResult,
)

logger = logging.getLogger(__name__)


class Client:
    """Handle-based batch client: eager submit (short-timeout POST, buffered
    retry on failure) + poll (auto-resend)."""

    def __init__(self, url: str = "http://127.0.0.1:8000", poll_interval: float = 2.0):
        self.url = url.rstrip("/")
        self.poll_interval = poll_interval
        self._http = httpx.Client(
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=30.0)
        )
        # submit-time POSTs must never stall the rollout loop: a dead scorer
        # costs ~1s per submit here, then the poll loop owns the retries.
        self._submit_http = httpx.Client(
            timeout=httpx.Timeout(connect=1.0, read=5.0, write=5.0, pool=5.0)
        )
        self._lock = threading.Lock()
        self._handles: dict[
            int, tuple[list[ScoreItem], dict[str, bool], int | None]
        ] = {}
        self._next_handle = count(1)
        self._resent: set[int] = set()

    def close(self) -> None:
        self._http.close()
        self._submit_http.close()

    def _send(
        self,
        items: list[ScoreItem],
        asr: bool,
        utmosv2: bool,
        p835: bool,
        sv: bool,
        http: httpx.Client | None = None,
    ) -> int | None:
        """POST /request once; None on transport failure or server 5xx."""
        client = http or self._http
        payload = ScoreRequest(
            items=items, asr=asr, utmosv2=utmosv2, p835=p835, sv=sv
        ).model_dump()
        try:
            r = client.post(f"{self.url}/request", json=payload)
        except httpx.TransportError:
            return None
        if r.status_code != 200:
            return None
        return int(r.json()["req_id"])

    def submit(
        self,
        items: list[ScoreItem],
        asr: bool = False,
        utmosv2: bool = False,
        p835: bool = False,
        sv: bool = False,
    ) -> int:
        """Register a batch, POST it eagerly, and return a local handle
        (≥1; -1 for empty items). Never raises on scorer downtime — the
        eager send runs on the short-timeout client (~1s worst case) and,
        on failure, the payload stays buffered for the poll loop to retry
        under its regular timeouts."""
        if not items:
            return -1
        with self._lock:
            handle = next(self._next_handle)
            self._handles[handle] = (
                items,
                {"asr": asr, "utmosv2": utmosv2, "p835": p835, "sv": sv},
                None,
            )
        rid = self._send(
            items,
            asr=asr,
            utmosv2=utmosv2,
            p835=p835,
            sv=sv,
            http=self._submit_http,
        )
        if rid is not None:
            with self._lock:
                if handle in self._handles:
                    self._handles[handle] = (
                        items,
                        {"asr": asr, "utmosv2": utmosv2, "p835": p835, "sv": sv},
                        rid,
                    )
        return handle

    def poll(self, handle: int) -> list[ScoreResult] | None:
        """One non-blocking round: results when ready, None while anything is
        in flight (queued, scoring, scorer down, or re-sent this round)."""
        if handle < 0:
            return None
        with self._lock:
            entry = self._handles.get(handle)
        if entry is None:
            return None
        items, services, rid = entry
        if rid is None:  # first send, or a re-send is due
            rid = self._send(items, **services)
            if rid is None:
                return None
            with self._lock:
                self._handles[handle] = (items, services, rid)
            return None
        try:
            r = self._http.post(f"{self.url}/lookup", json={"req_id": rid})
        except httpx.TransportError:
            return None
        if r.status_code == 200:
            with self._lock:
                self._handles.pop(handle, None)
                self._resent.discard(handle)
            return ScoreResponse.model_validate(r.json()).results
        if r.status_code == 202:
            return None
        # 404: unknown id (already consumed elsewhere or scorer restarted);
        # 5xx: scored-with-error (consumed). Either way re-send the same
        # payload under a fresh req_id on the next round.
        with self._lock:
            self._handles[handle] = (items, services, None)
        if handle not in self._resent:
            self._resent.add(handle)
            logger.warning(
                f"scorer req {rid} lost/failed (HTTP {r.status_code}) — re-sending"
            )
        return None

    def score(
        self,
        items: list[ScoreItem],
        asr: bool = False,
        utmosv2: bool = False,
        p835: bool = False,
        sv: bool = False,
    ) -> list[ScoreResult]:
        """Blocking convenience: submit + poll until results (infinite
        patience — scorer downtime and restarts are survived transparently)."""
        if not items:
            return []
        handle = self.submit(items, asr=asr, utmosv2=utmosv2, p835=p835, sv=sv)
        while (results := self.poll(handle)) is None:
            time.sleep(self.poll_interval)
        return results
