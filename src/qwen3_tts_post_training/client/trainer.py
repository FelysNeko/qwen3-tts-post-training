"""ZMQ client for the stateless scorer worker (workers/scorer).

Trainer binds two PUSH/PULL endpoints; scorer connects to both.
Audio stays on tmpfs; trainer owns lifecycle and deletes wavs after scoring.
Lazy init — first send_score triggers bind.
"""

from __future__ import annotations

import time

import zmq

from qwen3_tts_post_training.client.protocol import (
    ALL_FIELDS,
    ScoreItem,
    ScoreRequest,
    ScoreResponse,
)


class Client:
    """Pure batch client: send_score (non-blocking push) + recv_score (poll)."""

    def __init__(
        self,
        push_endpoint: str = "tcp://127.0.0.1:5555",
        pull_endpoint: str = "tcp://127.0.0.1:5556",
        timeout_s: float = 600.0,
    ):
        self.push_endpoint = push_endpoint
        self.pull_endpoint = pull_endpoint
        self.timeout_s = timeout_s
        self._ctx: zmq.Context | None = None
        self._push: zmq.Socket | None = None
        self._pull: zmq.Socket | None = None
        self._req_id = 0

    def _ensure_started(self) -> None:
        if self._push is not None:
            return
        self._ctx = zmq.Context.instance()
        self._push = self._ctx.socket(zmq.PUSH)
        self._push.setsockopt(zmq.LINGER, 0)
        self._push.setsockopt(zmq.SNDHWM, 1000)
        self._push.bind(self.push_endpoint)
        self._pull = self._ctx.socket(zmq.PULL)
        self._pull.setsockopt(zmq.LINGER, 0)
        self._pull.setsockopt(zmq.RCVHWM, 1000)
        self._pull.bind(self.pull_endpoint)

    def close(self) -> None:
        if self._push is None:
            return
        try:
            self._push.close(linger=0)
            self._pull.close(linger=0)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001, S110
            pass
        self._push = None
        self._pull = None

    def _send_json(self, obj: dict) -> None:
        assert self._push is not None
        self._push.send_json(obj)

    def _recv_json(self, timeout: float) -> dict:
        assert self._pull is not None
        poller = zmq.Poller()
        poller.register(self._pull, zmq.POLLIN)
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"scorer timed out after {timeout:.0f}s")
            evts = dict(poller.poll(int(remaining * 1000)))
            if self._pull not in evts:
                raise TimeoutError(f"scorer timed out after {timeout:.0f}s")
            try:
                return self._pull.recv_json(flags=zmq.NOBLOCK)
            except zmq.ZMQError:
                continue

    def send_score(self, items: list[ScoreItem], fields=None) -> int:
        """Non-blocking push. Returns req_id for recv_score. `fields=None`
        requests ALL_FIELDS; pass a ScoreField subset for need-based scoring."""
        if not items:
            return -1
        self._ensure_started()
        self._req_id += 1
        self._send_json(
            ScoreRequest(
                id=self._req_id,
                items=items,
                fields=ALL_FIELDS if fields is None else fields,
            ).model_dump(mode="json")  # frozenset -> list, StrEnum -> str
        )
        return self._req_id

    def recv_score(self, expect_id: int, timeout: float | None = None) -> list[dict]:
        self._ensure_started()
        timeout = self.timeout_s if timeout is None else timeout
        raw = self._recv_json(timeout=timeout)
        resp = ScoreResponse.model_validate(raw)
        if resp.id != expect_id:
            raise RuntimeError(
                f"scorer id mismatch {resp.id} != {expect_id} (order broken?)"
            )
        return [r.model_dump() for r in resp.results]

    def score(self, items: list[ScoreItem], fields=None) -> list[dict]:
        """Blocking convenience: send + recv."""
        if not items:
            return []
        rid = self.send_score(items, fields=fields)
        return self.recv_score(rid, timeout=self.timeout_s)
