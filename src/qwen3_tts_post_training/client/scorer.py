"""Scorer-side ZMQ worker client (mirror of trainer Client).

Trainer binds PUSH/PULL; this worker connects PULL/PUSH.
Encapsulates the connect block so serve.py has no raw zmq boilerplate.
"""

from __future__ import annotations

import zmq

from qwen3_tts_post_training.client.protocol import ScoreRequest, ScoreResponse


class Client:
    """Stateless scorer worker: recv_request (blocking poll) + send_response."""

    def __init__(
        self,
        push_endpoint: str = "tcp://127.0.0.1:5555",
        pull_endpoint: str = "tcp://127.0.0.1:5556",
    ):
        self.push_endpoint = push_endpoint
        self.pull_endpoint = pull_endpoint
        self._ctx: zmq.Context | None = None
        self._pull: zmq.Socket | None = None
        self._push: zmq.Socket | None = None

    def _ensure_connected(self) -> None:
        if self._pull is not None:
            return
        self._ctx = zmq.Context.instance()
        self._pull = self._ctx.socket(zmq.PULL)
        self._pull.setsockopt(zmq.LINGER, 0)
        self._pull.setsockopt(zmq.RCVHWM, 1000)
        self._pull.connect(self.push_endpoint)
        self._push = self._ctx.socket(zmq.PUSH)
        self._push.setsockopt(zmq.LINGER, 0)
        self._push.setsockopt(zmq.SNDHWM, 1000)
        self._push.connect(self.pull_endpoint)

    def close(self) -> None:
        if self._pull is None:
            return
        try:
            self._pull.close(linger=0)
            self._push.close(linger=0)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001, S110
            pass
        self._pull = None
        self._push = None

    def recv_request(self, timeout_ms: int = 500) -> ScoreRequest | None:
        """Poll with timeout_ms; returns ScoreRequest or None on timeout. Validates."""
        self._ensure_connected()
        assert self._pull is not None
        if self._pull.poll(timeout_ms) == 0:
            return None
        raw = self._pull.recv_json()
        return ScoreRequest.model_validate(raw)

    def send_response(self, resp: ScoreResponse) -> None:
        self._ensure_connected()
        assert self._push is not None
        self._push.send_json(resp.model_dump())
