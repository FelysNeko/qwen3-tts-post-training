"""JSON-line protocol between trainer and the scorer worker process.

Lock-step: one in-flight request; the response carries the same id.
Audio crosses the boundary as absolute file paths (rollout wavs live in
/dev/shm); scores come back raw — sigmoid/std/lambda composition lives in
qwen3_tts_post_training.reward, never here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class ScoreItem:
    wav: str
    text: str | None = None  # None → WER skipped (cer=None, transcript still returned)


def make_request(req_id: int, items: list[ScoreItem]) -> str:
    return json.dumps(
        {
            "id": req_id,
            "op": "score",
            "items": [{"wav": it.wav, "text": it.text} for it in items],
        },
        ensure_ascii=False,
    )


def make_ping(req_id: int) -> str:
    return json.dumps({"id": req_id, "op": "ping"})


def parse_response(line: str) -> dict:
    return json.loads(line)


class ScorerError(RuntimeError):
    pass
