"""Thin client for the resident scorer worker (workers/scorer).

Spawns the worker's own venv, talks JSON lines over stdin/stdout, restarts
once on death/timeout. Stdlib only — importable from any env that has the
root package installed (trainer does, via editable path dep).
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

from .protocol import ScoreItem, ScorerError, make_ping, make_request, parse_response

# Defaults point at playground assets until the assets/ migration (MD §8).
PLAYGROUND = Path("/home/felysneko/workspace/playground")
DEFAULTS = {
    "sv_dir": PLAYGROUND / "3D-Speaker",
    "sv_ref": PLAYGROUND / "audio" / "sv_ref_embedding.npy",
    "sv_ref_camp": PLAYGROUND / "audio" / "campplus" / "sv_ref_embedding.npy",
}


def repo_root() -> Path:
    env = os.environ.get("Q3TTS_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[3]


class ScorerClient:
    """Lock-step client. score() blocks; batching happens inside the worker."""

    def __init__(
        self,
        sv_dir: str | Path | None = None,
        sv_ref: str | Path | None = None,
        sv_ref_camp: str | Path | None = None,
        asr_model: str = "Qwen/Qwen3-ASR-1.7B-hf",
        mos_reps: int = 8,
        device: str = "cuda:0",
        timeout_s: float = 600.0,
    ):
        self.worker_dir = repo_root() / "workers" / "scorer"
        self.worker_py = self.worker_dir / ".venv" / "bin" / "python"
        self.serve_py = self.worker_dir / "main.py"
        args = [
            str(self.worker_py),
            str(self.serve_py),
            "--device",
            device,
            "--asr-model",
            asr_model,
            "--mos-reps",
            str(mos_reps),
        ]
        for flag, val in (
            ("--sv-dir", sv_dir),
            ("--sv-ref", sv_ref),
            ("--sv-ref-camp", sv_ref_camp),
        ):
            if val is not None:
                args += [flag, str(val)]
        self._args = args
        self.timeout_s = timeout_s
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue[str] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._req_id = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            self._args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(self.worker_dir),
        )
        self._lines = queue.Queue()
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()

    def _drain(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            self._lines.put(line)
        # EOF → sentinel so waiters fail fast instead of hanging on timeout
        self._lines.put("__EOF__")

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=30)
        except Exception:
            self._proc.kill()
        self._proc = None

    def __enter__(self) -> "ScorerClient":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- talk --------------------------------------------------------------

    def _roundtrip(self, line_req: str, timeout: float) -> dict:
        req_id = json.loads(line_req)["id"]
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(line_req + "\n")
        self._proc.stdin.flush()
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"scorer worker timed out after {timeout:.0f}s (req {req_id})"
                )
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(
                    f"scorer worker timed out after {timeout:.0f}s (req {req_id})"
                )
            if line == "__EOF__":
                raise ScorerError("scorer worker died (stdout closed)")
            line = line.strip()
            if not line:
                continue
            try:
                resp = parse_response(line)
            except json.JSONDecodeError:
                continue  # stray library print; real logs go to stderr
            if resp.get("id") != req_id:
                continue
            return resp

    def _call(self, line_req: str, timeout: float | None = None) -> dict:
        timeout = self.timeout_s if timeout is None else timeout
        try:
            self.start()
            return self._roundtrip(line_req, timeout)
        except (TimeoutError, ScorerError, OSError, BrokenPipeError):
            # one respawn + retry, then give up
            self.stop()
            self.start()
            return self._roundtrip(line_req, timeout)

    # -- api ---------------------------------------------------------------

    def ping(self) -> dict:
        self._req_id += 1
        resp = self._call(make_ping(self._req_id), timeout=120.0)
        if not resp.get("ok"):
            raise ScorerError(str(resp.get("error")))
        return resp

    def score(self, items: list[ScoreItem]) -> list[dict]:
        """Returns one dict per item: sim/sim_camp/cer/transcript/mos/dur
        (fields set to None where the corresponding scorer was skipped/failed;
        'error' key set if the whole item failed)."""
        if not items:
            return []
        self._req_id += 1
        resp = self._call(make_request(self._req_id, items))
        if not resp.get("ok"):
            raise ScorerError(str(resp.get("error")))
        results: list[dict] = resp["results"]
        if len(results) != len(items):
            raise ScorerError(
                f"worker returned {len(results)} results for {len(items)} items"
            )
        return results
