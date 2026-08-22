"""Resident scorer worker. Read JSON-line requests on stdin, write one
JSON-line response per request on stdout (stdout is reserved for the
protocol — sys.stdout is remapped to stderr so library prints can't corrupt
the channel). Logs and per-scorer timing go to stderr.

Run (normally spawned by ScorerClient, not by hand):
  workers/scorer/.venv/bin/python main.py --sv-dir ... --sv-ref ...

Lazy model loading: each scorer instantiates on first use; 'ping' reports
what is resident.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

PROTO = None  # dup of original fd 1, protocol-only


def log(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


def send(obj: dict) -> None:
    PROTO.write(json.dumps(obj, ensure_ascii=False) + "\n")
    PROTO.flush()


class Scorers:
    def __init__(self, args):
        self.args = args
        self._sv = None
        self._asr = None
        self._mos = None

    @property
    def sv(self):
        if self._sv is None:
            from .sv import SVScorer

            t0 = time.time()
            s = SVScorer(Path(self.args.sv_dir), self.args.device)
            if self.args.sv_ref:
                s.set_ref("eres2netv2", Path(self.args.sv_ref))
            if self.args.sv_ref_camp:
                s.set_ref("campplus", Path(self.args.sv_ref_camp))
            self._sv = s
            log(f"[load] sv {time.time() - t0:.1f}s")
        return self._sv

    @property
    def asr(self):
        if self._asr is None:
            from .asr import ASRScorer

            t0 = time.time()
            self._asr = ASRScorer(
                self.args.asr_model, self.args.device, self.args.asr_batch
            )
            log(f"[load] asr {time.time() - t0:.1f}s")
        return self._asr

    @property
    def mos(self):
        if self._mos is None:
            from .mos import MOSScorer

            t0 = time.time()
            self._mos = MOSScorer(
                fold=self.args.mos_fold,
                seed=self.args.mos_seed,
                num_repetitions=self.args.mos_reps,
            )
            log(f"[load] mos {time.time() - t0:.1f}s")
        return self._mos

    def loaded(self) -> dict:
        return {
            "sv": self._sv is not None,
            "asr": self._asr is not None,
            "mos": self._mos is not None,
        }


def handle_score(scorers: Scorers, items: list[dict]) -> dict:
    import soundfile as sf

    t_all = time.time()
    results = [
        {
            "wav": it["wav"],
            "sim": None,
            "sim_camp": None,
            "transcript": None,
            "cer": None,
            "mos": None,
            "dur": None,
            "error": None,
        }
        for it in items
    ]

    # duration + existence check
    ok_idx = []
    for i, it in enumerate(items):
        try:
            info = sf.info(it["wav"])
            results[i]["dur"] = info.frames / info.samplerate
            ok_idx.append(i)
        except Exception as e:  # noqa: BLE001 — per-item error containment
            results[i]["error"] = f"wav unreadable: {e}"
    if not ok_idx:
        return {"results": results, "timing": {}}

    # SV (per-file, fast)
    t0 = time.time()
    for i in ok_idx:
        try:
            results[i]["sim"] = scorers.sv.score(items[i]["wav"], "eres2netv2")
            if scorers.args.sv_ref_camp:
                results[i]["sim_camp"] = scorers.sv.score(items[i]["wav"], "campplus")
        except Exception as e:  # noqa: BLE001 — per-item error containment
            results[i]["error"] = f"sv: {e}"
    t_sv = time.time() - t0

    # ASR (batched)
    t0 = time.time()
    try:
        idx = ok_idx
        got = scorers.asr.score(
            [items[i]["wav"] for i in idx], [items[i].get("text") for i in idx]
        )
        for i, g in zip(idx, got):
            results[i]["transcript"] = g["transcript"]
            results[i]["cer"] = g["cer"]
    except Exception as e:  # noqa: BLE001 — keep worker alive per scorer
        for i in ok_idx:
            results[i]["error"] = (results[i]["error"] or "") + f"asr: {e}"
    t_asr = time.time() - t0

    # MOS (chunked, seeded)
    t0 = time.time()
    try:
        scores = scorers.mos.score([items[i]["wav"] for i in ok_idx])
        for i, m in zip(ok_idx, scores):
            results[i]["mos"] = m
    except Exception as e:  # noqa: BLE001 — keep worker alive per scorer
        for i in ok_idx:
            results[i]["error"] = (results[i]["error"] or "") + f"mos: {e}"
    t_mos = time.time() - t0

    return {
        "results": results,
        "timing": {
            "sv": round(t_sv, 2),
            "asr": round(t_asr, 2),
            "mos": round(t_mos, 2),
            "total": round(time.time() - t_all, 2),
        },
    }


def main() -> None:
    global PROTO
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--sv-dir", required=True, help="3D-Speaker root (pretrained/ inside)"
    )
    ap.add_argument("--sv-ref", default=None, help="ERes2NetV2 centroid npy")
    ap.add_argument(
        "--sv-ref-camp", default=None, help="CAM++ centroid npy (cross-monitor)"
    )
    ap.add_argument("--asr-model", default="Qwen/Qwen3-ASR-1.7B-hf")
    ap.add_argument("--asr-batch", type=int, default=8)
    ap.add_argument("--mos-fold", type=int, default=0)
    ap.add_argument("--mos-seed", type=int, default=42)
    ap.add_argument("--mos-reps", type=int, default=8)
    args = ap.parse_args()

    # reserve fd 1 for the protocol; everything else prints to stderr
    PROTO = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    log(f"[scorer] pid={os.getpid()} device={args.device}")
    scorers = Scorers(args)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            log(f"[scorer] non-JSON request line: {line[:120]}")
            continue
        rid = req.get("id")
        op = req.get("op")
        try:
            if op == "ping":
                send({"id": rid, "ok": True, "loaded": scorers.loaded()})
            elif op == "score":
                out = handle_score(scorers, req["items"])
                out["rss_mb"] = (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
                )
                log(f"[scorer] req {rid}: n={len(req['items'])} timing={out['timing']}")
                send({"id": rid, "ok": True, **out})
            else:
                send({"id": rid, "ok": False, "error": f"unknown op {op!r}"})
        except Exception as e:  # noqa: BLE001 — per-item error containment
            log(f"[scorer] req {rid} FATAL: {e!r}")
            send({"id": rid, "ok": False, "error": repr(e)})
    log("[scorer] stdin closed, exiting")


if __name__ == "__main__":
    main()
