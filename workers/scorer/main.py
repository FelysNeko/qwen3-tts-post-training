"""Scorer worker entrypoint — ZMQ PUSH/PULL batch service."""

from __future__ import annotations

import argparse
import logging
import os
import resource
import signal

from scorer.multi_object import Scorers

from qwen3_tts_post_training.client.protocol import ScoreResponse
from qwen3_tts_post_training.client.scorer import Client

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--sv-dir", required=True, help="3D-Speaker root (pretrained/ inside)"
    )
    ap.add_argument("--asr-model", default="Qwen/Qwen3-ASR-1.7B-hf")
    ap.add_argument("--asr-batch", type=int, default=8)
    ap.add_argument("--mos-fold", type=int, default=0)
    ap.add_argument("--mos-seed", type=int, default=42)
    ap.add_argument("--mos-reps", type=int, default=8)
    ap.add_argument(
        "--gpu-mel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="build UTMOS mel spectrograms on GPU (default on; ~20x faster, MOS within ~0.03 of the librosa path)",
    )
    ap.add_argument(
        "--push-endpoint",
        default="tcp://127.0.0.1:5555",
        help="ZMQ PULL connect endpoint (trainer PUSH binds here)",
    )
    ap.add_argument(
        "--pull-endpoint",
        default="tcp://127.0.0.1:5556",
        help="ZMQ PUSH connect endpoint (trainer PULL binds here)",
    )
    args = ap.parse_args()

    logger.info(f"pid={os.getpid()} device={args.device}")
    logger.info(f"push {args.push_endpoint} pull {args.pull_endpoint}")
    scorers = Scorers(args)
    worker = Client(args.push_endpoint, args.pull_endpoint)

    stop = {"flag": False}

    def _handle_sig(signum, _frame):
        logger.info(f"signal {signum}, exiting")
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    logger.info("connected, waiting for requests")
    while not stop["flag"]:
        req = worker.recv_request(timeout_ms=500)
        if req is None:
            continue
        results, timing = scorers.score(req.items, req.fields)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        logger.info(f"req {req.id}: n={len(req.items)} timing={timing}")
        resp = ScoreResponse(id=req.id, results=results, timing=timing, rss_mb=rss)
        worker.send_response(resp)

    worker.close()
    logger.info("exiting")


if __name__ == "__main__":
    main()
