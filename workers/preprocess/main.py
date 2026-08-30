"""Preprocess worker entrypoint — corpus → `.cache/{dataset.name}`.

Usage (any resident scorer worker — it is calibration-free):
    workers/preprocess/.venv/bin/python workers/preprocess/main.py \
      --dataset /path/to/corpus/Chinese(PRC)

The corpus wav dir is the input (sibling `{dataset.name}.jsonl` provides
{name, text}); the pool lands at `{cache-dir}/{dataset.name}`.

Four checksum-guarded stages (see preprocess/pipeline.py): filter →
enhanced(clearvoice 48k) → codes → embedding. The enhanced 48k output is
the only derived audio — all downstream consumers resample from it on the
fly. Interrupted runs resume per-clip; corrupt files are regenerated and
salvaged when the checksum reproduces.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from preprocess.pipeline import run_pipeline

from qwen3_tts_post_training.paths import repo_root

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="corpus wav dir; sibling {dir.name}.jsonl provides {name, text}",
    )
    ap.add_argument(
        "--model-path",
        default="/mnt/d/Repository/models/PhiLia093-TTS/",
        help="Qwen3-TTS ckpt (processor for filtering, speech_tokenizer for codes)",
    )
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=repo_root() / ".cache",
        help="cache ROOT location; the pool is {cache-dir}/{dataset.name}",
    )
    ap.add_argument("--min-tokens", type=int, default=2)
    ap.add_argument("--min-seconds", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=16, help="scoring chunk size")
    ap.add_argument(
        "--push-endpoint",
        default="tcp://127.0.0.1:5555",
        help="ZMQ PUSH bind (scorer PULL-connects here)",
    )
    ap.add_argument(
        "--pull-endpoint",
        default="tcp://127.0.0.1:5556",
        help="ZMQ PULL bind (scorer PUSH-connects here)",
    )
    ap.add_argument("--timeout", type=float, default=600.0, help="scorer timeout s")
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    assert args.dataset.is_dir(), f"dataset wav dir not found: {args.dataset}"
    assert Path(args.model_path).exists(), (
        f"TTS ckpt not found at {args.model_path!r} — pass --model-path"
    )

    import torch
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    from qwen3_tts_post_training.client.trainer import Client

    logger.info(f"dataset={args.dataset} device={args.device}")
    wrapper = Qwen3TTSModel.from_pretrained(
        args.model_path,
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = wrapper.processor
    speech_tokenizer = wrapper.model.speech_tokenizer

    def tokenize_text(text: str) -> int:
        ids = processor(text=text)["input_ids"]
        return len(ids[0]) if ids and isinstance(ids[0], list) else len(ids)

    client = Client(
        push_endpoint=args.push_endpoint,
        pull_endpoint=args.pull_endpoint,
        timeout_s=args.timeout,
    )
    try:
        out = run_pipeline(
            dataset=args.dataset,
            cache_root=args.cache_dir,
            tokenize_text=tokenize_text,
            speech_tokenizer=speech_tokenizer,
            client=client,
            device=args.device,
            model_path=args.model_path,
            min_tokens=args.min_tokens,
            min_seconds=args.min_seconds,
            batch=args.batch,
        )
    finally:
        client.close()
    logger.info(f"done: {out}")


if __name__ == "__main__":
    main()
