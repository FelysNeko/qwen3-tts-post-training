"""Preprocess worker entrypoint — corpus → `.cache/{namespace}`.

Usage (any resident scorer worker — it is calibration-free):
    workers/preprocess/.venv/bin/python workers/preprocess/main.py \
      --corpus-dir /path/to/corpus --namespace Cyrene/Chinese(PRC)

The corpus wav dir is `{corpus-dir}/{namespace}` (the namespace mirrors the
corpus hierarchy and IS the speaker name downstream); its sibling
`{corpus-dir}/{namespace}.jsonl` provides {name, text}. The pool lands at
`{cache-dir}/{namespace}`.

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

import torch
from preprocess.pipeline import run_pipeline

from qwen3_tts_post_training.paths import repo_root

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
        help="a dirctory that contains sub-dir like Cyrene/Chinese(PRC)",
    )
    parser.add_argument(
        "--namespace",
        type=Path,
        required=True,
        help="must in format like Cyrene/Chinese(PRC), and Cyrene/Chinese(PRC).jsonl must exist",
    )
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="Qwen3-TTS ckpt (processor for filtering, speech_tokenizer for codes)",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=repo_root() / ".cache",
        help="cache ROOT location; the pool is {cache-dir}/{namespace}",
    )
    parser.add_argument("--min-tokens", type=int, default=2)
    parser.add_argument("--min-seconds", type=float, default=0.1)
    parser.add_argument(
        "--random",
        action="store_true",
        help="shuffle the pool order (seeded — reproducible); SFT "
        "--per-pool-cap head slices then sample uniformly instead of "
        "taking the chapter-ordered corpus head",
    )
    parser.add_argument("--batch", type=int, default=16, help="scoring chunk size")
    parser.add_argument(
        "--scorer-url",
        default="http://127.0.0.1:8000",
        help="FastAPI scorer base URL (§50 request/lookup service)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    dataset = args.corpus_dir / args.namespace
    assert dataset.is_dir(), f"corpus wav dir not found: {dataset}"
    assert (args.corpus_dir / f"{args.namespace}.jsonl").exists(), (
        f"transcript jsonl not found: {args.corpus_dir / f'{args.namespace}.jsonl'}"
    )

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    from qwen3_tts_post_training.client.trainer import Client

    logger.info(f"using dataset: {dataset}")
    wrapper = Qwen3TTSModel.from_pretrained(
        args.model_path,
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = wrapper.processor
    speech_tokenizer = wrapper.model.speech_tokenizer

    def token_counter(text: str) -> int:
        ids = processor(text=text)["input_ids"]
        return len(ids[0]) if ids and isinstance(ids[0], list) else len(ids)

    client = Client(url=args.scorer_url)
    try:
        out = run_pipeline(
            dataset=dataset,
            namespace=args.namespace,
            cache_root=args.cache_dir,
            tokenize_text=token_counter,
            speech_tokenizer=speech_tokenizer,
            client=client,
            device=args.device,
            model_path=args.model_path,
            min_tokens=args.min_tokens,
            min_seconds=args.min_seconds,
            batch=args.batch,
            random_order=args.random,
        )
    finally:
        client.close()
    logger.info(f"done: {out}")


if __name__ == "__main__":
    main()
