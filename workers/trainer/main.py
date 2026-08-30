"""Thin CLI launcher for the trainer worker: `main.py grpo|sft [args]`."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace

from trainer.grpo.loop import TrainConfig, run_grpo
from trainer.sft.loop import SftConfig, run_sft

from qwen3_tts_post_training.paths import repo_root


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--model-path", type=str)
    shared.add_argument("--device", type=str)
    shared.add_argument("--cache-dir", type=str)
    shared.add_argument(
        "--namespace",
        type=str,
        help="cache selector: <repo>/.cache/{namespace} (e.g. Chinese(PRC))",
    )
    shared.add_argument("--lr", type=float)
    shared.add_argument("--warmup-steps", type=int)
    shared.add_argument("--weight-decay", type=float)
    shared.add_argument("--grad-clip", type=float)
    shared.add_argument("--seed", type=int)
    shared.add_argument("--out-dir", type=str)
    shared.add_argument("--ckpt-every", type=int)
    shared.add_argument("--resume", action="store_true")

    parser = argparse.ArgumentParser(description="trainer worker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sft = subparsers.add_parser("sft", parents=[shared])
    sft.add_argument("--speaker-audio", default=None)
    sft.add_argument("--limit", type=int, default=None)
    sft.add_argument("--batch-size", type=int, default=None)
    sft.add_argument("--grad-accum", type=int, default=None)
    sft.add_argument("--epochs", type=int, default=None)
    sft.add_argument("--log-every", type=int, default=None)
    sft.add_argument("--export-name", default=None)

    grpo = subparsers.add_parser("grpo", parents=[shared])
    grpo.add_argument("--lora-r", type=int, default=None)
    grpo.add_argument("--lora-alpha", type=float, default=None)
    grpo.add_argument("--speaker", default=None)
    grpo.add_argument("--num-prompts", type=int, default=None)
    grpo.add_argument("--group-size", type=int, default=None)
    grpo.add_argument("--token-budget", type=int, default=None)
    grpo.add_argument("--token-budget-infer", type=int, default=None)
    grpo.add_argument("--text-pool-path", default=None)
    grpo.add_argument("--num-steps", type=int, default=None)
    grpo.add_argument("--temperature", type=float, default=None)
    grpo.add_argument("--top-k", type=int, default=None)
    grpo.add_argument(
        "--sampler-impl", default=None, choices=["hf", "fast", "compiled", "graphed"]
    )
    grpo.add_argument("--variant", default=None, choices=["vanilla", "dr", "gspo"])
    grpo.add_argument("--kl-beta", type=float, default=None)
    grpo.add_argument("--scorer-push-endpoint", default=None)
    grpo.add_argument("--scorer-pull-endpoint", default=None)
    grpo.add_argument("--scorer-timeout", type=float, default=None)

    args = parser.parse_args()
    # --namespace = --cache-dir 的简写（<repo>/.cache/{namespace}，与
    # preprocess 的默认 cache-root 对齐）；显式 --cache-dir 与之互斥；
    # 两者皆缺在此直接拒绝，不溜到 run_* 的 assert
    if args.namespace is not None:
        if args.cache_dir is not None:
            parser.error("--namespace and --cache-dir are mutually exclusive")
        cache = repo_root() / ".cache" / args.namespace
        if not cache.is_dir():
            parser.error(f"cache dir not found: {cache}")
        args.cache_dir = str(cache)
    elif args.cache_dir is None:
        parser.error("--cache-dir (or --namespace) is required")
    # CLI 覆盖（未给的键 = None → 落回各分支 dataclass 的默认值）
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key not in ("command", "namespace") and value is not None
    }

    if args.command == "sft":
        run_sft(replace(SftConfig(), **overrides))
    else:
        run_grpo(replace(TrainConfig(), **overrides))


if __name__ == "__main__":
    main()
