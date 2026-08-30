"""Thin CLI launcher for the trainer worker: `main.py grpo|sft [args]`."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace

from trainer.grpo.loop import TrainConfig, run_grpo
from trainer.sft.loop import SftConfig, run_sft


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--model-path", type=str)
    shared.add_argument("--device", type=str)
    shared.add_argument("--dtype", type=str)
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
    sft.add_argument("--base-model-path", default=None)
    sft.add_argument("--cache-dir", default=None)
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
    grpo.add_argument("--metrics-path", default=None)
    grpo.add_argument("--variant", default=None, choices=["vanilla", "dr", "gspo"])
    grpo.add_argument("--kl-beta", type=float, default=None)
    grpo.add_argument("--scorer-push-endpoint", default=None)
    grpo.add_argument("--scorer-pull-endpoint", default=None)
    grpo.add_argument("--scorer-timeout", type=float, default=None)

    args = parser.parse_args()
    # CLI 覆盖（未给的键 = None → 落回各分支 dataclass 的默认值）
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key != "command" and value is not None
    }

    if args.command == "sft":
        run_sft(replace(SftConfig(), **overrides))
    else:
        run_grpo(replace(TrainConfig(), **overrides))


if __name__ == "__main__":
    main()
