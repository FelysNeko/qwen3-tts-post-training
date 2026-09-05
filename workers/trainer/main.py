"""Thin CLI launcher for the trainer worker: `main.py grpo|sft [args]`."""

from __future__ import annotations

import argparse
import logging
import os

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
    shared.add_argument(
        "--cache-dir",
        type=str,
        help="cache ROOT location (default <repo>/.cache); pools are "
        "{cache-dir}/{namespace} with namespace mirroring the corpus "
        "hierarchy, e.g. Cyrene/Chinese(PRC)",
    )
    shared.add_argument(
        "--namespaces",
        nargs="+",
        required=True,
        help="pool selector(s) under the cache root, e.g. "
        "Cyrene/Chinese(PRC); each namespace string IS its speaker name "
        "(export spk_id, GRPO sampling); sft takes K namespaces (K=1 = "
        "single-speaker); grpo takes the calibration pool selector(s) the "
        "text pool's speaker keys resolve against (multi-speaker pools: "
        "one .jsonl row per (speaker, text))",
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
    sft.add_argument(
        "--per-pool-cap",
        type=int,
        default=None,
        help="balanced head-slice per pool (train-side only; on a single "
        "pool this doubles as the debug limit)",
    )
    sft.add_argument(
        "--freeze",
        nargs="+",
        default=None,
        help="frozen components: subtalker talker text embedding blocks",
    )
    sft.add_argument("--sub-weight", type=float, default=None)
    sft.add_argument("--grad-checkpoint", action="store_true")
    sft.add_argument("--batch-size", type=int, default=None)
    sft.add_argument("--grad-accum", type=int, default=None)
    sft.add_argument("--epochs", type=int, default=None)
    sft.add_argument("--log-every", type=int, default=None)

    grpo = subparsers.add_parser("grpo", parents=[shared])
    grpo.add_argument("--lora-r", type=int, default=None)
    grpo.add_argument("--lora-alpha", type=float, default=None)
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
    grpo.add_argument("--logprob-micro", type=int, default=None)
    grpo.add_argument("--scorer-url", default=None)

    args = parser.parse_args()
    # CLI kwargs 直灌（未给的键 = None 已过滤 → 落回 dataclass 默认值）；
    # pool/speaker 选择器的存在性校验在 run_grpo/run_sft 内部 fail-loud
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key != "command" and value is not None
    }
    if args.command == "sft":
        run_sft(SftConfig(**overrides))
    else:
        # graphed KV pool + teacher-forcing peak need low-fragmentation
        # allocator segments (§47: micro=4 OOM'd with 1.58G stranded in
        # reserved-but-unallocated under the default allocator)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        run_grpo(TrainConfig(**overrides))


if __name__ == "__main__":
    main()
