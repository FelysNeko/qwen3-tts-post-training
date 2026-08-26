"""Thin CLI launcher for the GRPO loop (Phase 4)."""

from __future__ import annotations

import argparse

from trainer.loop import TrainConfig, run_grpo


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GRPO trainer worker")
    p.add_argument("--model-path", default=TrainConfig.model_path)
    p.add_argument("--device", default=TrainConfig.device)
    p.add_argument("--dtype", default=TrainConfig.dtype)
    p.add_argument("--lora-r", type=int, default=TrainConfig.lora_r)
    p.add_argument("--lora-alpha", type=float, default=TrainConfig.lora_alpha)
    p.add_argument("--speaker", default=TrainConfig.speaker)
    p.add_argument("--num-prompts", type=int, default=TrainConfig.num_prompts)
    p.add_argument("--group-size", type=int, default=TrainConfig.group_size)
    p.add_argument("--token-budget", type=int, default=TrainConfig.token_budget)
    p.add_argument(
        "--token-budget-infer", type=int, default=TrainConfig.token_budget_infer
    )
    p.add_argument("--text-pool-path", default=TrainConfig.text_pool_path)
    p.add_argument("--num-steps", type=int, default=TrainConfig.num_steps)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--temperature", type=float, default=TrainConfig.temperature)
    p.add_argument("--top-k", type=int, default=TrainConfig.top_k)
    p.add_argument(
        "--sampler-impl",
        default=TrainConfig.sampler_impl,
        choices=["hf", "fast", "compiled", "graphed"],
    )
    p.add_argument(
        "--variant", default=TrainConfig.variant, choices=["vanilla", "dr", "gspo"]
    )
    p.add_argument("--kl-beta", type=float, default=TrainConfig.kl_beta)
    p.add_argument("--warmup-steps", type=int, default=TrainConfig.warmup_steps)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    p.add_argument("--grad-clip", type=float, default=TrainConfig.grad_clip)
    p.add_argument(
        "--scorer-push-endpoint", default=TrainConfig.scorer_push_endpoint
    )
    p.add_argument(
        "--scorer-pull-endpoint", default=TrainConfig.scorer_pull_endpoint
    )
    p.add_argument("--scorer-timeout", type=float, default=TrainConfig.scorer_timeout)
    p.add_argument("--out-dir", default=TrainConfig.out_dir)
    p.add_argument("--ckpt-every", type=int, default=TrainConfig.ckpt_every)
    p.add_argument("--resume", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = TrainConfig(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        speaker=args.speaker,
        num_prompts=args.num_prompts,
        group_size=args.group_size,
        token_budget=args.token_budget,
        token_budget_infer=args.token_budget_infer,
        text_pool_path=args.text_pool_path,
        num_steps=args.num_steps,
        seed=args.seed,
        temperature=args.temperature,
        top_k=args.top_k,
        sampler_impl=args.sampler_impl,
        variant=args.variant,
        kl_beta=args.kl_beta,
        warmup_steps=args.warmup_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        scorer_push_endpoint=args.scorer_push_endpoint,
        scorer_pull_endpoint=args.scorer_pull_endpoint,
        scorer_timeout=args.scorer_timeout,
        out_dir=args.out_dir,
        ckpt_every=args.ckpt_every,
        resume=args.resume,
    )
    run_grpo(cfg)


if __name__ == "__main__":
    main()
