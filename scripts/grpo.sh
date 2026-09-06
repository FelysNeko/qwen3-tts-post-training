dir="runs/grpo/$(date +%s)"
mkdir -p "$dir"

setsid workers/trainer/.venv/bin/python workers/trainer/main.py grpo \
  --namespaces \
    "cyrene/Chinese(PRC)" \
    "castorice/Chinese(PRC)" \
    "aglaea/Chinese(PRC)" \
    "hyacine/Chinese(PRC)" \
    "cipher/Chinese(PRC)" \
    "hysilens/Chinese(PRC)" \
    "cerydra/Chinese(PRC)" \
  --text-pool-path archive/grpo.jsonl \
  --model-path runs/sft/1788731625/export \
  --device cuda:0 \
  --lora-r 16 \
  --lora-alpha 64 \
  --num-prompts 8 \
  --group-size 8 \
  --num-steps 400 \
  --seed 0 \
  --token-budget 1024 \
  --token-budget-infer 1024 \
  --temperature 0.9 \
  --top-k 50 \
  --sampler-impl graphed \
  --variant dr \
  --kl-beta 0.001 \
  --logprob-micro 0 \
  --lr 1e-5 \
  --warmup-steps 20 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --lam-mos 0.2 \
  --lam-p835 0.0 \
  --mos-role floor \
  --scorer-url http://127.0.0.1:8000 \
  --out-dir "$dir" \
  --ckpt-every 2 \
  > "$dir/trainer.log" 2>&1 < /dev/null &

echo "setsid pid $! @ $dir"

# ---- four-arm MOS-ablation differentials (edit the flags above) ----
# arm 2 (P835 driver): --num-steps 400 --lr 1e-5 --ckpt-every 2 --lam-mos 0 --lam-p835 0.4 --out-dir runs/grpo_arm2_p835
# arm 1 (no MOS):      --num-steps 400 --lr 1e-5 --ckpt-every 2 --lam-mos 0 --lam-p835 0 --out-dir runs/grpo_arm1_nomos
# arm 3 (raw UTMOS):   --num-steps 400 --lr 1e-5 --ckpt-every 2 --lam-mos 0.2 --mos-role raw --out-dir runs/grpo_arm3_rawmos
# arm 4 (defaults):    --num-steps 400 --lr 1e-5 --ckpt-every 2 --out-dir runs/grpo_arm4_default
# run order: 2 -> 1 -> 4 -> 3 (serial)
# single-GPU server: --device cuda:0 (shares the card with the scorer, ~21G/24G; dual-GPU keeps cuda:1)
