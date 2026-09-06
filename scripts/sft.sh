dir="runs/sft/$(date +%s)"
mkdir -p "$dir"

setsid workers/trainer/.venv/bin/python workers/trainer/main.py sft \
  --namespaces \
    "cyrene/Chinese(PRC)" \
    "castorice/Chinese(PRC)" \
    "aglaea/Chinese(PRC)" \
    "hyacine/Chinese(PRC)" \
    "cipher/Chinese(PRC)" \
    "hysilens/Chinese(PRC)" \
    "cerydra/Chinese(PRC)" \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --device cuda:0 \
  --batch-size 8 \
  --grad-accum 1 \
  --epochs 1 \
  --lr 5e-6 \
  --warmup-steps 50 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --sub-weight 0.3 \
  --seed 0 \
  --out-dir "$dir" \
  --ckpt-every 100 \
  --log-every 5 \
  > "$dir/trainer.log" 2>&1 < /dev/null &

echo "setsid pid $! @ $dir"
