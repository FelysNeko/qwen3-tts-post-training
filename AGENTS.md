# AGENTS.md

GRPO post-training pipeline for Qwen3-TTS CustomVoice. Python 3.12, uv-managed. No tests, no CI.

## Layout

- `src/qwen3_tts_post_training/` — core lib (pure torch, no model runtime): `reward/reward.py` (reward_v3), `scorers/` (JSON-lines protocol + subprocess client), `train/grpo.py` (losses). Installed editable into both workers.
- `workers/trainer/` — GRPO trainer worker (default `cuda:1`). Entry: `workers/trainer/main.py`.
- `workers/scorer/` — resident scoring worker (default `cuda:0`): SV + ASR/CER + MOS. Entry: `workers/scorer/main.py`.
- Each worker has its own `pyproject.toml` and `.venv`; deps are not unified at the root. Root `.venv` has only torch + ruff.

## Commands

```sh
uv sync                     # root: installs ruff into .venv
.venv/bin/ruff check .      # lint (all checks currently pass)
uv sync                     # inside workers/trainer or workers/scorer: worker deps
workers/trainer/.venv/bin/python workers/trainer/main.py [args]   # run trainer
workers/scorer/.venv/bin/python workers/scorer/main.py [args]     # run scorer standalone (rare; normally spawned)
```

- Ruff is the only check. `[tool.ruff.lint.per-file-ignores]` exempts vendored `workers/scorer/src/scorer/speakerlab/**` with `ALL` — do not "fix" style there.
- torch comes from the `pytorch-cu128` uv index; flash-attn is a pinned cu12 wheel in the root pyproject. Python must be 3.12 (`.python-version`).

## Runtime wiring (not obvious from filenames)

- Trainer spawns the scorer as a subprocess (`ScorerClient` in `src/qwen3_tts_post_training/scorers/client.py`): `workers/scorer/.venv/bin/python workers/scorer/main.py`, JSON lines over stdin/stdout, lock-step (one in-flight request). In `serve.py` stdout is dup'ed to an fd reserved for protocol and `sys.stdout` remapped to stderr — never print to stdout in scorer code.
- Trainer defaults to `cuda:1`, scorer to `cuda:0` (one CUDA context per process).
- Rollout wavs are written to per-run tmpfs dirs under `/dev/shm` and the scorer reads them by path; they are reclaimed by the OS.
- Scorer/SV model weights auto-fetch on first load via the upstream tools (no manual downloads): SV ckpts through the `modelscope` client, UTMOSv2 folds through `huggingface_hub` from the official `sarulab-speech/UTMOSv2`, ASR/wav2vec2 through transformers. See `workers/scorer/src/scorer/fetch.py`.
- Defaults are overridable: model ckpt `--model-path` (default `/mnt/d/Repository/models/PhiLia093-TTS/`, not present until downloaded — trainer raises a clear FileNotFoundError). SV assets resolve to the sibling `playground/` dir without a hardcoded username — `Q3TTS_ROOT` overrides repo root, `Q3TTS_PLAYGROUND` overrides the playground dir. Scorer worker `--sv-dir` can point at a local 3D-Speaker checkout.
- Checkpoints save LoRA deltas + codec head + optimizer state; `--resume` reads `out_dir/latest`.

## Vendored code — byte-faithful, do not modify

- `workers/scorer/src/scorer/speakerlab/**` — 3D-Speaker model defs; kept verbatim so SV ckpts load bit-identically.
- `workers/scorer/src/scorer/utmos/**` — bit-exact vs upstream UTMOSv2 @ cc2700db.
- MOS scoring is deterministic only with fixed seed + 8 averaged reps + `num_workers=0` (sequential); do not parallelize or "improve" this.

## Algorithm invariants

- Reference policy = the SAME model with LoRA disabled (`TrainerModel.set_adapter(False)`) — only one model in VRAM. Trainable: rsLoRA (r=16, α=64) on talker MLP only + `codec_head`; everything else frozen (MTP γ=0).
- Rollout must use `language="Auto"` and `repetition_penalty=None`; the teacher-forcing logprob reconstruction (`logprob.py`) is only valid under the Auto prefill layout and is checked bit-consistent vs generation.
- Logprobs replicate the sampling distribution (temperature, top_k, suppress band) — policy ratio/KL depend on this exactness.
- Rewards are composed by `reward_v3` (std-normalized SV/WER, MOS flameout when group std < eps); group resample criterion uses only SV/WER variance (MOS bimodal excluded). Docstrings cite external design docs (MD §N, playground notes) as design truth — don't "simplify" these decisions.
