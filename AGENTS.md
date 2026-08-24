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
- Defaults are overridable: model ckpt `--model-path` (default `/mnt/d/Repository/models/PhiLia093-TTS/`, not present until downloaded — trainer asserts a clear message). SV assets resolve to the sibling `playground/` dir without a hardcoded username — `Q3TTS_ROOT` overrides repo root, `Q3TTS_PLAYGROUND` overrides the playground dir. Scorer worker `--sv-dir` can point at a local 3D-Speaker checkout.
- Checkpoints save LoRA deltas + codec head + optimizer state; `--resume` reads `out_dir/latest`.

## Vendored code — byte-faithful, do not modify

- `workers/scorer/src/scorer/speakerlab/**` — 3D-Speaker model defs; kept verbatim so SV ckpts load bit-identically.
- `workers/scorer/src/scorer/utmos/**` — bit-exact vs upstream UTMOSv2 @ cc2700db.
- MOS scoring is deterministic only with fixed seed + 8 averaged reps + `num_workers=0` (sequential); do not parallelize or "improve" this.

## Algorithm invariants

- Reference policy = the SAME model with LoRA disabled (`TrainerModel.set_adapter(False)`) — only one model in VRAM. Trainable: rsLoRA (r=16, α=64) on talker MLP only + `codec_head`; everything else frozen (MTP γ=0).
- Rollout must use `language="Auto"`; sampling contract is fixed at temperature+top_k (top_p pinned to official 1.0, repetition_penalty pinned to None — deliberate deviation from the official inference default rep_penalty=1.05, which is a serving-only setting; RL rollout must stay stateless for the logprob reconstruction). The teacher-forcing logprob reconstruction (`logprob.py`) is only valid under the Auto prefill layout and is checked bit-consistent vs generation.
- Sampler impls (`--sampler-impl`, PROJECT_STATUS §9; all under `workers/trainer/src/trainer/samplers/`, ABC = `base.Sampler`, factory = `build_sampler`): `hf` (reference, `hf.py`), `fast` (hand-rolled eager loop in `eager.py`, `EagerSampler`, bit-equal vs `hf` incl. the RNG stream — verified greedy + seeded), `compiled` (same loop + `torch.compile(dynamic=None, epilogue_fusion=False)` on the two backbone forwards, `torch_compile.py`, `TorchCompileSampler`, ~2.4x, fixed batch = group_size, mismatched batch sizes raise — use `fast` for anything else), `graphed` (CUDA-graph decode in `cuda_graph.py`, `CudaGraphSampler`, ~4.8x: prefill eager into a static KV pool, then one captured 1-token step per replay; `flash_attn_with_kvcache` attention, fixed batch = group_size, mismatched batch sizes raise — use `fast` for anything else).
- `compiled` numerically drifts vs eager (inductor float reassociation; greedy argmax can flip at step 0) — bit-equality with `hf` does NOT hold; its contract is bitwise self-reproducibility after the built-in warmup (two dummy generations in `TorchCompileSampler.__init__` absorb dynamo's static->dynamic promotion). Do not remove the warmup and do not assume a fresh cold process reproduces a warm process's rollout.
- `graphed` non-obvious invariants (each one was a real bug, probe C1v7-C1v9): (1) capture MUST use an explicit `torch.cuda.Stream` — on WSL2 secondary GPUs (cuda:1) default-stream capture records an EMPTY graph (vLLM sidesteps the same bug with dedicated capture streams); (2) BOTH attention classes must be patched — the talker uses `Qwen3TTSTalkerAttention` (mrope) while the code predictor uses `Qwen3TTSAttention` (plain RoPE); a patch on only one leaves the other context-free (no cache reads/writes); (3) FA2 kernels return seq-major `[B, L, H, hd]` — reshape directly, never `.transpose(1, 2)` (transposing is invisible for L==1 decode steps but scrambles heads-vs-seq in multi-token prefills); (4) LoRA params are stored pre-transposed (`[in, r]` / `[r, out]`) — a strided `.t()` view makes cuBLAS materialize the weight at capture time, baking values into the graph so optimizer updates never reach replays.
- torch.compile notes: `epilogue_fusion=False` is required (RMSNorm/RoPE epilogue fusions lose fp32 precision, compounding over AR steps — vllm-omni's finding); dynamo cache limit is raised to 256 in `samplers/torch_compile.enable_compile` (LoRA on/off guard variants alone ≈ 200 graphs); LoRA adapter switching under compile is verified working (two guard variants, each compiled once).
- Logprobs are evaluated on the FULL temperature-scaled softmax (fp32) — the sampling-time top_k/suppress masks are deliberately NOT re-applied. Replaying a hard truncation makes log-probs discontinuous: a 1e-5 weight nudge (one clipped Adam step) flips borderline tokens across the k-th boundary → ±inf log-ratios → KL=inf / grad_norm ~1e4 (diagnosed C1v10; root cause behind every earlier NaN and runaway-policy smoke). Sampled tokens always lie in the sampling support, so unmasked log-probs are finite; top-50 mass at T=0.9 ≈ 1.
- Rewards (`reward_v3`, v3.1) are RAW-magnitude weighted sums (no within-group std division): Dr.GRPO subtracts the group mean itself, and std-normalizing amplifies pure ranking noise on flat groups to full scale (one Adam step along it collapsed the policy, smoke C1v8/C1v9). A component with within-group std < `flameout_eps` is zeroed (dead by construction); zero-signal groups (no WER spread) are skipped entirely (`needs_resample`). Docstrings cite external design docs (MD §N, playground notes) as design truth — don't "simplify" these decisions.
- Training layout (Fish Audio S2 style): `num_prompts` distinct prompts × `group_size` rollouts per step (default 8×8=64), one optimizer update per step via gradient accumulation (equal group weighting, per-group Dr.GRPO advantage). Linear LR warmup (`warmup_steps`, default 10) is REQUIRED — Adam's first step moves every param ±lr at once (sign step), jolting the distribution (observed KL 586 on step 1 without warmup, ~0.003 with). Guards that double as diagnostics: per-group runaway skip (`t_max > runaway_t_max`), flat-group skip, non-finite loss drop.

## Verification playbook — how a rollout/training change is proven correct

Layered isolation: each level only trusts the levels below it. When two impls disagree, drop one level down until they agree — the bug lives between the agreeing level and the disagreeing one. Every level compares with IDENTICAL inputs (same tensors, same seeds) so exactly one code path differs. Probe scripts live under `/tmp/opencode/` during a session (ephemeral — the method matters, not the script).

### L1 — kernel units (same tensors in, numbers out)
- A/B the two kernel families on identical tensors: `flash_attn_with_kvcache` vs `flash_attn_func` with the same q/k/v → expect bit-equal (0.0) or bf16 kernel-family tolerance (~0.004). Use REAL model tensors (post-projection, post-RoPE), not synthetic ones — a synthetic pass + real failure means the bug is in how the kernel is fed, not the kernel.
- Check kernel side effects explicitly: pool writes land at the right positions (`torch.equal(pool[:, :L], k_new)`), `cache_seqlens` is NOT mutated in place by the kernel (shared-seqlens designs depend on this).

### L2 — single-layer integration (hooks, not reconstructions)
- Monkeypatch a wrapper around `layer.self_attn.forward` that CAPTURES the real (hidden_states, position_embeddings, attention_mask, cache_position) during an actual model forward, then re-run both the original and the patched attention on the same captured inputs → `max_abs`. This found the head/seq transpose scramble (7.56) at a moment when L1 showed 0.0 — i.e. kernel correct, integration wrong. Prefer capturing real inputs over rebuilding them (rebuilt position embeddings reproduce your own misunderstanding, not the model's).

### L3 — full prefill equivalence (whole-model forward)
- Same prompt batch through both paths → prefill logits `max_abs` (expect ~0 vs the eager reference; 36.9 when broken). Then same-seed first-token check (sampled id must match: 404 == 404). Cheapest whole-model truth test — run it before any behavioral test.

### L4 — end-to-end rollout behavior
- Self-reproducibility: same seed, run twice, `torch.equal` on every output tensor (built into `CudaGraphSampler._warmup_graph`). Catches nondeterminism and state leakage between calls.
- EOS firing / step count: a healthy policy stops in a sane band (~150-250 steps for normal sentences). 511 ≈ max_new_tokens means no EOS — runaway detector AND correctness canary (first symptom of the prefill scramble).
- Cross-impl distribution-level equivalence: different kernel families will NOT be bit-equal (graphed 217 vs eager 178 steps is fine); the contract is same distribution, not same numbers.
- Wav decode sanity: durations plausible for the text length.
- CUDA-graph capture probes (in `_capture_graph`): replay output == eager output AND perturbing an input buffer changes the replay output — the second check catches empty/corrupt captures (WSL2 secondary-GPU default-stream capture records nothing; replay==eager==stale-buffer trivially passes the first check alone).

### L5 — RL interaction (the rollout is also a training input)
- Weight in-place pickup: perturb a trainable parameter in place → rollout must CHANGE; restore → must reproduce the original (catches graph weight-baking). Perturb a parameter that MATTERS: with zero-initialized `lora_b`, perturbing `lora_a` is a no-op ((x@A)@0 ≡ 0) — a false "baked" verdict.
- Gradient equivalence: same rollout codes + synthetic advantages → backward → total grad_norm comparable across sampler impls (58 vs 63 observed). Decouples "rollout path is wrong" from "training path is wrong".
- Update bound: after grad clip, one Adam step moves each param by ≤ lr (~1e-5). If observed KL explodes (grad 2976, KL 942) while |dw| ≤ lr, the update is fine — the METRIC is lying (discontinuity), see L6.
- Trajectory monitoring: KL / grad_norm / cer / sim across ≥6 steps. Bounded oscillation that decays = healthy transient; monotone blowup = divergence. KL=586 on step 1 with sane rollouts = Adam cold-start jolt (fix = warmup), NOT a broken path.

### L6 — number fingerprinting (read magnitudes as bug signatures)
- `inf` / NaN in log-ratios → hard discontinuity (top-k/suppress mask replay: a 1e-5 nudge flips the k-th boundary). Never ship a hard truncation inside a differentiable path.
- ~1e6 sentinel values (mean_R = 1000006) → division by an eps clamp (std≈0 group): the "guard" was the bug.
- max_abs 5-40 on hidden/logits with bit-equal kernels → layout scramble (heads vs seq), not arithmetic.
- Step count = max_new_tokens → no EOS → upstream prefill/position corruption.
- t_max ≈ lmax (979/1024) in TRAINING rollouts only → policy damage from a bad update (check the ckpt diff: per-param-group |dw| tells you which update misfired; a fresh-model vs ckpt diff of ~0.5 on lora_a is just two independent kaiming inits, not damage).

### Operating rules
- Experiments over theory: every hypothesis gets a minimal repro before it is believed; prior conclusions (including this file's) are re-verifiable, not authoritative.
- One variable per experiment: identical inputs/seeds, exactly one code path differing.
- Decouple before blaming: same weights × two samplers, same codes × surrogate loss — isolate which side of the pipeline a regression lives on before touching code.
- Heavy experiments: one at a time on this machine (two model stacks ≈ system OOM in WSL2); `setsid` background + short-poll so a shell timeout cannot kill or orphan the run.
