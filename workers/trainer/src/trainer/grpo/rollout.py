"""GRPO rollout pipeline: prompts → code groups → 24 kHz wavs → scorer →
`list[Sample]`.

`pipelined_rollout` is the single entry and the only consumer of the
scorer client: per prompt it rolls a group out under the current policy
(LoRA adapters on), renders wavs into a per-run tmpfs dir, submits them to
the scorer EAGERLY (the rollout∥score overlap lives exactly here — the
POST fires while later groups are still rolling out), then polls until
every group is scored. `compute_advantage` is a pure function of
(prompt, results): the `Prompt` carries its pool's whole calibration
(unit-norm centroid embedding + `RewardConfig`), so nothing else is
plumbed through — `list[Sample]` is the sole carrier of everything the
gradient update and the monitor consume.

Rollouts run under the current policy. Each group is seeded via
torch.manual_seed so identical (prompts, seed) reproduce the same codes.
Wavs go to a per-run tmpfs dir under /dev/shm — the scorer reads them by
path, and the trainer unlinks each group's wavs after scoring (an atexit
sweep catches leftovers from crashed runs).
"""

from __future__ import annotations

import json
import time
import wave
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from qwen3_tts_post_training.client.protocol import ScoreItem, ScoreResult
from qwen3_tts_post_training.client.trainer import Client
from qwen3_tts_post_training.text import cer, normalize
from trainer.grpo.grpo import ADV_DR, group_advantage
from trainer.grpo.reward import RewardBreakdown, RewardConfig, reward_v3
from trainer.grpo.samplers.base import Sampler
from trainer.model import ModelWrapper


def write_wav(path: str | Path, audio: np.ndarray, sr: int) -> Path:
    path = Path(path)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return path


def _cleanup_wavs(wav_paths: list[Path]) -> None:
    for p in wav_paths:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
    # try remove parent dir if empty
    if wav_paths:
        try:
            wav_paths[0].parent.rmdir()
        except OSError:
            pass


@dataclass
class Prompt:
    """One pool item plus its pool's whole calibration — self-contained by
    contract, so `compute_advantage` needs nothing besides (prompt, results).

    ``embedding`` is the pool's unit-norm centroid row [hidden], already on
    the training device (sims are caller-side: one matmul against it).
    ``reward_cfg`` is the per-speaker `RewardConfig` (sv stats from that
    pool's metrics.json + the run-level λ/mos_role knobs baked in at
    construction); ``reward_cfg.lam_p835 > 0`` also arms the scorer request.
    """

    loc: int
    speaker: str
    text: str
    embedding: torch.Tensor
    reward_cfg: RewardConfig


@dataclass
class Sample:
    """The sole data carrier from the pipeline to the gradient update and
    the monitor: one (prompt × group_size) rollout unit, fully scored.
    ``advantage`` is precomputed on the FULL group (never a slice) — the
    training side only slices it per micro-chunk."""

    prompt: Prompt
    codes: list[torch.Tensor]
    t_max: int
    advantage: torch.Tensor
    R: torch.Tensor
    bd: RewardBreakdown
    sim: torch.Tensor
    cer: torch.Tensor
    mos: torch.Tensor
    p835: torch.Tensor | None


@dataclass(frozen=True)
class RolloutParams:
    """Loop-position + sampling-contract scalars for one step — the only
    things `pipelined_rollout` needs beyond the self-contained prompts."""

    step: int
    seed_base: int
    temperature: float
    top_k: int
    token_budget: int


@dataclass
class RolloutResult:
    prompt: str
    codes: list[torch.Tensor]
    wav_paths: list[Path]
    fs: int
    cur_len: (
        int  # prefill length for token_budget accounting (no default: must be explicit)
    )


def rollout_group(
    sampler: Sampler,
    ttm: ModelWrapper,
    prompt: str,
    seed: int,
    tag: str,
    *,
    speaker: str,
    temperature: float,
    top_k: int,
    token_budget: int,
    work_dir: Path | None = None,
) -> RolloutResult:
    """Sample one group and render it to wav. Returns codes + wav paths.

    ``prompt`` is a single text, internally repeated `sampler.batch_size`
    times to form the GRPO group (homogeneous, no assert needed).
    ``speaker`` is the voice for this group (multi-speaker GRPO pools roll
    out one (speaker, text) pair per group).
    ``token_budget`` is total tokens (prefill cur_len + new) budget;
    ``max_new = token_budget - cur_len`` is derived inside the sampler.
    The non-varying part of the RL sampling contract is pinned here
    (do_sample=True governs both loops, subtalker trio at 0.9/50); probes
    that need other values call the sampler directly."""
    codes, cur_len = sampler.sample(
        prompt,
        seed=seed,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        token_budget=token_budget,
        subtalker_temperature=0.9,
        subtalker_top_k=50,
        speaker=speaker,
    )
    wavs, fs = ttm.decode(codes)
    if work_dir is None:
        work_dir = Path("/dev/shm") / f"grpo_{tag}"
    work_dir.mkdir(parents=True, exist_ok=True)
    wav_paths = [
        write_wav(work_dir / f"{tag}_{i}.wav", wav, fs) for i, wav in enumerate(wavs)
    ]
    return RolloutResult(prompt, codes, wav_paths, fs, cur_len=cur_len)


def _log_group(
    f,
    step: int,
    gi: int,
    speaker: str,
    prompt: str,
    cer=None,
    sim=None,
    skipped: bool = False,
    reason: str | None = None,
) -> None:
    """Per-group telemetry: one jsonl row per sampled group, written from
    quantities the skip/training path already computes. Offline analysis
    (per-source/per-length flat rates, who supplies the signal) joins rows
    via (step, gi) — draws are rng-replayable (seed * 1000003 + step)."""
    if f is None:
        return
    f.write(
        json.dumps(
            {
                "step": step,
                "gi": gi,
                "speaker": speaker,
                "chars": len(prompt),
                "cer_mean": round(float(cer.mean()), 5) if cer is not None else None,
                "cer_std": round(float(cer.std(unbiased=False)), 5)
                if cer is not None
                else None,
                "sim_std": round(float(sim.std(unbiased=False)), 5)
                if sim is not None
                else None,
                "skipped": skipped,
                "reason": reason,
            }
        )
        + "\n"
    )
    f.flush()  # survive hard freezes — this log exists to be post-mortem evidence


@dataclass
class GroupScore:
    """What `compute_advantage` derives from (prompt, results) alone — the
    pipeline zips it with the rollout's codes into a `Sample`."""

    advantage: torch.Tensor
    R: torch.Tensor
    bd: RewardBreakdown
    sim: torch.Tensor
    cer: torch.Tensor
    mos: torch.Tensor
    p835: torch.Tensor | None


def compute_advantage(prompt: Prompt, results: list[ScoreResult]) -> GroupScore:
    """Pure (prompt, results) → per-take score bundle + advantage.

    sim is caller-side (raw unit-norm embeddings @ the pool centroid riding
    on the prompt); CER is client-side (normalized edit distance between the
    group's prompt and each take's transcript); `reward_v3` assembles R with
    per-component flameout zeroing (dead components contribute nothing, so
    easy-text groups degenerate into sim-only advantages instead of being
    skipped — needs_resample was removed 2026-09-05, STATUS §54).
    The advantage is pinned to the Dr.GRPO form A = R − mean — the form
    shared by the fish/dr/gspo loss variants (the legacy std-normalized
    "vanilla" variant is not reachable from the pipelined rollout)."""
    device = prompt.embedding.device
    sim = (
        torch.tensor(
            [r.get_embedding_unwrap() for r in results],
            dtype=torch.float32,
            device=device,
        )
        @ prompt.embedding
    )
    cer_t = torch.tensor(
        [
            cer(normalize(prompt.text), normalize(r.get_transcript_unwrap()))
            for r in results
        ],
        dtype=torch.float32,
        device=device,
    )
    mos = torch.tensor(
        [r.get_utmosv2_unwrap() for r in results],
        dtype=torch.float32,
        device=device,
    )
    p835 = (
        torch.tensor(
            [r.get_p835_unwrap() for r in results],
            dtype=torch.float32,
            device=device,
        )
        if prompt.reward_cfg.lam_p835 > 0
        else None
    )
    R, bd = reward_v3(sim, cer_t, mos, prompt.reward_cfg, p835=p835)
    A, _, _ = group_advantage(R, ADV_DR)
    return GroupScore(
        advantage=A, R=R, bd=bd, sim=sim, cer=cer_t, mos=mos, p835=p835
    )


def pipelined_rollout(
    sampler: Sampler,
    client: Client,
    prompts: list[Prompt],
    params: RolloutParams,
    *,
    groups_f,
    skips: Counter,
) -> tuple[list[Sample], float, float]:
    """Roll every prompt's group, submit to the scorer as each rollout lands
    (submit never fails — an unreachable scorer just defers the send to the
    poll loop), then poll until every group comes back — the client absorbs
    scorer death and restarts (auto re-send), so a group is never dropped.
    Runaway groups (``t_max + cur_len >= token_budget``) skip BEFORE submit
    — that budget is the training-forward OOM guard, not a diagnostic.
    The trainer owns tmpfs wav unlink: every group's wavs die here, either
    at the runaway skip or right after its results arrive.
    Returns (samples, t_rollout, t_score)."""
    t0 = time.monotonic()
    todo: deque[tuple[int, Prompt, RolloutResult, int, int]] = deque()
    for gi, prompt in enumerate(prompts):
        seed = params.seed_base * 1000003 + params.step * 1009 + gi
        tag = f"step{params.step}g{gi}"
        rollout = rollout_group(
            sampler,
            sampler.ttm,
            prompt.text,
            seed,
            tag,
            speaker=prompt.speaker,
            temperature=params.temperature,
            top_k=params.top_k,
            token_budget=params.token_budget,
        )
        t_max = max(c.shape[0] for c in rollout.codes)
        if t_max + rollout.cur_len >= params.token_budget:
            skips["runaway"] += 1
            _log_group(
                groups_f,
                params.step,
                gi,
                prompt.speaker,
                prompt.text,
                skipped=True,
                reason="runaway",
            )
            _cleanup_wavs(rollout.wav_paths)
            continue
        handle = client.submit(
            [ScoreItem(wav_path=str(p)) for p in rollout.wav_paths],
            asr=True,
            utmosv2=True,
            p835=prompt.reward_cfg.lam_p835 > 0,
            sv=True,
        )
        todo.append((gi, prompt, rollout, t_max, handle))
    t_rollout = time.monotonic() - t0

    t1 = time.monotonic()
    samples: list[Sample] = []
    while todo:
        still: list[tuple[int, Prompt, RolloutResult, int, int]] = []
        for gi, prompt, rollout, t_max, handle in todo:
            results = client.poll(handle)
            if results is None:
                still.append((gi, prompt, rollout, t_max, handle))
                continue
            _cleanup_wavs(rollout.wav_paths)
            scored = compute_advantage(prompt, results)
            samples.append(
                Sample(
                    prompt=prompt,
                    codes=rollout.codes,
                    t_max=t_max,
                    advantage=scored.advantage,
                    R=scored.R,
                    bd=scored.bd,
                    sim=scored.sim,
                    cer=scored.cer,
                    mos=scored.mos,
                    p835=scored.p835,
                )
            )
            _log_group(
                groups_f,
                params.step,
                gi,
                prompt.speaker,
                prompt.text,
                cer=scored.cer,
                sim=scored.sim,
            )
        todo = deque(still)
        if todo:
            time.sleep(client.poll_interval)
    return samples, t_rollout, time.monotonic() - t1
