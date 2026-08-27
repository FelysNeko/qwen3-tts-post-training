"""Corpus → `.cache/{lang}` preprocessing: a disk-state `Snapshot` plus
functional stages.

    filter → clearvoice(48k) → codes(qwen_tts) → score(zmq)

The clearvoice 48k output is the ONLY derived audio artifact — every
downstream consumer resamples from it on the fly (SV `AF.resample`, ASR
`load_audio(path, 16000)`, MOS `librosa.resample`, speech_tokenizer
`encode(wav, sr=48000)`), so we keep no resampled copies.

`Cache.load` freezes the disk state: the validated+filtered corpus, the
still-valid cached asset rows (corpus checksum unchanged), and the
drop-reason log. The stage functions (`enhance_missing` / `score_missing` /
`sync`) are pure deltas on that snapshot — rows append live, so an
interrupted run keeps what it finished, and `sync` rewrites asset.jsonl
compacted before deriving metrics.json.

Validation is fail-loudly: a malformed manifest/asset line raises — no
silent skips. Manifest↔wav mismatches are not fatal: they are recorded in
`DropReasons`.

Scoring reuses `client/trainer.Client` (trainer-side bind of PUSH 5555 /
PULL 5556), so the resident scorer worker is oblivious to this caller. Run
the scorer WITHOUT --sv-ref/--metrics for preprocessing: `sim` comes back
None (the centroid only exists once every clip is embedded) and `vector`
carries the raw unit-norm ERes2NetV2 embedding. Codes extraction and scoring
overlap one chunk deep (extract chunk i+1 on cuda:1 while the scorer chews
chunk i on cuda:0) — the same zero-thread pipelining the GRPO loop uses.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from pydantic import BaseModel
from tqdm import tqdm

from qwen3_tts_post_training.client.protocol import ScoreItem
from qwen3_tts_post_training.client.trainer import Client

CLEARVOICE_SR = 48000

PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


@dataclass(frozen=True)
class Config:
    corpus: Path
    cache_dir: Path  # per-corpus dir (.cache/{lang})
    min_seconds: float = 0.1
    min_tokens: int = 2
    limit: int = 0  # first N manifest entries only (debug)

    @property
    def manifest(self) -> Path:
        return self.corpus.parent / f"{self.corpus.stem}.jsonl"

    @property
    def clearvoice_dir(self) -> Path:
        return self.cache_dir / "clearvoice"

    @property
    def asset_jsonl(self) -> Path:
        return self.cache_dir / "asset.jsonl"

    @property
    def metrics_json(self) -> Path:
        return self.cache_dir / "metrics.json"


class CorpusEntry(BaseModel):
    name: str
    text: str


class AssetEntry(CorpusEntry):
    transcript: str
    cer: float
    codes: list[list[int]]
    vector: list[float]
    utmosv2: float
    checksum: str


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), sr


def extract_codes(speech_tokenizer, wav: Path) -> list[list[int]]:
    audio, sr = load_mono(wav)
    codes = torch.as_tensor(speech_tokenizer.encode(audio, sr=sr).audio_codes[0])
    return codes.int().tolist()


def build_metrics(rows: list[dict], provenance: dict) -> dict:
    """Centroid = unit-norm mean of unit-norm per-clip vectors (same formula
    as playground/build_sv_reward.py); sim stats feed RewardConfig
    (sv_center/sv_scale), wer reflects domain ASR performance, utmosv2 stats
    are recorded for a future gate decision (tau stays 2.5 for now)."""
    assert rows, "no complete asset rows to build metrics from"
    vecs = np.stack([np.asarray(r["vector"], dtype=np.float64) for r in rows])
    norm = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    centroid = norm.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    sims = norm @ centroid
    cers = np.asarray([r["cer"] for r in rows], dtype=np.float64)
    mos = np.asarray([r["utmosv2"] for r in rows], dtype=np.float64)
    return {
        "centroid": centroid.tolist(),
        "sim": {"mean": float(sims.mean()), "std": float(sims.std())},
        "wer": float(cers.mean()),
        "utmosv2": {
            "mean": float(mos.mean()),
            "std": float(mos.std()),
            "percentiles": {str(p): float(np.percentile(mos, p)) for p in PERCENTILES},
        },
        "n_clips": len(rows),
        **provenance,
    }


@dataclass(frozen=True)
class DropReasons:
    orphan_corpus: tuple[str, ...]
    orphan_manifest: tuple[str, ...]
    less_than_min_tokens: tuple[str, ...]
    less_than_min_seconds: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus": {
                "orphan": self.orphan_corpus,
                "duration": self.less_than_min_seconds,
            },
            "manifest": {
                "orphan": self.orphan_manifest,
                "length": self.less_than_min_tokens,
            },
        }


@dataclass(frozen=True)
class Cache:
    """Immutable view of the disk state when preprocessing starts: the
    validated+filtered corpus, the still-valid cached asset rows, stale
    clearvoice wavs, and the drop-reason log. All stage computation is a pure
    delta on this."""

    config: Config
    corpus_entries: tuple[CorpusEntry, ...]
    asset_entries: tuple[AssetEntry, ...]
    drop_reasons: DropReasons

    @classmethod
    def load(cls, config: Config, token_counter: Callable) -> Cache:
        with open(config.manifest, encoding="utf-8") as file:
            corpus_entries = [CorpusEntry.model_validate_json(line) for line in file]

        wav_stems = [wav.stem for wav in config.corpus.glob("*.wav")]

        wav_stems_set = set(wav_stems)
        manifest_names_set = {e.name for e in corpus_entries}
        orphan_corpus_names = tuple(x for x in wav_stems if x not in manifest_names_set)
        orphan_manifest_names = tuple(
            e.name for e in corpus_entries if e.name not in wav_stems_set
        )

        corpus_entries = [e for e in corpus_entries if e.name in wav_stems_set]

        if config.limit:
            corpus_entries = corpus_entries[: config.limit]

        desired_corpus_entries = []
        less_than_min_tokens = []
        less_than_min_seconds = []

        for entry in corpus_entries:
            info = sf.info(str(config.corpus / f"{entry.name}.wav"))
            if token_counter(entry.text) < config.min_tokens:
                less_than_min_tokens.append(entry.name)
            elif info.frames / info.samplerate < config.min_seconds:
                less_than_min_seconds.append(entry.name)
            else:
                desired_corpus_entries.append(entry)

        drop_reasons = DropReasons(
            orphan_corpus=orphan_corpus_names,
            orphan_manifest=orphan_manifest_names,
            less_than_min_tokens=tuple(less_than_min_tokens),
            less_than_min_seconds=tuple(less_than_min_seconds),
        )

        desired_corpus_names = {e.name for e in desired_corpus_entries}

        config.cache_dir.mkdir(parents=True, exist_ok=True)
        config.asset_jsonl.touch(exist_ok=True)
        with open(config.asset_jsonl, encoding="utf-8") as file:
            asset_entries = [AssetEntry.model_validate_json(line) for line in file]
        asset_entries = tuple(
            entry
            for entry in asset_entries
            if entry.name in desired_corpus_names
            and entry.checksum == sha256(config.corpus / f"{entry.name}.wav")
        )

        return cls(
            config=config,
            corpus_entries=tuple(desired_corpus_entries),
            asset_entries=asset_entries,
            drop_reasons=drop_reasons,
        )


def enhance_missing(snapshot: Cache, *, device: str, overwrite: bool) -> None:
    from preprocess.clearvoice.decode import MossFormer2SE48KConfig, enhance
    from preprocess.clearvoice.load import (
        ensure_clearvoice,
        load_mossformer2_se_48k,
    )

    config = snapshot.config
    config.clearvoice_dir.mkdir(parents=True, exist_ok=True)
    todo = [
        e
        for e in snapshot.corpus_entries
        if overwrite or not (config.clearvoice_dir / f"{e.name}.wav").is_file()
    ]
    if not todo:
        return
    model = load_mossformer2_se_48k(ensure_clearvoice(), device)
    cfg = MossFormer2SE48KConfig()
    for entry in tqdm(todo, desc="clearvoice"):
        out = config.clearvoice_dir / f"{entry.name}.wav"
        if out.is_file() and not overwrite:
            continue
        audio, sr = load_mono(config.corpus / f"{entry.name}.wav")
        if sr != CLEARVOICE_SR:
            audio = AF.resample(torch.from_numpy(audio), sr, CLEARVOICE_SR).numpy()
        torch.manual_seed(0)  # kaldi fbank dither=1.0 consumes the global RNG
        enhanced = enhance(model, cfg, audio, device)
        sf.write(str(out), enhanced, CLEARVOICE_SR, subtype="PCM_16")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_missing(
    snapshot: Cache,
    *,
    speech_tokenizer,
    client: Client,
    batch: int,
    overwrite: bool,
) -> list[AssetEntry]:
    config = snapshot.config
    valid = {e.name for e in snapshot.asset_entries}
    todo = (
        list(snapshot.corpus_entries)
        if overwrite
        else [e for e in snapshot.corpus_entries if e.name not in valid]
    )
    print(f"[score] {len(snapshot.asset_entries)} cached rows, {len(todo)} to process")

    rows: list[AssetEntry] = []
    inflight: deque[tuple[int, list[CorpusEntry], list[list[list[int]]]]] = deque()

    def drain() -> None:
        rid, chunk, chunk_codes = inflight.popleft()
        results = client.recv_score(rid, timeout=client.timeout_s)
        for entry, codes, r in zip(chunk, chunk_codes, results):
            row = AssetEntry(
                name=entry.name,
                text=entry.text,
                transcript=r["transcript"],
                cer=r["cer"],
                codes=codes,
                vector=r["vector"],
                utmosv2=r["mos"],
                checksum=sha256(config.corpus / f"{entry.name}.wav"),
            )
            with open(config.asset_jsonl, "a", encoding="utf-8") as file:
                file.write(row.model_dump_json() + "\n")
            rows.append(row)

    t0 = time.monotonic()
    for i in tqdm(range(0, len(todo), batch), desc="score"):
        chunk = todo[i : i + batch]
        chunk_codes = [
            extract_codes(
                speech_tokenizer,
                config.clearvoice_dir / f"{e.name}.wav",
            )
            for e in chunk
        ]
        rid = client.send_score(
            [
                ScoreItem(
                    wav_path=str(config.clearvoice_dir / f"{e.name}.wav"),
                    text=e.text,
                )
                for e in chunk
            ]
        )
        inflight.append((rid, chunk, chunk_codes))
        if len(inflight) >= 2:
            drain()
    while inflight:
        drain()
    print(f"[score] done in {time.monotonic() - t0:.1f}s")
    return rows


def sync(
    snapshot: Cache,
    *,
    speech_tokenizer,
    client: Client,
    device: str,
    batch: int,
    model_path: str = "",
    overwrite: bool = False,
) -> None:
    config = snapshot.config
    enhance_missing(snapshot, device=device, overwrite=overwrite)
    incremental = score_missing(
        snapshot,
        speech_tokenizer=speech_tokenizer,
        client=client,
        batch=batch,
        overwrite=overwrite,
    )

    entries = snapshot.asset_entries + tuple(incremental)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    with open(config.asset_jsonl, "w", encoding="utf-8") as file:
        for each in entries:
            print(each.model_dump_json(), file=file)

    metrics = build_metrics(
        [e.model_dump() for e in entries],
        provenance={
            "dataset": str(config.corpus.resolve()),
            "model_path": model_path,
            "clearvoice": "MossFormer2_SE_48K",
            "sv_model": "eres2netv2",
            "min_tokens": config.min_tokens,
            "min_seconds": config.min_seconds,
            "dropped": snapshot.drop_reasons.to_dict(),
        },
    )
    with open(config.metrics_json, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    print(f"[metrics] written: {config.metrics_json}")


def run_pipeline(
    dataset: Path,
    cache_root: Path,
    tokenize_text,
    speech_tokenizer,
    client: Client,
    device: str,
    model_path: str,
    min_tokens: int,
    min_seconds: float,
    batch: int,
    limit: int,
) -> Path:
    config = Config(
        corpus=dataset,
        cache_dir=cache_root / dataset.name,
        min_tokens=min_tokens,
        min_seconds=min_seconds,
        limit=limit,
    )
    snapshot = Cache.load(config, token_counter=tokenize_text)
    sync(
        snapshot,
        speech_tokenizer=speech_tokenizer,
        client=client,
        device=device,
        batch=batch,
        model_path=model_path,
        overwrite=False,
    )
    return config.cache_dir
