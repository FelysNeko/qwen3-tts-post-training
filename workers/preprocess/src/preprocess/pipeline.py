"""Corpus → `.cache/{lang}` preprocessing: checksum-checked per-stage caches.

    filter → corpus → enhanced → codes → embedding

The enhanced 48k wav is the ONLY derived audio artifact — every downstream
consumer resamples from it on the fly (SV `AF.resample`, ASR
`load_audio(path, 16000)`, MOS `librosa.resample`, speech_tokenizer
`encode(wav, sr=48000)`), so we keep no resampled copies.

Every stage materializes its own artifact file — `enhanced/{name}.wav`,
`codes/{name}.npy`, `embedding/{name}.npy` — and each `AssetEntry` row
records the sha256 of all four artifacts (corpus wav included).
`precompute_task_table` turns disk + stored checksums into one `TaskRow` per
surviving clip (booleans = "the cached artifact still matches disk"). The
layers then run in dependency order:

- corpus: source of truth — a checksum mismatch invalidates every derived
  artifact outright, no salvage is possible.
- enhanced: regenerate from the corpus. Deterministic clearvoice turns a
  regenerated checksum equal to the stored one into proof that the old file
  was mere bit-rot (salvage — downstream stays valid); a different checksum
  cascades invalidation downward.
- codes: same salvage/cascade logic, re-extracted locally from a still-valid
  enhanced wav (no scorer round-trip needed).
- embedding: scorer round-trips (ASR/CER + SV vector + MOS) for everything
  still invalid, batched. The scorer (cuda:0) is the throughput bottleneck
  by ~8x (0.44s/clip vs 0.05s/clip for codes extraction) — client-side
  pipelining was measured at zero benefit and dropped (STATUS.md §16.8).

A table hole (missing/corrupt file) is therefore either filled in place or
propagated to everything that depends on it. `finalize` backfills per-row
`sim` against the freshly computed centroid (the centroid does not exist
before every clip is embedded), writes the centroid as `centroid.npy` beside
metrics.json (same np.save convention as codes/embedding), rewrites
asset.jsonl compacted, and ALWAYS rebuilds metrics.json wholesale — at ~2k
clips the centroid/sim pass is a millisecond matmul, incremental metrics
would be complexity with no payoff.

Scoring reuses `client/trainer.Client` (trainer-side bind of PUSH 5555 /
PULL 5556), so the resident scorer worker is oblivious to this caller. The
scorer is calibration-free — preprocessing requests
{vector, transcript, cer, mos} and the row `sim` is computed locally in
`finalize` from the raw unit-norm ERes2NetV2 vectors. Validation is
fail-loudly: a malformed manifest/asset line raises — no silent skips.
Manifest↔wav mismatches are not fatal: they are recorded in `DropReasons`.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from pydantic import BaseModel
from tqdm import tqdm

from qwen3_tts_post_training.client.protocol import ScoreField, ScoreItem
from qwen3_tts_post_training.client.trainer import Client

CLEARVOICE_SR = 48000

PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


@dataclass(frozen=True)
class Config:
    corpus_dir: Path
    cache_dir: Path  # per-corpus dir (.cache/{lang})
    min_seconds: float = 0.1
    min_tokens: int = 2
    limit: int = 0  # first N manifest entries only (debug)

    @property
    def manifest(self) -> Path:
        return self.corpus_dir.parent / f"{self.corpus_dir.stem}.jsonl"

    @property
    def enhanced_dir(self) -> Path:
        return self.cache_dir / "enhanced"

    @property
    def codes_dir(self) -> Path:
        return self.cache_dir / "codes"

    @property
    def embedding_dir(self) -> Path:
        return self.cache_dir / "embedding"

    @property
    def centroid_npy(self) -> Path:
        return self.cache_dir / "centroid.npy"

    @property
    def asset_jsonl(self) -> Path:
        return self.cache_dir / "asset.jsonl"

    @property
    def metrics_json(self) -> Path:
        return self.cache_dir / "metrics.json"


class CorpusEntry(BaseModel):
    name: str
    text: str


class Checksum(BaseModel):
    corpus: str
    enhanced: str
    codes: str
    embedding: str


class AssetEntry(CorpusEntry):
    transcript: str
    cer: float
    mos: float
    sim: float | None  # to centroid; backfilled by finalize
    checksum: Checksum

    def corpus_matches(self, config: Config) -> bool:
        path = config.corpus_dir / f"{self.name}.wav"
        return path.is_file() and sha256(path) == self.checksum.corpus

    def enhanced_matches(self, config: Config) -> bool:
        path = config.enhanced_dir / f"{self.name}.wav"
        return path.is_file() and sha256(path) == self.checksum.enhanced

    def codes_matches(self, config: Config) -> bool:
        path = config.codes_dir / f"{self.name}.npy"
        return path.is_file() and sha256(path) == self.checksum.codes

    def embedding_matches(self, config: Config) -> bool:
        path = config.embedding_dir / f"{self.name}.npy"
        return path.is_file() and sha256(path) == self.checksum.embedding


@dataclass(frozen=True)
class TaskRow:
    """Per-clip cache validity: True = the cached artifact (and the manifest
    text it was built from) still matches disk. Order = dependency order."""

    name: str
    corpus: bool
    enhanced: bool
    embedding: bool
    codes: bool

    @property
    def complete(self) -> bool:
        return self.corpus and self.enhanced and self.embedding and self.codes


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
    validated+filtered corpus, the cached asset rows keyed by clip name, and
    the drop-reason log. All layer computation is a pure delta on this."""

    config: Config
    corpus_entries: tuple[CorpusEntry, ...]
    asset_cache: dict[str, AssetEntry]
    drop_reasons: DropReasons

    @classmethod
    def load(cls, config: Config, token_counter: Callable) -> Cache:
        with open(config.manifest, encoding="utf-8") as file:
            corpus_entries = [CorpusEntry.model_validate_json(line) for line in file]

        wav_stems = [wav.stem for wav in config.corpus_dir.glob("*.wav")]

        wav_stems_set = set(wav_stems)
        manifest_names_set = {entry.name for entry in corpus_entries}
        orphan_corpus_names = tuple(
            stem for stem in wav_stems if stem not in manifest_names_set
        )
        orphan_manifest_names = tuple(
            entry.name for entry in corpus_entries if entry.name not in wav_stems_set
        )

        corpus_entries = [
            entry for entry in corpus_entries if entry.name in wav_stems_set
        ]

        if config.limit:
            corpus_entries = corpus_entries[: config.limit]

        desired_corpus_entries = []
        less_than_min_tokens = []
        less_than_min_seconds = []

        for entry in corpus_entries:
            wav_info = sf.info(str(config.corpus_dir / f"{entry.name}.wav"))
            if token_counter(entry.text) < config.min_tokens:
                less_than_min_tokens.append(entry.name)
            elif wav_info.frames / wav_info.samplerate < config.min_seconds:
                less_than_min_seconds.append(entry.name)
            else:
                desired_corpus_entries.append(entry)

        drop_reasons = DropReasons(
            orphan_corpus=orphan_corpus_names,
            orphan_manifest=orphan_manifest_names,
            less_than_min_tokens=tuple(less_than_min_tokens),
            less_than_min_seconds=tuple(less_than_min_seconds),
        )

        config.cache_dir.mkdir(parents=True, exist_ok=True)
        config.asset_jsonl.touch(exist_ok=True)
        with open(config.asset_jsonl, encoding="utf-8") as file:
            asset_cache = {
                entry.name: entry
                for entry in (AssetEntry.model_validate_json(line) for line in file)
            }

        return cls(
            config=config,
            corpus_entries=tuple(desired_corpus_entries),
            asset_cache=asset_cache,
            drop_reasons=drop_reasons,
        )

    def precompute_task_table(self) -> tuple[TaskRow, ...]:
        def task_row(entry: CorpusEntry) -> TaskRow:
            cached = self.asset_cache.get(entry.name)
            if cached is None or cached.text != entry.text:
                return TaskRow(entry.name, False, False, False, False)
            return TaskRow(
                name=entry.name,
                corpus=cached.corpus_matches(self.config),
                enhanced=cached.enhanced_matches(self.config),
                embedding=cached.embedding_matches(self.config),
                codes=cached.codes_matches(self.config),
            )

        return tuple(task_row(entry) for entry in self.corpus_entries)


def sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), sample_rate


def extract_codes(speech_tokenizer, wav_path: Path) -> np.ndarray:
    audio, sample_rate = load_mono(wav_path)
    codes = torch.as_tensor(
        speech_tokenizer.encode(audio, sr=sample_rate).audio_codes[0]
    )
    return codes.int().cpu().numpy()


def build_metrics(
    norms: np.ndarray,
    sims: np.ndarray,
    cer_values: np.ndarray,
    mos_values: np.ndarray,
    provenance: dict,
) -> dict:
    """Scalar calibration + provenance for metrics.json (the centroid itself
    lives beside it as centroid.npy). Sim stats feed RewardConfig
    (sv_center/sv_scale), wer reflects domain ASR performance, utmosv2 stats
    are recorded for a future gate decision (tau stays 2.5 for now)."""
    assert len(norms), "no complete rows to build metrics from"
    return {
        "sim": {"mean": float(sims.mean()), "std": float(sims.std())},
        "wer": float(cer_values.mean()),
        "utmosv2": {
            "mean": float(mos_values.mean()),
            "std": float(mos_values.std()),
            "percentiles": {
                str(percentile): float(np.percentile(mos_values, percentile))
                for percentile in PERCENTILES
            },
        },
        "n_clips": len(norms),
        **provenance,
    }


def apply_corpus_layer(table: tuple[TaskRow, ...]) -> tuple[TaskRow, ...]:
    """Corpus is the source of truth: a mismatch invalidates every derived
    artifact outright (nothing downstream can be salvaged)."""
    return tuple(
        replace(row, enhanced=False, embedding=False, codes=False)
        if not row.corpus
        else row
        for row in table
    )


def apply_enhanced_layer(
    cache: Cache, table: tuple[TaskRow, ...], device: str
) -> tuple[TaskRow, ...]:
    """Regenerate invalid enhanced wavs from the corpus. Deterministic
    clearvoice: a regenerated checksum equal to the stored one proves the
    old file was bit-rot (salvage — downstream untouched); a different
    checksum invalidates codes+embedding."""
    todo = [row for row in table if not row.enhanced]
    if not todo:
        return table

    from preprocess.clearvoice.decode import MossFormer2SE48KConfig, enhance
    from preprocess.clearvoice.load import (
        ensure_clearvoice,
        load_mossformer2_se_48k,
    )

    config = cache.config
    config.enhanced_dir.mkdir(parents=True, exist_ok=True)
    model = load_mossformer2_se_48k(ensure_clearvoice(), device)
    mossformer_config = MossFormer2SE48KConfig()

    updated: dict[str, TaskRow] = {}
    for row in tqdm(todo, desc="enhanced"):
        audio, sample_rate = load_mono(config.corpus_dir / f"{row.name}.wav")
        if sample_rate != CLEARVOICE_SR:
            audio = AF.resample(
                torch.from_numpy(audio), sample_rate, CLEARVOICE_SR
            ).numpy()
        torch.manual_seed(0)  # kaldi fbank dither=1.0 consumes the global RNG
        enhanced = enhance(model, mossformer_config, audio, device)
        enhanced_path = config.enhanced_dir / f"{row.name}.wav"
        sf.write(str(enhanced_path), enhanced, CLEARVOICE_SR, subtype="PCM_16")

        cached = cache.asset_cache.get(row.name)
        salvaged = (
            cached is not None and sha256(enhanced_path) == cached.checksum.enhanced
        )
        updated[row.name] = (
            replace(row, enhanced=True)
            if salvaged
            else replace(row, enhanced=True, codes=False, embedding=False)
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tuple(updated.get(row.name, row) for row in table)


def apply_codes_layer(
    cache: Cache, table: tuple[TaskRow, ...], speech_tokenizer
) -> tuple[TaskRow, ...]:
    """Repair codes whose enhanced wav is still valid by re-extracting them
    locally (no scorer round-trip). Same salvage/cascade logic as the
    enhanced layer; only rows with a valid embedding are eligible — the
    others are scored anyway and get their codes there."""
    todo = [row for row in table if not row.codes and row.embedding]
    if not todo:
        return table

    config = cache.config
    config.codes_dir.mkdir(parents=True, exist_ok=True)

    updated: dict[str, TaskRow] = {}
    for row in tqdm(todo, desc="codes"):
        codes = extract_codes(speech_tokenizer, config.enhanced_dir / f"{row.name}.wav")
        codes_path = config.codes_dir / f"{row.name}.npy"
        np.save(codes_path, codes)

        cached = cache.asset_cache.get(row.name)
        salvaged = cached is not None and sha256(codes_path) == cached.checksum.codes
        updated[row.name] = (
            replace(row, codes=True)
            if salvaged
            else replace(row, codes=False, embedding=False)
        )
    return tuple(updated.get(row.name, row) for row in table)


def score_missing(
    cache: Cache,
    table: tuple[TaskRow, ...],
    *,
    speech_tokenizer,
    client: Client,
    batch: int,
) -> list[AssetEntry]:
    """Scorer round-trips for every clip with an invalid embedding (fresh
    clips included), batched, fully serialized on purpose: the scorer
    (cuda:0) is the bottleneck by ~8x, and a depth-2 overlap was measured at
    ZERO benefit — the recv call absorbs the extraction window anyway
    (STATUS.md §16.8). Rows are appended to asset.jsonl live, so an
    interrupted run keeps what it finished."""
    config = cache.config
    todo = [row for row in table if not row.embedding]
    print(
        f"[score] {len(cache.corpus_entries) - len(todo)} cached clips, "
        f"{len(todo)} to process"
    )
    entries_by_name = {entry.name: entry for entry in cache.corpus_entries}
    config.codes_dir.mkdir(parents=True, exist_ok=True)
    config.embedding_dir.mkdir(parents=True, exist_ok=True)

    rows: list[AssetEntry] = []
    start = time.monotonic()
    for index in tqdm(range(0, len(todo), batch), desc="score"):
        chunk = todo[index : index + batch]
        for row in chunk:
            codes = extract_codes(
                speech_tokenizer, config.enhanced_dir / f"{row.name}.wav"
            )
            np.save(config.codes_dir / f"{row.name}.npy", codes)
        results = client.recv_score(
            client.send_score(
                [
                    ScoreItem(
                        wav_path=str(config.enhanced_dir / f"{row.name}.wav"),
                        text=entries_by_name[row.name].text,
                    )
                    for row in chunk
                ],
                fields={
                    ScoreField.VECTOR,
                    ScoreField.TRANSCRIPT,
                    ScoreField.CER,
                    ScoreField.MOS,
                },
            ),
            timeout=client.timeout_s,
        )
        for row, result in zip(chunk, results):
            embedding_path = config.embedding_dir / f"{row.name}.npy"
            np.save(embedding_path, np.asarray(result["vector"], dtype=np.float32))
            entry = entries_by_name[row.name]
            score_row = AssetEntry(
                name=row.name,
                text=entry.text,
                transcript=result["transcript"],
                cer=result["cer"],
                mos=result["mos"],
                sim=None,
                checksum=Checksum(
                    corpus=sha256(config.corpus_dir / f"{row.name}.wav"),
                    enhanced=sha256(config.enhanced_dir / f"{row.name}.wav"),
                    codes=sha256(config.codes_dir / f"{row.name}.npy"),
                    embedding=sha256(embedding_path),
                ),
            )
            with open(config.asset_jsonl, "a", encoding="utf-8") as file:
                file.write(score_row.model_dump_json() + "\n")
            rows.append(score_row)
    print(f"[score] done in {time.monotonic() - start:.1f}s")
    return rows


def finalize(
    cache: Cache,
    table: tuple[TaskRow, ...],
    incremental_rows: list[AssetEntry],
    *,
    model_path: str,
) -> None:
    """Compact asset.jsonl (valid cached rows + incremental rows), backfill
    per-row sim against the freshly computed centroid, write centroid.npy,
    rebuild metrics.json wholesale."""
    config = cache.config
    incremental_by_name = {row.name: row for row in incremental_rows}

    entries: list[AssetEntry] = []
    for entry in cache.corpus_entries:
        row = incremental_by_name.get(entry.name)
        if row is None and any(
            task_row.name == entry.name and task_row.complete for task_row in table
        ):
            row = cache.asset_cache[entry.name]
        assert row is not None, f"no valid row for clip {entry.name}"
        entries.append(row)

    vectors = np.stack(
        [np.load(config.embedding_dir / f"{entry.name}.npy") for entry in entries]
    ).astype(np.float64)  # centroid precision matches the playground reference
    norms = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    centroid = norms.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    sims = norms @ centroid
    np.save(config.centroid_npy, centroid)
    entries = [
        row.model_copy(update={"sim": float(sim)}) for row, sim in zip(entries, sims)
    ]

    config.cache_dir.mkdir(parents=True, exist_ok=True)
    with open(config.asset_jsonl, "w", encoding="utf-8") as file:
        for row in entries:
            print(row.model_dump_json(), file=file)

    metrics = build_metrics(
        norms,
        sims,
        np.asarray([row.cer for row in entries]),
        np.asarray([row.mos for row in entries]),
        provenance={
            "dataset": str(config.corpus_dir.resolve()),
            "model_path": model_path,
            "clearvoice": "MossFormer2_SE_48K",
            "sv_model": "eres2netv2",
            "min_tokens": config.min_tokens,
            "min_seconds": config.min_seconds,
            "dropped": cache.drop_reasons.to_dict(),
        },
    )
    with open(config.metrics_json, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    print(f"[metrics] written: {config.metrics_json}")


def sync(
    cache: Cache,
    *,
    speech_tokenizer,
    client: Client,
    device: str,
    batch: int,
    model_path: str,
) -> None:
    table = cache.precompute_task_table()
    table = apply_corpus_layer(table)
    table = apply_enhanced_layer(cache, table, device)
    table = apply_codes_layer(cache, table, speech_tokenizer)
    incremental_rows = score_missing(
        cache, table, speech_tokenizer=speech_tokenizer, client=client, batch=batch
    )
    finalize(cache, table, incremental_rows, model_path=model_path)


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
        corpus_dir=dataset,
        cache_dir=cache_root / dataset.name,
        min_tokens=min_tokens,
        min_seconds=min_seconds,
        limit=limit,
    )
    cache = Cache.load(config, token_counter=tokenize_text)
    sync(
        cache,
        speech_tokenizer=speech_tokenizer,
        client=client,
        device=device,
        batch=batch,
        model_path=model_path,
    )
    return config.cache_dir
