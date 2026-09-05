"""Corpus → `.cache/{namespace}` preprocessing: checksum-checked per-stage
caches. `namespace` is a corpus-mirroring relative path (`Speaker/Lang`,
e.g. `Cyrene/Chinese(PRC)`) and doubles as the SPEAKER name everywhere
downstream (SFT export spk_id, GRPO sampling) — there is no separate
speaker-naming configuration.

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
- codes: extracted locally wherever invalid, fresh clips included — no
  scorer round-trip is ever needed for codes. Salvage/cascade semantics as
  above.
- embedding: scorer pass 1 — request {VECTOR} for every clip with an
  invalid embedding artifact and write the npys. Pure artifact pass: a
  rewritten npy whose sha256 matches the cached row is bit-rot salvage
  (bit flips True — the row describes disk again); genuine drift or fresh
  clips keep bit False, which is how post detects a pool change.
- post_apply_embedding_layer: prune foreign embeddings, then materialize
  the centroid — reuse centroid.npy when nothing entered/left/changed the
  pool, recompute + persist otherwise. Every material layer also prunes
  artifacts of clips that left the filtered corpus scope, so expired
  content never lingers.
- text: scorer pass 2 — re-score {TRANSCRIPT, CER, MOS} ONLY for clips with
  no row yet or whose enhanced wav changed since `checksum.enhanced` was
  recorded. cer is deterministic and MOS is not even stable across scorer
  restarts (§16.9), so untouched clips keep their cached scores verbatim.
  Returns name → {transcript, cer, mos}; rows are live-appended with their
  sim against the materialized centroid.

Both scorer passes are batched and fully serialized on purpose: the scorer
(cuda:0) is the throughput bottleneck by ~8x, and client-side pipelining was
measured at zero benefit and dropped (STATUS.md §16.8). Rows are appended to
asset.jsonl live the moment a clip is complete, so an interrupted run keeps
what it finished (duplicates are fine — load keeps the last row per clip).

A table hole (missing/corrupt file) is therefore either filled in place or
propagated to everything that depends on it. `finalize` assembles the rows
new-overrides-old: transcript/cer/mos are write-time facts (this run's
re-scores win, everything else keeps its cached value), while the
disk-derived fields — checksums and sims against the materialized centroid
— are recomputed for the WHOLE pool on every run, so an embedding that
drifts (e.g. an SV model upgrade) refreshes everyone's sim without any
scorer round-trip. At ~2k clips that pass is a millisecond matmul —
incremental aggregates would be complexity with no payoff.

Scoring reuses `client/trainer.Client` (trainer-side bind of PUSH 5555 /
PULL 5556), so the resident scorer worker is oblivious to this caller. The
scorer is calibration-free — pass 1 requests {vector}, pass 2 requests
{transcript, cer, mos}, and the row `sim` is computed locally in `finalize`
from the raw unit-norm ERes2NetV2 vectors. Validation is fail-loudly: a
malformed manifest/asset line raises — no silent skips. Manifest↔wav
mismatches are not fatal: they are recorded in `DropReasons`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from pydantic import BaseModel
from tqdm import tqdm

from qwen3_tts_post_training.cache import CacheLayout
from qwen3_tts_post_training.client.protocol import ScoreItem, ScoreResult
from qwen3_tts_post_training.client.trainer import Client
from qwen3_tts_post_training.reward.text import cer, normalize

logger = logging.getLogger(__name__)

CLEARVOICE_SR = 48000

PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


@dataclass(frozen=True)
class Config:
    corpus_dir: Path
    cache_dir: Path  # pool dir ({cache-dir}/{namespace}, namespace-mirroring)
    min_seconds: float = 0.1
    min_tokens: int = 2
    # --random: shuffle the pool order (seeded 0 — same corpus → same order,
    # reproducible). The SFT worker's per-pool head slice then samples
    # uniformly instead of taking the chapter-ordered corpus head.
    random_order: bool = False

    @property
    def manifest(self) -> Path:
        return self.corpus_dir.parent / f"{self.corpus_dir.stem}.jsonl"

    @property
    def layout(self) -> CacheLayout:
        """The shared cache-layout view — path knowledge lives in
        qwen3_tts_post_training.cache, not here."""
        return CacheLayout(self.cache_dir)


class CorpusEntry(BaseModel):
    name: str
    text: str


class Checksum(BaseModel):
    corpus: str
    enhanced: str
    codes: str
    embedding: str

    @classmethod
    def from_disk(cls, config: Config, name: str) -> Checksum:
        """Authoritative artifact checksums, straight from disk — cached row
        values are used only for invalidation tests, never as data."""
        return cls(
            corpus=sha256(config.corpus_dir / f"{name}.wav"),
            enhanced=sha256(config.layout.enhanced_dir / f"{name}.wav"),
            codes=sha256(config.layout.codes_dir / f"{name}.npy"),
            embedding=sha256(config.layout.embedding_dir / f"{name}.npy"),
        )


class AssetEntry(CorpusEntry):
    transcript: str
    cer: float
    mos: float
    # sim to the current-pool centroid — derived from the embedding on disk
    # and recomputed for the whole pool at every finalize (identical values
    # when the pool is unchanged); transcript/cer/mos stay write-time facts
    sim: float
    checksum: Checksum

    def corpus_matches(self, config: Config) -> bool:
        path = config.corpus_dir / f"{self.name}.wav"
        return path.is_file() and sha256(path) == self.checksum.corpus

    def enhanced_matches(self, config: Config) -> bool:
        path = config.layout.enhanced_dir / f"{self.name}.wav"
        return path.is_file() and sha256(path) == self.checksum.enhanced

    def codes_matches(self, config: Config) -> bool:
        path = config.layout.codes_dir / f"{self.name}.npy"
        return path.is_file() and sha256(path) == self.checksum.codes

    def embedding_matches(self, config: Config) -> bool:
        path = config.layout.embedding_dir / f"{self.name}.npy"
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


class DropReasons(NamedTuple):
    """Why corpus clips are excluded from the pool. `_asdict()` yields the
    flat provenance dict for metrics.json — keys are the field names."""

    orphan_corpus: tuple[str, ...]
    orphan_manifest: tuple[str, ...]
    less_than_min_tokens: tuple[str, ...]
    less_than_min_seconds: tuple[str, ...]


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
        if config.random_order:
            # seeded (0): deterministic order per corpus, so salvage/reruns
            # and the SFT head slices stay reproducible
            random.Random(0).shuffle(corpus_entries)

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
        config.layout.asset_jsonl.touch(exist_ok=True)
        with open(config.layout.asset_jsonl, encoding="utf-8") as file:
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
    """Regenerate invalid enhanced wavs from the corpus. Exact bit semantics
    (bit = the row's checksum describes disk): a regenerated checksum equal
    to the stored one is bit-rot salvage — bit flips True, downstream
    untouched; a different checksum is genuine drift — the bit stays False
    and every derived artifact cascades to False (they were built from the
    old bytes)."""
    prune_foreign(cache, cache.config.layout.enhanced_dir, ".wav")
    todo = [row for row in table if not row.enhanced]
    if not todo:
        return table

    from preprocess.clearvoice.decode import MossFormer2SE48KConfig, enhance
    from preprocess.clearvoice.load import (
        ensure_clearvoice,
        load_mossformer2_se_48k,
    )

    config = cache.config
    config.layout.enhanced_dir.mkdir(parents=True, exist_ok=True)
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
        enhanced_path = config.layout.enhanced_dir / f"{row.name}.wav"
        sf.write(str(enhanced_path), enhanced, CLEARVOICE_SR, subtype="PCM_16")

        cached = cache.asset_cache.get(row.name)
        if cached is not None and (sha256(enhanced_path) == cached.checksum.enhanced):
            # bit-rot salvage: the row describes disk again — codes/embedding
            # were built from these same bytes and stay valid
            updated[row.name] = replace(row, enhanced=True)
        else:
            # genuine drift or a fresh clip: the row's enhanced checksum no
            # longer describes disk — everything derived from enhanced is
            # stale (the bit itself stays False and self-heals at finalize)
            updated[row.name] = replace(row, codes=False, embedding=False)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tuple(updated.get(row.name, row) for row in table)


def apply_codes_layer(
    cache: Cache, table: tuple[TaskRow, ...], speech_tokenizer
) -> tuple[TaskRow, ...]:
    """Codes artifacts: extract locally wherever invalid — no scorer
    round-trip ever needed for codes. Exact bit semantics (bit = the row's
    checksum describes disk): a regenerated checksum equal to the stored one
    is bit-rot salvage — bit flips True; genuine drift or a fresh clip keeps
    it False and it self-heals at finalize. Embedding is NOT downstream of
    codes (both derive from enhanced), so this layer never cascades — an
    enhanced drift already cascaded it in the enhanced layer."""
    prune_foreign(cache, cache.config.layout.codes_dir, ".npy")
    todo = [row for row in table if not row.codes]
    if not todo:
        return table

    config = cache.config
    config.layout.codes_dir.mkdir(parents=True, exist_ok=True)

    updated: dict[str, TaskRow] = {}
    for row in tqdm(todo, desc="codes"):
        codes = extract_codes(
            speech_tokenizer, config.layout.enhanced_dir / f"{row.name}.wav"
        )
        codes_path = config.layout.codes_dir / f"{row.name}.npy"
        np.save(codes_path, codes)

        cached = cache.asset_cache.get(row.name)
        if cached is not None and sha256(codes_path) == cached.checksum.codes:
            # bit-rot salvage: the row describes disk again — downstream
            # untouched (embedding derives from enhanced, not codes)
            updated[row.name] = replace(row, codes=True)

    return tuple(updated.get(row.name, row) for row in table)


def load_embeddings(cache: Cache) -> dict[str, np.ndarray]:
    """name → unit-norm float64 embedding for every corpus clip. The single
    full-pool load point per run: callers (post / collect_corpus_metrics /
    finalize) all consume this map — every corpus_entry's npy is on disk by the time
    this runs, because apply_embedding_layer materialized the invalid ones
    first."""
    vectors = np.stack(
        [
            np.load(cache.config.layout.embedding_dir / f"{entry.name}.npy")
            for entry in cache.corpus_entries
        ]
    ).astype(np.float64)
    norms = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return {entry.name: norm for entry, norm in zip(cache.corpus_entries, norms)}


def prune_foreign(cache: Cache, directory: Path, suffix: str) -> tuple[str, ...]:
    """Delete artifacts whose clip left the filtered corpus scope — expired
    content must not linger in the cache forever. Returns the removed clip
    names."""
    if not directory.is_dir():
        return ()
    keep = {entry.name for entry in cache.corpus_entries}
    foreign_names = tuple(
        path.stem for path in directory.glob(f"*{suffix}") if path.stem not in keep
    )
    for name in foreign_names:
        (directory / f"{name}{suffix}").unlink()
    if foreign_names:
        logger.info(
            f"prune {directory.name}: removed {len(foreign_names)} foreign artifacts"
        )
    return foreign_names


def apply_embedding_layer(
    cache: Cache,
    table: tuple[TaskRow, ...],
    client: Client,
    batch: int,
) -> tuple[TaskRow, ...]:
    """Scorer pass 1 — request {VECTOR} for every clip whose embedding
    artifact is invalid and write the npys. A rewritten npy whose sha256
    matches the cached row is bit-rot salvage — the bit flips True (the row
    describes disk again); genuine drift or a fresh clip keeps the bit
    False, which is exactly how `post_apply_embedding_layer` (consuming the
    returned table) detects a pool change. Batched, fully serialized on
    purpose: the scorer (cuda:0) is the bottleneck by ~8x, and client-side
    pipelining was measured at ZERO benefit (STATUS.md §16.8)."""
    config = cache.config
    todo = [row for row in table if not row.embedding]
    logger.info(
        f"{len(cache.corpus_entries) - len(todo)} cached clips, {len(todo)} to process"
    )
    if not todo:
        return table
    config.layout.embedding_dir.mkdir(parents=True, exist_ok=True)

    updated: dict[str, TaskRow] = {}
    start = time.monotonic()
    for index in tqdm(range(0, len(todo), batch), desc="embed"):
        chunk = todo[index : index + batch]
        results = client.score(
            [
                ScoreItem(wav_path=str(config.layout.enhanced_dir / f"{row.name}.wav"))
                for row in chunk
            ],
            sv=True,
        )
        for row, result in zip(chunk, results):
            embedding_path = config.layout.embedding_dir / f"{row.name}.npy"
            np.save(
                embedding_path,
                np.asarray(result.get_embedding_unwrap(), dtype=np.float32),
            )
            cached = cache.asset_cache.get(row.name)
            if cached is not None and (
                sha256(embedding_path) == cached.checksum.embedding
            ):
                # bit-rot salvage: the row describes disk again
                updated[row.name] = replace(row, embedding=True)
    logger.info(f"embedding done in {time.monotonic() - start:.1f}s")
    return tuple(updated.get(row.name, row) for row in table)


def post_apply_embedding_layer(
    cache: Cache,
    table: tuple[TaskRow, ...],
    name_to_norms: dict[str, np.ndarray],
) -> np.ndarray:
    """Materialize the centroid once the embedding pool has settled, reading
    the LIVE post-embedding-layer table: embedding bits are True only where
    the row's checksum describes disk (untouched or bit-rot salvage); bits
    left False by genuine drift or fresh clips — plus foreign embeddings
    pruned here — mean the pool changed, and the centroid is recomputed
    from the given norms and persisted. Unchanged pools reuse centroid.npy
    verbatim. The contract: after this function centroid.npy describes the
    embedding pool on disk. The return value feeds `collect_corpus_metrics`
    (row sims)
    and `finalize` (metrics sims)."""
    config = cache.config
    removed = prune_foreign(cache, config.layout.embedding_dir, ".npy")
    pool_changed = bool(removed) or any(not row.embedding for row in table)
    if not pool_changed and config.layout.centroid_npy.exists():
        return np.load(config.layout.centroid_npy)

    assert cache.corpus_entries, "no clips survived filtering — nothing to embed"
    # manifest order (or the seeded --random shuffle of it) keeps the
    # float64 summation deterministic
    norms = np.stack([name_to_norms[entry.name] for entry in cache.corpus_entries])
    centroid = norms.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    np.save(config.layout.centroid_npy, centroid)
    return centroid


def collect_corpus_metrics(
    cache: Cache,
    table: tuple[TaskRow, ...],
    name_to_norms: dict[str, np.ndarray],
    centroid: np.ndarray,
    client: Client,
    batch: int,
) -> dict[str, ScoreResult]:
    """Scorer pass 2 — re-score {TRANSCRIPT, CER, MOS} only for clips whose
    text scores are missing (fresh) or stale (the enhanced wav changed since
    their row was recorded). cer is deterministic and MOS is not even stable
    across scorer restarts, so untouched clips keep their cached scores
    verbatim — this pass is the ONLY thing that rewrites them.

    The centroid arrives materialized by `post_apply_embedding_layer`.
    Returns the per-clip `ScoreResult`s for `finalize` to merge over the
    cached rows; rows are live-appended with their sim (the clip's norm
    against the centroid), so an interrupted run keeps what it finished. The
    todo is a pure table read: under exact bit semantics, `enhanced=False`
    is precisely "fresh, or the enhanced wav drifted since the row recorded
    it" — text depends on nothing else."""
    config = cache.config
    entries_by_name = {entry.name: entry for entry in cache.corpus_entries}
    todo = [row for row in table if not row.enhanced]
    logger.info(f"{len(todo)} clips to score")
    if not todo:
        return {}

    text_results: dict[str, ScoreResult] = {}
    start = time.monotonic()
    with open(config.layout.asset_jsonl, "a", encoding="utf-8") as asset_file:
        for index in tqdm(range(0, len(todo), batch), desc="text"):
            chunk = todo[index : index + batch]
            results = client.score(
                [
                    ScoreItem(
                        wav_path=str(config.layout.enhanced_dir / f"{row.name}.wav")
                    )
                    for row in chunk
                ],
                asr=True,
                mos=True,
            )
            for row, result in zip(chunk, results):
                text_results[row.name] = result
                print(
                    AssetEntry(
                        name=row.name,
                        text=entries_by_name[row.name].text,
                        sim=float(name_to_norms[row.name] @ centroid),
                        checksum=Checksum.from_disk(config, row.name),
                        transcript=result.get_transcript_unwrap(),
                        cer=cer(
                            normalize(entries_by_name[row.name].text),
                            normalize(result.get_transcript_unwrap()),
                        ),
                        mos=result.get_mos_unwrap(),
                    ).model_dump_json(),
                    file=asset_file,
                )
            # a scored batch is durable — an interrupted run resumes after it
            asset_file.flush()
    logger.info(f"text scoring done in {time.monotonic() - start:.1f}s")
    return text_results


def finalize(
    cache: Cache,
    text_results: dict[str, ScoreResult],
    name_to_norms: dict[str, np.ndarray],
    centroid: np.ndarray,
    model_path: str,
) -> None:
    """Assemble asset.jsonl new-overrides-old: transcript/cer/mos are
    write-time facts (this run's re-scores win, everything else keeps its
    cached value), while the disk-derived fields — checksums via
    `Checksum.from_disk` and sims against the materialized centroid — are
    recomputed for the WHOLE pool on every run. Identical values when
    nothing changed; when embeddings drift (e.g. an SV model upgrade) every
    row's sim refreshes against the new centroid without any scorer
    round-trip — sim consumes the embedding. Then rebuild the authoritative
    pool aggregates (metrics.json) from the same norms: every metric shares
    the aligned {mean, std, percentiles} shape (sim stats feed RewardConfig
    sv_center/sv_scale, cer reflects domain ASR performance, mos stats are
    recorded for a future gate decision — tau stays 2.5)."""
    config = cache.config
    # manifest order (or the seeded --random shuffle of it) keeps the
    # float64 summation deterministic
    norms = np.stack([name_to_norms[entry.name] for entry in cache.corpus_entries])
    sims = norms @ centroid

    entries: list[AssetEntry] = []
    for entry, sim in zip(cache.corpus_entries, sims):
        cached = cache.asset_cache.get(entry.name)
        result = text_results.get(entry.name)
        assert cached is not None or result is not None, (
            f"no text scores for clip {entry.name}"
        )
        entries.append(
            AssetEntry(
                name=entry.name,
                text=entry.text,
                transcript=(
                    result.get_transcript_unwrap()
                    if result is not None
                    else cached.transcript
                ),
                cer=(
                    cer(
                        normalize(entry.text), normalize(result.get_transcript_unwrap())
                    )
                    if result is not None
                    else cached.cer
                ),
                mos=result.get_mos_unwrap() if result is not None else cached.mos,
                sim=float(sim),
                checksum=Checksum.from_disk(config, entry.name),
            )
        )

    config.cache_dir.mkdir(parents=True, exist_ok=True)
    with open(config.layout.asset_jsonl, "w", encoding="utf-8") as file:
        for row in entries:
            print(row.model_dump_json(), file=file)

    cer_values = np.asarray([row.cer for row in entries])
    mos_values = np.asarray([row.mos for row in entries])
    # speaker-conditioning reference: the pool's ERes2NetV2 medoid (max
    # mean-pairwise cosine — the clip closest to every other clip in the
    # space that hears channel/quality differences; STATUS §19.4). The SFT
    # worker selects in E2V2 but EMBEDS this clip with the model's own
    # speaker encoder — E2V2 vectors can't enter the talker slot.
    assert len(norms) >= 1, "empty pool"
    pairwise = norms @ norms.T
    mean_pairwise = (pairwise.sum(axis=1) - np.diagonal(pairwise)) / max(
        len(norms) - 1, 1
    )
    medoid_name = cache.corpus_entries[int(np.argmax(mean_pairwise))].name
    # aligned scalar shape: every metric is {mean, std, percentiles} over
    # the pool (sim stats feed RewardConfig sv_center/sv_scale; cer reflects
    # domain ASR performance; mos stats are recorded for a future gate
    # decision — tau stays 2.5)
    metrics = {
        "sim": {
            "mean": float(sims.mean()),
            "std": float(sims.std()),
            "percentiles": {
                str(percentile): float(np.percentile(sims, percentile))
                for percentile in PERCENTILES
            },
        },
        "cer": {
            "mean": float(cer_values.mean()),
            "std": float(cer_values.std()),
            "percentiles": {
                str(percentile): float(np.percentile(cer_values, percentile))
                for percentile in PERCENTILES
            },
        },
        "mos": {
            "mean": float(mos_values.mean()),
            "std": float(mos_values.std()),
            "percentiles": {
                str(percentile): float(np.percentile(mos_values, percentile))
                for percentile in PERCENTILES
            },
        },
        "n_clips": len(norms),
        "medoid": medoid_name,
        "dataset": str(config.corpus_dir.resolve()),
        "model_path": model_path,
        "min_tokens": config.min_tokens,
        "min_seconds": config.min_seconds,
        "dropped": cache.drop_reasons._asdict(),
    }
    with open(config.layout.metrics_json, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    logger.info(f"metrics written: {config.layout.metrics_json}")


def sync(
    cache: Cache,
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
    table = apply_embedding_layer(cache, table, client, batch)
    name_to_norms = load_embeddings(cache)
    centroid = post_apply_embedding_layer(cache, table, name_to_norms)
    text_results = collect_corpus_metrics(
        cache, table, name_to_norms, centroid, client, batch
    )
    finalize(cache, text_results, name_to_norms, centroid, model_path)


def run_pipeline(
    dataset: Path,
    namespace: Path,
    cache_root: Path,
    tokenize_text,
    speech_tokenizer,
    client: Client,
    device: str,
    model_path: str,
    min_tokens: int,
    min_seconds: float,
    batch: int,
    random_order: bool = False,
) -> Path:
    config = Config(
        corpus_dir=dataset,
        cache_dir=cache_root / namespace,
        min_tokens=min_tokens,
        min_seconds=min_seconds,
        random_order=random_order,
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
