"""Probe: preprocess pipeline (STATUS.md §16) — offline sections.

1. Protocol: `ScoreRequest.fields` is REQUIRED (missing raises); field
    subsets round-trip; unrequested ScoreResult fields are None while
    `get_*_unwrap` assert-crashes on them; nested ScoreResponse round-trip.
2. Reward injection: sv_center/sv_scale have NO defaults — bare RewardConfig()
    raises; a metrics.json holding the OLD playground pair (0.8585/0.0966)
    bit-exactly reproduces the explicit same pair through reward_v3; a
    shifted metrics injects its values.
3. Offline stages on a synthetic corpus (needs soundfile): Cache corpus
    one-to-one validation + filter, artifact-dir layout, task table bools
    (fresh = all-False, intact = all-True), corpus-layer cascade, corruption
    flips its own bit, prune (foreign artifacts + dropped clips),
    post_apply_embedding_layer (recompute on pool change, reuse otherwise),
    finalize wholesale assembly (sims recomputed + metrics scalars against
    hand references + flat dropped provenance + medoid = mean-pairwise
    argmax hand reference + CacheLayout.speaker_ref resolution and
    missing-key assert; embedding drift refreshes every sim AND the medoid
    while text facts stay cached), garbage rows raise.

Run in the preprocess venv (all sections):
    workers/preprocess/.venv/bin/python probes/probe_preprocess.py
Root venv runs sections 1-3 only (no soundfile).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "workers/preprocess/src"))

import numpy as np
import torch
from pydantic import ValidationError

from qwen3_tts_post_training.paths import repo_root

assert repo_root() == REPO

from qwen3_tts_post_training.cache import CacheLayout
from qwen3_tts_post_training.client.protocol import (
    ScoreField,
    ScoreItem,
    ScoreRequest,
    ScoreResponse,
    ScoreResult,
    Timing,
)
from qwen3_tts_post_training.reward.reward import RewardConfig, reward_v3

PASS = "PASS"


def check(name: str, cond: bool) -> None:
    print(f"{'✓' if cond else '✗'} {name}")
    assert cond, name


# ---------------------------------------------------------------- section 1
def section_protocol() -> None:
    full = ScoreResult(
        wav_path="/x.wav",
        embedding=[0.1, 0.2],
        transcript="你好",
        cer=0.0,
        mos=2.8,
    )
    full_rt = ScoreResult.model_validate(full.model_dump(mode="json"))
    check(
        "full round-trip",
        full_rt.get_embedding_unwrap() == [0.1, 0.2] and full_rt.mos == 2.8,
    )

    partial = ScoreResult(wav_path="/x.wav", cer=0.5)
    check(
        "unrequested fields None",
        partial.embedding is None
        and partial.transcript is None
        and partial.mos is None,
    )
    check("unwrap returns requested value", partial.get_cer_unwrap() == 0.5)
    try:
        partial.get_embedding_unwrap()
        check("unwrap on unrequested raises", False)
    except AssertionError:
        check("unwrap on unrequested raises", True)

    # fields is REQUIRED — no implicit score-everything default
    try:
        ScoreRequest(id=1, items=[ScoreItem(wav_path="/x.wav", text="t")])
        check("missing fields raises", False)
    except ValidationError:
        check("missing fields raises", True)

    subset = {ScoreField.EMBEDDING, ScoreField.CER}
    req2 = ScoreRequest(
        id=1, items=[ScoreItem(wav_path="/x.wav", text="t")], fields=subset
    )
    req2_rt = ScoreRequest.model_validate(req2.model_dump())
    check("fields frozenset round-trip", req2_rt.fields == subset)

    # the wire path is zmq send_json (plain json.dumps) — python-mode dumps
    # keep the frozenset and crash there; the client must use mode="json"
    json.dumps(req2.model_dump(mode="json"))
    check("request wire-serializable (mode=json)", True)

    # nested response round-trip through the scorer -> client hop
    resp = ScoreResponse(
        id=1,
        results=[full, partial],
        timing=Timing(sv=0.0, asr=0.0, mos=0.0),
        rss_mb=0,
    )
    resp_rt = ScoreResponse.model_validate(resp.model_dump(mode="json"))
    check(
        "nested response round-trip",
        resp_rt.results[0].get_embedding_unwrap() == [0.1, 0.2]
        and resp_rt.results[1].cer == 0.5,
    )

    check(
        "ScoreField values are result keys",
        {field.value for field in ScoreField}
        == set(ScoreResult.model_fields) - {"wav_path"},
    )


# ---------------------------------------------------------------- section 2
def section_reward_injection(tmp: Path) -> None:
    old = {"sim": {"mean": 0.8585, "std": 0.0966}}
    pool = CacheLayout(tmp / "pool")
    pool.cache_namespace_dir.mkdir(parents=True)
    pool.metrics_json.write_text(json.dumps(old))
    cfg = pool.reward_config()

    try:
        RewardConfig()
        check("RewardConfig() without calibration raises", False)
    except TypeError:
        check("RewardConfig() without calibration raises", True)

    check(
        "old playground pair == explicit construction",
        cfg.sv_center == 0.8585 and cfg.sv_scale == 0.0966,
    )

    np.save(pool.centroid_npy, np.asarray([1.0, 0.0]))
    check(
        "centroid loader value (sibling npy)",
        np.array_equal(pool.load_centroid(), np.asarray([1.0, 0.0])),
    )

    sim = torch.tensor([0.83, 0.88, 0.86, 0.90])
    cer = torch.tensor([0.10, 0.02, 0.05, 0.00])
    mos = torch.tensor([2.4, 3.1, 2.8, 3.0])
    R1, _bd1 = reward_v3(sim, cer, mos, RewardConfig(sv_center=0.8585, sv_scale=0.0966))
    R2, _bd2 = reward_v3(sim, cer, mos, cfg)
    check("reward_v3 bit-equal (file calib vs explicit pair)", torch.equal(R1, R2))

    shifted = CacheLayout(tmp / "pool_shifted")
    shifted.cache_namespace_dir.mkdir(parents=True)
    shifted.metrics_json.write_text(json.dumps({"sim": {"mean": 0.80, "std": 0.05}}))
    cfg3 = shifted.reward_config()
    check(
        "non-default injection",
        cfg3.sv_center == 0.80 and cfg3.sv_scale == 0.05,
    )
    R3, _ = reward_v3(sim, cer, mos, cfg3)
    check("injected config changes reward", not torch.equal(R1, R3))


# ---------------------------------------------------------------- section 3
def section_offline_stages(tmp: Path) -> None:
    try:
        import soundfile as sf
    except ImportError:
        print("SKIP offline stages (no soundfile in this venv)")
        return

    from preprocess.pipeline import (
        PERCENTILES,
        AssetEntry,
        Cache,
        Checksum,
        Config,
        TaskRow,
        apply_corpus_layer,
        finalize,
        post_apply_embedding_layer,
        prune_foreign,
        sha256,
    )

    corpus = tmp / "TestLang"
    corpus.mkdir()
    sr = 48000
    names = []
    for i in range(4):
        name = f"c{i}"
        names.append(name)
        t = np.arange(sr // 2, dtype=np.float32) / sr
        sf.write(str(corpus / f"{name}.wav"), 0.1 * np.sin(2 * np.pi * 220 * t), sr)
    jsonl = corpus.parent / "TestLang.jsonl"
    jsonl.write_text(
        "\n".join(
            json.dumps({"name": n, "text": f"句子{i}"}) for i, n in enumerate(names)
        )
    )

    def load_cache(**overrides) -> Cache:
        kwargs = {
            "corpus_dir": corpus,
            "cache_dir": tmp / "cache" / "TestLang",
            "min_tokens": 2,
            "min_seconds": 0.1,
        }
        kwargs.update(overrides)
        return Cache.load(Config(**kwargs), token_counter=lambda _: 3)

    snap = load_cache()
    check("corpus_entries one-to-one + filter", len(snap.corpus_entries) == 4)

    # orphan + absent recorded in DropReasons (dropped, not fatal)
    sf.write(str(corpus / "orphan.wav"), np.zeros(480, dtype=np.float32), sr)
    snap = load_cache()
    check(
        "orphan wav recorded + excluded",
        snap.drop_reasons.orphan_corpus == ("orphan",)
        and len(snap.corpus_entries) == 4,
    )
    (corpus / "orphan.wav").unlink()
    (corpus / names[0]).with_suffix(".wav").unlink()
    snap = load_cache()
    check(
        "manifest-absent recorded + excluded",
        snap.drop_reasons.orphan_manifest == (names[0],)
        and len(snap.corpus_entries) == 3,
    )
    t = np.arange(sr // 2, dtype=np.float32) / sr
    sf.write(str(corpus / f"{names[0]}.wav"), 0.1 * np.sin(2 * np.pi * 220 * t), sr)

    tight = load_cache(min_tokens=5)
    check("filter drops by tokens", len(tight.corpus_entries) == 0)

    config = Config(
        corpus_dir=corpus, cache_dir=tmp / "cache" / "TestLang", min_tokens=2
    )
    cache = load_cache()
    check(
        "layout: artifact dirs under cache_dir",
        config.layout.enhanced_dir == config.cache_dir / "enhanced"
        and config.layout.codes_dir == config.cache_dir / "codes"
        and config.layout.embedding_dir == config.cache_dir / "embedding",
    )

    # task table: fresh clips (no cached rows) are all-False
    table = cache.precompute_task_table()
    check(
        "task table all-False on empty cache",
        all(
            row == TaskRow(name, False, False, False, False)
            for row, name in zip(table, names)
        ),
    )

    # corpus layer cascades: corpus invalid -> everything downstream invalid
    broken = tuple(
        TaskRow(name, corpus=False, enhanced=True, embedding=True, codes=True)
        for name in names
    )
    cascaded = apply_corpus_layer(broken)
    check(
        "corpus layer cascades downstream",
        all(
            row == TaskRow(name, False, False, False, False)
            for row, name in zip(cascaded, names)
        ),
    )

    # finalize: kept-row checksum refresh + wholesale metrics rebuild over
    # synthetic rows/artifacts (rows carry their sim as a write-time
    # snapshot — finalize never rewrites sims)
    rng = np.random.default_rng(2)
    vectors = rng.normal(size=(4, 8)).astype(np.float32)
    hand_vectors = vectors.astype(np.float64)
    hand_norms = hand_vectors / np.linalg.norm(hand_vectors, axis=1, keepdims=True)
    hand_centroid = hand_norms.mean(axis=0)
    hand_centroid /= np.linalg.norm(hand_centroid)
    hand_sims = hand_norms @ hand_centroid

    def write_artifacts(name: str, vector: np.ndarray, sim: float) -> AssetEntry:
        config.layout.enhanced_dir.mkdir(parents=True, exist_ok=True)
        config.layout.codes_dir.mkdir(parents=True, exist_ok=True)
        config.layout.embedding_dir.mkdir(parents=True, exist_ok=True)
        (config.layout.enhanced_dir / f"{name}.wav").write_bytes(
            (corpus / f"{name}.wav").read_bytes()
        )
        np.save(config.layout.codes_dir / f"{name}.npy", np.zeros((3, 16), dtype=np.int32))
        np.save(config.layout.embedding_dir / f"{name}.npy", vector)
        return AssetEntry(
            name=name,
            text=f"句子{int(name[1:])}",
            transcript="x",
            cer=0.0,
            mos=2.5,
            sim=sim,
            checksum=Checksum(
                corpus=sha256(corpus / f"{name}.wav"),
                enhanced=sha256(config.layout.enhanced_dir / f"{name}.wav"),
                codes=sha256(config.layout.codes_dir / f"{name}.npy"),
                embedding=sha256(config.layout.embedding_dir / f"{name}.npy"),
            ),
        )

    cache.asset_cache.update(
        {
            name: write_artifacts(name, vector, float(sim))
            for name, vector, sim in zip(names, vectors, hand_sims)
        }
    )
    table = apply_corpus_layer(cache.precompute_task_table())
    check(
        "task table all-True on intact artifacts",
        all(row.complete for row in table),
    )

    # corrupt one enhanced wav -> enhanced bit False, corpus still True
    corrupted_path = config.layout.enhanced_dir / f"{names[1]}.wav"
    corrupted_path.write_bytes(corrupted_path.read_bytes() + b"\x00")
    table = cache.precompute_task_table()
    check(
        "corruption flips only its own bit",
        all(row.complete for row in table if row.name != names[1])
        and not table[1].enhanced,
    )
    corrupted_path.write_bytes((corpus / f"{names[1]}.wav").read_bytes())

    table = apply_corpus_layer(cache.precompute_task_table())

    # prune: artifacts of clips outside the filtered scope are removed
    (config.layout.embedding_dir / "ghost.npy").write_bytes(b"\x00")
    removed = prune_foreign(cache, config.layout.embedding_dir, ".npy")
    check(
        "prune removes foreign embeddings",
        removed == ("ghost",) and not (config.layout.embedding_dir / "ghost.npy").exists(),
    )

    # centroid materialization: recompute + persist on pool change, reuse
    # the stored npy otherwise (post reads the LIVE post-layer table —
    # embedding bits False = drifted or fresh)
    name_to_norms = dict(zip(names, hand_norms))
    config.layout.centroid_npy.unlink(missing_ok=True)
    changed_table = tuple(
        TaskRow(row.name, row.corpus, row.enhanced, False, row.codes) for row in table
    )
    centroid = post_apply_embedding_layer(cache, changed_table, name_to_norms)
    check(
        "post recomputes + persists centroid (hand reference)",
        np.allclose(centroid, hand_centroid, atol=1e-12)
        and config.layout.centroid_npy.exists(),
    )
    reused = post_apply_embedding_layer(cache, table, name_to_norms)
    check(
        "post reuses centroid.npy when pool unchanged",
        np.array_equal(reused, centroid),
    )

    finalize(cache, {}, name_to_norms, centroid, "probe")
    metrics = json.loads(config.layout.metrics_json.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in config.layout.asset_jsonl.read_text().splitlines()]
    check(
        "finalize recomputes sims (identical when pool unchanged)",
        all(row["sim"] == float(sim) for row, sim in zip(rows, hand_sims)),
    )
    check(
        "post's centroid.npy (hand reference, unit norm)",
        np.allclose(np.load(config.layout.centroid_npy), hand_centroid, atol=1e-12)
        and abs(np.linalg.norm(np.load(config.layout.centroid_npy)) - 1.0) < 1e-12,
    )
    check(
        "finalize rebuilds metrics wholesale (no centroid key)",
        "centroid" not in metrics and metrics["n_clips"] == 4,
    )
    check(
        "finalize metrics scalars (hand reference)",
        abs(metrics["sim"]["mean"] - float(hand_sims.mean())) < 1e-12
        and abs(metrics["sim"]["std"] - float(hand_sims.std())) < 1e-12
        and abs(
            metrics["sim"]["percentiles"]["50"] - float(np.percentile(hand_sims, 50))
        )
        < 1e-12
        and metrics["cer"]
        == {
            "mean": 0.0,
            "std": 0.0,
            "percentiles": {str(p): 0.0 for p in PERCENTILES},
        }
        and metrics["mos"]["mean"] == 2.5
        and metrics["mos"]["percentiles"]["50"] == 2.5,
    )
    check(
        "dropped provenance is flat DropReasons._asdict()",
        list(metrics["dropped"])
        == [
            "orphan_corpus",
            "orphan_manifest",
            "less_than_min_tokens",
            "less_than_min_seconds",
        ]
        and all(isinstance(v, list) for v in metrics["dropped"].values()),
    )

    # medoid: mean-pairwise argmax over the pool norms (hand reference) —
    # the SFT conditioning reference (STATUS §19.4)
    hand_pair = hand_norms @ hand_norms.T
    hand_mp = (hand_pair.sum(axis=1) - np.diagonal(hand_pair)) / (len(names) - 1)
    check(
        "finalize medoid (mean-pairwise argmax, hand reference)",
        metrics["medoid"] == names[int(np.argmax(hand_mp))],
    )
    ref = config.layout.speaker_ref()
    check(
        "layout.speaker_ref resolves the enhanced wav",
        ref == config.layout.enhanced_dir / f"{metrics['medoid']}.wav" and ref.exists(),
    )
    stripped_pool = CacheLayout(tmp / "pool_stripped")
    stripped_pool.cache_namespace_dir.mkdir(parents=True)
    stripped = json.loads(config.layout.metrics_json.read_text(encoding="utf-8"))
    del stripped["medoid"]
    stripped_pool.metrics_json.write_text(json.dumps(stripped), encoding="utf-8")
    try:
        stripped_pool.speaker_ref()
        check("speaker_ref on missing medoid raises", False)
    except AssertionError:
        check("speaker_ref on missing medoid raises", True)

    # embedding drift (e.g. an SV model upgrade): one npy changes direction
    # → the centroid moves and EVERY row's sim refreshes against it, while
    # transcript/cer/mos keep their cached values (scorer never sees this)
    drifted_vectors = vectors.copy()
    drifted_vectors[0] = drifted_vectors[0][::-1]  # new direction, same norm
    np.save(config.layout.embedding_dir / f"{names[0]}.npy", drifted_vectors[0])
    drifted_norm = drifted_vectors[0].astype(np.float64)
    drifted_norm /= np.linalg.norm(drifted_norm)
    drift_name_to_norms = {**name_to_norms, names[0]: drifted_norm}
    drift_table = tuple(
        TaskRow(row.name, row.corpus, row.enhanced, row.name == names[0], row.codes)
        for row in table
    )
    drift_centroid = post_apply_embedding_layer(cache, drift_table, drift_name_to_norms)
    finalize(cache, {}, drift_name_to_norms, drift_centroid, "probe")
    drift_norms = drifted_vectors.astype(np.float64) / np.linalg.norm(
        drifted_vectors.astype(np.float64), axis=1, keepdims=True
    )
    drift_sims = drift_norms @ drift_centroid
    rows = [json.loads(line) for line in config.layout.asset_jsonl.read_text().splitlines()]
    check(
        "embedding drift refreshes every sim (text facts untouched)",
        np.allclose([row["sim"] for row in rows], drift_sims, atol=1e-12)
        and all(row["transcript"] == "x" and row["cer"] == 0.0 for row in rows),
    )
    drift_metrics = json.loads(config.layout.metrics_json.read_text(encoding="utf-8"))
    drift_pair = drift_norms @ drift_norms.T
    drift_mp = (drift_pair.sum(axis=1) - np.diagonal(drift_pair)) / (len(names) - 1)
    check(
        "finalize medoid refreshes on embedding drift",
        drift_metrics["medoid"] == names[int(np.argmax(drift_mp))],
    )

    # a clip dropped from the corpus leaves the scope, and its artifacts
    # are pruned from every material layer's directory
    (corpus / f"{names[3]}.wav").unlink()
    dropped = load_cache()
    check(
        "dropped clip leaves the corpus scope",
        len(dropped.corpus_entries) == 3
        and names[3] not in {entry.name for entry in dropped.corpus_entries},
    )
    check(
        "dropped clip pruned from all artifact dirs",
        prune_foreign(dropped, config.layout.enhanced_dir, ".wav") == (names[3],)
        and prune_foreign(dropped, config.layout.codes_dir, ".npy") == (names[3],)
        and prune_foreign(dropped, config.layout.embedding_dir, ".npy") == (names[3],)
        and not (config.layout.enhanced_dir / f"{names[3]}.wav").exists(),
    )

    # asset validation: garbage raises (fail-loudly)
    with open(config.layout.asset_jsonl, "a", encoding="utf-8") as file:
        file.write("not json\n")
    try:
        load_cache()
        check("asset garbage raises", False)
    except ValidationError:
        check("asset garbage raises", True)


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        section_protocol()
        section_reward_injection(tmp)
        section_offline_stages(tmp)
    print(PASS)


if __name__ == "__main__":
    main()
