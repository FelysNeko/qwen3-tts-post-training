"""SFT training loop over the preprocess cache (`.cache/{namespace}/`).

Posture: SFT ALWAYS starts from a base ckpt (0.6B/1.7B — the finetuned
PhiLia093 ckpt is retired) and `export_custom_voice` turns the result into a
custom_voice speaker ckpt that the GRPO worker can continue from.

Data: `asset.jsonl` rows (name, text) + `codes/{name}.npy` ([T,16] int32) —
the codes are precomputed by the preprocess worker, so a training step is
purely `collate → teacher_forcing → CE` on the shared kernels
(`ModelWrapper.collate` / `ModelWrapper.teacher_forcing`).

Multi-speaker (`--namespaces`): K pools mix per batch — per-pool medoid vecs
stack to [b, hidden] into slot 6 (single-speaker = the K=1 broadcast case)
and `export_custom_voice` bakes one row per voice (slots 3000, 3001, ...),
so the GRPO worker can sample any exported speaker by name.

Loss (the single-shift pairing `teacher_forcing` was built for — the official
`sft_12hz.py` double-shifts both the labels (`inputs_embeds[:, :-1]` +
`labels[:, 1:]` through the internal HF CE) and the sub-talker hidden
selection; do NOT reproduce that here):

    loss = CE(talker_logits, codec_0_labels[:, 1:], ignore_index=-100)   # incl. EOS slot
         + 0.3 * CE(sub_talker_logits, talker_codec_ids[:, 1:])

Hyperparameters mirror the official script (AdamW wd=0.01, lr 2e-5, grad
accum → effective batch, clip 1.0) plus the linear LR warmup the GRPO loop
proved necessary (Adam's first step moves every param ±lr at once).

Checkpoint: rolling `latest.pt` (trainable state_dict + optimizer + batch
position) — full-FT ckpts are too large for per-step numbered files.

Export (`export_custom_voice`): white-list copy of the ckpt dir + config
surgery (`spk_id`/`spk_is_dialect` entry for the export speaker, reusing its
id when the name exists, allocating max+1 otherwise) + safetensors with
ONLY the `talker.*` keys (byte-layout match with the original
model.safetensors) and the reference-audio speaker embedding baked into
`codec_embedding.weight[id]`.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from qwen3_tts_post_training.cache import CacheLayout, load_multi_sft_dataset
from qwen3_tts_post_training.paths import repo_root
from qwen3_tts_post_training.system import gpu_allocated_mb, peak_rss_mb
from trainer.sft.model import (
    SftTrainerModel,
    extract_speaker_vec,
)

logger = logging.getLogger(__name__)


@dataclass
class SftConfig:
    # Pool selector(s): dirs are {cache_dir}/{namespace}, namespace
    # mirroring the corpus hierarchy. The NAMESPACE STRING is the speaker
    # name everywhere (export spk_id, GRPO sampling) — no separate
    # speaker-naming configuration. One namespace = single-speaker (the
    # K=1 case of the same path); K namespaces mix per batch.
    #
    # Speaker vecs are extracted from each pool's medoid
    # (`CacheLayout.speaker_ref`) and travel ONLY through slot 6
    # ([b, hidden] per batch); mixed batches mean identity cannot bake
    # into shared weights.
    namespaces: list[str]
    # Balanced head-slice per pool (train-side only; metrics/medoid stay
    # full-pool). None = use every clip of every pool.
    per_pool_cap: int | None = None

    # SFT starts ONLY from a base ckpt (asserted after load: base models
    # ship the in-model speaker encoder, custom_voice ckpts carry none) —
    # the finetuned PhiLia093 ckpt is retired; GRPO continues from
    # pipeline-produced custom_voice exports instead. Local dir or HF repo id.
    model_path: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    device: str = "cuda:1"

    # cache_dir = the cache ROOT location (default <repo>/.cache).
    cache_dir: str = str(repo_root() / ".cache")
    freeze: list[str] | None = None  # ⊆ {subtalker, talker, text}; None = 全训
    sub_weight: float = 0.3  # subtalker CE weight in loss = sem + w * sub

    batch_size: int = 2
    grad_accum: int = 4  # effective batch = batch_size * grad_accum (official)
    epochs: int = 3
    lr: float = 2e-5
    warmup_steps: int = 10  # linear LR ramp (same rationale as GRPO loop)
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    seed: int = 0
    out_dir: str = "runs/sft_v1"
    ckpt_every: int = 50  # optimizer steps between rolling-ckpt writes
    log_every: int = 10
    resume: bool = False


# ---------------------------------------------------------------------------
# ckpt (rolling latest.pt: trainable params + optimizer + batch position)
# ---------------------------------------------------------------------------


def _save_ckpt(cfg: SftConfig, model: SftTrainerModel, optimizer, pos: int) -> None:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = {
        "pos": pos,
        "params": {
            name: p.detach().cpu()
            for name, p in model.model.named_parameters()
            if p.requires_grad
        },
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, out / "latest.pt")


def _load_ckpt(cfg: SftConfig, model: SftTrainerModel, optimizer) -> int:
    """Restore the rolling ckpt. Returns the next batch position to run."""
    path = Path(cfg.out_dir) / "latest.pt"
    if not path.exists():
        return 0
    state = torch.load(path, map_location=cfg.device, weights_only=False)
    saved = state["params"]
    live = {name: p for name, p in model.model.named_parameters() if p.requires_grad}
    assert set(saved) == set(live), (
        "ckpt/trainable-set mismatch — the frozen-parameter policy changed "
        "since this ckpt was written"
    )
    for name, p in live.items():
        p.data.copy_(saved[name].to(cfg.device))
    optimizer.load_state_dict(state["optimizer"])
    logger.info(f"resumed at pos {state['pos']} from {path}")
    return state["pos"] + 1


# ---------------------------------------------------------------------------
# CustomVoice export
# ---------------------------------------------------------------------------

_EXPORT_COPY_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "configuration.json",
)


def export_custom_voice(
    cfg: SftConfig,
    model: SftTrainerModel,
    speakers: list[tuple[str, torch.Tensor]],
) -> Path:
    """Bake the reference-audio speaker(s) into a drop-in CustomVoice ckpt
    dir. `speakers` = [(spk_id_name, speaker_vec)] — one entry per voice;
    slots allocate max+1 (3000, 3001, ...), reusing a name's existing slot.

    The original model.safetensors holds ONLY `talker.*` tensors, so the
    export filters the state_dict the same way (speech_tokenizer and
    speaker_encoder live beside it, not inside it). `cfg.model_path` may be
    an HF repo id — resolve it to the local snapshot (cache-hit after
    training loaded the model from it)."""
    from huggingface_hub import snapshot_download

    src = Path(cfg.model_path)
    if not src.exists():
        src = Path(
            snapshot_download(
                cfg.model_path,
                allow_patterns=[*_EXPORT_COPY_FILES, "speech_tokenizer/*"],
            )
        )
    out = Path(cfg.out_dir) / "export"
    out.mkdir(parents=True, exist_ok=True)
    for name in _EXPORT_COPY_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
    if (src / "speech_tokenizer").is_dir():
        shutil.copytree(
            src / "speech_tokenizer", out / "speech_tokenizer", dirs_exist_ok=True
        )

    config_path = out / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # a from-base SFT produces a custom_voice speaker ckpt (official surgery)
    config["tts_model_type"] = "custom_voice"
    talker_config = config.get("talker_config", {})
    spk_id: dict = talker_config.get("spk_id", {})
    for name, _ in speakers:
        slot = spk_id[name] if name in spk_id else max(spk_id.values(), default=2999) + 1
        spk_id[name] = slot
    talker_config["spk_id"] = spk_id
    spk_dialect = talker_config.setdefault("spk_is_dialect", {})
    for name, _ in speakers:
        spk_dialect[name] = False
    config["talker_config"] = talker_config
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    state_dict = {
        k: v.detach().to("cpu")
        for k, v in model.model.state_dict().items()
        if k.startswith("talker.")
    }
    embedding = state_dict["talker.model.codec_embedding.weight"]
    baked = []
    for name, vec in speakers:
        slot = spk_id[name]
        embedding[slot] = vec.detach().to("cpu").to(embedding.dtype)
        baked.append(f"{name!r}@{slot}")
    save_file(state_dict, out / "model.safetensors", metadata={"format": "pt"})
    logger.info(f"exported speakers [{', '.join(baked)}] → {out}")
    return out


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------


def _assert_model_available(model_path: str) -> None:
    """Local ckpts must exist; HF repo ids (org/name, no filesystem hints)
    pass through to from_pretrained's own resolution."""
    path = Path(model_path)
    if path.is_absolute() or model_path.startswith(("./", "../")) or path.exists():
        assert path.exists(), (
            f"TTS ckpt not found at {model_path!r} — pass --model-path"
        )


def run_sft(cfg: SftConfig) -> None:
    _assert_model_available(cfg.model_path)
    assert cfg.namespaces, "sft requires --namespaces"
    layouts = [CacheLayout(Path(cfg.cache_dir) / ns) for ns in cfg.namespaces]
    speaker_names = list(cfg.namespaces)

    torch.manual_seed(cfg.seed)
    model = SftTrainerModel(cfg.model_path, device=cfg.device, freeze=cfg.freeze)
    assert model.model.speaker_encoder is not None, (
        f"{cfg.model_path} has no in-model speaker_encoder — SFT starts ONLY "
        "from a base ckpt (tts_model_type == 'base'); custom_voice ckpts "
        "carry none (continue GRPO from those, not SFT)"
    )
    model.model.train()

    speaker_vecs = torch.stack(
        [extract_speaker_vec(model, lay.speaker_ref()) for lay in layouts]
    )
    data = load_multi_sft_dataset(layouts, per_pool_cap=cfg.per_pool_cap)
    counts = Counter(tag for _, _, tag in data)
    logger.info(
        "pools: " + ", ".join(f"{n}={counts[i]}" for i, n in enumerate(speaker_names))
    )
    assert data, "empty dataset"
    batch_size = cfg.batch_size
    n_batches = (len(data) + batch_size - 1) // batch_size

    optimizer = torch.optim.AdamW(
        model.trainable_parameters, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    start_pos = _load_ckpt(cfg, model, optimizer) if cfg.resume else 0
    total_pos = cfg.epochs * n_batches

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    monitor_f = (out / "monitor.jsonl").open("a")

    logger.info(
        f"training {len(data)} clips: {n_batches} batches/epoch × {cfg.epochs} epochs"
        f" = {total_pos} steps | speaker vecs {tuple(speaker_vecs.shape)} |"
        f" {sum(p.numel() for p in model.trainable_parameters)} trainable params"
    )

    for epoch in range(cfg.epochs):
        order = list(range(len(data)))
        random.Random(cfg.seed * 1000003 + epoch).shuffle(order)
        for bi in range(n_batches):
            pos = epoch * n_batches + bi
            if pos < start_pos:
                continue
            t0 = time.monotonic()
            idx = order[bi * batch_size : (bi + 1) * batch_size]
            batch = model.collate([data[i][0] for i in idx], [data[i][1] for i in idx])
            tf = model.teacher_forcing(
                batch, speaker_vecs[[data[i][2] for i in idx]]
            )
            loss_sem = F.cross_entropy(
                tf.talker_logits.float().flatten(0, 1),
                batch.codec_0_labels[:, 1:].flatten(),
                ignore_index=-100,
            )
            loss_sub = F.cross_entropy(
                tf.sub_talker_logits.float().flatten(0, 1),
                tf.talker_codec_ids[:, 1:].flatten(),
            )
            loss = loss_sem + cfg.sub_weight * loss_sub
            if torch.isfinite(loss):
                (loss / cfg.grad_accum).backward()
            else:
                loss_sem = loss_sub = loss = None  # keep the step moving

            if (pos + 1) % cfg.grad_accum != 0:
                continue
            step = (pos + 1) // cfg.grad_accum
            lr_t = cfg.lr * min(1.0, step / max(1, cfg.warmup_steps))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_t
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.trainable_parameters, cfg.grad_clip
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % cfg.log_every == 0:
                loss_v = None if loss is None else round(loss.item(), 4)
                loss_sem_v = None if loss_sem is None else round(loss_sem.item(), 4)
                loss_sub_v = None if loss_sub is None else round(loss_sub.item(), 4)
                line = json.dumps(
                    {
                        "step": step,
                        "epoch": epoch,
                        "pos": pos,
                        "loss": loss_v,
                        "loss_sem": loss_sem_v,
                        "loss_sub": loss_sub_v,
                        "grad_norm": round(grad_norm.item(), 4),
                        "lr": f"{lr_t:.2e}",
                        "dur_s": round(time.monotonic() - t0, 2),
                        "rss_mb": peak_rss_mb(),
                        "gpu_alloc_mb": gpu_allocated_mb(cfg.device),
                    },
                    ensure_ascii=False,
                )
                logger.info(
                    f"step {step} epoch {epoch} loss {loss_v}"
                    f" (sem {loss_sem_v} sub {loss_sub_v})"
                    f" grad {round(grad_norm.item(), 4)} lr {lr_t:.2e}"
                    f" dur {round(time.monotonic() - t0, 2)}s"
                )
                monitor_f.write(line + "\n")
                monitor_f.flush()
            if step % cfg.ckpt_every == 0:
                _save_ckpt(cfg, model, optimizer, pos)

    _save_ckpt(cfg, model, optimizer, total_pos)
    monitor_f.close()
    export_custom_voice(
        cfg, model, list(zip(speaker_names, speaker_vecs, strict=True))
    )
