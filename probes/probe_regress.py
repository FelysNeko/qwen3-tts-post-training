"""Regression probe: batch.collate (vs official reference + legacy placement)
and the dense logprob algebra (sem shifted-select, sub placement, packing).
Synthetic tensors only — no model loads."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "workers/trainer/src"))

import torch
import torch.nn.functional as F
from trainer.model import ModelWrapper

torch.manual_seed(23)


def make_owner(config, real_ids):
    """ModelWrapper without __init__ (no ckpt load); a fake processor
    returns the canned [1, n] ids in call order (== zip order of texts)."""

    class _P:
        """Mimics the REAL processor: returns full assistant tokenization
        ([3 role + L + 5 specials]); collate is responsible for [:-5]."""

        def __init__(self, real_ids):
            self.calls = []
            self._ids = [
                torch.cat([ri.reshape(-1), torch.arange(900_000, 900_000 + 5)]).reshape(
                    1, -1
                )
                for ri in real_ids
            ]

        def __call__(self, text, return_tensors=None, padding=None):
            i = len(self.calls)
            self.calls.append(text)
            return {"input_ids": self._ids[i].clone()}

    owner = ModelWrapper.__new__(ModelWrapper)
    owner.model = SimpleNamespace(config=config, device=torch.device("cpu"))
    owner.device = "cpu"
    owner.processor = _P(real_ids)
    return owner


def collate(items, config):
    raise RuntimeError("unused: probe drives collate directly")


Q, V = 16, 97
tc = SimpleNamespace(
    codec_nothink_id=151001,
    codec_think_bos_id=151002,
    codec_think_eos_id=151003,
    codec_pad_id=151000,
    codec_bos_id=151998,
    codec_eos_token_id=151999,
    num_code_groups=Q,
)
cfg = SimpleNamespace(
    tts_pad_token_id=151643,
    tts_bos_token_id=151644,
    tts_eos_token_id=151645,
    talker_config=tc,
)

items = [
    (torch.randint(5, V // 2, (12,)), torch.randint(1, V, (5, Q))),
    (torch.randint(5, V // 2, (1, 8)), torch.randint(1, V, (2, Q))),  # [1,n] form
    (torch.randint(5, V // 2, (6,)), torch.randint(1, V, (9, Q))),
]
codes = [ac.clone() for _, ac in items]
text_lens = [ti.shape[-1] for ti, _ in items]
code_lens = [ac.shape[0] for _, ac in items]
pred_starts_old, spans = [], []
for ti, ac in items:
    l = ti.shape[-1] - 3
    pred_starts_old.append(8 + l + 1)  # legacy meta.pred_start (== codec bos pos)
    spans.append(10 + l)


# ---------- 1. official collate_fn reference (verbatim body) ----------
def official_collate(batch, config):
    t = max(b_["text_ids"].shape[1] + b_["audio_codes"].shape[0] for b_ in batch) + 8
    b = len(batch)
    input_ids = torch.zeros((b, t, 2), dtype=torch.long)
    codec_ids = torch.zeros((b, t, Q), dtype=torch.long)
    tem = torch.zeros((b, t), dtype=torch.bool)
    cem = torch.zeros((b, t), dtype=torch.bool)
    cmask = torch.zeros((b, t), dtype=torch.bool)
    am = torch.zeros((b, t), dtype=torch.long)
    labels = torch.full((b, t), -100, dtype=torch.long)
    c = config.talker_config
    for i, data in enumerate(batch):
        text_ids = data["text_ids"]
        audio_codec_0 = data["audio_codes"][:, 0]
        audio_codecs = data["audio_codes"]
        tl = text_ids.shape[1]
        q = audio_codec_0.shape[0]
        input_ids[i, :3, 0] = text_ids[0, :3]
        input_ids[i, 3:7, 0] = config.tts_pad_token_id
        input_ids[i, 7, 0] = config.tts_bos_token_id
        input_ids[i, 8 : 8 + tl - 3, 0] = text_ids[0, 3:]
        input_ids[i, 8 + tl - 3, 0] = config.tts_eos_token_id
        input_ids[i, 8 + tl - 2 : 8 + tl + q, 0] = config.tts_pad_token_id
        tem[i, : 8 + tl + q] = True
        input_ids[i, 3:8, 1] = torch.tensor(
            [
                c.codec_nothink_id,
                c.codec_think_bos_id,
                c.codec_think_eos_id,
                0,
                c.codec_pad_id,
            ]
        )
        input_ids[i, 8 : 8 + tl - 3, 1] = c.codec_pad_id
        input_ids[i, 8 + tl - 3, 1] = c.codec_pad_id
        input_ids[i, 8 + tl - 2, 1] = c.codec_bos_id
        input_ids[i, 8 + tl - 1 : 8 + tl - 1 + q, 1] = audio_codec_0
        input_ids[i, 8 + tl - 1 + q, 1] = c.codec_eos_token_id
        labels[i, 8 + tl - 1 : 8 + tl - 1 + q] = audio_codec_0
        labels[i, 8 + tl - 1 + q] = c.codec_eos_token_id
        codec_ids[i, 8 + tl - 1 : 8 + tl - 1 + q, :] = audio_codecs
        cem[i, 3 : 8 + tl + q] = True
        cem[i, 6] = False
        cmask[i, 8 + tl - 1 : 8 + tl - 1 + q] = True
        am[i, : 8 + tl + q] = True
    return {
        "input_ids": input_ids,
        "attention_mask": am,
        "text_embedding_mask": tem.unsqueeze(-1),
        "codec_embedding_mask": cem.unsqueeze(-1),
        "codec_0_labels": labels,
        "codec_ids": codec_ids,
        "codec_mask": cmask,
    }


# Drive the MERGED collate: fake processor returns deterministic ids per
# text (n = 12, 8, 6 — same as items' text ids lengths so geometry matches).
real_ids = [ti.reshape(-1) for ti, _ in items]
owner = make_owner(cfg, real_ids)
texts = [f"t{i}" for i in range(len(items))]
batch = owner.collate(texts, codes)
assert len(owner.processor.calls) == len(items)

off_items = [{"text_ids": ti.reshape(1, -1), "audio_codes": ac} for ti, ac in items]
ref = official_collate(off_items, cfg)
for k in ref:
    if not torch.equal(getattr(batch, k), ref[k]):
        d = getattr(batch, k) != ref[k]
        i0, j0 = d.nonzero()[0]
        print(
            f"DIFF {k} at [{i0},{j0}] ours={getattr(batch, k)[i0, max(0, j0 - 3) : j0 + 3].tolist()} ref={ref[k][i0, max(0, j0 - 3) : j0 + 3].tolist()}"
        )
        raise AssertionError(f"collate MISMATCH on {k}")
L = batch.input_ids.shape[1]
assert L == max(text_lens[i] + code_lens[i] for i in range(len(items))) + 8
print("1: collate byte-equal to official reference  PASS")

codec_ids, codec_mask = batch.codec_ids, batch.codec_mask
for i in range(len(items)):
    s = spans[i]
    assert torch.equal(codec_ids[i, s : s + code_lens[i]], codes[i])
    assert int(batch.input_ids[i, s - 1, 1]) == tc.codec_bos_id  # codec channel
    assert int(ref["codec_0_labels"][i, s + code_lens[i]]) == tc.codec_eos_token_id
print("   legacy-placement equivalence  PASS")

# ---------- 2/3. dense logprob algebra ----------
TEMP, STEMP = 0.9, 0.75


def token_log_probs(logits, tokens, temperature):
    logits = logits.float() / temperature
    return F.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


fake_logits = torch.randn(len(items), L, V) * 3
fake_hidden_grid = torch.randn(len(items), L, 64)
sub_grid_per_t = torch.randn(max(code_lens), Q - 1, V) * 2

predict_mask = codec_mask[:, 1:]
codes_flat = codec_ids[codec_mask]
sem_targets = torch.zeros_like(codec_ids[:, :, 0])
sem_targets[codec_mask] = codes_flat[:, 0]
sem_lp = token_log_probs(fake_logits[:, :-1], sem_targets[:, 1:], TEMP)

steps_flat = torch.cat([torch.arange(T) for T in code_lens])
sub_talker_logits = torch.stack([sub_grid_per_t[s] for s in steps_flat], dim=0)
sub_lp = token_log_probs(sub_talker_logits, codes_flat[:, 1:], STEMP)

b = len(items)
lengths = torch.tensor(code_lens)
max_j = predict_mask.shape[1]
n_idx, j_idx = predict_mask.nonzero(as_tuple=True)
sub_full = torch.zeros(b, max_j, Q - 1)
sub_full[n_idx, j_idx] = sub_lp
lp = torch.where(predict_mask, sem_lp, torch.zeros_like(sem_lp))
lp = torch.cat(
    [lp.unsqueeze(-1), sub_full.masked_fill(~predict_mask.unsqueeze(-1), 0.0)], dim=-1
).view(b, -1)
mask = predict_mask.unsqueeze(-1).expand(b, -1, Q).reshape(b, -1).float()

for i in range(len(items)):
    pp = pred_starts_old[i] + torch.arange(code_lens[i])
    new_rows = torch.nonzero(predict_mask[i]).squeeze(-1)
    assert torch.equal(new_rows, pp), f"shifted-select mismatch sample {i}"
    old_lps = token_log_probs(fake_logits[i, pp], codes[i][:, 0], TEMP)
    assert torch.equal(sem_lp[i][new_rows], old_lps), (
        f"codebook0 bit-equality failed {i}"
    )

    t_rows = j_idx[n_idx == i]
    blk = torch.cat(
        [
            token_log_probs(sub_grid_per_t[k], codec_ids[i, spans[i] + k, 1:], STEMP)[
                None
            ]
            for k in range(code_lens[i])
        ],
        dim=0,
    )
    assert torch.equal(sub_full[i][t_rows], blk), f"sub placement failed {i}"
    assert sub_full[i][~predict_mask[i]].abs().sum() == 0
    assert int(t_rows.min()) == spans[i] - 1

assert torch.isfinite(lp).all()
assert float(mask.sum()) == sum(code_lens) * Q
assert (lp[~mask.bool()] == 0).all()
unp = lp.view(b, max_j, Q)
assert torch.equal(unp[..., 0][predict_mask], sem_lp[predict_mask])
assert torch.equal(unp[..., 1:][predict_mask], sub_full[predict_mask])
counts = mask.view(b, max_j, Q).sum(dim=(1, 2)).long()
assert torch.equal(counts, lengths * Q)
print("2: sem shifted-select bit-equal vs legacy pred_start loop  PASS")
print("3: sub placement (no extra shift) + packing invariants  PASS")
print("ALL PASS")
