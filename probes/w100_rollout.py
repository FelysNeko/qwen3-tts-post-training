"""w100 四臂评测 rollout(双 GPU 双进程,每进程 2 臂)。

用法:
    python probes/w100_rollout.py <device> <arm1,arm2> [--cats c1,c2] [--tag T]

产物:runs/hp17b_w100_{arm}{tag}_eval/{voice}/{cat}_{i:02d}_{k}.wav
协议:graphed sampler(lmax=1024)、batch_size=4(4 take 一个 batch 共 seed 流)、
    T0.9/top-k50 + subtalker 0.9/50、language="Auto"、seed=1234+全局条目序号;
    token 预算:long=1024,其余=384;hit_cap: dur >= (budget-cur_len)*0.08-1.0。
断点续:组内任一 take 缺失则整组重生成(seed 定种,重生成幂等)。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
DEVICE = sys.argv[1]
ARMS = sys.argv[2].split(",")
CATS_FILTER = (
    set(sys.argv[sys.argv.index("--cats") + 1].split(","))
    if "--cats" in sys.argv
    else None
)
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""

with open(ROOT / "probes/tmp/general.json") as f:
    CATS = json.load(f)
BUDGET = {c: (1024 if c == "long" else 384) for c in CATS}
VOICES = ["cyrene", "castorice", "aglaea", "hyacine", "cipher", "hysilens", "cerydra"]
PROMPTS = [
    (c, i, t)
    for c, items in CATS.items()
    if CATS_FILTER is None or c in CATS_FILTER
    for i, t in enumerate(items)
]

from trainer.grpo.samplers.base import Sampler
from trainer.model import ModelWrapper


def export_dir(arm: str) -> Path:
    """arm 规格:'name'(→ runs/hp17b_w100_{name}/export)或 'name=dir'(显式路径)。"""
    if "=" in arm:
        _name, d = arm.split("=", 1)
        return ROOT / d
    return ROOT / f"runs/hp17b_w100_{arm}/export"

def eval_root(arm: str) -> Path:
    name = arm.split("=", 1)[0]
    if "=" in arm:
        return ROOT / f"runs/{name}_eval"
    return ROOT / f"runs/hp17b_w100_{name}_eval"

for arm in ARMS:
    out_root = eval_root(arm)
    mw = ModelWrapper(str(export_dir(arm)), device=DEVICE)
    for voice in VOICES:
        vdir = out_root / voice
        vdir.mkdir(parents=True, exist_ok=True)
        sam = Sampler.build(
            mw,
            impl="graphed",
            batch_size=4,
            lmax=1024,
        )
        for gi, (cat, pi, text) in enumerate(PROMPTS):
            paths = [vdir / f"{cat}_{pi:02d}_{k}.wav" for k in range(4)]
            if all(p.exists() for p in paths):
                continue
            budget = BUDGET[cat]
            codes, cur_len = sam.sample(
                text,
                seed=1234 + gi,
                do_sample=True,
                temperature=0.9,
                top_k=50,
                token_budget=budget,
                subtalker_temperature=0.9,
                subtalker_top_k=50,
                speaker=f"{voice}/Chinese(PRC)",
            )
            wavs, fs = mw.decode(codes)
            hit = 0
            for k, w in enumerate(wavs):
                sf.write(str(paths[k]), w, fs)
                hit += len(w) / fs >= (budget - cur_len) * 0.08 - 1.0
            print(
                f"{arm}/{voice}/{cat}_{pi:02d} cur_len={cur_len} "
                + "/".join(f"{len(w) / fs:.1f}" for w in wavs)
                + f"s hit={hit}",
                flush=True,
            )
            del codes, wavs
        del sam
        torch.cuda.empty_cache()
    del mw
    torch.cuda.empty_cache()
print(f"ROLLOUT_DONE {ARMS}", flush=True)
