"""w100 四臂打分(双 scorer 条目半切,方案 B)。

用法:python w100_score.py <half>
    half 0: bind 5555/5556,偶数全局组;half 1: bind 5557/5558,奇数全局组。
    同句恒定同进程 → MOS 进程偏移在臂间差分中相消。
报告:runs/hp17b_w100_{arm}_eval/report_h{0|1}.json,组键 {voice}/{cat}_{pi:02d},
    take 行 {dur, mos, cer, sim:{voice:..}} —— 与 w50 报告同构,分析脚本直接复用。
断点续:启动时读已有 report,组键齐全即跳过;每组完成即 flush。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
HALF = int(sys.argv[1])
PUSH = f"tcp://127.0.0.1:{5555 if HALF == 0 else 5557}"
PULL = f"tcp://127.0.0.1:{5556 if HALF == 0 else 5558}"

from qwen3_tts_post_training.cache import CacheLayout
from qwen3_tts_post_training.client.protocol import ScoreField, ScoreItem
from qwen3_tts_post_training.client.trainer import Client

ARMS = [
    "d_ep1=runs/d_ep1",
    "g40=runs/grpo_v2_s40/export",
]
VOICES = ["cyrene", "castorice", "aglaea", "hyacine", "cipher", "hysilens", "cerydra"]
with open(ROOT / "probes/tmp/general.json") as f:
    CATS = json.load(f)
PROMPTS = [(c, i, t) for c, items in CATS.items() for i, t in enumerate(items)]


def eval_dir(arm: str) -> Path:
    name = arm.split("=", 1)[0]
    if "=" in arm:
        return ROOT / f"runs/{name}_eval"
    return ROOT / f"runs/hp17b_w100_{name}_eval"

CENT = torch.stack(
    [
        torch.as_tensor(
            CacheLayout(ROOT / ".cache" / f"{v}/Chinese(PRC)").load_centroid(),
            dtype=torch.float32,
        )
        for v in VOICES
    ]
)
CENT /= CENT.norm(dim=1, keepdim=True)
FIELDS = frozenset({ScoreField.EMBEDDING, ScoreField.CER, ScoreField.MOS})
GROUPS = [
    (arm, voice, cat, pi, text)
    for arm in ARMS
    for voice in VOICES
    for (cat, pi, text) in PROMPTS
]

REPORTS = {arm: {} for arm in ARMS}
for arm in ARMS:
    p = eval_dir(arm) / f"report_h{HALF}.json"
    if p.exists():
        with open(p) as f:
            REPORTS[arm] = json.load(f)["groups"]


def flush() -> None:
    for arm in ARMS:
        p = eval_dir(arm) / f"report_h{HALF}.json"
        p.write_text(json.dumps({"groups": REPORTS[arm]}, ensure_ascii=False))


client = Client(push_endpoint=PUSH, pull_endpoint=PULL, timeout_s=600.0)
for gi, (arm, voice, cat, pi, text) in enumerate(GROUPS):
    if gi % 2 != HALF:
        continue
    key = f"{voice}/{cat}_{pi:02d}"
    if key in REPORTS[arm]:
        continue
    wavs = [
        eval_dir(arm) / f"{voice}/{cat}_{pi:02d}_{k}.wav" for k in range(4)
    ]
    results = client.score(
        [ScoreItem(wav_path=str(w), text=text) for w in wavs], fields=FIELDS
    )
    rows = []
    for w, r in zip(wavs, results):
        emb = torch.tensor(r.get_embedding_unwrap(), dtype=torch.float32)
        rows.append(
            {
                "dur": sf.info(w).duration,
                "mos": r.get_mos_unwrap(),
                "cer": r.get_cer_unwrap(),
                "sim": {v: float(s) for v, s in zip(VOICES, emb @ CENT.T)},
            }
        )
    REPORTS[arm][key] = rows
    flush()
    print(f"h{HALF} {arm}/{key} done", flush=True)

print(f"SCORE_H{HALF}_DONE", flush=True)
