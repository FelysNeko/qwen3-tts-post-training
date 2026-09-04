import sys

sys.path.insert(0, "/home/felys/workspace/qwen3-tts-post-training/workers/trainer/src")
import numpy as np
import torch
import torch.nn.functional as F
from trainer.grpo.grpo import GRPOConfig, column_weights, group_advantage, grpo_loss
from trainer.grpo.logprob import LogProbComputer
from trainer.lora import LoraTrainerModel

torch.manual_seed(0)
ttm = LoraTrainerModel("/home/felys/workspace/qwen3-tts-post-training/runs/sft_v1/export", device="cuda:1")
lpc = LogProbComputer(ttm, speaker="cyrene")

# 8 条真实 cache codes,长度刻意错开(测试 per-chunk padding 路径)
import json
import pathlib

cache = pathlib.Path("/home/felys/workspace/qwen3-tts-post-training/.cache/Chinese(PRC)")
with open(cache / "asset.jsonl") as f:
    rows = [json.loads(l) for l in f][:200]
rows.sort(key=lambda r: np.load(cache / "codes" / f"{r['name']}.npy").shape[0])
pick = [rows[0], rows[30], rows[60], rows[90], rows[120], rows[150], rows[180], rows[199]]
codes = [torch.from_numpy(np.load(cache / "codes" / f"{r['name']}.npy")).long() for r in pick]
lengths = [c.shape[0] for c in codes]
print("code lengths:", lengths)
text = "验收测试用的一句话。"
gs = 8
cfg = GRPOConfig(num_code_groups=ttm.talker.config.num_code_groups)
group_ids = torch.zeros(gs, dtype=torch.long, device="cuda:1")
R = torch.randn(gs, device="cuda:1")  # 合成 reward,A 全组算一次
A, _, _ = group_advantage(R, cfg.variant, group_ids, cfg.std_eps)

def get_grads(names):
    g = {}
    for n, p in zip(names, ttm.trainable_parameters):
        if p.requires_grad and p.grad is not None:
            g[n] = p.grad.detach().clone()
    return g

def zero_grads():
    for p in ttm.trainable_parameters:
        p.grad = None

# ---- ref: full vs micro=4 ----
ref_full = lpc.compute_ref([text]*gs, codes, 0.9, subtalker_temperature=0.9)
ref_micro = lpc.compute_ref([text]*gs, codes, 0.9, subtalker_temperature=0.9, micro=4)
d_all = (ref_full.log_probs - ref_micro.log_probs).abs()
valid = ref_full.mask.bool()
d_mean = d_all[valid].mean().item()
d_p99 = d_all[valid].quantile(0.99).item()
d_m = (ref_full.mask - ref_micro.mask).abs().max().item()
print(f"ref log_probs |Δ| mean={d_mean:.6f} p99={d_p99:.4f} max={d_all.max().item():.4f} | mask max_abs = {d_m}")
print(f"widths: full {tuple(ref_full.log_probs.shape)} micro {tuple(ref_micro.log_probs.shape)}")

# ---- policy: full-batch loss vs chunked loss+backward, grad 对比 ----
pol = lpc.compute_policy([text]*gs, codes, 0.9, subtalker_temperature=0.9)
loss_f, met_f = grpo_loss(pol.log_probs, ref_full.log_probs, R, pol.mask, group_ids, cfg)
zero_grads()
(loss_f).backward()
g_full = get_grads([f"p{i}" for i in range(len(ttm.trainable_parameters))])
norm_f = torch.sqrt(sum((g**2).sum() for g in g_full.values())).item()
W_total = (pol.mask * column_weights(pol.mask, cfg)).sum()
print(f"full: loss={loss_f.item():.6f} kl={met_f.kl.item():.6f} W={W_total.item():.1f} grad_norm={norm_f:.4f}")

zero_grads()
W_used = pol.log_probs.new_zeros(())
loss_acc = pol.log_probs.new_zeros(())
for i in range(0, gs, 4):
    sl = slice(i, i+4)
    pol_c = lpc.compute_policy([text]*4, codes[sl], 0.9, subtalker_temperature=0.9)
    w = pol_c.log_probs.shape[1]; pad = ref_full.log_probs.shape[1] - w
    lp_c = F.pad(pol_c.log_probs, (0, pad)) if pad else pol_c.log_probs
    mk_c = F.pad(pol_c.mask, (0, pad)) if pad else pol_c.mask
    loss_c, met_c = grpo_loss(lp_c, ref_full.log_probs[sl], A[sl], mk_c, group_ids[sl], cfg, advantage=A[sl])
    W_used = W_used + met_c.weight_mass
    loss_acc = loss_acc + loss_c.detach() * met_c.weight_mass
    (loss_c * met_c.weight_mass / W_total).backward()
g_micro = get_grads([f"p{i}" for i in range(len(ttm.trainable_parameters))])
norm_m = torch.sqrt(sum((g**2).sum() for g in g_micro.values())).item()
loss_m = (loss_acc / W_used).item()
print(f"micro: loss={loss_m:.6f} W_used={W_used.item():.1f} grad_norm={norm_m:.4f}")

# grad 逐参数对比(只看有梯度的)
gscale = max(g.abs().max() for g in g_full.values())
rel = max(((g_full[n] - g_micro[n]).abs().max() / gscale).item() for n in g_full if n in g_micro)
print(f"grad Δmax / global grad max = {rel:.5f}")
print("PASS" if d_mean < 0.02 and rel < 0.05 and abs(loss_f.item()-loss_m) < 5e-3 else "FAIL")
