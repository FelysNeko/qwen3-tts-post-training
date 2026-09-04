# probes — 回归与数值探针（合成张量，无需模型/音频）

- `probe_regress.py` — `FullTrainerModel.collate` vs 官方 collate_fn 逐字节相等（含 ragged、legacy 放置公式）、logprob dense 代数（sem shifted-select 位相等、sub 放置无额外偏移、packing 不变量）。
- `probe_gamma.py` — `GRPOConfig.subtalker_weight`（MTP γ）+ per-codebook 归一：γ=1 手算加权参考、γ=0 精确屏蔽（对扰动不变、梯度为 0、inf×0 护栏）、`num_code_groups=1` 旧路径等价、负值拒绝。
- `probe_micro.py` — GRPO logprob micro-batch 路径与全组前向的等价性（逐 token 位等、Dr.GRPO 基线只在全组上算一次）。
- `probe_preprocess.py` — 预处理管线（PROJECT_STATUS §16）：ScoreResult embedding/unwrap 防呆往返、metrics.json 注入 RewardConfig 与旧标定位等（reward_v3）、合成 corpus 离线阶段（load/filter/布局/asset 幂等、finalize 产物 metrics 标量手算参考、扁平 dropped 契约）、scorer set_ref(ndarray==npy) + 无 ref sim=None。
- `probe_clearvoice_ab.py` — vendored MossFormer2_SE_48K vs pip clearvoice 位等验证（2026-08-27 归档：max_abs=0.00e+00，4/4 clips 含 26s）。pip 已卸载，现跑会打印归档结论退出；重装 pip 包可复验。

## 评测基建（需模型/GPU）

- `tmp/general.json` — 8 类 × 4 句评测集（news/chat/multilingual/hard/emotional/ood/short/long；gitignore——含 NSFW 压力测试条目，不入库）。
- `w100_rollout.py` — 双 GPU 双进程 rollout：`python probes/w100_rollout.py {cuda:0|cuda:1} <arm1[,arm2...]> [--cats c1,c2] [--tag T]`。arm 规格 `name`（→ `runs/hp17b_w100_{name}/export`）或 `name=dir`（显式 export 路径 → eval 根 `runs/{name}_eval`）。graphed 采样、batch 4 take 共 seed（1234+全局序号）、断点续跑（跳过已存在 wav）、预算 long=1024/其余=384。
- `w100_score.py <half>` — 双 scorer 打分（half 0/1 按全局条目奇偶分半，端点 5555/5556 与 5557/5558）；report_h{0,1}.json 逐组 flush、断点续。产出：`{eval_root}/report_h{0,1}.json`（组键 `{voice}/{cat}_{pi:02d}`，take 行 {dur, mos, cer, sim}）。

`tmp/` — 归档草稿与已关闭实验的产物（gitignore；台词草稿、microbatch patch/设计稿、§19 native_embedding A/B、TODO 清单等）。

运行（preprocess venv 全过；root venv 只跑前 3 节，无 soundfile 自动 SKIP）：

```sh
workers/trainer/.venv/bin/python probes/probe_regress.py
.venv/bin/python probes/probe_gamma.py
workers/preprocess/.venv/bin/python probes/probe_preprocess.py
workers/preprocess/.venv/bin/python probes/probe_clearvoice_ab.py  # 归档，需临时重装 pip clearvoice
```
