# PROJECT_STATUS.md — Qwen3-TTS 后训练（Cyrene GRPO）

> 本文件为项目当前状态、迁移记录、标定数字复核与已知问题清单。
> 设计真相源仍在 `../playground/SV_REWARD_FINDINGS.md`，本文档只记录"当前机器上发生过什么、已验证什么、还差什么"。

最后更新：2026-08-26（§15 双后端重构 + 全 16 码本 logprob + MTP γ 显式化）

## 1. 目标

对 Qwen3-TTS CustomVoice 微调 ckpt（Cyrene, spk_id 3000）做 GRPO 后训练，reward = 三件套：

```
R = λ_sv·r_sv/std_batch(r_sv) + λ_wer·r_wer/std_batch(r_wer) + λ_mos·r_mos/max(std_batch(r_mos), eps)
```

- `r_sv  = sigmoid((sim_e2v2 − 0.8585) / 0.0966)`，监控用 CAM++
- `r_wer = 1 − CER_qwen3asr`（greedy + normalize()）
- `r_mos = max(0, 2.5 − mos_utmosv2fold0)`（**2026-08-23 已从 sigmoid 改为线性地板**，见 §10 附2）
- λ = (1.0, 1.0, 0.2)；advantage 层 Dr.GRPO（减均值不除 std）+ Clip-Higher ε 0.2/0.3 + KL Schulman k3 β=0.001

## 2. 迁移状态（旧机 → 新机）

| 项 | 状态 |
|---|---|
| 代码仓库 `qwen3-tts-post-training` | ✅ git 干净，已推 GitHub，三个 venv（root/trainer/scorer）全部可用 |
| 双 GPU 可见 | ✅ cuda:0=4070S 12G、cuda:1=5070Ti 16G，`torch 2.8.0+cu128` + flash-attn 正常 |
| 硬编码旧用户名路径（`felysneko`） | ✅ 已全部清除（scorers/client.py + AGENTS.md） |
| TTS 模型 ckpt `/mnt/d/Repository/models/PhiLia093-TTS/` | ✅ 已就位（4.3G，结构完整），rollout 已跑通 |
| playground SV 参考向量 | ✅ `audio/sv_ref_embedding.npy`（E2V2）、`audio/campplus/sv_ref_embedding.npy`（CAM++）已存在 |
| SV ckpt（eres2netv2/campplus） | ✅ modelscope 自动下载（已预拉） |
| UTMOSv2 权重（fold0-4） | ✅ 全部预拉完成（5×818MB） |
| Qwen3-ASR-1.7B-hf | ✅ 预拉完成（3.9G） |
| wav2vec2-base（UTMOS ssl） | ✅ 已自动下好 |
| SQUIM 权重（torchaudio） | ✅ 已下载（360M，实验后弃用） |
| SoX | 本环境不需要（预计算在 playground 已完成，产线不调 sox） |

权重来源已按"上游工具自管 cache"原则实现（`workers/scorer/src/scorer/fetch.py`）：
- SV：`modelscope.hub.file_download.model_file_download`（modelscope 自管 `~/.cache/modelscope`）
- UTMOS：`huggingface_hub.hf_hub_download`（HF 自管 `~/.cache/huggingface`），官方仓库 `sarulab-speech/UTMOSv2`
- ASR / wav2vec2：transformers 原生 from_pretrained

## 3. 标定数字复核（用 playground 产物实算）

| 量 | MD 记录 | 实测（playground 产物） | 结论 |
|---|---|---|---|
| E2V2 池规模 | 1779 条 | `audio/sv_names.json` 1779 | ✅ |
| sim→质心 mean/std | 0.8585 / 0.0966 | `sv_sim_to_ref.npy` 0.8585 / 0.0966 | ✅ 与 `RewardConfig` 一致 |
| pairwise mean/std | 0.7370 / 0.1272 | `sv_pairwise.npy` 0.7370 / 0.1272 | ✅ |
| MOS angry r7（三种子） | [2.19, 2.16, 2.16] | [2.186, 2.160, 2.160] | ✅ τ=2.5 锚点成立 |
| MOS 组均值 easy/hard/angry/sad | 3.01/2.88/2.74/2.70 | 3.07/2.84/2.71/2.84 | ⚠️ 定量微差，定性一致（easy 最高、angry 最低） |
| below-τ=2.5 占比（24=8×3种子） | — | easy 0/24、hard 2/24、angry 5/24、sad 2/24 | 护栏确实只拦"崩坏侧" |

> ⚠️ MOS 组均值与 MD 有 ~0.04-0.14 微差（探测口径/取整差异），排序趋势一致；如做正式报告建议用统一口径重算，不影响 τ=2.5 与 λ=0.2 的设计。

GRPO 超参（`TrainConfig` / `GRPOConfig` / `RewardConfig`，与 MD §七 表一致）：G=8、ε_low/ε_high=0.2/0.3、KL=Schulman k3、β=0.001、lr=5e-5、LoRA r=16 α=64 rsLoRA MLP-only、temp=0.9/top_k=50/top_p=1.0、rep_penalty=None、MTP γ=0。

## 4. 本次改动清单（2026-08-22）

1. **自动下载**（新增 `workers/scorer/src/scorer/fetch.py`；接入 `sv.py`/`utmos.py`/`mos.py`/`serve.py`）
   - SV / UTMOS / ASR 首次加载自动拉取，cache 全部由 HF / modelscope 自管。
2. **硬编码修复**
   - `scorers/client.py`：playground 路径改为 repo 兄弟目录解析 + `Q3TTS_PLAYGROUND` / `Q3TTS_ROOT` 环境变量覆盖，不再含用户名。
   - `loop.py`：model_path 默认 → `/mnt/d/Repository/models/PhiLia093-TTS/`，缺失时显式 `FileNotFoundError`。
3. **BUG：GRPO 组语义**（`loop.py::_pick_prompts`）
   - 原实现一组 8 个**不同** prompt → reward 组内 std 混入 prompt 差异，非标准 GRPO。
   - 改为**同一 prompt × group_size 次 rollout**（标准 GRPO：组内方差 = 同一文本的采样噪声）。
   - 同步影响：`needs_resample` 的组内方差、reward_v3 的 batch-std 现在语义正确。
4. **死锁隐患修复**（`scorers/client.py`）
   - scorer worker 的 `stderr=PIPE` 从不排空，下载进度/日志写满 64KB 缓冲会卡死 trainer。新增 stderr 转发线程。
5. **MOS 硬编码 cuda:0 修复**（`mos.py`/`utmos.py`/`serve.py`）→ 跟随 `--device`。
6. **speaker 参数化**（`loop.py` TrainConfig.speaker、`main.py --speaker`、Sampler/LogProbComputer 透传，默认 `cyrene`）。
7. **依赖**：`workers/scorer/pyproject.toml` 加 `modelscope>=1.17`（已 uv sync）。
8. `AGENTS.md` 同步更新。

> 注：`logprob.py` 的 `[:-5]` 魔法切片为模板稳定假设（assistant 模板不会变），按决策安全硬编码，不加运行时断言。

## 5. 已知问题 / 未决事项

- **needs_resample 常数重复**（`train/grpo.py`）：SV 标定 0.8585/0.0966 与 `RewardConfig` 重复硬编码，改标定易漏改。待统一为从 RewardConfig 取值。
- **ASR 失败语义**：scorer 列任一项为 None → `_scores_to_tensor` 抛错 → 当前 iteration `continue` 跳过（符合预期，不做部分返回）。
- **首次 score 超时**：client timeout 默认 600s；模型已预下载后首次加载仍含 1.7B ASR + UTMOS 冷启动，弱网下可能触及。若复现可调大。
- **组内 prompt 语义改动影响**：改完"同文本×8"后，监控指标（mean_R / r_*_mean）含义变为"单句 8 次 rollouts"，与 MD 的组内随机性实验口径一致；文本池需要多少条对应多少 step，注意 pool 与 num_steps 的关系。
- **eres2netv2 实测 215MB**（MD 未记录大小；初查误读为 818MB 的是 UTMOS fold0）。
- TTS ckpt 下载完成后需人工确认 `/mnt/d/Repository/models/PhiLia093-TTS/` 结构完整（config.json + 权重 + processor + generation_config.json）。

## 6. 运行

```sh
# 下载完成后
workers/trainer/.venv/bin/python workers/trainer/main.py grpo \
  --model-path /mnt/d/Repository/models/PhiLia093-TTS/ \
  --text-pool-path /home/felys/workspace/playground/cyrene_sft_chs_pool.txt \
  --num-steps N --seed 0
```

- trainer 默认 `cuda:1`（5070Ti），scorer 默认 `cuda:0`（4070S）。
- 无 `Q3TTS_*` 环境变量（曾在此声称的 `Q3TTS_ROOT`/`Q3TTS_PLAYGROUND` 系陈旧信息，2026-08-27 核实不存在）。
- 校验：`uv sync`（root）→ `.venv/bin/ruff check .`。

## 7. 未完成事项（源自 SV_REWARD_FINDINGS.md §六/§七/§八，按优先级）

### 阻塞训练
- [ ] **RL 文本池 + train/dev 划分**（用户负责，§七 缺口 2）：方向 held-out 拟 1600/179；池构成 = 普通 + 困难（绕口令/重复，LWR 增广）+ 情感文本。现只有 SFT 池 `cyrene_sft_chs_pool.txt`（1536 行），未做增广与划分。
- [ ] **1-epoch ckpt 重训 + 166 对 A/B 定夺冷启动版本**（§六/§七 缺口 6 时序①）：testzip 166 对已在 playground（`audio/testzip/{test,gen}/`），新 ckpt 到位后 ~20min 可跑，定夺冷启动版本。

### 设计文档已规划、本代码未实现（非阻塞，跑通后补）
- [ ] **CAM++ 交叉监控未接入训练循环**：§16.9 起协议只传 embedding（`sim_camp` 字段已删）；接入时走 caller 侧——`SVScorer.embed_wav('campplus')` + 调用方自持质心。MD §四/§七 要求监控用 CAM++ 盯 SV（防 hacking）。
- [ ] **UTMOS 5-fold 监控未接入**：scorer 默认 fold0；§四 定稿要求"5-fold UTMOS 定期 eval"。`ScorerClient` 也未暴露 `--mos-fold`。
- [ ] **below-τ 比例监控缺失**：§四 定稿要求"每 batch 记录 below-tau 比例（上升=退化警报）"。现 monitor 只有 `mos_dead`（组内 std<eps 熄火），不是 below-τ 占比。
- [ ] **时长/语速投机 + 熵坍缩警报**（§七 缺口 5）：组内时长中位数漂移、熵坍缩警报未实现。
- [ ] **长音频分段**（§四 工程规则）："长音频分段取 mean−λ·std 或最差段"未实现（注：§七 已搁置"时长卫兵"，但"长音频分段"未显式搁置，语义略有出入）。
- [ ] **优化路线**（§七）：decode 195ms→20-30ms——**torch.compile 已落地（2.4x，`--sampler-impl compiled`）**；**CUDA graph（Phase 3）已实现但被 5070Ti 捕获兼容问题阻塞**（见 §9 C1v5，需迁 cuda:0 或升级工具链）。`torch.compile` ASR 补测（§六，预估 ~2.4x）未做。
- [ ] **GVR 全量验证**（§七 缺口 6 时序③）：gen ~2.5h → sv + asr + mos（全量 1779 + GT 子样本 300）作 RL step-0 基线；推迟至 1-epoch ckpt 定稿后。

### 本轮实验新暴露（2026-08-23，听感测试产出）
- [ ] **时长/文本长度异常信号未实现**：黑化肥测试证明模型会系统性**截断长句尾段**，UTMOS 全打高分（2.78-3.27）抓不到 → 需加"时长 vs 文本预期"异常护栏（MD §四 工程规则里提过）。
- [ ] **w2v2_centroid 交叉监控未接入**：实验显示它对真实崩坏金标分位最高（95.6%）且在健康 take 上正确沉默——值得作为 CAM++ 之外的交叉信号。实现成本低（复用 wav2vec2 + 1779 质心）。
- [ ] **专名/OOD 词 CER 污染**：齁/昔涟/肏 等词 ASR 恒定误写（哦哦/洗脸/操），CER 被词表地板污染、组内相对排序失真 → 需 Qwen3-ASR `prompt="Vocabulary: ..."` 词表偏置（MD §三 已提，用户提供词表）。
- [ ] **OOD 文本会拉低 MOS**：极 OOD 文本（齁哦哦哦…）8 条 UTMOS 全 <2.5（1.70-2.27）；改写入分布后恢复（2.11-2.74）。GRPO 池应避免极 OOD 文本，或意识到"低 MOS 可能=文本离域而非模型崩"。

### 排在 GRPO 之后 / 用户提供
- [ ] **专名词表**（用户提供）→ Qwen3-ASR context 偏置复测专名句（§六/§三）。
- [ ] **instruct 数据构建 + swift 模板 user 轮 patch**（§六，已定方向，GRPO 后做）；GRPO 训练期间禁用 instruct（本轮实验提供声学佐证，见 §10 附）。
- [ ] **`build_sv_reward.py` 的 `--min-sim` / 说话人过滤参数**（§六）：26 条半语音仍在池。
- [ ] **playground.ipynb 分布图重跑**（§六，现指旧池）。
- [ ] **assets/ 物理迁移**（§八 迁移清单）：SV 池 npy/json、testzip 166 对、bake-off json 仍留在 playground（repo 兄弟目录，无环境变量机制——`Q3TTS_*` 已确认不存在）；`normalize()/cer()` 已迁入 core lib `reward/text.py`。
- [ ] **gvr.py 全量验证脚本本体**未迁入（scorer 常驻服务已就绪，打分调度逻辑等价，但 GVR 入口脚本未搬）。

## 8. 与设计文档的差异/发现（已核对实现）

1. **打分服务架构（已定：以现实现为准）**：MD §七 缺口 1 草图是"三 scorer 各自 venv 常驻 worker"（SV/MOS/WER 分进程）；实际实现合并为**一个** `workers/scorer`（SV+ASR+MOS 同进程同 CUDA context，transformers 5.15.1），TTS 单独在 `workers/trainer`（qwen-tts→transformers 4.57.3）。冲突解耦成立（4.57.3 只存在 trainer env），且比 MD 草图省一个常驻进程；§八 的"uv workspace 钉不同 transformers 大版本"验证因此被规避。此偏差为有意决策，后续以现实现为准。
2. **GRPO 组语义修复**（本次改动）：MD 组内随机性实验口径是"同一文本×8"，原代码却是 8 个不同 prompt（组内 std 混入 prompt 差异）→ 已按标准 GRPO 改为同文本×8。
3. **eres2netv2 ckpt 实为 215MB**（MD 未记录；初查 818MB 实为 UTMOS fold0）。
4. **MOS 组均值微差 —— 预期之中**：MD 记 easy/hard/angry/sad = 3.01/2.88/2.74/2.70，`seed_probe.json` 实算（3 种子平均）3.07/2.84/2.71/2.84。UTMOS 在固定协议（seed + 8 reps + 顺序处理）下逐值**确定可复现**，但这是两次不同口径的测量；且探测内部三种子组均值即漂 ±0.05-0.11（sad 2.77-2.88），落在 MD 自记的"单文件跨种子漂移 ±0.16"范围内——正是 MOS 定为 hinge 护栏（λ=0.2，不提供驱动梯度）的原因。
5. **MD 中 "sarulab/UTMOSv2" HF 引用为错误**：官方/公开仓库是 `sarulab-speech/UTMOSv2`（与官方 GitHub/Space 同 org）；`sarulab/UTMOSv2` 返回 401。vendored 注释已修正。以现在为准。
6. **`needs_resample` 标定常数重复**：见 §5；暂不处理（算法不变，仅收敛风险提示）。
7. **pool 规模说明**：文本池 = RL 训练用 prompt 集合；现训练循环每 step 只挑 1 个 prompt × 8 次 rollout，故池子条数 ≈ 不重复时的最大 step 数（超出后循环复用），与 `num_steps`（步数）无关。MD 拟 train 1600 / dev 179；`cyrene_sft_chs_pool.txt`（1536 行）是 SFT 文本池，规模接近但未做 RL 池的困难样本增广与划分，口径需在 §7 文本池任务里重新核对。

## 9. 打分性能剖析（2026-08-22 实测，cuda:0=4070S）

### decode 优化：阶段 0/1（2026-08-23，参考 vllm-omni qwen3_tts 实现）

**阶段 0 剖析（B=8 × ~16s 音频，205 步，总 29.9s）——推翻"HF 机器是瓶颈"的预判：**

| 区域 | 耗时 | 占比 | 明细 |
|---|---|---|---|
| cp（code predictor）前向 | 18.3s | 61% | 3180 次 × 5.75ms（5 层、[8,≤16,1024]） |
| main backbone 前向 | 7.2s | 24% | 213 次 × 33.9ms（28 层、[8,1,2048]） |
| 内环 HF 机器 | 3.2s | 11% | prepare/cache/processor，14 次/步 |
| 外环 HF 机器 | 0.6s | 2% | |
| vocoder + 写盘 | 0.5s | 2% | |

- **CPU-launch 瓶颈坐实**：1-token 前向 pipelined 37ms vs GPU 真算力 <1ms——GPU 基本闲置，~95% 时间在 Python/launch。
- 结论：eager 手写循环的天花板就是那 13% 机器开销；10 倍级收益必须靠 torch.compile（Phase 2）+ CUDA graph（Phase 3，vllm 路线：predictor re-prefill + 按 (bsz,seq) 捕图、主干预捕 1 张大图）。

**阶段 1（`trainer/fastgen.py`，`Sampler(impl="fast")` 开关，默认仍 "hf"）：**
- 手写双层解码循环（替换嵌套 HF generate），逐位复刻参考路径：同前向形状/顺序、同 logits 管线（fp32 cast → min_new → suppress → temperature → top_k → multinomial）、同 mrope/rope_deltas、同 RNG 流（每步 15 内 + 1 外次抽样）。
- **验证全过**：① greedy 位等价（B=3 混合长度、含 511 步贪婪复读序列，逐 token EQUAL）② 同种子全管线等价（RNG 流一致 → rollout 逐位复现）③ 自复现 ④ 冒烟（Sampler 集成路径）。
- 速度：148.5 → 135 ms/步（**+9%**，符合阶段 0 预判的天花板）。B=8 长句生成 34.3 → 26.9s。
- 附带发现：**greedy（含 subtalker greedy）必陷入复读不吐 EOS**（3/3 文本撞 512 帽）——该模型必须采样解码，也排除了"greedy 定长 batch"的优化路线。

**阶段 2 探针 + 落地（2026-08-23，`--sampler-impl compiled`，参考 vllm-omni）**

解耦实验：不动 fastgen、直接在参考 HF 路径上挂极简 compile（backbone ×2），量化"最少代码"的收益：

| 变体 | 结果 | 结论 |
|---|---|---|
| A：compile(dynamic=None, epilogue_fusion=False) | **2.41x（30.9 → 12.8s）** | 落地 |
| B：mode="reduce-overhead" | 崩：CUDA graph 输出被覆写（DynamicCache 跨步持有 RoPE 中间输出） | HF 动态循环 + 自动图 = 死路，坐实 Phase 3 需手写静态 cache |
| C：HF compile_config | 跑通但收益结构同 A | 不采用 |

Step-0 验证（三项全过）：
1. **确定性**：固定输入 compiled forward 逐位确定；warm 后全管线 3 连跑逐位相等。冷调用会经历 dynamo static→dynamic 图升级、与 warm 路径数值不同 → **解法 = 启动 warmup（2 次不同长度 dummy gen，~15s 一次性）**。vs eager 漂移属实（hidden max 1.49、greedy 第 0 步翻 argmax）——验收标准从"位等价"改为"warm 后自复现 + 分布级"。
2. **变长 recompile 风暴：无**。首个长度 20 张图完成动态升级后，4 个新长度 new_graphs=0；VRAM 稳定 4.0GB。
3. **LoRA 相容**：on/off 两个 guard 变体各编译一次（合计 ~204 图 → cache_size_limit 调 256）；切换开销 ~1.3s；扰动 lora_b 在 compiled 图里生效（on≠off）。

落地（阶段 1 代码上）：`fastgen.enable_compile()`（幂等）+ `FastSampler(compile=True)` 内建 warmup + `Sampler(impl="compiled")` + `--sampler-impl {hf,fast,compiled}` CLI。**B=8 长句 rollout ≈ 13s（eager 30.9s）。**

**compile 成本模型（2026-08-23 实测）**：训练循环内（单进程）**零重编译**——warmup 71s 一次性（`FastSampler.__init__` 内），之后任意文本长度/batch/LoRA on-off 全走缓存图（同形状 3 连跑 0.9s/0.95s/0.95s；冒烟 t_ref 首编后 29.4s→0.1s）。**跨进程重启要重付 ~71s**（dynamo 追踪不落盘；triton kernel 二进制有磁盘缓存 `$TMPDIR/torchinductor_<user>/`，715 个 .cubin 命中即不重编——`restart_warmup` 运行期间缓存 0 新写入证明 71s 全是追踪不是编译）。**WSL 重启清 /tmp → kernel 缓存也丢，完整重编 ~60-90s**。对 5000 步单进程训练，71s 摊到 25h 可忽略；代价只在频繁起停小实验/崩溃恢复时显现。

**Phase 3 探针（Step B + C1，2026-08-23，注意力内核方案定稿）**

| 探针 | 结果 |
|---|---|
| **transformers StaticCache + FA2** | ❌ 坏（80% hidden 偏移）。根因：`create_causal_mask` 对 StaticCache 取 kv_length=1024 → `flash_attention_mask` 见全 True → 塌缩 None → FA2 `is_causal` 对整个 1024 池注意（含未初始化尾段） |
| **transformers StaticCache + SDPA** | ✅ 语义正确（4% hidden / 0.5 logit 漂移），但 SDPA 掩码 kernel ≠ is_causal kernel（~0.8 bf16 漂移，mask 矩阵测试证实）。若走 SDPA 需全模型换 kernel，放弃 |
| **flash_attn_with_kvcache（vLLM 同款，FA2 原生）** | ✅ **方案定稿**。连续池 [B,LMAX,HK,HD] + 每步新 k/v 写 cache_seqlens 处 + 单 kernel 内写+注意；GQA 16/8 原生；**CUDA graph 捕获 + 内容可变 cache_seqlens 回放 + 逐步增长写缓存全过**（vs flash_attn_func 差 ≤0.004）。GRPO 组内 8 行同文本 → 每步所有行 cache_seqlens 相同（=步数），天然适配 |
| **每步构成**（eager fastgen, B=8, 254 步） | **cp 84ms(67%) + main 38ms(30%) + 胶水 4ms(3%)**——graph 两个 forward 即可，胶水不管（O(T²) has_eos 担忧不成立） |
| **内存** | 模型 4GB + 静态 KV（main LMAX=1024，8 kv 头）~0.94GB ≈ 5.1GB 峰值，16G 无压力 |
| **L_max** | 池最长 267 字→191 token→prefill≈202；LMAX=1024 覆盖 ~60s 音频，溢出回退 compiled |

> 路线修正：不切 SDPA（用户指出 vllm-omni 用 FA2——其 talker 走 vLLM PagedAttention 后端即 flash-attn 系；FA2 本身没问题，是 HF 的 FA2+StaticCache 整合坏了）。正解 = 复刻 vLLM 思路：固定形状 FA2 decode kernel（`flash_attn_with_kvcache`）+ 固定地址 KV 池 + 内容可变 cache_seqlens，无 padding 进入 kernel。

**C1v5 探针 + 结论（2026-08-23，`trainer/fastgraph.py` 已实现但 cuda:1 不可用）**

实现了 `impl="graphed"`（`fastgraph.py`：静态 KV 池 + `flash_attn_with_kvcache` 图捕获 + 图外采样/EOS + LoRA 兼容），但**在 trainer GPU（cuda:1 = 5070Ti）上捕获不可用**：

- **5070Ti（cuda:1，Blackwell GB203 + WSL2）上 CUDA graph 捕获整体不可靠**：flash-attn 全家族（func/varlen/kvcache）**和 torch 原生 op（纯 GEMM）都会烘焙内容可变输入**（replay 无视输入变化 → 每步输出恒定）；还出现空图（replay 0.1ms 空操作）与随机 "operation not permitted when stream is capturing"。
- **同一代码在 4070S（cuda:0）上完全正常**（C1v4：flash_attn_with_kvcache 捕获 + 内容可变回放 + 多步增长全过，vs flash_attn_func 差 ≤0.004）。
- 排查链：create_causal_mask 的 `.all()` CPU 同步（已用 attention_mask=None 绕开）→ LoRA 转置 Parameter GEMM（contiguous 转置可修捕获错误，但不解决烘焙）→ 最终定位为 GPU/工具链问题。
- 已加 **捕获验证 + 烘焙探针**（`_capture_graph`：replay vs eager 位等 + 扰动输入验证输出变化）→ cuda:1 上 `impl="graphed"` **快速失败并给出清晰报错**（不静默产垃圾）。

**结论：Phase 3（CUDA graph rollout）在本机不可用，`impl="graphed"` 彻底弃用**。发货加速器为 **`impl="compiled"`（torch.compile，2.4x，B=8 长句 ~12s，warm）**。

**C1v6 增补（2026-08-23，双 GPU 交换可行性 + 分发修复）**：

- **修复分发 bug**：`Sampler.sample` 只对 `impl=="fast"` 分发到 fast 路径，`graphed` 一直静默落到 HF generate（此前"graph 冒烟 self-repro 全过"是 HF 的假象，graph 路径从未真正端到端跑过）。已改为 `("fast","compiled","graphed")` 全分发。compiled 上轮已接好（冒烟 warm 数字两轮一致，10-16s），本次补的只是 graphed。
- **graph 路径端到端是坏的（不只在 cuda:1）**：分发修复后在 cuda:0 实测 `impl="graphed"` → **511 步不停（近 max_new_tokens），vs compiled/hf 190 步触发 EOS**——logits 语义漂移（静态池陈旧/捕获不全），且 35ms/步 比 compiled 单步（57ms）快不了多少、总时长反而更长（17.9s vs 10.9s）。**图路径不收敛到 EOS = 产垃圾，不可用。**
- **换 GPU 训练（4070S 训 + 5070Ti 打分）不值得**：4070S 唯一优势是图捕获正常，但图路径是坏的；compiled 两卡几乎同速（cuda:0 10.9s / cuda:1 12.2s）；4070S 更小（12GB）对 opt/ref/policy 训练更不利。**保持 5070Ti 训练 + 4070S 打分。**
- 实测（分发修复后，cuda:1，B=8 长句）：hf=30.4s，**compiled warm=12.2s**（自复现 ✓）；GRPO 循环 warm 稳态 ≈ rollout 10-16s + score 4.6s + ref 0.1s + opt 0.06s ≈ 15-21s/步。
- `_capture_graph` 保留捕获验证+烘焙探针（cuda:1 上 `impl="graphed"` 干净快速失败）；fastgraph.py 保留为研究资产，不再接入生产。

### 每 GRPO step（8 条）打分成本（warm）

| scorer | /条 | 8条/step | 占比 |
|---|---|---|---|
| SV（E2V2+CAM++） | ~0.07s | ~0.6s | 1% |
| ASR（Qwen3-ASR 1.7B） | ~0.32s | ~2.6s | 5% |
| **MOS（UTMOSv2 fold0, 8 reps）** | **~6s** | **~48s** | **94%** |

### MOS 内部构成（13.4s 长音频）

- CPU 特征构建（librosa mel 264ms + torchvision resize 175ms + ssl crop）≈ **640ms/rep × 8 = ~5.1s/条 ← 瓶颈**
- GPU 前向（batch=8 分摊）~0.3-0.5s/条
- load_audio 240ms/条

### 关键发现：MD 记录有误

- MD §四 实测 "8条×8reps≈46s"（5.75s/条）与现在一致 ✅
- 但 §七 搁置决策写 **"UTMOS 0.6s/条非瓶颈"——差 10 倍**（实际 ~6s/条）
- "打分非瓶颈、TTS 生成 48.8s/组是大头"按 20s 长音频算；RL 池是 2-13s 短句，gen 估计 ~10-20s/组，而 MOS 稳定 ~48s/组 → **MOS 与生成同级甚至更高，不再是"非瓶颈"**

### 行动

- [x] **GPU 化 MOS 特征构建（已实现，20.5x）**：librosa mel → torch.stft + mel filterbank（librosa.filters.mel 预计算一次）on GPU；RNG 仍走 numpy 顺序选 crop；resize 在 GPU 张量上。
  - 实现：`scorer/utmos/dataset.py::GPUSpecBuilder`，`--gpu-mel` 默认开启（`--no-gpu-mel` 回退 librosa）。
  - **一致性**：修掉两处细节后 MOS 与 librosa 路径差 ≤0.03 —— ① `power_to_db` 的 `top_db=80` 裁剪（漏了会 -0.72 偏移，当时还以为是 STFT 差异）；② torch.stft window 长度 = win_length（torch 内部自动 pad 到 n_fft）。
  - **确定性**：GPU 路径 Δ=0.0000 ✓
  - **速度**：warm 打分 8 条组 SV ~0.6s + ASR ~2.7s + MOS ~4s ≈ **7s/step**（原 ~51s，MOS 从 ~6s/条 → ~0.5s/条）；冷启动 12 条含模型加载 33.6s（原 109.5s）。
- [ ] 上线 5-fold UTMOS 监控前必须先 GPU 化（否则成本×5）。→ GPU 化已完成，5-fold 已可行。
- [ ] load_audio 重采样可改 torchaudio（次要）。

### GPU 化实测（2026-08-22）

| 路径 | MOS/条 | 8条打分 | 确定性 | MOS 偏差 |
|---|---|---|---|---|
| librosa (CPU) | ~7.9s | ~63s | Δ=0.0000 | 基准 |
| torch (GPU) | ~0.5s (warm) | ~4s | Δ=0.0000 | ≤0.03 vs librosa |

τ=2.5 锚点复测（angry r7）：CPU 2.375 / GPU 2.367，均低于 2.5 ✓。

> ⚠️ **发现：seed_probe.json 的 τ 校准口径不可复现**。原 `mos_seed_probe.py` 用 `num_workers=4` 跑出 angry r7≈2.16（MD 据此定 τ=2.5）；但文档自己确认 workers=4 不可复现。用生产确定性协议（vendored, num_workers=0）实测 angry r7 = 2.37-2.38（比 MD 高 ~0.2），仍低于 τ=2.5，护栏语义成立但余量更小。GPU 与 CPU 确定性路径一致（≤0.03）。

## 10. UTMOS 替代 A/B（2026-08-22，决定：**不替换**）

### 动机
MOS 护栏只需二分"崩坏 vs 正常"，UTMOS（3.9GB、TTA、域偏置）太重。测了 9 个候选能否替代。

### 数据
- 健康 68 = GT 30 + testzip_gen 30 + easy 8
- 合成分级崩坏 240 = 20 条 GT × 4 类人工破坏（clip/noise/repeat/silence）× 3 严重度
- 真实崩坏金标 = angry r7
- 另打分 256 条新鲜 rollout（4 setting × 4 rep × 16 行，`audio/rollout_hearing/`）

### 候选
UTMOS / SQUIM / wav2vec2 质心(对角白化距离) / wav2vec2 帧自相似 / 物理先验(clip/flatness/hf/sil_frac/sil_maxrun)

### 结果（OR 护栏 z-score 归一，健康 68 vs 合成 240）
| guard | AUC | rec@5% severe | angry_r7 z | 结论 |
|---|---|---|---|---|
| **UTMOS（现役）** | **0.774** | **0.463** | 1.02 | 仍是最佳单护栏 |
| centroid+flat+sill | 0.749 | 0.425 | 0.84 | 接近不赢 |
| centroid 单独 | 0.715 | 0.362 | 0.84 | 弱一档 |
| all-9 | 0.779 | 0.425 | 1.02 | 拼多多不涨 |

### 单人工件（per-artifact AUC）
- noise：w2v2_centroid **0.991** > phys_flatness 0.986 > utmos 0.916
- repeat：w2v2_centroid 0.890 > utmos 0.808
- silence：phys_sil_frac 0.839 > utmos 0.819
- clip：全员不行（utmos 0.554 / phys_clip 0.463）——剪波不适合做判别，留给 WER/SV
- **angry r7（真实崩坏）**：w2v2_centroid 打到 **95.6% 分位**（最强）> utmos 86.8%
- **SQUIM 全线拉胯**（AUC 0.685、angry r7 仅 17.6% 分位）→ **弃用**
- framesim 两种实现（mean 相邻帧 / 近同帧占比）都抓不住"段复读"——复读是**周期自相似**（lag 处相似）不是相邻帧相同；但 w2v2_centroid 意外覆盖了 repeat

### 决策
1. **不替换 UTMOS**：GPU 化后 0.5s/条、确定性已解决、τ=2.5 已标定，综合仍最强。轻量候选没有全面超过它。
2. **w2v2_centroid 记为最有价值的新信号**：零标签、确定性、复用已加载 wav2vec2；对真实崩坏金标分位最高、抓 noise/repeat 最强。可作为附加交叉信号，或未来重标 τ 时的候选。
3. **SQUIM 弃用**；物理先验可作廉价预筛（秒级抓明显 clip/silence），非替代。
4. ⚠️ **合成 ≠ 真实**：最终拍板仍等 256 rollout 的听感标签（`audio/rollout_hearing/tag_sheet.csv` 已生成 76 条精选）——用"天然好/坏"重跑最终 A/B。

### 附：instruct OOD 的声学证据（rollout 统计）
instruct="angry" 系统性改变声学画像：phys_flatness 中位数 0.096→0.024、phys_hf 0.26→0.20、w2v2_centroid 距离 837→1327（angry 组全部劣于 GT 中位数）——与 MD "instruct OOD 推飞音色" 结论一致，且**声学特征可见**，可作为训练期禁用 instruct 的客观佐证。

### 附2：组内排序测试（2026-08-23，听感 vs 打分器）
- 取「黑化肥」绕口令生成 8 条同条件 rollout，**全部 8 条都截断最后一句**（5.5-7.0s，应 ~8-10s）——系统性模型行为，UTMOS 给全高分（2.78-3.27）**抓不到截断**，佐证"时长/文本长度异常"护栏的必要性。
- 用户听感确定：倒三 = {00, 01, 06}（内部顺序不确定，06 最确定）。
- 各候选预测的倒三与人类交集：**全部仅 1/3（只有 06）或 0/3**（phys_hf）；centroid 出现 00/04 并列（均 +3.84σ）。
- **结论**：同质退化（全截断）的 8 条里，连人类自己都排不稳，没有任何候选能复现人类倒三——坐实 MD "质量项不做组内排序、只做地板/护栏" 的判断。护栏设计维持 UTMOS（λ=0.2、softplus 地板方向），w2v2_centroid 仅作交叉信号。
- **[x] r_mos 已改线性地板（2026-08-23）**：`reward.py::r_mos_fn` 从 `sigmoid((mos−2.5)/0.2)` → `max(0, 2.5−mos)`。验证：健康组（全 ≥τ）r_mos≡0 → std 精确=0 → `mos_dead=True` 必然触发（不再赌 std<eps）；混合组崩坏 take 得线性罚、健康 take 全 0；τ 锚点 angry_r7(2.37)→罚 0.13、好 take(3.0)→0。`mos_scale` 字段保留未用（留作将来 softplus 膝盖斜率）。

### 附3：Auto + 正常文本 组内测试（2026-08-23，最干净的一条证据）
- 「可这些…都只是猜想吧？万一我们完成了仪式，结果却又完全不是这么一回事呢？」language=Auto × 8，时长 8.4-9.2s（无截断）。
- **人耳：8 条无差别、都很好**；UTMOS 原始 2.68-3.18 全在 τ=2.5 之上。
- 组内 z 跨度：w2v2_centroid **0.20σ** / sil_frac 0.39σ（与"人耳无差别"一致）vs UTMOS **1.31σ** / flatness 1.29σ / hf 1.38σ（**捏造差异**）。
- **结论**：UTMOS 等在好 take 上会产生 ~1.3σ 的组内噪声——若用于组内排名即纯噪声 advantage；floor 设计（全在 τ 上 → 全零 → std≈0 → 熄火）正好吃掉它。centroid/sil_frac 在好 take 上正确沉默。**终局：UTMOS 当地板护栏、w2v2_centroid 当交叉监控、SQUIM/framesim/clip 弃用、补时长异常信号抓截断。**

## 11. 听感排序测试汇总（2026-08-23，Auto，G=8）

> 目的：验证"打分器组内排序 vs 人耳"，测试方法与产物已留档。

| 文本 | 类型 | 关键结果 |
|---|---|---|
| 黑化肥绕口令（长） | 域内 stress | **全 8 条截断尾句**（5.5-7.0s 应 ~8-10s），UTMOS 全高分抓不到 → 时长异常信号必要性 |
| 正常台词「可这些…都只是猜想」 | 域内普通 | 人耳 8 条无差别；UTMOS 等捏造 ~1.3σ 组内差 → **质量项必须当地板** |
| 「不要！不要这样！齁哦哦哦…肏了」（极 OOD） | 域外 | 8 条 UTMOS 全 <2.5（1.70-2.27）；排序"合理"（高分不崩但表现力差、低分音质损坏）——**文本离域导致** |
| 「哈哈哈哈~羽毛……犯规了…憋不住」（轻量改写） | 域内 | MOS 恢复 2.11-2.74，w01 三轴一致最好 / w00 三轴一致最差 |
| 「专人…妥善…识刻锚」（池台词） | 域内 | 干净；l01/l05 双强、l02 垫底；"识刻锚"被 ASR 误听（专名 OOD） |
| 「Q3 KPI + 四是四」（hard） | 域内长句 | **全 8 条健康**（MOS 2.65-3.04 全 >2.5），CER 全低——该"hard"文本实际不难 |
| hard 组 advantage 分析 | — | 健康组 advantage std=1.63（非小）全部来自 SV+WER，地板静默；h00 垫底（拖沓被 WER 抓 CER 0.111 + SV 0.762），与你耳朵一致；崩坏组（OOD）std=1.55 对照 |

### 汇总结论
1. **组内排序在"同质健康"和"同质崩坏"下连人耳都难**，打分器更差 → 质量项不做排序、只当地板（已落地）。
2. **在真有差异的组**（轻量改写、池台词、hard 组），打分器排序与人耳一致（w01/l01/h01 等三轴共识最好、00 系垫底）。
3. **OOD 文本降低 MOS 是文本离域而非模型崩坏**——GRPO 池应避免极 OOD 文本。
4. **hard 文本（Q3 KPI+绕口令）实际读得很健康**——之前 MD 把它当压力文本，但 2-epoch ckpt 已能干净读出；RL 池的"困难样本"需要更强的（重复/更长）。
5. **group size 论文佐证**：FlowTTS-GRPO (§3.3) CV3 用 **G=8**、F5-TTS 用 G=10，无消融；我们的 G=8 与其一致且有自身组内方差实证。

## 12. Token Budget 重构 + Sampler 契约收敛（2026-08-25）

**动因**：`max_new_tokens 4096` / `lmax 1024` / `runaway_t_max 400` 三参语义重叠（`lmax` 硬墙截 `max_new`，`runaway` 护 `OOM`），`hf` 为算 `cur_len` 临时 `new EagerSampler`（`hf.py:23 get_attr` 恶心代码）。

**改动**：
* `TrainConfig` 删三参，改 `token_budget 512` 训 / `token_budget_infer 1024` 推（`loop.py:62`），`max_new = token_budget - cur_len` 动态算，`lmax = token_budget_infer`（`cuda_graph.py:263` 静态池），`runaway` 改 `t_max+cur_len>=token_budget`（`loop.py:236`）
* `Sampler.sample` 改 `-> tuple[codes, cur_len]`（`base.py:92`），`Decoder.decode` 加 `@inference_mode`（`decoder.py:30`），`RolloutResult.cur_len: int` 去 `=0` 兜底（`rollout.py:26` 必填）
* `base.prefill_cur_len(processor, texts) -> cur_len` 纯函数（`base.py:36` `max_n+10`），`hf` 改 `prefill_cur_len` 不再 `import eager`（`hf.py:13`），`cur_len` 精确 `16` vs `eager 16` 已验
* `CLI` `--token-budget / --token-budget-infer`（`main.py:20`），`build_sampler(lmax=token_budget_infer)`

**Smoke（`8×8=64/步` 长池 `512/1024` 2步，`B8`）：**
* `graphed`：`t_max 271/268 total 327 <512` 未 `runaway`，`gpu_alloc 5.0G reserved 15.1G` 安全，`t_rollout 45s t_score 78→36s`
* `fast`：`t_max 278/258 total 334 <512`，`gpu_alloc 4.1G`，`t_rollout 168→170s` 慢 `graphed 4.3x` 符合预期
* `B8 T500 14.4G / T550 OOM` 实测墙仍在，`512` 对 `B8` 安全，`B4 T800 12.3G / T1000 14.4G` 可撑 `60s`

## 13. ZMQ 完全解耦 + 零线程批量流水（2026-08-26）

**动因**：`Popen + PIPE + fd dup` 魔改 `stdio` 侵入强、生命周期耦合、`/dev/shm` 永不删 `64 wav 46MB → 640 wav 460MB` 线性泄漏；`REQ/REP` 锁步无流水。

**架构**：
* `trainer bind PUSH 5555 + PULL 5556`（`client.py:38`），`scorer connect` 两个 `PULL/PUSH`（`serve.py:45`），`pyzmq 27.2` 双 `worker` 独 `venv`，`JSON` 单帧，`HWM 1000 LINGER 0`，`64 wav 2.5KB` 远不堵。
* **零线程批量**：`loop.py:228` `for gi 8 push (send_score 8 wav)` 非阻塞（`scorer T=1` 即算）→ `for gi 8 pull (recv_score Poller 600s)` 排水；`ZMQ` 缓冲即 `rollout ∥ score`，`wall = 8*rollout + last_score`（`204s→173s` 省 `30s`），保序单 `scorer` 无需 `id` 排序。
* **删档**：`scorer read-only never unlink`，`trainer recv 后 _cleanup_wavs` 删 `8 wav + rmdir`（`loop.py:117`），`B8` 峰 `64 wav 46MB → 0` 收敛。
* **双起**：无 `Popen`，`manual` 分别 `terminal` 起：`workers/scorer/.venv/bin/python workers/scorer/main.py --sv-dir ...` 后 `workers/trainer/.venv/bin/python workers/trainer/main.py grpo ...`；`ZMQ` 自动重连，无序亦可。

**验证**：`dummy scorer` `3×8 wav` `push 3 → pull 3` 保序 `id` 对齐 `PASS`（`pyzmq` 27.2），`graphed/fast` 仍 `B8 512` 安全。

## 14. 全库审计 + 死代码清理 + Sampler 单句 batch API（2026-08-26）

**动因**：全库只读审计发现死代码 9 项 + 未对齐 16 项；`cur_len = mask.shape[1]`（eager 含左 pad）与 `text_len`（graphed 去 pad）双口径靠"同文×8"巧合相等；`sample(texts: list[str])` 需 `assert len(set(texts))==1` 护同构，丑。

### 14.1 死代码清理（已落库）

* `loop.py`：删 `scorer_device` 字段 + CLI `--scorer-device`（ZMQ 解耦后无消费者）；删死函数 `_scores_to_tensor`（loop 内联 tensor 构造）；修 `token_budget_infer` 过期注释（`if None defaults to` 残留）
* `reward.py`：删死字段 `mos_scale` / `std_eps`（v3.1 RAW 后无消费者）；header docstring v3 → **v3.1 RAW 公式**（旧 `R=λ·r/std` 是 C1v8 前的公式，已误导）
* `client/scorer.py`：删 `self.last_raw`（写后无读）
* `asr.py:42`：`texts_ref: list[str | None]` → `list[str]`，删 `if ref else None` 分支——空 ref 直接算 cer（cer("") 按 `edit_distance/0` 语义返回），数据坏就让它炸，不静默产 None
* TYPE_CHECKING 全清：`samplers/base.py`、`rollout.py`、`samplers/__init__.py`、`utmos/model.py` 的 torch/TrainerModel/SimpleNamespace 改回直接 import（用户裁决：typing-only import 也无所谓，不要 `# noqa: F821`）
* `TrainConfig.kl_beta 0.01 → 0.001`：对齐 `GRPOConfig.kl_beta` 与 PROJECT_STATUS §1 β=0.001（原三处 10× 差）

### 14.2 Sampler API 收敛：单句 + 内部 batch 展开

* `Sampler.__init__` 加 `batch_size: int = 8`（GRPO group size），`build_sampler` 删除，工厂改为 **`Sampler.build` @staticmethod**（match/case，懒 import 四实现）；`samplers/__init__.py` **清空为 0 字节**——所有调用方改从 `trainer.samplers.base` 直引
* `sample(texts: list[str])` → **`sample(text: str)`**：内部 `[text] * self.batch_size` 展开成组，异构 assert 不再需要（类型系统层面杜绝异构 batch）
* `rollout_group(prompt: str, ...)` / `RolloutResult.prompt: str`（原 `prompts list`），`loop.py` 直传单 prompt
* `warmup_sample(text, token_budget)` 去 batch 参数；`TorchCompile/CudaGraph.__init__` 走 `super(batch_size=...)`，`self.batch` 别名删除统一 `self.batch_size`
* `prefill_cur_len(list[str])` max 版删除，仅存单句版 `(ids.shape[1]-8)+10`；别名 `prefill_cur_len_single` 已删
* `hf.py` `n = self.batch_size` 内联进 generate 参数

### 14.3 cur_len 单真源

* 真源 = `base.prefill_cur_len(processor, text) -> (n+10)`（head8+tail n+1+last1 代数恒等式，非经验拟合；实测 `你好。→12 / A*100→23 / 异长混批→15` 与 eager_manual/mask.shape 三方一致）
* `eager.py`：`cur_len = prefill_cur_len(...)`（原 `mask.shape[1]`）；`cuda_graph.py`：`text_len = prefill_cur_len(...)`（原 `mask.sum.max()`）——预算数字 `max_new/runaway` 三实现同源
* 物理层不动：eager 左 pad + mask 屏蔽（DynamicCache），graphed 紧凑写池 `0..text_len-1`（FA causal 扫全池不能有 pad）——预算归真源，布局各走各
* `eager.py first` 死计算留注释不删：上游 qwen-tts `modeling_qwen3_tts.py` generate 的 `tts_text_first_token` 段 streaming 下是真输入、non_streaming 下被 `[:, :-1]` 切掉，保留作上游 parity

### 14.4 审计遗留未决（按优先级）

* **P0**：`scorer/main.py:69` `scorers.score()` 无 try——坏 wav/ref 即 scorer 进程死、trainer 600s 超时才 skip（需 ErrorResponse 或维持 fail-loudly 决策）
* **P1**：`grpo.needs_resample(sim, sv_eps)` 死参（签名 SV/WER vs 实现 WER-only，C1v8 后有意为之）；`logprob.py device=self.model.device` vs `ttm.device`；`sv.py Kaldi.fbank(t.to(device))` GPU 兼容疑虑
* **P2**：`cuda_graph max_new=min(budget,lmax)-cur_len` vs eager/hf 仅 budget（训练 512<1024 无差）；protocol `sim_camp/transcript` 有产无消（CAM++ 监控 TODO）

**验证**：`ruff check .` All checks passed；`py_compile` 过；`prefill_cur_len` CPU 真 processor 探针 9 组（含 emoji/异长/混批）三方一致；`trainers main --help` 无 scorer-device。行为零变更（同文×8 数值路径不变），待下次 smoke 复核。

## 15. 双后端重构 + 全 16 码本 logprob + MTP γ 显式化（2026-08-26）

**动因**：SFT 复用规划（官方 dataset.py/collate/teacher_forcing）+ logprob 只盖列 0 的 IS 欠修正（θ 移动 `past_hidden` → 冻结权重的 predictor 条件分布在 policy/ref 间漂移，ratio 不修）。

### 15.1 结构重组

* `src/trainer/` 分层：共享核收敛进 `model.py`（`ModelWrapper` [CollateBatch + `collate` 单方法含完整布局数学 = 官方 collate_fn 逐字节移植探针验证，config 统一走 `self.model.config.talker_config`；`tokenize_assistant` 方法化——全部调用方本就持有 ttm，processor 管线归模型对象；`teacher_forcing` 共享前向核] + `lora.py` [LoraTrainerModel])；后端目录 `grpo/`（loop/logprob/rollout/decoder/samplers）与 `sft/`（占位）。原 `batch.py`/`model/` 已删档
* 官方 teacher_forcing 泄漏 bug 修正：全长前向 → `hidden[:, :-1][codec_mask[:, 1:]]` 取 p−1 隐态（官方选 p 位隐态把 c_p 的 1..15 层 embedding 漏进条件，且与生成时 `past_hidden` 不一致）
* `TrainerModel.teacher_forcing(batch, speaker_vec) -> TeacherForcing(sem_logits 稠密, predict_mask, codes_flat, sub_logits)`：故意不传 labels（内部 loss 硬编码温度 1.0）；sem_logits 返回稠密 `[:, :-1]` 以保 SFT 的 EOS 目标槽位
* grpo 后端消费 `LoraTrainerModel`；sft 两态皆可（方案 b = lora flavor，官方全参受 16G 显存排除）

### 15.2 全 16 码本 logprob + γ 显式化

* `logprob.py` 重写 dense：语义头 gather（与旧 pred_start 循环位等）+ `forward_sub_talker_finetune` 盖码本 1..15；打包契约 `[B, T*Q]`、列 `t*Q+j` = 时步 t 码本 j；pad 列填零防 −inf·0 NaN
* **MTP γ 决策链改写**：MD §四原 pin γ=0（"失败模式全在语义层，v1 不提取 15 码本 logprob"——当时自标临时简化）；Fish S2 §4.3 验证 Fast AR 参与 PG（共享序列 advantage + 各码本 KL + γ·fast）；现 `GRPOConfig.subtalker_weight=1.0` 默认（Fish 式）+ `subtalker_time_norm=True`（÷Q，整步权重 ≈ 语义单步），`0.0` 精确复现 MD 旧行为
* inf×0=NaN 护栏：γ=0 时零权重列的 exp 溢出不再泄漏进求和（`torch.where(w>0, …)` 在乘法前屏蔽，_clipped_loss 同）
* 未决：γ 数值消融未跑（1.0 vs 0.0 短程轨迹对比）；MLP-only vs all_linear 未评估（Fish 原文仅 MLP，PEFT 惯例是 all-linear，两者都只是出处不是证据）

### 15.3 验证

* collate vs 官方参考逐字节相等（含 `[1,n]` text_ids、ragged、最小 T=1 三用例）+ legacy 放置公式几何等价
* logprob dense 三层：语义头 vs 旧 meta/pred_start 循环**位相等**；sub naive-stacking 等价 + 时间位放置（无额外偏移，草稿期曾错置被此探针抓住）；packing 不变量（mask 计数 = ΣT×Q、off-mask 全零、全有限）
* γ 探针 5 组：γ=1+time_norm 对手算加权参考；γ=0 精确忽略 sub 列（对任意扰动不变）；num_code_groups=1 legacy 路径与改动前逐值一致；负 γ 拒绝；γ=0 梯度精确为零 + 加权均值耦合方向正确
* ruff / py_compile / 全模块 import / main --help 全过；合成探针已归档 `probes/`（`probe_regress` + `probe_gamma`，不再依赖 `/tmp`）；真模型 GPU smoke 未跑（同 codes 下新旧 ref 位等性 + KL/ratio 非 inf 待验）

### 15.4 长文本显存边界 + loop micro-batching 延后（2026-08-27）

* **实测边界**（`LoraTrainerModel`，`cuda:1 5070Ti 16G`，长文本池 16 句，`cur≈52–73`）：`B=8` 单次 `teacher_forcing`（28 层 talker + 5 层 predictor×`N`，`N=ΣT`）在 `token_budget=200`（`t≈201,N≈1109`）峰值 `13.3G` 可过，`250` 即 `14.9G OOM`；`512`（`t≈513,N≈3605`）`14.6G OOM`。短文本 `B=8,512` 同样 OOM——加入 sub-talker 全 16 码本后峰值翻倍，旧记录 `B8 T500 14.4G`（仅语义头）已失效
* **B=4 对照**：`350`（`13.0G` 可过）/`400`（`14.2G OOM`），印证 `O(B·T+Q·N)` 线性关系
* **决策**：`model.py` 的 micro-batching 已回退（用户要求先探极限，不提前做）；`loop` 的 micro-batching（组内 `B=8 → micro=2` 逐 micro `compute_*` + `backward` 后释放图，峰值 `O(micro·T)`）**延后到最后统一做**，且只需动 `grpo/loop.py` 单文件——`logprob`/`teacher_forcing` 已是动态 `B`，`train/grpo.py` 的 `group_advantage/_column_weights` 可直接 `import`。严格等价版需外提 `A_full/den_total` 并内联 `loss_t`（+25 行），近似版直接对切片调 `grpo_loss`（+8 行，`group=micro`）
* **scorer** 常驻 `cuda:0 4070S 9G`，`B8·G8@512` 长文本 smoke 在 `teacher_forcing` 段 OOM 已复现两次（`q_proj` 层 `SiLU` 分配 `36M` 失败），待 micro 落地后重跑

## 16. 预处理管线 `workers/preprocess` + `.cache` 数据契约（2026-08-27）

**目标**：把「corpus wav 目录 → 训练就绪数据」固化成第三个 worker（结构同 trainer/scorer：独立 pyproject + `.venv`），产出 `.cache/{lang}/`（enhanced wav + codes/embedding npy + asset.jsonl + centroid.npy + metrics.json）；训练侧标定全部改从 metrics.json 读，playground 依赖（sv_ref npy 路径 + 硬编码 0.8585/0.0966/2.5）就此解除。

### 16.1 输入 / 输出契约

- **输入**（唯一必填）：`/path/to/{target_language}`（wav 目录）；约定 sibling `{target_language}.jsonl`，每行 `{"name", "text"}`，name ↔ `{target_language}/{name}.wav` 一对一。首跑数据集：`../delta-me13/corpora/tts/cyrene/Chinese(PRC)/`（1825 条 48kHz mono，982MB）。
- **输出**（`.cache/{target_language}/`，`.gitignore` 已覆盖；v2 契约，2026-08-29 现状）：
  - `enhanced/{name}.wav` — MossFormer2_SE_48K 增强，48kHz PCM_16（时长不变）——**唯一派生音频**；下游全部现场重采样（SV `AF.resample`、ASR `load_audio(path, 16000)`、MOS `librosa.resample`、speech_tokenizer `encode(wav, sr=48000)`），不再自留 16k/24k 副本（原设计有三档，落地时核实零消费者后裁撤，省 ~2.3GB/语言）
  - `codes/{name}.npy` — `[T,16]` int32（speech_tokenizer 对 48k enhanced wav 提取，`encode` 内部自重采样；`np.save` 不用 `torch.save`——zip 容器有确定性风险）
  - `embedding/{name}.npy` — E2V2 192d float32 单位范数（`np.save`）
  - `asset.jsonl` — 每行 `{name, text, transcript, cer, mos, sim, checksum {corpus, enhanced, codes, embedding}}`：transcript/cer/mos 为写时事实（text 层落行）；`sim` 是磁盘派生字段——finalize 对当时质心**全员现算**，pool 未变 ⇒ 逐位等于旧值，SV 升级 ⇒ 质心重算 + 全员刷新（零 scorer 往返）；checksum 现算于磁盘，不覆盖 MOS 的跨进程非确定性（§16.9 末条）
  - `centroid.npy` — 单位范数 E2V2 质心（float64，`np.save`；`post_apply_embedding_layer` 物化——此后质心 ⟺ 磁盘池；finalize 无条件重建兜底）
  - `metrics.json` — `sim`/`cer`/`mos` 三者统一 `{mean, std, percentiles(1..99)}` 对齐形状（sim stats = `RewardConfig.sv_center/sv_scale` 标定，cer 反映域内 ASR，mos 只记录待 τ 规则）+ `n_clips` + 溯源参数（dataset/model_path/min_tokens/min_seconds + `dropped`＝`DropReasons._asdict()` 扁平四键）——centroid 已独立 npy，`clearvoice`/`sv_model` provenance 键已移除（五稿 2026-08-29）

### 16.2 任务表 + 八步 `sync()`（逐层幂等，中断重跑安全）

`sync()`（pipeline.py）一次跑完：`precompute_task_table` → corpus 层 → enhanced 层 → codes 层 → embedding 层 → post（质心物化）→ `collect_corpus_metrics`（text 打分）→ `finalize`（metrics 全量重建）。原始四阶段「产物存在即跳过」设计已被 §16.8 重构取代（只查存在性检测不到损坏/漂移）。

- **filter**：jsonl 加载 + 一对一 wav 校验；token 数走 `Qwen3TTSModel.processor`（codes 阶段反正要加载它），时长走 `sf.info`；`--min-tokens 2 --min-seconds 0.1`（同 playground 旧默认）；drop 归因 = `DropReasons` NamedTuple 四键（orphan_corpus/orphan_manifest/less_than_min_tokens/less_than_min_seconds）
- **任务表 + 四个 material 层**：`TaskRow(name, corpus, enhanced, embedding, codes)` 四 bool（= 行 checksum 是否仍描述磁盘），依赖序过层；精确 bit 语义——salvage（重生成 sha == 存储 sha ⇒ 纯 bit-rot）翻 True，真漂移/fresh 保持 False 自愈；corpus 层例外（源不匹配 → 下游全 False，源不可再生无 salvage），codes 层不做 embedding 级联（embedding 派生自 enhanced 而非 codes）；三个 material 层开头 `prune_foreign` 清理离开过滤范围的 clip 产物
- **打分路径**：codes 层仅 enhanced+embedding 有效的行**本地**重提取（不惊动 scorer）；embedding 层走 scorer `{EMBEDDING}`（batched 串行——depth-2 重叠已证伪，§16.8）；text 打分走 scorer `{TRANSCRIPT, CER, MOS}`，todo = `not row.enhanced` 纯表读，单 append 句柄 + 每批 flush 落行（崩溃韧性）

### 16.3 协议（core lib，字段级按需 + 强类型终态；演化全记录见 §16.9）

- `ScoreField(StrEnum) = {EMBEDDING, TRANSCRIPT, CER, MOS}`（值 = wire key）；`ScoreRequest.fields` **必填**（空集 client 侧 assert 拒绝）——「需要什么就打什么分」，scorer 按字段组 lazy-load 派发，未请求组 timing 记 0
- `ScoreResult` 公共可空字段（None = 未请求）+ 每字段一个 **`get_*_unwrap()`** 防呆读取（读错字段必炸，不让 None 悄悄漏到下游）；`recv_score` 返回 `list[ScoreResult]`（不降级 dict）；原 `sim`/`sim_camp` 字段已删——**sim 一律 caller 侧现算**（preprocess：finalize 对全池 embedding 现算；trainer：`vectors @ sv_centroid` 一次批量 matmul）
- 精度论证：float32 ⊂ float64 → JSON 往返逐位精确；dot 从 scorer 挪到 caller 只改求和顺序，相对误差 ~1e-7，低于 sv_scale≈0.097 与一切 flameout 阈值 4-5 个量级

### 16.4 训练侧消费 metrics.json

- trainer `--metrics-path` **必填**（assert）：`reward_config_from_metrics` 读 `sim.mean/std` → `RewardConfig.sv_center/sv_scale`；`load_centroid` 读 sibling `centroid.npy` → float32 renormalize（复刻旧 set_ref 配方）；`mos_tau` 维持 2.5（mos 统计只记录，待 gate 规则）
- loop `send_score({EMBEDDING, CER, MOS})`，sim = 本地 `vectors @ centroid`；旧 scorer `--sv-ref/--metrics` 旗已随「scorer 彻底无标定」删除（§16.9）——一个常驻 scorer 服务 preprocess + 训练，零重配置

### 16.5 ClearVoice：pip 依赖移除，VENDORED MossFormer2_SE_48K（2026-08-27 定稿）

**pip `clearvoice` 0.1.2 已从依赖里删除**，增强阶段改用 vendored 包 `workers/preprocess/src/preprocess/clearvoice/`。动因（pip 包三宗罪，全部源码核实）：

1. `checkpoint_dir: "checkpoints/MossFormer2_SE_48K"` 烧死在包内 yaml 且 **CWD 相对**，参数只从 yaml 解析（`parse_args(["--config",…])` 无视 sys.argv）→ 无 override API；检查/下载都发生在 `ClearVoice.__init__` 内部
2. `SpeechModel.__init__` 用 nvidia-smi 自动选「空显存最多」的 GPU 并 `torch.cuda.set_device`——进程级全局副作用
3. 依赖树污染：`numpy>=1.24.3,<2.0` + `librosa==0.10.2.post1` + opencv/scenedetect/gdown/pydub 等，仅为用一个模型

**Vendor 结构**（沿用 speakerlab/utmos 的 byte-faithful 惯例，ruff 全豁免 `clearvoice/mossformer2_se/**`）：

- `mossformer2_se/` — 模型定义 6 文件 VERBATIM（state-dict 键必须对上发布 ckpt）；外部依赖仅 einops + rotary-embedding-torch。唯一 byte-faithful 偏差：上游死导入 `from torchinfo import summary` 注释掉、torchinfo 依赖不装（用户裁决 2026-08-27）
- `decode.py` — `stft/istft/compute_fbank` 自 `utils/misc.py` 逐字移植 + decode 循环自 `utils/decode_batch.py`；配置常量 = 包内 yaml（win 1920/384/fft 1920/mel 60/hamming/48k）
- `load.py` — ckpt 加载移植自 `networks.py::SpeechModel`（`last_best_checkpoint` → ckpt → `state['model']` 键映射三连）；抓取 = 纯 `snapshot_download` 进 **canonical HF cache**（`alibabasglab/MossFormer2_SE_48K`），无 local_dir、无 chdir、无 symlink

**上游怪癖考古**（A/B 探针抓出，`probes/probe_clearvoice_ab.py`）：

- `one_time_decode_length` 护栏比较的是 `inputs.shape[0]`＝**batch 维**（1 > 960000 恒 False）→ tensor 模式下滑窗分支**不可达**，一切输入都走整段批量解码；滑窗分支本体还有 2D 张量三重索引 + 步进错缩进两处死 bug → vendor 忠实复刻"永远批量路径"的实际行为
- **位等验证**：同 seed（kaldi fbank `dither=1.0` 吃全局 torch RNG）下 vendor vs pip **max_abs=0.00e+00、4/4 clips identical**（含 26s 长条）——探针现为归档（pip 已卸载，重装可复验）
- 管线每 clip 前 `torch.manual_seed(0)` 固定 dither，保证任意重跑/断点续跑产物逐位一致

### 16.6 验证

- `--limit 8` 端到端（scorer 在位）→ schema/shape/维度检查
- codes round-trip：`ModelWrapper.decode(codes)` 时长 vs 原始 wav 合理
- **等价性回归**：手写含旧标定（0.8585/0.0966/2.5）的假 metrics.json → 注入路径与硬编码路径 `reward_v3` 逐值相同
- 幂等：重跑各阶段跳过已有产物；scorer `--metrics` 与 `--sv-ref` 两种启动 sim 逐值相同

### 16.7 落地记录（2026-08-27，全部完成）

- **协议/worker/消费三线全落地**：`ScoreResult` += `vector` + 可空 sim；`SVScorer.embed_wav/sim_to_ref`（无 ref 不再 raise）；`workers/preprocess`（pyproject+main+pipeline+`clearvoice/` vendored 包）；scorer `--metrics`；trainer `--metrics-path`（`reward/metrics.py::reward_config_from_metrics`，缺省回退硬编码 → 零行为变更）
- **ClearVoice 迁移收尾（§16.5）**：pip `clearvoice` 已卸载（连带 12 包），venv 只加 einops/rotary-embedding-torch（torchinfo 死导入已注释不装）；A/B 位等（0.0，含 26s）后切换；`.cache/_clearvoice` 权重目录已删——权重唯一居所 = canonical HF cache
- **probe**：`probes/probe_preprocess.py` 23 项全过（协议往返、注入等价、质心手算参考、合成 corpus 离线阶段、set_ref ndarray==npy、无 ref sim=None）；`probes/probe_clearvoice_ab.py` 位等记录归档
- **live smoke**（`--limit 4 --batch 2`，scorer 无 ref 模式，vendor 路径）：asset.jsonl 4 行全 schema 合规——codes `[95,16]`（×80ms=7.6s 与原 wav 精确吻合）、vector 单位范数、CER 0-0.1、utmosv2 2.22-2.62；metrics.json 质心单位范数；重跑幂等；codes decode round-trip 时长逐条相等（7.60=7.60s 等）
- **`--metrics` 数值等价**：scorer 重启后 sim 0.9514841 vs asset `dot(vector,centroid)` 0.9514841——质心加载与 embed 确定性双验证；scorer `--metrics` 与 `--sv-ref` 互斥由 CLI 断言
- ⚠️ **utmosv2 分布预警**：本 corpus（GT 音频）utmosv2 mean 2.40、p75=2.50——**约半数 GT 条目低于现行 τ=2.5**（旧 τ 由 playground angry 锚点标定，不随域迁移）。r_mos 地板只作用于 rollout 组（rollout MOS 分布待训练时观察），但"换 corpus 是否重标 τ"已从理论问题变成实际记录在案的数据点：metrics.json 的 percentiles 就是为这个决策留的
- 运行顺序注意：全量 preprocess 期间 scorer 须以**无 ref** 模式跑（质心要等全部向量）；结束后重启 scorer `--metrics .cache/{lang}/metrics.json` + 训练 `--metrics-path` 同路径

### 16.8 校验守卫的分阶段缓存重构（2026-08-28）

**动因**：§16.2 的「产物存在即跳过」只查存在性——文件损坏、corpus 变更、上游漂移都检测不到。重构为**每阶段独立产物文件 + 逐行四重 checksum + 任务表**（`precompute_task_table`）。

- **产物布局**：`enhanced/{name}.wav`（原 `clearvoice/`，「唯一派生音频」不变）+ `codes/{name}.npy`（`[T,16]` int32；用 `np.save` 不用 `torch.save`——zip 容器有确定性风险）+ `embedding/{name}.npy`（E2V2 192d float32）+ `asset.jsonl` + `metrics.json`
- **行 schema v2**：`{name, text, transcript, cer, mos, sim（finalize 回填）, checksum {corpus, enhanced, codes, embedding}}`——codes/vector 移入文件；新旧 schema 互不兼容（裁决：直接删缓存重跑，无迁移）
- **任务表 + 四层**：`TaskRow(name, corpus, enhanced, embedding, codes)` 四 bool（=缓存产物仍与磁盘一致），依赖序过层：corpus 层（源不匹配 → 下游全 False，源不可再生无 salvage）→ enhanced 层（重生成 → **新 sha == 存储 sha ⇒ 纯 bit-rot，salvage 下游不动**；≠ ⇒ 级联 codes/embedding False）→ codes 层（仅 enhanced+embedding 有效的行本地重提取，不惊动 scorer，同 salvage/级联）→ embedding 层（todo = embedding False；batched 串行；行即时 append 断点续跑）
- **depth-2 流水线证伪（2026-08-28 全量实测）**：1825 条全量跑埋点 `extract 94.1s + recv 787.9s ≈ wall 884.6s`——**重叠≈0**。结构原因：recv 返回时 scorer 恰好做完当前 chunk，接着在下一轮 extract 期间干等（~0.85s/chunk）；scorer 是 8x 瓶颈（0.44s/clip vs 提取 0.05s/clip）。GRPO loop 的 rollout∥score 重叠之所以有效，是 trainer 侧 rollout 本身长；这里已拆回 batched 串行，wall 不变、代码少一层 inflight/drain
- **centroid 独立成 `centroid.npy`（2026-08-28）**：与 codes/embedding 统一 np.save 约定（float64、单位范数）；metrics.json 不再内嵌 192-float JSON list，只留标量标定（sim/wer/utmosv2/provenance）。`reward.metrics.load_centroid(metrics_path)` 从 sibling 路径派生 npy——scorer CLI 与 wiring 零改动。选 npy 弃 .pt：消费方只当裸数组用（set_ref 吃 ndarray），且 centroid 不参与 checksum 体系（finalize 全量重建），torch 容器无加分项
- **设计裁决**：确定性（clearvoice 固定种子 + codes/embedding 确定）使「重生成 sha 相等 ⇒ 输入未变」成立——复合 hash 方案（embedding = H(corpus, enhanced, …)）的记账可省，表+级联即其语义；模型指纹（全局 provenance）**本期不做**（裁决 2026-08-28）；metrics **永远全量重建**（~2k 行是毫秒 matmul，选择性重算无价值）；行 sim 由 finalize 对新质心统一回填（质心一漂移全体 sim 刷新，不做增量）
- **验证**：probe 24 项（新增：表 bool 语义、corpus 级联、损坏只翻自己的位、finalize sim 回填+全量重建）；live smoke 三连——全新跑（4 clips 全落盘，codes `[95,16]` 等）→ 空闲幂等（`4 cached clips, 0 to process`）→ **salvage 实证**（人为损坏一条 enhanced → 重生成同字节 → 0 重打分 + asset.jsonl 逐字节不变）

### 16.9 Embedding-only 协议 + 字段级按需打分（2026-08-28）

**动因**：「需要什么就打什么分」——两个调用方字段集本就近乎不相交（trainer 用 sim/cer/mos，preprocess 用 vector/transcript/cer/mos），而 scorer 无条件跑三模型、全字段返回。

- **协议**：`ScoreField(StrEnum) = {VECTOR, TRANSCRIPT, CER, MOS}`（值 = result dict key）；`ScoreRequest.fields` 默认 ALL_FIELDS；`ScoreResult` **删除 `sim/sim_camp`**，四个字段 `| None`（None = 未请求）。scorer 按字段组派发 + lazy-load（VECTOR→SV、TRANSCRIPT/CER→ASR、MOS→MOS），未请求组 timing 记 0。**双向 wire 兼容**：旧 scorer 忽略 `fields` 键全量返回（多给不缺）；新 scorer 默认全量
- **scorer 彻底无标定**：`--sv-ref/--sv-ref-camp/--metrics` 三旗、`SVScorer.set_ref/sim_to_ref`、campplus 静默回退全部删除；`MODELS["campplus"]` 模型定义保留（embed 能力在，无协议入口）。**§16.7 的重启舞步作废**——一个常驻 scorer 永久服务 preprocess + 训练，零重配置
- **trainer**：`--metrics-path` 变必填（assert）；启动时 `load_centroid` → float32 renormalize（复刻旧 set_ref 配方）；loop `send_score(fields={VECTOR, CER, MOS})`（每 step 省 ~12k float 的 JSON），sim = `vectors @ sv_centroid` 一次批量 matmul
- **精度论证**：float32 ⊂ float64 → JSON 往返**逐位精确**；dot 位置从 scorer 挪到 trainer 只改求和顺序（numpy→torch），相对误差 ~1e-7，低于 sv_scale≈0.097 与一切 flameout 阈值 4-5 个量级；实证先例 = §16.7 的 0.9514841 == 0.9514841
- **明确不做**：字段集不接入 preprocess 产物/checksum（任务表阶段粒度已够）；跨监控（CAM++/w2v2）将来走 caller 侧（请求 vector 或另加模型参数），不走协议 sim 字段
- **验证**：probe 23 项（协议子集往返/默认全量/ScoreField==result keys/请求 mode=json 可序列化——第一版 `model_dump()` python mode 保留 frozenset 导致 zmq `send_json` 崩，已修并加探针防回归）；live 实测——损坏 embedding 重打分，transcript/cer/sim 对基线**逐位一致**（vector 传输无损 + 本地 matmul 等价旧 scorer dot）；同进程二次重打分 MOS delta=0.0
- **字段级补全贯彻（2026-08-28 续）**：`score_missing` 拆为 `embed_missing`（{VECTOR}，text 仍有效的行当场以 cached 合并落行）+ `score_text_missing`（{TRANSCRIPT,CER,MOS}，todo = 无行或 `checksum.enhanced` 不匹配——行内分数只依赖 enhanced+text，这是精确失效判据）；codes 提取整体搬回 `apply_codes_layer`（门控 `not codes`，吸收 fresh 提取——embedding-rot 行不再被无关重提 codes）；checksum 统一 `Checksum.from_disk` 现算，行内旧值只做失效判定。**修复路径的 MOS 漂移根除**：损坏 embedding → `[text] 0 clips to score` → 新行含 MOS 逐位一致（mos 2.447265625 == 2.447265625）；preprocess scorer 调用收敛到 `client.score()` 便捷方法（两步 send/recv 形式是 depth-2 遗留，trainer loop 的跨组流水线仍合法使用）
- **sim 快照化 + 行写入权收敛（2026-08-28 终稿）**：`ScoreRequest.fields` 变**必填**（`ALL_FIELDS` 删除，空 fields 在 client 侧 assert 拒绝；probe 断言缺 fields 必须 raise）；`embed_missing` → `apply_embedding_layer`（与其余三层同构：写 npy + bit 一律 True，**不写行**——行由磁盘重建，salvage 概念在此层失去意义）；**质心计算前移**到 `score_text_missing` 开头（此时全池 embedding 必在盘上）——行内 `sim` 当场实算，`sim=None` 占位 + finalize 回填机制消亡；`AssetEntry.sim` 必填，语义 = **写入时快照**（与 transcript/cer/mos 同一冻结原则，永不改写）；`finalize` = 权威刷新点：kept 行仅刷 checksum（磁盘现算，sim 不动）+ **无条件**重算/存质心 + 全量重建 metrics（兜底「clip 被裁但 text todo=0」的池变化）。行写入权 = text 层（活，保崩溃韧性）+ finalize（权威 compact）。验证：空闲跑 asset.jsonl+centroid.npy **逐字节不变**；损坏 embedding → `[embed] 1 / [text] 0` → 行含 mos+sim 位等（mos 2.447265625 / sim 0.9423833750032248 双双保持）
- **prune + post 层 + sim 派生化（2026-08-28 二稿，推翻上条的部分裁决）**：`--limit` 调试旗**整体删除**（调试 = probe 自建小 cache）；质心物化从 text pass 前移为独立 **`post_apply_embedding_layer`**（契约：此后 centroid.npy ⟺ 磁盘池；复用/重算由池变化信号决定）；三个 material 层开头各自 `prune_foreign`——离开过滤范围的 clip 其产物全部清理（过期内容不再无限留存）；**bit 语义精确化**（用户模型）：bit = 「行 checksum 是否仍描述磁盘」——embedding 层做 sha 对比，bit-rot（重生成字节相同）→ salvage 翻 True，真漂移/fresh → 保持 False；`post` 消费**活表**（`pre_embedding` 快照别名删除——旧语义下才需要），`removed ∨ any(not embedding)` 即池变化；**finalize 重写为 map-merge**：text pass 返回 `name_to_text`，finalize 按新覆盖旧组装 transcript/cer/mos，而 **checksum+sim 归为磁盘派生字段、每轮全员现算**（pool 未变 ⇒ 逐位等于旧值；embedding 漂移（如 SV 升级）⇒ 质心重算 + 全员 sim 刷新，零 scorer 往返——sim 消费 embedding 的彻底版，此前的「写入时快照」裁决就此作废）。验证：probe 28 项（新增漂移全员刷新/裁剪 clip 三目录清理/post 复用-重算双路径）；live——空闲跑双文件逐位一致（首轮为 sim 计算路径统一的 1-ULP 一次性迁移，次轮起定点）；损坏 embedding → **salvage → 质心复用** + `[text] 0` + 行含 mos+sim 位等。后续加固（未做）：metrics.json provenance 的 `sv_model` 加 ckpt sha256；centroid.npy 旁存池指纹做复用自校验
- **MOS 跨进程非确定性（本次实测发现，与协议改动正交）**：同一 clip 跨 scorer 进程重启 MOS 漂移 ~0.17（2.619→2.447），同进程内重复位等（0.0）——GPU 核级非确定性（cudnn autotune 类），任何 scorer 重启都会触发。行 checksum 不覆盖 MOS；metrics.json 的 utmos 统计因此混有跨运行的 run 级噪声（信息性用途，不影响 τ 决策方向，如需统一需全量重打分一次）
- **强类型协议 + 精确语义收尾（2026-08-28 三稿；ScoreResult 设计回收一版后定型）**：第一版把 payload 全隐藏（extra="allow" + model_post_init 收编 + property 独占读取）——功能验证通过但机制过巧，按用户裁决**回收**，定型为：`ScoreResult` 公共可空字段（None = 未请求）+ 每个可空字段一个 **`get_*_unwrap()`**（assert 剥离 None 的防呆读取——知道请求了什么的调用点用 unwrap，读错字段必炸而非让 None 悄悄漏到下游；直接属性访问是诚实类型）；`ScoreField.VECTOR → EMBEDDING`（值 `"vector"→"embedding"`），`recv_score` 直接返回 `list[ScoreResult]`（不再 `model_dump` 降级为 dict），trainer/preprocess 全部走 unwrap 读取；text 层 live-append 的 `append_row`（每行开一次文件）删除——**单个 append 模式句柄外提到批次循环外** + 每批 `flush()`（崩溃韧性等价旧逐行开关）。同时 `apply_enhanced_layer/apply_codes_layer` 收敛到与 embedding 层相同的精确 bit 语义（salvage→翻 True；漂移/fresh→保持 False 自愈），`score_text` todo 由此变纯表读 `not row.enhanced`（sha 复查删除），codes 层对 embedding 的**过度级联**删除（embedding 派生自 enhanced 而非 codes——qwen-tts 升级不再误触发全量重嵌）。验证：probe 28 项（协议段重写：unrequested=None + unwrap 防呆 raise/嵌套 ScoreResponse 往返/枚举值=wire keys）；live——新协议 scorer 重启后空闲双文件逐字节一致；损坏 embedding → salvage → 重嵌 1 条 4.7s + 质心复用 + `[text] 0` + 行含 mos+sim 位等
- **NamedTuple 化 + 命名收敛（2026-08-28 四稿）**：`DropReasons` dataclass→**NamedTuple**——手写 `to_dict`（连同 `corpus.orphan`/`duration`/`manifest.length` 那层重命名嵌套）删除，metrics.json 的 `dropped` provenance 改 `._asdict()`（stdlib API，扁平 + 字段本名，一次性变形；reward 端无消费者已验证）；probe 属性访问/kwargs 构造零改动。`score_text → collect_corpus_metrics`（返回注解顺手修正为 `dict[str, ScoreResult]`，上轮漏改）；`build_metrics` **内联进 finalize**（scalar 契约的验证随之折叠进 probe 的 finalize 块：sim mean/std 手算参考 + wer/utmos 标量 + 扁平 dropped key 契约，覆盖从「调助手函数」升级为「断言真实 finalize 产物」）。验证：probe 全过；live——首轮 metrics.json 因 dropped 变形重写（dropped=44 tokens/2 seconds，n_clips 1779），次轮起三文件全部字节稳定
- **metrics 键名对齐（2026-08-29 五稿）**：`wer → cer`、`utmosv2 → mos`（与产出它们的量同名），三者统一 `{mean, std, percentiles(1..99)}` 对齐形状——sim 也补上 percentiles；`clearvoice`/`sv_model` 两个 provenance 键移除（用户裁决：暂时不要）。reward 端零波及（`reward_config_from_metrics` 只读 sim.mean/std，`wer` 键本就无人消费）。验证：probe 标量检查更新（sim percentiles 手算参考 + cer 全零分布逐键相等 + mos mean/中位）；live——首轮 metrics.json 重写（sim 0.8709 / cer 0.0579 / mos 2.6），次轮三文件字节稳定
- **契约段同步（2026-08-29 六稿，纯文档）**：§16.1–16.4 仍停在 2026-08-27 原始设计（`clearvoice/` 目录名、v1 行 schema、内嵌 centroid、`--metrics` 旗、`sim/sim_camp` 可空、`vector` 字段名）——五轮演进只记在日期 bullet 里，契约段从未跟上，代码侧 grep 证实零残留（全部旧引用都在日期记录内）。重写为现状：§16.1 = v2 输出契约（enhanced/codes/embedding 三目录 + asset v2 行 + centroid.npy + metrics 对齐形状）；§16.2 = 任务表 + 八步 `sync()`（含原始四阶段设计的取代指针）；§16.3 = 字段级按需 + unwrap 终态；§16.4 = 训练侧必填消费。日期记录（§16.5–16.9）一律不动
## 17. SFT worker `workers/trainer/sft/`（2026-08-29）

**战略位置**：SFT 永远从 base ckpt 出发（`Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-Base`，HF repo id 直传自动抓取；已微调的 `PhiLia093` 退役），产出的 custom_voice export 是 GRPO worker 的续训起点。数据 = preprocess cache（`asset.jsonl` + `codes/*.npy`）——codes 的 12Hz token 空间跨模型尺寸同构（codec id 2149/2150/2148/2155/2156/2157 与 vocab 3072/2048 全同；0.6B 上 sem CE 起步 1.08 实证非 OOD）。

### 17.1 结构（薄封装，全部复用已验证内核）

- `model.py`：`SftTrainerModel(ModelWrapper)`——**全量 FT**（talker 全栈含 code_predictor/MTP 头，官方语义；GRPO 冻 predictor 的姿态在此不适用）；冻结仅 `speaker_encoder`（输出 detach 进 slot 6，天然无梯度）。`speech_tokenizer` 是普通包装对象（非 nn.Module），参数本就不在模型树。`load_base_speaker_encoder`：custom_voice ckpt 无 speaker encoder（仅 `tts_model_type == "base"` 时构造）→ 从 base safetensors 惰性抽 `speaker_encoder.*`（`safe_open`，76 张量 strict load）——SFT 从 base 出发时自动跳过
- speaker 条件：**用户指定一条参考音频**（`--speaker-audio`）→ soundfile 读（stereo→mono）→ librosa 24k 重采样 → 模型自带 `extract_speaker_embedding` → `[hidden]` 向量广播进 slot 6（teacher_forcing 的 spk 查表同路径），启动时提一次全程复用；官方逐条 `ref_mels→speaker_encoder().detach()` 的单说话人退化版
- `loop.py`：`load_sft_dataset`（asset 行 + codes npy → (text, [T,16] long)，缺 npy 即 assert）→ 每 epoch 种子洗牌切片 → `collate → teacher_forcing → CE`。**loss 单 shift 配对**：`CE(talker_logits, codec_0_labels[:,1:], ignore_index=-100) + 0.3·CE(sub_talker_logits, talker_codec_ids[:,1:])`——官方 `sft_12hz.py` 的 double-shift bug（`inputs_embeds[:, :-1]` + 内部 HF CE 再 shift 一次 → logits@p-2 vs 标签@p+1；`hidden_states[codec_mask[:, :-1]]` 选位 p 的 hidden 泄漏目标）不复现，`outputs.loss` 不可用
- 超参照抄官方：AdamW lr 2e-5 / wd 0.01 / grad_accum 4 / clip 1.0 + 线性 warmup 10（GRPO 教训：Adam 首步 sign jolt）；`model.train()`；fp32 CE
- ckpt：**滚动 `latest.pt`**（trainable state_dict + optimizer + 批位 pos；全量 FT 每步编号文件太贵）+ `--resume`（key 集合 assert 防 architecture 漂移）
- `export_custom_voice`：白名单拷贝（config/generation/preprocessor/tokenizer/vocab/merges + `speech_tokenizer/`）→ config 手术（`tts_model_type=custom_voice` + `spk_id`/`spk_is_dialect` 写入 `--export-name`，已存在复用其 slot，新名分配 max+1）→ safetensors **仅 `talker.*` 键**（与原版 model.safetensors 布局同构；官方脚本存全量 state_dict 反而多键）+ 参考音频 embedding 烘入 `codec_embedding.weight[slot]`；repo id 源经 `snapshot_download` 解析（训练后 cache 命中）

### 17.2 显存账与事故（16GB 卡现实）

- **1.7B 全量 FT + AdamW(bf16) 物理不可行**：1.92B 参数 → params/grads/exp_avg/exp_avg_sq 四份 bf16 = 14.3GiB 地板 > 16GB（官方脚本默认给 ≥24GB 卡，零冻结零 ckpt 零 8bit）；参数分布：28 层 73.5%、text_embedding 311M 16.2%（文本词表已全训，冻结损失最小）、code_predictor 9.1%。裁剪选项（冻 predictor+text_emb ~7GB / 8-bit Adam / Adafactor）已议未行——**先用 0.6B 实验**（用户裁决）
- 0.6B（906M 参数）实测：B2/accum4 GPU 5.6GB 稳跑；WSL2 **15GB 系统 RAM** 是另一条红线——首轮 smoke 死机复盘 = 中止的工具调用未杀 setsid 重跑进程 + scorer 常驻 3.2GB → 双模型栈系统 OOM（AGENTS「一次一个重实验」再证实）。稳态配方 **B1/accum4**：RAM 3.2GB / GPU 5.6GB / 单步 0.2-0.5s

### 17.3 验证（live smoke @ 0.6B-Base，8 clips 1 epoch）

- `speaker_vec [1024]`（自带 encoder）、trainable 906M；B1 两步 loss 2.49→2.83（sem 1.08→1.46，值域健康）、grad_norm 27-49、lr warmup 2e-6→4e-6
- 导出产物：custom_voice + `spk_id {cyrene: 3000}`、talker.* 前缀 403 张量、row3000 norm 10.44 vs base 0.015（未训练随机行被烘焙覆盖）
- **L4 decode sanity**：导出目录直接 `Qwen3TTSModel.from_pretrained` → `generate_custom_voice(speaker='cyrene', language='Auto')` → 4.16s/finite/peak 0.87（~20 字句长合理）

## 18. 入口收敛 + 日志系统 + 系统指标收敛（2026-08-29）

### 18.1 trainer 单入口（`main.py grpo|sft`）

- `sft_main.py` 已删，路由 = `main.py` 子命令（`dest="command", required=True`）。argparse 风格（用户定稿）：共享参数（model-path/device/dtype/lr/warmup-steps/weight-decay/grad-clip/seed/out-dir/ckpt-every + `--resume`）声明一次挂 `shared = argparse.ArgumentParser(add_help=False)`，经 `parents=[shared]` 装到两个子命令——**必须挂子解析器**：直接挂根解析器的话共享参数只能写在子命令之前（`main.py sft --lr x` 会 unrecognized）；子命令特有参数裸 `add_argument`，无 help 字符串
- CLI→Config 合并 = `dataclasses.replace(SftConfig()/TrainConfig(), **overrides)`，`overrides` = `vars(args)` 里非 None 键（None = 未给 → 落回各分支 dataclass 默认；两边 model_path/lr 默认不同必须经 None 回落保留；`is not None` 判断使 `--seed 0` 等 falsy 值正确穿透）。实测双分支覆盖 + 回落全过

### 18.2 日志系统（print → logging）

- 全仓 33 处 print → 模块级 `logger = logging.getLogger(__name__)`；三个入口 `main()` 加 `logging.basicConfig(level=INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")`（不配 handler 则 INFO 全静默——lastResort 只放 WARNING+）
- 消息纪律（用户要求）：**日志里无 `[tag]` 前缀、无 JSON**——来源由 `%(name)s` 承担；GRPO 4 处 skip/guard（scorer send/recv 失败、no_trainable_group、all_losses_nonfinite）= `warning` 级人话行；per-step monitor 行 = `step N | M groups, K skipped | loss ... (policy ..., kl ...) | grad ... lr ... | R ... | r_sv ... | sim ... | Ds` 一行
- **文件格式不动**：`monitor.jsonl`（grpo/sft）与 asset.jsonl 写入保持 JSON 原文（pipeline.py 两处 `print(..., file=...)` 是写文件不是日志，未转换）；`flush=True` 参数随 print 消失（StreamHandler 每条自动 flush）
- 零改动区：tqdm 4 循环（无 print 在循环体内）、vendored 三棵树

### 18.3 sys.path hack 清除 + 核心库瘦身

- **模块解析本就不靠 hack**：三个 worker 的 pyproject 都是 `uv_build` 后端 + `module-name` 配置，`uv sync` 默认把项目自身 editable 装进自己的 .venv（实测 `import scorer/preprocess/trainer` 直指 `src/`，无 path 介入）；scorer/preprocess 入口的两行 `sys.path.insert` 是历史残留，已删（连带仅为其存在的 `import sys`/`Path` 导入）。`[project.scripts]` 未加——bin 壳与解析无关，AGENTS 命令惯例统一 `.venv/bin/python workers/xxx/main.py`
- **`src/qwen3_tts_post_training/system.py`**（新）：`peak_rss_mb()`（getrusage ru_maxrss）/`current_rss_mb()`（/proc/self/statm，OSError → -1）/`gpu_allocated_mb(device)`/`gpu_reserved_mb(device)`（torch 分配器视角，round 1 位）——数值语义与原内联表达式逐字节一致；调用点收敛：grpo/sft monitor + scorer `ScoreResponse.rss_mb`，三个文件的 `import resource` 全清
- **`train/grpo.py` → `workers/trainer/src/trainer/grpo/grpo.py`**（`git mv`，唯一消费方 loop.py）：核心库瘦身为真正三方共享的 `reward/` + `client/` + `system.py`
