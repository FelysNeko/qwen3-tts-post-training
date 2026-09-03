# PROJECT_STATUS.md — Qwen3-TTS 后训练（Cyrene GRPO）

> 本文件为项目当前状态、迁移记录、标定数字复核与已知问题清单。
> 设计真相源仍在 `../playground/SV_REWARD_FINDINGS.md`，本文档只记录"当前机器上发生过什么、已验证什么、还差什么"。

最后更新：2026-09-02（§42 终裁：用户听感定基座 = runs/b_ep1，GRPO 就绪）

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

- trainer 的标定入口 = **cache 目录本身**（`--cache-dir`/`--namespace`，shared argparse，SFT/GRPO 同参；`--namespace {lang}` 简写 = `<repo>/.cache/{lang}`，与 preprocess 默认 cache-root 对齐）：`reward_config_from_metrics` 读 `<cache-dir>/metrics.json` 的 `sim.mean/std` → `RewardConfig.sv_center/sv_scale`；`load_centroid` 读 sibling `centroid.npy` → float32 renormalize（复刻旧 set_ref 配方）；`mos_tau` 维持 2.5（mos 统计只记录，待 gate 规则）。**`--metrics-path` 已删（2026-08-30）**：metrics.json 永远与训练数据同 cache，本地 cache + 外部 metrics 的跨池错配无人能 gating——旧旗标（§16.9 起必填、当日改 `--cache-dir` 可代）演进终结于单入口不变量。同日 `RewardConfig.sv_center/sv_scale` 默认值删除（0.8585/0.0966 playground 对退役）：两字段必填、`reward_v3` 的 cfg 必传——静默回退会让外来/过期标定看似健康；标定语义边界见「池几何 → metrics，策略/权重/护栏 → 代码默认值」
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

- `model.py`：`SftTrainerModel(ModelWrapper)`——**全量 FT**（talker 全栈含 code_predictor/MTP 头，官方语义；GRPO 冻 predictor 的姿态在此不适用）；冻结仅 `speaker_encoder`（输出 detach 进 slot 6，天然无梯度）。`speech_tokenizer` 是普通包装对象（非 nn.Module），参数本就不在模型树。**base-only 硬化（2026-08-30）**：曾有的 `load_base_speaker_encoder`（custom_voice 起点从 base safetensors 惰性借 76 张量）整体删除 + `--base-model-path` 旗删——用户既定姿态「SFT 只允许从 base 出发」，`run_sft` 加载后 assert in-model speaker_encoder，custom_voice 起点直接拒绝（续练走 GRPO，不是 SFT）
- speaker 条件：**cache metrics.json 的 `medoid` 条目**（池 E2V2 max-mean-pairwise clip，`--speaker-audio` 可覆盖）→ soundfile 读（stereo→mono）→ librosa 24k 重采样 → 模型自带 `extract_speaker_embedding` → `[hidden]` 向量广播进 slot 6（teacher_forcing 的 spk 查表同路径），启动时提一次全程复用；官方逐条 `ref_mels→speaker_encoder().detach()` 的单说话人退化版。选取在 E2V2（听得到通道/质量差异的空间）、嵌入用模型自带 encoder（E2V2 192d 进不了 slot）——决策与数字见 §19.4
- `loop.py`：`CacheLayout.load_sft_dataset`（core lib cache.py；asset 行 + codes npy → (text, [T,16] long)，缺 npy 即 assert）→ 每 epoch 种子洗牌切片 → `collate → teacher_forcing → CE`。**loss 单 shift 配对**：`CE(talker_logits, codec_0_labels[:,1:], ignore_index=-100) + 0.3·CE(sub_talker_logits, talker_codec_ids[:,1:])`——官方 `sft_12hz.py` 的 double-shift bug（`inputs_embeds[:, :-1]` + 内部 HF CE 再 shift 一次 → logits@p-2 vs 标签@p+1；`hidden_states[codec_mask[:, :-1]]` 选位 p 的 hidden 泄漏目标）不复现，`outputs.loss` 不可用
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

## 19. 原生 speaker encoder A/B：决策 = E2V2 维持不变（2026-08-30）

**背景**：`qwen3_tts` 自带 ECAPA-TDNN speaker encoder（`Qwen3TTSSpeakerEncoder`，24k mel128 → ASP+fc，输出不归一化；CustomVoice ckpt 不带权重，当时从 `Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-Base` 借——探针期复用了 SFT 的 `load_base_speaker_encoder`，该借用路径已于 2026-08-30 随 base-only 硬化删除，见 §17.1）。问题 = 能否替换外聘 ERes2NetV2 当 r_sv。**此前无任何决策记录**（playground SV bake-off §一 只测了 E2V2/CAM++/ECAPA，原生候选从未入册），本轮补测后**用户拍板：E2V2 不变**。探针 = `probes/native_embedding/`（脚本 + report*.json + 负样本集 + rollout 音频，gitignore 覆盖）。

### 19.1 GT 池几何（1779 条 enhanced wav，24k，fp32，跨进程位等）

* 提取：mel 配方逐行复刻 `modeling_qwen3_tts.extract_speaker_embedding`（n_fft 1024/hop 256/win 1024/fmax 12k），48k→24k librosa，全池 19s。0.6B(enc 1024)/1.7B(enc 2048) 行为几乎相同（σ 0.0124 vs 0.0126），尺寸无谓
* sim→质心：E2V2 **0.8709±0.0880** vs 原生 **0.9879±0.0124**（σ 紧 7×）；pairwise 0.758±0.117 vs **0.976±0.018**——原生把全池看成一个点（提纯的"说话人身份"提取器，丢掉的正是 r_sv 喂养的通道/质量/内容变化）
* 逐 clip 相关：P 0.871 / S 0.807；尾部重叠 bottom50 37/50（坏 clip 共识）、top50 17/50（好 take 上失明）
* ⚠️ 首跑撞上一次瞬态 GPU 故障（单 clip 非有限值 → centroid assert；守卫已加：非有限行诊断 + 只用有限行算质心；复跑两次位等 0.0）

### 19.2 rollout 组内 z-spread（16 组 × 8 = 128 takes，PhiLia093=1.7B custom_voice，种子 = 生产约定 `seed*1000003+step*1009+gi` step0）

* pooled within-group z_std：E2V2 0.296 vs 原生 0.190（**ratio 0.64，不是池上 σ 比的 1/7**）——池几何的悲观预判在 rollout 上没有全额兑现
* 分组型看才是重点：健康组（12/16）原生 ≈ E2V2 的 1/3（多为噪声压缩，MD §10 附3：健康 take 的 E2V2 散布大半是 take 间噪声）；**真退化组（g08/g10 OOD 崩坏）追平甚至反超**（0.462 vs 0.421）；**绕口令组 g12 sigmoid 地板反转**：E2V2 sim 塌到 0.66（z -2.4）→ r 贴地 → adv_std 0.018 < 原生 0.035（原生停在响应区）
* 全 takes spearman 0.791。结论从"基本不能换"上调为"值得 A/B"，引出负样本判别

### 19.3 负样本判别（17 条五轴，vs 各自池标定 z；report_negatives.json）

* 构成：官方 bucket 2 条（clone.wav/tokenizer_demo_1）+ LibriSpeech 6 说话人 + 同名台词 EN/JA/KO 三位 CV（**是不同配音演员**，不是同人错语种——初版标签已改 `other_va`）+ **域内中文轴**（官方 0.6B-CustomVoice 内置音色 vivian/serena/uncle_fu × 2 池文本，`builtin_zh.py` 现场生成，HF cache +2GB）
* 汇总：E2V2 均值 z **-6.01**（最差 clip -2.71）vs 原生 **-3.81**（最差 clip **-1.09**）；**双方 100% 分离负样本 vs 健康 take**
* **域内中文轴（公平对照，回应"E2V2 是 zh-cn 专精"的质疑）**：E2V2 对中文内置音色照旧 -5.1~-7.6σ（分离力是真实判别力，非域外推力）；原生对女声（vivian/serena）只有 -1.1~-1.5σ，与它健康 rollout 下界 -0.67 仅隔 0.4σ（serena_1 为全场最弱负样本）——**"干净语音高原"**：native 把"干净专业语音"与"目标说话人"混为一谈，男声（-3.4）才拉开
* qwen.ai 博客音频不可得：纯 CSR SPA，`/api/v2/article/retrieval` 服务端恒空列表，OSS 直链/列表无权限；负样本以等价素材重组

### 19.4 决策与定位

* **r_sv 主驱动维持 ERes2NetV2 不变**（三重验证：池几何 + rollout z-spread + 域内/域外负样本判别）
* 原生编码器定位：不采用；至多当"崩坏护栏/交叉监控"，且相对 CAM++ 无明显优势（信号 87% 与 E2V2 冗余、健康组第二意见幅度小）
* **SFT 条件向量随之定稿（2026-08-30 固化进管线）**：选 medoid **在 E2V2**（唯一听得到通道/质量/韵律差异的空间；原生空间 pairwise 0.976 排名压在窄带里，选 clip 分辨率不足），嵌入**用原生 encoder**（E2V2 192d 进不了 talker slot）——E2V2-medoid 那条 clip 的原生嵌入 vs 原生 centroid 的 cos = 0.9957/0.9951（0.6B/1.7B），数值上是同一向量，选真实 clip 取其 on-manifold + 可听审计 + 与 MD 旧池 medoid 连续（当前增强池 medoid 仍是 `side4_shitang_cyrene_109_f`，mean pairwise 0.8437 / 旧 0.8284）。落地：finalize 写 `medoid` 进 metrics.json（mean-pairwise argmax，空池 assert、n=1 退化为该 clip；mean_pairwise 数值不入册——池内聚度已由 sim 统计承载，无消费者）、`cache.CacheLayout.speaker_ref` 解析 sibling enhanced wav、SFT `--speaker-audio` 变可选（默认 = cache medoid，显式传入仍覆盖）；probe 32 项（medoid 手算参考 + 漂移刷新 + missing-key assert）。同日 `reward/metrics.py` 整体并入 `cache.py` 作 `CacheLayout` 方法（reward_config/load_centroid/speaker_ref + 每名 artifact 路径助手；preprocess `Config` 的 6 个重复 property 换成 `layout` 委托——layout 知识单源化，消费方一律复用）
* 遗留数据点（不阻塞）：原生编码器跨进程位等（E2V2 同）；`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` 已入 HF cache（可删可留）；torchaudio 无 sox 仅启动警告无害

## 20. PhiLia 解剖：谱分析 + 过拟合调查 + SFT/GRPO 策略推演（2026-08-30）

动机：用户连续追问「漂移这么大是不是过拟合 / 低秩吗 / epoch 降到 1 行不行 / OOD 用非原配女声冻结 subtalker / 全靠 GRPO 拟合 + GT 假装 rollout」。本轮全部用测量回答，暂无代码决策。工件（临时）：`/tmp/opencode/sv_spectrum.py` + `sv_report.json`（267 矩阵 SVD）、`/tmp/opencode/rollout_cer.py` + `rollout_cer.json`（128 rollout 的 ASR CER）、PhiLia 自带 `logging.jsonl`（167 条训练日志）。

### 20.1 谱分析：ΔW（PhiLia − 1.7B-Base）非低秩

- 分母：404 shared 键，267 变化、59 位等（多为 norm），shape 无 mismatch，base_only = `speaker_encoder.*`（base 带原生 encoder、PhiLia 不带——与结构认知一致）
- **结论：PhiLia 的微调更新是满秩对象**。ΔW r99 中位 919–1939（≈矩阵维度，effectively full-rank）；ΔW e@16 中位仅 4.6–13%（mlp 最扁平 4.6%，cp.lm_head 最高 13.1%）；ΔW stable rank 30–160；σ 谱平坦无主导方向（`layers.14.mlp.down_proj` σ₁=0.0099 → σ₈≈0.0052，e@512 才 51.5%）。W_philia 自身 stable rank 8.6–186、r99 620–1880（预训练权重典型重尾）
- 含义：**LoRA r=16 无法复现 PhiLia 的 SFT 漂移**（抓 ~5–11% 更新能量，复现需 rank ≈ r99 即 effectively full-rank）——反过来验证官方 SFT 全量 FT 的必然性；**不直接否定 GRPO 的 r=16**（RL 微扰与 SFT 累积漂移不是同类对象，待真实 GRPO run 的曲线裁决：reward 仍涨而 ‖B@A‖_F/‖W‖_F 饱和才是 rank 瓶颈证据）

### 20.2 漂移幅度 = Adam 天花板，不是异常

- 828 步 × lr（cosine 峰 ~4.96e-6 → 2e-6）≈ 每参数累积位移上限 3.2e-3（Adam 每步每参数 ≤ ±lr，m/√v 归一）；实测 maxabs/ceiling 中位 **0.30**、236/267 键 >0.25、mlp gate/up 峰值 **0.83**——梯度方向高度相干（单说话人单域两遍语料，batch 间不抵消，位移 ×steps 线性爬升而非 ×√steps 随机游走）
- 逐元素 RMS 中位仅 ~3.2e-5：每个分量挪得小，但 Adam 逐元素归一化让 12.6M 分量**各自独立**成高维更新场——「大」是错觉，「散（高秩）」是本质
- 例外识别：`codec_embedding.weight` maxabs/ceiling = 4882×——非训练漂移，是导出手术把 speaker 向量写进 slot 3000 行的产物
- TTS 微调漂移大 vs LLM 直觉的成因：初始 CE 3.30 nats（base 对该声音真无知，非 LLM 式近优点微调）+ 数据零多样性（梯度不相消）+ 说话人身份是散布全解码路径的连续细节（slot 向量只是钥匙，talker 全部权重是「钥匙→声学」映射）+ 双 AR 传导（talker 隐状态动 → predictor 输入分布动，PhiLia 里 cp 5 层 + 15 lm_heads 全动）

### 20.3 过拟合三线裁决 + 「续写」真因（用户耳朵纠偏）

- **训练线**：loss 中段平台（后半程 50 步均值仅降 0.012）、token_acc 稳 0.71（epoch 2 见重复数据无冲高 = 无帧级记忆化）、grad_norm 15–25 无塌缩
- **泛化线**：128 条 novel-text rollout（16 个未见 prompt × 8，scorer ASR 同标定）——叙事组 CER g6=**0.000**、g5=0.004、g15=0.019、g4=0.029，**低于人声语料基线 0.058**（同 ASR 同域）；文本跟随完好
- **高 CER 组解构**：g8（0.62，拟声「齁噢噢」）、g12/g14（0.26–0.27，绕口令）属 ASR 困难材料；g3（0.62）音长 25.5s vs 文本 ~10s——最初误判「即兴续写」，**用户听后纠偏：是退化生成**——尾句短语复读循环（「就在梦里…你也会回来的，对不对？」×3、prompt 本体重来一次）+ 背诵作者完整训练集的台词（银河猫猫侠 idle 对白；grep 证实全部不在我们 1779 条 cache——作者语料是我们子集的超集）
- **机制**：裸文本 prompt 无边界 → 无 EOS 决策 → AR 越过文本后落回高先验已背材料 + 短语循环；rollout 契约 pin `repetition_penalty=None`（logprob 重建无状态所需）+ T0.9/top-k50 放大循环固化。g6 等叙事句精确停住（6.3–7.0s）证明「读完就停」的能力在，是**边界判定**问题不是能力损伤
- **结论**：漂移大 ≠ 过拟合；内容级记忆存在但只在无边界提示下暴露（CE 平台 ≠ 内容没背下，2 epoch 足够背「说了什么」）；对 RL 管线 = runaway/needs_resample 样本来源，对 serving = 必须官方 prompt 模板 + rep_penalty 1.05
- 音频证据：`probes/native_embedding/rollouts/`（`g{gi}_{k}.wav` 无补零；g3_3/g3_0 25.5s、g6_0 6.3s 对照）

### 20.4 1.7B 全量 SFT 显存账（本机）

- 参数量实测：0.6B = 0.91B/1.83GB；1.7B 系 = 1.92B/3.83GB（bf16）。纯 bf16 训练（torch AdamW 状态继承参数 dtype，无 fp32 master）固定开销 = weights+grads+m+v = 4×参数量×2B：0.6B = **6.77 GiB**，1.7B = **14.31 GiB**
- **cuda:0（12G）：1.7B 全量 FT 不可能**（固定 14.31 > 可用 ~11.5）；**cuda:1（16G）：B1+grad-accum 可行但薄**（可用 ~15.5 GiB，14.31 + tokenizer/speaker_encoder 驻留 + B1 激活（hidden 2048、T~500 很小）≈ 14.8–15.3，余量 ~1 GiB，碎片化是主要敌人）；**0.6B 宽裕**（余量近半）
- 存在性证明：PhiLia `logging.jsonl` `memory(GiB)=15.96` @ B4——同配方（纯 bf16 + torch AdamW）在更大卡上跑通；其 bf16 Adam 收敛曲线健康 = 该规模实证可行。ckpt 落盘先 `.to("cpu")` 无 GPU 尖峰
- 待办：3 分钟探针（加载 1.7B SftTrainerModel → AdamW → 一次 teacher-forcing fwd/bwd @ B1 → 读 `max_memory_allocated`）把算术变测量值；bitsandbytes 未装（8-bit AdamW 可再省 ~6 GiB，连 12G 卡都能跑，属后备）
- **决策点**：1-epoch A/B 用 0.6B 先探路还是直接 1.7B？

### 20.5 三条悬而未决的路线（全部待拍板，未写码）

1. **epoch 1 A/B**：epoch 2 几乎零收益（loss −0.012）且有内容记忆代价 → 1 epoch 值得试；但 epoch 数与 cosine 绑定（414 步天花板减半，音色特化充足性未知，E2V2 sim 才测得出）；lr 要有意识定——PhiLia 实证峰 ~5e-6 vs 我们继承官方 2e-5（1 epoch 下位移上限 = PhiLia 总漂移 2.6 倍，激进）。跑法：只跑 1-epoch 臂（PhiLia = 2-epoch 臂现成），评估链 = novel-text CER（叙事组）+ t_max/失控率 + 背诵检测器（ASR 转写与语料 n-gram 重叠）+ E2V2 sim + 耳朵
2. **OOD（非原配女声 + 冻结 subtalker 只调 talker）**：机制一行可行（GRPO 有冻结先例），但 (a) predictor「冻结 ≠ 不受影响」（talker 隐状态移动改变其输入分布），(b) talker 出的 codebook 0 携带韵律 → 代打数据会污染（同女声或可接受），(c) **目标歧义未解**：(i) 跨 vec 迁移给 Cyrene（「vec 无关内容能力」假设未验证）vs (ii) 造辅助音色（干净但产物是另一个声音）；更轻的替代：官方模板 + rep_penalty 先修边界病，base 模型路由 OOD 内容
3. **GRPO + GT**：「GT 混进 rollout group 假装是其 rollout」机制全通（codes 无辜、scorer 走 enhanced wav 路径、GT 长度 < token_budget；centroid 恰为 GT 池均值 → GT 的 r_sv ≈ sigmoid(0)=0.5，不会碾压 group——标定巧合反而利好稳定），**但「全靠 GRPO 从头拟合」不成立**（advantage 缩水的模仿信号比 CE 弱几个量级 + off-policy 偏差 + 把背诵病搬进 RL）。同意图的干净形式 = **混合目标 `L = L_GRPO + λ·L_CE(GT codes)`**：有正统先例（InstructGPT PPO 混预训练 CE 防漂移 λ≈9.75、RRHF reward 加权 NLL、RAFT/ReST-EM、SLiC-HF），现代 RL 框架的 SFT-anchor 标准开关；我们数据切分白送「CE 锚在语料文本、RL 探索在 pool 文本」的分离；λ 按梯度范数配平起步，背诵检测器盯梢；predictor 冻结与否与此正交

## 21. cache_dir/namespace 解耦：trainer 双字段 + preprocess 免疫（2026-08-30）

- **动因（用户纠偏）**：旧 trainer CLI 把 `--namespace` 当固定根下的"简写"、与 `--cache-dir`（池完整路径）互斥；config 只持有解析后的字符串，表达不了"哪个池"。第一版实现（cache.py 里的互斥 resolver）当场被否——**cache_dir 与 namespace 是正交的**：cache_dir = `.cache` 根**位置**，namespace = 池名，池 = `{cache_dir}/{namespace}`，可独立组合。resolver 已删，cache.py 保持纯 layout。
- **trainer 侧（解耦落地，用户三轮纠偏后定稿）**：`TrainConfig` / `SftConfig` 持 `cache_dir`（根，默认 `<repo>/.cache`）+ `namespace` 双字段；**`namespace` 无默认必填（`namespace: str` 置于类首，类型即约束）**，无 accessor——`run_grpo`/`run_sft` 调用点直拼 `CacheLayout(Path(cfg.cache_dir) / cfg.namespace)` 一次。CLI：`--namespace` 用 argparse 原生 `required=True`，kwargs 直灌 `SftConfig(**overrides)` / `TrainConfig(**overrides)`（`replace` 桥与 `cfg = cfg or Config()` 默认实例随之删除，存在性不前移——池缺失由 `run_*` 内 layout 加载自然报错）；`--cache-dir` 语义为 cache 根。
- **preprocess 侧（两轮纠偏后定稿）**：**namespace 不是 preprocess 的概念**——语料是任意路径的输入目录，池名 = `{dataset.name}` 是固有行为，无需字段承载。CLI = `--dataset`（必填）+ `--cache-dir`（根语义，由 `--cache-root` 更名，默认 `<repo>/.cache`）；`run_pipeline(dataset, cache_root, ...)` 内部 `Config(cache_dir=cache_root / dataset.name)`，preprocess `Config` 与全包代码不出现 namespace 字样。
- **验证**：ruff 全绿；probe_preprocess PASS；config 冒烟 = `SftConfig(namespace="Chinese(PRC)").layout` 解析到真实池 + `reward_config` 0.8709/0.0880、`TrainConfig(cache_dir="/x", namespace=...)` 拼接正确、缺 namespace 双 config assert；CLI 错误路径（缺 `--namespace` / 池不存在）实测报错；preprocess 真实路径冒烟 + `--help`。

## 22. 全链路首跑、评估体系与消融定案（2026-08-31）

当天主线：SFT→GRPO 全链路跑通并四轮评估 → sub_weight 剂量实验 → 组件冻结消融（两轮设计）→ MOS 跨进程事故与协议修复 → runs/ 清理 → GRPO 池重构 810。两次死机（WSL2 dxg 幽灵显存累积，`wsl --shutdown` 唯一清法；驱动 616.56 + 内核 6.18.40.1）均非代码问题。

### 22.1 全链路首跑

- **SFT-1ep**（`runs/sft_v1/export`，cyrene @ slot 3000）：B4/accum1、lr 5e-6、warmup 20、wd 0.01，0.3s/步，loss 2.50→2.08（sem 1.09→0.81）。配方为用户拍板（1 epoch 来自 §20.5-1 的推演，峰值 lr 5e-6 对齐 PhiLia 实证）。
- **GRPO v5**（100 步，graphed，8×8=64 rollout/步，lr 1e-6，budget 448）：KL p50 0.0086（单步尖峰即恢复）、r_sv 0.583→0.596、无 runaway。**micro-batch**（`--logprob-micro 4`：logprob 分块 + 逐组切片 backward，advantage 全组算一次再切）16G 显存下跑通；数学等价 probe PASS（W 精确相等、loss 差 0.1%、grad 差 1%），完整 diff 存 `probes/microbatch.patch` + `probes/microbatch_design.md`。
- **操作协议入 AGENTS.md**：新实验必须先重启 scorer（fresh socket）等 `connected` 再起 trainer——scorer 跨 trainer 重启存活会产生 stale 响应（"scorer id mismatch"组静默跳过）。robustness try/except 按用户裁定 revert（scorer 读不到 wav 裸崩为已知边界），`empty_cache` 保留。
- **设备教训**：nvidia-smi 枚举与 CUDA 运行时索引**相反**（cuda:0=4070S 12G、cuda:1=5070Ti 16G），定位设备只信 CUDA。

### 22.2 评估体系与四轮结果（12 in-sample / 8 OOD / 5 极 OOD / 4×8 稳定性）

| 集 | SFT-0.3 | GRPO | 结论 |
|---|---|---|---|
| in-sample 12 | 0.8761/8.54%/2.766* | **0.8937**/6.51%/2.822* | GRPO 三指标全胜（*MOS 为修正前列，见 §22.3） |
| OOD-mild 8 | 0.7910/5.05% | 0.8149/6.12% | sim 泛化 +0.024，CER 微亏，MOS 平 |
| 极 OOD 5 | 0.8763/28.49% | 0.8731/26.72% | 增益归零 = 训练分布收窄 |
| 稳定性 4×8 | dur std 13.6s | **0.9s** | GRPO 消灭 runaway 尾巴（15×）；sim std 0.0385→0.0177 |

配套：SFT-2ep 消融（dur std 减半至 6.3s 但仍有 ~70s 残余跑飞；GRPO 对 2ep 仍全面胜）→「欠拟合 EOS」假设部分成立但不改变 GRPO 价值。产物全在 `runs/eval/`（结构化分区 + README 索引/对照表）。

### 22.3 MOS 跨进程不可比事故（协议修复）

用户质疑 SFT-0.3 基线「不可能差这么多」→ 复测（同代码路径重生成 + 逐字节 wav 对比）揪出：**sim/CER 12/12 逐位一致，MOS 12/12 全部漂移（−0.31~+0.24）**——wav 完全相同而分数变化 = scorer 进程重启的 GPU kernel 非确定性（AGENTS 已载的 within-lifetime-only 位稳定）。当天反复死机重启导致各臂 MOS 分属 4 个 scorer 进程，**此前所有跨进程 MOS 对照无效**。修复：全部臂的存量 wav 在单 scorer 进程统一重打（`runs/eval/mos/rescore_mos.json` = **权威 MOS**）。修正后排位：FRZ-TALK 3.085 > SFT-0.1 2.895 > FRZ-SUB 2.863 > GRPO 2.822 > SFT-0.3 2.787（GRPO 的 MOS 实为微亏，它买的是 sim/CER）。

### 22.4 sub_weight 剂量实验：0.3→0.1 近砍半 CER

用户拍板「更简单：常量 0.1，不做 decay 不冻结」→ `SftConfig.sub_weight` + `--sub-weight`（约 3 行，loss = sem + w·sub）。结果（sw01，全量 FT）：in-sample **0.8795/4.75%/2.895**（CER 8.54→4.75%，sim/MOS 同涨）；hard **24.9%** 同时好于 SFT-0.3（28.5%）与 GRPO（26.7%）。剂量-效应干净：CER FRZ(2.2) < 0.1(4.8) < GRPO(6.5) < 0.3(8.5)——subtalker 训练信号越强文本跟随越差。decay（线性 0.3→0）已设计未实施，是剂量曲线的自然延伸。

### 22.5 组件冻结消融（第一版，后被重构）

`--freeze sub|talker`（v1）：FRZ-SUB（只训 talker 侧 764.2M）0.8831/2.18%/2.852；FRZ-TALK（predictor-only 141.6M）0.8576/2.15%/3.085、hard sim 崩至 0.7594 而 MOS 2.471 双集第一——**音色/文本=talker、声学质量=predictor 的二分**首次显形，与 GRPO 冻结 predictor 的姿态自洽；Fish S2 §4.3 对照（同 Dr.GRPO/γ·fast/rsLoRA-MLP-only/Schulman KL/异步打分；异于我们多 DAPO clip、predictor 冻结、他们 ASR 置信度惩罚）已讲解归档。

### 22.6 对称消融重构（用户三轮纠偏后定稿）

用户连续纠偏：frz 逻辑补集写法 → 「为什么不训练 text_projection（混淆变量！）」→ **亲手重写** `SftTrainerModel` 为可组合集合 API：`freeze: list[str] ⊆ {subtalker, talker, text}`，text_projection（共享入口，codebook embedding 不经它）**双臂均训**，唯一变量 = 哪个生成栈学习。落地修复两处 `parameters()` 生成器直接调 `requires_grad_` 的 AttributeError（用户原稿 bug）；CLI `nargs="+"` 空格分隔（用户否决逗号 lambda + 防呆校验，均已从简）；冒烟 5 配置 PASS（905.8 / 764.2 / 147.9 / 141.6 / 757.9M）。**新 `{"talker"}` 臂**（sft_frz_talkhead，从 base）：in-sample 0.8660/**1.77%**/3.090、hard 0.8109/**18.4%**/2.230——**CER 双集全场最佳**（超过 GRPO），MOS 与 predictor-only 持平（text_projection 对 MOS 中性），sim 代价 hard 放大（0.858 vs 0.811）。三线证据（旧消融/剂量曲线/对称对）收敛：**talker 侧承载音色，predictor 侧承载声学质量+文本跟随**。

### 22.7 runs/ 清理 + 文档

- eval/ 分区：`insample/{sft_vs_grpo,frz,talker}`、`hard/*`、`ood8/`、`stability/`、`verify/`、`mos/`（权威）、`logs/` + README（口径、对照表、清理记录）。
- 清理 34.4G→15G：grpo_v1-v4 整删、**grpo_v5 全部 ckpt 含 latest 删除（最终 GRPO 策略已无 ckpt，复评需重训 ~2h）**、现役 SFT 臂 latest.pt 删（不可 --resume，重训 ~4min）、sft_ep2/sft_frz_talk 的 export 暂留（用户指示）。数字结论全在 eval/ 报告，无信息损失。
- 根 README 补 Setup + Basic usage（四步 CLI，对照三 worker 实际 argparse）。

### 22.8 GRPO 池重构：100 → 800 训练 + 10 留出

用户发现 v5 池仅 100 条（8×100 步 = 每条平均 ~8 次访问）。澄清：RL 复用 prompt 合法（每次访问都是当时策略下的全新 rollout，复用的是 prompt 不是 rollout）；但池多样性是真泛化轴（§22.2 极 OOD 归零佐证）。新池：源 `../delta-me13/corpora/sft/cyrene/chs.jsonl`（279 对话 → 1647 assistant 回复）拆行 + 去纯标点 + 去 SFT 整段重复（1542）→ 1557 去重（与旧池构建数字一致，管线保真）→ `MIN_LEN=10` 过滤 → 1371 → **最长前 810**（用户拍板，弃随机方案）→ 800 训练 `probes/grpo.txt`（21/31/119 字）+ **10 留出 `probes/insample.txt`**（22/30/48），种子 20260831。**协议升级：留出集从未入训——in-domain 泛化与池内记忆首次可分离测量**（旧"in-sample"实为从训练池抽样，被污染）。

### 22.9 未决 / 下一步

1. **扩池 GRPO 重跑**（v6）：`--text-pool-path probes/grpo.txt`，验证打 `probes/insample.txt`——检验 OOD 泛化是否随池多样性改善（需先重训 GRPO ~2h）。
2. **decay**（0.3→0 线性）已设计未实施；剂量曲线指向 predictor 干扰可进一步压低。
3. CE 锚混合目标（§20.5-3：`L = L_GRPO + λ·L_CE(GT)`）待拍板。
4. FRZ 系数字的 MOS 均以 `mos/rescore_mos.json` 为准；新臂评估必须与参照臂同 scorer 进程打分。

## 23. 分工四臂 v2：GRPO 基座与 OOD teaching 路线定标（2026-08-31）

用户重做四臂对称消融（决策导向：选 GRPO 基座 + 为「非本人 OOD 音频 teaching forcing 教特殊场景」路线定方向）。旧臂 export 已被清理，本轮全从 base 重训。**评估协议升级：insample = `probes/insample.txt` 8 条从未入训的留出文本（池内记忆与泛化首次分离）；hard = `probes/hard.txt` 8 条（4 拟声 + 4 绕口令）；64 wav 同一 scorer 进程打分**。完整报告 = `runs/abl2_eval/report.md`。

### 23.1 设置与主结果

- 四臂（base ckpt、lr 5e-6、1ep、B4/accum1、warmup 20、同 seed 0）：`base`（全量 905.8M，sub 0.3）/ `sw002`（全量，sub 0.02）/ `subfrz`（冻 code_predictor，764.2M）/ `talkfrz`（冻 talker.model+codec_head，147.9M）；text_projection 双冻结臂均训（唯一变量 = 哪个生成栈学习）。
- 主表（mean）：insample sim/CER/MOS/dur → base **0.8686**/2.74/2.869/8.1s；sw002 0.8531/3.23/2.949/7.2s；subfrz 0.8626/4.19/**3.101**/6.9s；talkfrz 0.8423/**1.78**/2.833/6.8s。hard → base 0.7748/19.2/2.327；sw002 0.7913/27.5/2.252；subfrz 0.7915/17.4/**2.361**；talkfrz 0.7451/**12.3**/2.212。

### 23.2 分工定案（v1 消融强化版）

1. **文本跟随 + 极端文本鲁棒性 = predictor 侧**：talkfrz 全胜——hard 剪 runaway（每臂恰 1 条 77-79s，臂间无差）后 CER **9.1%** vs 其余 19-31%；拟声组 21.7% vs 29-40%；绕口令组 2.9% vs 6-15%。
2. **音色 = talker 侧**：talkfrz 双集 sim 垫底且 hard 崩至 0.745——冻 talker 后音色只剩 slot 向量 + text_projection 间接通路。
3. **MOS = predictor 健康度**：subfrz（predictor 零接触）双集第一；talkfrz（predictor 被单域 CE 重训）垫底。
4. **语速（风格代理）被全参微调拉长**：全量臂 7.25-8.11s vs 冻结臂 6.8s（同文本同种子）。sub_weight 剂量 0.3→0.02 对 CER **非单调**（hard 19.2→27.5 反升）——§22.4 的"越小越好"外推不成立（n=8，弱结论）。
5. 训练末态 sem：base/sw002/subfrz ≈ 0.81，talkfrz 1.064（sem 头冻在 codec_head，只能间接拖动）。

### 23.3 两个决策

- **GRPO 基座 → subfrz 型 export**：GRPO 可训练面（talker MLP r=16）要求基座 talker 已充分适配（talkfrz 的 talker 是 base 原重，r=16 补不回全秩漂移，§20.1）；GRPO 冻结面（predictor）应交最健康者（subfrz 零接触、MOS 第一）。备选 = base 臂（官方全参路线，insample CER 第二）。
- **OOD teaching-forcing → 两段式**：第一段 subfrz 型适配 talker（sim/MOS 双高），第二段冻 talker 只训 predictor(+text_projection) 吃 OOD 教学——音色由已适配的冻结尾保护。talkfrz 臂即该路线彩排：predictor 训练交付全部极端文本收益，hard-sim 0.745 的崩塌是"冻了 talker 但 talker 没适配过"的反面教材。

### 23.4 环境与杂项

- 旧 scorer 的 `--sv-dir /tmp/opencode/sv` 目录已不存在；核实 `ensure_sv_ckpt` 语义 = 目录缺失自动走 modelscope fetch，原命令可安全重启。SFT/评估生成全程不依赖 scorer，fresh scorer 只需在打分前 connected。
- 本轮 eval 进程 `c.close()` 后挂住不退出（DONE 已打、report 已写），kill -9 收尾；下次脚本在 close 后加显式 `sys.exit`/context 处理。
- `runs/abl2_*/latest.pt`（全参臂 ~7.4G×3）待用户处置；export 各 ~1.8-2.4G 保留。

## 24. 8-take 重评估 + sub_weight 证伪 + embedding 考古 + text_embedding 冻结定案（2026-08-31）

### 24.1 8-take 重评估（部分完成，scorer 队列被用户取消）

- **512 wav 已全部生成**（4 臂 × [8 insample + 4 hard 绕口令 + 4 ood 拟声] × 8 take；graphed B8、T0.9/topk50、seed 1234+i 组内共享流、budget 1024；`runs/abl2_eval8/{臂}/{集}/{prompt:02d}_{take}.wav`）。CER/MOS 未打：runaway 长剪（77-80s，顶满 budget）上 ASR+MOS ~30 min/组（怪物组实测 asr 875s + mos 878s），用户裁定取消。
- **Runaway 率 = 本轮最硬发现**（dur≥60s）：ood 集 talkfrz **15/32（47%）** > sw002 11/32 > base 5/32 > **subfrz 2/32**；insample/hard 四臂均 ≤2。**EOS 判定随 predictor 训练劣化**（非单调：0.02 比 0.3 更差）。runaway 是 prompt 特异的（ood p0「齁噢快停下」p3「咕咕噜」为重灾区，绕口令几乎安全），且全部顶满 token_budget。
- **SV-only 全量打分**（`runs/abl2_eval8/sv_report.json/md`）：健康 take 的 sim 四臂同档（insample 0.84-0.88、std 全部 ~0.03——稳定性无差）；**runaway take 的 sim 崩至 0.30-0.46**（voice 随胡言乱语漂走，健康 0.84+，双峰）。§23.2 的「talkfrz hard sim 崩塌」实为 runaway 伪影——单 take 评估抽中什么全凭运气。
- 增量落盘教训：分数只在组完成后可见，中间态不可存——后续评估脚本逐组 flush + `os._exit(0)` 防 zmq 挂。

### 24.2 sub_weight 旋钮证伪：Adam 尺度不变性

- 用户问「为什么 sw002（w=0.02）和 base（w=0.3）没区别」→ 权重漂移实测：predictor 层 rel-L2 **0.39% vs 0.38%**——15 倍 loss 权重差，漂移零差。
- **机制**：predictor 参数只从 `w·g_sub` 拿梯度，Adam 更新 = `lr·m/√v` 对常数缩放不变——**loss 缩放类旋钮被优化器原样吞掉**。freeze 有效是因为断梯度（`requires_grad=False`），无量可归一。talker 侧混合梯度 `g_sem + w·g_sub` 中 g_sem 主导 + grad-clip 再归一，w 同样失效（三臂 0.21-0.23%）。
- **w=0 决定性实验**（abl2_sw000，用户设计以防代码 bug）：predictor 86 键中 **71 键（5 层 + 15 lm_head + norms）漂移 <0.001% 真零**——旋钮机械上正确；15 张 codec_embedding 表漂移 0.05-0.30% 源自 **sem 路径共享**（`modeling_qwen3_tts.py:1988`：ref 码 1-15 的输入嵌入就是 predictor 的表，sem loss 经 talker 输入路径训练它们，与 sub_weight 无关）。
- **§22.4 剂量曲线（0.3→0.1 CER 近砍半）就此作废**：既然 w=0.02 ≡ w=0.3，0.1 亦然；当时 n=12 单 take 评估，差距 = 单 take 噪声 + run 间方差。若真要调 predictor 学习量，正确旋钮 = 按参数组分 lr（Adam 更新随 lr 线性缩放）；GRPO 的 γ 同为 loss 缩放，不变性警告适用（好在 GRPO 的 predictor 本就冻结）。

### 24.3 embedding 拓扑（17 表 3 组，345.8M = 38% 参数）

| 组 | 键 | 形状 | 参数 |
|---|---|---|---|
| ① text_embedding | `talker.model.text_embedding.weight` | (151936, 2048) | 311.2M |
| ② talker 主 codec 表 | `talker.model.codec_embedding.weight` | (3072, 1024) | 3.1M |
| ③ predictor 15 表 | `code_predictor.model.codec_embedding.{0..14}` | 各 (2048, 1024) | 31.5M |

- ③ 是**共享表**：predictor 输入 + talker 输入路径（teacher forcing 的 ref 码嵌入）双用——sem/sub 两条 loss 都到得了。② row 3000 = 导出手术烤 speaker 向量处（官方同款）。text_projection（6.3M）与 codec_head（3.1M）是投影/输出头，不是查表。词表 151936 = **Qwen2 系词表**（Qwen2.5/Qwen3 沿用），TTS 仓库不附带 tokenizer；权重级同源性本地不可验证（ASR 对照无效——两边各自训过，rel-L2 1.135）。
- 1.7B 账本：text_embedding 同为 311.2M（词表/维度共享设计），③ 翻倍至 63M，embedding 合计 380.4M（19.7%）——**冻 embedding 为 1.7B 全参 SFT 省 2.28 GiB Adam 状态**（14.31 → ~12 GiB 地板，§20.4 的 16G 卡账本重算入口）。

### 24.4 官方配方考古（0.6B-Base → 官方 CustomVoice 权重漂移）

| 组件 | 官方漂移 | 裁决 |
|---|---|---|
| **① text_embedding** | **0.005%** | **官方冻结**（bf16 噪声级）——用户直觉实锤 |
| ② 主表（9 烤入行外，3063 行） | 1.395% | 训练 |
| ② 主表 9 行 | norm 9.6-10.2 | 烤 speaker 向量（官方 9 内置音色，行 2861-2878 + 3010-3066；我们的 slot 3000 恰好不在其内） |
| ③ 15 共享表 | 0.69-2.01%（mean 1.45%） | 训练 |
| talker 层 / codec_head / text_projection / predictor 层 | 0.8-1.6% | 全训（**官方 SFT 不冻 predictor**） |

官方 = 全参 FT 减 text_embedding；漂移尺度 ~1.5%（lr 2e-5 多 epoch）vs 我们 ~0.2%（5e-6 1ep），同族不同强度。

### 24.5 定案：SFT 无条件冻结 text_embedding

- `SftTrainerModel.__init__` 硬编码 `talker.model.text_embedding` 的 `requires_grad_(False)`——**无 CLI 开关**（用户裁定：信官方，不值得暴露旋钮）。docstring 记录官方漂移依据 + Adam 稀疏行论证（罕见 token 行在 Adam 下与常用行同速 sign-step）+ 显存红利（311M×3 bf16 ≈ 1.87G）。
- 冒烟 5 配置 PASS：None **594.6M** / {subtalker} 453.0M / {talker} 147.9M / {talker,text} 141.6M / {subtalker,text} 446.8M；断言 text_embedding 永不出现在 trainable。ruff 全绿。
- **旧臂语义注记**：旧 `{"talker"}` 臂因 `talker.model.parameters()` 意外已含 ① 冻结（147.9M 从未含它），而 subfrz/全量臂在训 ①——新不变量使所有配置一致，此后重训臂与 §22/§23 数字严格可比性以「① 冻结」为新基线。GRPO 不受影响（LoRA 目标本不含 embedding）。
- 未做（用户裁定不跑训练）：官方对齐臂重训对照、健康 take 的 CER/MOS 补打（fast 脚本骨架在 `/tmp/opencode/eval8_fast.py`，跳 dur≥60s）、`runs/abl2_*/latest.pt`（~22G）待处置。
- **公开脚本比对（2026-08-31 续）**：`QwenLM/Qwen3-TTS` `finetuning/sft_12hz.py` 全脚本无一处 `requires_grad_(False)`，`AdamW(qwen3tts.model.parameters())` 全量入优化器，text_embedding 在前向图中活跃——**公开脚本会训练 text_embedding，与官方 CV 产物（0.005%）矛盾 → 官方内部配方 ≠ 公开脚本**（内部多一步冻结或由另一管线产出）。我们的无条件冻结 = 对齐官方**产物**，非复刻公开脚本。附带核实：`"spk_id": {name: 3000}` 硬编码 slot 3000（官方 CV 的 9 槽来自另一多音色管线）；`loss = outputs.loss + 0.3 * sub_talker_loss` 即我们继承的 0.3 常数——在 Adam 下同被尺度不变性吞掉（§24.2 对官方自己同样成立）；speaker_encoder 未冻结但 `.detach()` 切梯度 → `grad=None` → AdamW 跳过（意外安全）；§15/§17 记录的 double-shift 与隐态选位问题原样在。
- **旧训练的 text_embedding 漂移解剖（2026-08-31 续）**：聚合 rel-L2 全训练臂 0.0201-0.0202%（sft_v1/abl2_base/sw002/sw000 逐位同档——sem 路径唯一驱动，sub 项贡献为零，Adam 不变性第四次实证；talkfrz 0.0000% 结构冻结）。行级（sft_v1）：151,936 行中仅 **4,262 行**漂移 >1e-4，与语料唯一 token 行（4,289）99.2% 重合，未用行 0.00% 位不动——「梯度只到过语料用过的行」。行相对漂移 median 0.09%（max 7.26%，某罕见 token），绝对值 ~37% Adam 天花板（方向部分相干）。即：训 text_embedding 的实际效果 = **领域剧本词表的隐空间重写**，147k 无关行零风险也无零收益。

### 24.6 官方 CV×Base 全量普查 + SVD 谱（402 共享键逐键 + 266 变更矩阵 svdvals；`probes/official_cv_census/report.json`）

- **普查：官方除 text_embedding 外无其他刻意冻结**。逐位冻结的 106 键 = 全部 RMSNorm 权重（104 层内 norm + 2 final norm）——**我们的 abl2_base 同样位冻结 130 键、交集 106**：同一批 norm 在两个独立训练里都不动 → **涌现性位冻结**（RMSNorm 梯度 < bf16 ulp，lr 2e-5 官方也不例外），非刻意设计。① text_embedding 的"0.005%"实为**管线手术**：仅 3 行被点编辑（151671-151673 = `<tts_pad>`/`<tts_text_bos>`/`<tts_text_eod>`，max-abs 9e-4），训练零接触。speaker_encoder 76 键整体不随 CV 发布（§17.1 已知）。
- **SVD：官方 ΔW 非低秩，§20.1 结论获官方产物级再证**。变更矩阵 svdvals：talker_layers r99 med 777（485-906）、e@16 med 21.8%、srank 38；pred_layers r99 820 / e@16 13.8%；pred_heads r99 769 / e@16 18.7%；codec_pred_tables r99 732 / e@16 27.2%；codec_head r99 844 / e@16 12.2%；codec_main（剔 9 烤入行）r99 880 / e@16 14.5%。最结构化的是 text_projection（e@16 38.5%、srank 24.6）——文本入口方向最相干。
- **我们 abl2_base 同框**：更弥散（talker_layers r99 931、e@16 仅 8.4%、srank 61）——小漂移区（5e-6×445 步）的 ΔW 比官方（2e-5 多 epoch，漂移 6-10 倍大）更接近随机方向。**LoRA r=16 的能量捕获上限 = e@16：官方 ~12-27%、我们 ~8.4%**——「voice SFT 是满秩对象，r=16 复现不了」从 PhiLia 单例升级为双例（官方产品 + 我们的 run），且给出本机配方下的定量界。

## 25. abl3:新基线(text_embedding 冻结)四臂重训 + 预算受控 8-take 全量评估（2026-08-31）

- **归档**:abl2 全系(五臂 + 两代评估 + 512 wav)→ `runs/archive/abl2_20260831/`。
- **重训**:四臂(同 §23 recipe/seed)在 §24.5 新不变量下,trainable 冒烟 147.9/453.0/594.6/594.6M ✓,exports = `runs/abl3_*/export`。
- **评估协议升级**:`token_budget = 311`(**= max cur_len 61 + 250**,prompt 计入预算;音频硬上限 ~20s,runaway 撞墙不再产生 79s 怪物)→ 生成 4 分钟、512 剪单 scorer 全量打分 12 分钟零超时。`hit_cap` = 贴预算墙。报告 = `runs/abl3_eval8/report.md` + 逐 take `report.json`。
- **主表**(mean±std):复现——MOS subfrz 双集第一(2.982/2.382)、sim base/subfrz 领先 talkfrz 垫底(hard 0.711)、EOS 免疫 subfrz(cap 0/0/12%)vs talkfrz(5/12/**50%**)。**修正**——insample CER 四臂 2.80-3.17%(std 2.6-5.6pp)全在噪声内,§22.6 的 talkfrz CER 优势为单 take 抽奖;sw002≈base 再证。GRPO 基座推荐不变:**subfrz 型**。
- **事故**:双 scorer 在线(§24 会话僵尸未清 + 新起一只)→ PUSH 分流 + recv 挂死 30 min,WSL 重启清场后单 scorer 重打、作废中间 report。**新增硬规则:起 scorer 前必须 `pgrep -fc scorer/main[.]py` 计数 = 0,启动后断言 = 1**。

## 26. blocks-vs-tables 归因:bothfrz / embfrz(2026-08-31)

- **新增 `--freeze embedding`**(`sft/model.py` + main.py help):冻主 codec 表 `talker.model.codec_embedding` + predictor 15 表 `code_predictor.model.codec_embedding`(text_embedding 本就无条件冻);冒烟对账 560.0M 可训 + 34.6M 表 = 594.6M ✓。
- **两臂**(配方/评估 = §25 同款):`bothfrz` = `--freeze talker subtalker`(仅 text_projection 6.3M 可训)、`embfrz` = `--freeze embedding`(560M 可训)。
- **embfrz ≈ base 全指标噪声内** → 1ep@5e-6 下 codec 表(34.6M)惰性,学习由 stack 承担;**"微调威力源自 embedding"假设证伪**(限定:短训练口径)。
- **bothfrz = 意外主角**:仅入口投影可训即 **MOS/CER 全场第一 + ood cap 12%**,但 **sim 崩塌**(hard 0.660)= 换声。text_projection = 全模型最高杠杆 6.3M;UTMOS/ASR 身份盲,**sim 必须留在 reward**(r_v3 已含)。报告:`runs/abl4_eval8/report.md`。

## 27. embonly 补做:归因矩阵闭合(2026-09-01)

- §26 bothfrz 系误冻(`talker.model.parameters()` 连主表、`code_predictor.parameters()` 连 15 表)——实际仅 text_projection 6.3M 可训。补做正确版 `--freeze blocks`(新增组件):双 stack(layers+norm)+ codec_head + predictor lm_head 冻,可训 = proj+双表 40,897,536(冒烟对账 ✓;首版漏冻 lm_head,复烟抓出)。
- **embonly ≈ bothfrz 全指标**(sim 0.778/0.664/0.770,MOS 3.32/3.09/2.66):加 34.6M 可训表行为零变化 → §26 结论升级为**双向证伪**;**音色锚定在 stack**(训 stack 臂 sim≥0.87,不训臂塌 0.66-0.78);换声漂移由 text_projection 单独产生。报告见 `runs/abl4_eval8/report.md` 修正节。

## 28. 阶段结论汇总：分工消融 + blocks-vs-tables 归因（2026-09-01 定稿）

> §22-§27 的蒸馏版，实验数据详见各节与 `runs/abl{2,3,4}_eval8/report*.md`。

**A. GRPO 基座与训练面**
1. **基座选 subfrz 型**（冻 predictor，只训 talker）：MOS 双集第一（2.982/2.382）、EOS 免疫唯一（ood cap 12% vs base 47%/talkfrz 50%）、sim 保持 0.85+。talkfrz 型 EOS 最差，永不入选（§25）。
2. **CER 排序须 8-take**：insample 四臂 2.80-3.17%（take 级 std 2.6-5.6pp），单 take 的"talkfrz CER 最佳"是抽奖（§23→§25 修正）。
3. **sub_weight 旋钮 = 死旋钮**：Adam 二阶归一使 loss 标量缩放不改变 predictor 纯梯度参数的步长（w=0.02 vs 0.3 漂移 0.39% vs 0.38%）；w=0 决定性实验证旋钮机械正确（71/86 键真零，15 张 predictor 表经 sem 路径共享照训）。§22.4 剂量曲线作废（§24.2）。
4. **线性 warmup 必须保留**、Dr.GRPO 无 std 除法、reward 必含 sim（§26/§27 的身份盲指标教训）。

**B. embedding 考古与冻结不变量**
5. **SFT 无条件冻 `talker.model.text_embedding`**（官方对齐：官方 CV vs Base 漂移 0.005% = 仅 3 行手工编辑 tts_pad/tts_text_bos/tts_text_eod@151671-3；Qwen2 词表血统）；无 CLI 开关（§24.5）。
6. 官方 CV 真实配方：codec 表/层/头全训（1.0-1.6%），9 内置音色烤入行 2861-2878+3010-3066（slot 3000 不冲突），speaker_encoder 不随 CV 发布，106 个 RMSNorm 位冻结=涌现；公开 `sft_12hz.py` 无任何冻结 ≠ 官方内部配方（§24.4/24.6）。
7. 旧训练的 text_embedding 漂移 = 4,262 行（语料 99.2% token），rare 行 Adam sign-step 同速——稀疏更新危害的实锤，也是冻它的理由（§24.5）。

**C. blocks-vs-tables 归因（四点矩阵，§25-§27）**
8. **codec 表双向惰性**：冻掉（embfrz≈base）无损失；可训但 stack 冻（embonly 40.9M）也救不了。1ep@5e-6 口径。
9. **音色锚定在 transformer stack**：训 stack 臂 sim ≥0.87；不训臂塌 0.66-0.78，与其余可训什么无关。
10. **text_projection（6.3M）= 全模型最高杠杆小块**：单独可训即产生全部换声漂移 + MOS 3.3/CER 2-4% 的身份盲"全场第一"。GRPO reward 摘 sim = 给策略留换声捷径。
11. SVD 备忘：官方 ΔW 非低秩（r99 777-935，e@16 12-27%），我们更弥散 → LoRA r=16 的捕获上限 = e@16（§24.6）。

**D. 评估与运维协议**
12. **评估 token_budget 必须计入 prompt cur_len**（311 = max 61 + 250）：runaway 从 79s 灾难降级为 20s 可计量事件（hit_cap），打分 1.5h 超时 → 12 分钟；hit_cap = 更灵敏的 EOS 失败探测（§25）。
13. 撞墙 take 的 CER/MOS 是截断污染，严肃比较用健康 take 口径（§25）。
14. **scorer 单实例硬规则**：起前 `pgrep -fc` = 0、起后 = 1；双实例（僵尸未清）→ PUSH 分流 recv 挂死（§25 事故）；诊断工具一律 `uvx`（py-spy），不直装 venv。

## 29. 多说话人 SFT 支持（2026-09-01）

- **CLI**：`main.py sft --namespaces {a} {b} ... [--per-pool-cap N] [--export-names ...]`（`--namespace` 的 required 移入 main.py 断言：grpo/单人 sft 必须、多臂忽略；`--speaker-audio/--limit` 在多臂模式拒绝）。
- **实现**：`cache.load_multi_sft_dataset`（(text, codes, speaker_tag) 三元组 + per-pool 头部均衡切片，只影响训练、metrics/medoid 仍全池）；`run_sft` 单/多统一 `[b, hidden]` slot-6 路径（单人 = K=1 广播，`teacher_forcing` 零改动）；`export_custom_voice` 多音色（spk_id 3000/3001/... 逐个烤入行），GRPO 可按名字采样任意导出音色。
- **冒烟 5/5**（`/tmp/opencode/smoke_multi.py`，合成双池）：broadcast ≡ [b,hidden] 全同行（逐行跟随各自 vec）、双池训练 + 导出 spk_id/烤入行逐位等于重提取 vec、单人回归 cyrene@3000、多音色 export 经 ModelWrapper 加载成功；CLI 端到端另验。
- **运维**：`/tmp` 是 12G tmpfs——冒烟/实验的 `latest.pt`（全量 FT ≈1.2G/个）会塞满它（今日实录）；大产物一律写 repo 盘。

## 30. cache 架构:namespace 贯通 + speaker ≡ namespace(2026-09-01)

- **层级化**:cache 从平铺 `.cache/{lang}/` 升级为镜像 corpus 层级的 `.cache/{speaker}/{lang}/`;preprocess CLI = `--corpus-dir /path/to/corpus --namespace Cyrene/Chinese(PRC)`(wav 目录 = `{corpus-dir}/{namespace}`,manifest = 同名 `.jsonl`——`Config.manifest` 原逻辑天然兼容);pool 落盘 `cache_root / namespace`(pipeline.run_pipeline 新增 namespace 参数,替换拍扁层级的 `dataset.name`)。
- **speaker ≡ namespace(硬规则)**:namespace 字符串就是 speaker 名,贯通 preprocess 池 → SFT 导出 spk_id → GRPO 采样;**删除全部手动口子**——`--speaker-audio`、`--export-name(s)`、SFT `--namespace`/`--limit`、GRPO `--speaker`(TrainConfig.speaker 字段一并删,`speaker = namespace`)。CLI 统一:两者都走 shared `--namespaces`(sft K 个,K=1 即单人;grpo 恰好 1 个,run_grpo assert,列表形态留给未来多角色 GRPO)。
- **SFT 代码简化**:run_sft 删单人/多臂分支(统一 K 池路径,单人 = K=1);`--limit` 语义被 `--per-pool-cap` 覆盖(单池 cap = 旧 limit);`--sub-weight` 保留(§24.2 的漏梯度验证实验待跑,用户拍板)。
- **迁移**:`.cache/Chinese(PRC)` → `.cache/Cyrene/Chinese(PRC)`(纯 mv,零重算);旧 export(sft_v1/abl3_*)spk_id 仍是 "cyrene"——GRPO 续训它们需显式提供 namespace 对齐的 export,新导出全部带 namespace 名。
- **验证**:冒烟 5/5(重跑,断言改 namespace 名)+ CLI 端到端(K=1 `--namespaces poolA` → `poolA@3000`)+ ruff;tmpfs 教训沿用(大 ckpt 一律写 repo 盘 runs/)。

## 31. preprocess 增量验证:cyrene/Chinese(PRC) +30(2026-09-01)

- **架构首验**:新层级 corpus(`delta-me13/corpora/tts/{speaker}/{lang}`)+ `--corpus-dir/--namespace` 路径式 CLI 全通;存量池随 speaker≡namespace 改名 `.cache/cyrene/Chinese(PRC)`(小写,跟磁盘)。
- **增量路径**:语料 1855 = 池 1779 + 76 新名;实际处理 **30**(checksum 守卫:1779 缓存全跳过,零重算——salvage 路径按设计工作);终态 1809 行。
- **filter 拦截 46**:44 条纯省略号/点(min_tokens 拦)+ **2 条 0 秒空 wav**(`chapter4_77_cyrene_128`-adjacent 批次,`翻开吧,永恒的一页。`/`逃不掉哦~` 有文本无音频——**语料导出损坏,待用户侧修**)。
- **产物抽检**:codes [T,16]/emb (192,)/enhanced wav/逐行 sim-cer-mos 全部就位;metrics 重建 sim 0.8701±0.0900(旧 0.8709±0.0880,+30 clip 的自然漂移)、medoid `side4_shitang_cyrene_109_f`。
- **⚠️ 数据构成发现(待拍板)**:池内按名字后缀含**非 cyrene 说话人**——wangxi 195 条(sim 0.864,接近 cyrene 的 0.871)、zuozhe 20 条(0.897);cyrene 变体 cyrenely 21 条 sim 0.797(明显另一个人声?)。非 cyrene ≈ 215/1809 ≈ 12%——单说话人 SFT 的身份汤问题,选择:保留 / 按后缀过滤 / 未来按说话人拆池(多说话人 SFT 的现成素材)。**旧池(1779)同样含此构成,非本次回归**。

## 32. aglaea/Chinese(PRC) 首池建成(2026-09-01)

- **第二个说话人池**:corpus `delta-me13/corpora/tts/aglaea/Chinese(PRC)`(954 wav+jsonl,双向零差)→ 池 928 行 = 954 − 25(≤4字,min_tokens)− 1(0.04s 音效 stub,min_seconds);全程 8.5 分钟(enhancement+embedding+scoring)。
- **首份 aglaea 标定**:sim 0.8791±0.0939(p50 0.905)、cer 0.0608、**mos 3.103**(cyrene 是 2.60——aglaea 音频质量评分显著更高);medoid `archive_aglaea_12`;centroid.npy 就位。
- **构成**:aglaea 906 + `aglaeahy` 10(变体)+ `Ev_archive_vo_avatar_*` 12(系统语音)——**零跨角色污染**(对照 cyrene 的 wangxi 215 条)。
- **多说话人 SFT 前置条件已齐**:cyrene/Chinese(PRC) 1809 + aglaea/Chinese(PRC) 928,`sft --namespaces "cyrene/Chinese(PRC)" "aglaea/Chinese(PRC)"` 即可跑;第三角色待数据。

## 33. hysilens / hyacine 首池建成(2026-09-01)

- **语料体检**:两家各 4 语言完美对齐;本批只跑 **Chinese(PRC)**(用户指定,其余 3 语言语料在位未处理)。
- **hysilens/Chinese(PRC)**:432 → 池 413(拦 19);sim 0.8824±0.0887、cer 0.0361、mos 2.757;medoid `chapter4_58_hysilens_102`。**构成:hysilens 358 + helektra 50(12%,另一说话人/变体,同 wangxi 模式,待拍板)+ 系统 5**。
- **hyacine/Chinese(PRC)**:797 → 池 781(拦 16);sim 0.8463±0.0965、cer 0.0647、mos 2.626;medoid `archive_hyacine_4`。构成:hyacine 762 + hyacinetitan 10(泰坦形态变体)+ 系统 9,零污染。
- **四池格局**:cyrene 1809 / aglaea 928 / hysilens 413 / hyacine 781 —— 多说话人 SFT 材料齐;`--per-pool-cap` 建议压到 400(hysilens 最小池)。

## 34. castorice/Chinese(PRC) 首池建成(2026-09-01)

- 1627 语料 → 池 **1556**(拦 71:短文本 + 1 条坏音频);sim 0.8709±0.1191(p50 0.909)、cer 0.066、mos 2.620;medoid `chapter4_26_castorice_200`。
- 构成:castorice 1519 + `castoricehy` 15 + `castoricetitan` 12(泰坦形态)+ 系统 10——零跨角色污染。
- **事故复盘**:打分阶段遇系统级死机(load 10.6,非重启);scorer 陪葬 → 原 preprocess 卡 recv 600s 超时循环,杀掉重启后 **checksum 守卫按完整链 salvage**(432 条已完整,1124 条从打分续),全程仅 ~10 分钟重建——幂等设计的实战验证。
- **五池格局**:cyrene 1809 / castorice 1556 / aglaea 928 / hyacine 781 / hysilens 413;多说话人 SFT 用 `--per-pool-cap 400`。

## 35. cipher/Chinese(PRC) 首池建成(2026-09-01)

- 683 语料 → 池 **661**(拦 22);sim 0.8389±0.1377、cer 0.0989、mos 2.617;medoid `chapter4_32_cipher_182`。
- 构成:cipher 600 + `shaocipher` 49(变体,同 hy/titan 模式)+ 系统 12——零跨角色污染。
- **六池格局**:cyrene 1809 / castorice 1556 / aglaea 928 / hyacine 781 / cipher 661 / hysilens 413;`--per-pool-cap 400`。注意 cipher 的 sim std(0.138)与 cer(0.099)是六池最高,内部分散度偏大。

## 36. cipher 重剪枝:shaocipher 移除(2026-09-01)

- 用户从语料移除 49 条 shaocipher → 重跑 preprocess:**管线自动剪枝**(sync 对账 manifest,0 条重处理,秒级完成);池 661 → **612**(= 629 语料 − 17 filter),shaocipher 残留 0,构成 cipher 600 + 系统 12。
- metrics 全池重建:sim 0.8455±0.1323、cer 0.0930、mos 2.638,medoid 不变(`chapter4_32_cipher_182`)。
- 运维教训追加:**scorer 的 launch 与 verify 禁止写在同一条命令里**——setsid 段的裸 `scorer/main.py` 字符串会被 verify 的 pgrep 自匹配,报出幽灵 count=2(ps 实证只有一只);双实例断言必须用独立命令 + `pgrep -af` 人工过目。

## 37. cerydra/Chinese(PRC) 首池建成——七池集齐(2026-09-01)

- 385 语料 → 池 **374**(拦 11);构成 cerydra 374 含系统 11(抽样后缀普查零污染)。
- 七池格局:cyrene 1809 / castorice 1556 / aglaea 928 / hyacine 781 / cipher 612 / hysilens 413 / cerydra 374;多说话人 SFT `--per-pool-cap 370` 对齐最小池。

## 38. preprocess `--random` + 七池乱序重跑(2026-09-01)

- **新 flag**:`preprocess --random`——`Config.random_order` → `Cache.load` 在 manifest 解析后以 `random.Random(0)` 定种乱序 `corpus_entries`(同语料同序,可复现);asset/任务表/metrics 全部顺序无关,行为不变。
- **动机**:SFT `--per-pool-cap` 头部切片从"章节有序的语料头"变成均匀抽样。
- **七池重跑**:全部 salvage(0 to process,每池 ~10s),乱序生效、内容逐行对齐无损。
- **顺带**:cyrene 语料被用户清了 3 条(1855→1852,坏 wav 清理),池同步 1809→**1806**。
- 运维:多池批量 = 单 setsid bash 顺序循环 + 单 scorer(全绿零事故)。

## 39. 多说话人漂移归因实验:无 multi-only 漂移 + 七音色分离通过(2026-09-01)

- **三臂**(全量 FT,唯一冻结 = 无条件 text_embedding;lr 5e-6、B4/accum1、warmup 20、wd 0.01、seed 0;`--per-pool-cap 370` 乱序切片,单人臂与 multi 的 cyrene 子集同 370 条):
  A `single370-1ep`(93 步)/ B `single370-6ep`(558 步,步数对照)/ C `multi7×370-1ep`(648 步,7 池混合)+ REF = abl3_base(445 步)。
- **普查(per-key 加权 rel-L2 vs base)**:所有组件步数成比例,C 全部落在单人臂带内 → **没有任何 multi-only 漂移**;多人混合在权重层面不产生单人所没有的漂移谱。text_embedding 全臂 0.00000(bit 冻结验证)。
- **方法论坑(重要)**:第一轮普查的"主表 0.77 vs 0.41 multi-only"是**伪影**——`codec_embedding` rows ≥3000 是 export 烘焙的音色槽(C 烘 7 个 vs 单人 1 个),不是训练漂移。census 必须排除槽位行;修正后主表非槽位漂移 C=0.00157 ≈ 步数插值(B 0.00246 / REF 0.00125),"单人臂饱和到同一张表"同样是槽位伪影(非槽位 pairwise 仅 0.001-0.002)。
- **身份分离矩阵(112 wav = 7 音色 × 4 prompt × 4 take,graphed,T=0.9/top-k50,seed 1234+i)**:7×7 sim 矩阵**全对角占优**,diag 0.824-0.893,margin(diag−最优 offdiag)+0.04(hyacine,与该池 corpus sim 最低一致)~+0.16(cerydra);per-voice CER 0.024-0.041 全部健康。
- **结论**:多人 SFT 可用,身份走 slot-6/烘焙槽机制如设计;多人 GRPO 基座**无需因"多人冲突"新增冻结**——§26-28 单人归因(subfrz 型)直接沿用。
- **根因修复**:`export_custom_voice` 的 spk 表键统一小写(官方约定;sampler 查表 `.lower()`);三个 export 的 config.json 已磁盘手术同步。

## 40. 1.7B 本地全量 FT:可行(--grad-checkpoint --adam-fused)(2026-09-02)

- **问题**:1.7B(`Qwen3-TTS-12Hz-1.7B-Base`,已在 HF 缓存)全量 FT(唯一冻结 = text_embedding,1.61B 可训)在 16GB 5070Ti 上 B1/accum4 是否可行。
- **直接答案:裸跑不行**——第一个 `optimizer.step()` 即 OOM(14.93/15.89GiB)。真凶不是激活:**bf16 AdamW 状态 w 3.9 + g 3.2 + m+v 6.4 = 13.5GB**,foreach step 还要瞬态物化 ~3GB;grad checkpointing 单开无效(激活本来就不是大头,15.07GB 仍 OOM 在 step)。
- **修法(两个新旗标,均默认关)**:`--grad-checkpoint`(双 stack 重算激活,`use_reentrant=False`)压激活 + `--adam-fused`(fused kernel,无大瞬态,数学同)压 step 瞬态。组合后 **峰值 13.98GB,~1.9GB 余量**,12 条 cyrene smoke 端到端通过(训+导出,@3000,3.6GB safetensors)。
- **速度**:1.7B B1 ~1.3s/批(5070Ti)→ 2590 条全量 1ep ≈ 14 分钟;hidden=2048(0.6B 的两倍),speaker vecs (1, 2048)。
- **顺带修复**:§39 小写修复漏了 export 的第二处 `spk_id[name]` 查表(重导出 KeyError)——补 `name.lower()`。

### §40 修订(同日):fused 强制化

- `--adam-fused` 开关移除,AdamW `fused=True` 硬编码(SftConfig 无字段、无 CLI)——与 flash-attention-2 同一待遇:数学恒等,只换 kernel 路径,没有理由留开关。1.7B 所需旗标只剩 `--grad-checkpoint`。

## 41. 1.7B 七池全量 SFT + graphed 评测(2026-09-02)

- **训练**:`Qwen3-TTS-12Hz-1.7B-Base` 七池全量 6470 条(无 cap,cyrene 28% / cerydra 4.7% 不平衡按"全音频"语义接受),1.61B 可训,1617 更新,~25 分钟(0.9s/update),峰值 VRAM 14.72GB 零 OOM,sem loss 2.49→~0.9。export 七音色 @3000-3006(3.6GB)。
- **graphed sampler 在 1.7B 开箱即用**(零改动,hidden 2048):112 wav(7 音色 × 2 句 × 8 take,T=0.9/top-k50)无 runaway(max 12.2s,全部正常 EOS)。
- **CER 质变**:0.000-0.0099(0.6B capped 是 0.024-0.041)——1.7B 可懂度大幅更优。
- **身份分离矩阵**:diag 0.824-0.908;margin 五升二薄——cyrene +0.21 / castorice +0.14 / aglaea +0.15 / cipher +0.09 / cerydra +0.18 均优于 0.6B;**hyacine −0.006(转负!)与 hysilens +0.048 变薄**,两者弱轴都是 hyacine↔cyrene——疑似全量训练的池不平衡(cyrene 28% 梯度占比)把共享权重往 cyrene 声学拉,盖过小池。0.6B capped 时 hyacine 也有同轴混淆(0.813),全量放大了它。
- **MOS**:cerydra 3.29 / aglaea 3.24 最高(超各自池均值),cipher 2.74 / hysilens 2.63 最低(与池质量一致)。
- **结论**:1.7B 全量多人 SFT 整体强于 0.6B capped(CER/大池 margin/MOS),代价是小池(hyacine/hysilens)身份被大池侵蚀——下一轮候选:cap 对齐(如 400)或池加权采样。

## 42. 1.7B 超参四臂:质量选 D(5e-6/B8),身份选 B(1e-5/B8),warmup 50 固化(2026-09-02)

- 四臂(lr1e-5/5e-6 × 等效B4/B8 = B1×accum4/8,warmup 50,七池全量 6470)全部 exit=0,各 ~27 min;打分双 GPU 双 scorer **按条目半切**(同句恒定同进程 → MOS 进程偏移臂间相消;3136 wav,h1/h2 report 分文件)。
- **质量**:D 全场最优(ALL CER 6.60%/MOS 2.947/ood CER 37.5%);**身份**:lr 1e-5 双臂小池 margin 翻倍(hyacine +.087 vs +.048);**warmup 50 vs 20**(C vs §41 单变量):hyacine −.006→+.048,长 warmup 救小池一半,固化默认。
- OOD 是唯一 runaway 源(A 21/D 24 每类 32 take,其余 224 全零),prompt 特异,不入排序依据。
- 推荐:D(质量)或 B(身份);报告 runs/hp17b_eval_report.md,日志 runs/hp17b_*/train.log + runs/hp17b_{rollout,score_h1,score_h2}.log。

### §42 修正(同日):锁 B(lr1e-5/B8)+ warmup 50 固化

- 显著性检验:hyacine margin B-vs-D **+6.4σ**(hysilens/cipher/cerydra +3.5~4.1σ)——身份是全场唯一强信号;MOS D−B=0.047(3σ)在听感阈以下,CER(去 ood)差 ~2σ 弱信号,sim 四臂持平。
- warmup 50 vs 20(C vs §41)单变量:hyacine −0.006→+0.048,与上条互为独立证据,**warmup 50 固化为 SFT 默认**。
- GRPO 可行性:B 组内 std_sim 0.0343 全场最高、死组 2/196 最少,组内 MOS std 与 §41 实训水平同档——B 的 export 可直接作 GRPO 基座。
- 元结论:1ep SFT 超参平坦区,停止 lr/batch 调参;后续杠杆 = 池平衡(cap/加权采样)→ GRPO。
- OOD 顶墙两型(真 runaway 1e-5 / 长但稳 5e-6),单 dur 阈值会误判,须 sim 联合判据。

### §42 附录:漂移普查 × 官方 1.7B CV 对比

- 发现 `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` 并完成同架构对照(per-key 加权 rel-L2 + delta SVD;官方烘焙槽位散布 2861-3066,**排除须取双方槽位并集**,§39 的 ≥3000 规则对官方不适用)。
- **我们(B)全组件漂移 0.25-0.8%,官方 3.4-7.0%——温和 10-30 倍,量级完全合理**;官方 text_embedding 仅动 0.012% ≈ 冻结,二次验证无条件冻结。
- SVD:官方漂移幅度大且相对集中(fc1 PR≈395,结构化适应);我们弥散微扰(PR≈1414-1808)——1 epoch 温和 SGD 形态。§41 漂移 ≈ B 一半,与 lr 比例一致。
- 数据:`runs/hp17b_census17b.json`。

### §42 续:B-ep2(resume +1 epoch)= 当前全场最优

- B(lr1e-5/B8/w50)resume 续训 1 epoch: MOS 2.900→**2.972**(超 D 成为全场最高)、去ood MOS **3.073**、hyacine margin +0.081→**+0.100**(历史最佳,身份质量双赢)、sim 0.8328;代价 ALL CER +0.82pp(ood 顶墙 15→19,去ood 仅+0.18pp)、aglaea margin 略降。
- 漂移全组件 ×1.5(0.004-0.012),仍为官方 1/5-1/10;**epoch 外推:到官方漂移量级还有 3-5 epoch 空间,epoch 是比 lr 更大的杠杆**。
- ep1 export 备份 `runs/hp17b_lr1e5_bs8/export_ep1`;评测 `runs/hp17b_lr1e5_bs8_ep2_eval/`,普查 `runs/hp17b_census17b_ep2.json`。

### §42 终:B-ep3 曲线见顶回落,ep2 = 最优基座

- B 续训三连(ep1/2/3):MOS 2.900/**2.972**/2.895,去ood MOS 2.994/**3.073**/2.996,去ood CER 1.25/1.43/1.53 单调爬,margin 三epoch稳定(hyacine +.098),sim 缓涨至 0.836,ood 顶墙 14-19 波动。
- **质量曲线 U 型,ep2 见顶:B-ep2 = 最终推荐 GRPO 基座**(`runs/hp17b_lr1e5_bs8/export`,注意其中已是 ep3 权重——**ep2 权重在 `export_ep2`**)。ep3 回退瓶颈 = 数据重复暴露,非漂移量(×1.3 仍远低于官方)。
- **GRPO 基座正式导出:`runs/b_ep2`**(= export_ep2 的拷贝,bit 校验 0.000000;custom_voice 七音色 @3000-3006 小写键,3.83GB)。GRPO `--model-path runs/b_ep2` 即用。

### §42 终裁(用户听感):基座 = B-ep1(`runs/b_ep1`)

- 用户 A/B 试听(ep1 vs ep2 同句官方栈合成):**ep1 效果最好**——耳朵为最终裁判,推翻指标侧的 ep2 推荐(MOS +0.072 属听感不可闻差异;ep1 本就 CER 占优)。
- **GRPO 基座:`runs/b_ep1`**(bit 校验 = export_ep1,七音色 @3000-3006);`runs/b_ep2` 保留作对照。
- 教训:小 MOS 差(≪0.1)不可作为基座选择依据;epoch 曲线 "ep2 峰" 对 UTMOS 成立,对人耳不成立。
