# GRPO Run 1 — 超参数与启动方案(待审)

> 生成于 2026-09-04。依据:AGENTS.md(当前版)+ STATUS §43-§47 + 代码内 `TrainConfig`/`GRPOConfig` 实际默认值。
> 审阅通过后按下文「启动序列」执行。**除 `--num-steps 30` 外,全部走默认 config(§47 定档),不传冗余旗标。**

---

## 1. 启动前状态(已验证)

- ✅ 两张 GPU 0 MiB 空(chrome 残留 + 冒烟 scorer 已清;cuda:1 的 5558 MiB 陈旧计数随进程退出消失,**无需重启 WSL**)
- ✅ 无 trainer/scorer/probe 残留进程;ZMQ 5555/5556 空闲;/dev/shm 无 grpo_* 残留
- ✅ ruff 全绿;fused AdamW 已进 GRPO loop(本次唯一代码改动,冒烟通过)

## 2. 基座与数据

| 项 | 值 | 出处 |
|---|---|---|
| 基座 | `runs/d_ep1`(5e-6/B8/warmup50/1ep/seed0 重训,export bit 校验,七音色 @3000-3006)| §44 |
| LoRA 起点 | 全零(lora_a kaiming / lora_b 零初始化)| 代码 |
| 文本池 | `private_dataset/grpo_pool_v1.jsonl` **3200 条** = delta 964(in-character)+ 共享 2236(七角色均分 316-320)| §45 |
| 池长度 | p50 75 字 / p90 145 / max 200;>140 共 ~467 条(budget 896 下已实测可跑,零撞墙)| §45/§47 |
| 池格式 | `{"speaker": "{voice}/chinese(prc)", "text": ...}` 全小写,speaker 随行 | §45 |

## 3. 训练超参数(TrainConfig,全部默认值)

### 优化
| 参数 | 值 | 备注 |
|---|---|---|
| lr | **1e-6**(用户 9ccbaf1 @13:19 提交定档)| linear warmup **20 步**(默认已同步 20)|
| optimizer | AdamW **fused=True** | wd 0.01;本次由 foreach 切 fused(与 SFT 对齐,消 foreach 瞬态)|
| grad clip | 1.0 | |
| variant | **dr**(Dr.GRPO)| 原始量级优势,组内减均值;raw reward 不除 std(C1v8/9 教训)|

### 批量布局(Fish S2)
| 参数 | 值 |
|---|---|
| num_prompts × group_size | **8 × 8 = 64** rollout/步,1 次 optimizer update/步 |
| 每组 | 同 (speaker, text) × 8 take,组内 Dr.GRPO 优势;梯度按组均权累积 |
| 抽样 | 每步 `rng.sample(pool, 8)`,speaker 镜像池比例(约 17% cyrene/hyacine…均匀)|
| seed | 0 |

### Rollout / 显存定档(§47 阶梯实测)
| 参数 | 值 | 实测依据 |
|---|---|---|
| token_budget | **896** | max cur_len+t_max = 864 + 32 余量;8/8 自然 EOS |
| token_budget_infer(lmax)| **896** | rollout 峰值 5.1G |
| sampler | **graphed** | prefill eager → 1-token 图重放,~4.8× |
| ref logprob | **全组 B8 单次** | 推理模式峰值仅 9.4G |
| policy logprob | **micro=2** | micro=4 在 T>800 必 OOM(§47 阶梯)|
| 显存峰值 | ~14.4G / 15.74G(5070 Ti)| expandable_segments(main.py grpo 路径自动注入)|

### 采样契约(rollout 与 logprob 重构共享)
| 参数 | 值 |
|---|---|
| temperature / top_k | 0.9 / 50(semantic + subtalker 同)|
| top_p / repetition_penalty | **钉死 1.0 / None**(官方 serving 默认 rep=1.05 有意偏离,RL 必须无状态)|
| language | Auto(Auto prefill 布局是 logprob 重构合法性的前提)|
| MTP γ(subtalker_weight)| 1.0(Fish 式,`subtalker_time_norm=True` ÷Q);=0 可复现 MD 语义单通路 |

### 奖励(v3.1,raw 量级加权和,组内 std<eps 分量熄火)
| 项 | 值 |
|---|---|
| R | `1.0·r_sv + 1.0·r_wer + 0.2·r_mos` |
| r_sv | `sigmoid((sim − sv_center)/sv_scale)` — **per-speaker 校准**,metrics.json 注入 |
| r_wer | `1 − CER`,CER 归一化口径(NFKC/小写/去标点空白符号/中文数字→阿拉伯,§47 已验证)|
| r_mos | `max(0, 2.5 − mos)` hinge 地板,健康区恒 0 → 按构造熄火,只挡不驱动 |
| λ_mos | 0.2 |

**7 池校准(sv_center/sv_scale ← 各池 metrics.json sim.mean/std)**:

| 池 | sv_center | sv_scale |
|---|---|---|
| aglaea | 0.8791 | 0.0939 |
| castorice | 0.8709 | 0.1191 |
| cerydra | 0.8766 | 0.1073 |
| cipher | 0.8455 | 0.1323 |
| cyrene | 0.8709 | 0.0875 |
| hyacine | 0.8463 | 0.0965 |
| hysilens | 0.8824 | 0.0887 |

## 4. 运行参数

| 参数 | 值 |
|---|---|
| out_dir | `runs/grpo_v1`(默认)|
| num_steps | **50**(用户定档;每步 ckpt 可 `--resume` 续跑加长)|
| ckpt_every | 1(每步存 LoRA delta + codec head + optimizer)|
| monitor | on(per-step jsonl:loss/kl/grad/R/cer/sim/t_max/per_speaker{sim,cer}/skips)|
| scorer_timeout | 1800s |

## 5. 启动序列(铁律:scorer 先行,错峰加载)

```sh
# 1) scorer 先起(cuda:0 = 4070S),等 "connected" 日志再动 trainer
setsid workers/scorer/.venv/bin/python workers/scorer/main.py \
  --sv-dir /tmp/opencode/sv --device cuda:0 \
  --push-endpoint tcp://127.0.0.1:5555 --pull-endpoint tcp://127.0.0.1:5556 \
  > runs/grpo_v1_scorer.log 2>&1 < /dev/null &

# 2) 确认 connected 后,trainer(cuda:1 = 5070 Ti)
setsid env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  workers/trainer/.venv/bin/python workers/trainer/main.py grpo \
  --namespaces "cyrene/Chinese(PRC)" "castorice/Chinese(PRC)" "aglaea/Chinese(PRC)" \
               "hyacine/Chinese(PRC)" "cipher/Chinese(PRC)" "hysilens/Chinese(PRC)" \
               "cerydra/Chinese(PRC)" \
  --text-pool-path private_dataset/grpo_pool_v1.jsonl \
  --num-steps 50 --warmup-steps 20 \
  > runs/grpo_v1_trainer.log 2>&1 < /dev/null &
```

## 6. 监控要点(健康判据)

**健康轨迹**:
- KL 0.001-0.05 区间(lr 1e-6 下预期 <0.02;冒烟实测 0.003;>0.5 异常,>1 停);**warmup 20 步内 lr 从 1e-6 爬到 1e-5,KL 近零属预期,不是信号缺失**
- grad_norm 0.5-3(贴 clip 1.0 属正常;持续 >10 异常)
- cer_mean 全池起点 ~5-10%(基座短句水平),慢变量
- sim_mean 稳在各自池 center 附近(±0.02)
- t_max 中短句应 150-400;≈ budget-cur_len(撞墙)只允许出现在 >140 字长句
- flat 跳过:全池短句占比高,预期 10-25%(长池是 31%);>50% 说明奖励信号塌了

**停跑条件**:KL > 1 或持续上行;loss NaN;runaway skip 连续出现于短句组;CUDA OOM。

## 7. 已知风险与既定决策(不阻塞,知情运行)

1. **级联脆弱性未修**(用户裁决暂缓):打分若超 1800s → trainer 删 wav → scorer 读已删文件崩。缓解 = timeout 1800 + 4070S 已无桌面争用。崩了再修。
2. **`\n` 多行 OOD 未探针**:池内 delta 侧多行文本(§44 记 54%)从未单独验证模型生成行为;scoring 侧 CER 已证明安全(归一化剥 `\n`)。**处置:前 5 步盯多行行的 t_max/EOS,异常再抽听**——不阻塞启动。
3. **MOS 分量大部分时间熄火**:设计如此(hinge + 组内 std 小即零),r_mos=0 是健康态不是故障。
4. **d_ep1 8 类基线未测**(用户决定跳过):E2 配对评测时补跑 `w100_rollout.py "d_ep1=runs/d_ep1"` 即可,每步 ckpt 不损失任何中间态。
5. **AGENTS 小漂移**:锚点行池构成写 982/2218,§45 正典为 964/2236(总数一致);dump 后顺手修正。

## 8. 训练后(E2,预备)

1. `w100_rollout.py cuda:1 "grpo_v1=runs/grpo_v1/step_00029.pt的export?"` — 注:GRPO ckpt 是 LoRA delta,评测需先合成/export 或按 w100 的 arm 机制加载;具体在 E2 启动前确认(留到 30 步跑完再说)。
2. 8 类 general.json 配对:d_ep1 基线 vs grpo_v1;判据见 §6(E2 判据:in-character CER ↓ / 共享池不升 / sim ≥ 基线−0.01 / long_2 不退化)。
