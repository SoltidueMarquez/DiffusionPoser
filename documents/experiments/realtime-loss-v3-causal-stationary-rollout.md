# Realtime Loss v3 + Causal Stationary + Stable Rollout 实验记录

> The authoritative C04 profile is `configs/experiments/c04-loss-v3-stable-rollout.json`, the active baseline tag is `baseline/c04-loss-v3-stable-rollout`, and the compact result record is `documents/experiments/realtime-loss-v3-causal-stationary-rollout-results.json`. Legacy filesystem paths below are historical provenance only and must not be used as current commands or artifact locations.

## 结论

`C04 causal-stationary-loss-v3-stable-rollout` 未通过组合验收，9 项检查中通过 6 项、失败 3 项。

- 通过：训练稳定性、梯度比例、姿态保护、tracker 保护、重连指标和长 rollout 指标。
- 失败：stationary 原始输出越界率、`F1@0.7` 绝对增益、`false-lock@0.7` 退化上限。
- 决策：保留本分支的实现与实验数据，不进入多 seed 30k，不增加 Sensor-only MLP Proposal。
- 后续优先处理 stationary 输出校准和 false-lock 代价，再单独验证 rollout；当前证据不支持把失败归因于 MLP 缺失。

机器可读结果保存在
`documents/experiments/realtime-loss-v3-causal-stationary-rollout-results.json`。

## 基线与运行入口

- baseline tag：`baseline/c04-loss-v3-stable-rollout`
- baseline commit：`e8d93edc9d12e5725a6612a12891fc576538e8d6`
- cleanup branch：`codex/c04-cleanup-refactor`
- 实验日期：`2026-07-16`
- 完整可恢复入口：

```powershell
conda run --no-capture-output -n diffusionposer5070 python -m scripts.experiments.run_profile `
  --experiment-config configs/experiments/c04-loss-v3-stable-rollout.json `
  --stage validate
```

所有 Python 阶段均由脚本使用
`conda run --no-capture-output -n diffusionposer5070` 执行。阶段状态和完整日志位于
`runs/realtime-loss-v3-causal-stationary-rollout/orchestration/`。

## 实现内容

- 因果标签与迁移：
  `data_loaders/stationary_label_config.py`、
  `data_loaders/realtime_pose_kinematics.py`、
  `data_converter/relabel_realtime_pose_stationary.py`。
- Loss v3：
  `diffusion/gaussian_diffusion.py`、
  `diffusion/realtime_pose/`。
- 双层 rollout：
  `train/training_loop.py`、
  `train/realtime_rollout.py`。
- Runtime `0.5/0.7` 双阈值评估：
  `eval/evaluate_realtime_pose_rollout.py`。
- 自动实验与汇总：
  `scripts/experiments/run_profile.py` 与
  `configs/experiments/c04-loss-v3-stable-rollout.json`。

保持模型 `214→154`、61 帧 schema、Resolver、ONNX/Sentis 输出和 Unity 接口不变。

## 数据产物

| 产物 | 路径或规模 |
|---|---|
| source | `dataset/generated/sources/realtime_pose_stationary5_v1/amass_60hz_v3_causal_stationary` |
| task | `dataset/generated/tasks/realtime_pose_stationary5_v1/amass_60hz_v3_causal_stationary_rollout9_tasks/20260716_171854_causal_stationary_rollout9_seed10` |
| task 数量 | train `17,526`，test `2,608` |
| normalizer | `dataset/generated/normalizers/realtime_pose_stationary5_v1/amass_60hz_v3_causal_stationary_train/20260716_172957_causal_stationary_train_seed10` |
| normalizer 样本 | `17,526` 个 task，10 个 mask/task，共 `10,690,860` 帧 |
| longseq eval | `dataset/generated/longseq_eval/realtime_pose_stationary5_v1/amass_60hz_v3_causal_stationary_pre_stage_a/20260716_pre_stage_a_seed10` |
| longseq 规模 | 18 条序列，共 `49,326` 帧，单条 `2,018–4,869` 帧 |

Stationary 元数据：

```text
stationary_label_method=joint_center_speed_causal_fast_release_v2
stationary_static_speed=0.03
stationary_moving_speed=0.25
stationary_causal_window=5
stationary_release_mode=fast_release_min
```

## 梯度标定

使用固定 16 个 batch、seed 10、batch size 16 和 Stage A 20k checkpoint 标定。
权重限制为 `[1e-6, 100]`，没有权重触发限制；所有实测比例与目标比例一致。

| Loss group | 目标比例 | 固化权重 |
|---|---:|---:|
| local rotation | 0.20 | 3.7483991513 |
| body geometry | 0.25 | 0.2056174121 |
| tracker relative position | 0.15 | 0.0664902285 |
| tracker relative rotation | 0.05 | 0.0581069459 |
| no-Hip yaw | 0.05 | 8.7213458665 |
| no-Hip height | 0.05 | 0.0232212027 |
| stationary regression | 0.10 | 0.0202359978 |
| stationary margin | 0.05 | 0.0226605547 |
| stationary range | 0.10 | 0.0076066353 |
| contact height | 0.025 | 0.0152887753 |
| contact velocity | 0.025 | 0.0000203435 |
| joint velocity | 0.04 | 0.0003846126 |
| rotation velocity | 0.04 | 0.0006603078 |
| yaw velocity | 0.02 | 0.0009690314 |

原始标定文件：
`runs/realtime-loss-v3-causal-stationary-rollout/calibration/realtime-loss-v3-calibration.json`。

## C00 控制评估

- checkpoint：
  `runs/realtime-loss-v2-rollout8/screening/A02_loss_v2_h1/20260715_231025_A02_loss_v2_h1_seed10_5k/model000005000.pt`
- 不训练，直接在因果 stationary v2 的相同 longseq 集上评估。
- 输出：
  `output/realtime-loss-v3-causal-stationary-rollout/C00_a02_causal_stationary/longseq_eval_summary.json`
- 耗时：`2,597.911 s`。

## C04 训练

- 初始化：
  `runs/realtime_pose_stationary5_v1/v2_stage_a_rotation_fix_20k_20260714/20260714_140005_stage_a_rotation_fix_20k_seed10/model000020000.pt`
- 配置：seed 10、batch 16、LR `1e-5`、5k step、mask `30/30/20/20`。
- Short rollout：概率 `0.5`，固定 H=1。
- Long rollout：概率 `0.25`，H=2 → H=2–4 → H=2–8 课程。
- run：
  `runs/realtime-loss-v3-causal-stationary-rollout/C04_causal_stationary_loss_v3_stable_rollout/20260716_183107_C04_causal_stationary_loss_v3_stable_rollout_seed10_5k`
- checkpoint：上述目录中的 `model000005000.pt`。
- 训练耗时：`7,559.277 s`，约 2 小时 6 分钟。

最近 1k step：

| 指标 | 数值 |
|---|---:|
| loss | 0.01036953 |
| simple loss | 0.00369144 |
| aux loss | 0.00029173 |
| grad norm（clip 前） | 0.290958 |
| 梯度裁剪比例 | 0% |
| short rollout 事件比例 | 0.488 |
| long rollout 事件比例 | 0.264 |
| long rollout 平均 H | 5.0434 |

## C00 与 C04 评估结果

C04 输出：
`output/realtime-loss-v3-causal-stationary-rollout/C04_causal_stationary_loss_v3_stable_rollout/longseq_eval_summary.json`。
评估耗时 `2,020.876 s`。C00 与 C04 均使用同一 18 条 longseq、同一 mask timeline 和非 EMA 权重。

### Stationary

| 指标 | C00 | C04 | 变化 | 验收 |
|---|---:|---:|---:|---|
| F1@0.5 | 0.2376 | 0.3155 | +0.0779 | 仅记录 |
| false-lock@0.5 | 0.1704 | 0.3232 | +0.1527 | 变差 |
| missed-lock@0.5 | 0.7323 | 0.5111 | -0.2212 | 改善 |
| F1@0.7 | 0.1810 | 0.2544 | +0.0734 | 失败，要求 `≥+0.10` |
| false-lock@0.7 | 0.1008 | 0.1813 | +0.0805 | 失败，最多允许 `+0.03` |
| missed-lock@0.7 | 0.7860 | 0.6337 | -0.1523 | 通过，要求改善 `≥0.10` |
| 原始输出越界率 | 47.78% | 28.40% | -19.38 pp | 失败，要求 `<5%` |

### 全局姿态与重连

| 指标 | C00 | C04 | 相对变化 |
|---|---:|---:|---:|
| 全局 MPJPE（cm） | 12.0065 | 10.3802 | -13.55% |
| 全局 MPJRE（deg） | 17.8599 | 17.3554 | -2.82% |
| 全局 MPJVE（cm/s） | 116.5078 | 68.3605 | -41.33% |
| Jitter（m/s³） | 5497.0669 | 3040.4738 | -44.69% |
| PJ | 1270.9211 | 1042.0860 | -18.01% |
| AUJ | 7134.0889 | 5275.0543 | -26.06% |
| reconnect MPJVE（cm/s） | 164.6997 | 139.7338 | -15.16% |
| reconnect Jitter（m/s³） | 10359.5403 | 8558.5243 | -17.39% |
| reconnect PJ | 1905.8482 | 1632.5900 | -14.34% |
| reconnect AUJ | 10159.5697 | 8298.8229 | -18.32% |

### 姿态与 Tracker 保护项

| 指标 | C00 | C04 | 相对变化 |
|---|---:|---:|---:|
| full-six MPJPE（cm） | 4.3445 | 4.0008 | -7.91% |
| full-six MPJRE（deg） | 13.5168 | 12.8675 | -4.80% |
| full-six MPJVE（cm/s） | 64.4947 | 52.2993 | -18.91% |
| standard-three MPJPE（cm） | 20.3564 | 17.4435 | -14.31% |
| standard-three MPJRE（deg） | 22.0585 | 21.4033 | -2.97% |
| standard-three MPJVE（cm/s） | 162.9222 | 76.7729 | -52.88% |
| tracker position（cm） | 4.8305 | 3.9661 | -17.89% |
| tracker rotation（deg） | 23.2857 | 17.1667 | -26.28% |

所有保护项均未退化，且多数明显改善。

## 验收表

| 检查 | 结果 |
|---|---|
| stationary 原始输出越界率 `<5%` | 失败 |
| `F1@0.7` 绝对提高 `≥0.10` | 失败 |
| `missed-lock@0.7` 改善 `≥0.10` | 通过 |
| `false-lock@0.7` 恶化不超过 `0.03` | 失败 |
| reconnect 至少两项改善 `5%` | 通过，四项均改善 |
| reconnect 关键指标无一恶化超过 `5%` | 通过 |
| full-six、standard-three、tracker 无超过 `5%` 的退化 | 通过 |
| 最近 1k step 梯度裁剪率 `<20%` | 通过，实际 0% |
| loss group 梯度比例位于 `0.5×–2×` | 通过 |

## 失败诊断

1. Stationary 分支从“漏锁”转向了“过度锁定”。`missed-lock@0.7` 改善
   `0.1523`，但 `false-lock@0.7` 同时恶化 `0.0805`。这与 margin loss
   对 active/inactive 分别归一化、近似等权处理两类的方向一致：它提高了 active
   输出，却没有体现 runtime 中 false lock 更高的代价。
2. Range loss 只在越界后提供 SmoothL1 软惩罚，且仍受统一 timestep attenuation。
   它把越界率从 `47.78%` 降至 `28.40%`，说明梯度有效，但不足以形成接近
   `[0,1]` 的输出参数化，因此 `<5%` 目标无法靠当前权重达到。
3. Rollout 不是本轮失败的主要来源。重连四项改善 `14.34%–18.32%`，全局
   MPJVE、Jitter 和 AUJ 分别改善 `41.33%`、`44.69%` 和 `26.06%`，且姿态与
   tracker 保护项全部通过。
4. 因果 stationary v2 标签的因果性、快速释放、恢复和元数据测试全部通过。
   当前结果更直接指向输出校准与分类代价设计问题，而不是需要额外 MLP。

## 测试记录

| 命令 | 结果 |
|---|---|
| `conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/data_pipeline -q` | 108 passed |
| `conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/train -q` | 79 passed, 1 skipped |
| `conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/eval -q` | 42 passed |
| `conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke -q` | 358 passed, 1 skipped |

生成数据、训练产物、评估输出和 checkpoint 均保留在本地且不进入提交。
