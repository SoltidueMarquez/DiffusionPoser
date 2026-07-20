# 论文实验参考协议

## 文档定位

本文档规定论文实验应包含的对比方法、评估协议、指标、消融和报告格式，是
`documents/PAPER_LONG_TERM_GOAL.md` 的配套长期参考。

- 本文档不是某次实验的结果记录，不绑定 baseline tag、branch、checkpoint 或 seed。
- 具体实验开始前，必须另建 experiment profile 或实验记录，冻结数据划分、代码 commit、随机种子、超参数和验收阈值。
- 不提交 `dataset/`、`runs/`、`output/`、`save/`、checkpoint 或生成的二进制数据。
- 只有在相同输入、相同数据划分、相同骨架评价集合和相同 evaluator 下重新得到的结果，才允许放入同一张主对比表。
- 不直接把不同论文表格中的数字拼接为本文结果，因为帧率、骨架、关节集合、Root 对齐和 Jitter 定义可能不同。

## 总体实验结构

论文实验固定分为五组：

1. 标准三点 Tracker 主对比；
2. 六点 Tracker 主对比；
3. 动态掉线与重连专项；
4. 关键模块消融；
5. 实时性能与 Unity/Sentis 一致性。

其中：

- `standard_three` 是与外部方法公平比较的基础协议；
- `dynamic_dropout`、持续 no-Hip 和重连是本文核心鲁棒性协议；
- `full_six` 用于比较增加 Pelvis 和双脚 Tracker 后的重建能力；
- C00/C04 等历史实验属于内部筛选或消融，不能代替外部方法主对比。

## 公平比较的统一协议

### 数据与帧率

- 主实验统一使用 AMASS，目标帧率为 60 Hz。
- 应建立并冻结一套公开可说明的 60 Hz 协议：
  - P1：同分布/常规划分；
  - P2：跨数据集泛化划分。
- 具体 AMASS 子集、subject、sequence manifest 和文件 hash 必须在正式实验记录中保存。
- 当前仓库自定义 `data_loaders/splits/` 可以继续用于开发，但只有在所有外部基线都按相同划分重训时，才能用于论文直接对比。

### 因果性

- 所有方法只能使用当前及过去观测，不得使用未来帧。
- 本文最终方法固定使用过去 60 帧历史生成当前第 61 帧。
- 外部方法若原版使用未来帧，必须切换到零未来帧版本或重新训练因果版本。
- SparsePoser 主对比使用严格因果的 `Ours-0` 配置，不使用其包含未来帧的版本。
- 若外部方法无法统一为 60 帧历史，应在表中明确写出上下文长度，不得隐去额外历史信息。

### 输入与历史

- 三点协议只输入 Head、Left Hand、Right Hand 的位置与旋转。
- 六点协议输入 Head、Left Hand、Right Hand、Pelvis/Hip、Left Foot、Right Foot 的位置与旋转。
- 无效 Tracker 必须通过显式 valid mask 表示，不使用当前帧 GT 或 stale fill。
- 所有长序列评估使用 predicted history；GT history 结果只能作为 oracle 上界，不能作为主结果。
- 所有方法共享同一 Tracker dropout timeline、同一初始状态和同一 warm-up 帧数。

### 骨架与坐标

- 外部主对比统一在公共 SMPL/AMASS 关节集合上计算指标，优先使用各方法都能映射的 22 个关节。
- 本文最终方法原生 body.fbx 24 骨骼结果单独用于 Unity/运行时表，不能直接与外部 22 关节 MPJPE 混表。
- 主表必须明确 MPJPE 是 global/world-space 还是 pelvis-aligned。
- 本文以 global/world-space MPJPE 为主指标，同时补充 pelvis-aligned/root-relative MPJPE，用于区分 Root 漂移与局部姿态误差。
- 所有方法使用同一 body shape、rest pose、Tracker offset 和 FK 实现；若无法统一，必须单列说明。

### 随机性与统计

- 最终模型和关键基线至少运行 3 个 seed，报告 `mean ± std`。
- 长序列结果以 sequence 为统计单元，建议同时给出 95% bootstrap confidence interval。
- 扩散模型在线评估固定采样 seed，每帧只允许一次实际部署采样；不得使用 best-of-N 选择 GT 最近结果。
- 筛选实验可以单 seed，但不得作为论文最终结论。

## 实验一：标准三点 Tracker 主对比

### 输入

`standard_three = Head + Left Hand + Right Hand`，三个 Tracker 均提供 6DoF。

### 必须比较的方法

| 方法 | 类型 | 对比目的 |
| --- | --- | --- |
| AvatarPoser | 确定性 Transformer | 经典三点全身姿态基线 |
| AGRoL | 条件扩散 | 与本文扩散路线直接比较 |
| RPM-Reactive / RPM-Smooth | Rolling prediction | 比较预测历史、缺失观测与过渡平滑性 |
| FisherPoser | 因果概率姿态模型 | 比较较新的三点概率建模方法 |

**AvatarJLM 和原始 DiffusionPoser 均不进入本文三点对比基线范围。** 不在主表、动态三点表或必做基线复现实验中加入这两个方法。本文最终方法在结果表中统一写作 `Ours`，不把原始 DiffusionPoser 作为额外 baseline。

### 主表指标

| 指标 | 单位 | 趋势 | 含义 |
| --- | --- | --- | --- |
| Global MPJPE | cm | ↓ | 世界坐标下全身关节位置误差，包含 Root 误差 |
| Pelvis-aligned MPJPE | cm | ↓ | 去除 Root 平移后的局部姿态位置误差 |
| MPJRE | deg | ↓ | 平均关节旋转误差 |
| MPJVE | cm/s | ↓ | 平均关节速度误差 |
| Jitter | m/s³ | ↓ | 关节位置三阶差分的高频抖动 |
| Tracker position mean / P95 | cm | ↓ | 有效 Head/Hands 的位置对齐误差 |
| Tracker rotation mean / P95 | deg | ↓ | 有效 Head/Hands 的旋转对齐误差 |

标准三点主结论必须同时考虑：

- MPJPE/MPJRE：姿态准确性；
- MPJVE/Jitter：运动稳定性；
- Tracker mean/P95：实际控制点是否跟手。

不能只凭单个 MPJPE 判断方法优劣，也不能用过度平滑换取低 Jitter 而不报告 Tracker P95。

## 实验二：六点 Tracker 主对比

### 输入

`full_six = Head + Left Hand + Right Hand + Pelvis/Hip + Left Foot + Right Foot`。

### 必须比较的方法

| 方法 | 约束 |
| --- | --- |
| Final IK | 使用同一校准后六点 Tracker、同一骨架和同一求解预算 |
| SparsePoser `Ours-0` | 严格因果、零未来帧；优先统一为相同 60 帧历史 |

**原始 DiffusionPoser 不进入本文六点对比基线范围。** 六点主表以 Final IK 和 SparsePoser `Ours-0` 为外部基线，本文最终方法统一写作 `Ours`。可在消融表中额外加入只用 `full_six` 训练的本文方法 specialist，作为统一模型的 oracle specialist 参考。

### 六点主表指标

六点主表使用与三点相同的核心指标：

- Global MPJPE；
- Pelvis-aligned MPJPE；
- MPJRE；
- MPJVE；
- Jitter；
- 六个 Tracker position mean/P95；
- 六个 Tracker rotation mean/P95。

Final IK 可能获得很低的端点位置误差，但全身旋转、自然性和时间稳定性未必更好，因此必须同时报告 MPJRE、MPJVE 和 Jitter，不能只比较 Tracker 对齐。

## 实验三：动态掉线与重连专项

### 必测场景

| 场景 | 说明 |
| --- | --- |
| `full_six` | 六点稳定输入 |
| `standard_three` | Head + 双手稳定输入 |
| `static_sparse` | 片段内固定的其他合法 3–5 点组合 |
| `dynamic_dropout` | 非 Head Tracker 随机掉线并恢复 |
| 6→3 | 从六点切换为标准三点 |
| 3→6/reconnect | 三点运行后恢复 Pelvis 与双脚 |
| no-Hip | Pelvis/Hip 单独或持续缺失 |
| single-foot missing | 左脚或右脚单独缺失 |
| hand loss | 单手或双手暂时缺失，Head 始终有效 |

每种掉线至少覆盖 `0.1 / 0.5 / 1.0 / 2.0 s`。如果训练分布没有覆盖某个时长，应将其标记为 out-of-distribution robustness test。

### 对比方法

- RPM-Reactive 与 RPM-Smooth；
- 本文最终方法 `Ours`；
- AvatarPoser、AGRoL 和 FisherPoser 只有在使用相同 valid mask、相同掉线训练数据重新适配后，才作为 `-mask-adapted` 基线加入；
- 不把不支持缺失观测且未适配的方法失败结果包装成不公平的主结论。

### 动态专项指标

| 指标 | 单位 | 趋势 | 含义 |
| --- | --- | --- | --- |
| Dropout MPJPE / MPJRE | cm / deg | ↓ | 掉线期间姿态精度 |
| Dropout MPJVE / Jitter | cm/s / m/s³ | ↓ | 掉线期间运动稳定性 |
| Peak Jerk (PJ) | evaluator 原始单位 | ↓ | 状态切换后一秒内最大 jerk |
| Area Under Jerk (AUJ) | evaluator 原始单位 | ↓ | 状态切换后一秒内累计 jerk |
| Root XZ mean error | cm | ↓ | Root 平均水平位置误差 |
| Root XZ final error | cm | ↓ | 长序列末尾 Root 水平误差 |
| Root drift | cm/min | ↓ | 持续 no-Hip 时的累计漂移速度 |
| Root yaw mean error | deg | ↓ | Root 水平朝向误差 |
| Reconnect peak jump XZ | cm | ↓ | 重连瞬间的位置跳变 |
| Reconnect peak jump yaw | deg | ↓ | 重连瞬间的朝向跳变 |
| Recovery time | s / frames | ↓ | 重连后恢复到稳定误差范围的时间 |

PJ/AUJ 分别在 tracking→synthesis 和 synthesis→tracking 两种转换中报告；对应本项目为 6→3/dropout 与 3→6/reconnect。

## 实验四：关键模块消融

所有消融从同一最终配置出发，一次只改变一个因素，避免无法解释的全排列。

### A. 生成模型

| 变体 | 目的 |
| --- | --- |
| Deterministic regression | 验证扩散建模本身的收益 |
| Ours causal diffusion | 本文最终因果扩散模型 |

比较 MPJPE、MPJRE、MPJVE、Jitter、Tracker P95 和推理延迟。

### B. 动态 Tracker 条件

| 变体 | 目的 |
| --- | --- |
| `standard_three` specialist | 三点专用上界/参考 |
| `full_six` specialist | 六点专用上界/参考 |
| unified without dynamic dropout | 验证只有静态组合是否足够 |
| unified dynamic 3–6 | 最终统一模型 |

重点比较四类 Tracker pattern 的分组 MPJPE、MPJVE、Jitter 和 Tracker P95。

### C. Predicted-history rollout

| 变体 | 目的 |
| --- | --- |
| no rollout / GT-history training | 暴露 train-test history gap |
| H=1 rollout | 验证短 rollout |
| H=1 + H=2–8 rollout | 验证长 rollout 与重连稳定性 |

主指标为完整 predicted-history 长序列的 MPJVE、Jitter、PJ、AUJ、Root drift 和 reconnect 指标。

### D. Stationary/contact

| 变体 | 目的 |
| --- | --- |
| no stationary/contact | 姿态基础模型 |
| stationary output only | 验证五维因果静止监督 |
| stationary + contact losses | 验证接触几何与速度约束 |
| full bounded stationary runtime | 最终运行时方案 |

Stationary 指标只用于本文内部消融，不能要求没有对应输出的外部方法报告。

### E. Runtime Root Resolver

| 变体 | 目的 |
| --- | --- |
| model output only | 观察原始 Root 漂移 |
| direct Hip correction / no reconnect smoothing | 观察硬切换问题 |
| full Resolver | 验证 no-Hip、Head anchor 与重连策略 |

主指标为 Root XZ/yaw error、Root drift、reconnect peak jump、recovery time、PJ 和 AUJ。

### F. DDIM 步数

至少测试 `1 / 5 / 10 / 20` 个采样步，报告：

- MPJPE、MPJRE；
- MPJVE、Jitter；
- Tracker P95；
- 单帧 latency、P95 latency、FPS。

该实验用于给出精度—稳定性—速度折中，不允许只选择最快或最准确的一端。

`root_heading_delta_sincos` 目前是任务表示的一部分，不默认作为论文创新点，因此不列为必做消融。只有在论文贡献中明确声称 Root delta 表示优于绝对 yaw 时，才增加“absolute yaw vs delta sin/cos”实验。

## Stationary 与接触指标

### Stationary 分类/概率

| 指标 | 趋势 | 说明 |
| --- | --- | --- |
| Stationary F1@0.7 | ↑ | Unity runtime 阈值下的综合分类结果 |
| False-lock rate@0.7 | ↓ | 实际运动却被错误锁定的比例 |
| Missed-lock rate@0.7 | ↓ | 实际静止却未锁定的比例 |
| Pre-clamp out-of-bounds ratio | ↓ | 有界投影前超出 `[0,1]` 的比例 |
| Probability jitter | ↓ | 相邻帧 stationary probability 抖动 |

F1 不能单独作为验收依据；必须同时检查 false lock 与 missed lock，尤其不能通过过度预测静止来提高 recall。

### 接触结果

| 指标 | 单位 | 趋势 | 说明 |
| --- | --- | --- | --- |
| Contact foot velocity / foot skating | m/s | ↓ | GT 接触期间预测脚部移动速度 |
| Floating-foot ratio | ratio | ↓ | GT 接触时预测脚高于阈值的比例 |
| Ground-penetration ratio | ratio | ↓ | 预测脚穿过地面的比例 |
| no-Hip contact foot velocity | m/s | ↓ | Hip 缺失时的脚接触稳定性 |
| no-Hip floating-foot ratio | ratio | ↓ | Hip 缺失时的悬空脚比例 |

## 实验五：实时性能与部署闭环

### 必报指标

| 指标 | 条件 |
| --- | --- |
| 参数量 | 模型网络参数总数 |
| 模型文件大小 | checkpoint、ONNX/Sentis 分别报告 |
| DDIM steps | 与精度结果一一对应 |
| Mean latency | batch=1、单模型独占设备 |
| P95 latency | batch=1、包含必要前后处理 |
| FPS | 与 latency 在同一环境测量 |
| Peak GPU memory | 同一 batch 和采样步数 |
| Unity/Sentis end-to-end latency | 包括 Tracker 编码、模型、Resolver 和骨骼写回 |

如果论文宣称 60 Hz 实时运行，目标硬件上 Unity/Sentis 的 P95 端到端延迟应不超过 `16.67 ms`。若未达到，应准确表述为 online/interactive，而不是稳定 60 Hz real-time。

性能测试禁止与其他 GPU 训练或评估并行；历史并行评估产生的 FPS 不能作为论文实时性能结果。

### Python 与 Unity 一致性

必须验证：

- 相同输入和随机种子下 ONNX/Sentis 与 Python 输出误差；
- Tracker codec 解码一致性；
- 单帧 Resolver 一致性；
- 长序列 Root 累计误差；
- dropout/reconnect 状态机行为；
- 最终 body.fbx local rotation、Root XZ/yaw 和 pelvis height 的运行时一致性。

## 建议论文表格

### Table 1：标准三点主对比

```text
Method | Global MPJPE | PA-MPJPE | MPJRE | MPJVE | Jitter | Tracker Pos P95 | Tracker Rot P95
```

外部基线行固定为 AvatarPoser、AGRoL、RPM-Reactive、RPM-Smooth 和 FisherPoser；不加入 AvatarJLM 或原始 DiffusionPoser。另设 `Ours` 行填写本文最终方法结果。

### Table 2：六点主对比

```text
Method | Global MPJPE | PA-MPJPE | MPJRE | MPJVE | Jitter | Tracker Pos P95 | Tracker Rot P95
```

外部基线行固定为 Final IK 和 SparsePoser `Ours-0`；不加入原始 DiffusionPoser。另设 `Ours` 行填写本文最终六点方法结果。

### Table 3：动态掉线与重连

```text
Method/Pattern | Dropout MPJPE | MPJVE | Jitter | PJ | AUJ | Root drift | Reconnect jump XZ/yaw | Recovery time
```

### Table 4：Stationary/contact 消融

```text
Variant | F1@0.7 | False lock | Missed lock | OOB | Foot velocity | Floating ratio | Penetration ratio
```

### Table 5：实时性能

```text
Method/Steps | Params | Model size | Mean latency | P95 latency | FPS | Peak memory | Unity E2E latency
```

## 当前 evaluator 覆盖情况

当前 `eval/evaluate_realtime_pose_rollout.py` 已覆盖或部分覆盖：

- MPJPE、root-relative MPJPE、MPJRE、MPJVE、Jitter；
- Tracker position/rotation mean 与 P95；
- Root XZ mean/final error、Root drift；
- reconnect peak jump、recovery time；
- Tracker pattern、6→3、3→6/reconnect、no-Hip duration 分组；
- Stationary F1、false lock、missed lock、OOB 和 probability jitter；
- foot skating/contact velocity、floating foot、ground penetration；
- PJ、AUJ。

正式实验前仍需确认或补齐：

1. 公共 22-joint 外部比较 evaluator；
2. 明确的 global MPJPE 与 pelvis-aligned MPJPE 命名；
3. Root yaw error 的 degree 输出；
4. 固定一秒窗口的 tracking→synthesis 与 synthesis→tracking PJ/AUJ；
5. batch=1 mean/P95 latency、显存和 Unity 端到端性能汇总；
6. 统一 AMASS 60 Hz P1/P2 manifests；
7. AvatarPoser、AGRoL、RPM、FisherPoser、SparsePoser 和 Final IK 的统一输入适配与结果导入。

## 外部方法参考

- [AvatarPoser](https://arxiv.org/abs/2207.13784)
- [AGRoL](https://openaccess.thecvf.com/content/CVPR2023/html/Du_Avatars_Grow_Legs_Generating_Smooth_Human_Motion_From_Sparse_Tracking_CVPR_2023_paper.html)
- [RPM](https://openaccess.thecvf.com/content/CVPR2025/html/Barquero_From_Sparse_Signal_to_Smooth_Motion_Real-Time_Motion_Generation_with_CVPR_2025_paper.html)
- [FisherPoser](https://openaccess.thecvf.com/content/CVPR2026/html/Xia_FisherPoser_Human_Motion_Estimation_from_Sparse_Observations_with_Hierarchical_Region-Wise_CVPR_2026_paper.html)
- [SparsePoser](https://arxiv.org/abs/2311.02191)

## 结论使用原则

- 三点主结论以 MPJPE/MPJRE 和 MPJVE/Jitter 的共同结果为准。
- 六点主结论必须同时超过或解释与 Final IK 的端点精度差异，以及与 SparsePoser 的全身姿态和时间质量差异。
- 动态掉线主结论重点看 MPJVE、Jitter、PJ、AUJ、Root drift、reconnect jump 和 recovery time。
- Stationary/contact 主结论重点看 false lock 与实际脚部滑动，不以 F1 单项决定。
- 实时性主结论以 Unity/Sentis batch=1 P95 end-to-end latency 为准。
- 所有未通过预注册验收标准的结果保留为负结果或局限性，不改写成已验证贡献。
