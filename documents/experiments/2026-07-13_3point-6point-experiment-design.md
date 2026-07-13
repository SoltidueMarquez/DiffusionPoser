# 3-point / 6-point 稀疏追踪实验设计

记录日期：2026-07-13

状态：AMASS-only 设计稿，等待实验前复核

适用项目：DiffusionPoser `realtime_pose_stationary5_v1`

## 1. 实验目标

本组实验只回答两个核心问题：

1. 同一个学习模型能否在不切换 checkpoint 的情况下同时支持标准 VR 3-point 和完整 6-point 输入？
2. 在 6-point 输入下，学习式动作先验相对于传统全身 IK 是否能生成更准确、更连续、更自然的全身动作？

传感器掉线重连、长序列漂移和物理优化器开关将在后续独立协议中展开。本协议只评估传感器稳定在线时的 3-point / 6-point 主结果，并为后续实验提供固定基线。

## 2. 输入定义

所有 tracker 都提供世界坐标下的位置和旋转，即 6DoF。传感器顺序固定为：

| 索引 | 名称 | 3-point | 6-point |
| ---: | --- | :---: | :---: |
| 0 | head | 有效 | 有效 |
| 1 | left wrist/controller | 有效 | 有效 |
| 2 | right wrist/controller | 有效 | 有效 |
| 3 | waist/pelvis | 无效 | 有效 |
| 4 | left foot | 无效 | 有效 |
| 5 | right foot | 无效 | 有效 |

3-point 严格表示 HMD + 左右手柄，不允许用 waist、GT pelvis、GT root translation 或未来帧补充信息。6-point 严格表示 head、双手、waist、双脚均有效。

## 3. 待验证假设

### H1：统一模型支持两种稳定传感器配置

同一个 AMASS checkpoint 在 3-point 和 6-point 下都能稳定推理；相对于分别训练的专用模型，统一模型的 MPJPE 和 MPJRE 相对退化不超过 5%。

### H2：增加三个下半身 tracker 带来可测量收益

同一个统一模型从 3-point 切换到 6-point 后，下半身位置误差、root translation error 和 foot skating 均显著下降。比较采用逐序列配对统计，而不是只比较全数据均值。

### H3：6-point 学习模型优于纯 IK

在 tracker 拟合误差相近的前提下，统一学习模型应当比传统 IK 获得更低的 MPJPE、MPJVE 和 Jitter。IK 可能在被观测末端的瞬时位置误差上更低，因此必须同时报告 tracker adherence 与全身动作质量，不能只选择对学习模型有利的指标。

## 4. AMASS 数据与协议

AMASS 是 3-point / 6-point 的主控实验数据集。使用 GT SMPL/SMPL-X 动作通过 FK 生成 head、双手、waist 和双脚 6DoF tracker。GT 动作保留为评测真值；不能把 GT pelvis 或未声明的 GT 关节作为模型输入。

主表采用公开三点工作使用的 AMASS A-P1 60 FPS 协议，与当前 Unity 和 `realtime_pose_stationary5_v1` 运行频率一致。正式运行前从公开实现导出准确的 train/validation/test 序列 manifest，并在仓库中版本冻结；所有方法使用完全相同的 manifest。不得将同一原始序列切成相邻窗口后分配到不同 split。

AMASS 训练时使用一个统一 checkpoint，稳定输入采样比例固定为：

- 50% 窗口使用 clean 3-point mask；
- 50% 窗口使用 clean 6-point mask。

测试时从同一条 GT 序列生成配对的 3-point 和 6-point 输入，使传感器数量收益能够逐序列比较。

正式实验前需要额外冻结以下 AMASS 数据产物：

1. A-P1 的 train/validation/test 原始序列 manifest，以及去重和坏序列排除记录；
2. 统一的目标骨架、公共评测关节列表和 body-shape 处理配置；
3. 固定的 tracker attachment 与 tracker-to-joint calibration 配置；
4. 每条测试序列配对生成的 3-point/6-point 6DoF tracker 流、GT 动作和有效帧 mask；
5. 全序列评测清单、统一 warm-up 长度和动作类别元数据。

训练可以继续按窗口采样；验证和测试必须从序列开头连续滚动到结尾，除规定的首帧初始化外不得中途重置。

## 5. 比较方法

### 5.1 3-point 基线

AMASS 三点主表采用公开三点工作中的代表性学习式基线：

- FinalIK：传统 IK 锚点；
- AvatarPoser：Transformer 三点姿态恢复；
- AGRoL：三点条件扩散模型；
- EgoPoser：面向头手输入的实时姿态模型；
- SAGE：分层生成式三点模型；
- AvatarJLM：joint-level 三点建模；
- HMD-Poser：HMD 场景模型；
- RPM-Reactive / RPM-Smooth：在线滚动预测三点基线；
- Ours-Unified：本项目统一模型。

三点主结论优先依据所有方法在同一预处理、同一骨架映射和同一评测脚本下重跑的结果。论文原表数字只用于 sanity check，不能与本项目重跑结果直接混合排名。

实现优先级分为两层：FinalIK、AvatarPoser、EgoPoser、HMD-Poser 和 RPM 属于主表必做基线；AGRoL、SAGE、AvatarJLM 在公开代码、权重和骨架能够公平对齐时加入扩展表。无法重跑的方法只引用原论文结果，不进入同列排名。

### 5.2 6-point 基线

六点主表只在 AMASS 上报告：

- FinalIK：Unity 全身 IK；
- DLS-IK：在统一骨架上实现的 damped least-squares 可复现基线；
- SparsePoser：六个 6DoF tracking device 的学习式基线；
- Ours-Unified：与三点主表完全相同的 AMASS checkpoint；
- Ours-Unified + Tracker IK：检查 tracker 末端对齐后处理的影响。

SparsePoser 的核心是先用 skeleton-aware convolutional autoencoder 学习动作流形，再用 learned IK 网络把手脚末端调整到六个 tracker。它比纯 IK 更接近“学习先验 + 末端约束”的六点方案，因此用于判断本项目的收益是否不仅来自“用了神经网络”。

## 6. 公平性约束

所有方法必须满足以下条件：

1. 使用同一批测试序列、同一 FPS 和相同 warm-up 区间；
2. 只使用当前帧及历史帧，禁止 future frames；
3. 使用相同 tracker 定义、坐标系和 tracker-to-joint calibration；
4. 使用相同目标骨架、body shape 或统一的公共关节子集；
5. 第一帧允许统一 T-pose/标定初始化，之后禁止用 GT pose、GT root 或 GT contact 重新初始化；
6. 在线推理过程中禁止周期性重置隐状态或根节点；
7. IK 使用上一帧预测作为在线初始化，首帧使用统一 T-pose；
8. Tracker IK、滤波器等后处理必须在方法名称中显式标注，不能只给 Ours 使用未披露的后处理；
9. 所有延迟均以 batch size 1、相同硬件、同步计时报告。

不同方法骨架不一致时，先转到共同世界坐标关节集合再计算指标。公共集合至少包含 pelvis、hips、knees、ankles/feet、spine、neck/head、shoulders、elbows、wrists。骨长差异必须通过统一 shape/calibration 处理，不能通过 Procrustes alignment 消除全局平移和朝向错误。

## 7. 指标

### 7.1 姿态与动态

- MPJRE（degree）：平均关节旋转误差；
- MPJPE（cm）：世界坐标平均关节位置误差；
- MPJVE（cm/s）：平均关节速度误差；
- Jitter：关节 jerk 的序列统计；所有方法使用同一中央差分、边界处理和单位定义；
- Upper-body PE / Lower-body PE：上、下半身位置误差；
- Root position error / heading error：根节点全局位置和朝向误差。

### 7.2 观测一致性

- Tracker position error（cm）：预测骨架对应 tracker attachment 与输入 tracker 的距离；
- Tracker rotation error（degree）；
- Hand PE、Waist PE、Foot PE：按 tracker 类型分别统计，不能只报告六点平均值。

### 7.3 物理合理性辅助指标

- Foot skating velocity / skating frame ratio；
- Ground penetration depth / penetration frame ratio；
- Floating height；
- Contact precision、recall 和 F1。

物理指标在 clean 3-point / 6-point 附表中报告，用于描述基础模型的动作质量；物理优化器的因果收益由后续单独的 on/off 消融验证。

### 7.4 实时性

- 网络推理 latency：mean、P50、P95、P99；
- 端到端 latency：包含 tracker 编码、网络和该结果行显式启用的 Tracker IK；
- FPS、显存峰值和参数量。

## 8. 实验矩阵

### E01：AMASS clean 3-point 主比较

目的：验证三点姿态恢复精度。所有方法使用 head + 双手输入，报告姿态、动态、tracker adherence 和物理辅助指标。

### E02：AMASS clean 6-point vs IK

目的：验证完整六点下学习先验相对传统 IK 的价值。主比较为 FinalIK、DLS-IK、SparsePoser、Ours-Unified；Tracker IK 作为显式附加行。

### E03：同一模型的 3-point vs 6-point 配对比较

目的：只改变输入 mask，不改变 checkpoint。对每条测试序列计算 `metric(3-point) - metric(6-point)`，报告均值、95% bootstrap confidence interval 和六点获益的序列比例。

### E04：统一模型 vs 专用模型

训练三个 AMASS-60 模型：

- `Ours-3-only`：只见 clean 3-point；
- `Ours-6-only`：只见 clean 6-point；
- `Ours-Unified`：3-point / 6-point 各占 50%。

该实验验证统一输入能力是否以明显精度损失为代价。统一模型相对对应专用模型的 MPJPE/MPJRE 相对退化阈值设为 5%。

## 9. 主结果表模板

### Table A：AMASS 3-point

| Method | MPJRE ↓ | MPJPE ↓ | MPJVE ↓ | Jitter ↓ | Lower PE ↓ | Root PE ↓ | Foot skate ↓ | Hand tracker PE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

### Table B：AMASS 6-point

| Method | MPJRE ↓ | MPJPE ↓ | MPJVE ↓ | Jitter ↓ | Lower PE ↓ | Root PE ↓ | Foot skate ↓ | Mean tracker PE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

### Table C：统一能力与传感器收益

| Model | 3-point MPJPE ↓ | 6-point MPJPE ↓ | 3-point Lower PE ↓ | 6-point Lower PE ↓ | 3-point Jitter ↓ | 6-point Jitter ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

## 10. 统计与报告规则

- 所有指标先按序列求均值，再在序列层面汇总，避免长序列支配结果；
- 主表报告 mean，附录报告 standard deviation 和 95% bootstrap confidence interval；
- 3-point vs 6-point 使用配对 bootstrap；
- 每个随机训练设置至少运行 3 个 seed；
- 不只报告最佳 seed；主表使用 3 个 seed 的均值；
- 同时保存 per-sequence CSV，支持后续按动作类别、速度和移动距离分析；
- 定性视频必须使用预先固定的序列列表，不能按最终效果挑选。

## 11. 当前模型进入实验前的必要改动

当前 `realtime_pose_stationary5_v1` 已有六个 tracker 通道和 `sensor_valid`，但仍要求 waist 始终有效、每帧至少三个 tracker，并固定 61 帧/60 FPS。正式实验前至少需要：

1. 解除 waist tracker 必须有效的约束，使 head + 双手成为合法 3-point；
2. 无 waist 时改用 head-relative/global-motion estimator 估计 root heading、root XZ 和 pelvis height；
3. 训练采样器显式生成 clean 3-point 与 clean 6-point 的 50/50 配对 mask；
4. 评测器增加公共骨架映射、分身体区域误差和分 tracker adherence；
5. IK baseline 禁止调用 GT root，并使用与模型相同的 calibration；
6. 将网络和 Tracker IK 的运行时间分项记录。

## 12. 参考协议

- AvatarPoser: https://arxiv.org/abs/2207.13784
- EgoPoser: https://arxiv.org/abs/2308.06493
- HMD-Poser: https://arxiv.org/abs/2403.03561
- SparsePoser: https://arxiv.org/abs/2311.02191
- RPM: https://arxiv.org/abs/2504.05265
