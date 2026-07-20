# 论文长期目标与任务定义

## 文档定位

本文档定义 DiffusionPoser 当前论文长期保持不变的研究任务与验收方向。

- 本文档回答“论文要解决什么问题”。
- `schemas/`、`documents/schema_registry.md` 和运行时契约回答“当前数据与接口如何实现”。
- `documents/experiments/` 回答“某个具体方法假设是否得到实验支持”。
- 论文对比方法、指标、消融和公平性要求见 [`documents/PAPER_EXPERIMENT_REFERENCE.md`](PAPER_EXPERIMENT_REFERENCE.md)。
- 模型结构、loss 权重、rollout 课程、Resolver 版本和训练超参数可以迭代，但不得在没有明确决策记录的情况下改变本文档定义的研究任务。
- 实验尚未通过验收的模块只能表述为待验证方法，不能提前写成论文结论。

## 一句话任务定义

本文研究**动态稀疏 Tracker 条件下的因果实时全身姿态重建**：在 VR/实时角色驱动场景中，仅使用当前及历史的稀疏 6DoF Tracker 观测和过去 60 帧的模型预测历史，实时重建当前帧全身姿态，并在 Tracker 缺失、动态掉线和重新连接时保持长序列姿态准确、运动平滑、身体接触稳定且 Root 运动不漂移。

## 研究场景

系统面向 60 Hz 在线人体动作重建与 Unity 角色驱动。可用 Tracker 集合最多包含：

1. Head
2. Left Hand
3. Right Hand
4. Pelvis/Hip
5. Left Foot
6. Right Foot

合法输入帧满足：

- Head Tracker 始终有效；
- 每帧至少三个 Tracker 有效；
- Pelvis/Hip Tracker 可以缺失；
- 有效 Tracker 提供经过统一坐标和校准处理的世界位置与旋转；
- 无效 Tracker 通过显式 `sensor_valid` 表示，不允许用当前帧 GT 或过期观测填充。

论文需要同时覆盖以下观测条件，而不能只报告固定六点配置：

- `full_six`：六个 Tracker 全部有效；
- `standard_three`：Head 与双手三个 Tracker；
- `static_sparse`：一个片段内保持不变的其他合法稀疏组合；
- `dynamic_dropout`：运行过程中发生 Tracker 掉线与恢复。

## 因果输入与输出

### 输入

在目标时刻 `t`，模型只能使用：

- 当前帧及历史 Tracker 观测、旋转和有效性；
- `t-60` 到 `t-1` 的 60 帧姿态与 Root 历史；
- 在线运行时已经可获得的上一帧最终 Resolver 状态、`floor_y` 和坐标原点状态。

真实部署中的历史必须来自模型预测和运行时最终状态。训练阶段可以使用受控扰动或 rollout 构造预测历史，但不能把依赖未来帧或当前帧 GT 的信息引入推理契约。

### 输出

系统在每个时刻生成当前第 61 帧的：

- 24 个 body.fbx 骨骼的 local rotation delta 6D；
- Root heading delta；
- Root XZ motion delta；
- Pelvis local height；
- Pelvis、左右脚和左右手的五维 `stationary_prob_5`。

模型输出经过与训练语义一致的 Runtime Root Resolver 得到最终世界 Root、姿态和关节位置，再用于下一帧历史和 Unity 角色驱动。

## 核心研究问题

### RQ1：动态稀疏观测下能否准确重建全身姿态？

模型应在六点、标准三点和其他合法稀疏组合下保持合理的全身位置与旋转精度，并充分利用新增 Tracker，而不是只对单一设备布局过拟合。

### RQ2：预测历史会不会导致长序列误差累积？

在线推理不能持续依赖 GT 历史。论文需要验证 rollout 训练或等价方法能否减轻 exposure bias，并改善长序列速度误差、抖动和累计漂移。

### RQ3：Tracker 掉线与重连时是否稳定？

系统需要在 Hip 或其他非 Head Tracker 缺失时继续工作，并在 Tracker 恢复后平滑回到观测约束，避免 Root 跳变、朝向突变和速度尖峰。

### RQ4：接触状态是否有助于减少滑步和错误锁定？

`stationary_prob_5` 必须严格因果，并在脚步及其他候选接触部位上平衡 false lock 与 missed lock。接触辅助不能以明显牺牲姿态精度或动态动作自由度为代价。

### RQ5：训练、Python 评估与 Unity Runtime 是否语义一致？

模型、Normalizer、Tracker codec、Resolver、ONNX/Sentis 和 Unity 解码必须共享同一明确契约。离线指标改善只有在运行时能够复现时才构成有效论文证据。

## 方法模块及其职责

以下模块服务于同一个论文任务，不分别改变任务定义：

| 模块 | 在论文任务中的职责 |
| --- | --- |
| 因果扩散补全模型 | 根据历史姿态与当前稀疏 Tracker 条件生成当前全身姿态 |
| 动态 Tracker mask | 覆盖六点、三点、静态稀疏和动态掉线条件 |
| Rollout 训练 | 缩小 GT 历史训练与预测历史推理之间的分布差异 |
| `stationary_prob_5` | 提供严格因果的身体静止/候选接触信息，辅助减少滑步 |
| Runtime Root Resolver | 融合模型 Root motion 与当前可用 Tracker，处理 Hip 缺失和重连 |
| Unity/Sentis 导出 | 验证方法能够在目标实时运行环境中保持同一语义 |

具体模块可以被替换或消融。只有被对照实验支持的模块才能作为最终贡献保留。

## 论文长期目标

### G1：姿态准确性

在 `full_six` 和 `standard_three` 上取得有竞争力的 MPJPE、MPJRE 与 Tracker 对齐误差，并报告不同 Tracker 模式下的分组结果。

### G2：动态稀疏鲁棒性

在 `static_sparse`、`dynamic_dropout`、持续 no-Hip 和 3→6/6→3 转换中保持可用，不因单个非 Head Tracker 缺失而失效。

### G3：长序列稳定性

在完整 predicted-history rollout 下控制 MPJVE、Jitter、PJ、AUJ 和 Root 漂移，重点报告掉线与重连窗口内的指标。

### G4：接触可靠性

同时报告 stationary F1、false-lock、missed-lock、输出范围以及脚部接触速度/高度，不能只用单一 F1 掩盖错误锁定。

### G5：实时部署闭环

保持单模型、单 motion 输出和统一运行时契约；验证 Python 与 Unity 长序列回放的一致性，并报告目标硬件上的推理延迟或帧率。

## 最低论文证据要求

在将某个版本称为论文最终方法前，至少应具备：

1. 与明确基线和主要相关方法在同一数据划分、同一 Tracker 条件下的比较；
2. `full_six`、`standard_three`、`static_sparse` 和 `dynamic_dropout` 的分组结果；
3. 完整 predicted-history 长序列评估，而非只做独立单帧或 GT-history 评估；
4. Hip 掉线、持续 no-Hip、6→3 和 3→6/reconnect 的专项结果；
5. Rollout、stationary/contact 和 Resolver 关键设计的消融实验；
6. 多个随机种子的重复实验或置信区间；
7. Python 与 Unity/Sentis 的数值或行为一致性验证；
8. 对失败案例、适用边界和实时性能的诚实报告。

具体数值阈值应在每轮实验开始前写入 experiment profile 或实验记录，不能根据结果事后修改。

## 论文主张边界

本文当前不主张解决：

- 缺失 Head 或少于三个有效 Tracker 的任意欠约束输入；
- 使用未来帧的离线平滑或动作修复；
- 从文本、音频或场景语义生成动作；
- 原始硬件坐标标定、Tracker 身份识别或通用跨骨架 retargeting；
- 任意地形上的 Root Y 估计；当前运行时 Root world y 由 `floor_y` 决定；
- 未经实验验收的 stationary、contact、rollout 或 Resolver 改动已经优于基线。

## 预期论文主线

论文应围绕以下逻辑展开：

1. 稀疏 Tracker 全身姿态重建在真实在线运行中存在动态设备可用性和预测历史累积误差；
2. 因果扩散补全模型提供多解动作分布建模能力；
3. 动态 mask 与 rollout 训练使模型适应不同 Tracker 组合和自身预测历史；
4. 因果 stationary 建模与 Runtime Root Resolver 分别处理接触稳定和 Root 漂移/重连；
5. 长序列、掉线重连和 Unity Runtime 实验共同验证方法，而不是仅依赖逐帧姿态误差。

最终摘要、贡献列表和实验结论必须以已经完成的对照实验为依据；本文档只固定研究问题和目标，不预先宣布实验成功。
