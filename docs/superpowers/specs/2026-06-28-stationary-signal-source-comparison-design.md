# Stationary Signal Source Comparison Design

日期：2026-06-28

## 背景

`realtime_pose_stationary5_v1` schema 已经把 `stationary_prob_5` 固定为特征通道 `149:154`。当前 feature-only 训练会把这 5 个概率作为 `pred_x0` 的一部分一起预测。模型也已经支持可选的 `stationaryHead`，导出 ONNX 时可以额外输出 `stationary_prob5`。Unity 端现有推理管线在检测到 head 输出时会覆盖 `PosePrediction.StationaryProb5`。

本设计只回答一个问题：Unity 物理/接触控制应该消费 feature 通道里的 `stationary_prob_5`，还是消费 `stationaryHead` 的额外输出。

## 目标

- 保持 `realtime_pose_stationary5_v1` schema 不变。
- 保持 source、task、normalizer 数据契约不变。
- 让 Unity runtime 可以显式切换 stationary 信号源。
- 在同一批 replay 和评估数据上比较不同 stationary 信号源的接触控制表现。
- 用 false lock、missed lock、jitter、transition lag 和 Unity replay 物理指标决定默认 runtime 策略。

## 非目标

- 不修改 AMASS/source 生成逻辑。
- 不修改 task/normalizer 目录结构。
- 不引入新的 schema 名。
- 不在第一阶段修改物理 solver 的核心接触求解逻辑。
- 不在第一阶段加入 blend、低通、滞回等后处理，除非评估证明需要第二阶段处理。

## 候选信号源

### FeatureChannel

从 denoised feature 的 `stationary_prob_5` 通道读取概率，通道范围为 `149:154`。

优点：

- 不需要额外 ONNX 输出。
- 与当前 schema 完全一致。
- 训练、导出、Unity 兼容路径最短。

风险：

- stationary 概率只受整体 denoise loss 间接监督，可能被 pose 主任务稀释。
- 与姿态特征共用输出空间，概率校准不一定适合接触阈值。

### StationaryHead

从模型额外输出 `stationary_prob5` 读取概率。训练时使用 `target_stationary_prob_5` 的 BCE 监督。

优点：

- 监督目标直接对应 stationary 分类/概率。
- 可以独立调 loss 权重和 head-only fine-tune。
- 更适合作为 Unity 接触控制的显式信号。

风险：

- 多一个训练、导出、runtime 分支。
- 如果联合训练权重不合适，可能影响 pose 主任务。
- 如果 head 输出抖动，需要第二阶段增加滤波或滞回。

### Auto

运行时默认策略。第一阶段定义为：如果模型资产声明存在 `stationary_prob5` head 输出，则使用 `StationaryHead`；否则使用 `FeatureChannel`。

## 实验矩阵

第一阶段只做三组。

| 组 | 模型 | Unity stationary 来源 | 用途 |
| --- | --- | --- | --- |
| A | 当前 feature-only 100k | FeatureChannel | 当前 Unity baseline |
| C | 从 A 初始化，冻结非 stationary head 参数，只训练 stationary head | StationaryHead | 尽量固定 pose 主干，干净比较 stationary 信号源 |
| B | `use_stationary_head=true` 联合训练 | StationaryHead | 观察联合训练是否进一步改善 stationary 和接触表现 |

执行顺序为 A、C、B。C 的优先级高于 B，因为它能减少 pose 模型变化带来的干扰。

## Unity Runtime 设计

新增一个显式策略枚举：

```csharp
public enum StationarySignalSource
{
    Auto,
    FeatureChannel,
    StationaryHead
}
```

推理管线按策略写入 `PosePrediction.StationaryProb5`：

- `FeatureChannel`：始终保留 feature 解码出的 `stationary_prob_5`。
- `StationaryHead`：要求 Sentis runner 能取到 `stationary_prob5`，否则记录错误并回退或失败，具体行为由调用场景配置。
- `Auto`：有 head 输出时使用 head，否则使用 feature。

`PosePrediction` 增加或保留调试字段：

- `StationaryProb5`：最终给物理系统消费的概率。
- `HasStationaryHead`：模型是否提供 head 输出。
- `StationarySignalSourceUsed`：本帧实际采用的信号源，方便 replay 和日志对齐。

## Python 评估设计

新增离线评估模块：

- `eval/stationary_signal_metrics.py`
  - 只实现纯指标函数。
  - 输入预测概率、GT 概率、阈值和可选关节名。
  - 输出 per-joint 和 aggregate 指标。

- `eval/evaluate_stationary_signal_source.py`
  - CLI 入口。
  - 加载 checkpoint、normalizer、task/replay 数据。
  - 同时支持 feature channel 输出和 stationary head 输出。
  - 写出 JSON/CSV 报告。

输出目录：

```text
outputs/stationary_signal_compare/<timestamp>/
  metrics_summary.json
  per_clip_metrics.csv
  per_joint_metrics.csv
  unity_replay_summary.json
```

## 指标

指标按 Unity 物理风险排序。

### false_lock_rate

GT 处于运动状态，但预测 stationary 概率超过阈值。这个指标优先级最高，因为误锁会让脚、手或 pelvis 被物理系统错误固定。

### missed_lock_rate

GT 处于静止状态，但预测 stationary 概率低于阈值。这个指标对应漏锁，常见表现是脚滑或接触不稳。

### stationary_f1

在阈值 `0.5` 和 `0.7` 下计算 precision、recall、F1。必须输出 per-joint 指标，左右脚权重最高，左右手次之，pelvis 用于辅助判断整体稳定性。

### transition_lag_frames

统计 GT stationary 状态切换时，预测信号到达相同状态所需的帧数。动到静和静到动分开统计。

### prob_jitter

统计连续帧 stationary 概率变化，衡量信号抖动。高 jitter 会造成 Unity 接触状态频繁开关。

### Unity replay 物理指标

Unity 自动 replay 侧记录：

- 脚滑距离。
- 接触状态切换次数。
- 物理修正量。
- 穿地或爆炸次数。
- 每帧实际使用的 `StationarySignalSourceUsed`。

## 判胜规则

第一优先级是降低误锁。

1. 如果某方案 `false_lock_rate` 明显更高，直接淘汰。
2. 在 `false_lock_rate` 接近时，选择 `missed_lock_rate` 更低、脚滑更少的方案。
3. 如果 `StationaryHead` 离线指标更好但 Unity replay 中 jitter 明显更高，不直接采用 head，而进入第二阶段的 filter/hysteresis 设计。
4. 如果 `FeatureChannel` 和 `StationaryHead` 表现接近，选择 `FeatureChannel`，减少 runtime 分支。
5. 如果 head-only fine-tune 的 C 明显优于 A，最终策略倾向于 schema 保留 feature 通道，但 Unity 物理/接触控制使用 `StationaryHead`。

## 测试计划

Python 冒烟测试：

- `tests/smoke/eval/test_stationary_signal_metrics.py`
- 覆盖 false lock、missed lock、F1、jitter、transition lag 的 toy 数据。
- 覆盖空事件、全静止、全运动和阈值边界。

Unity 侧验证：

- 确认 `FeatureChannel` 策略不会读取 head 输出。
- 确认 `StationaryHead` 策略在 head 输出存在时覆盖 feature。
- 确认 `Auto` 在 head 存在时使用 head，在 head 缺失时回退 feature。
- replay summary 必须记录每帧实际使用的 stationary 信号源。

## 实施阶段

第一阶段：

1. 增加 Unity stationary 信号源选择。
2. 增加 Python stationary 指标函数和 CLI。
3. 跑 A baseline。
4. 从当前 A checkpoint 初始化 C，只训练 stationary head。
5. 导出 C 到 Unity。
6. 对同一批 replay 比较 `FeatureChannel` 和 `StationaryHead`。

第二阶段只在第一阶段证明有必要时启动：

- 增加 threshold sweep。
- 增加 hysteresis 或低通滤波。
- 对物理 solver 的接触阈值进行独立调参。

## 验收标准

- 能在 Unity 中显式选择 stationary 信号源。
- 能在 Python 离线评估中分别得到 feature channel 和 stationary head 的指标。
- A 和 C 至少完成同一批 replay 的对比报告。
- 报告中包含 per-joint 指标和 Unity replay 物理指标。
- 最终能根据判胜规则明确推荐 `FeatureChannel`、`StationaryHead` 或进入第二阶段后处理。
