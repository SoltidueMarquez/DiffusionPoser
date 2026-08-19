# RealtimePose Predictor + 单帧 DiT Python Contract

本文档只描述当前 Predictor + 单帧 DiT 主链路。所有 source、Task Store、normalizer
和 checkpoint 由调用方显式指定路径。

## 时间与参考系

```text
motion context offsets       = [-10,...,-1]
Predictor tracker offsets          = [-10,...,0]
Predictor tracker previous offset  = -11
Predictor output offsets           = [0,...,10]
Predictor free-running max steps   = 30
Runtime/model FPS                  = 30
Pose dimension               = 24 * 6 = 144
Predictor sparse dimension         = 54
```

Predictor 与 DiT 均假定已有完整 10 帧 Pose 历史，不存在 padding、history length 或
valid mask。所有 Pose 和 Tracker 特征统一表达在当前预测时刻的 Head-yaw 参考系
`C_n`：原点为当前 Head 的 XZ 与当前 floor，高度轴保持世界 Y，朝向为当前 Head
yaw。Predictor rolling 每前进一步都用该步 Head 重新构造 `C_n`。

Predictor 54D sparse 通道顺序为：三点 global rotation6D 18D、三点 relative
rotation6D 18D、三点 position 9D、三点 position delta 9D。其中：

```text
relative_rotation[t] = R[t-1]^T @ R[t]
position_delta[t]    = p[t] - p[t-1]
```

三点固定为 Head、左 Wrist、右 Wrist。rotation6D 不做直接差分。

## Source

30Hz source 字段：

| 字段 | 形状 |
|---|---:|
| `body_pose_body_fbx_local_delta_6d` | `[T,144]` |
| `root_pos_world` | `[T,3]` |
| `root_yaw` | `[T]` |
| `pelvis_height` | `[T,1]` |
| `tracker_pos_world` | `[T,6,3]` |
| `tracker_rot_world_6d` | `[T,6,6]` |
| `joints_world` | `[T,24,3]` |
| `stationary_prob_5` | `[T,5]` |
| `joint_offsets_parent` | `[24,3]` |
| `joint_rest_local_rotations_6d` | `[24,6]` |

Predictor sequence Dataset 启动时一次性预载所选 split。每条 source 只执行一次整段
FK，内存中仅常驻 `joint_rotations_world_6d [T,24,6]`、
`tracker_positions_world [T,6,3]`、`tracker_rotations_world_6d [T,6,6]`、
`floor_y [T]` 与 `joint_offsets_parent [24,3]`；原始 body pose 和世界旋转矩阵在
预载后释放。每个样本从内存切出 offset `-11..+40` 的 52 帧，3x3 旋转矩阵在
batch 进入设备后重建。

## 单帧 Task Store

Task Store 保存原始物理值，不保存归一化结果。设 shard 样本数为 `M`：

| 字段 | 形状 | 语义 |
|---|---:|---|
| `motion_context_clean` | `[M,10,144]` | 当前 `C_n` 下的过去 10 帧 Pose |
| `core_tracker_context_clean` | `[M,11,54]` | 核心三点过去 10 帧与当前 sparse 特征 |
| `current_pose_target_clean` | `[M,144]` | 当前帧 diffusion target |
| `current_tracker_continuous` | `[M,6,9]` | 当前六点 position + rotation6D |
| `previous_pose_target_clean` | `[M,144]` | 当前 `C_n` 下上一帧 GT Pose |
| `previous_head_position_current_ref` | `[M,3]` | 上一 Head 在当前 `C_n` 的位置 |
| `target_joints_head_ref` | `[M,24,3]` | 当前几何监督 |
| `target_root_position_head_ref` | `[M,3]` | 当前 Root 几何监督 |
| `target_root_yaw_world` / `target_hip_height` | `[M]` | Root 监督 |
| `previous_contact_target` / `contact_target` | `[M,2]` | 接触监督 |
| `joint_offsets_parent` | `[M,24,3]` | 骨架 offset |
| `joint_rest_local_rotations_6d` | `[M,24,6]` | rest local rotation |
| `task_seed` / `current_frame` | `[M]` | 稳定任务标识和绝对当前帧 |

Dataset 输出：

```text
x                       [B,144]
motion_context          [B,10,144]
core_tracker_context    [B,11,54]
current_tracker_raw     [B,6,10]
tracker_available       [B,6]
```

`current_tracker_raw` 通道为 position `0:3`、rotation6D `3:9`、available `9`。
不可用 Tracker 的前 9 维为零。训练 sampler 只以 1:1 产生核心三点和全部六点，
模型不接收全局 scenario id。

## Normalizer

```text
pose_mean.pt / pose_scale.pt          [144]
tracker_mean.pt / tracker_std.pt      [6,9]
predictor_sparse_mean.pt / predictor_sparse_std.pt [54]
```

Predictor 与 DiT 共用 Pose normalizer。Dataset 加载 Task Store 时才归一化。

## Predictor

`RealtimePosePredictor` 输入 `[B,10,144]` 与 `[B,11,54]`，输出
`[B,11,144]`。10 个 motion token 后拼接 11 个零 prediction slot；motion query
先对 tracker token 做 cross-attention，再进入 4 层 TransformerEncoder。输出仅
解码 prediction slots。

监督覆盖 11 帧：normalized rotation6D MSE、SO(3) rotation velocity、
Head-aligned FK position 与 FK joint velocity，权重均为 1。训练只有一个阶段，
每个 batch 均匀采样 `fr in [0,30]`；前 `fr` 步 `eval + no_grad`，且每步只回填
horizon 0，最后一步恢复训练模式并反传。30Hz 下的 30 步闭环对应 1 秒。

默认训练 100,000 步，AdamW 使用 `lr=3e-4`、`weight_decay=1e-4`；完成 50,000
步后学习率除以 30。checkpoint 使用同一步号的 `model/ema/optXXXXXXXXX.pt`
三件套，默认保留最近 3 组，`checkpoint_max_keep=0` 时不清理；
`model_latest.pt` 固定写入最新 EMA 推理权重，不参与清理。`--resume_checkpoint
latest` 从最近的带步号 `model*.pt` 恢复模型、optimizer、EMA 与 step。

## 单帧 DiT 与冻结边界

DiT diffusion state、普通 noise、模型输出和 sample 均为 `[B,144]`。公开条件：

```text
motion_context           [B,10,144]
predictor_pose_horizon         [B,11,144]
current_joint_condition  [B,24,10]
```

`predictor_pose_horizon[:,0]` 是 Predictor current prior，`[:,1:]` 是未来 10 帧条件。每层
只更新当前 24 个 joint token：先做 spatial self-attention，再由每个当前关节
对该关节的 `[-10..-1,1..10]` 20 帧上下文做 temporal cross-attention，最后
执行 timestep AdaLN MLP。输出 Pose `[B,144]` 和 contact logits `[B,2]`。

DiT 训练中的 Predictor 始终 `eval()`、`requires_grad=False` 且在 `torch.no_grad()`
中执行。DiT optimizer、EMA 和 checkpoint 不含 Predictor 权重；`args.json` 记录
`predictor_model_path`。冻结 Predictor 与 DiT 始终共享 Task Store 中同一份干净、
完整的 10 帧历史，不添加人工历史扰动。部署历史的时间累积误差由 Predictor
单阶段训练中的 0～30 步闭环回填建模。

## IK-Inpainting

IK 从反归一化的 `predictor_pose_horizon[:,0]` 初始化。Head、双手始终 available；
直接 Tracker 旋转先写入 IK。双臂始终求解；有 Hip 时求解躯干；Hip 与对应
Foot 同时 available 时才求解该腿。Foot available 但 Hip unavailable 时只写入
Foot 直接旋转并提供 Foot position condition。

单帧 inpainting：

```text
pose           [B,144]
valid          [B,24]
release_level  [B,24]
known_noise    [B,144]
```

available Tracker 的 source reliability 固定为 1。中间 IK 关节按约束类型、
FABRIK residual 与校准值计算 confidence；`release_level =
sin((1-confidence)*pi/2)`。普通 diffusion noise 与 known noise 独立，known noise
在完整 DDIM 轨迹中固定。

最终 projection 先把全部 rotation6D 投影到 SO(3)，再仅对所有 available Tracker
的直接关节旋转执行硬覆盖。IK 中间关节不硬覆盖。

## Runtime 与评估

Runtime 固定每秒调用 30 次。初始化必须提供 10 帧完整 world/deployed Pose history，以及核心三点
offset `-11..-1` 的 11 帧 Tracker；也接受已经包含当前 offset 0 的 12 帧形式。
每步只读取当前六点 Tracker，依次执行 Predictor、IK、单帧 DDIM、hard projection、
Head-anchored resolver，并把 deployed 当前 Pose 追加到 history。60/90Hz 显示插值由
Python 模型输出之后的客户端完成。

`RuntimeStepResult` 主要字段：

```text
predictor_pose_horizon    [11,144]
raw_pred_pose       [144]
deployed_pred_pose  [144]
inpaint_confidence  [24]
contact_logits      [2]
current_head_yaw_world scalar
```

长序列评估用 source 帧 0～10 初始化，从帧 11 开始闭环运行；按照 RPM P2，帧
11～29 只用于预热，正式指标从帧 30 开始统计。之后的 Pose history 只使用
deployed 输出，Tracker 只读取当前帧；`--max_frames` 只限制预热后的计分帧数。
`eval.evaluate_realtime_pose_predictor`
可在不加载 DiT 的情况下独立报告普通 RPM-P2/MC 指标：前 22 个 SMPL 关节的
parent-local axis-angle MPJRE、世界关节 MPJPE、MPJVE、预测 Jitter 与 GT Jitter。
指标先按每条序列计算，再对有效序列等权平均；不足 2 帧时 MPJVE 为 `null`，不足
4 帧时 Jitter 为 `null`。11 帧 horizon rotation 以及前 30/30 帧后的闭环误差仅作为
Predictor 诊断。组合评估的静态配置覆盖 core only、
core+Hip、core+单脚、core+双脚、core+Hip+单脚和 all six 共 8 种，并分别报告
DiT raw 与 DiT deployed 的同组 RPM-P2/MC 指标；全局 Predictor-only 基线只计算一次。core only 与 all six
标记为训练见过的配置。`--tracker_configs` 可选择评估子集；基础报告只运行
core only 与 all six，正式完整报告运行全部 8 种。

`predictor_sparse_*` 统计键、`predictor_sparse_mean.pt/std.pt`、模型参数名与
`predictor_pose_horizon` 都是当前唯一契约；旧 `rpm_*` Task Store、normalizer、
checkpoint 和 DiT checkpoint 不兼容，需要按当前链路重新生成或重新训练。
