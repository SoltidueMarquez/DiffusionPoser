# RealPose Python Contract

本文档是当前 Python 主链路的数据契约。旧 task、旧 normalizer 和旧 checkpoint 与本契约不兼容，读取时直接报错，不提供兼容分支。

## 时间窗口与共同参考系

运行时维护最近 60 帧密集世界状态，模型只读取 10 个历史锚点和当前帧：

```text
history anchor indices = [0, 7, 13, 20, 26, 33, 39, 46, 52, 59]
frame offsets          = [-60, -53, -47, -40, -34, -27, -21, -14, -8, -1, 0]
```

Pose、Tracker 和 Head 路径必须使用同一组锚点。11 帧全部表达在当前参考系 `C_n`：

- 原点为当前 Head 的水平位置与当前 `floor_y`；
- 朝向为当前 Head yaw；
- Pose 世界旋转左乘当前 yaw 的逆旋转；
- Tracker 位置和旋转直接从世界状态变换到 `C_n`；
- Head 路径表示每个锚点相对当前 Head 原点和朝向的绝对位置与 yaw。

冷启动历史左侧锚点无效。所有无效锚点字段均在归一化后清为字面零，`window_valid_mask=False`；当前帧始终有效。

## Source

60 Hz realtime source 保持以下字段：

- `body_pose_body_fbx_local_delta_6d: [T,144]`
- `root_pos_world: [T,3]`
- `root_yaw: [T]`
- `pelvis_height: [T,1]`
- `tracker_pos_world: [T,6,3]`
- `tracker_rot_world_6d: [T,6,6]`
- `joints_world: [T,24,3]`
- `joint_offsets_parent: [24,3]`
- `joint_rest_local_rotations_6d: [24,6]`
- `stationary_prob_5: [T,5]`

Source 可以复用；task、normalizer 和 checkpoint 必须重建。

## Task Store

Task Store 是 task 生成阶段写入磁盘的未归一化数据。设样本数为 `M`、Source 数为 `S`，每个样本固定保存五套 Tracker 场景：

| shard 字段 | 形状 | 语义 |
|---|---:|---|
| `pose_window_clean` | `[M,11,144]` | 当前参考系下的干净 Pose 窗口 |
| `tracker_window_continuous` | `[M,11,6,9]` | 未拼接状态通道的 Tracker 位置与 rotation6D |
| `head_path_window` | `[M,11,5]` | 当前参考系下的 Head 路径 |
| `configured` / `measured_valid` | `[M,5,61,6]` | 五套场景的密集 61 帧 Tracker 状态 |
| `target_joints_head_ref` | `[M,24,3]` | 当前帧关节位置监督 |
| `target_root_position_head_ref` | `[M,3]` | 当前 Root 在当前 Head 参考系中的位置 |
| `target_root_yaw_world` | `[M]` | 当前 Pelvis forward 世界 yaw |
| `target_hip_height` | `[M]` | 当前 Pelvis 世界高度 |
| `current_head_yaw_world` | `[M]` | 当前 Head 世界 yaw |
| `current_head_position_world` | `[M,3]` | 当前 Head 世界位置 |
| `floor_y` | `[M]` | 当前地面世界高度 |
| `future_leg_target` | `[M,3,8,6]` | 未来 3 帧双腿 rotation6D |
| `contact_target` | `[M,2]` | 当前左右脚接触监督 |
| `source_index` / `start_frame` | `[M]` | Source 索引与密集历史起始帧 |

Store 根目录还包含 `generation_plan.jsonl` 与对应 SHA-256；每个 split 包含：

- `task_store.json`：`generation_plan_hash`、`split`、`sample_count`、`source_count`、`two_point_phase_counts`、`config_names`、`tracker_feature_dim`、`schema_fields` 和 `shards`；
- `sources.jsonl`：Source 索引、路径、帧数、帧率及镜像标记；
- `source_joint_offsets_parent.npy: [S,24,3]`；
- `source_joint_rest_local_rotations_6d.npy: [S,24,6]`；
- `shards/shard_*/`：上表各字段的独立 `.npy` 以及 normalizer 聚合使用的 `stats.npz`。

## 模型 Batch

Dataset 从 Task Store 选择一套 Tracker 场景，按虚拟会话起点重新累计状态并抽取 10 个历史锚点和当前帧。以下形状均包含 batch 轴 `B`：

| 字段 | 形状 | 语义 |
|---|---:|---|
| `x` | `[B,11,144]` | 干净的 10 帧历史 Pose 与当前 GT，完整参与扩散；Dataset 不重复返回 Task Store 的 `pose_window_clean` 名称 |
| `history_pose_observation` | `[B,10,144]` | 历史 Pose 条件；训练时可扰动，部署时为真实历史预测 |
| `tracker_window` | `[B,11,6,13]` | 同步锚点的完整 Tracker 历史与当前 Tracker |
| `head_path_window` | `[B,11,5]` | 同步锚点在 `C_n` 下的 Head 路径 |
| `history_region_confidence` | `[B,10,5]` | 历史 Pose 逐帧、逐身体区域可信度 |
| `window_valid_mask` | `[B,11]` | 锚点有效性；当前帧恒为 `True` |
| `frame_offsets` | `[B,11]` | 固定真实帧偏移 |
| `configured` / `measured_valid` | `[B,11,6]` | 所选场景在同步锚点上的 Tracker 状态 |
| `d_off` / `d_on` | `[B,11,6]` | 从虚拟会话起点重新累计的掉线与在线帧数 |
| `current_tracker_raw` | `[B,6,13]` | 当前几何 Loss 与 hard projection 使用的未归一化 Tracker |
| `hard_rotation_state` | `[B,6]` | 当前帧 hard rotation 集合 |
| `joint_offsets_parent` | `[B,24,3]` | 目标骨架父子偏移 |
| `joint_rest_local_rotations_6d` | `[B,24,6]` | 目标骨架 rest local rotation6D |
| `target_joints_head_ref` | `[B,24,3]` | 当前帧关节位置监督 |
| `target_root_position_head_ref` | `[B,3]` | 当前 Root 在 `C_n` 下的位置监督 |
| `target_root_yaw_world` | `[B]` | 当前 Pelvis forward 世界 yaw 监督 |
| `target_hip_height` | `[B]` | 当前 Pelvis 世界高度监督 |
| `current_head_yaw_world` | `[B]` | 当前 Head 世界 yaw |
| `current_head_position_world` | `[B,3]` | 当前 Head 世界位置 |
| `floor_y` | `[B]` | 当前地面世界高度 |
| `future_leg_target` | `[B,3,8,6]` | 未来 3 帧双腿监督 |
| `contact_target` | `[B,2]` | 左右脚接触监督 |
| `history_length` | `[B]` | 当前虚拟会话可见的密集历史帧数 |
| `scenario_id` / `scenario` | `[B]` | 所选 Tracker 场景的索引与名称 |
| `start_frame` / `task_id` / `source_path` | `[B]` | 样本定位与追踪信息 |

模型内部及扩散采样输出为 `[B,11,144]`；外部 `reconstruct_batch` 和 `RuntimeStepResult` 只返回当前帧 `[B,144]`。

## 144D Pose 与历史扰动

每帧 Pose 由 24 个关节的全局 rotation6D 拼接而成。标准扩散作用于完整 `[B,11,144]` 窗口，但历史 Pose 只是可能含误差的参考信息，部署时只使用当前帧输出。因此 diffusion reconstruction loss 先对全部有效历史帧取平均，再令该历史平均项的总权重为 `0.1`、当前帧权重为 `1.0`；存在有效历史时二者除以 `1.1` 保持 loss 尺度稳定，冷启动没有有效历史时只计算当前帧。无效历史帧不参与 loss。

历史 Pose 扰动只修改独立条件 `history_pose_observation`，干净窗口仍作为扩散监督。扰动在反归一化后的 SO(3) 空间执行：

```text
history_noise_prob = 0.8
history_noise_min_deg = 2.0
history_noise_max_deg = 10.0
history_noise_temporal_rho = 0.95
history_noise_region_ratio = 0.75
history_noise_joint_ratio = 0.25
sigma(c) = 2° + (1-c) * 8°
```

时间相关系数按锚点实际帧间隔计算。验证集不添加合成噪声，当前 Pose、Tracker 和 Head 路径不添加此噪声。

## 13D Tracker

每个 Tracker 固定为：

```text
position3 + rotation6D + configured + measured_valid + d_off + d_on
```

通道偏移为 `0:3`、`3:9`、`9`、`10`、`11`、`12`。无效测量的前 9 维严格清零。只有前 9 维连续量参与 normalizer；状态与持续时间不参与。`d_off/d_on` 输入模型前除以 60，物理状态仍使用整数帧数。

Task store 保留密集的 `configured/measured_valid: [M,5,61,6]`。Dataset 从虚拟会话起点重新累计 `d_on/d_off` 和 hard state，再抽取模型锚点，不能直接继承被裁掉历史的持续时间。

`TrackerReliabilityConfig` 由模型持有；Runtime 必须直接读取模型配置，不提供独立覆盖入口。`duration_cap`、kappa 与 hard rotation gate 始终使用同一套参数。

## Head 路径

每个锚点节点为：

\[
p_k^{head}=[x_k^{C_n},z_k^{C_n},h_k,\sin(\psi_k-\psi_n),\cos(\psi_k-\psi_n)].
\]

当前节点恒为 `[0,0,h_n,0,1]`。该字段表示过去一秒内 Head 走过的绝对路径，不是锚点间增量。

## Normalizer

- `pose_mean.pt`、`pose_scale.pt`: `[144]`，其中 `pose_scale = pose_std + eps`，所有 Pose 物理空间转换统一使用该尺度
- `tracker_mean.pt`、`tracker_std.pt`: `[6,9]`
- `head_path_xz_mean.pt`、`head_path_xz_std.pt`: `[2]`
- `head_height_mean.pt`、`head_height_std.pt`: 标量

Head 路径的 XZ 与高度分别归一化，sin/cos 保持原值。normalizer 元数据必须与 task 的 generation-plan hash 一致。

## 时空 DiT 与投影

- `WindowObservationEncoder` 逐帧输出 Tracker state、position、rotation token，保留 11 帧时间轴，不使用历史 GRU summary；
- 每层依次执行 Tracker cross-attention、24 关节 Spatial Self-Attention、同关节 11 帧 Temporal Self-Attention；
- Temporal Attention 使用 causal mask，当前帧能读取全部历史，历史帧不能读取未来；
- 训练时 `model_kwargs` 只传递一个 `y` 字典，模型预测、Loss 和 hard projection 从同一份条件与监督数据读取，不在顶层重复条件字段；
- 历史 Pose、Tracker 窗口、Head 路径和置信度由 `prepare_conditioning()` 每个目标帧编码一次，在 DDIM 步间复用；
- future-leg 与 contact head 只读取最后当前帧 token；
- 所有 11 帧执行 rotation6D 到 SO(3) 投影，hard Tracker rotation 只替换最后当前帧；
- 训练阶段不执行模型 rollout；长序列闭环只用于评估，运行时每步只追加当前部署预测，不重写过去历史。
