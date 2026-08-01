# RealPose140 Contract

本文档是当前 Python 本地主链路的数据结构唯一说明。代码和产物不保存 schema 名称、版本或表示方式元数据；结构调整时直接同步修改本文档与实现。

## 时序任务

- 历史长度：60 帧。
- 当前目标：第 61 帧。
- `seq_len = 61`，`target_start = 60`，`target_length = 1`。

## Source

Source 保存从 AMASS 转换得到的完整运动缓存：

- `body_pose_body_fbx_local_delta_6d`: `[T, 144]`，24 个关节的 body.fbx local delta rotation6D。
- `root_pos_world`: `[T, 3]`，其中 world y 固定为 0。
- `root_yaw`: `[T]`。
- `root_heading_delta_sincos`: `[T, 2]`。
- `root_delta_xz_ref`: `[T, 2]`。
- `pelvis_height`: `[T, 1]`，等于 `joints_world[:, 0, 1]`。
- `tracker_pos_world`: `[T, 6, 3]`。
- `tracker_rot_world_6d`: `[T, 6, 6]`。
- `joints_world`: `[T, 24, 3]`。
- `joint_offsets_parent`: `[24, 3]`。
- `joint_rest_local_rotations_6d`: `[24, 6]`。
- `stationary_prob_5`: `[T, 5]`。

可选的 `body_fbx_rest_json` 是 rest 文件来源路径。Source 的 `metadata` 必须包含与实际数组一致的 `frames` 和 `target_fps`；当前主链路只接受 `target_fps = 60`。其余只保存路径、镜像标记、Tracker 顺序和 stationary 关节列表等来源信息，不保存结构名称或版本。

## 140 维姿态

- `0:138`：Pelvis 之外 23 个关节的 rotation6D，每个关节 6 维。
- `138:140`：Root yaw 相对当前 Head yaw 的 `[sin, cos]`。
- 所有关节旋转都表达在当前 Head yaw 参考系中。
- Pelvis 位移、高度、stationary 概率和 Tracker 不进入 140 维姿态。

## Task

持久化 task 的核心条件与目标：

- `pose_history`: `[60, 140]`。
- `tracker_window`: `[61, 6, 12]`。
- `current_target`: `[140]`。
- `known_target`: `[140]`。
- `known_mask`: `[140]`，rotation6D 按完整关节原子化，Root yaw 的 sin/cos 同时已知或未知。
- `valid_frame_mask`: `[60]`。

用于损失、重建与评估的辅助数组：

- `target_joints_head_ref`、`prev_joints_head_ref`: `[24, 3]`。
- `target_root_position_head_ref`: `[3]`。
- `target_root_yaw_world`、`target_hip_height`、`current_head_yaw_world`、`floor_y`: 标量。
- `current_head_position_world`: `[3]`。
- `joint_offsets_parent`: `[24, 3]`。
- `joint_rest_local_rotations_6d`: `[24, 6]`。
- `configured`、`measured_valid`、`missing_age`: `[61, 6]`。
- `scenario`、`source_path`、`start_frame`、`target_start`、`target_length`、`valid_length`、`source_frames`、`seq_len`、`rollout_step`、`max_rollout_steps`: 来源与窗口信息。

Task manifest 必须记录 `source_frames`、`target_fps`、`is_mirrored`、`max_rollout_steps` 和 `rollout_task_paths`。其中 `rollout_task_paths` 的数量固定为 `max_rollout_steps - 1`。

Tracker 最后一维：

- `0:3`：位置。
- `3:9`：rotation6D。
- `9`：configured。
- `10`：measured valid。
- `11`：归一化 missing age。

## Normalizer

- `pose_mean.pt`、`pose_std.pt`：`[140]`。
- `tracker_mean.pt`、`tracker_std.pt`：`[6, 9]`，只统计 Tracker 的位置与 rotation6D。
- Tracker 无有效测量时，归一化后的前 9 维置零。
- `normalizer_meta.json` 只保存统计来源、样本数和数值计算信息。

## 模型接口

- 当前扩散状态：`[B, 140]`。
- 历史姿态：`[B, 60, 140]`。
- Tracker 条件：`[B, 61, 6, 12]`。
- 输出：`[B, 140]`。
