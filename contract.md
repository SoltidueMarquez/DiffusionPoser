# RealPose140 Contract

本文档是当前 Python 本地主链路的数据结构唯一说明。代码和产物不保存 schema 名称、版本或表示方式元数据；结构调整时直接同步修改本文档与实现。

## 时序任务

- 历史长度：60 帧。
- 当前目标：第 61 帧。
- 一个基础窗口连续保存最多 4 个目标帧，因此默认覆盖 64 帧；训练可以请求 1～4 步。
- 每个 source 默认在所有合法起点中确定性保留 20 个窗口。source、起点和 task 均按稳定键排序。

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

Source 的 `metadata` 必须包含与实际数组一致的 `frames` 和 `target_fps`；当前主链路只接受 `target_fps = 60`。

## 140 维姿态

- `0:138`：Pelvis 之外 23 个关节的 rotation6D，每个关节 6 维。
- `138:140`：Root yaw 相对当前 Head yaw 的 `[sin, cos]`。
- 所有关节旋转都表达在当前目标帧的 Head yaw 参考系中。
- Pelvis 位移、高度、stationary 概率和 Tracker 不进入 140 维姿态。

## 生成计划

`generation_plan.jsonl` 在数据物化前写入。其 SHA-256 保存于 `generation_plan.sha256`，绝对路径和时间戳不参与计划内容。`task_id` 只由 split、source ID 和起始帧决定。

每个基础窗口包含五套固定索引的 Tracker 配置：

0. `fixed_six`
1. `fixed_three`
2. `three_to_six`
3. `six_to_three`
4. `dropout`

同一配置贯穿四个 rollout 窗口。切换或掉线事件位于这些窗口的公共帧内；dropout 从六点配置中选择 1～2 个非 Head Tracker，持续 1～60 帧。Head 始终 configured 且 measured valid。`missing_age` 使用额外 60 帧前缀计算，最大为 60。

## mmap Task Store

每个 split 的 `task_store.json` 记录 generation-plan hash、样本数、source 数、最大 rollout 步数和 shard 列表。默认每个 shard 包含 4096 个基础窗口，数据为未压缩 `.npy`，Dataset 使用 `mmap_mode="r"` 读取。

一个含 `M` 个窗口、最大四步的 shard 包含：

- `pose_history`: `[M, 60, 140] float32`，只保存 step 0。
- `current_target`: `[M, 4, 140] float32`。
- `tracker_continuous`: `[M, 4, 61, 6, 9] float32`，未应用掉线。
- `full_known_target`: `[M, 4, 140] float32`，保存六个 Tracker 全有效时的硬条件值。
- `configured`: `[M, 5, 64, 6] uint8`。
- `measured_valid`: `[M, 5, 64, 6] uint8`。
- `missing_age`: `[M, 5, 64, 6] uint8`。
- `target_joints_head_ref`、`prev_joints_head_ref`: `[M, 4, 24, 3] float32`。
- `target_root_position_head_ref`: `[M, 4, 3] float32`。
- `target_root_yaw_world`、`target_hip_height`、`current_head_yaw_world`、`floor_y`: `[M, 4] float32`。
- `current_head_position_world`: `[M, 4, 3] float32`。
- `source_index`、`start_frame`: `[M] int32`。

`joint_offsets_parent` 与 `joint_rest_local_rotations_6d` 按 source 各保存一次，由 `source_index` 引用。shard 使用 `open_memmap` 写入临时目录，完成后原子重命名。

## Dataset 与 batch

Dataset 接收 `TaskRequest(task_index, config_index, rollout_steps)`。普通整数索引等价于配置 0、单步请求。每个 worker 最多缓存两个打开的 shard。

Dataset 从所选配置切出 61 帧状态，把连续量与 `configured`、`measured_valid`、归一化 `missing_age` 拼成 `[61, 6, 12]`。无效测量的前 9 维严格置零。当前帧 `measured_valid` 映射为 `[140] known_mask`，`known_target` 的 unknown 位置在归一化后再次置零。`valid_frame_mask` 固定为 `[60]` 全 True。

对外 batch 保持：

- 当前扩散状态：`[B, 140]`。
- 历史姿态：`[B, 60, 140]`。
- Tracker 条件：`[B, 61, 6, 12]`。
- `known_target`、`known_mask`：`[B, 140]`。
- 输出：`[B, 140]`。

rollout 子项不保存或返回 GT `pose_history`。训练和重建均从 step 0 的历史持续推进，保留所有先前预测。

## Normalizer

- `pose_mean.pt`、`pose_std.pt`: `[140]`。
- `tracker_mean.pt`、`tracker_std.pt`: `[6, 9]`。
- Pose 只统计每个基础窗口的 `pose_history + current_target[:, 0]`。
- Tracker 只统计未遮挡的 `tracker_continuous[:, 0]`，六个 Tracker 全部计入。
- 每个 shard 保存 float64 `sum/sumsq/count`，normalizer 按 shard 编号合并，std 使用 population 定义。
- `normalizer_meta.json` 保存 generation-plan hash、split、样本数和观察数；训练只校验 task 与 normalizer 的 plan hash 一致。
- 场景权重、rollout 概率和 rollout 长度不影响 normalizer。
