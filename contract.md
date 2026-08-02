# RealPose Python Contract

本文档是当前 Python 主链路的数据结构唯一说明。旧 task、normalizer 与 checkpoint 不兼容，读取时必须明确报错，不提供兼容路径。

## 时序与参考系

- 历史长度固定为 60 帧，当前目标为第 61 帧。
- `pose_history` 与 `tracker_history` 统一表达在上一帧 Head-yaw/floor 参考系 \(C_{n-1}\)。
- `current_target` 与 `current_tracker` 表达在当前参考系 \(C_n\)。
- rollout 最多物化 4 个连续目标；下一步历史只能追加 `deployed_pred_xstart.detach()`，不得读取后续 GT pose history。
- 冷启动时三类历史统一左侧补零，补零帧对应 `valid_frame_mask=False`；padding 是归一化后模型输入空间中的字面量零。
- 虚拟会话首帧从零状态重新推进 `d_off/d_on` 与 hard hysteresis，不得继承被裁掉历史的持续时间。首帧 Head trajectory 的平移/yaw 增量为零，`cos(delta_yaw)=1`。
- rollout 的有效历史长度按 `min(60,H+s)` 增长；同一 rollout 必须沿同一虚拟会话状态线连续推进，不能逐步重置 duration。

参考系原点为当前 Head 水平位置与地面：

\[
o_n^W=[p^W_{H,n,x}, floor_y, p^W_{H,n,z}]^\top.
\]

Head forward 水平投影退化时沿用上一合法 yaw。

## Source

原始 AMASS realtime source 保持现有缓存字段：

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

只接受 60 Hz source。Source 本身可以复用；task 及其下游产物必须重建。

## Batch

| 字段 | 形状 | 语义 |
|---|---:|---|
| `x` / `current_target` | `[B,144]` | 当前完整扩散状态/监督，位于 \(C_n\) |
| `pose_history` | `[B,60,144]` | 过去 deployed pose，位于 \(C_{n-1}\) |
| `tracker_history` | `[B,60,6,13]` | 过去 Tracker，位于 \(C_{n-1}\) |
| `current_tracker` | `[B,6,13]` | 归一化当前 Tracker，位于 \(C_n\) |
| `current_tracker_raw` | `[B,6,13]` | 未归一化当前 Tracker，供投影和几何损失 |
| `trajectory_history` | `[B,60,5]` | 过去 Head trajectory |
| `current_trajectory` | `[B,1,5]` | 当前 Head trajectory |
| `valid_frame_mask` | `[B,60]` | 历史有效帧 |
| `history_length` | `[B]` | 本步有效历史帧数，范围 `[0,60]` |
| `hard_rotation_state` | `[B,6]` | 本帧固定 hard 集合 |
| `joint_offsets_parent` | `[B,24,3]` | 目标骨架父子偏移 |
| `future_leg_target` | `[B,3,8,6]` | 未来 3 帧双腿监督 |
| `contact_target` | `[B,2]` | 左右脚接触监督 |
| `raw_pred_xstart` | `[B,144]` | 投影前预测 |
| `deployed_pred_xstart` | `[B,144]` | SO(3) 与 hard rotation 投影后预测 |

新链路不存在 `known_target`、`known_mask`、`inpaint_mask` 或 `inpaint_cond`。

## 144D pose

144 维由 24 个 joint 的全局 rotation6D 顺序拼接，全部表达在目标参考系中。扩散前向过程对完整 144D 加噪；hard joint 仍必须在 raw 分支接受生成监督。

## 13D Tracker

每个 Tracker 的固定顺序为：

`position3 + rotation6D + configured + measured_valid + d_off + d_on`

对应 offset：

- `0:3`: position
- `3:9`: rotation6D
- `9`: configured
- `10`: measured_valid
- `11`: d_off
- `12`: d_on

无效测量的前 9 维必须严格清零。只有前 9 维连续量参与 Tracker normalizer；配置、有效性、duration 和 hard 状态不参与统计。`d_off/d_on` 对模型输入除以 60，物理状态仍保存整数帧数。

## Head trajectory

\[
\tau_n=[\Delta x^{C_{n-1}},\Delta z^{C_{n-1}},h_{head},\sin\Delta\psi,\cos\Delta\psi].
\]

`delta xz` 与 `sin/cos` 不缩放；只有 Head 相对地面高度使用独立 normalizer。Head position 不进入普通 position measurement token。

## 可靠性与 hard rotation gate

默认配置：

- `D_warm_pos=15`
- `D_warm_rot=15`
- `D_h=15`
- duration cap `60`

\[
\kappa_i^a=c_i v_i\min(1,d_i^{on}/D_{warm}^a),
\qquad
\rho_r^a=1-\prod_i(1-A_{r,i}^a\kappa_i^a).
\]

Head rotation 始终 hard。其他 Tracker 的 hard 状态直接定义为 `configured & measured_valid & (d_on >= D_h)`：掉线或取消配置时立即退出，恢复连续有效 15 帧后进入。恢复期的渐进影响只由 \(\kappa\) 负责，hard 判定不依赖上一帧状态。本帧 hard 集合在所有 DDIM step 间保持不变。

## 五类训练场景

| ID | 名称 | 规则 |
|---:|---|---|
| 0 | `fixed_six` | 六点持续配置有效 |
| 1 | `fixed_three` | 仅 Head 与双手持续配置有效 |
| 2 | `three_to_six` | Hip 与双脚在切换帧恢复 |
| 3 | `six_to_three` | Hip 与双脚在切换帧取消配置 |
| 4 | `two_point_dropout_reconnect` | 两个不同非 Head Tracker 同步掉线重连 |

默认采样权重各 `0.2`，CLI 可覆盖。切换 target 位于事件后 0～14 帧。两点掉线持续 5～30 帧，掉线中与重连后 0～14 帧按 1:1 采样。可靠度只由配置、测量有效性和连续恢复时长决定。事件必须按 `global_seed + source_id` 在绝对时间线上确定，重叠窗口状态一致。

训练默认以 `cold_start_prob=0.1` 把样本替换为部分历史冷启动；命中时 `H` 在 `[0,59]` 均匀采样，否则使用完整 60 帧历史。验证集保持完整历史，长序列评估另外报告 `cold_start_0_59`、`steady_state_60_plus` 和全帧指标。

## mmap task store

当前 shard 至少包含：

- `pose_history: [M,60,144]`
- `current_target: [M,K,144]`
- `tracker_history_continuous: [M,K,60,6,9]`
- `current_tracker_continuous: [M,K,6,9]`
- `trajectory_history: [M,K,60,5]`
- `current_trajectory: [M,K,1,5]`
- `configured/measured_valid/d_off/d_on/hard_rotation_state: [M,5,60+K,6]`
- `history_head_yaw_world/current_head_yaw_world: [M,K]`
- `future_leg_target: [M,K,3,8,6]`
- `contact_target: [M,K,2]`
- 当前 FK、Root 和 source 索引辅助字段。

`task_store.json` 必须记录 `tracker_feature_dim=13`、五类 `config_names` 和完整 `schema_fields`。缺少任一新字段时直接拒绝读取。

## Normalizer

- `pose_mean.pt`、`pose_std.pt`: `[144]`
- `tracker_mean.pt`、`tracker_std.pt`: `[6,9]`
- `head_height_mean.pt`、`head_height_std.pt`: 标量

normalizer 元数据必须与 task generation-plan hash 一致。场景权重不参与 normalizer 统计。

## 模型与扩散输出

- DynamicObservationEncoder: `S:[B,6,D]`、`M_pos:[B,5,D]`、`M_rot:[B,6,D]`、`U:[B,6,D]`。
- RegionalMotionEncoder 只读取 past 条件，输出 global/pelvis/left-leg/right-leg temporal token 与 latent。
- TargetDiT 输出 raw `[B,144]`、future-leg `[B,3,8,6]`、contact logits `[B,2]`。
- `prepare_conditioning()` 每个目标帧只执行一次。
- Projected DDIM 默认每一步投影，支持 `all_steps`、`late_steps`、`final_step`。
- Resolver 只严格检查 hard Tracker rotation，不对 soft Tracker 做硬约束。
