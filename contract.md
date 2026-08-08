# RealPose Python Contract

本文档是当前 Python 主链路的数据结构唯一说明。旧 task、normalizer 与 checkpoint 不兼容，读取时必须明确报错，不提供兼容路径。

## 时序与参考系

- 历史长度固定为 60 帧，当前目标为第 61 帧。
- `pose_history` 与 `tracker_history` 统一表达在上一帧 Head-yaw/floor 参考系 \(C_{n-1}\)。
- `current_target` 与 `current_tracker` 表达在当前参考系 \(C_n\)。
- rollout 最多物化 15 个连续目标，默认仍为 4；下一步历史只能追加 `deployed_pred_xstart.detach()`，不得读取后续 GT pose history。
- 后续 `K-1` 帧主 loss 默认以 `rollout_frame_weighting=uniform` 保持原算术平均；受控实验可用 `linear_late` 按 `1…K-1` 归一化加权。该参数不改变第一帧主 loss、joint/rotation velocity、foot-slide、时序状态或 task schema。
- 冷启动时三类历史统一左侧补零，补零帧对应 `valid_frame_mask=False`；padding 是归一化后模型输入空间中的字面量零。
- 虚拟会话首帧从零状态重新推进 `d_off/d_on` 与 hard hysteresis，不得继承被裁掉历史的持续时间。首帧 Head trajectory 的平移/yaw 增量为零，`cos(delta_yaw)=1`。
- rollout 的有效历史长度按 `min(60,H+s)` 增长；同一 rollout 必须沿同一虚拟会话状态线连续推进，不能逐步重置 duration。

参考系原点为当前 Head 水平位置与地面：

\[
o_n^W=[p^W_{H,n,x}, floor_y, p^W_{H,n,z}]^\top.
\]

Head forward 水平投影退化时沿用上一合法 yaw。

`target_root_yaw_world` 与最终 `predicted_root_yaw_world` 统一表示 Pelvis forward
的世界 heading，并 wrap 到 `[-π,π)`。它们不得直接读取 source 的 Actor Root
分解量 `root_yaw`；必须满足：

\[
wrap(current\_head\_yaw\_world + decode(current\_target).pelvis\_heading)
= target\_root\_yaw\_world.
\]

当 Pelvis local residual 接近 180° 时，source Actor Root heading 与该 Pelvis
heading 可以相差约 π；这是两种语义的差异，不是 checkpoint 误差。

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

TAID 方案草案中曾写过 15D Tracker，但当前稳定 Python/Unity 契约明确保持
13D：不新增 velocity 或 `delta_e` 字段。位置/旋转 innovation 及其时间差只从
`tracker_history`、`pose_history`、当前 13D Tracker 和 Head trajectory 内部派生。

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

训练默认以 `cold_start_prob=0.1` 把样本替换为部分历史冷启动；命中时 `H` 在 `[0,59]` 均匀采样，否则使用完整 60 帧历史。训练实验可用 `cold_start_history_weights` 覆盖 `[0]`、`[1,4]`、`[5,14]`、`[15,29]`、`[30,59]` 五个历史桶的概率，并用 `cold_start_scenario_weights` 只覆盖部分历史样本的场景概率；未提供时必须保持上述默认分布。验证集保持完整历史，长序列评估另外报告 `cold_start_0_59`、`steady_state_60_plus`、`by_startup_phase` 和全帧指标。

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

`task_store.json` 必须记录 `tracker_feature_dim=13`、五类 `config_names`、完整 `schema_fields` 以及 `two_point_phase_counts`。后者统计两点掉线与重连样本，数量必须与 `sample_count` 一致并满足近似 1:1。缺少任一新字段时直接拒绝读取。

`K` 允许 `1…15`，但 `seq_len/max_seq_len` 始终为 61；K15 只表示离线 task 额外物化基础目标之后的14个连续目标和共75帧 Tracker 状态，不扩展 TargetDiT 输入长度。生成器的默认 `K=4` 保持旧行为。限量生成默认选择排序列表前缀；受控实验可使用 `limit_selection=stratified`，按 seed 从排序后的 source 等距区间确定性抽取。

## Normalizer

- `pose_mean.pt`、`pose_std.pt`: `[144]`
- `tracker_mean.pt`、`tracker_std.pt`: `[6,9]`
- `head_height_mean.pt`、`head_height_std.pt`: 标量

normalizer 元数据必须与 task generation-plan hash 一致。场景权重不参与 normalizer 统计。K15 配对实验允许逐字节复用正式 K4 的六个统计文件，但必须先证明两份 task 来自同一 source、均满足144D Pose/13D Tracker/五场景契约，并在新元数据中绑定 K15 generation-plan hash、记录原 metadata 与全部统计文件 SHA256；Dataset 仍执行严格 plan-hash 匹配，不提供 mismatch 绕过。

## 模型与扩散输出

- DynamicObservationEncoder: `S:[B,6,D]`、`M_pos:[B,5,D]`、`M_rot:[B,6,D]`、`U:[B,6,D]`。
- RegionalMotionEncoder 只读取 past 条件，输出 global/pelvis/left-leg/right-leg temporal token 与 latent。
- TargetDiT 输出 raw `[B,144]`、future-leg `[B,3,8,6]`、contact logits `[B,2]`。
- `prepare_conditioning()` 每个目标帧只执行一次。
- Projected DDIM 默认每一步投影，支持 `all_steps`、`late_steps`、`final_step`。
- Resolver 只严格检查 hard Tracker rotation，不对 soft Tracker 做硬约束。

## TAID B0～B6 条件契约

TAID 由 `--taid_ablation` 显式选择，默认 `B0`。B0 不创建 TAID 参数，保持旧
TargetDiT state dict 与前向路径；B1～B6 仍输出完整 raw `[B,144]`，不改变
Diffusion schedule、Projected DDIM 或 Unity 稳定输入输出。

角色由 13D 中的 `configured/measured_valid/d_on` 确定：未配置、Missing、
Uncertain、Anchor 四种身份不合并。Head 强制为 Anchor。第一版固定：

- `K0=5`、`K1=15`、`KR=15`；
- `alpha=valid*clip((d_on-K0)/(K1-K0),0,1)`；
- `beta=valid*(1-alpha)*min(1,d_on/KR)`；
- `region_coverage=1-prod(1-alpha*coverage)`。

Anchor Prior 只读 deployed pose history、Head trajectory、region coverage，以及
逐 Tracker 独立融合的 `[state_token, position_token, rotation_token,
history_summary]`。其中 `history_summary:[B,6,D]` 来自已存在且冻结的
DynamicObservationEncoder 60帧 Tracker history GRU；四类 token 先在每个 Tracker
内部融合为 `[B,6,D]`，再乘 `alpha` 并除以现有 `sum(alpha)`。六个结果不再直接
求和，而是按 Head、LeftHand、RightHand、Hip、LeftFoot、RightFoot 的固定顺序
展平为 `[B,6D]`，经无 bias 的 `anchor_slot_projection:[D,6D]` 压回 `[B,D]`。
该投影初始化为六个单位矩阵横向拼接，因此训练开始时与旧加权平均逐元素等价，
随后才允许不同身体槽位学习不同作用。Uncertain/Missing/未配置 Tracker 的当前和
历史观测都不得通过共享聚合旁路进入。Prior 输出：

- `prior_pose_raw/model: [B,144]`；
- `prior_root_head: [B,4]`，依次为 Head-relative Actor Root `xyz+yaw`；
- `prior_contact_logits: [B,2]`；
- `prior_joint_velocity_head: [B,24,3]`，仅用于 B1 训练期速度监督；
- `region_coverage: [B,5]`。

`prior_root_head.xyz` 是 TAID 内部辅助状态，只用于 Prior 内部 FK、FK innovation
和 TargetDiT 条件，不得覆盖部署 Root。部署 Root 的唯一真值仍是 hard rotation
projection 后的 144D pose，经 Head-Anchored World Resolver 解析得到的结果：Head
位置严格对齐当前 Head Tracker，Actor Root 的 Y 位于 floor。`prior_root_head.yaw`
仍严格由 `prior_pose_raw` 的 Pelvis heading 派生，不存在独立 yaw head。

B1 的 `prior_fk_loss` 必须使用投影后的 `deployed_pred_xstart` 并调用与 runtime
一致的 Torch Resolver；内部 `prior_root_head.xyz + prior_pose_raw` FK 只产生
`prior_internal_fk_loss`、Root gap 和 joint gap 诊断，不进入总损失。这是当前实现
相对“让 Prior Root xyz 辅助部署 Resolver”提案的受控差异，目的是保持既有
144D/Unity 稳定接口和 Head/floor 世界解析契约。

当前跨 Tracker 聚合固定为六槽投影，`args.json` 必须记录
`taid_prior_tracker_aggregation=fixed_slots`。本轮只改变跨 Tracker 聚合容量，
不改变 Tracker-history 编码、velocity监督或任何 loss 权重。29Z及更早 B1
checkpoint 不含 `anchor_slot_projection`，禁止恢复或续训；正式 B0 仍可按缺失
全部 TAID 参数的既有规则初始化 B1。

Posterior innovation 固定为 `[B,6,6]`：位置使用米，旋转使用 SO(3) Log 的弧度。
Head residual 恒为零；无效测量的 residual、`delta_e` 和 innovation token 必须严格
为零。上一帧位置 residual 先按 `current_trajectory` 的 Head yaw 增量从
`C_(n-1)` 旋到 `C_n`。第一版尺度按 Head/LHand/RHand/Hip/LFoot/RFoot 固定为：

- position `(1.0,0.25,0.25,0.20,0.20,0.20)` 米；
- rotation `(1.0,0.50,0.50,0.35,0.35,0.35)` 弧度；
- 每维使用 `3*tanh(value/scale/3)` 确定性截断；
- shared MLP 为 `12→128→64`，再适配到 TargetDiT latent。

消融含义固定：B1=Prior only；B2=Prior absolute condition；B3=加 Uncertain
绝对观测；B4=以 FK innovation 替代绝对观测；B5=加固定区域路由；B6=把
Uncertain hard 权重换成互补连续 `alpha/beta`。B1/B2 的直接 posterior
`beta=0`，B3～B5 使用 Uncertain hard beta，B6 使用连续 beta。固定路由为 Hip→Torso（弱到
双腿）、Hand→对应 Arm（弱到 Torso）、Foot→对应 Leg（contact 时弱到
Torso），Head 不进入 innovation。路由仅限制直接条件注入，不截断全身
self-attention。

B1 仅训练 Prior；B2～B6 冻结 Prior，并对用于 FK/条件的 Prior 输出
stop-gradient。每个目标帧只执行一次 Prior/innovation，全部 DDIM step 复用。
Python runtime 可只读导出 `taid_prior_root_head` 和 `taid_prior_joints_head`；B0
必须报告不可用，不能以零误差代替。长序列评估按 condition、history phase 和
startup phase 汇总 Prior Root/FK 与 deployed Resolver 的差距。
`init_checkpoint` 只允许 B0→B1 或 B1→B2～B6；`resume_checkpoint` 必须与当前
ablation 完全相同。
