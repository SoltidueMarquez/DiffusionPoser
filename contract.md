# RealPose Python Contract

本文档是当前 Python 主链路的数据契约。source、Task Store 和 normalizer 都由调用方传入的实际目录定义，不读取 manifest、meta、hash 或 latest 指针。

## 目录契约

以 RPM-P2 为例，调用方传入的目录应直接落到实际协议产物：

```text
source_dir/
  HumanEva/...
  CMU/...
  M/...

task_dir/
  train/shards/shard_*/...
  test/shards/shard_*/...

normalizer_dir/
  pose_mean.pt
  pose_scale.pt
  tracker_mean.pt
  tracker_std.pt
  head_path_xz_mean.pt
  head_path_xz_std.pt
  head_height_mean.pt
  head_height_std.pt
```

例如 `task_dir=artifacts/tasks/RPM-P2`、`normalizer_dir=artifacts/normalizer/RPM-P2`。Dataset 再根据 `split=train/test` 进入对应子目录；调用方不能只传 `artifacts/tasks` 或 `artifacts/normalizer` 并期待程序自动选择 RPM 协议或最新产物。

## 时间窗口与共同参考系

运行时维护最近 60 帧密集世界状态。Tracker/Head 条件由 10 个历史锚点和当前帧组成，扩散目标由当前帧和未来 10 帧组成：

```text
dense tracker condition length = 61
history anchor count           = 10
condition window length        = 11
future frame count             = 10
diffusion horizon length       = 11
model token length             = 21

history anchor indices = [0, 7, 13, 20, 26, 33, 39, 46, 52, 59]
model frame offsets    = [-60, -53, -47, -40, -34, -27, -21, -14, -8, -1,
                           0,   1,   2,   3,   4,   5,   6,   7,  8,  9, 10]
```

10 个历史 Pose 与 11 个目标 Pose 全部表达在当前参考系 `C_n`；Tracker 和 Head 路径只保存 10 个历史槽和当前槽，不保存未来观测：

- 原点为当前 Head 的水平位置与当前 `floor_y`；
- 朝向为当前 Head yaw；
- Pose 世界旋转左乘当前 yaw 的逆旋转；
- Tracker 位置和旋转直接从世界状态变换到 `C_n`；
- Head 路径表示每个锚点相对当前 Head 原点和朝向的绝对位置与 yaw。

冷启动历史左侧锚点无效。所有无效历史条件在归一化后清为字面零，`window_valid_mask=False`；当前条件槽始终有效，11 个扩散目标始终存在且不使用这个 mask。source 至少需要 71 帧，所有 Task 起点必须满足 `current_absolute + 10 < frame_count`。

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

source key 由 `.npz` 相对 source 根目录的路径产生，`.npy/.npz` 后缀在 split 匹配时忽略；`M/` 首级目录表示镜像。source 文件不要求内嵌 metadata。

## Task Store

Task Store 是 task 生成阶段写入磁盘的未归一化数据。设样本数为 `M`、Source 数为 `S`，每个样本固定保存五套 Tracker 场景：

| shard 字段 | 形状 | 语义 |
|---|---:|---|
| `history_pose_clean` | `[M,10,144]` | 当前参考系下的 10 个历史 Pose 条件 |
| `pose_target_horizon_clean` | `[M,11,144]` | 当前帧与未来 10 帧的联合扩散目标；索引 0 为当前帧 |
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
| `previous_contact_target` | `[M,2]` | 上一帧左右脚接触监督 |
| `contact_target` | `[M,2]` | 当前左右脚接触监督 |
| `joint_offsets_parent` | `[M,24,3]` | 当前任务对应的骨架父子偏移 |
| `joint_rest_local_rotations_6d` | `[M,24,6]` | 当前任务对应的骨架 rest local rotation6D |
| `task_seed` / `start_frame` | `[M]` | 稳定任务种子与密集历史起始帧；种子由 `split + source相对路径 + start_frame` 生成 |

Task Store 只使用目录结构：

- `<task_dir>/<split>/shards/shard_*/`：上表各字段的独立 `.npy`；
- 每个 shard 的所有字段首维必须一致；
- 每个 shard 包含 normalizer 聚合使用的 `stats.npz`；
- shard 名称按字典序决定读取顺序，不存在额外索引或元数据文件。

旧字段 `pose_window_clean`、`future_leg_target` 或缺少上述新字段的 Task Store 会被明确拒绝，不提供兼容读取路径。每个 shard 的 Pose normalizer 统计只聚合 10 个历史锚点和当前目标，不纳入未来 10 个密集目标帧。

## 模型 Batch

Dataset 从 Task Store 选择一套 Tracker 场景，按虚拟会话起点重新累计状态并抽取 10 个历史锚点和当前帧。以下形状均包含 batch 轴 `B`：

| 字段 | 形状 | 语义 |
|---|---:|---|
| `x` | `[B,11,144]` | 当前帧与未来 10 帧的联合 diffusion target；`x[:,0]` 为当前帧 |
| `history_pose_observation` | `[B,10,144]` | 历史 Pose 条件；训练时可扰动，部署时为真实历史预测 |
| `head_path_window` | `[B,11,5]` | 同步锚点在 `C_n` 下的 Head 路径 |
| `history_region_confidence` | `[B,10,5]` | 历史 Pose 逐帧、逐身体区域可信度 |
| `window_valid_mask` | `[B,11]` | 锚点有效性；当前帧恒为 `True` |
| `frame_offsets` | `[B,21]` | 10 个历史槽加 11 个目标槽的固定真实帧偏移 |
| `configured` / `measured_valid` | `[B,11,6]` | 所选场景在同步锚点上的 Tracker 状态 |
| `d_off` / `d_on` | `[B,11,6]` | 从虚拟会话起点重新累计的掉线与在线帧数 |
| `tracker_window_raw` | `[B,11,6,13]` | IK 与几何 Loss 使用的未归一化 Tracker 窗口 |
| `hard_rotation_state_window` | `[B,11,6]` | 保留的旧诊断字段；不再进入当前采样控制链 |
| `joint_offsets_parent` | `[B,24,3]` | 目标骨架父子偏移 |
| `joint_rest_local_rotations_6d` | `[B,24,6]` | 目标骨架 rest local rotation6D |
| `target_joints_head_ref` | `[B,24,3]` | 当前帧关节位置监督 |
| `target_root_position_head_ref` | `[B,3]` | 当前 Root 在 `C_n` 下的位置监督 |
| `target_root_yaw_world` | `[B]` | 当前 Pelvis forward 世界 yaw 监督 |
| `target_hip_height` | `[B]` | 当前 Pelvis 世界高度监督 |
| `current_head_yaw_world` | `[B]` | 当前 Head 世界 yaw |
| `current_head_position_world` | `[B,3]` | 当前 Head 世界位置 |
| `floor_y` | `[B]` | 当前地面世界高度 |
| `previous_contact_target` | `[B,2]` | 上一帧左右脚接触监督 |
| `contact_target` | `[B,2]` | 左右脚接触监督 |
| `history_length` | `[B]` | 当前虚拟会话可见的密集历史帧数 |
| `scenario_id` / `scenario` | `[B]` | 所选 Tracker 场景的索引与名称 |
| `start_frame` / `task_id` | `[B]` | 样本定位与稳定任务标识；`task_id` 是 `task_seed` 的固定十六进制表示 |

`history_region_confidence` 只描述过去 10 个历史条件槽位，用于历史 Pose 条件和训练期历史扰动；它不参与当前帧逐关节置信度计算，也不参与 `T_soft` 或当前逐关节 inpainting 控制。

Dataset 不返回 `source_path`，也不接受 `folder_path` 过滤。需要按数据集或序列筛选时，应使用 split 或单独的 Task Store 目录。

训练不构造 IK、rolling prior 或 inpainting 张量。Runtime 在采样时只构造两个动态张量：

| 字段 | 形状 | 语义 |
|---|---:|---|
| `inpaint_pose` | `[B,11,144]` | 当前 IK 均值与上一轮 horizon 左移后的未来先验，按 Pose normalizer 归一化 |
| `inpaint_confidence` | `[B,11,24]` | 逐目标帧、逐关节标量置信度，范围 `[0,1]` |

如需 mask，在当前 DDIM 步由 `confidence > 0`、`t >= T_soft` 与 `t > 0` 局部派生，不持久化 `inpaint_kind` 或独立 `inpaint_mask`。模型 diffusion state、模型输出和扩散采样结果均为 `[B,11,144]`。`history_pose_observation: [B,10,144]` 是固定条件，不进入 diffusion Markov chain。`RuntimeStepResult` 返回 `raw_pred_pose_horizon/deployed_pred_pose_horizon: [11,144]` 和 `inpaint_confidence: [11,24]`，只解析 horizon 0 并追加到密集历史。

## 144D Pose 与历史扰动

每帧 Pose 由 24 个关节的全局 rotation6D 拼接而成。训练入口令 `x_start = x`，对 `[B,11,144]` 联合目标执行 `q_sample()`。同一样本的 11 帧共享一个 diffusion timestep，各帧各特征独立采样噪声；`feature_w: [144]` 广播到 horizon 轴，reconstruction loss 对 horizon 和 feature 共同求均值。

历史 Pose 扰动只修改独立条件 `history_pose_observation`，不会改变联合 diffusion target `x`。扰动在反归一化后的 SO(3) 空间执行：

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

通道偏移为 `0:3`、`3:9`、`9`、`10`、`11`、`12`。无效测量的前 9 维严格清零。只有前 9 维连续量参与 normalizer；状态与持续时间不参与。`tracker_window_raw` 中的 `d_off/d_on` 仍以 60 归一化；Runtime 使用当前密集帧的整数 `d_on` 计算在线置信度。

Task store 保留密集的 `configured/measured_valid: [M,5,61,6]`。Dataset 从虚拟会话起点重新累计 `d_on/d_off` 和 hard state，再抽取模型锚点，不能直接继承被裁掉历史的持续时间。

当前 inpainting 只使用一套 Tracker 在线置信度：`w_t = valid_t * clamp(d_on / tracker_confidence_warmup, 0, 1)`。该配置属于 Runtime/采样器，不挂到 DiT 参数或 checkpoint 上。`kappa_position/kappa_rotation` 和 `hard_rotation_state` 可作历史诊断字段，不再决定当前 inpainting 强度。

## Head 路径

每个锚点节点为：

\[
p_k^{head}=[x_k^{C_n},z_k^{C_n},h_k,\sin(\psi_k-\psi_n),\cos(\psi_k-\psi_n)].
\]

当前节点恒为 `[0,0,h_n,0,1]`。该字段表示过去一秒内 Head 走过的绝对路径，不是锚点间增量。

## Normalizer

- `pose_mean.pt`、`pose_scale.pt`: `[144]`；`pose_scale` 由标准差稳定化后再加 `eps`，所有 Pose 物理空间转换统一使用该尺度
- `tracker_mean.pt`、`tracker_std.pt`: `[6,9]`
- `head_path_xz_mean.pt`、`head_path_xz_std.pt`: `[2]`
- `head_height_mean.pt`、`head_height_std.pt`: 标量

Head 路径的 XZ 与高度分别归一化，sin/cos 保持原值。normalizer 目录只包含上述八个张量文件，不读取 metadata，也不自动判断它与 task 是否匹配。

## 时空 DiT 与投影

- 模型 token 网格为 `[B,21,24,D]`：`0:10` 是历史 Pose 条件，`10:21` 是当前与未来 10 帧的带噪目标；
- Tracker position 先经 IK 转换成同一 144D Pose 状态；DiT 不再实例化或执行 Tracker state/position/rotation cross-attention；
- 每层只执行 24 关节 Spatial Self-Attention 与同关节 21 槽 Temporal Self-Attention；
- 模型只维护 `token_valid_mask: [B,21]`。冷启动无效历史不能作为 attention key；11 个目标 token 始终有效；
- Temporal Attention 的历史 query 只能读取有效历史 key，不能读取任何目标 token；目标 query 可读取全部有效历史和全部 11 个目标，目标窗口内部双向 attention；
- 训练时 `model_kwargs` 只传递一个 `y` 字典，模型预测与 Loss 从同一份条件和监督数据读取，不在顶层重复条件字段；
- 历史 Pose 与 Head 路径由 `prepare_conditioning()` 每个目标帧编码一次，在 DDIM 步间复用；
- 历史和目标使用独立输入投影；共享一个 diffusion timestep embedding，21 个 frame-offset embedding 区分帧位置；
- DiT 不接收 `inpaint_confidence` 或 `inpaint_kind`，不存在 constraint/confidence embedding；置信度只在模型调用前决定哪些 `x_t` 关节被条件覆盖；
- 解码最后 11 个槽，输出 `[B,11,144]`；`contact_head` 只读取当前目标槽，不存在 `future_leg_head`；
- SO(3) projection 只对 `[B,11,144]` 的所有 rotation6D 执行合法化；最终步也不再用 Tracker rotation 强制覆盖当前帧；
- DDIM 的 state、初始 noise 和每步更新均为 `[B,11,144]`。每个实时帧另采样一份 `known_noise: [B,11,144]` 并在完整去噪轨迹复用；
- 训练阶段不执行 IK、inpainting 或模型 rollout，不使用未来 GT 伪造 rolling prior；长序列闭环只用于评估，运行时每步只追加当前部署预测，不重写过去历史。

## IK-Inpainting

- 有上一轮 horizon 时，`previous_horizon[1]` 正好对齐新一轮当前帧，重编码到当前 Head-yaw 参考系后作为 IK 初值；尚无 horizon 时使用上一帧已部署 Pose，首帧再回退到 rest rotation。初值本身不标成当前观测；
- 固定执行两轮 FABRIK：Hip 有效时求解躯干到 Head；手臂求解 Shoulder–Elbow–Wrist；Hip 与对应 Foot 同时有效时求解 Hip–Knee–Ankle–Foot；缺失必要 Tracker 时整条链跳过；
- 从旧骨向量到新骨向量的 shortest-arc rotation 左乘上一帧 global rotation，使上一帧 twist 随 IK soft 条件连续保留；直接测得的 Head、Pelvis、Wrist、Foot rotation 在 IK 后覆盖；

### 当前与未来置信度

- 当前 IK 只生成 `[B,144]` Pose 初值，不评价自身准确性，也不参与置信度计算；
- 当前 inpainting 使用固定 Tracker→区域 mapping：`Torso ← Head、Hip`，左右手臂分别由对应 Hand 覆盖，左右腿分别由对应 Foot 覆盖；五区域再按 `TARGET_JOINT_REGIONS` 展开成只读 `[24,6]` 二值矩阵；
- 每个 Tracker 的在线置信度为 `w_t = valid_t * clamp(d_on_t / tracker_confidence_warmup, 0, 1)`；当前逐关节置信度唯一公式为 `c[j,0] = max_t(mapping[j,t] * w_t)`，禁止平均、noisy-or、骨链 `min` 或额外 solved mask；
- 采样入口保留 `--tracker_confidence_warmup`、`--future_confidence_decay` 和 `--fabrik_iterations`；这些参数不属于 DiT 构造参数；
- 上一轮世界 horizon 用索引 `2..10` 对齐新一轮未来 `1..9`，最远帧重复旧末帧，并重新编码到当前 Head-yaw 参考系；首次推理的未来置信度全为 `0`；
- 未来置信度严格为 `c[j,k]=c[j,0]*future_confidence_decay^k`，`k=1..10`，默认 decay 为 `0.9`；
- 统一 DDIM timestep 为 `t`，逐关节使用连续阈值 `T_soft(c)=(1-c)*T_max`，不平方、不取整。当 `c>0`、`t>=T_soft` 且 `t>0` 时，用同一全局 `t` 下的 `q(inpaint_pose,t,known_noise)` 覆盖该关节；当 `t<T_soft` 时停止覆盖，由当前 diffusion state 继续去噪。`c=1` 在全部 `t>0` 步骤持续注入，但最终 `t=0` 与其他置信度一起释放，允许模型保留此前修正并完成最后一次前向；`c=0` 从不注入。同一个实时帧在完整 DDIM 轨迹中复用同一份 `known_noise`。
- `pose_history_mode=ground_truth` 只在当前帧评估结束后用当前 GT 替换历史状态，并立即清空预测 horizon；下一帧 IK 读取上一帧 GT Pose，未来 rolling prior 保持无效且置信度为零，不读取未来 GT。

## Loss、运行时与长序列结果

- diffusion reconstruction、global rotation、local rotation 和 rotation velocity loss 作用于全部 11 帧；
- rotation velocity 比较相邻目标帧 `R_h^T R_(h+1)` 的 SO(3) 夹角，共 10 组，默认权重为 `1.0`；
- Tracker rotation/position、FK、Head-reference joint、root yaw、Head-to-Root XZ、contact 和 contact-slide 只作用于 horizon 0；`deployed_pred_xstart` 只是全 horizon 的 SO(3) 合法化结果，不包含 Tracker hard 覆盖；
- `previous_pose_target` 是未扰动的最后一个历史 Pose；cold-start contact-slide 由最后历史槽有效性屏蔽；
- 长序列结果同时保存 `reference_pose_horizon_raw`、`raw_pred_pose_horizon_raw`、`deployed_pred_pose_horizon_raw: [N,T,11,144]` 和 `pose_horizon_valid_mask: [N,T,11]`；序列尾部缺少的未来参考帧填 NaN；
- 当前 MPJRE/MPJPE/MPJVE/Jitter/contact 等指标继续从 horizon 0 计算；另外报告 raw/deployed 的 horizon 0 到 10 全身 MPJRE，以及未来 1:10 宏平均。当前阶段不计算未来 MPJPE 或未来 root 位置误差；
- 旧 Task Store、旧单帧 checkpoint/args 和联合 11 帧模型的 Unity/Sentis 导出均明确拒绝。Unity runtime schema 仍保持旧单帧接口，不生成联合模型 ONNX。
