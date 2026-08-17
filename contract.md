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

`stationary_prob_5` 只表示 5 个候选关节的世界空间静止概率，不直接作为脚部
地面接触真值。Task 和长序列评估取左右脚通道后，再乘以脚高软门控：脚高相对
`floor_y` 不超过 5cm 时完整保留，5cm 到 10cm 线性衰减，达到 10cm 时归零。

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
| `target_root_yaw_world` | `[M]` | 在完整 Source 上因果展开的 Pelvis 世界 Y-twist heading；退化帧沿用上一可靠值 |
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
| `hard_rotation_state_window` | `[B,11,6]` | 只用于最终部署 Pose 的直接 Tracker rotation hard projection |
| `joint_offsets_parent` | `[B,24,3]` | 目标骨架父子偏移 |
| `joint_rest_local_rotations_6d` | `[B,24,6]` | 目标骨架 rest local rotation6D |
| `target_joints_head_ref` | `[B,24,3]` | 当前帧关节位置监督 |
| `target_root_position_head_ref` | `[B,3]` | 当前 Root 在 `C_n` 下的位置监督 |
| `target_root_yaw_world` | `[B]` | 在完整 Source 上因果展开的 Pelvis 世界 Y-twist heading 监督 |
| `target_hip_height` | `[B]` | 当前 Pelvis 世界高度监督 |
| `current_head_yaw_world` | `[B]` | 当前 Head 世界 yaw |
| `current_head_position_world` | `[B,3]` | 当前 Head 世界位置 |
| `floor_y` | `[B]` | 当前地面世界高度 |
| `previous_contact_target` | `[B,2]` | 上一帧左右脚接触监督 |
| `contact_target` | `[B,2]` | 左右脚接触监督 |
| `history_length` | `[B]` | 当前虚拟会话可见的密集历史帧数 |
| `scenario_id` / `scenario` | `[B]` | 所选 Tracker 场景的索引与名称 |
| `start_frame` / `task_id` | `[B]` | 样本定位与稳定任务标识；`task_id` 是 `task_seed` 的固定十六进制表示 |

`history_region_confidence` 只描述过去 10 个历史条件槽位，用于历史 Pose 条件和训练期历史扰动；它不参与当前帧逐关节置信度或当前逐关节 inpainting 控制。

Dataset 不返回 `source_path`，也不接受 `folder_path` 过滤。需要按数据集或序列筛选时，应使用 split 或单独的 Task Store 目录。

训练与 Runtime 使用同一套当前帧 IK 条件构造逻辑，并在模型调用前构造以下动态条件：

| 字段 | 形状 | 语义 |
|---|---:|---|
| `inpaint_pose` | `[B,11,144]` | 索引 0 为当前 IK Pose；第一轮未来 1..10 严格为零 |
| `inpaint_valid` | `[B,11,24]` | 索引 0 等于 IK `updated_mask`；第一轮未来 1..10 严格为 `False` |
| `release_level` | `[B,11,24]` | 当前逐关节在物理噪声坐标中的释放阈值，范围 `[0,1]` |
| `current_joint_condition` | `[B,24,10]` | 当前逐关节 Tracker position、有效性、IK confidence 与约束类型；只加到 current token |

`current_joint_condition` 的通道固定为：`0:3` 是归一化 Tracker position，`3` 是 position valid，`4` 是 IK updated valid，`5` 是 IK confidence，`6:10` 是四种 constraint type 的 one-hot。六个 Tracker position 只 scatter 到 Head、双 Wrist、Pelvis、双 Foot；无效位置在归一化后重新清零。position 使用现有 `tracker_mean/std[:,0:3]`，禁用 normalizer 时保留 Head-yaw 参考系下的米制数值。

当前 DDPM/DDIM 步的 active mask 由 `inpaint_valid`、实际 `alpha_bar_t` 和 `release_level` 局部计算，不进入 DiT。模型 diffusion state、模型输出和扩散采样结果均为 `[B,11,144]`。`history_pose_observation: [B,10,144]` 是固定条件，不进入 diffusion Markov chain。`RuntimeStepResult` 继续返回 `raw_pred_pose_horizon/deployed_pred_pose_horizon: [11,144]` 和诊断用 `inpaint_confidence: [11,24]`；后者只有 horizon 0 可非零。Runtime 只解析 horizon 0 并追加到密集历史。

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

当前 inpainting 的 Tracker 来源可靠度为 `w_t = configured_t * measured_valid_t * clamp(d_on / tracker_confidence_warmup, 0, 1)`，再与 IK 约束类型质量和归一化端点残差联合计算逐关节 confidence。该配置由训练与 Runtime 共享并写入 `args.json`，但不改变 DiT 参数结构。`kappa_position/kappa_rotation` 只作历史诊断，`hard_rotation_state` 只决定最终直接 Tracker rotation hard projection。

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
- Tracker position 一路经 IK 转换成 144D Pose state，另一路以逐关节 10D 轻量条件保留端点几何；DiT 不实例化或执行 Tracker cross-attention；
- 每层只执行 24 关节 Spatial Self-Attention 与同关节 21 槽 Temporal Self-Attention；
- 模型只维护 `token_valid_mask: [B,21]`。冷启动无效历史不能作为 attention key；11 个目标 token 始终有效；
- Temporal Attention 的历史 query 只能读取有效历史 key，不能读取任何目标 token；目标 query 可读取全部有效历史和全部 11 个目标，目标窗口内部双向 attention；
- 训练时 `model_kwargs` 只传递一个 `y` 字典，模型预测与 Loss 从同一份条件和监督数据读取，不在顶层重复条件字段；
- 历史 Pose、Head 路径与当前 10D joint condition 由 `prepare_conditioning()` 每个目标帧编码一次，在 DDIM 步间复用；
- 保留逐锚点 Head 路径编码；另从归一化 Head 路径按真实 `-1/-8/-60/0` 帧偏移确定性构造 14D 运动摘要，通道固定为 `[vx_1,vz_1,vx_8,vz_8,vx_60,vz_60,yaw_rate_8,yaw_rate_60,current_height,vertical_velocity_8,vertical_velocity_60,valid_1,valid_8,valid_60]`，经单层线性投影后只加到当前 Pelvis token；
- 历史和目标使用独立输入投影；共享一个 diffusion timestep embedding，21 个 frame-offset embedding 区分帧位置；
- 当前 joint condition 只经过 `Linear(10, latent_dim)` 并加到 token 10；不直接写入未来 token，未来只能通过 temporal self-attention 读取当前条件；
- DiT 不接收旧的 `inpaint_confidence` 或 `inpaint_kind` 字段。soft inpainting 仍由 `release_level` 外部控制，模型仅通过 10D joint condition 看到 confidence 与 constraint type 的语义副本；
- 解码最后 11 个槽，输出 `[B,11,144]`；`contact_head` 只读取当前目标槽，不存在 `future_leg_head`；
- SO(3) projection 先合法化 `[B,11,144]` 的所有 rotation6D，再只对 horizon 0 中处于 hard 状态的 Tracker 关节覆盖实测旋转；
- DDIM 的 state、初始 noise 和每步更新均为 `[B,11,144]`。每个实时帧另采样一份 `known_noise: [B,11,144]` 并在完整去噪轨迹复用；
- 训练阶段在 history corruption 后执行当前帧 IK 与 inpainting，不使用未来 GT 或预测 horizon；长序列闭环只用于评估，运行时每步只追加当前部署预测，不重写过去历史。

## IK-Inpainting

- 训练使用经过 history corruption 的最后一个有效历史 Pose，Runtime 使用上一帧实际部署 Pose 并重编码到当前 Head-yaw 参考系；无有效上一帧时统一回退到 rest rotation。初值本身不标成当前观测；
- 固定执行两轮 FABRIK：Hip 有效时求解躯干到 Head；手臂求解 Shoulder–Elbow–Wrist；Hip 与对应 Foot 同时有效时求解 Hip–Knee–Ankle–Foot；缺失必要 Tracker 时整条链跳过；
- 从旧骨向量到新骨向量的 shortest-arc rotation 左乘上一帧 global rotation，使上一帧 twist 随 IK soft 条件连续保留；直接测得的 Head、Pelvis、Wrist、Foot rotation 在 IK 后覆盖；

### 当前帧部分 IK 与物理噪声置信度

- IK 返回 `pose: [B,24,6]`、`updated_mask/direct_rotation_mask/constraint_type/position_residual/confidence: [B,24]`。约束编号固定为 `DIRECT_ROTATION=0`、`POSITION_SOLVED=1`、`DIRECTION_ONLY=2`、`INHERITED=3`；`INHERITED` 必须满足 `updated_mask=False、confidence=0`；
- 有效 Head、双 Wrist、Pelvis、双 Foot rotation 标为 `DIRECT_ROTATION`。当前 shortest-arc FABRIK 只更新 swing 并继承初始化 twist，所以 Spine、Shoulder/Elbow、Hip/Knee/Ankle 的真实链更新统一标为 `DIRECTION_ONLY`；本轮不得产生 `POSITION_SOLVED`；
- Torso 只有 Head 与 Hip 同时有效时求解，腿只有 Hip 与对应 Foot 同时有效时求解，手臂由对应 Wrist 激活；Collar、Hand 和未激活骨链保持 `INHERITED`；
- Tracker 来源可靠度为 `w_t=configured_t*measured_valid_t*clamp(d_on_t/tracker_confidence_warmup,0,1)`。Torso 取 `min(head,hip)`，腿取 `min(hip,foot)`，手臂取对应 Wrist，直接 rotation 取自身 Tracker；禁止区域 `max` mapping；
- 链约束使用最终 FK 后的 `endpoint_residual/chain_length`。逐关节公式为 `confidence=source_reliability*constraint_quality*exp(-residual_ratio/residual_scale)`；直接 rotation 的质量固定为 1 且不受位置 residual 降权，继承类型固定为 0；
- `--ik_direction_only_quality` 与 `--ik_residual_scale` 没有生产默认值，必须由 `eval/calibrate_realtime_pose_ik.py` 在 materialized train task 上拟合并显式传给训练/采样，随后写入 checkpoint `args.json`。当前不产生 `POSITION_SOLVED`，因此 `--ik_position_solved_quality` 可为空；
- confidence 映射为 `release_level=sin((1-confidence)*pi/2)`。每步使用实际 `noise_level_t=sqrt(1-alpha_bar_t)`，仅当 `inpaint_valid & (noise_level_t>=release_level)` 时，以同一 `known_noise` 重建 `q(inpaint_pose,t)` 并覆盖模型输入；未激活位置逐元素保持当前 diffusion state；
- 不存在 `t=0` 特判。confidence=1 的直接 rotation 可持续到最低噪声步，中低 confidence 约束会更早释放；训练 epsilon target 必须由被修改后的 `x_model` 与 GT `x_start` 重算；
- 第一轮显式禁用 future rolling prior。CLI 名称暂时保留，但传入 `--use_future_rolling_prior` 时 Runtime 立即报错；未来 1..10 的 `inpaint_valid` 和诊断 confidence 恒为零，不保存上一轮预测 horizon；
- `pose_history_mode=ground_truth` 只在当前帧评估结束后用当前 GT 替换历史状态；下一帧 IK 只读取上一帧 GT Pose，不读取任何未来 GT。

校准命令示例：

```powershell
conda run -n diffusionposer5070 python -m eval.calibrate_realtime_pose_ik --data_dir <task_store> --split train --output output/ik_calibration.json
```

## Loss、运行时与长序列结果

- diffusion reconstruction、global rotation、local rotation 和 rotation velocity loss 作用于全部 11 帧；
- rotation velocity 比较相邻目标帧 `R_h^T R_(h+1)` 的 SO(3) 夹角，共 10 组，默认权重为 `1.0`；
- Tracker rotation/position、FK、Head-reference joint、root yaw、Head-to-Root XZ、Hip height、contact 和 contact-slide 只作用于 horizon 0；root yaw 使用 Pelvis 旋转的最近世界 Y-twist heading，按 15° 尺度的二维 circular loss 监督，XZ 与 Hip height 使用 5cm 尺度的 Huber loss；`deployed_pred_xstart` 在全 horizon SO(3) 合法化后，仅对 horizon 0 的 hard Tracker 旋转执行覆盖；Head 始终 hard，其他 Tracker 仅在 configured、measured_valid 且 `d_on >= d_hard` 时 hard；
- `previous_pose_target` 是未扰动的最后一个历史 Pose；cold-start contact-slide 由最后历史槽有效性屏蔽；
- 长序列结果同时保存 `reference_pose_horizon_raw`、`raw_pred_pose_horizon_raw`、`deployed_pred_pose_horizon_raw: [N,T,11,144]` 和 `pose_horizon_valid_mask: [N,T,11]`；序列尾部缺少的未来参考帧填 NaN；
- 当前 MPJRE/MPJPE/MPJVE/Jitter/contact 等指标继续从 horizon 0 计算；另外报告 raw/deployed 的 horizon 0 到 10 全身 MPJRE，以及未来 1:10 宏平均。当前阶段不计算未来 MPJPE 或未来 root 位置误差；
- 长序列评估 metadata 继续记录 rolling-prior 开关和 decay，但第一轮有效运行的开关只能为 `False`；`rp1g<decay>` 命名仅保留给第三轮重新实现后使用；
- `target_root_yaw_world` 的标签语义已经变化，因此 Task Store 必须重新生成，配套 normalizer 也必须从新 Task Store 重算；新增 `pelvis_head_motion_input` 后不提供旧 checkpoint 兼容迁移，必须从头训练。联合 11 帧模型的 Unity/Sentis 导出仍拒绝；Unity runtime schema 保持旧单帧接口，不生成联合模型 ONNX。
