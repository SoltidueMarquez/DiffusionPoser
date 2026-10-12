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

启用 `--rpm_hand_dropout` 时，Predictor Dataset 额外构造
`tracker_available [52,6]`。左右手按 RPM 官方训练 masker 独立以 10% 概率
丢弃一个连续 segment，长度 `L ~ U(1,41)`；随机数使用 Python `random` 的官方
调用顺序，并由 train-only base seed 与 task id 派生，因此不受 worker 数量影响。

## 单帧 Task Store

Task Store 保存原始物理值，不保存归一化结果。设 shard 样本数为 `M`：

| 字段 | 形状 | 语义 |
|---|---:|---|
| `motion_context_clean` | `[M,10,144]` | 当前 `C_n` 下的过去 10 帧 Pose |
| `core_tracker_context_clean` | `[M,11,54]` | 核心三点过去 10 帧与当前 sparse 特征 |
| `current_pose_target_clean` | `[M,144]` | 当前帧 GT Pose；Dataset 加载后用于构造 residual target |
| `current_tracker_continuous` | `[M,6,9]` | 当前六点 position + rotation6D |
| `previous_pose_target_clean` | `[M,144]` | 当前 `C_n` 下上一帧 GT Pose |
| `target_joints_head_ref` | `[M,24,3]` | 当前几何监督 |
| `target_root_position_head_ref` | `[M,3]` | 当前 Root 几何监督 |
| `target_root_yaw_world` / `target_hip_height` | `[M]` | Root 监督 |
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
不可用 Tracker 的前 9 维为零。训练 sampler 对 Hip、LeftFoot、RightFoot 的所有
开关组合做确定性轮换，长期等概率覆盖全部 8 种配置；模型不接收 scenario id。
`--rpm_hand_dropout` 只在 Dataset 读取时叠加确定性左右手 gap，不改变 Task Store
字段或 normalizer。缺失 sparse 通道在归一化域置零；当前手 Tracker 同步更新
`tracker_available` 并清零连续量。训练默认 seed 为 7，训练期评估使用 seed+1；
RPM-P2 官方测试 gap seed=6 只由正式评估入口读取，不进入训练。

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

## Predictor residual diffusion 与冻结边界

令 `p = predictor_pose_horizon[:,0]`，完整 normalized rotation6D residual 为
`r = GT - p`，逐关节修正条件为 `m [B,24]`。DiT diffusion state、noise、target、
模型输出和 sample 均为 `[B,144]` residual：

```text
x_start = broadcast6(m) * r
raw_pred_pose = p + predicted_residual
```

扩散基础 schedule 默认使用 50 个 timestep，并固定 `predict_xstart=True`。训练使用
完整 50 步 schedule；采样默认从该 schedule respacing 到 10 步，亦可用同一新
checkpoint 做 50/20/10/8/5-step 消融。

DiT 公开条件：

```text
motion_context          [B,10,144]
predictor_current       [B,24,6]       # 由 horizon[:,0] 提供
predictor_future        [B,10,24,6]    # 由 horizon[:,1:11] 提供
tracker_geometry        [B,6,9]
tracker_available       [B,6] bool
ik_residual             [B,24,6]
ik_gap                  [B,24]
ik_confidence           [B,24]
denoise_strength        [B,24]
constraint_type         [B,24]
```

实现接口使用完整 `predictor_pose_horizon [B,11,144]` 传递 current 与 future。
历史与未来分别经过不同 Linear 和 role embedding，加入共同 joint/region identity
及固定 frame offset，拼成 `temporal_context [B,24,20,D]`，顺序严格为
`[-10,...,-1,+1,...,+10]`。

当前 24 个 joint query 先执行一次 Tracker cross-attention。K/V 物理形状固定为
`[B,6,D]`，但 `key_padding_mask = ~tracker_available` 会把缺失 Tracker 从
attention logits 中排除，因此逻辑 K/V 长度是 3～6；只把缺失 token 置零不能替代
这个 mask。随后进入 4 个 block，每层依次执行 24 关节 spatial self-attention、
同关节 20 帧 temporal cross-attention 和 timestep AdaLN MLP。固定模型配置为
`D=192, layers=4, heads=6, mlp_ratio=4, dropout=0, max_seq_len=21`，参数量小于
5M。`LayerNorm → Linear(D,6)` 输出头零初始化，直接输出 residual；`m` 已经作用于
训练 target，不在模型输出上再次相乘。

第二阶段冻结训练中的 Predictor 始终 `eval()`、`requires_grad=False` 且在 `torch.no_grad()`
中执行。DiT optimizer、EMA 和 checkpoint 不含 Predictor 权重；`args.json` 记录
`predictor_model_path`。冻结 Predictor 与 DiT 始终共享 Task Store 中同一份干净
Pose 历史；启用 RPM hand dropout 时，只对 Predictor sparse context 与当前手部
Tracker availability 施加同一确定性 mask。部署 Pose 历史的时间累积误差仍由
Predictor 单阶段训练中的 0～30 步闭环回填建模。

## Predictor 与 DiT 联合微调

冻结 DiT 阶段收敛后，使用 `python -m train.train_realtime_pose_joint` 加载同一套
Predictor 与 DiT checkpoint 做短程联合微调。联合阶段继续使用相同 Task Store、
8 种 Tracker 配置和条件加权 residual，不改变训练与部署张量契约。

令可训练 Predictor 输出为 `p`。IK、DiT condition 和 diffusion target 使用
`stopgrad(p)`，避免 Predictor 通过移动 residual target 或 IK 条件降低 diffusion
loss；最终姿态仍使用未 detach 的 `p`：

```text
p_condition = stopgrad(p)
x_start = broadcast6(m) * (GT - p_condition)
raw_pred_pose = p + predicted_residual
loss = dit_loss + predictor_loss_weight * mse(p, GT)
```

因此 Predictor 从最终姿态辅助损失和当前帧 pose MSE 获得梯度，DiT 继续学习条件
加权 residual。optimizer 使用两个参数组，默认 DiT `lr=1e-5`、Predictor
`lr=1e-6`，默认联合训练 20,000 step。

联合 checkpoint 必须成对使用。`modelXXXXXXXXX.pt` / `emaXXXXXXXXX.pt` 保存 DiT，
`predictorXXXXXXXXX.pt` / `predictor_emaXXXXXXXXX.pt` 保存 Predictor；
`model_latest.pt` 与 `predictor_latest.pt` 是同一步的两份 EMA 推理权重。

## IK residual 与修正条件

IK 从反归一化的 `predictor_pose_horizon[:,0]` 初始化。Head、双手始终 available；
直接 Tracker 旋转先写入 IK。双臂始终求解；有 Hip 时求解躯干；对应 Foot
available 时即求解该腿。没有 Hip 时，以 Head 对齐后的 Predictor 骨盆位置作为
腿链根节点。`ik_residual = normalized(IK) - p` 只作为 DiT joint condition，
不注入 diffusion state。

Predictor/IK gap 使用原始旋转的 SO(3) geodesic angle，单位为弧度：

```text
d_j = ||log((R_predictor_j)^T R_ik_j)||_2
u_j = clamp((d_j - gap_low) / (gap_high - gap_low), 0, 1)
demand_j = 3*u_j^2 - 2*u_j^3
m_j = 0.05 + 0.95 * support_j * ik_confidence_j * demand_j
```

`support` 固定为直接旋转 `1.0`、方向/位置骨链约束 `0.35`、继承未约束 `0.0`。
`m` 既作为条件显式告诉 DiT 当前关节的建议修正需求，也只在训练 target 中作用
一次；模型输出不再重复乘 `m`。校准在全部 8 种 Tracker 配置的
`updated_mask=True` 关节上统计
Predictor/IK gap，全局 P25/P90 分别写入 `ik_gap_low/high`；两者相差小于
`1e-4` 弧度时校准失败。校准 JSON 同时保留 `ik_direction_only_quality` 与
`ik_residual_scale`。训练和采样缺少 gap 校准值时 fail fast。

最终 projection 先把全部 rotation6D 投影到 SO(3)，再仅对所有 available Tracker
的直接关节旋转执行硬覆盖。IK 中间关节不硬覆盖。

## Runtime 与评估

Runtime 固定每秒调用 30 次。初始化必须提供 10 帧完整 world/deployed Pose history，以及核心三点
offset `-11..-1` 的 11 帧 Tracker；也接受已经包含当前 offset 0 的 12 帧形式。
每步只读取当前六点 Tracker，依次执行 Predictor、IK、10-step deterministic DDIM
（`eta=0`）、hard projection、
Head-anchored resolver，并把 deployed 当前 Pose 追加到 history。60/90Hz 显示插值由
Python 模型输出之后的客户端完成。

`RuntimeStepResult` 主要字段：

```text
predictor_pose_horizon    [11,144]
raw_pred_pose       [144]
deployed_pred_pose  [144]
ik_gap              [24]
ik_confidence       [24]
denoise_strength    [24]
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
DiT raw 与 DiT deployed 的同组 RPM-P2/MC 指标；全局 Predictor-only 基线只计算一次。
全部 8 种配置都属于训练分布。`--tracker_configs` 可选择评估子集；正式报告运行
全部 8 种，并同时报告 Tracker error 与端到端延迟。

动态 Tracker 退化实验由 `sample.evaluate_progressive_tracker_dropout` 独立运行。
帧 11～29 使用完整六点预热，正式计分区间等分四段，并按指定排列依次移除
Hip、Left Foot、Right Foot，使当前帧 availability 按 `6→5→4→3` 变化。
正式实验覆盖三个 optional Tracker 的全部 6 种移除顺序；每种顺序分别对整条拼接
序列计算 MPJRE、MPJPE、MPJVE 与预测 Jitter，再报告跨顺序均值和总体标准差。
分阶段指标仅作退化诊断，整条指标保留 Tracker 切换边界的速度与 Jitter。动态报告
不重复计算静态 6/3 Tracker 或 Predictor-only baseline。

动态 Tracker 添加实验由 `sample.evaluate_progressive_tracker_addition` 独立运行。
帧 11～29 使用 Head、Left Hand、Right Hand 三点预热，正式计分区间同样等分
四段，并按指定排列依次添加 Hip、Left Foot、Right Foot，使当前帧 availability
按 `3→4→5→6` 变化。正式实验覆盖全部 6 种添加顺序；每个 `add_order` 记录
逆序的 `paired_drop_order`，两条路径经过相同 Tracker mask 集合。整条指标保留
Tracker 添加边界的速度与 Jitter，报告不读取退化 JSON，也不重复生成静态 baseline。

`predictor_sparse_*` 统计键、`predictor_sparse_mean.pt/std.pt`、模型参数名与
`predictor_pose_horizon` 都是当前唯一契约。已有 Predictor checkpoint、当前 Task Store
及 Pose/Tracker normalizer 可直接复用；绝对 Pose diffusion/旧 IK trajectory 条件的
DiT checkpoint 与本 residual 架构不兼容，不提供兼容加载路径，必须重新训练 DiT。

## FLUIDUnity 离线 Demo

Unity 只录制六点 Tracker，固定顺序为 Head、LeftWrist、RightWrist、Hip、
LeftFoot、RightFoot。录制 JSON 顶层保存 `floorY` 和 `frames`；每帧保存
`time`、`positions [6,3]`、`rotations [6,4]`（quaternion xyzw）与
`available [6]`。位置和旋转均为校准后的 Unity TrackingOrigin 局部坐标。

`python -m sample.infer_unity_recording` 将录制重采样为 30Hz，运行当前
Predictor + 单帧 DiT runtime，并输出 Unity 可直接回放的 JSON。每帧只保存
`time`、Actor `rootPosition/rootRotation`、`pelvisLocalPosition` 和 SMPL24
固定顺序的 `localRotations [24,4]`。该 Demo 文件不包含 schema 或版本字段。
录制中的 Hip 异常时可传入 `--ignore_hip`，它只在本次推理中把 Hip availability
全程设为 false，不修改原始 JSON；Head、双手和双脚仍按五点配置进入 runtime。

## FLUIDUnity ONNX 部署

`export.export_sentis_denoiser` 导出 batch=1、FP32、opset 15 的两个模型。
`predictor.onnx` 输入归一化的 `motion_context [1,10,144]` 和
`core_tracker_context [1,11,54]`，输出归一化的 `predictor_pose_horizon [1,11,144]`。
历史 Pose 必须每帧转换到当前 Head-yaw 坐标系；缺失的 sparse 特征在归一化后
置零，速度有效性同时依赖相邻帧。上述特征处理由调用端完成。

`pose_sampler.onnx` 输入如下，除 `constraint_type` 为 int64 外全部为 float32：

| 输入 | 形状与语义 |
| --- | --- |
| motion_context | [1,10,144]，归一化历史 Pose |
| predictor_pose_horizon | [1,11,144]，Predictor 输出 |
| current_tracker_raw | [1,6,10]，当前 Head-yaw 系位置、forward/up 6D 旋转、0/1 availability |
| ik_residual | [1,24,6]，归一化 IK 姿态减 Predictor 当前姿态 |
| ik_gap / ik_confidence / denoise_strength | 各 [1,24]，现有 IK 条件 |
| constraint_type | [1,24]，现有 IK 约束类别索引 |
| noise | [1,144]，显式标准正态初始噪声 |

输出 `deployed_pose [1,144]` 仍在归一化域。调用端反归一化后使用现有
Head-anchored resolver 和 body rest 转换为 Unity 姿态，并反馈部署旋转到历史。
IK、历史管理与显示插值不在 ONNX 图中。

Sampler 固定采用 50 个基础 diffusion timestep、`ts_respace="10"`、`eta=0`；
实际 timestep 映射和系数作为常量保存。条件编码一次，10 次 DiT 共享权重，
仅最后一步投影，不再次乘 denoise strength，也不在图中生成随机数。
部署副本将 DiT block 的无仿射 LayerNorm 表达为中心化值乘方差的逆平方根，
保留原 eps 和 AdaLN 广播运算，避免 Unity Inference 2.6.1 错融合三维 scale/bias。
训练模型及用于对照的原 PyTorch 模型不做替换。
部署副本的时间 Attention 保留原 Q/K/V 和输出投影，用 `(K @ Qᵀ)ᵀ` 与
`(Vᵀ @ Pᵀ)ᵀ` 等价表达两次矩阵乘；保持双侧缩放和原时间轴 Softmax，
避免 Unity 的 M=1 向量批乘路径。此替换不涉及空间或 Tracker Attention，
不改变部署输入输出、权重、精度及采样步数。
同一次 Sampler 调用中，各 DiT block 的时间上下文归一化及 K/V 联合投影
在去噪循环外各执行一次，结果通过局部 tuple 传给 10 步；Q 仍逐步计算。
K/V 不写入跨调用状态，不同 block 不共享，下一 tick 从新条件重新计算。
`runtime_config.json` 的 normalizer 保存六组已加载统计和单独的
`normalizer_eps`：Pose 使用 pose_scale 直接相除，Tracker/sparse 使用 std+eps。
`ik` 保存实际解析参数，`ik_position_solved_quality=null` 表示沿用几何计算的
quality。骨架文件原样复制为 `body_fbx_rest.json`。

在仓库根目录执行：

```powershell
conda run -n diffusionposer5070 python -m export.export_sentis_denoiser `
  --predictor_model_path artifacts/unity_demo/predictor/model_latest.pt `
  --model_path artifacts/unity_demo/dit/ema000100000.pt `
  --normalizer_dir artifacts/unity_demo/normalizer `
  --ik_calibration_path artifacts/unity_demo/ik_calibration.json `
  --body_fbx_rest_json ../FLUIDUnity/Assets/StreamingAssets/FLUID/body_fbx_rest.json `
  --output_dir output/unity_onnx
```

将模块名换为 `export.check_unity_onnx`，其余参数相同，即执行最小对照。
默认录制为 `recording_20260829_164358.json`，也可用 `--input` 指定。
对照固定忽略 Hip，处理 90 个重采样帧、从第 12 帧开始推理，跳过 30 个推理
输出后比较连续 10 帧，容差为 atol=1e-4、rtol=1e-3。闭环历史使用 ONNX 输出；
原 PyTorch sampler 在相同输入下产生参考，不改变模型采样语义。
输出 `comparison.json`、`onnx_pose_result.json` 和 `reference_inputs.npz`。
NPZ 各输入带首轴 10（对照 tick），其后保留原 batch 轴；参考输出键为
`expected_predictor`、`expected_deployed_pose`，供 Unity 导入对照复用。
CPU ONNX 对照不代表 Unity GPU 兼容性或性能验收。

### 椅子扶手的可选残差碰撞后处理

该功能只存在于部署模块，普通 Python 采样与训练不接入。
`--collision_postprocess` 默认关闭；开启后必须传 Unity 编辑器标定的
`--collision_profile` 与 `--ts_respace 5`。开启后导出 `pose_sampler_condition.onnx`、
`pose_sampler_step.onnx`、`pose_sampler_projection.onnx` 和 `collision_compute.json`；
普通五步模型仍为 `pose_sampler_5step.onnx`。碰撞链不接受单个 `--sampler_filename`。
Unity 在推理启动时选择模型，修改开关后需重启推理，不在运行中切换模型。
开启却缺模型、标定或匹配的配置时终止并报告错误。

碰撞命令链保留普通 sampler 的全部外部输入，另加下列 GPU 快照；输出仍为
`deployed_pose [1,144]`，另有 posterior correction 求出的头部高度偏移。旋转历史仍使用 sampler 输出；高度单独保存在已接受的序列状态，显示层脚部补偿不写回历史。

| 新增输入 | 类型、形状与含义 |
| --- | --- |
| armrest_geometry | float32 [1,2,15]；两个 OBB 各为中心 xyz、单位轴 x/y/z 各 xyz、半尺寸 xyz |
| contact_mode | Unity int32 [1,2,2]（Python 参考为 int64）；上肢×扶手，0 停用、1 盒体避碰、2 盒体及有限上表面约束 |
| seated_context | float32 [1,16]；前 15 项为座面 OBB（中心、三轴、半尺寸），最后一项为接触模式 0/1/2；不再包含骨盆托举或软锚定 |
| display_from_pose | float32 [1,106]；未施加坐姿补偿的显示姿态：根位置 3、根四元数 4、骨盆局部位置 3、24 个局部四元数 96 |
| head_height_context | float32 [1,3]；上一次接受的头部 Y 偏移（角色米制）、与本次输入的时间差（秒，最大 0.1）、上次前后偏移（米）；新序列为 0、1/30、0 |

| correction_context | float32 [1,366]；前360项为五个采样阶段各72个已接受姿态修正量，第360项为时序有效权重，第361–365项为各阶段实际骨盆前后修正位移(m)。只使用每阶段后48个脊柱／髋／膝坐标；首帧及重置为全零 |

六项快照只进入碰撞 Shader，不进入 DiT。分段网络内部接口固定 batch=1：

- 条件模型输入为普通 sampler 去掉 `noise` 的八项输入。输出 `joint_condition [1,24,D]`、
  `tracker_tokens [1,6,D]`、`tracker_available [1,6]`（int32），以及每个 block 的
  `key_i [24,H,20,D/H]` 与 `value_i [24,H,D/H,20]`；D/H 取导出模型的配置。
- 单步模型输入上述全部缓存、`state [1,144]` 和 `timestep [1]`（ONNX int64，Unity Tensor<int>），
  输出 `predicted_residual [1,144]`。五个固定 Worker 使用同一模型资产和同一条件缓存。
- 投影模型输入 `predictor_pose_horizon [1,11,144]`、`edited_residual [1,144]`、
  `current_tracker_raw [1,6,10]`，输出 `deployed_pose [1,144]`。
- `collision_compute.json` 保存 `models`、`condition_tensors`（名称、形状、类型）、
  `sampling_steps`、执行顺序下的 `timesteps`、每步 `ddim` 系数
  （recip、recipm1、sqrt_alpha_prev、sqrt_one_minus_alpha_prev、variance）、
  `pose_mean/pose_scale` 和原有 settings/profile/parents/restLocalPositions。

初始化时录制并复用一个 CommandBuffer：Predictor、GPU IK、条件准备一次，随后五次
单步 DiT、碰撞 Shader、DDIM，最后投影。中间没有 CPU 读回或跨帧等待；请求结束只读回
最终姿态。当前普通模型关闭路径不改变。旧碰撞 ONNX 已移除；编辑器对照使用普通模型、分段旁路与当前 Shader 链。

头部高度以独立变量 `dy` 与残差共同求解，角色米制上限 `head_y_limit`（场景当前试用 0.05）。候选和最终重建均使用 `pelvis.y=tracker_head_y-relative_head_y+dy`；头部侧向位置和跟踪旋转继续硬投影，前后自由度见下文。测量项新增 `head_y_weight*dy^2+head_y_smooth_weight*(dy-dy_prev)^2`，同样除以 `2*m^2`，当前权重 1／0.5。每帧边界同时满足 `abs(dy)<=limit` 与 `abs(dy-dy_prev)<=head_y_speed*min(dt,0.1)`，当前速度 0.15 m/s；平滑的是误差偏移，不滤波原始头部运动。优化坐标为 `dy/0.1`，使用相同中心差分与步长，无扩散残差先验；每个采样阶段以上一阶段高度热启动，最佳候选联合保存姿态与高度。场景设置上限为 0 时恢复硬高度，旁路输出高度严格为 0。回零为软损失，其他目标仍可能使离座后的高度偏移非零。

Compute 增加高度和前后各一对差分候选，总计 149×4；Work 为 377 float（前373项保存姿态、高度、回退与骨盆位移状态，后4项保存前后自由度），Values 为 596 float4。投影后 Pack 将 144 维姿态、dy 与五阶段 360 维姿态修正及5个米制骨盆修正再加末尾前后偏移打包为 511 float，一次最终 GPU 读回；`GpuIkStep.Output` 仍为 144 维，另提供 `HeadYOffset` 和 `Corrections[365]`，`FluidSequenceRuntime.Finish` 接收二者并仅在结果接受时提交历史。高度不改写 tracker、HMD 相机、训练输入或 ONNX 形状；无需重训/重导出网络。Python 参考启用高度时使用 `forward_with_height` 返回 `(edited_residual, dy)` 并将 dy 传入几何；仅返回残差的 `forward` 在非零高度上限或修正量时序权重下拒绝执行，防止调用者静默丢弃高度。分段参考链另以 `height_offset` 暴露末阶段结果。

位置和 OBB 统一到本次推理的 Head-yaw 坐标系，长度单位为角色坐标中的米；
四元数采用 Unity xyzw。根位置/旋转是该 Head-yaw 系中的全局量，骨盆位置与
骨骼四元数仍为各自父节点局部量。环境世界坐标先通过实际角色显示父节点的
逆变换，再去除本次 Head 位置和 yaw；不重复应用 Tracker 已包含的 bodyScale。
仅支持正的统一角色缩放、保持竖直的角色父坐标系与水平座面/扶手；拒绝退化
或剪切 OBB。模型骨架及代理使用标定时的角色尺度，不逐帧读取蒙皮。

`collision_profile.json` 保存 `capsules [2,4]`（左右臂 × 肘侧/中段/腕侧/手掌）和
`seatCapsules [3]`（臀部、左大腿、右大腿，骨骼 0/1/2）以及 `restLocalPositions [24,3]`。每个胶囊含 `bone`、骨骼局部端点 `start/end [3]`、
正半径 `radius`；骨骼顺序固定为 `[[18,18,18,20],[19,19,19,21]]`。几何由 PuppetMaster
生成工具创建的 CapsuleCollider 导出，其余全身 Collider 不进入配置或计算。
模型配套 JSON 保存骨架父节点和实际优化参数；Shader 从配置上传一次常量。
编辑 Collider 后重新导出 profile 并同步 `collision_compute.json.profile`；现有
condition/step/projection 分段 ONNX 不含碰撞几何常量，因此无需重训或重新导出它们。
角色骨架必须是当前 SMPL24 层级。模型、配套 JSON 与标定作为 Unity 普通资源
放在 `Assets/FLUID/Models`，由 Inspector 引用；不作为 StreamingAssets TextAsset 加载。

每个 DiT 干净残差输出后，编辑关节 16、17、18、19、3、6、9、12、1、2、4、5 对应的
72 个**归一化** rotation6D 残差分量；Predictor 固定。默认内层 5 次更新，
中心差分步幅 0.001、初始步长 0.0001、测量尺度 0.01，
`h_k=h_0(1-0.99*k/steps)`。先验方差为当前 respaced sampler 的
`max(1-alpha_bar_t, 0.000001)`，无 Langevin 噪声。平方误差求和，
双腕、Hip、双脚的有效位置误差只惩罚超过 0.02 的部分，安全间隙 0.002。
每个阶段先比较原始残差与“原始残差＋上次已接受的同阶段修正”两个起点，仅在当前目标激活且总能量严格降低时使用历史候选。每轮按原步长、1/2、1/4、1/8 评估四个候选，选第一个有限且总能量不增加的结果；均不满足时保留更新前姿态及高度。高度候选仍投影到原有位置／速度边界；20次内循环与5次DiT调用不变。四候选各计算四个显示插值位置，使用独立几何 kernel 并行评估，再统一验收，不增加CPU读回。诊断 Trace[1100+3*k:1103+3*k]（前20轮）记录起点能量、接受能量和步长比例（失败为0）。
更新统一为 `r_A <- r_A - h_k * grad_A(E)`，其中
`E = ||r_A-r_hat_A||^2/(2*sigma_t^2) + (sum(d^2)+back_collision+torso_curve+chest_head+sum(e_wrist^2)+body_tracking_weight*sum(e_body^2)+spine_bend+head_y_weight*dy^2+head_y_smooth_weight*(dy-dy_prev)^2)/(2*m^2)`。
另加修正量时序项 `0.5*correction_smooth_weight*w*sum((delta-delta_prev)^2)`，其中 `delta=r_A-r_hat_A`，只作用于脊柱 3/6/9/12、髋 1/2、膝 4/5。每个采样阶段使用上一已接受帧的同阶段修正，解析梯度为 `weight*w*(delta-delta_prev)`，最佳候选使用同一能量；当前场景权重 500，设为 0 关闭。新序列及超过 0.1 s 的间隔不施加此项；有效间隔权重 `w=min(1,(1/30)/dt)`。历史方向差向量先乘 pose_scale、旋转到本帧 Head-yaw，再除 pose_scale；不使用归一化均值。旁路和无活动目标输出零修正，未接受的 GPU 请求不推进历史。Python 单阶段 `forward_with_height` 的 `correction_context [1,74]` 接收当前阶段72姿态量、有效权重与实际骨盆位移；分段参考的 `corrections [1,365]` 供调用端在接受后存储。此项约束修正变化而非锁定上一帧绝对姿态。
骨盆前后连续性项沿座面局部 Z 的水平归一化方向 n，保存每阶段 `d=dot(n,p_selected-p_original)`，在下一已接受帧相同阶段以 `p_original_current+n*d_previous` 的前后投影为目标。令 `e=dot(n,p_candidate-p_original_current)-d_previous`，增加平滑目标 `pelvis_forward_smooth_weight*w*release*e²/(2*m²)`。每次步长回退额外要求总能量下降至少覆盖 `pelvis_forward_smooth_weight*w*release*pelvis_forward_update_cost*abs(dot(n,p_trial-p_current))/(2*m²)`；该位移代价只参与接受判断，不参与有限差分梯度，避免绝对值零点干扰方向。小收益移动被拒绝，足够大的碰撞收益仍允许更新，不另加显示平滑或固定锚点。仅座面顶面模式启用；目标与当前显示骨盆前后距离/dt 在 0.15 m/s 以下全权重，0.15–0.6 m/s 线性释放，0.6 m/s 以上关闭。新序列、跟踪中断和丢弃请求沿用原历史提交规则，旁路／无活动目标写零位移。米制位移是椅子轴上的标量，不随 Head-yaw 旋转；姿态修正仍执行原坐标变换。有限差分、热启动和步长回退共用此目标。Trace[1012:1016] 记录原始前后位置、目标、有效权重、最终位移；重跑 head_height.json 另记录五阶段 pelvisForwardCorrections。网络 ONNX 不变，部署上下文及一次读回扩展为上述形状，需同步更新两端；旧形状不混用。

仍保留扩散残差先验，不使用额外混合回拉或 `postedit_w` 参数；
最佳候选筛选也使用该能量，不另设保真上限。`e_body` 为 Hip／双脚的有效位置误差，默认 `body_tracking_weight=0.15`；双腕权重仍为 1，跟踪旋转不变，头部 Y 按上述有界软约束处理。
躯干与双臂、下肢共同参与损失，149 个候选（原点、72 对姿态扰动及高度、前后各一对扰动）计算共同目标的中心差分。
各候选重新计算跟踪旋转投影、头部定位、骨盆位置、局部姿态插值和全身 FK；仅非活动全局旋转可复用。
脊柱分配项使用终点 3/6/9/12/15 的父子局部单位四元数（统一到 w>=0），取局部俯仰代理量 `b=2*q.x`。
对相邻骨段加入 `spine_bend_scale^2 * sum(max(abs(b[i+1]-b[i])-spine_bend_tolerance,0)^2)`。
头部 15 的全局旋转仍由跟踪器固定，只通过其父子关系参与损失，不新增活动变量。
默认尺度 0.45 米、容差 0.15（小角度约 8.6°），保留站立录制的自然差异，抑制坐下时局部大幅反折；这是部署姿态正则，不是人体关节硬限位。
它取代相对原始残差的局部旋转保真项，避免把已有折叠当作应保留的参考；即使无碰撞，超出容差也参与同一 posterior correction，并沿原采样路径反馈历史。
新增累计躯干项 `torso_curve_scale^2 * max(sum(abs(b[0:4]))-torso_curve_limit,0)^2`，默认 0.2／0.7；刚体整体前倾不产生额外局部弯曲。胸—头项从 `inverse(q[9])*q[15]` 的单位四元数提取 `abs(2*q.x)`，超过 `chest_head_limit=0.35` 后平方惩罚，尺度 `chest_head_scale=0.15`。两项同样进入激活判定，不能被“无接触”提前退出跳过；都只在最终候选姿态计算。
`profile.backCapsules` 存储预留躯干 Collider 的骨骼局部胶囊（当前骨骼 3、9），`profile.backrest[15]` 为靠背在座面坐标系中的中心、三轴、半尺寸，长度采用角色米制。GPU 从原 `seated_context` 的座面 OBB 转换至靠背局部系；靠背无需额外动态几何输入。背部胶囊逐个对靠背 OBB 计算穿透平方损失，并按有限 x/y 范围限制靠背 +Z 前表面，避免穿到后侧。座面上下文 mode=0 同时关闭靠背。形状与靠背相对位置变动须重新导出；运行时不使用 PhysX 解算，也不拟合网格。
上述距离均为角色坐标米制；统一缩放角色后世界距离也随之缩放。
如要求任意角色缩放下仍固定世界 2 mm，须按该缩放重新导出间隙常量，当前
上述输入不另传尺度标量。

小臂和手掌都采用胶囊轴线与 OBB 的最小有符号距离减半径；端点按各候选
的绑定骨骼位置和旋转变换，小臂保留轴向扭转的完整旋转。
每臂对每个扶手取三段小臂平方惩罚的最大值，再加手掌惩罚，避免重叠段重复计权。
臀部和双侧大腿与座面配对，三个独立部位的平方穿透相加，再乘 `seat_collision_weight`（参考默认 1，场景当前 2），避免最深接触遮蔽另一侧梯度；不增加向下吸附。该权重只乘座面接触项，不放大靠背、扶手、跟踪或姿态正则，GPU 与 Python 参考一致；诊断穿透仍报告未加权的实际代理距离。
上方模式在有限的、按代理半径扩展的座面/扶手 xz 范围内约束上表面。
碰撞评估包含旋转投影、Tracker 旋转覆盖、0.25/0.5/0.75/1.0 的局部姿态
插值和 FK；独立骨盆托举及对应脊柱 CCD、双臂补偿已删除。脚锁、腿部 IK 和防穿地仍在显示端执行。
每次优化比较初始残差及有限的有效迭代，保留最低目标值；初始无碰撞且脊柱分配在容差内则原样返回。
Shader 在 GPU 上记录无约束违例标记，跳过余下无效候选评估；不增加 CPU 往返。
编辑后的残差参与 epsilon 重算和 DDIM 更新，五步仍只调用五次 DiT。

Unity 的接触状态使用上次有效显示上肢和当前 Tracker，默认接近 0.03、
抬离 0.05、范围滞回 0.02；单次推理内固定。ONNX 的显示过渡只使用请求时
快照；四个采样点不保证连续轨迹零穿透。实际显示诊断单独写入会话目录的
`collision_*.jsonl`，含配置、环境快照、穿透/约束剩余量、腕误差/位置、
推理耗时、显示延迟和跳过样本计数；关闭时只写关闭标记，不执行几何检查。
实际显示不再追加推手修正。

标定、导出、检查命令与本机验收结果见 [部署说明](export/collision_postprocess.md)。


坐姿头部前后自由度（2026-10-12）：沿座面局部 Z 的水平归一化方向 n，候选根位置增加 n*df，保持侧向位置、头部跟踪旋转和原 Y 自由度。当前场景 head_forward_limit=0（前后硬定位；此前0.02 m为试验值）、head_forward_weight=1、head_forward_smooth_weight=0.5、head_forward_speed=0.05 m/s。优化坐标 df/0.1，共同测量增加 (weight*df²+smooth*(df-df_prev)²)/(2*m²)。顶面接触模式2内服从位置/速度边界；离开模式2后按速度上限向0释放。五个采样阶段共用本帧速度边界，上一阶段值热启动，最佳候选同时保存姿态、dy、df；10次迭代。HeightOutput=[dy,df]，Pack[510]=df。GpuIkStep 另提供 HeadForwardOffset 和本次提交时的 HeadForwardAxis（Head-yaw 局部坐标），Finish 转回世界方向重建根位置；只在接受结果时提交 df 历史。新序列清零，丢弃请求不提交；旁路输出0。Python forward_with_height 的 forward_context/initial_forward 分别为上次接受值与本阶段初值，前后结果通过 forward_result["offset"] 返回；分段参考 chain.forward_offset 暴露最终值。此变量不修改 tracker 输入或 HMD 相机，网络形状不变。Trace[1030+2*k:1032+2*k] 保存前5轮 df/0.1 与梯度，Trace[1020:1022] 保存末次候选归一化 df 和最终输出米制 df。


骨盆前后历史基准修正（2026-10-12）：每阶段 p_original_current 使用该阶段未修正残差重建、且不含头部前后偏移的骨盆位置。GPU 从初始候选的座椅轴投影扣除 df_initial；Python 用 forward=0 重建基准。于是 d=dot(n,p_selected-p_original_current) 包含姿态变化与完整 df_selected，而非只包含 df_selected-df_initial。时序目标、动作释放判断和历史提交共用这个基准；不新增字段或权重，10次迭代和头部±2cm保持。已有运行中的旧语义历史不能沿用，更新代码后重新启动推理。

当前保留较稳回放配置：10次迭代，头部前后自由度关闭，Y上限0.05m，阶段位置差释放判断。原始tracker运动释放试验已撤回，运行时与Python恢复上述3项/2项头部上下文；原网络不变。
