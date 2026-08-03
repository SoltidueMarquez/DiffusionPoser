# DiffusionPoser 审阅项目图

本文件提供当前主链路的导航与审阅锚点。每次结论仍需回到当前 HEAD 源码、`contract.md`、active checkpoint 参数和实际评估产物刷新。

## 1. 任务目标与应用场景

目标是在 60 Hz 在线场景中，根据 Head、双手以及可选 Hip/双脚 Tracker 的世界测量，连续重建人体 24 关节全身姿态和 Actor Root，使 Unity 角色在 Tracker 配置变化、临时掉线和重连时仍能稳定运动。

典型应用是 VR/实时数字人全身驱动：

- 最低固定输入是 Head、左手和右手三点。
- 可动态增加 Hip、左脚和右脚，形成六点配置。
- 允许三点/六点切换，以及六点配置下两个非 Head Tracker 同步掉线和重连。
- 每帧输出 24 关节世界旋转、Unity local delta rotation、Root 世界位置/yaw、Pelvis 高度和世界关节位置。
- 在线 runtime 从空历史冷启动，不依赖 GT pose warmup。

当前上游假设：Unity 已把每个 Tracker 标定并转换到对应关节，算法直接假定 Tracker 与映射关节的位置和旋转重合。当前训练不覆盖安装偏移、标定误差、测量噪声、outlier、单点掉线或任意掉线组合。

## 2. 核心表示

### 时序和坐标系

- 历史长度固定 60 帧，当前目标是第 61 帧。
- `pose_history`、`tracker_history` 位于上一帧 Head-yaw/floor 参考系 `C_(n-1)`。
- `current_target`、`current_tracker` 位于当前参考系 `C_n`。
- 当前 Head 水平位置与 `floor_y` 定义参考系原点；Head yaw 定义水平朝向。
- Head forward 水平投影退化时沿用上一合法 yaw。

### 主要张量

| 字段 | 形状 | 审阅语义 |
|---|---:|---|
| `x/current_target` | `[B,144]` | 当前 24×rotation6D 完整扩散目标 |
| `pose_history` | `[B,60,144]` | 过去 deployed pose |
| `tracker_history` | `[B,60,6,13]` | 过去测量与状态 |
| `current_tracker` | `[B,6,13]` | 归一化当前条件 |
| `current_tracker_raw` | `[B,6,13]` | 投影、FK 和几何 loss 使用的物理量 |
| `trajectory_history` | `[B,60,5]` | past-only Head 轨迹缓存 |
| `current_trajectory` | `[B,1,5]` | 当前 Head 运动控制 |
| `valid_frame_mask` | `[B,60]` | 左补零历史的有效帧 |
| `hard_rotation_state` | `[B,6]` | 本帧 DDIM 全程固定的 hard 集合 |
| `future_leg_target` | `[B,3,8,6]` | 未来三帧双腿辅助监督 |
| `contact_target` | `[B,2]` | 左右脚接触监督 |

每个 Tracker 的 13D 顺序固定为：

```text
position3 + rotation6D + configured + measured_valid + d_off + d_on
```

前 9 维无效测量必须清零；只有前 9 维参与 Tracker normalizer。`d_off/d_on` 的物理值是 `[0,60]` 整数帧数，模型输入除以 60。

Head trajectory 为：

```text
[delta_x_prev_head_ref, delta_z_prev_head_ref,
 normalized_head_height, sin(delta_yaw), cos(delta_yaw)]
```

## 3. 端到端代码链路

### A. Source 转换

- 入口：`data_converter/amass_to_realtime_pose.py`
- 几何：`data_loaders/body_fbx_kinematics.py`
- 验证：`data_loaders/realtime_pose_validation.py`
- 输入：AMASS/SMPL-H、SMPL body model、Unity `body_fbx_rest.json`
- 输出：60 Hz source NPZ，含 local delta pose、Root、Tracker、world joints、骨架和 stationary 标签。

Source 可以沿用；当前 task、normalizer、checkpoint 与旧契约不兼容。

### B. Tracker 场景与 task store

- 场景常量：`data_loaders/sensor_masking.py`
- 可靠性配置：`data_loaders/realtime_pose_config.py`
- 绝对事件线：`data_loaders/tracker_timeline.py`
- 可靠性/hard：`data_loaders/tracker_reliability.py`
- 生成入口：`data_loaders/generate_realtime_pose_tasks.py`
- mmap 存储：`data_loaders/realtime_pose_task_store.py`

五类训练场景为 `fixed_six`、`fixed_three`、`three_to_six`、`six_to_three` 和 `two_point_dropout_reconnect`。动态事件按 `global_seed + source_id` 固定在 source 绝对时间线上，重叠窗口应获得相同状态。

### C. Dataset、冷启动和归一化

- Dataset：`data_loaders/realtime_pose_dataset.py`
- DataLoader：`data_loaders/get_data.py`
- Normalizer 生成：`data_loaders/compute_realtime_pose_normalizer.py`
- Normalizer：`utils/normalizer.py`

`TaskRequest` 同时选择 task、场景、rollout 长度和初始 history length。冷启动命中时，history length 在 0～59 中采样；Dataset 从虚拟会话起点重放 duration/hard 状态，三类历史左补零。

### D. 条件编码和 TargetDiT

- 当前观察：`model/realtime_pose_observation_encoder.py`
- past-only motion prior：`model/realtime_pose_motion_encoder.py`
- 主模型：`model/realtime_pose_target_dit.py`
- 工厂：`utils/model_util.py`

DynamicObservationEncoder 输出 state、非 Head position、全部 rotation、history summary、逐 Tracker 可靠性 `kappa` 和五区域覆盖度 `rho`。

RegionalMotionEncoder 只读取过去 pose、Tracker history summary 和 trajectory history，输出 global/pelvis/left-leg/right-leg 的 60 帧 temporal token 与四个 latent。当前 trajectory 单独编码，不得进入 past-only latent。

TargetDiT 把 144D 拆成 24 个 joint token，按 torso、左右臂、左右腿路由 context、position、rotation、motion prior 和 trajectory FiLM，输出 raw 144D、future-leg 和 contact logits。

### E. 训练与 raw/deployed 分离

- 训练入口：`train/train_diffusionposer.py`
- 训练循环：`train/training_loop.py`
- 扩散：`diffusion/gaussian_diffusion.py`
- 几何损失：`diffusion/realtime_pose_losses.py`
- rollout 时序损失：`diffusion/realtime_pose_temporal_losses.py`
- 投影：`diffusion/realtime_pose_projection.py`

完整 144D 都被加噪。TargetDiT 先产生 `raw_pred_xstart`；随后所有 joint 先投影到合法 SO(3)，hard Tracker 对应 joint 再被测量旋转替换，得到 `deployed_pred_xstart`。

- raw：diffusion、global/local rotation、Tracker rotation 学习。
- deployed：FK、Tracker position、Root/world geometry、rollout 和部署缓存。
- 辅助头：未来双腿、脚接触；foot-slide 在 rollout 相邻 deployed 帧上计算。

### F. 采样和在线 runtime

- 单窗口：`sample/reconstruct_stream.py`
- 离线连续 rollout：`sample/reconstruct_rollout.py`
- 状态化 runtime：`sample/realtime_pose_runtime.py`
- Projected DDIM：`GaussianDiffusion.projected_ddim_sample[_loop]`

每帧只运行一次 `prepare_conditioning()`；DDIM 多步复用其输出。投影后的 `x0` 必须重新推导 epsilon，再更新下一噪声状态。最终 deployed pose 经一次 Head-Anchored World Resolver 恢复 Root、世界关节和 Unity local rotation。

### G. 长序列评估

- 闭环生成：`sample/evaluate_longseq_eval_set.py`
- 指标：`eval/evaluate_realtime_pose.py`

长序列从首帧空历史开始逐帧运行 `RealtimePoseRuntime`，不使用 GT warmup。重点指标包括：

- raw/deployed rotation error；
- MPJRE、MPJPE、MPJVE、MPJAE；
- raw/deployed hard Tracker rotation、soft Tracker rotation；
- 非 Head Tracker position error；
- Root yaw/XZ、Pelvis height、Root step delta；
- future-leg、contact、foot slide；
- sampling/e2e latency 和 60 FPS 预算命中率。

必须按场景、`d_off`、重连 `d_on=1..15`、hard/soft 状态以及 cold-start/steady-state 分组，不只看全局均值。

## 4. 当前必须守住的不变量

1. 新链路没有 `known_target/known_mask/inpaint_mask/inpaint_cond`。
2. `configured` 表示设备配置，`measured_valid` 表示本帧有测量，二者不可互换。
3. `d_off` 只在 configured-but-missing 时累积；`d_on` 只在 configured-and-valid 时累积。
4. Head 始终 configured、valid 且 rotation hard；Head position 不进入普通 position token。
5. 非 Head Tracker 重连后先 soft，连续有效 15 帧后 hard。
6. hard 集合在同一帧所有 DDIM step 中固定。
7. motion prior 严格 past-only；当前 Tracker 和当前 trajectory 不得泄漏进去。
8. rollout 和 runtime 缓存只写 deployed prediction。
9. 冷启动 padding 是归一化空间的字面量零，duration 不继承裁掉历史。
10. Resolver 只验证 hard rotation；soft 测量不能被误写成严格约束。

## 5. 文档与工程边界

- `contract.md` 是 Python 数据结构唯一契约；维度变化只在此维护。
- `documents/算法架构.md` 描述设计目标和公式；仍需核对当前代码是否完整实现。
- `README.md` 是运行入口说明，不应作为张量语义权威来源。
- 当前 README 明确提示 Unity 与导出链路尚未同步；在同步和跨 runtime 测试完成前，不宣称 Unity 部署闭环已完成。
- 旧实验 notes、旧 x277/211D/214D 文件和旧 checkpoint 只能作为历史背景，不能证明当前 144D 动态链路的效果。

## 6. 测试路由

- 数据/状态：`tests/smoke/data_pipeline/test_dynamic_task_contract.py`、`test_tracker_reliability.py`
- 架构：`tests/smoke/train/test_realtime_pose_architecture.py`
- raw/deployed 与 rollout：`tests/smoke/train/test_raw_deployed_training.py`
- 梯度：`tests/smoke/train/test_training_gradients.py`
- Projected DDIM：`tests/smoke/sample/test_projected_ddim.py`
- 冷启动 runtime：`tests/smoke/sample/test_realtime_pose_runtime.py`
- 长序列与指标：`tests/smoke/sample/test_evaluate_longseq_eval_set.py`、`tests/smoke/eval/test_realtime_pose_eval.py`

测试文件存在不等于测试已在当前环境通过；每次报告必须给出实际执行命令和结果。
