# Loss Ablation V1 实验记录

## 基线

- baseline tag: `baseline/loss-ablation-v1`
- baseline commit: `4a7d5f2 Add tracker corrections to Unity stream simulation`
- experiment branch: `codex/loss-ablation-v1`
- 创建日期: `2026-05-27`

## 实验目的

- 对 loss 相关改动做第一轮 ablation，记录每次改动的动机、配置和结果。
- 历史实验记录中的 `realtime_pose_v2_contact` 已废弃；新实验应使用当前 `realtime_pose_body_fbx_local_root_y0_v1` stationary5 契约。

## 产物约定

- 不提交 `dataset/`、`runs/`、`output/`、`save/` 或 checkpoint/data 二进制产物。
- 训练、采样和评估命令默认使用 `conda run -n diffusionposer5070 <command>`。
- 每个实验条目记录代码改动摘要、关键参数、使用的数据切分、checkpoint/tag、指标和主观观察。

## 实验条目

### 001 - non-hip Huber tracker pos loss

- 改动摘要: `sensor_reprojection_pos_loss` 改为非 hip tracker 的 vector-distance Huber，并按 timestep 对低噪声阶段加权。
- 训练配置: 建议首轮使用 `--tracker_pos_loss_weight 10 --tracker_pos_huber_beta 0.05 --tracker_pos_timestep_min_weight 0.1 --tracker_pos_timestep_gamma 2.0`
- 采样/评估配置:
- 结果指标:
- 结论:

### 002 - IK warm-start diffusion refinement

- 改动摘要: 新增 `tracker_pose` IK 初始化采样路径，先用第 60 帧历史 pose 和当前 valid tracker 的位置/旋转做第 61 帧粗解，再通过已有 `init_image + skip_timesteps` 从中间扩散步去噪；不修改训练和 forward noising。
- 关键文件: `sample/ik_initializer.py`, `sample/reconstruct_stream.py`, `sample/simulate_unity_stream.py`, `sample/evaluate_unity_stream_source.py`, `utils/parser_util.py`, `tests/smoke/sample/*`。
- 采样参数默认值: `--ik_init_mode random --ik_init_timestep -1 --ik_init_iterations 16 --ik_init_lr 0.03 --ik_init_pos_weight 1.0 --ik_init_rot_weight 0.2 --ik_init_reg_weight 0.01 --ik_init_delta_limit 0.15`。
- 训练配置: 不重训。
- 测试命令: `conda run -n diffusionposer5070 pytest tests/smoke/sample` 通过，17 passed；`conda run -n diffusionposer5070 pytest tests/smoke/train/test_train_entrypoint.py` 通过，8 passed。
- Run 目录: 未生成训练、采样或评估 artifact。
- 结果指标: 待后续用真实 checkpoint 对比 `random diffusion`、`IK only`、`IK-init diffusion` 和 `IK-init + post IK`。
- 结论: 代码路径已接入并通过 smoke tests；当前分支可进入真实 checkpoint 的采样消融。
