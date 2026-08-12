# Realtime Pose v2 训练前固定长序列基线（2026-07-14）

> 历史记录：本文的复制版 eval set、manifest 与当时的 normalizer 路径仅用于复盘旧实验。当前长序列评估直接读取 `source_dir + split_dir + split`，操作方式见 `documents/复现.md`。

## 定位

本记录用于 Stage A 开始前的工程连通性基线。模型仍是旧分布训练得到的
`model000100000.pt`，但输入使用 v2 normalizer 和外部 Root Resolver，因此这些数值
不能作为正式论文结果；后续每 5k checkpoint 应复用相同 eval set、mask 和采样配置作纵向比较。

## 代码快照

- Python：`a060b89 feat: add sparse tracker root resolver v2 pipeline`
- Unity：`204ba27 feat: add realtime sparse tracker root resolver v2`
- 快照前回归：Python `311 passed, 1 skipped`；Unity batchmode contract v2 self-test passed。
- 评测时额外修复了 longseq 参数加载：checkpoint 只恢复 model/diffusion 参数，不能覆盖评测 seed 和 Tracker 扰动配置。
- 参数隔离修复后完整 smoke：`312 passed, 1 skipped`。

## 固定输入

- Eval set：`dataset/generated/longseq_eval/realtime_pose_stationary5_v1/amass_60hz_v2_pre_stage_a/20260714_pre_stage_a_seed10`
- Manifest SHA256：`1f7f0a3a2e1b7456738e321d6ce8a2709cfa8cde303c7c440fa4c26921640696`
- 当前序列：`CMU/55/55_13_poses.npz`，4869 帧，60 FPS，非镜像。
- Checkpoint：`runs/realtime_pose_stationary5_v1/stationary5_target_dit/20260626_192142_s5_tdit_seed10/model000100000.pt`
- Checkpoint SHA256：`52db75718dc85d7ed9a957037a14ed13f7139b7fbc799cc9595a7a29aa6be3a2`
- Normalizer：`dataset/generated/normalizers/realtime_pose_stationary5_v1/amass_60hz_v2_train/20260714_003252_v2_base_online_seed10`
- `mean.pt` SHA256：`511ca2bd8129d2e5721182330ccdbbbd18d3658be65b01e507405b53a1a7eb07`
- `std.pt` SHA256：`5ec680598aeeeec482cb0bf54ab2d3c46940962ab51c105d27f7077e5602dc0`

固定运行配置：DDIM10、`history_pose_source=predicted`、`warmup_target_source=first_frame`、
Root correction 开启、Tracker IK 关闭、视频关闭。Tracker mask 为
`fixed_categories/standard_three`、seed 10；全程仅 Head/LeftHand/RightHand 有效，
latency、burst dropout、outlier、position/rotation noise 全为 0。

## 结果

结果目录：
`output/longseq_eval/realtime_pose_stationary5_v1/pre_stage_a_model100k_standard_three_seed10`

- 总帧数 4869；warmup 60；计分帧 4809。
- Root-relative MPJPE：264.015 mm。
- Root XZ mean/final error：29.300 / 16.603 cm。
- Root drift：12.852 cm/min。
- Head/Hand reprojection position mean/P95：12.222 / 36.501 cm。
- Head/Hand reprojection rotation mean/P95：71.776 / 164.838 deg。
- Stationary F1：0.2106；false-lock 0.7448；missed-lock 0.4921。
- DDIM p50/p95：44.660 / 56.262 ms。
- Resolver p50/p95：0.794 / 1.006 ms。
- End-to-end p50/p95：46.845 / 58.506 ms；20.948 FPS。
- Root source：61 帧 reset、4808 帧 head_fk；无 reconnect，reconnect alpha 恒为 0。
- 输出 NPZ 共 48 个数组，所有数值数组均为有限值；每帧有效 Tracker 数恒为 3。

第一次诊断运行被 checkpoint 中的训练扰动参数污染，已保留在
`pre_stage_a_model100k_standard_three_seed10_invalid_checkpoint_args_override`，不得作为基线。
