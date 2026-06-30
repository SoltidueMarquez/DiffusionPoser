# GlobalPose Oracle Tracker Eval Notes

记录时间：2026-06-30

## 当前状态

- GlobalPose 官方仓库已放在 `dataset/external/globalpose/repo`，当前 commit 为 `eef0e3b59b10104a80970261c3ad46d6c4681f29`。
- 官方资产已准备：
  - `dataset/external/globalpose/repo/models/SMPL_male.pkl`
  - `dataset/external/globalpose/repo/data/weights.pt`
  - `dataset/external/globalpose/repo/data/test_datasets/dipimu.pt`
  - `dataset/external/globalpose/repo/data/test_datasets/totalcapture_dipcalib.pt`
  - `dataset/external/globalpose/repo/data/test_datasets/totalcapture_officalib.pt`
- `globalpose38` 已创建，当前是 `Python 3.8.20 + PyTorch 2.4.1 CPU`。
- 官方 `GPNet.forward_frame` 已完成单帧 smoke。完整 `test.py` 尚未全量跑，因为当前 `globalpose38` 不是 CUDA PyTorch，全量 CPU 运行成本过高。
- 已新增 DiffusionPoser 转换器：`data_converter/globalpose_to_realtime_pose.py`。
- 已新增 GlobalPose 风格平移漂移指标：`eval/globalpose_metrics.py`。

## 环境说明

尝试安装 CUDA PyTorch 时遇到两个限制：

- PyTorch wheel 下载多次触发 SSL `ASN1: NOT_ENOUGH_DATA`。
- Conda 安装 CUDA PyTorch 多次长时间求解/下载超时。

因此当前先用 CPU 版 PyTorch 完成 GlobalPose 官方单帧 smoke、资产校验和 DiffusionPoser oracle tracker 数据转换。后续要复现论文官方数值，需要先把 `globalpose38` 替换成 CUDA PyTorch。

## 已生成的正式数据

正式 TotalCapture Official oracle tracker source：

```text
dataset/generated/sources/realtime_pose_stationary5_v1/globalpose_totalcapture_officalib_oracle_tracker/
```

结果：45 条 source，来自 `totalcapture_officalib.pt`。

正式 test task set 使用短路径名，避免 Windows 路径过长：

```text
dataset/generated/tasks/realtime_pose_stationary5_v1/gp_tc_off_ot_tasks/20260701_002641_ft1/
```

结果：45 个 full-tracker test task，每条 GlobalPose 序列 1 个 61 帧窗口。

正式 longseq eval set：

```text
dataset/generated/longseq_eval/realtime_pose_stationary5_v1/globalpose_totalcapture_officalib_oracle_tracker/ft1_longseq/
```

结果：45 条连续序列，帧数范围 2018 到 6775，总帧数 176249。

## 复现命令

转换 TotalCapture Official source：

```powershell
conda run --no-capture-output -n diffusionposer5070 python -m data_converter.globalpose_to_realtime_pose `
  --globalpose_dataset dataset/external/globalpose/repo/data/test_datasets/totalcapture_officalib.pt `
  --dataset_name totalcapture_officalib `
  --source_set_name globalpose_totalcapture_officalib_oracle_tracker `
  --overwrite
```

生成 full-tracker test tasks：

```powershell
conda run --no-capture-output -n diffusionposer5070 python -m data_loaders.generate_realtime_pose_tasks `
  --source_set_name globalpose_totalcapture_officalib_oracle_tracker `
  --task_set_name gp_tc_off_ot_tasks `
  --splits test `
  --split_dir= `
  --samples_per_file 1 `
  --mask_policy full `
  --run_name ft1 `
  --overwrite
```

生成 longseq eval set：

```powershell
conda run --no-capture-output -n diffusionposer5070 python -m data_loaders.build_realtime_longseq_eval_set `
  --task_dir dataset/generated/tasks/realtime_pose_stationary5_v1/gp_tc_off_ot_tasks `
  --task_run latest `
  --output_root dataset/generated/longseq_eval/realtime_pose_stationary5_v1/globalpose_totalcapture_officalib_oracle_tracker `
  --run_name ft1_longseq `
  --split test `
  --min_frames 2000 `
  --overwrite
```

## 验证

已运行：

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/data_pipeline/test_globalpose_converter.py -q
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/eval/test_globalpose_metrics.py -q
```

结果：全部通过。

已手动验证：

- 正式 source manifest 行数：45。
- 正式 task manifest 行数：45。
- 读取第一个 task 后，`encode_realtime_pose_features` 输出 `(61, 214) float32`。
- 正式 longseq manifest 行数：45。

## 解释和限制

这个 oracle tracker 版本只用于验证 DiffusionPoser 在 GlobalPose/TotalCapture 动作分布上的流程和重建能力：

```text
GlobalPose GT pose/tran -> SMPL FK -> tracker_pos_world/tracker_rot_world_6d -> realtime_pose source/task
```

它不能和 GlobalPose 论文里的 IMU benchmark 数字直接公平比较，因为 GlobalPose 论文输入是 IMU，而这里给 DiffusionPoser 的 tracker 来自 GT pose/tran。

对于 TotalCapture Official Calibration 和 DIP Calibration，oracle tracker 由 GT pose/tran 派生，标定差异不会进入 DiffusionPoser 输入；如果后面要比较校准鲁棒性，必须设计 IMU 或预测 tracker 输入版本，而不是这个 oracle tracker 版本。

## 后续

- 安装 CUDA PyTorch 到 `globalpose38` 后，全量运行官方 `test.py`，保存官方控制台输出和 `data/temp/results`。
- 用现有 checkpoint 在 `gp_tc_off_ot_tasks` 或 `ft1_longseq` 上跑 DiffusionPoser 采样/rollout。
- 对 DiffusionPoser 输出接入 `eval/globalpose_metrics.py` 的 1m-7m travelled-distance translation drift。
- 如需覆盖 `totalcapture_dipcalib.pt` 和 `dipimu.pt`，复用同一转换器分别生成对应 source/task/longseq eval set。
