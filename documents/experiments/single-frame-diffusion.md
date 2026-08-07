# 单帧扩散实验记录

## 基线

- baseline tag: `baseline/single-frame-diffusion`
- baseline commit: `908a8e1 修改坐标系与添加loss`
- experiment branch: `codex/single-frame-diffusion`
- 创建日期: `2026-08-07`

## 实验目的

- 将当前工作区中的单帧扩散、历史条件、训练损失、长序列推理与评估改动从多帧实验分支中独立出来。
- 后续单帧方案的训练、采样和消融均在本分支继续，避免与 11 帧共同扩散方案混淆。

## 产物约定

- 不提交 `dataset/`、`runs/`、`output/`、`save/`、checkpoint 或生成的二进制数据。
- 训练、采样和评估命令默认使用 `conda run -n diffusionposer5070 <command>`。

## 实验条目

### 001 - 分离当前单帧工作区

- 改动摘要: 将分支创建时工作区内已有的模型、扩散、损失、训练、长序列推理、评估、文档与 smoke test 修改整体归入单帧实验分支。
- 关键文件: `model/realtime_pose_spatiotemporal_dit.py`、`diffusion/gaussian_diffusion.py`、`diffusion/realtime_pose_losses.py`、`train/training_loop.py`、`sample/realtime_pose_runtime.py`、`eval/evaluate_realtime_pose.py`。
- 测试命令: 本次仅做 Git 分支归档，未重新运行测试；当前工作区已有对应 `tests/smoke/` 修改。
- Run 目录: 不适用。
- 结果指标: 不适用。
- 结论: 单帧改动已与原 `codex/dynamic-topology-inpainting-v2` 分支建立清晰边界。

### 002 - 单帧长序列抖动来源诊断

- 改动摘要: 增加同条件重复采样诊断，并为长序列评估增加预测历史/GT 历史切换，用于隔离扩散随机噪声和历史误差反馈。
- 关键文件: `sample/diagnose_realtime_pose_sampling_variance.py`、`sample/evaluate_longseq_eval_set.py`。
- 检查命令: `conda run -n diffusionposer5070 python -m py_compile sample/evaluate_longseq_eval_set.py sample/diagnose_realtime_pose_sampling_variance.py`。
- Run 目录: `D:\dp\single_frame_jitter_diagnostics_90k_20260807`。
- 结果指标: 固定噪声使 MPJVE 下降 81.78%、jitter 下降 93.73%，但 MPJPE 上升 55.50%；GT 历史使 MPJPE 下降 77.66%、Root yaw 下降 89.41%，但 jitter 上升 67.22%。
- 结论: 逐帧独立扩散噪声是高频抖动的主要来源；预测历史误差是绝对姿态和 root 长程漂移的主要来源，同时预测历史也提供了明显的时间低通作用。
