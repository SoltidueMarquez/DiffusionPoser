# Root-Relative Global Rotation 实验记录

## 基线

- baseline tag: `baseline/root-relative-global-rotation`
- baseline commit: `4a7d5f2 Add tracker corrections to Unity stream simulation`
- experiment branch: `codex/root-relative-global-rotation`
- 创建日期: `2026-06-02`

## 实验目的

- 将主链路姿态字段改为 `body_pose_root_global_6d`，评估从 SMPL24 parent-local rotation 切到 root-yaw-relative global rotation 的收益和代价。
- 目标是让 tracker/global rotation 对齐更直接，同时保留 `root_yaw_delta_sincos` 对水平朝向的显式建模。
- 该实验不直接追求兼容已废弃的数据、normalizer、checkpoint 或 Unity runtime 资产。

## 产物约定

- 不提交 `dataset/`、`runs/`、`output/`、`save/` 或 checkpoint/data 二进制产物。
- 训练、采样和评估命令默认使用 `conda run -n diffusionposer5070 <command>`。
- 实验产物必须使用独立目录，避免覆盖当前 stationary5 主链路。

## 实验条目

### 001 - root-relative global rotation

- 改动摘要: 待实现。计划把 24 个关节旋转编码为 `inverse(Yaw(root_yaw)) @ global_rotation[j]`，再在 FK/loss/runtime 中恢复 source global rotation。
- 关键文件: `data_loaders/realtime_pose_kinematics.py`、`data_converter/amass_to_realtime_pose.py`、`data_loaders/sensor_masking.py`、`diffusion/gaussian_diffusion.py`、`export/write_unity_runtime_assets.py`
- 训练配置: 待定。
- 测试命令: `conda run -n diffusionposer5070 pytest tests/smoke/data_pipeline tests/smoke/train tests/smoke/export`
- Run 目录: `runs/root-relative-global-rotation`
- 结果指标: 待定。
- 结论: 待定。
