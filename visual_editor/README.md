# RealtimePose Studio

`visual_editor/` 是本地 realtime_pose_v1 查看和 task 导出工具。

## 启动 API

```powershell
conda run -n diffusionposer5070 python -m visual_editor.server --source_dir dataset/AMASS_realtime_pose_60hz --data_dir dataset/AMASS_realtime_pose_60hz_tasks --result_dir output --output_dir visual_editor/.runtime/exports
```

## 数据

- Source：`realtime_pose_v1` `.npz`，包含 `body_pose_parent_6d/root_pos_world/root_yaw/tracker_pos_world/joints_world` 等源数组。
- Task：`materialized_realtime_pose_v1` `.npz`，包含 source 数组、`sensor_valid`、`inpaint_mask` 和 61 帧窗口元数据。
- Result：采样结果 `.npz`，建议包含 `reference_features`、`conditioned_features`、`reconstructed_features`。

默认导出的 task 固定为 61 帧 full-tracker 窗口，`sensor_valid` 全 1。需要做评估或可视化对比时，可以在 exporter 中选择固定 tracker pattern；训练阶段的随机遮盖由 Dataset 动态完成，不需要从编辑器导出多份训练 mask。
