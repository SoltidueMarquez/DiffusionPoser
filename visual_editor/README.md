# RealtimePose Studio

`visual_editor/` 是本地 realtime pose source、task 和 result 查看/导出工具。默认 schema 跟随 registry 的 `realtime_pose_stationary5_v1`，legacy exact schema `realtime_pose_body_fbx_local_root_y0_v1` 仍可读取。

## 启动 API

```powershell
conda run --no-capture-output -n diffusionposer5070 python -m visual_editor.server --result_dir output --output_dir visual_editor/.runtime/exports
```

未显式传 `--source_dir/--data_dir` 时，API 会根据 `configs/data_roots.local.json` 或 `configs/data_roots.example.json` 的 `generated_root` 解析默认 source/task 目录。

## 数据

- Source：默认读取 registry default schema `realtime_pose_stationary5_v1` `.npz`，同时保留 legacy exact schema `realtime_pose_body_fbx_local_root_y0_v1` 读取能力；字段包含 `body_pose_body_fbx_local_delta_6d/pose_representation/root_pos_world/root_yaw/root_heading_delta_sincos/root_delta_xz_ref/pelvis_height/stationary_prob_5/tracker_pos_world/tracker_rot_world_6d/joints_world/joint_offsets_parent/joint_rest_local_rotations_6d`。
- Task：默认读取 `materialized_realtime_pose_stationary5_v1` `.npz`，legacy exact `materialized_realtime_pose_body_fbx_local_root_y0_v1` 仍可读取；包含 source 数组、`sensor_valid`、`inpaint_mask` 和 61 帧窗口元数据。
- Result：采样结果 `.npz`，建议包含 `reference_features`、`conditioned_features`、`reconstructed_features` 或对应 `*_raw` 字段。

默认导出的 task 固定为 61 帧 full-tracker 窗口。训练阶段的随机遮盖由 Dataset 动态完成，不需要从编辑器导出多份训练 mask。
