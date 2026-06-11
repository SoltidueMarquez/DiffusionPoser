# RealtimePose Studio

`visual_editor/` 是本地 `realtime_pose_body_fbx_local_root_y0_v1` source、task 和 result 查看/导出工具。

## 启动 API

```powershell
conda run -n diffusionposer5070 python -m visual_editor.server --source_dir dataset/AMASS_realtime_pose_body_fbx_local_root_y0_60hz --data_dir dataset/AMASS_realtime_pose_body_fbx_local_root_y0_60hz_tasks --result_dir output --output_dir visual_editor/.runtime/exports
```

## 数据

- Source：`realtime_pose_body_fbx_local_root_y0_v1` `.npz`，包含 `body_pose_body_fbx_local_delta_6d/pose_representation/root_pos_world/root_yaw/root_heading_delta_sincos/root_delta_xz_ref/pelvis_height/foot_contact/tracker_pos_world/tracker_rot_world_6d/joints_world/joint_offsets_parent/joint_rest_local_rotations_6d`。
- Task：`materialized_realtime_pose_body_fbx_local_root_y0_v1` `.npz`，包含 source 数组、`sensor_valid`、`inpaint_mask` 和 61 帧窗口元数据。
- Result：采样结果 `.npz`，建议包含 `reference_features`、`conditioned_features`、`reconstructed_features` 或对应 `*_raw` 字段。

默认导出的 task 固定为 61 帧 full-tracker 窗口。训练阶段的随机遮盖由 Dataset 动态完成，不需要从编辑器导出多份训练 mask。
