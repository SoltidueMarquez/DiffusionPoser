# realtime_pose_stationary5_v1 Plan

项目默认 schema 已经切换为 `realtime_pose_stationary5_v1`：

- 61 帧窗口，前 60 帧历史条件，第 61 帧生成 `body_pose_body_fbx_local_delta_6d + root_heading_delta_sincos + root_delta_xz_ref + pelvis_height + stationary_prob_5`。
- 模型输入输出 `[B, 214, 61]`。
- actor root world y 固定为 0，`pelvis_height` 表示 pelvis local offset y。
- hip/waist tracker 必须有效，每帧至少 3 个 tracker valid。
- `stationary_prob_5` 在转换阶段落盘，由 `joints_world` 的 5 个候选接触关节速度派生。
- Unity runtime、ONNX dummy input 与 smoke tests 均围绕 `realtime_pose_stationary5_v1`。
- `realtime_pose_body_fbx_local_root_y0_v1` 是 legacy alias，仍可读取、训练和导出；但 source/task/normalizer/checkpoint/runtime asset 的 exact metadata 不能与默认 schema 混用。
- 旧 `realtime_pose_v2_contact` / `root_yaw_global_6d` 数据、normalizer、checkpoint、Unity runtime asset 已废弃且不再保留兼容入口。
