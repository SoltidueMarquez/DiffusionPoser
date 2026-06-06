# realtime_pose_v2_contact Plan

项目主链路已经切换为 `realtime_pose_v2_contact`：

- 61 帧窗口，前 60 帧历史条件，第 61 帧生成 `body_pose_root_global_6d + root_yaw_delta_sincos + root_delta_xz_ref + root_height + foot_contact`。
- 模型输入输出 `[B,211,61]`。
- hip/waist tracker 必须有效，每帧至少 3 个 tracker valid。
- `foot_contact` 在转换阶段落盘，由 `joints_world` 派生。
- Unity runtime、ONNX dummy input、Visual Editor、smoke tests 均围绕 `realtime_pose_v2_contact`。
