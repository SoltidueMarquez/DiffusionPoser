# realtime_pose_v1 Plan

项目主链路已经切换为 `realtime_pose_v1`：

- 61 帧窗口，前 60 帧历史条件，第 61 帧生成 `body_pose_parent_6d + root_yaw_delta_sincos`。
- 模型输入输出 `[B,206,61]`。
- hip/waist tracker 必须有效，每帧至少 3 个 tracker valid。
- `contact` 不落盘，由 `joints_world` 动态派生。
- Unity runtime、ONNX dummy input、Visual Editor、smoke tests 均围绕 `realtime_pose_v1`。
