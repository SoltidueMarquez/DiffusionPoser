realtime_pose_stationary5_v1：固定 body_fbx_local + root_y0 前提下，使用 61 帧窗口、214 维特征和 stationary_prob_5 的实时姿态重建契约。

该 schema 是 `realtime_pose_body_fbx_local_root_y0_v1` 的 canonical 名称。旧名仍作为 alias 注册，便于已有 task、checkpoint 和 Unity 资产逐步迁移。

核心约束：
- 条件输入为 `[B, 214, 61]`，扩散目标是第 61 帧的 154 维主特征。
- 第 61 帧的 `0:154` 为补全目标，历史 60 帧和 tracker 观测通道不进入 inpaint mask。
- schema 内的 actor root y 固定为 0；运行时最终 Root world y 等于 `floor_y`，人体高度由 pelvis height 表示。
- `stationary_prob_5` 位于 `149:154`，表示 pelvis、左右脚、左右手的静止概率。
- `feature_contract_version=2`；Head 始终有效且每帧至少三个 Tracker，Hip 可缺失。
- Tracker 编解码使用 `tracker_codec_v2`，Root 后处理使用模型外的 `runtime_root_resolver_v1`。
- 额外 `StationaryHead` 已删除；ONNX/Sentis 只有一个 motion 输出，主通道 149:154 继续训练和评测。
