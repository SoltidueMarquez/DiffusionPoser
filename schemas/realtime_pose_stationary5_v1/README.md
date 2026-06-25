realtime_pose_stationary5_v1：固定 body_fbx_local + root_y0 前提下，使用 61 帧窗口、214 维特征和 stationary_prob_5 的实时姿态重建契约。

该 schema 是 `realtime_pose_body_fbx_local_root_y0_v1` 的 canonical 名称。旧名仍作为 alias 注册，便于已有 task、checkpoint 和 Unity 资产逐步迁移。

核心约束：
- 输入输出特征均为 `[B, 214, 61]`。
- 第 61 帧的 `0:154` 为补全目标，历史 60 帧和 tracker 观测通道不进入 inpaint mask。
- actor root 的 world y 固定为 0，pelvis 高度使用 pelvis bone 的 local offset y。
- `stationary_prob_5` 位于 `149:154`，表示 pelvis、左右脚、左右手的静止概率。
