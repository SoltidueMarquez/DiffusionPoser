# Runtime Root Resolver v1

`runtime_root_resolver_v1` 是 Python 和 Unity 独立实现共同遵循的行为契约，不要求共享源码。模型维度保持 214/154，Root 校正只发生在模型外。

Tracker 顺序为 Head、LeftHand、RightHand、Pelvis、LeftFoot、RightFoot。合法帧必须满足 `head_valid && valid_count >= 3`。输入必须是 `synthetic_joint_world` 或 `calibrated_joint_world`，`raw_device_world` 必须拒绝。

Tracker codec 使用列向量：

```text
p_ref   = R_yaw(ref_yaw)^T (p_world - ref_root_world)
p_world = R_yaw(ref_yaw) p_ref + ref_root_world
R_ref   = R_yaw(ref_yaw)^T R_world
R_world = R_yaw(ref_yaw) R_ref
```

Rotation 6D 依次保存 forward/Z 列和 up/Y 列，解码时正交化。Hip 有效时 reference 由当前 Hip observation 和静态 pelvis offset 在推理前计算；Hip 无效时使用上一帧最终 Root/yaw。`root_delta_xz_ref` 始终用上一帧最终 yaw 旋转到世界系后累加。

Hip 有效时 Root XZ/yaw/height 跟随 Hip target，yaw 不与模型混合；连续观测可用 0.03s 时间常数抗抖。Hip 无效时 yaw 积分模型 yaw delta，Root XZ 使用 Head-FK 绝对锚定，pelvis height 的 Head Y 修正单帧限制在 ±0.10m，修正后必须重新 FK。

Hip 重连用 timestamp 累计 0.1s：position/height 使用 smoothstep，yaw 使用最短角插值，目标每帧更新。origin revision 改变时有坐标变换则转换状态，否则重置；timestamp 倒退或间隔超过 0.25s 也重置。reset 帧 delta 为零。

下一帧历史只能来自最终 pose、Root、yaw、pelvis height 和 joints。Python/Unity golden tests 的 codec 和单帧 Resolver 容差为 `1e-4`，长序列 Root 累计容差为 `1e-3 m/rad`。
