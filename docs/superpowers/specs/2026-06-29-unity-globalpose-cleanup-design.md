# Unity GlobalPose 物理链路深度清理设计

## 背景

Unity 端 realtime pose 物理链路已经从早期的 GlobalPose 专用 driver 过渡到 `RealtimePosePhysicsMethodAsset` / `IRealtimePosePhysicsMethodRuntime` 的方法资产架构，但旧的 GlobalPose 兼容路径仍留在通用层里。当前最大的结构问题是：

- `SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/RealtimePoseGlobalPoseMethodAsset.cs` 同时包含 asset、runtime、frame input 构建、stationary 信号注入、contact probe、diagnostics、self-test、tau/GRF 输出等职责。
- `RealtimePosePhysicsDriver` 仍在 `methodAsset == null` 时自动创建 GlobalPose runtime，并保留 `StartGlobalPosePhysics` / `StopGlobalPosePhysics` 旧 API。
- `PhysicsStepSnapshot` 和 `PhysicsDebugRouter` 的通用字段中包含 `ConeQpFitError`、`ResidualNorm`、`FirstLsqrResidualForce` 等 GlobalPose 专属指标。

本次清理的目标是切断通用 physics driver 对 GlobalPose 的硬编码依赖，同时保留 GlobalPose 作为一个明确的 physics method 实现。

## 目标

- `RealtimePosePhysicsDriver` 只负责物理 session 生命周期、native dynamics state、method runtime 调用、debug snapshot 管理。
- GlobalPose 相关逻辑集中在 `GlobalPoseFlow` 目录，作为一个插件式 method 实现存在。
- 删除 GlobalPose legacy fallback 和 obsolete `StartGlobalPosePhysics` / `StopGlobalPosePhysics` API。
- 将 GlobalPose runtime 大文件拆成职责明确的小文件，降低后续改 contact、diagnostics、self-test 时的误伤范围。
- 不手工编辑二进制 Unity scene/asset 文件；序列化迁移通过 Unity Editor 保存或专用 editor validation/migration 工具完成。

## 非目标

- 不改 GlobalPose solver 的数学行为、contact 判断、LSQR/QP 流程、stationary_prob_5 语义。
- 不重命名算法概念中的 `GlobalPose`。它仍是论文/方法名，应保留在 method asset、runtime、settings、solver 类名里。
- 不同时清理 DiffusionPoser 推理、replay input、actor pose decoder 等非 physics method 主线。
- 不手工 patch `.unity`、`.asset` 这类当前以 Unity binary serialization 保存的文件。

## 推荐方案

采用两层清理：

1. 通用 physics 层去 GlobalPose 化。
2. GlobalPose method 层内部拆分。

不采用“一次性重命名所有 GlobalPose 类/目录/资产”的方案，因为这会制造大量 Unity GUID、序列化引用和场景引用风险，而收益主要是命名洁癖，不解决实际边界问题。

## 通用 Physics 层设计

`RealtimePosePhysicsDriver` 保留以下职责：

- 解析 armature 路径并创建/销毁 Carticulate native dynamic handle。
- 管理 `stateQuat`、`stateRoot`、`stateVel`、`zeroVelocity` 等 dynamics state buffer。
- 初始化并调用 `RealtimePosePhysicsMethodAsset.CreateRuntime()`。
- 调用 runtime 的 `Initialize`、`StartSession`、`StopSession`、`TryStep`、`Shutdown`。
- 将 `RealtimePosePhysicsMethodDiagnostics` 合并进 `PhysicsStepSnapshot`。
- 将 integrated state 应用到 target SMPL transforms。

`RealtimePosePhysicsDriver` 删除以下内容：

- `legacyRealtimePoseGlobalPosePhysicsSettings`
- `legacySyncReferenceAngularVelocityFromJointTransforms`
- `UsesLegacyGlobalPoseFallback`
- `HasLegacyGlobalPoseSettings`
- `StartGlobalPosePhysics()`
- `StopGlobalPosePhysics()`
- `methodAsset == null` 时创建 `RealtimePoseGlobalPoseMethodRuntime` 的 fallback

新的失败策略：

- `methodAsset == null` 时初始化失败，并通过 `PublishError(RealtimePosePhysicsDebugTopic.Validation, "...")` 明确提示需要配置 physics method asset。
- `ResolveConfiguredMethodName()` 在未配置 asset 时返回类似 `"<unconfigured>"`，不再默认返回 GlobalPose method name。

`RealtimePosePhysicsDriverEditor` 删除 legacy fallback helpbox，改成：

- 显示 assigned method asset 的 method name。
- 未配置 method asset 时显示 error/warning helpbox。
- Runtime tools 仍使用 `StartPhysicsSession`、`StopPhysicsSession`、`RunNetPyTryStepSelfTest`。

## GlobalPose Method 层设计

将 `RealtimePoseGlobalPoseMethodAsset.cs` 拆成以下文件，均保留在 `Physics/GlobalPoseFlow/`：

- `RealtimePoseGlobalPoseMethodAsset.cs`
  - 只保留 `ScriptableObject` asset、serialized settings 引用、`syncReferenceAngularVelocityFromJointTransforms` 配置和 `CreateRuntime()`。

- `RealtimePoseGlobalPoseMethodRuntime.cs`
  - 持有 settings、solver、frame buffers、diagnostics state。
  - 实现 `IRealtimePosePhysicsMethodRuntime`、`IRealtimePosePhysicsMethodSnapshotProvider`、`IRealtimePosePhysicsSelfTestRuntime`。
  - 保留 lifecycle 和 `TryStep()` 主流程，但把 frame 构建、diagnostics、自测细节委托出去。

- `RealtimePoseGlobalPoseFrameBuilder.cs`
  - 从 reference root、SMPL joints、Actor 构造 `CarticulateGpNetPySolver.FrameInput`。
  - 注入 `stationaryProb5`。
  - 执行 Unity physics probe。
  - 生成 `GpPredJointWorld15`。
  - 输出必要的 mapping trace。

- `RealtimePoseGlobalPoseDiagnosticsBuilder.cs`
  - 从 solver state 构造 `RealtimePoseGlobalPoseMethodDiagnostics`。
  - 构造 `PhysicsContactObservation[]`。
  - 生成 tau csv、GRF json、virtual force override、summary。
  - 保持 gizmo/debug 需要的数据结构不变。

- `RealtimePoseGlobalPoseSelfTest.cs`
  - 包装 `CarticulateGpNetPySelfTestRunner.TryRunSingleStep`。
  - 统一 self-test message 和 diagnostics 构造。

- `RealtimePoseGlobalPoseReferenceVelocitySync.cs`
  - 从 reference joint transforms 同步 Actor bone angular velocity。
  - 隔离 `FindReferenceBoneForTransform` 和 previous rotation cache 逻辑。

拆分原则：

- 不改变 public serialized field 名称，避免破坏 `RealtimePoseGlobalPoseMethod.asset`。
- 新 helper 优先 `internal sealed` 或 `internal static`，不扩大公开 API。
- 共享 buffer 由 runtime 创建和持有，helper 通过明确参数使用，避免 helper 隐式分配大数组。
- 中文注释只解释边界和坐标/语义原因，不复述 C# 语法。

## Diagnostics 边界

保留通用 diagnostics 的跨方法字段：

- `MethodName`
- `Summary`
- `ActiveContactMask`
- `ActiveContactCount`
- `Contacts`
- `TauCsv`
- `GrfJson`
- `HasVirtualForceOverride`
- `VirtualForceOverride`

GlobalPose 专属字段只存在于 `RealtimePoseGlobalPoseMethodDiagnostics`：

- `ConeQpFitError`
- `ResidualNorm`
- `ConeQpVarCount`
- `FirstLsqrResidualForce`
- `HasFirstLsqrResidualForce`

`PhysicsStepSnapshot` 应优先依赖 `MethodDiagnostics` 保存 method-specific diagnostics。若现有 `PhysicsDebugRouter.FormatStepSummary` 还需要显示 QP/residual，改为：

- 先使用 `snapshot.MethodSummary`。
- 只有在 `snapshot.MethodDiagnostics is RealtimePoseGlobalPoseMethodDiagnostics` 时才读取 GlobalPose 专属指标。

这样通用 snapshot 不再需要直接暴露 GlobalPose 专属字段。

## 序列化迁移策略

当前 Unity 项目的 scene/asset 文件表现为 Unity binary serialization，不适合手工修改。清理时按以下顺序处理：

1. 先确认所有使用 `RealtimePosePhysicsDriver` 的场景或 prefab 都已显式配置 `methodAsset`。
2. 如发现空 `methodAsset`，通过 Unity Editor 或 editor migration 工具设置为 `Assets/Projects/RealtimePose/Physics/GlobalPose/RealtimePoseGlobalPoseMethod.asset`。
3. 删除 legacy serialized fields 后，在 Unity Editor 中打开并保存相关 scene/asset，让 Unity 自动丢弃 orphan serialized data。
4. 不用 `FormerlySerializedAs` 保留 `globalPosePhysicsSettings` 等旧字段，因为深度清理目标是移除隐式兼容路径。

如果需要自动检查，可新增 editor-only validation 菜单：

- 扫描场景/prefab 中的 `RealtimePosePhysicsDriver`。
- 报告 `methodAsset == null` 的对象路径。
- 可选地一键填入默认 GlobalPose method asset。

该工具只作为迁移辅助，不进入 runtime 依赖。

## 错误处理

- driver 未配置 method asset：初始化失败，发布 validation error。
- method runtime 初始化失败：driver shutdown native handle，避免半初始化状态。
- GlobalPose helper 遇到缺失 reference joint/probe config：保持当前 fallback 行为，返回空 contact/probe 数据，不改变 solver 数学路径。
- self-test 只在 runtime 初始化且 method 支持 `IRealtimePosePhysicsSelfTestRuntime` 时可用。

## 验证计划

最小验证：

- Unity 编译通过。
- `RealtimePose_DiffusionPoser_Test` 场景中 `RealtimePosePhysicsDriver.methodAsset` 非空。
- Inspector 不再显示 legacy GlobalPose fallback。
- Play Mode 下可 initialize driver、start physics session、stop physics session。
- `NetPy Physics Self Test` 能跑通并输出 GlobalPose diagnostics summary。

运行链路验证：

- replay/inference 完成后，`RealtimePosePhysicsBridge` 能继续提供 `stationaryProb5`。
- GlobalPose solver 能继续接收 stationary/contact/probe 输入并输出 `qddot`。
- gizmo 仍能显示 probe point、reference contact point、solver contact point、solver force、motion vector。
- tau csv / GRF json / virtual force override 在 debug visualization 需要时仍可用。

回归关注点：

- `methodAsset == null` 的旧场景会从“默认 GlobalPose”变成“初始化失败”。这是预期行为，但迁移前必须先扫场景/prefab。
- 删除 obsolete `StartGlobalPosePhysics` / `StopGlobalPosePhysics` 可能影响外部脚本；应先全项目 `rg` 确认无调用。
- 移除 snapshot 上的 GlobalPose 专属字段后，所有显示 QP/residual 的代码必须改为读取 `RealtimePoseGlobalPoseMethodDiagnostics`。

## 实施顺序

1. 添加或运行 editor validation，确认所有 driver 都配置 method asset。
2. 清理 `RealtimePosePhysicsDriver` 的 legacy GlobalPose fallback 和 obsolete API。
3. 调整 custom inspector 的 method 配置提示。
4. 收窄 `PhysicsStepSnapshot`，把 GlobalPose 专属指标读取迁移到 method diagnostics。
5. 拆分 `RealtimePoseGlobalPoseMethodAsset.cs`，保持行为不变。
6. 在 Unity Editor 中保存受影响 scene/asset，清理 orphan serialized fields。
7. 执行编译、self-test、replay/inference 物理链路验证。

## 成功标准

- 通用 physics driver 不再直接引用 `RealtimePoseGlobalPoseMethodRuntime` 或 `RealtimePoseGlobalPosePhysicsSettings`。
- GlobalPose method 仍可通过 `RealtimePoseGlobalPoseMethod.asset` 正常运行。
- `methodAsset` 成为必填配置，旧 fallback 路径完全移除。
- GlobalPose runtime 文件职责清晰，单文件不再混合 asset、frame builder、diagnostics 和 self-test。
- 现有 realtime pose + stationary_prob_5 + physics debug 可视化行为保持一致。
