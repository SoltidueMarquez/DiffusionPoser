# Kimodo Unity Motion Browser Design

## 背景

目标是在 Unity Editor 中增加一个独立的 Kimodo 动作资产浏览和预览工具，用于查看、筛选和推进 Kimodo 生成的数据进入 DiffusionPoser 数据链路。工具形态类似 Asset Browser：预览发生在 EditorWindow 内部，不依赖当前打开的 scene，不进入 Play Mode，也不把播放对象写回场景。

当前上下文：

- Kimodo 产物目录是 `D:\Projects\SchoolWorkProjects\kimodo\artifacts\kimodo_raw`。
- 已生成样例 `walk_turn_wave_01.npz/.bvh`，NPZ 包含 `posed_joints [T,77,3]`、`local_rot_mats [T,77,3,3]`、`global_rot_mats [T,77,3,3]`、`root_positions [T,3]`、`foot_contacts [T,6]` 和 `global_root_heading [T,2]`。
- Unity 工程是 `D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity`。
- Unity 里已有 RealtimePose JSON replay、测试场景、`AI4Animation.Actor`、`ActorPoseApplier` 和 DiffusionPoser runtime 逻辑，但这些面向 `realtime_pose_stationary5_v1`，不应直接拿来伪装 Kimodo raw 预览。

## 目标

1. 在 Unity EditorWindow 内浏览 Kimodo 动作，不要求打开特定 scene。
2. 选中动作后直接在窗口中预览。
3. 支持把动作应用到用户指定的 `AI4Animation.Actor` 模板；Actor 模板会被复制到隐藏预览上下文，不修改原对象或当前 scene。
4. 没有 Actor 模板或映射不足时，自动绘制 Kimodo 77 关节点调试骨架。
5. 每条动作导入成独立 Unity asset，不维护全局 library SO。
6. 在同一窗口提供第二层数据链路入口：生成 preview package、转换 pseudo-AMASS、生成 DiffusionPoser replay/source/task/normalizer。
7. 保持 Kimodo preview 数据格式和 RealtimePose schema 严格分离；只有第二层转换后的产物才使用 `realtime_pose_stationary5_v1`。

## 非目标

- 不把 raw `.npz/.bvh` 复制进 Unity 工程。
- 不在第一版实现完整 Humanoid retarget；第一版只检测并保留入口。
- 不把每帧动作大数组直接写入 ScriptableObject 字段。
- 不让预览窗口修改当前 scene 的 Actor、材质、相机或播放状态。
- 不新增未注册 schema，也不放宽 DiffusionPoser exact `schema_name` 约束。

## 最终方案

采用“每条 motion 一个 SO + 若干 TextAsset 引用”的 Unity 内 motion package。

```text
Assets/Projects/RealtimePose/KimodoBrowser/Motions/
  walk_turn_wave_01/
    walk_turn_wave_01.asset
    manifest.json
    skeleton.json
    joint_positions_world.bytes
    local_rotations_xyzw.bytes
    global_rotations_xyzw.bytes
    root_positions.bytes
    foot_contacts.bytes
```

不创建 `KimodoMotionLibrary.asset`。Browser 打开或手动刷新时通过：

```text
AssetDatabase.FindAssets("t:KimodoMotionAsset", searchFolders)
```

扫描 `KimodoMotionAsset`。

## Unity Asset 职责

`KimodoMotionAsset` 是单条动作的索引、状态和 Unity 引用容器。它保存：

- `displayName`
- `sourceNpzPath`
- `sourceBvhPath`
- `sourceSha256`
- `fps`
- `frameCount`
- `jointCount`
- `manifest: TextAsset`
- `skeleton: TextAsset`
- `jointPositionsWorld: TextAsset`
- `localRotationsXyzw: TextAsset`
- `globalRotationsXyzw: TextAsset`
- `rootPositions: TextAsset`
- `footContacts: TextAsset`
- `status: New / Approved / Rejected / Converted`
- `tags`
- `notes`
- `defaultBindingProfile`
- `convertedReplayJson`
- `convertedSourcePath`

SO 只管理元信息、状态、引用和 Unity 专属配置。动作帧数据保存在 `.bytes` 中，通过 `TextAsset.bytes` 读取。

## Motion Package 格式

`manifest.json` 只保存契约、来源、shape、hash 和 stream 描述：

```json
{
  "format": "kimodo_motion_preview",
  "version": 1,
  "sourceKind": "kimodo_raw_npz",
  "sourceNpzPath": "D:/Projects/SchoolWorkProjects/kimodo/artifacts/kimodo_raw/walk_turn_wave_01.npz",
  "sourceBvhPath": "D:/Projects/SchoolWorkProjects/kimodo/artifacts/kimodo_raw/walk_turn_wave_01.bvh",
  "sourceSha256": "...",
  "fps": 30,
  "frameCount": 180,
  "jointCount": 77,
  "unit": "meter",
  "coordinateSystem": "unity_world",
  "rootJointIndex": 0,
  "streams": {
    "jointPositionsWorld": {
      "path": "joint_positions_world.bytes",
      "dtype": "float32",
      "shape": [180, 77, 3],
      "sha256": "..."
    },
    "localRotations": {
      "path": "local_rotations_xyzw.bytes",
      "dtype": "float32",
      "shape": [180, 77, 4],
      "quaternionOrder": "xyzw",
      "sha256": "..."
    }
  }
}
```

`skeleton.json` 保存 77 关节骨架定义：

- `jointNames`
- `parentIndices`
- `aliases`
- `sourceBvhJointNames`
- `smpl24Mapping`
- `humanoidMappingCandidates`

`.bytes` 文件使用 little-endian：

- `joint_positions_world.bytes`: float32 `[T, 77, 3]`
- `local_rotations_xyzw.bytes`: float32 `[T, 77, 4]`
- `global_rotations_xyzw.bytes`: float32 `[T, 77, 4]`
- `root_positions.bytes`: float32 `[T, 3]`
- `foot_contacts.bytes`: uint8 `[T, 6]`

Python 生成包时负责把 Kimodo rotation matrix 转成 Unity 约定的 quaternion，并在 manifest 中记录坐标系和单位。Unity 加载时先校验 shape 与文件大小，必要时再校验 sha256。

## EditorWindow 设计

窗口名：

```text
RealtimePose/Kimodo Motion Browser
```

布局：

- 左侧 `Library`：扫描 `KimodoMotionAsset`，显示文件名、帧数、时长、状态、标签、是否有 BVH、是否已转换。
- 中间 `Preview`：独立 3D 预览区，不进入 Play Mode。
- 底部 `Timeline`：播放、暂停、前后单帧、帧滑条、速度、循环、root 轨迹、脚接触显示开关。
- 右侧 `Target / Pipeline`：Actor 模板、绑定 profile、映射状态、导入/转换按钮、日志输出。

预览不依赖当前 scene。实现上使用隐藏 preview scene 或 `PreviewRenderUtility` 加内部相机；优先选隐藏 preview scene，因为要同时管理 Actor 副本、debug skeleton、轨迹和脚接触标记。

## Actor 预览行为

预览优先级：

1. 如果指定了带 `AI4Animation.Actor` 的 prefab、模型或场景对象，窗口复制一份到隐藏预览上下文。
2. 第一版优先支持 SMPL/body.fbx Actor，骨名可匹配 `Pelvis/L_Hip/...`、`m_avg_Pelvis/...` 和本仓库 `DefaultPoseSkeletons` 的 canonical alias。
3. 能映射的骨骼照常驱动，缺失骨骼在右侧映射面板标红。
4. 如果 Actor 缺失或有效映射过少，显示 77 关节点调试骨架。
5. 如果目标是 Humanoid `Animator.avatar.isHuman`，第一版只识别并提示需要后续 Humanoid retarget 模块，不把它混入 SMPL/body.fbx 逻辑。

Actor 驱动分两步：

- 用 `skeleton.json` 中的 `smpl24Mapping` 从 Kimodo 77 关节映射到 RealtimePose/SMPL 24 主骨骼。
- 对 SMPL/body.fbx Actor 使用 local rotation 优先；若 rotation 坐标系或 rest pose 不满足，则 fallback 到 joint position 方向约束驱动调试骨架，而不是强行扭曲 Actor。

手指骨第一版只在 77 关节点调试骨架中显示，不要求驱动目标 Actor 手指。

## 导入流程

用户选择 raw `.npz` 后点击 `Import Preview`：

```text
Unity EditorWindow
  -> conda run --no-capture-output --prefix D:\Anaconda\envs\kimodo_gui python -m kimodo_gui.export_unity_preview
  -> 生成临时 package
  -> 复制 manifest/skeleton/bytes 到 Assets/Projects/RealtimePose/KimodoBrowser/Motions/<motion_name>/
  -> AssetDatabase.ImportAsset
  -> 创建 KimodoMotionAsset.asset 并引用 TextAsset
  -> 自动选中新 asset
```

导入不会复制 raw `.npz/.bvh`，只保存 `sourceNpzPath/sourceBvhPath/sourceSha256` 用于追溯。

## 第二层 DiffusionPoser Pipeline

窗口右侧对选中 `KimodoMotionAsset` 提供操作：

- `Convert to pseudo-AMASS`
- `Build RealtimePose Source`
- `Build RealtimePose Tasks`
- `Build Normalizer`
- `Export RealtimePose Replay`

所有命令由 Unity 调用 Python/conda：

```powershell
conda run --no-capture-output --prefix D:\Anaconda\envs\kimodo_gui python -m ...
conda run --no-capture-output -n diffusionposer5070 python -m ...
```

执行日志流式显示在窗口里，并写入 `D:\Projects\SchoolWorkProjects\kimodo\runs`。转换成功后，`KimodoMotionAsset` 更新：

- `status = Converted`
- `convertedReplayJson`
- `convertedSourcePath`

RealtimePose replay/source/task/normalizer 必须使用 exact `schema_name`，默认 `realtime_pose_stationary5_v1`。

## 错误处理

导入阶段：

- source hash 不匹配：提示重新生成 package。
- stream 文件大小与 shape 不一致：拒绝加载。
- manifest version 不支持：提示升级 importer。
- `.bytes` 缺失：标记 motion asset broken。

预览阶段：

- Actor 模板丢失：fallback 到 debug skeleton。
- 映射不足：显示缺失骨骼列表，继续驱动可映射骨骼或 fallback。
- rotation 数据异常：当前帧标红，保留点位骨架预览。
- 预览上下文销毁失败：窗口关闭时强制清理隐藏对象。

Pipeline 阶段：

- Python 命令失败：保留日志，不更新 `Converted` 状态。
- DiffusionPoser schema mismatch：直接失败，不自动按 canonical name 放宽。
- 输出目录已有产物：默认生成新 run id，不覆盖。

## 测试策略

Unity 编辑器测试：

- `KimodoMotionAsset` 创建后能引用所有 TextAsset。
- manifest/stream shape 校验失败时给出明确错误。
- Browser 能通过 `AssetDatabase.FindAssets` 找到 motion asset。
- 没有 Actor 时 debug skeleton 预览路径可创建。
- SMPL/body.fbx Actor 映射至少覆盖 24 主骨骼中的核心躯干、四肢和头部。

Python 测试：

- 从 mock Kimodo NPZ 生成 `manifest.json + skeleton.json + .bytes`。
- stream 文件大小、dtype、shape 与 manifest 一致。
- rotation matrix 到 quaternion 的输出无 NaN/Inf。
- source sha256 写入并可校验。

手动验收：

- 导入 `walk_turn_wave_01.npz`。
- 在窗口内无 scene 依赖播放 180 帧动作。
- 没有 Actor 时可看到 77 关节点骨架。
- 指定 SMPL/body.fbx Actor 模板后可在窗口里预览模型动作。
- 标记 Approved 后可执行第二层转换命令并记录日志。

## 后续扩展

- Humanoid retarget：使用 `Animator`/`HumanPoseHandler` 或专门绑定 profile。
- BVH 对照预览：导入 BVH 层级并与 NPZ 点位同屏比较。
- 质量标注导出：Approved/Rejected/tags/notes 写回 JSONL，用于训练集筛选。
- 批量导入：从 raw 目录一次生成多个 `KimodoMotionAsset`。
- 差异视图：Kimodo raw、DiffusionPoser replay reference、model inference 三路对比。

## 审批状态

已确认的产品决策：

- 做两层工具：raw Kimodo 预览 + DiffusionPoser pipeline。
- Unity 允许调用本机 Python/conda 命令。
- 预览窗口与当前 scene 脱钩，行为类似 asset browser。
- Actor 是预览模板，会复制到窗口内部上下文，不直接驱动 scene 对象。
- 没有 Actor 时绘制 77 关节点骨架。
- 第一版优先 SMPL/body.fbx Actor，Humanoid 作为后续扩展。
- 动作数据采用每条 motion 独立 SO：`KimodoMotionAsset.asset + TextAsset manifest/skeleton + .bytes streams`。
- 不创建全局 `KimodoMotionLibrary.asset`，Browser 通过 AssetDatabase 扫描 motion asset。
