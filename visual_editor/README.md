# RealtimePose Studio

`visual_editor/` 是本地 realtime pose source、task 和 result 的查看与导出工具。默认 schema 为
`realtime_pose_stationary5_v1`；已注册的 legacy exact schema
`realtime_pose_body_fbx_local_root_y0_v1` 仍可读取。

## 启动 API

```powershell
conda run --no-capture-output -n diffusionposer5070 python -m visual_editor.server
```

未显式指定 source 或 task 路径时，API 从
`configs/artifact_roots.local.json` 或 `configs/artifact_roots.example.json` 解析
AMASS、SMPL、source、task、结果和编辑器运行目录。环境变量
`REALTIME_POSE_EDITOR_*` 仍可为单次启动覆盖对应路径。

## 数据契约

- Source 和 task 必须携带所选 exact `schema_name`，以及
  `pose_representation="body_fbx_local_delta_6d"`、`root_y_policy="fixed_zero"` 和
  `pelvis_height_mode="pelvis_local_offset_y"`。
- 编辑器导出的 task 固定为 61 帧窗口，第 61 帧补全；可选 tracker pattern 为
  `full_six`、`standard_three`、`static_sparse` 和 `dynamic_dropout`。
- Result 为采样输出 `.npz`，建议包含 `reference_features`、`conditioned_features`、
  `reconstructed_features` 或对应的 `*_raw` 字段。
