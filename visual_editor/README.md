# RealtimePose Studio

`visual_editor/` 是本地 realtime pose source、task 和 result 的查看与导出工具。数据字段和维度统一以仓库根目录的 [contract.md](../contract.md) 为准。

## 启动 API

```powershell
conda run -n diffusionposer5070 python -m visual_editor.server --source_dir dataset/AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz --data_dir dataset/AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz_tasks --result_dir output --output_dir visual_editor/.runtime/exports
```
