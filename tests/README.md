# 测试目录

当前仓库的冒烟测试统一放在 `tests/smoke/`。这些测试使用临时目录和小规模构造数据，目标是在正式训练、采样或导出前快速验证 `realtime_pose_body_fbx_local_root_y0_v1` 主链路没有被破坏。

```text
tests/
├── smoke/
│   ├── data_pipeline/  # root-y0 source/task generator/Dataset/normalizer
│   ├── train/          # DiT forward、训练入口、单 batch loss
│   ├── sample/         # 61 帧实时重建和可视化辅助
│   ├── eval/           # realtime_pose 评估指标
│   └── export/         # Unity/Sentis runtime assets
└── README.md
```

常用命令：

```powershell
conda run -n diffusionposer5070 pytest tests/smoke
```

按领域单独执行：

```powershell
conda run -n diffusionposer5070 pytest tests/smoke/data_pipeline
conda run -n diffusionposer5070 pytest tests/smoke/train
conda run -n diffusionposer5070 pytest tests/smoke/sample
conda run -n diffusionposer5070 pytest tests/smoke/eval
conda run -n diffusionposer5070 pytest tests/smoke/export
```
