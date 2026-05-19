# 测试目录

当前仓库的冒烟测试统一放在 `tests/smoke/`。这些测试使用临时目录和小规模构造数据，目标是在正式训练或导出前快速验证主链路没有被破坏。

```text
tests/
├── smoke/
│   ├── data_pipeline/  # X277 schema、缺失任务生成、Dataset、normalizer
│   ├── train/   # 训练入口目录检查、checkpoint resume 逻辑
│   ├── sample/  # 在线逐帧重建窗口、补全可视化
│   ├── eval/    # current277 评估指标
│   └── export/  # Unity/Sentis 运行时资产导出辅助逻辑
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

新增冒烟测试时优先放进对应领域子目录；如果是更细的单元测试或长耗时集成测试，再单独建立 `tests/unit/` 或 `tests/integration/`。
