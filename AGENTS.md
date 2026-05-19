# AGENTS.md

本文件给 Codex 和其他代码代理提供仓库级协作约定。修改代码前先阅读本文件，再结合就近目录里的说明和现有实现风格执行。

## 基本环境

- 默认工作目录是仓库根目录。
- Python、pytest、训练、采样、评估和导出命令优先使用 Anaconda 环境：

```powershell
conda run -n diffusionposer5070 <command>
```

- 如果需要交互式调试，可以先执行：

```powershell
conda activate diffusionposer5070
```

## 项目结构

- `data_converter/`：AMASS/SMPL-H 到 X277 特征的转换脚本。
- `data_loaders/`：X277 数据读取、缺失任务生成、normalizer、sensor mask 规则。
- `model/`：DiffusionPoser/DiT 模型结构。
- `diffusion/`：Gaussian diffusion、schedule、loss、respace 等扩散核心逻辑。
- `train/`：训练入口、训练循环、日志平台和 checkpoint 逻辑。
- `sample/`：在线逐帧重建、采样辅助、可视化。
- `eval/`：current277 评估入口和指标计算。
- `export/`：Unity/Sentis 运行时资产导出。
- `tests/smoke/`：主链路冒烟测试，按 `data_pipeline/`、`train/`、`sample/`、`eval/`、`export/` 分组。
- `dataset/`、`runs/`、`save/`、`output/`：数据和训练/导出产物，默认不要改动或提交。

## 代码风格

- 保持研究代码直白可读，优先复用本仓库已有函数、命名和目录边界。
- Python 文件使用清晰的模块级函数，避免把复杂逻辑堆在 `if __name__ == "__main__":` 下。
- 公共函数、类、CLI 参数命名要表达研究语义，例如 `create_model_and_diffusion`、`X277MissingTaskDataset`、`build_stream_window`。
- 张量维度超过两个语义轴时，在函数边界或关键变换附近写清形状，例如 `[B, C, T]`、`[T, 277]`、`[B, T, D]`。
- 不要混用 `sensor_missing_labels`、`valid_frame_mask`、`inpaint_mask` 的语义。
- 新增或修改注释时优先使用中文解释“为什么这样做”和“变量代表什么”，不要只复述 Python 语法。
- 保持 CLI 脚本薄；可复用逻辑应进入可导入函数，便于测试。

## 测试约定

冒烟测试统一放在 `tests/smoke/`：

```powershell
conda run -n diffusionposer5070 pytest tests/smoke
```

按领域单独运行：

```powershell
conda run -n diffusionposer5070 pytest tests/smoke/data_pipeline
conda run -n diffusionposer5070 pytest tests/smoke/train
conda run -n diffusionposer5070 pytest tests/smoke/sample
conda run -n diffusionposer5070 pytest tests/smoke/eval
conda run -n diffusionposer5070 pytest tests/smoke/export
```

如果改动训练、数据生成、在线重建、评估或导出链路，至少运行相关领域的冒烟测试；改动跨模块契约时运行完整 `tests/smoke`。

## 数据和产物约束

- 不要把真实数据集、checkpoint、导出 `.npz/.pt/.mp4` 等大文件加入版本控制。
- 不要在没有明确需求时重写 `dataset/`、`runs/`、`save/`、`output/`。
- 修改生成脚本时优先使用临时目录或小规模构造数据做测试。
- 对路径参数保持项目根目录相对路径默认值，避免把本机绝对路径写入通用配置。

## Current277 任务契约

- `full_reconstruction_current` 固定为 10 帧历史条件 + 第 11 帧单帧补全。
- `seq_len` 和 `valid_length` 必须等于 11；`target_start` 必须等于 10，`target_length` 必须等于 1。
- 只在第 11 帧标记 `sensor_missing_labels` 和 `inpaint_mask`；前 10 帧只能作为条件输入。
- 不保留旧 100/150 帧 materialized task 的读取兼容，不做自动裁剪、自动改写 mask 或静默回退。旧任务数据必须用当前生成脚本重新生成。
- 训练、采样、评估和 Unity/Sentis 导出都应以 11 帧窗口为当前任务事实；checkpoint 里的旧 `seq_len` 不能覆盖这个任务契约。

## 变更原则

- 先读现有实现，再按当前模块边界做最小必要修改。
- 不要回滚或覆盖用户已有未提交改动。
- 只有用户明确要求“添加兼容”时才保留或新增兼容路径；其他需求变更应直接切换到新方案，不要同时维护旧方案。
- 新增功能需要配套轻量 smoke test，尤其是数据维度、mask 语义、checkpoint 恢复、采样/评估输出格式。
- 发现 README 或旧文档与代码不一致时，可以同步更新；不要为了整理文档扩大代码改动范围。
