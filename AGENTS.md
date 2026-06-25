# AGENTS.md

## 沟通语言

- 默认使用中文回答和解释；只有在用户明确要求其他语言，或必须保留英文术语、变量名、API 名称时才使用英文。

本文件给 Codex 和其他代码代理提供仓库级协作约定。修改代码前先阅读本文件，再结合就近目录里的说明和现有实现风格执行。

## 基本环境

- 默认工作目录是仓库根目录。
- Python、pytest、训练、采样、评估和导出命令优先使用 Anaconda 环境；给用户生成或实际执行 `conda run` 命令时，默认带 `--no-capture-output`，方便面板实时显示日志：

```powershell
conda run --no-capture-output -n diffusionposer5070 <command>
```

## 项目结构

- `data_converter/`：AMASS/SMPL-H 到 registry 中已注册 realtime pose schema 源数据的转换脚本。
- `data_loaders/`：已注册 schema 的 task 生成、Dataset、normalizer、tracker pattern 规则。
- `model/`：DiffusionPoser/DiT 模型结构。
- `diffusion/`：Gaussian diffusion、schedule、loss、respace 等扩散核心逻辑。
- `train/`：训练入口、训练循环、日志平台和 checkpoint 逻辑。
- `sample/`：实时 61 帧窗口重建、采样辅助、可视化辅助。
- `eval/`：realtime pose 评估入口和指标计算。
- `export/`：Unity/Sentis 运行时资产导出。
- `tests/smoke/`：主链路冒烟测试。
- `dataset/`、`runs/`、`save/`、`output/`：数据和训练/导出产物，默认不要改动或提交。

## 代码风格

- 保持研究代码直白可读，优先复用本仓库已有函数、命名和目录边界。
- Python 文件使用清晰的模块级函数，避免把复杂逻辑堆在 `if __name__ == "__main__":` 下。
- 公共函数、类、CLI 参数命名要表达研究语义，例如 `create_model_and_diffusion`、`RealtimePoseTaskDataset`、`build_realtime_inpaint_mask`。
- 张量维度超过两个语义轴时，在函数边界或关键变换附近写清形状，例如 `[B, C, T]`、`[T, 211]`、`[B, T, D]`。
- 不要混用 `sensor_valid`、`valid_frame_mask`、`inpaint_mask` 的语义。
- 新增或修改注释时优先使用中文解释“为什么这样做”和“变量代表什么”，不要只复述 Python 语法。
- 保持 CLI 脚本薄；可复用逻辑应进入可导入函数，便于测试。

## 测试约定

冒烟测试统一放在 `tests/smoke/`：

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke
```

按领域单独运行：

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/data_pipeline
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/train
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/sample
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/eval
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/export
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/visual_editor
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/schemas
```

如果改动训练、数据生成、在线重建、评估或导出链路，至少运行相关领域的冒烟测试；改动跨模块契约时运行完整 `tests/smoke`。

## Schema 契约

- 当前默认 schema 是 `realtime_pose_stationary5_v1`；legacy exact name `realtime_pose_body_fbx_local_root_y0_v1` 仍在 registry 中注册，可继续训练和导出。
- 仓库允许维护 registry 中明确注册的 legacy schema alias。未注册的旧数据、task、normalizer、checkpoint 或 Unity runtime asset 不应新增兼容入口。
- `schema_name` 只表达数据产物和 Unity runtime 必须共同理解的稳定契约。实验名、模型结构、loss、训练超参、ablation 变化不进入 `schema_name`，应放在 run/config/experiment name 中。
- 每个 trainable/exportable canonical schema 必须有 `schemas/<canonical_schema>/README.md`、adapter、contract smoke test、最小训练测试和最小导出测试。
- legacy alias 可以共享 canonical adapter 和 README，但 canonical README 与 registry 文档必须记录 alias 关系；alias 仍必须有最小训练测试和最小导出测试覆盖 exact `schema_name`。
- resume checkpoint 必须 exact `schema_name` 匹配；不能因为 `schema_canonical_name` 相同而放宽恢复条件。
- 固定为 60 帧历史条件 + 第 61 帧单帧补全。
- `seq_len = 61`，`target_start = 60`，`target_length = 1`。
- `feature_dim = 214`，模型输入输出均为 `[B, 214, 61]`。
- source/task/normalizer/runtime asset 必须包含所选 exact `schema_name` 和 `pose_representation="body_fbx_local_delta_6d"`。
- source/task/normalizer/runtime asset 必须显式包含 `root_y_policy="fixed_zero"` 和 `pelvis_height_mode="pelvis_local_offset_y"`。
- actor root 的 world y 固定为 0；`root_pos_world[:, 1]` 必须全为 0。
- `pelvis_height` 表示 pelvis bone 的 local offset y，必须等于 `joints_world[:, 0, 1]`。
- 通道 `0:144` 是 `body_pose_body_fbx_local_delta_6d`，`144:146` 是 `root_heading_delta_sincos`。
- 通道 `146:148` 是 `root_delta_xz_ref`，`148:149` 是 `pelvis_height`，`149:154` 是 `stationary_prob_5`。
- 通道 `154:172` 是 `tracker_pos_ref`，`172:208` 是 `tracker_rot_ref_6d`，`208:214` 是 `sensor_valid`。
- `inpaint_mask` 只允许覆盖第 61 帧的 `0:154`。
- hip/waist tracker 必须始终 valid；每帧至少 3 个 tracker valid。
- 默认 task generator 每个窗口只写 full-tracker task；训练随机遮盖在 `RealtimePoseTaskDataset` 中动态发生。
- invalid tracker 的 `tracker_pos_ref/tracker_rot_ref_6d` 在归一化后置零，不使用 GT 或上一帧 stale fill。
- 当前帧 tracker 只能用 `root_yaw_{t-1}` 转到参考局部系，不能使用 GT `root_yaw_t`。
- 必须保存 `stationary_prob_5`，由转换阶段从 `joints_world` 的 pelvis、左右脚、左右手速度派生。

## 变更原则

- 先读现有实现，再按当前模块边界做最小必要修改。
- 不要回滚或覆盖用户已有未提交改动。
- 只有用户明确要求“添加兼容”且 schema 已进入 registry 时才保留或新增 legacy 路径；其他需求变更应直接切换到新方案，不要隐式维护未注册旧方案。
- 新增功能需要配套轻量 smoke test，尤其是数据维度、mask 语义、checkpoint 恢复、采样/评估输出格式。
- 发现 README 或旧文档与代码不一致时，可以同步更新；不要为了整理文档扩大代码改动范围。
