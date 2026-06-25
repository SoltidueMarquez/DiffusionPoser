# Schema Registry

本文档约定 DiffusionPoser realtime pose schema 的命名、注册、产物路径和 legacy 维护规则。代码层面的入口是 `schemas/registry.py`，当前默认 schema 是 `realtime_pose_stationary5_v1`。

## 命名规则

- `schema_name` 使用短而稳定的契约名，例如 `realtime_pose_stationary5_v1`。
- `body_fbx_local`、`root_y0`、61 帧窗口、214 维特征等已经是当前项目固定前提，不必继续堆进新名字。
- 名称应表达数据和 Unity runtime 必须共同理解的语义差异，例如新增/删除通道、改变通道顺序、改变 `inpaint_mask` 范围、改变 tracker 观测语义、改变 `root_y_policy` 或 `pelvis_height_mode`。
- 名称不表达实验身份。模型结构、loss、训练超参、数据子集、ablation、run id、导出目标目录等变化进入 config、run name、experiment name 或 export name。
- v1/v2 只在契约不兼容时递增；同一契约下的 bugfix、性能优化和训练策略调整不改 schema。

## 何时新增 Schema

需要新增 schema 的情况：
- source/task/normalizer/runtime asset 的通道布局、维度、顺序或含义发生变化。
- Unity runtime 解码规则需要和旧 runtime 区分。
- 训练或导出入口需要拒绝旧产物，避免 silent mismatch。
- `pose_representation`、`root_y_policy`、`pelvis_height_mode` 或 target/inpaint 范围发生变化。

只改 run/config 的情况：
- 更换 `model_arch`、层数、heads、latent dim、diffusion steps 或 loss 权重。
- 更换 AMASS split、采样数量、训练 batch size、学习率、checkpoint 保留策略。
- 增加 ablation、对比实验或导出目录。
- 只修复实现 bug，但产物契约和 Unity runtime 语义不变。

## 新增 Adapter

新增 canonical schema 时在 `schemas/<canonical_schema>/` 下建立并维护这些文件：

| 文件 | 责任 |
| --- | --- |
| `contract.py` | 定义 `SCHEMA_NAME`、维度、通道起点、metadata 常量和 `build_<schema>_spec()`。 |
| `adapter.py` | 暴露 adapter，校验 source/task metadata，构建 inpaint mask，连接 Unity schema builder。 |
| `unity.py` | 生成 Unity runtime 需要的 feature schema 和 runtime rules。 |
| `README.md` | 用一行摘要和要点说明该 schema 的稳定契约、通道布局和 legacy 关系。 |

然后在 `schemas/registry.py` 注册 adapter。每个 trainable/exportable canonical schema 必须同时具备：

- schema README。
- adapter 和 spec。
- contract smoke test。
- 最小训练入口测试。
- 最小导出/runtime asset 测试。

legacy alias 不要求单独建立 `schemas/<alias>/` 目录；它可以共享 canonical adapter、contract 和 README。共享时必须满足两点：canonical README 记录 alias 关系，`documents/schema_registry.md` 的 Legacy 规则记录 exact/canonical 区别。alias 仍必须有最小训练和最小导出 smoke 覆盖 exact `schema_name`。

当前 `realtime_pose_stationary5_v1` 的 legacy alias 是 `realtime_pose_body_fbx_local_root_y0_v1`，二者共用 `schemas/realtime_pose_stationary5_v1/` 下的 adapter 和 README。最小链路测试在 `tests/smoke/schemas/test_stationary5_train_export.py` 中同时覆盖 canonical exact name 和 legacy exact name。

## Smoke 测试

运行所有 schema smoke：

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/schemas -q
```

只跑当前 canonical exact schema 的最小 source/task/normalizer/export 与训练参数 smoke：

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest `
  "tests/smoke/schemas/test_stationary5_train_export.py::test_stationary5_schema_toy_source_task_normalizer_export[realtime_pose_stationary5_v1]" `
  "tests/smoke/schemas/test_stationary5_train_export.py::test_stationary5_schema_training_args_accept_exact_name[realtime_pose_stationary5_v1]" `
  -q
```

只跑 legacy exact name 时，把参数化节点中的 schema 改为 `realtime_pose_body_fbx_local_root_y0_v1`。新增 schema 后应给它自己的 contract test 和 train/export 最小测试；不要只依赖 registry 存在性测试。

## 数据和产物路径

AMASS 原始路径和生成产物路径优先通过 `configs/data_roots.local.json` 管理；没有 local 文件时读取 `configs/data_roots.example.json`。相对路径按仓库根目录解析：

```json
{
  "amass_root": "dataset/AMASS",
  "smpl_model_dir": "dataset/body_models",
  "body_fbx_rest_json": "",
  "generated_root": "dataset/generated"
}
```

`amass_root`、`smpl_model_dir` 和 `body_fbx_rest_json` 指向原始输入或 Unity 导出的 rest 数据，不放进 `generated_root`。`generated_root` 下只放可重建产物。若复制下面命令，默认省略 `--data_roots_config`，让脚本自动选择 local 或 example；需要固定本机路径时，先创建 `configs/data_roots.local.json`，再显式添加该参数。

| 类型 | 路径模板 |
| --- | --- |
| source | `generated_root/sources/<schema_name>/<source_set_name>` |
| task set root | `generated_root/tasks/<schema_name>/<task_set_name>` |
| normalizer set root | `generated_root/normalizers/<schema_name>/<normalizer_name>` |
| training run root | `runs/<schema_name>/<experiment_name>` |
| export | `output/<schema_name>/<export_name>` |

task 和 normalizer 的实际产物位于 set root 下的 timestamped 子目录；`latest_tasks.*` 与 `latest_normalizer.*` 指向最近一次具体产物目录。训练入口和 pipeline 可以从 set root 解析 latest，直接调用导出脚本时应传具体 normalizer 产物目录。

转换、task 生成和 normalizer 统计入口支持 set/name 参数：

```powershell
conda run --no-capture-output -n diffusionposer5070 python -m data_converter.amass_to_realtime_pose `
  --schema realtime_pose_stationary5_v1 `
  --source_set_name amass_60hz

conda run --no-capture-output -n diffusionposer5070 python -m data_loaders.generate_realtime_pose_tasks `
  --schema realtime_pose_stationary5_v1 `
  --source_set_name amass_60hz `
  --task_set_name amass_60hz_tasks

conda run --no-capture-output -n diffusionposer5070 python -m data_loaders.compute_realtime_pose_normalizer `
  --schema realtime_pose_stationary5_v1 `
  --task_set_name amass_60hz_tasks `
  --normalizer_name amass_60hz_train
```

`scripts.run_realtime_pose_pipeline` 还使用 `--experiment_name` 和 `--export_name` 推导 `runs/<schema>/<experiment>` 与 `output/<schema>/<export>`。显式传入 `--source_dir`、`--task_dir`、`--normalizer_dir`、`--save_dir` 或 `--export_dir` 会覆盖推导路径，但不会放宽 metadata 校验。直接调用导出脚本时，`--normalizer_dir` 应指向具体 normalizer 产物目录；pipeline 会在传参前把 normalizer set 根解析到 latest 产物目录。

## Legacy 规则

- 只有 registry 中明确注册的 legacy alias 才允许继续维护。当前 legacy exact name 是 `realtime_pose_body_fbx_local_root_y0_v1`，canonical name 是 `realtime_pose_stationary5_v1`。
- legacy alias 可以共享 canonical adapter 和 README；当前 alias 由 `schemas/realtime_pose_stationary5_v1/adapter.py` 注册，README 也在该 canonical 目录下记录 alias 关系。
- exact `schema_name` 是 artifact 身份；source/task/normalizer/checkpoint/Unity runtime asset 必须按 exact name 匹配。
- `schema_canonical_name` 只用于说明两个 exact names 共享同一通道契约，不能用于 checkpoint resume 或 artifact 加载时的宽松匹配。
- resume checkpoint 必须 exact `schema_name` 匹配；canonical 相同但 exact name 不同也应拒绝恢复。
- legacy alias 要继续 trainable/exportable，必须在 canonical README 和 registry 文档中记录 alias 关系，并保留覆盖 exact alias name 的最小训练测试和最小导出测试。
- 未注册的旧 `realtime_pose_v2_contact`、`root_yaw_global_6d`、X277/current277 或其他历史产物不得新增隐式兼容路径；如确需恢复，先显式设计并注册 schema。
