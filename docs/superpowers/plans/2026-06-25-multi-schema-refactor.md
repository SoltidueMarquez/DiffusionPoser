# Multi-Schema Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 DiffusionPoser 的 realtime pose 链路重构为多 schema 共存架构，让旧 schema 仍可训练和导出，同时后续 schema 只新增契约差异，不复制整条工程链路。

**Architecture:** 新增 `schemas/` registry 和 adapter 层，`data_converter/`、`data_loaders/`、`train/`、`sample/`、`eval/`、`export/` 继续作为通用主链路。数据来源路径和生成产物路径独立放到 `configs/data_roots.local.json` 与 `utils/artifact_paths.py`，schema 只表达数据语义，不保存本机绝对路径。

**Tech Stack:** Python、NumPy、PyTorch、pytest、现有 Anaconda 环境 `diffusionposer5070`、Unity/Sentis JSON runtime assets。

---

## 命名与兼容策略

新默认 schema 名采用简化命名：

```text
realtime_pose_stationary5_v1
```

旧名继续注册并保持可用：

```text
realtime_pose_body_fbx_local_root_y0_v1
```

二者在第一轮重构中使用同一个 adapter 和相同通道布局。旧名用于历史 source/task/normalizer/checkpoint/runtime asset；新名用于后续新生成产物。以后如果仍是 stationary5 但通道或任务契约不兼容，新增 `realtime_pose_stationary5_v2`。如果 stationary 语义变化，新增语义名，例如 `realtime_pose_stationary7_v1`。

## 文件结构

新增：

```text
schemas/
  __init__.py
  base.py
  registry.py
  realtime_pose_stationary5_v1/
    __init__.py
    adapter.py
    contract.py
    unity.py
    README.md

configs/
  data_roots.example.json

utils/
  data_roots.py
  artifact_paths.py

tests/smoke/schemas/
  test_schema_registry.py
  test_stationary5_contract.py
  test_stationary5_train_export.py
  test_artifact_paths.py
```

修改：

```text
.gitignore
AGENTS.md
README.md
data_loaders/sensor_masking.py
data_loaders/realtime_pose_contract.py
data_loaders/realtime_pose_dataset.py
data_loaders/generate_realtime_pose_tasks.py
data_loaders/compute_realtime_pose_normalizer.py
data_converter/amass_to_realtime_pose.py
utils/parser_util.py
utils/normalizer.py
train/training_loop.py
train/train_diffusionposer.py
sample/simulate_unity_stream.py
sample/evaluate_unity_stream_source.py
sample/evaluate_longseq_eval_set.py
export/write_unity_runtime_assets.py
export/export_sentis_denoiser.py
scripts/run_realtime_pose_pipeline.py
tests/smoke/realtime_pose_fixtures.py
```

## Task 1: 新增 schema 基础接口和 registry

**Files:**
- Create: `schemas/__init__.py`
- Create: `schemas/base.py`
- Create: `schemas/registry.py`
- Test: `tests/smoke/schemas/test_schema_registry.py`

- [ ] **Step 1: 写 registry 失败测试**

```python
from schemas.registry import get_default_schema_name, get_schema_adapter, get_schema_spec, list_schema_names


def test_stationary5_default_schema_registered():
    assert get_default_schema_name() == "realtime_pose_stationary5_v1"
    assert "realtime_pose_stationary5_v1" in list_schema_names(trainable_only=True)
    assert "realtime_pose_body_fbx_local_root_y0_v1" in list_schema_names(trainable_only=True)


def test_legacy_schema_name_keeps_exact_identity():
    spec = get_schema_spec("realtime_pose_body_fbx_local_root_y0_v1")
    assert spec.name == "realtime_pose_body_fbx_local_root_y0_v1"
    assert spec.canonical_name == "realtime_pose_stationary5_v1"
    assert spec.feature_dim == 214


def test_adapter_round_trip_lookup():
    adapter = get_schema_adapter("realtime_pose_stationary5_v1")
    assert adapter.spec.name == "realtime_pose_stationary5_v1"
    assert adapter.spec.pose_representation == "body_fbx_local_delta_6d"
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/schemas/test_schema_registry.py -q
```

Expected: FAIL，提示 `schemas` 模块不存在。

- [ ] **Step 3: 实现 `schemas/base.py`**

定义 `SchemaSpec`，保留现有 `SchemaSpec` 所有字段，并补充：

```python
canonical_name: str
one_line: str
seq_len: int = 61
target_start: int = 60
target_length: int = 1
trainable: bool = True
exportable: bool = True
```

定义 `SchemaAdapter` Protocol，至少包含：

```python
spec: SchemaSpec
validate_source(...)
validate_task(...)
build_inpaint_mask(...)
build_unity_feature_schema(...)
```

- [ ] **Step 4: 实现 `schemas/registry.py`**

提供：

```python
register_schema(adapter: SchemaAdapter) -> None
get_schema_spec(schema_name: str | None) -> SchemaSpec
get_schema_adapter(schema_name: str | None) -> SchemaAdapter
list_schema_names(trainable_only: bool = False, exportable_only: bool = False) -> tuple[str, ...]
get_default_schema_name() -> str
```

默认名为 `realtime_pose_stationary5_v1`。

- [ ] **Step 5: 运行测试确认通过**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/schemas/test_schema_registry.py -q
```

Expected: PASS。

- [ ] **Step 6: Commit**

```powershell
git add schemas tests/smoke/schemas/test_schema_registry.py
git commit -m "refactor: add realtime pose schema registry"
```

## Task 2: 抽出 stationary5 adapter，并保留旧名 alias

**Files:**
- Create: `schemas/realtime_pose_stationary5_v1/__init__.py`
- Create: `schemas/realtime_pose_stationary5_v1/contract.py`
- Create: `schemas/realtime_pose_stationary5_v1/adapter.py`
- Create: `schemas/realtime_pose_stationary5_v1/unity.py`
- Create: `schemas/realtime_pose_stationary5_v1/README.md`
- Modify: `schemas/registry.py`
- Test: `tests/smoke/schemas/test_stationary5_contract.py`

- [ ] **Step 1: 写 stationary5 契约测试**

测试两种 schema name 都得到 214 维、61 帧、target slice 0:154、tracker slice 154:214。

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/schemas/test_stationary5_contract.py -q
```

Expected: FAIL，提示 adapter 未注册。

- [ ] **Step 3: 实现 `contract.py`**

从 `data_loaders/sensor_masking.py` 迁移 stationary5 通道常量：

```text
0:144 body_pose_body_fbx_local_delta_6d
144:146 root_heading_delta_sincos
146:148 root_delta_xz_ref
148:149 pelvis_height
149:154 stationary_prob_5
154:172 tracker_pos_ref
172:208 tracker_rot_ref_6d
208:214 sensor_valid
```

- [ ] **Step 4: 实现 `adapter.py`**

实现 `Stationary5Adapter`，构造两个实例：

```python
CANONICAL_ADAPTER = Stationary5Adapter(name="realtime_pose_stationary5_v1", canonical_name="realtime_pose_stationary5_v1")
LEGACY_ADAPTER = Stationary5Adapter(name="realtime_pose_body_fbx_local_root_y0_v1", canonical_name="realtime_pose_stationary5_v1")
```

- [ ] **Step 5: 实现 `unity.py`**

把 `export/write_unity_runtime_assets.py::build_realtime_pose_feature_schema` 里的 stationary5 runtime schema 生成逻辑搬进 adapter，保持 JSON 字段不变。

- [ ] **Step 6: 写 README**

第一行必须是：

```text
realtime_pose_stationary5_v1：固定 body_fbx_local + root_y0 前提下，使用 61 帧窗口、214 维特征和 stationary_prob_5 的实时姿态重建契约。
```

- [ ] **Step 7: 运行测试确认通过**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/schemas/test_stationary5_contract.py -q
```

Expected: PASS。

- [ ] **Step 8: Commit**

```powershell
git add schemas/realtime_pose_stationary5_v1 schemas/registry.py tests/smoke/schemas/test_stationary5_contract.py
git commit -m "refactor: add stationary5 schema adapter"
```

## Task 3: 把 `sensor_masking.py` 改成兼容门面

**Files:**
- Modify: `data_loaders/sensor_masking.py`
- Test: `tests/smoke/data_pipeline/test_realtime_pose_data.py`
- Test: `tests/smoke/schemas/test_schema_registry.py`

- [ ] **Step 1: 写兼容性断言**

确认旧 import 仍可用：

```python
from data_loaders.sensor_masking import DEFAULT_REALTIME_POSE_SCHEMA_NAME, REALTIME_POSE_SCHEMA_NAMES, get_schema_spec


def test_sensor_masking_exports_schema_registry_compatibility():
    assert DEFAULT_REALTIME_POSE_SCHEMA_NAME == "realtime_pose_stationary5_v1"
    assert "realtime_pose_body_fbx_local_root_y0_v1" in REALTIME_POSE_SCHEMA_NAMES
    assert get_schema_spec("realtime_pose_stationary5_v1").feature_dim == 214
```

- [ ] **Step 2: 运行测试确认失败或旧默认名不符**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/schemas/test_schema_registry.py -q
```

- [ ] **Step 3: 修改 `sensor_masking.py`**

保留现有通道常量和 tracker pattern 函数，删除本地 `SCHEMA_SPECS` 权威定义，改为从 `schemas.registry` 导入：

```python
from schemas.base import SchemaSpec
from schemas.registry import get_default_schema_name, get_schema_spec, list_schema_names

REALTIME_POSE_SCHEMA_NAME = get_default_schema_name()
DEFAULT_REALTIME_POSE_SCHEMA_NAME = REALTIME_POSE_SCHEMA_NAME
REALTIME_POSE_SCHEMA_NAMES = list_schema_names()
```

- [ ] **Step 4: 修正默认名引发的旧测试**

旧测试中如果只是断言默认名等于旧长名，改成断言默认 schema 的 feature_dim 和契约字段。不要删除旧长名可用性测试。

- [ ] **Step 5: 运行数据契约 smoke**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/schemas tests/smoke/data_pipeline/test_realtime_pose_data.py -q
```

Expected: PASS。

- [ ] **Step 6: Commit**

```powershell
git add data_loaders/sensor_masking.py tests/smoke
git commit -m "refactor: route sensor schema lookup through registry"
```

## Task 4: 引入数据路径配置和产物路径推导

**Files:**
- Create: `configs/data_roots.example.json`
- Modify: `.gitignore`
- Create: `utils/data_roots.py`
- Create: `utils/artifact_paths.py`
- Test: `tests/smoke/schemas/test_artifact_paths.py`

- [ ] **Step 1: 写路径测试**

覆盖：

```python
def test_artifact_paths_include_schema_name(tmp_path):
    roots = DataRoots(amass_root=tmp_path / "AMASS", generated_root=tmp_path / "generated")
    assert source_root(roots, "realtime_pose_stationary5_v1", "amass_train").as_posix().endswith(
        "generated/sources/realtime_pose_stationary5_v1/amass_train"
    )
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/schemas/test_artifact_paths.py -q
```

- [ ] **Step 3: 创建 `configs/data_roots.example.json`**

内容：

```json
{
  "amass_root": "dataset/AMASS",
  "smpl_model_dir": "dataset/body_models",
  "body_fbx_rest_json": "",
  "generated_root": "dataset/generated"
}
```

- [ ] **Step 4: 修改 `.gitignore`**

添加：

```text
configs/data_roots.local.json
dataset/generated/
```

- [ ] **Step 5: 实现 `utils/data_roots.py`**

使用标准库 `json`，优先读取显式 `--data_roots_config`，否则读取 `configs/data_roots.local.json`，不存在时回退 example。

- [ ] **Step 6: 实现 `utils/artifact_paths.py`**

提供：

```python
source_root(roots, schema_name, source_set_name)
task_root(roots, schema_name, task_set_name)
normalizer_root(roots, schema_name, normalizer_name)
run_root(schema_name, experiment_name)
export_root(schema_name, export_name)
```

- [ ] **Step 7: 运行测试确认通过**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/schemas/test_artifact_paths.py -q
```

- [ ] **Step 8: Commit**

```powershell
git add .gitignore configs/data_roots.example.json utils/data_roots.py utils/artifact_paths.py tests/smoke/schemas/test_artifact_paths.py
git commit -m "feat: add schema-aware artifact paths"
```

## Task 5: 改造 AMASS source 转换入口

**Files:**
- Modify: `data_converter/amass_to_realtime_pose.py`
- Test: `tests/smoke/data_pipeline/test_realtime_pose_stationary5_data.py`

- [ ] **Step 1: 写 source manifest provenance 测试**

测试转换输出 manifest 包含：

```json
{
  "schema_name": "realtime_pose_stationary5_v1",
  "schema_canonical_name": "realtime_pose_stationary5_v1",
  "raw_dataset": "AMASS",
  "raw_root_key": "amass_root",
  "converter_args": {}
}
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/data_pipeline/test_realtime_pose_stationary5_data.py -q
```

- [ ] **Step 3: 修改 CLI 参数**

新增：

```text
--data_roots_config
--source_set_name
```

保留 `--amass_dir`、`--output_dir` 显式覆盖能力；如果没有显式传入，则从 `data_roots + schema_name + source_set_name` 推导路径。

- [ ] **Step 4: 写 source metadata**

每个 `.npz` 和 manifest entry 都写入 schema metadata、raw relative path、converter args。绝对 AMASS 路径只写入顶层 manifest provenance，不写入单个样本契约字段。

- [ ] **Step 5: 运行 source 测试**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/data_pipeline/test_realtime_pose_stationary5_data.py -q
```

- [ ] **Step 6: Commit**

```powershell
git add data_converter/amass_to_realtime_pose.py tests/smoke/data_pipeline/test_realtime_pose_stationary5_data.py
git commit -m "feat: record schema-aware source provenance"
```

## Task 6: 改造 task 生成和 normalizer 产物目录

**Files:**
- Modify: `data_loaders/generate_realtime_pose_tasks.py`
- Modify: `data_loaders/compute_realtime_pose_normalizer.py`
- Test: `tests/smoke/data_pipeline/test_realtime_pose_pipeline.py`

- [ ] **Step 1: 写 task/normalizer 默认路径测试**

验证未传 `--output_dir` 时路径包含：

```text
dataset/generated/tasks/<schema_name>/<task_set_name>
dataset/generated/normalizers/<schema_name>/<normalizer_name>
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/data_pipeline/test_realtime_pose_pipeline.py -q
```

- [ ] **Step 3: 修改 task CLI**

新增：

```text
--data_roots_config
--task_set_name
```

保留 `--source_dir` 和 `--output_dir` 显式覆盖。

- [ ] **Step 4: 修改 normalizer CLI**

新增：

```text
--data_roots_config
--normalizer_name
```

保留 `--task_dir` 和 `--output_dir` 显式覆盖。

- [ ] **Step 5: 更新 manifest**

task manifest 和 normalizer meta 写入 `schema_name`、`schema_canonical_name`、`source_dir`、`task_dir`、`generated_root`。

- [ ] **Step 6: 运行 pipeline 测试**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/data_pipeline/test_realtime_pose_pipeline.py -q
```

- [ ] **Step 7: Commit**

```powershell
git add data_loaders/generate_realtime_pose_tasks.py data_loaders/compute_realtime_pose_normalizer.py tests/smoke/data_pipeline/test_realtime_pose_pipeline.py
git commit -m "feat: route task and normalizer outputs by schema"
```

## Task 7: 解除训练入口单 schema 限制

**Files:**
- Modify: `utils/parser_util.py`
- Modify: `train/training_loop.py`
- Modify: `train/train_diffusionposer.py`
- Test: `tests/smoke/train/test_train_entrypoint.py`
- Test: `tests/smoke/train/test_resume_checkpoint.py`

- [ ] **Step 1: 写多 schema choices 测试**

验证 `--schema realtime_pose_body_fbx_local_root_y0_v1` 和 `--schema realtime_pose_stationary5_v1` 都能 parse。

- [ ] **Step 2: 写 resume schema mismatch 测试**

构造 args.json 中 schema 为旧名，CLI schema 为新名，期望报错。旧名和新名虽然同 adapter，但 checkpoint 恢复必须按 exact `schema_name` 一致，避免混用产物。

- [ ] **Step 3: 运行测试确认失败**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/train/test_train_entrypoint.py tests/smoke/train/test_resume_checkpoint.py -q
```

- [ ] **Step 4: 修改 parser**

`TRAIN_REALTIME_POSE_SCHEMA_NAMES` 改为：

```python
tuple(list_schema_names(trainable_only=True))
```

`--input_feats` 默认从所选 schema 推导；如果 argparse 阶段不能依赖另一个参数，则在训练启动后用 schema 覆盖并校验。

- [ ] **Step 5: 修改 `validate_root_y0_training_args`**

重命名为 `validate_realtime_pose_training_args`。删除 `schema.name != REALTIME_POSE_SCHEMA_NAME` 限制，保留：

```text
input_feats == schema.feature_dim
seq_len == schema.seq_len
max_seq_len == schema.seq_len
```

- [ ] **Step 6: 更新 checkpoint metadata**

`args.json` 同时写：

```json
{
  "schema": "<exact schema_name>",
  "schema_name": "<exact schema_name>",
  "schema_canonical_name": "<canonical schema name>",
  "pose_representation": "body_fbx_local_delta_6d",
  "root_y_policy": "fixed_zero",
  "pelvis_height_mode": "pelvis_local_offset_y"
}
```

- [ ] **Step 7: 运行训练测试**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/train/test_train_entrypoint.py tests/smoke/train/test_resume_checkpoint.py -q
```

- [ ] **Step 8: Commit**

```powershell
git add utils/parser_util.py train/training_loop.py train/train_diffusionposer.py tests/smoke/train
git commit -m "refactor: allow trainable schemas through training entrypoint"
```

## Task 8: 让 export 通过 adapter 生成 Unity runtime asset

**Files:**
- Modify: `export/write_unity_runtime_assets.py`
- Modify: `export/export_sentis_denoiser.py`
- Test: `tests/smoke/export/test_runtime_assets.py`

- [ ] **Step 1: 写新旧 schema export 测试**

同一个 toy normalizer 分别用旧名和新名导出，断言 `feature_schema.json` 的 `schemaName` 等于 exact schema name。

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/export/test_runtime_assets.py -q
```

- [ ] **Step 3: 修改 `write_unity_runtime_assets.py`**

`build_realtime_pose_feature_schema` 改为：

```python
adapter = get_schema_adapter(schema_name)
return adapter.build_unity_feature_schema(...)
```

保留 `build_normalizer` 和 `build_ddim_schedule` 的公共逻辑，但 schema 信息从 `adapter.spec` 取。

- [ ] **Step 4: 修改 Sentis 导出**

`export_sentis_denoiser.py` 从 checkpoint args 读取 exact schema name；如果 CLI 显式 `--schema`，必须和 checkpoint schema 一致。

- [ ] **Step 5: 运行 export 测试**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/export/test_runtime_assets.py -q
```

- [ ] **Step 6: Commit**

```powershell
git add export/write_unity_runtime_assets.py export/export_sentis_denoiser.py tests/smoke/export/test_runtime_assets.py
git commit -m "refactor: delegate Unity feature schema to adapters"
```

## Task 9: 改造 sample/eval 读取 checkpoint schema

**Files:**
- Modify: `sample/simulate_unity_stream.py`
- Modify: `sample/evaluate_unity_stream_source.py`
- Modify: `sample/evaluate_longseq_eval_set.py`
- Modify: `eval/evaluate_realtime_pose.py`
- Test: `tests/smoke/sample/test_simulate_unity_stream.py`
- Test: `tests/smoke/sample/test_evaluate_unity_stream_source.py`
- Test: `tests/smoke/eval/test_realtime_pose_rollout.py`

- [ ] **Step 1: 写 checkpoint schema 读取测试**

构造 args.json 中 schema 为旧名，sample CLI 不传 schema 时应使用 checkpoint schema；传入不同 schema 时应报错。

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/sample tests/smoke/eval -q
```

- [ ] **Step 3: 修改 sample/eval schema 选择规则**

统一规则：

```text
1. 如果 checkpoint args 有 schema，默认使用 checkpoint schema。
2. 如果 CLI 显式传 --schema，必须等于 checkpoint schema。
3. 如果没有 checkpoint schema，使用 CLI schema。
4. 如果都没有，使用 DEFAULT_REALTIME_POSE_SCHEMA_NAME。
```

- [ ] **Step 4: 运行 sample/eval 测试**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/sample tests/smoke/eval -q
```

- [ ] **Step 5: Commit**

```powershell
git add sample eval tests/smoke/sample tests/smoke/eval
git commit -m "refactor: resolve sample schemas from checkpoint metadata"
```

## Task 10: 更新一键 pipeline 脚本

**Files:**
- Modify: `scripts/run_realtime_pose_pipeline.py`
- Test: `tests/smoke/data_pipeline/test_realtime_pose_pipeline.py`

- [ ] **Step 1: 写 pipeline 参数测试**

验证 pipeline 默认 schema 是 `realtime_pose_stationary5_v1`，并且所有阶段传递同一个 exact schema name。

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/data_pipeline/test_realtime_pose_pipeline.py -q
```

- [ ] **Step 3: 修改 pipeline**

新增：

```text
--data_roots_config
--source_set_name
--task_set_name
--normalizer_name
--experiment_name
--export_name
```

默认输出全部通过 `utils/artifact_paths.py` 推导。

- [ ] **Step 4: 保留显式路径覆盖**

如果用户传 `--source_dir`、`--task_dir`、`--normalizer_dir`、`--save_dir`、`--export_dir`，优先使用显式路径，并在日志中打印 `schema_name` 与路径。

- [ ] **Step 5: 运行 pipeline smoke**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/data_pipeline/test_realtime_pose_pipeline.py -q
```

- [ ] **Step 6: Commit**

```powershell
git add scripts/run_realtime_pose_pipeline.py tests/smoke/data_pipeline/test_realtime_pose_pipeline.py
git commit -m "feat: make realtime pipeline schema-aware"
```

## Task 11: 补齐最小 legacy schema train/export 矩阵

**Files:**
- Create/Modify: `tests/smoke/schemas/test_stationary5_train_export.py`
- Modify: `tests/smoke/realtime_pose_fixtures.py`

- [ ] **Step 1: 写参数化测试**

对以下 schema 参数化：

```python
@pytest.mark.parametrize(
    "schema_name",
    [
        "realtime_pose_stationary5_v1",
        "realtime_pose_body_fbx_local_root_y0_v1",
    ],
)
```

覆盖 toy source、task、normalizer、tiny train args、runtime asset export。

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/schemas/test_stationary5_train_export.py -q
```

- [ ] **Step 3: 调整 fixtures**

`build_toy_realtime_source`、`write_toy_source_dataset` 默认用 `DEFAULT_REALTIME_POSE_SCHEMA_NAME`，但允许传 exact schema name，并写入相同 exact schema name。

- [ ] **Step 4: 运行 schema smoke**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/schemas -q
```

- [ ] **Step 5: Commit**

```powershell
git add tests/smoke/schemas tests/smoke/realtime_pose_fixtures.py
git commit -m "test: cover train and export for legacy schemas"
```

## Task 12: 文档和 AGENTS 约定更新

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`
- Create/Modify: `documents/schema_registry.md`

- [ ] **Step 1: 更新 `AGENTS.md`**

把“不兼容且不再保留旧 schema”改为：

```text
仓库允许维护 registry 中明确注册的 legacy schema；每个 trainable/exportable schema 必须有 README、adapter、contract smoke test、最小训练测试和最小导出测试。
```

- [ ] **Step 2: 更新 README**

说明新默认 schema、旧 schema 训练/导出方式、路径配置文件和产物目录。

- [ ] **Step 3: 新增 schema registry 文档**

内容包含：

```text
schema_name 命名规则
什么时候新增 schema
什么时候只改 run/config
如何新增 adapter
如何运行单 schema smoke
如何处理旧数据路径
```

- [ ] **Step 4: 运行文档相关 smoke**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/schemas -q
```

- [ ] **Step 5: Commit**

```powershell
git add AGENTS.md README.md documents/schema_registry.md
git commit -m "docs: document multi-schema workflow"
```

## Task 13: 全链路验证

**Files:**
- No source changes unless verification exposes a bug.

- [ ] **Step 1: 运行 schema smoke**

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/schemas -q
```

Expected: PASS。

- [ ] **Step 2: 运行数据链路 smoke**

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/data_pipeline -q
```

Expected: PASS。

- [ ] **Step 3: 运行训练 smoke**

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/train -q
```

Expected: PASS。

- [ ] **Step 4: 运行 sample/eval/export smoke**

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/sample tests/smoke/eval tests/smoke/export -q
```

Expected: PASS。

- [ ] **Step 5: 运行完整 smoke**

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke
```

Expected: PASS。

- [ ] **Step 6: Commit 验证修复**

如果验证阶段有修复：

```powershell
git add <changed-files>
git commit -m "fix: complete multi-schema smoke coverage"
```

## 实施注意事项

- 不要一次性删除旧 schema 名；旧数据、normalizer、checkpoint、Unity asset 依赖 exact schema name。
- 不要把 AMASS 绝对路径写进 schema contract；绝对路径只属于 `configs/data_roots.local.json`。
- 不要复制 `train/`、`diffusion/`、`export/`、`sample/` 目录；schema 差异只能进 adapter。
- resume checkpoint 必须按 exact schema name 匹配，不按 canonical name 放宽。
- 新增 schema 时必须先写 README 第一行、contract test、最小 train/export test。
