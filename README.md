# DiffusionPoser RealtimePose

当前默认 schema 是 `realtime_pose_stationary5_v1`，姿态表示为 `body_fbx_local_delta_6d`。legacy exact name `realtime_pose_body_fbx_local_root_y0_v1` 仍在 registry 中注册，可继续训练和导出；恢复 checkpoint、读取 task/normalizer 和导出 Unity runtime asset 时必须使用产物里记录的 exact `schema_name`，不能只按 canonical name 放宽匹配。

`schema_name` 只描述数据产物和 Unity runtime 必须共同理解的稳定契约。实验名、模型结构、loss 或训练超参变化应放在 run/config 名称中，不应写进 `schema_name`。

## Schema 契约

固定窗口仍是 `seq_len=61`、`target_start=60`、`target_length=1`，模型输入输出为 `[B, 214, 61]`，第 61 帧补全 target `0:154`。

| 范围 | 维度 | 含义 |
| --- | ---: | --- |
| `0:144` | 144 | `body_pose_body_fbx_local_delta_6d`，24 个 body.fbx local rotation delta 6D，pelvis/root delta 固定 identity |
| `144:146` | 2 | `root_heading_delta_sincos` |
| `146:148` | 2 | `root_delta_xz_ref`，上一帧 root heading 参考系下 actor root XZ 位移；actor root y 固定为 0 |
| `148:149` | 1 | `pelvis_height`，pelvis local offset y；root-y0 下数值等于 pelvis world y，FK 时覆盖 bone 0 的 local offset y |
| `149:154` | 5 | `stationary_prob_5` |
| `154:172` | 18 | `tracker_pos_ref` |
| `172:208` | 36 | `tracker_rot_ref_6d` |
| `208:214` | 6 | `sensor_valid` |

source/task/normalizer/runtime asset 必须包含所选 exact `schema_name`、`pose_representation="body_fbx_local_delta_6d"`、`root_y_policy="fixed_zero"` 和 `pelvis_height_mode="pelvis_local_offset_y"`。新 source 还必须包含 Unity Editor 从 `Assets/Models/body.fbx` 导出的 `body_fbx_rest.json` 对应 rest 数据：`joint_offsets_parent` 和 `joint_rest_local_rotations_6d`。

## 数据根目录

本机路径优先写在 `configs/data_roots.local.json`；没有 local 文件时会读取 `configs/data_roots.example.json`。相对路径按仓库根目录解析。如果尚未创建 local 文件并只想使用 example 配置，下面命令中的 `--data_roots_config configs/data_roots.local.json` 可以省略。

```json
{
  "amass_root": "dataset/AMASS",
  "smpl_model_dir": "dataset/body_models",
  "body_fbx_rest_json": "../SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Models/DiffusionPoser/body_fbx_rest.json",
  "generated_root": "dataset/generated"
}
```

默认产物布局按 schema 和 set/name 分层：

| 产物 | 默认位置 |
| --- | --- |
| source set | `dataset/generated/sources/realtime_pose_stationary5_v1/amass_60hz` |
| task set | `dataset/generated/tasks/realtime_pose_stationary5_v1/amass_60hz_tasks` |
| normalizer | `dataset/generated/normalizers/realtime_pose_stationary5_v1/amass_60hz_train` |
| training runs | `runs/realtime_pose_stationary5_v1/<experiment_name>` |
| export output | `output/realtime_pose_stationary5_v1/<export_name>` |

## 数据流

先在 Unity Editor 执行菜单：

```text
RealtimePose/Export body_fbx_rest.json
```

然后用 schema、set/name 参数生成 source、task 和 normalizer：

```powershell
conda run --no-capture-output -n diffusionposer5070 python -m data_converter.amass_to_realtime_pose `
  --schema realtime_pose_stationary5_v1 `
  --data_roots_config configs/data_roots.local.json `
  --source_set_name amass_60hz `
  --target_fps 60 `
  --overwrite

conda run --no-capture-output -n diffusionposer5070 python -m data_loaders.generate_realtime_pose_tasks `
  --schema realtime_pose_stationary5_v1 `
  --data_roots_config configs/data_roots.local.json `
  --source_set_name amass_60hz `
  --task_set_name amass_60hz_tasks `
  --split_dir data_loaders/splits `
  --splits train test `
  --samples_per_file 4 `
  --mask_policy full `
  --overwrite

conda run --no-capture-output -n diffusionposer5070 python -m data_loaders.compute_realtime_pose_normalizer `
  --schema realtime_pose_stationary5_v1 `
  --data_roots_config configs/data_roots.local.json `
  --task_set_name amass_60hz_tasks `
  --normalizer_name amass_60hz_train `
  --split train `
  --overwrite
```

`--output_dir`、`--source_dir`、`--task_dir` 和 `--normalizer_dir` 仍可显式覆盖上述布局；覆盖后仍必须保证目录内 metadata 的 exact `schema_name` 与 CLI 一致。

## Pipeline

`scripts.run_realtime_pose_pipeline` 会用同一套 `--schema`、`--source_set_name`、`--task_set_name`、`--normalizer_name`、`--experiment_name` 和 `--export_name` 解析路径。未显式指定目录时，它会使用 `configs/data_roots.local.json` 和 `generated_root/sources|tasks|normalizers/<schema>/<name>`。

```powershell
conda run --no-capture-output -n diffusionposer5070 python -m scripts.run_realtime_pose_pipeline `
  --schema realtime_pose_stationary5_v1 `
  --data_roots_config configs/data_roots.local.json `
  --source_set_name amass_60hz `
  --task_set_name amass_60hz_tasks `
  --normalizer_name amass_60hz_train `
  --experiment_name stationary5_target_dit `
  --export_name stationary5_unity `
  --stop_after export
```

如需继续使用旧 exact name，把 `--schema` 改为 `realtime_pose_body_fbx_local_root_y0_v1`，并使用对应 legacy 产物目录。不要把 canonical schema 和 legacy exact schema 的 task、normalizer、checkpoint 或 Unity asset 混用。

## 训练与导出

也可以跳过 pipeline，直接对 task/normalizer set 根目录或具体 run 目录训练；直接导出时 `--normalizer_dir` 指向具体 normalizer 产物目录，或使用 pipeline 代为解析 latest 指针：

```powershell
conda run --no-capture-output -n diffusionposer5070 python -m train.train_diffusionposer `
  --schema realtime_pose_stationary5_v1 `
  --model_arch target_dit `
  --input_feats 214 `
  --data_dir dataset/generated/tasks/realtime_pose_stationary5_v1/amass_60hz_tasks `
  --data_split train `
  --normalizer_dir dataset/generated/normalizers/realtime_pose_stationary5_v1/amass_60hz_train `
  --save_dir runs/realtime_pose_stationary5_v1/stationary5_target_dit `
  --run_name auto

conda run --no-capture-output -n diffusionposer5070 python -m export.write_unity_runtime_assets `
  --schema realtime_pose_stationary5_v1 `
  --output_dir output/realtime_pose_stationary5_v1/stationary5_unity `
  --normalizer_dir dataset/generated/normalizers/realtime_pose_stationary5_v1/amass_60hz_train/<normalizer_run> `
  --normalize_input

conda run --no-capture-output -n diffusionposer5070 python -m export.export_sentis_denoiser `
  --schema realtime_pose_stationary5_v1 `
  --model_arch target_dit `
  --model_path runs/realtime_pose_stationary5_v1/stationary5_target_dit/<run>/model000000001.pt `
  --output_dir output/realtime_pose_stationary5_v1/stationary5_unity `
  --normalizer_dir dataset/generated/normalizers/realtime_pose_stationary5_v1/amass_60hz_train/<normalizer_run>
```

Unity runtime 直接解码 body.fbx local delta：`bone.localRotation = restLocalRotation * decodedDelta`。Actor root y 固定为 0，`pelvis_height` 通过 pelvis bone `localPosition.y` 生效；`WorldOffset` 只用于显示偏移，不参与模型语义。

## 测试

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/schemas -q
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke
```

更多 schema 维护规则见 `documents/schema_registry.md`。
