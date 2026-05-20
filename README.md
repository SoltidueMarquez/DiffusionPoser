# DiffusionPoser RealtimePose v1

本仓库主链路已经切换为 `realtime_pose_v1`。旧 X277/current277 数据、task、checkpoint、Unity schema 不再兼容，旧数据需要重新转换和重新生成任务。

## 固定任务契约

- `schema_name = realtime_pose_v1`
- `seq_len = 61`
- `target_start = 60`
- `target_length = 1`
- `feature_dim = 206`
- 输入张量为 `[B, 206, 61]`
- 第 61 帧只生成 `body_pose_parent_6d + root_yaw_delta_sincos`，即通道 `0:146`
- tracker 条件和 `sensor_valid` 永远不参与 diffusion loss

通道布局：

| 范围 | 维度 | 含义 |
| --- | ---: | --- |
| `0:144` | 144 | `body_pose_parent_6d` |
| `144:146` | 2 | `root_yaw_delta_sincos` |
| `146:164` | 18 | `tracker_pos_ref` |
| `164:200` | 36 | `tracker_rot_ref_6d` |
| `200:206` | 6 | `sensor_valid` |

## 数据流程

1. AMASS 转换为 realtime 源数据：

```powershell
conda run -n diffusionposer5070 python -m data_converter.amass_to_realtime_pose --amass_dir dataset/AMASS --smpl_model_dir dataset/body_models --output_dir dataset/AMASS_realtime_pose_60hz --target_fps 60 --overwrite
```

2. 计算 normalizer：

```powershell
conda run -n diffusionposer5070 python -m data_loaders.compute_realtime_pose_normalizer --source_dir dataset/AMASS_realtime_pose_60hz --output_dir dataset/meta_AMASS_realtime_pose_60hz --split_dir data_loaders/splits --split train --overwrite
```

3. 生成 61 帧 task：

```powershell
conda run -n diffusionposer5070 python -m data_loaders.generate_realtime_pose_tasks --source_dir dataset/AMASS_realtime_pose_60hz --output_dir dataset/AMASS_realtime_pose_60hz_tasks --split_dir data_loaders/splits --splits train test --samples_per_file 4 --mask_policy full --overwrite
```

task generator 默认每个 61 帧窗口只写出 1 条 full-tracker task，`sensor_valid` 全 1。训练用 tracker 随机遮盖发生在 Dataset 阶段；评估、可视化和 debug 如果需要固定 sparse tracker，可显式使用 `--mask_policy fixed_patterns --fixed_tracker_patterns mixed-sparse`。

## 训练

```powershell
conda run -n diffusionposer5070 python -m train.train_diffusionposer --data_dir dataset/AMASS_realtime_pose_60hz_tasks --data_split train --normalizer_dir dataset/meta_AMASS_realtime_pose_60hz --save_dir runs/realtime_pose_v1_train --overwrite
```

模型默认 `input_feats=206, seq_len=61, max_seq_len=61`，DiT 中包含 learnable frame positional embedding，并保留 diffusion timestep embedding。

可选训练增强参数：

```powershell
--tracker_mask_policy dynamic_categories --tracker_mask_seed 10 --tracker_mask_fill zero --tracker_mask_categories all --tracker_pos_noise_std 0.005 --tracker_rot_noise_std 0.01 --non_hip_tracker_dropout_prob 0.1 --history_pose_noise_std 0.01 --history_yaw_noise_std 0.01 --root_yaw_ref_noise_std 0.01
```

train split 默认 `--tracker_mask_policy auto`，等价于 `dynamic_categories`；非 train split 默认使用 task 内固定 `sensor_valid`，保证 eval/sample 可复现。invalid tracker 的 `tracker_pos_ref/tracker_rot_ref_6d` 在归一化后固定置零，本版不使用 GT 填充或上一帧 stale fill。

训练 loss：

```text
L_total = L_denoise + 10.0 * L_yaw + 2.0 * L_fk + 0.5 * L_joint_vel + 0.5 * L_foot_lock
```

`contact` 不保存到数据中；foot-lock 训练项和 Unity 端规则 contact 都由 `joints_world` 动态派生。

## 导出

Unity runtime assets 只导出 `realtime_pose_v1`：

```powershell
conda run -n diffusionposer5070 python -m export.write_unity_runtime_assets --output_dir <UnityModelDir> --normalizer_dir dataset/meta_AMASS_realtime_pose_60hz --normalize_input
```

ONNX/Sentis 导出固定 dummy input `[1,206,61]`：

```powershell
conda run -n diffusionposer5070 python -m export.export_sentis_denoiser --model_path runs/realtime_pose_v1_train/model000000001.pt --output_dir <UnityModelDir> --normalizer_dir dataset/meta_AMASS_realtime_pose_60hz
```

Unity 端应维护 61 帧 ring buffer，用上一帧 `root_yaw` 编码 tracker。若 hip invalid 或 total valid trackers 小于 3，应保持上一帧或进入 fail-safe，不调用 denoiser。

## Unity 端目录

Unity 项目不在本仓库内部，默认位于同级目录：

```text
../SIGGRAPH2024Unity
```

RealtimePose 相关代码和资产集中在：

```text
../SIGGRAPH2024Unity/Assets/Projects/RealtimePose/
├─ Scripts/
│  ├─ Core/        # DiffusionPoserRealtimeDriver、RealtimePoseInferencePipeline
│  ├─ Features/    # realtime_pose_v1 schema、normalizer、feature encoder
│  ├─ Input/       # tracker frame/source 和 valid tracker 校验
│  ├─ Inference/   # Sentis denoiser runner、DDIM sampler/schedule
│  ├─ Output/      # pose decoder、prediction、actor applier
│  ├─ Debug/       # tensor dump 和输出诊断
│  └─ Editor/      # RealtimePose 测试场景构建和校验
├─ Models/DiffusionPoser/
│  ├─ feature_schema.json
│  ├─ normalizer.json
│  ├─ ddim_schedule.json
│  └─ diffusionposer_denoiser.onnx  # 训练 realtime checkpoint 后再导出
└─ Scenes/
   └─ RealtimePose_DiffusionPoser_Test.unity
```

`Models/DiffusionPoser/` 只允许放 `realtime_pose_v1` 运行时资产；旧 277 维资产不要再放回这个目录。

## RealtimePose Studio

可视化编辑器已改为扫描 realtime source、task 和 reconstruction result：

```powershell
conda run -n diffusionposer5070 python -m visual_editor.server --source_dir dataset/AMASS_realtime_pose_60hz --data_dir dataset/AMASS_realtime_pose_60hz_tasks --result_dir output --output_dir visual_editor/.runtime/exports
```

Viewer 使用 `root_pos_world + root_yaw + body_pose_parent_6d` 对应的 joints 显示骨架，并直接显示 `tracker_pos_world`。task exporter 强制 61 帧窗口、hip valid 和每帧至少 3 个 tracker。

## 测试

```powershell
conda run -n diffusionposer5070 pytest tests/smoke
```
