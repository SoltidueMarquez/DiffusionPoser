# DiffusionPoser RealtimePose

当前主链路是 `realtime_pose_v2_contact`；旧 X277/current277 数据、task、checkpoint、Unity schema 不兼容。

## Schema 契约

所有 schema 固定使用 `seq_len=61`、`target_start=60`、`target_length=1`，输入张量为 `[B, C, 61]`。

| schema | feature_dim | target_dim | target |
| --- | ---: | ---: | --- |
| `realtime_pose_v2_motion` | 209 | 149 | pose + yaw + root_delta_xz_ref + root_height |
| `realtime_pose_v2_contact` | 211 | 151 | v2_motion + left/right foot contact |

推荐 v2_contact 通道布局：

| 范围 | 维度 | 含义 |
| --- | ---: | --- |
| `0:144` | 144 | `body_pose_root_global_6d` |
| `144:146` | 2 | `root_yaw_delta_sincos` |
| `146:148` | 2 | `root_delta_xz_ref` |
| `148:149` | 1 | `root_height` |
| `149:151` | 2 | `foot_contact` |
| `151:169` | 18 | `tracker_pos_ref` |
| `169:205` | 36 | `tracker_rot_ref_6d` |
| `205:211` | 6 | `sensor_valid` |

source/task/normalizer/runtime asset 必须带 `pose_representation="root_yaw_global_6d"`；旧 `body_pose_parent_6d` 数据与当前 schema 不兼容，需要重新生成。

`sensor_valid` 和 `foot_contact` 在 normalizer 中固定 `mean=0,std=1`。invalid tracker 的 position/rotation 通道在归一化后置零。

## 一站式脚本

推荐直接从 AMASS 转换、task 生成、normalizer 统计一路跑到训练：

```powershell
conda run -n diffusionposer5070 python -m scripts.run_realtime_pose_pipeline `
  --schema realtime_pose_v2_contact `
  --amass_dir dataset/AMASS `
  --smpl_model_dir dataset/body_models `
  --source_dir dataset/AMASS_realtime_pose_v2_60hz `
  --normalizer_dir dataset/meta_AMASS_realtime_pose_v2_60hz `
  --task_dir dataset/AMASS_realtime_pose_v2_60hz_tasks `
  --save_dir runs/realtime_pose_v2_contact_target_dit `
  --model_arch target_dit `
  --num_steps 1000000 `
  --train_batch_size 64 `
  --overwrite
```

`--reuse_source_dir` 只接受已经包含当前 schema 全部字段的 v2 source；默认不启用复用，避免把旧特征格式混入当前训练链路。

如果前面步骤已经完成，可以用 `--start_at tasks`、`--start_at normalizer` 或 `--start_at train` 从中间继续；只想检查将要执行的命令，用 `--dry_run`。如果确实要强制重跑 source SMPL forward，添加 `--rebuild_source`。

VSCode 里可以直接运行：

- `Tasks: Run Task` -> `realtime_pose_v2: one-stop convert-to-train`
- `Run and Debug` -> `realtime_pose_v2 | one-stop convert-to-train`

## 数据流程

```powershell
conda run -n diffusionposer5070 python -m data_converter.amass_to_realtime_pose `
  --amass_dir dataset/AMASS `
  --smpl_model_dir dataset/body_models `
  --output_dir dataset/AMASS_realtime_pose_v2_60hz `
  --target_fps 60 `
  --overwrite

conda run -n diffusionposer5070 python -m data_loaders.generate_realtime_pose_tasks `
  --schema realtime_pose_v2_contact `
  --source_dir dataset/AMASS_realtime_pose_v2_60hz `
  --output_dir dataset/AMASS_realtime_pose_v2_60hz_tasks `
  --split_dir data_loaders/splits `
  --splits train test `
  --samples_per_file 4 `
  --mask_policy full `
  --overwrite

conda run -n diffusionposer5070 python -m data_loaders.compute_realtime_pose_normalizer `
  --schema realtime_pose_v2_contact `
  --task_dir dataset/AMASS_realtime_pose_v2_60hz_tasks `
  --output_dir dataset/meta_AMASS_realtime_pose_v2_60hz `
  --split train `
  --overwrite
```

## 训练

```powershell
conda run -n diffusionposer5070 python -m train.train_diffusionposer `
  --schema realtime_pose_v2_contact `
  --model_arch target_dit `
  --input_feats 211 `
  --data_dir dataset/AMASS_realtime_pose_v2_60hz_tasks `
  --data_split train `
  --normalizer_dir dataset/meta_AMASS_realtime_pose_v2_60hz `
  --save_dir runs/realtime_pose_v2_contact_target_dit `
  --overwrite
```

常用自回归增强：

```powershell
--history_pose_noise_std 0.02 --history_yaw_noise_std 0.02 `
--history_pose_dropout_prob 0.05 --history_pose_replace_prob 0.05 `
--tracker_latency_max_frames 2 --tracker_burst_dropout_prob 0.05 --tracker_outlier_prob 0.01 `
--predicted_history_cache_dir output/pred_history_cache --predicted_history_prob 0.25
```

## Rollout

```powershell
conda run -n diffusionposer5070 python -m sample.reconstruct_rollout `
  --schema realtime_pose_v2_contact `
  --model_path runs/realtime_pose_v2_contact_target_dit/model000050000.pt `
  --data_dir dataset/AMASS_realtime_pose_v2_60hz_tasks `
  --data_split test `
  --normalizer_dir dataset/meta_AMASS_realtime_pose_v2_60hz `
  --output_dir output/realtime_pose_v2_rollout

conda run -n diffusionposer5070 python -m eval.evaluate_realtime_pose_rollout `
  --input_dir output/realtime_pose_v2_rollout `
  --output_json output/realtime_pose_v2_rollout/rollout_eval_summary.json
```

## 导出

```powershell
conda run -n diffusionposer5070 python -m export.write_unity_runtime_assets `
  --schema realtime_pose_v2_contact `
  --output_dir <UnityModelDir> `
  --normalizer_dir dataset/meta_AMASS_realtime_pose_v2_60hz `
  --normalize_input

conda run -n diffusionposer5070 python -m export.export_sentis_denoiser `
  --schema realtime_pose_v2_contact `
  --model_arch target_dit `
  --model_path runs/realtime_pose_v2_contact_target_dit/model000000001.pt `
  --output_dir <UnityModelDir> `
  --normalizer_dir dataset/meta_AMASS_realtime_pose_v2_60hz
```

Unity 端 ring buffer 必须用上一帧 predicted `root_yaw` 编码 tracker；v2 运行时需要积分 `root_delta_xz_ref`，用 `root_height` 驱动 pelvis 高度，并用 `foot_contact` 做 foot lock / IK。

## 测试

```powershell
conda run -n diffusionposer5070 pytest tests/smoke
```
