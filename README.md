# DiffusionPoser RealtimePose

当前 Python 本地主链路的数据维度、字段和时序结构统一记录在 [contract.md](contract.md)。代码与生成产物不保存额外的结构名称或版本元数据。

## 数据链路

先从 Unity Editor 导出 `body_fbx_rest.json`，再重新生成 source、task 和 normalizer：

```powershell
conda run -n diffusionposer5070 python -m data_converter.amass_to_realtime_pose `
  --body_fbx_rest_json ..\SIGGRAPH2024Unity\Assets\Projects\RealtimePose\Models\DiffusionPoser\body_fbx_rest.json `
  --amass_dir dataset/AMASS `
  --smpl_model_dir dataset/body_models `
  --output_dir dataset/AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz `
  --target_fps 60 `
  --overwrite

conda run -n diffusionposer5070 python -m data_loaders.generate_realtime_pose_tasks `
  --source_dir dataset/AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz `
  --output_dir dataset/AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz_tasks `
  --split_dir data_loaders/splits `
  --splits train test `
  --base_windows_per_source 20 `
  --max_rollout_steps 4 `
  --shard_size 4096 `
  --seed 10

conda run -n diffusionposer5070 python -m data_loaders.compute_realtime_pose_normalizer `
  --task_dir dataset/AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz_tasks `
  --output_dir dataset/meta_AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz `
  --split train
```

## 训练

```powershell
conda run -n diffusionposer5070 python -m train.train_diffusionposer `
  --model_arch target_dit `
  --data_dir dataset/AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz_tasks `
  --data_split train `
  --normalizer_dir dataset/meta_AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz `
  --save_dir runs/realtime_pose_body_fbx_local_root_y0_stationary5_target_dit `
  --rollout_steps 4 `
  --rollout_prob 0.25 `
  --scenario_weights 1 1 1 1 1 `
  --overwrite
```

Task 生成后由未压缩 `.npy` shard 直接 mmap 读取。每个基础动作窗口保存固定五套 Tracker 配置，训练期只选择配置与是否读取后续 rollout step；这些训练参数不会改变 normalizer。

Unity 与导出链路暂未同步到这次 Python 本地改动。
