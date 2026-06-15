# DiffusionPoser RealtimePose

当前主链路是 `realtime_pose_body_fbx_local_root_y0_v1`，姿态表示为 `body_fbx_local_delta_6d`。上一代 body_fbx local schema、已删除的 `realtime_pose_v2_contact` / `root_yaw_global_6d` source、task、normalizer、checkpoint、Unity runtime asset 不能和当前链路混用。

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

source/task/normalizer/runtime asset 必须包含 `pose_representation="body_fbx_local_delta_6d"`。新 source 还必须包含 Unity Editor 从 `Assets/Models/body.fbx` 导出的 `body_fbx_rest.json` 对应 rest 数据：`joint_offsets_parent` 和 `joint_rest_local_rotations_6d`。

## 数据流

先在 Unity Editor 执行菜单：

```text
RealtimePose/Export body_fbx_rest.json
```

然后重新生成 source、task、normalizer、checkpoint 和 Unity assets：

```powershell
conda run -n diffusionposer5070 python -m data_converter.amass_to_realtime_pose `
  --schema realtime_pose_body_fbx_local_root_y0_v1 `
  --body_fbx_rest_json ..\SIGGRAPH2024Unity\Assets\Projects\RealtimePose\Models\DiffusionPoser\body_fbx_rest.json `
  --amass_dir dataset/AMASS `
  --smpl_model_dir dataset/body_models `
  --output_dir dataset/AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz `
  --target_fps 60 `
  --overwrite

conda run -n diffusionposer5070 python -m data_loaders.generate_realtime_pose_tasks `
  --schema realtime_pose_body_fbx_local_root_y0_v1 `
  --source_dir dataset/AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz `
  --output_dir dataset/AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz_tasks `
  --split_dir data_loaders/splits `
  --splits train test `
  --samples_per_file 4 `
  --mask_policy full `
  --overwrite

conda run -n diffusionposer5070 python -m data_loaders.compute_realtime_pose_normalizer `
  --schema realtime_pose_body_fbx_local_root_y0_v1 `
  --task_dir dataset/AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz_tasks `
  --output_dir dataset/meta_AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz `
  --split train `
  --overwrite
```

## 训练与导出

```powershell
conda run -n diffusionposer5070 python -m train.train_diffusionposer `
  --schema realtime_pose_body_fbx_local_root_y0_v1 `
  --model_arch target_dit `
  --input_feats 214 `
  --data_dir dataset/AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz_tasks `
  --data_split train `
  --normalizer_dir dataset/meta_AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz `
  --save_dir runs/realtime_pose_body_fbx_local_root_y0_stationary5_target_dit `
  --overwrite

conda run -n diffusionposer5070 python -m export.write_unity_runtime_assets `
  --schema realtime_pose_body_fbx_local_root_y0_v1 `
  --output_dir ..\SIGGRAPH2024Unity\Assets\Projects\RealtimePose\Models\DiffusionPoser `
  --normalizer_dir dataset/meta_AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz\<normalizer_run> `
  --normalize_input

conda run -n diffusionposer5070 python -m export.export_sentis_denoiser `
  --schema realtime_pose_body_fbx_local_root_y0_v1 `
  --model_arch target_dit `
  --model_path runs\realtime_pose_body_fbx_local_root_y0_stationary5_target_dit\<run>\model000000001.pt `
  --output_dir ..\SIGGRAPH2024Unity\Assets\Projects\RealtimePose\Models\DiffusionPoser `
  --normalizer_dir dataset/meta_AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz\<normalizer_run>
```

Unity runtime 现在直接解码 body.fbx local delta：`bone.localRotation = restLocalRotation * decodedDelta`。Actor root y 固定为 0，`pelvis_height` 通过 pelvis bone `localPosition.y` 生效；`WorldOffset` 只用于显示偏移，不参与模型语义。

## 测试

```powershell
conda run -n diffusionposer5070 pytest tests/smoke
```
