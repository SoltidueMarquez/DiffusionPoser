# DiffusionPoser RealtimePose

当前 Python 主链路采用全目录式数据输入。source、Task Store 和 normalizer 都由调用方传入的实际目录定义，不读取 manifest、meta、hash、时间戳目录或 `latest_*` 指针。字段、维度和时序语义统一以 [contract.md](contract.md) 为准，完整复现步骤见 [documents/复现.md](documents/复现.md)。

## 推荐目录

以 RPM-P2 主实验为例：

```text
artifacts/
  source/
    HumanEva/...
    CMU/...
    M/...
  tasks/
    RPM-P2/
      train/shards/shard_*/
      test/shards/shard_*/
  normalizer/
    RPM-P2/
      pose_mean.pt
      pose_scale.pt
      tracker_mean.pt
      tracker_std.pt
      head_path_xz_mean.pt
      head_path_xz_std.pt
      head_height_mean.pt
      head_height_std.pt
  runs/
    RPM-P2/
```

目录名本身表示协议和 split。命令必须传入上面的实际目录，不会自动选择“最新产物”。

## 数据链路

先从 Unity Editor 导出 `body_fbx_rest.json`，再依次生成 source、RPM-P2 Task Store 和 RPM-P2 train normalizer：

```powershell
conda run -n diffusionposer5070 python -m data_converter.amass_to_realtime_pose `
  --body_fbx_rest_json ..\SIGGRAPH2024Unity\Assets\Projects\RealtimePose\Models\DiffusionPoser\body_fbx_rest.json `
  --amass_dir dataset/AMASS `
  --smpl_model_dir dataset/body_models `
  --output_dir artifacts/source `
  --target_fps 60

conda run -n diffusionposer5070 python -m data_loaders.generate_realtime_pose_tasks `
  --source_dir artifacts/source `
  --split_dir data_loaders/splits/RPM-P2 `
  --output_dir artifacts/tasks/RPM-P2 `
  --splits train test `
  --base_windows_per_source 20 `
  --shard_size 4096 `
  --seed 10

conda run -n diffusionposer5070 python -m data_loaders.compute_realtime_pose_normalizer `
  --task_dir artifacts/tasks/RPM-P2 `
  --split train `
  --output_dir artifacts/normalizer/RPM-P2
```

已有输出目录默认会报错。确认需要替换明确目录时，为相应命令添加 `--overwrite`。Task 生成先写同级临时目录，所有 shard 完成后才切换为正式目录。

## 训练

```powershell
conda run -n diffusionposer5070 python -m train.train_diffusionposer `
  --model_arch spatiotemporal_dit `
  --data_dir artifacts/tasks/RPM-P2 `
  --data_split train `
  --normalizer_dir artifacts/normalizer/RPM-P2 `
  --save_dir artifacts/runs/RPM-P2 `
  --run_name main `
  --scenario_weights 1 1 1 1 1
```

`--run_name` 仅属于训练 run；Task Store 和 normalizer 不使用它。训练 checkpoint 的 `latest_run` 仍保留：pipeline 使用 `--resume_latest`，直接调用训练入口时使用 `--resume_checkpoint latest`。它们不参与数据目录解析。

## 长序列评估

长序列评估直接读取 source 与 RPM split，不再构建复制版 eval set：

```powershell
conda run -n diffusionposer5070 python -m sample.evaluate_longseq_eval_set `
  --source_dir artifacts/source `
  --split_dir data_loaders/splits/RPM-P2 `
  --split test `
  --normalizer_dir artifacts/normalizer/RPM-P2 `
  --model_path artifacts/runs/RPM-P2/main/model000100000.pt `
  --output_dir output/RPM-P2/model100k
```

`--min_frames` 默认是 `0`；需要只评估超长序列时再显式提高门槛。

## 当前边界

- 旧 Task Store 不兼容，需要按新目录结构重新生成。
- 旧 normalizer 只要目录中有八个 `.pt` 文件即可直接复用；额外的旧 meta 文件会被忽略。
- 模型输入、Loss、144D Pose、Tracker 语义和评价指标没有因本次目录重构而变化。
- Unity/Sentis 导出仍是单独维护的旧接口，不作为当前 Python 主链路的运行保证。
