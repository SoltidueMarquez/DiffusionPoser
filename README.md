# DiffusionPoser RealtimePose

当前 Python 主链路由两个独立训练的模型组成：Predictor Transformer 根据完整的过去
10 帧 Pose 与 Head/双手 Tracker 预测当前及未来 10 帧；单帧 DiT 使用 Predictor
先验和当前可用 Tracker 执行 IK-Inpainting，只恢复当前 144D Pose。字段、维度和
参考系语义以 [contract.md](contract.md) 为准。

## 产物目录

```text
artifacts/
  source/                       # 转换后的 RPM P2 30Hz source
  predictor/
    tasks/train/shards/...      # 单帧 DiT Task Store
    tasks/test/shards/...
    normalizer/                 # Pose、Tracker、Predictor sparse 统计
    runs/predictor/             # 单阶段 Predictor
    runs/dit/                   # 冻结 Predictor 条件下的单帧 DiT
    output/longseq_report.json
```

程序只使用命令行给出的明确目录。数据、normalizer 和 checkpoint 应按当前契约
重新生成。

## 数据准备

```powershell
conda run -n diffusionposer5070 python -m data_converter.amass_to_realtime_pose `
  --amass_dir dataset/AMASS `
  --smpl_model_dir dataset/body_models `
  --body_fbx_rest_json path/to/body_fbx_rest.json `
  --output_dir artifacts/source

conda run -n diffusionposer5070 python -m data_loaders.generate_realtime_pose_tasks `
  --source_dir artifacts/source `
  --split_dir data_loaders/splits/RPM-P2 `
  --output_dir artifacts/predictor/tasks `
  --splits train test

conda run -n diffusionposer5070 python -m data_loaders.compute_realtime_pose_normalizer `
  --task_dir artifacts/predictor/tasks `
  --output_dir artifacts/predictor/normalizer
```

## 单阶段 Predictor 训练

```powershell
conda run -n diffusionposer5070 python -m train.train_realtime_pose_predictor `
  --source_dir artifacts/source `
  --split_dir data_loaders/splits/RPM-P2 `
  --normalizer_dir artifacts/predictor/normalizer `
  --save_dir artifacts/predictor/runs/predictor `
  --num_steps 100000 `
  --precision bf16 --batch_size 512 --num_workers 4 `
  --checkpoint_max_keep 3
```

每个 batch 均匀采样 0～30 个 closed-loop step，并只回填 Predictor horizon 0。
在 30Hz 下，最多 30 个 closed-loop step 对应 1 秒部署误差累积。
默认 AdamW 为 `lr=3e-4`、`weight_decay=1e-4`，完成 50,000 步后学习率除以 30。
训练保存 model、EMA、optimizer 和 `args.json`；`model_latest.pt` 始终写入最新
EMA 推理权重。
`checkpoint_max_keep=3` 表示仅保留最近 3 组编号 model/EMA/optimizer，
`model_latest.pt` 不参与清理；设为 0 可保留全部编号 checkpoint。
恢复时可加 `--resume_checkpoint latest`，它会读取最近的带步号 checkpoint。

Predictor Dataset 会在训练启动时把 train 的最小训练字段预载到主进程内存，
之后 batch 不再读取 source 文件或重复执行 FK。常驻量约为每个 source 帧 796
bytes，另加每条 source 288 bytes 的骨架 offsets；启动日志会打印实际常驻 GiB
和预载耗时。

## 单帧 DiT 训练

可先基于冻结 Predictor 校准 IK confidence：

```powershell
conda run -n diffusionposer5070 python -m eval.calibrate_realtime_pose_ik `
  --data_dir artifacts/predictor/tasks `
  --normalizer_dir artifacts/predictor/normalizer `
  --predictor_model_path artifacts/predictor/runs/predictor/model_latest.pt `
  --output artifacts/predictor/ik_calibration.json
```

再训练单帧 DiT：

```powershell
conda run -n diffusionposer5070 python -m train.train_diffusionposer `
  --model_arch current_dit `
  --data_dir artifacts/predictor/tasks `
  --normalizer_dir artifacts/predictor/normalizer `
  --predictor_model_path artifacts/predictor/runs/predictor/model_latest.pt `
  --ik_calibration_path artifacts/predictor/ik_calibration.json `
  --save_dir artifacts/predictor/runs/dit `
  --latent_dim 384 --layers 6 `
  --log_interval 10 `
  --precision bf16 --batch_size 512 --num_workers 4 `
  --gradient_clip
```

Predictor 在 DiT 训练中始终冻结，不进入 DiT optimizer 或 checkpoint。
Predictor 与 DiT 共享 Task Store 中的干净 10 帧历史；不再添加人工历史扰动。
BF16 只覆盖 Predictor/DiT forward，模型输出会转回 FP32 后再计算 IK、diffusion 与
几何 loss；CUDA FP32 matmul 同时启用 TF32。

## 长序列评估

两种评估都从 source 帧 11 开始闭环运行，并按 RPM P2 跳过第一秒，直到
source 帧 30 才开始统计正式指标。`--max_frames` 只限制预热后的计分帧数。
正式报告使用前 22 个 SMPL 关节，包含 MPJRE、MPJPE、MPJVE、预测 Jitter 和
GT Jitter；Predictor horizon 与前后 30 个生成帧误差作为额外诊断保留。

只评估 Predictor、无需加载 DiT：

```powershell
conda run -n diffusionposer5070 python -m eval.evaluate_realtime_pose_predictor `
  --source_dir artifacts/source `
  --split_dir data_loaders/splits/RPM-P2 `
  --normalizer_dir artifacts/predictor/normalizer `
  --predictor_model_path artifacts/predictor/runs/predictor/model_latest.pt `
  --output_json artifacts/predictor/output/predictor_longseq_report.json
```

评估 Predictor + DiT：

```powershell
conda run -n diffusionposer5070 python -m sample.evaluate_longseq_eval_set `
  --source_dir artifacts/source `
  --split_dir data_loaders/splits/RPM-P2 `
  --split test `
  --normalizer_dir artifacts/predictor/normalizer `
  --predictor_model_path artifacts/predictor/runs/predictor/model_latest.pt `
  --dit_model_path artifacts/predictor/runs/dit/<run>/model000100000.pt `
  --ts_respace ddim5 `
  --output_json artifacts/predictor/output/longseq_report.json
```

基础报告可额外传入 `--tracker_configs core_only all_six`，只运行训练见过的
核心三点与全部六点；不传该参数时默认运行全部 8 种静态配置。

评估使用 11 个 source 帧做 GT burn-in。独立入口报告 Predictor current、horizon
和 30 帧前后闭环指标；组合入口只回填 deployed Pose，覆盖 8 种静态 Tracker
组合，并报告一次 Predictor-only 基线以及各配置的 DiT raw/deployed 指标。

## 验证

```powershell
conda run -n diffusionposer5070 pytest tests/smoke
```

本轮不包含 Tracker 断线/重连、PCAF、future tracker 和 Unity/Sentis 双模型导出。
