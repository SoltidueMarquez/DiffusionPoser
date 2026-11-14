# LM 与 ΠGDM 场景引导复现

所有必要输入、固定参数和入口均随当前仓库提供。无需 FLUIDUnity 工程，也不读取旧的
`output/scene_guidance_*` 实验报告。模型使用已经跟踪的 `artifacts/unity_demo/` 权重、
checkpoint 参数与 normalizer；不需要训练或下载 AMASS/SMPL 数据。

## 环境与输入检查

使用项目 `requirements.txt` 对应环境。正式采样/计时需要支持当前 PyTorch 的 CUDA GPU；
原实验为 RTX 5070、PyTorch 2.7.0+cu128、FP32、batch 1、5 步 DDIM。
以下命令在仓库根目录执行，输入校验无需 GPU：

```powershell
conda run -n diffusionposer5070 python -m eval.evaluate_scene_guidance --check_inputs
conda run -n diffusionposer5070 python -m eval.evaluate_scene_guidance --config experiments/scene_guidance/sequence.json --check_inputs
```

`inputs_manifest.json` 固定模型和输入的 SHA256。JSON/Python 文本计算哈希时统一 LF 换行，
避免 Windows/Linux checkout 差异。缺少文件或仓库内固定输入发生变化时会直接报错。
需要换模型或输入时，请使用另一个文件路径并通过配置或 CLI 指定；实际使用文件的哈希
仍会写入本次 `experiment.json`。

## 固定快照实验

```powershell
conda run -n diffusionposer5070 python -m eval.evaluate_scene_guidance --config experiments/scene_guidance/snapshot.json
```

- 65 个快照，均匀保留 20 个调参帧，测量剩余 45 帧；默认不重新调参。
- 无修正、PostEdit、LM、ΠGDM 共用固定条件、Predictor 输出和按帧号/种子生成的噪声。
- 种子 10/20/30，预热 20 次，每组重复计时 5 次。
- 输出到 `output/scene_guidance_snapshot/`：`report.html`、逐次 JSONL、逐帧/汇总 CSV、
  姿态 NPZ、对照图、输入与代码来源、基线一致性检查。

## 长序列闭环实验

```powershell
conda run -n diffusionposer5070 python -m eval.evaluate_scene_guidance --config experiments/scene_guidance/sequence.json
```

- 构造的坐下/扶手接近 Tracker 轨迹为 1108 帧、约 36.9 秒、30 Hz，没有全身姿态真值。
- 前 11 帧用于初始化，连续生成 1097 帧，其中前 30 帧为启动段，计分 1067 帧。
- PostEdit、LM、ΠGDM 各自反馈自己的上一帧输出；种子 10/20，每帧采样一次、历史更新一次。
- 世界扶手从快照恢复，并验证与 Tracker 轨迹对齐；忽略 Hip Tracker，保留有效足部 Tracker。
- 输出到 `output/scene_guidance_sequence/`：`report.html`、逐帧指标、计时 JSONL、汇总 CSV/JSON、
  各方法各种子的姿态/世界关节/Tracker/时间戳 NPZ。

两份配置使用相同的观测尺度 0.5 mm、安全间隙 2 mm。PostEdit 最多 5 次更新、步长
0.000074；LM 最多 5 次更新、初始阻尼 0.001；ΠGDM 引导强度 10、协方差倍率 0.1。
算法实现保留已有实验语义：LM 编辑干净残差，ΠGDM 通过 DiT VJP 引导 DDIM 状态。
ΠGDM 为胶囊—OBB SDF 的局部线性适配，不代表完整 InPose 系统。

## 短程验证与参数覆盖

```powershell
conda run -n diffusionposer5070 python -m eval.evaluate_scene_guidance --methods lm pigdm --max_frames 3 --seeds 10 --warmup 1 --repeats 1 --output_dir output/scene_guidance_snapshot_smoke
conda run -n diffusionposer5070 python -m eval.evaluate_scene_guidance --config experiments/scene_guidance/sequence.json --methods lm pigdm --max_frames 3 --seeds 10 --warmup 1 --output_dir output/scene_guidance_sequence_smoke
conda run -n diffusionposer5070 pytest tests/smoke/eval
```

CLI 覆盖配置。相对路径以仓库根目录为基准；绝对路径可用于其他输入和输出位置。
`--max_frames 0`（默认）表示完整运行；正数限制正式计分帧，长序列仍先完成初始化与启动段。
短程结果会标记 `short_run=true`，不能作为正式效率结论。快照协议自动保留无修正基线。
结果目录必须为空，重复运行时用 `--output_dir` 指定新的目录。

可加 `--tune` 在快照的调参集上扫描原有参数网格，按每种方法的速度—平均穿透 Pareto 点
筛选后再测测试集；测试帧不参与筛选。长序列只接受固定配置。

## 输入与指标范围

`inputs/` 中保存四个文件：

| 文件 | 来源和用途 |
| --- | --- |
| `captured_replay_inputs.json` | 原 `output/collision_compute/` 的 65 帧；仅保留实验读取的输入字段，数值不变 |
| `chair_probe.json` | 原 `output/collision_comparison/before/` 的构造 Tracker 轨迹 |
| `body_fbx_rest.json` | 从 FLUIDUnity 固定下来的骨架 rest 数据 |
| `collision_profile.json` | 从 FLUIDUnity 固定下来的胶囊碰撞标定 |

几何约束使用 8 个手臂/手掌胶囊与 2 个扶手 OBB。穿透帧判定阈值为 0.001 mm，安全间隙违例
单独统计；Tracker 误差为有效设备位置误差，不是全身 MPJPE。
`sampler_ms` 包含条件准备、5 次 DiT、引导和最终投影；不包含 Predictor/IK、文件读取或绘图。
几何 CUDA Graph 初始化开销单独记录，逐帧准备计入引导时间。所有方法共享执行缓存，
但姿态变化后重新计算 SDF/Jacobian。长序列按帧轮换方法顺序，减小温度/负载漂移影响。
这些输入只覆盖同一构造场景，硬件负载也会影响耗时。

核心代码位于 `eval/scene_guidance.py`、`eval/scene_guidance_cache.py`；
`scene_guidance_setup.py` 管理共享初始化和校验，`scene_guidance_benchmark.py` 组织快照实验，
`scene_guidance_sequence.py` 组织闭环实验。新结果继续写入被 Git 忽略的 `output/`。

## 整理后的验证记录

2026-11-14，在上述 RTX 5070 环境完成：

- `tests/smoke/eval`：31 项通过，包含固定配置、调参模式和长序列独立入口的 GPU 短程运行。
- runtime、碰撞后处理和 temporal attention 相关 smoke test：52 项通过。
- 完整快照：2700 次采样零失败，保存的 540 个姿态与原固定参数实验逐值一致。
- 完整长序列：两种子、三方法共 6402 次计分采样零失败；全部 6582 个输出姿态及其世界关节、
  Tracker、时间戳与原长序列实验逐值一致。耗时和显存按本次进程重新测量。

原结果仅用于这次迁移后的数值对照，不是新入口的依赖；新克隆无需取得这些旧结果。
