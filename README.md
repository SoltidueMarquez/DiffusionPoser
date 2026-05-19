# DiffusionPoser

DiffusionPoser 论文复现项目索引。

当前仓库已经接通训练链路：离线 X277 传感器缺失任务、DiffusionPoser DiT 模型、Gaussian Diffusion 训练损失、checkpoint 保存和基础日志。

## 快速入口

### 安装依赖

```powershell
pip install -r requirements.txt
```

`requirements.txt` 当前固定了 PyTorch `2.7.0+cu128`，如本机 CUDA 版本不同，需要按本机环境调整 PyTorch 安装项。

### X277 缺失任务生成

先把 `dataset/AMASS_x277_60hz` 中的 `x: [T, 277]` 转成固定长度缺失任务。默认每个源文件生成 4 条任务，序列长度为 11：

```powershell
python -m data_loaders.generate_x277_missing_tasks --source_dir dataset/AMASS_x277_60hz --output_dir dataset/AMASS_x277_60hz_missing_tasks --split_dir data_loaders/splits --splits train test --seq_len 11 --samples_per_file 4 --seed 10 --overwrite
```

`data_loaders/splits/train.txt` 和 `data_loaders/splits/test.txt` 已经从 StableMotion 参考项目复制到本项目，可直接用于 `stablemotion_split_key` 过滤。

```powershell
python -m data_loaders.generate_x277_missing_tasks --source_dir dataset/AMASS_x277_60hz --output_dir dataset/AMASS_x277_60hz_missing_tasks --split_dir data_loaders/splits --splits train test --seq_len 11 --overwrite
```

生成目录形如 `dataset/AMASS_x277_60hz_missing_tasks/train/manifest.jsonl` 和 `tasks/*.npz`。训练时直接把根目录传给 `--data_dir`：

当前 `full_reconstruction_current` 任务固定为 DiffusionPoser 风格的单帧在线补全：每条固定窗口固定 11 帧，前 10 帧作为历史条件，第 11 帧标记为重建目标。第 11 帧会补全 body/root/contact，以及该帧离线 tracker 的 position/rotation。

```powershell
python -m train.train_diffusionposer --data_dir dataset/AMASS_x277_60hz_missing_tasks --data_split train --save_dir runs/x277_train --overwrite
```

### AMASS 转换脚本

`data_converter/amass_to_x277.py` 用于把 AMASS SMPL/SMPL-H 动作转换成项目使用的 X277 帧特征：

```powershell
python data_converter/amass_to_x277.py --amass_dir dataset/AMASS --smpl_model_dir dataset/body_models --output_dir dataset/processed/amass_x277_60hz
```

AMASS、SMPL、SMPL-H 通常需要登录和许可证确认，不能直接从仓库自动下载。

## 目录索引

```text
DiffusionPoser/
├─ README.md
├─ requirements.txt
├─ train/
│  ├─ train_diffusionposer.py   # 训练主入口：解析参数、创建数据/模型/扩散器、启动训练循环
│  ├─ training_loop.py          # 训练循环：mask 构造、loss 计算、优化器、EMA、checkpoint
│  └─ train_platforms.py        # 日志平台：NoPlatform、TensorboardPlatform、可选 ClearML
├─ model/
│  └─ diffusionposer_dit.py     # DiffusionPoserDiT：基于 TransformerEncoder 的轻量 DiT 骨架
├─ diffusion/
│  ├─ gaussian_diffusion.py     # Gaussian diffusion 核心实现与训练/采样损失
│  ├─ respace.py                # 扩散时间步重采样与 SpacedDiffusion
│  ├─ resample.py               # schedule sampler
│  ├─ losses.py                 # SNR、KL、离散高斯似然等损失工具
│  ├─ nn.py                     # diffusion 通用神经网络工具函数
│  └─ logger.py                 # key-value 日志系统
├─ data_loaders/
│  ├─ get_data.py               # DataLoader 工厂：读取真实 X277 缺失任务
│  ├─ generate_x277_missing_tasks.py # 离线生成 train/test 传感器缺失任务
│  ├─ sensor_masking.py         # 6 个传感器到 X277 tracker 维度的映射与 mask 生成
│  └─ x277_dataset.py           # 读取离线任务并输出 [B, 283, T] batch
├─ data_converter/
│  └─ amass_to_x277.py          # AMASS/SMPL-H 到 X277 特征转换和可视化导出
├─ utils/
│  ├─ parser_util.py            # 训练命令行参数
│  ├─ model_util.py             # 模型和 diffusion 创建函数
│  ├─ dist_util.py              # 设备选择和分布式训练占位工具
│  └─ fixseed.py                # 随机种子固定
└─ sample/
   └─ __init__.py               # 采样模块占位
```

## 关键文件说明

| 文件 | 作用 |
| --- | --- |
| `train/train_diffusionposer.py` | 训练主入口。调用 `train_args()` 读取参数，创建 DataLoader、模型和 diffusion 对象，并启动 `TrainLoop`。 |
| `train/training_loop.py` | 最小训练闭环。负责构造 inpainting mask、调用 `diffusion.training_losses()`、反向传播、保存 `model*.pt` 和 `opt*.pt`。 |
| `model/diffusionposer_dit.py` | DiffusionPoser DiT 模型。输入输出形状为 `[B, C, T]`，默认 `C=283`。 |
| `utils/parser_util.py` | 所有训练参数定义，包括数据、模型、扩散和训练配置。 |
| `utils/model_util.py` | `create_model_and_diffusion()` 和 `create_gaussian_diffusion()`。 |
| `data_loaders/get_data.py` | 数据入口。训练必须提供离线任务目录 `--data_dir`。 |
| `data_loaders/generate_x277_missing_tasks.py` | 离线生成固定长度 X277 传感器缺失任务，可按 split 文件生成 train/test。 |
| `data_loaders/sensor_masking.py` | 定义 6 个传感器顺序、tracker 维度映射、缺失标签和 inpaint mask 生成。 |
| `data_loaders/x277_dataset.py` | 读取任务 manifest 和源 X277 `.npz`，输出固定长度训练 batch。 |
| `data_converter/amass_to_x277.py` | 完整 AMASS 转换脚本，包含重采样、SMPL 前向、Unity 坐标转换、X277 特征构造、manifest 和可视化导出。 |

## 数据约定

训练 batch 当前预期字段：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `x` | `[B, 283, T]` | 前 277 维为 X277，后 6 维为传感器缺失标签。 |
| `valid_frame_mask` | `[B, T]` | 有效帧标记。 |
| `attention_mask` | `[B, T]` | 与 `valid_frame_mask` 同义，提供给模型忽略 padding 帧。 |
| `sensor_missing_labels` | `[B, 6, T]` | 6 个 tracker 在每帧是否缺失。 |
| `inpaint_mask` | `[B, 283, T]` | `True` 表示该位置需要扩散补全并参与 loss。 |

`TrainLoop.mask_manager()` 使用离线生成的 `inpaint_mask`，缺少该字段会直接报错，避免误跑随机或无监督 mask。`X277MissingTaskDataset` 只接受原生 11 帧 materialized task：前 10 帧是历史条件，第 11 帧是唯一补全目标。已有 100/150 帧旧任务不会被自动裁剪，必须用当前生成脚本重新生成。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--save_dir` | 必填 | checkpoint、`args.json` 和日志输出目录。 |
| `--overwrite` | `False` | 允许复用已经存在的 `save_dir`。 |
| `--cuda` | `False` | 启用 CUDA。 |
| `--device` | `0` | CUDA 设备编号。 |
| `--batch_size` | `8` | batch size。 |
| `--input_feats` | `283` | 训练输入特征维度。 |
| `--seq_len` | `11` | 固定训练帧数；10 帧历史 + 第 11 帧补全。 |
| `--num_workers` | `0` | DataLoader worker 数量。 |
| `--layers` | `8` | Transformer 层数。 |
| `--heads` | `8` | attention heads。 |
| `--latent_dim` | `512` | Transformer hidden width。 |
| `--diffusion_steps` | `50` | diffusion 时间步数量。 |
| `--noise_schedule` | `cosine` | diffusion beta schedule。 |
| `--num_steps` | `10000` | 训练步数。 |
| `--resume_checkpoint` | 空 | 从 `model*.pt` 恢复模型参数。 |
| `--train_platform_type` | `NoPlatform` | 可选 `NoPlatform` 或 `TensorboardPlatform`。 |

## 当前待补齐

- `sample/` 目前只有占位文件，采样和评估入口尚未实现。
- 源码中部分中文注释存在编码异常，后续可以统一转为 UTF-8 后再清理。
