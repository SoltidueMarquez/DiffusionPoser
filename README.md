# DiffusionPoser

DiffusionPoser 论文复现项目索引。

当前仓库已经接通最小训练链路：随机稀疏传感器数据集、DiffusionPoser DiT 模型、Gaussian Diffusion 训练损失、checkpoint 保存和基础日志。真实 DiffusionPoser 数据集读取仍在 `data_loaders/get_data.py` 中预留接口。

## 快速入口

### 安装依赖

```powershell
pip install -r requirements.txt
```

`requirements.txt` 当前固定了 PyTorch `2.7.0+cu128`，如本机 CUDA 版本不同，需要按本机环境调整 PyTorch 安装项。

### Smoke Training

不传 `--data_dir` 时会使用随机数据跑通训练链路：

```powershell
python -m train.train_diffusionposer --save_dir runs/smoke --overwrite --num_steps 10 --log_interval 1 --save_interval 5
```

使用 GPU：

```powershell
python -m train.train_diffusionposer --cuda --device 0 --save_dir runs/smoke_cuda --overwrite --num_steps 10
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
│  ├─ get_data.py               # DataLoader 工厂；真实数据接口预留在这里
│  └─ smoke_dataset.py          # 随机稀疏传感器数据集，用于 smoke training
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
| `model/diffusionposer_dit.py` | DiffusionPoser DiT 模型。输入输出形状为 `[B, C, T]`，默认 `C=190`。 |
| `utils/parser_util.py` | 所有训练参数定义，包括数据、模型、扩散和训练配置。 |
| `utils/model_util.py` | `create_model_and_diffusion()` 和 `create_gaussian_diffusion()`。 |
| `data_loaders/get_data.py` | 数据入口。当前 `data_dir` 为空时使用 smoke dataset；传入真实数据目录会抛出 `NotImplementedError`。 |
| `data_loaders/smoke_dataset.py` | 随机生成 `x`、`valid_frame_mask`、`sensor_mask`，用于验证训练代码是否跑通。 |
| `data_converter/amass_to_x277.py` | 完整 AMASS 转换脚本，包含重采样、SMPL 前向、Unity 坐标转换、X277 特征构造、manifest 和可视化导出。 |

## 数据约定

训练 batch 当前预期字段：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `x` | `[B, C, T]` | 动作特征序列。 |
| `valid_frame_mask` | `[B, T]` | 有效帧标记。 |
| `sensor_mask` | `[B, C, T]` | 已观测传感器特征位置；`False` 表示训练时需要补全。 |

`TrainLoop.mask_manager()` 会把 `sensor_mask` 转成 diffusion 使用的 `inpaint_cond` 和 `y.mask`。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--save_dir` | 必填 | checkpoint、`args.json` 和日志输出目录。 |
| `--overwrite` | `False` | 允许复用已经存在的 `save_dir`。 |
| `--cuda` | `False` | 启用 CUDA。 |
| `--device` | `0` | CUDA 设备编号。 |
| `--batch_size` | `8` | batch size。 |
| `--input_feats` | `190` | 训练输入特征维度。 |
| `--seq_len` | `60` | smoke dataset 序列长度。 |
| `--mask_ratio` | `0.6` | 随机 mask 比例。 |
| `--layers` | `8` | Transformer 层数。 |
| `--heads` | `8` | attention heads。 |
| `--latent_dim` | `512` | Transformer hidden width。 |
| `--diffusion_steps` | `50` | diffusion 时间步数量。 |
| `--noise_schedule` | `cosine` | diffusion beta schedule。 |
| `--num_steps` | `10000` | 训练步数。 |
| `--resume_checkpoint` | 空 | 从 `model*.pt` 恢复模型参数。 |
| `--train_platform_type` | `NoPlatform` | 可选 `NoPlatform` 或 `TensorboardPlatform`。 |

## 当前待补齐

- `data_loaders/get_data.py` 还没有接入转换后的真实 DiffusionPoser 数据。
- `sample/` 目前只有占位文件，采样和评估入口尚未实现。
- 源码中部分中文注释存在编码异常，后续可以统一转为 UTF-8 后再清理。
