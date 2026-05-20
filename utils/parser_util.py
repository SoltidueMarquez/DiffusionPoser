from __future__ import annotations

import json
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

from data_loaders.sensor_masking import (
    LAST_FRAME_RECONSTRUCTION_SEQ_LEN,
    TASK_MODE_FULL_RECONSTRUCTION_CURRENT,
    TASK_MODES,
)


def train_args():
    parser = ArgumentParser(description="Train a DiffusionPoser current277 diffusion reconstruction model.")
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return parser.parse_args()


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"yes", "true", "t", "1", "y"}:
        return True
    if value in {"no", "false", "f", "0", "n"}:
        return False
    raise ValueError(f"无法解析布尔参数：{value}")


def parse_and_load_from_model(
    parser: ArgumentParser,
    argv: list[str] | None = None,
    ignore_keys: set[str] | None = None,
):
    """先解析命令行，再用 checkpoint 同目录的 `args.json` 补齐默认值。"""

    args = parser.parse_args(argv)
    model_path = getattr(args, "model_path", "")
    if not model_path:
        return args

    checkpoint_args = load_args_json(Path(model_path))
    if not checkpoint_args:
        return args

    ignore_keys = set(ignore_keys or set())
    for key, value in checkpoint_args.items():
        if key in ignore_keys or not hasattr(args, key):
            continue
        try:
            default_value = parser.get_default(key)
        except Exception:
            default_value = None
        if getattr(args, key) == default_value:
            setattr(args, key, value)
    return args


def add_base_options(parser: ArgumentParser):
    group = parser.add_argument_group("base")
    group.add_argument("--cuda", default=True, type=str2bool, help="是否优先使用 CUDA。")
    group.add_argument("--device", default=0, type=int, help="CUDA device index.")
    group.add_argument("--seed", default=10, type=int, help="Random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size.")


def add_data_options(parser: ArgumentParser):
    group = parser.add_argument_group("dataset")
    group.add_argument(
        "--task_mode",
        default=TASK_MODE_FULL_RECONSTRUCTION_CURRENT,
        choices=TASK_MODES,
        type=str,
        help="训练/采样任务语义。",
    )
    group.add_argument(
        "--data_dir",
        required=True,
        type=str,
        help="离线生成的 X277 传感器缺失任务目录。",
    )
    group.add_argument("--data_split", default="train", type=str, help="Dataset split name.")
    group.add_argument(
        "--normalizer_dir",
        default="dataset/meta_AMASS_current277_60hz",
        type=str,
        help="X277 normalizer/meta 目录；normalize_input 读取 mean.pt/std.pt，weighted_loss 读取 feature_w.pt。",
    )
    group.add_argument(
        "--normalize_input",
        default=True,
        type=str2bool,
        help="是否使用 normalizer_dir/mean.pt 和 std.pt 标准化 X277 输入特征。",
    )
    group.add_argument(
        "--preload_data",
        default=False,
        type=str2bool,
        help="是否在 Dataset 初始化时把所有 task npz 解压到内存；正式训练默认关闭。",
    )
    group.add_argument(
        "--input_feats",
        default=283,
        type=int,
        help="X277 特征 277 维 + 6 维传感器缺失标签。",
    )
    group.add_argument(
        "--seq_len",
        default=LAST_FRAME_RECONSTRUCTION_SEQ_LEN,
        type=int,
        help="训练/采样窗口帧数；当前任务固定为 11，即 10 帧历史 + 第 11 帧补全。",
    )
    group.add_argument("--num_workers", default=0, type=int, help="DataLoader worker 数量。")


def add_sampling_options(parser: ArgumentParser):
    group = parser.add_argument_group("sampling")
    group.add_argument("--model_path", required=True, type=str, help="Path to model#########.pt.")
    group.add_argument(
        "--output_dir",
        default="",
        type=str,
        help="测试结果输出目录；留空时根据 checkpoint 目录自动派生。",
    )
    group.add_argument(
        "--folder_path",
        default="",
        type=str,
        help="可选：只遍历测试集中某个子目录，用于缩小测试范围。",
    )
    group.add_argument(
        "--visualize_num",
        default=0,
        type=int,
        help="要可视化的样本数；0 表示不输出可视化，<0 表示全部样本。",
    )
    group.add_argument(
        "--visualize_fps",
        default=20.0,
        type=float,
        help="可视化视频帧率。",
    )
    group.add_argument(
        "--x277_fps",
        default=60.0,
        type=float,
        help="X277 特征解码时使用的原始运动帧率；visualize_fps 只控制导出视频帧率。",
    )
    group.add_argument(
        "--use_ema",
        default=True,
        type=str2bool,
        help="若 checkpoint 同目录存在 EMA 权重，则优先加载。",
    )


def add_model_options(parser: ArgumentParser):
    group = parser.add_argument_group("model")
    group.add_argument("--layers", default=8, type=int, help="Number of transformer layers.")
    group.add_argument("--heads", default=8, type=int, help="Number of attention heads.")
    group.add_argument("--latent_dim", default=512, type=int, help="Transformer hidden width.")
    group.add_argument("--dropout", default=0.0, type=float, help="Transformer dropout.")
    group.add_argument("--zero_init", action="store_true", help="Zero-init the final projection.")


def add_diffusion_options(parser: ArgumentParser):
    group = parser.add_argument_group("diffusion")
    group.add_argument("--noise_schedule", default="cosine", choices=["linear", "cosine"], type=str)
    group.add_argument("--diffusion_steps", default=50, type=int)
    group.add_argument("--sigma_small", default=True, type=str2bool)
    group.add_argument("--predict_xstart", default=1, type=int)
    group.add_argument("--ts_respace", default="", type=str, help="Optional DDIM respacing, e.g. ddim20.")


def add_training_options(parser: ArgumentParser):
    group = parser.add_argument_group("training")
    group.add_argument("--save_dir", required=True, type=str, help="Directory to save checkpoints and args.json.")
    group.add_argument(
        "--overwrite",
        default=True,
        action=BooleanOptionalAction,
        help="默认允许写入已有 save_dir；如需保护已有目录，请传 --no-overwrite。",
    )
    group.add_argument(
        "--train_platform_type",
        default="NoPlatform",
        choices=["NoPlatform", "TensorboardPlatform"],
        type=str,
    )
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="AdamW weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="达到该 step 后停止训练；0 表示关闭。")
    group.add_argument("--log_interval", default=1_000, type=int, help="Log every N steps.")
    group.add_argument("--save_interval", default=50_000, type=int, help="Save every N steps.")
    group.add_argument(
        "--checkpoint_max_keep",
        default=0,
        type=int,
        help="最多保留最近 N 组 model/opt/ema checkpoint；0 表示不清理。",
    )
    group.add_argument("--num_steps", default=1_000_000, type=int, help="Total training steps.")
    group.add_argument(
        "--resume_checkpoint",
        default="",
        type=str,
        help="Path to model#########.pt, or latest/auto to resume from the newest model*.pt in save_dir.",
    )
    group.add_argument("--gradient_clip", action="store_true", help="Clip grad-norm at 1.0.")
    group.add_argument("--weighted_loss", action="store_true", help="从 normalizer_dir/feature_w_file 读取逐维 loss 权重。")
    group.add_argument("--feature_w_file", default="feature_w.pt", type=str, help="Feature weight file name.")
    group.add_argument("--snr_gamma", default=0.0, type=float, help="SNR loss weighting gamma; 0 disables.")
    group.add_argument("--l1_loss", action="store_true", help="Use L1 loss instead of L2.")
    group.add_argument("--model_ema", action="store_true", help="Track EMA of model parameters.")
    group.add_argument("--model_ema_steps", type=int, default=10, help="How often to update EMA.")
    group.add_argument("--model_ema_decay", type=float, default=0.995, help="EMA decay.")
    group.add_argument("--model_ema_update_after", type=int, default=5000, help="Start EMA updates after N steps.")
    group.add_argument(
        "--eval_during_training",
        action="store_true",
        help="Reserved flag; current training loop logs a warning and skips in-loop evaluation.",
    )


def load_args_json(model_path: Path) -> dict:
    """读取 checkpoint 同目录下的 `args.json`。"""

    args_path = model_path.with_name("args.json")
    if not args_path.exists():
        return {}
    with args_path.open("r", encoding="utf-8") as file:
        return json.load(file)
