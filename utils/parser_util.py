from argparse import ArgumentParser


def train_args():
    parser = ArgumentParser(description="Train a DiffusionPoser fix-only diffusion reconstruction model.")
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


def add_base_options(parser: ArgumentParser):
    group = parser.add_argument_group("base")
    group.add_argument("--cuda", default=True, type=str2bool, help="是否优先使用 CUDA。")
    group.add_argument("--device", default=0, type=int, help="CUDA device index.")
    group.add_argument("--seed", default=10, type=int, help="Random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size.")


def add_data_options(parser: ArgumentParser):
    group = parser.add_argument_group("dataset")
    group.add_argument(
        "--data_dir",
        required=True,
        type=str,
        help="离线生成的 X277 传感器缺失任务目录。",
    )
    group.add_argument("--data_split", default="train", type=str, help="Dataset split name.")
    group.add_argument(
        "--normalizer_dir",
        default="dataset/meta_AMASS_x277_60hz",
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
        default=100,
        type=int,
        help="训练序列固定裁剪/padding 后的帧数。",
    )
    group.add_argument("--num_workers", default=0, type=int, help="DataLoader worker 数量。")


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
    group.add_argument("--overwrite", action="store_true", help="Allow writing into an existing save_dir.")
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
    group.add_argument("--resume_checkpoint", default="", type=str, help="Path to model#########.pt.")
    group.add_argument("--gradient_clip", action="store_true", help="Clip grad-norm at 1.0.")
    group.add_argument("--weighted_loss", action="store_true", help="从 normalizer_dir/feature_w_file 读取逐维 loss 权重。")
    group.add_argument("--feature_w_file", default="feature_w.pt", type=str, help="Feature weight file name.")
    group.add_argument("--snr_gamma", default=0.0, type=float, help="SNR loss weighting gamma; 0 disables.")
    group.add_argument("--l1_loss", action="store_true", help="Use L1 loss instead of L2.")
    group.add_argument("--model_ema", action="store_true", help="Track EMA of model parameters.")
    group.add_argument("--model_ema_steps", type=int, default=10, help="How often to update EMA.")
    group.add_argument("--model_ema_decay", type=float, default=0.995, help="EMA decay.")
    group.add_argument("--model_ema_update_after", type=int, default=5000, help="Start EMA updates after N steps.")
    group.add_argument("--eval_during_training", action="store_true", help="Run evaluation while training.")
