from argparse import ArgumentParser


def train_args():
    parser = ArgumentParser(description="Train a DiffusionPoser diffusion reconstruction model.")
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return parser.parse_args()


def add_base_options(parser: ArgumentParser):
    group = parser.add_argument_group("base")
    group.add_argument("--cuda", action="store_true", help="Use CUDA when available.")
    group.add_argument("--device", default=0, type=int, help="CUDA device index.")
    group.add_argument("--seed", default=10, type=int, help="Random seed.")
    group.add_argument("--batch_size", default=8, type=int, help="Batch size.")


def add_data_options(parser: ArgumentParser):
    group = parser.add_argument_group("dataset")
    group.add_argument("--data_dir", default="", type=str, help="预处理后的 DiffusionPoser 数据目录。")
    group.add_argument("--data_split", default="train", type=str, help="Dataset split name.")
    group.add_argument("--input_feats", default=190, type=int, help="每一帧的动作特征维度，DiffusionPoser 默认 190。")
    group.add_argument("--seq_len", default=60, type=int, help="Smoke dataset 的序列长度；真实数据接入后由数据集决定。")
    group.add_argument("--smoke_num_batches", default=2, type=int, help="未提供 data_dir 时用于 smoke training 的 batch 数。")


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
    group.add_argument("--sigma_small", action="store_true", default=True)
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
    group.add_argument("--lr", default=1e-4, type=float)
    group.add_argument("--weight_decay", default=0.0, type=float)
    group.add_argument("--lr_anneal_steps", default=0, type=int)
    group.add_argument("--log_interval", default=100, type=int)
    group.add_argument("--save_interval", default=1_000, type=int)
    group.add_argument("--num_steps", default=10_000, type=int)
    group.add_argument("--resume_checkpoint", default="", type=str)
    group.add_argument("--gradient_clip", action="store_true")
    group.add_argument("--snr_gamma", default=0.0, type=float)
    group.add_argument("--l1_loss", action="store_true")
    group.add_argument("--mask_ratio", default=0.6, type=float, help="随机训练时每个有效帧中待补全特征比例。")
    group.add_argument("--model_ema", action="store_true")
    group.add_argument("--model_ema_steps", type=int, default=10)
    group.add_argument("--model_ema_decay", type=float, default=0.995)
    group.add_argument("--model_ema_update_after", type=int, default=5000)
    group.add_argument("--eval_during_training", action="store_true")
