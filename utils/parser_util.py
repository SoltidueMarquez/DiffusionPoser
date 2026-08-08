from __future__ import annotations

import json
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

from data_loaders.realtime_pose_config import RealtimePoseLossWeights
from data_loaders.sensor_masking import (
    DEFAULT_SCENARIO_WEIGHTS,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_DIM,
    REALTIME_POSE_WINDOW_LENGTH,
    TASK_MODE_REALTIME_POSE,
    TASK_MODES,
)


_LOSS_DEFAULTS = RealtimePoseLossWeights()


def build_train_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train the 144D realtime pose reconstruction model.")
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return parser


def train_args(argv: list[str] | None = None):
    return build_train_arg_parser().parse_args(argv)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"yes", "true", "t", "1", "y"}:
        return True
    if value in {"no", "false", "f", "0", "n"}:
        return False
    raise ValueError(f"无法解析布尔参数：{value}")


def parse_and_load_from_model(parser: ArgumentParser, argv: list[str] | None = None, ignore_keys: set[str] | None = None):
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
    group.add_argument("--cuda", default=True, type=str2bool)
    group.add_argument("--device", default=0, type=int)
    group.add_argument("--seed", default=10, type=int)
    group.add_argument("--batch_size", default=64, type=int)


def add_data_options(parser: ArgumentParser):
    group = parser.add_argument_group("dataset")
    group.add_argument("--task_mode", default=TASK_MODE_REALTIME_POSE, choices=TASK_MODES, type=str)
    group.add_argument("--data_dir", required=True, type=str, help="realtime_pose materialized task 目录。")
    group.add_argument("--data_split", default="train", type=str)
    group.add_argument(
        "--normalizer_dir",
        default="dataset/meta_AMASS_realtime_pose_144d_pelvis_residual_root_y0_stationary5_60hz",
        type=str,
    )
    group.add_argument("--normalize_input", default=True, type=str2bool)
    group.add_argument("--input_feats", default=REALTIME_POSE_TARGET_DIM, type=int)
    group.add_argument("--seq_len", default=REALTIME_POSE_SEQ_LEN, type=int)
    group.add_argument("--num_workers", default=0, type=int)
    group.add_argument(
        "--cold_start_prob",
        default=0.1,
        type=float,
        help="训练时使用 0～59 帧部分历史的概率；其余样本保留完整 60 帧历史。",
    )
    group.add_argument(
        "--scenario_weights",
        nargs=5,
        default=list(DEFAULT_SCENARIO_WEIGHTS),
        type=float,
        metavar=("SIX", "THREE", "3TO6", "6TO3", "TWO_DROP_RECONNECT"),
    )


def add_sampling_options(parser: ArgumentParser):
    group = parser.add_argument_group("sampling")
    group.add_argument("--model_path", required=True, type=str)
    group.add_argument("--output_dir", default="", type=str)
    group.add_argument("--folder_path", default="", type=str)
    group.add_argument("--visualize_num", default=0, type=int)
    group.add_argument("--visualize_fps", default=20.0, type=float)
    group.add_argument("--source_fps", default=60.0, type=float)
    group.add_argument("--use_ema", default=True, type=str2bool)
    group.add_argument(
        "--projected_ddim_mode",
        default="all_steps",
        choices=["all_steps", "late_steps", "final_step"],
    )
    group.add_argument("--projected_ddim_late_steps", default=5, type=int)


def add_model_options(parser: ArgumentParser):
    group = parser.add_argument_group("model")
    group.add_argument("--layers", default=8, type=int)
    group.add_argument("--heads", default=8, type=int)
    group.add_argument("--latent_dim", default=512, type=int)
    group.add_argument("--dropout", default=0.0, type=float)
    group.add_argument("--zero_init", action="store_true")
    group.add_argument("--max_seq_len", default=REALTIME_POSE_WINDOW_LENGTH, type=int)
    group.add_argument(
        "--model_arch",
        default="spatiotemporal_dit",
        choices=["spatiotemporal_dit"],
        type=str,
    )


def add_diffusion_options(parser: ArgumentParser):
    group = parser.add_argument_group("diffusion")
    group.add_argument("--noise_schedule", default="cosine", choices=["linear", "cosine"], type=str)
    group.add_argument("--diffusion_steps", default=50, type=int)
    group.add_argument("--sigma_small", default=True, type=str2bool)
    group.add_argument("--predict_xstart", default=1, type=int)
    group.add_argument("--ts_respace", default="", type=str)


def add_training_options(parser: ArgumentParser):
    group = parser.add_argument_group("training")
    group.add_argument("--save_dir", required=True, type=str)
    group.add_argument("--run_name", default="auto", type=str)
    group.add_argument("--overwrite", default=False, action=BooleanOptionalAction)
    group.add_argument("--train_platform_type", default="NoPlatform", choices=["NoPlatform", "TensorboardPlatform"], type=str)
    group.add_argument("--lr", default=1e-4, type=float)
    group.add_argument("--weight_decay", default=0.0, type=float)
    group.add_argument("--lr_anneal_steps", default=0, type=int)
    group.add_argument("--log_interval", default=1_000, type=int)
    group.add_argument("--save_interval", default=5_000, type=int)
    group.add_argument("--checkpoint_max_keep", default=0, type=int)
    group.add_argument("--num_steps", default=1_000_000, type=int)
    group.add_argument("--resume_checkpoint", default="", type=str)
    group.add_argument("--gradient_clip", action="store_true")
    group.add_argument("--weighted_loss", action="store_true")
    group.add_argument("--feature_w_file", default="feature_w.pt", type=str)
    group.add_argument("--snr_gamma", default=0.0, type=float)
    group.add_argument("--l1_loss", action="store_true")
    group.add_argument("--aux_loss_weight", default=1.0, type=float)
    group.add_argument("--rotation_loss_weight", default=_LOSS_DEFAULTS.global_rotation, type=float)
    group.add_argument("--fk_loss_weight", default=_LOSS_DEFAULTS.fk, type=float)
    group.add_argument("--local_rot_loss_weight", default=_LOSS_DEFAULTS.local_rotation, type=float)
    group.add_argument("--tracker_pos_loss_weight", default=_LOSS_DEFAULTS.tracker_position, type=float)
    group.add_argument("--tracker_pos_huber_beta", default=0.05, type=float)
    group.add_argument("--tracker_rot_loss_weight", default=_LOSS_DEFAULTS.tracker_rotation, type=float)
    group.add_argument("--root_loss_weight", default=_LOSS_DEFAULTS.root, type=float)
    group.add_argument(
        "--head_ref_joint_distance_loss_weight",
        default=_LOSS_DEFAULTS.head_ref_joint_distance,
        type=float,
    )
    group.add_argument("--head_to_root_xz_loss_weight", default=_LOSS_DEFAULTS.head_to_root_xz, type=float)
    group.add_argument("--future_leg_loss_weight", default=_LOSS_DEFAULTS.future_leg, type=float)
    group.add_argument("--contact_loss_weight", default=_LOSS_DEFAULTS.contact, type=float)
    group.add_argument("--contact_slide_loss_weight", default=_LOSS_DEFAULTS.contact_slide, type=float)
    group.add_argument("--history_noise_prob", default=0.8, type=float)
    group.add_argument("--history_noise_min_deg", default=2.0, type=float)
    group.add_argument("--history_noise_max_deg", default=10.0, type=float)
    group.add_argument("--history_noise_temporal_rho", default=0.95, type=float)
    group.add_argument("--history_noise_region_ratio", default=0.75, type=float)
    group.add_argument("--history_noise_joint_ratio", default=0.25, type=float)
    group.add_argument("--model_ema", default=True, action=BooleanOptionalAction)
    group.add_argument("--model_ema_steps", type=int, default=10)
    group.add_argument("--model_ema_decay", type=float, default=0.995)
    group.add_argument("--model_ema_update_after", type=int, default=5000)
    group.add_argument("--eval_during_training", action="store_true")
    group.add_argument("--eval_split", default="test", type=str)
    group.add_argument("--eval_num_batches", default=4, type=int)


def load_args_json(model_path: Path) -> dict:
    args_path = model_path.with_name("args.json")
    if not args_path.exists():
        return {}
    with args_path.open("r", encoding="utf-8") as file:
        return json.load(file)
