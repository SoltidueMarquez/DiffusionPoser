from __future__ import annotations

import json
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

from data_loaders.realtime_pose_config import RealtimePoseLossWeights
from data_loaders.sensor_masking import (
    REALTIME_POSE_MODEL_TOKEN_LENGTH,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_DIM,
    TASK_MODE_REALTIME_POSE,
    TASK_MODES,
)
from utils.training_precision import TRAINING_PRECISIONS


_LOSS_DEFAULTS = RealtimePoseLossWeights()


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).lower()
    if text in {"yes", "true", "t", "1", "y"}:
        return True
    if text in {"no", "false", "f", "0", "n"}:
        return False
    raise ValueError(f"无法解析布尔参数：{value}")


def build_train_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="训练冻结 Predictor 条件下的单帧 realtime pose DiT。")
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_ik_inpainting_options(parser)
    add_training_options(parser)
    return parser


def train_args(argv: list[str] | None = None):
    return apply_ik_calibration(build_train_arg_parser().parse_args(argv))


def apply_ik_calibration(args):
    calibration = str(getattr(args, "ik_calibration_path", "") or "").strip()
    if not calibration:
        return args
    path = Path(calibration).expanduser().resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    recommended = report["recommended_parameters"]
    if getattr(args, "ik_direction_only_quality", None) is None:
        args.ik_direction_only_quality = float(recommended["ik_direction_only_quality"])
    if getattr(args, "ik_residual_scale", None) is None:
        args.ik_residual_scale = float(recommended["ik_residual_scale"])
    args.ik_calibration_path = str(path)
    return args


def parse_and_load_from_model(
    parser: ArgumentParser,
    argv: list[str] | None = None,
    ignore_keys: set[str] | None = None,
):
    args = parser.parse_args(argv)
    model_path = str(
        getattr(args, "dit_model_path", "") or getattr(args, "model_path", "") or ""
    )
    if not model_path:
        return apply_ik_calibration(args)
    checkpoint_args = load_args_json(Path(model_path))
    ignored = set(ignore_keys or set()) | {"predictor_model_path", "dit_model_path", "model_path"}
    for key, value in checkpoint_args.items():
        if key in ignored or not hasattr(args, key):
            continue
        if getattr(args, key) == parser.get_default(key):
            setattr(args, key, value)
    return apply_ik_calibration(args)


def add_base_options(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("base")
    group.add_argument("--cuda", default=True, type=str2bool)
    group.add_argument("--device", default=0, type=int)
    group.add_argument("--seed", default=10, type=int)
    group.add_argument("--batch_size", default=64, type=int)
    group.add_argument(
        "--precision", default="fp32", choices=TRAINING_PRECISIONS
    )


def add_data_options(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("dataset")
    group.add_argument("--task_mode", default=TASK_MODE_REALTIME_POSE, choices=TASK_MODES)
    group.add_argument("--data_dir", required=True)
    group.add_argument("--data_split", default="train")
    group.add_argument("--normalizer_dir", required=True)
    group.add_argument("--normalize_input", default=True, type=str2bool)
    group.add_argument("--input_feats", default=REALTIME_POSE_TARGET_DIM, type=int)
    group.add_argument("--seq_len", default=REALTIME_POSE_SEQ_LEN, type=int)
    group.add_argument("--num_workers", default=0, type=int)


def add_model_options(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("model")
    group.add_argument("--model_arch", default="current_dit", choices=("current_dit",))
    group.add_argument("--layers", default=6, type=int)
    group.add_argument("--heads", default=8, type=int)
    group.add_argument("--latent_dim", default=384, type=int)
    group.add_argument("--dropout", default=0.0, type=float)
    group.add_argument("--zero_init", action="store_true")
    group.add_argument("--max_seq_len", default=REALTIME_POSE_MODEL_TOKEN_LENGTH, type=int)


def add_diffusion_options(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("diffusion")
    group.add_argument("--noise_schedule", default="cosine", choices=("linear", "cosine"))
    group.add_argument("--diffusion_steps", default=50, type=int)
    group.add_argument("--sigma_small", default=True, type=str2bool)
    group.add_argument("--predict_xstart", default=1, type=int)
    group.add_argument("--ts_respace", default="")


def add_ik_inpainting_options(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("ik inpainting")
    group.add_argument("--ik_calibration_path", default="")
    group.add_argument("--fabrik_iterations", default=2, type=int)
    group.add_argument("--ik_direction_only_quality", default=None, type=float)
    group.add_argument("--ik_residual_scale", default=None, type=float)
    group.add_argument("--ik_position_solved_quality", default=None, type=float)


def add_sampling_options(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("sampling")
    group.add_argument("--predictor_model_path", required=True)
    group.add_argument("--dit_model_path", required=True)
    group.add_argument("--output_dir", default="")
    group.add_argument("--visualize_num", default=0, type=int)
    group.add_argument("--visualize_fps", default=20.0, type=float)
    group.add_argument("--use_ema", default=True, type=str2bool)
    add_ik_inpainting_options(parser)


def add_training_options(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("training")
    group.add_argument("--predictor_model_path", required=True)
    group.add_argument("--save_dir", required=True)
    group.add_argument("--run_name", default="auto")
    group.add_argument("--overwrite", default=False, action=BooleanOptionalAction)
    group.add_argument(
        "--train_platform_type",
        default="NoPlatform",
        choices=("NoPlatform", "TensorboardPlatform"),
    )
    group.add_argument("--lr", default=1e-4, type=float)
    group.add_argument("--weight_decay", default=0.0, type=float)
    group.add_argument("--log_interval", default=10, type=int)
    group.add_argument("--save_interval", default=5000, type=int)
    group.add_argument("--num_steps", default=1_000_000, type=int)
    group.add_argument("--resume_checkpoint", default="")
    group.add_argument("--gradient_clip", action="store_true")
    group.add_argument("--weighted_loss", action="store_true")
    group.add_argument("--feature_w_file", default="feature_w.pt")
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
    group.add_argument("--head_ref_joint_distance_loss_weight", default=_LOSS_DEFAULTS.head_ref_joint_distance, type=float)
    group.add_argument("--head_to_root_xz_loss_weight", default=_LOSS_DEFAULTS.head_to_root_xz, type=float)
    group.add_argument("--hip_height_loss_weight", default=_LOSS_DEFAULTS.hip_height, type=float)
    group.add_argument("--rotation_velocity_loss_weight", default=_LOSS_DEFAULTS.rotation_velocity, type=float)
    group.add_argument("--contact_loss_weight", default=_LOSS_DEFAULTS.contact, type=float)
    group.add_argument("--contact_slide_loss_weight", default=_LOSS_DEFAULTS.contact_slide, type=float)
    group.add_argument("--model_ema_decay", default=0.995, type=float)
    group.add_argument("--eval_during_training", action="store_true")
    group.add_argument("--eval_split", default="test")
    group.add_argument("--eval_num_batches", default=4, type=int)


def load_args_json(model_path: Path) -> dict:
    path = model_path.with_name("args.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
