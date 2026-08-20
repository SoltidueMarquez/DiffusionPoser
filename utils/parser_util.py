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


def build_joint_finetune_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="联合微调 realtime pose Predictor 与 DiT。")
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_ik_inpainting_options(parser)
    add_training_options(parser)
    group = parser.add_argument_group("joint finetuning")
    group.add_argument("--dit_model_path", required=True)
    group.add_argument("--predictor_lr", default=1e-6, type=float)
    group.add_argument("--predictor_loss_weight", default=1.0, type=float)
    parser.set_defaults(
        lr=1e-5,
        num_steps=20_000,
        save_interval=1_000,
        run_name="joint_finetune",
    )
    return parser


def joint_finetune_args(argv: list[str] | None = None):
    parser = build_joint_finetune_arg_parser()
    # 只继承 DiT 的结构、扩散和 loss 契约；联合阶段自己的数据路径、学习率和
    # 运行长度必须由本次命令决定，不能被上一阶段的 args.json 覆盖。
    args = parse_and_load_from_model(
        parser,
        argv,
        ignore_keys={
            "cuda",
            "device",
            "seed",
            "batch_size",
            "precision",
            "data_dir",
            "data_split",
            "normalizer_dir",
            "normalize_input",
            "num_workers",
            "save_dir",
            "run_name",
            "overwrite",
            "train_platform_type",
            "lr",
            "weight_decay",
            "log_interval",
            "save_interval",
            "num_steps",
            "resume_checkpoint",
            "gradient_clip",
            "eval_during_training",
            "eval_split",
            "eval_num_batches",
            "predictor_lr",
            "predictor_loss_weight",
        },
    )
    args.joint_finetune = True
    return args


def apply_ik_calibration(args):
    calibration = str(getattr(args, "ik_calibration_path", "") or "").strip()
    if not calibration:
        return args
    path = Path(calibration).expanduser().resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    recommended = report["recommended_parameters"]
    for argument_name in (
        "ik_direction_only_quality",
        "ik_residual_scale",
        "ik_gap_low",
        "ik_gap_high",
    ):
        if getattr(args, argument_name, None) is not None:
            continue
        if argument_name not in recommended:
            raise ValueError(
                f"IK 校准文件缺少 recommended_parameters.{argument_name}：{path}"
            )
        setattr(args, argument_name, float(recommended[argument_name]))
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
    # `ts_respace` 是采样策略而非 checkpoint 结构；不能让训练 args.json 中的
    # 空值覆盖 runtime 的默认 10-step 或用户显式指定的消融步数。
    ignored = set(ignore_keys or set()) | {
        "predictor_model_path",
        "dit_model_path",
        "model_path",
        "ts_respace",
    }
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
    group.add_argument("--layers", default=4, type=int)
    group.add_argument("--heads", default=6, type=int)
    group.add_argument("--latent_dim", default=192, type=int)
    group.add_argument("--dropout", default=0.0, type=float)
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
    group.add_argument("--ik_gap_low", default=None, type=float)
    group.add_argument("--ik_gap_high", default=None, type=float)
    group.add_argument("--ik_direction_support", default=0.35, type=float)
    group.add_argument("--ik_untracked_strength", default=0.05, type=float)


def add_sampling_options(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("sampling")
    group.add_argument("--predictor_model_path", required=True)
    group.add_argument("--dit_model_path", required=True)
    group.add_argument("--output_dir", default="")
    group.add_argument("--visualize_num", default=0, type=int)
    group.add_argument("--visualize_fps", default=20.0, type=float)
    group.add_argument("--use_ema", default=True, type=str2bool)
    add_ik_inpainting_options(parser)
    # 训练仍使用 50 个基础 timestep；采样入口默认构造确定性的 10-step DDIM。
    parser.set_defaults(ts_respace="10")


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
    group.add_argument("--model_ema_decay", default=0.995, type=float)
    group.add_argument("--eval_during_training", action="store_true")
    group.add_argument("--eval_split", default="test")
    group.add_argument("--eval_num_batches", default=4, type=int)


def load_args_json(model_path: Path) -> dict:
    path = model_path.with_name("args.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
