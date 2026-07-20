from __future__ import annotations

import json
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from collections.abc import Sequence
from pathlib import Path

from diffusion.realtime_pose import REALTIME_POSE_LOSS_DEFAULTS
from train.realtime_rollout import REALTIME_ROLLOUT_V3_DEFAULTS
from data_loaders.sensor_masking import (
    DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SEQ_LEN,
    TASK_MODE_REALTIME_POSE,
    TASK_MODES,
    TRACKER_MASK_FILL_MODES,
    TRACKER_MASK_POLICIES,
    get_schema_spec,
)
from schemas.registry import list_schema_names
from utils.default_artifact_paths import default_realtime_pose_normalizer_root
from utils.schema_resolution import has_explicit_schema_arg, resolve_runtime_schema


TRAIN_REALTIME_POSE_SCHEMA_NAMES = tuple(list_schema_names(trainable_only=True))
NORMALIZER_DIR_ARG = "--normalizer_dir"


def has_explicit_option_arg(argv: Sequence[str] | None, option_name: str) -> bool:
    values = sys.argv[1:] if argv is None else list(argv)
    return any(value == option_name or value.startswith(f"{option_name}=") for value in values)


def apply_data_path_defaults(
    args,
    *,
    normalizer_dir_explicit: bool = False,
    force_normalizer_dir: bool = False,
):
    if not hasattr(args, "normalizer_dir"):
        return args
    if normalizer_dir_explicit:
        setattr(args, "_normalizer_dir_auto_default", False)
        return args
    value = str(getattr(args, "normalizer_dir", "") or "").strip()
    if force_normalizer_dir or not value:
        schema_name = str(getattr(args, "schema", DEFAULT_REALTIME_POSE_SCHEMA_NAME) or DEFAULT_REALTIME_POSE_SCHEMA_NAME)
        artifact_roots_config = str(getattr(args, "artifact_roots_config", "") or "").strip() or None
        args.normalizer_dir = str(
            default_realtime_pose_normalizer_root(
                schema_name=schema_name,
                artifact_roots_config=artifact_roots_config,
            )
        )
        setattr(args, "_normalizer_dir_auto_default", True)
    return args


def _install_data_path_default_parser(parser: ArgumentParser) -> None:
    if getattr(parser, "_realtime_pose_data_path_defaults_installed", False):
        return
    original_parse_args = parser.parse_args

    def parse_args_with_data_defaults(args=None, namespace=None):
        parsed = original_parse_args(args, namespace)
        return apply_data_path_defaults(
            parsed,
            normalizer_dir_explicit=has_explicit_option_arg(args, NORMALIZER_DIR_ARG),
        )

    parser.parse_args = parse_args_with_data_defaults
    setattr(parser, "_realtime_pose_data_path_defaults_installed", True)


def train_args(argv: list[str] | None = None):
    parser = ArgumentParser(
        description="Train a realtime_pose_stationary5_v1 diffusion reconstruction model.",
        allow_abbrev=False,
    )
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return parser.parse_args(argv)


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
        if getattr(args, key) == default_value or bool(getattr(args, f"_{key}_auto_default", False)):
            setattr(args, key, value)
    return args


def parse_and_load_runtime_schema_from_model(
    parser: ArgumentParser,
    argv: list[str] | None = None,
    ignore_keys: set[str] | None = None,
):
    """
    解析采样/运行时入口参数，并按 checkpoint exact schema 修正 args.schema。

    调用方 parser 必须使用 allow_abbrev=False；否则 argparse 可能接受
    --sche 这类缩写，而 has_explicit_schema_arg 只把精确的 --schema 形式
    视为用户显式指定 schema。
    """
    cli_schema_explicit = has_explicit_schema_arg(argv)
    normalizer_dir_explicit = has_explicit_option_arg(argv, NORMALIZER_DIR_ARG)
    effective_ignore_keys = set(ignore_keys or set())
    if cli_schema_explicit:
        effective_ignore_keys.add("schema")
    if not normalizer_dir_explicit:
        effective_ignore_keys.add("normalizer_dir")
    args = parse_and_load_from_model(parser, argv=argv, ignore_keys=effective_ignore_keys)
    model_path = getattr(args, "model_path", "")
    checkpoint_args = load_args_json(Path(model_path)) if model_path else {}
    args.schema = resolve_runtime_schema(
        cli_schema=getattr(args, "schema", None),
        checkpoint_args=checkpoint_args,
        cli_schema_explicit=cli_schema_explicit,
    )
    apply_data_path_defaults(
        args,
        normalizer_dir_explicit=normalizer_dir_explicit,
        force_normalizer_dir=not normalizer_dir_explicit,
    )
    return args


def add_base_options(parser: ArgumentParser):
    group = parser.add_argument_group("base")
    group.add_argument("--artifact_roots_config", default="", type=str)
    group.add_argument("--cuda", default=True, type=str2bool)
    group.add_argument("--device", default=0, type=int)
    group.add_argument("--seed", default=10, type=int)
    group.add_argument("--batch_size", default=64, type=int)


def add_data_options(parser: ArgumentParser):
    group = parser.add_argument_group("dataset")
    default_schema = get_schema_spec(DEFAULT_REALTIME_POSE_SCHEMA_NAME)
    group.add_argument("--task_mode", default=TASK_MODE_REALTIME_POSE, choices=TASK_MODES, type=str)
    group.add_argument("--schema", default=DEFAULT_REALTIME_POSE_SCHEMA_NAME, choices=TRAIN_REALTIME_POSE_SCHEMA_NAMES, type=str)
    group.add_argument("--data_dir", required=True, type=str, help="realtime_pose materialized task 目录。")
    group.add_argument("--data_split", default="train", type=str)
    group.add_argument("--normalizer_dir", default="", type=str)
    group.add_argument("--normalize_input", default=True, type=str2bool)
    group.add_argument("--preload_data", default=False, type=str2bool)
    group.add_argument("--input_feats", default=default_schema.feature_dim, type=int)
    group.add_argument("--seq_len", default=REALTIME_POSE_SEQ_LEN, type=int)
    group.add_argument("--num_workers", default=0, type=int)
    group.add_argument("--tracker_pos_noise_std", default=0.0, type=float)
    group.add_argument("--tracker_rot_noise_std", default=0.0, type=float)
    group.add_argument("--non_head_tracker_dropout_prob", default=0.0, type=float)
    group.add_argument("--tracker_mask_policy", default="auto", choices=TRACKER_MASK_POLICIES, type=str)
    group.add_argument("--tracker_mask_seed", default=0, type=int)
    group.add_argument("--tracker_mask_fill", default="zero", choices=TRACKER_MASK_FILL_MODES, type=str)
    group.add_argument("--tracker_mask_categories", nargs="+", default=["all"], type=str)
    group.add_argument("--history_pose_noise_std", default=0.02, type=float)
    group.add_argument("--history_yaw_noise_std", default=0.02, type=float)
    group.add_argument("--root_yaw_ref_noise_std", default=0.0, type=float)
    group.add_argument("--history_pose_dropout_prob", default=0.05, type=float)
    group.add_argument("--history_pose_replace_prob", default=0.05, type=float)
    group.add_argument("--history_yaw_replace_prob", default=0.0, type=float)
    group.add_argument("--history_root_yaw_drift_std", default=0.0, type=float)
    group.add_argument("--tracker_latency_max_frames", default=2, type=int)
    group.add_argument("--tracker_burst_dropout_prob", default=0.05, type=float)
    group.add_argument("--tracker_outlier_prob", default=0.01, type=float)
    group.add_argument("--rollout_steps", default=1, type=int)
    _install_data_path_default_parser(parser)


def add_sampling_options(parser: ArgumentParser):
    group = parser.add_argument_group("sampling")
    group.add_argument("--model_path", required=True, type=str)
    group.add_argument("--output_dir", default="", type=str)
    group.add_argument("--folder_path", default="", type=str)
    group.add_argument("--visualize_num", default=0, type=int)
    group.add_argument("--visualize_fps", default=20.0, type=float)
    group.add_argument("--source_fps", default=60.0, type=float)
    group.add_argument("--use_ema", default=True, type=str2bool)
    group.add_argument("--ik_init_mode", default="random", choices=["random", "tracker_pose"], type=str)
    group.add_argument("--ik_init_timestep", default=-1, type=int)
    group.add_argument("--ik_init_iterations", default=16, type=int)
    group.add_argument("--ik_init_lr", default=0.03, type=float)
    group.add_argument("--ik_init_pos_weight", default=1.0, type=float)
    group.add_argument("--ik_init_rot_weight", default=0.2, type=float)
    group.add_argument("--ik_init_reg_weight", default=0.01, type=float)
    group.add_argument("--ik_init_delta_limit", default=0.15, type=float)


def add_model_options(parser: ArgumentParser):
    group = parser.add_argument_group("model")
    group.add_argument("--layers", default=8, type=int)
    group.add_argument("--heads", default=8, type=int)
    group.add_argument("--latent_dim", default=512, type=int)
    group.add_argument("--dropout", default=0.0, type=float)
    group.add_argument("--zero_init", action="store_true")
    group.add_argument("--max_seq_len", default=REALTIME_POSE_SEQ_LEN, type=int)
    group.add_argument("--model_arch", default="target_dit", choices=["full_feature_dit", "target_dit"], type=str)


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
    group.add_argument("--init_checkpoint", default="", type=str)
    group.add_argument("--gradient_clip", action="store_true")
    group.add_argument("--weighted_loss", action="store_true")
    group.add_argument("--feature_w_file", default="feature_w.pt", type=str)
    group.add_argument("--snr_gamma", default=0.0, type=float)
    group.add_argument("--l1_loss", action="store_true")
    for option_name, default in REALTIME_POSE_LOSS_DEFAULTS.items():
        group.add_argument(f"--{option_name}", default=default, type=float)
    for option_name, default in REALTIME_ROLLOUT_V3_DEFAULTS.items():
        group.add_argument(f"--{option_name}", default=default, type=type(default))
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
