from __future__ import annotations

import argparse
import subprocess
import sys
from argparse import BooleanOptionalAction
from pathlib import Path

from data_loaders.sensor_masking import DEFAULT_REALTIME_POSE_SCHEMA_NAME, REALTIME_POSE_SCHEMA_NAMES, get_schema_spec


PIPELINE_STAGES = ("convert", "tasks", "normalizer", "train")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run realtime_pose data conversion, task generation, normalizer, and training in one command."
    )
    paths = parser.add_argument_group("paths")
    paths.add_argument("--amass_dir", default="dataset/AMASS", type=str)
    paths.add_argument("--smpl_model_dir", default="dataset/body_models", type=str)
    paths.add_argument("--source_dir", default="dataset/AMASS_realtime_pose_v2_60hz", type=str)
    paths.add_argument("--normalizer_dir", default="dataset/meta_AMASS_realtime_pose_v2_60hz", type=str)
    paths.add_argument("--task_dir", default="dataset/AMASS_realtime_pose_v2_60hz_tasks", type=str)
    paths.add_argument("--split_dir", default="data_loaders/splits", type=str)
    paths.add_argument("--save_dir", default="runs/realtime_pose_v2_contact_target_dit", type=str)

    pipeline = parser.add_argument_group("pipeline")
    pipeline.add_argument("--schema", default=DEFAULT_REALTIME_POSE_SCHEMA_NAME, choices=REALTIME_POSE_SCHEMA_NAMES)
    pipeline.add_argument("--start_at", default="convert", choices=PIPELINE_STAGES)
    pipeline.add_argument("--stop_after", default="train", choices=PIPELINE_STAGES)
    pipeline.add_argument("--dry_run", action="store_true")
    pipeline.add_argument("--run_name", default="auto", type=str)
    pipeline.add_argument("--overwrite", action=BooleanOptionalAction, default=False)

    convert = parser.add_argument_group("convert")
    convert.add_argument("--skip_convert", action="store_true")
    convert.add_argument("--rebuild_source", action="store_true")
    convert.add_argument("--reuse_source_dir", default="", type=str)
    convert.add_argument("--target_fps", default=60.0, type=float)
    convert.add_argument("--convert_batch_size", default=256, type=int)
    convert.add_argument("--convert_limit", default=0, type=int)
    convert.add_argument("--mirror", action=BooleanOptionalAction, default=True)
    convert.add_argument("--skip_existing", action=BooleanOptionalAction, default=True)
    convert.add_argument("--allow_partial", action="store_true")

    normalizer = parser.add_argument_group("normalizer")
    normalizer.add_argument("--skip_normalizer", action="store_true")
    normalizer.add_argument("--normalizer_split", default="train", type=str)

    tasks = parser.add_argument_group("tasks")
    tasks.add_argument("--skip_tasks", action="store_true")
    tasks.add_argument("--splits", nargs="+", default=["train", "test"])
    tasks.add_argument("--samples_per_file", default=4, type=int)
    tasks.add_argument("--mask_policy", default="full", choices=["full", "fixed_patterns"])
    tasks.add_argument("--fixed_tracker_patterns", nargs="+", default=["all"])
    tasks.add_argument("--short_source_policy", default="skip", choices=["skip", "error"])

    train = parser.add_argument_group("train")
    train.add_argument("--skip_train", action="store_true")
    train.add_argument("--resume_latest", action="store_true")
    train.add_argument("--model_arch", default="target_dit", choices=["full_feature_dit", "target_dit"])
    train.add_argument("--cuda", default=True, type=str2bool)
    train.add_argument("--device", default=0, type=int)
    train.add_argument("--train_batch_size", default=64, type=int)
    train.add_argument("--num_workers", default=0, type=int)
    train.add_argument("--num_steps", default=1_000_000, type=int)
    train.add_argument("--save_interval", default=5_000, type=int)
    train.add_argument("--log_interval", default=1_000, type=int)
    train.add_argument("--checkpoint_max_keep", default=3, type=int)
    train.add_argument("--lr", default=1e-4, type=float)
    train.add_argument("--train_platform_type", default="TensorboardPlatform", choices=["NoPlatform", "TensorboardPlatform"])
    train.add_argument("--layers", default=8, type=int)
    train.add_argument("--heads", default=8, type=int)
    train.add_argument("--latent_dim", default=512, type=int)
    train.add_argument("--diffusion_steps", default=50, type=int)
    train.add_argument("--ts_respace", default="", type=str)
    train.add_argument("--model_ema", action=BooleanOptionalAction, default=True)
    train.add_argument("--gradient_clip", action=BooleanOptionalAction, default=True)
    train.add_argument("--history_pose_noise_std", default=0.02, type=float)
    train.add_argument("--history_yaw_noise_std", default=0.02, type=float)
    train.add_argument("--history_pose_dropout_prob", default=0.05, type=float)
    train.add_argument("--history_pose_replace_prob", default=0.05, type=float)
    train.add_argument("--history_yaw_replace_prob", default=0.0, type=float)
    train.add_argument("--tracker_latency_max_frames", default=2, type=int)
    train.add_argument("--tracker_burst_dropout_prob", default=0.05, type=float)
    train.add_argument("--tracker_outlier_prob", default=0.01, type=float)
    train.add_argument("--predicted_history_cache_dir", default="", type=str)
    train.add_argument("--predicted_history_prob", default=0.0, type=float)
    return parser


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def selected_stages(start_at: str, stop_after: str) -> tuple[str, ...]:
    start_index = PIPELINE_STAGES.index(start_at)
    stop_index = PIPELINE_STAGES.index(stop_after)
    if stop_index < start_index:
        raise ValueError(f"stop_after={stop_after} 不能早于 start_at={start_at}")
    return PIPELINE_STAGES[start_index: stop_index + 1]


def add_flag(args: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        args.append(flag)


def add_bool_value(args: list[str], name: str, value: bool) -> None:
    args.extend([name, "true" if value else "false"])


def run_python_module(module: str, args: list[str], dry_run: bool) -> None:
    command = [sys.executable, "-m", module, *args]
    printable = " ".join(quote_part(part) for part in command)
    print(f"[realtime_pose_pipeline] {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=project_root(), check=True)


def quote_part(value: str) -> str:
    if any(char.isspace() for char in value):
        return f'"{value}"'
    return value


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_path(path: str) -> str:
    return str(Path(path))


def build_convert_args(args: argparse.Namespace) -> list[str]:
    command = [
        "--schema", args.schema,
        "--amass_dir", normalize_path(args.amass_dir),
        "--smpl_model_dir", normalize_path(args.smpl_model_dir),
        "--output_dir", normalize_path(args.source_dir),
        "--target_fps", str(args.target_fps),
        "--batch_size", str(args.convert_batch_size),
    ]
    if args.reuse_source_dir and not args.rebuild_source:
        command.extend(["--reuse_source_dir", normalize_path(args.reuse_source_dir)])
    if int(args.convert_limit) > 0:
        command.extend(["--limit", str(args.convert_limit)])
    add_flag(command, bool(args.mirror), "--mirror")
    if args.rebuild_source:
        add_flag(command, bool(args.overwrite), "--overwrite")
    else:
        add_flag(command, bool(args.skip_existing), "--skip_existing")
        add_flag(command, bool(args.overwrite), "--rebuild_manifest")
    add_flag(command, bool(args.allow_partial), "--allow_partial")
    return command


def build_normalizer_args(args: argparse.Namespace) -> list[str]:
    command = [
        "--schema", args.schema,
        "--task_dir", normalize_path(args.task_dir),
        "--output_dir", normalize_path(args.normalizer_dir),
        "--split", args.normalizer_split,
        "--run_name", args.run_name,
    ]
    add_flag(command, bool(args.overwrite), "--overwrite")
    return command


def build_task_args(args: argparse.Namespace) -> list[str]:
    command = [
        "--schema", args.schema,
        "--source_dir", normalize_path(args.source_dir),
        "--output_dir", normalize_path(args.task_dir),
        "--split_dir", normalize_path(args.split_dir),
        "--splits", *[str(split) for split in args.splits],
        "--samples_per_file", str(args.samples_per_file),
        "--mask_policy", args.mask_policy,
        "--fixed_tracker_patterns", *[str(pattern) for pattern in args.fixed_tracker_patterns],
        "--short_source_policy", args.short_source_policy,
        "--run_name", args.run_name,
    ]
    add_flag(command, bool(args.overwrite), "--overwrite")
    return command


def build_train_args(args: argparse.Namespace) -> list[str]:
    schema = get_schema_spec(args.schema)
    command = [
        "--schema", schema.name,
        "--model_arch", args.model_arch,
        "--input_feats", str(schema.feature_dim),
        "--data_dir", normalize_path(args.task_dir),
        "--data_split", "train",
        "--normalizer_dir", normalize_path(args.normalizer_dir),
        "--save_dir", normalize_path(args.save_dir),
        "--run_name", args.run_name,
        "--batch_size", str(args.train_batch_size),
        "--num_workers", str(args.num_workers),
        "--num_steps", str(args.num_steps),
        "--save_interval", str(args.save_interval),
        "--log_interval", str(args.log_interval),
        "--checkpoint_max_keep", str(args.checkpoint_max_keep),
        "--lr", str(args.lr),
        "--train_platform_type", args.train_platform_type,
        "--layers", str(args.layers),
        "--heads", str(args.heads),
        "--latent_dim", str(args.latent_dim),
        "--diffusion_steps", str(args.diffusion_steps),
        "--history_pose_noise_std", str(args.history_pose_noise_std),
        "--history_yaw_noise_std", str(args.history_yaw_noise_std),
        "--history_pose_dropout_prob", str(args.history_pose_dropout_prob),
        "--history_pose_replace_prob", str(args.history_pose_replace_prob),
        "--history_yaw_replace_prob", str(args.history_yaw_replace_prob),
        "--tracker_latency_max_frames", str(args.tracker_latency_max_frames),
        "--tracker_burst_dropout_prob", str(args.tracker_burst_dropout_prob),
        "--tracker_outlier_prob", str(args.tracker_outlier_prob),
        "--tracker_mask_policy", "dynamic_categories",
        "--tracker_mask_categories", "all",
    ]
    add_bool_value(command, "--cuda", bool(args.cuda))
    command.extend(["--device", str(args.device)])
    if args.ts_respace:
        command.extend(["--ts_respace", args.ts_respace])
    if args.predicted_history_cache_dir:
        command.extend(["--predicted_history_cache_dir", normalize_path(args.predicted_history_cache_dir)])
        command.extend(["--predicted_history_prob", str(args.predicted_history_prob)])
    add_flag(command, bool(args.model_ema), "--model_ema")
    add_flag(command, bool(args.gradient_clip), "--gradient_clip")
    add_flag(command, bool(args.overwrite), "--overwrite")
    add_flag(command, bool(args.resume_latest), "--resume_checkpoint")
    if bool(args.resume_latest):
        command.append("latest")
    return command


def run_pipeline(args: argparse.Namespace) -> None:
    stages = selected_stages(args.start_at, args.stop_after)
    if "convert" in stages and not args.skip_convert:
        run_python_module("data_converter.amass_to_realtime_pose", build_convert_args(args), dry_run=args.dry_run)
    if "tasks" in stages and not args.skip_tasks:
        run_python_module("data_loaders.generate_realtime_pose_tasks", build_task_args(args), dry_run=args.dry_run)
    if "normalizer" in stages and not args.skip_normalizer:
        run_python_module("data_loaders.compute_realtime_pose_normalizer", build_normalizer_args(args), dry_run=args.dry_run)
    if "train" in stages and not args.skip_train:
        run_python_module("train.train_diffusionposer", build_train_args(args), dry_run=args.dry_run)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
