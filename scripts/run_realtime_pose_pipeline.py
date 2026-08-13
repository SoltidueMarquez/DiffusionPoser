from __future__ import annotations

import argparse
import subprocess
import sys
from argparse import BooleanOptionalAction
from pathlib import Path

from data_loaders.sensor_masking import REALTIME_POSE_TARGET_DIM


PIPELINE_STAGES = ("convert", "tasks", "normalizer", "train")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run realtime_pose data conversion, task generation, normalizer, and training in one command."
    )
    paths = parser.add_argument_group("paths")
    paths.add_argument("--amass_dir", default="dataset/AMASS", type=str)
    paths.add_argument("--smpl_model_dir", default="dataset/body_models", type=str)
    paths.add_argument(
        "--source_dir",
        default="dataset/AMASS_realtime_pose_body_fbx_local_pelvis_residual_root_y0_stationary5_60hz",
        type=str,
    )
    paths.add_argument(
        "--normalizer_dir",
        default="dataset/meta_AMASS_realtime_pose_144d_pelvis_residual_root_y0_stationary5_60hz",
        type=str,
    )
    paths.add_argument(
        "--task_dir",
        default="dataset/AMASS_realtime_pose_144d_pelvis_residual_root_y0_stationary5_60hz_tasks",
        type=str,
    )
    paths.add_argument("--split_dir", default="data_loaders/splits", type=str)
    paths.add_argument("--save_dir", required=True, type=str)

    pipeline = parser.add_argument_group("pipeline")
    pipeline.add_argument("--start_at", default="convert", choices=PIPELINE_STAGES)
    pipeline.add_argument("--stop_after", default="train", choices=PIPELINE_STAGES)
    pipeline.add_argument("--dry_run", action="store_true")
    pipeline.add_argument("--run_name", default="auto", type=str)
    pipeline.add_argument("--overwrite", action=BooleanOptionalAction, default=False)

    convert = parser.add_argument_group("convert")
    convert.add_argument("--skip_convert", action="store_true")
    convert.add_argument("--target_fps", default=60.0, type=float)
    convert.add_argument("--convert_batch_size", default=256, type=int)
    convert.add_argument("--convert_num_workers", default=1, type=int)
    convert.add_argument("--convert_worker_torch_threads", default=1, type=int)
    convert.add_argument("--convert_limit", default=0, type=int)
    convert.add_argument("--body_fbx_rest_json", default="", type=str)
    convert.add_argument("--mirror", action=BooleanOptionalAction, default=True)
    convert.add_argument("--skip_existing", action=BooleanOptionalAction, default=True)
    convert.add_argument("--allow_partial", action="store_true")

    normalizer = parser.add_argument_group("normalizer")
    normalizer.add_argument("--skip_normalizer", action="store_true")
    normalizer.add_argument("--normalizer_split", default="train", type=str)

    tasks = parser.add_argument_group("tasks")
    tasks.add_argument("--skip_tasks", action="store_true")
    tasks.add_argument("--splits", nargs="+", default=["train", "test"])
    tasks.add_argument("--base_windows_per_source", default=20, type=int)
    tasks.add_argument("--shard_size", default=4096, type=int)
    tasks.add_argument("--short_source_policy", default="skip", choices=["skip", "error"])

    train = parser.add_argument_group("train")
    train.add_argument("--skip_train", action="store_true")
    train.add_argument("--resume_latest", action="store_true")
    train.add_argument(
        "--model_arch",
        default="spatiotemporal_dit",
        choices=["spatiotemporal_dit"],
    )
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
    train.add_argument("--scenario_weights", nargs=5, default=[0.2] * 5, type=float)
    train.add_argument("--history_noise_prob", default=0.8, type=float)
    train.add_argument("--history_noise_min_deg", default=2.0, type=float)
    train.add_argument("--history_noise_max_deg", default=10.0, type=float)
    train.add_argument("--history_noise_temporal_rho", default=0.95, type=float)
    train.add_argument("--history_noise_region_ratio", default=0.75, type=float)
    train.add_argument("--history_noise_joint_ratio", default=0.25, type=float)
    train.add_argument("--rotation_loss_weight", default=1.0, type=float)
    train.add_argument("--local_rot_loss_weight", default=1.0, type=float)
    train.add_argument("--fk_loss_weight", default=2.0, type=float)
    train.add_argument("--tracker_pos_loss_weight", default=10.0, type=float)
    train.add_argument("--tracker_rot_loss_weight", default=1.0, type=float)
    train.add_argument("--root_loss_weight", default=1.0, type=float)
    train.add_argument(
        "--head_ref_joint_distance_loss_weight", default=1.0, type=float
    )
    train.add_argument("--head_to_root_xz_loss_weight", default=1.0, type=float)
    train.add_argument("--rotation_velocity_loss_weight", default=1.0, type=float)
    train.add_argument("--contact_loss_weight", default=0.1, type=float)
    train.add_argument("--contact_slide_loss_weight", default=0.1, type=float)
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


def add_boolean_optional_flag(args: list[str], name: str, value: bool) -> None:
    """向使用 BooleanOptionalAction 的下游 parser 显式传递 true/false。"""

    args.append(name if value else f"--no-{name.removeprefix('--')}")


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


def stage_is_disabled(stage: str, args: argparse.Namespace) -> bool:
    return bool(getattr(args, f"skip_{stage}", False))


def build_stage_args(stage: str, args: argparse.Namespace) -> tuple[str, list[str]]:
    if stage == "convert":
        return "data_converter.amass_to_realtime_pose", build_convert_args(args)
    if stage == "tasks":
        return "data_loaders.generate_realtime_pose_tasks", build_task_args(args)
    if stage == "normalizer":
        return "data_loaders.compute_realtime_pose_normalizer", build_normalizer_args(args)
    if stage == "train":
        return "train.train_diffusionposer", build_train_args(args)
    raise ValueError(f"未知 pipeline stage: {stage}")


def run_stage(stage: str, args: argparse.Namespace) -> None:
    module, module_args = build_stage_args(stage, args)
    run_python_module(module, module_args, dry_run=bool(args.dry_run))


def build_convert_args(args: argparse.Namespace) -> list[str]:
    command = [
        "--amass_dir", normalize_path(args.amass_dir),
        "--smpl_model_dir", normalize_path(args.smpl_model_dir),
        "--output_dir", normalize_path(args.source_dir),
        "--target_fps", str(args.target_fps),
        "--batch_size", str(args.convert_batch_size),
        "--num_workers", str(args.convert_num_workers),
        "--worker_torch_threads", str(args.convert_worker_torch_threads),
    ]
    if int(args.convert_limit) > 0:
        command.extend(["--limit", str(args.convert_limit)])
    if args.body_fbx_rest_json:
        command.extend(["--body_fbx_rest_json", normalize_path(args.body_fbx_rest_json)])
    add_flag(command, bool(args.mirror), "--mirror")
    add_flag(command, bool(args.skip_existing) and not bool(args.overwrite), "--skip_existing")
    add_flag(command, bool(args.overwrite), "--overwrite")
    add_flag(command, bool(args.allow_partial), "--allow_partial")
    return command


def build_normalizer_args(args: argparse.Namespace) -> list[str]:
    command = [
        "--task_dir", normalize_path(args.task_dir),
        "--output_dir", normalize_path(args.normalizer_dir),
        "--split", args.normalizer_split,
    ]
    add_flag(command, bool(args.overwrite), "--overwrite")
    return command


def build_task_args(args: argparse.Namespace) -> list[str]:
    command = [
        "--source_dir", normalize_path(args.source_dir),
        "--output_dir", normalize_path(args.task_dir),
        "--split_dir", normalize_path(args.split_dir),
        "--splits", *[str(split) for split in args.splits],
        "--base_windows_per_source", str(args.base_windows_per_source),
        "--shard_size", str(args.shard_size),
        "--short_source_policy", args.short_source_policy,
    ]
    add_flag(command, bool(args.overwrite), "--overwrite")
    return command


def build_train_args(args: argparse.Namespace) -> list[str]:
    command = [
        "--model_arch", args.model_arch,
        "--input_feats", str(REALTIME_POSE_TARGET_DIM),
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
        "--scenario_weights", *[str(value) for value in args.scenario_weights],
        "--history_noise_prob", str(args.history_noise_prob),
        "--history_noise_min_deg", str(args.history_noise_min_deg),
        "--history_noise_max_deg", str(args.history_noise_max_deg),
        "--history_noise_temporal_rho", str(args.history_noise_temporal_rho),
        "--history_noise_region_ratio", str(args.history_noise_region_ratio),
        "--history_noise_joint_ratio", str(args.history_noise_joint_ratio),
        "--rotation_loss_weight", str(args.rotation_loss_weight),
        "--local_rot_loss_weight", str(args.local_rot_loss_weight),
        "--fk_loss_weight", str(args.fk_loss_weight),
        "--tracker_pos_loss_weight", str(args.tracker_pos_loss_weight),
        "--tracker_rot_loss_weight", str(args.tracker_rot_loss_weight),
        "--root_loss_weight", str(args.root_loss_weight),
        "--head_ref_joint_distance_loss_weight",
        str(args.head_ref_joint_distance_loss_weight),
        "--head_to_root_xz_loss_weight", str(args.head_to_root_xz_loss_weight),
        "--rotation_velocity_loss_weight", str(args.rotation_velocity_loss_weight),
        "--contact_loss_weight", str(args.contact_loss_weight),
        "--contact_slide_loss_weight", str(args.contact_slide_loss_weight),
    ]
    add_bool_value(command, "--cuda", bool(args.cuda))
    command.extend(["--device", str(args.device)])
    if args.ts_respace:
        command.extend(["--ts_respace", args.ts_respace])
    add_boolean_optional_flag(command, "--model_ema", bool(args.model_ema))
    add_flag(command, bool(args.gradient_clip), "--gradient_clip")
    add_flag(command, bool(args.overwrite), "--overwrite")
    add_flag(command, bool(args.resume_latest), "--resume_checkpoint")
    if bool(args.resume_latest):
        command.append("latest")
    return command


def run_pipeline(args: argparse.Namespace) -> None:
    stages = selected_stages(args.start_at, args.stop_after)
    for stage in stages:
        if stage_is_disabled(stage, args):
            print(f"[realtime_pose_pipeline] skip {stage}: --skip_{stage}", flush=True)
            continue

        run_stage(stage, args)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
