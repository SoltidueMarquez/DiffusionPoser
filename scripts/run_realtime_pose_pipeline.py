from __future__ import annotations

import argparse
import subprocess
import sys
from argparse import BooleanOptionalAction
from pathlib import Path


PIPELINE_STAGES = (
    "convert",
    "tasks",
    "normalizer",
    "predictor",
    "calibrate",
    "train",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 Predictor + 单帧 DiT Python 训练流水线。")
    paths = parser.add_argument_group("paths")
    paths.add_argument("--amass_dir", default="dataset/AMASS")
    paths.add_argument("--smpl_model_dir", default="dataset/body_models")
    paths.add_argument("--body_fbx_rest_json", default="")
    paths.add_argument("--source_dir", default="dataset/realtime_pose_source")
    paths.add_argument("--task_dir", default="dataset/realtime_pose_tasks")
    paths.add_argument("--normalizer_dir", default="dataset/realtime_pose_normalizer")
    paths.add_argument("--split_dir", default="data_loaders/splits/RPM-P2")
    paths.add_argument("--predictor_save_dir", default="runs/predictor")
    paths.add_argument("--predictor_model_path", default="")
    paths.add_argument("--save_dir", required=True)
    paths.add_argument("--ik_calibration_path", default="output/ik_calibration.json")
    pipeline = parser.add_argument_group("pipeline")
    pipeline.add_argument("--start_at", choices=PIPELINE_STAGES, default="convert")
    pipeline.add_argument("--stop_after", choices=PIPELINE_STAGES, default="train")
    pipeline.add_argument("--dry_run", action="store_true")
    pipeline.add_argument("--overwrite", action=BooleanOptionalAction, default=False)
    for stage in PIPELINE_STAGES:
        pipeline.add_argument(f"--skip_{stage}", action="store_true")
    convert = parser.add_argument_group("convert")
    convert.add_argument("--convert_batch_size", default=256, type=int)
    convert.add_argument("--convert_num_workers", default=1, type=int)
    convert.add_argument("--convert_worker_torch_threads", default=1, type=int)
    tasks = parser.add_argument_group("tasks")
    tasks.add_argument("--splits", nargs="+", default=["train", "test"])
    tasks.add_argument("--base_windows_per_source", default=20, type=int)
    tasks.add_argument("--shard_size", default=4096, type=int)
    train = parser.add_argument_group("training")
    train.add_argument("--device", default=0, type=int)
    train.add_argument("--batch_size", default=64, type=int)
    train.add_argument("--num_workers", default=0, type=int)
    train.add_argument("--predictor_num_steps", default=100_000, type=int)
    train.add_argument("--dit_num_steps", default=1_000_000, type=int)
    train.add_argument("--predictor_windows_per_source", default=128, type=int)
    train.add_argument("--save_interval", default=5000, type=int)
    train.add_argument("--checkpoint_max_keep", default=3, type=int)
    train.add_argument("--log_interval", default=10, type=int)
    train.add_argument("--run_name", default="auto")
    train.add_argument("--predictor_resume_checkpoint", default="")
    train.add_argument("--dit_resume_checkpoint", default="")
    train.add_argument("--ik_direction_only_quality", default=None, type=float)
    train.add_argument("--ik_residual_scale", default=None, type=float)
    return parser


def selected_stages(start_at: str, stop_after: str) -> tuple[str, ...]:
    first = PIPELINE_STAGES.index(start_at)
    last = PIPELINE_STAGES.index(stop_after)
    if last < first:
        raise ValueError("stop_after 不能早于 start_at。")
    return PIPELINE_STAGES[first : last + 1]


def run_python_module(module: str, args: list[str], dry_run: bool) -> None:
    command = [sys.executable, "-m", module, *args]
    print("[realtime_pose_pipeline] " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)


def build_stage_args(stage: str, args: argparse.Namespace) -> tuple[str, list[str]]:
    if stage == "convert":
        return "data_converter.amass_to_realtime_pose", build_convert_args(args)
    if stage == "tasks":
        return "data_loaders.generate_realtime_pose_tasks", build_task_args(args)
    if stage == "normalizer":
        return "data_loaders.compute_realtime_pose_normalizer", build_normalizer_args(args)
    if stage == "predictor":
        return "train.train_realtime_pose_predictor", build_predictor_args(args)
    if stage == "calibrate":
        return "eval.calibrate_realtime_pose_ik", build_calibration_args(args)
    if stage == "train":
        return "train.train_diffusionposer", build_train_args(args)
    raise ValueError(f"未知 stage：{stage}")


def build_convert_args(args) -> list[str]:
    result = [
        "--amass_dir", str(args.amass_dir),
        "--smpl_model_dir", str(args.smpl_model_dir),
        "--output_dir", str(args.source_dir),
        "--batch_size", str(args.convert_batch_size),
        "--num_workers", str(args.convert_num_workers),
        "--worker_torch_threads", str(args.convert_worker_torch_threads),
        *(["--overwrite"] if args.overwrite else []),
    ]
    if str(args.body_fbx_rest_json).strip():
        result.extend(["--body_fbx_rest_json", str(args.body_fbx_rest_json)])
    return result


def build_task_args(args) -> list[str]:
    return [
        "--source_dir", str(args.source_dir),
        "--output_dir", str(args.task_dir),
        "--split_dir", str(args.split_dir),
        "--splits", *args.splits,
        "--base_windows_per_source", str(args.base_windows_per_source),
        "--shard_size", str(args.shard_size),
        *(["--overwrite"] if args.overwrite else []),
    ]


def build_normalizer_args(args) -> list[str]:
    return [
        "--task_dir", str(args.task_dir),
        "--output_dir", str(args.normalizer_dir),
        *(["--overwrite"] if args.overwrite else []),
    ]


def _predictor_model_path(args) -> str:
    return args.predictor_model_path or str(Path(args.predictor_save_dir) / "model_latest.pt")


def build_predictor_args(args) -> list[str]:
    result = [
        "--source_dir", str(args.source_dir),
        "--split_dir", str(args.split_dir),
        "--normalizer_dir", str(args.normalizer_dir),
        "--save_dir", str(args.predictor_save_dir),
        "--windows_per_source", str(args.predictor_windows_per_source),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--num_steps", str(args.predictor_num_steps),
        "--save_interval", str(args.save_interval),
        "--checkpoint_max_keep", str(args.checkpoint_max_keep),
        "--log_interval", str(args.log_interval),
        "--device", str(args.device),
    ]
    if str(args.predictor_resume_checkpoint).strip():
        result.extend(["--resume_checkpoint", str(args.predictor_resume_checkpoint)])
    return result


def build_calibration_args(args) -> list[str]:
    return [
        "--data_dir", str(args.task_dir),
        "--normalizer_dir", str(args.normalizer_dir),
        "--predictor_model_path", _predictor_model_path(args),
        "--output", str(args.ik_calibration_path),
        "--device", str(args.device),
    ]


def build_train_args(args) -> list[str]:
    result = [
        "--model_arch", "current_dit",
        "--data_dir", str(args.task_dir),
        "--normalizer_dir", str(args.normalizer_dir),
        "--predictor_model_path", _predictor_model_path(args),
        "--ik_calibration_path", str(args.ik_calibration_path),
        "--save_dir", str(args.save_dir),
        "--run_name", str(args.run_name),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--num_steps", str(args.dit_num_steps),
        "--save_interval", str(args.save_interval),
        "--log_interval", str(args.log_interval),
        "--device", str(args.device),
    ]
    if args.ik_direction_only_quality is not None:
        result.extend(["--ik_direction_only_quality", str(args.ik_direction_only_quality)])
    if args.ik_residual_scale is not None:
        result.extend(["--ik_residual_scale", str(args.ik_residual_scale)])
    if str(args.dit_resume_checkpoint).strip():
        result.extend(["--resume_checkpoint", str(args.dit_resume_checkpoint)])
    if args.overwrite:
        result.append("--overwrite")
    return result


def run_pipeline(args: argparse.Namespace) -> None:
    for stage in selected_stages(args.start_at, args.stop_after):
        if getattr(args, f"skip_{stage}"):
            continue
        module, values = build_stage_args(stage, args)
        run_python_module(module, values, bool(args.dry_run))


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
