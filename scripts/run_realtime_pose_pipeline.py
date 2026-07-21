from __future__ import annotations

import argparse
import json
import subprocess
import sys
from argparse import BooleanOptionalAction
from dataclasses import dataclass
from pathlib import Path

from data_converter.amass_to_realtime_pose import DEFAULT_SOURCE_SET_NAME
from data_loaders.compute_realtime_pose_normalizer import DEFAULT_NORMALIZER_NAME
from data_loaders.generate_realtime_pose_tasks import DEFAULT_TASK_SET_NAME
from data_loaders.realtime_pose_dataset import (
    load_source_reference_task_marker,
    read_task_manifest,
    reject_materialized_entry,
    validate_source_reference_entry,
)
from data_loaders.sensor_masking import DEFAULT_REALTIME_POSE_SCHEMA_NAME, REALTIME_POSE_SCHEMA_NAMES, get_schema_spec
from train.realtime_rollout import REALTIME_LR_DEFAULTS, REALTIME_ROLLOUT_DEFAULTS
from utils.artifact_paths import export_root, normalizer_root, run_root, source_root, task_root
from utils.artifact_roots import load_artifact_roots
from utils.run_dirs import resolve_latest_or_self


PIPELINE_STAGES = ("convert", "tasks", "normalizer", "train", "export")
SOURCE_USABLE_STATUSES = {"converted", "skipped_existing", "reused_source", "upgraded_existing_source"}


@dataclass(frozen=True)
class StageResult:
    stage: str
    status: str
    message: str = ""
    returncode: int = 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run realtime_pose data conversion, task generation, normalizer, and training in one command."
    )
    paths = parser.add_argument_group("paths")
    paths.add_argument("--artifact_roots_config", default="", type=str)
    paths.add_argument("--source_set_name", default=DEFAULT_SOURCE_SET_NAME, type=str)
    paths.add_argument("--task_set_name", default=DEFAULT_TASK_SET_NAME, type=str)
    paths.add_argument("--normalizer_name", default=DEFAULT_NORMALIZER_NAME, type=str)
    paths.add_argument("--amass_dir", default="", type=str)
    paths.add_argument("--smpl_model_dir", default="", type=str)
    paths.add_argument("--source_dir", default="", type=str)
    paths.add_argument("--normalizer_dir", default="", type=str)
    paths.add_argument("--task_dir", default="", type=str)
    paths.add_argument("--split_dir", default="data_loaders/splits", type=str)
    paths.add_argument("--save_dir", default="", type=str)
    paths.add_argument("--export_dir", default="", type=str)

    pipeline = parser.add_argument_group("pipeline")
    pipeline.add_argument("--schema", default=DEFAULT_REALTIME_POSE_SCHEMA_NAME, choices=REALTIME_POSE_SCHEMA_NAMES)
    pipeline.add_argument("--start_at", default="convert", choices=PIPELINE_STAGES)
    pipeline.add_argument("--stop_after", default="train", choices=PIPELINE_STAGES)
    pipeline.add_argument("--dry_run", action="store_true")
    pipeline.add_argument("--run_name", default="auto", type=str)
    pipeline.add_argument("--experiment_name", default="", type=str)
    pipeline.add_argument("--export_name", default="", type=str)
    pipeline.add_argument("--overwrite", action=BooleanOptionalAction, default=False)
    pipeline.add_argument(
        "--continue_on_error",
        "--keep_going",
        dest="continue_on_error",
        action="store_true",
        help="某个阶段失败后尽量继续后续阶段；最终仍会用非零退出汇总失败。",
    )
    pipeline.add_argument(
        "--resume_pipeline",
        action="store_true",
        help="跳过已有可用的 source/task/normalizer 产物；训练阶段仍通过 --resume_latest 控制。",
    )

    convert = parser.add_argument_group("convert")
    convert.add_argument("--skip_convert", action="store_true")
    convert.add_argument("--rebuild_source", action="store_true")
    convert.add_argument("--reuse_source_dir", default="", type=str)
    convert.add_argument("--target_fps", default=60.0, type=float)
    convert.add_argument("--convert_batch_size", default=256, type=int)
    convert.add_argument("--convert_num_workers", default=1, type=int)
    convert.add_argument("--convert_worker_torch_threads", default=1, type=int)
    convert.add_argument("--convert_limit", default=0, type=int)
    convert.add_argument("--body_fbx_rest_json", default="", type=str)
    convert.add_argument("--mirror", action=BooleanOptionalAction, default=True)
    convert.add_argument("--skip_existing", action=BooleanOptionalAction, default=True)
    convert.add_argument(
        "--allow_partial",
        action=BooleanOptionalAction,
        default=True,
        help="批量重建数据时允许 converter 记录并跳过少量失败/过短样本；可用 --no-allow_partial 恢复严格模式。",
    )

    normalizer = parser.add_argument_group("normalizer")
    normalizer.add_argument("--skip_normalizer", action="store_true")
    normalizer.add_argument("--normalizer_split", default="train", type=str)
    normalizer.add_argument("--normalizer_windows_per_source", default=2, type=int)
    normalizer.add_argument("--normalizer_convergence_windows_per_source", default=4, type=int)
    normalizer.add_argument("--normalizer_check_convergence", default=True, type=str2bool)
    normalizer.add_argument("--normalizer_tracker_mask_seed", default=10, type=int)

    tasks = parser.add_argument_group("tasks")
    tasks.add_argument("--skip_tasks", action="store_true")
    tasks.add_argument("--splits", nargs="+", default=["train", "test"])
    tasks.add_argument("--samples_per_source", default=2, type=int)
    tasks.add_argument("--mask_policy", default="full", choices=["full", "fixed_patterns"])
    tasks.add_argument("--fixed_tracker_patterns", nargs="+", default=[])
    tasks.add_argument("--patterns_per_source", default=1, type=int)
    tasks.add_argument("--task_rollout_steps", default=2, type=int)
    tasks.add_argument("--short_source_policy", default="skip", choices=["skip", "error"])

    train = parser.add_argument_group("train")
    train.add_argument("--skip_train", action="store_true")
    train.add_argument("--resume_latest", action="store_true")
    train.add_argument("--init_checkpoint", default="", type=str)
    train.add_argument("--model_arch", default="target_dit", choices=["full_feature_dit", "target_dit"])
    train.add_argument("--cuda", default=True, type=str2bool)
    train.add_argument("--device", default=0, type=int)
    train.add_argument("--train_batch_size", default=64, type=int)
    train.add_argument("--num_workers", default=0, type=int)
    train.add_argument("--source_cache_max_mib", default=512, type=int)
    train.add_argument("--num_steps", default=1_000_000, type=int)
    train.add_argument("--save_interval", default=5_000, type=int)
    train.add_argument("--log_interval", default=1_000, type=int)
    train.add_argument("--checkpoint_max_keep", default=3, type=int)
    train.add_argument("--lr", default=5e-5, type=float)
    for option_name, default in REALTIME_LR_DEFAULTS.items():
        train.add_argument(f"--{option_name}", default=default, type=type(default))
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
    train.add_argument("--tracker_latency_max_frames", default=0, type=int)
    train.add_argument("--tracker_burst_dropout_prob", default=0.0, type=float)
    train.add_argument("--tracker_outlier_prob", default=0.0, type=float)
    train.add_argument("--rollout_steps", default=1, type=int)
    for option_name, default in REALTIME_ROLLOUT_DEFAULTS.items():
        train.add_argument(f"--{option_name}", default=default, type=type(default))
    train.add_argument("--tracker_mask_policy", default="task", choices=["task", "fixed_categories", "dynamic_categories"])
    train.add_argument("--tracker_mask_categories", nargs="+", default=["all"])
    train.add_argument("--eval_during_training", action=BooleanOptionalAction, default=True)
    train.add_argument("--eval_num_batches", default=4, type=int)

    export = parser.add_argument_group("export")
    export.add_argument("--skip_export", action="store_true")
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


def path_arg_is_empty(value: object) -> bool:
    if value is None:
        return True
    return not str(value).strip()


def add_optional_path_arg(command: list[str], flag: str, value: object) -> None:
    if not path_arg_is_empty(value):
        command.extend([flag, normalize_path(str(value))])


def add_artifact_roots_arg(command: list[str], args: argparse.Namespace) -> None:
    if not path_arg_is_empty(getattr(args, "artifact_roots_config", "")):
        command.extend(["--artifact_roots_config", normalize_path(str(args.artifact_roots_config))])


def load_pipeline_artifact_roots(args: argparse.Namespace):
    return load_artifact_roots(getattr(args, "artifact_roots_config", "") or None)


def first_non_empty(*values: object, default: str = "auto") -> str:
    for value in values:
        text = str(value).strip() if value is not None else ""
        if text:
            return text
    return default


def resolve_pipeline_source_dir(args: argparse.Namespace) -> Path:
    if not path_arg_is_empty(getattr(args, "source_dir", "")):
        return Path(args.source_dir)
    # 与 converter/task resolver 使用同一套 schema 和 set 名，避免 pipeline 检查旧目录。
    return source_root(
        load_pipeline_artifact_roots(args),
        schema_name=str(args.schema),
        source_set_name=str(args.source_set_name),
    )


def resolve_pipeline_task_dir(args: argparse.Namespace) -> Path:
    if not path_arg_is_empty(getattr(args, "task_dir", "")):
        return Path(args.task_dir)
    return task_root(
        load_pipeline_artifact_roots(args),
        schema_name=str(args.schema),
        task_set_name=str(args.task_set_name),
    )


def resolve_pipeline_normalizer_dir(args: argparse.Namespace) -> Path:
    if not path_arg_is_empty(getattr(args, "normalizer_dir", "")):
        return Path(args.normalizer_dir)
    return normalizer_root(
        load_pipeline_artifact_roots(args),
        schema_name=str(args.schema),
        normalizer_name=str(args.normalizer_name),
    )


def resolve_pipeline_save_dir(args: argparse.Namespace) -> Path:
    if not path_arg_is_empty(getattr(args, "save_dir", "")):
        return Path(args.save_dir)
    experiment_name = first_non_empty(getattr(args, "experiment_name", ""), getattr(args, "run_name", ""))
    return run_root(
        schema_name=str(args.schema),
        experiment_name=experiment_name,
        roots=load_pipeline_artifact_roots(args),
    )


def resolve_pipeline_export_dir(args: argparse.Namespace) -> Path:
    if not path_arg_is_empty(getattr(args, "export_dir", "")):
        return Path(args.export_dir)
    export_name = first_non_empty(
        getattr(args, "export_name", ""),
        getattr(args, "experiment_name", ""),
        getattr(args, "run_name", ""),
    )
    return export_root(
        schema_name=str(args.schema),
        export_name=export_name,
        roots=load_pipeline_artifact_roots(args),
    )


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
    if stage == "export":
        return "export.write_unity_runtime_assets", build_export_args(args)
    raise ValueError(f"未知 pipeline stage: {stage}")


def run_stage(stage: str, args: argparse.Namespace) -> None:
    module, module_args = build_stage_args(stage, args)
    run_python_module(module, module_args, dry_run=bool(args.dry_run))


def should_skip_completed_stage(stage: str, args: argparse.Namespace) -> tuple[bool, str]:
    if not bool(getattr(args, "resume_pipeline", False)):
        return False, ""
    if bool(getattr(args, "overwrite", False)):
        return False, ""
    if stage == "convert" and not bool(getattr(args, "rebuild_source", False)):
        source_dir = resolve_pipeline_source_dir(args)
        if has_usable_source_manifest(source_dir=source_dir, schema_name=str(args.schema)):
            return True, f"复用已有 source manifest: {source_dir / 'manifest.jsonl'}"
    if stage == "tasks":
        task_dir = resolve_latest_or_self(resolve_pipeline_task_dir(args), kind="tasks")
        if has_task_manifests(task_dir=task_dir, splits=list(args.splits), schema_name=str(args.schema)):
            return True, f"复用已有 task 产物: {task_dir}"
    if stage == "normalizer":
        normalizer_dir = resolve_latest_or_self(resolve_pipeline_normalizer_dir(args), kind="normalizer")
        if has_normalizer_artifact(normalizer_dir=normalizer_dir, schema_name=str(args.schema)):
            return True, f"复用已有 normalizer 产物: {normalizer_dir}"
    return False, ""


def dependency_block_message(stage: str, failed_stages: set[str], args: argparse.Namespace) -> str:
    """阶段失败后只在依赖产物仍可用时继续，避免误用旧的 latest 产物。"""

    if stage == "tasks" and "convert" in failed_stages:
        if has_usable_source_manifest(source_dir=resolve_pipeline_source_dir(args), schema_name=str(args.schema)):
            return ""
        return "convert 失败且 source manifest 中没有可用 source，跳过 tasks。"
    if stage == "normalizer" and "tasks" in failed_stages:
        task_dir = resolve_latest_or_self(resolve_pipeline_task_dir(args), kind="tasks")
        if has_task_manifests(
            task_dir=task_dir,
            splits=[str(args.normalizer_split)],
            schema_name=str(args.schema),
        ):
            return ""
        return "tasks 失败且找不到可用于 normalizer_split 的 task manifest，跳过 normalizer。"
    if stage == "train":
        if "tasks" in failed_stages:
            task_dir = resolve_latest_or_self(resolve_pipeline_task_dir(args), kind="tasks")
            if not has_task_manifests(task_dir=task_dir, splits=["train"], schema_name=str(args.schema)):
                return "tasks 失败且找不到 train task manifest，跳过 train。"
        if "normalizer" in failed_stages and not bool(getattr(args, "skip_normalizer", False)):
            normalizer_dir = resolve_latest_or_self(resolve_pipeline_normalizer_dir(args), kind="normalizer")
            if not has_normalizer_artifact(normalizer_dir=normalizer_dir, schema_name=str(args.schema)):
                return "normalizer 失败且找不到可用 normalizer 产物，跳过 train。"
    return ""


def has_usable_source_manifest(source_dir: Path, schema_name: str) -> bool:
    manifest_path = source_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return False
    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("status") not in SOURCE_USABLE_STATUSES:
                    continue
                if str(entry.get("schema_name", schema_name)) != schema_name:
                    continue
                return True
    except (OSError, json.JSONDecodeError):
        return False
    return False


def has_task_manifests(task_dir: Path, splits: list[str], schema_name: str) -> bool:
    if not task_dir.exists():
        return False
    schema = get_schema_spec(schema_name)
    for split in splits:
        manifest_path = task_dir / str(split) / "manifest.jsonl"
        if not manifest_path.exists():
            return False
        try:
            load_source_reference_task_marker(manifest_path, schema_name=schema.name)
            entries = read_task_manifest(manifest_path)
            if not entries:
                return False
            for entry in entries:
                reject_materialized_entry(entry, source=str(manifest_path))
                validate_source_reference_entry(entry, schema=schema, required_rollout_steps=1)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
    return True


def has_normalizer_artifact(normalizer_dir: Path, schema_name: str) -> bool:
    mean_path = normalizer_dir / "mean.pt"
    std_path = normalizer_dir / "std.pt"
    meta_path = normalizer_dir / "normalizer_meta.json"
    convergence_path = normalizer_dir / "normalizer_convergence.json"
    if not (mean_path.exists() and std_path.exists() and meta_path.exists() and convergence_path.exists()):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        str(meta.get("schema_name", "")) == schema_name
        and meta.get("normalizer_convergence_passed") is True
        and int(meta.get("windows_per_source", 0)) == 2
        and int(meta.get("convergence_windows_per_source", 0)) == 4
        and str(meta.get("task_manifest_sha256", ""))
        and str(meta.get("source_manifest_sha256", ""))
        and convergence.get("passed") is True
    )


def format_pipeline_failures(failures: list[StageResult]) -> str:
    parts = [
        f"{failure.stage}(returncode={failure.returncode}): {failure.message}"
        for failure in failures
    ]
    return "realtime_pose pipeline 存在失败阶段：" + "; ".join(parts)


def build_convert_args(args: argparse.Namespace) -> list[str]:
    command = [
        "--schema", str(args.schema),
        "--source_set_name", str(args.source_set_name),
        "--target_fps", str(args.target_fps),
        "--batch_size", str(args.convert_batch_size),
        "--num_workers", str(args.convert_num_workers),
        "--worker_torch_threads", str(args.convert_worker_torch_threads),
    ]
    add_artifact_roots_arg(command, args)
    add_optional_path_arg(command, "--amass_dir", getattr(args, "amass_dir", ""))
    add_optional_path_arg(command, "--smpl_model_dir", getattr(args, "smpl_model_dir", ""))
    add_optional_path_arg(command, "--output_dir", getattr(args, "source_dir", ""))
    if args.reuse_source_dir and not args.rebuild_source:
        command.extend(["--reuse_source_dir", normalize_path(args.reuse_source_dir)])
    if int(args.convert_limit) > 0:
        command.extend(["--limit", str(args.convert_limit)])
    if args.body_fbx_rest_json:
        command.extend(["--body_fbx_rest_json", normalize_path(args.body_fbx_rest_json)])
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
        "--schema", str(args.schema),
        "--task_set_name", str(args.task_set_name),
        "--normalizer_name", str(args.normalizer_name),
        "--split", args.normalizer_split,
        "--windows_per_source", str(args.normalizer_windows_per_source),
        "--convergence_windows_per_source", str(args.normalizer_convergence_windows_per_source),
        "--check_convergence", str(bool(args.normalizer_check_convergence)).lower(),
        "--tracker_mask_seed", str(args.normalizer_tracker_mask_seed),
        "--run_name", args.run_name,
    ]
    add_artifact_roots_arg(command, args)
    add_optional_path_arg(command, "--task_dir", getattr(args, "task_dir", ""))
    add_optional_path_arg(command, "--output_dir", getattr(args, "normalizer_dir", ""))
    add_flag(command, bool(args.overwrite), "--overwrite")
    return command


def build_task_args(args: argparse.Namespace) -> list[str]:
    command = [
        "--schema", str(args.schema),
        "--source_set_name", str(args.source_set_name),
        "--task_set_name", str(args.task_set_name),
        "--split_dir", normalize_path(args.split_dir),
        "--splits", *[str(split) for split in args.splits],
        "--samples_per_source", str(args.samples_per_source),
        "--mask_policy", args.mask_policy,
        "--patterns_per_source", str(args.patterns_per_source),
        "--rollout_steps", str(args.task_rollout_steps),
        "--short_source_policy", args.short_source_policy,
        "--run_name", args.run_name,
    ]
    if args.fixed_tracker_patterns:
        command.extend(["--fixed_tracker_patterns", *[str(pattern) for pattern in args.fixed_tracker_patterns]])
    add_artifact_roots_arg(command, args)
    add_optional_path_arg(command, "--source_dir", getattr(args, "source_dir", ""))
    add_optional_path_arg(command, "--output_dir", getattr(args, "task_dir", ""))
    add_flag(command, bool(args.overwrite), "--overwrite")
    return command


def build_export_args(args: argparse.Namespace) -> list[str]:
    schema = get_schema_spec(args.schema)
    normalizer_dir = resolve_latest_or_self(resolve_pipeline_normalizer_dir(args), kind="normalizer")
    command = [
        "--schema", schema.name,
        "--output_dir", normalize_path(resolve_pipeline_export_dir(args)),
        "--diffusion_steps", str(args.diffusion_steps),
        "--normalizer_dir", normalize_path(normalizer_dir),
        "--normalize_input",
    ]
    return command


def build_train_args(args: argparse.Namespace) -> list[str]:
    schema = get_schema_spec(args.schema)
    command = [
        "--schema", schema.name,
        "--model_arch", args.model_arch,
        "--input_feats", str(schema.feature_dim),
        "--data_dir", normalize_path(resolve_pipeline_task_dir(args)),
        "--data_split", "train",
        "--normalizer_dir", normalize_path(resolve_pipeline_normalizer_dir(args)),
        "--save_dir", normalize_path(resolve_pipeline_save_dir(args)),
        "--run_name", args.run_name,
        "--batch_size", str(args.train_batch_size),
        "--num_workers", str(args.num_workers),
        "--source_cache_max_mib", str(args.source_cache_max_mib),
        "--num_steps", str(args.num_steps),
        "--save_interval", str(args.save_interval),
        "--log_interval", str(args.log_interval),
        "--checkpoint_max_keep", str(args.checkpoint_max_keep),
        "--lr", str(args.lr),
        "--lr_warmup_start", str(args.lr_warmup_start),
        "--lr_warmup_steps", str(args.lr_warmup_steps),
        "--lr_min", str(args.lr_min),
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
        "--tracker_mask_policy", str(args.tracker_mask_policy),
        "--tracker_mask_categories", *[str(value) for value in args.tracker_mask_categories],
        "--rollout_steps", str(args.rollout_steps),
        "--short_rollout_prob", str(args.short_rollout_prob),
        "--short_rollout_loss_weight", str(args.short_rollout_loss_weight),
        "--long_rollout_prob", str(args.long_rollout_prob),
        "--long_rollout_loss_weight", str(args.long_rollout_loss_weight),
        "--rollout_h1_start_step", str(args.rollout_h1_start_step),
        "--rollout_h2_start_step", str(args.rollout_h2_start_step),
        "--rollout_h4_start_step", str(args.rollout_h4_start_step),
        "--rollout_h8_start_step", str(args.rollout_h8_start_step),
        "--rollout_prob_ramp_steps", str(args.rollout_prob_ramp_steps),
        "--rollout_max_horizon_prob", str(args.rollout_max_horizon_prob),
        "--long_rollout_transition_prob", str(args.long_rollout_transition_prob),
        "--long_rollout_smooth_l1_beta", str(args.long_rollout_smooth_l1_beta),
        "--rollout_ddim_steps", str(args.rollout_ddim_steps),
        "--eval_num_batches", str(args.eval_num_batches),
    ]
    add_bool_value(command, "--cuda", bool(args.cuda))
    command.extend(["--device", str(args.device)])
    if args.ts_respace:
        command.extend(["--ts_respace", args.ts_respace])
    if args.init_checkpoint:
        command.extend(["--init_checkpoint", normalize_path(args.init_checkpoint)])
    add_flag(command, bool(args.model_ema), "--model_ema")
    add_flag(command, bool(args.gradient_clip), "--gradient_clip")
    add_flag(command, bool(args.eval_during_training), "--eval_during_training")
    add_flag(command, bool(args.overwrite), "--overwrite")
    add_flag(command, bool(args.resume_latest), "--resume_checkpoint")
    if bool(args.resume_latest):
        command.append("latest")
    return command


def run_pipeline(args: argparse.Namespace) -> None:
    stages = selected_stages(args.start_at, args.stop_after)
    failures: list[StageResult] = []
    failed_stages: set[str] = set()
    for stage in stages:
        if stage_is_disabled(stage, args):
            print(f"[realtime_pose_pipeline] skip {stage}: --skip_{stage}", flush=True)
            continue

        should_skip, skip_message = should_skip_completed_stage(stage, args)
        if should_skip:
            print(f"[realtime_pose_pipeline] skip {stage}: {skip_message}", flush=True)
            continue

        if bool(getattr(args, "continue_on_error", False)):
            blocked_message = dependency_block_message(stage=stage, failed_stages=failed_stages, args=args)
            if blocked_message:
                print(f"[realtime_pose_pipeline] skip {stage}: {blocked_message}", flush=True)
                continue

        try:
            run_stage(stage, args)
        except subprocess.CalledProcessError as exc:
            failed_stages.add(stage)
            result = StageResult(
                stage=stage,
                status="failed",
                message=str(exc),
                returncode=int(exc.returncode),
            )
            failures.append(result)
            print(f"[realtime_pose_pipeline] failed {stage}: {exc}", flush=True)
            if not bool(getattr(args, "continue_on_error", False)):
                raise

    if failures:
        raise RuntimeError(format_pipeline_failures(failures))


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
