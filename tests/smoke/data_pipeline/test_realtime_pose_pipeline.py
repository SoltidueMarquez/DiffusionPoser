from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import data_loaders.compute_realtime_pose_normalizer as normalizer_computer
import data_loaders.generate_realtime_pose_tasks as task_generator
from data_loaders.realtime_pose_contract import runtime_contract_metadata
from data_loaders.sensor_masking import (
    REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_LENGTH,
    REALTIME_POSE_TARGET_START,
    TASK_MODE_REALTIME_POSE,
    get_schema_spec,
)
from data_loaders.stationary_label_config import stationary_label_metadata
from scripts import run_realtime_pose_pipeline as pipeline
from tests.smoke.realtime_pose_fixtures import write_toy_source_dataset
from utils.run_dirs import read_latest_pointer, write_latest_pointer


CONVERT_MODULE = "data_converter.amass_to_realtime_pose"
TASK_MODULE = "data_loaders.generate_realtime_pose_tasks"
NORMALIZER_MODULE = "data_loaders.compute_realtime_pose_normalizer"
EXPORT_MODULE = "export.write_unity_runtime_assets"


def write_artifact_roots_config(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "artifact_roots.json"
    generated_root = tmp_path / "generated"
    config_path.write_text(
        json.dumps(
            {
                "amass_root": str(tmp_path / "AMASS"),
                "generated_root": str(generated_root),
                "runs_root": str(tmp_path / "runs"),
                "outputs_root": str(tmp_path / "output"),
            }
        ),
        encoding="utf-8",
    )
    return config_path, generated_root


def parse_pipeline_args(tmp_path: Path, *extra: str):
    return pipeline.build_arg_parser().parse_args(
        [
            "--amass_dir",
            str(tmp_path / "AMASS"),
            "--smpl_model_dir",
            str(tmp_path / "body_models"),
            "--source_dir",
            str(tmp_path / "sources"),
            "--task_dir",
            str(tmp_path / "tasks"),
            "--normalizer_dir",
            str(tmp_path / "normalizer"),
            "--split_dir",
            "",
            "--splits",
            "train",
            "--schema",
            REALTIME_POSE_SCHEMA_NAME,
            *extra,
        ]
    )


def parse_schema_aware_pipeline_args(tmp_path: Path, *extra: str):
    config_path, _ = write_artifact_roots_config(tmp_path)
    return pipeline.build_arg_parser().parse_args(
        [
            "--amass_dir",
            str(tmp_path / "AMASS"),
            "--smpl_model_dir",
            str(tmp_path / "body_models"),
            "--artifact_roots_config",
            str(config_path),
            "--source_set_name",
            "toy_sources",
            "--task_set_name",
            "toy_tasks",
            "--normalizer_name",
            "toy_norm",
            "--split_dir",
            "",
            "--splits",
            "train",
            *extra,
        ]
    )


def assert_arg_value(command: list[str], flag: str, expected: str) -> None:
    assert flag in command
    assert command[command.index(flag) + 1] == expected


def write_usable_source_manifest(source_dir: Path) -> None:
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    source_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "status": "converted",
        "schema_name": schema.name,
        "pose_representation": schema.pose_representation,
        "source_relative_path": "toy.npz",
        "output_path": "toy.npz",
    }
    (source_dir / "manifest.jsonl").write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")


def write_latest_task_artifact(task_root: Path, split: str = "train") -> Path:
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    task_dir = task_root / "20260101_000000_rtp_tasks"
    split_dir = task_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    source_dir = task_root.parent / "sources"
    source_manifest = source_dir / "manifest.jsonl"
    if not source_manifest.exists():
        write_usable_source_manifest(source_dir)
    entry = {
        "task_id": "toy_source_reference",
        "task_format": schema.task_format,
        "schema_name": schema.name,
        "schema_canonical_name": str(schema.canonical_name),
        "pose_representation": schema.pose_representation,
        "root_y_policy": schema.root_y_policy,
        "pelvis_height_mode": schema.pelvis_height_mode,
        "source_path": str(source_dir / "toy.npz"),
        "source_relative_path": "toy.npz",
        "stablemotion_split_key": "toy",
        "source_frames": 70,
        "samples_per_source": 2,
        "sampling_seed": 10,
        "max_rollout_steps": 1,
        "seq_len": REALTIME_POSE_SEQ_LEN,
        "feature_dim": schema.feature_dim,
        "target_start": REALTIME_POSE_TARGET_START,
        "target_length": REALTIME_POSE_TARGET_LENGTH,
        "task_mode": TASK_MODE_REALTIME_POSE,
        **runtime_contract_metadata(),
        **stationary_label_metadata(),
    }
    (split_dir / "manifest.jsonl").write_text(
        json.dumps(entry, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    task_generator.write_task_output_marker(
        source_dir=source_dir,
        output_dir=task_dir,
        split_dir=None,
        schema_name=schema.name,
        generated_root=task_root.parent,
        source_set_name="toy",
        task_set_name="toy",
    )
    write_latest_pointer(
        root_dir=task_root,
        kind="tasks",
        output_dir=task_dir,
        metadata={"schema_name": REALTIME_POSE_SCHEMA_NAME, "splits": [split]},
    )
    return task_dir


def write_latest_normalizer_artifact(normalizer_root: Path) -> Path:
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    normalizer_dir = normalizer_root / "20260101_000000_normalizer"
    normalizer_dir.mkdir(parents=True, exist_ok=True)
    (normalizer_dir / "mean.pt").write_bytes(b"mean")
    (normalizer_dir / "std.pt").write_bytes(b"std")
    (normalizer_dir / "normalizer_meta.json").write_text(
        json.dumps(
            {
                "schema_name": schema.name,
                "windows_per_source": 2,
                "convergence_windows_per_source": 4,
                "normalizer_convergence_passed": True,
                "task_manifest_sha256": "task-sha",
                "source_manifest_sha256": "source-sha",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (normalizer_dir / "normalizer_convergence.json").write_text(
        json.dumps({"passed": True}),
        encoding="utf-8",
    )
    write_latest_pointer(
        root_dir=normalizer_root,
        kind="normalizer",
        output_dir=normalizer_dir,
        metadata={"schema_name": schema.name},
    )
    return normalizer_dir


def test_pipeline_continue_on_error_runs_later_stages_when_partial_source_exists(monkeypatch, tmp_path):
    write_usable_source_manifest(tmp_path / "sources")
    calls: list[str] = []

    def fake_run_python_module(module: str, args: list[str], dry_run: bool) -> None:
        calls.append(module)
        if module == CONVERT_MODULE:
            raise subprocess.CalledProcessError(7, ["python", "-m", module])

    monkeypatch.setattr(pipeline, "run_python_module", fake_run_python_module)
    args = parse_pipeline_args(tmp_path, "--stop_after", "normalizer", "--continue_on_error")

    with pytest.raises(RuntimeError, match="convert"):
        pipeline.run_pipeline(args)

    assert calls == [CONVERT_MODULE, TASK_MODULE, NORMALIZER_MODULE]


def test_pipeline_default_stops_on_stage_failure(monkeypatch, tmp_path):
    calls: list[str] = []

    def fake_run_python_module(module: str, args: list[str], dry_run: bool) -> None:
        calls.append(module)
        raise subprocess.CalledProcessError(7, ["python", "-m", module])

    monkeypatch.setattr(pipeline, "run_python_module", fake_run_python_module)
    args = parse_pipeline_args(tmp_path, "--stop_after", "normalizer")

    with pytest.raises(subprocess.CalledProcessError):
        pipeline.run_pipeline(args)

    assert calls == [CONVERT_MODULE]


def test_resume_pipeline_skips_completed_data_artifacts(monkeypatch, tmp_path):
    write_usable_source_manifest(tmp_path / "sources")
    write_latest_task_artifact(tmp_path / "tasks", split="train")
    write_latest_normalizer_artifact(tmp_path / "normalizer")
    calls: list[str] = []

    def fake_run_python_module(module: str, args: list[str], dry_run: bool) -> None:
        calls.append(module)

    monkeypatch.setattr(pipeline, "run_python_module", fake_run_python_module)
    args = parse_pipeline_args(tmp_path, "--stop_after", "normalizer", "--resume_pipeline")

    pipeline.run_pipeline(args)

    assert calls == []


def test_pipeline_passes_parallel_converter_args(tmp_path):
    args = parse_pipeline_args(
        tmp_path,
        "--convert_num_workers",
        "3",
        "--convert_worker_torch_threads",
        "2",
    )

    convert_args = pipeline.build_convert_args(args)

    assert convert_args[convert_args.index("--num_workers") + 1] == "3"
    assert convert_args[convert_args.index("--worker_torch_threads") + 1] == "2"


def test_pipeline_default_schema_is_stationary5():
    args = pipeline.build_arg_parser().parse_args([])

    assert args.schema == "realtime_pose_stationary5_v1"


def test_pipeline_schema_aware_commands_defer_paths_to_resolvers(tmp_path):
    config_path = tmp_path / "artifact_roots.json"
    args = parse_schema_aware_pipeline_args(
        tmp_path,
        "--schema",
        "realtime_pose_stationary5_v1",
    )

    convert_args = pipeline.build_convert_args(args)
    task_args = pipeline.build_task_args(args)
    normalizer_args = pipeline.build_normalizer_args(args)

    for command in (convert_args, task_args, normalizer_args):
        assert_arg_value(command, "--schema", "realtime_pose_stationary5_v1")
        assert_arg_value(command, "--artifact_roots_config", str(config_path))

    assert_arg_value(convert_args, "--source_set_name", "toy_sources")
    assert "--output_dir" not in convert_args

    assert_arg_value(task_args, "--source_set_name", "toy_sources")
    assert_arg_value(task_args, "--task_set_name", "toy_tasks")
    assert "--source_dir" not in task_args
    assert "--output_dir" not in task_args

    assert_arg_value(normalizer_args, "--task_set_name", "toy_tasks")
    assert_arg_value(normalizer_args, "--normalizer_name", "toy_norm")
    assert "--task_dir" not in normalizer_args
    assert "--output_dir" not in normalizer_args


def test_pipeline_explicit_paths_override_schema_aware_resolvers(tmp_path):
    explicit_source_dir = tmp_path / "explicit_sources"
    explicit_task_dir = tmp_path / "explicit_tasks"
    explicit_normalizer_dir = tmp_path / "explicit_normalizer"
    args = parse_schema_aware_pipeline_args(
        tmp_path,
        "--source_dir",
        str(explicit_source_dir),
        "--task_dir",
        str(explicit_task_dir),
        "--normalizer_dir",
        str(explicit_normalizer_dir),
        "--schema",
        "realtime_pose_stationary5_v1",
    )

    convert_args = pipeline.build_convert_args(args)
    task_args = pipeline.build_task_args(args)
    normalizer_args = pipeline.build_normalizer_args(args)

    assert_arg_value(convert_args, "--output_dir", str(explicit_source_dir))

    assert_arg_value(task_args, "--source_dir", str(explicit_source_dir))
    assert_arg_value(task_args, "--output_dir", str(explicit_task_dir))

    assert_arg_value(normalizer_args, "--task_dir", str(explicit_task_dir))
    assert_arg_value(normalizer_args, "--output_dir", str(explicit_normalizer_dir))


def test_pipeline_stages_include_export_without_changing_default_stop_after():
    args = pipeline.build_arg_parser().parse_args([])

    assert "export" in pipeline.PIPELINE_STAGES
    assert args.stop_after == "train"
    assert pipeline.selected_stages("convert", "export") == ("convert", "tasks", "normalizer", "train", "export")
    assert pipeline.selected_stages("export", "export") == ("export",)


def test_pipeline_train_save_dir_uses_schema_and_experiment_name(tmp_path):
    args = parse_schema_aware_pipeline_args(
        tmp_path,
        "--schema",
        "realtime_pose_stationary5_v1",
        "--experiment_name",
        "toy_exp",
    )

    train_args = pipeline.build_train_args(args)

    assert_arg_value(
        train_args,
        "--save_dir",
        str(tmp_path / "runs" / "realtime_pose_stationary5_v1" / "toy_exp"),
    )


def test_pipeline_task_defaults_rebuild_final_distribution_with_adjacent_rollout(tmp_path):
    args = parse_schema_aware_pipeline_args(tmp_path)
    task_args = pipeline.build_task_args(args)
    assert_arg_value(task_args, "--mask_policy", "full")
    assert_arg_value(task_args, "--patterns_per_source", "1")
    assert_arg_value(task_args, "--rollout_steps", "2")
    assert "--fixed_tracker_patterns" not in task_args


def test_pipeline_train_command_uses_single_process_curriculum(tmp_path):
    args = parse_schema_aware_pipeline_args(
        tmp_path,
        "--num_steps", "130000",
        "--rollout_steps", "9",
        "--tracker_mask_policy", "dynamic_categories",
        "--tracker_mask_categories", "all",
        "--resume_latest",
    )
    train_args = pipeline.build_train_args(args)
    assert_arg_value(train_args, "--num_steps", "130000")
    assert_arg_value(train_args, "--lr", "5e-05")
    assert_arg_value(train_args, "--rollout_steps", "9")
    assert_arg_value(train_args, "--rollout_h1_start_step", "30000")
    assert_arg_value(train_args, "--rollout_h2_start_step", "60000")
    assert_arg_value(train_args, "--rollout_h4_start_step", "70000")
    assert_arg_value(train_args, "--rollout_h8_start_step", "90000")
    assert_arg_value(train_args, "--tracker_mask_policy", "dynamic_categories")
    assert "--resume_checkpoint" in train_args
    assert "latest" in train_args


def test_pipeline_explicit_save_dir_overrides_experiment_name(tmp_path):
    explicit_save_dir = tmp_path / "explicit_runs"
    args = parse_schema_aware_pipeline_args(
        tmp_path,
        "--save_dir",
        str(explicit_save_dir),
        "--experiment_name",
        "toy_exp",
    )

    train_args = pipeline.build_train_args(args)

    assert_arg_value(train_args, "--save_dir", str(explicit_save_dir))


def test_pipeline_export_command_uses_schema_and_export_name(tmp_path):
    args = parse_schema_aware_pipeline_args(
        tmp_path,
        "--schema",
        "realtime_pose_stationary5_v1",
        "--export_name",
        "toy_export",
    )

    module, export_args = pipeline.build_stage_args("export", args)

    assert module == EXPORT_MODULE
    assert_arg_value(export_args, "--schema", "realtime_pose_stationary5_v1")
    assert_arg_value(
        export_args,
        "--output_dir",
        str(tmp_path / "output" / "realtime_pose_stationary5_v1" / "toy_export"),
    )
    assert_arg_value(export_args, "--diffusion_steps", str(args.diffusion_steps))
    assert_arg_value(export_args, "--normalizer_dir", str(pipeline.resolve_pipeline_normalizer_dir(args)))
    assert "--normalize_input" in export_args


def test_pipeline_export_uses_latest_normalizer_artifact(tmp_path):
    _, generated_root = write_artifact_roots_config(tmp_path)
    normalizer_root = generated_root / "normalizers" / REALTIME_POSE_SCHEMA_NAME / "toy_norm"
    latest_normalizer_dir = write_latest_normalizer_artifact(normalizer_root)
    args = parse_schema_aware_pipeline_args(
        tmp_path,
        "--schema",
        REALTIME_POSE_SCHEMA_NAME,
        "--export_name",
        "toy_export",
    )

    export_args = pipeline.build_export_args(args)

    assert_arg_value(export_args, "--normalizer_dir", str(latest_normalizer_dir))
    assert str(latest_normalizer_dir) != str(normalizer_root)


def test_pipeline_explicit_export_dir_overrides_export_name(tmp_path):
    explicit_export_dir = tmp_path / "explicit_export"
    args = parse_schema_aware_pipeline_args(
        tmp_path,
        "--export_dir",
        str(explicit_export_dir),
        "--export_name",
        "toy_export",
    )

    export_args = pipeline.build_export_args(args)

    assert_arg_value(export_args, "--output_dir", str(explicit_export_dir))


def test_pipeline_skip_export_does_not_run_stage(monkeypatch, tmp_path):
    calls: list[str] = []

    def fake_run_python_module(module: str, args: list[str], dry_run: bool) -> None:
        calls.append(module)

    monkeypatch.setattr(pipeline, "run_python_module", fake_run_python_module)
    args = parse_schema_aware_pipeline_args(
        tmp_path,
        "--skip_export",
        "--start_at",
        "export",
        "--stop_after",
        "export",
    )

    pipeline.run_pipeline(args)

    assert calls == []


def test_task_generation_resolver_uses_schema_aware_defaults_from_artifact_roots(tmp_path):
    config_path, generated_root = write_artifact_roots_config(tmp_path)

    args = task_generator.build_argument_parser().parse_args(
        [
            "--artifact_roots_config",
            str(config_path),
            "--schema",
            REALTIME_POSE_SCHEMA_NAME,
            "--source_set_name",
            "toy_source",
            "--task_set_name",
            "toy_tasks",
        ]
    )
    resolved = task_generator.resolve_task_generation_paths(args)

    assert resolved.source_dir == generated_root / "sources" / REALTIME_POSE_SCHEMA_NAME / "toy_source"
    assert resolved.output_dir == generated_root / "tasks" / REALTIME_POSE_SCHEMA_NAME / "toy_tasks"


def test_task_generation_resolver_keeps_explicit_paths(tmp_path):
    config_path, _ = write_artifact_roots_config(tmp_path)
    explicit_source_dir = tmp_path / "explicit_sources"
    explicit_output_dir = tmp_path / "explicit_tasks"

    args = task_generator.build_argument_parser().parse_args(
        [
            "--artifact_roots_config",
            str(config_path),
            "--source_dir",
            str(explicit_source_dir),
            "--output_dir",
            str(explicit_output_dir),
            "--schema",
            REALTIME_POSE_SCHEMA_NAME,
            "--source_set_name",
            "toy_source",
            "--task_set_name",
            "toy_tasks",
        ]
    )
    resolved = task_generator.resolve_task_generation_paths(args)

    assert resolved.source_dir == explicit_source_dir
    assert resolved.output_dir == explicit_output_dir


def test_normalizer_resolver_uses_schema_aware_defaults_from_artifact_roots(tmp_path):
    config_path, generated_root = write_artifact_roots_config(tmp_path)

    args = normalizer_computer.build_argument_parser().parse_args(
        [
            "--artifact_roots_config",
            str(config_path),
            "--schema",
            REALTIME_POSE_SCHEMA_NAME,
            "--task_set_name",
            "toy_tasks",
            "--normalizer_name",
            "toy_norm",
        ]
    )
    resolved = normalizer_computer.resolve_normalizer_paths(args)

    assert resolved.task_dir == generated_root / "tasks" / REALTIME_POSE_SCHEMA_NAME / "toy_tasks"
    assert resolved.output_dir == generated_root / "normalizers" / REALTIME_POSE_SCHEMA_NAME / "toy_norm"


def test_normalizer_resolver_keeps_explicit_paths(tmp_path):
    config_path, _ = write_artifact_roots_config(tmp_path)
    explicit_task_dir = tmp_path / "explicit_tasks"
    explicit_output_dir = tmp_path / "explicit_normalizer"

    args = normalizer_computer.build_argument_parser().parse_args(
        [
            "--artifact_roots_config",
            str(config_path),
            "--task_dir",
            str(explicit_task_dir),
            "--output_dir",
            str(explicit_output_dir),
            "--schema",
            REALTIME_POSE_SCHEMA_NAME,
            "--task_set_name",
            "toy_tasks",
            "--normalizer_name",
            "toy_norm",
        ]
    )
    resolved = normalizer_computer.resolve_normalizer_paths(args)

    assert resolved.task_dir == explicit_task_dir
    assert resolved.output_dir == explicit_output_dir


def test_task_and_normalizer_metadata_record_schema_aware_roots(tmp_path):
    config_path, generated_root = write_artifact_roots_config(tmp_path)
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    source_dir = generated_root / "sources" / schema.name / "toy_source"
    task_root = generated_root / "tasks" / schema.name / "toy_tasks"
    normalizer_root = generated_root / "normalizers" / schema.name / "toy_norm"
    write_toy_source_dataset(source_dir, schema_name=schema.name)

    task_generator.main(
        [
            "--artifact_roots_config",
            str(config_path),
            "--source_set_name",
            "toy_source",
            "--task_set_name",
            "toy_tasks",
            "--schema",
            schema.name,
            "--split_dir",
            "",
            "--splits",
            "train",
            "--samples_per_source",
            "1",
            "--overwrite",
        ]
    )
    task_output_dir = read_latest_pointer(task_root, kind="tasks")
    assert task_output_dir is not None
    marker = json.loads((task_output_dir / task_generator.TASK_OUTPUT_MARKER).read_text(encoding="utf-8"))
    manifest_entry = json.loads((task_output_dir / "train" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    latest_task = json.loads((task_root / "latest_tasks.json").read_text(encoding="utf-8"))

    for payload in (marker, manifest_entry, latest_task):
        assert payload["schema_canonical_name"] == schema.canonical_name
        assert payload["generated_root"] == str(generated_root)
        assert payload["source_set_name"] == "toy_source"
        assert payload["task_set_name"] == "toy_tasks"
        assert payload["source_dir"] == str(source_dir)
        assert payload["output_dir"] == str(task_output_dir)

    normalizer_computer.main(
        [
            "--artifact_roots_config",
            str(config_path),
            "--task_set_name",
            "toy_tasks",
            "--normalizer_name",
            "toy_norm",
            "--schema",
            schema.name,
            "--split",
            "train",
            "--windows_per_source",
            "2",
            "--convergence_windows_per_source",
            "4",
            "--check_convergence",
            "false",
            "--overwrite",
        ]
    )
    normalizer_output_dir = read_latest_pointer(normalizer_root, kind="normalizer")
    assert normalizer_output_dir is not None
    normalizer_meta = json.loads((normalizer_output_dir / "normalizer_meta.json").read_text(encoding="utf-8"))
    latest_normalizer = json.loads((normalizer_root / "latest_normalizer.json").read_text(encoding="utf-8"))

    for payload in (normalizer_meta, latest_normalizer):
        assert payload["schema_canonical_name"] == schema.canonical_name
        assert payload["generated_root"] == str(generated_root)
        assert payload["task_set_name"] == "toy_tasks"
        assert payload["normalizer_name"] == "toy_norm"
        assert payload["task_dir"] == str(task_output_dir)
        assert payload["output_dir"] == str(normalizer_output_dir)
        for key, value in stationary_label_metadata().items():
            assert payload[key] == value
def test_pipeline_allows_partial_conversion_by_default(tmp_path):
    args = parse_pipeline_args(tmp_path)

    convert_args = pipeline.build_convert_args(args)

    assert "--allow_partial" in convert_args


def test_pipeline_can_disable_partial_conversion_for_strict_runs(tmp_path):
    args = parse_pipeline_args(tmp_path, "--no-allow_partial")

    convert_args = pipeline.build_convert_args(args)

    assert "--allow_partial" not in convert_args
