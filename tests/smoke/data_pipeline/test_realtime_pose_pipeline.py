from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from data_loaders.sensor_masking import REALTIME_POSE_SCHEMA_NAME, get_schema_spec
from scripts import run_realtime_pose_pipeline as pipeline
from utils.run_dirs import write_latest_pointer


CONVERT_MODULE = "data_converter.amass_to_realtime_pose"
TASK_MODULE = "data_loaders.generate_realtime_pose_tasks"
NORMALIZER_MODULE = "data_loaders.compute_realtime_pose_normalizer"


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
    task_dir = task_root / "20260101_000000_rtp_tasks"
    split_dir = task_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "manifest.jsonl").write_text("", encoding="utf-8")
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
        json.dumps({"schema_name": schema.name}, ensure_ascii=False),
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
