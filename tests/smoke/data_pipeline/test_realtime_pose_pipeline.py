from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import run_realtime_pose_pipeline as pipeline
from data_loaders.generate_realtime_pose_tasks import shard_fields
from data_loaders.sensor_masking import TRACKER_FEATURE_DIM, TRACKER_PATTERN_CATEGORIES
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
            "--save_dir",
            str(tmp_path / "runs"),
            "--splits",
            "train",
            *extra,
        ]
    )


def write_usable_source_manifest(source_dir: Path) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "status": "converted",
        "source_relative_path": "toy.npz",
        "output_path": "toy.npz",
    }
    (source_dir / "manifest.jsonl").write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")


def write_latest_task_artifact(task_root: Path, split: str = "train") -> Path:
    task_dir = task_root / "20260101_000000_rtp_tasks"
    split_dir = task_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "task_store.json").write_text(
        json.dumps(
            {
                "generation_plan_hash": "temporary",
                "split": split,
                "sample_count": 1,
                "source_count": 1,
                "two_point_phase_counts": {"dropout": 1, "reconnect": 0},
                "config_names": list(TRACKER_PATTERN_CATEGORIES),
                "tracker_feature_dim": TRACKER_FEATURE_DIM,
                "schema_fields": sorted(shard_fields()),
                "shards": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    write_latest_pointer(
        root_dir=task_root,
        kind="tasks",
        output_dir=task_dir,
        metadata={"splits": [split]},
    )
    return task_dir


def write_latest_normalizer_artifact(normalizer_root: Path) -> Path:
    normalizer_dir = normalizer_root / "20260101_000000_normalizer"
    normalizer_dir.mkdir(parents=True, exist_ok=True)
    (normalizer_dir / "pose_mean.pt").write_bytes(b"mean")
    (normalizer_dir / "pose_scale.pt").write_bytes(b"scale")
    (normalizer_dir / "tracker_mean.pt").write_bytes(b"mean")
    (normalizer_dir / "tracker_std.pt").write_bytes(b"std")
    (normalizer_dir / "head_path_xz_mean.pt").write_bytes(b"mean")
    (normalizer_dir / "head_path_xz_std.pt").write_bytes(b"std")
    (normalizer_dir / "head_height_mean.pt").write_bytes(b"mean")
    (normalizer_dir / "head_height_std.pt").write_bytes(b"std")
    (normalizer_dir / "normalizer_meta.json").write_text("{}\n", encoding="utf-8")
    write_latest_pointer(
        root_dir=normalizer_root,
        kind="normalizer",
        output_dir=normalizer_dir,
        metadata={},
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


def test_pipeline_passes_new_task_store_and_training_sampling_args(tmp_path):
    args = parse_pipeline_args(
        tmp_path,
        "--base_windows_per_source",
        "7",
        "--shard_size",
        "128",
        "--history_noise_prob",
        "0.7",
        "--scenario_weights",
        "5",
        "4",
        "3",
        "2",
        "1",
    )
    task_args = pipeline.build_task_args(args)
    train_args = pipeline.build_train_args(args)
    assert task_args[task_args.index("--base_windows_per_source") + 1] == "7"
    assert task_args[task_args.index("--shard_size") + 1] == "128"
    assert "--samples_per_file" not in task_args
    assert train_args[train_args.index("--history_noise_prob") + 1] == "0.7"
    assert "--rollout_steps" not in train_args
    weights = train_args.index("--scenario_weights")
    assert train_args[weights + 1 : weights + 6] == ["5.0", "4.0", "3.0", "2.0", "1.0"]
    invalid_reliability_options = {
        "--d_warm_pos",
        "--d_warm_rot",
        "--d_hard",
        "--tracker_duration_cap",
    }
    assert invalid_reliability_options.isdisjoint(train_args)
