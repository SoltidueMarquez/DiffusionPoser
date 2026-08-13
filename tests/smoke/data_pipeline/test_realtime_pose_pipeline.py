from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import run_realtime_pose_pipeline as pipeline


def _args(tmp_path: Path, *extra: str):
    return pipeline.build_arg_parser().parse_args(
        [
            "--amass_dir", str(tmp_path / "AMASS"),
            "--smpl_model_dir", str(tmp_path / "body_models"),
            "--source_dir", str(tmp_path / "sources"),
            "--task_dir", str(tmp_path / "tasks"),
            "--normalizer_dir", str(tmp_path / "normalizer"),
            "--split_dir", str(tmp_path / "splits"),
            "--save_dir", str(tmp_path / "runs"),
            "--splits", "train",
            *extra,
        ]
    )


def test_pipeline_stops_at_first_failed_stage(monkeypatch, tmp_path):
    calls: list[str] = []

    def fail(module: str, args: list[str], dry_run: bool) -> None:
        del args, dry_run
        calls.append(module)
        raise subprocess.CalledProcessError(7, module)

    monkeypatch.setattr(pipeline, "run_python_module", fail)
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.run_pipeline(_args(tmp_path, "--stop_after", "normalizer"))
    assert calls == ["data_converter.amass_to_realtime_pose"]


def test_pipeline_passes_parallel_converter_args(tmp_path):
    args = _args(
        tmp_path,
        "--convert_num_workers", "3",
        "--convert_worker_torch_threads", "2",
    )
    values = pipeline.build_convert_args(args)
    assert values[values.index("--num_workers") + 1] == "3"
    assert values[values.index("--worker_torch_threads") + 1] == "2"


def test_pipeline_passes_joint_horizon_training_args(tmp_path):
    args = _args(
        tmp_path,
        "--base_windows_per_source", "7",
        "--shard_size", "128",
        "--history_noise_prob", "0.7",
        "--rotation_velocity_loss_weight", "1.5",
        "--contact_slide_loss_weight", "0.2",
        "--scenario_weights", "5", "4", "3", "2", "1",
    )
    task_args = pipeline.build_task_args(args)
    train_args = pipeline.build_train_args(args)
    assert task_args[task_args.index("--base_windows_per_source") + 1] == "7"
    assert task_args[task_args.index("--shard_size") + 1] == "128"
    assert train_args[train_args.index("--history_noise_prob") + 1] == "0.7"
    assert train_args[train_args.index("--rotation_velocity_loss_weight") + 1] == "1.5"
    assert train_args[train_args.index("--contact_slide_loss_weight") + 1] == "0.2"
    assert "--future_leg_loss_weight" not in train_args
    weights = train_args.index("--scenario_weights")
    assert train_args[weights + 1 : weights + 6] == ["5.0", "4.0", "3.0", "2.0", "1.0"]
