from __future__ import annotations

import json
import shutil
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from data_loaders.compute_realtime_pose_normalizer import compute_realtime_pose_normalizer
from data_loaders.generate_realtime_pose_tasks import main as generate_tasks
from data_loaders.realtime_pose_dataset import (
    RealtimePoseBatchSampler,
    RealtimePoseTaskDataset,
    TaskRequest,
)
from data_loaders.realtime_pose_geometry import (
    decode_target_head_rotations_np,
    derive_hip_height_from_head_np,
    extract_forward_yaw_np,
    extract_forward_yaw_torch,
    resolve_root_head_reference_np,
)
from data_loaders.realtime_pose_kinematics import JOINT_INDEX, rotation_6d_to_matrix_np
from data_loaders.sensor_masking import REALTIME_POSE_TARGET_DIM, TRACKER_FEATURE_DIM
from data_loaders.tracker_timeline import classify_tracker_window, compute_missing_age
from tests.smoke.realtime_pose_fixtures import write_toy_source_dataset
from utils.normalizer import RealtimePoseNormalizer
from utils.run_dirs import read_latest_pointer


def _generate_store(source_dir: Path, task_root: Path, base_windows: int = 5) -> Path:
    counts = generate_tasks(
        [
            "--source_dir",
            str(source_dir),
            "--output_dir",
            str(task_root),
            "--splits",
            "train",
            "--split_dir",
            "",
            "--base_windows_per_source",
            str(base_windows),
            "--max_rollout_steps",
            "4",
            "--shard_size",
            "2",
            "--seed",
            "17",
        ]
    )
    assert counts == {"train": base_windows * 2}
    task_dir = read_latest_pointer(task_root, "tasks")
    assert task_dir is not None
    return task_dir


def _duplicate_source_and_reorder_manifest(source_dir: Path, reverse: bool) -> None:
    first = source_dir / "ACCAD" / "toy_realtime.npz"
    second = source_dir / "CMU" / "toy_realtime_2.npz"
    second.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(first, second)
    entries = [
        {
            "status": "converted",
            "source_relative_path": "ACCAD/toy_realtime.npz",
            "stablemotion_split_key": "ACCAD/toy_realtime",
            "output_path": "ACCAD/toy_realtime.npz",
            "frames": 90,
            "target_fps": 60.0,
            "is_mirrored": False,
        },
        {
            "status": "converted",
            "source_relative_path": "CMU/toy_realtime_2.npz",
            "stablemotion_split_key": "CMU/toy_realtime_2",
            "output_path": "CMU/toy_realtime_2.npz",
            "frames": 90,
            "target_fps": 60.0,
            "is_mirrored": False,
        },
    ]
    if reverse:
        entries.reverse()
    with (source_dir / "manifest.jsonl").open("w", encoding="utf-8") as file:
        for entry in entries:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _compute_normalizer(task_dir: Path, output_root: Path) -> Path:
    result = compute_realtime_pose_normalizer(
        Namespace(
            task_dir=str(task_dir),
            output_dir=str(output_root),
            split="train",
            eps=1e-8,
            run_name="smoke",
        )
    )
    return Path(result["output_dir"])


def test_generation_plan_shards_and_normalizer_are_deterministic(tmp_path):
    source_a = tmp_path / "source_a"
    source_b = tmp_path / "source_b"
    write_toy_source_dataset(source_a, frame_count=90)
    write_toy_source_dataset(source_b, frame_count=90)
    _duplicate_source_and_reorder_manifest(source_a, reverse=False)
    _duplicate_source_and_reorder_manifest(source_b, reverse=True)
    task_a = _generate_store(source_a, tmp_path / "tasks_a")
    task_b = _generate_store(source_b, tmp_path / "tasks_b")

    assert (task_a / "generation_plan.sha256").read_text() == (
        task_b / "generation_plan.sha256"
    ).read_text()
    assert (task_a / "generation_plan.jsonl").read_bytes() == (
        task_b / "generation_plan.jsonl"
    ).read_bytes()
    npy_a = sorted(path.relative_to(task_a / "train") for path in (task_a / "train").rglob("*.npy"))
    npy_b = sorted(path.relative_to(task_b / "train") for path in (task_b / "train").rglob("*.npy"))
    assert npy_a == npy_b
    for relative in npy_a:
        np.testing.assert_array_equal(np.load(task_a / "train" / relative), np.load(task_b / "train" / relative))

    normalizer_a = RealtimePoseNormalizer(_compute_normalizer(task_a, tmp_path / "normalizer_a"))
    normalizer_b = RealtimePoseNormalizer(_compute_normalizer(task_b, tmp_path / "normalizer_b"))
    torch.testing.assert_close(normalizer_a.pose_mean, normalizer_b.pose_mean)
    torch.testing.assert_close(normalizer_a.pose_std, normalizer_b.pose_std)
    torch.testing.assert_close(normalizer_a.tracker_mean, normalizer_b.tracker_mean)
    torch.testing.assert_close(normalizer_a.tracker_std, normalizer_b.tracker_std)


def test_five_task_configs_are_continuous_across_four_rollout_steps(tmp_path):
    source_dir = tmp_path / "source"
    write_toy_source_dataset(source_dir, frame_count=90)
    _duplicate_source_and_reorder_manifest(source_dir, reverse=False)
    task_dir = _generate_store(source_dir, tmp_path / "tasks", base_windows=1)
    dataset = RealtimePoseTaskDataset(task_dir, split="train", normalize_input=False)
    assert not hasattr(dataset, "entries")
    assert dataset.task_id_at(0)
    expected = ("fixed_six", "fixed_three", "three_to_six", "six_to_three", "dropout")
    for config_index, scenario in enumerate(expected):
        item = dataset[TaskRequest(0, config_index, 4)]
        sequence = [item, *item["rollout"]]
        assert all(step["scenario"] == scenario for step in sequence)
        for step in sequence:
            assert step["configured"][:, 0].all()
            assert step["measured_valid"][:, 0].all()
            assert classify_tracker_window(
                step["configured"].numpy(), step["measured_valid"].numpy()
            ) == scenario
            assert int(step["missing_age"].max()) <= 60
        for left, right in zip(sequence, sequence[1:]):
            torch.testing.assert_close(left["configured"][1:], right["configured"][:-1])
            torch.testing.assert_close(left["measured_valid"][1:], right["measured_valid"][:-1])
            torch.testing.assert_close(left["missing_age"][1:], right["missing_age"][:-1])


def test_shard_dataset_mask_geometry_normalizer_and_root_resolver(tmp_path):
    source_dir = tmp_path / "source"
    write_toy_source_dataset(source_dir, frame_count=90)
    _duplicate_source_and_reorder_manifest(source_dir, reverse=False)
    task_dir = _generate_store(source_dir, tmp_path / "tasks", base_windows=1)
    dataset = RealtimePoseTaskDataset(task_dir, split="train", normalize_input=False)
    item = dataset[TaskRequest(0, 4, 1)]
    assert item["pose_history"].shape == (60, REALTIME_POSE_TARGET_DIM)
    assert item["tracker_window"].shape == (61, 6, TRACKER_FEATURE_DIM)
    invalid = ~item["measured_valid"]
    assert np.all(item["tracker_window"].numpy()[..., :9][invalid.numpy()] == 0.0)
    assert torch.all(item["known_target"][~item["known_mask"]] == 0.0)

    fixed = dataset[TaskRequest(0, 0, 1)]
    target = fixed["x"].numpy()
    rest_rot = rotation_6d_to_matrix_np(fixed["joint_rest_local_rotations_6d"].numpy())
    rotations, root_yaw_head = decode_target_head_rotations_np(target, rest_rot)
    head_height = float(fixed["tracker_window"][-1, 0, 1])
    derived_height = float(
        derive_hip_height_from_head_np(rotations, fixed["joint_offsets_parent"].numpy(), head_height)
    )
    assert derived_height == pytest.approx(float(fixed["target_hip_height"]), abs=1e-4)
    root, height, joints = resolve_root_head_reference_np(
        rotations,
        float(root_yaw_head),
        fixed["joint_offsets_parent"].numpy(),
        head_height,
        hip_measured_valid=False,
    )
    assert root[1] == 0.0
    assert height == pytest.approx(float(fixed["target_hip_height"]), abs=1e-4)
    np.testing.assert_allclose(joints[JOINT_INDEX["head"]], [0.0, head_height, 0.0], atol=1e-5)

    normalizer_dir = _compute_normalizer(task_dir, tmp_path / "normalizer")
    normalized = RealtimePoseTaskDataset(
        task_dir, split="train", normalizer_dir=normalizer_dir, normalize_input=True
    )[TaskRequest(0, 4, 1)]
    invalid = ~normalized["measured_valid"]
    assert np.all(normalized["tracker_window"].numpy()[..., :9][invalid.numpy()] == 0.0)
    assert torch.all(normalized["known_target"][~normalized["known_mask"]] == 0.0)


def test_batch_sampler_decides_rollout_before_dataset_reads_future_steps(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    write_toy_source_dataset(source_dir, frame_count=90)
    _duplicate_source_and_reorder_manifest(source_dir, reverse=False)
    task_dir = _generate_store(source_dir, tmp_path / "tasks", base_windows=2)
    dataset = RealtimePoseTaskDataset(task_dir, split="train", normalize_input=False)
    original = dataset._step_to_item
    read_steps: list[int] = []

    def recording_read(*args, **kwargs):
        read_steps.append(int(kwargs["step"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(dataset, "_step_to_item", recording_read)
    no_rollout = RealtimePoseBatchSampler(dataset, 2, 7, [1] * 5, 4, 0.0, False, False)
    requests = next(iter(no_rollout))
    assert all(request.rollout_steps == 1 for request in requests)
    for request in requests:
        dataset[request]
    assert read_steps == [0, 0]

    read_steps.clear()
    full_rollout = RealtimePoseBatchSampler(dataset, 2, 7, [9, 1, 1, 1, 1], 4, 1.0, False, False)
    requests = next(iter(full_rollout))
    assert all(request.rollout_steps == 4 for request in requests)
    for request in requests:
        item = dataset[request]
        assert len(item["rollout"]) == 3
        assert all("pose_history" not in step for step in item["rollout"])
    assert read_steps == [0, 1, 2, 3, 0, 1, 2, 3]


def test_missing_age_and_head_yaw_semantics_remain_stable():
    configured = np.ones((80, 6), dtype=bool)
    measured = configured.copy()
    measured[5:70, 1] = False
    age = compute_missing_age(configured, measured)
    assert age[5, 1] == 1
    assert age[64, 1] == age[69, 1] == 60
    assert age[70, 1] == 0

    rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 4, axis=0)
    rotations[1, :, 2] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    rotations[2, :, 2] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    rotations[3, :, 2] = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
    numpy_yaw = extract_forward_yaw_np(rotations, initial_yaw=0.25)
    torch_yaw = extract_forward_yaw_torch(torch.from_numpy(rotations), initial_yaw=0.25)
    np.testing.assert_allclose(torch_yaw.numpy(), numpy_yaw, atol=1e-7)
    assert numpy_yaw[1] == pytest.approx(numpy_yaw[0])
    assert numpy_yaw[3] == pytest.approx(numpy_yaw[2])
