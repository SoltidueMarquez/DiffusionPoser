from __future__ import annotations

import json
from argparse import Namespace

import numpy as np
import pytest
import torch

from data_loaders.compute_realtime_pose_normalizer import compute_realtime_pose_normalizer
from data_loaders.generate_realtime_pose_tasks import main as generate_tasks
from data_loaders.realtime_pose_dataset import RealtimePoseTaskDataset
from data_loaders.realtime_pose_geometry import (
    decode_target_head_rotations_np,
    derive_hip_height_from_head_np,
    extract_forward_yaw_np,
    extract_forward_yaw_torch,
    resolve_root_head_reference_np,
)
from data_loaders.realtime_pose_kinematics import JOINT_INDEX, rotation_6d_to_matrix_np
from data_loaders.sensor_masking import (
    REALTIME_POSE_TARGET_DIM,
    TRACKER_FEATURE_DIM,
)
from data_loaders.tracker_timeline import (
    build_tracker_timeline,
    candidate_starts_by_scenario,
    compute_missing_age,
)
from tests.smoke.realtime_pose_fixtures import write_toy_source_dataset
from utils.normalizer import RealtimePoseNormalizer
from utils.run_dirs import read_latest_pointer


def test_missing_age_and_all_five_scenarios_are_deterministic():
    configured = np.ones((80, 6), dtype=bool)
    measured = configured.copy()
    measured[5:70, 1] = False
    age = compute_missing_age(configured, measured)
    assert age[4, 1] == 0
    assert age[5, 1] == 1
    assert age[64, 1] == 60
    assert age[69, 1] == 60
    assert age[70, 1] == 0

    timeline_a = build_tracker_timeline("subject/clip", 1000, global_seed=17)
    timeline_b = build_tracker_timeline("subject/clip", 1000, global_seed=17)
    np.testing.assert_array_equal(timeline_a.configured, timeline_b.configured)
    np.testing.assert_array_equal(timeline_a.measured_valid, timeline_b.measured_valid)
    np.testing.assert_array_equal(timeline_a.missing_age, timeline_b.missing_age)
    candidates = candidate_starts_by_scenario(timeline_a)
    assert all(candidates[name] for name in ("fixed_six", "fixed_three", "three_to_six", "six_to_three", "dropout"))
    # 重叠窗口中的同一绝对帧必须携带相同计数。
    np.testing.assert_array_equal(timeline_a.window(100).missing_age[1:], timeline_a.window(101).missing_age[:-1])


def test_head_yaw_numpy_torch_share_causal_vertical_fallback():
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 4, axis=0)
    rotations[1, :, 2] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    rotations[2, :, 2] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    rotations[3, :, 2] = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
    numpy_yaw = extract_forward_yaw_np(rotations, initial_yaw=0.25)
    torch_yaw = extract_forward_yaw_torch(torch.from_numpy(rotations), initial_yaw=0.25)
    np.testing.assert_allclose(torch_yaw.numpy(), numpy_yaw, atol=1e-7)
    assert numpy_yaw[1] == pytest.approx(numpy_yaw[0])
    assert numpy_yaw[3] == pytest.approx(numpy_yaw[2])


def test_140d_task_dataset_normalizer_and_root_resolver(tmp_path):
    source_dir = tmp_path / "source"
    task_root = tmp_path / "tasks"
    normalizer_root = tmp_path / "normalizer"
    write_toy_source_dataset(source_dir, frame_count=1000)
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
            "--samples_per_file",
            "1",
            "--seed",
            "17",
        ]
    )
    assert counts == {"train": 5}
    task_dir = read_latest_pointer(task_root, "tasks")
    assert task_dir is not None
    manifest_path = task_dir / "train" / "manifest.jsonl"
    entries = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert {entry["scenario"] for entry in entries} == {
        "fixed_six",
        "fixed_three",
        "three_to_six",
        "six_to_three",
        "dropout",
    }

    dataset = RealtimePoseTaskDataset(task_dir, split="train", normalize_input=False)
    item = dataset[0]
    assert tuple(item["pose_history"].shape) == (60, REALTIME_POSE_TARGET_DIM)
    assert tuple(item["tracker_window"].shape) == (61, 6, TRACKER_FEATURE_DIM)
    assert tuple(item["x"].shape) == (REALTIME_POSE_TARGET_DIM,)
    assert tuple(item["known_mask"].shape) == (REALTIME_POSE_TARGET_DIM,)
    invalid = ~item["measured_valid"]
    assert np.all(item["tracker_window"].numpy()[..., :9][invalid.numpy()] == 0.0)

    task_path = manifest_path.parent / entries[0]["task_path"]
    with np.load(task_path, allow_pickle=False) as task:
        target = task["current_target"]
        rest_rot = rotation_6d_to_matrix_np(task["joint_rest_local_rotations_6d"])
        rotations, root_yaw_head = decode_target_head_rotations_np(target, rest_rot)
        head_height = float(task["tracker_window"][-1, 0, 1])
        derived_height = float(
            derive_hip_height_from_head_np(rotations, task["joint_offsets_parent"], head_height)
        )
        assert derived_height == pytest.approx(float(task["target_hip_height"]), abs=1e-4)
        root, height, joints = resolve_root_head_reference_np(
            rotations,
            float(root_yaw_head),
            task["joint_offsets_parent"],
            head_height,
            hip_measured_valid=False,
        )
        assert root[1] == 0.0
        assert height == pytest.approx(float(task["target_hip_height"]), abs=1e-4)
        np.testing.assert_allclose(joints[JOINT_INDEX["head"]], [0.0, head_height, 0.0], atol=1e-5)

    meta = compute_realtime_pose_normalizer(
        Namespace(
            task_dir=str(task_dir),
            output_dir=str(normalizer_root),
            split="train",
            eps=1e-8,
            run_name="smoke",
            overwrite=False,
        )
    )
    normalizer = RealtimePoseNormalizer(meta["output_dir"])
    assert tuple(normalizer.pose_mean.shape) == (140,)
    assert tuple(normalizer.tracker_mean.shape) == (6, 9)
    normalized_dataset = RealtimePoseTaskDataset(
        task_dir,
        split="train",
        normalizer_dir=meta["output_dir"],
        normalize_input=True,
    )
    normalized = normalized_dataset[0]
    invalid = ~normalized["measured_valid"]
    assert np.all(normalized["tracker_window"].numpy()[..., :9][invalid.numpy()] == 0.0)
