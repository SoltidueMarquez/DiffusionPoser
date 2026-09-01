from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from data_loaders.compute_realtime_pose_normalizer import (
    compute_realtime_pose_normalizer,
)
from data_loaders.generate_realtime_pose_tasks import (
    build_task_bundle_row,
    compute_source_joint_rotations_world,
    generate_realtime_pose_tasks,
    shard_fields,
)
from data_loaders.realtime_pose_dataset import (
    RealtimePoseBatchSampler,
    RealtimePoseTaskDataset,
)
from data_loaders.realtime_pose_kinematics import (
    rotation_6d_forward_up_np,
    rotation_6d_to_matrix_np,
)
from data_loaders.realtime_pose_predictor_dataset import RealtimePosePredictorSequenceDataset
from data_loaders.realtime_pose_predictor_features import (
    build_predictor_sparse_availability_mask_np,
    build_predictor_sparse_features_np,
)
from data_loaders.rpm_hand_dropout import (
    build_rpm_dit_training_availability,
)
from data_loaders.sensor_masking import (
    STATIC_OPTIONAL_TRACKER_MASKS,
    TRAIN_TRACKER_ENDPOINTS,
)
from tests.smoke.realtime_pose_fixtures import (
    build_toy_realtime_source,
    write_toy_source_dataset,
)


def test_new_task_store_fields_and_shapes():
    expected = {
        "motion_context_clean": (10, 144),
        "core_tracker_context_clean": (11, 54),
        "current_pose_target_clean": (144,),
        "current_tracker_continuous": (6, 9),
    }
    schema = shard_fields()
    for name, shape in expected.items():
        assert schema[name][0] == shape


def test_task_row_does_not_read_future_tracker():
    source = build_toy_realtime_source(frame_count=64)
    rotations = compute_source_joint_rotations_world(source)
    row = build_task_bundle_row(source, rotations, source["root_yaw"], 20, 1)
    changed = {name: np.asarray(value).copy() for name, value in source.items() if name != "metadata"}
    changed["tracker_pos_world"][21:] += 1000.0
    changed["tracker_rot_world_6d"][21:] *= -1.0
    second = build_task_bundle_row(changed, rotations, source["root_yaw"], 20, 1)
    np.testing.assert_array_equal(
        row["core_tracker_context_clean"], second["core_tracker_context_clean"]
    )


def test_predictor_54d_feature_order_and_so3_relative_rotation():
    tracker = np.zeros((12, 6, 9), dtype=np.float32)
    angles = np.linspace(0.0, 0.55, 12)
    rotations = np.zeros((12, 6, 3, 3), dtype=np.float64)
    for time, angle in enumerate(angles):
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.asarray([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
        rotations[time] = rotation
        tracker[time, :, :3] = np.asarray([time, 2 * time, -time], dtype=np.float32)
    tracker[..., 3:9] = rotation_6d_forward_up_np(rotations)
    sparse = build_predictor_sparse_features_np(tracker)
    assert sparse.shape == (11, 54)
    np.testing.assert_allclose(sparse[0, :18], tracker[1, :3, 3:9].reshape(-1))
    expected_relative = rotations[0, :3].transpose(0, 2, 1) @ rotations[1, :3]
    actual_relative = rotation_6d_to_matrix_np(sparse[0, 18:36].reshape(3, 6))
    np.testing.assert_allclose(actual_relative, expected_relative, atol=1e-6)
    np.testing.assert_allclose(sparse[0, 36:45], tracker[1, :3, :3].reshape(-1))
    np.testing.assert_allclose(
        sparse[0, 45:54], (tracker[1, :3, :3] - tracker[0, :3, :3]).reshape(-1)
    )


def test_training_sampler_uses_all_eight_optional_tracker_masks():
    assert TRAIN_TRACKER_ENDPOINTS == STATIC_OPTIONAL_TRACKER_MASKS
    assert len(TRAIN_TRACKER_ENDPOINTS) == 8


def test_training_sampler_produces_all_eight_masks_equally():
    class _Dataset:
        indices_by_shard = [list(range(16))]

        def __len__(self):
            return 16

    sampler = RealtimePoseBatchSampler(
        _Dataset(), batch_size=4, seed=10, shuffle=True, drop_last=True
    )
    config_indices = [
        request.config_index for batch in sampler for request in batch
    ]
    assert set(config_indices) == set(range(8))
    assert all(config_indices.count(index) == 2 for index in range(8))


def test_materialized_task_normalizer_and_predictor_sequence_contract(
    tmp_path, monkeypatch
):
    source_dir = tmp_path / "source"
    write_toy_source_dataset(source_dir, frame_count=70)
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "train.txt").write_text(
        "ACCAD/toy_realtime\n", encoding="utf-8"
    )
    task_dir = tmp_path / "tasks"
    counts = generate_realtime_pose_tasks(
        SimpleNamespace(
            source_dir=str(source_dir),
            output_dir=str(task_dir),
            split_dir=str(split_dir),
            splits=["train"],
            seq_len=11,
            base_windows_per_source=2,
            shard_size=2,
            short_source_policy="error",
            limit=0,
            seed=10,
            overwrite=False,
        )
    )
    assert counts == {"train": 2}

    normalizer_dir = tmp_path / "normalizer"
    compute_realtime_pose_normalizer(
        SimpleNamespace(
            task_dir=str(task_dir),
            output_dir=str(normalizer_dir),
            split="train",
            eps=1e-8,
            overwrite=False,
        )
    )
    dataset = RealtimePoseTaskDataset(
        task_dir,
        split="train",
        normalizer_dir=normalizer_dir,
    )
    sample = dataset[0]
    assert sample["x"].shape == (144,)
    assert sample["motion_context"].shape == (10, 144)
    assert sample["core_tracker_context"].shape == (11, 54)
    assert sample["current_tracker_raw"].shape == (6, 10)

    predictor_dataset = RealtimePosePredictorSequenceDataset(
        source_dir=source_dir,
        split_dir=split_dir,
        split="train",
        windows_per_source=1,
        seed=10,
    )
    assert predictor_dataset.resident_bytes > 0
    assert not predictor_dataset.sequences[0].joint_rotations_world_6d.flags.writeable

    def reject_disk_read(*args, **kwargs):
        raise AssertionError("Predictor __getitem__ 不应再次读取 source 文件。")

    monkeypatch.setattr(np, "load", reject_disk_read)
    predictor_sample = predictor_dataset[0]
    assert predictor_sample["joint_rotations_world_6d"].shape == (52, 24, 6)
    assert predictor_sample["tracker_positions_world"].shape == (52, 6, 3)
    assert predictor_sample["tracker_available"].shape == (52, 6)


def test_task_and_predictor_datasets_apply_same_deterministic_hand_dropout(
    tmp_path, monkeypatch
):
    source_dir = tmp_path / "source"
    write_toy_source_dataset(source_dir, frame_count=70)
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "train.txt").write_text(
        "ACCAD/toy_realtime\n", encoding="utf-8"
    )
    task_dir = tmp_path / "tasks"
    generate_realtime_pose_tasks(
        SimpleNamespace(
            source_dir=str(source_dir),
            output_dir=str(task_dir),
            split_dir=str(split_dir),
            splits=["train"],
            seq_len=11,
            base_windows_per_source=1,
            shard_size=1,
            short_source_policy="error",
            limit=0,
            seed=10,
            overwrite=False,
        )
    )
    monkeypatch.setattr(
        "data_loaders.realtime_pose_dataset.stable_rpm_hand_dropout_seed",
        lambda *args: 127,
    )
    task_dataset = RealtimePoseTaskDataset(
        task_dir,
        split="train",
        normalize_input=False,
        rpm_hand_dropout=True,
    )
    task_sample = task_dataset[0]
    expected_sparse_mask = build_predictor_sparse_availability_mask_np(
        build_rpm_dit_training_availability(seed=127)
    )
    assert not task_sample["tracker_available"][1:3].any()
    assert not task_sample["current_tracker_raw"][1:3].any()
    assert not task_sample["core_tracker_context"].numpy()[~expected_sparse_mask].any()

    monkeypatch.setattr(
        "data_loaders.realtime_pose_predictor_dataset.stable_rpm_hand_dropout_seed",
        lambda *args: 127,
    )
    predictor_dataset = RealtimePosePredictorSequenceDataset(
        source_dir=source_dir,
        split_dir=split_dir,
        split="train",
        windows_per_source=1,
        seed=10,
        rpm_hand_dropout=True,
    )
    predictor_sample = predictor_dataset[0]
    assert not predictor_sample["tracker_available"][1:41, 1].any()
    assert not predictor_sample["tracker_available"][7:25, 2].any()
