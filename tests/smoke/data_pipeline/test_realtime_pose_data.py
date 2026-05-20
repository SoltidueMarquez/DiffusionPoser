from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import data_converter.amass_to_realtime_pose as amass_converter
from data_converter.amass_to_realtime_pose import build_realtime_pose_features
from data_loaders.compute_realtime_pose_normalizer import compute_realtime_pose_normalizer
from data_loaders.generate_realtime_pose_tasks import main as generate_realtime_pose_tasks_main
from data_loaders.realtime_pose_dataset import (
    RealtimePoseTaskDataset,
    encode_realtime_pose_features,
    load_materialized_task_npz,
    load_realtime_task_arrays,
)
from data_loaders.sensor_masking import (
    HIP_TRACKER_INDEX,
    REALTIME_POSE_INPUT_DIM,
    REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_DIM,
    REALTIME_POSE_TARGET_START,
    SENSOR_VALID_DIM,
    SENSOR_VALID_START,
    TRACKER_PATTERN_CATEGORIES,
    TRACKER_POS_REF_START,
    TRACKER_ROT_REF_START,
)
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source, write_toy_source_dataset


def test_converter_feature_builder_outputs_realtime_schema_shapes():
    class SmplMotion:
        pass

    source = build_toy_realtime_source(frame_count=4)
    motion = SmplMotion()
    motion.joint_positions = source["joints_world"]
    rotations = np.zeros((4, 24, 3, 3), dtype=np.float32)
    rotations[..., 0, 0] = 1.0
    rotations[..., 1, 1] = 1.0
    rotations[..., 2, 2] = 1.0
    motion.joint_rotations = rotations

    features = build_realtime_pose_features(motion)
    assert features["body_pose_parent_6d"].shape == (4, 144)
    assert features["root_pos_world"].shape == (4, 3)
    assert features["root_yaw_delta_sincos"].shape == (4, 2)
    assert features["tracker_pos_world"].shape == (4, 6, 3)
    assert features["tracker_rot_world_6d"].shape == (4, 6, 6)
    assert features["joints_world"].shape == (4, 24, 3)
    assert "contact" not in features


def test_task_generator_defaults_to_full_tracker_tasks(tmp_path):
    source_dir = tmp_path / "sources"
    output_dir = tmp_path / "tasks"
    write_toy_source_dataset(source_dir)

    generate_realtime_pose_tasks_main(
        [
            "--source_dir",
            str(source_dir),
            "--output_dir",
            str(output_dir),
            "--splits",
            "train",
            "--samples_per_file",
            "2",
            "--split_dir",
            "",
            "--overwrite",
        ]
    )
    manifest_path = output_dir / "train" / "manifest.jsonl"
    entries = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 2
    assert {entry["tracker_pattern"] for entry in entries} == {"full-trackers"}
    assert {entry["mask_policy"] for entry in entries} == {"full"}

    for entry in entries:
        task = load_materialized_task_npz(manifest_dir=manifest_path.parent, task_path=entry["task_path"])
        sensor_valid = task["sensor_valid"].astype(bool)
        assert sensor_valid.all()
        assert sensor_valid[:, HIP_TRACKER_INDEX].all()
        assert (sensor_valid.sum(axis=1) >= 3).all()
        assert task["inpaint_mask"].shape == (REALTIME_POSE_SEQ_LEN, REALTIME_POSE_INPUT_DIM)
        assert task["inpaint_mask"][REALTIME_POSE_TARGET_START, :REALTIME_POSE_TARGET_DIM].all()
        assert not task["inpaint_mask"][:, REALTIME_POSE_TARGET_DIM:].any()


def test_task_generator_default_split_filter_rejects_unmatched_source(tmp_path):
    source_dir = tmp_path / "sources"
    output_dir = tmp_path / "tasks"
    write_toy_source_dataset(source_dir)

    with pytest.raises(RuntimeError, match="没有匹配"):
        generate_realtime_pose_tasks_main(
            [
                "--source_dir",
                str(source_dir),
                "--output_dir",
                str(output_dir),
                "--splits",
                "train",
                "--samples_per_file",
                "1",
                "--overwrite",
            ]
        )


def test_task_generator_overwrite_rejects_unsafe_directories(tmp_path):
    source_dir = tmp_path / "sources"
    write_toy_source_dataset(source_dir)

    with pytest.raises(ValueError, match="source_dir"):
        generate_realtime_pose_tasks_main(
            [
                "--source_dir",
                str(source_dir),
                "--output_dir",
                str(source_dir),
                "--split_dir",
                "",
                "--overwrite",
            ]
        )

    output_dir = tmp_path / "non_task_dir"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(ValueError, match="标记"):
        generate_realtime_pose_tasks_main(
            [
                "--source_dir",
                str(source_dir),
                "--output_dir",
                str(output_dir),
                "--split_dir",
                "",
                "--overwrite",
            ]
        )
    assert (output_dir / "keep.txt").exists()


def test_task_generator_fixed_patterns_keeps_constraints_and_covers_categories(tmp_path):
    source_dir = tmp_path / "sources"
    output_dir = tmp_path / "tasks"
    write_toy_source_dataset(source_dir)

    generate_realtime_pose_tasks_main(
        [
            "--source_dir",
            str(source_dir),
            "--output_dir",
            str(output_dir),
            "--splits",
            "train",
            "--samples_per_file",
            "1",
            "--split_dir",
            "",
            "--mask_policy",
            "fixed_patterns",
            "--fixed_tracker_patterns",
            "all",
            "--overwrite",
        ]
    )
    manifest_path = output_dir / "train" / "manifest.jsonl"
    entries = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    categories = {entry["tracker_pattern"] for entry in entries}
    assert set(TRACKER_PATTERN_CATEGORIES).issubset(categories)
    assert {entry["mask_policy"] for entry in entries} == {"fixed_patterns"}

    for entry in entries:
        task = load_materialized_task_npz(manifest_dir=manifest_path.parent, task_path=entry["task_path"])
        sensor_valid = task["sensor_valid"].astype(bool)
        assert sensor_valid[:, HIP_TRACKER_INDEX].all()
        assert (sensor_valid.sum(axis=1) >= 3).all()


def test_dataset_outputs_206_by_61_and_reference_uses_previous_yaw(tmp_path):
    source_dir = tmp_path / "sources"
    output_dir = tmp_path / "tasks"
    write_toy_source_dataset(source_dir)
    generate_realtime_pose_tasks_main(
        [
            "--source_dir",
            str(source_dir),
            "--output_dir",
            str(output_dir),
            "--splits",
            "train",
            "--samples_per_file",
            "1",
            "--split_dir",
            "",
            "--overwrite",
        ]
    )
    dataset = RealtimePoseTaskDataset(output_dir, split="train", normalize_input=False)
    item = dataset[0]
    assert tuple(item["x"].shape) == (REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SEQ_LEN)
    assert tuple(item["conditioned_x"].shape) == (REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SEQ_LEN)
    assert item["inpaint_mask"][:REALTIME_POSE_TARGET_DIM, REALTIME_POSE_TARGET_START].all()
    assert not item["inpaint_mask"][REALTIME_POSE_TARGET_DIM:, :].any()

    entry = dataset.entries[0]
    task = dataset.load_task(0, entry)
    arrays = load_realtime_task_arrays(task, seq_len=REALTIME_POSE_SEQ_LEN)
    base = encode_realtime_pose_features(arrays)
    changed_current = {key: value.copy() for key, value in arrays.items()}
    changed_current["root_yaw"][REALTIME_POSE_TARGET_START] += 1.0
    current_encoded = encode_realtime_pose_features(changed_current)
    np.testing.assert_allclose(base[REALTIME_POSE_TARGET_START, 146:200], current_encoded[REALTIME_POSE_TARGET_START, 146:200])

    changed_prev = {key: value.copy() for key, value in arrays.items()}
    changed_prev["root_yaw"][REALTIME_POSE_TARGET_START - 1] += 1.0
    prev_encoded = encode_realtime_pose_features(changed_prev)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(base[REALTIME_POSE_TARGET_START, 146:200], prev_encoded[REALTIME_POSE_TARGET_START, 146:200])


def test_dataset_dynamic_tracker_mask_samples_legal_categories(tmp_path):
    source_dir = tmp_path / "sources"
    output_dir = tmp_path / "tasks"
    write_toy_source_dataset(source_dir)
    generate_realtime_pose_tasks_main(
        [
            "--source_dir",
            str(source_dir),
            "--output_dir",
            str(output_dir),
            "--splits",
            "train",
            "--samples_per_file",
            "1",
            "--split_dir",
            "",
            "--overwrite",
        ]
    )

    dataset = RealtimePoseTaskDataset(
        output_dir,
        split="train",
        normalize_input=False,
        tracker_mask_policy="dynamic_categories",
        tracker_mask_seed=123,
    )
    categories = []
    masks = []
    for _ in range(len(TRACKER_PATTERN_CATEGORIES) * 2):
        item = dataset[0]
        sensor_valid = item["sensor_valid"].numpy().T.astype(bool)
        categories.append(item["tracker_pattern"])
        masks.append(sensor_valid.tobytes())
        assert sensor_valid[:, HIP_TRACKER_INDEX].all()
        assert (sensor_valid.sum(axis=1) >= 3).all()

    assert set(TRACKER_PATTERN_CATEGORIES).issubset(set(categories))
    assert len(set(masks)) > 1


def test_dataset_dynamic_mask_and_augmentation_are_seed_reproducible(tmp_path):
    source_dir = tmp_path / "sources"
    output_dir = tmp_path / "tasks"
    write_toy_source_dataset(source_dir)
    generate_realtime_pose_tasks_main(
        [
            "--source_dir",
            str(source_dir),
            "--output_dir",
            str(output_dir),
            "--splits",
            "train",
            "--samples_per_file",
            "1",
            "--split_dir",
            "",
            "--overwrite",
        ]
    )

    def collect(seed: int):
        dataset = RealtimePoseTaskDataset(
            output_dir,
            split="train",
            normalize_input=False,
            tracker_mask_policy="dynamic_categories",
            tracker_mask_seed=seed,
            tracker_pos_noise_std=0.01,
            tracker_rot_noise_std=0.01,
            history_pose_noise_std=0.01,
            history_yaw_noise_std=0.01,
            root_yaw_ref_noise_std=0.01,
        )
        return [
            (
                item["tracker_pattern"],
                item["sensor_valid"].numpy().copy(),
                item["x"].numpy().copy(),
            )
            for item in (dataset[0] for _ in range(4))
        ]

    seq_a = collect(seed=321)
    seq_b = collect(seed=321)
    seq_c = collect(seed=322)
    for item_a, item_b in zip(seq_a, seq_b):
        assert item_a[0] == item_b[0]
        np.testing.assert_array_equal(item_a[1], item_b[1])
        np.testing.assert_allclose(item_a[2], item_b[2])
    assert any(not np.allclose(item_a[2], item_c[2]) for item_a, item_c in zip(seq_a, seq_c))


def test_dataset_fixed_tracker_mask_is_reproducible_and_zero_fills_invalid_channels(tmp_path):
    source_dir = tmp_path / "sources"
    output_dir = tmp_path / "tasks"
    write_toy_source_dataset(source_dir)
    generate_realtime_pose_tasks_main(
        [
            "--source_dir",
            str(source_dir),
            "--output_dir",
            str(output_dir),
            "--splits",
            "test",
            "--samples_per_file",
            "1",
            "--split_dir",
            "",
            "--overwrite",
        ]
    )

    dataset_a = RealtimePoseTaskDataset(
        output_dir,
        split="test",
        normalize_input=False,
        tracker_mask_policy="fixed_categories",
        tracker_mask_seed=10,
        tracker_mask_categories=["upper-body"],
    )
    dataset_b = RealtimePoseTaskDataset(
        output_dir,
        split="test",
        normalize_input=False,
        tracker_mask_policy="fixed_categories",
        tracker_mask_seed=10,
        tracker_mask_categories=["upper-body"],
    )
    item_a = dataset_a[0]
    item_b = dataset_b[0]
    np.testing.assert_array_equal(item_a["sensor_valid"].numpy(), item_b["sensor_valid"].numpy())

    x = item_a["x"].numpy()
    sensor_valid = item_a["sensor_valid"].numpy().astype(bool)
    assert not sensor_valid.all()
    for tracker_index in range(sensor_valid.shape[0]):
        missing = ~sensor_valid[tracker_index]
        if not missing.any():
            continue
        pos_start = TRACKER_POS_REF_START + tracker_index * 3
        rot_start = TRACKER_ROT_REF_START + tracker_index * 6
        assert np.allclose(x[pos_start:pos_start + 3, missing], 0.0)
        assert np.allclose(x[rot_start:rot_start + 6, missing], 0.0)


def test_normalizer_keeps_sensor_valid_as_binary_condition(tmp_path):
    source_dir = tmp_path / "sources"
    normalizer_dir = tmp_path / "meta"
    output_dir = tmp_path / "tasks"
    write_toy_source_dataset(source_dir)

    compute_realtime_pose_normalizer(
        SimpleNamespace(
            source_dir=str(source_dir),
            output_dir=str(normalizer_dir),
            split_dir="",
            split="train",
            eps=1e-8,
            overwrite=True,
        )
    )
    generate_realtime_pose_tasks_main(
        [
            "--source_dir",
            str(source_dir),
            "--output_dir",
            str(output_dir),
            "--splits",
            "train",
            "--samples_per_file",
            "1",
            "--split_dir",
            "",
            "--overwrite",
        ]
    )

    dataset = RealtimePoseTaskDataset(
        output_dir,
        split="train",
        normalizer_dir=normalizer_dir,
        normalize_input=True,
        tracker_mask_policy="fixed_categories",
        tracker_mask_seed=10,
        tracker_mask_categories=["upper-body"],
    )
    sensor_values = []
    for index in range(len(dataset)):
        item = dataset[index]
        values = item["x"][SENSOR_VALID_START:SENSOR_VALID_START + SENSOR_VALID_DIM].numpy()
        sensor_values.append(values)
        x = item["x"].numpy()
        sensor_valid = item["sensor_valid"].numpy().astype(bool)
        for tracker_index in range(sensor_valid.shape[0]):
            missing = ~sensor_valid[tracker_index]
            if not missing.any():
                continue
            pos_start = TRACKER_POS_REF_START + tracker_index * 3
            rot_start = TRACKER_ROT_REF_START + tracker_index * 6
            assert np.allclose(x[pos_start:pos_start + 3, missing], 0.0)
            assert np.allclose(x[rot_start:rot_start + 6, missing], 0.0)
    sensor_values = np.concatenate([values.reshape(-1) for values in sensor_values])
    assert set(np.unique(sensor_values).tolist()).issubset({0.0, 1.0})


def test_converter_fails_on_partial_conversion_by_default(monkeypatch, tmp_path):
    amass_dir = tmp_path / "AMASS"
    output_dir = tmp_path / "converted"
    bad_path = amass_dir / "bad_motion.npz"
    bad_path.parent.mkdir(parents=True)
    bad_path.write_bytes(b"not a real motion")

    monkeypatch.setattr(sys, "argv", ["prog", "--amass_dir", str(amass_dir), "--output_dir", str(output_dir)])
    monkeypatch.setattr(amass_converter, "validate_shared_args", lambda args: None)
    monkeypatch.setattr(amass_converter, "iter_amass_motion_files", lambda amass_path: [bad_path])
    monkeypatch.setattr(amass_converter, "SmplModelCache", lambda model_dir: object())

    def fail_convert(path, args, model_cache, mirror_variant=False):
        raise ValueError("boom")

    monkeypatch.setattr(amass_converter, "convert_one_motion", fail_convert)
    with pytest.raises(RuntimeError, match="allow_partial"):
        amass_converter.main()

    manifest_path = output_dir / "manifest.jsonl"
    entries = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert entries[0]["status"] == "failed"
