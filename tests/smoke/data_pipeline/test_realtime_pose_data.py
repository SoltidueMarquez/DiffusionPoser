from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import data_converter.amass_to_realtime_pose as amass_converter
from data_converter.amass_to_realtime_pose import build_realtime_pose_features
from data_loaders.compute_realtime_pose_normalizer import compute_realtime_pose_normalizer
from data_loaders.generate_realtime_pose_tasks import (
    make_task_id,
    load_realtime_source,
    main as generate_realtime_pose_tasks_main,
    read_source_entries,
)
from data_loaders.body_fbx_kinematics import build_synthetic_body_fbx_rest
from data_loaders.realtime_pose_kinematics import fk_body_fbx_local_torch
from data_loaders.realtime_pose_dataset import (
    RealtimePoseTaskDataset,
    encode_realtime_pose_features,
    load_materialized_task_npz,
    load_realtime_task_arrays,
)
from data_loaders.sensor_masking import (
    DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    HIP_TRACKER_INDEX,
    POSE_REPRESENTATION_KEY,
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
    get_schema_spec,
)
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source, write_toy_source_dataset
from utils.run_dirs import read_latest_pointer


def latest_artifact_dir(root: Path, kind: str) -> Path:
    latest = read_latest_pointer(root, kind=kind)
    assert latest is not None
    return latest


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

    body_fbx_rest = build_synthetic_body_fbx_rest()
    features = build_realtime_pose_features(
        motion,
        schema_name=REALTIME_POSE_SCHEMA_NAME,
        body_fbx_rest=body_fbx_rest,
    )
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    assert features[schema.body_pose_key].shape == (4, 144)
    assert str(features[POSE_REPRESENTATION_KEY].item()) == schema.pose_representation
    assert features["root_pos_world"].shape == (4, 3)
    assert features[schema.root_heading_delta_key].shape == (4, 2)
    assert features["tracker_pos_world"].shape == (4, 6, 3)
    assert features["tracker_rot_world_6d"].shape == (4, 6, 6)
    assert features["joints_world"].shape == (4, 24, 3)
    assert features["root_delta_xz_ref"].shape == (4, 2)
    assert features[schema.pelvis_height_key].shape == (4, 1)
    assert features["foot_contact"].shape == (4, 2)
    assert features["joint_rest_local_rotations_6d"].shape == (24, 6)


def test_converter_cli_defaults_to_current_recommended_schema():
    args = amass_converter.parse_args([])
    assert args.schema == DEFAULT_REALTIME_POSE_SCHEMA_NAME


def test_converter_reuses_existing_v2_source_without_smpl(monkeypatch, tmp_path):
    amass_dir = tmp_path / "AMASS"
    reuse_dir = tmp_path / "reuse_v2"
    output_dir = tmp_path / "converted_v2"
    fake_amass_path = amass_dir / "ACCAD" / "toy_realtime.npz"
    fake_amass_path.parent.mkdir(parents=True)
    fake_amass_path.write_bytes(b"reuse path does not load this file")

    reuse_path = reuse_dir / "ACCAD" / "toy_realtime.npz"
    reuse_path.parent.mkdir(parents=True)
    np.savez(reuse_path, **build_toy_realtime_source(frame_count=REALTIME_POSE_SEQ_LEN))

    def fail_smpl(*args, **kwargs):
        raise AssertionError("reuse path should not run SMPL forward")

    monkeypatch.setattr(amass_converter, "run_smpl_forward", fail_smpl)
    args = SimpleNamespace(
        amass_dir=amass_dir,
        output_dir=output_dir,
        target_fps=60.0,
        batch_size=1,
        schema=REALTIME_POSE_SCHEMA_NAME,
        reuse_source_dir=reuse_dir,
        skip_existing=False,
        overwrite=True,
    )
    record = amass_converter.convert_one_motion(
        path=fake_amass_path,
        args=args,
        model_cache=object(),
        mirror_variant=False,
    )
    assert record["status"] == "reused_source"
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    with np.load(output_dir / "ACCAD" / "toy_realtime.npz", allow_pickle=False) as data:
        assert "root_delta_xz_ref" in data.files
        assert schema.pelvis_height_key in data.files
        assert "foot_contact" in data.files
        assert "joint_rest_local_rotations_6d" in data.files
        assert str(data[POSE_REPRESENTATION_KEY].item()) == schema.pose_representation
        metadata = json.loads(str(data["metadata"].item()))
    assert metadata["schema_name"] == REALTIME_POSE_SCHEMA_NAME
    assert metadata["pose_representation"] == schema.pose_representation


def test_source_manifest_reused_entries_are_usable(tmp_path):
    source_dir = tmp_path / "converted"
    source_path = source_dir / "ACCAD" / "toy_realtime.npz"
    source_path.parent.mkdir(parents=True)
    np.savez(source_path, **build_toy_realtime_source(frame_count=REALTIME_POSE_SEQ_LEN))
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    manifest_entry = {
        "status": "reused_source",
        "schema_name": schema.name,
        "pose_representation": schema.pose_representation,
        "output_path": "ACCAD/toy_realtime.npz",
        "source_relative_path": "ACCAD/toy_realtime.npz",
        "stablemotion_split_key": "ACCAD/toy_realtime",
        "frames": REALTIME_POSE_SEQ_LEN,
    }
    with (source_dir / "manifest.jsonl").open("w", encoding="utf-8") as file:
        file.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
    entries = read_source_entries(source_dir)
    assert len(entries) == 1
    assert entries[0]["source_path"] == str(source_path)


def test_realtime_source_loader_rejects_legacy_parent_local_pose(tmp_path):
    source = build_toy_realtime_source(frame_count=REALTIME_POSE_SEQ_LEN)
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    legacy_source = {
        key: value
        for key, value in source.items()
        if key not in {schema.body_pose_key, POSE_REPRESENTATION_KEY}
    }
    legacy_source["body_pose_parent_6d"] = source[schema.body_pose_key]
    legacy_path = tmp_path / "legacy_source.npz"
    np.savez(legacy_path, **legacy_source)

    with pytest.raises(ValueError, match="legacy body_pose_parent_6d"):
        load_realtime_source(legacy_path, schema_name=schema.name)


def test_fk_reconstructs_pelvis_from_ground_root_offset():
    class SmplMotion:
        pass

    source = build_toy_realtime_source(frame_count=4)
    source["joints_world"][:, 0, 1] = 0.92
    motion = SmplMotion()
    motion.joint_positions = source["joints_world"]
    rotations = np.zeros((4, 24, 3, 3), dtype=np.float32)
    rotations[..., 0, 0] = 1.0
    rotations[..., 1, 1] = 1.0
    rotations[..., 2, 2] = 1.0
    motion.joint_rotations = rotations

    body_fbx_rest = build_synthetic_body_fbx_rest()
    features = build_realtime_pose_features(motion, body_fbx_rest=body_fbx_rest)
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    np.testing.assert_allclose(features[schema.pelvis_height_key][0], np.asarray([0.92], dtype=np.float32))
    expected_root_pos = source["joints_world"][0, 0] - body_fbx_rest.pelvis_local_position
    np.testing.assert_allclose(features["root_pos_world"][0], expected_root_pos, atol=1e-6)
    np.testing.assert_allclose(features["joint_offsets_parent"][0], body_fbx_rest.pelvis_local_position)

    pred_joints = fk_body_fbx_local_torch(
        body_pose_local_delta_6d=torch.from_numpy(features[schema.body_pose_key][:1]).float(),
        actor_root_pos_world=torch.from_numpy(features["root_pos_world"][:1]).float(),
        root_heading=torch.from_numpy(features["root_yaw"][:1]).float(),
        rest_local_positions=torch.from_numpy(features["joint_offsets_parent"][None]).float(),
        rest_local_rotations_6d=torch.from_numpy(features["joint_rest_local_rotations_6d"][None]).float(),
    )
    np.testing.assert_allclose(
        pred_joints[0, 0].detach().numpy(),
        source["joints_world"][0, 0],
        atol=1e-6,
    )


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
            "--schema",
            REALTIME_POSE_SCHEMA_NAME,
            "--split_dir",
            "",
            "--overwrite",
        ]
    )
    task_output_dir = latest_artifact_dir(output_dir, kind="tasks")
    assert (output_dir / "latest_tasks.json").exists()
    manifest_path = task_output_dir / "train" / "manifest.jsonl"
    entries = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 2
    assert {entry["tracker_pattern"] for entry in entries} == {"full-trackers"}
    assert {entry["mask_policy"] for entry in entries} == {"full"}
    assert {entry[POSE_REPRESENTATION_KEY] for entry in entries} == {get_schema_spec(REALTIME_POSE_SCHEMA_NAME).pose_representation}

    for entry in entries:
        task = load_materialized_task_npz(manifest_dir=manifest_path.parent, task_path=entry["task_path"])
        sensor_valid = task["sensor_valid"].astype(bool)
        assert sensor_valid.all()
        assert sensor_valid[:, HIP_TRACKER_INDEX].all()
        assert (sensor_valid.sum(axis=1) >= 3).all()
        assert task["inpaint_mask"].shape == (REALTIME_POSE_SEQ_LEN, REALTIME_POSE_INPUT_DIM)
        assert task["inpaint_mask"][REALTIME_POSE_TARGET_START, :REALTIME_POSE_TARGET_DIM].all()
        assert not task["inpaint_mask"][:, REALTIME_POSE_TARGET_DIM:].any()


def test_task_id_stays_short_for_long_amass_paths():
    task_id = make_task_id(
        split="train",
        stablemotion_split_key=(
            "BioMotionLab_NTroje/rub042/very_long_nested_subject_name/"
            "rub042_0027_circle_walk_stageii_poses"
        ),
        sample_index=12,
        pattern_index=3,
        pattern_category="full-trackers",
    )

    assert len(task_id) <= 72
    assert "full_trackers" not in task_id
    digest = task_id.split("_")[-1]
    assert len(digest) == 16
    assert all(char in "0123456789abcdef" for char in digest)


def test_task_generator_skips_short_sources_by_default(tmp_path):
    source_dir = tmp_path / "sources"
    output_dir = tmp_path / "tasks"
    write_toy_source_dataset(source_dir, frame_count=REALTIME_POSE_SEQ_LEN - 3)

    counts = generate_realtime_pose_tasks_main(
        [
            "--source_dir",
            str(source_dir),
            "--output_dir",
            str(output_dir),
            "--splits",
            "train",
            "--samples_per_file",
            "2",
            "--schema",
            REALTIME_POSE_SCHEMA_NAME,
            "--split_dir",
            "",
            "--overwrite",
        ]
    )

    assert counts["train"] == 0
    task_output_dir = latest_artifact_dir(output_dir, kind="tasks")
    assert (task_output_dir / "train" / "manifest.jsonl").read_text(encoding="utf-8") == ""
    report_path = task_output_dir / "train" / "skipped_short_sources.jsonl"
    records = [json.loads(line) for line in report_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["source_frames"] == REALTIME_POSE_SEQ_LEN - 3
    assert records[0]["required_frames"] == REALTIME_POSE_SEQ_LEN


def test_task_generator_strict_short_source_policy_raises(tmp_path):
    source_dir = tmp_path / "sources"
    output_dir = tmp_path / "tasks"
    write_toy_source_dataset(source_dir, frame_count=REALTIME_POSE_SEQ_LEN - 3)

    with pytest.raises(ValueError, match="至少需要"):
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
                "--schema",
                REALTIME_POSE_SCHEMA_NAME,
                "--split_dir",
                "",
                "--short_source_policy",
                "error",
                "--overwrite",
            ]
        )


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
                "--schema",
                REALTIME_POSE_SCHEMA_NAME,
                "--overwrite",
            ]
        )
    assert not output_dir.exists()


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
    with pytest.raises(ValueError, match="latest_tasks"):
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
            "--schema",
            REALTIME_POSE_SCHEMA_NAME,
            "--split_dir",
            "",
            "--mask_policy",
            "fixed_patterns",
            "--fixed_tracker_patterns",
            "all",
            "--overwrite",
        ]
    )
    task_output_dir = latest_artifact_dir(output_dir, kind="tasks")
    manifest_path = task_output_dir / "train" / "manifest.jsonl"
    entries = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    categories = {entry["tracker_pattern"] for entry in entries}
    assert set(TRACKER_PATTERN_CATEGORIES).issubset(categories)
    assert {entry["mask_policy"] for entry in entries} == {"fixed_patterns"}

    for entry in entries:
        task = load_materialized_task_npz(manifest_dir=manifest_path.parent, task_path=entry["task_path"])
        sensor_valid = task["sensor_valid"].astype(bool)
        assert sensor_valid[:, HIP_TRACKER_INDEX].all()
        assert (sensor_valid.sum(axis=1) >= 3).all()


def test_dataset_outputs_schema_dim_by_61_and_reference_uses_previous_yaw(tmp_path):
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
            "--schema",
            REALTIME_POSE_SCHEMA_NAME,
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
    schema = dataset.schema
    tracker_slice = slice(schema.tracker_pos_slice().start, schema.tracker_rot_slice().stop)
    changed_current = {key: value.copy() for key, value in arrays.items()}
    changed_current["root_yaw"][REALTIME_POSE_TARGET_START] += 1.0
    current_encoded = encode_realtime_pose_features(changed_current)
    np.testing.assert_allclose(
        base[REALTIME_POSE_TARGET_START, tracker_slice],
        current_encoded[REALTIME_POSE_TARGET_START, tracker_slice],
    )

    changed_prev = {key: value.copy() for key, value in arrays.items()}
    changed_prev["root_yaw"][REALTIME_POSE_TARGET_START - 1] += 1.0
    prev_encoded = encode_realtime_pose_features(changed_prev)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            base[REALTIME_POSE_TARGET_START, tracker_slice],
            prev_encoded[REALTIME_POSE_TARGET_START, tracker_slice],
        )


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
            "--schema",
            REALTIME_POSE_SCHEMA_NAME,
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
            "--schema",
            REALTIME_POSE_SCHEMA_NAME,
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


def test_dataset_history_condition_corruption_only_indexes_history_frames(tmp_path):
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
            "--schema",
            REALTIME_POSE_SCHEMA_NAME,
            "--split_dir",
            "",
            "--overwrite",
        ]
    )

    dataset = RealtimePoseTaskDataset(
        output_dir,
        split="train",
        normalize_input=False,
        history_pose_dropout_prob=1.0,
        history_pose_replace_prob=1.0,
        history_yaw_replace_prob=1.0,
        history_root_yaw_drift_std=0.01,
    )
    item = dataset[0]

    conditioned = item["conditioned_x"].numpy()
    clean = item["x"].numpy()
    assert conditioned.shape[1] == REALTIME_POSE_SEQ_LEN
    assert np.allclose(conditioned[:144, :REALTIME_POSE_TARGET_START], 0.0)
    assert not np.allclose(clean[:144, :REALTIME_POSE_TARGET_START], 0.0)
    assert np.allclose(conditioned[:REALTIME_POSE_TARGET_DIM, REALTIME_POSE_TARGET_START], 0.0)


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
            "--schema",
            REALTIME_POSE_SCHEMA_NAME,
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
            "--schema",
            REALTIME_POSE_SCHEMA_NAME,
            "--split_dir",
            "",
            "--overwrite",
        ]
    )
    compute_realtime_pose_normalizer(
        SimpleNamespace(
            task_dir=str(output_dir),
            output_dir=str(normalizer_dir),
            split="train",
            schema=REALTIME_POSE_SCHEMA_NAME,
            eps=1e-8,
            overwrite=True,
        )
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


def test_task_normalizer_ignores_invalid_tracker_zero_fill_in_stats(tmp_path):
    source_dir = tmp_path / "sources"
    task_dir = tmp_path / "tasks"
    normalizer_dir = tmp_path / "meta"
    write_toy_source_dataset(source_dir)
    generate_realtime_pose_tasks_main(
        [
            "--source_dir",
            str(source_dir),
            "--output_dir",
            str(task_dir),
            "--splits",
            "train",
            "--samples_per_file",
            "1",
            "--schema",
            REALTIME_POSE_SCHEMA_NAME,
            "--split_dir",
            "",
            "--mask_policy",
            "fixed_patterns",
            "--fixed_tracker_patterns",
            "full-trackers",
            "upper-body",
            "--overwrite",
        ]
    )

    meta = compute_realtime_pose_normalizer(
        SimpleNamespace(
            task_dir=str(task_dir),
            output_dir=str(normalizer_dir),
            split="train",
            schema=REALTIME_POSE_SCHEMA_NAME,
            eps=1e-8,
            overwrite=True,
        )
    )
    assert meta["matched_tasks"] == 2

    task_output_dir = latest_artifact_dir(task_dir, kind="tasks")
    manifest_path = task_output_dir / "train" / "manifest.jsonl"
    entries = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tracker_valid_counts = np.zeros(6, dtype=np.int64)
    all_features = []
    all_valid = []
    for entry in entries:
        task = load_materialized_task_npz(manifest_path.parent, entry["task_path"], schema_name=REALTIME_POSE_SCHEMA_NAME)
        arrays = load_realtime_task_arrays(task, seq_len=REALTIME_POSE_SEQ_LEN, schema_name=REALTIME_POSE_SCHEMA_NAME)
        all_features.append(encode_realtime_pose_features(arrays, schema_name=REALTIME_POSE_SCHEMA_NAME))
        all_valid.append(arrays["sensor_valid"])
        tracker_valid_counts += arrays["sensor_valid"].sum(axis=0).astype(np.int64)
    assert meta["tracker_valid_observation_counts"] == tracker_valid_counts.astype(int).tolist()

    features = np.concatenate(all_features, axis=0)
    valid_all = np.concatenate(all_valid, axis=0)
    partial_trackers = np.where((tracker_valid_counts > 0) & (tracker_valid_counts < len(entries) * REALTIME_POSE_SEQ_LEN))[0]
    assert partial_trackers.size > 0
    chosen = None
    for tracker_index in partial_trackers.tolist():
        candidate_slice = slice(TRACKER_POS_REF_START + tracker_index * 3, TRACKER_POS_REF_START + tracker_index * 3 + 3)
        expected = features[valid_all[:, tracker_index], candidate_slice].mean(axis=0)
        biased = features[:, candidate_slice].mean(axis=0)
        if not np.allclose(expected, biased, atol=1e-6):
            chosen = (candidate_slice, expected, biased)
            break
    assert chosen is not None
    pos_slice, expected_mean, biased_mean = chosen
    normalizer_output_dir = latest_artifact_dir(normalizer_dir, kind="normalizer")
    saved_mean = torch.load(normalizer_output_dir / "mean.pt", map_location="cpu", weights_only=True).numpy()[pos_slice]
    np.testing.assert_allclose(saved_mean, expected_mean, atol=1e-6)
    assert not np.allclose(saved_mean, biased_mean, atol=1e-6)


def test_converter_fails_on_partial_conversion_by_default(monkeypatch, tmp_path):
    amass_dir = tmp_path / "AMASS"
    output_dir = tmp_path / "converted"
    bad_path = amass_dir / "bad_motion.npz"
    bad_path.parent.mkdir(parents=True)
    bad_path.write_bytes(b"not a real motion")

    monkeypatch.setattr(amass_converter, "validate_shared_args", lambda args: None)
    monkeypatch.setattr(amass_converter, "iter_amass_motion_files", lambda amass_path: [bad_path])
    monkeypatch.setattr(amass_converter, "SmplModelCache", lambda model_dir: object())

    def fail_convert(path, args, model_cache, mirror_variant=False):
        raise ValueError("boom")

    monkeypatch.setattr(amass_converter, "convert_one_motion", fail_convert)
    with pytest.raises(RuntimeError, match="allow_partial"):
        amass_converter.main(["--amass_dir", str(amass_dir), "--output_dir", str(output_dir)])

    manifest_path = output_dir / "manifest.jsonl"
    entries = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert entries[0]["status"] == "failed"
