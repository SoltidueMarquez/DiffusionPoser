from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import data_converter.amass_to_realtime_pose as amass_converter
from data_converter.amass_to_realtime_pose import build_realtime_pose_features
from data_converter.amass_smpl_utils import MotionSource, SMPL_JOINT_COUNT, SMPL_PARENTS, run_smpl_forward
from data_loaders.compute_realtime_pose_normalizer import compute_realtime_pose_normalizer
from data_loaders.generate_realtime_pose_tasks import (
    TASK_OUTPUT_MARKER,
    make_task_id,
    load_realtime_source,
    main as generate_realtime_pose_tasks_main,
    read_source_entries,
    resolve_manifest_file,
    save_task_npz,
    temporary_task_path,
)
from data_loaders.body_fbx_kinematics import build_synthetic_body_fbx_rest, fk_body_fbx_local_delta
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
    REALTIME_POSE_TARGET_LENGTH,
    REALTIME_POSE_TARGET_START,
    SENSOR_VALID_DIM,
    SENSOR_VALID_START,
    TRACKER_PATTERN_CATEGORIES,
    TRACKER_POS_REF_START,
    TRACKER_ROT_REF_START,
    get_schema_spec,
)
from data_loaders.stationary_label_config import stationary_label_metadata
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source, build_toy_source_metadata, write_toy_source_dataset
from utils.run_dirs import read_latest_pointer


def latest_artifact_dir(root: Path, kind: str) -> Path:
    latest = read_latest_pointer(root, kind=kind)
    assert latest is not None
    return latest


def write_task_variant(source_task: Path, output_task: Path, drop_keys: set[str] | None = None, **overrides) -> None:
    drop_keys = drop_keys or set()
    with np.load(source_task, allow_pickle=False) as data:
        payload = {key: data[key].copy() for key in data.files if key not in drop_keys}
    payload.update(overrides)
    np.savez(output_task, **payload)


def generate_single_task_manifest(tmp_path: Path) -> tuple[Path, dict]:
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
    task_output_dir = latest_artifact_dir(output_dir, kind="tasks")
    manifest_path = task_output_dir / "train" / "manifest.jsonl"
    entry = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])
    return manifest_path, entry


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
    assert features["stationary_prob_5"].shape == (4, 5)
    assert np.all((features["stationary_prob_5"] >= 0.0) & (features["stationary_prob_5"] <= 1.0))
    assert features["joint_rest_local_rotations_6d"].shape == (24, 6)
    np.testing.assert_allclose(features["root_pos_world"][:, 1], 0.0, atol=1e-6)
    np.testing.assert_allclose(features[schema.pelvis_height_key][:, 0], features["joints_world"][:, 0, 1], atol=1e-6)


def test_converter_cli_defaults_to_current_recommended_schema():
    args = amass_converter.parse_args([])
    assert args.schema == DEFAULT_REALTIME_POSE_SCHEMA_NAME
    args = amass_converter.parse_args(["--height_threshold", "0.03"])
    assert args.height_threshold == pytest.approx(0.03)
    args = amass_converter.parse_args(["--num_workers", "3", "--worker_torch_threads", "2"])
    assert args.num_workers == 3
    assert args.worker_torch_threads == 2


def test_converter_default_body_fbx_rest_arg_uses_default_json(monkeypatch):
    calls = []

    def fake_load_body_fbx_rest(path):
        calls.append(path)
        return object()

    monkeypatch.setattr(amass_converter, "load_body_fbx_rest", fake_load_body_fbx_rest)
    args = amass_converter.parse_args([])

    assert amass_converter.resolve_body_fbx_rest_for_schema(args) is not None
    assert calls == [None]


def test_converter_parallel_work_items_include_mirror_variants():
    motion_files = [Path("a.npz"), Path("b.npz")]
    args = SimpleNamespace(mirror=True)

    items = amass_converter.build_conversion_work_items(args=args, motion_files=motion_files)

    assert items == [
        (Path("a.npz"), False),
        (Path("a.npz"), True),
        (Path("b.npz"), False),
        (Path("b.npz"), True),
    ]


def test_run_smpl_forward_does_not_request_mesh_vertices():
    class FakeSmplModel:
        diffusionposer_model_type = "smpl"
        num_betas = 10

        def __init__(self):
            self.parents = torch.as_tensor(SMPL_PARENTS, dtype=torch.long)
            self.parameter = torch.nn.Parameter(torch.zeros(()))
            self.return_verts_values = []

        def parameters(self):
            yield self.parameter

        def __call__(self, **kwargs):
            self.return_verts_values.append(kwargs["return_verts"])
            assert kwargs["return_verts"] is False
            batch_size = int(kwargs["global_orient"].shape[0])
            joints = torch.zeros((batch_size, SMPL_JOINT_COUNT, 3), dtype=torch.float32)
            joints[:, :, 0] = kwargs["transl"][:, :1]
            return SimpleNamespace(joints=joints)

    class FakeModelCache:
        def __init__(self, model):
            self.model = model

        def get(self, gender):
            return self.model

    frame_count = 3
    source = MotionSource(
        path=Path("toy.npz"),
        relative_path=Path("toy.npz"),
        poses=np.zeros((frame_count, 66), dtype=np.float64),
        trans=np.zeros((frame_count, 3), dtype=np.float64),
        betas=np.zeros(10, dtype=np.float64),
        gender="neutral",
        source_fps=60.0,
    )
    model = FakeSmplModel()

    motion = run_smpl_forward(source=source, model_cache=FakeModelCache(model), batch_size=2)

    assert model.return_verts_values == [False, False, False]
    assert motion.joint_positions.shape == (frame_count, SMPL_JOINT_COUNT, 3)
    assert motion.joint_rotations.shape == (frame_count, SMPL_JOINT_COUNT, 3, 3)
    assert motion.rest_joints.shape == (SMPL_JOINT_COUNT, 3)
    assert not hasattr(motion, "vertices")
    assert not hasattr(motion, "rest_vertices")


def test_resolve_manifest_file_prefers_existing_repo_relative_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    repo_relative_path = Path("dataset") / "toy_source.npz"
    repo_relative_path.parent.mkdir()
    repo_relative_path.write_bytes(b"exists")

    assert resolve_manifest_file(source_dir, repo_relative_path.as_posix()) == repo_relative_path
    assert resolve_manifest_file(source_dir, "missing/toy_source.npz") == source_dir / "missing" / "toy_source.npz"


def test_converter_reuses_existing_root_y0_source_without_smpl(monkeypatch, tmp_path):
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
        assert "stationary_prob_5" in data.files
        assert "joint_rest_local_rotations_6d" in data.files
        assert str(data[POSE_REPRESENTATION_KEY].item()) == schema.pose_representation
        metadata = json.loads(str(data["metadata"].item()))
    assert metadata["schema_name"] == REALTIME_POSE_SCHEMA_NAME
    assert metadata["pose_representation"] == schema.pose_representation
    assert metadata["root_y_policy"] == schema.root_y_policy
    assert metadata["pelvis_height_mode"] == schema.pelvis_height_mode
    for key, value in stationary_label_metadata().items():
        assert metadata[key] == value


@pytest.mark.parametrize(
    ("drop_key", "override", "match"),
    [
        ("stationary_label_method", {}, "stationary label metadata"),
        (None, {"stationary_label_method": "joint_speed_v0"}, "stationary_label_method"),
    ],
)
def test_converter_rejects_stationary_label_metadata_mismatch_on_reuse(
    monkeypatch,
    tmp_path,
    drop_key,
    override,
    match,
):
    amass_dir = tmp_path / "AMASS"
    reuse_dir = tmp_path / "reuse_bad_stationary"
    output_dir = tmp_path / "converted"
    fake_amass_path = amass_dir / "ACCAD" / "toy_realtime.npz"
    fake_amass_path.parent.mkdir(parents=True)
    fake_amass_path.write_bytes(b"reuse path does not load this file")

    source = build_toy_realtime_source(frame_count=REALTIME_POSE_SEQ_LEN)
    metadata = build_toy_source_metadata(frame_count=REALTIME_POSE_SEQ_LEN)
    if drop_key is not None:
        metadata.pop(drop_key)
    metadata.update(override)
    source["metadata"] = np.asarray(json.dumps(metadata))
    reuse_path = reuse_dir / "ACCAD" / "toy_realtime.npz"
    reuse_path.parent.mkdir(parents=True)
    np.savez(reuse_path, **source)

    monkeypatch.setattr(
        amass_converter,
        "run_smpl_forward",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no smpl")),
    )
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
    with pytest.raises(ValueError, match=match):
        amass_converter.convert_one_motion(path=fake_amass_path, args=args, model_cache=object(), mirror_variant=False)


def test_converter_rejects_legacy_body_fbx_source_reuse(monkeypatch, tmp_path):
    amass_dir = tmp_path / "AMASS"
    reuse_dir = tmp_path / "reuse_old_body_fbx"
    output_dir = tmp_path / "converted"
    fake_amass_path = amass_dir / "ACCAD" / "toy_realtime.npz"
    fake_amass_path.parent.mkdir(parents=True)
    fake_amass_path.write_bytes(b"reuse path does not load this file")

    source = build_toy_realtime_source(frame_count=REALTIME_POSE_SEQ_LEN)
    source["root_pos_world"][:, 1] = 1.25
    source["metadata"] = np.asarray(
        json.dumps(
            {
                "schema_name": "realtime_pose_body_fbx_local_v1",
                "pose_representation": "body_fbx_local_delta_6d",
                "root_y_policy": "actor_root_from_pelvis",
                "pelvis_height_mode": "actor_root_y",
            }
        )
    )
    reuse_path = reuse_dir / "ACCAD" / "toy_realtime.npz"
    reuse_path.parent.mkdir(parents=True)
    np.savez(reuse_path, **source)

    monkeypatch.setattr(amass_converter, "run_smpl_forward", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no smpl")))
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
    with pytest.raises(ValueError, match="schema_name"):
        amass_converter.convert_one_motion(path=fake_amass_path, args=args, model_cache=object(), mirror_variant=False)


def test_converter_rejects_root_y0_reuse_with_nonzero_root_y(monkeypatch, tmp_path):
    amass_dir = tmp_path / "AMASS"
    reuse_dir = tmp_path / "reuse_bad_root_y"
    output_dir = tmp_path / "converted"
    fake_amass_path = amass_dir / "ACCAD" / "toy_realtime.npz"
    fake_amass_path.parent.mkdir(parents=True)
    fake_amass_path.write_bytes(b"reuse path does not load this file")

    source = build_toy_realtime_source(frame_count=REALTIME_POSE_SEQ_LEN)
    source["root_pos_world"][:, 1] = 0.5
    source["metadata"] = np.asarray(json.dumps(build_toy_source_metadata(frame_count=REALTIME_POSE_SEQ_LEN)))
    reuse_path = reuse_dir / "ACCAD" / "toy_realtime.npz"
    reuse_path.parent.mkdir(parents=True)
    np.savez(reuse_path, **source)

    monkeypatch.setattr(amass_converter, "run_smpl_forward", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no smpl")))
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
    with pytest.raises(ValueError, match="root-y0"):
        amass_converter.convert_one_motion(path=fake_amass_path, args=args, model_cache=object(), mirror_variant=False)


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


def test_task_generation_scan_rejects_old_body_fbx_source_without_manifest(tmp_path):
    source_dir = tmp_path / "sources"
    output_dir = tmp_path / "tasks"
    source_path = source_dir / "ACCAD" / "old_body_fbx.npz"
    source_path.parent.mkdir(parents=True)
    source = build_toy_realtime_source(frame_count=REALTIME_POSE_SEQ_LEN)
    source["metadata"] = np.asarray(
        json.dumps(
            {
                "schema_name": "realtime_pose_body_fbx_local_v1",
                "pose_representation": "body_fbx_local_delta_6d",
                "root_y_policy": "actor_root_from_pelvis",
                "pelvis_height_mode": "actor_root_y",
            }
        )
    )
    np.savez(source_path, **source)

    with pytest.raises(ValueError, match="schema_name"):
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
    expected_root_pos[1] = 0.0
    np.testing.assert_allclose(features["root_pos_world"][0], expected_root_pos, atol=1e-6)
    np.testing.assert_allclose(features["joint_offsets_parent"][0], body_fbx_rest.pelvis_local_position)

    fk_offsets = features["joint_offsets_parent"].copy()
    fk_offsets[0, 1] = features[schema.pelvis_height_key][0, 0]
    pred_joints = fk_body_fbx_local_torch(
        body_pose_local_delta_6d=torch.from_numpy(features[schema.body_pose_key][:1]).float(),
        actor_root_pos_world=torch.from_numpy(features["root_pos_world"][:1]).float(),
        root_heading=torch.from_numpy(features["root_yaw"][:1]).float(),
        rest_local_positions=torch.from_numpy(fk_offsets[None]).float(),
        rest_local_rotations_6d=torch.from_numpy(features["joint_rest_local_rotations_6d"][None]).float(),
    )
    np.testing.assert_allclose(
        pred_joints[0, 0].detach().numpy(),
        source["joints_world"][0, 0],
        atol=1e-6,
    )
    arrays = dict(features)
    arrays["sensor_valid"] = np.ones((4, 6), dtype=bool)
    encoded = encode_realtime_pose_features(arrays, schema_name=REALTIME_POSE_SCHEMA_NAME)
    tracker_ref = encoded[:, schema.tracker_pos_slice()].reshape(4, 6, 3)
    np.testing.assert_allclose(tracker_ref[:, :, 1], features["tracker_pos_world"][:, :, 1], atol=1e-6)


def test_body_fbx_numpy_fk_matches_torch_fk():
    source = build_toy_realtime_source(frame_count=4)
    rest = build_synthetic_body_fbx_rest()
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    offsets = np.repeat(source["joint_offsets_parent"][None], 4, axis=0).astype(np.float32)
    offsets[:, 0, 1] = source[schema.pelvis_height_key][:, 0]

    numpy_joints, numpy_rotations = fk_body_fbx_local_delta(
        body_pose_local_delta_6d=source[schema.body_pose_key],
        actor_root_pos_world=source["root_pos_world"],
        root_heading=source["root_yaw"],
        rest=rest,
        local_offsets=offsets,
    )
    torch_joints, torch_rotations = fk_body_fbx_local_torch(
        body_pose_local_delta_6d=torch.from_numpy(source[schema.body_pose_key]).float(),
        actor_root_pos_world=torch.from_numpy(source["root_pos_world"]).float(),
        root_heading=torch.from_numpy(source["root_yaw"]).float(),
        rest_local_positions=torch.from_numpy(offsets).float(),
        rest_local_rotations_6d=torch.from_numpy(
            np.repeat(source["joint_rest_local_rotations_6d"][None], 4, axis=0)
        ).float(),
        return_global_rot=True,
    )

    np.testing.assert_allclose(numpy_joints, torch_joints.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(numpy_rotations, torch_rotations.detach().numpy(), atol=1e-6)


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
        assert int(np.asarray(task["target_start"]).item()) == REALTIME_POSE_TARGET_START
        assert int(np.asarray(task["target_length"]).item()) == REALTIME_POSE_TARGET_LENGTH


def test_task_loader_rejects_missing_root_y_policy(tmp_path):
    manifest_path, entry = generate_single_task_manifest(tmp_path)
    broken_task = manifest_path.parent / "missing_root_y_policy.npz"
    write_task_variant(manifest_path.parent / entry["task_path"], broken_task, drop_keys={"root_y_policy"})

    with pytest.raises(KeyError, match="root_y_policy"):
        load_materialized_task_npz(manifest_dir=manifest_path.parent, task_path=broken_task.name, schema_name=REALTIME_POSE_SCHEMA_NAME)


def test_task_loader_rejects_missing_or_wrong_target_contract(tmp_path):
    manifest_path, entry = generate_single_task_manifest(tmp_path)
    source_task = manifest_path.parent / entry["task_path"]
    missing_target = manifest_path.parent / "missing_target_start.npz"
    write_task_variant(source_task, missing_target, drop_keys={"target_start"})
    with pytest.raises(KeyError, match="target_start"):
        load_materialized_task_npz(manifest_dir=manifest_path.parent, task_path=missing_target.name, schema_name=REALTIME_POSE_SCHEMA_NAME)

    wrong_target = manifest_path.parent / "wrong_target_start.npz"
    write_task_variant(source_task, wrong_target, target_start=np.int64(0))
    with pytest.raises(ValueError, match="target_start"):
        load_materialized_task_npz(manifest_dir=manifest_path.parent, task_path=wrong_target.name, schema_name=REALTIME_POSE_SCHEMA_NAME)

    missing_length = manifest_path.parent / "missing_target_length.npz"
    write_task_variant(source_task, missing_length, drop_keys={"target_length"})
    with pytest.raises(KeyError, match="target_length"):
        load_materialized_task_npz(manifest_dir=manifest_path.parent, task_path=missing_length.name, schema_name=REALTIME_POSE_SCHEMA_NAME)


def test_task_loader_rejects_root_y0_invariant_error(tmp_path):
    manifest_path, entry = generate_single_task_manifest(tmp_path)
    source_task = manifest_path.parent / entry["task_path"]
    with np.load(source_task, allow_pickle=False) as data:
        bad_root_pos = data["root_pos_world"].copy()
    bad_root_pos[:, 1] = 0.25
    broken_task = manifest_path.parent / "bad_root_y0.npz"
    write_task_variant(source_task, broken_task, root_pos_world=bad_root_pos)

    with pytest.raises(ValueError, match="root-y0"):
        load_materialized_task_npz(manifest_dir=manifest_path.parent, task_path=broken_task.name, schema_name=REALTIME_POSE_SCHEMA_NAME)


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


def test_task_npz_temp_path_does_not_exceed_final_path_length(tmp_path):
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    target_len = 259
    stem_len = target_len - len(str(task_dir)) - 1 - len(".npz")
    if stem_len < 8:
        pytest.skip("临时目录路径过长，无法构造接近 Windows MAX_PATH 的最终文件名")
    task_path = task_dir / (("x" * stem_len) + ".npz")
    temp_path = temporary_task_path(task_path)

    assert len(str(temp_path)) <= len(str(task_path))
    save_task_npz(task_path, compress=False, value=np.asarray([1], dtype=np.float32))
    assert task_path.exists()
    assert not temp_path.exists()


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


def test_task_generator_accepts_legacy_marked_task_root_without_latest_pointer(tmp_path):
    source_dir = tmp_path / "sources"
    output_dir = tmp_path / "tasks"
    write_toy_source_dataset(source_dir)
    legacy_run = output_dir / "20240101_000000_rtp_tasks_seed10"
    legacy_run.mkdir(parents=True)
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    marker = {
        "schema_name": schema.name,
        "task_format": schema.task_format,
        POSE_REPRESENTATION_KEY: schema.pose_representation,
        "root_y_policy": schema.root_y_policy,
        "pelvis_height_mode": schema.pelvis_height_mode,
        "source_dir": str(source_dir),
        "split_dir": "",
    }
    (legacy_run / TASK_OUTPUT_MARKER).write_text(json.dumps(marker, ensure_ascii=False), encoding="utf-8")

    counts = generate_realtime_pose_tasks_main(
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

    assert counts["train"] == 1
    assert legacy_run.exists()
    assert (output_dir / "latest_tasks.json").exists()
    assert latest_artifact_dir(output_dir, kind="tasks") != legacy_run


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


def test_rollout_task_generation_and_dataset_shapes(tmp_path):
    source_dir = tmp_path / "sources"
    output_dir = tmp_path / "tasks"
    write_toy_source_dataset(source_dir, frame_count=REALTIME_POSE_SEQ_LEN + 1)
    generate_realtime_pose_tasks_main(
        [
            "--source_dir",
            str(source_dir),
            "--output_dir",
            str(output_dir),
            "--splits",
            "train",
            "--samples_per_file",
            "3",
            "--rollout_steps",
            "2",
            "--schema",
            REALTIME_POSE_SCHEMA_NAME,
            "--split_dir",
            "",
            "--overwrite",
        ]
    )

    task_run_dir = latest_artifact_dir(output_dir, "tasks")
    manifest_path = task_run_dir / "train" / "manifest.jsonl"
    entries = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert entries
    for entry in entries:
        assert entry["max_rollout_steps"] == 2
        assert len(entry["rollout_task_paths"]) == 1
        assert entry["start_frame"] == 0
        base_task = load_materialized_task_npz(manifest_dir=manifest_path.parent, task_path=entry["task_path"])
        rollout_task = load_materialized_task_npz(
            manifest_dir=manifest_path.parent,
            task_path=entry["rollout_task_paths"][0],
        )
        assert int(np.asarray(base_task["start_frame"]).item()) == 0
        assert int(np.asarray(rollout_task["start_frame"]).item()) == 1
        assert int(np.asarray(rollout_task["start_frame"]).item()) + REALTIME_POSE_SEQ_LEN <= int(
            np.asarray(rollout_task["source_frames"]).item()
        )

    dataset = RealtimePoseTaskDataset(
        output_dir,
        split="train",
        normalize_input=False,
        enable_rollout=True,
        rollout_steps=2,
    )
    item = dataset[0]
    assert tuple(item["x"].shape) == (REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SEQ_LEN)
    assert tuple(item["conditioned_x"].shape) == (REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SEQ_LEN)
    assert len(item["rollout"]) == 1
    assert tuple(item["rollout"][0]["x"].shape) == (REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SEQ_LEN)
    assert tuple(item["rollout"][0]["conditioned_x"].shape) == (REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SEQ_LEN)


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
