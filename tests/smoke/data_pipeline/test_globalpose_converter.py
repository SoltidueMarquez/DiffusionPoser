from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import data_converter.globalpose_to_realtime_pose as globalpose_converter
from data_converter.amass_smpl_utils import SmplMotion
from data_loaders.body_fbx_kinematics import build_synthetic_body_fbx_rest
from data_loaders.realtime_pose_contract import load_source_metadata, validate_realtime_source_contract
from data_loaders.sensor_masking import REALTIME_POSE_SEQ_LEN, get_schema_spec


def write_toy_globalpose_dataset(path: Path, frame_count: int = REALTIME_POSE_SEQ_LEN + 4) -> None:
    pose = torch.zeros((frame_count, 72), dtype=torch.float32)
    tran = torch.zeros((frame_count, 3), dtype=torch.float32)
    tran[:, 0] = torch.linspace(0.0, 0.2, frame_count)
    identity = torch.eye(3, dtype=torch.float32)
    payload = {
        "name": ["toy_seq"],
        "pose": [pose],
        "tran": [tran],
        "aS": [torch.zeros((frame_count, 6, 3), dtype=torch.float32)],
        "wS": [torch.zeros((frame_count, 6, 3), dtype=torch.float32)],
        "mS": [torch.zeros((frame_count, 6, 3), dtype=torch.float32)],
        "RIS": [identity.repeat(frame_count, 6, 1, 1)],
        "RIM": [identity.repeat(6, 1, 1)],
        "RSB": [identity.repeat(6, 1, 1)],
    }
    torch.save(payload, path)


def fake_smpl_motion(frame_count: int) -> SmplMotion:
    joints = np.zeros((frame_count, 24, 3), dtype=np.float64)
    joints[:, :, 1] = np.linspace(0.9, 1.2, 24, dtype=np.float64)
    joints[:, :, 0] += np.linspace(0.0, 0.2, frame_count, dtype=np.float64)[:, None]
    rotations = np.repeat(np.eye(3, dtype=np.float64)[None, None], frame_count * 24, axis=0).reshape(
        frame_count,
        24,
        3,
        3,
    )
    parents = np.array(
        [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
        dtype=np.int64,
    )
    return SmplMotion(
        raw_joint_positions=joints.copy(),
        joint_positions=joints,
        joint_rotations=rotations,
        rest_joints=joints[:1].copy(),
        parents=parents,
    )


def test_globalpose_converter_resolves_schema_aware_output_dir(tmp_path):
    config_path = tmp_path / "data_roots.json"
    generated_root = tmp_path / "generated"
    amass_root = tmp_path / "AMASS"
    dataset_path = tmp_path / "totalcapture_officalib.pt"
    config_path.write_text(
        json.dumps({"amass_root": str(amass_root), "generated_root": str(generated_root)}),
        encoding="utf-8",
    )

    args = globalpose_converter.parse_args(
        [
            "--data_roots_config",
            str(config_path),
            "--globalpose_dataset",
            str(dataset_path),
            "--source_set_name",
            "globalpose_totalcapture_officalib_oracle",
        ]
    )
    resolved = globalpose_converter.resolve_converter_paths(args)

    assert resolved.output_dir == (
        generated_root
        / "sources"
        / "realtime_pose_stationary5_v1"
        / "globalpose_totalcapture_officalib_oracle"
    )


def test_globalpose_converter_writes_oracle_source_and_manifest(tmp_path, monkeypatch):
    dataset_path = tmp_path / "totalcapture_officalib.pt"
    write_toy_globalpose_dataset(dataset_path)
    output_dir = tmp_path / "sources"
    schema = get_schema_spec("realtime_pose_stationary5_v1")

    def fake_builder(*, pose_axis_angle, tran, source, model_cache, batch_size):
        assert pose_axis_angle.shape[1] == 72
        assert tran.shape[1] == 3
        return fake_smpl_motion(pose_axis_angle.shape[0])

    monkeypatch.setattr(globalpose_converter, "build_smpl_motion_from_globalpose_sequence", fake_builder)
    args = globalpose_converter.resolve_converter_paths(
        globalpose_converter.parse_args(
            [
                "--globalpose_dataset",
                str(dataset_path),
                "--dataset_name",
                "totalcapture_officalib",
                "--output_dir",
                str(output_dir),
                "--source_set_name",
                "globalpose_totalcapture_officalib_oracle",
                "--overwrite",
            ]
        )
    )

    counts = globalpose_converter.convert_globalpose_dataset(
        args,
        body_fbx_rest=build_synthetic_body_fbx_rest(),
        model_cache=None,
    )

    assert counts == {"converted": 1, "skipped_existing": 0, "failed": 0}
    source_path = output_dir / "totalcapture_officalib" / "toy_seq.npz"
    assert source_path.exists()
    with np.load(source_path, allow_pickle=False) as data:
        validate_realtime_source_contract(data, schema=schema, source=str(source_path))
        metadata = load_source_metadata(data, source=str(source_path))
        assert metadata["raw_dataset"] == "GlobalPose"
        assert metadata["globalpose_dataset_name"] == "totalcapture_officalib"
        assert metadata["globalpose_sequence_name"] == "toy_seq"
        assert metadata["tracker_source"] == "oracle_gt_pose_tran"
        assert data["tracker_pos_world"].shape == (REALTIME_POSE_SEQ_LEN + 4, 6, 3)

    manifest_path = output_dir / "manifest.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["source_relative_path"] == "totalcapture_officalib/toy_seq.npz"
    assert records[0]["stablemotion_split_key"] == "GlobalPose/totalcapture_officalib/toy_seq"
