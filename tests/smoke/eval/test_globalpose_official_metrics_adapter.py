from __future__ import annotations

import numpy as np
import pytest

from data_converter.amass_smpl_utils import AMASS_TO_UNITY
from data_loaders.body_fbx_kinematics import SOURCE_FK_TO_BODY_FBX_BASIS
from data_loaders.realtime_pose_kinematics import rotation_6d_forward_up_np
from data_loaders.sensor_masking import REALTIME_POSE_SCHEMA_NAME, get_schema_spec
from eval.globalpose_metrics import compute_translation_drift
from eval.globalpose_official_metrics_adapter import (
    aggregate_translation_drift_by_sequence,
    compute_roundtrip_diagnostics,
    decode_body_fbx_delta_to_smpl_local_rotations,
    ensure_numpy_legacy_aliases_for_chumpy,
    GlobalPoseMotion,
    motion_from_rollout_payload,
)


def rotation_z(angle: float) -> np.ndarray:
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.asarray(
        [
            [cos_angle, -sin_angle, 0.0],
            [sin_angle, cos_angle, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def identity_body_features(frame_count: int) -> np.ndarray:
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    features = np.zeros((frame_count, schema.feature_dim), dtype=np.float32)
    identity_6d = rotation_6d_forward_up_np(np.eye(3, dtype=np.float64)).astype(np.float32)
    features[:, schema.body_pose_slice()] = np.tile(identity_6d, 24)
    return features


def straight_walk(length_m: float = 10.0, frame_count: int = 101) -> np.ndarray:
    tran = np.zeros((frame_count, 3), dtype=np.float32)
    tran[:, 0] = np.linspace(0.0, length_m, frame_count, dtype=np.float32)
    return tran


def test_decode_body_fbx_delta_recovers_non_root_smpl_local_rotations():
    frame_count = 2
    source_local_smpl = np.repeat(np.eye(3, dtype=np.float64)[None, None], frame_count * 24, axis=0).reshape(
        frame_count,
        24,
        3,
        3,
    )
    source_local_smpl[0, 1] = rotation_z(0.25)
    source_local_smpl[1, 1] = rotation_z(-0.5)

    source_local_unity = AMASS_TO_UNITY[None, None] @ source_local_smpl @ AMASS_TO_UNITY.T[None, None]
    basis = SOURCE_FK_TO_BODY_FBX_BASIS.astype(np.float64)
    body_delta = basis[None, None] @ source_local_unity @ basis.T[None, None]
    body_delta[:, 0] = np.eye(3)
    body_pose_6d = rotation_6d_forward_up_np(body_delta).reshape(frame_count, -1)

    decoded = decode_body_fbx_delta_to_smpl_local_rotations(body_pose_6d, root_yaw=np.zeros(frame_count))

    np.testing.assert_allclose(decoded[:, 1], source_local_smpl[:, 1], atol=1e-6)
    expected_root = np.repeat(np.eye(3, dtype=np.float64)[None], frame_count, axis=0)
    np.testing.assert_allclose(decoded[:, 0], expected_root, atol=1e-6)


def test_motion_from_rollout_payload_uses_metadata_sequence_name_and_smpl_coordinates():
    frame_count = 2
    features = identity_body_features(frame_count)
    predicted_joints_world = np.zeros((frame_count, 24, 3), dtype=np.float32)
    predicted_joints_world[:, 0] = np.asarray([[1.0, 3.0, 2.0], [2.0, 5.0, 4.0]], dtype=np.float32)
    expected_tran = predicted_joints_world[:, 0].astype(np.float64) @ AMASS_TO_UNITY
    payload = {
        "predicted_features_raw": features[None],
        "root_yaw_predicted": np.zeros((1, frame_count), dtype=np.float32),
        "predicted_joints_world": predicted_joints_world[None],
        "metadata": np.asarray({"source_relative_path": "totalcapture_officalib/s1_rom2.npz"}, dtype=object),
    }

    motion = motion_from_rollout_payload(payload, kind="predicted")

    assert motion.sequence_name == "s1_rom2"
    assert motion.pose.shape == (frame_count, 24, 3, 3)
    np.testing.assert_allclose(motion.tran, expected_tran, atol=1e-6)


def test_aggregate_translation_drift_by_sequence_matches_globalpose_sequence_averaging():
    target = straight_walk()
    seq_a = compute_translation_drift(target * np.asarray([0.9, 1.0, 1.0], dtype=np.float32), target)
    seq_b = compute_translation_drift(target * np.asarray([0.8, 1.0, 1.0], dtype=np.float32), target)

    aggregate = aggregate_translation_drift_by_sequence([seq_a, seq_b], window_sizes=range(1, 8))

    assert aggregate[7]["sequence_count"] == 2
    assert aggregate[7]["mean_m"] == pytest.approx(1.05, abs=1e-5)
    assert aggregate[7]["drift_percent"] == pytest.approx(15.0, abs=1e-5)


def test_compute_roundtrip_diagnostics_ignores_constant_translation_offset():
    frame_count = 4
    pose = np.repeat(np.eye(3, dtype=np.float32)[None, None], frame_count * 24, axis=0).reshape(
        frame_count,
        24,
        3,
        3,
    )
    target_tran = straight_walk(length_m=0.3, frame_count=frame_count)
    recovered_tran = target_tran + np.asarray([0.2, -0.1, 0.05], dtype=np.float32)
    recovered = GlobalPoseMotion(sequence_name="s1_walk", pose=pose, tran=recovered_tran)
    target = GlobalPoseMotion(sequence_name="s1_walk", pose=pose.copy(), tran=target_tran)

    diagnostics = compute_roundtrip_diagnostics(recovered, target)

    assert diagnostics["nonroot_local_angle_deg"] == pytest.approx(0.0, abs=1e-6)
    assert diagnostics["metric_nonroot_local_angle_deg"] == pytest.approx(0.0, abs=1e-6)
    assert diagnostics["translation_delta_rmse_m"] == pytest.approx(0.0, abs=1e-6)


def test_compute_roundtrip_diagnostics_reports_metric_nonroot_without_ignored_joints():
    frame_count = 2
    target_pose = np.repeat(np.eye(3, dtype=np.float32)[None, None], frame_count * 24, axis=0).reshape(
        frame_count,
        24,
        3,
        3,
    )
    recovered_pose = target_pose.copy()
    recovered_pose[:, 22] = rotation_z(0.5)
    tran = straight_walk(length_m=0.1, frame_count=frame_count)
    recovered = GlobalPoseMotion(sequence_name="s1_walk", pose=recovered_pose, tran=tran)
    target = GlobalPoseMotion(sequence_name="s1_walk", pose=target_pose, tran=tran.copy())

    diagnostics = compute_roundtrip_diagnostics(recovered, target)

    assert diagnostics["nonroot_local_angle_deg"] > 0.0
    assert diagnostics["metric_nonroot_local_angle_deg"] == pytest.approx(0.0, abs=1e-6)


def test_ensure_numpy_legacy_aliases_for_chumpy_installs_missing_aliases():
    ensure_numpy_legacy_aliases_for_chumpy()

    assert np.int is int
    assert np.float is float
    assert np.complex is complex
    assert np.object is object
    assert np.unicode is str
    assert np.str is str
