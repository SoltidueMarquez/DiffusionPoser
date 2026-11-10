import numpy as np

from data_converter.amass_smpl_utils import AMASS_TO_UNITY, local_to_global_rotations
from data_loaders.body_fbx_kinematics import (
    build_synthetic_body_fbx_rest, extract_root_heading_from_source_pelvis_up,
    fk_body_fbx_local_delta, source_global_rotations_to_body_fbx_local_delta_6d,
)
from data_loaders.realtime_pose_kinematics import SMPL_PARENTS
from sample.analyze_dropout_visual_quality import retarget_rpm_to_source
from sample.prepare_dropout_baseline_inputs import mask_rpm_hand_features
from sample.three_point_baseline_data import matrix_to_first_two_columns_6d


def test_rpm_gap_mask_uses_official_feature_channels():
    sparse = np.ones((10, 54), np.float32)
    masked = mask_rpm_hand_features(sparse, [[(2, 5)], [(6, 8)]])
    left = [*range(6, 12), *range(24, 30), *range(39, 42), *range(48, 51)]
    right = [*range(12, 18), *range(30, 36), *range(42, 45), *range(51, 54)]
    assert not masked[2:5, left].any()
    assert not masked[6:8, right].any()
    assert masked[5].all()  # 官方恢复首帧全部通道恢复，不套用 Ours 的速度遮挡规则。
    assert masked[:, [0, 18, 36, 45]].all()
    assert sparse.all()


def test_retarget_preserves_common_skeleton_and_observed_head():
    from scipy.spatial.transform import Rotation
    rest = build_synthetic_body_fbx_rest()
    local = Rotation.random(4 * 24, random_state=8).as_matrix().reshape(4, 24, 3, 3)
    local[:, 22:] = np.eye(3)
    world = AMASS_TO_UNITY @ local_to_global_rotations(local, SMPL_PARENTS) @ AMASS_TO_UNITY.T
    yaw = extract_root_heading_from_source_pelvis_up(world[:, 0])
    delta = source_global_rotations_to_body_fbx_local_delta_6d(world, yaw)
    translation = np.arange(12).reshape(4, 3).astype(np.float32)
    expected, expected_rotations = fk_body_fbx_local_delta(delta, translation, yaw, rest)
    motion = matrix_to_first_two_columns_6d(local[:, :22]).reshape(4, 132)
    positions, rotations, predicted_yaw = retarget_rpm_to_source(motion, expected[:, 15], rest)
    np.testing.assert_allclose(positions, expected, atol=2e-6)
    np.testing.assert_allclose(rotations, expected_rotations, atol=2e-6)
    np.testing.assert_allclose(predicted_yaw, yaw, atol=1e-6)
