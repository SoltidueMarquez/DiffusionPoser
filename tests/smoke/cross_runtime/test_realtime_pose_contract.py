from __future__ import annotations

import numpy as np

from data_loaders.realtime_pose_dataset import encode_realtime_pose_features
from data_loaders.sensor_masking import (
    REALTIME_POSE_INPUT_DIM,
    REALTIME_POSE_SEQ_LEN,
    ROOT_YAW_DELTA_START,
    TRACKER_COUNT,
    TRACKER_POS_REF_START,
)
from data_loaders.tracker_codec import (
    TRACKER_REF_SOURCE_CURRENT_HIP,
    build_tracker_reference_np,
    encode_tracker_positions_np,
)
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source


def test_feature_major_golden_layout_matches_unity_indexing():
    source = build_toy_realtime_source(frame_count=REALTIME_POSE_SEQ_LEN)
    sensor_valid = np.ones((REALTIME_POSE_SEQ_LEN, TRACKER_COUNT), dtype=bool)
    features = encode_realtime_pose_features({**source, "sensor_valid": sensor_valid})

    feature_major = features.T.reshape(-1)
    for frame_index in (0, REALTIME_POSE_SEQ_LEN - 2, REALTIME_POSE_SEQ_LEN - 1):
        for channel in (0, ROOT_YAW_DELTA_START, TRACKER_POS_REF_START, REALTIME_POSE_INPUT_DIM - 1):
            unity_index = channel * REALTIME_POSE_SEQ_LEN + frame_index
            assert feature_major[unity_index] == features[frame_index, channel]


def test_tracker_reference_golden_uses_current_hip_when_valid():
    source = build_toy_realtime_source(frame_count=REALTIME_POSE_SEQ_LEN)
    sensor_valid = np.ones((REALTIME_POSE_SEQ_LEN, TRACKER_COUNT), dtype=bool)
    features = encode_realtime_pose_features({**source, "sensor_valid": sensor_valid})

    frame_index = REALTIME_POSE_SEQ_LEN - 1
    previous_root = np.concatenate([source["root_pos_world"][:1], source["root_pos_world"][:-1]], axis=0)
    previous_yaw = np.concatenate([source["root_yaw"][:1], source["root_yaw"][:-1]], axis=0)
    ref_pos, ref_yaw, ref_source = build_tracker_reference_np(
        tracker_pos_world=source["tracker_pos_world"],
        tracker_rot_world_6d=source["tracker_rot_world_6d"],
        sensor_valid=sensor_valid,
        previous_final_root_pos_world=previous_root,
        previous_final_root_yaw=previous_yaw,
        pelvis_offset_parent=source["joint_offsets_parent"][0],
    )
    expected = encode_tracker_positions_np(source["tracker_pos_world"], ref_pos, ref_yaw)[frame_index, 0]
    actual = features[frame_index, TRACKER_POS_REF_START:TRACKER_POS_REF_START + 3]
    assert ref_source[frame_index] == TRACKER_REF_SOURCE_CURRENT_HIP
    np.testing.assert_allclose(actual, expected, atol=1e-6)
