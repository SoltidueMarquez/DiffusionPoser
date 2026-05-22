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


def test_tracker_reference_golden_uses_previous_frame_yaw():
    source = build_toy_realtime_source(frame_count=REALTIME_POSE_SEQ_LEN)
    sensor_valid = np.ones((REALTIME_POSE_SEQ_LEN, TRACKER_COUNT), dtype=bool)
    features = encode_realtime_pose_features({**source, "sensor_valid": sensor_valid})

    frame_index = REALTIME_POSE_SEQ_LEN - 1
    tracker_world = source["tracker_pos_world"][frame_index, 0]
    root_pos = source["root_pos_world"][frame_index]
    previous_yaw = source["root_yaw"][frame_index - 1]
    cos_yaw = np.cos(previous_yaw)
    sin_yaw = np.sin(previous_yaw)
    rotation = np.asarray(
        [
            [cos_yaw, 0.0, sin_yaw],
            [0.0, 1.0, 0.0],
            [-sin_yaw, 0.0, cos_yaw],
        ],
        dtype=np.float32,
    )
    expected = (tracker_world - root_pos) @ rotation
    actual = features[frame_index, TRACKER_POS_REF_START:TRACKER_POS_REF_START + 3]
    np.testing.assert_allclose(actual, expected, atol=1e-6)
