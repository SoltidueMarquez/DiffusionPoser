from __future__ import annotations

import numpy as np

from data_loaders.realtime_pose_dataset import encode_realtime_pose_features
from sample.visualization import decode_realtime_pose_joints, decode_root_yaw_from_delta
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source


def test_realtime_pose_fk_visualization_helpers():
    source = build_toy_realtime_source(frame_count=3)
    source["sensor_valid"] = np.ones((3, 6), dtype=bool)
    features = encode_realtime_pose_features(source)
    joints = decode_realtime_pose_joints(
        features=features,
        root_pos_world=source["root_pos_world"],
        root_yaw=source["root_yaw"],
        joint_offsets_parent=source["joint_offsets_parent"],
    )
    assert joints.shape == (3, 24, 3)
    yaw = decode_root_yaw_from_delta(0.5, np.asarray([0.0, 1.0], dtype=np.float32))
    assert abs(yaw - 0.5) < 1e-6
