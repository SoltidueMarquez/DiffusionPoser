from __future__ import annotations

import numpy as np

from data_loaders.realtime_pose_kinematics import derive_foot_contact_prob_2
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source


def test_stationary_source_field_is_available_for_contact_supervision():
    source = build_toy_realtime_source(frame_count=70)
    assert "stationary_prob_5" in source


def test_foot_contact_probability_softly_rejects_stationary_feet_above_floor():
    stationary = np.zeros((3, 5), dtype=np.float32)
    stationary[:, 1:3] = np.asarray([0.8, 0.6], dtype=np.float32)
    joints = np.zeros((3, 24, 3), dtype=np.float32)
    floor_y = np.asarray([0.0, 0.1, -0.1], dtype=np.float32)
    relative_foot_height = np.asarray(
        [[0.04, 0.05], [0.075, 0.075], [0.10, 0.20]],
        dtype=np.float32,
    )
    joints[:, 10:12, 1] = floor_y[:, None] + relative_foot_height

    contact = derive_foot_contact_prob_2(
        stationary_prob_5=stationary,
        joints_world=joints,
        floor_y=floor_y,
    )

    np.testing.assert_allclose(
        contact,
        [[0.8, 0.6], [0.4, 0.3], [0.0, 0.0]],
        atol=1e-6,
    )
