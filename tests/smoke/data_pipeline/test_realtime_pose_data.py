from __future__ import annotations

import numpy as np
import pytest
import torch

from data_converter.amass_to_realtime_pose import parse_args
from data_loaders.realtime_pose_kinematics import derive_stationary_prob_5
from data_loaders.sensor_masking import (
    BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY,
    REALTIME_POSE_FPS,
)
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source
from utils.normalizer import RealtimePoseNormalizer


def test_reusable_source_keeps_local_pose_fk_and_stationary_fields():
    source = build_toy_realtime_source(frame_count=70)
    assert source[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY].shape == (70, 144)
    assert source["stationary_prob_5"].shape == (70, 5)
    assert source["joint_rest_local_rotations_6d"].shape == (24, 6)
    np.testing.assert_allclose(source["root_pos_world"][:, 1], 0.0, atol=1e-7)


def test_realtime_pose_converter_and_stationary_speed_use_30hz():
    args = parse_args([])
    assert REALTIME_POSE_FPS == 30.0
    assert args.target_fps == 30.0
    assert str(args.output_dir).endswith("_30hz")

    joints = np.zeros((3, 24, 3), dtype=np.float32)
    joints[:, :, 0] = np.arange(3, dtype=np.float32)[:, None] * 0.005
    probability = derive_stationary_prob_5(joints, median_window=1)
    np.testing.assert_allclose(probability, 0.4, atol=1e-6)


def test_pose_scale_is_the_single_pose_conversion_contract(tmp_path):
    writer = RealtimePoseNormalizer(tmp_path, eps=0.25, disable=True)
    writer.save(
        pose_mean=torch.ones(144),
        pose_scale=torch.full((144,), 2.0),
        tracker_mean=torch.zeros(6, 9),
        tracker_std=torch.ones(6, 9),
        predictor_sparse_mean=torch.zeros(54),
        predictor_sparse_std=torch.ones(54),
    )

    normalizer = RealtimePoseNormalizer(tmp_path, eps=1e-8)
    assert normalizer.eps == pytest.approx(1e-8)
    torch.testing.assert_close(normalizer.pose_scale, torch.full((144,), 2.0))
    raw = torch.full((2, 144), 5.0)
    normalized = normalizer.normalize_pose(raw)
    torch.testing.assert_close(normalized, torch.full((2, 144), 2.0))
    torch.testing.assert_close(normalizer.inverse_pose(normalized), raw)
