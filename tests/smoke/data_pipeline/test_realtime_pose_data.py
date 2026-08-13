from __future__ import annotations

import numpy as np
import pytest
import torch

from data_loaders.sensor_masking import BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source
from utils.normalizer import RealtimePoseNormalizer


def test_reusable_source_keeps_local_pose_fk_and_stationary_fields():
    source = build_toy_realtime_source(frame_count=70)
    assert source[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY].shape == (70, 144)
    assert source["stationary_prob_5"].shape == (70, 5)
    assert source["joint_rest_local_rotations_6d"].shape == (24, 6)
    np.testing.assert_allclose(source["root_pos_world"][:, 1], 0.0, atol=1e-7)


def test_old_mean_std_normalizer_fails_fast(tmp_path):
    torch.save(torch.zeros(214), tmp_path / "mean.pt")
    torch.save(torch.ones(214), tmp_path / "std.pt")
    with pytest.raises(FileNotFoundError):
        RealtimePoseNormalizer(tmp_path)


def test_pose_scale_is_the_single_pose_conversion_contract(tmp_path):
    writer = RealtimePoseNormalizer(tmp_path, eps=0.25, disable=True)
    writer.save(
        pose_mean=torch.ones(144),
        pose_scale=torch.full((144,), 2.0),
        tracker_mean=torch.zeros(6, 9),
        tracker_std=torch.ones(6, 9),
        head_path_xz_mean=torch.zeros(2),
        head_path_xz_std=torch.ones(2),
        head_height_mean=torch.tensor(0.0),
        head_height_std=torch.tensor(1.0),
    )

    normalizer = RealtimePoseNormalizer(tmp_path, eps=1e-8)
    assert normalizer.eps == pytest.approx(1e-8)
    torch.testing.assert_close(normalizer.pose_scale, torch.full((144,), 2.0))
    raw = torch.full((2, 144), 5.0)
    normalized = normalizer.normalize_pose(raw)
    torch.testing.assert_close(normalized, torch.full((2, 144), 2.0))
    torch.testing.assert_close(normalizer.inverse_pose(normalized), raw)
