from __future__ import annotations

import numpy as np
import torch

from data_loaders.realtime_pose_ik import RealtimePoseIKResult
from diffusion.realtime_pose_inpainting import (
    apply_realtime_pose_inpainting,
    build_realtime_pose_inpainting_condition,
    confidence_to_release_level,
)


def _ik_result():
    pose = torch.zeros(1, 24, 6)
    pose[..., 2] = 1.0
    pose[..., 4] = 1.0
    updated = torch.zeros(1, 24, dtype=torch.bool)
    updated[:, :3] = True
    confidence = torch.zeros(1, 24)
    confidence[:, 0] = 1.0
    confidence[:, 1] = 0.5
    confidence[:, 2] = 0.1
    return RealtimePoseIKResult(
        pose=pose,
        updated_mask=updated,
        direct_rotation_mask=updated.clone(),
        constraint_type=torch.zeros(1, 24, dtype=torch.long),
        position_residual=torch.zeros(1, 24),
        confidence=confidence,
    )


def test_single_frame_inpainting_contract_and_fixed_known_noise():
    known = torch.randn(1, 144)
    condition = build_realtime_pose_inpainting_condition(
        _ik_result(), torch.zeros(144), torch.ones(144), known_noise=known
    )
    assert condition.pose.shape == (1, 144)
    assert condition.valid.shape == (1, 24)
    assert condition.release_level.shape == (1, 24)
    assert condition.known_noise.data_ptr() == known.data_ptr()


def test_release_level_keeps_high_confidence_longer_and_absent_bits_unchanged():
    condition = build_realtime_pose_inpainting_condition(
        _ik_result(), None, None, known_noise=torch.zeros(1, 144)
    )
    assert condition.release_level[0, 0] < condition.release_level[0, 1]
    assert condition.release_level[0, 1] < condition.release_level[0, 2]
    x_t = torch.randn(1, 144)
    injected, active = apply_realtime_pose_inpainting(
        x_t,
        torch.tensor([0]),
        condition,
        np.asarray([0.9999, 0.5], dtype=np.float64),
    )
    assert active.shape == (1, 24)
    torch.testing.assert_close(
        injected.reshape(1, 24, 6)[:, 3:], x_t.reshape(1, 24, 6)[:, 3:]
    )


def test_confidence_release_endpoints():
    release = confidence_to_release_level(torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(release, torch.tensor([1.0, 0.0]), atol=1e-6, rtol=0)
