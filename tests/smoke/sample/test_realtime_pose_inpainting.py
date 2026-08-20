from __future__ import annotations

import torch
import pytest

from data_loaders.realtime_pose_config import IKInpaintingConfig
from data_loaders.realtime_pose_ik import (
    DIRECT_ROTATION,
    DIRECTION_ONLY,
    INHERITED,
    RealtimePoseIKResult,
)
from diffusion.realtime_pose_inpainting import build_realtime_pose_inpainting_condition


IDENTITY_6D = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
YAW_90_6D = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def _config() -> IKInpaintingConfig:
    return IKInpaintingConfig(
        fabrik_iterations=1,
        direction_only_quality=0.8,
        residual_scale=0.5,
        gap_low=0.1,
        gap_high=0.5,
        direction_support=0.35,
        untracked_strength=0.05,
    )


def _ik_result() -> RealtimePoseIKResult:
    pose = IDENTITY_6D.repeat(24).reshape(1, 24, 6)
    pose[:, 1:4] = YAW_90_6D
    constraint_type = torch.full((1, 24), INHERITED, dtype=torch.long)
    constraint_type[:, 0:2] = DIRECT_ROTATION
    constraint_type[:, 2] = DIRECTION_ONLY
    updated = constraint_type != INHERITED
    return RealtimePoseIKResult(
        pose=pose,
        updated_mask=updated,
        direct_rotation_mask=constraint_type == DIRECT_ROTATION,
        constraint_type=constraint_type,
        position_residual=torch.zeros(1, 24),
        confidence=torch.ones(1, 24),
    )


def test_ik_gap_controls_direct_direction_and_untracked_strength():
    predictor = IDENTITY_6D.repeat(24)[None]
    condition = build_realtime_pose_inpainting_condition(
        _ik_result(),
        predictor,
        pose_mean=None,
        pose_scale=None,
        config=_config(),
    )
    assert condition.ik_residual.shape == (1, 24, 6)
    assert condition.ik_gap.shape == (1, 24)
    assert condition.denoise_strength[0, 0] == 0.05
    assert condition.denoise_strength[0, 1] == 1.0
    torch.testing.assert_close(
        condition.denoise_strength[0, 2],
        torch.tensor(0.05 + 0.95 * 0.35),
    )
    assert condition.denoise_strength[0, 3] == 0.05


def test_strength_is_monotonic_in_gap():
    predictor = IDENTITY_6D.repeat(24)[None].repeat(3, 1)
    base = _ik_result()
    result = RealtimePoseIKResult(
        pose=base.pose.repeat(3, 1, 1),
        updated_mask=base.updated_mask.repeat(3, 1),
        direct_rotation_mask=base.direct_rotation_mask.repeat(3, 1),
        constraint_type=base.constraint_type.repeat(3, 1),
        position_residual=base.position_residual.repeat(3, 1),
        confidence=base.confidence.repeat(3, 1),
    )
    # 第 1 个 direct joint 首项 gap 低于 low，后两项高于 high。
    result.pose[0, 1] = IDENTITY_6D
    condition = build_realtime_pose_inpainting_condition(
        result, predictor, None, None, _config()
    )
    assert torch.all(
        condition.denoise_strength[1:, 1]
        >= condition.denoise_strength[:-1, 1]
    )


def test_gap_calibration_is_required_and_must_have_nonzero_range():
    with pytest.raises(ValueError, match="gap 尚未校准"):
        IKInpaintingConfig(
            direction_only_quality=0.8,
            residual_scale=0.5,
        ).validate()
    with pytest.raises(ValueError, match="至少比 gap_low 大"):
        IKInpaintingConfig(
            direction_only_quality=0.8,
            residual_scale=0.5,
            gap_low=0.2,
            gap_high=0.20001,
        ).validate()
