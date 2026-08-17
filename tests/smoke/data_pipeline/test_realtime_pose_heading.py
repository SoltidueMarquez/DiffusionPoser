from __future__ import annotations

import math

import numpy as np
import torch

from data_loaders.realtime_pose_geometry import (
    extract_continuous_rotation_heading_np,
    extract_rotation_heading_np,
    extract_rotation_heading_torch,
)
from data_loaders.realtime_pose_kinematics import (
    make_yaw_rotation_np,
    make_yaw_rotation_torch,
)


def _pitch_rotation_np(angle: np.ndarray) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    result = np.zeros((angle.shape[0], 3, 3), dtype=np.float64)
    result[:, 0, 0] = 1.0
    result[:, 1, 1] = cosine
    result[:, 1, 2] = -sine
    result[:, 2, 1] = sine
    result[:, 2, 2] = cosine
    return result


def _pitch_rotation_torch(angle: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    result = torch.zeros((*angle.shape, 3, 3), dtype=angle.dtype, device=angle.device)
    result[..., 0, 0] = 1.0
    result[..., 1, 1] = cosine
    result[..., 1, 2] = -sine
    result[..., 2, 1] = sine
    result[..., 2, 2] = cosine
    return result


def test_swing_twist_heading_stays_fixed_through_vertical_forward_axis():
    yaw = 0.7
    pitch = np.radians(np.linspace(-170.0, 170.0, 69))
    rotations = make_yaw_rotation_np(np.full(pitch.shape, yaw)) @ _pitch_rotation_np(
        pitch
    )

    extracted = extract_rotation_heading_np(rotations)

    np.testing.assert_allclose(extracted, yaw, atol=1e-6)


def test_continuous_heading_unwraps_pi_and_holds_true_singularity():
    yaws = np.radians(np.asarray([179.0, 181.0, 183.0]))
    rotations = make_yaw_rotation_np(yaws)
    rotations[1] = _pitch_rotation_np(np.asarray([math.pi]))[0]

    extracted = extract_continuous_rotation_heading_np(
        rotations,
        initial_yaw=float(yaws[0]),
    )

    assert extracted[0] == np.float32(yaws[0])
    assert extracted[1] == extracted[0]
    assert extracted[2] > math.pi
    assert extracted[2] == np.float32(yaws[2])


def test_numpy_torch_heading_match_and_torch_gradient_is_finite():
    yaw = torch.tensor([0.2, -0.8], dtype=torch.float64, requires_grad=True)
    pitch = torch.tensor([math.pi / 2.0, -1.2], dtype=torch.float64)
    rotations = make_yaw_rotation_torch(yaw) @ _pitch_rotation_torch(pitch)

    torch_heading = extract_rotation_heading_torch(rotations)
    numpy_heading = extract_rotation_heading_np(rotations.detach().numpy())
    torch_heading.square().sum().backward()

    np.testing.assert_allclose(torch_heading.detach().numpy(), numpy_heading, atol=1e-7)
    assert yaw.grad is not None
    assert torch.isfinite(yaw.grad).all()
