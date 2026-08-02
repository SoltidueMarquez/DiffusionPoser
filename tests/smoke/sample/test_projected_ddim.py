from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from data_loaders.realtime_pose_kinematics import rotation_6d_to_matrix_torch
from data_loaders.sensor_masking import TRACKER_TO_JOINT
from diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)
from diffusion.realtime_pose_projection import (
    project_realtime_pose_xstart,
    project_rotation_6d_to_so3,
)


IDENTITY_6D = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])


def _tracker(batch_size: int = 2) -> torch.Tensor:
    tracker = torch.zeros(batch_size, 6, 13)
    tracker[..., 3:9] = IDENTITY_6D
    tracker[..., 9:11] = 1.0
    tracker[..., 12] = 1.0
    return tracker


def test_so3_projection_is_idempotent_and_hard_rotation_is_exact():
    torch.manual_seed(1)
    raw = torch.randn(2, 144)
    tracker = _tracker()
    hard = torch.zeros(2, 6, dtype=torch.bool)
    hard[:, 0] = True
    hard[:, 3] = True
    deployed = project_realtime_pose_xstart(raw, tracker, hard)
    repeated = project_realtime_pose_xstart(deployed, tracker, hard)
    torch.testing.assert_close(deployed, repeated, atol=1e-5, rtol=1e-5)
    rotations = rotation_6d_to_matrix_torch(deployed.reshape(2, 24, 6))
    identity = torch.eye(3).expand_as(rotations)
    torch.testing.assert_close(rotations.transpose(-1, -2) @ rotations, identity, atol=1e-5, rtol=1e-5)
    assert torch.all(torch.linalg.det(rotations) > 0.9999)
    for tracker_index in (0, 3):
        joint_index = TRACKER_TO_JOINT[tracker_index]
        torch.testing.assert_close(deployed[:, joint_index * 6 : joint_index * 6 + 6], tracker[:, tracker_index, 3:9])


def test_soft_tracker_is_not_replaced():
    raw = project_rotation_6d_to_so3(torch.randn(1, 24, 6)).reshape(1, 144)
    tracker = _tracker(batch_size=1)
    tracker[:, 1, 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    hard = torch.zeros(1, 6, dtype=torch.bool)
    hard[:, 0] = True
    deployed = project_realtime_pose_xstart(raw, tracker, hard)
    wrist = TRACKER_TO_JOINT[1]
    torch.testing.assert_close(deployed[:, wrist * 6 : wrist * 6 + 6], raw[:, wrist * 6 : wrist * 6 + 6])


class _ConstantXStart(nn.Module):
    def __init__(self, value: torch.Tensor):
        super().__init__()
        self.register_buffer("value", value)

    def forward(self, x, timestep, **kwargs):
        del timestep, kwargs
        return self.value.expand_as(x)


class _AuxiliaryXStart(_ConstantXStart):
    def forward(self, x, timestep, return_aux_outputs=False, **kwargs):
        value = super().forward(x, timestep, **kwargs)
        if not return_aux_outputs:
            return value
        return value, {
            "future_leg": torch.zeros(x.shape[0], 3, 8, 6, device=x.device),
            "contact_logits": torch.ones(x.shape[0], 2, device=x.device),
        }


def test_projected_ddim_recomputes_epsilon_from_deployed_xstart():
    diffusion = GaussianDiffusion(
        betas=np.asarray([0.1, 0.2], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    x = torch.tensor([[0.5, -0.25]], dtype=torch.float32)
    raw_xstart = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    deployed_xstart = torch.tensor([[0.3, -0.4]], dtype=torch.float32)
    timestep = torch.tensor([1], dtype=torch.long)
    result = diffusion.projected_ddim_sample(
        _ConstantXStart(raw_xstart),
        x,
        timestep,
        projection_fn=lambda _: deployed_xstart,
        eta=0.0,
        projection_mode="all_steps",
    )
    alpha = torch.tensor(diffusion.alphas_cumprod[1], dtype=x.dtype)
    alpha_prev = torch.tensor(diffusion.alphas_cumprod_prev[1], dtype=x.dtype)
    epsilon = (x - torch.sqrt(alpha) * deployed_xstart) / torch.sqrt(1.0 - alpha)
    expected = torch.sqrt(alpha_prev) * deployed_xstart + torch.sqrt(1.0 - alpha_prev) * epsilon
    torch.testing.assert_close(result["sample"], expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(result["raw_pred_xstart"], raw_xstart)
    torch.testing.assert_close(result["deployed_pred_xstart"], deployed_xstart)


def test_projected_ddim_returns_final_auxiliary_heads():
    diffusion = GaussianDiffusion(
        betas=np.asarray([0.1, 0.2], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    result = diffusion.projected_ddim_sample_loop(
        _AuxiliaryXStart(torch.zeros(1, 4)),
        shape=(1, 4),
        projection_fn=lambda value: value,
        device=torch.device("cpu"),
    )
    assert result["auxiliary_outputs"]["future_leg"].shape == (1, 3, 8, 6)
    assert result["auxiliary_outputs"]["contact_logits"].shape == (1, 2)
