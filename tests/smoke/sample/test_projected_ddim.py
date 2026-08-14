from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from data_loaders.realtime_pose_kinematics import rotation_6d_to_matrix_torch
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


def test_so3_projection_is_idempotent_without_tracker_overwrite():
    torch.manual_seed(1)
    raw = torch.randn(2, 11, 144)
    deployed = project_realtime_pose_xstart(raw)
    repeated = project_realtime_pose_xstart(deployed)
    torch.testing.assert_close(deployed, repeated, atol=1e-5, rtol=1e-5)
    rotations = rotation_6d_to_matrix_torch(deployed.reshape(2, 11, 24, 6))
    identity = torch.eye(3).expand_as(rotations)
    torch.testing.assert_close(rotations.transpose(-1, -2) @ rotations, identity, atol=1e-5, rtol=1e-5)
    assert torch.all(torch.linalg.det(rotations) > 0.9999)


def test_projection_only_legalizes_each_rotation6d():
    raw = project_rotation_6d_to_so3(torch.randn(1, 11, 24, 6)).reshape(1, 11, 144)
    deployed = project_realtime_pose_xstart(raw)
    torch.testing.assert_close(deployed, raw, atol=1e-5, rtol=1e-5)


def test_projection_rejects_legacy_single_frame_state():
    with torch.no_grad(), np.testing.assert_raises(ValueError):
        project_realtime_pose_xstart(
            torch.zeros(1, 144),
        )


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
            "contact_logits": torch.ones(x.shape[0], 2, device=x.device),
        }


def test_projected_ddim_keeps_soft_prediction_until_final_projection():
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
    )
    alpha = torch.tensor(diffusion.alphas_cumprod[1], dtype=x.dtype)
    alpha_prev = torch.tensor(diffusion.alphas_cumprod_prev[1], dtype=x.dtype)
    epsilon = (x - torch.sqrt(alpha) * raw_xstart) / torch.sqrt(1.0 - alpha)
    expected = torch.sqrt(alpha_prev) * raw_xstart + torch.sqrt(1.0 - alpha_prev) * epsilon
    torch.testing.assert_close(result["sample"], expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(result["raw_pred_xstart"], raw_xstart)
    torch.testing.assert_close(result["deployed_pred_xstart"], raw_xstart)

    final = diffusion.projected_ddim_sample(
        _ConstantXStart(raw_xstart),
        x,
        torch.tensor([0]),
        projection_fn=lambda _: deployed_xstart,
        eta=0.0,
    )
    torch.testing.assert_close(final["sample"], deployed_xstart)


def test_projected_ddim_returns_final_auxiliary_heads():
    diffusion = GaussianDiffusion(
        betas=np.asarray([0.1, 0.2], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    result = diffusion.projected_ddim_sample_loop(
        _AuxiliaryXStart(torch.zeros(1, 11, 144)),
        shape=(1, 11, 144),
        projection_fn=lambda value: value,
        device=torch.device("cpu"),
    )
    assert result["sample"].shape == (1, 11, 144)
    assert result["raw_pred_xstart"].shape == (1, 11, 144)
    assert result["deployed_pred_xstart"].shape == (1, 11, 144)
    assert set(result["auxiliary_outputs"]) == {"contact_logits"}
    assert result["auxiliary_outputs"]["contact_logits"].shape == (1, 2)


def test_projected_ddim_only_projects_once_at_final_step():
    diffusion = GaussianDiffusion(
        betas=np.asarray([0.1, 0.15, 0.2, 0.25], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    model = _ConstantXStart(torch.zeros(1, 4))
    projection_calls = []

    def projection_fn(value):
        projection_calls.append(1)
        return value

    diffusion.projected_ddim_sample_loop(
        model,
        shape=(1, 4),
        projection_fn=projection_fn,
        device=torch.device("cpu"),
    )
    assert len(projection_calls) == 1
