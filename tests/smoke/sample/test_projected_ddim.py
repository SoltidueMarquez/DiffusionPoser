from __future__ import annotations

import numpy as np
import torch
from torch import nn

from data_loaders.realtime_pose_kinematics import rotation_6d_to_matrix_torch
from data_loaders.sensor_masking import TRACKER_TO_JOINT
from diffusion.gaussian_diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType
from diffusion.realtime_pose_projection import project_realtime_pose_xstart


IDENTITY_6D = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])


def _tracker(mask=(True, True, True, False, False, False)):
    tracker = torch.zeros(1, 6, 10)
    tracker[..., 3:9] = IDENTITY_6D
    tracker[..., 9] = torch.tensor(mask)
    return tracker


def test_single_frame_projection_legalizes_so3_and_writes_available_trackers():
    raw = torch.randn(1, 144)
    tracker = _tracker()
    deployed = project_realtime_pose_xstart(raw, tracker).reshape(1, 24, 6)
    rotations = rotation_6d_to_matrix_torch(deployed)
    identity = torch.eye(3).expand_as(rotations)
    torch.testing.assert_close(rotations.transpose(-1, -2) @ rotations, identity, atol=1e-5, rtol=1e-5)
    for tracker_index in range(3):
        torch.testing.assert_close(
            deployed[0, TRACKER_TO_JOINT[tracker_index]], IDENTITY_6D, atol=1e-5, rtol=1e-5
        )


class _Constant(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.register_buffer("value", value)

    def forward(self, x, timestep, **kwargs):
        del timestep, kwargs
        return self.value.expand_as(x)


def test_projected_ddim_is_single_frame_and_projects_only_final_step():
    diffusion = GaussianDiffusion(
        betas=np.asarray([0.1, 0.2], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    calls = []
    result = diffusion.projected_ddim_sample_loop(
        _Constant(torch.zeros(1, 144)),
        shape=(1, 144),
        projection_fn=lambda value: calls.append(1) or value,
        predictor_current=IDENTITY_6D.repeat(24)[None],
        device=torch.device("cpu"),
    )
    assert result["sample"].shape == (1, 144)
    torch.testing.assert_close(result["sample"], torch.zeros(1, 144))
    torch.testing.assert_close(
        result["raw_pred_pose"], IDENTITY_6D.repeat(24)[None]
    )
    assert len(calls) == 1
