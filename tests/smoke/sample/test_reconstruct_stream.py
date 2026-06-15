from __future__ import annotations

import numpy as np
import torch

from data_loaders.sensor_masking import (
    HIP_TRACKER_INDEX,
    REALTIME_POSE_INPUT_DIM,
    REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_DIM,
    REALTIME_POSE_TARGET_START,
    SMPL_JOINT_COUNT,
    get_schema_spec,
)
from sample.ik_initializer import build_tracker_pose_init_image
from sample.reconstruct_stream import reconstruct_batch, save_reconstruction
from sample.simulate_unity_stream import IDENTITY_6D


class RecordingDiffusion:
    def __init__(self):
        self.noise = "unset"

    def p_sample_loop(self, model, shape, noise, clip_denoised, model_kwargs):
        del model, clip_denoised, model_kwargs
        self.noise = noise
        return torch.ones(shape)


class RecordingWarmStartDiffusion:
    def __init__(self):
        self.num_timesteps = 10
        self.init_image = None
        self.skip_timesteps = None

    def p_sample_loop(self, model, shape, noise, clip_denoised, model_kwargs, init_image=None, skip_timesteps=0):
        del model, noise, clip_denoised, model_kwargs
        self.init_image = init_image
        self.skip_timesteps = skip_timesteps
        return torch.ones(shape)


def test_reconstruct_batch_starts_from_sampler_noise_and_keeps_conditions():
    conditioned = torch.full((1, REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SEQ_LEN), 2.0)
    batch = {
        "conditioned_x": conditioned,
        "valid_frame_mask": torch.ones(1, REALTIME_POSE_SEQ_LEN, dtype=torch.bool),
    }
    diffusion = RecordingDiffusion()

    reconstructed = reconstruct_batch(
        model=object(),
        diffusion=diffusion,
        batch=batch,
        device=torch.device("cpu"),
        use_ddim=False,
    )

    assert diffusion.noise is None
    assert torch.all(reconstructed[:, :REALTIME_POSE_TARGET_DIM, REALTIME_POSE_TARGET_START] == 1.0)
    condition_mask = torch.ones_like(conditioned, dtype=torch.bool)
    condition_mask[:, :REALTIME_POSE_TARGET_DIM, REALTIME_POSE_TARGET_START] = False
    assert torch.all(reconstructed[condition_mask] == conditioned[condition_mask])


def test_reconstruct_batch_passes_ik_init_image_and_start_timestep():
    conditioned = torch.zeros((1, REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SEQ_LEN), dtype=torch.float32)
    init_image = torch.full_like(conditioned, 0.5)
    batch = {
        "conditioned_x": conditioned,
        "valid_frame_mask": torch.ones(1, REALTIME_POSE_SEQ_LEN, dtype=torch.bool),
    }
    diffusion = RecordingWarmStartDiffusion()

    reconstruct_batch(
        model=object(),
        diffusion=diffusion,
        batch=batch,
        device=torch.device("cpu"),
        use_ddim=False,
        init_image=init_image,
        start_timestep=3,
    )

    assert diffusion.init_image is not None
    assert diffusion.skip_timesteps == 6
    torch.testing.assert_close(diffusion.init_image, init_image)


def test_tracker_pose_init_image_only_changes_target_frame_target_channels():
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    conditioned = torch.zeros((1, schema.feature_dim, REALTIME_POSE_SEQ_LEN), dtype=torch.float32)
    conditioned[:, schema.body_pose_slice(), REALTIME_POSE_TARGET_START - 1] = torch.from_numpy(
        np.tile(IDENTITY_6D, SMPL_JOINT_COUNT)
    )
    conditioned[:, schema.root_yaw_delta_slice(), REALTIME_POSE_TARGET_START - 1] = torch.tensor([0.0, 1.0])
    conditioned[:, schema.stationary_prob_slice(), REALTIME_POSE_TARGET_START - 1] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0])
    conditioned[:, schema.tracker_pos_slice(HIP_TRACKER_INDEX), REALTIME_POSE_TARGET_START] = torch.tensor([0.0, 0.9, 0.0])
    conditioned[:, schema.tracker_rot_slice(HIP_TRACKER_INDEX), REALTIME_POSE_TARGET_START] = torch.from_numpy(IDENTITY_6D)
    conditioned[:, schema.sensor_valid_slice(), REALTIME_POSE_TARGET_START] = 1.0
    before = conditioned.clone()

    init_image = build_tracker_pose_init_image(
        conditioned_x=conditioned,
        schema_name=schema.name,
        joint_offsets_parent=torch.zeros(1, SMPL_JOINT_COUNT, 3),
        iterations=0,
    )

    changed_mask = init_image != before
    allowed_mask = torch.zeros_like(conditioned, dtype=torch.bool)
    allowed_mask[:, schema.target_slice(), REALTIME_POSE_TARGET_START] = True
    assert not changed_mask[~allowed_mask].any()
    assert torch.allclose(
        init_image[:, schema.body_pose_slice(), REALTIME_POSE_TARGET_START],
        conditioned[:, schema.body_pose_slice(), REALTIME_POSE_TARGET_START - 1],
    )
    assert torch.allclose(init_image[:, schema.root_height_slice(), REALTIME_POSE_TARGET_START], torch.tensor([[0.9]]))


def test_save_reconstruction_writes_raw_and_normalized_features(tmp_path):
    class OffsetNormalizer:
        def inverse(self, features):
            return features + 10.0

    reference = torch.zeros((1, REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SEQ_LEN), dtype=torch.float32)
    reconstructed = reference.clone()
    reconstructed[:, 0, REALTIME_POSE_TARGET_START] = 1.0
    inpaint_mask = torch.zeros_like(reference, dtype=torch.bool)
    inpaint_mask[:, :REALTIME_POSE_TARGET_DIM, REALTIME_POSE_TARGET_START] = True
    path = tmp_path / "result.npz"

    save_reconstruction(
        path=path,
        reference=reference,
        conditioned=reference,
        reconstructed=reconstructed,
        inpaint_mask=inpaint_mask,
        normalizer=OffsetNormalizer(),
    )

    with np.load(path, allow_pickle=False) as data:
        assert "reference_features_raw" in data.files
        assert "reference_features_normalized" in data.files
        assert data["input_feature_space"].item() == "normalized"
        np.testing.assert_allclose(data["reference_features_normalized"], 0.0)
        np.testing.assert_allclose(data["reference_features_raw"], 10.0)
        np.testing.assert_allclose(data["reference_features"], data["reference_features_raw"])


def test_save_reconstruction_without_normalizer_does_not_write_normalized_fields(tmp_path):
    reference = torch.zeros((1, REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SEQ_LEN), dtype=torch.float32)
    inpaint_mask = torch.zeros_like(reference, dtype=torch.bool)
    path = tmp_path / "raw_result.npz"

    save_reconstruction(
        path=path,
        reference=reference,
        conditioned=reference,
        reconstructed=reference,
        inpaint_mask=inpaint_mask,
        normalizer=None,
    )

    with np.load(path, allow_pickle=False) as data:
        assert data["input_feature_space"].item() == "raw"
        assert "reference_features_raw" in data.files
        assert "reference_features_normalized" not in data.files
