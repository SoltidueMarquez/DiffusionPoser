from __future__ import annotations

import numpy as np
import torch

from data_loaders.sensor_masking import REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SEQ_LEN, REALTIME_POSE_TARGET_DIM, REALTIME_POSE_TARGET_START
from sample.reconstruct_stream import reconstruct_batch, save_reconstruction


class RecordingDiffusion:
    def __init__(self):
        self.noise = "unset"

    def p_sample_loop(self, model, shape, noise, clip_denoised, model_kwargs):
        del model, clip_denoised, model_kwargs
        self.noise = noise
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
