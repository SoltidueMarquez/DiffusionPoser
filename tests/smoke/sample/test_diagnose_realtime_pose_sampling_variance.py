from __future__ import annotations

import numpy as np
import pytest
import torch

from sample.diagnose_realtime_pose_sampling_variance import (
    save_repeat_sampling_artifact,
)


def test_repeat_sampling_artifact_preserves_both_noise_streams(tmp_path):
    repeat_noise = torch.arange(2 * 11 * 144, dtype=torch.float32).reshape(2, 11, 144)
    repeat_known_noise = -repeat_noise
    output_path = tmp_path / "repeat_samples.npz"

    save_repeat_sampling_artifact(
        output_path,
        repeat_noise=repeat_noise,
        repeat_known_noise=repeat_known_noise,
        predicted_root_yaw_world=np.asarray([0.1, 0.2], dtype=np.float32),
    )

    with np.load(output_path, allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload["noise"], repeat_noise.numpy())
        np.testing.assert_array_equal(
            payload["known_noise"], repeat_known_noise.numpy()
        )
        np.testing.assert_array_equal(
            payload["predicted_root_yaw_world"],
            np.asarray([0.1, 0.2], dtype=np.float32),
        )


def test_repeat_sampling_artifact_rejects_mismatched_known_noise(tmp_path):
    with pytest.raises(ValueError, match="同形"):
        save_repeat_sampling_artifact(
            tmp_path / "repeat_samples.npz",
            repeat_noise=torch.zeros(2, 11, 144),
            repeat_known_noise=torch.zeros(1, 11, 144),
        )
