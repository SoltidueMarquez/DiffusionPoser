from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from export.write_unity_runtime_assets import X277_FEATURE_DIM, X277_MODEL_INPUT_DIM, build_normalizer


class RuntimeAssetsTest(unittest.TestCase):
    def test_x277_normalizer_maps_runtime_sensor_labels_to_training_scale(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            normalizer_dir = Path(tmp_dir)
            torch.save(torch.zeros(X277_FEATURE_DIM), normalizer_dir / "mean.pt")
            torch.save(torch.ones(X277_FEATURE_DIM), normalizer_dir / "std.pt")

            payload = build_normalizer(
                feature_dim=X277_MODEL_INPUT_DIM,
                normalizer_dir=normalizer_dir,
                normalize_input=True,
                strict=True,
            )

        label_mean = np.asarray(payload["mean"][X277_FEATURE_DIM:X277_MODEL_INPUT_DIM], dtype=np.float32)
        label_std = np.asarray(payload["std"][X277_FEATURE_DIM:X277_MODEL_INPUT_DIM], dtype=np.float32)
        np.testing.assert_allclose(label_mean, np.full(6, 0.5, dtype=np.float32))
        np.testing.assert_allclose(label_std, np.full(6, 0.5, dtype=np.float32))

        runtime_labels = np.asarray([0.0, 1.0], dtype=np.float32)
        normalized = (runtime_labels - label_mean[:2]) / label_std[:2]
        np.testing.assert_allclose(normalized, np.asarray([-1.0, 1.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
