from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_loaders.sensor_masking import MODEL_INPUT_DIM, ROOT_YAW_START, X277_FEATURE_DIM
from eval.evaluate_current277 import evaluate_file


class Current277EvaluationTest(unittest.TestCase):
    def test_metrics_use_target_frames_not_context_frames(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "stream_outputs.npz"
            reference = np.zeros((3, X277_FEATURE_DIM), dtype=np.float32)
            reconstructed = np.zeros_like(reference)

            # 第 0 帧是上下文帧，即使误差很大也不应进入主重建指标。
            reconstructed[0, ROOT_YAW_START] = 1000.0
            reconstructed[1, ROOT_YAW_START] = 5.0

            inpaint_mask = np.zeros((3, MODEL_INPUT_DIM), dtype=bool)
            inpaint_mask[1, ROOT_YAW_START] = True
            sensor_missing_labels = np.zeros((3, 6), dtype=bool)
            valid_frame_mask = np.ones(3, dtype=bool)

            np.savez(
                path,
                reference_motion=reference,
                reconstructed_motion=reconstructed,
                sensor_missing_labels=sensor_missing_labels,
                inpaint_mask=inpaint_mask,
                valid_frame_mask=valid_frame_mask,
            )

            result = evaluate_file(path)

        self.assertEqual(result["target_frames"], 1)
        self.assertEqual(result["context_frames"], 2)
        normal_online = result["scenarios"]["normal_online"]
        self.assertEqual(normal_online["frames"], 1)
        self.assertEqual(normal_online["root_yaw_abs_degree"], 5.0)


if __name__ == "__main__":
    unittest.main()
