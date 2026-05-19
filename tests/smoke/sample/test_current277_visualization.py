from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_loaders.sensor_masking import (
    BODY_VEL_DIM,
    BODY_VEL_START,
    MODEL_INPUT_DIM,
    SENSOR_LABEL_DIM,
    TRACKER_POS_DIM,
    TRACKER_POS_START,
    X277_FEATURE_DIM,
)
from sample.visualization import (
    FULL_RECONSTRUCTION_VISUALIZATION_NOTE,
    HAS_VISUALIZATION_BACKEND,
    SMPL_JOINT_COUNT,
    decode_x277_joint_positions_from_body_velocity,
    decode_x277_tracker_positions,
    render_full_reconstruction_visualization,
    restore_missing_tracker_positions_for_visualization,
)


class Current277FullVisualizationTest(unittest.TestCase):
    def test_decode_body_velocity_joint_positions_shape(self):
        motion = build_toy_current277_motion(frame_count=5)

        decoded = decode_x277_joint_positions_from_body_velocity(motion, target_fps=60.0)

        self.assertEqual(decoded.shape, (5, SMPL_JOINT_COUNT, 3))
        self.assertTrue(np.isfinite(decoded).all())

    def test_missing_tracker_visualization_uses_reference_position(self):
        reference_motion = build_toy_current277_motion(frame_count=5)
        conditioned_motion = reference_motion.copy()
        missing_sensor_indices = (1, 4)
        for sensor_index in missing_sensor_indices:
            pos_start = TRACKER_POS_START + sensor_index * TRACKER_POS_DIM
            conditioned_motion[2, pos_start : pos_start + TRACKER_POS_DIM] = 0.0

        sensor_missing_labels = np.zeros((5, SENSOR_LABEL_DIM), dtype=bool)
        sensor_missing_labels[2, list(missing_sensor_indices)] = True

        reference_trackers = decode_x277_tracker_positions(reference_motion)
        conditioned_trackers = decode_x277_tracker_positions(conditioned_motion)
        restored_trackers = restore_missing_tracker_positions_for_visualization(
            conditioned_trackers=conditioned_trackers,
            reference_trackers=reference_trackers,
            sensor_missing_labels=sensor_missing_labels,
        )

        self.assertFalse(np.allclose(conditioned_trackers[2, 1], reference_trackers[2, 1]))
        self.assertFalse(np.allclose(conditioned_trackers[2, 4], reference_trackers[2, 4]))
        np.testing.assert_allclose(restored_trackers[2, 1], reference_trackers[2, 1], atol=1e-6)
        np.testing.assert_allclose(restored_trackers[2, 4], reference_trackers[2, 4], atol=1e-6)
        np.testing.assert_allclose(restored_trackers[2, 0], conditioned_trackers[2, 0], atol=1e-6)

    @unittest.skipUnless(HAS_VISUALIZATION_BACKEND, "缺少可视化后端，跳过完整补全 mp4 smoke test。")
    def test_render_full_reconstruction_visualization_writes_mp4(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_path = root / "reconstruction.mp4"

            reference_motion = build_toy_current277_motion(frame_count=5)
            inpaint_mask = np.zeros((5, MODEL_INPUT_DIM), dtype=bool)
            inpaint_mask[1:4, :X277_FEATURE_DIM] = True

            conditioned_motion = reference_motion.copy()
            conditioned_motion[inpaint_mask[:, :X277_FEATURE_DIM]] = 0.0
            reconstructed_motion = reference_motion.copy()
            reconstructed_motion[1:4, BODY_VEL_START : BODY_VEL_START + BODY_VEL_DIM] *= 0.85

            sensor_missing_labels = np.zeros((5, SENSOR_LABEL_DIM), dtype=bool)
            sensor_missing_labels[2, 1] = True
            sensor_missing_labels[2, 4] = True

            meta = render_full_reconstruction_visualization(
                reference_motion=reference_motion,
                conditioned_motion=conditioned_motion,
                reconstructed_motion=reconstructed_motion,
                sensor_missing_labels=sensor_missing_labels,
                inpaint_mask=inpaint_mask,
                output_path=output_path,
                fps=5.0,
                title="toy full reconstruction",
                valid_length=5,
                x277_fps=60.0,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(meta["visualization"], "current277_full_reconstruction")
            self.assertEqual(meta["frame_count"], 5)
            self.assertEqual(meta["note"], FULL_RECONSTRUCTION_VISUALIZATION_NOTE)
            self.assertTrue(meta["frames"][2]["is_reconstruction_target"])


def build_toy_current277_motion(frame_count: int) -> np.ndarray:
    """构造一个极小 current277 序列，用来验证完整补全可视化链路。"""

    motion = np.zeros((frame_count, X277_FEATURE_DIM), dtype=np.float32)
    for frame_index in range(frame_count):
        base_x = frame_index * 0.03
        tracker_points = np.asarray(
            [
                [base_x, 1.72, 0.05],
                [base_x - 0.45, 1.18, 0.02],
                [base_x + 0.45, 1.18, 0.02],
                [base_x, 0.95, 0.00],
                [base_x - 0.16, 0.05, 0.12],
                [base_x + 0.16, 0.05, 0.12],
            ],
            dtype=np.float32,
        )
        motion[
            frame_index,
            TRACKER_POS_START : TRACKER_POS_START + SENSOR_LABEL_DIM * TRACKER_POS_DIM,
        ] = tracker_points.reshape(-1)
        body_velocity = np.zeros((SMPL_JOINT_COUNT, 3), dtype=np.float32)
        body_velocity[:, 0] = 0.03 * 60.0
        motion[frame_index, BODY_VEL_START : BODY_VEL_START + BODY_VEL_DIM] = body_velocity.reshape(-1)
        motion[frame_index, 270:272] = np.asarray([0.03, 0.0], dtype=np.float32)
        motion[frame_index, 272] = 0.0
        motion[frame_index, 273:277] = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    return motion


if __name__ == "__main__":
    unittest.main()
