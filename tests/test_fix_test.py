from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_loaders.sensor_masking import (
    MODEL_INPUT_DIM,
    SENSOR_LABEL_DIM,
    SENSOR_NAMES,
    X277_FEATURE_DIM,
    build_inpaint_mask_from_sensor_missing_labels,
    sensor_feature_slices,
)
from data_loaders.manifest_utils import filter_entries_by_folder_path
from sample.visualization import (
    HAS_VISUALIZATION_BACKEND,
    SMPL_JOINT_COUNT,
    VISUALIZATION_LIMITATION_NOTE,
    decode_x277_joint_positions,
    decode_x277_tracker_positions,
    render_fix_visualization,
)


class FixMaskBuildTest(unittest.TestCase):
    def test_build_inpaint_mask_from_sensor_missing_labels(self):
        sensor_missing_labels = np.zeros((2, 5, 6), dtype=bool)
        sensor_missing_labels[0, 1:4, 1] = True
        sensor_missing_labels[0, 2:5, 4] = True
        sensor_missing_labels[1, :2, 3] = True

        valid_frame_mask = np.array(
            [
                [True, True, True, True, True],
                [True, True, False, False, False],
            ],
            dtype=bool,
        )

        inpaint_mask = build_inpaint_mask_from_sensor_missing_labels(sensor_missing_labels, valid_frame_mask)
        self.assertEqual(inpaint_mask.shape, (2, 5, MODEL_INPUT_DIM))
        self.assertFalse(inpaint_mask[:, :, X277_FEATURE_DIM:MODEL_INPUT_DIM].any())

        expected = np.zeros_like(inpaint_mask)
        for batch_index in range(2):
            for frame_index in range(5):
                if not valid_frame_mask[batch_index, frame_index]:
                    continue
                for sensor_index in np.flatnonzero(sensor_missing_labels[batch_index, frame_index]):
                    pos_slice, rot_slice = sensor_feature_slices(int(sensor_index))
                    expected[batch_index, frame_index, pos_slice] = True
                    expected[batch_index, frame_index, rot_slice] = True

        np.testing.assert_array_equal(inpaint_mask, expected)


class X277DatasetFolderFilterTest(unittest.TestCase):
    def test_folder_path_filters_dataset_entries(self):
        entries = [
            {
                "task_id": "keep_me",
                "source_relative_path": "target/subfolder/keep_me.npz",
                "source_path": str(Path("/tmp/source/target/subfolder/keep_me.npz")),
            },
            {
                "task_id": "drop_me",
                "source_relative_path": "other/place/drop_me.npz",
                "source_path": str(Path("/tmp/source/other/place/drop_me.npz")),
            },
        ]

        filtered = filter_entries_by_folder_path(entries, folder_path="target/subfolder")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["task_id"], "keep_me")


class X277SkeletonDecodeTest(unittest.TestCase):
    def test_decode_x277_joint_positions_shape_and_tracker_points(self):
        features = np.zeros((3, X277_FEATURE_DIM), dtype=np.float32)
        head_tracker_root = np.array([0.0, 1.8, 0.1], dtype=np.float32)
        features[:, 216:219] = head_tracker_root

        decoded = decode_x277_joint_positions(features, target_fps=60.0)
        trackers = decode_x277_tracker_positions(features)

        self.assertEqual(decoded.shape, (3, SMPL_JOINT_COUNT, 3))
        self.assertEqual(trackers.shape, (3, SENSOR_LABEL_DIM, 3))
        self.assertTrue(np.isfinite(decoded).all())
        self.assertTrue(np.isfinite(trackers).all())
        np.testing.assert_allclose(trackers[0, 0], head_tracker_root, atol=1e-6)


class FixVisualizationSmokeTest(unittest.TestCase):
    @unittest.skipUnless(HAS_VISUALIZATION_BACKEND, "缺少可视化后端，跳过 mp4 smoke test。")
    def test_render_fix_visualization_writes_mp4(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_path = root / "repair.mp4"

            reference_motion = np.zeros((4, X277_FEATURE_DIM), dtype=np.float32)
            corrupted_motion = np.zeros_like(reference_motion)
            repaired_motion = np.zeros_like(reference_motion)
            sensor_missing_labels = np.zeros((4, SENSOR_LABEL_DIM), dtype=bool)

            for frame_index in range(4):
                base = np.array([frame_index * 0.05, 0.0, 0.0], dtype=np.float32)
                for tracker_index in range(6):
                    offset = np.array([0.1 * tracker_index, 0.03 * tracker_index, 0.02 * tracker_index], dtype=np.float32)
                    reference_motion[frame_index, 216 + tracker_index * 3 : 216 + (tracker_index + 1) * 3] = base + offset
                    repaired_motion[frame_index, 216 + tracker_index * 3 : 216 + (tracker_index + 1) * 3] = base + offset
                    corrupted_motion[frame_index, 216 + tracker_index * 3 : 216 + (tracker_index + 1) * 3] = base + offset

            sensor_missing_labels[1, 1] = True
            sensor_missing_labels[1, 4] = True
            sensor_missing_labels[2, 3] = True

            corrupted_motion[1, 216 + 1 * 3 : 216 + 2 * 3] = 0.0
            corrupted_motion[1, 216 + 4 * 3 : 216 + 5 * 3] = 0.0
            corrupted_motion[2, 216 + 3 * 3 : 216 + 4 * 3] = 0.0
            repaired_motion[1, 216 + 1 * 3 : 216 + 2 * 3] = np.array([0.11, 0.02, 0.03], dtype=np.float32)
            repaired_motion[1, 216 + 4 * 3 : 216 + 5 * 3] = np.array([0.4, 0.12, 0.08], dtype=np.float32)
            repaired_motion[2, 216 + 3 * 3 : 216 + 4 * 3] = np.array([0.2, 0.0, 0.0], dtype=np.float32)

            meta = render_fix_visualization(
                reference_motion=reference_motion,
                corrupted_motion=corrupted_motion,
                repaired_motion=repaired_motion,
                sensor_missing_labels=sensor_missing_labels,
                output_path=output_path,
                fps=5.0,
                title="toy sample",
                valid_length=4,
                x277_fps=60.0,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(meta["frame_count"], 4)
            self.assertEqual(len(meta["frames"]), 4)
            self.assertEqual(meta["note"], VISUALIZATION_LIMITATION_NOTE)
            self.assertListEqual(meta["frames"][1]["missing_sensor_names"], [SENSOR_NAMES[1], SENSOR_NAMES[4]])


if __name__ == "__main__":
    unittest.main()
