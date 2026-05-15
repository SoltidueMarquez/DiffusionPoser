import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loaders.compute_x277_normalizer import main as compute_x277_normalizer_main
from data_loaders.generate_x277_missing_tasks import main as generate_missing_tasks_main
from data_loaders.sensor_masking import (
    MODEL_INPUT_DIM,
    SENSOR_LABEL_DIM,
    X277_FEATURE_DIM,
    apply_sensor_missing_interval,
    sensor_feature_slices,
    tracker_feature_mask,
)
from data_loaders.x277_dataset import X277MissingTaskDataset
from utils.normalizer import X277Normalizer


class SensorMaskingTest(unittest.TestCase):
    def test_sensor_index_only_targets_tracker_position_and_rotation(self):
        seq_len = 8
        for sensor_index in range(SENSOR_LABEL_DIM):
            sensor_missing_labels = np.zeros((seq_len, SENSOR_LABEL_DIM), dtype=bool)
            inpaint_mask = np.zeros((seq_len, MODEL_INPUT_DIM), dtype=bool)

            apply_sensor_missing_interval(
                sensor_missing_labels=sensor_missing_labels,
                inpaint_mask=inpaint_mask,
                start=2,
                length=3,
                sensor_indices=[sensor_index],
            )

            expected = np.zeros_like(inpaint_mask)
            pos_slice, rot_slice = sensor_feature_slices(sensor_index)
            expected[2:5, pos_slice] = True
            expected[2:5, rot_slice] = True

            np.testing.assert_array_equal(inpaint_mask, expected)
            self.assertTrue(sensor_missing_labels[2:5, sensor_index].all())
            self.assertFalse(sensor_missing_labels[:2, sensor_index].any())
            self.assertFalse(sensor_missing_labels[5:, sensor_index].any())


class X277MissingTaskDatasetTest(unittest.TestCase):
    def test_generator_and_dataset_create_fixed_length_training_batch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = tmp_path / "AMASS_x277_60hz"
            source_file = source_dir / "CMU" / "142" / "sample_poses.npz"
            source_file.parent.mkdir(parents=True)

            x277 = np.arange(35 * X277_FEATURE_DIM, dtype=np.float32).reshape(35, X277_FEATURE_DIM)
            np.savez(source_file, x=x277)

            manifest_entry = {
                "status": "converted",
                "feature_dim": X277_FEATURE_DIM,
                "frames": 35,
                "output_path": str(source_file),
                "source_relative_path": "CMU/142/sample_poses.npz",
                "stablemotion_split_key": "CMU/142/sample_poses.npy",
                "is_mirrored": False,
            }
            with (source_dir / "manifest.jsonl").open("w", encoding="utf-8") as file:
                file.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")

            split_dir = tmp_path / "splits"
            split_dir.mkdir()
            (split_dir / "train.txt").write_text("CMU/142/sample_poses.npy\n", encoding="utf-8")

            output_dir = tmp_path / "missing_tasks"
            generate_missing_tasks_main(
                [
                    "--source_dir",
                    str(source_dir),
                    "--output_dir",
                    str(output_dir),
                    "--split_dir",
                    str(split_dir),
                    "--splits",
                    "train",
                    "--seq_len",
                    "100",
                    "--samples_per_file",
                    "2",
                    "--seed",
                    "123",
                ]
            )

            manifest_path = output_dir / "train" / "manifest.jsonl"
            manifest_lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(manifest_lines), 2)

            dataset = X277MissingTaskDataset(data_dir=output_dir, split="train", seq_len=100, normalize_input=False)
            self.assertEqual(len(dataset), 2)

            item = dataset[0]
            self.assertEqual(tuple(item["x"].shape), (MODEL_INPUT_DIM, 100))
            self.assertEqual(tuple(item["valid_frame_mask"].shape), (100,))
            self.assertEqual(tuple(item["attention_mask"].shape), (100,))
            self.assertEqual(tuple(item["sensor_missing_labels"].shape), (SENSOR_LABEL_DIM, 100))
            self.assertEqual(tuple(item["inpaint_mask"].shape), (MODEL_INPUT_DIM, 100))
            self.assertEqual(int(item["valid_frame_mask"].sum().item()), 35)

            self.assertTrue(torch.all(item["x"][:, 35:] == 0))
            self.assertFalse(item["valid_frame_mask"][35:].any())
            self.assertFalse(item["inpaint_mask"][:, 35:].any())
            self.assertFalse(item["inpaint_mask"][X277_FEATURE_DIM:MODEL_INPUT_DIM].any())

            labels = item["sensor_missing_labels"].numpy()
            np.testing.assert_array_equal(item["x"][X277_FEATURE_DIM:MODEL_INPUT_DIM].numpy().astype(bool), labels)

            missing_sensor_count = int(labels.any(axis=1).sum())
            self.assertGreaterEqual(missing_sensor_count, 1)
            self.assertLessEqual(missing_sensor_count, 4)
            self.assertGreaterEqual(SENSOR_LABEL_DIM - missing_sensor_count, 2)

            mask = item["inpaint_mask"].numpy()
            allowed_features = tracker_feature_mask(range(SENSOR_LABEL_DIM))
            self.assertFalse(mask[~allowed_features].any())

            loader = DataLoader(dataset, batch_size=2, shuffle=False)
            batch = next(iter(loader))
            self.assertEqual(tuple(batch["x"].shape), (2, MODEL_INPUT_DIM, 100))
            self.assertEqual(tuple(batch["valid_frame_mask"].shape), (2, 100))
            self.assertEqual(tuple(batch["inpaint_mask"].shape), (2, MODEL_INPUT_DIM, 100))

            normalizer_dir = tmp_path / "meta"
            mean = np.arange(X277_FEATURE_DIM, dtype=np.float32)
            std = np.full(X277_FEATURE_DIM, 2.0, dtype=np.float32)
            X277Normalizer(base_dir=normalizer_dir, disable=True).save(mean=mean, std=std)

            normalized_dataset = X277MissingTaskDataset(
                data_dir=output_dir,
                split="train",
                seq_len=100,
                normalizer_dir=normalizer_dir,
                normalize_input=True,
            )
            normalized_item = normalized_dataset[0]
            expected_x277 = (x277[:35] - mean) / std
            np.testing.assert_allclose(
                normalized_item["x"][:X277_FEATURE_DIM, :35].T.numpy(),
                expected_x277,
                rtol=1e-6,
                atol=1e-6,
            )
            self.assertTrue(torch.all(normalized_item["x"][:, 35:] == 0))

            normalized_labels = normalized_item["sensor_missing_labels"].numpy()
            expected_label_channels = np.where(normalized_labels[:, :35], 1.0, -1.0)
            np.testing.assert_array_equal(
                normalized_item["x"][X277_FEATURE_DIM:MODEL_INPUT_DIM, :35].numpy(),
                expected_label_channels,
            )
            self.assertTrue(torch.all(normalized_item["x"][X277_FEATURE_DIM:MODEL_INPUT_DIM, 35:] == 0))
            np.testing.assert_array_equal(normalized_item["inpaint_mask"].numpy(), item["inpaint_mask"].numpy())


class X277NormalizerComputationTest(unittest.TestCase):
    def test_compute_normalizer_uses_train_split_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = tmp_path / "AMASS_x277_60hz"
            train_file = source_dir / "CMU" / "001" / "train_poses.npz"
            test_file = source_dir / "CMU" / "001" / "test_poses.npz"
            train_file.parent.mkdir(parents=True)

            train_x = np.stack(
                [
                    np.arange(X277_FEATURE_DIM, dtype=np.float32),
                    np.arange(X277_FEATURE_DIM, dtype=np.float32) + 2.0,
                ],
                axis=0,
            )
            test_x = np.full((2, X277_FEATURE_DIM), 1000.0, dtype=np.float32)
            np.savez(train_file, x=train_x)
            np.savez(test_file, x=test_x)

            entries = [
                {
                    "status": "converted",
                    "feature_dim": X277_FEATURE_DIM,
                    "frames": int(train_x.shape[0]),
                    "output_path": str(train_file),
                    "source_relative_path": "CMU/001/train_poses.npz",
                    "stablemotion_split_key": "CMU/001/train_poses.npy",
                    "is_mirrored": False,
                },
                {
                    "status": "converted",
                    "feature_dim": X277_FEATURE_DIM,
                    "frames": int(test_x.shape[0]),
                    "output_path": str(test_file),
                    "source_relative_path": "CMU/001/test_poses.npz",
                    "stablemotion_split_key": "CMU/001/test_poses.npy",
                    "is_mirrored": False,
                },
            ]
            with (source_dir / "manifest.jsonl").open("w", encoding="utf-8") as file:
                for entry in entries:
                    file.write(json.dumps(entry, ensure_ascii=False) + "\n")

            split_dir = tmp_path / "splits"
            split_dir.mkdir()
            (split_dir / "train.txt").write_text("CMU/001/train_poses.npy\n", encoding="utf-8")
            (split_dir / "test.txt").write_text("CMU/001/test_poses.npy\n", encoding="utf-8")

            output_dir = tmp_path / "meta"
            meta = compute_x277_normalizer_main(
                [
                    "--source_dir",
                    str(source_dir),
                    "--output_dir",
                    str(output_dir),
                    "--split_dir",
                    str(split_dir),
                    "--split",
                    "train",
                    "--overwrite",
                ]
            )

            mean = torch.load(output_dir / "mean.pt", map_location="cpu", weights_only=True).numpy()
            std = torch.load(output_dir / "std.pt", map_location="cpu", weights_only=True).numpy()
            np.testing.assert_allclose(mean, train_x.mean(axis=0), rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(std, train_x.std(axis=0), rtol=1e-6, atol=1e-6)
            self.assertEqual(meta["matched_source_files"], 1)
            self.assertEqual(meta["total_frames"], int(train_x.shape[0]))
            self.assertTrue((output_dir / "normalizer_meta.json").exists())


if __name__ == "__main__":
    unittest.main()
