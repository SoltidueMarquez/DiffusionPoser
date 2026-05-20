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
    CONTACT_START,
    MODEL_INPUT_DIM,
    ROOT_DELTA_START,
    ROOT_YAW_START,
    SENSOR_LABEL_DIM,
    TASK_MODE_FULL_RECONSTRUCTION_CURRENT,
    TASK_FORMAT_CURRENT277_BODY_RECONSTRUCTION_V2,
    TRACKER_POS_DIM,
    TRACKER_POS_START,
    TRACKER_ROT_DIM,
    TRACKER_ROT_START,
    X277_FEATURE_DIM,
    apply_sensor_missing_interval,
    create_full_reconstruction_task,
    sensor_feature_slices,
)
from data_loaders.x277_dataset import X277MissingTaskDataset
from utils.normalizer import X277Normalizer


class SensorMaskingTest(unittest.TestCase):
    def test_sensor_missing_interval_only_sets_missing_label(self):
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

            np.testing.assert_array_equal(inpaint_mask, np.zeros_like(inpaint_mask))
            self.assertTrue(sensor_missing_labels[2:5, sensor_index].all())
            self.assertFalse(sensor_missing_labels[:2, sensor_index].any())
            self.assertFalse(sensor_missing_labels[5:, sensor_index].any())

    def test_full_reconstruction_masks_body_root_contact_without_tracker_targets(self):
        rng = np.random.default_rng(7)
        sensor_missing_labels, inpaint_mask, intervals, target_start, target_length = create_full_reconstruction_task(
            seq_len=11,
            valid_length=11,
            rng=rng,
            num_intervals=1,
            all_sensor_dropout_prob=1.0,
        )

        self.assertEqual(target_start, 10)
        self.assertEqual(target_length, 1)
        self.assertEqual(len(intervals), 1)

        target = target_start
        self.assertTrue(inpaint_mask[target, 0:216].all())
        self.assertTrue(inpaint_mask[target, ROOT_DELTA_START : ROOT_DELTA_START + 2].all())
        self.assertTrue(inpaint_mask[target, ROOT_YAW_START : ROOT_YAW_START + 1].all())
        self.assertTrue(inpaint_mask[target, CONTACT_START : CONTACT_START + 4].all())
        self.assertFalse(inpaint_mask[:target_start].any())
        self.assertFalse(inpaint_mask[:, X277_FEATURE_DIM:MODEL_INPUT_DIM].any())

        self.assertTrue(sensor_missing_labels[target].all())
        self.assertFalse(sensor_missing_labels[:target_start].any())
        for sensor_index in range(SENSOR_LABEL_DIM):
            pos_slice, rot_slice = sensor_feature_slices(sensor_index)
            self.assertFalse(inpaint_mask[target, pos_slice].any())
            self.assertFalse(inpaint_mask[target, rot_slice].any())


class X277MissingTaskDatasetTest(unittest.TestCase):
    def test_generator_creates_full_reconstruction_current_task(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = tmp_path / "AMASS_current277_60hz"
            source_file = source_dir / "CMU" / "142" / "sample_poses.npz"
            source_file.parent.mkdir(parents=True)

            x277 = np.arange(20 * X277_FEATURE_DIM, dtype=np.float32).reshape(20, X277_FEATURE_DIM)
            np.savez(source_file, x=x277)
            with (source_dir / "manifest.jsonl").open("w", encoding="utf-8") as file:
                file.write(
                    json.dumps(
                        {
                            "status": "converted",
                            "feature_dim": X277_FEATURE_DIM,
                            "frames": 20,
                            "output_path": str(source_file),
                            "source_relative_path": "CMU/142/sample_poses.npz",
                            "stablemotion_split_key": "CMU/142/sample_poses.npy",
                            "is_mirrored": False,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            split_dir = tmp_path / "splits"
            split_dir.mkdir()
            (split_dir / "train.txt").write_text("CMU/142/sample_poses.npy\n", encoding="utf-8")
            output_dir = tmp_path / "full_tasks"
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
                    "--task_mode",
                    TASK_MODE_FULL_RECONSTRUCTION_CURRENT,
                    "--seq_len",
                    "11",
                    "--samples_per_file",
                    "1",
                    "--all_sensor_dropout_prob",
                    "1.0",
                    "--seed",
                    "123",
                ]
            )

            manifest_entry = json.loads((output_dir / "train" / "manifest.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(manifest_entry["task_mode"], TASK_MODE_FULL_RECONSTRUCTION_CURRENT)
            self.assertEqual(manifest_entry["schema_name"], "current277_v1")
            target_start = int(manifest_entry["target_start"])
            target_length = int(manifest_entry["target_length"])
            target = target_start
            self.assertEqual(target_start, 10)
            self.assertEqual(target_length, 1)
            self.assertEqual(manifest_entry["task_format"], TASK_FORMAT_CURRENT277_BODY_RECONSTRUCTION_V2)
            self.assertEqual(manifest_entry["tracker_target_mode"], "derived_from_body")
            self.assertEqual(manifest_entry["missing_tracker_condition"], "zero")

            with np.load(output_dir / "train" / manifest_entry["task_path"], allow_pickle=False) as task_data:
                inpaint_mask = task_data["inpaint_mask"].astype(bool)
                sensor_missing_labels = task_data["sensor_missing_labels"].astype(bool)
            self.assertTrue(inpaint_mask[target, 0:216].all())
            self.assertTrue(inpaint_mask[target, ROOT_DELTA_START : ROOT_DELTA_START + 2].all())
            self.assertTrue(inpaint_mask[target, CONTACT_START : CONTACT_START + 4].all())
            self.assertFalse(inpaint_mask[:target_start].any())
            self.assertFalse(inpaint_mask[:, X277_FEATURE_DIM:MODEL_INPUT_DIM].any())
            for sensor_index in range(SENSOR_LABEL_DIM):
                pos_slice, rot_slice = sensor_feature_slices(sensor_index)
                self.assertFalse(inpaint_mask[target, pos_slice].any())
                self.assertFalse(inpaint_mask[target, rot_slice].any())
            self.assertTrue(sensor_missing_labels[target].all())
            self.assertFalse(sensor_missing_labels[:target_start].any())

    def test_generator_and_dataset_create_fixed_length_training_batch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = tmp_path / "AMASS_current277_60hz"
            source_file = source_dir / "CMU" / "142" / "sample_poses.npz"
            source_file.parent.mkdir(parents=True)

            x277 = np.arange(11 * X277_FEATURE_DIM, dtype=np.float32).reshape(11, X277_FEATURE_DIM)
            np.savez(source_file, x=x277)

            manifest_entry = {
                "status": "converted",
                "feature_dim": X277_FEATURE_DIM,
                "frames": 11,
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
                    "11",
                    "--samples_per_file",
                    "2",
                    "--all_sensor_dropout_prob",
                    "0.0",
                    "--seed",
                    "123",
                ]
            )

            manifest_path = output_dir / "train" / "manifest.jsonl"
            manifest_lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(manifest_lines), 2)
            first_entry = json.loads(manifest_lines[0])
            first_task_path = output_dir / "train" / first_entry["task_path"]
            with np.load(first_task_path, allow_pickle=False) as task_data:
                self.assertIn("x277", task_data.files)
                self.assertEqual(tuple(task_data["x277"].shape), (11, X277_FEATURE_DIM))
                np.testing.assert_array_equal(task_data["x277"], x277)

            dataset = X277MissingTaskDataset(data_dir=output_dir, split="train", seq_len=11, normalize_input=False)
            self.assertEqual(len(dataset), 2)

            item = dataset[0]
            self.assertEqual(tuple(item["x"].shape), (MODEL_INPUT_DIM, 11))
            self.assertEqual(tuple(item["conditioned_x"].shape), (MODEL_INPUT_DIM, 11))
            self.assertEqual(tuple(item["valid_frame_mask"].shape), (11,))
            self.assertEqual(tuple(item["attention_mask"].shape), (11,))
            self.assertEqual(tuple(item["sensor_missing_labels"].shape), (SENSOR_LABEL_DIM, 11))
            self.assertEqual(tuple(item["inpaint_mask"].shape), (MODEL_INPUT_DIM, 11))
            self.assertEqual(int(item["valid_frame_mask"].sum().item()), 11)

            self.assertFalse(item["inpaint_mask"][X277_FEATURE_DIM:MODEL_INPUT_DIM].any())

            labels = item["sensor_missing_labels"].numpy()
            np.testing.assert_array_equal(item["x"][X277_FEATURE_DIM:MODEL_INPUT_DIM].numpy().astype(bool), labels)

            mask = item["inpaint_mask"].numpy()
            self.assertTrue(labels.any())
            target_frame = int(item["target_start"])
            self.assertEqual(target_frame, 10)
            self.assertEqual(int(item["target_length"]), 1)
            self.assertTrue(mask[:, :target_frame].sum() == 0)
            self.assertTrue(mask[:, target_frame + 1 :].sum() == 0)
            self.assertTrue(labels[:, :target_frame].sum() == 0)
            self.assertTrue(labels[:, target_frame + 1 :].sum() == 0)
            self.assertTrue(mask[:216].any())
            self.assertTrue(mask[ROOT_DELTA_START : ROOT_DELTA_START + 2].any())
            self.assertTrue(mask[ROOT_YAW_START : ROOT_YAW_START + 1].any())
            self.assertTrue(mask[CONTACT_START : CONTACT_START + 4].any())
            self.assertFalse(mask[TRACKER_POS_START : TRACKER_POS_START + SENSOR_LABEL_DIM * TRACKER_POS_DIM].any())
            self.assertFalse(mask[TRACKER_ROT_START : TRACKER_ROT_START + SENSOR_LABEL_DIM * TRACKER_ROT_DIM].any())

            conditioned_x = item["conditioned_x"].numpy()
            full_x = item["x"].numpy()
            for sensor_index in np.flatnonzero(labels[:, target_frame]):
                pos_slice, rot_slice = sensor_feature_slices(int(sensor_index))
                self.assertTrue(np.all(conditioned_x[pos_slice, target_frame] == 0.0))
                self.assertTrue(np.all(conditioned_x[rot_slice, target_frame] == 0.0))
                self.assertFalse(np.all(full_x[pos_slice, target_frame] == 0.0))
                self.assertFalse(np.all(full_x[rot_slice, target_frame] == 0.0))

            loader = DataLoader(dataset, batch_size=2, shuffle=False)
            batch = next(iter(loader))
            self.assertEqual(tuple(batch["x"].shape), (2, MODEL_INPUT_DIM, 11))
            self.assertEqual(tuple(batch["conditioned_x"].shape), (2, MODEL_INPUT_DIM, 11))
            self.assertEqual(tuple(batch["valid_frame_mask"].shape), (2, 11))
            self.assertEqual(tuple(batch["inpaint_mask"].shape), (2, MODEL_INPUT_DIM, 11))

            normalizer_dir = tmp_path / "meta"
            mean = np.arange(X277_FEATURE_DIM, dtype=np.float32)
            std = np.full(X277_FEATURE_DIM, 2.0, dtype=np.float32)
            X277Normalizer(base_dir=normalizer_dir, disable=True).save(mean=mean, std=std)

            normalized_dataset = X277MissingTaskDataset(
                data_dir=output_dir,
                split="train",
                seq_len=11,
                normalizer_dir=normalizer_dir,
                normalize_input=True,
            )
            normalized_item = normalized_dataset[0]
            expected_x277 = (x277 - mean) / std
            np.testing.assert_allclose(
                normalized_item["x"][:X277_FEATURE_DIM, :11].T.numpy(),
                expected_x277,
                rtol=1e-6,
                atol=1e-6,
            )

            normalized_labels = normalized_item["sensor_missing_labels"].numpy()
            expected_label_channels = np.where(normalized_labels[:, :11], 1.0, -1.0)
            np.testing.assert_array_equal(
                normalized_item["x"][X277_FEATURE_DIM:MODEL_INPUT_DIM, :11].numpy(),
                expected_label_channels,
            )
            normalized_conditioned_x = normalized_item["conditioned_x"].numpy()
            normalized_labels = normalized_item["sensor_missing_labels"].numpy()
            normalized_target = int(normalized_item["target_start"])
            for sensor_index in np.flatnonzero(normalized_labels[:, normalized_target]):
                pos_slice, rot_slice = sensor_feature_slices(int(sensor_index))
                self.assertTrue(np.all(normalized_conditioned_x[pos_slice, normalized_target] == 0.0))
                self.assertTrue(np.all(normalized_conditioned_x[rot_slice, normalized_target] == 0.0))
            np.testing.assert_array_equal(normalized_item["inpaint_mask"].numpy(), item["inpaint_mask"].numpy())

            # 新训练路径只支持 materialized task；缺少 x277 时应尽早报错，而不是回退读取源数据。
            legacy_task_path = output_dir / "train" / "tasks" / "legacy_without_x277.npz"
            with np.load(first_task_path, allow_pickle=False) as task_data:
                legacy_task = {key: task_data[key].copy() for key in task_data.files if key != "x277"}
            np.savez(legacy_task_path, **legacy_task)
            legacy_manifest = output_dir / "legacy" / "manifest.jsonl"
            legacy_manifest.parent.mkdir()
            legacy_manifest.write_text(
                json.dumps(
                    {
                        "task_id": "legacy_without_x277",
                        "task_path": "../train/tasks/legacy_without_x277.npz",
                        "seq_len": 11,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            legacy_dataset = X277MissingTaskDataset(
                data_dir=output_dir,
                split="legacy",
                seq_len=11,
                normalize_input=False,
            )
            with self.assertRaises(KeyError):
                _ = legacy_dataset[0]

            # 旧长窗口任务必须重新生成；Dataset 不再把 100/150 帧任务裁成 10+1。
            legacy_long_task_path = output_dir / "train" / "tasks" / "legacy_long_window.npz"
            np.savez(
                legacy_long_task_path,
                x277=np.zeros((150, X277_FEATURE_DIM), dtype=np.float32),
                sensor_missing_labels=np.zeros((150, SENSOR_LABEL_DIM), dtype=bool),
                inpaint_mask=np.zeros((150, MODEL_INPUT_DIM), dtype=bool),
                start_frame=np.int64(0),
                valid_length=np.int64(150),
                source_frames=np.int64(150),
                seq_len=np.int64(150),
            )
            legacy_long_manifest = output_dir / "legacy_long" / "manifest.jsonl"
            legacy_long_manifest.parent.mkdir()
            legacy_long_manifest.write_text(
                json.dumps(
                    {
                        "task_id": "legacy_long_window",
                        "task_path": "../train/tasks/legacy_long_window.npz",
                        "seq_len": 150,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                X277MissingTaskDataset(
                    data_dir=output_dir,
                    split="legacy_long",
                    seq_len=11,
                    normalize_input=False,
                )


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
