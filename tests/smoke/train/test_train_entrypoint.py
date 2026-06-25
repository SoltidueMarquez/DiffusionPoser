from __future__ import annotations

import json
import tempfile
import sys
import types
import unittest
from argparse import ArgumentParser, Namespace
from pathlib import Path
from unittest.mock import patch

from data_loaders.sensor_masking import POSE_REPRESENTATION_KEY, REALTIME_POSE_SCHEMA_NAME, get_schema_spec
import train.train_diffusionposer as train_entrypoint
from train.train_diffusionposer import prepare_save_dir, resolve_save_dir, save_args
from train.train_platforms import TensorboardPlatform
from utils.parser_util import add_data_options, add_training_options, train_args


CANONICAL_SCHEMA_NAME = "realtime_pose_stationary5_v1"
LEGACY_SCHEMA_NAME = "realtime_pose_body_fbx_local_root_y0_v1"


class TrainEntrypointTest(unittest.TestCase):
    def _write_resume_checkpoint(self, tmp_dir: str, schema_name: str) -> tuple[Path, Path]:
        save_dir = Path(tmp_dir) / "run"
        save_dir.mkdir()
        checkpoint = save_dir / "model000000010.pt"
        checkpoint.write_bytes(b"")
        schema = get_schema_spec(schema_name)
        (save_dir / "args.json").write_text(
            json.dumps(
                {
                    "task_mode": "realtime_pose_reconstruction",
                    "schema": schema.name,
                    "schema_name": schema.name,
                    "schema_canonical_name": schema.canonical_name,
                    "input_feats": schema.feature_dim,
                    "seq_len": schema.seq_len,
                    "max_seq_len": schema.seq_len,
                    "model_arch": "full_feature_dit",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return save_dir, checkpoint

    def test_resume_without_cli_schema_inherits_checkpoint_exact_schema_and_default_normalizer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir, checkpoint = self._write_resume_checkpoint(tmp_dir, LEGACY_SCHEMA_NAME)
            args = Namespace(
                save_dir=str(save_dir),
                resume_checkpoint=str(checkpoint),
                schema=CANONICAL_SCHEMA_NAME,
                normalizer_dir=str(Path("dataset/generated/normalizers") / CANONICAL_SCHEMA_NAME / "amass_60hz_train"),
                _normalizer_dir_auto_default=True,
            )

            with patch(
                "utils.default_artifact_paths.load_data_roots",
                return_value=Namespace(generated_root=Path("dataset/generated")),
            ):
                train_entrypoint.resolve_resume_schema_and_paths(
                    args,
                    cli_schema_explicit=False,
                    normalizer_dir_explicit=False,
                )

            self.assertEqual(args.schema, LEGACY_SCHEMA_NAME)
            self.assertEqual(Path(args.save_dir), save_dir.resolve())
            self.assertEqual(Path(args.resume_checkpoint), checkpoint.resolve())
            self.assertEqual(
                Path(args.normalizer_dir).as_posix(),
                f"dataset/generated/normalizers/{LEGACY_SCHEMA_NAME}/amass_60hz_train",
            )

    def test_resume_rejects_explicit_canonical_schema_for_legacy_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir, checkpoint = self._write_resume_checkpoint(tmp_dir, LEGACY_SCHEMA_NAME)
            args = Namespace(
                save_dir=str(save_dir),
                resume_checkpoint=str(checkpoint),
                schema=CANONICAL_SCHEMA_NAME,
                normalizer_dir="",
                _normalizer_dir_auto_default=True,
            )

            with self.assertRaisesRegex(ValueError, "checkpoint schema"):
                train_entrypoint.resolve_resume_schema_and_paths(
                    args,
                    cli_schema_explicit=True,
                    normalizer_dir_explicit=False,
                )

    def test_resume_rejects_explicit_legacy_schema_for_canonical_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir, checkpoint = self._write_resume_checkpoint(tmp_dir, CANONICAL_SCHEMA_NAME)
            args = Namespace(
                save_dir=str(save_dir),
                resume_checkpoint=str(checkpoint),
                schema=LEGACY_SCHEMA_NAME,
                normalizer_dir="",
                _normalizer_dir_auto_default=True,
            )

            with self.assertRaisesRegex(ValueError, "checkpoint schema"):
                train_entrypoint.resolve_resume_schema_and_paths(
                    args,
                    cli_schema_explicit=True,
                    normalizer_dir_explicit=False,
                )

    def test_resume_inherits_schema_but_preserves_explicit_normalizer_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir, checkpoint = self._write_resume_checkpoint(tmp_dir, LEGACY_SCHEMA_NAME)
            args = Namespace(
                save_dir=str(save_dir),
                resume_checkpoint=str(checkpoint),
                schema=CANONICAL_SCHEMA_NAME,
                normalizer_dir="custom/normalizer",
                _normalizer_dir_auto_default=False,
            )

            train_entrypoint.resolve_resume_schema_and_paths(
                args,
                cli_schema_explicit=False,
                normalizer_dir_explicit=True,
            )

            self.assertEqual(args.schema, LEGACY_SCHEMA_NAME)
            self.assertEqual(args.normalizer_dir, "custom/normalizer")

    def test_data_options_default_to_previous_history_and_tracker_augmentation(self):
        parser = ArgumentParser()
        add_data_options(parser)

        args = parser.parse_args(["--data_dir", "dataset/tasks"])

        self.assertEqual(args.schema, CANONICAL_SCHEMA_NAME)
        self.assertEqual(args.schema, REALTIME_POSE_SCHEMA_NAME)
        self.assertEqual(args.history_pose_noise_std, 0.02)
        self.assertEqual(args.history_yaw_noise_std, 0.02)
        self.assertEqual(args.history_pose_dropout_prob, 0.05)
        self.assertEqual(args.history_pose_replace_prob, 0.05)
        self.assertEqual(args.tracker_latency_max_frames, 2)
        self.assertEqual(args.tracker_burst_dropout_prob, 0.05)
        self.assertEqual(args.tracker_outlier_prob, 0.01)

    def test_data_options_reject_legacy_training_schema(self):
        parser = ArgumentParser()
        add_data_options(parser)

        with self.assertRaises(SystemExit):
            parser.parse_args(["--data_dir", "dataset/tasks", "--schema", "realtime_pose_v2_contact"])

    def test_train_args_rejects_schema_abbreviation(self):
        with self.assertRaises(SystemExit):
            train_args(
                [
                    "--data_dir",
                    "dataset/tasks",
                    "--save_dir",
                    "save/test_run",
                    "--sche",
                    CANONICAL_SCHEMA_NAME,
                ]
            )

    def test_data_options_accept_trainable_realtime_pose_schemas(self):
        for schema_name in (CANONICAL_SCHEMA_NAME, LEGACY_SCHEMA_NAME):
            parser = ArgumentParser()
            add_data_options(parser)

            args = parser.parse_args(["--data_dir", "dataset/tasks", "--schema", schema_name])

            self.assertEqual(args.schema, schema_name)

    def test_training_options_default_to_protect_existing_save_dir(self):
        parser = ArgumentParser()
        add_training_options(parser)

        args = parser.parse_args(["--save_dir", "save/test_run"])

        self.assertFalse(args.overwrite)
        self.assertEqual(args.run_name, "auto")
        self.assertEqual(args.save_interval, 5_000)
        self.assertTrue(args.model_ema)
        self.assertEqual(args.tracker_pos_loss_weight, 10.0)
        self.assertEqual(args.tracker_pos_huber_beta, 0.05)
        self.assertEqual(args.tracker_pos_timestep_min_weight, 0.1)
        self.assertEqual(args.tracker_pos_timestep_gamma, 2.0)

    def test_training_options_can_disable_default_model_ema(self):
        parser = ArgumentParser()
        add_training_options(parser)

        args = parser.parse_args(["--save_dir", "save/test_run", "--no-model_ema"])

        self.assertFalse(args.model_ema)

    def test_training_options_can_enable_overwrite(self):
        parser = ArgumentParser()
        add_training_options(parser)

        args = parser.parse_args(["--save_dir", "save/test_run", "--overwrite"])

        self.assertTrue(args.overwrite)

    def test_existing_save_dir_is_allowed_when_overwrite_enabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir) / "run"
            save_dir.mkdir()
            args = Namespace(save_dir=str(save_dir), overwrite=True, resume_checkpoint="")

            prepare_save_dir(args)

            self.assertTrue(save_dir.exists())

    def test_resume_writes_resume_args_without_overwriting_original_args(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir) / "run"
            save_dir.mkdir()
            original = save_dir / "args.json"
            original.write_text('{"old": true}', encoding="utf-8")
            args = Namespace(
                save_dir=str(save_dir),
                resume_checkpoint=str(save_dir / "model000000010.pt"),
                lr=1e-4,
            )

            save_args(args)

            self.assertEqual(original.read_text(encoding="utf-8"), '{"old": true}')
            self.assertTrue((save_dir / "resume_args.json").exists())

    def test_save_args_writes_exact_schema_and_canonical_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir) / "run"
            save_dir.mkdir()
            args = Namespace(
                save_dir=str(save_dir),
                resume_checkpoint="",
                schema=LEGACY_SCHEMA_NAME,
                lr=1e-4,
            )
            schema = get_schema_spec(LEGACY_SCHEMA_NAME)

            save_args(args)

            payload = json.loads((save_dir / "args.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], LEGACY_SCHEMA_NAME)
            self.assertEqual(payload["schema_name"], LEGACY_SCHEMA_NAME)
            self.assertEqual(payload["schema_canonical_name"], schema.canonical_name)
            self.assertEqual(payload[POSE_REPRESENTATION_KEY], schema.pose_representation)
            self.assertEqual(payload["root_y_policy"], schema.root_y_policy)
            self.assertEqual(payload["pelvis_height_mode"], schema.pelvis_height_mode)

    def test_fresh_training_resolves_save_dir_to_unique_run_child_and_latest_pointer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = Path(tmp_dir) / "runs"
            args = Namespace(
                save_dir=str(run_root),
                overwrite=False,
                resume_checkpoint="",
                run_name="debug run",
                schema=REALTIME_POSE_SCHEMA_NAME,
                model_arch="target_dit",
                seed=123,
            )

            resolve_save_dir(args)
            resolved = Path(args.save_dir)
            prepare_save_dir(args)

            self.assertEqual(resolved.parent, run_root.resolve())
            self.assertIn("debug_run", resolved.name)
            self.assertEqual((run_root / "latest_run.txt").read_text(encoding="utf-8"), str(resolved))
            self.assertTrue((run_root / "latest_run.json").exists())

    def test_fresh_training_run_child_avoids_same_second_collision(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = Path(tmp_dir) / "runs"
            args = Namespace(
                save_dir=str(run_root),
                overwrite=False,
                resume_checkpoint="",
                run_name="same",
                schema=REALTIME_POSE_SCHEMA_NAME,
                model_arch="target_dit",
                seed=123,
            )

            resolve_save_dir(args)
            first = Path(args.save_dir)
            prepare_save_dir(args)
            second_args = Namespace(**vars(args))
            second_args.save_dir = str(run_root)
            second_args.resume_checkpoint = ""

            resolve_save_dir(second_args)
            second = Path(second_args.save_dir)

            self.assertNotEqual(first, second)
            self.assertEqual(second.parent, run_root.resolve())

    def test_existing_save_dir_can_still_be_protected_when_overwrite_disabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir) / "run"
            save_dir.mkdir()
            args = Namespace(save_dir=str(save_dir), overwrite=False, resume_checkpoint="")

            with self.assertRaises(FileExistsError):
                prepare_save_dir(args)

    def test_existing_save_dir_is_allowed_for_resume(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir) / "run"
            save_dir.mkdir()
            args = Namespace(
                save_dir=str(save_dir),
                overwrite=False,
                resume_checkpoint=str(save_dir / "model000000010.pt"),
            )

            prepare_save_dir(args)

            self.assertTrue(save_dir.exists())

    def test_tensorboard_platform_falls_back_when_run_path_cannot_be_opened(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir) / "run"
            save_dir.mkdir()
            writer_log_dirs = []

            class FakeSummaryWriter:
                def __init__(self, log_dir):
                    writer_log_dirs.append(str(log_dir))
                    if len(writer_log_dirs) == 1:
                        raise FileNotFoundError("bad tensorboard path")

                def close(self):
                    pass

            fake_tensorboard = types.ModuleType("torch.utils.tensorboard")
            fake_tensorboard.SummaryWriter = FakeSummaryWriter
            with patch.dict(sys.modules, {"torch.utils.tensorboard": fake_tensorboard}):
                platform = TensorboardPlatform(str(save_dir))
                platform.close()

            fallback_path = Path((save_dir / "tensorboard_log_dir.txt").read_text(encoding="utf-8"))
            self.assertEqual(writer_log_dirs[0], str(save_dir))
            self.assertEqual(writer_log_dirs[1], str(fallback_path))
            self.assertTrue(fallback_path.exists())
            self.assertTrue(str(fallback_path).isascii())

    def test_tensorboard_platform_uses_fallback_for_windows_non_ascii_run_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir) / "中文_run"
            save_dir.mkdir()
            writer_log_dirs = []

            class FakeSummaryWriter:
                def __init__(self, log_dir):
                    writer_log_dirs.append(str(log_dir))

                def close(self):
                    pass

            fake_tensorboard = types.ModuleType("torch.utils.tensorboard")
            fake_tensorboard.SummaryWriter = FakeSummaryWriter
            with patch.dict(sys.modules, {"torch.utils.tensorboard": fake_tensorboard}):
                platform = TensorboardPlatform(str(save_dir))
                platform.close()

            fallback_path = Path((save_dir / "tensorboard_log_dir.txt").read_text(encoding="utf-8"))
            expected_first_log_dir = str(fallback_path) if sys.platform.startswith("win") else str(save_dir)
            self.assertEqual(writer_log_dirs[0], expected_first_log_dir)
            if sys.platform.startswith("win"):
                self.assertTrue(str(fallback_path).isascii())


if __name__ == "__main__":
    unittest.main()
