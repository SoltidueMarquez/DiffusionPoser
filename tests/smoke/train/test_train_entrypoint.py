from __future__ import annotations

import tempfile
import sys
import types
import unittest
from argparse import ArgumentParser, Namespace
from pathlib import Path
from unittest.mock import patch

from train.train_diffusionposer import prepare_save_dir, resolve_save_dir, save_args
from train.train_platforms import TensorboardPlatform
from utils import model_util
from utils.parser_util import (
    add_data_options,
    add_model_options,
    add_sampling_options,
    add_training_options,
    apply_ik_calibration,
)


class TrainEntrypointTest(unittest.TestCase):
    def test_ik_calibration_report_fills_missing_training_parameters(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            calibration_path = Path(tmp_dir) / "ik_calibration.json"
            calibration_path.write_text(
                """{
                    "recommended_parameters": {
                        "ik_direction_only_quality": 0.42,
                        "ik_residual_scale": 0.08
                    }
                }""",
                encoding="utf-8",
            )
            args = Namespace(
                ik_calibration_path=str(calibration_path),
                ik_direction_only_quality=None,
                ik_residual_scale=None,
            )

            resolved = apply_ik_calibration(args)

            self.assertEqual(resolved.ik_direction_only_quality, 0.42)
            self.assertEqual(resolved.ik_residual_scale, 0.08)
            self.assertEqual(resolved.ik_calibration_path, str(calibration_path.resolve()))

    def test_explicit_ik_parameter_can_override_calibration_report(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            calibration_path = Path(tmp_dir) / "ik_calibration.json"
            calibration_path.write_text(
                """{
                    "recommended_parameters": {
                        "ik_direction_only_quality": 0.42,
                        "ik_residual_scale": 0.08
                    }
                }""",
                encoding="utf-8",
            )
            args = Namespace(
                ik_calibration_path=str(calibration_path),
                ik_direction_only_quality=0.6,
                ik_residual_scale=None,
            )

            resolved = apply_ik_calibration(args)

            self.assertEqual(resolved.ik_direction_only_quality, 0.6)
            self.assertEqual(resolved.ik_residual_scale, 0.08)

    def test_model_factory_does_not_attach_runtime_reliability_configuration(self):
        args = Namespace(
            input_feats=144,
            latent_dim=64,
            layers=1,
            heads=8,
            dropout=0.0,
            zero_init=True,
            max_seq_len=21,
            model_arch="spatiotemporal_dit",
            d_warm_pos=999,
            tracker_duration_cap=999,
        )

        with (
            patch("utils.model_util.RealtimePoseSpatioTemporalDiT") as model_constructor,
            patch("utils.model_util.create_gaussian_diffusion", return_value=object()),
        ):
            model_util.create_model_and_diffusion(args)

        constructor_kwargs = model_constructor.call_args.kwargs
        self.assertNotIn("reliability_config", constructor_kwargs)
        self.assertNotIn("ik_inpainting_config", constructor_kwargs)

    def test_model_options_do_not_expose_unpropagated_reliability_controls(self):
        parser = ArgumentParser()
        add_model_options(parser)

        invalid_options = (
            "--d_warm_pos",
            "--d_warm_rot",
            "--d_hard",
            "--tracker_duration_cap",
        )
        for option in invalid_options:
            self.assertNotIn(option, parser._option_string_actions)

    def test_data_options_only_expose_materialized_task_controls(self):
        parser = ArgumentParser()
        add_data_options(parser)

        args = parser.parse_args(["--data_dir", "dataset/tasks"])

        self.assertEqual(args.cold_start_prob, 0.1)
        self.assertNotIn("--rollout_steps", parser._option_string_actions)
        self.assertNotIn("--history_pose_noise_std", parser._option_string_actions)
        self.assertNotIn("--tracker_latency_max_frames", parser._option_string_actions)
        self.assertNotIn("--tracker_mask_policy", parser._option_string_actions)

    def test_sampling_options_expose_runtime_rolling_prior_controls(self):
        parser = ArgumentParser()
        add_sampling_options(parser)
        args = parser.parse_args(["--model_path", "model.pt"])

        self.assertEqual(args.tracker_confidence_warmup, 15)
        self.assertEqual(args.fabrik_iterations, 2)
        self.assertIsNone(args.ik_direction_only_quality)
        self.assertIsNone(args.ik_residual_scale)
        self.assertIsNone(args.ik_position_solved_quality)
        self.assertFalse(args.use_future_rolling_prior)
        self.assertEqual(args.future_confidence_decay, 0.9)
        enabled = parser.parse_args(
            ["--model_path", "model.pt", "--use_future_rolling_prior"]
        )
        self.assertTrue(enabled.use_future_rolling_prior)
        self.assertNotIn("--projected_ddim_mode", parser._option_string_actions)
        self.assertNotIn("--projected_ddim_late_steps", parser._option_string_actions)

        for option in (
            "--ik_init_mode",
            "--ik_init_timestep",
            "--ik_init_iterations",
            "--ik_init_lr",
            "--ik_init_pos_weight",
            "--ik_init_rot_weight",
            "--ik_init_reg_weight",
            "--ik_init_delta_limit",
        ):
            self.assertNotIn(option, parser._option_string_actions)

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
        self.assertEqual(args.head_ref_joint_distance_loss_weight, 1.0)
        self.assertEqual(args.hip_height_loss_weight, 1.0)
        self.assertEqual(args.contact_slide_loss_weight, 0.1)
        self.assertNotIn(
            "--temporal_rotation_loss_weight", parser._option_string_actions
        )
        self.assertEqual(args.history_noise_prob, 0.8)
        self.assertNotIn("--world_joint_loss_weight", parser._option_string_actions)
        self.assertNotIn("--rollout_prob", parser._option_string_actions)
        removed_legacy_losses = (
            "--pelvis_fk_loss_weight",
            "--pelvis_offset_loss_weight",
            "--pelvis_consistency_loss_weight",
            "--transition_loss_weight",
        )
        for option in removed_legacy_losses:
            self.assertNotIn(option, parser._option_string_actions)

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

    def test_fresh_training_resolves_save_dir_to_unique_run_child_and_latest_pointer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = Path(tmp_dir) / "runs"
            args = Namespace(
                save_dir=str(run_root),
                overwrite=False,
                resume_checkpoint="",
                run_name="debug run",
                model_arch="spatiotemporal_dit",
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
                model_arch="spatiotemporal_dit",
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

            if sys.platform.startswith("win"):
                fallback_path = Path(
                    (save_dir / "tensorboard_log_dir.txt").read_text(encoding="utf-8")
                )
                self.assertEqual(writer_log_dirs[0], str(fallback_path))
                self.assertTrue(str(fallback_path).isascii())
            else:
                self.assertEqual(writer_log_dirs[0], str(save_dir))


if __name__ == "__main__":
    unittest.main()
