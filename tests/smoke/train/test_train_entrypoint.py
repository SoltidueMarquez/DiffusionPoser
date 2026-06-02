from __future__ import annotations

import tempfile
import unittest
from argparse import ArgumentParser, Namespace
from pathlib import Path

from train.train_diffusionposer import prepare_save_dir, resolve_save_dir, save_args
from utils.parser_util import add_training_options


class TrainEntrypointTest(unittest.TestCase):
    def test_training_options_default_to_protect_existing_save_dir(self):
        parser = ArgumentParser()
        add_training_options(parser)

        args = parser.parse_args(["--save_dir", "save/test_run"])

        self.assertFalse(args.overwrite)
        self.assertEqual(args.run_name, "auto")
        self.assertEqual(args.save_interval, 5_000)
        self.assertEqual(args.tracker_pos_huber_beta, 0.05)
        self.assertEqual(args.tracker_pos_timestep_min_weight, 0.1)
        self.assertEqual(args.tracker_pos_timestep_gamma, 2.0)

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
                schema="realtime_pose_v2_contact",
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
                schema="realtime_pose_v2_contact",
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


if __name__ == "__main__":
    unittest.main()
