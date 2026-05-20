from __future__ import annotations

import tempfile
import unittest
from argparse import ArgumentParser, Namespace
from pathlib import Path

from train.train_diffusionposer import prepare_save_dir
from utils.parser_util import add_training_options


class TrainEntrypointTest(unittest.TestCase):
    def test_training_options_default_to_overwrite(self):
        parser = ArgumentParser()
        add_training_options(parser)

        args = parser.parse_args(["--save_dir", "save/test_run"])

        self.assertTrue(args.overwrite)

    def test_training_options_can_disable_overwrite(self):
        parser = ArgumentParser()
        add_training_options(parser)

        args = parser.parse_args(["--save_dir", "save/test_run", "--no-overwrite"])

        self.assertFalse(args.overwrite)

    def test_existing_save_dir_is_allowed_when_overwrite_enabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir) / "run"
            save_dir.mkdir()
            args = Namespace(save_dir=str(save_dir), overwrite=True, resume_checkpoint="")

            prepare_save_dir(args)

            self.assertTrue(save_dir.exists())

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
