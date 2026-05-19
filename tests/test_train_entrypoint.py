from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from train.train_diffusionposer import prepare_save_dir


class TrainEntrypointTest(unittest.TestCase):
    def test_existing_save_dir_requires_overwrite_or_resume(self):
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
