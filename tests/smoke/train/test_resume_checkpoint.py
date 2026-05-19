from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from train.training_loop import TrainLoop, find_latest_model_checkpoint, find_resume_checkpoint


class DummyModel:
    def train(self):
        return self

    def eval(self):
        return self


class ResumeCheckpointResolutionTest(unittest.TestCase):
    def test_latest_checkpoint_uses_largest_model_step(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir)
            (save_dir / "model000000002.pt").write_bytes(b"")
            (save_dir / "model000000010.pt").write_bytes(b"")
            (save_dir / "opt000000999.pt").write_bytes(b"")
            (save_dir / "ema000000999.pt").write_bytes(b"")

            latest = find_latest_model_checkpoint(save_dir)

            self.assertIsNotNone(latest)
            self.assertEqual(latest.name, "model000000010.pt")

    def test_resume_checkpoint_supports_latest_keyword(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir)
            (save_dir / "model000000004.pt").write_bytes(b"")
            (save_dir / "model000000008.pt").write_bytes(b"")

            resolved = Path(find_resume_checkpoint(save_dir, "latest"))

            self.assertEqual(resolved.name, "model000000008.pt")

    def test_missing_explicit_checkpoint_falls_back_to_latest(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir)
            (save_dir / "model000000006.pt").write_bytes(b"")

            resolved = Path(find_resume_checkpoint(save_dir, save_dir / "model000000002.pt"))

            self.assertEqual(resolved.name, "model000000006.pt")

    def test_empty_resume_checkpoint_keeps_fresh_training(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir)
            (save_dir / "model000000006.pt").write_bytes(b"")

            self.assertEqual(find_resume_checkpoint(save_dir, ""), "")

    def test_latest_keyword_requires_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError):
                find_resume_checkpoint(Path(tmp_dir), "latest")


class TrainLoopStepAccountingTest(unittest.TestCase):
    def test_first_saved_checkpoint_uses_completed_step_one(self):
        loop = object.__new__(TrainLoop)
        loop.model = DummyModel()
        loop.data = [{"x": torch.zeros(1)}]
        loop.device = torch.device("cpu")
        loop.num_epochs = 3
        loop.step = 0
        loop.resume_step = 0
        loop.num_steps = 1
        loop.lr_anneal_steps = 0
        loop.log_interval = 0
        loop.save_interval = 1

        optimized_steps = []
        logged_steps = []
        saved_steps = []
        loop.run_step = lambda batch: optimized_steps.append(loop.step + loop.resume_step)
        loop.log_step = lambda: logged_steps.append(loop.step + loop.resume_step)
        loop.report_metrics = lambda: None
        loop.save = lambda: saved_steps.append(loop.step + loop.resume_step)
        loop.evaluate = lambda: None

        loop.run_loop()

        self.assertEqual(optimized_steps, [0])
        self.assertEqual(logged_steps, [1])
        self.assertEqual(saved_steps, [1])
        self.assertEqual(loop.step, 1)

    def test_resume_at_num_steps_does_not_run_extra_batch(self):
        loop = object.__new__(TrainLoop)
        loop.model = DummyModel()
        loop.data = [{"x": torch.zeros(1)}]
        loop.device = torch.device("cpu")
        loop.num_epochs = 1
        loop.step = 0
        loop.resume_step = 10
        loop.num_steps = 10
        loop.lr_anneal_steps = 0
        loop.run_step = lambda batch: self.fail("run_step should not be called after reaching num_steps")

        loop.run_loop()

        self.assertEqual(loop.step, 0)


if __name__ == "__main__":
    unittest.main()
