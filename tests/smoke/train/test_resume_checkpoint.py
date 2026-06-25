from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from data_loaders.sensor_masking import (
    REALTIME_POSE_INPUT_DIM,
    REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SEQ_LEN,
    TASK_MODE_REALTIME_POSE,
)
import train.training_loop as training_loop
from schemas.registry import get_schema_spec
from train.training_loop import (
    TrainLoop,
    find_latest_model_checkpoint,
    find_latest_run_dir,
    find_resume_checkpoint,
    validate_resume_checkpoint_contract,
)


CANONICAL_SCHEMA_NAME = "realtime_pose_stationary5_v1"
LEGACY_SCHEMA_NAME = "realtime_pose_body_fbx_local_root_y0_v1"


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

    def test_latest_keyword_uses_latest_run_pointer_when_save_dir_is_run_root(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = Path(tmp_dir)
            run_dir = run_root / "20260526_120000_debug"
            run_dir.mkdir()
            (run_dir / "model000000012.pt").write_bytes(b"")
            (run_root / "latest_run.json").write_text(
                json.dumps({"save_dir": str(run_dir)}),
                encoding="utf-8",
            )

            resolved = Path(find_resume_checkpoint(run_root, "latest"))

            self.assertEqual(resolved, run_dir / "model000000012.pt")
            self.assertEqual(find_latest_run_dir(run_root), run_dir)

    def test_missing_explicit_checkpoint_raises_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir)
            (save_dir / "model000000006.pt").write_bytes(b"")

            with self.assertRaises(FileNotFoundError):
                find_resume_checkpoint(save_dir, save_dir / "model000000002.pt")

    def test_empty_resume_checkpoint_keeps_fresh_training(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir)
            (save_dir / "model000000006.pt").write_bytes(b"")

            self.assertEqual(find_resume_checkpoint(save_dir, ""), "")

    def test_latest_keyword_requires_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError):
                find_resume_checkpoint(Path(tmp_dir), "latest")

    def test_resume_contract_accepts_current_root_y0_args(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir)
            checkpoint = save_dir / "model000000010.pt"
            checkpoint.write_bytes(b"")
            (save_dir / "args.json").write_text(
                json.dumps(
                    {
                        "task_mode": TASK_MODE_REALTIME_POSE,
                        "schema": REALTIME_POSE_SCHEMA_NAME,
                        "input_feats": REALTIME_POSE_INPUT_DIM,
                        "seq_len": REALTIME_POSE_SEQ_LEN,
                        "max_seq_len": REALTIME_POSE_SEQ_LEN,
                        "model_arch": "full_feature_dit",
                        "root_y_policy": "fixed_zero",
                        "pelvis_height_mode": "pelvis_local_offset_y",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            args = Namespace(schema=REALTIME_POSE_SCHEMA_NAME, model_arch="full_feature_dit")

            validate_resume_checkpoint_contract(checkpoint, args)

    def test_resume_contract_rejects_legacy_schema_even_when_dim_matches(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir)
            checkpoint = save_dir / "model000000010.pt"
            checkpoint.write_bytes(b"")
            (save_dir / "args.json").write_text(
                json.dumps(
                    {
                        "task_mode": TASK_MODE_REALTIME_POSE,
                        "schema": "realtime_pose_v2_contact",
                        "input_feats": REALTIME_POSE_INPUT_DIM,
                        "seq_len": REALTIME_POSE_SEQ_LEN,
                        "max_seq_len": REALTIME_POSE_SEQ_LEN,
                        "model_arch": "full_feature_dit",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            args = Namespace(schema=REALTIME_POSE_SCHEMA_NAME, model_arch="full_feature_dit")

            with self.assertRaisesRegex(ValueError, "schema"):
                validate_resume_checkpoint_contract(checkpoint, args)

    def test_realtime_pose_training_args_accept_trainable_legacy_schema_name(self):
        schema = get_schema_spec(LEGACY_SCHEMA_NAME)
        args = Namespace(
            schema=LEGACY_SCHEMA_NAME,
            input_feats=schema.feature_dim,
            seq_len=schema.seq_len,
            max_seq_len=schema.seq_len,
        )

        validator = getattr(training_loop, "validate_realtime_pose_training_args", None)

        self.assertIsNotNone(validator)
        self.assertEqual(validator(args).name, LEGACY_SCHEMA_NAME)

    def test_resume_contract_rejects_canonical_cli_with_legacy_checkpoint_schema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir)
            checkpoint = save_dir / "model000000010.pt"
            checkpoint.write_bytes(b"")
            canonical = get_schema_spec(CANONICAL_SCHEMA_NAME)
            legacy = get_schema_spec(LEGACY_SCHEMA_NAME)
            (save_dir / "args.json").write_text(
                json.dumps(
                    {
                        "task_mode": TASK_MODE_REALTIME_POSE,
                        "schema": LEGACY_SCHEMA_NAME,
                        "schema_name": LEGACY_SCHEMA_NAME,
                        "schema_canonical_name": legacy.canonical_name,
                        "input_feats": canonical.feature_dim,
                        "seq_len": canonical.seq_len,
                        "max_seq_len": canonical.seq_len,
                        "model_arch": "full_feature_dit",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            args = Namespace(schema=CANONICAL_SCHEMA_NAME, model_arch="full_feature_dit")

            with self.assertRaisesRegex(ValueError, "schema=.*expected"):
                validate_resume_checkpoint_contract(checkpoint, args)

    def test_resume_contract_rejects_legacy_cli_with_canonical_checkpoint_schema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir)
            checkpoint = save_dir / "model000000010.pt"
            checkpoint.write_bytes(b"")
            canonical = get_schema_spec(CANONICAL_SCHEMA_NAME)
            legacy = get_schema_spec(LEGACY_SCHEMA_NAME)
            (save_dir / "args.json").write_text(
                json.dumps(
                    {
                        "task_mode": TASK_MODE_REALTIME_POSE,
                        "schema": CANONICAL_SCHEMA_NAME,
                        "schema_name": CANONICAL_SCHEMA_NAME,
                        "schema_canonical_name": canonical.canonical_name,
                        "input_feats": legacy.feature_dim,
                        "seq_len": legacy.seq_len,
                        "max_seq_len": legacy.seq_len,
                        "model_arch": "full_feature_dit",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            args = Namespace(schema=LEGACY_SCHEMA_NAME, model_arch="full_feature_dit")

            with self.assertRaisesRegex(ValueError, "schema=.*expected"):
                validate_resume_checkpoint_contract(checkpoint, args)


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

    def test_lr_anneal_steps_does_not_stop_training(self):
        loop = object.__new__(TrainLoop)
        loop.model = DummyModel()
        loop.data = [{"x": torch.zeros(1)}, {"x": torch.zeros(1)}]
        loop.device = torch.device("cpu")
        loop.num_epochs = 1
        loop.step = 0
        loop.resume_step = 0
        loop.num_steps = 2
        loop.lr_anneal_steps = 1
        loop.log_interval = 0
        loop.save_interval = 0

        optimized_steps = []
        loop.run_step = lambda batch: optimized_steps.append(loop.step + loop.resume_step)
        loop.log_step = lambda: None
        loop.report_metrics = lambda: None
        loop.save = lambda: None
        loop.evaluate = lambda: None

        loop.run_loop()

        self.assertEqual(optimized_steps, [0, 1])
        self.assertEqual(loop.step, 2)

    def test_anneal_lr_updates_optimizer_param_groups(self):
        loop = object.__new__(TrainLoop)
        loop.lr = 0.1
        loop.lr_anneal_steps = 10
        loop.step = 5
        loop.resume_step = 0
        loop.opt = type("DummyOpt", (), {"param_groups": [{"lr": 0.1}, {"lr": 0.1}]})()

        loop._anneal_lr()

        self.assertAlmostEqual(loop.opt.param_groups[0]["lr"], 0.05)
        self.assertAlmostEqual(loop.opt.param_groups[1]["lr"], 0.05)


if __name__ == "__main__":
    unittest.main()
