from __future__ import annotations

from pathlib import Path

import pytest

from train.training_loop import (
    find_latest_model_checkpoint,
    find_resume_checkpoint,
    parse_resume_step_from_filename,
)


def test_latest_checkpoint_and_step_resolution(tmp_path):
    (tmp_path / "model000000002.pt").write_bytes(b"")
    (tmp_path / "model000000010.pt").write_bytes(b"")
    assert find_latest_model_checkpoint(tmp_path).name == "model000000010.pt"
    assert Path(find_resume_checkpoint(tmp_path, "latest")).name == "model000000010.pt"
    assert parse_resume_step_from_filename(tmp_path / "model000000010.pt") == 10


def test_explicit_missing_checkpoint_does_not_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_resume_checkpoint(tmp_path, tmp_path / "model000000001.pt")


def test_empty_resume_means_fresh_training(tmp_path):
    assert find_resume_checkpoint(tmp_path, "") == ""
