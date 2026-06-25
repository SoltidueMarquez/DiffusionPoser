from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from data_loaders.sensor_masking import (
    REALTIME_POSE_BODY_FBX_LOCAL_ROOT_Y0_SCHEMA_NAME,
    REALTIME_POSE_SCHEMA_NAME,
)
from sample import build_predicted_history_cache, reconstruct_rollout, reconstruct_stream


SAMPLE_ENTRYPOINTS = (
    reconstruct_stream,
    reconstruct_rollout,
    build_predicted_history_cache,
)


def write_checkpoint_args(tmp_path: Path, schema_name: str) -> Path:
    model_path = tmp_path / "model000000000.pt"
    model_path.write_bytes(b"")
    (tmp_path / "args.json").write_text(json.dumps({"schema": schema_name}), encoding="utf-8")
    return model_path


def minimal_model_args(tmp_path: Path, model_path: Path, *extra_args: str) -> list[str]:
    return [
        "--model_path",
        str(model_path),
        "--data_dir",
        str(tmp_path / "data"),
        "--cuda",
        "false",
        *extra_args,
    ]


def guard_after_schema_validation(monkeypatch: pytest.MonkeyPatch, module) -> None:
    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("continued past schema validation")

    monkeypatch.setattr(module.dist_util, "setup_dist", fail_if_called)


@pytest.mark.parametrize("module", SAMPLE_ENTRYPOINTS)
def test_sample_model_loader_rejects_legacy_checkpoint_with_explicit_canonical_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module,
) -> None:
    model_path = write_checkpoint_args(tmp_path, REALTIME_POSE_BODY_FBX_LOCAL_ROOT_Y0_SCHEMA_NAME)
    guard_after_schema_validation(monkeypatch, module)

    with pytest.raises(ValueError, match="checkpoint schema"):
        module.main(
            minimal_model_args(
                tmp_path,
                model_path,
                "--schema",
                REALTIME_POSE_SCHEMA_NAME,
            )
        )


@pytest.mark.parametrize("module", SAMPLE_ENTRYPOINTS)
def test_sample_model_loader_rejects_canonical_checkpoint_with_explicit_legacy_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module,
) -> None:
    model_path = write_checkpoint_args(tmp_path, REALTIME_POSE_SCHEMA_NAME)
    guard_after_schema_validation(monkeypatch, module)

    with pytest.raises(ValueError, match="checkpoint schema"):
        module.main(
            minimal_model_args(
                tmp_path,
                model_path,
                "--schema",
                REALTIME_POSE_BODY_FBX_LOCAL_ROOT_Y0_SCHEMA_NAME,
            )
        )


@pytest.mark.parametrize("module", SAMPLE_ENTRYPOINTS)
def test_sample_model_loader_uses_checkpoint_exact_schema_when_cli_schema_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module,
) -> None:
    model_path = write_checkpoint_args(tmp_path, REALTIME_POSE_BODY_FBX_LOCAL_ROOT_Y0_SCHEMA_NAME)
    captured: dict[str, str] = {}

    class StopAfterDatasetSchema:
        def __init__(self, *args, **kwargs):
            del args
            captured["schema_name"] = kwargs["schema_name"]
            raise RuntimeError("stop after schema capture")

    monkeypatch.setattr(module.dist_util, "setup_dist", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.dist_util, "dev", lambda: torch.device("cpu"))
    monkeypatch.setattr(module, "RealtimePoseTaskDataset", StopAfterDatasetSchema)

    with pytest.raises(RuntimeError, match="stop after schema capture"):
        module.main(minimal_model_args(tmp_path, model_path))

    assert captured["schema_name"] == REALTIME_POSE_BODY_FBX_LOCAL_ROOT_Y0_SCHEMA_NAME


@pytest.mark.parametrize("module", SAMPLE_ENTRYPOINTS)
def test_sample_model_loader_parsers_reject_schema_abbreviation(tmp_path: Path, module) -> None:
    parser = module.build_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            minimal_model_args(
                tmp_path,
                tmp_path / "model000000000.pt",
                "--sche",
                REALTIME_POSE_SCHEMA_NAME,
            )
        )
