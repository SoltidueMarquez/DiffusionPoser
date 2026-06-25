from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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


def write_checkpoint_args(tmp_path: Path, schema_name: str, schema_key: str = "schema") -> Path:
    model_path = tmp_path / "model000000000.pt"
    model_path.write_bytes(b"")
    (tmp_path / "args.json").write_text(json.dumps({schema_key: schema_name}), encoding="utf-8")
    return model_path


def minimal_model_args(tmp_path: Path, model_path: Path, *extra_args: str) -> list[str]:
    return [
        "--model_path",
        str(model_path),
        "--data_dir",
        str(tmp_path / "data"),
        "--output_dir",
        str(tmp_path / "out"),
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


@pytest.mark.parametrize("schema_key", ("schema", "schema_name"))
@pytest.mark.parametrize("module", SAMPLE_ENTRYPOINTS)
def test_sample_model_loader_uses_checkpoint_exact_schema_when_cli_schema_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module,
    schema_key: str,
) -> None:
    model_path = write_checkpoint_args(
        tmp_path,
        REALTIME_POSE_BODY_FBX_LOCAL_ROOT_Y0_SCHEMA_NAME,
        schema_key=schema_key,
    )
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


def test_build_predicted_history_cache_omitted_schema_uses_temp_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = write_checkpoint_args(tmp_path, REALTIME_POSE_BODY_FBX_LOCAL_ROOT_Y0_SCHEMA_NAME)
    created_dirs: list[Path] = []

    def record_mkdir(self, *args, **kwargs):
        del args, kwargs
        created_dirs.append(self.resolve())

    class StopAfterDatasetSchema:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("stop after output dir capture")

    monkeypatch.setattr(Path, "mkdir", record_mkdir)
    monkeypatch.setattr(build_predicted_history_cache.dist_util, "setup_dist", lambda *args, **kwargs: None)
    monkeypatch.setattr(build_predicted_history_cache.dist_util, "dev", lambda: torch.device("cpu"))
    monkeypatch.setattr(build_predicted_history_cache, "RealtimePoseTaskDataset", StopAfterDatasetSchema)

    with pytest.raises(RuntimeError, match="stop after output dir capture"):
        build_predicted_history_cache.main(minimal_model_args(tmp_path, model_path))

    assert created_dirs == [(tmp_path / "out").resolve()]


def test_reconstruct_rollout_default_output_dir_uses_generic_stationary5_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = write_checkpoint_args(tmp_path, REALTIME_POSE_SCHEMA_NAME)
    saved_paths: list[Path] = []

    class DummyDataset:
        pass

    monkeypatch.setattr(reconstruct_rollout.dist_util, "setup_dist", lambda *args, **kwargs: None)
    monkeypatch.setattr(reconstruct_rollout.dist_util, "dev", lambda: torch.device("cpu"))
    monkeypatch.setattr(reconstruct_rollout, "RealtimePoseTaskDataset", lambda *args, **kwargs: DummyDataset())
    monkeypatch.setattr(reconstruct_rollout, "create_model_and_diffusion", lambda args: (object(), object()))
    monkeypatch.setattr(
        reconstruct_rollout,
        "load_checkpoint_model",
        lambda model, model_path, device, use_ema: (model, "checkpoint"),
    )
    monkeypatch.setattr(
        reconstruct_rollout,
        "rollout_dataset",
        lambda **kwargs: {"metadata": np.asarray({"schema_name": REALTIME_POSE_SCHEMA_NAME}, dtype=object)},
    )
    monkeypatch.setattr(reconstruct_rollout, "save_rollout", lambda path, payload: saved_paths.append(path))

    result = reconstruct_rollout.main(
        [
            "--model_path",
            str(model_path),
            "--data_dir",
            str(tmp_path / "data"),
            "--cuda",
            "false",
        ]
    )

    expected_path = (Path("output/realtime_pose_stationary5_rollout") / "rollout_result.npz").resolve()
    assert saved_paths == [expected_path]
    assert result["output_path"] == expected_path
    assert "body_fbx_local_root_y0" not in expected_path.as_posix()


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
