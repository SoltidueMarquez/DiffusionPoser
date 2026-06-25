from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from data_loaders.generate_realtime_pose_tasks import main as generate_realtime_pose_tasks_main
from data_loaders.realtime_pose_dataset import RealtimePoseTaskDataset
from data_loaders.sensor_masking import REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SCHEMA_NAME, REALTIME_POSE_SEQ_LEN, get_schema_spec
from tests.smoke.realtime_pose_fixtures import write_toy_source_dataset
from utils.run_dirs import read_latest_pointer
from visual_editor.models import StudioConfig
from visual_editor.realtime_pose import load_task_npz
from visual_editor.services import MotionStudioService


GENERATED_SOURCE_ROOT = "dataset/generated/sources/realtime_pose_stationary5_v1/amass_60hz"
GENERATED_TASK_ROOT = "dataset/generated/tasks/realtime_pose_stationary5_v1/amass_60hz_tasks"
LEGACY_PARENT_LOCAL_MARKERS = (
    "dataset/AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz",
    "dataset/meta_AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz",
)


def normalized_path_text(value) -> str:
    return str(value).replace("\\", "/")


def assert_generated_layout_path(value, expected_suffix: str) -> None:
    text = normalized_path_text(value)
    assert text.endswith(expected_suffix)
    for marker in LEGACY_PARENT_LOCAL_MARKERS:
        assert marker not in text


def import_server_with_optional_dependency_stubs(monkeypatch):
    class FakeFastAPI:
        def __init__(self, *args, **kwargs):
            self.state = types.SimpleNamespace()

        def add_middleware(self, *args, **kwargs):
            return None

        def get(self, *args, **kwargs):
            return lambda fn: fn

        def post(self, *args, **kwargs):
            return lambda fn: fn

        def patch(self, *args, **kwargs):
            return lambda fn: fn

    class FakeHTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class FakeBaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_dump(self, exclude_none: bool = False):
            payload = dict(self.__dict__)
            if exclude_none:
                payload = {key: value for key, value in payload.items() if value is not None}
            return payload

    def fake_field(default=None, default_factory=None, **kwargs):
        del kwargs
        if default_factory is not None:
            return default_factory()
        return default

    fake_fastapi = types.ModuleType("fastapi")
    fake_fastapi.FastAPI = FakeFastAPI
    fake_fastapi.HTTPException = FakeHTTPException
    fake_middleware = types.ModuleType("fastapi.middleware")
    fake_cors = types.ModuleType("fastapi.middleware.cors")
    fake_cors.CORSMiddleware = object
    fake_pydantic = types.ModuleType("pydantic")
    fake_pydantic.BaseModel = FakeBaseModel
    fake_pydantic.Field = fake_field
    monkeypatch.setitem(sys.modules, "fastapi", fake_fastapi)
    monkeypatch.setitem(sys.modules, "fastapi.middleware", fake_middleware)
    monkeypatch.setitem(sys.modules, "fastapi.middleware.cors", fake_cors)
    monkeypatch.setitem(sys.modules, "pydantic", fake_pydantic)
    sys.modules.pop("visual_editor.server", None)

    from visual_editor import server

    return server


def write_task_variant(source_task: Path, output_task: Path, drop_keys: set[str] | None = None, **overrides) -> None:
    drop_keys = drop_keys or set()
    with np.load(source_task, allow_pickle=False) as data:
        payload = {key: data[key].copy() for key in data.files if key not in drop_keys}
    payload.update(overrides)
    np.savez(output_task, **payload)


def generate_first_task(tmp_path: Path) -> Path:
    source_dir = tmp_path / "sources"
    task_dir = tmp_path / "tasks"
    write_toy_source_dataset(source_dir)
    generate_realtime_pose_tasks_main(
        [
            "--source_dir",
            str(source_dir),
            "--output_dir",
            str(task_dir),
            "--splits",
            "train",
            "--samples_per_file",
            "1",
            "--schema",
            REALTIME_POSE_SCHEMA_NAME,
            "--split_dir",
            "",
            "--overwrite",
        ]
    )
    task_output_dir = read_latest_pointer(task_dir, kind="tasks")
    assert task_output_dir is not None
    manifest_path = task_output_dir / "train" / "manifest.jsonl"
    entry = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])
    return manifest_path.parent / entry["task_path"]


def test_visual_editor_server_defaults_use_generated_artifact_layout(monkeypatch, tmp_path):
    monkeypatch.setenv("REALTIME_POSE_EDITOR_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("REALTIME_POSE_EDITOR_OUTPUT_DIR", str(tmp_path / "exports"))
    monkeypatch.delenv("REALTIME_POSE_EDITOR_SOURCE_DIR", raising=False)
    monkeypatch.delenv("REALTIME_POSE_EDITOR_DATA_DIR", raising=False)

    server = import_server_with_optional_dependency_stubs(monkeypatch)

    config = server.config_from_env()
    args = server.build_argument_parser().parse_args([])

    assert_generated_layout_path(config.source_dir, GENERATED_SOURCE_ROOT)
    assert_generated_layout_path(config.data_dir, GENERATED_TASK_ROOT)
    assert_generated_layout_path(args.source_dir, GENERATED_SOURCE_ROOT)
    assert_generated_layout_path(args.data_dir, GENERATED_TASK_ROOT)


def test_visual_editor_server_preserves_env_path_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("REALTIME_POSE_EDITOR_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("REALTIME_POSE_EDITOR_OUTPUT_DIR", str(tmp_path / "exports"))
    monkeypatch.setenv("REALTIME_POSE_EDITOR_SOURCE_DIR", "custom/source")
    monkeypatch.setenv("REALTIME_POSE_EDITOR_DATA_DIR", "custom/tasks")

    server = import_server_with_optional_dependency_stubs(monkeypatch)

    config = server.config_from_env()
    args = server.build_argument_parser().parse_args([])

    assert normalized_path_text(config.source_dir) == "custom/source"
    assert normalized_path_text(config.data_dir) == "custom/tasks"
    assert normalized_path_text(args.source_dir) == "custom/source"
    assert normalized_path_text(args.data_dir) == "custom/tasks"


def test_visual_editor_ai_index_defaults_to_generated_layout():
    index_path = Path("visual_editor/ai_index.json")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    api = payload["entrypoints"]["api"]
    assert GENERATED_SOURCE_ROOT in api
    assert GENERATED_TASK_ROOT in api
    for marker in LEGACY_PARENT_LOCAL_MARKERS:
        assert marker not in api
    assert "body_fbx_local_root_y0" in payload["data_contract"]["source"]
    assert REALTIME_POSE_SCHEMA_NAME in payload["data_contract"]["source"]


def test_visual_editor_readme_examples_use_generated_layout():
    readme = Path("visual_editor/README.md").read_text(encoding="utf-8")

    assert GENERATED_SOURCE_ROOT in readme
    assert GENERATED_TASK_ROOT in readme
    for marker in LEGACY_PARENT_LOCAL_MARKERS:
        assert marker not in readme


def test_visual_editor_launcher_defaults_use_generated_layout():
    launcher_paths = [
        Path("visual_editor/electron/main.cjs"),
        Path("visual_editor/scripts/start.ps1"),
        Path("visual_editor/scripts/start_web.ps1"),
    ]

    for path in launcher_paths:
        text = normalized_path_text(path.read_text(encoding="utf-8"))
        assert GENERATED_SOURCE_ROOT in text
        assert GENERATED_TASK_ROOT in text
        for marker in LEGACY_PARENT_LOCAL_MARKERS:
            assert marker not in text


def test_visual_editor_task_loader_rejects_missing_contract_metadata(tmp_path):
    source_task = generate_first_task(tmp_path)
    assert load_task_npz(source_task)["sensor_valid"].shape[0] == REALTIME_POSE_SEQ_LEN

    missing_policy = source_task.parent / "missing_root_y_policy.npz"
    write_task_variant(source_task, missing_policy, drop_keys={"root_y_policy"})
    with pytest.raises(KeyError, match="root_y_policy"):
        load_task_npz(missing_policy)

    missing_mode = source_task.parent / "missing_pelvis_height_mode.npz"
    write_task_variant(source_task, missing_mode, drop_keys={"pelvis_height_mode"})
    with pytest.raises(KeyError, match="pelvis_height_mode"):
        load_task_npz(missing_mode)


def test_visual_editor_task_loader_rejects_root_y0_invariant_errors(tmp_path):
    source_task = generate_first_task(tmp_path)
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    with np.load(source_task, allow_pickle=False) as data:
        bad_root_pos = data["root_pos_world"].copy()
        bad_pelvis_height = data[schema.pelvis_height_key].copy()

    bad_root_pos[:, 1] = 0.2
    bad_root_task = source_task.parent / "bad_root_y0.npz"
    write_task_variant(source_task, bad_root_task, root_pos_world=bad_root_pos)
    with pytest.raises(ValueError, match="root-y0"):
        load_task_npz(bad_root_task)

    bad_pelvis_height[:, 0] += 0.3
    bad_pelvis_task = source_task.parent / "bad_pelvis_height.npz"
    write_task_variant(source_task, bad_pelvis_task, **{schema.pelvis_height_key: bad_pelvis_height})
    with pytest.raises(ValueError, match="joints_world"):
        load_task_npz(bad_pelvis_task)


def test_realtime_pose_studio_scans_frames_and_exports_tasks(tmp_path):
    source_dir = tmp_path / "sources"
    task_dir = tmp_path / "tasks"
    result_dir = tmp_path / "results"
    export_dir = tmp_path / "exports"
    runtime_dir = tmp_path / "runtime"
    write_toy_source_dataset(source_dir)
    generate_realtime_pose_tasks_main(
        [
            "--source_dir",
            str(source_dir),
            "--output_dir",
            str(task_dir),
            "--splits",
            "train",
            "--samples_per_file",
            "1",
            "--schema",
            REALTIME_POSE_SCHEMA_NAME,
            "--split_dir",
            "",
            "--overwrite",
        ]
    )
    result_dir.mkdir()
    np.savez(
        result_dir / "toy_result.npz",
        schema_name=np.asarray(REALTIME_POSE_SCHEMA_NAME),
        reference_features=np.zeros((1, REALTIME_POSE_SEQ_LEN, REALTIME_POSE_INPUT_DIM), dtype=np.float32),
        reconstructed_features=np.zeros((1, REALTIME_POSE_SEQ_LEN, REALTIME_POSE_INPUT_DIM), dtype=np.float32),
    )
    service = MotionStudioService(
        StudioConfig.from_paths(
            amass_dir=tmp_path / "amass",
            source_dir=source_dir,
            data_dir=task_dir,
            result_dir=result_dir,
            output_dir=export_dir,
            runtime_dir=runtime_dir,
            realtime_pose_fps=60.0,
        )
    )
    payload = service.library_payload()
    kinds = {asset["kind"] for asset in payload["assets"]}
    assert {"source", "task", "result"}.issubset(kinds)
    result_asset = next(asset for asset in payload["assets"] if asset["kind"] == "result")
    assert result_asset["frame_count"] == REALTIME_POSE_SEQ_LEN
    source_asset = next(asset for asset in payload["assets"] if asset["kind"] == "source")
    frames = service.frames_payload(asset_id=source_asset["asset_id"], track_id="realtime_source", start=0, count=2)
    assert frames["count"] == 2
    assert len(frames["frames"][0]["trackers"]) == 6
    assert len(frames["frames"][0]["sensor_valid"]) == 6
    assert "contact" not in frames["frames"][0]
    assert "sensor_missing_labels" not in frames["frames"][0]

    project = service.edit.create_project(asset_id=source_asset["asset_id"], track_id="realtime_source")
    exported = service.edit.export(
        project_id=project["project_id"],
        request={
            "output_dir": str(export_dir),
            "frame_start": 60,
            "frame_end": 60,
            "tracker_patterns": ["full-trackers", "mixed-sparse"],
            "split": "train",
            "export_name": "studio_export",
        },
    )
    dataset = RealtimePoseTaskDataset(exported["export_dir"], split="train", normalize_input=False)
    assert len(dataset) == 2
    assert exported["mask_policy"] == "fixed_patterns"
    assert tuple(dataset[0]["x"].shape) == (REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SEQ_LEN)
