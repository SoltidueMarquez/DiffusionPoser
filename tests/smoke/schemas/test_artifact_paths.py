from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def _load_path_helpers():
    assert importlib.util.find_spec("utils.data_roots") is not None, "utils.data_roots module is missing"
    assert importlib.util.find_spec("utils.artifact_paths") is not None, "utils.artifact_paths module is missing"

    from utils.artifact_paths import export_root, normalizer_root, run_root, source_root, task_root
    from utils.data_roots import DataRoots, load_data_roots

    return DataRoots, load_data_roots, source_root, task_root, normalizer_root, run_root, export_root


def test_data_roots_can_be_constructed_with_optional_paths(tmp_path):
    DataRoots, *_ = _load_path_helpers()

    roots = DataRoots(amass_root=tmp_path / "AMASS", generated_root=tmp_path / "generated")

    assert roots.amass_root == tmp_path / "AMASS"
    assert roots.generated_root == tmp_path / "generated"
    assert roots.smpl_model_dir is None
    assert roots.body_fbx_rest_json is None


def test_source_root_uses_generated_schema_set_layout(tmp_path):
    DataRoots, _, source_root, *_ = _load_path_helpers()
    roots = DataRoots(amass_root=tmp_path / "AMASS", generated_root=tmp_path / "generated")

    path = source_root(roots, "realtime_pose_stationary5_v1", "amass_train")

    assert path == tmp_path / "generated" / "sources" / "realtime_pose_stationary5_v1" / "amass_train"


def test_artifact_roots_include_schema_name(tmp_path):
    DataRoots, _, _, task_root, normalizer_root, run_root, export_root = _load_path_helpers()
    schema_name = "realtime_pose_stationary5_v1"
    roots = DataRoots(amass_root=tmp_path / "AMASS", generated_root=tmp_path / "generated")

    task_path = task_root(roots, schema_name, "train_task")
    normalizer_path = normalizer_root(roots, schema_name, "train_norm")
    run_path = run_root(schema_name, "exp_a", base_dir=tmp_path / "runs")
    export_path = export_root(schema_name, "unity_a", base_dir=tmp_path / "output")

    assert task_path == tmp_path / "generated" / "tasks" / schema_name / "train_task"
    assert normalizer_path == tmp_path / "generated" / "normalizers" / schema_name / "train_norm"
    assert run_path == tmp_path / "runs" / schema_name / "exp_a"
    assert export_path == tmp_path / "output" / schema_name / "unity_a"


def test_load_data_roots_parses_relative_paths_and_empty_optional_path(tmp_path):
    _, load_data_roots, *_ = _load_path_helpers()
    config_path = tmp_path / "roots.json"
    config_path.write_text(
        json.dumps(
            {
                "amass_root": "AMASS",
                "smpl_model_dir": "body_models",
                "body_fbx_rest_json": "",
                "generated_root": "generated",
            }
        ),
        encoding="utf-8",
    )

    roots = load_data_roots(config_path=config_path, project_root=tmp_path)

    assert roots.amass_root == tmp_path / "AMASS"
    assert roots.smpl_model_dir == tmp_path / "body_models"
    assert roots.body_fbx_rest_json is None
    assert roots.generated_root == tmp_path / "generated"


@pytest.mark.parametrize("missing_field", ["amass_root", "generated_root"])
def test_load_data_roots_requires_required_fields(tmp_path, missing_field):
    _, load_data_roots, *_ = _load_path_helpers()
    payload = {
        "amass_root": "AMASS",
        "generated_root": "generated",
    }
    payload.pop(missing_field)
    config_path = tmp_path / f"missing_{missing_field}.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=missing_field):
        load_data_roots(config_path=config_path, project_root=tmp_path)


@pytest.mark.parametrize("bad_value", ["", "   ", "schema/name", r"schema\name"])
def test_artifact_paths_reject_empty_or_path_like_schema_and_names(tmp_path, bad_value):
    DataRoots, _, source_root, task_root, normalizer_root, run_root, export_root = _load_path_helpers()
    roots = DataRoots(amass_root=tmp_path / "AMASS", generated_root=tmp_path / "generated")

    builders = [
        lambda value: source_root(roots, value, "amass_train"),
        lambda value: source_root(roots, "realtime_pose_stationary5_v1", value),
        lambda value: task_root(roots, value, "train_task"),
        lambda value: task_root(roots, "realtime_pose_stationary5_v1", value),
        lambda value: normalizer_root(roots, value, "train_norm"),
        lambda value: normalizer_root(roots, "realtime_pose_stationary5_v1", value),
        lambda value: run_root(value, "exp_a", base_dir=tmp_path / "runs"),
        lambda value: run_root("realtime_pose_stationary5_v1", value, base_dir=tmp_path / "runs"),
        lambda value: export_root(value, "unity_a", base_dir=tmp_path / "output"),
        lambda value: export_root("realtime_pose_stationary5_v1", value, base_dir=tmp_path / "output"),
    ]

    for build_path in builders:
        with pytest.raises(ValueError):
            build_path(bad_value)
