from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.artifact_paths import export_root, normalizer_root, run_root, source_root, task_root
from utils.artifact_roots import ArtifactRoots, load_artifact_roots


def make_roots(tmp_path: Path) -> ArtifactRoots:
    return ArtifactRoots(
        workspace_root=tmp_path,
        amass_root=tmp_path / "AMASS",
        smpl_model_dir=tmp_path / "body_models",
        body_fbx_rest_json=None,
        generated_root=tmp_path / "generated",
        runtime_contract_root=tmp_path / "runtime_contracts",
        runs_root=tmp_path / "runs",
        outputs_root=tmp_path / "output",
        external_root=tmp_path / "external",
        archive_root=tmp_path / "archive",
        manifest_root=tmp_path / "manifests",
    )


def test_artifact_roots_can_be_constructed_with_optional_paths(tmp_path: Path) -> None:
    roots = make_roots(tmp_path)

    assert roots.amass_root == (tmp_path / "AMASS").resolve()
    assert roots.generated_root == (tmp_path / "generated").resolve()
    assert roots.body_fbx_rest_json is None
    assert roots.root_for("runtime_contract_root") == (tmp_path / "runtime_contracts").resolve()


def test_artifact_paths_use_schema_set_layout(tmp_path: Path) -> None:
    roots = make_roots(tmp_path)
    schema_name = "realtime_pose_stationary5_v1"

    assert source_root(roots, schema_name, "amass_train") == (
        tmp_path / "generated" / "sources" / schema_name / "amass_train"
    )
    assert task_root(roots, schema_name, "train_task") == (
        tmp_path / "generated" / "tasks" / schema_name / "train_task"
    )
    assert normalizer_root(roots, schema_name, "train_norm") == (
        tmp_path / "generated" / "normalizers" / schema_name / "train_norm"
    )
    assert run_root(schema_name, "exp_a", roots=roots) == tmp_path / "runs" / schema_name / "exp_a"
    assert export_root(schema_name, "unity_a", roots=roots) == (
        tmp_path / "output" / schema_name / "unity_a"
    )


def test_load_artifact_roots_parses_relative_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "roots.json"
    config_path.write_text(
        json.dumps(
            {
                "amass_root": "AMASS",
                "smpl_model_dir": "body_models",
                "body_fbx_rest_json": "runtime_contracts/rest.json",
                "generated_root": "generated",
                "runtime_contract_root": "runtime_contracts",
                "runs_root": "runs",
                "outputs_root": "output",
                "external_root": "external",
                "archive_root": "archive",
                "manifest_root": "manifests",
            }
        ),
        encoding="utf-8",
    )

    roots = load_artifact_roots(config_path=config_path, project_root=tmp_path)

    assert roots.amass_root == tmp_path / "AMASS"
    assert roots.smpl_model_dir == tmp_path / "body_models"
    assert roots.body_fbx_rest_json == tmp_path / "runtime_contracts/rest.json"
    assert roots.runtime_contract_root == tmp_path / "runtime_contracts"


def test_load_artifact_roots_prefers_explicit_then_local_then_example(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    example_path = configs_dir / "artifact_roots.example.json"
    local_path = configs_dir / "artifact_roots.local.json"
    explicit_path = tmp_path / "explicit.json"

    for path, generated_root in (
        (example_path, "generated_from_example"),
        (local_path, "generated_from_local"),
        (explicit_path, "generated_from_explicit"),
    ):
        path.write_text(
            json.dumps({"amass_root": "AMASS", "generated_root": generated_root}),
            encoding="utf-8",
        )

    explicit_roots = load_artifact_roots(config_path=explicit_path, project_root=tmp_path)
    local_roots = load_artifact_roots(project_root=tmp_path)
    local_path.unlink()
    example_roots = load_artifact_roots(project_root=tmp_path)

    assert explicit_roots.generated_root == tmp_path / "generated_from_explicit"
    assert local_roots.generated_root == tmp_path / "generated_from_local"
    assert example_roots.generated_root == tmp_path / "generated_from_example"


@pytest.mark.parametrize("missing_field", ["amass_root", "generated_root"])
def test_load_artifact_roots_requires_required_fields(tmp_path: Path, missing_field: str) -> None:
    payload = {"amass_root": "AMASS", "generated_root": "generated"}
    payload.pop(missing_field)
    config_path = tmp_path / f"missing_{missing_field}.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=missing_field):
        load_artifact_roots(config_path=config_path, project_root=tmp_path)


@pytest.mark.parametrize(
    "field_name",
    ["amass_root", "smpl_model_dir", "body_fbx_rest_json", "generated_root", "runtime_contract_root"],
)
@pytest.mark.parametrize("bad_value", [["AMASS"], {"path": "AMASS"}])
def test_load_artifact_roots_rejects_non_string_path_fields(
    tmp_path: Path,
    field_name: str,
    bad_value: object,
) -> None:
    payload = {"amass_root": "AMASS", "generated_root": "generated"}
    payload[field_name] = bad_value
    config_path = tmp_path / f"bad_{field_name}.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=field_name):
        load_artifact_roots(config_path=config_path, project_root=tmp_path)


@pytest.mark.parametrize("bad_value", ["", "   ", ".", "..", "schema/name", r"schema\name"])
def test_artifact_paths_reject_empty_or_path_like_schema_and_names(tmp_path: Path, bad_value: str) -> None:
    roots = make_roots(tmp_path)
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
