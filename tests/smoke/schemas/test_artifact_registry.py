from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.artifact_registry import fingerprint_path, load_artifact_registry
from utils.artifact_roots import load_artifact_roots


def _write_roots_config(project_root: Path) -> Path:
    path = project_root / "configs" / "artifact_roots.local.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "workspace_root": ".",
                "amass_root": "store/active/raw/AMASS",
                "smpl_model_dir": "store/active/raw/body_models",
                "generated_root": "store/active/generated",
                "runs_root": "store/active/runs",
                "outputs_root": "store/active/output",
                "external_root": "store/active/external",
                "archive_root": "store/archive/2026-07-cleanup",
                "manifest_root": "store/manifests",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_artifact_roots_resolve_relative_to_project_and_registry_rejects_escape(tmp_path):
    config_path = _write_roots_config(tmp_path)
    roots = load_artifact_roots(config_path=config_path, project_root=tmp_path)

    assert roots.generated_root == (tmp_path / "store/active/generated").resolve()
    registry_path = tmp_path / "configs" / "artifact_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    {
                        "id": "safe.asset",
                        "kind": "fixture",
                        "root_key": "generated_root",
                        "relative_path": "sources/c04",
                        "retention": "active",
                        "status": "fixture",
                        "schema_name": "realtime_pose_stationary5_v1",
                        "dependencies": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    registry = load_artifact_registry(registry_path, project_root=tmp_path)

    assert registry.resolve("safe.asset", roots) == roots.generated_root / "sources/c04"

    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    {
                        "id": "escape.asset",
                        "kind": "fixture",
                        "root_key": "generated_root",
                        "relative_path": "../escape",
                        "retention": "active",
                        "status": "fixture",
                        "schema_name": None,
                        "dependencies": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes"):
        load_artifact_registry(registry_path, project_root=tmp_path)


def test_fingerprint_path_is_stable_for_file_order(tmp_path):
    root = tmp_path / "asset"
    (root / "nested").mkdir(parents=True)
    (root / "nested" / "b.txt").write_text("two", encoding="utf-8")
    (root / "a.txt").write_text("one", encoding="utf-8")

    first = fingerprint_path(root, hash_files=True)
    second = fingerprint_path(root, hash_files=True)

    assert first == second
    assert first["file_count"] == 2
    assert first["size_bytes"] == 6
    assert isinstance(first["tree_sha256"], str)
