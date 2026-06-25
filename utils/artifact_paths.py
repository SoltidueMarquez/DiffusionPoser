from __future__ import annotations

from pathlib import Path
from typing import Any


def source_root(roots: Any, schema_name: str, source_set_name: str) -> Path:
    schema = _validate_path_name(schema_name, "schema_name")
    source_set = _validate_path_name(source_set_name, "source_set_name")
    return Path(roots.generated_root) / "sources" / schema / source_set


def task_root(roots: Any, schema_name: str, task_set_name: str) -> Path:
    schema = _validate_path_name(schema_name, "schema_name")
    task_set = _validate_path_name(task_set_name, "task_set_name")
    return Path(roots.generated_root) / "tasks" / schema / task_set


def normalizer_root(roots: Any, schema_name: str, normalizer_name: str) -> Path:
    schema = _validate_path_name(schema_name, "schema_name")
    normalizer = _validate_path_name(normalizer_name, "normalizer_name")
    return Path(roots.generated_root) / "normalizers" / schema / normalizer


def run_root(schema_name: str, experiment_name: str, base_dir: str | Path = "runs") -> Path:
    schema = _validate_path_name(schema_name, "schema_name")
    experiment = _validate_path_name(experiment_name, "experiment_name")
    return Path(base_dir) / schema / experiment


def export_root(schema_name: str, export_name: str, base_dir: str | Path = "output") -> Path:
    schema = _validate_path_name(schema_name, "schema_name")
    export = _validate_path_name(export_name, "export_name")
    return Path(base_dir) / schema / export


def _validate_path_name(value: str, field_name: str) -> str:
    # schema/name 只允许作为单级目录名，避免把上层路径拼进产物目录。
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if "/" in normalized or "\\" in normalized:
        raise ValueError(f"{field_name} must not contain path separators: {normalized}")
    return normalized
