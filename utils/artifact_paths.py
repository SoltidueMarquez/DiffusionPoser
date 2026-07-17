from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.artifact_roots import load_artifact_roots


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


def longseq_eval_root(roots: Any, schema_name: str, eval_set_name: str) -> Path:
    schema = _validate_path_name(schema_name, "schema_name")
    eval_set = _validate_path_name(eval_set_name, "eval_set_name")
    return Path(roots.generated_root) / "longseq_eval" / schema / eval_set


def run_root(
    schema_name: str,
    experiment_name: str,
    base_dir: str | Path | None = None,
    *,
    roots: Any | None = None,
) -> Path:
    schema = _validate_path_name(schema_name, "schema_name")
    experiment = _validate_path_name(experiment_name, "experiment_name")
    if base_dir is not None and roots is not None:
        raise ValueError("run_root accepts either base_dir or roots, not both")
    resolved_base = Path(base_dir) if base_dir is not None else Path(
        (roots or load_artifact_roots()).runs_root
    )
    return resolved_base / schema / experiment


def export_root(
    schema_name: str,
    export_name: str,
    base_dir: str | Path | None = None,
    *,
    roots: Any | None = None,
) -> Path:
    schema = _validate_path_name(schema_name, "schema_name")
    export = _validate_path_name(export_name, "export_name")
    if base_dir is not None and roots is not None:
        raise ValueError("export_root accepts either base_dir or roots, not both")
    resolved_base = Path(base_dir) if base_dir is not None else Path(
        (roots or load_artifact_roots()).outputs_root
    )
    return resolved_base / schema / export


def _validate_path_name(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if normalized in {".", ".."}:
        raise ValueError(f"{field_name} must not be a relative directory marker: {normalized}")
    if "/" in normalized or "\\" in normalized:
        raise ValueError(f"{field_name} must not contain path separators: {normalized}")
    return normalized
