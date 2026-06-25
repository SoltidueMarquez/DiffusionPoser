from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DataRoots:
    amass_root: Path
    generated_root: Path
    smpl_model_dir: Path | None = None
    body_fbx_rest_json: Path | None = None

    def __post_init__(self) -> None:
        self.amass_root = Path(self.amass_root)
        self.generated_root = Path(self.generated_root)
        if self.smpl_model_dir is not None:
            self.smpl_model_dir = Path(self.smpl_model_dir)
        if self.body_fbx_rest_json is not None:
            self.body_fbx_rest_json = Path(self.body_fbx_rest_json)


def load_data_roots(config_path: str | Path | None = None, project_root: str | Path | None = None) -> DataRoots:
    project_root_path = _default_project_root() if project_root is None else Path(project_root).resolve()
    selected_config_path = _select_config_path(config_path, project_root_path)
    payload = _read_json_object(selected_config_path)

    return DataRoots(
        amass_root=_required_path(payload, "amass_root", project_root_path),
        smpl_model_dir=_optional_path(payload.get("smpl_model_dir"), "smpl_model_dir", project_root_path),
        body_fbx_rest_json=_optional_path(payload.get("body_fbx_rest_json"), "body_fbx_rest_json", project_root_path),
        generated_root=_required_path(payload, "generated_root", project_root_path),
    )


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _select_config_path(config_path: str | Path | None, project_root: Path) -> Path:
    if config_path is not None:
        explicit_path = Path(config_path).expanduser()
        return explicit_path if explicit_path.is_absolute() else project_root / explicit_path

    local_path = project_root / "configs" / "data_roots.local.json"
    if local_path.exists():
        return local_path
    return project_root / "configs" / "data_roots.example.json"


def _read_json_object(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Data roots config must be a JSON object: {config_path}")
    return payload


def _required_path(payload: dict[str, Any], field_name: str, project_root: Path) -> Path:
    if field_name not in payload:
        raise ValueError(f"Missing required data root field: {field_name}")
    path = _optional_path(payload[field_name], field_name, project_root)
    if path is None:
        raise ValueError(f"Missing required data root field: {field_name}")
    return path


def _optional_path(value: Any, field_name: str, project_root: Path) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(f"Data root field must be a string path: {field_name}")

    path_text = os.fspath(value).strip()
    if not path_text:
        return None

    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return project_root / path
