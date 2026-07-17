from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactRoots:
    """本机训练资产的物理根目录。

    路径配置与 Git 工作树分离，避免切换 worktree 时复制 AMASS、任务和
    checkpoint。配置文件允许使用相对路径，统一相对于代码仓库根目录解析。
    """

    workspace_root: Path
    amass_root: Path
    generated_root: Path
    runtime_contract_root: Path
    runs_root: Path
    outputs_root: Path
    external_root: Path
    archive_root: Path
    manifest_root: Path
    smpl_model_dir: Path | None = None
    body_fbx_rest_json: Path | None = None

    def __post_init__(self) -> None:
        """统一将调用方传入的路径标准化，避免测试和 CLI 走出两套语义。"""

        for field_name in (
            "workspace_root",
            "amass_root",
            "generated_root",
            "runtime_contract_root",
            "runs_root",
            "outputs_root",
            "external_root",
            "archive_root",
            "manifest_root",
            "smpl_model_dir",
            "body_fbx_rest_json",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, Path(value).expanduser().resolve())

    def root_for(self, root_key: str) -> Path:
        roots = {
            "workspace_root": self.workspace_root,
            "amass_root": self.amass_root,
            "generated_root": self.generated_root,
            "runtime_contract_root": self.runtime_contract_root,
            "runs_root": self.runs_root,
            "outputs_root": self.outputs_root,
            "external_root": self.external_root,
            "archive_root": self.archive_root,
            "manifest_root": self.manifest_root,
            "smpl_model_dir": self.smpl_model_dir,
            "body_fbx_rest_json": self.body_fbx_rest_json,
        }
        try:
            root = roots[root_key]
        except KeyError as error:
            raise ValueError(f"unknown artifact root key: {root_key}") from error
        if root is None:
            raise ValueError(f"artifact root is not configured: {root_key}")
        return Path(root)


def load_artifact_roots(
    config_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> ArtifactRoots:
    project_root_path = _default_project_root() if project_root is None else Path(project_root).resolve()
    selected_config_path = select_roots_config_path(config_path, project_root_path)
    payload = read_roots_payload(selected_config_path)

    workspace_root = _optional_path(payload.get("workspace_root", "."), "workspace_root", project_root_path)
    if workspace_root is None:
        raise ValueError("workspace_root must not be empty")
    archive_root = _optional_path(
        payload.get("archive_root", "artifactStore/DiffusionPoser/archive"),
        "archive_root",
        project_root_path,
    )
    if archive_root is None:
        raise ValueError("archive_root must not be empty")

    generated_root = _required_path(payload, "generated_root", project_root_path)

    return ArtifactRoots(
        workspace_root=workspace_root,
        amass_root=_required_path(payload, "amass_root", project_root_path),
        smpl_model_dir=_optional_path(payload.get("smpl_model_dir"), "smpl_model_dir", project_root_path),
        body_fbx_rest_json=_optional_path(
            payload.get("body_fbx_rest_json"), "body_fbx_rest_json", project_root_path
        ),
        generated_root=generated_root,
        runtime_contract_root=_required_path_or_default(
            payload,
            "runtime_contract_root",
            generated_root.parent / "runtime_contracts",
            project_root_path,
        ),
        runs_root=_required_path_or_default(payload, "runs_root", project_root_path / "runs", project_root_path),
        outputs_root=_required_path_or_default(
            payload, "outputs_root", project_root_path / "output", project_root_path
        ),
        external_root=_required_path_or_default(
            payload, "external_root", project_root_path / "dataset" / "external", project_root_path
        ),
        archive_root=archive_root,
        manifest_root=_required_path_or_default(
            payload,
            "manifest_root",
            archive_root.parent.parent / "manifests",
            project_root_path,
        ),
    )


def select_roots_config_path(config_path: str | Path | None, project_root: Path) -> Path:
    if config_path is not None:
        explicit_path = Path(config_path).expanduser()
        return explicit_path if explicit_path.is_absolute() else project_root / explicit_path

    candidates = (
        project_root / "configs" / "artifact_roots.local.json",
        project_root / "configs" / "artifact_roots.example.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no roots configuration found under {project_root / 'configs'}")


def read_roots_payload(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"artifact roots config must be a JSON object: {config_path}")
    return payload


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _required_path(payload: dict[str, Any], field_name: str, project_root: Path) -> Path:
    if field_name not in payload:
        raise ValueError(f"missing required artifact root field: {field_name}")
    path = _optional_path(payload[field_name], field_name, project_root)
    if path is None:
        raise ValueError(f"missing required artifact root field: {field_name}")
    return path


def _required_path_or_default(
    payload: dict[str, Any],
    field_name: str,
    default: Path,
    project_root: Path,
) -> Path:
    if field_name not in payload:
        return default
    path = _optional_path(payload[field_name], field_name, project_root)
    if path is None:
        raise ValueError(f"artifact root must not be empty: {field_name}")
    return path


def _optional_path(value: Any, field_name: str, project_root: Path) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(f"artifact root field must be a string path: {field_name}")

    path_text = os.fspath(value).strip()
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()
