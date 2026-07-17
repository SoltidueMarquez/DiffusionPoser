from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Iterable

from utils.filesystem_paths import filesystem_path, path_exists
from utils.artifact_roots import ArtifactRoots


@dataclass(frozen=True)
class ArtifactRecord:
    """逻辑资产到本地根目录的稳定映射。

    ``relative_path`` 永远相对 ``root_key``，因此实验 profile 不保存个人机器的
    绝对路径，也不会在切换 worktree 后意外读取旧工作树里的产物。
    """

    artifact_id: str
    kind: str
    root_key: str
    relative_path: Path
    retention: str
    status: str
    schema_name: str | None
    dependencies: tuple[str, ...]
    legacy_relative_path: Path | None = None
    relocate_from_relative_path: Path | None = None
    expected_file_count: int | None = None
    expected_size_bytes: int | None = None
    expected_tree_sha256: str | None = None

    def resolve(self, roots: ArtifactRoots) -> Path:
        return _resolve_under_root(roots.root_for(self.root_key), self.relative_path, self.artifact_id)

    def resolve_relocation_source(self, roots: ArtifactRoots) -> Path:
        if self.relocate_from_relative_path is None:
            raise ValueError(f"artifact has no relocation source path: {self.artifact_id}")
        return _resolve_under_root(
            roots.root_for(self.root_key),
            self.relocate_from_relative_path,
            self.artifact_id,
        )


@dataclass(frozen=True)
class ArtifactRegistry:
    path: Path
    schema_version: int
    records: dict[str, ArtifactRecord]

    def get(self, artifact_id: str) -> ArtifactRecord:
        try:
            return self.records[artifact_id]
        except KeyError as error:
            raise KeyError(f"unknown artifact id: {artifact_id}") from error

    def resolve(self, artifact_id: str, roots: ArtifactRoots) -> Path:
        return self.get(artifact_id).resolve(roots)


def load_artifact_registry(
    path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> ArtifactRegistry:
    project_root_path = _default_project_root() if project_root is None else Path(project_root).resolve()
    registry_path = Path(path) if path is not None else project_root_path / "configs" / "artifact_registry.json"
    if not registry_path.is_absolute():
        registry_path = project_root_path / registry_path
    registry_path = registry_path.resolve()
    payload = _read_json_object(registry_path)
    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ValueError(f"unsupported artifact registry schema_version: {schema_version!r}")
    entries = payload.get("assets")
    if not isinstance(entries, list):
        raise ValueError("artifact registry assets must be a list")

    records: dict[str, ArtifactRecord] = {}
    for entry in entries:
        record = _parse_record(entry)
        if record.artifact_id in records:
            raise ValueError(f"duplicate artifact id: {record.artifact_id}")
        records[record.artifact_id] = record
    _validate_dependencies(records)
    return ArtifactRegistry(path=registry_path, schema_version=schema_version, records=records)


def validate_artifacts(
    registry: ArtifactRegistry,
    roots: ArtifactRoots,
    artifact_ids: Iterable[str],
    *,
    require_expected_fingerprint: bool = False,
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for artifact_id in artifact_ids:
        record = registry.get(artifact_id)
        path = record.resolve(roots)
        if not path_exists(path):
            raise FileNotFoundError(f"artifact is missing: {artifact_id} -> {path}")
        if require_expected_fingerprint and (
            record.expected_file_count is None
            or record.expected_size_bytes is None
            or record.expected_tree_sha256 is None
        ):
            raise ValueError(f"artifact has no verified fingerprint: {artifact_id}")
        resolved[artifact_id] = path
    return resolved


def fingerprint_path(path: Path, *, hash_files: bool) -> dict[str, int | str | None]:
    """计算可迁移资产的确定性指纹。

    清单中的 tree hash 会编码相对路径、文件大小与每个文件的 SHA-256。大目录只在
    实际迁移/验证时启用 ``hash_files=True``，普通 profile 校验不重新扫描数十 GiB。
    """

    path = Path(path)
    if not path_exists(path):
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    total_size = 0
    file_count = 0
    for relative_path, file_path in _iter_files(path):
        size = file_path.stat().st_size
        total_size += size
        file_count += 1
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        if hash_files:
            digest.update(_sha256_file(file_path).encode("ascii"))
        digest.update(b"\n")
    return {
        "file_count": file_count,
        "size_bytes": total_size,
        "tree_sha256": digest.hexdigest() if hash_files else None,
    }


def write_file_inventory(
    path: Path,
    inventory_path: Path,
    *,
    artifact_id: str,
) -> dict[str, int | str]:
    """先写出文件级 SHA-256 清单，再返回与 ``fingerprint_path`` 一致的树指纹。

    清单使用 JSONL，避免为包含数十万个 task 文件的资产在内存中累积大列表。
    ``relative_path`` 始终相对资产根，因此目录重定位不会改变清单含义。
    """

    path = Path(path)
    if not path_exists(path):
        raise FileNotFoundError(path)
    inventory_path = Path(inventory_path)
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = inventory_path.with_suffix(inventory_path.suffix + ".tmp")
    digest = hashlib.sha256()
    total_size = 0
    file_count = 0
    with open(filesystem_path(temporary), "w", encoding="utf-8", newline="\n") as file:
        file.write(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "diffusionposer_artifact_file_inventory",
                    "artifact_id": artifact_id,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        for relative_path, file_path in _iter_files(path):
            size = file_path.stat().st_size
            sha256 = _sha256_file(file_path)
            total_size += size
            file_count += 1
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(size).encode("ascii"))
            digest.update(b"\0")
            digest.update(sha256.encode("ascii"))
            digest.update(b"\n")
            file.write(
                json.dumps(
                    {
                        "relative_path": relative_path,
                        "size_bytes": size,
                        "sha256": sha256,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    os.replace(filesystem_path(temporary), filesystem_path(inventory_path))
    return {
        "file_count": file_count,
        "size_bytes": total_size,
        "tree_sha256": digest.hexdigest(),
    }


def _iter_files(path: Path) -> Iterable[tuple[str, Path]]:
    """按稳定顺序遍历文件，不为数十万任务文件一次性构建路径列表。"""

    if path.is_file():
        # 单文件资产的逻辑位置由 registry 管理；临时迁移名不应改变其内容指纹。
        yield ".", path
        return
    root_path = filesystem_path(path)
    for directory, dir_names, file_names in os.walk(root_path):
        dir_names.sort()
        for file_name in sorted(file_names):
            file_path = Path(directory) / file_name
            yield os.path.relpath(str(file_path), root_path).replace("\\", "/"), file_path


def _parse_record(payload: Any) -> ArtifactRecord:
    if not isinstance(payload, dict):
        raise ValueError("artifact record must be an object")
    artifact_id = _required_string(payload, "id")
    dependencies = payload.get("dependencies", [])
    if not isinstance(dependencies, list) or not all(isinstance(item, str) for item in dependencies):
        raise ValueError(f"artifact dependencies must be a string list: {artifact_id}")
    return ArtifactRecord(
        artifact_id=artifact_id,
        kind=_required_string(payload, "kind"),
        root_key=_required_string(payload, "root_key"),
        relative_path=_safe_relative_path(_required_string(payload, "relative_path"), artifact_id),
        retention=_required_string(payload, "retention"),
        status=_required_string(payload, "status"),
        schema_name=_optional_string(payload.get("schema_name"), "schema_name", artifact_id),
        dependencies=tuple(dependencies),
        legacy_relative_path=_optional_relative_path(
            payload.get("legacy_relative_path"), artifact_id
        ),
        relocate_from_relative_path=_optional_relative_path(
            payload.get("relocate_from_relative_path"), artifact_id
        ),
        expected_file_count=_optional_nonnegative_int(
            payload.get("expected_file_count"), "expected_file_count", artifact_id
        ),
        expected_size_bytes=_optional_nonnegative_int(
            payload.get("expected_size_bytes"), "expected_size_bytes", artifact_id
        ),
        expected_tree_sha256=_optional_string(
            payload.get("expected_tree_sha256"), "expected_tree_sha256", artifact_id
        ),
    )


def _validate_dependencies(records: dict[str, ArtifactRecord]) -> None:
    for record in records.values():
        for dependency in record.dependencies:
            if dependency not in records:
                raise ValueError(f"artifact dependency is not registered: {record.artifact_id} -> {dependency}")


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"artifact registry must be a JSON object: {path}")
    return payload


def _required_string(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"artifact record requires non-empty {field_name}")
    return value.strip()


def _optional_string(value: Any, field_name: str, artifact_id: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string for {artifact_id}")
    return value.strip()


def _safe_relative_path(value: str, artifact_id: str) -> Path:
    normalized = value.replace("\\", "/").strip()
    pure = PurePath(normalized)
    if pure.is_absolute() or any(part == ".." for part in pure.parts):
        raise ValueError(f"artifact relative_path escapes its root: {artifact_id}")
    return Path(normalized)


def _resolve_under_root(root: Path, relative_path: Path, artifact_id: str) -> Path:
    root = root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"artifact path escapes root: {artifact_id}")
    return candidate


def _optional_relative_path(value: Any, artifact_id: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"legacy_relative_path must be a non-empty string for {artifact_id}")
    return _safe_relative_path(value, artifact_id)


def _optional_nonnegative_int(value: Any, field_name: str, artifact_id: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer for {artifact_id}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(filesystem_path(path), "rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]
