from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any, Iterable, Sequence

from utils.artifact_registry import (
    ArtifactRecord,
    ArtifactRegistry,
    fingerprint_path,
    load_artifact_registry,
    write_file_inventory,
)
from utils.artifact_roots import ArtifactRoots, load_artifact_roots
from utils.filesystem_paths import ensure_directory, filesystem_path, path_exists
from data_loaders.body_fbx_kinematics import load_body_fbx_rest


@dataclass(frozen=True)
class MigrationTarget:
    record: ArtifactRecord
    source: Path
    destination: Path


@dataclass(frozen=True)
class DeletionTarget:
    candidate_id: str
    root_key: str
    relative_path: Path
    path: Path
    category: str
    reproduction_command: str
    retirement_reason: str
    dependencies: tuple[str, ...]
    preconditions: tuple[str, ...]
    registry_asset_ids: tuple[str, ...]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, mirror, migrate, verify, and safely retire DiffusionPoser artifacts.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--action",
        required=True,
        choices=("plan", "mirror", "migrate", "relocate", "verify", "predelete", "delete"),
    )
    parser.add_argument("--asset-id", action="append", dest="asset_ids", default=[])
    parser.add_argument("--source-workspace", default=None, type=Path)
    parser.add_argument("--artifact-roots-config", default=None, type=Path)
    parser.add_argument("--artifact-registry", default=None, type=Path)
    parser.add_argument("--manifest-path", default=None, type=Path)
    parser.add_argument("--cleanup-config", default=None, type=Path)
    parser.add_argument("--deletion-manifest", default=None, type=Path)
    parser.add_argument("--confirm-manifest-sha256", default="", type=str)
    parser.add_argument("--remove-cleanup-config", action="store_true")
    parser.add_argument("--purge-deletion-audit", action="store_true")
    parser.add_argument("--write-registry-fingerprints", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    roots = load_artifact_roots(
        config_path=_resolve_from_project(args.artifact_roots_config, project_root),
        project_root=project_root,
    )
    registry = load_artifact_registry(
        path=_resolve_from_project(args.artifact_registry, project_root),
        project_root=project_root,
    )
    if args.action == "predelete":
        source_workspace = _require_source_workspace(args.source_workspace, project_root)
        cleanup_config = _require_path(args.cleanup_config, "--cleanup-config", project_root)
        manifest_path = _resolve_manifest_path(args.manifest_path, roots, args.action)
        manifest = predelete_candidates(
            cleanup_config=cleanup_config,
            source_workspace=source_workspace,
            registry=registry,
            roots=roots,
            manifest_path=manifest_path,
        )
        _write_json(manifest_path, manifest)
        print(f"[manage_artifacts] predelete manifest={manifest_path}")
        return 0

    if args.action == "delete":
        deletion_manifest = _require_path(args.deletion_manifest, "--deletion-manifest", project_root)
        delete_prepared_manifest(
            manifest_path=deletion_manifest,
            expected_manifest_sha256=str(args.confirm_manifest_sha256),
            registry_path=registry.path,
            registry=registry,
            roots=roots,
            remove_cleanup_config=bool(args.remove_cleanup_config),
            purge_deletion_audit=bool(args.purge_deletion_audit),
        )
        if args.purge_deletion_audit:
            print(f"[manage_artifacts] deleted candidates and purged audit={deletion_manifest}")
        else:
            print(f"[manage_artifacts] deleted manifest={deletion_manifest}")
        return 0

    if not args.asset_ids:
        raise ValueError("--asset-id is required for plan, mirror, migrate, relocate, and verify")

    records = resolve_records_with_dependencies(registry, args.asset_ids)
    manifest_path = _resolve_manifest_path(args.manifest_path, roots, args.action)

    if args.action == "verify":
        manifest = verify_artifacts(registry, roots, records, manifest_path)
        if args.write_registry_fingerprints:
            update_registry_fingerprints(registry.path, manifest["assets"])
        _write_json(manifest_path, manifest)
        print(f"[manage_artifacts] verified manifest={manifest_path}")
        return 0

    if args.action == "relocate":
        targets = build_relocation_targets(records, roots)
        manifest = relocate_artifacts(targets, registry, roots, manifest_path)
        if args.write_registry_fingerprints:
            update_registry_fingerprints(registry.path, manifest["assets"])
        print(f"[manage_artifacts] relocated manifest={manifest_path}")
        return 0

    source_workspace = _require_source_workspace(args.source_workspace, project_root)
    targets = build_targets(records, roots, source_workspace)
    if args.action == "plan":
        manifest = plan_migration(targets, source_workspace, roots)
        _write_json(manifest_path, manifest)
        print(f"[manage_artifacts] plan={manifest_path}")
        return 0

    if args.action == "mirror":
        requested_ids = set(args.asset_ids)
        targets = [target for target in targets if target.record.artifact_id in requested_ids]
        if not targets:
            raise ValueError("mirror requires at least one requested asset with a legacy source path")
        manifest = mirror_artifacts(targets, registry, source_workspace, roots, manifest_path)
        if args.write_registry_fingerprints:
            update_registry_fingerprints(registry.path, manifest["assets"])
        print(f"[manage_artifacts] mirrored manifest={manifest_path}")
        return 0

    manifest = migrate_artifacts(targets, registry, source_workspace, roots, manifest_path)
    if args.write_registry_fingerprints:
        update_registry_fingerprints(registry.path, manifest["assets"])
    print(f"[manage_artifacts] migrated manifest={manifest_path}")
    return 0


def resolve_records_with_dependencies(
    registry: ArtifactRegistry,
    asset_ids: Iterable[str],
) -> list[ArtifactRecord]:
    """按依赖拓扑排序，确保 source/task/run 等父资产先迁移并可验证。"""

    ordered: list[ArtifactRecord] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(artifact_id: str) -> None:
        if artifact_id in visited:
            return
        if artifact_id in visiting:
            raise ValueError(f"artifact dependency cycle detected at: {artifact_id}")
        visiting.add(artifact_id)
        record = registry.get(artifact_id)
        for dependency in record.dependencies:
            visit(dependency)
        visiting.remove(artifact_id)
        visited.add(artifact_id)
        ordered.append(record)

    for artifact_id in asset_ids:
        visit(artifact_id)
    _reject_overlapping_records(ordered)
    return ordered


def build_targets(
    records: Iterable[ArtifactRecord],
    roots: ArtifactRoots,
    source_workspace: Path,
) -> list[MigrationTarget]:
    targets: list[MigrationTarget] = []
    for record in records:
        if record.legacy_relative_path is None:
            # checkpoint/summary logical IDs are nested under a copied run/output asset.
            # They remain in the manifest and are verified after their parent is in place.
            continue
        source = _safe_source_path(source_workspace, record.legacy_relative_path)
        destination = record.resolve(roots)
        if source == destination:
            raise ValueError(f"legacy source and destination are identical: {record.artifact_id}")
        targets.append(MigrationTarget(record=record, source=source, destination=destination))
    return targets


def build_relocation_targets(
    records: Iterable[ArtifactRecord],
    roots: ArtifactRoots,
) -> list[MigrationTarget]:
    """构造 artifact store 内部的路径缩短目标。

    重定位只允许在同一个 ``root_key`` 内执行。实际执行前后都计算完整
    SHA-256；重定位后的任一检查失败时，工具会把路径原子地移回原位置。
    """

    targets: list[MigrationTarget] = []
    for record in records:
        if record.relocate_from_relative_path is None:
            continue
        source = record.resolve_relocation_source(roots)
        destination = record.resolve(roots)
        if source == destination:
            raise ValueError(f"relocation source and destination are identical: {record.artifact_id}")
        targets.append(MigrationTarget(record=record, source=source, destination=destination))
    _reject_overlapping_target_paths(targets)
    return targets


def plan_migration(
    targets: Iterable[MigrationTarget],
    source_workspace: Path,
    roots: ArtifactRoots,
) -> dict[str, Any]:
    assets = []
    for target in targets:
        if not path_exists(target.source):
            raise FileNotFoundError(f"legacy artifact is missing: {target.record.artifact_id} -> {target.source}")
        if path_exists(target.destination):
            raise FileExistsError(f"artifact destination already exists: {target.record.artifact_id} -> {target.destination}")
        assets.append(
            _asset_manifest_entry(
                target,
                state="planned",
                source_fingerprint=fingerprint_path(target.source, hash_files=False),
                source_schema_metadata=inspect_schema_metadata(target.record, target.source),
            )
        )
    return _manifest_header("plan", source_workspace, roots, assets)


def mirror_artifacts(
    targets: Iterable[MigrationTarget],
    registry: ArtifactRegistry,
    source_workspace: Path,
    roots: ArtifactRoots,
    manifest_path: Path,
) -> dict[str, Any]:
    """Copy assets into the store while retaining the original source path."""

    targets = list(targets)
    assets = [_asset_manifest_entry(target, state="pending") for target in targets]
    manifest = _manifest_header("mirror", source_workspace, roots, assets)
    _write_json(manifest_path, manifest)

    for index, target in enumerate(targets):
        entry = manifest["assets"][index]
        if not path_exists(target.source):
            raise FileNotFoundError(
                f"mirror source is missing: {target.record.artifact_id} -> {target.source}"
            )

        entry["state"] = "fingerprinting_source"
        _write_json(manifest_path, manifest)
        source_inventory = _inventory_path(manifest_path, target.record.artifact_id)
        source_fingerprint = write_file_inventory(
            target.source,
            source_inventory,
            artifact_id=target.record.artifact_id,
        )
        source_schema_metadata = inspect_schema_metadata(target.record, target.source)
        source_contract_validation = _validate_record_semantics(target.record, target.source)
        entry["source_fingerprint"] = source_fingerprint
        entry["source_file_inventory"] = str(source_inventory)
        entry["source_schema_metadata"] = source_schema_metadata
        entry["source_contract_validation"] = source_contract_validation
        _write_json(manifest_path, manifest)

        if path_exists(target.destination):
            entry["state"] = "verifying_existing_destination"
            _write_json(manifest_path, manifest)
            destination_fingerprint = fingerprint_path(target.destination, hash_files=True)
            _assert_same_fingerprint(
                target.record.artifact_id,
                source_fingerprint,
                destination_fingerprint,
            )
            destination_schema_metadata = inspect_schema_metadata(target.record, target.destination)
            _assert_same_schema_metadata(
                target.record.artifact_id,
                source_schema_metadata,
                destination_schema_metadata,
            )
            destination_contract_validation = _validate_record_semantics(
                target.record,
                target.destination,
            )
        else:
            entry["state"] = "copying"
            _write_json(manifest_path, manifest)
            staging = _staging_path(target.destination)
            if path_exists(staging):
                raise FileExistsError(f"mirror staging path already exists: {staging}")
            ensure_directory(target.destination.parent)
            entry["staging"] = str(staging)
            _copy_path(target.source, staging)

            entry["state"] = "verifying_copy"
            _write_json(manifest_path, manifest)
            copied_fingerprint = fingerprint_path(staging, hash_files=True)
            _assert_same_fingerprint(target.record.artifact_id, source_fingerprint, copied_fingerprint)
            copied_schema_metadata = inspect_schema_metadata(target.record, staging)
            _assert_same_schema_metadata(
                target.record.artifact_id,
                source_schema_metadata,
                copied_schema_metadata,
            )
            _validate_record_semantics(target.record, staging)
            os.replace(filesystem_path(staging), filesystem_path(target.destination))
            destination_fingerprint = fingerprint_path(target.destination, hash_files=True)
            _assert_same_fingerprint(
                target.record.artifact_id,
                source_fingerprint,
                destination_fingerprint,
            )
            destination_schema_metadata = inspect_schema_metadata(target.record, target.destination)
            _assert_same_schema_metadata(
                target.record.artifact_id,
                source_schema_metadata,
                destination_schema_metadata,
            )
            destination_contract_validation = _validate_record_semantics(
                target.record,
                target.destination,
            )

        _assert_registry_fingerprint(target.record, destination_fingerprint)
        _validate_destination_dependencies(target, registry, roots)
        entry["destination_fingerprint"] = destination_fingerprint
        entry["destination_schema_metadata"] = destination_schema_metadata
        entry["destination_contract_validation"] = destination_contract_validation
        entry.pop("staging", None)
        entry["state"] = "verified_and_source_retained"
        entry["finished_at"] = _now()
        _write_json(manifest_path, manifest)
    return manifest


def predelete_candidates(
    *,
    cleanup_config: Path,
    source_workspace: Path,
    registry: ArtifactRegistry,
    roots: ArtifactRoots,
    manifest_path: Path,
) -> dict[str, Any]:
    """Fingerprint exact deletion candidates without removing any path."""

    if not cleanup_config.exists():
        raise FileNotFoundError(f"cleanup configuration is missing: {cleanup_config}")
    cleanup_payload = _read_json_if_exists(cleanup_config)
    if cleanup_payload.get("schema_version") != 1:
        raise ValueError("cleanup configuration requires schema_version=1")
    raw_candidates = cleanup_payload.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("cleanup configuration requires a non-empty candidates list")
    global_preconditions = _string_list(
        cleanup_payload.get("required_preconditions", []),
        "required_preconditions",
    )
    targets = _parse_deletion_targets(
        raw_candidates,
        source_workspace=source_workspace,
        registry=registry,
        roots=roots,
        global_preconditions=global_preconditions,
    )
    _reject_overlapping_deletion_targets(targets)
    _reject_active_asset_overlap(targets, registry, roots)

    assets = [_deletion_manifest_entry(target, state="pending") for target in targets]
    manifest = _manifest_header("predelete", source_workspace, roots, assets)
    manifest["kind"] = "diffusionposer_predelete_manifest"
    manifest["cleanup_config"] = str(cleanup_config.resolve())
    manifest["cleanup_config_sha256"] = _sha256_file(cleanup_config)
    manifest["review_status"] = "human_review_required"
    manifest["required_preconditions"] = list(global_preconditions)
    _write_json(manifest_path, manifest)

    for index, target in enumerate(targets):
        entry = manifest["assets"][index]
        entry["state"] = "fingerprinting"
        _write_json(manifest_path, manifest)
        inventory_path = _inventory_path(manifest_path, target.candidate_id)
        fingerprint = write_file_inventory(
            target.path,
            inventory_path,
            artifact_id=target.candidate_id,
        )
        entry["file_inventory"] = str(inventory_path)
        entry["fingerprint"] = fingerprint
        entry["state"] = "ready_for_human_review"
        entry["fingerprinted_at"] = _now()
        _write_json(manifest_path, manifest)
    return manifest


def delete_prepared_manifest(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    registry_path: Path,
    registry: ArtifactRegistry,
    roots: ArtifactRoots,
    remove_cleanup_config: bool,
    purge_deletion_audit: bool,
) -> None:
    """按已确认的 predelete 清单删除，并可在全部成功后移除本轮审计文件。"""

    manifest_path = manifest_path.resolve()
    if not expected_manifest_sha256.strip():
        raise ValueError("--confirm-manifest-sha256 is required for delete")
    actual_manifest_sha256 = _sha256_file(manifest_path)
    if actual_manifest_sha256.lower() != expected_manifest_sha256.strip().lower():
        raise RuntimeError(
            "deletion manifest SHA-256 does not match the explicit confirmation: "
            f"{actual_manifest_sha256}"
        )

    manifest = _read_json_if_exists(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "diffusionposer_predelete_manifest":
        raise ValueError(f"not a predelete manifest: {manifest_path}")
    if manifest.get("action") != "predelete":
        raise ValueError(f"predelete manifest action is invalid: {manifest.get('action')!r}")
    source_workspace_text = manifest.get("source_workspace")
    if not isinstance(source_workspace_text, str) or not source_workspace_text.strip():
        raise ValueError("predelete manifest has no source workspace")
    source_workspace = Path(source_workspace_text).resolve()
    if not source_workspace.is_dir():
        raise FileNotFoundError(f"predelete source workspace is missing: {source_workspace}")
    raw_assets = manifest.get("assets")
    if not isinstance(raw_assets, list) or not raw_assets:
        raise ValueError("predelete manifest has no candidate assets")

    targets, expected_fingerprints = _parse_manifest_deletion_targets(
        raw_assets,
        source_workspace=source_workspace,
        registry=registry,
        roots=roots,
    )
    _reject_overlapping_deletion_targets(targets)
    _reject_active_asset_overlap(targets, registry, roots)

    manifest["delete_started_at"] = _now()
    _write_json(manifest_path, manifest)
    entries_by_id = {str(entry["id"]): entry for entry in raw_assets if isinstance(entry, dict)}
    for target in targets:
        entry = entries_by_id[target.candidate_id]
        if entry.get("state") != "ready_for_human_review":
            raise RuntimeError(
                f"candidate is not ready for explicit deletion review: {target.candidate_id}"
            )
        entry["state"] = "revalidating_before_delete"
        _write_json(manifest_path, manifest)
        try:
            current_fingerprint = fingerprint_path(target.path, hash_files=True)
            _assert_same_fingerprint(
                target.candidate_id,
                expected_fingerprints[target.candidate_id],
                current_fingerprint,
            )
        except Exception as error:
            entry["state"] = "fingerprint_mismatch"
            entry["delete_error"] = str(error)
            _write_json(manifest_path, manifest)
            raise

        entry["state"] = "deleting"
        _write_json(manifest_path, manifest)
        try:
            _remove_path(target.path)
        except OSError as error:
            entry["state"] = "deletion_failed"
            entry["delete_error"] = str(error)
            _write_json(manifest_path, manifest)
            raise
        entry["state"] = "deleted"
        entry["deleted_at"] = _now()
        _write_json(manifest_path, manifest)

    try:
        removed_asset_ids = _remove_deleted_registry_records(registry_path, registry, roots, targets)
    except Exception as error:
        manifest["registry_update_error"] = str(error)
        _write_json(manifest_path, manifest)
        raise
    manifest["removed_registry_asset_ids"] = removed_asset_ids
    manifest["delete_finished_at"] = _now()
    manifest["review_status"] = "deleted_after_explicit_hash_confirmation"
    _write_json(manifest_path, manifest)

    if remove_cleanup_config:
        cleanup_config = manifest.get("cleanup_config")
        if not isinstance(cleanup_config, str) or not cleanup_config.strip():
            raise ValueError("predelete manifest has no cleanup configuration path")
        cleanup_path = Path(cleanup_config).resolve()
        if cleanup_path.exists():
            _remove_path(cleanup_path)

    if purge_deletion_audit:
        _purge_predelete_audit(manifest_path)


def migrate_artifacts(
    targets: Iterable[MigrationTarget],
    registry: ArtifactRegistry,
    source_workspace: Path,
    roots: ArtifactRoots,
    manifest_path: Path,
) -> dict[str, Any]:
    targets = list(targets)
    previous = _read_json_if_exists(manifest_path)
    previous_by_id = {
        str(entry["id"]): entry
        for entry in previous.get("assets", [])
        if isinstance(entry, dict) and entry.get("id")
    }
    assets = [
        previous_by_id.get(target.record.artifact_id, _asset_manifest_entry(target, state="pending"))
        for target in targets
    ]
    manifest = _manifest_header("migrate", source_workspace, roots, assets)
    _write_json(manifest_path, manifest)

    for index, target in enumerate(targets):
        entry = manifest["assets"][index]
        source_exists = path_exists(target.source)
        destination_exists = path_exists(target.destination)
        if not source_exists and destination_exists:
            _resume_verified_asset(target, entry)
            _write_json(manifest_path, manifest)
            continue
        if not source_exists:
            raise FileNotFoundError(f"legacy artifact is missing: {target.record.artifact_id} -> {target.source}")
        if destination_exists:
            _finish_verified_destination(
                target,
                registry,
                roots,
                entry,
                manifest_path,
                manifest,
            )
            _write_json(manifest_path, manifest)
            continue
        existing_staging = _find_staging_paths(target.destination)
        if existing_staging:
            if len(existing_staging) != 1:
                raise RuntimeError(
                    f"multiple staging copies require manual inspection for {target.record.artifact_id}: "
                    f"{existing_staging}"
                )
            _recover_staging_copy(target, registry, roots, entry, existing_staging[0])
            _write_json(manifest_path, manifest)
            continue

        entry["state"] = "fingerprinting_source"
        _write_json(manifest_path, manifest)
        source_inventory = _inventory_path(manifest_path, target.record.artifact_id)
        source_fingerprint = write_file_inventory(
            target.source,
            source_inventory,
            artifact_id=target.record.artifact_id,
        )
        source_schema_metadata = inspect_schema_metadata(target.record, target.source)
        entry["source_fingerprint"] = source_fingerprint
        entry["source_file_inventory"] = str(source_inventory)
        entry["source_schema_metadata"] = source_schema_metadata
        entry["state"] = "copying"
        _write_json(manifest_path, manifest)

        staging = _staging_path(target.destination)
        if path_exists(staging):
            raise FileExistsError(f"migration staging path already exists: {staging}")
        ensure_directory(target.destination.parent)
        entry["staging"] = str(staging)
        _copy_path(target.source, staging)

        entry["state"] = "verifying_copy"
        _write_json(manifest_path, manifest)
        copied_fingerprint = fingerprint_path(staging, hash_files=True)
        _assert_same_fingerprint(target.record.artifact_id, source_fingerprint, copied_fingerprint)
        copied_schema_metadata = inspect_schema_metadata(target.record, staging)
        _assert_same_schema_metadata(target.record.artifact_id, source_schema_metadata, copied_schema_metadata)

        os.replace(filesystem_path(staging), filesystem_path(target.destination))
        destination_fingerprint = fingerprint_path(target.destination, hash_files=True)
        _assert_same_fingerprint(target.record.artifact_id, source_fingerprint, destination_fingerprint)
        destination_schema_metadata = inspect_schema_metadata(target.record, target.destination)
        _assert_same_schema_metadata(target.record.artifact_id, source_schema_metadata, destination_schema_metadata)
        _validate_destination_dependencies(target, registry, roots)

        # 删除旧路径只发生在副本、最终目标、schema metadata 和依赖都已验证之后。
        entry["destination_fingerprint"] = destination_fingerprint
        entry["destination_schema_metadata"] = destination_schema_metadata
        entry.pop("staging", None)
        entry["state"] = "removing_source"
        _write_json(manifest_path, manifest)
        try:
            _remove_path(target.source)
        except OSError as error:
            entry["state"] = "source_removal_failed"
            entry["source_removal_error"] = str(error)
            _write_json(manifest_path, manifest)
            raise
        entry["state"] = "verified_and_source_removed"
        entry["finished_at"] = _now()
        _write_json(manifest_path, manifest)
    return manifest


def relocate_artifacts(
    targets: Iterable[MigrationTarget],
    registry: ArtifactRegistry,
    roots: ArtifactRoots,
    manifest_path: Path,
) -> dict[str, Any]:
    """在同一 artifact root 内原子重命名，并在失败时回滚。

    这是为了消除 Windows 的长路径，而不是删除资产。每个目录在重命名前
    先做完整指纹；目标路径和 schema/dependency 校验通过后才记录为完成。
    """

    targets = list(targets)
    previous = _read_json_if_exists(manifest_path)
    previous_by_id = {
        str(entry["id"]): entry
        for entry in previous.get("assets", [])
        if isinstance(entry, dict) and entry.get("id")
    }
    assets = [
        previous_by_id.get(target.record.artifact_id, _asset_manifest_entry(target, state="pending"))
        for target in targets
    ]
    manifest = _manifest_header("relocate", None, roots, assets)
    _write_json(manifest_path, manifest)

    for index, target in enumerate(targets):
        entry = manifest["assets"][index]
        source_exists = path_exists(target.source)
        destination_exists = path_exists(target.destination)
        if not source_exists and destination_exists:
            _resume_relocated_asset(target, entry)
            _write_json(manifest_path, manifest)
            continue
        if not source_exists:
            raise FileNotFoundError(
                f"relocation source is missing: {target.record.artifact_id} -> {target.source}"
            )
        if destination_exists:
            raise FileExistsError(
                f"relocation destination already exists: {target.record.artifact_id} -> {target.destination}"
            )

        entry["state"] = "fingerprinting_source"
        _write_json(manifest_path, manifest)
        source_inventory = _inventory_path(manifest_path, target.record.artifact_id)
        source_fingerprint = write_file_inventory(
            target.source,
            source_inventory,
            artifact_id=target.record.artifact_id,
        )
        source_schema_metadata = inspect_schema_metadata(target.record, target.source)
        entry["source_fingerprint"] = source_fingerprint
        entry["source_file_inventory"] = str(source_inventory)
        entry["source_schema_metadata"] = source_schema_metadata
        entry["state"] = "relocating"
        _write_json(manifest_path, manifest)
        ensure_directory(target.destination.parent)

        moved = False
        try:
            os.replace(filesystem_path(target.source), filesystem_path(target.destination))
            moved = True
            destination_fingerprint = fingerprint_path(target.destination, hash_files=True)
            _assert_same_fingerprint(target.record.artifact_id, source_fingerprint, destination_fingerprint)
            destination_schema_metadata = inspect_schema_metadata(target.record, target.destination)
            _assert_same_schema_metadata(
                target.record.artifact_id,
                source_schema_metadata,
                destination_schema_metadata,
            )
            _validate_destination_dependencies(target, registry, roots)
        except Exception:
            if moved and path_exists(target.destination) and not path_exists(target.source):
                os.replace(filesystem_path(target.destination), filesystem_path(target.source))
                entry["state"] = "rolled_back"
                entry["rolled_back_at"] = _now()
                _write_json(manifest_path, manifest)
            raise

        entry["destination_fingerprint"] = destination_fingerprint
        entry["destination_schema_metadata"] = destination_schema_metadata
        entry["state"] = "verified_and_relocated"
        entry["finished_at"] = _now()
        _write_json(manifest_path, manifest)
    return manifest


def verify_artifacts(
    registry: ArtifactRegistry,
    roots: ArtifactRoots,
    records: Iterable[ArtifactRecord],
    manifest_path: Path,
) -> dict[str, Any]:
    prior = _read_json_if_exists(manifest_path)
    expected_by_id = {
        str(entry["id"]): entry
        for entry in prior.get("assets", [])
        if isinstance(entry, dict) and entry.get("id")
    }
    assets = []
    for record in records:
        destination = record.resolve(roots)
        if not path_exists(destination):
            raise FileNotFoundError(f"registered artifact is missing: {record.artifact_id} -> {destination}")
        fingerprint = fingerprint_path(destination, hash_files=True)
        expected = expected_by_id.get(record.artifact_id, {})
        expected_fingerprint = expected.get("destination_fingerprint") or expected.get("source_fingerprint")
        if isinstance(expected_fingerprint, dict):
            _assert_same_fingerprint(record.artifact_id, expected_fingerprint, fingerprint)
        _assert_registry_fingerprint(record, fingerprint)
        metadata = inspect_schema_metadata(record, destination)
        for dependency in record.dependencies:
            dependency_path = registry.resolve(dependency, roots)
            if not path_exists(dependency_path):
                raise FileNotFoundError(
                    f"registered dependency is missing: {record.artifact_id} -> {dependency_path}"
                )
        assets.append(
            {
                "id": record.artifact_id,
                "source": expected.get("source"),
                "destination": str(destination),
                "destination_fingerprint": fingerprint,
                "destination_schema_metadata": metadata,
                "state": "verified",
                "finished_at": _now(),
            }
        )
    return _manifest_header("verify", None, roots, assets)


def update_registry_fingerprints(registry_path: Path, assets: Iterable[dict[str, Any]]) -> None:
    """只回填本次显式验证的资产，避免配置文件被宽泛扫描结果污染。"""

    values = {
        str(asset["id"]): asset.get("destination_fingerprint")
        for asset in assets
        if isinstance(asset.get("destination_fingerprint"), dict)
    }
    with registry_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    for entry in payload.get("assets", []):
        fingerprint = values.get(str(entry.get("id")))
        if fingerprint is None:
            continue
        entry["expected_file_count"] = fingerprint["file_count"]
        entry["expected_size_bytes"] = fingerprint["size_bytes"]
        entry["expected_tree_sha256"] = fingerprint["tree_sha256"]
        if entry.get("kind") == "body_fbx_rest_contract":
            entry["status"] = "verified_mirrored"
        else:
            entry["status"] = "verified_active" if entry.get("retention") == "active" else "verified_archive"
    _write_json(registry_path, payload)


def inspect_schema_metadata(record: ArtifactRecord, path: Path) -> list[dict[str, Any]]:
    """读取轻量 metadata 中的 schema 标记；没有标记时保留空列表而不猜测。"""

    if record.schema_name is None:
        return []
    findings: list[dict[str, Any]] = []
    for metadata_path in _metadata_candidates(path):
        payload = _read_metadata_payload(metadata_path)
        if payload is None:
            continue
        values = sorted(_collect_schema_values(payload))
        if not values:
            continue
        if any(value != record.schema_name for value in values):
            raise ValueError(
                f"schema metadata mismatch for {record.artifact_id}: {metadata_path} -> {values}"
            )
        findings.append({"path": str(metadata_path), "schema_values": values})
    return findings


def _metadata_candidates(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix in {".json", ".jsonl"} else []
    names = {
        ".realtime_pose_tasks.json",
        "args.json",
        "config.json",
        "latest_normalizer.json",
        "latest_tasks.json",
        "manifest.json",
        "manifest.jsonl",
        "metadata.json",
        "normalizer_meta.json",
        "summary.json",
    }
    candidates = [candidate for candidate in (path / name for name in sorted(names)) if path_exists(candidate)]
    child_names = sorted(os.listdir(filesystem_path(path))) if path_exists(path) else []
    for child_name in child_names:
        child = path / child_name
        if os.path.isdir(filesystem_path(child)):
            candidates.extend(candidate for candidate in (child / name for name in sorted(names)) if path_exists(candidate))
    return candidates


def _resume_verified_asset(target: MigrationTarget, entry: dict[str, Any]) -> None:
    """校验已落位的目标后跳过，允许长时间复制在 metadata 阶段安全恢复。"""

    if entry.get("state") not in {"verified_and_source_removed", "resumed_verified"}:
        raise RuntimeError(
            f"artifact has no verified prior migration state: {target.record.artifact_id}"
        )
    # 首次完成时已经比较了源与目标的完整 hash。仅因为后续资产失败而续跑时，
    # 不应重复扫描数百 GiB；每个条目的原始验证结果仍保留在 manifest 中。
    if entry.get("state") in {"verified_and_source_removed", "resumed_verified"}:
        entry["state"] = "resumed_verified"
        entry["resumed_at"] = _now()
        return
    expected = entry.get("destination_fingerprint") or entry.get("source_fingerprint")
    if not isinstance(expected, dict):
        raise RuntimeError(f"artifact has no prior fingerprint: {target.record.artifact_id}")
    actual = fingerprint_path(target.destination, hash_files=True)
    _assert_same_fingerprint(target.record.artifact_id, expected, actual)
    metadata = inspect_schema_metadata(target.record, target.destination)
    entry["destination_fingerprint"] = actual
    entry["destination_schema_metadata"] = metadata
    entry["state"] = "resumed_verified"
    entry["resumed_at"] = _now()


def _finish_verified_destination(
    target: MigrationTarget,
    registry: ArtifactRegistry,
    roots: ArtifactRoots,
    entry: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    """恢复目标已校验、但旧目录尚未移除的中断迁移。"""

    resumable_states = {
        "verifying_copy",
        "revalidating_existing_destination",
        "removing_source",
        "source_removal_failed",
    }
    if entry.get("state") not in resumable_states:
        raise FileExistsError(
            f"artifact destination already exists without resumable migration state: "
            f"{target.record.artifact_id} -> {target.destination}"
        )
    expected_source = entry.get("source_fingerprint")
    if not isinstance(expected_source, dict):
        raise RuntimeError(f"artifact has no source fingerprint for recovery: {target.record.artifact_id}")

    entry["state"] = "revalidating_existing_destination"
    _write_json(manifest_path, manifest)
    destination_fingerprint = fingerprint_path(target.destination, hash_files=True)
    _assert_same_fingerprint(target.record.artifact_id, expected_source, destination_fingerprint)
    destination_schema_metadata = inspect_schema_metadata(target.record, target.destination)
    source_schema_metadata = entry.get("source_schema_metadata", [])
    if isinstance(source_schema_metadata, list):
        _assert_same_schema_metadata(
            target.record.artifact_id,
            source_schema_metadata,
            destination_schema_metadata,
        )
    _validate_destination_dependencies(target, registry, roots)

    entry["destination_fingerprint"] = destination_fingerprint
    entry["destination_schema_metadata"] = destination_schema_metadata
    entry["state"] = "removing_source"
    _write_json(manifest_path, manifest)
    try:
        _remove_path(target.source)
    except OSError as error:
        entry["state"] = "source_removal_failed"
        entry["source_removal_error"] = str(error)
        _write_json(manifest_path, manifest)
        raise
    entry["state"] = "verified_and_source_removed"
    entry["finished_at"] = _now()
    entry.pop("source_removal_error", None)


def _resume_relocated_asset(target: MigrationTarget, entry: dict[str, Any]) -> None:
    """中断恢复时确认已经落到短路径的资产仍与迁移前指纹一致。"""

    if entry.get("state") not in {"verified_and_relocated", "resumed_relocated"}:
        raise RuntimeError(
            f"artifact has no verified prior relocation state: {target.record.artifact_id}"
        )
    expected = entry.get("destination_fingerprint") or entry.get("source_fingerprint")
    if not isinstance(expected, dict):
        raise RuntimeError(f"artifact has no prior relocation fingerprint: {target.record.artifact_id}")
    actual = fingerprint_path(target.destination, hash_files=True)
    _assert_same_fingerprint(target.record.artifact_id, expected, actual)
    entry["destination_fingerprint"] = actual
    entry["destination_schema_metadata"] = inspect_schema_metadata(
        target.record,
        target.destination,
    )
    entry["state"] = "resumed_relocated"
    entry["resumed_at"] = _now()


def _recover_staging_copy(
    target: MigrationTarget,
    registry: ArtifactRegistry,
    roots: ArtifactRoots,
    entry: dict[str, Any],
    staging: Path,
) -> None:
    """在复制已完成但进程中断时，用完整 hash 恢复而不重复复制大资产。"""

    entry["state"] = "recovering_staging_copy"
    source_fingerprint = fingerprint_path(target.source, hash_files=True)
    previous_source_fingerprint = entry.get("source_fingerprint")
    if isinstance(previous_source_fingerprint, dict):
        _assert_same_fingerprint(
            target.record.artifact_id,
            previous_source_fingerprint,
            source_fingerprint,
        )
    source_schema_metadata = inspect_schema_metadata(target.record, target.source)
    staging_fingerprint = fingerprint_path(staging, hash_files=True)
    _assert_same_fingerprint(target.record.artifact_id, source_fingerprint, staging_fingerprint)
    staging_schema_metadata = inspect_schema_metadata(target.record, staging)
    _assert_same_schema_metadata(
        target.record.artifact_id,
        source_schema_metadata,
        staging_schema_metadata,
    )
    os.replace(filesystem_path(staging), filesystem_path(target.destination))
    destination_fingerprint = fingerprint_path(target.destination, hash_files=True)
    _assert_same_fingerprint(target.record.artifact_id, source_fingerprint, destination_fingerprint)
    destination_schema_metadata = inspect_schema_metadata(target.record, target.destination)
    _assert_same_schema_metadata(
        target.record.artifact_id,
        source_schema_metadata,
        destination_schema_metadata,
    )
    _validate_destination_dependencies(target, registry, roots)
    _remove_path(target.source)
    entry["source_fingerprint"] = source_fingerprint
    entry["source_schema_metadata"] = source_schema_metadata
    entry["destination_fingerprint"] = destination_fingerprint
    entry["destination_schema_metadata"] = destination_schema_metadata
    entry.pop("staging", None)
    entry["state"] = "verified_and_source_removed"
    entry["recovered_at"] = _now()


def _find_staging_paths(destination: Path) -> list[Path]:
    parent = destination.parent
    if not path_exists(parent):
        return []
    prefix = f".{destination.name}.migration-"
    return sorted(
        parent / name
        for name in os.listdir(filesystem_path(parent))
        if name.startswith(prefix)
    )


def _read_metadata_payload(path: Path) -> Any:
    try:
        if path.suffix == ".jsonl":
            with open(filesystem_path(path), "r", encoding="utf-8") as file:
                first_line = file.readline()
            return json.loads(first_line) if first_line.strip() else None
        with open(filesystem_path(path), "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _collect_schema_values(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {"schema", "schema_name"} and isinstance(nested, str):
                result.add(nested)
            result.update(_collect_schema_values(nested))
    elif isinstance(value, list):
        for nested in value:
            result.update(_collect_schema_values(nested))
    return result


def _assert_same_schema_metadata(artifact_id: str, source: list[dict[str, Any]], destination: list[dict[str, Any]]) -> None:
    source_values = sorted({value for item in source for value in item["schema_values"]})
    destination_values = sorted({value for item in destination for value in item["schema_values"]})
    if source_values != destination_values:
        raise RuntimeError(
            f"schema metadata changed during migration for {artifact_id}: {source_values} != {destination_values}"
        )


def _validate_destination_dependencies(
    target: MigrationTarget,
    registry: ArtifactRegistry,
    roots: ArtifactRoots,
) -> None:
    for dependency in target.record.dependencies:
        dependency_path = registry.resolve(dependency, roots)
        if not path_exists(dependency_path):
            raise FileNotFoundError(
                f"artifact dependency is missing after migration: {target.record.artifact_id} -> {dependency_path}"
            )


def _assert_registry_fingerprint(record: ArtifactRecord, fingerprint: dict[str, Any]) -> None:
    expected = {
        "file_count": record.expected_file_count,
        "size_bytes": record.expected_size_bytes,
        "tree_sha256": record.expected_tree_sha256,
    }
    if all(value is None for value in expected.values()):
        return
    if expected != fingerprint:
        raise RuntimeError(f"registered fingerprint mismatch for {record.artifact_id}: {expected} != {fingerprint}")


def _assert_same_fingerprint(artifact_id: str, left: dict[str, Any], right: dict[str, Any]) -> None:
    if left != right:
        raise RuntimeError(f"artifact fingerprint mismatch after copy: {artifact_id}: {left} != {right}")


_DELETION_ROOT_KEYS = frozenset(
    {"workspace_root", "archive_root", "outputs_root", "runs_root", "external_root"}
)
_GLOB_CHARACTERS = frozenset("*?[]{}")


def _validate_record_semantics(record: ArtifactRecord, path: Path) -> dict[str, Any]:
    if record.kind != "body_fbx_rest_contract":
        return {}
    rest = load_body_fbx_rest(path)
    return {
        "bone_count": len(rest.bone_names),
        "parent_count": int(rest.parents.shape[0]),
        "tracker_joint_indices": [int(index) for index in rest.tracker_joint_indices.tolist()],
        "parsed_source_path": str(rest.source_path),
    }


def _parse_deletion_targets(
    raw_candidates: Iterable[Any],
    *,
    source_workspace: Path,
    registry: ArtifactRegistry,
    roots: ArtifactRoots,
    global_preconditions: tuple[str, ...],
) -> list[DeletionTarget]:
    targets: list[DeletionTarget] = []
    candidate_ids: set[str] = set()
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            raise ValueError("cleanup candidates must be objects")
        candidate_id = _required_candidate_string(raw_candidate.get("id"), "candidate id")
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate cleanup candidate id: {candidate_id}")
        candidate_ids.add(candidate_id)
        root_key = _required_candidate_string(
            raw_candidate.get("root_key", "workspace_root"),
            f"root_key for {candidate_id}",
        )
        if root_key not in _DELETION_ROOT_KEYS:
            raise ValueError(f"cleanup candidate has unsupported root_key: {candidate_id} -> {root_key}")
        relative_path = _safe_deletion_relative_path(
            raw_candidate.get("relative_path"),
            candidate_id,
        )
        path = _resolve_deletion_path(
            root_key=root_key,
            relative_path=relative_path,
            source_workspace=source_workspace,
            roots=roots,
            candidate_id=candidate_id,
        )
        if not path_exists(path):
            raise FileNotFoundError(f"cleanup candidate is missing: {candidate_id} -> {path}")
        category = _required_candidate_string(raw_candidate.get("category"), f"category for {candidate_id}")
        reproduction_command = _optional_candidate_string(
            raw_candidate.get("reproduction_command"),
            f"reproduction_command for {candidate_id}",
        )
        retirement_reason = _optional_candidate_string(
            raw_candidate.get("retirement_reason"),
            f"retirement_reason for {candidate_id}",
        )
        if not reproduction_command and not retirement_reason:
            raise ValueError(
                f"cleanup candidate requires reproduction_command or retirement_reason: {candidate_id}"
            )
        dependencies = _string_list(raw_candidate.get("dependencies", []), f"dependencies for {candidate_id}")
        preconditions = global_preconditions + _string_list(
            raw_candidate.get("preconditions", []),
            f"preconditions for {candidate_id}",
        )
        registry_asset_ids = _string_list(
            raw_candidate.get("registry_asset_ids", []),
            f"registry_asset_ids for {candidate_id}",
        )
        for artifact_id in registry_asset_ids:
            registry.get(artifact_id)
        targets.append(
            DeletionTarget(
                candidate_id=candidate_id,
                root_key=root_key,
                relative_path=relative_path,
                path=path,
                category=category,
                reproduction_command=reproduction_command,
                retirement_reason=retirement_reason,
                dependencies=dependencies,
                preconditions=preconditions,
                registry_asset_ids=registry_asset_ids,
            )
        )
    return targets


def _parse_manifest_deletion_targets(
    raw_assets: Iterable[Any],
    *,
    source_workspace: Path,
    registry: ArtifactRegistry,
    roots: ArtifactRoots,
) -> tuple[list[DeletionTarget], dict[str, dict[str, Any]]]:
    targets = _parse_deletion_targets(
        raw_assets,
        source_workspace=source_workspace,
        registry=registry,
        roots=roots,
        global_preconditions=(),
    )
    expected_fingerprints: dict[str, dict[str, Any]] = {}
    for raw_asset in raw_assets:
        if not isinstance(raw_asset, dict):
            raise ValueError("predelete manifest assets must be objects")
        candidate_id = _required_candidate_string(raw_asset.get("id"), "manifest candidate id")
        expected_fingerprints[candidate_id] = _required_fingerprint(
            raw_asset.get("fingerprint"),
            candidate_id,
        )
        expected_path = _resolve_deletion_path(
            root_key=_required_candidate_string(raw_asset.get("root_key"), f"root_key for {candidate_id}"),
            relative_path=_safe_deletion_relative_path(raw_asset.get("relative_path"), candidate_id),
            source_workspace=source_workspace,
            roots=roots,
            candidate_id=candidate_id,
        )
        stored_path = raw_asset.get("path")
        if not isinstance(stored_path, str) or Path(stored_path).resolve() != expected_path:
            raise ValueError(f"predelete manifest candidate path changed: {candidate_id}")
    return targets, expected_fingerprints


def _required_candidate_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"cleanup candidate requires {field_name}")
    return value.strip()


def _optional_candidate_string(value: Any, field_name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"cleanup candidate {field_name} must be a string")
    return value.strip()


def _string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a string list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{field_name} must contain non-empty strings")
    return tuple(item.strip() for item in value)


def _safe_deletion_relative_path(value: Any, candidate_id: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"cleanup candidate requires relative_path: {candidate_id}")
    normalized = value.replace("\\", "/").strip()
    pure = PurePath(normalized)
    if (
        pure.is_absolute()
        or normalized in {"", "."}
        or any(part == ".." for part in pure.parts)
        or any(character in normalized for character in _GLOB_CHARACTERS)
    ):
        raise ValueError(f"cleanup candidate path must be exact and remain under its root: {candidate_id}")
    return Path(normalized)


def _resolve_deletion_path(
    *,
    root_key: str,
    relative_path: Path,
    source_workspace: Path,
    roots: ArtifactRoots,
    candidate_id: str,
) -> Path:
    root = source_workspace if root_key == "workspace_root" else roots.root_for(root_key)
    root = root.resolve()
    candidate = (root / relative_path).resolve()
    if not _path_is_inside(root, candidate) or candidate == root:
        raise ValueError(f"cleanup candidate path escapes or removes its root: {candidate_id}")
    return candidate


def _deletion_manifest_entry(target: DeletionTarget, *, state: str) -> dict[str, Any]:
    return {
        "id": target.candidate_id,
        "root_key": target.root_key,
        "relative_path": target.relative_path.as_posix(),
        "path": str(target.path),
        "category": target.category,
        "reproduction_command": target.reproduction_command,
        "retirement_reason": target.retirement_reason,
        "dependencies": list(target.dependencies),
        "preconditions": list(target.preconditions),
        "registry_asset_ids": list(target.registry_asset_ids),
        "state": state,
    }


def _reject_overlapping_deletion_targets(targets: Iterable[DeletionTarget]) -> None:
    target_list = list(targets)
    for index, first in enumerate(target_list):
        for second in target_list[index + 1 :]:
            if _paths_overlap(first.path, second.path):
                raise ValueError(
                    "cleanup candidates overlap and cannot be deleted independently: "
                    f"{first.candidate_id}, {second.candidate_id}"
                )


def _reject_active_asset_overlap(
    targets: Iterable[DeletionTarget],
    registry: ArtifactRegistry,
    roots: ArtifactRoots,
) -> None:
    for target in targets:
        for record in registry.records.values():
            if record.retention != "active":
                continue
            record_path = record.resolve(roots)
            writable = record.kind.startswith("writable_") or record.status.startswith("writable_")
            if writable:
                if _path_is_inside(target.path, record_path):
                    raise ValueError(
                        "cleanup candidate contains an active writable root: "
                        f"{target.candidate_id} -> {record.artifact_id}"
                    )
                continue
            if _paths_overlap(target.path, record_path):
                raise ValueError(
                    "cleanup candidate overlaps an active asset: "
                    f"{target.candidate_id} -> {record.artifact_id}"
                )


def _required_fingerprint(value: Any, candidate_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"predelete manifest has no fingerprint: {candidate_id}")
    file_count = value.get("file_count")
    size_bytes = value.get("size_bytes")
    tree_sha256 = value.get("tree_sha256")
    if (
        not isinstance(file_count, int)
        or file_count < 0
        or not isinstance(size_bytes, int)
        or size_bytes < 0
        or not isinstance(tree_sha256, str)
        or len(tree_sha256) != 64
    ):
        raise ValueError(f"predelete manifest fingerprint is invalid: {candidate_id}")
    return {
        "file_count": file_count,
        "size_bytes": size_bytes,
        "tree_sha256": tree_sha256.lower(),
    }


def _path_is_inside(root: Path, path: Path) -> bool:
    return path == root or root in path.parents


def _paths_overlap(first: Path, second: Path) -> bool:
    return _path_is_inside(first, second) or _path_is_inside(second, first)


def _remove_deleted_registry_records(
    registry_path: Path,
    registry: ArtifactRegistry,
    roots: ArtifactRoots,
    targets: Iterable[DeletionTarget],
) -> list[str]:
    target_list = list(targets)
    removed_ids = {
        artifact_id
        for target in target_list
        for artifact_id in target.registry_asset_ids
    }
    for record in registry.records.values():
        record_path = record.resolve(roots)
        if any(_path_is_inside(target.path, record_path) for target in target_list):
            removed_ids.add(record.artifact_id)
    for artifact_id in removed_ids:
        record = registry.get(artifact_id)
        if record.retention == "active":
            raise RuntimeError(f"refusing to remove active registry asset: {artifact_id}")
    for record in registry.records.values():
        if record.artifact_id in removed_ids:
            continue
        dangling_dependencies = sorted(set(record.dependencies) & removed_ids)
        if dangling_dependencies:
            raise RuntimeError(
                f"deletion would leave registry dependency references: {record.artifact_id} -> "
                f"{dangling_dependencies}"
            )

    with registry_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    entries = payload.get("assets")
    if not isinstance(entries, list):
        raise ValueError(f"artifact registry has no assets list: {registry_path}")
    payload["assets"] = [
        entry for entry in entries if isinstance(entry, dict) and entry.get("id") not in removed_ids
    ]
    _write_json(registry_path, payload)
    return sorted(removed_ids)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_path(source: Path, destination: Path) -> None:
    if source.is_file():
        shutil.copy2(filesystem_path(source), filesystem_path(destination))
    else:
        shutil.copytree(filesystem_path(source), filesystem_path(destination), copy_function=shutil.copy2)


def _remove_path(path: Path) -> None:
    if path.is_file() or path.is_symlink():
        os.unlink(filesystem_path(path))
    else:
        shutil.rmtree(filesystem_path(path), onerror=_retry_remove_readonly)


def _retry_remove_readonly(function: Any, path: str, exception_info: Any) -> None:
    """Windows 下只对待删除的历史源文件撤销只读属性后重试。"""

    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        function(path)
    except OSError:
        raise exception_info[1]


def _staging_path(destination: Path) -> Path:
    return destination.parent / f".{destination.name}.migration-{uuid.uuid4().hex}"


def _asset_manifest_entry(
    target: MigrationTarget,
    *,
    state: str,
    source_fingerprint: dict[str, Any] | None = None,
    source_schema_metadata: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": target.record.artifact_id,
        "kind": target.record.kind,
        "retention": target.record.retention,
        "schema_name": target.record.schema_name,
        "source": str(target.source),
        "destination": str(target.destination),
        "dependencies": list(target.record.dependencies),
        "state": state,
        "source_fingerprint": source_fingerprint,
        "source_schema_metadata": source_schema_metadata,
    }


def _manifest_header(
    action: str,
    source_workspace: Path | None,
    roots: ArtifactRoots,
    assets: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "diffusionposer_artifact_migration",
        "action": action,
        "created_at": _now(),
        "source_workspace": None if source_workspace is None else str(source_workspace),
        "artifact_roots": {
            "workspace_root": str(roots.workspace_root),
            "amass_root": str(roots.amass_root),
            "generated_root": str(roots.generated_root),
            "runtime_contract_root": str(roots.runtime_contract_root),
            "runs_root": str(roots.runs_root),
            "outputs_root": str(roots.outputs_root),
            "external_root": str(roots.external_root),
            "archive_root": str(roots.archive_root),
            "manifest_root": str(roots.manifest_root),
        },
        "assets": assets,
    }


def _safe_source_path(source_workspace: Path, relative_path: Path) -> Path:
    root = source_workspace.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"legacy artifact path escapes source workspace: {relative_path}")
    return candidate


def _reject_overlapping_records(records: Iterable[ArtifactRecord]) -> None:
    records = list(records)
    for index, first in enumerate(records):
        if first.legacy_relative_path is None:
            continue
        for second in records[index + 1 :]:
            if second.legacy_relative_path is None:
                continue
            left = first.legacy_relative_path
            right = second.legacy_relative_path
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(
                    f"selected artifact records overlap and cannot be copied independently: "
                    f"{first.artifact_id}, {second.artifact_id}"
                )


def _reject_overlapping_target_paths(targets: Iterable[MigrationTarget]) -> None:
    """拒绝任何源/目标包含关系，避免一次重定位覆盖另一资产。"""

    target_list = list(targets)
    paths = [
        (target.record.artifact_id, path)
        for target in target_list
        for path in (target.source, target.destination)
    ]
    for index, (first_id, first_path) in enumerate(paths):
        for second_id, second_path in paths[index + 1 :]:
            if first_id == second_id:
                continue
            if first_path == second_path or first_path in second_path.parents or second_path in first_path.parents:
                raise ValueError(
                    "selected relocation paths overlap and cannot be moved independently: "
                    f"{first_id}, {second_id}"
                )


def _require_source_workspace(source_workspace: Path | None, project_root: Path) -> Path:
    if source_workspace is None:
        raise ValueError("--source-workspace is required for plan, mirror, migrate, and predelete")
    resolved = _resolve_from_project(source_workspace, project_root).resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise FileNotFoundError(f"source workspace does not exist: {resolved}")
    return resolved


def _require_path(path: Path | None, option_name: str, project_root: Path) -> Path:
    if path is None:
        raise ValueError(f"{option_name} is required")
    resolved = _resolve_from_project(path, project_root).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{option_name} does not exist: {resolved}")
    return resolved


def _resolve_manifest_path(path: Path | None, roots: ArtifactRoots, action: str) -> Path:
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = roots.manifest_root / f"{stamp}_artifact_{action}.json"
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _inventory_path(manifest_path: Path, artifact_id: str) -> Path:
    safe_id = "".join(character if character.isalnum() or character in {".", "_", "-"} else "_" for character in artifact_id)
    return manifest_path.parent / f"{manifest_path.stem}.files" / f"{safe_id}.jsonl"


def _purge_predelete_audit(manifest_path: Path) -> None:
    """只移除当前 predelete manifest 派生的 JSONL 目录，避免误删其他迁移审计。"""
    inventory_root = manifest_path.parent / f"{manifest_path.stem}.files"
    if inventory_root.exists() or inventory_root.is_symlink():
        _remove_path(inventory_root)
    if manifest_path.exists() or manifest_path.is_symlink():
        _remove_path(manifest_path)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    temporary.replace(path)


def _resolve_from_project(path: Path | None, project_root: Path) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return project_root / path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
