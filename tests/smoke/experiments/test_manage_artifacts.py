from __future__ import annotations

import json
import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.experiments.manage_artifacts import (
    MigrationTarget,
    _finish_verified_destination,
    build_relocation_targets,
    build_targets,
    delete_prepared_manifest,
    migrate_artifacts,
    mirror_artifacts,
    predelete_candidates,
    relocate_artifacts,
    resolve_records_with_dependencies,
    update_registry_fingerprints,
    verify_artifacts,
)
from utils.artifact_registry import ArtifactRecord, ArtifactRegistry, fingerprint_path, load_artifact_registry
from utils.artifact_roots import ArtifactRoots
from data_loaders.body_fbx_kinematics import SMPL_JOINT_NAMES, SMPL_PARENTS, TRACKER_JOINT_INDICES


def _write_registry(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    {
                        "id": "raw.fixture",
                        "kind": "raw",
                        "root_key": "amass_root",
                        "relative_path": ".",
                        "legacy_relative_path": "dataset/raw",
                        "retention": "active",
                        "status": "migration_pending",
                        "schema_name": None,
                        "dependencies": [],
                    },
                    {
                        "id": "source.fixture",
                        "kind": "source",
                        "root_key": "generated_root",
                        "relative_path": "sources/realtime_pose_stationary5_v1/c04",
                        "legacy_relative_path": "dataset/generated/source_c04",
                        "retention": "active",
                        "status": "migration_pending",
                        "schema_name": "realtime_pose_stationary5_v1",
                        "dependencies": ["raw.fixture"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _roots(tmp_path: Path) -> ArtifactRoots:
    store = tmp_path / "artifact_store"
    return ArtifactRoots(
        workspace_root=tmp_path,
        amass_root=store / "active/raw/AMASS",
        smpl_model_dir=store / "active/raw/body_models",
        generated_root=store / "active/generated",
        runtime_contract_root=store / "active/runtime_contracts",
        runs_root=store / "active/runs",
        outputs_root=store / "active/output",
        external_root=store / "active/external",
        archive_root=store / "archive/2026-07-cleanup",
        manifest_root=store / "manifests",
    )


def _write_body_fbx_rest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "boneNames": list(SMPL_JOINT_NAMES),
        "parents": [int(value) for value in SMPL_PARENTS.tolist()],
        "restLocalPositions": [{"x": 0.0, "y": 0.05, "z": 0.0} for _ in SMPL_JOINT_NAMES],
        "restLocalRotations": [{"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0} for _ in SMPL_JOINT_NAMES],
        "trackerJointIndices": [int(value) for value in TRACKER_JOINT_INDICES.tolist()],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_migration_copies_verifies_and_only_then_removes_source(tmp_path):
    source_workspace = tmp_path / "legacy_workspace"
    (source_workspace / "dataset/raw").mkdir(parents=True)
    (source_workspace / "dataset/raw" / "amass.txt").write_text("raw", encoding="utf-8")
    source = source_workspace / "dataset/generated/source_c04"
    source.mkdir(parents=True)
    (source / "metadata.json").write_text(
        json.dumps({"schema_name": "realtime_pose_stationary5_v1"}),
        encoding="utf-8",
    )
    (source / "sample.bin").write_bytes(b"fixture")

    registry_path = tmp_path / "artifact_registry.json"
    _write_registry(registry_path)
    registry = load_artifact_registry(registry_path, project_root=tmp_path)
    roots = _roots(tmp_path)
    records = resolve_records_with_dependencies(registry, ["source.fixture"])
    targets = build_targets(records, roots, source_workspace)
    manifest_path = roots.manifest_root / "fixture_migration.json"

    manifest = migrate_artifacts(targets, registry, source_workspace, roots, manifest_path)

    destination = roots.generated_root / "sources/realtime_pose_stationary5_v1/c04"
    assert not source.exists()
    assert destination.exists()
    assert manifest_path.exists()
    assert all(item["state"] == "verified_and_source_removed" for item in manifest["assets"])
    source_entry = next(item for item in manifest["assets"] if item["id"] == "source.fixture")
    assert source_entry["source_fingerprint"] == source_entry["destination_fingerprint"]
    assert source_entry["destination_schema_metadata"]
    assert Path(source_entry["source_file_inventory"]).exists()

    verified = verify_artifacts(registry, roots, records, manifest_path)
    assert {item["id"] for item in verified["assets"]} == {"raw.fixture", "source.fixture"}
    update_registry_fingerprints(registry_path, verified["assets"])
    updated = json.loads(registry_path.read_text(encoding="utf-8"))
    source_config = next(item for item in updated["assets"] if item["id"] == "source.fixture")
    assert source_config["status"] == "verified_active"
    assert source_config["expected_tree_sha256"] == source_entry["destination_fingerprint"]["tree_sha256"]


def test_relocation_hashes_and_moves_active_asset_to_short_path(tmp_path):
    registry_path = tmp_path / "artifact_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    {
                        "id": "source.fixture",
                        "kind": "source",
                        "root_key": "generated_root",
                        "relative_path": "c04/source",
                        "relocate_from_relative_path": "sources/realtime_pose_stationary5_v1/very_long_c04_source_name",
                        "retention": "active",
                        "status": "verified_active",
                        "schema_name": "realtime_pose_stationary5_v1",
                        "dependencies": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    roots = _roots(tmp_path)
    source = roots.generated_root / "sources/realtime_pose_stationary5_v1/very_long_c04_source_name"
    source.mkdir(parents=True)
    (source / "metadata.json").write_text(
        json.dumps({"schema_name": "realtime_pose_stationary5_v1"}),
        encoding="utf-8",
    )
    (source / "sample.bin").write_bytes(b"fixture")

    registry = load_artifact_registry(registry_path, project_root=tmp_path)
    records = resolve_records_with_dependencies(registry, ["source.fixture"])
    targets = build_relocation_targets(records, roots)
    manifest = relocate_artifacts(
        targets,
        registry,
        roots,
        roots.manifest_root / "fixture_relocation.json",
    )

    destination = roots.generated_root / "c04/source"
    assert not source.exists()
    assert destination.exists()
    assert manifest["assets"][0]["state"] == "verified_and_relocated"
    assert manifest["assets"][0]["source_fingerprint"] == manifest["assets"][0]["destination_fingerprint"]
    assert Path(manifest["assets"][0]["source_file_inventory"]).exists()


def test_recovery_removes_partially_deleted_source_when_destination_matches_manifest(tmp_path):
    roots = _roots(tmp_path)
    source = tmp_path / "legacy/globalpose"
    destination = roots.archive_root / "external/globalpose"
    source.mkdir(parents=True)
    destination.mkdir(parents=True)
    (source / "keep.txt").write_text("keep", encoding="utf-8")
    (source / "removed_before_retry.txt").write_text("removed", encoding="utf-8")
    (destination / "keep.txt").write_text("keep", encoding="utf-8")
    (destination / "removed_before_retry.txt").write_text("removed", encoding="utf-8")
    (source / "removed_before_retry.txt").unlink()

    record = ArtifactRecord(
        artifact_id="external.fixture",
        kind="external",
        root_key="archive_root",
        relative_path=Path("external/globalpose"),
        retention="archive",
        status="migration_pending",
        schema_name=None,
        dependencies=(),
    )
    registry = ArtifactRegistry(path=tmp_path / "registry.json", schema_version=1, records={record.artifact_id: record})
    target = MigrationTarget(record=record, source=source, destination=destination)
    entry = {
        "id": record.artifact_id,
        "state": "verifying_copy",
        "source_fingerprint": fingerprint_path(destination, hash_files=True),
        "source_schema_metadata": [],
    }
    manifest = {"assets": [entry]}
    manifest_path = tmp_path / "recovery.json"

    _finish_verified_destination(target, registry, roots, entry, manifest_path, manifest)

    assert not source.exists()
    assert entry["state"] == "verified_and_source_removed"
    assert entry["destination_fingerprint"] == entry["source_fingerprint"]


def test_mirror_body_rest_retains_unity_source_and_preserves_contract(tmp_path):
    roots = _roots(tmp_path)
    (roots.smpl_model_dir / "model.bin").parent.mkdir(parents=True)
    (roots.smpl_model_dir / "model.bin").write_bytes(b"model")
    source_workspace = tmp_path / "workspace"
    source_path = source_workspace / "unity/body_fbx_rest.json"
    _write_body_fbx_rest(source_path)
    registry_path = tmp_path / "artifact_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    {
                        "id": "raw.body_models", "kind": "body_model", "root_key": "smpl_model_dir",
                        "relative_path": ".", "retention": "active", "status": "verified_active",
                        "schema_name": None, "dependencies": []
                    },
                    {
                        "id": "runtime.body_fbx_rest", "kind": "body_fbx_rest_contract",
                        "root_key": "runtime_contract_root", "relative_path": "schema/body_fbx_rest.json",
                        "legacy_relative_path": "unity/body_fbx_rest.json", "retention": "active",
                        "status": "verified_mirrored", "schema_name": "realtime_pose_stationary5_v1",
                        "dependencies": ["raw.body_models"]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    registry = load_artifact_registry(registry_path, project_root=tmp_path)
    target = build_targets(
        resolve_records_with_dependencies(registry, ["runtime.body_fbx_rest"]),
        roots,
        source_workspace,
    )[-1]

    manifest = mirror_artifacts(
        [target],
        registry,
        source_workspace,
        roots,
        roots.manifest_root / "mirror.json",
    )

    destination = roots.runtime_contract_root / "schema/body_fbx_rest.json"
    assert source_path.exists()
    assert destination.exists()
    assert manifest["assets"][0]["state"] == "verified_and_source_retained"
    assert manifest["assets"][0]["source_fingerprint"] == manifest["assets"][0]["destination_fingerprint"]
    assert manifest["assets"][0]["destination_contract_validation"]["bone_count"] == 24
    assert manifest["assets"][0]["destination_contract_validation"]["tracker_joint_indices"] == [15, 20, 21, 0, 10, 11]


def test_predelete_writes_inventory_then_delete_revalidates_manifest(tmp_path):
    roots = _roots(tmp_path)
    workspace = tmp_path / "legacy_workspace"
    retired = workspace / "output/retired"
    retired.mkdir(parents=True)
    (retired / "result.txt").write_text("retired", encoding="utf-8")
    registry_path = tmp_path / "artifact_registry.json"
    registry_path.write_text(json.dumps({"schema_version": 1, "assets": []}), encoding="utf-8")
    registry = load_artifact_registry(registry_path, project_root=tmp_path)
    cleanup_config = tmp_path / "cleanup.json"
    cleanup_config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "required_preconditions": ["fixture review"],
                "candidates": [
                    {
                        "id": "retired", "root_key": "workspace_root", "relative_path": "output/retired",
                        "category": "fixture", "retirement_reason": "fixture", "dependencies": [],
                        "registry_asset_ids": []
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest_path = roots.manifest_root / "predelete.json"

    manifest = predelete_candidates(
        cleanup_config=cleanup_config,
        source_workspace=workspace,
        registry=registry,
        roots=roots,
        manifest_path=manifest_path,
    )

    entry = manifest["assets"][0]
    assert retired.exists()
    assert entry["state"] == "ready_for_human_review"
    assert Path(entry["file_inventory"]).exists()
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    delete_prepared_manifest(
        manifest_path=manifest_path,
        expected_manifest_sha256=digest,
        registry_path=registry_path,
        registry=registry,
        roots=roots,
        remove_cleanup_config=False,
        purge_deletion_audit=False,
    )
    assert not retired.exists()
    deleted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert deleted_manifest["assets"][0]["state"] == "deleted"


def test_delete_can_purge_its_predelete_audit_after_success(tmp_path):
    roots = _roots(tmp_path)
    workspace = tmp_path / "legacy_workspace"
    retired = workspace / "output/retired"
    retired.mkdir(parents=True)
    (retired / "result.txt").write_text("retired", encoding="utf-8")
    registry_path = tmp_path / "artifact_registry.json"
    registry_path.write_text(json.dumps({"schema_version": 1, "assets": []}), encoding="utf-8")
    registry = load_artifact_registry(registry_path, project_root=tmp_path)
    cleanup_config = tmp_path / "cleanup.json"
    cleanup_config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidates": [
                    {
                        "id": "retired",
                        "root_key": "workspace_root",
                        "relative_path": "output/retired",
                        "category": "fixture",
                        "retirement_reason": "fixture",
                        "dependencies": [],
                        "registry_asset_ids": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = roots.manifest_root / "predelete.json"
    manifest = predelete_candidates(
        cleanup_config=cleanup_config,
        source_workspace=workspace,
        registry=registry,
        roots=roots,
        manifest_path=manifest_path,
    )
    inventory_path = Path(manifest["assets"][0]["file_inventory"])
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    delete_prepared_manifest(
        manifest_path=manifest_path,
        expected_manifest_sha256=digest,
        registry_path=registry_path,
        registry=registry,
        roots=roots,
        remove_cleanup_config=True,
        purge_deletion_audit=True,
    )

    assert not retired.exists()
    assert not cleanup_config.exists()
    assert not inventory_path.exists()
    assert not manifest_path.exists()
    assert not inventory_path.parent.exists()


def test_predelete_rejects_active_assets_and_runtime_contract_root(tmp_path):
    roots = _roots(tmp_path)
    active_run = roots.runs_root / "c04"
    active_run.mkdir(parents=True)
    (active_run / "model000005000.pt").write_bytes(b"checkpoint")
    (roots.amass_root / "sequence.npz").parent.mkdir(parents=True)
    (roots.amass_root / "sequence.npz").write_bytes(b"amass")
    (roots.smpl_model_dir / "model.pkl").parent.mkdir(parents=True)
    (roots.smpl_model_dir / "model.pkl").write_bytes(b"body-model")
    for relative_path in ("c04/source", "c04/task", "c04/normalizer", "c04/longseq"):
        directory = roots.generated_root / relative_path
        directory.mkdir(parents=True)
        (directory / "metadata.json").write_text("{}", encoding="utf-8")
    runtime_contract = roots.runtime_contract_root / "schema/body_fbx_rest.json"
    _write_body_fbx_rest(runtime_contract)
    workspace = tmp_path
    registry_path = tmp_path / "artifact_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    {
                        "id": "raw.amass", "kind": "raw", "root_key": "amass_root",
                        "relative_path": ".", "retention": "active", "status": "verified_active",
                        "schema_name": None, "dependencies": []
                    },
                    {
                        "id": "raw.body_models", "kind": "body_model", "root_key": "smpl_model_dir",
                        "relative_path": ".", "retention": "active", "status": "verified_active",
                        "schema_name": None, "dependencies": []
                    },
                    {
                        "id": "source.c04", "kind": "source", "root_key": "generated_root",
                        "relative_path": "c04/source", "retention": "active", "status": "verified_active",
                        "schema_name": "realtime_pose_stationary5_v1", "dependencies": []
                    },
                    {
                        "id": "task.c04", "kind": "task", "root_key": "generated_root",
                        "relative_path": "c04/task", "retention": "active", "status": "verified_active",
                        "schema_name": "realtime_pose_stationary5_v1", "dependencies": ["source.c04"]
                    },
                    {
                        "id": "normalizer.c04", "kind": "normalizer", "root_key": "generated_root",
                        "relative_path": "c04/normalizer", "retention": "active", "status": "verified_active",
                        "schema_name": "realtime_pose_stationary5_v1", "dependencies": ["source.c04"]
                    },
                    {
                        "id": "longseq.c04", "kind": "longseq", "root_key": "generated_root",
                        "relative_path": "c04/longseq", "retention": "active", "status": "verified_active",
                        "schema_name": "realtime_pose_stationary5_v1", "dependencies": ["source.c04"]
                    },
                    {
                        "id": "checkpoint.c04.model", "kind": "checkpoint", "root_key": "runs_root",
                        "relative_path": "c04/model000005000.pt", "retention": "active", "status": "verified_active",
                        "schema_name": "realtime_pose_stationary5_v1", "dependencies": []
                    },
                    {
                        "id": "runtime.body_fbx_rest", "kind": "body_fbx_rest_contract",
                        "root_key": "runtime_contract_root", "relative_path": "schema/body_fbx_rest.json",
                        "retention": "active", "status": "verified_mirrored",
                        "schema_name": "realtime_pose_stationary5_v1", "dependencies": []
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    registry = load_artifact_registry(registry_path, project_root=tmp_path)
    cleanup_config = tmp_path / "cleanup.json"
    protected_candidates = {
        "amass": "artifact_store/active/raw/AMASS",
        "body-models": "artifact_store/active/raw/body_models",
        "source": "artifact_store/active/generated/c04/source",
        "task": "artifact_store/active/generated/c04/task",
        "normalizer": "artifact_store/active/generated/c04/normalizer",
        "longseq": "artifact_store/active/generated/c04/longseq",
        "checkpoint": "artifact_store/active/runs/c04/model000005000.pt",
    }
    for candidate_id, relative_path in protected_candidates.items():
        cleanup_config.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "candidates": [
                        {
                            "id": candidate_id,
                            "root_key": "workspace_root",
                            "relative_path": relative_path,
                            "category": "fixture",
                            "retirement_reason": "fixture",
                            "dependencies": [],
                            "registry_asset_ids": [],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        try:
            predelete_candidates(
                cleanup_config=cleanup_config,
                source_workspace=workspace,
                registry=registry,
                roots=roots,
                manifest_path=roots.manifest_root / f"blocked-{candidate_id}.json",
            )
        except ValueError as error:
            assert "active asset" in str(error)
        else:
            raise AssertionError(f"active {candidate_id} candidate must be rejected")

    cleanup_config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidates": [
                    {
                        "id": "body-rest", "root_key": "runtime_contract_root",
                        "relative_path": "schema/body_fbx_rest.json", "category": "fixture",
                        "retirement_reason": "fixture", "dependencies": [], "registry_asset_ids": []
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    try:
        predelete_candidates(
            cleanup_config=cleanup_config,
            source_workspace=workspace,
            registry=registry,
            roots=roots,
            manifest_path=roots.manifest_root / "blocked-runtime.json",
        )
    except ValueError as error:
        assert "unsupported root_key" in str(error)
    else:
        raise AssertionError("runtime contract candidate must be rejected")
