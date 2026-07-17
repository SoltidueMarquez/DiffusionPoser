from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.experiments.run_profile import (
    _collect_training_output,
    _summary_markdown,
    _write_run_records,
    build_summary,
    main,
    validate_profile,
)
from utils.artifact_registry import load_artifact_registry
from utils.artifact_roots import ArtifactRoots
from utils.experiment_profile import load_experiment_profile, render_command_args, resolve_profile_artifacts


def _roots(tmp_path: Path) -> ArtifactRoots:
    store = tmp_path / "store"
    return ArtifactRoots(
        workspace_root=tmp_path,
        amass_root=store / "raw/AMASS",
        smpl_model_dir=store / "raw/body_models",
        generated_root=store / "generated",
        runtime_contract_root=store / "runtime_contracts",
        runs_root=store / "runs",
        outputs_root=store / "output",
        external_root=store / "external",
        archive_root=store / "archive",
        manifest_root=store / "manifests",
    )


def test_profile_resolves_registered_paths_and_writes_run_summary(tmp_path):
    roots = _roots(tmp_path)
    source = roots.generated_root / "sources/c04"
    run_root = roots.runs_root / "c04"
    summary = roots.outputs_root / "c04/longseq_eval_summary.json"
    source.mkdir(parents=True)
    run_root.mkdir(parents=True)
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"metric": 1.0}), encoding="utf-8")

    registry_path = tmp_path / "configs/artifact_registry.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    {
                        "id": "source", "kind": "source", "root_key": "generated_root",
                        "relative_path": "sources/c04", "retention": "active", "status": "verified_active",
                        "schema_name": "realtime_pose_stationary5_v1", "dependencies": []
                    },
                    {
                        "id": "run", "kind": "run", "root_key": "runs_root",
                        "relative_path": "c04", "retention": "active", "status": "verified_active",
                        "schema_name": "realtime_pose_stationary5_v1", "dependencies": ["source"]
                    },
                    {
                        "id": "summary", "kind": "summary", "root_key": "outputs_root",
                        "relative_path": "c04/longseq_eval_summary.json", "retention": "active", "status": "verified_active",
                        "schema_name": "realtime_pose_stationary5_v1", "dependencies": ["run"]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    profile_path = tmp_path / "configs/experiments/c04.json"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "c04-fixture",
                "schema_name": "realtime_pose_stationary5_v1",
                "baseline_status": "training_seed_unaccepted_stationary",
                "artifacts": {"source": "source", "run_root": "run", "evaluation_summary": "summary"},
                "commands": {
                    "train": {"module": "example.module", "args": ["--data", "${artifact:source}", "--parent", "${artifact_parent:source}"]}
                }
            }
        ),
        encoding="utf-8",
    )
    registry = load_artifact_registry(registry_path, project_root=tmp_path)
    profile = load_experiment_profile(profile_path)
    artifact_paths = resolve_profile_artifacts(profile, registry, roots)

    assert render_command_args(profile.command("train"), artifact_paths) == [
        "--data", str(source), "--parent", str(source.parent)
    ]
    validation = validate_profile(profile, registry, roots, artifact_paths)
    assert validation["artifacts"]["source"]["id"] == "source"

    result = build_summary(profile, registry, artifact_paths, stage="train", command_args=["--data", str(source)])
    result["status"] = "dry_run"
    _write_run_records(run_root, result, stage="train")
    assert (run_root / "run_manifest.json").exists()
    assert (run_root / "run_summary.md").exists()


def test_c04_profile_keeps_the_baseline_as_an_unaccepted_training_seed():
    project_root = Path(__file__).resolve().parents[3]
    profile = load_experiment_profile(
        project_root / "configs/experiments/c04-loss-v3-stable-rollout.json"
    )
    registry = load_artifact_registry(project_root=project_root)

    assert profile.payload["baseline_status"] == "training_seed_unaccepted_stationary"
    assert profile.payload["code"]["tag"] == "baseline/c04-loss-v3-stable-rollout"
    assert profile.artifacts["init_checkpoint"] == "checkpoint.c04.model"
    assert profile.artifacts["runtime_contract"] == "runtime.body_fbx_rest"
    assert profile.artifacts["evaluation_ema_checkpoint"] == "checkpoint.c04.ema"
    assert profile.artifacts["run_root"] == "run.c04.followup"
    assert registry.get("run.c04").status == "verified_active"
    assert registry.get("run.c04.followup").status == "writable_root"


def test_training_output_records_resolved_args_checkpoints_and_last_metrics(tmp_path):
    run_dir = tmp_path / "runs/c04/20260717_warm_start"
    run_dir.mkdir(parents=True)
    (run_dir / "args.json").write_text(
        json.dumps({"cuda": True, "num_steps": 1, "save_dir": str(run_dir)}),
        encoding="utf-8",
    )
    (run_dir / "model000000001.pt").write_bytes(b"model")
    (run_dir / "ema000000001.pt").write_bytes(b"ema")
    (run_dir / "opt000000001.pt").write_bytes(b"optimizer")
    (run_dir / "progress.csv").write_text(
        "step,loss,tracker_relative_pos_loss\n1,0.25,0.125\n",
        encoding="utf-8",
    )

    output = _collect_training_output(run_dir)

    assert output["training_arguments"]["cuda"] is True
    assert output["training_arguments"]["num_steps"] == 1
    assert [checkpoint["kind"] for checkpoint in output["checkpoints"]] == ["ema", "model", "opt"]
    assert all(len(checkpoint["sha256"]) == 64 for checkpoint in output["checkpoints"])
    assert output["last_metrics"] == {
        "step": 1.0,
        "loss": 0.25,
        "tracker_relative_pos_loss": 0.125,
    }
    markdown = _summary_markdown(
        {
            "profile_id": "c04-fixture",
            "stage": "train",
            "status": "completed",
            "baseline_status": "training_seed_unaccepted_stationary",
            "runtime_code_commit": "e8d93ed",
            "profile_sha256": "profile-sha",
            "schema_name": "realtime_pose_stationary5_v1",
            "artifacts": {},
            "command_args": ["--num_steps", "1"],
            "training_output": output,
            "comparison": {"reference": "C00"},
            "acceptance": {"stationary_f1_at_0_7_gain_ge": 0.1},
            "recorded_result": {"accepted": False},
        }
    )

    assert "## Resolved Training Arguments" in markdown
    assert "model000000001.pt" in markdown
    assert "## Last Training Metrics" in markdown
    assert "## Comparison" in markdown
    assert "## Acceptance" in markdown


def test_dry_run_does_not_overwrite_latest_real_run_record(tmp_path):
    roots = _roots(tmp_path)
    source = roots.generated_root / "c04/source"
    source.mkdir(parents=True)
    run_root = roots.runs_root / "c04"
    run_root.mkdir(parents=True)
    manifest_path = run_root / "run_manifest.json"
    manifest_path.write_text('{"status":"completed"}\n', encoding="utf-8")

    roots_config = tmp_path / "configs/artifact_roots.json"
    roots_config.parent.mkdir(parents=True)
    roots_config.write_text(
        json.dumps(
            {
                "workspace_root": str(roots.workspace_root),
                "amass_root": str(roots.amass_root),
                "smpl_model_dir": str(roots.smpl_model_dir),
                "generated_root": str(roots.generated_root),
                "runtime_contract_root": str(roots.runtime_contract_root),
                "runs_root": str(roots.runs_root),
                "outputs_root": str(roots.outputs_root),
                "external_root": str(roots.external_root),
                "archive_root": str(roots.archive_root),
                "manifest_root": str(roots.manifest_root),
            }
        ),
        encoding="utf-8",
    )
    registry_path = tmp_path / "configs/artifact_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    {
                        "id": "source", "kind": "source", "root_key": "generated_root",
                        "relative_path": "c04/source", "retention": "active", "status": "verified_active",
                        "schema_name": "realtime_pose_stationary5_v1", "dependencies": []
                    },
                    {
                        "id": "run", "kind": "run", "root_key": "runs_root",
                        "relative_path": "c04", "retention": "active", "status": "writable_root",
                        "schema_name": "realtime_pose_stationary5_v1", "dependencies": ["source"]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    profile_path = tmp_path / "configs/experiments/dry-run.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "dry-run-fixture",
                "schema_name": "realtime_pose_stationary5_v1",
                "baseline_status": "training_seed_unaccepted_stationary",
                "artifacts": {"source": "source", "run_root": "run"},
                "commands": {
                    "train": {
                        "module": "nonexistent.module",
                        "args": ["--data_dir", "${artifact:source}", "--save_dir", "${artifact:run_root}"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert main([
        "--experiment-config", str(profile_path),
        "--artifact-roots-config", str(roots_config),
        "--artifact-registry", str(registry_path),
        "--stage", "train",
        "--dry-run",
    ]) == 0
    assert manifest_path.read_text(encoding="utf-8") == '{"status":"completed"}\n'
