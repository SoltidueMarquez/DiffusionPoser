from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SKILL_SCRIPT = (
    PROJECT_ROOT
    / ".codex/skills/realtime-poser-experiment/scripts/experiment_record.py"
)
SPEC = importlib.util.spec_from_file_location("realtime_poser_experiment_record", SKILL_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return completed.stdout.strip()


def _init_repo(path: Path, name: str) -> tuple[str, str]:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-b", "main", str(path)], check=True, capture_output=True)
    _git(path, "config", "user.name", "Experiment Test")
    _git(path, "config", "user.email", "experiment-test@example.com")
    _git(path, "remote", "add", "origin", f"https://example.invalid/{name}.git")
    (path / "baseline.txt").write_text(f"{name} baseline\n", encoding="utf-8")
    _git(path, "add", "baseline.txt")
    _git(path, "commit", "-m", f"{name} baseline")
    return _git(path, "rev-parse", "HEAD"), _git(path, "show", "-s", "--format=%s", "HEAD")


def _path_record(path: str | None, note: str) -> dict[str, str | None]:
    return {"path": path, "note": note}


def _record_payload(
    experiment_id: str,
    *,
    dp_baseline: str,
    dp_experiment: str,
    dp_baseline_subject: str,
    dp_experiment_subject: str,
    unity_commit: str,
    unity_subject: str,
    unity_participation: str = "reference_only",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "summary": "验证双仓库实验记录",
        "experiment_type": "code_change",
        "status": "ready",
        "created_at": "2026-07-20T12:00:00+08:00",
        "script": f"scripts/experiments/{experiment_id}.ps1",
        "repositories": {
            "diffusionposer": {
                "root": ".",
                "remote": "https://example.invalid/diffusionposer.git",
                "branch_at_snapshot": "main",
                "participation": "primary",
                "changed": True,
                "baseline_commit": dp_baseline,
                "baseline_subject": dp_baseline_subject,
                "experiment_commit": dp_experiment,
                "experiment_subject": dp_experiment_subject,
            },
            "unity": {
                "root": "../unity",
                "remote": "https://example.invalid/unity.git",
                "branch_at_snapshot": "main",
                "participation": unity_participation,
                "changed": False,
                "baseline_commit": unity_commit,
                "baseline_subject": unity_subject,
                "experiment_commit": unity_commit,
                "experiment_subject": unity_subject,
            },
        },
        "paths": {
            "dataset": _path_record("dataset/example", "实验输入。"),
            "task": _path_record(None, "本测试不使用 task。"),
            "normalizer": _path_record(None, "本测试不使用 normalizer。"),
            "input_checkpoint": _path_record(None, "本测试不使用 checkpoint。"),
            "run_dir": _path_record(f"runs/{experiment_id}", "运行目录。"),
            "log_dir": _path_record(f"runs/{experiment_id}/logs", "日志目录。"),
            "output_checkpoint": _path_record(None, "本测试不生成 checkpoint。"),
            "sample_output": _path_record(None, "本测试不采样。"),
            "eval_output": _path_record(None, "本测试不评估。"),
            "export_output": _path_record(None, "本测试不导出。"),
            "unity_assets": _path_record(None, "Unity 为 reference_only。"),
        },
        "runtime": {
            "manifest": f"runs/{experiment_id}/experiment_runtime.json",
            "diffusionposer_commit": None,
            "unity_commit": None,
            "command": None,
        },
        "tests": [],
        "result": {"metrics": {}, "conclusion": None, "failure_reason": None},
    }


def _write_record(repo: Path, payload: dict[str, Any]) -> Path:
    record_path = repo / "documents/experiments" / f"{payload['experiment_id']}.md"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False).rstrip()
    record_path.write_text(
        f"---\n{frontmatter}\n---\n\n# {payload['experiment_id']}｜{payload['summary']}\n",
        encoding="utf-8",
    )
    return record_path


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    dp_repo = tmp_path / "diffusionposer"
    unity_repo = tmp_path / "unity"
    dp_baseline, dp_baseline_subject = _init_repo(dp_repo, "diffusionposer")
    unity_commit, unity_subject = _init_repo(unity_repo, "unity")

    experiment_id = "EXP-20260720-001"
    script_path = dp_repo / "scripts/experiments" / f"{experiment_id}.ps1"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("Write-Host 'fixture'\n", encoding="utf-8")
    _git(dp_repo, "add", script_path.relative_to(dp_repo).as_posix())
    _git(dp_repo, "commit", "-m", f"experiment({experiment_id}): fixture")
    dp_experiment = _git(dp_repo, "rev-parse", "HEAD")
    dp_experiment_subject = _git(dp_repo, "show", "-s", "--format=%s", "HEAD")

    payload = _record_payload(
        experiment_id,
        dp_baseline=dp_baseline,
        dp_experiment=dp_experiment,
        dp_baseline_subject=dp_baseline_subject,
        dp_experiment_subject=dp_experiment_subject,
        unity_commit=unity_commit,
        unity_subject=unity_subject,
    )
    record_path = _write_record(dp_repo, payload)
    return dp_repo, unity_repo, record_path, payload


def test_next_id_uses_date_and_highest_existing_sequence(tmp_path: Path):
    records = tmp_path / "documents/experiments"
    records.mkdir(parents=True)
    (records / "EXP-20260720-001.md").write_text("", encoding="utf-8")
    (records / "EXP-20260720-003.md").write_text("", encoding="utf-8")
    (records / "legacy-name.md").write_text("", encoding="utf-8")
    (records / "EXP-20260719-099.md").write_text("", encoding="utf-8")

    assert MODULE.next_experiment_id(records, date(2026, 7, 20)) == "EXP-20260720-004"


def test_markdown_template_has_parseable_yaml_frontmatter():
    template_path = (
        PROJECT_ROOT
        / ".codex/skills/realtime-poser-experiment/assets/experiment-record.md"
    )
    lines = template_path.read_text(encoding="utf-8").splitlines()
    closing_index = next(index for index, line in enumerate(lines[1:], start=1) if line == "---")

    payload = yaml.safe_load("\n".join(lines[1:closing_index]))

    assert payload["schema_version"] == 1
    assert payload["experiment_id"] == "{{EXPERIMENT_ID}}"
    assert payload["repositories"]["unity"]["changed"] is False


def test_pre_run_allows_dirty_reference_only_unity_with_warning(tmp_path: Path):
    dp_repo, unity_repo, record_path, _ = _fixture(tmp_path)
    (unity_repo / "unrelated-local-change.txt").write_text("not part of experiment\n", encoding="utf-8")

    result = MODULE.validate_record(
        record_path,
        diffusionposer_root=dp_repo,
        unity_root=unity_repo,
        phase="pre-run",
    )

    assert result["ok"] is True
    assert result["warnings"]
    assert "reference_only" in result["warnings"][0]


def test_pre_run_rejects_dirty_participating_unity(tmp_path: Path):
    dp_repo, unity_repo, record_path, payload = _fixture(tmp_path)
    payload["repositories"]["unity"]["participation"] = "participating"
    _write_record(dp_repo, payload)
    (unity_repo / "runtime-change.cs").write_text("// dirty\n", encoding="utf-8")

    with pytest.raises(MODULE.RecordValidationError, match="参与型 Unity 运行前必须保持工作区干净"):
        MODULE.validate_record(
            record_path,
            diffusionposer_root=dp_repo,
            unity_root=unity_repo,
            phase="pre-run",
        )


def test_pre_run_accepts_clean_participating_unity_commit(tmp_path: Path):
    dp_repo, unity_repo, record_path, payload = _fixture(tmp_path)
    unity_baseline = payload["repositories"]["unity"]["baseline_commit"]
    (unity_repo / "runtime-change.cs").write_text("// experiment change\n", encoding="utf-8")
    _git(unity_repo, "add", "runtime-change.cs")
    _git(unity_repo, "commit", "-m", f"experiment({payload['experiment_id']}): Unity fixture")
    unity_experiment = _git(unity_repo, "rev-parse", "HEAD")
    unity_subject = _git(unity_repo, "show", "-s", "--format=%s", "HEAD")
    unity_record = payload["repositories"]["unity"]
    unity_record.update(
        {
            "participation": "participating",
            "changed": True,
            "baseline_commit": unity_baseline,
            "experiment_commit": unity_experiment,
            "experiment_subject": unity_subject,
        }
    )
    _write_record(dp_repo, payload)

    result = MODULE.validate_record(
        record_path,
        diffusionposer_root=dp_repo,
        unity_root=unity_repo,
        phase="pre-run",
    )

    assert result["ok"] is True
    assert result["warnings"] == []


def test_validate_rejects_script_missing_from_worktree(tmp_path: Path):
    dp_repo, unity_repo, record_path, payload = _fixture(tmp_path)
    (dp_repo / payload["script"]).unlink()

    with pytest.raises(MODULE.RecordValidationError, match="实验脚本不存在"):
        MODULE.validate_record(
            record_path,
            diffusionposer_root=dp_repo,
            unity_root=unity_repo,
            phase="record",
        )


def test_validate_rejects_unknown_full_commit_sha(tmp_path: Path):
    dp_repo, unity_repo, record_path, payload = _fixture(tmp_path)
    payload["repositories"]["diffusionposer"]["experiment_commit"] = "0" * 40
    _write_record(dp_repo, payload)

    with pytest.raises(MODULE.RecordValidationError, match="不存在于对应仓库"):
        MODULE.validate_record(
            record_path,
            diffusionposer_root=dp_repo,
            unity_root=unity_repo,
            phase="record",
        )


def test_close_rejects_runtime_manifest_commit_mismatch(tmp_path: Path):
    dp_repo, unity_repo, record_path, payload = _fixture(tmp_path)
    experiment_commit = payload["repositories"]["diffusionposer"]["experiment_commit"]
    payload["status"] = "completed"
    payload["runtime"]["diffusionposer_commit"] = experiment_commit
    payload["runtime"]["unity_commit"] = payload["repositories"]["unity"]["experiment_commit"]
    payload["runtime"]["command"] = "conda run --no-capture-output -n diffusionposer5070 python -m fixture"
    payload["result"]["conclusion"] = "测试完成。"
    manifest_path = dp_repo / payload["runtime"]["manifest"]
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "experiment_id": payload["experiment_id"],
                "status": "completed",
                "diffusionposer_commit": "f" * 40,
                "unity_commit": payload["repositories"]["unity"]["experiment_commit"],
            }
        ),
        encoding="utf-8",
    )
    _write_record(dp_repo, payload)

    with pytest.raises(MODULE.RecordValidationError, match="runtime manifest 的 DiffusionPoser commit 不一致"):
        MODULE.validate_record(
            record_path,
            diffusionposer_root=dp_repo,
            unity_root=unity_repo,
            phase="close",
        )
