from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from utils.artifact_registry import ArtifactRegistry, load_artifact_registry
from utils.artifact_roots import ArtifactRoots, load_artifact_roots
from utils.filesystem_paths import ensure_directory, filesystem_path, path_exists
from utils.experiment_profile import (
    ExperimentProfile,
    load_experiment_profile,
    render_command_args,
    resolve_profile_artifacts,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a registered DiffusionPoser experiment profile.",
        allow_abbrev=False,
    )
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument("--artifact-roots-config", default=None, type=Path)
    parser.add_argument("--artifact-registry", default=None, type=Path)
    parser.add_argument("--stage", required=True, choices=("validate", "train", "evaluate", "summarize"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args, overrides = parser.parse_known_args(argv)
    overrides = _strip_override_separator(overrides)
    project_root = Path(__file__).resolve().parents[2]
    profile = load_experiment_profile(_resolve_from_project(args.experiment_config, project_root))
    roots = load_artifact_roots(
        config_path=_resolve_from_project(args.artifact_roots_config, project_root),
        project_root=project_root,
    )
    registry = load_artifact_registry(
        path=_resolve_from_project(args.artifact_registry, project_root),
        project_root=project_root,
    )
    artifact_paths = resolve_profile_artifacts(profile, registry, roots)
    validation = validate_profile(profile, registry, roots, artifact_paths)

    if args.stage == "validate":
        print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.stage == "summarize":
        output_dir = _summary_output_dir(artifact_paths)
        summary = build_summary(profile, registry, artifact_paths, stage="summarize", command_args=[])
        _write_run_records(output_dir, summary, stage="summarize")
        print(f"[run_profile] summary={output_dir / 'run_summary.md'}")
        return 0

    command = profile.command(args.stage)
    command_args = render_command_args(command, artifact_paths)
    command_args.extend(overrides)
    output_dir = _stage_output_dir(args.stage, command_args, artifact_paths)
    summary = build_summary(profile, registry, artifact_paths, stage=args.stage, command_args=command_args)
    summary["validation"] = validation
    summary["started_at"] = _now()

    invocation = [sys.executable, "-m", command.module, *command_args]
    print("[run_profile] " + subprocess.list2cmdline(invocation))
    if args.dry_run:
        summary["status"] = "dry_run"
        summary["planned_output_dir"] = str(output_dir)
        # dry-run 不代表一次训练，不能覆盖 run 根目录中最近一次真实训练的审计记录。
        print("[run_profile] dry-run: no run manifest was written")
        return 0

    completed = subprocess.run(invocation, cwd=project_root, check=False)
    if args.stage == "train":
        output_dir = _resolve_train_output_dir(command_args, artifact_paths)
        summary["training_output"] = _collect_training_output(output_dir)
    elif args.stage == "evaluate":
        summary["evaluation_summary"] = _read_json_if_exists(
            artifact_paths.get("evaluation_summary", Path(""))
        )
    summary["finished_at"] = _now()
    summary["exit_code"] = completed.returncode
    summary["status"] = "completed" if completed.returncode == 0 else "failed"
    _write_run_records(output_dir, summary, stage=args.stage)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    print(f"[run_profile] manifest={output_dir / 'run_manifest.json'}")
    return 0


def validate_profile(
    profile: ExperimentProfile,
    registry: ArtifactRegistry,
    roots: ArtifactRoots,
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "profile_id": profile.profile_id,
        "schema_name": profile.schema_name,
        "profile_sha256": profile.profile_sha256,
        "artifacts": {},
    }
    for name, path in artifact_paths.items():
        artifact_id = profile.artifacts[name]
        record = registry.get(artifact_id)
        exists = path_exists(path)
        writable = record.status in {"writable_root", "writable_file"}
        if not exists and not writable:
            raise FileNotFoundError(f"profile artifact is missing: {name} -> {path}")
        missing_dependencies = []
        for dependency in record.dependencies:
            dependency_path = registry.resolve(dependency, roots)
            dependency_record = registry.get(dependency)
            dependency_writable = dependency_record.status in {"writable_root", "writable_file"}
            if not path_exists(dependency_path) and not dependency_writable:
                missing_dependencies.append(f"{dependency} -> {dependency_path}")
        if missing_dependencies:
            raise FileNotFoundError(
                f"profile artifact dependencies are missing for {name}: {missing_dependencies}"
            )
        result["artifacts"][name] = {
            "id": artifact_id,
            "path": str(path),
            "exists": exists,
            "writable": writable,
            "schema_name": record.schema_name,
            "retention": record.retention,
            "expected_file_count": record.expected_file_count,
            "expected_size_bytes": record.expected_size_bytes,
            "expected_tree_sha256": record.expected_tree_sha256,
        }
    return result


def build_summary(
    profile: ExperimentProfile,
    registry: ArtifactRegistry,
    artifact_paths: dict[str, Path],
    *,
    stage: str,
    command_args: list[str],
) -> dict[str, Any]:
    comparison = profile.payload.get("comparison", {})
    result_record = _read_json_if_exists(profile.path.parents[2] / str(comparison.get("result_record", "")))
    evaluation_summary = _read_json_if_exists(artifact_paths.get("evaluation_summary", Path("")))
    return {
        "profile_id": profile.profile_id,
        "profile_path": str(profile.path),
        "profile_sha256": profile.profile_sha256,
        "profile_code": profile.payload.get("code", {}),
        "runtime_code_commit": _git_commit(profile.path.parents[2]),
        "baseline_status": profile.payload.get("baseline_status"),
        "schema_name": profile.schema_name,
        "stage": stage,
        "command_args": command_args,
        "artifacts": {
            name: _artifact_summary(registry, profile.artifacts[name], path)
            for name, path in artifact_paths.items()
        },
        "comparison": comparison,
        "acceptance": profile.payload.get("acceptance", {}),
        "recorded_result": result_record,
        "evaluation_summary": evaluation_summary,
        "retention": profile.payload.get("retention", {}),
    }


def _artifact_summary(registry: ArtifactRegistry, artifact_id: str, path: Path) -> dict[str, Any]:
    record = registry.get(artifact_id)
    return {
        "id": artifact_id,
        "path": str(path),
        "schema_name": record.schema_name,
        "dependencies": list(record.dependencies),
        "expected_file_count": record.expected_file_count,
        "expected_size_bytes": record.expected_size_bytes,
        "expected_tree_sha256": record.expected_tree_sha256,
    }


def _stage_output_dir(stage: str, command_args: list[str], artifact_paths: dict[str, Path]) -> Path:
    option = "--save_dir" if stage == "train" else "--output_dir"
    configured = _last_option_value(command_args, option)
    if configured:
        return Path(configured).resolve()
    return _summary_output_dir(artifact_paths)


def _resolve_train_output_dir(command_args: list[str], artifact_paths: dict[str, Path]) -> Path:
    run_root = _stage_output_dir("train", command_args, artifact_paths)
    pointer = run_root / "latest_run.json"
    payload = _read_json_if_exists(pointer)
    if isinstance(payload, dict) and payload.get("save_dir"):
        run_dir = Path(str(payload["save_dir"])).resolve()
        if path_exists(run_dir):
            return run_dir
    return run_root


def _summary_output_dir(artifact_paths: dict[str, Path]) -> Path:
    return artifact_paths.get("run_root", next(iter(artifact_paths.values()))).resolve()


def _write_run_records(output_dir: Path, summary: dict[str, Any], *, stage: str) -> None:
    ensure_directory(output_dir)
    manifest = {
        "kind": "diffusionposer_run_manifest",
        "created_at": _now(),
        "stage": stage,
        **summary,
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    _write_json(output_dir / "run_summary.json", summary)
    with open(filesystem_path(output_dir / "run_summary.md"), "w", encoding="utf-8", newline="\n") as file:
        file.write(_summary_markdown(summary))


def _collect_training_output(run_dir: Path) -> dict[str, Any]:
    """记录训练实际落盘的参数、checkpoint 与最后一次日志指标。

    train 入口会把解析后的参数写入 run 子目录；这里在子进程结束后读取它，避免只记录
    profile 的默认参数而遗漏 CLI 覆盖。checkpoint 使用内容 SHA-256，便于后续清理前复核。
    """
    result: dict[str, Any] = {
        "path": str(run_dir),
        "exists": path_exists(run_dir),
        "checkpoints": [],
        "last_metrics": None,
    }
    if not path_exists(run_dir) or not run_dir.is_dir():
        return result

    for args_name in ("args.json", "resume_args.json"):
        args_path = run_dir / args_name
        args_payload = _read_json_if_exists(args_path)
        if isinstance(args_payload, dict):
            result["training_arguments_path"] = str(args_path)
            result["training_arguments"] = args_payload
            break

    checkpoint_pattern = re.compile(r"^(model|ema|opt)(\d{9})\.pt$")
    checkpoints: list[dict[str, Any]] = []
    for checkpoint_path in sorted(run_dir.iterdir()):
        if not checkpoint_path.is_file():
            continue
        match = checkpoint_pattern.fullmatch(checkpoint_path.name)
        if match is None:
            continue
        checkpoints.append(
            {
                "kind": match.group(1),
                "step": int(match.group(2)),
                "path": str(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
                "sha256": _sha256_file(checkpoint_path),
            }
        )
    result["checkpoints"] = checkpoints
    result["last_metrics"] = _read_last_progress_metrics(run_dir / "progress.csv")
    return result


def _read_last_progress_metrics(progress_path: Path) -> dict[str, float | str | None] | None:
    """读取 progress.csv 的最后一行，保留每个 loss 项以便实验摘要可追溯。"""
    if not path_exists(progress_path):
        return None
    try:
        with open(filesystem_path(progress_path), "r", encoding="utf-8", newline="") as file:
            rows = csv.DictReader(file)
            last_row = None
            for row in rows:
                last_row = row
    except OSError:
        return None
    if last_row is None:
        return None
    return {key: _parse_metric_value(value) for key, value in last_row.items()}


def _parse_metric_value(value: str | None) -> float | str | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(filesystem_path(path), "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary_markdown(summary: dict[str, Any]) -> str:
    status = str(summary.get("status", "recorded"))
    lines = [
        f"# {summary['profile_id']}",
        "",
        f"- stage: {summary['stage']}",
        f"- status: {status}",
        f"- baseline_status: {summary.get('baseline_status')}",
        f"- runtime_code_commit: {summary.get('runtime_code_commit')}",
        f"- profile_sha256: {summary['profile_sha256']}",
        f"- schema_name: {summary['schema_name']}",
        "",
        "## Artifacts",
    ]
    for name, artifact in summary["artifacts"].items():
        lines.append(f"- {name}: {artifact['id']} -> {artifact['path']}")
    command_args = summary.get("command_args")
    if isinstance(command_args, list) and command_args:
        lines.extend(("", "## Invocation", "```text", subprocess.list2cmdline(command_args), "```"))
    training_output = summary.get("training_output")
    if isinstance(training_output, dict):
        lines.extend(("", "## Training Output", f"- path: {training_output.get('path')}") )
        arguments_path = training_output.get("training_arguments_path")
        if arguments_path:
            lines.append(f"- resolved_arguments: {arguments_path}")
        for checkpoint in training_output.get("checkpoints", []):
            if not isinstance(checkpoint, dict):
                continue
            lines.append(
                "- checkpoint: "
                f"{checkpoint.get('kind')} step={checkpoint.get('step')} "
                f"sha256={checkpoint.get('sha256')} path={checkpoint.get('path')}"
            )
        arguments = training_output.get("training_arguments")
        if isinstance(arguments, dict):
            lines.extend(("", "## Resolved Training Arguments", "```json"))
            lines.append(json.dumps(arguments, ensure_ascii=False, indent=2, sort_keys=True))
            lines.append("```")
        last_metrics = training_output.get("last_metrics")
        if isinstance(last_metrics, dict):
            lines.extend(("", "## Last Training Metrics", "```json"))
            lines.append(json.dumps(last_metrics, ensure_ascii=False, indent=2, sort_keys=True))
            lines.append("```")
    comparison = summary.get("comparison")
    if isinstance(comparison, dict) and comparison:
        lines.extend(("", "## Comparison", "```json"))
        lines.append(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))
        lines.append("```")
    acceptance = summary.get("acceptance")
    if isinstance(acceptance, dict) and acceptance:
        lines.extend(("", "## Acceptance", "```json"))
        lines.append(json.dumps(acceptance, ensure_ascii=False, indent=2, sort_keys=True))
        lines.append("```")
    recorded = summary.get("recorded_result")
    if isinstance(recorded, dict):
        lines.extend(("", "## Recorded Decision", f"- accepted: {recorded.get('accepted')}"))
        decision = recorded.get("decision")
        if isinstance(decision, dict) and decision.get("next_focus"):
            lines.append(f"- next_focus: {decision['next_focus']}")
    return "\n".join(lines) + "\n"


def _last_option_value(values: list[str], option_name: str) -> str | None:
    selected: str | None = None
    for index, value in enumerate(values):
        if value == option_name and index + 1 < len(values):
            selected = values[index + 1]
        elif value.startswith(f"{option_name}="):
            selected = value.split("=", 1)[1]
    return selected


def _read_json_if_exists(path: Path) -> Any:
    if not path or not path_exists(path) or not os.path.isfile(filesystem_path(path)):
        return None
    try:
        with open(filesystem_path(path), "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(filesystem_path(temporary), "w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(filesystem_path(temporary), filesystem_path(path))


def _git_commit(project_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _resolve_from_project(path: Path | None, project_root: Path) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return project_root / path


def _strip_override_separator(values: list[str]) -> list[str]:
    return values[1:] if values[:1] == ["--"] else values


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
