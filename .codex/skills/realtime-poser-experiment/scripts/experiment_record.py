from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import yaml


EXPERIMENT_ID_PATTERN = re.compile(r"^EXP-(\d{8})-(\d{3})$")
COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
EXPERIMENT_TYPES = {"code_change", "loss_only"}
STATUSES = {"draft", "ready", "running", "completed", "failed", "abandoned"}
PATH_KEYS = {
    "dataset",
    "task",
    "normalizer",
    "input_checkpoint",
    "run_dir",
    "log_dir",
    "output_checkpoint",
    "sample_output",
    "eval_output",
    "export_output",
    "unity_assets",
}


class RecordValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("\n".join(errors))
        self.errors = errors


def next_experiment_id(records_dir: Path, target_date: date | None = None) -> str:
    """按上海本地日期和现有文件名分配当天的下一个实验编号。"""
    current_date = target_date or datetime.now(ZoneInfo("Asia/Shanghai")).date()
    date_token = current_date.strftime("%Y%m%d")
    highest = 0
    if records_dir.exists():
        for path in records_dir.iterdir():
            if not path.is_file():
                continue
            match = EXPERIMENT_ID_PATTERN.fullmatch(path.stem)
            if match is None or match.group(1) != date_token:
                continue
            highest = max(highest, int(match.group(2)))
    if highest >= 999:
        raise ValueError(f"{date_token} 已存在 999 个实验编号，无法继续分配。")
    return f"EXP-{date_token}-{highest + 1:03d}"


def load_record(record_path: Path) -> dict[str, Any]:
    text = record_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise RecordValidationError(["实验记录必须以 YAML frontmatter 的 '---' 开始。"])
    try:
        closing_index = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration as exc:
        raise RecordValidationError(["实验记录缺少 YAML frontmatter 的结束 '---'。"])
    payload = yaml.safe_load("\n".join(lines[1:closing_index]))
    if not isinstance(payload, dict):
        raise RecordValidationError(["YAML frontmatter 必须解析为对象。"])
    return payload


def validate_record(
    record_path: Path,
    *,
    diffusionposer_root: Path | None = None,
    unity_root: Path | None = None,
    phase: str = "record",
) -> dict[str, Any]:
    record_path = record_path.resolve()
    payload = load_record(record_path)
    errors: list[str] = []
    warnings: list[str] = []

    experiment_id = _required_string(payload, "experiment_id", errors)
    if experiment_id and EXPERIMENT_ID_PATTERN.fullmatch(experiment_id) is None:
        errors.append("experiment_id 必须匹配 EXP-YYYYMMDD-NNN。")
    if experiment_id and record_path.name != f"{experiment_id}.md":
        errors.append(f"记录文件名必须是 {experiment_id}.md。")
    summary = _required_string(payload, "summary", errors)
    if summary and (len(summary) > 80 or "\n" in summary):
        errors.append("summary 必须是一句不超过 80 个字符的简洁说明。")
    _validate_record_text(record_path, experiment_id, summary, errors)
    if payload.get("schema_version") != 1:
        errors.append("schema_version 必须为 1。")
    if payload.get("experiment_type") not in EXPERIMENT_TYPES:
        errors.append("experiment_type 必须是 code_change 或 loss_only。")
    status = payload.get("status")
    if status not in STATUSES:
        errors.append(f"status 必须是 {sorted(STATUSES)} 之一。")
    created_at = _required_string(payload, "created_at", errors)
    if created_at:
        try:
            datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError:
            errors.append("created_at 必须是 ISO 8601 时间。")

    repositories = payload.get("repositories")
    if not isinstance(repositories, dict):
        repositories = {}
        errors.append("repositories 必须是对象。")
    dp_record = repositories.get("diffusionposer")
    unity_record = repositories.get("unity")
    if not isinstance(dp_record, dict):
        dp_record = {}
        errors.append("repositories.diffusionposer 必须是对象。")
    if not isinstance(unity_record, dict):
        unity_record = {}
        errors.append("repositories.unity 必须是对象。")

    dp_root = _resolve_diffusionposer_root(record_path, diffusionposer_root, errors)
    resolved_unity_root = _resolve_unity_root(dp_root, unity_root, unity_record, errors)
    _validate_recorded_root("diffusionposer", dp_record, dp_root, dp_root, errors)
    _validate_recorded_root("unity", unity_record, dp_root, resolved_unity_root, errors)
    dp_commits = _validate_repository_record(
        "diffusionposer", dp_record, dp_root, expected_participation={"primary"}, errors=errors
    )
    unity_commits = _validate_repository_record(
        "unity",
        unity_record,
        resolved_unity_root,
        expected_participation={"participating", "reference_only"},
        errors=errors,
    )
    if unity_record.get("participation") == "reference_only" and unity_record.get("changed") is not False:
        errors.append("reference_only Unity 必须设置 changed: false。")
    if dp_root is not None:
        expected_record = dp_root / "documents/experiments" / f"{experiment_id}.md"
        if record_path != expected_record.resolve():
            errors.append(f"实验记录必须位于 {expected_record}。")

    script = _required_string(payload, "script", errors)
    if experiment_id and script and script != f"scripts/experiments/{experiment_id}.ps1":
        errors.append(f"script 必须是 scripts/experiments/{experiment_id}.ps1。")
    if script and dp_root is not None:
        script_path = _safe_repo_path(dp_root, script, "script", errors)
        if script_path is not None and not script_path.is_file():
            errors.append(f"实验脚本不存在：{script_path}")
        elif script_path is not None:
            script_text = script_path.read_text(encoding="utf-8")
            if "{{" in script_text or "}}" in script_text:
                errors.append("实验脚本仍包含模板占位符。")
        experiment_commit = dp_commits.get("experiment")
        if experiment_commit and not _git_object_exists(dp_root, f"{experiment_commit}:{Path(script).as_posix()}"):
            errors.append("DiffusionPoser experiment_commit 中不包含实验脚本。")

    _validate_paths(payload.get("paths"), errors)
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        errors.append("runtime 必须是对象。")
    _required_string(runtime, "manifest", errors, prefix="runtime.")
    tests = payload.get("tests")
    if not isinstance(tests, list):
        errors.append("tests 必须是数组。")
    else:
        for index, test in enumerate(tests):
            if not isinstance(test, dict):
                errors.append(f"tests[{index}] 必须是对象。")
                continue
            _required_string(test, "command", errors, prefix=f"tests[{index}].")
            _required_string(test, "result", errors, prefix=f"tests[{index}].")
    result = payload.get("result")
    if not isinstance(result, dict):
        errors.append("result 必须是对象。")
    elif not isinstance(result.get("metrics"), dict):
        errors.append("result.metrics 必须是对象。")

    if phase == "pre-run":
        _validate_pre_run(
            record_path,
            experiment_id,
            status,
            dp_root,
            resolved_unity_root,
            dp_commits,
            unity_commits,
            unity_record,
            errors,
            warnings,
        )
    elif phase == "close":
        _validate_close(
            payload,
            experiment_id,
            status,
            dp_root,
            dp_commits,
            unity_commits,
            unity_record,
            errors,
        )
    elif phase != "record":
        errors.append("phase 必须是 record、pre-run 或 close。")

    if errors:
        raise RecordValidationError(errors)
    return {
        "ok": True,
        "experiment_id": experiment_id,
        "phase": phase,
        "diffusionposer_root": str(dp_root),
        "unity_root": str(resolved_unity_root),
        "warnings": warnings,
    }


def _resolve_diffusionposer_root(
    record_path: Path, configured_root: Path | None, errors: list[str]
) -> Path | None:
    candidate = configured_root.resolve() if configured_root is not None else _discover_git_root(record_path.parent)
    if candidate is None or not _is_git_repo(candidate):
        errors.append("无法解析 DiffusionPoser Git 仓库。")
        return None
    return candidate


def _resolve_unity_root(
    dp_root: Path | None,
    configured_root: Path | None,
    record: dict[str, Any],
    errors: list[str],
) -> Path | None:
    if configured_root is not None:
        candidate = configured_root.resolve()
    elif dp_root is not None and isinstance(record.get("root"), str) and record["root"].strip():
        candidate = (dp_root / record["root"]).resolve()
    else:
        candidate = None
    if candidate is None or not _is_git_repo(candidate):
        errors.append("无法解析 Unity Git 仓库。")
        return None
    return candidate


def _validate_repository_record(
    name: str,
    record: dict[str, Any],
    root: Path | None,
    *,
    expected_participation: set[str],
    errors: list[str],
) -> dict[str, str]:
    prefix = f"repositories.{name}."
    _required_string(record, "root", errors, prefix=prefix)
    remote = _required_string(record, "remote", errors, prefix=prefix)
    _required_string(record, "branch_at_snapshot", errors, prefix=prefix)
    participation = _required_string(record, "participation", errors, prefix=prefix)
    if participation and participation not in expected_participation:
        errors.append(f"{prefix}participation 必须是 {sorted(expected_participation)} 之一。")
    changed = record.get("changed")
    if not isinstance(changed, bool):
        errors.append(f"{prefix}changed 必须是布尔值。")
    if root is not None and remote:
        try:
            actual_remote = _git(root, "config", "--get", "remote.origin.url")
        except RuntimeError:
            actual_remote = ""
        if actual_remote != remote:
            errors.append(f"{prefix}remote 与仓库 origin 不一致。")

    result: dict[str, str] = {}
    for role in ("baseline", "experiment"):
        commit_key = f"{role}_commit"
        subject_key = f"{role}_subject"
        commit = _required_string(record, commit_key, errors, prefix=prefix)
        subject = _required_string(record, subject_key, errors, prefix=prefix)
        if commit and COMMIT_PATTERN.fullmatch(commit) is None:
            errors.append(f"{prefix}{commit_key} 必须是完整 40 位 SHA。")
            continue
        if commit:
            result[role] = commit.lower()
        if root is not None and commit and COMMIT_PATTERN.fullmatch(commit):
            if not _git_object_exists(root, f"{commit}^{{commit}}"):
                errors.append(f"{prefix}{commit_key} 不存在于对应仓库。")
            elif subject:
                actual_subject = _git(root, "show", "-s", "--format=%s", commit)
                if actual_subject != subject:
                    errors.append(f"{prefix}{subject_key} 与实际 commit subject 不一致。")

    baseline = result.get("baseline")
    experiment = result.get("experiment")
    if isinstance(changed, bool) and baseline and experiment:
        if changed and baseline == experiment:
            errors.append(f"{prefix}changed 为 true 时 baseline 与 experiment commit 不能相同。")
        if not changed and baseline != experiment:
            errors.append(f"{prefix}changed 为 false 时 baseline 与 experiment commit 必须相同。")
    return result


def _validate_paths(paths: Any, errors: list[str]) -> None:
    if not isinstance(paths, dict):
        errors.append("paths 必须是对象。")
        return
    missing = sorted(PATH_KEYS - set(paths))
    if missing:
        errors.append(f"paths 缺少字段：{missing}")
    for key in sorted(PATH_KEYS & set(paths)):
        entry = paths[key]
        if not isinstance(entry, dict):
            errors.append(f"paths.{key} 必须包含 path 和 note。")
            continue
        path_value = entry.get("path")
        note = entry.get("note")
        has_path = isinstance(path_value, str) and bool(path_value.strip())
        has_note = isinstance(note, str) and bool(note.strip())
        if not has_path and not has_note:
            errors.append(f"paths.{key} 未使用时必须在 note 中写明原因。")
        if key in {"run_dir", "log_dir"} and not has_path:
            errors.append(f"paths.{key}.path 不能为空。")


def _validate_pre_run(
    record_path: Path,
    experiment_id: str,
    status: Any,
    dp_root: Path | None,
    unity_root: Path | None,
    dp_commits: dict[str, str],
    unity_commits: dict[str, str],
    unity_record: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    if status != "ready":
        errors.append("pre-run 校验要求 status: ready。")
    if dp_root is not None and dp_commits.get("experiment"):
        if _git(dp_root, "rev-parse", "HEAD").lower() != dp_commits["experiment"]:
            errors.append("DiffusionPoser HEAD 与 experiment_commit 不一致。")
        try:
            expected_record = record_path.relative_to(dp_root).as_posix()
        except ValueError:
            expected_record = ""
        unexpected = [path for path in _dirty_paths(dp_root) if path != expected_record]
        if unexpected:
            errors.append(f"DiffusionPoser 运行前存在记录文件之外的未提交改动：{unexpected}")

    participation = unity_record.get("participation")
    if unity_root is not None and unity_commits.get("experiment"):
        current_head = _git(unity_root, "rev-parse", "HEAD").lower()
        if current_head != unity_commits["experiment"]:
            errors.append("Unity HEAD 与记录的 experiment_commit 不一致。")
        if participation == "participating":
            dirty = _dirty_paths(unity_root)
            if dirty:
                errors.append(f"参与型 Unity 运行前必须保持工作区干净：{dirty}")
        elif participation == "reference_only":
            dirty = _dirty_paths(unity_root)
            if dirty:
                warnings.append(f"Unity 为 reference_only，以下未提交改动不属于实验：{dirty}")


def _validate_close(
    payload: dict[str, Any],
    experiment_id: str,
    status: Any,
    dp_root: Path | None,
    dp_commits: dict[str, str],
    unity_commits: dict[str, str],
    unity_record: dict[str, Any],
    errors: list[str],
) -> None:
    if status not in {"completed", "failed", "abandoned"}:
        errors.append("close 校验要求 completed、failed 或 abandoned 状态。")
        return
    result = payload.get("result", {})
    if status == "completed" and not _nonempty_string(result.get("conclusion")):
        errors.append("completed 实验必须填写 result.conclusion。")
    if status == "failed" and not _nonempty_string(result.get("failure_reason")):
        errors.append("failed 实验必须填写 result.failure_reason。")
    if status == "abandoned":
        if not (_nonempty_string(result.get("conclusion")) or _nonempty_string(result.get("failure_reason"))):
            errors.append("abandoned 实验必须填写结论或放弃原因。")
        return

    runtime = payload.get("runtime", {})
    dp_runtime = runtime.get("diffusionposer_commit")
    unity_runtime = runtime.get("unity_commit")
    if dp_runtime != dp_commits.get("experiment"):
        errors.append("runtime.diffusionposer_commit 与实验 commit 不一致。")
    if unity_runtime != unity_commits.get("experiment"):
        errors.append("runtime.unity_commit 与记录的 Unity experiment_commit 不一致。")
    if not _nonempty_string(runtime.get("command")):
        errors.append("completed/failed 实验必须填写 runtime.command。")

    manifest_value = runtime.get("manifest")
    if dp_root is None or not _nonempty_string(manifest_value):
        return
    manifest_path = _safe_repo_path(dp_root, manifest_value, "runtime.manifest", errors)
    if manifest_path is None or not manifest_path.is_file():
        errors.append(f"runtime manifest 不存在：{manifest_path}")
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"runtime manifest 无法读取：{exc}")
        return
    if manifest.get("experiment_id") != experiment_id:
        errors.append("runtime manifest 的 experiment_id 不一致。")
    if manifest.get("diffusionposer_commit") != dp_commits.get("experiment"):
        errors.append("runtime manifest 的 DiffusionPoser commit 不一致。")
    if manifest.get("unity_commit") != unity_commits.get("experiment"):
        errors.append("runtime manifest 的 Unity commit 不一致。")
    if manifest.get("status") != status:
        errors.append("runtime manifest 的 status 与实验记录不一致。")


def _validate_record_text(
    record_path: Path, experiment_id: str, summary: str, errors: list[str]
) -> None:
    text = record_path.read_text(encoding="utf-8")
    if "{{" in text or "}}" in text:
        errors.append("实验记录仍包含模板占位符。")
    lines = text.splitlines()
    closing_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_index = index
            break
    if closing_index is None:
        return
    title = next((line.strip() for line in lines[closing_index + 1 :] if line.strip()), "")
    expected_title = f"# {experiment_id}｜{summary}"
    if experiment_id and summary and title != expected_title:
        errors.append(f"实验记录标题必须是：{expected_title}")


def _required_string(
    payload: dict[str, Any], key: str, errors: list[str], *, prefix: str = ""
) -> str:
    value = payload.get(key)
    if not _nonempty_string(value):
        errors.append(f"{prefix}{key} 必须是非空字符串。")
        return ""
    return str(value).strip()


def _validate_recorded_root(
    name: str,
    record: dict[str, Any],
    base_root: Path | None,
    actual_root: Path | None,
    errors: list[str],
) -> None:
    value = record.get("root")
    if base_root is None or actual_root is None or not _nonempty_string(value):
        return
    raw_path = Path(str(value))
    recorded_root = raw_path.resolve() if raw_path.is_absolute() else (base_root / raw_path).resolve()
    if recorded_root != actual_root:
        errors.append(f"repositories.{name}.root 与实际仓库路径不一致。")


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _discover_git_root(start: Path) -> Path | None:
    completed = subprocess.run(
        ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        return None
    return Path(completed.stdout.strip()).resolve()


def _is_git_repo(root: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"git {' '.join(args)} 执行失败。")
    return completed.stdout.strip()


def _git_object_exists(root: Path | None, object_name: str) -> bool:
    if root is None:
        return False
    completed = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", object_name],
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def _dirty_paths(root: Path) -> list[str]:
    output = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    paths: list[str] = []
    for line in output.splitlines():
        if len(line) < 4:
            continue
        value = line[3:]
        if " -> " in value:
            value = value.split(" -> ", 1)[1]
        paths.append(value.strip('"').replace("\\", "/"))
    return paths


def _safe_repo_path(root: Path, value: str, field: str, errors: list[str]) -> Path | None:
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        errors.append(f"{field} 必须位于 DiffusionPoser 仓库内。")
        return None
    return candidate


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    normalized = value.replace("-", "")
    try:
        return datetime.strptime(normalized, "%Y%m%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("日期必须是 YYYY-MM-DD 或 YYYYMMDD。") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Allocate and validate realtime poser experiment records.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    next_id = subparsers.add_parser("next-id", help="Print the next EXP-YYYYMMDD-NNN identifier.")
    next_id.add_argument("--records-dir", default=Path("documents/experiments"), type=Path)
    next_id.add_argument("--date", default=None, type=_parse_date)

    validate = subparsers.add_parser("validate", help="Validate an experiment record and repo state.")
    validate.add_argument("--record", required=True, type=Path)
    validate.add_argument("--diffusionposer-root", default=None, type=Path)
    validate.add_argument("--unity-root", default=None, type=Path)
    validate.add_argument("--phase", choices=("record", "pre-run", "close"), default="record")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "next-id":
        print(next_experiment_id(args.records_dir, args.date))
        return 0
    try:
        result = validate_record(
            args.record,
            diffusionposer_root=args.diffusionposer_root,
            unity_root=args.unity_root,
            phase=args.phase,
        )
    except RecordValidationError as exc:
        print(json.dumps({"ok": False, "errors": exc.errors}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
