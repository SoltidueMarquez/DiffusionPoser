from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data_loaders.realtime_pose_dataset import (
    find_manifest_path,
    load_source_reference_task_marker,
    read_task_manifest,
    reject_materialized_entry,
    resolve_source_reference_path,
    validate_source_reference_entry,
)
from data_loaders.sensor_masking import (
    DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SCHEMA_NAMES,
    get_schema_spec,
)
from utils.normalizer import RealtimePoseNormalizer


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json_object(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是 object：{path}")
    return value


def validate_prepared_task(
    *,
    source_dir: Path,
    task_dir: Path,
    manifest_split: str,
    expected_data_split: str,
    schema_name: str,
    samples_per_source: int,
    rollout_steps: int,
    sampling_seed: int,
    mask_policy: str,
    patterns_per_source: int,
) -> dict[str, object]:
    """严格验证可复用的 source-reference task，并返回后续 SHA 绑定信息。"""

    source_dir = source_dir.resolve()
    task_dir = task_dir.resolve()
    source_manifest_path = source_dir / "manifest.jsonl"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"source manifest 不存在：{source_manifest_path}")
    source_manifest_sha256 = sha256_file(source_manifest_path)

    manifest_path = find_manifest_path(task_dir, split=str(manifest_split)).resolve()
    marker = load_source_reference_task_marker(manifest_path, schema_name=schema_name)
    marker_source_dir = Path(str(marker.get("source_dir", ""))).resolve()
    if marker_source_dir != source_dir:
        raise ValueError(f"task marker source_dir={marker_source_dir}，期望 {source_dir}")
    marker_source_manifest = Path(str(marker["source_manifest_path"])).resolve()
    if marker_source_manifest != source_manifest_path:
        raise ValueError(
            f"task marker 绑定的 source manifest={marker_source_manifest}，期望 {source_manifest_path}"
        )
    if str(marker["source_manifest_sha256"]) != source_manifest_sha256:
        raise ValueError("task marker 的 source manifest SHA256 与当前 source 不一致。")

    schema = get_schema_spec(schema_name)
    entries = read_task_manifest(manifest_path)
    if not entries:
        raise RuntimeError(f"task manifest 为空：{manifest_path}")
    task_ids: set[str] = set()
    for entry in entries:
        reject_materialized_entry(entry, source=str(manifest_path))
        validate_source_reference_entry(entry, schema=schema, required_rollout_steps=int(rollout_steps))
        task_id = str(entry["task_id"])
        if task_id in task_ids:
            raise ValueError(f"task manifest 包含重复 task_id：{task_id}")
        task_ids.add(task_id)
        if str(entry.get("split", "")) != str(expected_data_split):
            raise ValueError(
                f"任务 {task_id} split={entry.get('split')!r}，期望 {expected_data_split!r}"
            )
        if int(entry["samples_per_source"]) != int(samples_per_source):
            raise ValueError(
                f"任务 {task_id} samples_per_source={entry['samples_per_source']}，期望 {samples_per_source}"
            )
        if int(entry["max_rollout_steps"]) != int(rollout_steps):
            raise ValueError(
                f"任务 {task_id} max_rollout_steps={entry['max_rollout_steps']}，期望 {rollout_steps}"
            )
        if int(entry.get("sampling_seed", -1)) != int(sampling_seed):
            raise ValueError(
                f"任务 {task_id} sampling_seed={entry.get('sampling_seed')!r}，期望 {sampling_seed}"
            )
        if str(entry.get("mask_policy", "")) != str(mask_policy):
            raise ValueError(
                f"任务 {task_id} mask_policy={entry.get('mask_policy')!r}，期望 {mask_policy!r}"
            )
        if int(entry.get("patterns_per_source", -1)) != int(patterns_per_source):
            raise ValueError(
                "任务 "
                f"{task_id} patterns_per_source={entry.get('patterns_per_source')!r}，"
                f"期望 {patterns_per_source}"
            )
        resolve_source_reference_path(entry, marker)

    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": source_manifest_sha256,
        "entry_count": len(entries),
        "split": str(expected_data_split),
        "sampling_seed": int(sampling_seed),
        "mask_policy": str(mask_policy),
        "patterns_per_source": int(patterns_per_source),
    }


def validate_prepared_normalizer(
    *,
    normalizer_dir: Path,
    task_summary: dict[str, object],
    schema_name: str,
    windows_per_source: int,
    convergence_windows_per_source: int,
    tracker_mask_seed: int,
) -> dict[str, object]:
    """验证 normalizer 数值契约、采样规模、收敛报告及 task/source SHA 绑定。"""

    normalizer = RealtimePoseNormalizer(base_dir=normalizer_dir, schema_name=schema_name)
    resolved_dir = normalizer.base_dir.resolve()
    meta_path = resolved_dir / "normalizer_meta.json"
    convergence_path = resolved_dir / "normalizer_convergence.json"
    meta = read_json_object(meta_path)
    convergence = read_json_object(convergence_path)

    expected_values = {
        "schema_name": str(schema_name),
        "split": "train",
        "windows_per_source": int(windows_per_source),
        "convergence_windows_per_source": int(convergence_windows_per_source),
        "sampling_epoch": 0,
        "tracker_mask_seed": int(tracker_mask_seed),
        "task_manifest_sha256": str(task_summary["manifest_sha256"]),
        "source_manifest_sha256": str(task_summary["source_manifest_sha256"]),
    }
    for key, expected in expected_values.items():
        if meta.get(key) != expected:
            raise ValueError(f"normalizer {key}={meta.get(key)!r}，期望 {expected!r}")
    if meta.get("normalizer_convergence_passed") is not True:
        raise ValueError("normalizer_meta.json 未记录通过收敛门禁。")

    convergence_expected = {
        "schema_name": str(schema_name),
        "official_windows_per_source": int(windows_per_source),
        "comparison_windows_per_source": int(convergence_windows_per_source),
        "sampling_epoch": 0,
    }
    for key, expected in convergence_expected.items():
        if convergence.get(key) != expected:
            raise ValueError(f"normalizer convergence {key}={convergence.get(key)!r}，期望 {expected!r}")
    if convergence.get("passed") is not True or convergence.get("finite") is not True:
        raise ValueError("normalizer 收敛报告未通过或包含 NaN/Inf。")
    if convergence.get("failed_conditions") not in ([], None):
        raise ValueError("normalizer 收敛报告仍包含失败条件。")

    return {
        "normalizer_dir": str(resolved_dir),
        "windows_per_source": int(windows_per_source),
        "convergence_windows_per_source": int(convergence_windows_per_source),
        "convergence_passed": True,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="严格校验可复用的 realtime pose task/normalizer。")
    parser.add_argument("--source_dir", required=True, type=str)
    parser.add_argument("--task_dir", required=True, type=str)
    parser.add_argument("--manifest_split", required=True, type=str)
    parser.add_argument("--expected_data_split", required=True, type=str)
    parser.add_argument("--schema", default=DEFAULT_REALTIME_POSE_SCHEMA_NAME, choices=REALTIME_POSE_SCHEMA_NAMES)
    parser.add_argument("--samples_per_source", required=True, type=int)
    parser.add_argument("--rollout_steps", required=True, type=int)
    parser.add_argument("--sampling_seed", required=True, type=int)
    parser.add_argument("--mask_policy", required=True, type=str)
    parser.add_argument("--patterns_per_source", required=True, type=int)
    parser.add_argument("--normalizer_dir", default="", type=str)
    parser.add_argument("--windows_per_source", default=0, type=int)
    parser.add_argument("--convergence_windows_per_source", default=0, type=int)
    parser.add_argument("--tracker_mask_seed", default=-1, type=int)
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = build_argument_parser().parse_args(argv)
    task_summary = validate_prepared_task(
        source_dir=Path(args.source_dir),
        task_dir=Path(args.task_dir),
        manifest_split=str(args.manifest_split),
        expected_data_split=str(args.expected_data_split),
        schema_name=str(args.schema),
        samples_per_source=int(args.samples_per_source),
        rollout_steps=int(args.rollout_steps),
        sampling_seed=int(args.sampling_seed),
        mask_policy=str(args.mask_policy),
        patterns_per_source=int(args.patterns_per_source),
    )
    result: dict[str, object] = {"task": task_summary}
    if str(args.normalizer_dir).strip():
        if int(args.windows_per_source) <= 0 or int(args.convergence_windows_per_source) <= 0:
            raise ValueError("校验 normalizer 时必须提供正数 K/K2 窗口数量。")
        if int(args.tracker_mask_seed) < 0:
            raise ValueError("校验 normalizer 时必须提供非负 tracker_mask_seed。")
        result["normalizer"] = validate_prepared_normalizer(
            normalizer_dir=Path(args.normalizer_dir),
            task_summary=task_summary,
            schema_name=str(args.schema),
            windows_per_source=int(args.windows_per_source),
            convergence_windows_per_source=int(args.convergence_windows_per_source),
            tracker_mask_seed=int(args.tracker_mask_seed),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
