from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from data_loaders.sensor_masking import (
    DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    MIN_VALID_TRACKERS,
    REALTIME_POSE_SCHEMA_NAMES,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_LENGTH,
    REALTIME_POSE_TARGET_START,
    TASK_MASK_POLICIES,
    TASK_MASK_POLICY_FIXED_PATTERNS,
    TASK_MASK_POLICY_FULL,
    TASK_MODE_REALTIME_POSE,
    TRACKER_PATTERN_CATEGORIES,
    create_realtime_inpaint_mask,
    get_schema_spec,
    make_tracker_pattern,
    make_window_patterns,
    normalize_tracker_pattern_categories,
    repeat_pattern_sensor_valid,
    validate_sensor_valid,
)


SOURCE_KEYS = {
    "body_pose_parent_6d",
    "root_pos_world",
    "root_yaw",
    "root_yaw_delta_sincos",
    "tracker_pos_world",
    "tracker_rot_world_6d",
    "joints_world",
    "joint_offsets_parent",
}
TASK_OUTPUT_MARKER = ".realtime_pose_tasks.json"
USABLE_SOURCE_STATUSES = {"converted", "skipped_existing", "reused_source", "upgraded_existing_source"}
SHORT_SOURCE_POLICY_SKIP = "skip"
SHORT_SOURCE_POLICY_ERROR = "error"
SHORT_SOURCE_POLICIES = (SHORT_SOURCE_POLICY_SKIP, SHORT_SOURCE_POLICY_ERROR)
SHORT_SOURCE_REPORT_NAME = "skipped_short_sources.jsonl"


@dataclass(frozen=True)
class SplitTaskPlan:
    split: str
    entries: list[dict]
    seed: int


@dataclass(frozen=True)
class TaskGenerationPlan:
    source_dir: Path
    output_dir: Path
    split_dir: Path | None
    split_plans: list[SplitTaskPlan]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate realtime_pose materialized tasks.")
    paths = parser.add_argument_group("paths")
    paths.add_argument("--source_dir", default="dataset/AMASS_realtime_pose_v2_60hz", type=str)
    paths.add_argument("--output_dir", default="dataset/AMASS_realtime_pose_v2_60hz_tasks", type=str)
    paths.add_argument("--split_dir", default="data_loaders/splits", type=str)

    task = parser.add_argument_group("task")
    task.add_argument("--splits", nargs="+", default=["train"], type=str)
    task.add_argument("--seq_len", default=REALTIME_POSE_SEQ_LEN, type=int)
    task.add_argument("--schema", default=DEFAULT_REALTIME_POSE_SCHEMA_NAME, choices=REALTIME_POSE_SCHEMA_NAMES, type=str)
    task.add_argument("--samples_per_file", default=4, type=int)
    task.add_argument("--mask_policy", default=TASK_MASK_POLICY_FULL, choices=TASK_MASK_POLICIES, type=str)
    task.add_argument("--fixed_tracker_patterns", nargs="+", default=["all"], type=str)
    task.add_argument("--patterns_per_window", default=len(TRACKER_PATTERN_CATEGORIES), type=int)
    task.add_argument("--min_valid_trackers", default=MIN_VALID_TRACKERS, type=int)
    task.add_argument("--ensure_pattern_categories", default=True, type=str2bool)
    task.add_argument("--short_source_policy", default=SHORT_SOURCE_POLICY_SKIP, choices=SHORT_SOURCE_POLICIES, type=str)
    task.add_argument("--limit", default=0, type=int)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--seed", default=10, type=int)
    runtime.add_argument("--compress_tasks", action="store_true")
    runtime.add_argument("--manifest_flush_interval", default=100, type=int)
    runtime.add_argument("--overwrite", action="store_true")
    return parser


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def generate_realtime_pose_tasks(args: argparse.Namespace) -> dict[str, int]:
    plan = plan_realtime_pose_task_generation(args)
    return execute_realtime_pose_task_generation(plan=plan, args=args)


def plan_realtime_pose_task_generation(args: argparse.Namespace) -> TaskGenerationPlan:
    """
    先完成所有不会改文件系统的校验。

    这样 split 配错、source 不存在或没有匹配样本时，不会留下空 output_dir 或 marker。
    """

    if int(args.seq_len) != REALTIME_POSE_SEQ_LEN:
        raise ValueError(f"realtime_pose 固定 seq_len={REALTIME_POSE_SEQ_LEN}，实际为 {args.seq_len}")
    if int(args.min_valid_trackers) < MIN_VALID_TRACKERS:
        raise ValueError(f"min_valid_trackers 至少为 {MIN_VALID_TRACKERS}")

    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    split_dir = Path(args.split_dir).resolve() if args.split_dir else None
    if not source_dir.exists():
        raise FileNotFoundError(f"{args.schema} 源目录不存在：{source_dir}")

    source_entries = read_source_entries(source_dir)
    if not source_entries:
        raise RuntimeError(f"{source_dir} 中没有 {args.schema} 源数据。")

    split_plans: list[SplitTaskPlan] = []
    for split_index, split in enumerate(args.splits):
        split_keys = read_split_keys(split_dir, split)
        split_entries = filter_entries_by_split(source_entries, split_keys)
        if args.limit > 0:
            split_entries = split_entries[: args.limit]
        if not split_entries:
            raise RuntimeError(f"split={split} 没有匹配 {args.schema} 源数据。")
        split_plans.append(
            SplitTaskPlan(
                split=split,
                entries=split_entries,
                seed=int(args.seed) + split_index,
            )
        )

    validate_task_output_dir_available(
        source_dir=source_dir,
        output_dir=output_dir,
        overwrite=bool(args.overwrite),
    )
    return TaskGenerationPlan(
        source_dir=source_dir,
        output_dir=output_dir,
        split_dir=split_dir,
        split_plans=split_plans,
    )


def execute_realtime_pose_task_generation(plan: TaskGenerationPlan, args: argparse.Namespace) -> dict[str, int]:
    prepare_task_output_dir(
        source_dir=plan.source_dir,
        output_dir=plan.output_dir,
        overwrite=bool(args.overwrite),
        split_dir=plan.split_dir,
        schema_name=str(args.schema),
    )

    counts = {}
    for split_plan in plan.split_plans:
        rng = np.random.default_rng(split_plan.seed)
        counts[split_plan.split] = generate_split_tasks(
            entries=split_plan.entries,
            output_dir=plan.output_dir,
            split=split_plan.split,
            rng=rng,
            args=args,
            source_split_dir=plan.split_dir,
        )
    return counts


def prepare_task_output_dir(
    source_dir: Path,
    output_dir: Path,
    overwrite: bool,
    split_dir: Path | None,
    schema_name: str,
) -> None:
    """准备 task 输出目录，并在覆盖已有目录前做路径安全检查。"""

    validate_task_output_dir_available(source_dir=source_dir, output_dir=output_dir, overwrite=overwrite)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_task_output_marker(source_dir=source_dir, output_dir=output_dir, split_dir=split_dir, schema_name=schema_name)


def validate_task_output_dir_available(source_dir: Path, output_dir: Path, overwrite: bool) -> None:
    """只校验 output_dir 是否可用，不创建或删除任何文件。"""

    if not output_dir.exists():
        return
    if not overwrite:
        raise FileExistsError(f"输出目录已存在：{output_dir}，如需重建请添加 --overwrite")
    validate_task_output_dir_for_overwrite(source_dir=source_dir, output_dir=output_dir)


def validate_task_output_dir_for_overwrite(source_dir: Path, output_dir: Path) -> None:
    """禁止 `--overwrite` 删除源数据目录、仓库级目录或未知非空目录。"""

    if output_dir == source_dir:
        raise ValueError(f"output_dir 不能与 source_dir 相同：{output_dir}")
    try:
        source_inside_output = source_dir.is_relative_to(output_dir)
    except AttributeError:
        source_inside_output = str(source_dir).startswith(str(output_dir))
    if source_inside_output:
        raise ValueError(f"output_dir 不能是 source_dir 的上级目录：output_dir={output_dir}, source_dir={source_dir}")

    protected_names = {"dataset", "runs", "save", "output"}
    if output_dir.name in protected_names:
        raise ValueError(f"拒绝覆盖仓库级产物目录：{output_dir}")

    children = list(output_dir.iterdir())
    if not children:
        return
    marker_path = output_dir / TASK_OUTPUT_MARKER
    if not marker_path.exists():
        raise ValueError(
            f"拒绝覆盖没有 {TASK_OUTPUT_MARKER} 标记的非空目录：{output_dir}。"
            "请确认路径无误后手动清理，或选择新的 output_dir。"
        )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    schema = get_schema_spec(marker.get("schema_name"))
    if marker.get("task_format") != schema.task_format:
        raise ValueError(f"output_dir 标记不是合法 realtime_pose task 目录：{marker_path}")


def write_task_output_marker(source_dir: Path, output_dir: Path, split_dir: Path | None, schema_name: str) -> None:
    schema = get_schema_spec(schema_name)
    marker = {
        "schema_name": schema.name,
        "task_format": schema.task_format,
        "source_dir": str(source_dir),
        "split_dir": str(split_dir) if split_dir is not None else "",
    }
    marker_path = output_dir / TASK_OUTPUT_MARKER
    marker_path.write_text(json.dumps(marker, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def read_source_entries(source_dir: Path) -> list[dict]:
    manifest_path = source_dir / "manifest.jsonl"
    if manifest_path.exists():
        entries = []
        with manifest_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("status", "converted") not in USABLE_SOURCE_STATUSES:
                    continue
                path = resolve_manifest_file(source_dir, entry.get("output_path") or entry.get("source_relative_path"))
                entries.append(
                    {
                        "source_path": str(path),
                        "source_relative_path": normalize_slashes(str(entry.get("source_relative_path") or path.name)),
                        "stablemotion_split_key": normalize_split_key(
                            str(entry.get("stablemotion_split_key") or Path(entry.get("source_relative_path") or path.name).with_suffix(".npy"))
                        ),
                        "frames": int(entry.get("frames") or 0),
                    }
                )
        return entries

    entries = []
    for path in sorted(source_dir.rglob("*.npz")):
        if "tasks" in path.parts:
            continue
        entries.append(
            {
                "source_path": str(path),
                "source_relative_path": path.relative_to(source_dir).as_posix(),
                "stablemotion_split_key": normalize_split_key(path.relative_to(source_dir).with_suffix(".npy").as_posix()),
                "frames": 0,
            }
        )
    return entries


def resolve_manifest_file(base_dir: Path, value: str | None) -> Path:
    if not value:
        raise KeyError("manifest entry 缺少 output_path/source_relative_path")
    path = Path(value)
    return path if path.is_absolute() else base_dir / normalize_slashes(value)


def read_split_keys(split_dir: Path | None, split: str) -> set[str] | None:
    if split_dir is None:
        return None
    path = split_dir / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"找不到 split 文件：{path}")
    return {normalize_split_key(line) for line in path.read_text(encoding="utf-8").splitlines() if normalize_split_key(line)}


def filter_entries_by_split(entries: list[dict], split_keys: set[str] | None) -> list[dict]:
    if split_keys is None:
        return entries
    return [entry for entry in entries if normalize_split_key(entry["stablemotion_split_key"]) in split_keys]


def normalize_split_key(raw_key: str) -> str:
    key = normalize_slashes(str(raw_key).strip())
    if not key:
        return ""
    key = key.split(",", 1)[0].strip()
    if key.endswith(".npy") or key.endswith(".npz"):
        key = key[:-4]
    return key


def normalize_slashes(path: str) -> str:
    return path.replace("\\", "/")


def generate_split_tasks(
    entries: list[dict],
    output_dir: Path,
    split: str,
    rng: np.random.Generator,
    args: argparse.Namespace,
    source_split_dir: Path | None,
) -> int:
    output_split_dir = output_dir / split
    task_dir = output_split_dir / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_split_dir / "manifest.jsonl"
    written = 0
    skipped_short_sources: list[dict] = []
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        progress = tqdm(total=len(entries) * int(args.samples_per_file), desc=f"生成 {split} realtime tasks", unit="window")
        for entry in entries:
            source_path = Path(entry["source_path"])
            known_frames = int(entry.get("frames") or 0)
            if 0 < known_frames < REALTIME_POSE_SEQ_LEN:
                if should_raise_for_short_source(args):
                    raise ValueError(format_short_source_error(source_path=source_path, source_frames=known_frames))
                skipped_short_sources.append(make_short_source_record(entry, source_path, known_frames))
                progress.update(int(args.samples_per_file))
                continue

            schema = get_schema_spec(args.schema)
            source = load_realtime_source(source_path, schema_name=schema.name)
            source_frames = int(source["body_pose_parent_6d"].shape[0])
            if source_frames < REALTIME_POSE_SEQ_LEN:
                if should_raise_for_short_source(args):
                    raise ValueError(format_short_source_error(source_path=source_path, source_frames=source_frames))
                skipped_short_sources.append(make_short_source_record(entry, source_path, source_frames))
                progress.update(int(args.samples_per_file))
                continue

            for sample_index in range(int(args.samples_per_file)):
                start_frame = int(rng.integers(0, source_frames - REALTIME_POSE_SEQ_LEN + 1))
                clip = clip_source(source, start_frame=start_frame, seq_len=REALTIME_POSE_SEQ_LEN)
                patterns = build_window_patterns(rng=rng, args=args)
                for pattern_index, pattern in enumerate(patterns):
                    sensor_valid = repeat_pattern_sensor_valid(pattern, seq_len=REALTIME_POSE_SEQ_LEN)
                    validate_sensor_valid(sensor_valid, min_valid_trackers=int(args.min_valid_trackers))
                    task_id = make_task_id(
                        split=split,
                        stablemotion_split_key=entry["stablemotion_split_key"],
                        sample_index=sample_index,
                        pattern_index=pattern_index,
                        pattern_category=pattern.category,
                    )
                    task_rel_path = Path("tasks") / f"{task_id}.npz"
                    task_path = output_split_dir / task_rel_path
                    save_task_npz(
                        task_path=task_path,
                        compress=bool(args.compress_tasks),
                        **clip,
                        sensor_valid=sensor_valid,
                        inpaint_mask=create_realtime_inpaint_mask(schema_name=schema.name),
                        start_frame=np.int64(start_frame),
                        valid_length=np.int64(REALTIME_POSE_SEQ_LEN),
                        source_frames=np.int64(source_frames),
                        seq_len=np.int64(REALTIME_POSE_SEQ_LEN),
                    )
                    manifest_entry = {
                        "task_id": task_id,
                        "task_path": task_rel_path.as_posix(),
                        "split": split,
                        "source_path": str(source_path),
                        "source_relative_path": normalize_slashes(entry["source_relative_path"]),
                        "stablemotion_split_key": normalize_slashes(entry["stablemotion_split_key"]),
                        "start_frame": start_frame,
                        "valid_length": REALTIME_POSE_SEQ_LEN,
                        "source_frames": source_frames,
                        "seq_len": REALTIME_POSE_SEQ_LEN,
                        "feature_dim": schema.feature_dim,
                        "task_format": schema.task_format,
                        "schema_name": schema.name,
                        "task_mode": TASK_MODE_REALTIME_POSE,
                        "target_start": REALTIME_POSE_TARGET_START,
                        "target_length": REALTIME_POSE_TARGET_LENGTH,
                        "mask_policy": str(args.mask_policy),
                        "split_dir": str(source_split_dir) if source_split_dir is not None else "",
                        "tracker_pattern": pattern.category,
                        "tracker_pattern_detail": pattern.to_dict(),
                    }
                    manifest_file.write(json.dumps(manifest_entry, ensure_ascii=False, sort_keys=True) + "\n")
                    written += 1
                    if args.manifest_flush_interval > 0 and written % int(args.manifest_flush_interval) == 0:
                        manifest_file.flush()
                progress.update(1)
        progress.close()
    write_short_source_report(output_split_dir=output_split_dir, split=split, records=skipped_short_sources)
    return written


def should_raise_for_short_source(args: argparse.Namespace) -> bool:
    return str(getattr(args, "short_source_policy", SHORT_SOURCE_POLICY_SKIP)) == SHORT_SOURCE_POLICY_ERROR


def format_short_source_error(source_path: Path, source_frames: int) -> str:
    return f"{source_path} 至少需要 {REALTIME_POSE_SEQ_LEN} 帧，实际为 {source_frames}"


def make_short_source_record(entry: dict, source_path: Path, source_frames: int) -> dict:
    return {
        "source_path": str(source_path),
        "source_relative_path": normalize_slashes(entry.get("source_relative_path", "")),
        "stablemotion_split_key": normalize_slashes(entry.get("stablemotion_split_key", "")),
        "source_frames": int(source_frames),
        "required_frames": REALTIME_POSE_SEQ_LEN,
    }


def write_short_source_report(output_split_dir: Path, split: str, records: list[dict]) -> None:
    if not records:
        return
    report_path = output_split_dir / SHORT_SOURCE_REPORT_NAME
    with report_path.open("w", encoding="utf-8") as report_file:
        for record in records:
            report_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        f"[generate_realtime_pose_tasks] split={split} skipped_short_sources={len(records)} report={report_path}",
        flush=True,
    )


def build_window_patterns(rng: np.random.Generator, args: argparse.Namespace):
    """根据 task 生成策略返回本窗口要写出的固定 tracker pattern。"""

    if str(args.mask_policy) == TASK_MASK_POLICY_FULL:
        return [make_tracker_pattern("full-trackers", rng)]
    if str(args.mask_policy) != TASK_MASK_POLICY_FIXED_PATTERNS:
        raise ValueError(f"未知 task mask_policy: {args.mask_policy}")

    fixed_patterns = list(getattr(args, "fixed_tracker_patterns", []) or [])
    if fixed_patterns:
        categories = normalize_tracker_pattern_categories(fixed_patterns)
        return [make_tracker_pattern(category, rng) for category in categories]
    return make_window_patterns(
        rng=rng,
        patterns_per_window=int(args.patterns_per_window),
        ensure_pattern_categories=bool(args.ensure_pattern_categories),
    )


def load_realtime_source(path: Path, schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME) -> dict[str, np.ndarray]:
    schema = get_schema_spec(schema_name)
    with np.load(path, allow_pickle=False) as data:
        required_keys = set(SOURCE_KEYS)
        if schema.supports_root_motion:
            required_keys.update({"root_delta_xz_ref", "root_height"})
        if schema.supports_contact:
            required_keys.add("foot_contact")
        missing = sorted(required_keys.difference(data.files))
        if missing:
            raise KeyError(f"{path} 缺少 {schema.name} 源字段：{missing}")
        source = {key: data[key].astype(np.float32, copy=True) for key in required_keys}
    frame_count = source["body_pose_parent_6d"].shape[0]
    expected_shapes = {
        "body_pose_parent_6d": (frame_count, 144),
        "root_pos_world": (frame_count, 3),
        "root_yaw": (frame_count,),
        "root_yaw_delta_sincos": (frame_count, 2),
        "tracker_pos_world": (frame_count, 6, 3),
        "tracker_rot_world_6d": (frame_count, 6, 6),
        "joints_world": (frame_count, 24, 3),
        "joint_offsets_parent": (24, 3),
    }
    if schema.supports_root_motion:
        expected_shapes["root_delta_xz_ref"] = (frame_count, 2)
        expected_shapes["root_height"] = (frame_count, 1)
    if schema.supports_contact:
        expected_shapes["foot_contact"] = (frame_count, 2)
    for key, shape in expected_shapes.items():
        if tuple(source[key].shape) != shape:
            raise ValueError(f"{path} 字段 {key} 应为 {shape}，实际为 {tuple(source[key].shape)}")
    return source


def clip_source(source: dict[str, np.ndarray], start_frame: int, seq_len: int) -> dict[str, np.ndarray]:
    end_frame = int(start_frame) + int(seq_len)
    return {
        key: value.copy() if key == "joint_offsets_parent" else value[start_frame:end_frame].copy()
        for key, value in source.items()
    }


def save_task_npz(task_path: Path, compress: bool, **arrays) -> None:
    temp_path = task_path.with_name(task_path.name + ".tmp")
    if temp_path.exists():
        temp_path.unlink()
    with temp_path.open("wb") as file:
        if compress:
            np.savez_compressed(file, **arrays)
        else:
            np.savez(file, **arrays)
    temp_path.replace(task_path)


def make_task_id(split: str, stablemotion_split_key: str, sample_index: int, pattern_index: int, pattern_category: str) -> str:
    key = normalize_slashes(stablemotion_split_key)
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", Path(key).with_suffix("").as_posix()).strip("_")
    digest = hashlib.sha1(f"{split}:{key}:{sample_index}:{pattern_index}:{pattern_category}".encode("utf-8")).hexdigest()[:8]
    safe_category = re.sub(r"[^A-Za-z0-9_]+", "_", pattern_category).strip("_")
    return f"{stem}_s{sample_index:04d}_p{pattern_index:02d}_{safe_category}_{digest}"


def main(argv: list[str] | None = None) -> dict[str, int]:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    counts = generate_realtime_pose_tasks(args)
    for split, count in counts.items():
        print(f"[generate_realtime_pose_tasks] split={split} tasks={count}")
    return counts


if __name__ == "__main__":
    main()
