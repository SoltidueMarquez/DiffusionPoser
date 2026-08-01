from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from tqdm.auto import tqdm

from data_loaders.body_fbx_kinematics import BodyFbxRest, fk_body_fbx_local_delta_root_y0
from data_loaders.realtime_pose_geometry import (
    build_known_target_np,
    build_pose_target_np,
    build_tracker_window_np,
    extract_forward_yaw_np,
)
from data_loaders.realtime_pose_kinematics import (
    SMPL_JOINT_NAMES,
    SMPL_PARENTS,
    TRACKER_JOINT_INDICES,
    rotation_6d_to_matrix_np,
)
from data_loaders.realtime_pose_task_store import (
    SHARD_STATS_FILE,
    STORE_METADATA_FILE,
    ShardWriter,
    write_generation_plan,
    write_json,
)
from data_loaders.realtime_pose_validation import load_realtime_metadata, validate_realtime_source_arrays
from data_loaders.sensor_masking import (
    BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY,
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_DIM,
    REALTIME_POSE_TARGET_START,
    TRACKER_CONTINUOUS_DIM,
    TRACKER_COUNT,
)
from data_loaders.tracker_timeline import (
    build_task_config_plan,
    materialize_task_configurations,
    stable_context_seed,
)
from utils.run_dirs import timestamped_child_dir, write_latest_pointer


USABLE_SOURCE_STATUSES = {"converted", "skipped_existing", "reused_source", "upgraded_existing_source"}
SHORT_SOURCE_POLICY_SKIP = "skip"
SHORT_SOURCE_POLICY_ERROR = "error"
SHORT_SOURCE_POLICIES = (SHORT_SOURCE_POLICY_SKIP, SHORT_SOURCE_POLICY_ERROR)
TASK_OUTPUT_MARKER = ".realtime_pose_tasks.json"


@dataclass(frozen=True)
class SplitTaskPlan:
    split: str
    sources: list[dict]
    tasks: list[dict]


@dataclass(frozen=True)
class TaskGenerationPlan:
    source_dir: Path
    output_root: Path
    output_dir: Path
    split_plans: list[SplitTaskPlan]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成确定性 mmap realtime pose task store。")
    paths = parser.add_argument_group("paths")
    paths.add_argument(
        "--source_dir",
        default="dataset/AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz",
    )
    paths.add_argument(
        "--output_dir",
        default="dataset/AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz_tasks",
    )
    paths.add_argument("--split_dir", default="data_loaders/splits")

    task = parser.add_argument_group("task")
    task.add_argument("--splits", nargs="+", default=["train"])
    task.add_argument("--seq_len", default=REALTIME_POSE_SEQ_LEN, type=int)
    task.add_argument("--base_windows_per_source", default=20, type=int)
    task.add_argument("--max_rollout_steps", default=4, type=int)
    task.add_argument("--shard_size", default=4096, type=int)
    task.add_argument("--short_source_policy", default=SHORT_SOURCE_POLICY_SKIP, choices=SHORT_SOURCE_POLICIES)
    task.add_argument("--limit", default=0, type=int)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--seed", default=10, type=int)
    runtime.add_argument("--run_name", default="auto")
    return parser


def generate_realtime_pose_tasks(args: argparse.Namespace) -> dict[str, int]:
    plan = plan_realtime_pose_task_generation(args)
    plan.output_dir.mkdir(parents=True, exist_ok=False)
    write_task_output_marker(plan)
    all_plan_entries = [task for split_plan in plan.split_plans for task in split_plan.tasks]
    plan_hash = write_generation_plan(plan.output_dir, all_plan_entries)

    counts: dict[str, int] = {}
    for split_plan in plan.split_plans:
        counts[split_plan.split] = materialize_split(
            split_plan=split_plan,
            output_dir=plan.output_dir,
            plan_hash=plan_hash,
            shard_size=int(args.shard_size),
            max_rollout_steps=int(args.max_rollout_steps),
        )
    write_latest_pointer(
        root_dir=plan.output_root,
        kind="tasks",
        output_dir=plan.output_dir,
        metadata={"source_dir": str(plan.source_dir), "counts": counts, "generation_plan_hash": plan_hash},
    )
    return counts


def plan_realtime_pose_task_generation(args: argparse.Namespace) -> TaskGenerationPlan:
    if int(args.seq_len) != REALTIME_POSE_SEQ_LEN:
        raise ValueError(f"当前任务固定 seq_len={REALTIME_POSE_SEQ_LEN}。")
    if not 1 <= int(args.max_rollout_steps) <= 4:
        raise ValueError("max_rollout_steps 必须在 [1,4]。")
    if int(args.base_windows_per_source) < 1:
        raise ValueError("base_windows_per_source 必须大于等于 1。")
    if int(args.shard_size) < 1:
        raise ValueError("shard_size 必须大于等于 1。")

    source_dir = Path(args.source_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir 不存在: {source_dir}")
    if output_root == source_dir or source_dir.is_relative_to(output_root):
        raise ValueError("output_dir 不能等于或包含 source_dir。")
    output_root.mkdir(parents=True, exist_ok=True)
    label = "rtp_140d" if str(args.run_name).lower() in {"", "auto"} else str(args.run_name)
    output_dir = timestamped_child_dir(output_root, label)

    entries = read_source_entries(source_dir)
    if not entries:
        raise RuntimeError(f"{source_dir} 中没有可用 source。")
    split_dir = Path(args.split_dir).resolve() if args.split_dir else None
    split_plans: list[SplitTaskPlan] = []
    required_frames = REALTIME_POSE_HISTORY_LENGTH + int(args.max_rollout_steps)
    for raw_split in args.splits:
        split = str(raw_split)
        split_entries = filter_entries_by_split(entries, read_split_keys(split_dir, split))
        if int(args.limit) > 0:
            split_entries = split_entries[: int(args.limit)]
        sources: list[dict] = []
        tasks: list[dict] = []
        for entry in split_entries:
            source_frames = int(entry["source_frames"])
            if source_frames < required_frames:
                if str(args.short_source_policy) == SHORT_SOURCE_POLICY_ERROR:
                    raise ValueError(
                        f"{entry['source_path']} 至少需要 {required_frames} 帧，实际为 {source_frames}"
                    )
                continue
            source_index = len(sources)
            source_info = {**entry, "source_index": source_index}
            sources.append(source_info)
            starts = select_window_starts(
                frame_count=source_frames,
                count=int(args.base_windows_per_source),
                max_rollout_steps=int(args.max_rollout_steps),
                global_seed=int(args.seed),
                split=split,
                source_id=str(entry["stablemotion_split_key"]),
            )
            for start_frame in starts:
                task_id = make_task_id(split, str(entry["stablemotion_split_key"]), start_frame)
                tasks.append(
                    {
                        "split": split,
                        "source_id": str(entry["stablemotion_split_key"]),
                        "source_relative_path": str(entry["source_relative_path"]),
                        "source_index": source_index,
                        "source_frames": source_frames,
                        "start_frame": int(start_frame),
                        "task_id": task_id,
                        "configs": build_task_config_plan(
                            task_id=task_id,
                            global_seed=int(args.seed),
                            max_rollout_steps=int(args.max_rollout_steps),
                        ),
                    }
                )
        if not tasks:
            raise RuntimeError(f"split={split} 没有可生成的 task。")
        split_plans.append(SplitTaskPlan(split=split, sources=sources, tasks=tasks))
    return TaskGenerationPlan(source_dir, output_root, output_dir, split_plans)


def select_window_starts(
    frame_count: int,
    count: int,
    max_rollout_steps: int,
    global_seed: int,
    split: str,
    source_id: str,
) -> list[int]:
    available = int(frame_count) - (REALTIME_POSE_HISTORY_LENGTH + int(max_rollout_steps)) + 1
    if available <= 0:
        return []
    if available <= int(count):
        return list(range(available))
    rng = np.random.Generator(
        np.random.PCG64(stable_context_seed(global_seed, split, source_id, "window"))
    )
    starts: list[int] = []
    for interval_index in range(int(count)):
        low = interval_index * available // int(count)
        high = (interval_index + 1) * available // int(count)
        starts.append(int(rng.integers(low, high)))
    return sorted(starts)


def materialize_split(
    split_plan: SplitTaskPlan,
    output_dir: Path,
    plan_hash: str,
    shard_size: int,
    max_rollout_steps: int,
) -> int:
    split_dir = output_dir / split_plan.split
    split_dir.mkdir(parents=True, exist_ok=False)
    source_lookup = {str(source["stablemotion_split_key"]): source for source in split_plan.sources}
    source_offsets = np.empty((len(split_plan.sources), 24, 3), dtype=np.float32)
    source_rest_rotations = np.empty((len(split_plan.sources), 24, 6), dtype=np.float32)
    source_metadata: list[dict] = []
    source_cache: dict[str, tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]] = {}
    recorded_sources: set[str] = set()

    shards: list[dict] = []
    fields = shard_fields(max_rollout_steps)
    for shard_index, first in enumerate(range(0, len(split_plan.tasks), int(shard_size))):
        shard_tasks = split_plan.tasks[first : first + int(shard_size)]
        relative_dir = Path("shards") / f"shard_{shard_index:05d}"
        writer = ShardWriter(split_dir / relative_dir, len(shard_tasks), fields)
        pose_sum = np.zeros(REALTIME_POSE_TARGET_DIM, dtype=np.float64)
        pose_sumsq = np.zeros_like(pose_sum)
        pose_count = 0
        tracker_sum = np.zeros((TRACKER_COUNT, TRACKER_CONTINUOUS_DIM), dtype=np.float64)
        tracker_sumsq = np.zeros_like(tracker_sum)
        tracker_count = np.zeros((TRACKER_COUNT, 1), dtype=np.float64)

        for row_index, task in enumerate(tqdm(shard_tasks, desc=f"生成 {split_plan.split} shard {shard_index}", unit="task")):
            source_id = str(task["source_id"])
            cached = source_cache.get(source_id)
            if cached is None:
                source_info = source_lookup[source_id]
                source_path = Path(source_info["source_path"])
                source = load_realtime_source(source_path)
                if int(source[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY].shape[0]) != int(source_info["source_frames"]):
                    raise ValueError(f"{source_path} 帧数与 source manifest 不一致。")
                joint_rotations = compute_source_joint_rotations_world(source)
                tracker_rotations = rotation_6d_to_matrix_np(source["tracker_rot_world_6d"])
                head_yaws = extract_forward_yaw_np(tracker_rotations[:, 0])
                cached = (source, joint_rotations, head_yaws)
                source_cache.clear()
                source_cache[source_id] = cached
                source_index = int(source_info["source_index"])
                source_offsets[source_index] = source["joint_offsets_parent"]
                source_rest_rotations[source_index] = source["joint_rest_local_rotations_6d"]
                if source_id not in recorded_sources:
                    source_metadata.append(
                        {
                            "source_index": source_index,
                            "source_id": source_id,
                            "source_path": str(source_path),
                            "source_relative_path": str(source_info["source_relative_path"]),
                            "source_frames": int(source_info["source_frames"]),
                            "target_fps": float(source_info["target_fps"]),
                            "is_mirrored": bool(source_info["is_mirrored"]),
                        }
                    )
                    recorded_sources.add(source_id)
            row = build_task_bundle_row(
                source=cached[0],
                joint_rotations_world=cached[1],
                head_yaws=cached[2],
                start_frame=int(task["start_frame"]),
                source_index=int(task["source_index"]),
                config_plans=task["configs"],
                max_rollout_steps=max_rollout_steps,
            )
            writer.write_row(row_index, row)
            pose = np.concatenate([row["pose_history"], row["current_target"][0:1]], axis=0).astype(np.float64)
            pose_sum += pose.sum(axis=0)
            pose_sumsq += np.square(pose).sum(axis=0)
            pose_count += pose.shape[0]
            tracker = row["tracker_continuous"][0].astype(np.float64)
            tracker_sum += tracker.sum(axis=0)
            tracker_sumsq += np.square(tracker).sum(axis=0)
            tracker_count += tracker.shape[0]

        writer.finish()
        np.savez(
            split_dir / relative_dir / SHARD_STATS_FILE,
            pose_sum=pose_sum,
            pose_sumsq=pose_sumsq,
            pose_count=np.int64(pose_count),
            tracker_sum=tracker_sum,
            tracker_sumsq=tracker_sumsq,
            tracker_count=tracker_count,
        )
        shards.append({"index": shard_index, "row_count": len(shard_tasks), "path": relative_dir.as_posix()})

    source_metadata.sort(key=lambda value: int(value["source_index"]))
    write_npy_atomic(split_dir / "source_joint_offsets_parent.npy", source_offsets)
    write_npy_atomic(split_dir / "source_joint_rest_local_rotations_6d.npy", source_rest_rotations)
    with (split_dir / "sources.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for source in source_metadata:
            file.write(json.dumps(source, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(
        split_dir / STORE_METADATA_FILE,
        {
            "generation_plan_hash": plan_hash,
            "split": split_plan.split,
            "sample_count": len(split_plan.tasks),
            "source_count": len(split_plan.sources),
            "max_rollout_steps": max_rollout_steps,
            "config_names": ["fixed_six", "fixed_three", "three_to_six", "six_to_three", "dropout"],
            "shards": shards,
        },
    )
    return len(split_plan.tasks)


def shard_fields(max_rollout_steps: int) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    steps = int(max_rollout_steps)
    return {
        "pose_history": ((REALTIME_POSE_HISTORY_LENGTH, REALTIME_POSE_TARGET_DIM), np.dtype("float32")),
        "current_target": ((steps, REALTIME_POSE_TARGET_DIM), np.dtype("float32")),
        "tracker_continuous": ((steps, REALTIME_POSE_SEQ_LEN, TRACKER_COUNT, TRACKER_CONTINUOUS_DIM), np.dtype("float32")),
        "full_known_target": ((steps, REALTIME_POSE_TARGET_DIM), np.dtype("float32")),
        "configured": ((5, REALTIME_POSE_HISTORY_LENGTH + steps, TRACKER_COUNT), np.dtype("uint8")),
        "measured_valid": ((5, REALTIME_POSE_HISTORY_LENGTH + steps, TRACKER_COUNT), np.dtype("uint8")),
        "missing_age": ((5, REALTIME_POSE_HISTORY_LENGTH + steps, TRACKER_COUNT), np.dtype("uint8")),
        "target_joints_head_ref": ((steps, 24, 3), np.dtype("float32")),
        "prev_joints_head_ref": ((steps, 24, 3), np.dtype("float32")),
        "target_root_position_head_ref": ((steps, 3), np.dtype("float32")),
        "target_root_yaw_world": ((steps,), np.dtype("float32")),
        "target_hip_height": ((steps,), np.dtype("float32")),
        "current_head_yaw_world": ((steps,), np.dtype("float32")),
        "current_head_position_world": ((steps, 3), np.dtype("float32")),
        "floor_y": ((steps,), np.dtype("float32")),
        "source_index": ((), np.dtype("int32")),
        "start_frame": ((), np.dtype("int32")),
    }


def build_task_bundle_row(
    source: dict[str, np.ndarray],
    joint_rotations_world: np.ndarray,
    head_yaws: np.ndarray,
    start_frame: int,
    source_index: int,
    config_plans: list[dict],
    max_rollout_steps: int,
) -> dict[str, np.ndarray | int]:
    steps = int(max_rollout_steps)
    configured, measured_valid, missing_age = materialize_task_configurations(
        config_plans,
        frame_count=REALTIME_POSE_HISTORY_LENGTH + steps,
    )
    row: dict[str, np.ndarray | int] = {
        "pose_history": np.empty((REALTIME_POSE_HISTORY_LENGTH, REALTIME_POSE_TARGET_DIM), dtype=np.float32),
        "current_target": np.empty((steps, REALTIME_POSE_TARGET_DIM), dtype=np.float32),
        "tracker_continuous": np.empty((steps, REALTIME_POSE_SEQ_LEN, TRACKER_COUNT, TRACKER_CONTINUOUS_DIM), dtype=np.float32),
        "full_known_target": np.empty((steps, REALTIME_POSE_TARGET_DIM), dtype=np.float32),
        "configured": configured.astype(np.uint8),
        "measured_valid": measured_valid.astype(np.uint8),
        "missing_age": missing_age,
        "target_joints_head_ref": np.empty((steps, 24, 3), dtype=np.float32),
        "prev_joints_head_ref": np.empty((steps, 24, 3), dtype=np.float32),
        "target_root_position_head_ref": np.empty((steps, 3), dtype=np.float32),
        "target_root_yaw_world": np.empty(steps, dtype=np.float32),
        "target_hip_height": np.empty(steps, dtype=np.float32),
        "current_head_yaw_world": np.empty(steps, dtype=np.float32),
        "current_head_position_world": np.empty((steps, 3), dtype=np.float32),
        "floor_y": np.empty(steps, dtype=np.float32),
        "source_index": int(source_index),
        "start_frame": int(start_frame),
    }
    full_state = np.ones((REALTIME_POSE_SEQ_LEN, TRACKER_COUNT), dtype=bool)
    zero_age = np.zeros_like(full_state, dtype=np.float32)
    for step in range(steps):
        step_start = int(start_frame) + step
        frame_slice = slice(step_start, step_start + REALTIME_POSE_SEQ_LEN)
        current_absolute = step_start + REALTIME_POSE_TARGET_START
        current_head_yaw = float(head_yaws[current_absolute])
        current_head_position = source["tracker_pos_world"][current_absolute, 0].astype(np.float32)
        floor_y = float(source["root_pos_world"][current_absolute, 1])
        pose_window = build_pose_target_np(
            joint_rotations_world[frame_slice],
            source["root_yaw"][frame_slice],
            current_head_yaw,
        )
        tracker_window = build_tracker_window_np(
            source["tracker_pos_world"][frame_slice],
            source["tracker_rot_world_6d"][frame_slice],
            current_head_position,
            floor_y,
            current_head_yaw,
            full_state,
            full_state,
            zero_age,
        )
        full_known_target, _ = build_known_target_np(
            tracker_window[REALTIME_POSE_TARGET_START],
            current_head_yaw,
            source["tracker_rot_world_6d"][current_absolute],
        )
        cos_yaw = np.cos(current_head_yaw)
        sin_yaw = np.sin(current_head_yaw)
        yaw_inv = np.asarray(
            [[cos_yaw, 0.0, -sin_yaw], [0.0, 1.0, 0.0], [sin_yaw, 0.0, cos_yaw]],
            dtype=np.float64,
        )
        origin = np.asarray([current_head_position[0], floor_y, current_head_position[2]], dtype=np.float64)
        joints_head = np.einsum(
            "ij,taj->tai",
            yaw_inv,
            source["joints_world"][frame_slice].astype(np.float64) - origin[None, None],
        ).astype(np.float32)
        root_head = yaw_inv @ (source["root_pos_world"][current_absolute].astype(np.float64) - origin)
        if step == 0:
            row["pose_history"][:] = pose_window[:REALTIME_POSE_HISTORY_LENGTH]
        row["current_target"][step] = pose_window[REALTIME_POSE_TARGET_START]
        row["tracker_continuous"][step] = tracker_window[..., :TRACKER_CONTINUOUS_DIM]
        row["full_known_target"][step] = full_known_target
        row["target_joints_head_ref"][step] = joints_head[REALTIME_POSE_TARGET_START]
        row["prev_joints_head_ref"][step] = joints_head[REALTIME_POSE_TARGET_START - 1]
        row["target_root_position_head_ref"][step] = root_head.astype(np.float32)
        row["target_root_yaw_world"][step] = source["root_yaw"][current_absolute]
        row["target_hip_height"][step] = source["pelvis_height"][current_absolute, 0]
        row["current_head_yaw_world"][step] = current_head_yaw
        row["current_head_position_world"][step] = current_head_position
        row["floor_y"][step] = floor_y
    return row


def compute_source_joint_rotations_world(source: dict[str, np.ndarray]) -> np.ndarray:
    rest_rotations = rotation_6d_to_matrix_np(source["joint_rest_local_rotations_6d"])
    rest = BodyFbxRest(
        bone_names=tuple(SMPL_JOINT_NAMES),
        parents=SMPL_PARENTS.copy(),
        rest_local_positions=source["joint_offsets_parent"].astype(np.float32),
        rest_local_rotations=rest_rotations.astype(np.float32),
        tracker_joint_indices=TRACKER_JOINT_INDICES.copy(),
        source_path=None,
    )
    _, rotations = fk_body_fbx_local_delta_root_y0(
        body_pose_local_delta_6d=source[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY],
        actor_root_pos_world=source["root_pos_world"],
        root_heading=source["root_yaw"],
        pelvis_height=source["pelvis_height"],
        rest=rest,
    )
    return rotations


def load_realtime_source(path: Path, schema_name: str | None = None) -> dict[str, np.ndarray]:
    del schema_name
    with np.load(path, allow_pickle=False) as data:
        validate_realtime_source_arrays(data, path=path)
        fields = (
            BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY,
            "root_pos_world",
            "root_yaw",
            "pelvis_height",
            "tracker_pos_world",
            "tracker_rot_world_6d",
            "joints_world",
            "joint_offsets_parent",
            "joint_rest_local_rotations_6d",
        )
        return {key: np.asarray(data[key]).copy() for key in fields}


def read_source_entries(source_dir: Path) -> list[dict]:
    manifest_path = source_dir / "manifest.jsonl"
    entries: list[dict] = []
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("status", "converted") not in USABLE_SOURCE_STATUSES:
                    continue
                raw_path = entry.get("output_path") or entry.get("source_relative_path")
                path = Path(str(raw_path))
                if not path.is_absolute():
                    path = source_dir / path
                relative = str(entry.get("source_relative_path") or path.relative_to(source_dir))
                entries.append(
                    {
                        "source_path": str(path.resolve()),
                        "source_relative_path": normalize_slashes(relative),
                        "stablemotion_split_key": normalize_split_key(
                            str(entry.get("stablemotion_split_key") or Path(relative).with_suffix(".npy"))
                        ),
                        "source_frames": int(entry["frames"]),
                        "target_fps": float(entry["target_fps"]),
                        "is_mirrored": bool(entry.get("is_mirrored", normalize_slashes(relative).startswith("M/"))),
                    }
                )
    else:
        for path in sorted(source_dir.rglob("*.npz")):
            relative = path.relative_to(source_dir).as_posix()
            with np.load(path, allow_pickle=False) as data:
                metadata = load_realtime_metadata(data, path)
                source_frames = validate_realtime_source_arrays(data, metadata=metadata, path=path)
            entries.append(
                {
                    "source_path": str(path.resolve()),
                    "source_relative_path": relative,
                    "stablemotion_split_key": normalize_split_key(Path(relative).with_suffix(".npy").as_posix()),
                    "source_frames": source_frames,
                    "target_fps": float(metadata["target_fps"]),
                    "is_mirrored": bool(metadata.get("is_mirrored", relative.startswith("M/"))),
                }
            )
    return sorted(entries, key=lambda value: str(value["stablemotion_split_key"]))


def read_split_keys(split_dir: Path | None, split: str) -> set[str] | None:
    if split_dir is None:
        return None
    path = split_dir / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"找不到 split 文件: {path}")
    return {normalize_split_key(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def filter_entries_by_split(entries: list[dict], split_keys: set[str] | None) -> list[dict]:
    if split_keys is None:
        return list(entries)
    return [entry for entry in entries if normalize_split_key(entry["stablemotion_split_key"]) in split_keys]


def normalize_split_key(raw_key: str) -> str:
    key = normalize_slashes(str(raw_key).strip()).split(",", 1)[0].strip()
    if key.endswith((".npy", ".npz")):
        key = key[:-4]
    return key


def normalize_slashes(path: str) -> str:
    return str(path).replace("\\", "/")


def make_task_id(split: str, source_id: str, start_frame: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", Path(source_id).stem).strip("_") or "source"
    digest = hashlib.sha256(f"{split}\x1f{source_id}\x1f{int(start_frame)}".encode("utf-8")).hexdigest()[:16]
    return f"{stem[:24]}_f{int(start_frame):08d}_{digest}"


def write_npy_atomic(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    array = open_memmap(temporary, mode="w+", dtype=value.dtype, shape=value.shape)
    array[:] = value
    array.flush()
    del array
    temporary.replace(path)


def write_task_output_marker(plan: TaskGenerationPlan) -> None:
    write_json(plan.output_dir / TASK_OUTPUT_MARKER, {"source_dir": str(plan.source_dir)})


def main(argv: list[str] | None = None) -> dict[str, int]:
    args = build_argument_parser().parse_args(argv)
    counts = generate_realtime_pose_tasks(args)
    for split, count in counts.items():
        print(f"[generate_realtime_pose_tasks] split={split} tasks={count}")
    return counts


if __name__ == "__main__":
    main()
