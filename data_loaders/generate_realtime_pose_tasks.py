from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from data_loaders.body_fbx_kinematics import (
    BodyFbxRest,
    fk_body_fbx_local_delta_root_y0,
)
from data_loaders.realtime_pose_geometry import (
    build_pose_target_np,
    build_tracker_measurements_np,
    extract_continuous_rotation_heading_np,
    extract_forward_yaw_np,
)
from data_loaders.realtime_pose_kinematics import (
    SMPL_JOINT_NAMES,
    SMPL_PARENTS,
    TRACKER_JOINT_INDICES,
    rotation_6d_to_matrix_np,
)
from data_loaders.realtime_pose_predictor_features import build_predictor_step_features_np
from data_loaders.realtime_pose_task_store import SHARD_STATS_FILE, ShardWriter
from data_loaders.realtime_pose_validation import validate_realtime_source_arrays
from data_loaders.sensor_masking import (
    BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY,
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_DIM,
    PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH,
    PREDICTOR_SPARSE_DIM,
    TRACKER_CONTINUOUS_DIM,
    TRACKER_COUNT,
)


SHORT_SOURCE_POLICY_SKIP = "skip"
SHORT_SOURCE_POLICY_ERROR = "error"
SHORT_SOURCE_POLICIES = (SHORT_SOURCE_POLICY_SKIP, SHORT_SOURCE_POLICY_ERROR)
MINIMUM_CURRENT_FRAME = PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH


@dataclass(frozen=True)
class SplitTaskPlan:
    split: str
    sources: list[dict]
    tasks: list[dict]


@dataclass(frozen=True)
class TaskGenerationPlan:
    source_dir: Path
    output_dir: Path
    split_plans: list[SplitTaskPlan]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="生成 Predictor + 单帧 DiT 共用的 mmap task store。"
    )
    paths = parser.add_argument_group("paths")
    paths.add_argument(
        "--source_dir",
        default=(
            "dataset/AMASS_realtime_pose_body_fbx_local_"
            "pelvis_residual_root_y0_stationary5_30hz"
        ),
    )
    paths.add_argument(
        "--output_dir",
        default="dataset/AMASS_realtime_pose_predictor_current_dit_tasks",
    )
    paths.add_argument("--split_dir", default="data_loaders/splits")

    task = parser.add_argument_group("task")
    task.add_argument("--splits", nargs="+", default=["train"])
    task.add_argument("--seq_len", default=REALTIME_POSE_SEQ_LEN, type=int)
    task.add_argument("--base_windows_per_source", default=20, type=int)
    task.add_argument("--shard_size", default=4096, type=int)
    task.add_argument(
        "--short_source_policy",
        default=SHORT_SOURCE_POLICY_SKIP,
        choices=SHORT_SOURCE_POLICIES,
    )
    task.add_argument("--limit", default=0, type=int)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--seed", default=10, type=int)
    runtime.add_argument("--overwrite", action="store_true")
    return parser


def generate_realtime_pose_tasks(args: argparse.Namespace) -> dict[str, int]:
    plan = plan_realtime_pose_task_generation(args)
    output_dir = plan.output_dir
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and not bool(args.overwrite):
        raise FileExistsError(
            f"task 输出目录已存在: {output_dir}；请指定新目录或使用 --overwrite。"
        )

    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        counts = {
            split_plan.split: materialize_split(
                split_plan=split_plan,
                output_dir=temporary_dir,
                shard_size=int(args.shard_size),
            )
            for split_plan in plan.split_plans
        }
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temporary_dir.replace(output_dir)
        return counts
    except BaseException:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise


def plan_realtime_pose_task_generation(
    args: argparse.Namespace,
) -> TaskGenerationPlan:
    if int(args.seq_len) != REALTIME_POSE_SEQ_LEN:
        raise ValueError(f"当前任务固定 seq_len={REALTIME_POSE_SEQ_LEN}。")
    if int(args.base_windows_per_source) < 1:
        raise ValueError("base_windows_per_source 必须大于等于 1。")
    if int(args.shard_size) < 1:
        raise ValueError("shard_size 必须大于等于 1。")

    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    repository_root = Path(__file__).resolve().parents[1]
    if output_dir.parent == output_dir or output_dir == repository_root:
        raise ValueError(f"拒绝将 task output_dir 指向磁盘根目录或仓库根目录：{output_dir}")
    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir 不存在: {source_dir}")
    if (
        output_dir == source_dir
        or output_dir.is_relative_to(source_dir)
        or source_dir.is_relative_to(output_dir)
    ):
        raise ValueError("source_dir 与 output_dir 不能相同或相互包含。")

    entries = read_source_entries(source_dir)
    if not entries:
        raise RuntimeError(f"{source_dir} 中没有可用 source。")
    split_dir = Path(args.split_dir).resolve() if args.split_dir else None
    split_plans: list[SplitTaskPlan] = []
    minimum_frames = MINIMUM_CURRENT_FRAME + 1

    for raw_split in args.splits:
        split = str(raw_split)
        split_keys = read_split_keys(split_dir, split)
        split_entries = filter_entries_by_split(entries, split_keys)
        if int(args.limit) > 0:
            split_entries = split_entries[: int(args.limit)]
        sources: list[dict] = []
        tasks: list[dict] = []
        for entry in split_entries:
            source_path = Path(entry["source_path"])
            with np.load(source_path, allow_pickle=False) as payload:
                frame_count = validate_realtime_source_arrays(
                    payload, path=source_path
                )
            if frame_count < minimum_frames:
                if str(args.short_source_policy) == SHORT_SOURCE_POLICY_ERROR:
                    raise ValueError(
                        f"{source_path} 至少需要 {minimum_frames} 帧，实际为 {frame_count}"
                    )
                continue
            sources.append({**entry, "source_frames": frame_count})
            current_frames = select_window_starts(
                frame_count=frame_count,
                count=int(args.base_windows_per_source),
                max_rollout_steps=0,
                global_seed=int(args.seed),
                split=split,
                source_id=str(entry["stablemotion_split_key"]),
            )
            for current_frame in current_frames:
                tasks.append(
                    {
                        "split": split,
                        "source_id": str(entry["stablemotion_split_key"]),
                        "current_frame": int(current_frame),
                        "task_seed": make_task_seed(
                            split,
                            str(entry["stablemotion_split_key"]),
                            current_frame,
                        ),
                    }
                )
        if not tasks:
            raise RuntimeError(f"split={split} 没有可生成的 task。")
        split_plans.append(SplitTaskPlan(split=split, sources=sources, tasks=tasks))
    return TaskGenerationPlan(source_dir, output_dir, split_plans)


def select_window_starts(
    frame_count: int,
    count: int,
    max_rollout_steps: int,
    global_seed: int,
    split: str,
    source_id: str,
) -> list[int]:
    """沿 source 时间轴分层抽样当前帧；本函数不生成任何 Tracker 场景。"""

    first = MINIMUM_CURRENT_FRAME
    last = int(frame_count) - 1 - int(max_rollout_steps)
    if last < first:
        return []
    candidates = np.arange(first, last + 1, dtype=np.int64)
    requested = min(int(count), int(candidates.size))
    if requested <= 0:
        return []
    seed = stable_seed(global_seed, split, source_id, "task_windows")
    rng = np.random.Generator(np.random.PCG64(seed))
    selected: list[int] = []
    for interval_index in range(requested):
        low = interval_index * candidates.size // requested
        high = (interval_index + 1) * candidates.size // requested
        selected.append(int(candidates[int(rng.integers(low, high))]))
    return sorted(selected)


def materialize_split(
    split_plan: SplitTaskPlan,
    output_dir: Path,
    shard_size: int,
) -> int:
    split_dir = output_dir / split_plan.split
    split_dir.mkdir(parents=True, exist_ok=False)
    source_lookup = {
        str(source["stablemotion_split_key"]): source
        for source in split_plan.sources
    }
    source_cache: dict[
        str,
        tuple[dict[str, np.ndarray], np.ndarray, np.ndarray],
    ] = {}

    for shard_index, first in enumerate(
        range(0, len(split_plan.tasks), int(shard_size))
    ):
        shard_tasks = split_plan.tasks[first : first + int(shard_size)]
        relative_dir = Path("shards") / f"shard_{shard_index:05d}"
        writer = ShardWriter(
            split_dir / relative_dir,
            len(shard_tasks),
            shard_fields(),
        )
        pose_sum = np.zeros(REALTIME_POSE_TARGET_DIM, dtype=np.float64)
        pose_sumsq = np.zeros_like(pose_sum)
        pose_count = 0
        tracker_sum = np.zeros(
            (TRACKER_COUNT, TRACKER_CONTINUOUS_DIM), dtype=np.float64
        )
        tracker_sumsq = np.zeros_like(tracker_sum)
        tracker_count = np.zeros((TRACKER_COUNT, 1), dtype=np.float64)
        predictor_sparse_sum = np.zeros(PREDICTOR_SPARSE_DIM, dtype=np.float64)
        predictor_sparse_sumsq = np.zeros_like(predictor_sparse_sum)
        predictor_sparse_count = 0

        for row_index, task in enumerate(
            tqdm(
                shard_tasks,
                desc=f"生成 {split_plan.split} shard {shard_index}",
                unit="task",
            )
        ):
            source_id = str(task["source_id"])
            cached = source_cache.get(source_id)
            if cached is None:
                source_info = source_lookup[source_id]
                source = load_realtime_source(Path(source_info["source_path"]))
                joint_rotations = compute_source_joint_rotations_world(source)
                tracker_rotations = rotation_6d_to_matrix_np(
                    source["tracker_rot_world_6d"]
                )
                head_yaws = extract_forward_yaw_np(tracker_rotations[:, 0])
                root_yaws = extract_continuous_rotation_heading_np(
                    joint_rotations[:, 0], initial_yaw=float(head_yaws[0])
                )
                cached = (source, joint_rotations, root_yaws)
                # Task 按 source 排序，单项 cache 足以避免无界占用 worker 内存。
                source_cache.clear()
                source_cache[source_id] = cached
            row = build_task_bundle_row(
                source=cached[0],
                joint_rotations_world=cached[1],
                root_yaws_world=cached[2],
                current_frame=int(task["current_frame"]),
                task_seed=int(task["task_seed"]),
            )
            writer.write_row(row_index, row)

            poses = np.concatenate(
                [
                    np.asarray(row["motion_context_clean"], dtype=np.float64),
                    np.asarray(row["current_pose_target_clean"], dtype=np.float64)[
                        None
                    ],
                ],
                axis=0,
            )
            pose_sum += poses.sum(axis=0)
            pose_sumsq += np.square(poses).sum(axis=0)
            pose_count += int(poses.shape[0])
            tracker = np.asarray(
                row["current_tracker_continuous"], dtype=np.float64
            )
            tracker_sum += tracker
            tracker_sumsq += np.square(tracker)
            tracker_count += 1
            sparse = np.asarray(
                row["core_tracker_context_clean"], dtype=np.float64
            )
            predictor_sparse_sum += sparse.sum(axis=0)
            predictor_sparse_sumsq += np.square(sparse).sum(axis=0)
            predictor_sparse_count += int(sparse.shape[0])

        writer.finish()
        np.savez(
            split_dir / relative_dir / SHARD_STATS_FILE,
            pose_sum=pose_sum,
            pose_sumsq=pose_sumsq,
            pose_count=np.int64(pose_count),
            tracker_sum=tracker_sum,
            tracker_sumsq=tracker_sumsq,
            tracker_count=tracker_count,
            predictor_sparse_sum=predictor_sparse_sum,
            predictor_sparse_sumsq=predictor_sparse_sumsq,
            predictor_sparse_count=np.int64(predictor_sparse_count),
        )
    return len(split_plan.tasks)


def shard_fields() -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    return {
        "motion_context_clean": (
            (REALTIME_POSE_HISTORY_LENGTH, REALTIME_POSE_TARGET_DIM),
            np.dtype("float32"),
        ),
        "core_tracker_context_clean": (
            (PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH, PREDICTOR_SPARSE_DIM),
            np.dtype("float32"),
        ),
        "current_pose_target_clean": (
            (REALTIME_POSE_TARGET_DIM,),
            np.dtype("float32"),
        ),
        "current_tracker_continuous": (
            (TRACKER_COUNT, TRACKER_CONTINUOUS_DIM),
            np.dtype("float32"),
        ),
        "previous_pose_target_clean": (
            (REALTIME_POSE_TARGET_DIM,),
            np.dtype("float32"),
        ),
        "target_joints_head_ref": ((24, 3), np.dtype("float32")),
        "target_root_position_head_ref": ((3,), np.dtype("float32")),
        "target_root_yaw_world": ((), np.dtype("float32")),
        "target_hip_height": ((), np.dtype("float32")),
        "current_head_yaw_world": ((), np.dtype("float32")),
        "current_head_position_world": ((3,), np.dtype("float32")),
        "floor_y": ((), np.dtype("float32")),
        "joint_offsets_parent": ((24, 3), np.dtype("float32")),
        "joint_rest_local_rotations_6d": ((24, 6), np.dtype("float32")),
        "task_seed": ((), np.dtype("uint64")),
        "current_frame": ((), np.dtype("int32")),
    }


def build_task_bundle_row(
    source: dict[str, np.ndarray],
    joint_rotations_world: np.ndarray,
    root_yaws_world: np.ndarray,
    current_frame: int,
    task_seed: int,
) -> dict[str, np.ndarray | int | float]:
    """物化连续过去 10 帧、核心三点 Predictor 条件和单帧监督。"""

    current = int(current_frame)
    frame_count = int(joint_rotations_world.shape[0])
    if current < MINIMUM_CURRENT_FRAME or current >= frame_count:
        raise ValueError(
            f"current_frame 必须位于 [{MINIMUM_CURRENT_FRAME},{frame_count - 1}]。"
        )
    motion_indices = np.arange(current - REALTIME_POSE_HISTORY_LENGTH, current)
    tracker_indices = np.arange(
        current - PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH, current + 1
    )
    current_head_position = np.asarray(
        source["tracker_pos_world"][current, 0], dtype=np.float32
    )
    floor_y = float(source["root_pos_world"][current, 1])

    predictor_features = build_predictor_step_features_np(
        motion_rotations_world=joint_rotations_world[motion_indices],
        tracker_positions_world_with_previous=source["tracker_pos_world"][
            tracker_indices
        ],
        tracker_rotations_world_6d_with_previous=source[
            "tracker_rot_world_6d"
        ][tracker_indices],
        floor_y=floor_y,
    )
    # Task 中的 history、Predictor sparse、target 与当前 Tracker 必须共享同一个 C_n。
    current_head_yaw = float(predictor_features.current_head_yaw_world)
    current_pose = build_pose_target_np(
        joint_rotations_world[current : current + 1], current_head_yaw
    )[0]
    previous_pose = build_pose_target_np(
        joint_rotations_world[current - 1 : current], current_head_yaw
    )[0]
    current_tracker = build_tracker_measurements_np(
        source["tracker_pos_world"][current : current + 1],
        source["tracker_rot_world_6d"][current : current + 1],
        current_head_position,
        floor_y,
        current_head_yaw,
    )[0]
    cos_yaw = np.cos(current_head_yaw)
    sin_yaw = np.sin(current_head_yaw)
    yaw_inverse = np.asarray(
        [
            [cos_yaw, 0.0, -sin_yaw],
            [0.0, 1.0, 0.0],
            [sin_yaw, 0.0, cos_yaw],
        ],
        dtype=np.float64,
    )
    origin = np.asarray(
        [current_head_position[0], floor_y, current_head_position[2]],
        dtype=np.float64,
    )
    target_joints_head = np.einsum(
        "ij,aj->ai",
        yaw_inverse,
        source["joints_world"][current].astype(np.float64) - origin[None],
    ).astype(np.float32)
    target_root_head = yaw_inverse @ (
        source["root_pos_world"][current].astype(np.float64) - origin
    )
    return {
        "motion_context_clean": predictor_features.motion_context.astype(np.float32),
        "core_tracker_context_clean": predictor_features.core_tracker_context.astype(
            np.float32
        ),
        "current_pose_target_clean": current_pose.astype(np.float32),
        "current_tracker_continuous": current_tracker.astype(np.float32),
        "previous_pose_target_clean": previous_pose.astype(np.float32),
        "target_joints_head_ref": target_joints_head,
        "target_root_position_head_ref": target_root_head.astype(np.float32),
        "target_root_yaw_world": np.float32(root_yaws_world[current]),
        "target_hip_height": np.float32(source["pelvis_height"][current, 0]),
        "current_head_yaw_world": np.float32(current_head_yaw),
        "current_head_position_world": current_head_position,
        "floor_y": np.float32(floor_y),
        "joint_offsets_parent": source["joint_offsets_parent"].astype(np.float32),
        "joint_rest_local_rotations_6d": source[
            "joint_rest_local_rotations_6d"
        ].astype(np.float32),
        "task_seed": np.uint64(task_seed),
        "current_frame": np.int32(current),
    }


def compute_source_joint_rotations_world(
    source: dict[str, np.ndarray],
) -> np.ndarray:
    rest_rotations = rotation_6d_to_matrix_np(
        source["joint_rest_local_rotations_6d"]
    )
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
    with np.load(path, allow_pickle=False) as payload:
        validate_realtime_source_arrays(payload, path=path)
        fields = (
            BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY,
            "root_pos_world",
            "root_yaw",
            "pelvis_height",
            "tracker_pos_world",
            "tracker_rot_world_6d",
            "joints_world",
            "stationary_prob_5",
            "joint_offsets_parent",
            "joint_rest_local_rotations_6d",
        )
        return {key: np.asarray(payload[key]).copy() for key in fields}


def read_source_entries(source_dir: Path) -> list[dict]:
    entries: list[dict] = []
    seen_keys: set[str] = set()
    for path in sorted(source_dir.rglob("*.npz")):
        relative = path.relative_to(source_dir).as_posix()
        source_key = normalize_split_key(relative)
        if source_key in seen_keys:
            raise ValueError(f"source 目录出现重复相对键: {source_key}")
        seen_keys.add(source_key)
        entries.append(
            {
                "source_path": str(path.resolve()),
                "source_relative_path": relative,
                "stablemotion_split_key": source_key,
                "is_mirrored": relative.startswith("M/"),
            }
        )
    return entries


def read_split_keys(split_dir: Path | None, split: str) -> set[str] | None:
    if split_dir is None:
        return None
    path = split_dir / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"找不到 split 文件: {path}")
    return {
        normalize_split_key(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def filter_entries_by_split(
    entries: list[dict], split_keys: set[str] | None
) -> list[dict]:
    if split_keys is None:
        return list(entries)
    return [
        entry
        for entry in entries
        if normalize_split_key(entry["stablemotion_split_key"]) in split_keys
    ]


def normalize_split_key(raw_key: str) -> str:
    key = str(raw_key).strip().replace("\\", "/").split(",", 1)[0].strip()
    if key.endswith((".npy", ".npz")):
        key = key[:-4]
    return key


def make_task_id(split: str, source_id: str, current_frame: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", Path(source_id).stem).strip("_")
    digest = hashlib.sha256(
        f"{split}\x1f{source_id}\x1f{int(current_frame)}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{stem[:24] or 'source'}_f{int(current_frame):08d}_{digest}"


def make_task_seed(split: str, source_id: str, current_frame: int) -> int:
    payload = f"{split}\x1f{source_id}\x1f{int(current_frame)}".encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:8], byteorder="little", signed=False
    )


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:8], byteorder="little", signed=False
    )


def main(argv: list[str] | None = None) -> dict[str, int]:
    args = build_argument_parser().parse_args(argv)
    counts = generate_realtime_pose_tasks(args)
    for split, count in counts.items():
        print(f"[generate_realtime_pose_tasks] split={split} tasks={count}")
    return counts


if __name__ == "__main__":
    main()
