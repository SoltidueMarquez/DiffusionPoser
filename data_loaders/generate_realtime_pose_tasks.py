from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
from data_loaders.realtime_pose_validation import (
    load_realtime_metadata,
    validate_realtime_source_arrays,
)
from data_loaders.sensor_masking import (
    BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY,
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_LENGTH,
    REALTIME_POSE_TARGET_START,
    TRACKER_COUNT,
)
from data_loaders.tracker_timeline import (
    build_tracker_timeline,
    candidate_starts_by_scenario,
    classify_tracker_window,
    sample_balanced_starts,
    stable_source_seed,
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
    entries: list[dict]


@dataclass(frozen=True)
class TaskGenerationPlan:
    source_dir: Path
    output_root: Path
    output_dir: Path
    split_plans: list[SplitTaskPlan]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 140 维 Head-anchor realtime pose tasks。")
    paths = parser.add_argument_group("paths")
    paths.add_argument(
        "--source_dir",
        default="dataset/AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz",
        type=str,
    )
    paths.add_argument(
        "--output_dir",
        default="dataset/AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz_tasks",
        type=str,
    )
    paths.add_argument("--split_dir", default="data_loaders/splits", type=str)

    task = parser.add_argument_group("task")
    task.add_argument("--splits", nargs="+", default=["train"], type=str)
    task.add_argument("--seq_len", default=REALTIME_POSE_SEQ_LEN, type=int)
    task.add_argument("--samples_per_file", default=4, type=int, help="每个 source、每类场景的窗口数。")
    task.add_argument(
        "--scenario_weights",
        nargs=5,
        default=[1.0, 1.0, 1.0, 1.0, 1.0],
        type=float,
        metavar=("SIX", "THREE", "3TO6", "6TO3", "DROPOUT"),
        help="五类场景的相对采样权重；默认等比例。",
    )
    task.add_argument("--rollout_steps", default=1, type=int)
    task.add_argument("--short_source_policy", default=SHORT_SOURCE_POLICY_SKIP, choices=SHORT_SOURCE_POLICIES)
    task.add_argument("--limit", default=0, type=int)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--seed", default=10, type=int)
    runtime.add_argument("--compress_tasks", action="store_true")
    runtime.add_argument("--manifest_flush_interval", default=100, type=int)
    runtime.add_argument("--run_name", default="auto", type=str)
    return parser


def generate_realtime_pose_tasks(args: argparse.Namespace) -> dict[str, int]:
    plan = plan_realtime_pose_task_generation(args)
    plan.output_dir.mkdir(parents=True, exist_ok=False)
    write_task_output_marker(plan)
    counts: dict[str, int] = {}
    for split_plan in plan.split_plans:
        counts[split_plan.split] = generate_split_tasks(
            entries=split_plan.entries,
            output_dir=plan.output_dir,
            split=split_plan.split,
            args=args,
        )
    write_latest_pointer(
        root_dir=plan.output_root,
        kind="tasks",
        output_dir=plan.output_dir,
        metadata={
            "source_dir": str(plan.source_dir),
            "counts": counts,
        },
    )
    return counts


def plan_realtime_pose_task_generation(args: argparse.Namespace) -> TaskGenerationPlan:
    if int(args.seq_len) != REALTIME_POSE_SEQ_LEN:
        raise ValueError(f"当前任务固定 seq_len={REALTIME_POSE_SEQ_LEN}。")
    if int(args.rollout_steps) < 1:
        raise ValueError("rollout_steps 必须大于等于 1。")
    if int(args.samples_per_file) < 1:
        raise ValueError("samples_per_file 必须大于等于 1。")

    source_dir = Path(args.source_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir 不存在: {source_dir}")
    if output_root == source_dir or source_dir.is_relative_to(output_root):
        raise ValueError("output_dir 不能等于或包含 source_dir。")
    output_root.mkdir(parents=True, exist_ok=True)
    run_label = "rtp_140d" if str(args.run_name).lower() in {"", "auto"} else str(args.run_name)
    output_dir = timestamped_child_dir(output_root, run_label)

    entries = read_source_entries(source_dir=source_dir)
    if not entries:
        raise RuntimeError(f"{source_dir} 中没有可用 source。")
    split_dir = Path(args.split_dir).resolve() if args.split_dir else None
    split_plans: list[SplitTaskPlan] = []
    for split in args.splits:
        keys = read_split_keys(split_dir=split_dir, split=str(split))
        split_entries = filter_entries_by_split(entries=entries, split_keys=keys)
        if int(args.limit) > 0:
            split_entries = split_entries[: int(args.limit)]
        if not split_entries:
            raise RuntimeError(f"split={split} 没有匹配 source。")
        split_plans.append(SplitTaskPlan(split=str(split), entries=split_entries))
    return TaskGenerationPlan(
        source_dir=source_dir,
        output_root=output_root,
        output_dir=output_dir,
        split_plans=split_plans,
    )


def generate_split_tasks(
    entries: list[dict],
    output_dir: Path,
    split: str,
    args: argparse.Namespace,
) -> int:
    output_split_dir = output_dir / split
    task_dir = output_split_dir / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_split_dir / "manifest.jsonl"
    written = 0
    rollout_steps = int(args.rollout_steps)
    required_frames = REALTIME_POSE_SEQ_LEN + rollout_steps - 1

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for entry in tqdm(entries, desc=f"生成 {split} 140D tasks", unit="source"):
            source_path = Path(entry["source_path"])
            source = load_realtime_source(source_path)
            source_frames = int(source[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY].shape[0])
            if source_frames != int(entry["source_frames"]):
                raise ValueError(
                    f"{source_path} 实际帧数 {source_frames} 与 source manifest 的 "
                    f"{int(entry['source_frames'])} 不一致"
                )
            if source_frames < required_frames:
                if str(args.short_source_policy) == SHORT_SOURCE_POLICY_ERROR:
                    raise ValueError(f"{source_path} 至少需要 {required_frames} 帧，实际为 {source_frames}")
                continue

            source_id = normalize_slashes(entry["stablemotion_split_key"])
            timeline = build_tracker_timeline(source_id, source_frames, global_seed=int(args.seed))
            candidates = candidate_starts_by_scenario(timeline)
            max_start = source_frames - required_frames
            candidates = {
                category: [start for start in starts if start <= max_start]
                for category, starts in candidates.items()
            }
            selection_rng = np.random.default_rng(stable_source_seed(source_id, int(args.seed) + 7919))
            selected = sample_balanced_starts(
                candidates,
                scenario_sample_counts(int(args.samples_per_file), args.scenario_weights),
                selection_rng,
            )
            if not selected:
                continue

            joint_rotations_world = compute_source_joint_rotations_world(source)
            tracker_rotations_world = rotation_6d_to_matrix_np(source["tracker_rot_world_6d"])
            head_yaws = extract_forward_yaw_np(tracker_rotations_world[:, 0])

            for sample_index, (base_scenario, start_frame) in enumerate(selected):
                task_id = make_task_id(split, source_id, sample_index, base_scenario)
                task_rel_path = Path("tasks") / f"{task_id}.npz"
                rollout_task_paths: list[str] = []
                for rollout_step in range(rollout_steps):
                    step_start = start_frame + rollout_step
                    timeline_window = timeline.window(step_start)
                    scenario = classify_tracker_window(
                        timeline_window.configured,
                        timeline_window.measured_valid,
                    ) or base_scenario
                    arrays = build_task_arrays(
                        source=source,
                        source_path=source_path,
                        source_frames=source_frames,
                        joint_rotations_world=joint_rotations_world,
                        head_yaws=head_yaws,
                        timeline_window=timeline_window,
                        start_frame=step_start,
                        scenario=scenario,
                    )
                    rel_path = task_rel_path if rollout_step == 0 else Path("tasks") / f"{task_id}_r{rollout_step:02d}.npz"
                    if rollout_step > 0:
                        rollout_task_paths.append(rel_path.as_posix())
                    save_task_npz(
                        task_path=output_split_dir / rel_path,
                        compress=bool(args.compress_tasks),
                        **arrays,
                        rollout_step=np.int64(rollout_step),
                        max_rollout_steps=np.int64(rollout_steps),
                    )

                manifest_entry = {
                    "task_id": task_id,
                    "task_path": task_rel_path.as_posix(),
                    "split": split,
                    "source_path": str(source_path),
                    "source_relative_path": normalize_slashes(entry["source_relative_path"]),
                    "stablemotion_split_key": source_id,
                    "source_frames": source_frames,
                    "target_fps": float(entry["target_fps"]),
                    "is_mirrored": bool(entry["is_mirrored"]),
                    "start_frame": start_frame,
                    "scenario": base_scenario,
                    "seq_len": REALTIME_POSE_SEQ_LEN,
                    "max_rollout_steps": rollout_steps,
                    "rollout_task_paths": rollout_task_paths,
                }
                manifest_file.write(json.dumps(manifest_entry, ensure_ascii=False, sort_keys=True) + "\n")
                written += 1
                if int(args.manifest_flush_interval) > 0 and written % int(args.manifest_flush_interval) == 0:
                    manifest_file.flush()
    return written


def scenario_sample_counts(samples_per_file: int, weights: list[float] | tuple[float, ...]) -> dict[str, int]:
    """把相对权重确定性分配为总计 `5 * samples_per_file` 个窗口。"""

    from data_loaders.sensor_masking import TRACKER_PATTERN_CATEGORIES

    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (5,) or np.any(values < 0.0) or not np.any(values > 0.0):
        raise ValueError("scenario_weights 必须是五个非负数，且至少一个大于零。")
    total = int(samples_per_file) * 5
    expected = values / values.sum() * total
    counts = np.floor(expected).astype(np.int64)
    remainder = total - int(counts.sum())
    order = np.argsort(-(expected - counts), kind="stable")
    counts[order[:remainder]] += 1
    return {
        category: int(count)
        for category, count in zip(TRACKER_PATTERN_CATEGORIES, counts.tolist())
    }


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


def build_task_arrays(
    source: dict[str, np.ndarray],
    source_path: Path,
    source_frames: int,
    joint_rotations_world: np.ndarray,
    head_yaws: np.ndarray,
    timeline_window,
    start_frame: int,
    scenario: str,
) -> dict[str, np.ndarray]:
    frame_slice = slice(start_frame, start_frame + REALTIME_POSE_SEQ_LEN)
    current_absolute = start_frame + REALTIME_POSE_TARGET_START
    current_head_yaw = float(head_yaws[current_absolute])
    current_head_position = source["tracker_pos_world"][current_absolute, 0].astype(np.float32)
    floor_y = float(source["root_pos_world"][current_absolute, 1])

    pose_window = build_pose_target_np(
        joint_rotations_world[frame_slice],
        source["root_yaw"][frame_slice],
        current_head_yaw,
    )
    tracker_window = build_tracker_window_np(
        tracker_pos_world=source["tracker_pos_world"][frame_slice],
        tracker_rot_world_6d=source["tracker_rot_world_6d"][frame_slice],
        current_head_pos_world=current_head_position,
        floor_y=floor_y,
        current_head_yaw=current_head_yaw,
        configured=timeline_window.configured,
        measured_valid=timeline_window.measured_valid,
        missing_age_norm=timeline_window.missing_age_norm,
    )
    known_target, known_mask = build_known_target_np(
        current_tracker_features=tracker_window[REALTIME_POSE_TARGET_START],
        current_head_yaw=current_head_yaw,
        current_tracker_rot_world_6d=source["tracker_rot_world_6d"][current_absolute],
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
    root_head = (yaw_inv @ (source["root_pos_world"][current_absolute].astype(np.float64) - origin)).astype(np.float32)

    return {
        "pose_history": pose_window[:REALTIME_POSE_HISTORY_LENGTH],
        "tracker_window": tracker_window,
        "current_target": pose_window[REALTIME_POSE_TARGET_START],
        "known_target": known_target,
        "known_mask": known_mask,
        "valid_frame_mask": np.ones(REALTIME_POSE_HISTORY_LENGTH, dtype=bool),
        "target_joints_head_ref": joints_head[REALTIME_POSE_TARGET_START],
        "prev_joints_head_ref": joints_head[REALTIME_POSE_TARGET_START - 1],
        "target_root_position_head_ref": root_head,
        "target_root_yaw_world": np.asarray(source["root_yaw"][current_absolute], dtype=np.float32),
        "target_hip_height": np.asarray(source["pelvis_height"][current_absolute, 0], dtype=np.float32),
        "current_head_yaw_world": np.asarray(current_head_yaw, dtype=np.float32),
        "current_head_position_world": current_head_position,
        "floor_y": np.asarray(floor_y, dtype=np.float32),
        "joint_offsets_parent": source["joint_offsets_parent"].astype(np.float32),
        "joint_rest_local_rotations_6d": source["joint_rest_local_rotations_6d"].astype(np.float32),
        "configured": timeline_window.configured,
        "measured_valid": timeline_window.measured_valid,
        "missing_age": timeline_window.missing_age,
        "scenario": np.asarray(scenario),
        "source_path": np.asarray(str(source_path)),
        "start_frame": np.int64(start_frame),
        "target_start": np.int64(REALTIME_POSE_TARGET_START),
        "target_length": np.int64(REALTIME_POSE_TARGET_LENGTH),
        "valid_length": np.int64(REALTIME_POSE_SEQ_LEN),
        "source_frames": np.int64(source_frames),
        "seq_len": np.int64(REALTIME_POSE_SEQ_LEN),
    }


def load_realtime_source(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        validate_realtime_source_arrays(data, path=path)
        # 主任务只载入 FK、Tracker 与评估标签真正需要的数组。旧 Root XZ、
        # heading delta 和 stationary_prob_5 继续留在可复用 source 中，但不进入 task 内存。
        task_source_fields = (
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
        source = {key: np.asarray(data[key]).copy() for key in task_source_fields}
    return source


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
                if not path.is_absolute() and not path.exists():
                    path = source_dir / path
                relative = str(entry.get("source_relative_path") or path.name)
                if "frames" not in entry or "target_fps" not in entry:
                    raise KeyError(f"{manifest_path} 的 source 记录缺少 frames/target_fps: {relative}")
                entries.append(
                    {
                        "source_path": str(path),
                        "source_relative_path": normalize_slashes(relative),
                        "stablemotion_split_key": normalize_split_key(
                            str(entry.get("stablemotion_split_key") or Path(relative).with_suffix(".npy"))
                        ),
                        "source_frames": int(entry["frames"]),
                        "target_fps": float(entry["target_fps"]),
                        "is_mirrored": bool(entry.get("is_mirrored", normalize_slashes(relative).startswith("M/"))),
                    }
                )
        return entries

    for path in sorted(source_dir.rglob("*.npz")):
        relative = path.relative_to(source_dir).as_posix()
        with np.load(path, allow_pickle=False) as data:
            metadata = load_realtime_metadata(data, path)
            source_frames = validate_realtime_source_arrays(data, metadata=metadata, path=path)
        entries.append(
            {
                "source_path": str(path),
                "source_relative_path": relative,
                "stablemotion_split_key": normalize_split_key(Path(relative).with_suffix(".npy").as_posix()),
                "source_frames": source_frames,
                "target_fps": float(metadata["target_fps"]),
                "is_mirrored": bool(metadata.get("is_mirrored", relative.startswith("M/"))),
            }
        )
    return entries


def read_split_keys(split_dir: Path | None, split: str) -> set[str] | None:
    if split_dir is None:
        return None
    path = split_dir / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"找不到 split 文件: {path}")
    return {normalize_split_key(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def filter_entries_by_split(entries: list[dict], split_keys: set[str] | None) -> list[dict]:
    if split_keys is None:
        return entries
    return [entry for entry in entries if normalize_split_key(entry["stablemotion_split_key"]) in split_keys]


def normalize_split_key(raw_key: str) -> str:
    key = normalize_slashes(str(raw_key).strip()).split(",", 1)[0].strip()
    if key.endswith((".npy", ".npz")):
        key = key[:-4]
    return key


def normalize_slashes(path: str) -> str:
    return str(path).replace("\\", "/")


def make_task_id(split: str, source_id: str, sample_index: int, category: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", Path(source_id).stem).strip("_") or "source"
    stem = stem[:24]
    digest = hashlib.sha1(f"{split}:{source_id}:{sample_index}:{category}".encode("utf-8")).hexdigest()[:14]
    safe_category = re.sub(r"[^A-Za-z0-9_]+", "_", category)[:12]
    return f"{stem}_s{sample_index:04d}_{safe_category}_{digest}"


def save_task_npz(task_path: Path, compress: bool, **arrays) -> None:
    task_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = task_path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        if compress:
            np.savez_compressed(file, **arrays)
        else:
            np.savez(file, **arrays)
    temporary.replace(task_path)


def write_task_output_marker(plan: TaskGenerationPlan) -> None:
    marker = {
        "source_dir": str(plan.source_dir),
    }
    (plan.output_dir / TASK_OUTPUT_MARKER).write_text(
        json.dumps(marker, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> dict[str, int]:
    args = build_argument_parser().parse_args(argv)
    counts = generate_realtime_pose_tasks(args)
    for split, count in counts.items():
        print(f"[generate_realtime_pose_tasks] split={split} tasks={count}")
    return counts


if __name__ == "__main__":
    main()
