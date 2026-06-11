from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from data_loaders.generate_realtime_pose_tasks import clip_source, load_realtime_source, normalize_slashes, save_task_npz
from data_loaders.realtime_pose_contract import validate_realtime_task_contract, validate_root_y0_invariants
from data_loaders.sensor_masking import (
    LEGACY_BODY_POSE_PARENT_KEY,
    POSE_REPRESENTATION_BODY_FBX_LOCAL_DELTA_6D,
    POSE_REPRESENTATION_KEY,
    REALTIME_POSE_INPUT_DIM,
    REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_LENGTH,
    REALTIME_POSE_TARGET_START,
    TASK_MODE_REALTIME_POSE,
    TRACKER_NAMES,
    TRACKER_PATTERN_CATEGORIES,
    TASK_MASK_POLICY_FIXED_PATTERNS,
    TASK_MASK_POLICY_FULL,
    create_realtime_inpaint_mask,
    get_schema_spec,
    make_tracker_pattern,
    repeat_pattern_sensor_valid,
    validate_pose_representation,
    validate_sensor_valid,
)


SENSOR_NAMES = tuple(TRACKER_NAMES)
KEYFRAME_TARGETS = ("root",) + SENSOR_NAMES
DEFAULT_API_FRAME_LIMIT = 240


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_token(value: str, fallback: str = "item") -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return token or fallback


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSONL: {exc}") from exc


def normalize_relative_path(value: str) -> str:
    return value.replace("\\", "/").strip("/")


def resolve_manifest_file(base_dir: Path, value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else base_dir / path


def file_stat_payload(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def load_task_npz(path: Path) -> dict[str, np.ndarray]:
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    with np.load(path, allow_pickle=False) as data:
        task = {key: data[key].copy() for key in data.files}
    validate_realtime_task_contract(task, schema=schema, source=str(path))
    validate_sensor_valid(task["sensor_valid"])
    return task


def load_stream_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key].copy() for key in data.files}
    return payload


def validate_realtime_motion_arrays(payload: dict[str, np.ndarray], path: Path | None = None) -> None:
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    label = f"{path} " if path else ""
    if LEGACY_BODY_POSE_PARENT_KEY in payload:
        raise ValueError(f"{label}contains legacy {LEGACY_BODY_POSE_PARENT_KEY}; regenerate realtime_pose data.")
    if POSE_REPRESENTATION_KEY not in payload:
        raise KeyError(f"{label}missing `{POSE_REPRESENTATION_KEY}`")
    validate_pose_representation(payload[POSE_REPRESENTATION_KEY], schema_name=schema.name, source=str(path or "payload"))
    frame_count = int(np.asarray(payload[schema.body_pose_key]).shape[0])
    expected = {
        schema.body_pose_key: (frame_count, 144),
        "root_pos_world": (frame_count, 3),
        "root_yaw": (frame_count,),
        schema.root_heading_delta_key: (frame_count, 2),
        "tracker_pos_world": (frame_count, 6, 3),
        "tracker_rot_world_6d": (frame_count, 6, 6),
        "joints_world": (frame_count, 24, 3),
        "joint_offsets_parent": (24, 3),
    }
    if schema.supports_root_motion:
        expected["root_delta_xz_ref"] = (frame_count, 2)
        expected[schema.pelvis_height_key] = (frame_count, 1)
    if schema.supports_contact:
        expected["foot_contact"] = (frame_count, 2)
    if schema.pose_representation == POSE_REPRESENTATION_BODY_FBX_LOCAL_DELTA_6D:
        expected["joint_rest_local_rotations_6d"] = (24, 6)
    for key, shape in expected.items():
        if key not in payload:
            raise KeyError(f"{label}missing `{key}`")
        if tuple(np.asarray(payload[key]).shape) != shape:
            raise ValueError(f"{label}{key} must be {shape}, got {np.asarray(payload[key]).shape}")
    validate_root_y0_invariants(payload, schema=schema, source=str(path or "payload"))


def array_to_list(array: np.ndarray) -> list:
    return np.asarray(array).astype(float).tolist()


def build_realtime_pose_frames(
    *,
    asset_id: str,
    track_id: str,
    motion: dict[str, np.ndarray],
    task: dict[str, np.ndarray] | None,
    start: int,
    count: int,
    fps: float,
    frame_offset: int = 0,
) -> dict[str, Any]:
    validate_realtime_motion_arrays(motion)
    frame_count = int(motion[get_schema_spec(REALTIME_POSE_SCHEMA_NAME).body_pose_key].shape[0])
    start_index = max(0, int(start) + int(frame_offset))
    count_value = max(1, min(int(count), DEFAULT_API_FRAME_LIMIT))
    end_index = min(frame_count, start_index + count_value)
    if start_index >= end_index:
        raise ValueError(f"invalid frame range start={start}, offset={frame_offset}, count={count}, T={frame_count}")

    sensor_valid = task.get("sensor_valid").astype(bool) if task is not None and "sensor_valid" in task else None
    inpaint_mask = task.get("inpaint_mask").astype(bool) if task is not None and "inpaint_mask" in task else None
    frames = []
    for frame_index in range(start_index, end_index):
        frames.append(
            {
                "asset_id": asset_id,
                "track_id": track_id,
                "frame": int(frame_index - int(frame_offset)),
                "source_frame": int(frame_index),
                "time": float(frame_index / fps),
                "root": array_to_list(motion["root_pos_world"][frame_index]),
                "root_yaw": float(motion["root_yaw"][frame_index]),
                "trackers": array_to_list(motion["tracker_pos_world"][frame_index]),
                "joints": array_to_list(motion["joints_world"][frame_index]),
                "sensor_valid": array_to_list(sensor_valid[frame_index]) if sensor_valid is not None else [True] * 6,
                "inpaint_target": bool(inpaint_mask[frame_index, :REALTIME_POSE_INPUT_DIM].any()) if inpaint_mask is not None else False,
                "valid": True,
            }
        )
    return {
        "schema_name": "realtime_pose_studio_frames_v2",
        "asset_id": asset_id,
        "track_id": track_id,
        "start": int(start),
        "frame_offset": int(frame_offset),
        "count": len(frames),
        "frame_count": frame_count,
        "fps": float(fps),
        "frames": frames,
    }


def write_realtime_task_dataset(
    *,
    source: dict[str, np.ndarray],
    source_label: str,
    source_path: Path,
    source_relative_path: str,
    project_id: str,
    output_dir: Path,
    request: dict[str, Any],
) -> dict[str, Any]:
    validate_realtime_motion_arrays(source)
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    frame_count = int(source[schema.body_pose_key].shape[0])
    target_frames = resolve_target_frames(request=request, frame_count=frame_count)
    split = safe_token(str(request.get("split") or "train"), fallback="train")
    export_name = request.get("export_name") or f"{safe_token(project_id)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    export_dir = output_dir / safe_token(str(export_name), fallback="realtime_pose_tasks")
    if export_dir.exists():
        raise FileExistsError(f"export directory already exists: {export_dir}")

    pattern_categories = resolve_pattern_categories(request)
    mask_policy = TASK_MASK_POLICY_FULL if pattern_categories == ["full-trackers"] else TASK_MASK_POLICY_FIXED_PATTERNS
    rng = np.random.default_rng(int(request.get("seed", 10)))
    split_dir = export_dir / split
    task_dir = split_dir / "tasks"
    task_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = split_dir / "manifest.jsonl"
    tasks: list[dict[str, Any]] = []

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for target_frame in target_frames:
            start_frame = int(target_frame) - REALTIME_POSE_TARGET_START
            clip = clip_source(source, start_frame=start_frame, seq_len=REALTIME_POSE_SEQ_LEN)
            task_arrays = dict(clip)
            task_arrays.update(
                {
                    "schema_name": np.asarray(schema.name),
                    "task_format": np.asarray(schema.task_format),
                    POSE_REPRESENTATION_KEY: np.asarray(schema.pose_representation),
                    "root_y_policy": np.asarray(schema.root_y_policy),
                    "pelvis_height_mode": np.asarray(schema.pelvis_height_mode),
                }
            )
            for pattern_index, category in enumerate(pattern_categories):
                pattern = make_tracker_pattern(category, rng)
                sensor_valid = repeat_pattern_sensor_valid(pattern, seq_len=REALTIME_POSE_SEQ_LEN)
                validate_sensor_valid(sensor_valid)
                task_id = f"{safe_token(project_id)}_f{target_frame:06d}_p{pattern_index:02d}_{safe_token(category)}"
                task_rel_path = Path("tasks") / f"{task_id}.npz"
                task_path = split_dir / task_rel_path
                save_task_npz(
                    task_path=task_path,
                    compress=False,
                    **task_arrays,
                    sensor_valid=sensor_valid,
                    inpaint_mask=create_realtime_inpaint_mask(schema_name=schema.name),
                    start_frame=np.int64(start_frame),
                    target_start=np.int64(REALTIME_POSE_TARGET_START),
                    target_length=np.int64(REALTIME_POSE_TARGET_LENGTH),
                    valid_length=np.int64(REALTIME_POSE_SEQ_LEN),
                    source_frames=np.int64(frame_count),
                    seq_len=np.int64(REALTIME_POSE_SEQ_LEN),
                )
                manifest_entry = {
                    "task_id": task_id,
                    "task_path": task_rel_path.as_posix(),
                    "split": split,
                    "source_path": str(source_path),
                    "source_relative_path": normalize_slashes(source_relative_path),
                    "stablemotion_split_key": normalize_slashes(source_relative_path or source_path.with_suffix(".npy").name),
                    "start_frame": start_frame,
                    "valid_length": REALTIME_POSE_SEQ_LEN,
                    "source_frames": frame_count,
                    "seq_len": REALTIME_POSE_SEQ_LEN,
                    "feature_dim": schema.feature_dim,
                    "task_format": schema.task_format,
                    "schema_name": schema.name,
                    POSE_REPRESENTATION_KEY: schema.pose_representation,
                    "root_y_policy": schema.root_y_policy,
                    "pelvis_height_mode": schema.pelvis_height_mode,
                    "task_mode": TASK_MODE_REALTIME_POSE,
                    "target_start": REALTIME_POSE_TARGET_START,
                    "target_length": REALTIME_POSE_TARGET_LENGTH,
                    "mask_policy": mask_policy,
                    "tracker_pattern": category,
                    "tracker_pattern_detail": pattern.to_dict(),
                    "edit_project_id": project_id,
                }
                manifest_file.write(json.dumps(manifest_entry, ensure_ascii=False, sort_keys=True) + "\n")
                tasks.append({"task_id": task_id, "target_frame": int(target_frame), "task_path": str(task_path)})

    dataset_index = {
        "schema_name": "realtime_pose_studio_export_v2",
        "pose_representation": schema.pose_representation,
        "created_at": utc_now(),
        "export_dir": str(export_dir),
        "split": split,
        "manifest_path": str(manifest_path),
        "task_count": len(tasks),
        "tasks": tasks,
        "source": {
            "label": source_label,
            "source_path": str(source_path),
            "source_relative_path": source_relative_path,
            "pose_representation": schema.pose_representation,
        },
        "project_id": project_id,
        "mask_policy": mask_policy,
        "tracker_patterns": pattern_categories,
    }
    write_json(export_dir / "dataset_index.json", dataset_index)
    return dataset_index


def resolve_target_frames(request: dict[str, Any], frame_count: int) -> list[int]:
    if request.get("target_frames"):
        frames = sorted({int(frame) for frame in request["target_frames"]})
    else:
        start = int(request.get("frame_start", REALTIME_POSE_TARGET_START))
        end = int(request.get("frame_end", frame_count - 1))
        stride = max(1, int(request.get("stride", 1)))
        frames = list(range(start, end + 1, stride))
    invalid = [frame for frame in frames if frame < REALTIME_POSE_TARGET_START or frame >= frame_count]
    if invalid:
        raise ValueError(f"target_frames must be in [{REALTIME_POSE_TARGET_START}, {frame_count - 1}], invalid={invalid}")
    if not frames:
        raise ValueError("no export target frames")
    return frames


def resolve_pattern_categories(request: dict[str, Any]) -> list[str]:
    value = request.get("tracker_pattern") or request.get("tracker_patterns") or "full-trackers"
    if isinstance(value, str):
        categories = [value]
    else:
        categories = [str(item) for item in value]
    if "all" in categories:
        categories = list(TRACKER_PATTERN_CATEGORIES)
    unknown = [category for category in categories if category not in TRACKER_PATTERN_CATEGORIES]
    if unknown:
        raise ValueError(f"unknown tracker pattern categories: {unknown}")
    return categories
