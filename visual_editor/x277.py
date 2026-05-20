from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from data_loaders.sensor_masking import (
    CONTACT_DIM,
    CONTACT_START,
    CURRENT277_SCHEMA_NAME,
    HISTORY_CONTEXT_FRAMES,
    LAST_FRAME_RECONSTRUCTION_SEQ_LEN,
    MODEL_INPUT_DIM,
    SENSOR_LABEL_DIM,
    SENSOR_NAMES,
    TASK_MODE_FULL_RECONSTRUCTION_CURRENT,
    X277_FEATURE_DIM,
    apply_sensor_missing_interval,
    mark_current_reconstruction_targets,
)
from sample.visualization import (
    KINEMATIC_CHAINS,
    SMPL_JOINT_NAMES,
    build_smpl_like_joints_from_tracker_points,
    decode_x277_tracker_positions,
    make_yaw_rotation,
)

TRACKER_TARGETS = tuple(SENSOR_NAMES)
KEYFRAME_TARGETS = ("root",) + TRACKER_TARGETS
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


def path_identity(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path.absolute())


def file_stat_payload(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path_identity(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def resolve_manifest_file(base_dir: Path, value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else base_dir / path


def validate_x277_array(x277: np.ndarray, path: Path | None = None) -> np.ndarray:
    x = np.asarray(x277, dtype=np.float32)
    location = f"{path} " if path else ""
    if x.ndim != 2 or x.shape[1] != X277_FEATURE_DIM:
        raise ValueError(f"{location}x277 must be [T, {X277_FEATURE_DIM}], got {tuple(x.shape)}")
    if x.shape[0] <= 0:
        raise ValueError(f"{location}x277 has no valid frames")
    if not np.isfinite(x).all():
        raise ValueError(f"{location}x277 contains NaN or Inf")
    return x.copy()


def load_x277_from_npz(path: Path, key: str = "x") -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            raise KeyError(f"{path} missing `{key}`")
        return validate_x277_array(data[key], path=path)


def load_task_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        task = {key: data[key].copy() for key in data.files}
    required = {"x277", "sensor_missing_labels", "inpaint_mask", "start_frame", "valid_length", "source_frames", "seq_len"}
    missing = sorted(required.difference(task))
    if missing:
        raise KeyError(f"{path} missing materialized task fields: {missing}")
    validate_x277_array(task["x277"], path=path)
    return task


def load_stream_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key].copy() for key in data.files}
    for key in ("reference_motion", "conditioned_motion", "reconstructed_motion"):
        if key not in payload:
            raise KeyError(f"{path} missing `{key}`")
        validate_x277_array(payload[key], path=path)
    return payload


def array_to_list(array: np.ndarray) -> list:
    return np.asarray(array).tolist()


def decode_root_trajectory(x277: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = validate_x277_array(x277)
    root_position = np.zeros(3, dtype=np.float64)
    root_yaw = 0.0
    positions: list[np.ndarray] = []
    yaws: list[float] = []
    for row in x:
        prev_root_rotation = make_yaw_rotation(np.asarray([root_yaw], dtype=np.float64))[0]
        delta_xz = row[270:272].astype(np.float64)
        delta_world = np.asarray([delta_xz[0], 0.0, delta_xz[1]], dtype=np.float64) @ prev_root_rotation.T
        root_position = root_position + delta_world
        root_yaw = root_yaw + math.radians(float(row[272]))
        positions.append(root_position.copy())
        yaws.append(root_yaw)
    return np.stack(positions, axis=0).astype(np.float32), np.asarray(yaws, dtype=np.float32)


def encode_root_deltas(x277: np.ndarray, root_positions: np.ndarray, root_yaws: np.ndarray) -> None:
    positions = np.asarray(root_positions, dtype=np.float64)
    yaws = np.asarray(root_yaws, dtype=np.float64)
    if positions.shape != (x277.shape[0], 3):
        raise ValueError(f"root_positions must be [T, 3], got {positions.shape}")
    previous_position = np.zeros(3, dtype=np.float64)
    previous_yaw = 0.0
    for frame_index, position in enumerate(positions):
        delta_world = position - previous_position
        prev_rotation = make_yaw_rotation(np.asarray([previous_yaw], dtype=np.float64))[0]
        delta_local = np.asarray([delta_world[0], 0.0, delta_world[2]], dtype=np.float64) @ prev_rotation
        x277[frame_index, 270:272] = delta_local[[0, 2]].astype(np.float32)
        previous_position = position
        previous_yaw = float(yaws[frame_index])


def tracker_world_to_root_local(tracker_world: np.ndarray, root_positions: np.ndarray, root_yaws: np.ndarray) -> np.ndarray:
    world = np.asarray(tracker_world, dtype=np.float64)
    roots = np.asarray(root_positions, dtype=np.float64)
    yaws = np.asarray(root_yaws, dtype=np.float64)
    if world.shape != (roots.shape[0], SENSOR_LABEL_DIM, 3):
        raise ValueError(f"tracker_world must be [T, 6, 3], got {world.shape}")
    local = np.zeros_like(world)
    for frame_index in range(world.shape[0]):
        rotation = make_yaw_rotation(np.asarray([float(yaws[frame_index])], dtype=np.float64))[0]
        local[frame_index] = (world[frame_index] - roots[frame_index][None]) @ rotation
    return local.astype(np.float32)


def skeletons_from_trackers(tracker_world: np.ndarray) -> np.ndarray:
    trackers = np.asarray(tracker_world, dtype=np.float64)
    if trackers.ndim != 3 or trackers.shape[1:] != (SENSOR_LABEL_DIM, 3):
        raise ValueError(f"tracker_world must be [T, 6, 3], got {trackers.shape}")
    return np.stack([build_smpl_like_joints_from_tracker_points(frame).astype(np.float32) for frame in trackers], axis=0)


def encode_body_velocity_from_skeletons(
    x277: np.ndarray,
    joints_world: np.ndarray,
    root_yaws: np.ndarray,
    fps: float,
) -> None:
    if fps <= 0:
        raise ValueError("fps must be positive")
    joints = np.asarray(joints_world, dtype=np.float64)
    if joints.shape[0] != x277.shape[0] or joints.shape[1:] != (len(SMPL_JOINT_NAMES), 3):
        raise ValueError(f"joints_world must be [T, 24, 3], got {joints.shape}")
    yaws = np.asarray(root_yaws, dtype=np.float64)
    velocities = np.zeros((x277.shape[0], len(SMPL_JOINT_NAMES), 3), dtype=np.float32)
    for frame_index in range(1, x277.shape[0]):
        rotation = make_yaw_rotation(np.asarray([float(yaws[frame_index])], dtype=np.float64))[0]
        world_velocity = (joints[frame_index] - joints[frame_index - 1]) * float(fps)
        velocities[frame_index] = (world_velocity @ rotation).astype(np.float32)
    x277[:, 144:216] = velocities.reshape(x277.shape[0], -1)


def interpolate_track(base_track: np.ndarray, keyframes: list[dict[str, Any]]) -> np.ndarray:
    track = np.asarray(base_track, dtype=np.float32).copy()
    valid_keyframes = sorted(
        (
            {"frame": int(item["frame"]), "position": np.asarray(item["position"], dtype=np.float32)}
            for item in keyframes
            if "frame" in item and "position" in item
        ),
        key=lambda item: item["frame"],
    )
    valid_keyframes = [item for item in valid_keyframes if 0 <= item["frame"] < track.shape[0]]
    if not valid_keyframes:
        return track
    for item in valid_keyframes:
        track[item["frame"]] = item["position"]
    if len(valid_keyframes) == 1:
        return track
    for left, right in zip(valid_keyframes[:-1], valid_keyframes[1:]):
        start = left["frame"]
        end = right["frame"]
        if end <= start:
            continue
        alpha = np.linspace(0.0, 1.0, end - start + 1, dtype=np.float32)[:, None]
        track[start : end + 1] = left["position"][None] * (1.0 - alpha) + right["position"][None] * alpha
    return track


def build_x277_frames(
    *,
    asset_id: str,
    track_id: str,
    x277: np.ndarray,
    task: dict[str, np.ndarray] | None,
    start: int,
    count: int,
    fps: float,
    frame_offset: int = 0,
) -> dict[str, Any]:
    x = validate_x277_array(x277)
    start_index = max(0, int(start) + int(frame_offset))
    count_value = max(1, min(int(count), DEFAULT_API_FRAME_LIMIT))
    end_index = min(x.shape[0], start_index + count_value)
    if start_index >= end_index:
        raise ValueError(f"invalid frame range start={start}, offset={frame_offset}, count={count}, T={x.shape[0]}")

    root_positions, root_yaws = decode_root_trajectory(x)
    tracker_world = decode_x277_tracker_positions(x)
    joints_world = skeletons_from_trackers(tracker_world)
    sensor_missing_labels = task.get("sensor_missing_labels").astype(bool) if task and "sensor_missing_labels" in task else None
    inpaint_mask = task.get("inpaint_mask").astype(bool) if task and "inpaint_mask" in task else None
    valid_frame_mask = task.get("valid_frame_mask").astype(bool) if task and "valid_frame_mask" in task else None

    frames = []
    for frame_index in range(start_index, end_index):
        frames.append(
            {
                "asset_id": asset_id,
                "track_id": track_id,
                "frame": int(frame_index - int(frame_offset)),
                "source_frame": int(frame_index),
                "time": float(frame_index / fps),
                "root": array_to_list(root_positions[frame_index]),
                "root_yaw": float(root_yaws[frame_index]),
                "trackers": array_to_list(tracker_world[frame_index]),
                "joints": array_to_list(joints_world[frame_index]),
                "contact": array_to_list(x[frame_index, CONTACT_START : CONTACT_START + CONTACT_DIM]),
                "sensor_missing_labels": array_to_list(sensor_missing_labels[frame_index]) if sensor_missing_labels is not None else [],
                "inpaint_target": bool(inpaint_mask[frame_index, :X277_FEATURE_DIM].any()) if inpaint_mask is not None else False,
                "valid": bool(valid_frame_mask[frame_index]) if valid_frame_mask is not None else True,
            }
        )
    return {
        "asset_id": asset_id,
        "track_id": track_id,
        "start": int(start),
        "frame_offset": int(frame_offset),
        "count": len(frames),
        "frame_count": int(x.shape[0]),
        "fps": float(fps),
        "frames": frames,
    }


def build_amass_frames(
    *,
    asset_id: str,
    track_id: str,
    joints: np.ndarray,
    start: int,
    count: int,
    fps: float,
    frame_offset: int = 0,
) -> dict[str, Any]:
    values = np.asarray(joints, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (len(SMPL_JOINT_NAMES), 3):
        raise ValueError(f"AMASS joints must be [T, 24, 3], got {values.shape}")
    start_index = max(0, int(start) + int(frame_offset))
    count_value = max(1, min(int(count), DEFAULT_API_FRAME_LIMIT))
    end_index = min(values.shape[0], start_index + count_value)
    if start_index >= end_index:
        raise ValueError(f"invalid AMASS frame range start={start}, count={count}, T={values.shape[0]}")
    tracker_indices = [15, 20, 21, 0, 10, 11]
    frames = []
    for frame_index in range(start_index, end_index):
        frame_joints = values[frame_index]
        root = frame_joints[0]
        frames.append(
            {
                "asset_id": asset_id,
                "track_id": track_id,
                "frame": int(frame_index - int(frame_offset)),
                "source_frame": int(frame_index),
                "time": float(frame_index / fps),
                "root": array_to_list(root),
                "root_yaw": 0.0,
                "trackers": array_to_list(frame_joints[tracker_indices]),
                "joints": array_to_list(frame_joints),
                "contact": [0.0, 0.0, 0.0, 0.0],
                "sensor_missing_labels": [],
                "inpaint_target": False,
                "valid": True,
            }
        )
    return {
        "asset_id": asset_id,
        "track_id": track_id,
        "start": int(start),
        "frame_offset": int(frame_offset),
        "count": len(frames),
        "frame_count": int(values.shape[0]),
        "fps": float(fps),
        "frames": frames,
    }


def resolve_target_frames(request: dict[str, Any], frame_count: int) -> list[int]:
    if "target_frames" in request and request["target_frames"]:
        frames = sorted({int(frame) for frame in request["target_frames"]})
    else:
        start = int(request.get("frame_start", HISTORY_CONTEXT_FRAMES))
        end = int(request.get("frame_end", frame_count - 1))
        stride = max(1, int(request.get("stride", 1)))
        frames = list(range(start, end + 1, stride))
    invalid = [frame for frame in frames if frame < HISTORY_CONTEXT_FRAMES or frame >= frame_count]
    if invalid:
        raise ValueError(f"target_frames must be in [{HISTORY_CONTEXT_FRAMES}, {frame_count - 1}], invalid={invalid}")
    if not frames:
        raise ValueError("no export target frames")
    return frames


def parse_sensor_indices(values: Any) -> list[int]:
    if values is None:
        return []
    if isinstance(values, (str, int)):
        values = [values]
    indices: list[int] = []
    for value in values:
        if isinstance(value, str) and not value.isdigit():
            if value not in SENSOR_NAMES:
                raise ValueError(f"unknown sensor name: {value}")
            index = SENSOR_NAMES.index(value)
        else:
            index = int(value)
        if index < 0 or index >= SENSOR_LABEL_DIM:
            raise ValueError(f"sensor index out of range: {index}")
        if index not in indices:
            indices.append(index)
    return indices


def write_current277_task_dataset(
    *,
    edited: np.ndarray,
    source_label: str,
    source_path: Path,
    source_relative_path: str,
    project_id: str,
    output_dir: Path,
    request: dict[str, Any],
) -> dict[str, Any]:
    x = validate_x277_array(edited)
    target_frames = resolve_target_frames(request=request, frame_count=x.shape[0])
    missing_sensors = parse_sensor_indices(request.get("missing_sensors", []))
    split = safe_token(str(request.get("split") or "train"), fallback="train")
    export_name = request.get("export_name") or f"{safe_token(project_id)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    export_dir = output_dir / safe_token(str(export_name), fallback="edited_dataset")
    if export_dir.exists():
        raise FileExistsError(f"export directory already exists: {export_dir}")

    split_dir = export_dir / split
    task_dir = split_dir / "tasks"
    task_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = split_dir / "manifest.jsonl"
    manifest_entries: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []

    for target_frame in target_frames:
        start_frame = int(target_frame) - HISTORY_CONTEXT_FRAMES
        clip = x[start_frame : target_frame + 1].astype(np.float32, copy=True)
        sensor_missing_labels = np.zeros((LAST_FRAME_RECONSTRUCTION_SEQ_LEN, SENSOR_LABEL_DIM), dtype=bool)
        inpaint_mask = np.zeros((LAST_FRAME_RECONSTRUCTION_SEQ_LEN, MODEL_INPUT_DIM), dtype=bool)
        mark_current_reconstruction_targets(inpaint_mask=inpaint_mask, start=HISTORY_CONTEXT_FRAMES, length=1)
        if missing_sensors:
            apply_sensor_missing_interval(
                sensor_missing_labels=sensor_missing_labels,
                inpaint_mask=inpaint_mask,
                start=HISTORY_CONTEXT_FRAMES,
                length=1,
                sensor_indices=missing_sensors,
            )

        task_id = f"{safe_token(project_id)}_f{int(target_frame):06d}"
        task_rel_path = Path("tasks") / f"{task_id}.npz"
        task_path = split_dir / task_rel_path
        np.savez(
            task_path,
            x277=clip,
            sensor_missing_labels=sensor_missing_labels,
            inpaint_mask=inpaint_mask,
            start_frame=np.int64(start_frame),
            valid_length=np.int64(LAST_FRAME_RECONSTRUCTION_SEQ_LEN),
            source_frames=np.int64(x.shape[0]),
            seq_len=np.int64(LAST_FRAME_RECONSTRUCTION_SEQ_LEN),
        )
        manifest_entry = {
            "task_id": task_id,
            "task_path": task_rel_path.as_posix(),
            "split": split,
            "source_path": str(source_path),
            "source_relative_path": source_relative_path,
            "stablemotion_split_key": source_relative_path or source_path.with_suffix(".npy").name,
            "start_frame": start_frame,
            "valid_length": LAST_FRAME_RECONSTRUCTION_SEQ_LEN,
            "source_frames": int(x.shape[0]),
            "seq_len": LAST_FRAME_RECONSTRUCTION_SEQ_LEN,
            "feature_dim": MODEL_INPUT_DIM,
            "task_format": "materialized_current277_last_frame_reconstruction_v1",
            "schema_name": CURRENT277_SCHEMA_NAME,
            "task_mode": TASK_MODE_FULL_RECONSTRUCTION_CURRENT,
            "target_start": HISTORY_CONTEXT_FRAMES,
            "target_length": 1,
            "missing_intervals": [
                {
                    "start": HISTORY_CONTEXT_FRAMES,
                    "length": 1,
                    "sensor_indices": missing_sensors,
                    "sensor_names": [SENSOR_NAMES[index] for index in missing_sensors],
                }
            ],
            "edit_project_id": project_id,
        }
        manifest_entries.append(manifest_entry)
        tasks.append({"task_id": task_id, "target_frame": int(target_frame), "task_path": str(task_path)})

    with manifest_path.open("w", encoding="utf-8") as file:
        for entry in manifest_entries:
            file.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    dataset_index = {
        "schema_name": "x277_motion_studio_edited_dataset_v1",
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
        },
        "project_id": project_id,
    }
    write_json(export_dir / "dataset_index.json", dataset_index)
    write_json(
        export_dir / "edit_log.json",
        {
            "schema_name": "x277_motion_studio_edit_log_v1",
            "created_at": utc_now(),
            "project_id": project_id,
            "export_request": request,
            "target_frames": target_frames,
            "missing_sensors": missing_sensors,
        },
    )
    return dataset_index
