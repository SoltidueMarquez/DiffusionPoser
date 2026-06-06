from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data_loaders.sensor_masking import (
    POSE_REPRESENTATION_KEY,
    REALTIME_POSE_V2_CONTACT_SCHEMA_NAME,
    SMPL_JOINT_COUNT,
    get_schema_spec,
    validate_pose_representation,
)

from .config import DIFFUSIONPOSER_ROOT


@dataclass(frozen=True)
class ReplayArrays:
    payload: dict[str, Any]
    frame_indices: np.ndarray
    target_features_raw: np.ndarray
    reference_joints_world: np.ndarray
    root_yaw: np.ndarray
    root_pos_world: np.ndarray


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_source_npz(replay_payload: dict[str, Any], replay_json: Path) -> Path:
    metadata = replay_payload.get("metadata") or {}
    source_path = metadata.get("sourcePath") or (metadata.get("sourceMetadata") or {}).get("source_path")
    if not source_path:
        raise KeyError("replay JSON metadata must contain `sourcePath` for source joint_offsets_parent lookup.")

    path = Path(str(source_path))
    candidates = [path] if path.is_absolute() else [DIFFUSIONPOSER_ROOT / path, replay_json.parent / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"source npz not found from replay metadata: {source_path}")


def select_frame_range(frame_count: int, frame_start: int, requested_count: int) -> tuple[int, int]:
    start = int(frame_start)
    if start < 0 or start >= frame_count:
        raise ValueError(f"frame_start must be in [0,{frame_count - 1}], got {frame_start}")
    count = frame_count - start if int(requested_count) <= 0 else int(requested_count)
    if count <= 0 or start + count > frame_count:
        raise ValueError(f"invalid frame range [{start},{start + count}) for frame_count={frame_count}")
    return start, count


def load_replay_arrays(replay_json: Path, frame_start: int, frame_count: int) -> ReplayArrays:
    payload = load_json(replay_json)
    if payload.get("schemaName") != REALTIME_POSE_V2_CONTACT_SCHEMA_NAME:
        raise ValueError(f"replay schemaName must be {REALTIME_POSE_V2_CONTACT_SCHEMA_NAME}, got {payload.get('schemaName')}")

    schema = get_schema_spec(REALTIME_POSE_V2_CONTACT_SCHEMA_NAME)
    validate_pose_representation(
        payload.get("poseRepresentation"),
        schema_name=schema.name,
        source=str(replay_json),
    )
    total_frames = int(payload["frameCount"])
    start, count = select_frame_range(total_frames, frame_start, frame_count)
    feature_len = int(payload.get("targetFeatureLength") or schema.target_dim)
    if feature_len != schema.target_dim:
        raise ValueError(f"targetFeatureLength must be {schema.target_dim}, got {feature_len}")

    target_features = np.asarray(payload["targetFeaturesRaw"], dtype=np.float32).reshape(total_frames, feature_len)
    joints = np.asarray(payload["referenceJointsWorld"], dtype=np.float32).reshape(total_frames, SMPL_JOINT_COUNT, 3)
    root_yaw = np.asarray(payload["rootYaw"], dtype=np.float32).reshape(total_frames)
    root_pos = np.asarray(payload["rootPosWorld"], dtype=np.float32).reshape(total_frames, 3)
    frame_indices = np.arange(start, start + count, dtype=np.int64)
    return ReplayArrays(
        payload=payload,
        frame_indices=frame_indices,
        target_features_raw=target_features[start : start + count],
        reference_joints_world=joints[start : start + count],
        root_yaw=root_yaw[start : start + count],
        root_pos_world=root_pos[start : start + count],
    )


def load_source_offsets(source_npz: Path) -> np.ndarray:
    schema = get_schema_spec(REALTIME_POSE_V2_CONTACT_SCHEMA_NAME)
    with np.load(source_npz, allow_pickle=False) as data:
        if POSE_REPRESENTATION_KEY not in data.files:
            raise KeyError(f"{source_npz} missing `{POSE_REPRESENTATION_KEY}`")
        validate_pose_representation(data[POSE_REPRESENTATION_KEY], schema_name=schema.name, source=str(source_npz))
        if "joint_offsets_parent" not in data.files:
            raise KeyError(f"{source_npz} missing `joint_offsets_parent`")
        offsets = np.asarray(data["joint_offsets_parent"], dtype=np.float32)
    if offsets.shape != (SMPL_JOINT_COUNT, 3):
        raise ValueError(f"joint_offsets_parent must be [24,3], got {offsets.shape}")
    return offsets
