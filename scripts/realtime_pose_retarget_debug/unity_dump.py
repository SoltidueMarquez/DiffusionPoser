from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from data_loaders.realtime_pose_kinematics import SMPL_JOINT_NAMES, rotation_6d_to_matrix_np
from data_loaders.sensor_masking import REALTIME_POSE_V2_CONTACT_SCHEMA_NAME, SMPL_JOINT_COUNT, get_schema_spec

from .replay_io import ReplayArrays, load_json


def vector_payload_to_np(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        return np.asarray([value.get("x", 0.0), value.get("y", 0.0), value.get("z", 0.0)], dtype=np.float64)
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return np.asarray(value[:3], dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


def quaternion_payload_to_matrix(value: Any) -> np.ndarray:
    if not isinstance(value, dict):
        return np.eye(3, dtype=np.float64)
    x = float(value.get("x", 0.0))
    y = float(value.get("y", 0.0))
    z = float(value.get("z", 0.0))
    w = float(value.get("w", 1.0))
    norm = max(math.sqrt(x * x + y * y + z * z + w * w), 1e-8)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    relative = np.asarray(a, dtype=np.float64).T @ np.asarray(b, dtype=np.float64)
    cos_angle = (float(np.trace(relative)) - 1.0) * 0.5
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cos_angle)))))


def compare_unity_dump(unity_dump_json: Path, replay: ReplayArrays) -> dict[str, Any]:
    if not unity_dump_json.exists():
        raise FileNotFoundError(f"Unity dump JSON not found: {unity_dump_json}")

    dump = load_json(unity_dump_json)
    frame_by_index = {int(frame): i for i, frame in enumerate(replay.frame_indices.tolist())}
    schema = get_schema_spec(REALTIME_POSE_V2_CONTACT_SCHEMA_NAME)
    pose_rot = rotation_6d_to_matrix_np(replay.target_features_raw[:, schema.body_pose_slice()].reshape(-1, SMPL_JOINT_COUNT, 6))

    compared = 0
    max_angle = 0.0
    max_forward_angle = 0.0
    worst = {"frameIndex": -1, "boneIndex": -1, "boneName": ""}
    rows: list[dict[str, Any]] = []
    for frame in dump.get("frames", []):
        frame_index = int(frame.get("frameIndex", -1))
        if frame_index not in frame_by_index:
            continue
        local_frame = frame_by_index[frame_index]
        for bone in frame.get("bones", []):
            bone_index = int(bone.get("boneIndex", -1))
            if bone_index < 0 or bone_index >= SMPL_JOINT_COUNT:
                continue
            if bone.get("decodedLocalRotation") is not None:
                unity_matrix = quaternion_payload_to_matrix(bone.get("decodedLocalRotation"))
            else:
                unity_matrix = np.stack(
                    [
                        vector_payload_to_np(bone.get("decodedLocalRight")),
                        vector_payload_to_np(bone.get("decodedLocalUp")),
                        vector_payload_to_np(bone.get("decodedLocalForward")),
                    ],
                    axis=-1,
                )
            python_matrix = pose_rot[local_frame, bone_index]
            angle = rotation_angle_degrees(python_matrix, unity_matrix)
            forward_dot = float(np.dot(python_matrix[:, 2], unity_matrix[:, 2]))
            forward_angle = float(math.degrees(math.acos(max(-1.0, min(1.0, forward_dot)))))
            compared += 1
            if angle > max_angle:
                max_angle = angle
                worst = {
                    "frameIndex": frame_index,
                    "boneIndex": bone_index,
                    "boneName": bone.get("boneName", SMPL_JOINT_NAMES[bone_index]),
                }
            max_forward_angle = max(max_forward_angle, forward_angle)
            rows.append(
                {
                    "frameIndex": frame_index,
                    "playbackMode": frame.get("playbackMode", ""),
                    "boneIndex": bone_index,
                    "boneName": bone.get("boneName", SMPL_JOINT_NAMES[bone_index]),
                    "localRotationAngleDeg": angle,
                    "localForwardAngleDeg": forward_angle,
                }
            )

    return {
        "path": str(unity_dump_json),
        "comparedBoneFrames": compared,
        "maxLocalRotationAngleDeg": max_angle,
        "maxLocalForwardAngleDeg": max_forward_angle,
        "decoderLooksAligned": bool(compared > 0 and max_angle < 0.1),
        "worstLocalRotation": worst,
        "rows": rows,
    }
