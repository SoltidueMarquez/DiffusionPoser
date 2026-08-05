"""旧 Unity/Sentis 导出入口，当前不属于受支持的 Python 主链路。

该实现仍绑定旧 feature schema 与旧运行时接口，暂时保留仅供后续迁移参考。
在完成当前 spatiotemporal_dit 契约迁移前，不要把它作为可运行 CLI 或回归契约。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


DIFFUSIONPOSER_ROOT = Path(__file__).resolve().parents[1]
if str(DIFFUSIONPOSER_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSIONPOSER_ROOT))


from data_loaders.generate_realtime_pose_tasks import load_realtime_source  # noqa: E402
from data_loaders.realtime_pose_kinematics import (  # noqa: E402
    SMPL_PARENTS,
    make_yaw_rotation_np,
)
from data_loaders.sensor_masking import (  # noqa: E402
    BODY_POSE_DIM,
    DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    ROOT_DELTA_XZ_DIM,
    ROOT_HEIGHT_DIM,
    ROOT_YAW_DELTA_DIM,
    SMPL_JOINT_COUNT,
    STATIONARY_JOINT_INDICES,
    STATIONARY_JOINT_NAMES,
    STATIONARY_PROB_DIM,
    TRACKER_COUNT,
    TRACKER_NAMES,
    get_schema_spec,
)


DEFAULT_REPLAY_FPS = 60.0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export realtime_pose source as a Unity JSON replay stream.")
    parser.add_argument("--source_npz", required=True, type=str)
    parser.add_argument("--output_json", required=True, type=str)
    parser.add_argument("--schema", default=DEFAULT_REALTIME_POSE_SCHEMA_NAME, type=str)
    parser.add_argument("--fps", default=DEFAULT_REPLAY_FPS, type=float)
    parser.add_argument("--frame_start", default=0, type=int)
    parser.add_argument("--frame_count", default=0, type=int, help="0 means export until the end of the source.")
    parser.add_argument(
        "--identity_6d_rotations",
        action="store_true",
        help="Debug resource: set all replay 6D rotations to identity and rebuild reference joints with source offsets.",
    )
    return parser


def read_source_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        if "metadata" not in data.files:
            return {}
        value = data["metadata"]
    try:
        text = str(value.item())
    except Exception:
        text = str(value)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def load_sensor_valid(path: Path, frame_count: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "sensor_valid" in data.files:
            sensor_valid = np.asarray(data["sensor_valid"], dtype=bool)
        else:
            sensor_valid = np.ones((frame_count, TRACKER_COUNT), dtype=bool)
    expected_shape = (frame_count, TRACKER_COUNT)
    if sensor_valid.shape != expected_shape:
        raise ValueError(f"sensor_valid 应为 {expected_shape}，实际为 {sensor_valid.shape}")
    return sensor_valid


def slice_time(array: np.ndarray, start: int, count: int) -> np.ndarray:
    return np.asarray(array[start : start + count])


def select_frame_range(frame_count: int, frame_start: int, requested_count: int) -> tuple[int, int]:
    start = int(frame_start)
    if start < 0 or start >= frame_count:
        raise ValueError(f"frame_start 必须在 [0,{frame_count - 1}] 内，实际为 {frame_start}")
    count = frame_count - start if int(requested_count) <= 0 else int(requested_count)
    if count <= 0:
        raise ValueError(f"frame_count 必须为正数或 0，实际为 {requested_count}")
    if start + count > frame_count:
        raise ValueError(f"请求帧范围 [{start},{start + count}) 超出源长度 {frame_count}")
    return start, count


def flatten_float(array: np.ndarray) -> list[float]:
    values = np.asarray(array, dtype=np.float32).reshape(-1)
    return [float(value) for value in values.tolist()]


def flatten_int(array: np.ndarray) -> list[int]:
    values = np.asarray(array).reshape(-1)
    return [int(value) for value in values.tolist()]


def build_identity_body_pose_6d(frame_count: int) -> np.ndarray:
    identity_joint = np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    return np.tile(identity_joint, (int(frame_count), SMPL_JOINT_COUNT)).astype(np.float32)


def build_identity_tracker_rotations_6d(frame_count: int) -> np.ndarray:
    identity_joint = np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    return np.tile(identity_joint, (int(frame_count), TRACKER_COUNT, 1)).astype(np.float32)


def compute_identity_6d_reference_joints(
    root_pos_world: np.ndarray,
    root_yaw: np.ndarray,
    joint_offsets_parent: np.ndarray,
) -> np.ndarray:
    """用 source FK 的真实 offsets 生成 identity 6D 对应的 reference joints。

    这里故意保留 root yaw 和 root position，只把所有 joint rotation 置为 identity；
    这样 Unity 里看到的就是当前 source rotation convention 的默认 bind/rest 姿态。
    """

    root_pos = np.asarray(root_pos_world, dtype=np.float64)
    yaw = np.asarray(root_yaw, dtype=np.float64).reshape(-1)
    offsets = np.asarray(joint_offsets_parent, dtype=np.float64)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos_world should be [T,3], got {root_pos.shape}")
    if yaw.shape != (root_pos.shape[0],):
        raise ValueError(f"root_yaw should be [T], got {yaw.shape}")
    if offsets.shape != (SMPL_JOINT_COUNT, 3):
        raise ValueError(f"joint_offsets_parent should be [24,3], got {offsets.shape}")

    yaw_rotations = make_yaw_rotation_np(yaw)
    joints = np.zeros((root_pos.shape[0], SMPL_JOINT_COUNT, 3), dtype=np.float64)
    for joint_index, parent_index in enumerate(SMPL_PARENTS.tolist()):
        if parent_index < 0:
            joints[:, joint_index] = root_pos + np.einsum("tij,j->ti", yaw_rotations, offsets[joint_index])
        else:
            joints[:, joint_index] = joints[:, parent_index] + np.einsum(
                "tij,j->ti",
                yaw_rotations,
                offsets[joint_index],
            )
    return joints.astype(np.float32)


def build_target_features_raw(
    source: dict[str, np.ndarray],
    start: int,
    count: int,
    target_dim: int,
    schema_name: str,
) -> np.ndarray:
    target = np.zeros((count, target_dim), dtype=np.float32)
    schema = get_schema_spec(schema_name)

    body_pose = slice_time(source[schema.body_pose_key], start, count)
    root_yaw_delta = slice_time(source[schema.root_heading_delta_key], start, count)
    root_delta = slice_time(source["root_delta_xz_ref"], start, count)
    pelvis_height = slice_time(source[schema.pelvis_height_key], start, count)
    stationary_prob = slice_time(source["stationary_prob_5"], start, count) if schema.supports_stationary_prob else None

    if body_pose.shape != (count, BODY_POSE_DIM):
        raise ValueError(f"{schema.body_pose_key} should be [{count},{BODY_POSE_DIM}], got {body_pose.shape}")
    if root_yaw_delta.shape != (count, ROOT_YAW_DELTA_DIM):
        raise ValueError(f"{schema.root_heading_delta_key} should be [{count},{ROOT_YAW_DELTA_DIM}], got {root_yaw_delta.shape}")
    if root_delta.shape != (count, ROOT_DELTA_XZ_DIM):
        raise ValueError(f"root_delta_xz_ref 应为 [{count},{ROOT_DELTA_XZ_DIM}]，实际为 {root_delta.shape}")
    if pelvis_height.shape != (count, ROOT_HEIGHT_DIM):
        raise ValueError(f"{schema.pelvis_height_key} should be [{count},{ROOT_HEIGHT_DIM}], got {pelvis_height.shape}")
    if stationary_prob is not None and stationary_prob.shape != (count, STATIONARY_PROB_DIM):
        raise ValueError(f"stationary_prob_5 应为 [{count},{STATIONARY_PROB_DIM}]，实际为 {stationary_prob.shape}")

    target[:, 0:BODY_POSE_DIM] = body_pose
    cursor = BODY_POSE_DIM
    target[:, cursor : cursor + ROOT_YAW_DELTA_DIM] = root_yaw_delta
    cursor += ROOT_YAW_DELTA_DIM
    target[:, cursor : cursor + ROOT_DELTA_XZ_DIM] = root_delta
    cursor += ROOT_DELTA_XZ_DIM
    target[:, cursor : cursor + ROOT_HEIGHT_DIM] = pelvis_height
    cursor += ROOT_HEIGHT_DIM
    if stationary_prob is not None:
        target[:, cursor : cursor + STATIONARY_PROB_DIM] = stationary_prob
        cursor += STATIONARY_PROB_DIM
    if cursor != target_dim:
        raise ValueError(f"target_dim={target_dim} 与 schema target cursor={cursor} 不一致")
    return target


def build_unity_replay_stream_payload(
    source_npz: Path,
    schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    fps: float = DEFAULT_REPLAY_FPS,
    frame_start: int = 0,
    frame_count: int = 0,
    identity_6d_rotations: bool = False,
    source_override: dict[str, np.ndarray] | None = None,
    source_metadata_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schema = get_schema_spec(schema_name)

    path = Path(source_npz).resolve()
    source = source_override if source_override is not None else load_realtime_source(path, schema_name=schema.name)
    total_frames = int(source["tracker_pos_world"].shape[0])
    if "sensor_valid" in source:
        sensor_valid = np.asarray(source["sensor_valid"], dtype=bool)
        expected_shape = (total_frames, TRACKER_COUNT)
        if sensor_valid.shape != expected_shape:
            raise ValueError(f"sensor_valid 应为 {expected_shape}，实际为 {sensor_valid.shape}")
    else:
        sensor_valid = load_sensor_valid(path, total_frames)
    start, count = select_frame_range(total_frames, frame_start, frame_count)
    metadata = source_metadata_override if source_metadata_override is not None else read_source_metadata(path)

    # Unity JsonUtility 对嵌套数组支持很弱，所以所有逐帧张量都按 frame-major 展平成一维数组。
    tracker_positions = slice_time(source["tracker_pos_world"], start, count)
    tracker_rotations = slice_time(source["tracker_rot_world_6d"], start, count)
    valid = slice_time(sensor_valid, start, count).astype(np.int32)
    joints = slice_time(source["joints_world"], start, count)
    root_heading = slice_time(source["root_yaw"], start, count)
    root_pos = slice_time(source["root_pos_world"], start, count)
    stationary_prob = slice_time(source["stationary_prob_5"], start, count) if schema.supports_stationary_prob else None
    target_features_raw = build_target_features_raw(source, start, count, schema.target_dim, schema_name=schema.name)
    if identity_6d_rotations:
        target_features_raw[:, 0:BODY_POSE_DIM] = build_identity_body_pose_6d(count)
        tracker_rotations = build_identity_tracker_rotations_6d(count)
        joints = compute_identity_6d_reference_joints(
            root_pos_world=root_pos,
            root_yaw=root_heading,
            joint_offsets_parent=np.asarray(source["joint_offsets_parent"], dtype=np.float32),
        )

    payload = {
        "schemaName": schema.name,
        "poseRepresentation": schema.pose_representation,
        "fps": float(fps),
        "frameStart": int(start),
        "frameCount": int(count),
        "sourceFrameCount": int(total_frames),
        "trackerCount": TRACKER_COUNT,
        "trackerNames": list(TRACKER_NAMES),
        "jointCount": SMPL_JOINT_COUNT,
        "trackerPositions": flatten_float(tracker_positions),
        "trackerRotations6d": flatten_float(tracker_rotations),
        "sensorValid": flatten_int(valid),
        "referenceJointsWorld": flatten_float(joints),
        "rootHeading": flatten_float(root_heading),
        "rootPosWorld": flatten_float(root_pos),
        "targetFeatureLength": int(schema.target_dim),
        "targetFeaturesRaw": flatten_float(target_features_raw),
        "metadata": {
            "sourcePath": str(path),
            "poseRepresentation": schema.pose_representation,
            "sourceMetadata": metadata,
            "layout": "frame-major-flat",
            "debugIdentity6dRotations": bool(identity_6d_rotations),
            "debugReferenceJointsWorld": "identity_6d_fk_joint_offsets_parent" if identity_6d_rotations else "source_joints_world",
            "trackerPositionsShape": [count, TRACKER_COUNT, 3],
            "trackerRotations6dShape": [count, TRACKER_COUNT, 6],
            "sensorValidShape": [count, TRACKER_COUNT],
            "referenceJointsWorldShape": [count, SMPL_JOINT_COUNT, 3],
            "rootHeadingShape": [count],
            "rootPosWorldShape": [count, 3],
            "targetFeaturesRawShape": [count, int(schema.target_dim)],
        },
    }
    if stationary_prob is not None:
        payload["stationaryProbDim"] = STATIONARY_PROB_DIM
        payload["stationaryProbJointIndices"] = [int(value) for value in STATIONARY_JOINT_INDICES]
        payload["stationaryProbJointNames"] = list(STATIONARY_JOINT_NAMES)
        payload["stationaryProb5"] = flatten_float(stationary_prob)
        payload["metadata"]["stationaryProb5Shape"] = [count, STATIONARY_PROB_DIM]
    return payload


def write_unity_replay_stream(
    source_npz: Path,
    output_json: Path,
    schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    fps: float = DEFAULT_REPLAY_FPS,
    frame_start: int = 0,
    frame_count: int = 0,
    identity_6d_rotations: bool = False,
    source_override: dict[str, np.ndarray] | None = None,
    source_metadata_override: dict[str, Any] | None = None,
) -> Path:
    payload = build_unity_replay_stream_payload(
        source_npz=source_npz,
        schema_name=schema_name,
        fps=fps,
        frame_start=frame_start,
        frame_count=frame_count,
        identity_6d_rotations=identity_6d_rotations,
        source_override=source_override,
        source_metadata_override=source_metadata_override,
    )
    path = Path(output_json).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    return path


def main(argv: list[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    output_path = write_unity_replay_stream(
        source_npz=Path(args.source_npz),
        output_json=Path(args.output_json),
        schema_name=str(args.schema),
        fps=float(args.fps),
        frame_start=int(args.frame_start),
        frame_count=int(args.frame_count),
        identity_6d_rotations=bool(args.identity_6d_rotations),
    )
    print(f"[write_unity_replay_stream] wrote {output_path}")
    return output_path


if __name__ == "__main__":
    main()
