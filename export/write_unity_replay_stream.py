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
    FOOT_CONTACT_DIM,
    REALTIME_POSE_V2_CONTACT_SCHEMA_NAME,
    ROOT_DELTA_XZ_DIM,
    ROOT_HEIGHT_DIM,
    ROOT_YAW_DELTA_DIM,
    SMPL_JOINT_COUNT,
    TRACKER_COUNT,
    TRACKER_NAMES,
    get_schema_spec,
)


DEFAULT_REPLAY_FPS = 60.0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export realtime_pose_v2_contact npz source as a Unity JSON replay stream.")
    parser.add_argument("--source_npz", required=True, type=str)
    parser.add_argument("--output_json", required=True, type=str)
    parser.add_argument("--schema", default=REALTIME_POSE_V2_CONTACT_SCHEMA_NAME, type=str)
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


def build_target_features_raw(source: dict[str, np.ndarray], start: int, count: int, target_dim: int) -> np.ndarray:
    target = np.zeros((count, target_dim), dtype=np.float32)
    schema = get_schema_spec(REALTIME_POSE_V2_CONTACT_SCHEMA_NAME)

    body_pose = slice_time(source[schema.body_pose_key], start, count)
    root_yaw_delta = slice_time(source["root_yaw_delta_sincos"], start, count)
    root_delta = slice_time(source["root_delta_xz_ref"], start, count)
    root_height = slice_time(source["root_height"], start, count)
    foot_contact = slice_time(source["foot_contact"], start, count)

    if body_pose.shape != (count, BODY_POSE_DIM):
        raise ValueError(f"{schema.body_pose_key} should be [{count},{BODY_POSE_DIM}], got {body_pose.shape}")
    if root_yaw_delta.shape != (count, ROOT_YAW_DELTA_DIM):
        raise ValueError(f"root_yaw_delta_sincos 应为 [{count},{ROOT_YAW_DELTA_DIM}]，实际为 {root_yaw_delta.shape}")
    if root_delta.shape != (count, ROOT_DELTA_XZ_DIM):
        raise ValueError(f"root_delta_xz_ref 应为 [{count},{ROOT_DELTA_XZ_DIM}]，实际为 {root_delta.shape}")
    if root_height.shape != (count, ROOT_HEIGHT_DIM):
        raise ValueError(f"root_height 应为 [{count},{ROOT_HEIGHT_DIM}]，实际为 {root_height.shape}")
    if foot_contact.shape != (count, FOOT_CONTACT_DIM):
        raise ValueError(f"foot_contact 应为 [{count},{FOOT_CONTACT_DIM}]，实际为 {foot_contact.shape}")

    target[:, 0:BODY_POSE_DIM] = body_pose
    cursor = BODY_POSE_DIM
    target[:, cursor : cursor + ROOT_YAW_DELTA_DIM] = root_yaw_delta
    cursor += ROOT_YAW_DELTA_DIM
    target[:, cursor : cursor + ROOT_DELTA_XZ_DIM] = root_delta
    cursor += ROOT_DELTA_XZ_DIM
    target[:, cursor : cursor + ROOT_HEIGHT_DIM] = root_height
    cursor += ROOT_HEIGHT_DIM
    target[:, cursor : cursor + FOOT_CONTACT_DIM] = foot_contact
    return target


def build_unity_replay_stream_payload(
    source_npz: Path,
    schema_name: str = REALTIME_POSE_V2_CONTACT_SCHEMA_NAME,
    fps: float = DEFAULT_REPLAY_FPS,
    frame_start: int = 0,
    frame_count: int = 0,
    identity_6d_rotations: bool = False,
) -> dict[str, Any]:
    schema = get_schema_spec(schema_name)
    if schema.name != REALTIME_POSE_V2_CONTACT_SCHEMA_NAME:
        raise ValueError(f"Unity replay stream 只支持 {REALTIME_POSE_V2_CONTACT_SCHEMA_NAME}，实际为 {schema.name}")

    path = Path(source_npz).resolve()
    source = load_realtime_source(path, schema_name=schema.name)
    total_frames = int(source["tracker_pos_world"].shape[0])
    sensor_valid = load_sensor_valid(path, total_frames)
    start, count = select_frame_range(total_frames, frame_start, frame_count)
    metadata = read_source_metadata(path)

    # Unity JsonUtility 对嵌套数组支持很弱，所以所有逐帧张量都按 frame-major 展平成一维数组。
    tracker_positions = slice_time(source["tracker_pos_world"], start, count)
    tracker_rotations = slice_time(source["tracker_rot_world_6d"], start, count)
    valid = slice_time(sensor_valid, start, count).astype(np.int32)
    joints = slice_time(source["joints_world"], start, count)
    root_yaw = slice_time(source["root_yaw"], start, count)
    root_pos = slice_time(source["root_pos_world"], start, count)
    foot_contact = slice_time(source["foot_contact"], start, count)
    target_features_raw = build_target_features_raw(source, start, count, schema.target_dim)
    if identity_6d_rotations:
        target_features_raw[:, 0:BODY_POSE_DIM] = build_identity_body_pose_6d(count)
        tracker_rotations = build_identity_tracker_rotations_6d(count)
        joints = compute_identity_6d_reference_joints(
            root_pos_world=root_pos,
            root_yaw=root_yaw,
            joint_offsets_parent=np.asarray(source["joint_offsets_parent"], dtype=np.float32),
        )

    return {
        "schemaName": schema.name,
        "poseRepresentation": schema.pose_representation,
        "fps": float(fps),
        "frameStart": int(start),
        "frameCount": int(count),
        "sourceFrameCount": int(total_frames),
        "trackerCount": TRACKER_COUNT,
        "trackerNames": list(TRACKER_NAMES),
        "jointCount": SMPL_JOINT_COUNT,
        "footContactDim": FOOT_CONTACT_DIM,
        "trackerPositions": flatten_float(tracker_positions),
        "trackerRotations6d": flatten_float(tracker_rotations),
        "sensorValid": flatten_int(valid),
        "referenceJointsWorld": flatten_float(joints),
        "rootYaw": flatten_float(root_yaw),
        "rootPosWorld": flatten_float(root_pos),
        "footContact": flatten_float(foot_contact),
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
            "rootYawShape": [count],
            "rootPosWorldShape": [count, 3],
            "footContactShape": [count, FOOT_CONTACT_DIM],
            "targetFeaturesRawShape": [count, int(schema.target_dim)],
        },
    }


def write_unity_replay_stream(
    source_npz: Path,
    output_json: Path,
    schema_name: str = REALTIME_POSE_V2_CONTACT_SCHEMA_NAME,
    fps: float = DEFAULT_REPLAY_FPS,
    frame_start: int = 0,
    frame_count: int = 0,
    identity_6d_rotations: bool = False,
) -> Path:
    payload = build_unity_replay_stream_payload(
        source_npz=source_npz,
        schema_name=schema_name,
        fps=fps,
        frame_start=frame_start,
        frame_count=frame_count,
        identity_6d_rotations=identity_6d_rotations,
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
