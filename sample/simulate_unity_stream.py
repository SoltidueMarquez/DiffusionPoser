from __future__ import annotations

import argparse
import time
from argparse import BooleanOptionalAction
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from data_loaders.realtime_pose_dataset import zero_missing_tracker_channels
from data_loaders.realtime_pose_contract import (
    COORDINATE_CONVENTION_VERSION,
    FEATURE_CONTRACT_VERSION,
    JOINT_MAPPING_VERSION,
    TRACKER_SPACE_CALIBRATED_JOINT_WORLD,
)
from data_loaders.realtime_pose_kinematics import (
    SMPL_PARENTS,
    TRACKER_JOINT_INDICES,
    fk_body_fbx_local_torch,
    fk_root_global_torch,
    integrate_root_delta_xz_ref,
    make_yaw_rotation_np,
    rotation_6d_forward_up_np,
    rotation_6d_to_matrix_np,
)
from data_loaders.sensor_masking import (
    DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    HEAD_TRACKER_INDEX,
    HIP_TRACKER_INDEX,
    MIN_VALID_TRACKERS,
    POSE_REPRESENTATION_BODY_FBX_LOCAL_DELTA_6D,
    REALTIME_POSE_SCHEMA_NAMES,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_START,
    ROOT_DELTA_XZ_DIM,
    SMPL_JOINT_COUNT,
    STATIONARY_PROB_DIM,
    TRACKER_COUNT,
    get_schema_spec,
)
from data_loaders.tracker_codec import (
    REFERENCE_POLICY_VERSION,
    TRACKER_CODEC_VERSION,
    build_tracker_reference_np,
    encode_tracker_positions_np,
    encode_tracker_rotations_np,
)
from sample.ik_initializer import IK_INIT_MODE_TRACKER_POSE, resolve_ik_init_timestep, validate_ik_init_mode
from sample.reconstruct_stream import (
    build_ik_init_image_for_batch,
    build_realtime_inpaint_mask,
    reconstruct_batch,
    tensor_bct_to_numpy_btc,
)
from sample.runtime_root_resolver import (
    RESOLVER_CONTRACT_VERSION,
    RootSource,
    RuntimeRootResolver,
)
from sample.utils import load_checkpoint_model
from utils import dist_util
from utils.artifact_roots import load_artifact_roots
from utils.model_util import create_model_and_diffusion
from utils.normalizer import RealtimePoseNormalizer
from utils.parser_util import (
    add_base_options,
    add_diffusion_options,
    add_model_options,
    add_sampling_options,
    load_args_json,
    parse_and_load_from_model,
    str2bool,
)
from utils.schema_resolution import has_explicit_schema_arg, resolve_runtime_schema


IDENTITY_6D = np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
INVALID_FRAME_POLICY_HOLD = "hold"
INVALID_FRAME_POLICY_RAISE = "raise"
INVALID_FRAME_POLICIES = (INVALID_FRAME_POLICY_HOLD, INVALID_FRAME_POLICY_RAISE)
DEFAULT_TRACKER_IK_ITERATIONS = 4
DEFAULT_TRACKER_IK_LR = 0.04
DEFAULT_TRACKER_IK_BLEND = 0.4
DEFAULT_TRACKER_IK_TARGET_SMOOTHING = 0.6
DEFAULT_TRACKER_IK_DELTA_LIMIT = 0.08
TRACKER_IK_REGULARIZATION_WEIGHT = 0.01


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate Unity realtime tracker-only input in Python.",
        allow_abbrev=False,
    )
    add_base_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sampling_options(parser)
    default_schema = get_schema_spec(DEFAULT_REALTIME_POSE_SCHEMA_NAME)
    stream = parser.add_argument_group("unity_stream")
    stream.add_argument("--tracker_stream_path", required=True, type=str, help="包含 tracker_pos_world/sensor_valid 的 npz。")
    stream.add_argument("--schema", default=None, choices=REALTIME_POSE_SCHEMA_NAMES, type=str)
    stream.add_argument("--input_feats", default=default_schema.feature_dim, type=int)
    stream.add_argument("--seq_len", default=REALTIME_POSE_SEQ_LEN, type=int)
    stream.add_argument("--normalizer_dir", default="", type=str)
    stream.add_argument("--normalize_input", default=True, type=str2bool)
    stream.add_argument("--initial_root_yaw", default=0.0, type=float)
    stream.add_argument(
        "--invalid_frame_policy",
        default=INVALID_FRAME_POLICY_HOLD,
        choices=INVALID_FRAME_POLICIES,
        type=str,
        help="tracker 有效性不满足运行时合约时，hold 表示沿用上一帧输出。",
    )
    stream.add_argument("--assume_identity_tracker_rot", action="store_true")
    stream.add_argument("--limit", default=0, type=int)
    stream.add_argument(
        "--root_correction",
        default=True,
        action=BooleanOptionalAction,
        help="用 waist tracker 的真实 transform 修正预测 root yaw/root xz/root height。",
    )
    stream.add_argument(
        "--tracker_ik",
        default=True,
        action=BooleanOptionalAction,
        help="在推理后用已知 head/hands/feet tracker 做 position IK，修正 body pose。",
    )
    stream.add_argument("--tracker_ik_iterations", default=DEFAULT_TRACKER_IK_ITERATIONS, type=int)
    stream.add_argument("--tracker_ik_lr", default=DEFAULT_TRACKER_IK_LR, type=float)
    stream.add_argument("--tracker_ik_blend", default=DEFAULT_TRACKER_IK_BLEND, type=float)
    stream.add_argument("--tracker_ik_target_smoothing", default=DEFAULT_TRACKER_IK_TARGET_SMOOTHING, type=float)
    stream.add_argument("--tracker_ik_delta_limit", default=DEFAULT_TRACKER_IK_DELTA_LIMIT, type=float)
    return parser


def full_valid_sensor_mask(frame_count: int) -> np.ndarray:
    return np.ones((int(frame_count), TRACKER_COUNT), dtype=bool)


def identity_tracker_rotations(frame_count: int) -> np.ndarray:
    return np.tile(IDENTITY_6D, (int(frame_count), TRACKER_COUNT, 1)).astype(np.float32)


def load_tracker_stream(
    path: Path,
    assume_identity_tracker_rot: bool = False,
    limit: int = 0,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        expected_metadata = {
            "feature_contract_version": FEATURE_CONTRACT_VERSION,
            "tracker_space": TRACKER_SPACE_CALIBRATED_JOINT_WORLD,
            "joint_mapping_version": JOINT_MAPPING_VERSION,
            "coordinate_convention_version": COORDINATE_CONVENTION_VERSION,
            "tracker_codec_version": TRACKER_CODEC_VERSION,
            "reference_policy_version": REFERENCE_POLICY_VERSION,
            "resolver_contract_version": RESOLVER_CONTRACT_VERSION,
        }
        missing_metadata = [key for key in (*expected_metadata, "calibration_version") if key not in data.files]
        if missing_metadata:
            raise ValueError(f"{path} missing calibrated tracker contract metadata: {missing_metadata}")
        for key, expected in expected_metadata.items():
            value = np.asarray(data[key]).item()
            actual = int(value) if isinstance(expected, int) else str(value)
            if actual != expected:
                raise ValueError(f"{path} {key}={actual!r}, expected {expected!r}")
        calibration_version = str(np.asarray(data["calibration_version"]).item()).strip()
        if not calibration_version:
            raise ValueError(f"{path} calibration_version must be non-empty")

        if "tracker_pos_world" not in data.files:
            raise KeyError(f"{path} 缺少 tracker_pos_world 字段。")
        tracker_pos_world = np.asarray(data["tracker_pos_world"], dtype=np.float32)
        if tracker_pos_world.ndim != 3 or tracker_pos_world.shape[1:] != (TRACKER_COUNT, 3):
            raise ValueError(f"tracker_pos_world 应为 [T,{TRACKER_COUNT},3]，实际为 {tracker_pos_world.shape}")

        if "tracker_rot_world_6d" in data.files:
            tracker_rot_world_6d = np.asarray(data["tracker_rot_world_6d"], dtype=np.float32)
        elif assume_identity_tracker_rot:
            tracker_rot_world_6d = identity_tracker_rotations(tracker_pos_world.shape[0])
        else:
            raise KeyError(
                f"{path} 缺少 tracker_rot_world_6d；如果只想调试位置流，请显式传 --assume_identity_tracker_rot。"
            )
        if tracker_rot_world_6d.shape != (tracker_pos_world.shape[0], TRACKER_COUNT, 6):
            raise ValueError(
                f"tracker_rot_world_6d 应为 [T,{TRACKER_COUNT},6]，实际为 {tracker_rot_world_6d.shape}"
            )

        sensor_valid = (
            np.asarray(data["sensor_valid"], dtype=bool)
            if "sensor_valid" in data.files
            else full_valid_sensor_mask(tracker_pos_world.shape[0])
        )
        if sensor_valid.shape != (tracker_pos_world.shape[0], TRACKER_COUNT):
            raise ValueError(f"sensor_valid 应为 [T,{TRACKER_COUNT}]，实际为 {sensor_valid.shape}")

        joint_offsets_parent = (
            np.asarray(data["joint_offsets_parent"], dtype=np.float32)
            if "joint_offsets_parent" in data.files
            else None
        )
        joint_rest_local_rotations_6d = (
            np.asarray(data["joint_rest_local_rotations_6d"], dtype=np.float32)
            if "joint_rest_local_rotations_6d" in data.files
            else None
        )
        if joint_offsets_parent is not None and joint_offsets_parent.shape != (SMPL_JOINT_COUNT, 3):
            raise ValueError(f"joint_offsets_parent 应为 [{SMPL_JOINT_COUNT},3]，实际为 {joint_offsets_parent.shape}")
        if joint_rest_local_rotations_6d is not None and joint_rest_local_rotations_6d.shape != (SMPL_JOINT_COUNT, 6):
            raise ValueError(
                f"joint_rest_local_rotations_6d 应为 [{SMPL_JOINT_COUNT},6]，实际为 {joint_rest_local_rotations_6d.shape}"
            )

        frame_count_total = tracker_pos_world.shape[0]
        timestamp_seconds = (
            np.asarray(data["timestamp_seconds"], dtype=np.float64)
            if "timestamp_seconds" in data.files
            else np.arange(frame_count_total, dtype=np.float64) / 60.0
        )
        floor_y = (
            np.asarray(data["floor_y"], dtype=np.float32)
            if "floor_y" in data.files
            else np.zeros(frame_count_total, dtype=np.float32)
        )
        tracking_origin_revision = (
            np.asarray(data["tracking_origin_revision"], dtype=np.int64)
            if "tracking_origin_revision" in data.files
            else np.zeros(frame_count_total, dtype=np.int64)
        )
        previous_to_current_world = (
            np.asarray(data["previous_to_current_world"], dtype=np.float32)
            if "previous_to_current_world" in data.files
            else None
        )
        if timestamp_seconds.shape != (frame_count_total,):
            raise ValueError(f"timestamp_seconds must be [{frame_count_total}], got {timestamp_seconds.shape}")
        if floor_y.shape != (frame_count_total,):
            raise ValueError(f"floor_y must be [{frame_count_total}], got {floor_y.shape}")
        if tracking_origin_revision.shape != (frame_count_total,):
            raise ValueError(
                f"tracking_origin_revision must be [{frame_count_total}], got {tracking_origin_revision.shape}"
            )
        if previous_to_current_world is not None and previous_to_current_world.shape != (frame_count_total, 4, 4):
            raise ValueError(
                f"previous_to_current_world must be [{frame_count_total},4,4], got {previous_to_current_world.shape}"
            )

    frame_count = tracker_pos_world.shape[0]
    if int(limit) > 0:
        frame_count = min(frame_count, int(limit))
    stream = {
        "tracker_pos_world": tracker_pos_world[:frame_count],
        "tracker_rot_world_6d": tracker_rot_world_6d[:frame_count],
        "sensor_valid": sensor_valid[:frame_count],
        "timestamp_seconds": timestamp_seconds[:frame_count],
        "floor_y": floor_y[:frame_count],
        "tracking_origin_revision": tracking_origin_revision[:frame_count],
        "feature_contract_version": np.asarray(FEATURE_CONTRACT_VERSION, dtype=np.int64),
        "tracker_space": np.asarray(TRACKER_SPACE_CALIBRATED_JOINT_WORLD),
        "calibration_version": np.asarray(calibration_version),
        "joint_mapping_version": np.asarray(JOINT_MAPPING_VERSION),
        "coordinate_convention_version": np.asarray(COORDINATE_CONVENTION_VERSION),
        "tracker_codec_version": np.asarray(TRACKER_CODEC_VERSION),
        "reference_policy_version": np.asarray(REFERENCE_POLICY_VERSION),
        "resolver_contract_version": np.asarray(RESOLVER_CONTRACT_VERSION),
    }
    if previous_to_current_world is not None:
        stream["previous_to_current_world"] = previous_to_current_world[:frame_count]
    if joint_offsets_parent is not None:
        stream["joint_offsets_parent"] = joint_offsets_parent
    if joint_rest_local_rotations_6d is not None:
        stream["joint_rest_local_rotations_6d"] = joint_rest_local_rotations_6d
    return stream


def sensor_validity_ok(sensor_valid: np.ndarray) -> bool:
    valid = np.asarray(sensor_valid, dtype=bool)
    return bool(
        valid.shape == (TRACKER_COUNT,)
        and valid[HEAD_TRACKER_INDEX]
        and valid.sum() >= MIN_VALID_TRACKERS
    )


def estimate_root_pos_from_hip_tracker(
    tracker_pos_world: np.ndarray,
    root_yaw: float = 0.0,
    joint_offsets_parent: np.ndarray | None = None,
    schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME,
) -> np.ndarray:
    root_pos = np.asarray(tracker_pos_world[HIP_TRACKER_INDEX], dtype=np.float32).copy()
    if joint_offsets_parent is not None and is_body_fbx_local_schema(schema_name):
        offsets = np.asarray(joint_offsets_parent, dtype=np.float32)
        if offsets.shape != (SMPL_JOINT_COUNT, 3):
            raise ValueError(f"joint_offsets_parent 应为 [{SMPL_JOINT_COUNT},3]，实际为 {offsets.shape}")
        yaw_rotation = make_yaw_rotation_np(np.asarray([float(root_yaw)], dtype=np.float64))[0]
        root_pos = root_pos - (yaw_rotation @ offsets[0].astype(np.float64)).astype(np.float32)
    root_pos[1] = 0.0
    return root_pos


def estimate_root_pos_from_available_trackers(
    tracker_pos_world: np.ndarray,
    sensor_valid: np.ndarray,
    root_yaw: float = 0.0,
    fallback_root_pos_world: np.ndarray | None = None,
    joint_offsets_parent: np.ndarray | None = None,
    schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME,
) -> np.ndarray:
    """有腰时使用实测 pelvis；无腰时沿用预测 root，首帧用 head XZ 初始化。"""

    tracker_pos = np.asarray(tracker_pos_world, dtype=np.float32)
    valid = np.asarray(sensor_valid, dtype=bool)
    if tracker_pos.shape != (TRACKER_COUNT, 3):
        raise ValueError(f"单帧 tracker_pos_world 应为 [{TRACKER_COUNT},3]，实际为 {tracker_pos.shape}")
    if valid.shape != (TRACKER_COUNT,):
        raise ValueError(f"单帧 sensor_valid 应为 [{TRACKER_COUNT}]，实际为 {valid.shape}")
    if valid[HIP_TRACKER_INDEX]:
        return estimate_root_pos_from_hip_tracker(
            tracker_pos,
            root_yaw=root_yaw,
            joint_offsets_parent=joint_offsets_parent,
            schema_name=schema_name,
        )
    if fallback_root_pos_world is not None:
        root_pos = np.asarray(fallback_root_pos_world, dtype=np.float32).copy()
    elif valid[HEAD_TRACKER_INDEX]:
        root_pos = tracker_pos[HEAD_TRACKER_INDEX].copy()
    else:
        first_valid = np.flatnonzero(valid)
        if first_valid.size == 0:
            raise ValueError("至少需要一个有效 tracker 来初始化 root 位置。")
        root_pos = tracker_pos[int(first_valid[0])].copy()
    root_pos[1] = 0.0
    return root_pos


def estimate_root_yaw_from_hip_tracker(
    tracker_rot_world_6d: np.ndarray,
    sensor_valid: np.ndarray,
    fallback_yaw: float,
) -> float:
    """从 waist tracker 的世界旋转取水平 forward yaw；退化旋转时沿用模型预测。"""

    valid = np.asarray(sensor_valid, dtype=bool)
    if valid.shape != (TRACKER_COUNT,) or not valid[HIP_TRACKER_INDEX]:
        return float(fallback_yaw)
    tracker_rot = np.asarray(tracker_rot_world_6d, dtype=np.float32)
    if tracker_rot.shape != (TRACKER_COUNT, 6):
        raise ValueError(f"单帧 tracker_rot_world_6d 应为 [{TRACKER_COUNT},6]，实际为 {tracker_rot.shape}")

    hip_rotation = rotation_6d_to_matrix_np(tracker_rot[HIP_TRACKER_INDEX: HIP_TRACKER_INDEX + 1])[0]
    forward = hip_rotation[:, 2]
    horizontal_norm = float(np.linalg.norm(forward[[0, 2]]))
    if horizontal_norm < 1e-6:
        return float(fallback_yaw)
    return float(np.arctan2(float(forward[0]), float(forward[2])))


def encode_single_root_delta_xz_ref(
    prev_root_pos_world: np.ndarray,
    prev_root_yaw: float,
    root_pos_world: np.ndarray,
) -> np.ndarray:
    prev_pos = np.asarray(prev_root_pos_world, dtype=np.float64)
    root_pos = np.asarray(root_pos_world, dtype=np.float64)
    delta_world = root_pos - prev_pos
    delta_world[1] = 0.0
    yaw_rotation = make_yaw_rotation_np(np.asarray([float(prev_root_yaw)], dtype=np.float64))[0]
    delta_ref = delta_world @ yaw_rotation
    return delta_ref[[0, 2]].astype(np.float32)


def is_body_fbx_local_schema(schema_name: str) -> bool:
    return get_schema_spec(schema_name).pose_representation == POSE_REPRESENTATION_BODY_FBX_LOCAL_DELTA_6D


def apply_root_y0_actor_root(
    actor_root_pos_world: np.ndarray,
    schema_name: str,
) -> np.ndarray:
    """root-y0 schema 中 actor root 只承载 XZ 和 yaw，y 始终固定为 0。"""

    schema = get_schema_spec(schema_name)
    root_pos = np.asarray(actor_root_pos_world, dtype=np.float32).copy()
    if schema.supports_root_motion:
        root_pos[1] = 0.0
    return root_pos.astype(np.float32)


def fk_joints_from_target(
    target_raw: np.ndarray,
    root_pos_world: np.ndarray,
    root_yaw: float,
    joint_offsets_parent: np.ndarray,
    schema_name: str,
    joint_rest_local_rotations_6d: np.ndarray | None = None,
    return_global_rot: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    schema = get_schema_spec(schema_name)
    target = np.asarray(target_raw, dtype=np.float32)
    offsets = np.asarray(joint_offsets_parent, dtype=np.float32).copy()
    if offsets.shape != (SMPL_JOINT_COUNT, 3):
        raise ValueError(f"joint_offsets_parent 应为 [{SMPL_JOINT_COUNT},3]，实际为 {offsets.shape}")
    if schema.supports_root_motion:
        offsets[0, 1] = float(target[schema.root_height_slice()][0])
    root_pos = apply_root_y0_actor_root(
        actor_root_pos_world=root_pos_world,
        schema_name=schema.name,
    )

    with torch.no_grad():
        if is_body_fbx_local_schema(schema.name):
            result = fk_body_fbx_local_torch(
                body_pose_local_delta_6d=torch.from_numpy(target[schema.body_pose_slice()][None]).float(),
                actor_root_pos_world=torch.from_numpy(root_pos[None]).float(),
                root_heading=torch.tensor([float(root_yaw)], dtype=torch.float32),
                rest_local_positions=torch.from_numpy(offsets[None]).float(),
                rest_local_rotations_6d=(
                    None
                    if joint_rest_local_rotations_6d is None
                    else torch.from_numpy(np.asarray(joint_rest_local_rotations_6d, dtype=np.float32)[None]).float()
                ),
                return_global_rot=return_global_rot,
            )
        else:
            result = fk_root_global_torch(
                body_pose_root_global_6d=torch.from_numpy(target[schema.body_pose_slice()][None]).float(),
                root_pos_world=torch.from_numpy(root_pos[None]).float(),
                root_yaw=torch.tensor([float(root_yaw)], dtype=torch.float32),
                parent_offsets=torch.from_numpy(offsets[None]).float(),
                return_global_rot=return_global_rot,
            )
    if return_global_rot:
        joints, global_rot = result
        return joints.numpy()[0].astype(np.float32), global_rot.numpy()[0].astype(np.float32)
    return result.numpy()[0].astype(np.float32)


def fk_tracker_positions_from_target(
    target_raw: np.ndarray,
    root_pos_world: np.ndarray,
    root_yaw: float,
    joint_offsets_parent: np.ndarray,
    schema_name: str,
    joint_rest_local_rotations_6d: np.ndarray | None = None,
) -> np.ndarray:
    """用预测 target 做 FK，返回 6 个 tracker 对应关节的世界位置。"""

    joints = fk_joints_from_target(
        target_raw=target_raw,
        root_pos_world=root_pos_world,
        root_yaw=root_yaw,
        joint_offsets_parent=joint_offsets_parent,
        schema_name=schema_name,
        joint_rest_local_rotations_6d=joint_rest_local_rotations_6d,
    )
    return joints[TRACKER_JOINT_INDICES].astype(np.float32)


def tracker_ik_joint_mask(tracker_indices: np.ndarray) -> np.ndarray:
    """
    根据已知 tracker 端点找出需要参与 IK 的祖先关节。
    root yaw/root xz 已经由 waist tracker 单独强约束，所以这里冻结 0 号 pelvis local rotation，
    避免 IK 把整个人体姿态重新整体扭转。
    """

    mask = np.zeros((SMPL_JOINT_COUNT,), dtype=bool)
    for tracker_index in np.asarray(tracker_indices, dtype=np.int64).reshape(-1):
        joint_index = int(TRACKER_JOINT_INDICES[int(tracker_index)])
        while joint_index >= 0:
            if joint_index != 0:
                mask[joint_index] = True
            joint_index = int(SMPL_PARENTS[joint_index])
    return mask


def normalize_body_pose_6d(body_pose_parent_6d: np.ndarray) -> np.ndarray:
    rotations = rotation_6d_to_matrix_np(
        np.asarray(body_pose_parent_6d, dtype=np.float32).reshape(SMPL_JOINT_COUNT, 6)
    )
    return rotation_6d_forward_up_np(rotations).reshape(-1).astype(np.float32)


def blend_body_pose_6d(base_body_pose_parent_6d: np.ndarray, target_body_pose_parent_6d: np.ndarray, blend: float) -> np.ndarray:
    base = np.asarray(base_body_pose_parent_6d, dtype=np.float32).reshape(-1)
    target = np.asarray(target_body_pose_parent_6d, dtype=np.float32).reshape(-1)
    if base.shape != (SMPL_JOINT_COUNT * 6,) or target.shape != (SMPL_JOINT_COUNT * 6,):
        raise ValueError(f"body_pose_root_global_6d 应为 [{SMPL_JOINT_COUNT * 6}]。")
    weight = float(np.clip(float(blend), 0.0, 1.0))
    mixed = base + weight * (target - base)
    return normalize_body_pose_6d(mixed)


def clamp_body_pose_delta(
    body_pose_parent_6d: np.ndarray,
    reference_body_pose_parent_6d: np.ndarray | None,
    delta_limit: float,
) -> np.ndarray:
    if reference_body_pose_parent_6d is None or float(delta_limit) <= 0.0:
        return normalize_body_pose_6d(body_pose_parent_6d)
    pose = np.asarray(body_pose_parent_6d, dtype=np.float32).reshape(SMPL_JOINT_COUNT, 6)
    reference = np.asarray(reference_body_pose_parent_6d, dtype=np.float32).reshape(SMPL_JOINT_COUNT, 6)
    delta = pose - reference
    delta_norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    scale = np.minimum(1.0, float(delta_limit) / np.maximum(delta_norm, 1e-8))
    return normalize_body_pose_6d(reference + delta * scale)


def smooth_tracker_positions_for_ik(
    tracker_pos_world: np.ndarray,
    sensor_valid: np.ndarray,
    previous_smoothed_pos_world: np.ndarray | None,
    previous_valid: np.ndarray | None,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对 IK target 做 EMA，降低 tracker 噪声直接传到关节角的高频抖动。
    只平滑连续有效的 tracker；新变为有效的 tracker 直接使用当前测量，避免 stale target。
    """

    measured = np.asarray(tracker_pos_world, dtype=np.float32)
    valid = np.asarray(sensor_valid, dtype=bool)
    if measured.shape != (TRACKER_COUNT, 3):
        raise ValueError(f"单帧 tracker_pos_world 应为 [{TRACKER_COUNT},3]，实际为 {measured.shape}")
    if valid.shape != (TRACKER_COUNT,):
        raise ValueError(f"单帧 sensor_valid 应为 [{TRACKER_COUNT}]，实际为 {valid.shape}")
    alpha = float(np.clip(float(smoothing), 0.0, 0.99))
    if (
        alpha <= 0.0
        or previous_smoothed_pos_world is None
        or previous_valid is None
        or np.asarray(previous_smoothed_pos_world).shape != (TRACKER_COUNT, 3)
        or np.asarray(previous_valid).shape != (TRACKER_COUNT,)
    ):
        return measured.copy(), valid.copy()

    smoothed = measured.copy()
    reusable = valid & np.asarray(previous_valid, dtype=bool)
    previous = np.asarray(previous_smoothed_pos_world, dtype=np.float32)
    smoothed[reusable] = alpha * previous[reusable] + (1.0 - alpha) * measured[reusable]
    return smoothed.astype(np.float32), valid.copy()


def apply_tracker_position_ik(
    predicted_frame_raw: np.ndarray,
    root_pos_world: np.ndarray,
    root_yaw: float,
    tracker_pos_world: np.ndarray,
    sensor_valid: np.ndarray,
    joint_offsets_parent: np.ndarray | None,
    schema_name: str,
    joint_rest_local_rotations_6d: np.ndarray | None = None,
    enabled: bool = True,
    iterations: int = DEFAULT_TRACKER_IK_ITERATIONS,
    lr: float = DEFAULT_TRACKER_IK_LR,
    initial_body_pose_parent_6d: np.ndarray | None = None,
    previous_body_pose_parent_6d: np.ndarray | None = None,
    blend: float = DEFAULT_TRACKER_IK_BLEND,
    delta_limit: float = DEFAULT_TRACKER_IK_DELTA_LIMIT,
) -> np.ndarray:
    """
    用已知 tracker 位置对预测 pose 做一小步运行时 IK 投影。

    输入/输出都是单帧 raw feature `[211]`。IK 只修改 body pose root-global 6D，
    root yaw/root xz/root height 由 root correction 负责，因此这里不改变 root 状态。
    """

    corrected = np.asarray(predicted_frame_raw, dtype=np.float32).copy()
    if not enabled or joint_offsets_parent is None or int(iterations) <= 0 or float(blend) <= 0.0:
        return corrected

    schema = get_schema_spec(schema_name)
    tracker_pos = np.asarray(tracker_pos_world, dtype=np.float32)
    valid = np.asarray(sensor_valid, dtype=bool)
    offsets = np.asarray(joint_offsets_parent, dtype=np.float32).copy()
    if tracker_pos.shape != (TRACKER_COUNT, 3):
        raise ValueError(f"单帧 tracker_pos_world 应为 [{TRACKER_COUNT},3]，实际为 {tracker_pos.shape}")
    if valid.shape != (TRACKER_COUNT,):
        raise ValueError(f"单帧 sensor_valid 应为 [{TRACKER_COUNT}]，实际为 {valid.shape}")
    if offsets.shape != (SMPL_JOINT_COUNT, 3):
        raise ValueError(f"joint_offsets_parent 应为 [{SMPL_JOINT_COUNT},3]，实际为 {offsets.shape}")

    ik_tracker_indices = np.flatnonzero(valid)
    ik_tracker_indices = ik_tracker_indices[ik_tracker_indices != HIP_TRACKER_INDEX]
    if ik_tracker_indices.size == 0:
        return corrected

    trainable_joint_mask = tracker_ik_joint_mask(ik_tracker_indices)
    if not trainable_joint_mask.any():
        return corrected

    if schema.supports_root_motion:
        offsets[0, 1] = float(corrected[schema.root_height_slice()][0])

    target_joint_indices = torch.as_tensor(TRACKER_JOINT_INDICES[ik_tracker_indices], dtype=torch.long)
    target_positions = torch.from_numpy(tracker_pos[ik_tracker_indices]).float()[None]
    root_pos_np = apply_root_y0_actor_root(
        actor_root_pos_world=root_pos_world,
        schema_name=schema.name,
    )
    root_pos = torch.from_numpy(root_pos_np[None]).float()
    root_yaw_tensor = torch.tensor([float(root_yaw)], dtype=torch.float32)
    parent_offsets = torch.from_numpy(offsets[None]).float()
    rest_rot_6d = (
        None
        if joint_rest_local_rotations_6d is None
        else torch.from_numpy(np.asarray(joint_rest_local_rotations_6d, dtype=np.float32)[None]).float()
    )
    original_body_pose = corrected[schema.body_pose_slice()].copy()
    original_pose = torch.from_numpy(original_body_pose.reshape(1, SMPL_JOINT_COUNT, 6)).float()
    initial_pose = original_pose.clone()
    if initial_body_pose_parent_6d is not None:
        warm_start = np.asarray(initial_body_pose_parent_6d, dtype=np.float32).reshape(SMPL_JOINT_COUNT, 6)
        initial_pose[:, trainable_joint_mask] = torch.from_numpy(warm_start[trainable_joint_mask]).float()[None]
    trainable_mask = torch.from_numpy(trainable_joint_mask).bool()
    pose = initial_pose.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([pose], lr=float(lr))

    for _ in range(int(iterations)):
        optimizer.zero_grad(set_to_none=True)
        if is_body_fbx_local_schema(schema.name):
            joints = fk_body_fbx_local_torch(
                body_pose_local_delta_6d=pose.reshape(1, -1),
                actor_root_pos_world=root_pos,
                root_heading=root_yaw_tensor,
                rest_local_positions=parent_offsets,
                rest_local_rotations_6d=rest_rot_6d,
            )
        else:
            joints = fk_root_global_torch(
                body_pose_root_global_6d=pose.reshape(1, -1),
                root_pos_world=root_pos,
                root_yaw=root_yaw_tensor,
                parent_offsets=parent_offsets,
            )
        error = joints[:, target_joint_indices] - target_positions
        position_loss = error.square().mean()
        regularization = (pose[:, trainable_mask] - original_pose[:, trainable_mask]).square().mean()
        loss = position_loss + TRACKER_IK_REGULARIZATION_WEIGHT * regularization
        loss.backward()
        if pose.grad is not None:
            pose.grad[:, ~trainable_mask] = 0.0
        optimizer.step()
        with torch.no_grad():
            pose[:, ~trainable_mask] = original_pose[:, ~trainable_mask]

    ik_body_pose = normalize_body_pose_6d(pose.detach().cpu().numpy().reshape(-1))
    blended_body_pose = blend_body_pose_6d(original_body_pose, ik_body_pose, blend=blend)
    corrected[schema.body_pose_slice()] = clamp_body_pose_delta(
        blended_body_pose,
        reference_body_pose_parent_6d=previous_body_pose_parent_6d,
        delta_limit=float(delta_limit),
    )
    return corrected


def encode_unity_tracker_frame(
    tracker_pos_world: np.ndarray,
    tracker_rot_world_6d: np.ndarray,
    sensor_valid: np.ndarray,
    reference_root_yaw: float,
    schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    root_pos_world: np.ndarray | None = None,
    joint_offsets_parent: np.ndarray | None = None,
) -> np.ndarray:
    """
    把 Unity 当前帧 tracker transform 编成模型 raw feature。

    运行时只能使用上一帧预测 root_yaw 作为 reference yaw；当前帧 target
    通道会在采样前置零，所以这里仅填 tracker 条件和 sensor_valid。
    """

    schema = get_schema_spec(schema_name)
    tracker_pos = np.asarray(tracker_pos_world, dtype=np.float32)
    tracker_rot = np.asarray(tracker_rot_world_6d, dtype=np.float32)
    valid = np.asarray(sensor_valid, dtype=bool)
    if tracker_pos.shape != (TRACKER_COUNT, 3):
        raise ValueError(f"单帧 tracker_pos_world 应为 [{TRACKER_COUNT},3]，实际为 {tracker_pos.shape}")
    if tracker_rot.shape != (TRACKER_COUNT, 6):
        raise ValueError(f"单帧 tracker_rot_world_6d 应为 [{TRACKER_COUNT},6]，实际为 {tracker_rot.shape}")
    if valid.shape != (TRACKER_COUNT,):
        raise ValueError(f"单帧 sensor_valid 应为 [{TRACKER_COUNT}]，实际为 {valid.shape}")

    if root_pos_world is None:
        root_position_yaw = estimate_root_yaw_from_hip_tracker(
            tracker_rot_world_6d=tracker_rot,
            sensor_valid=valid,
            fallback_yaw=reference_root_yaw,
        )
        root_pos = estimate_root_pos_from_available_trackers(
            tracker_pos,
            sensor_valid=valid,
            root_yaw=root_position_yaw,
            joint_offsets_parent=joint_offsets_parent,
            schema_name=schema.name,
        )
    else:
        root_pos = np.asarray(root_pos_world, dtype=np.float32).copy()
    root_pos[1] = 0.0

    tracker_pos_ref = encode_tracker_positions_np(
        tracker_pos_world=tracker_pos,
        ref_root_pos_world=root_pos,
        ref_root_yaw=np.asarray(float(reference_root_yaw), dtype=np.float32),
    )
    tracker_rot_ref_6d = encode_tracker_rotations_np(
        tracker_rot_world_6d=tracker_rot,
        ref_root_yaw=np.asarray(float(reference_root_yaw), dtype=np.float32),
    )

    features = np.zeros((schema.feature_dim,), dtype=np.float32)
    features[schema.tracker_pos_slice()] = tracker_pos_ref.reshape(-1).astype(np.float32)
    features[schema.tracker_rot_slice()] = tracker_rot_ref_6d.reshape(-1).astype(np.float32)
    features[schema.sensor_valid_slice()] = valid.astype(np.float32)
    for tracker_index in range(TRACKER_COUNT):
        if valid[tracker_index]:
            continue
        features[schema.tracker_pos_slice(tracker_index)] = 0.0
        features[schema.tracker_rot_slice(tracker_index)] = 0.0
    return features


def initial_target_feature(schema_name: str, root_height: float = 0.0) -> np.ndarray:
    schema = get_schema_spec(schema_name)
    features = np.zeros((schema.feature_dim,), dtype=np.float32)
    features[schema.body_pose_slice()] = np.tile(IDENTITY_6D, SMPL_JOINT_COUNT)
    features[schema.root_yaw_delta_slice()] = np.asarray([0.0, 1.0], dtype=np.float32)
    if schema.supports_root_motion:
        features[schema.root_delta_xz_slice()] = np.zeros((ROOT_DELTA_XZ_DIM,), dtype=np.float32)
        features[schema.root_height_slice()] = np.asarray([float(root_height)], dtype=np.float32)
    if schema.supports_stationary_prob:
        features[schema.stationary_prob_slice()] = np.zeros((STATIONARY_PROB_DIM,), dtype=np.float32)
    return features


def normalize_conditioned_window(
    window_raw: np.ndarray,
    normalizer: RealtimePoseNormalizer | None,
    schema_name: str,
) -> np.ndarray:
    schema = get_schema_spec(schema_name)
    conditioned = window_raw.copy()
    if normalizer is not None:
        conditioned = np.asarray(normalizer.normalize(conditioned), dtype=np.float32)
        sensor_valid = np.asarray(window_raw[:, schema.sensor_valid_slice()], dtype=bool)
        zero_missing_tracker_channels(features=conditioned, sensor_valid=sensor_valid, schema_name=schema.name)
    conditioned[REALTIME_POSE_TARGET_START, schema.target_slice()] = 0.0
    return conditioned.astype(np.float32, copy=False)


def inverse_feature_window(
    window: np.ndarray,
    normalizer: RealtimePoseNormalizer | None,
) -> np.ndarray:
    if normalizer is None:
        return window.astype(np.float32, copy=False)
    return np.asarray(normalizer.inverse(window), dtype=np.float32)


@dataclass
class UnityStreamState:
    schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME
    initial_root_yaw: float = 0.0
    invalid_frame_policy: str = INVALID_FRAME_POLICY_HOLD
    history_raw: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=REALTIME_POSE_TARGET_START))
    current_root_yaw: float = field(init=False)
    current_root_pos_world: np.ndarray | None = None
    last_output_raw: np.ndarray | None = None
    last_root_pos_world: np.ndarray | None = None
    last_validity_ok: bool = True
    tracker_ik_smoothed_pos_world: np.ndarray | None = None
    tracker_ik_smoothed_valid: np.ndarray | None = None
    root_resolver: RuntimeRootResolver | None = None
    current_pelvis_height: float = 0.0
    final_joints_world: np.ndarray | None = None
    last_preliminary_joints_world: np.ndarray | None = None
    last_preliminary_root_yaw: float | None = None
    last_root_source: RootSource = RootSource.RESET
    last_reconnect_alpha: float = 0.0
    last_resolver_elapsed_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.invalid_frame_policy not in INVALID_FRAME_POLICIES:
            raise ValueError(f"未知 invalid_frame_policy={self.invalid_frame_policy}")
        self.schema = get_schema_spec(self.schema_name)
        self.current_root_yaw = float(self.initial_root_yaw)

    def has_full_history(self) -> bool:
        return len(self.history_raw) == REALTIME_POSE_TARGET_START

    def make_initial_history_frame(
        self,
        tracker_feature_raw: np.ndarray,
        root_height: float = 0.0,
        target_feature_raw: np.ndarray | None = None,
    ) -> np.ndarray:
        frame = np.asarray(tracker_feature_raw, dtype=np.float32).copy()
        target = (
            initial_target_feature(self.schema.name, root_height=root_height)
            if target_feature_raw is None
            else np.asarray(target_feature_raw, dtype=np.float32)
        )
        frame[self.schema.target_slice()] = target[self.schema.target_slice()]
        return frame

    def append_warmup_frame(
        self,
        tracker_feature_raw: np.ndarray,
        root_pos_world: np.ndarray,
        root_height: float = 0.0,
        target_feature_raw: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.current_root_pos_world is None:
            self.current_root_pos_world = np.asarray(root_pos_world, dtype=np.float32).copy()
        history_frame = self.make_initial_history_frame(
            tracker_feature_raw,
            root_height=root_height,
            target_feature_raw=target_feature_raw,
        )
        self.history_raw.append(history_frame)
        self.last_output_raw = history_frame.copy()
        self.last_root_pos_world = self.current_root_pos_world.copy()
        return history_frame.copy(), self.current_root_pos_world.copy()

    def build_window_raw(self, tracker_feature_raw: np.ndarray) -> np.ndarray:
        if not self.has_full_history():
            raise RuntimeError("Unity stream history 尚未填满 60 帧。")
        current = np.asarray(tracker_feature_raw, dtype=np.float32).copy()
        current[self.schema.target_slice()] = 0.0
        return np.concatenate([np.stack(list(self.history_raw), axis=0), current[None]], axis=0)

    def append_output_frame(
        self,
        tracker_feature_raw: np.ndarray,
        output_frame_raw: np.ndarray,
        root_pos_world: np.ndarray,
        history_target_raw: np.ndarray | None = None,
    ) -> None:
        history_frame = np.asarray(tracker_feature_raw, dtype=np.float32).copy()
        target_source = (
            np.asarray(output_frame_raw, dtype=np.float32)
            if history_target_raw is None
            else np.asarray(history_target_raw, dtype=np.float32)
        )
        history_frame[self.schema.target_slice()] = target_source[self.schema.target_slice()]
        self.history_raw.append(history_frame)
        self.last_output_raw = history_frame.copy()
        self.last_root_pos_world = np.asarray(root_pos_world, dtype=np.float32).copy()

    def update_tracker_ik_targets(self, tracker_pos_world: np.ndarray, sensor_valid: np.ndarray, smoothing: float) -> np.ndarray:
        smoothed, smoothed_valid = smooth_tracker_positions_for_ik(
            tracker_pos_world=tracker_pos_world,
            sensor_valid=sensor_valid,
            previous_smoothed_pos_world=self.tracker_ik_smoothed_pos_world,
            previous_valid=self.tracker_ik_smoothed_valid,
            smoothing=float(smoothing),
        )
        self.tracker_ik_smoothed_pos_world = smoothed.copy()
        self.tracker_ik_smoothed_valid = smoothed_valid.copy()
        return smoothed

    def accept_prediction(
        self,
        tracker_feature_raw: np.ndarray,
        predicted_frame_raw: np.ndarray,
        fallback_root_pos_world: np.ndarray,
        history_target_raw: np.ndarray | None = None,
        tracker_pos_world: np.ndarray | None = None,
        tracker_rot_world_6d: np.ndarray | None = None,
        sensor_valid: np.ndarray | None = None,
        joint_offsets_parent: np.ndarray | None = None,
        joint_rest_local_rotations_6d: np.ndarray | None = None,
        root_correction: bool = True,
        tracker_ik: bool = True,
        tracker_ik_iterations: int = DEFAULT_TRACKER_IK_ITERATIONS,
        tracker_ik_lr: float = DEFAULT_TRACKER_IK_LR,
        tracker_ik_blend: float = DEFAULT_TRACKER_IK_BLEND,
        tracker_ik_target_smoothing: float = DEFAULT_TRACKER_IK_TARGET_SMOOTHING,
        tracker_ik_delta_limit: float = DEFAULT_TRACKER_IK_DELTA_LIMIT,
        timestamp: float = 0.0,
        floor_y: float = 0.0,
        tracking_origin_revision: int = 0,
        previous_to_current_world: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del history_target_raw
        self.last_preliminary_joints_world = None
        self.last_preliminary_root_yaw = None
        predicted = np.asarray(predicted_frame_raw, dtype=np.float32).copy()
        prev_root_yaw = float(self.current_root_yaw)
        prev_root_pos = (
            np.asarray(fallback_root_pos_world, dtype=np.float32).copy()
            if self.current_root_pos_world is None
            else self.current_root_pos_world.copy()
        )
        previous_body_pose = (
            None
            if self.last_output_raw is None
            else np.asarray(self.last_output_raw[self.schema.body_pose_slice()], dtype=np.float32).copy()
        )
        yaw_delta = predicted[self.schema.root_yaw_delta_slice()]
        model_root_yaw = float(prev_root_yaw + np.arctan2(float(yaw_delta[0]), float(yaw_delta[1])))

        if self.schema.supports_root_motion:
            model_root_pos = integrate_root_delta_xz_ref(
                prev_root_pos_world=prev_root_pos[None],
                prev_root_yaw=np.asarray([prev_root_yaw], dtype=np.float32),
                root_delta_xz_ref=predicted[self.schema.root_delta_xz_slice()][None],
            )[0]
            model_root_pos = apply_root_y0_actor_root(
                actor_root_pos_world=model_root_pos,
                schema_name=self.schema.name,
            )
        else:
            model_root_pos = np.asarray(fallback_root_pos_world, dtype=np.float32).copy()

        root_pos = model_root_pos
        self.current_root_yaw = model_root_yaw
        self.last_preliminary_root_yaw = float(model_root_yaw)

        # 这里保存的是 Resolver/Tracker IK 之前、直接由模型 Root 与姿态得到的 FK。
        # 评测需要把模型原始几何与运行时校正后的最终关节严格分开。
        if joint_offsets_parent is not None:
            self.last_preliminary_joints_world = fk_joints_from_target(
                target_raw=predicted,
                root_pos_world=model_root_pos,
                root_yaw=model_root_yaw,
                joint_offsets_parent=joint_offsets_parent,
                schema_name=self.schema.name,
                joint_rest_local_rotations_6d=joint_rest_local_rotations_6d,
            ).astype(np.float32, copy=True)

        if tracker_pos_world is not None and tracker_rot_world_6d is not None and sensor_valid is not None:
            offsets = (
                np.zeros((SMPL_JOINT_COUNT, 3), dtype=np.float32)
                if joint_offsets_parent is None
                else np.asarray(joint_offsets_parent, dtype=np.float32)
            )
            if self.root_resolver is None:
                self.root_resolver = RuntimeRootResolver(pelvis_offset_parent=offsets[0])

            if root_correction:
                model_height = (
                    float(predicted[self.schema.root_height_slice()][0])
                    if self.schema.supports_root_motion
                    else self.current_pelvis_height
                )

                def fk_callback(root: np.ndarray, yaw: float, pelvis_height: float) -> np.ndarray:
                    fk_target = predicted.copy()
                    if self.schema.supports_root_motion:
                        fk_target[self.schema.root_height_slice()] = np.asarray([pelvis_height], dtype=np.float32)
                    return fk_joints_from_target(
                        target_raw=fk_target,
                        root_pos_world=root,
                        root_yaw=yaw,
                        joint_offsets_parent=offsets,
                        schema_name=self.schema.name,
                        joint_rest_local_rotations_6d=joint_rest_local_rotations_6d,
                    )

                preliminary_joints = self.last_preliminary_joints_world
                if preliminary_joints is None:
                    preliminary_joints = fk_callback(model_root_pos, model_root_yaw, model_height)
                    self.last_preliminary_joints_world = preliminary_joints.astype(np.float32, copy=True)
                model_root_delta = (
                    predicted[self.schema.root_delta_xz_slice()]
                    if self.schema.supports_root_motion
                    else np.zeros(ROOT_DELTA_XZ_DIM, dtype=np.float32)
                )
                resolver_started = time.perf_counter()
                resolved = self.root_resolver.resolve(
                    tracker_pos_world=tracker_pos_world,
                    tracker_rot_world_6d=tracker_rot_world_6d,
                    sensor_valid=sensor_valid,
                    timestamp=float(timestamp),
                    floor_y=float(floor_y),
                    tracking_origin_revision=int(tracking_origin_revision),
                    model_root_delta_xz_ref=model_root_delta,
                    model_yaw_delta_sincos=predicted[self.schema.root_yaw_delta_slice()],
                    model_pelvis_height=model_height,
                    fk_callback=fk_callback,
                    preliminary_joints_world=preliminary_joints,
                    previous_to_current_world=previous_to_current_world,
                )
                self.last_resolver_elapsed_ms = (time.perf_counter() - resolver_started) * 1000.0
                root_pos = resolved.final_root_pos_world.copy()
                self.current_root_yaw = float(resolved.final_root_yaw)
                self.current_pelvis_height = float(resolved.final_pelvis_height)
                self.final_joints_world = resolved.final_joints_world.copy()
                self.last_root_source = resolved.root_source
                self.last_reconnect_alpha = float(resolved.reconnect_alpha)
                if self.schema.supports_root_motion:
                    predicted[self.schema.root_delta_xz_slice()] = resolved.final_root_delta_xz_ref
                    predicted[self.schema.root_height_slice()] = np.asarray(
                        [resolved.final_pelvis_height], dtype=np.float32
                    )
                predicted[self.schema.root_yaw_delta_slice()] = resolved.final_yaw_delta_sincos

            ik_tracker_pos_world = (
                self.update_tracker_ik_targets(
                    tracker_pos_world=tracker_pos_world,
                    sensor_valid=sensor_valid,
                    smoothing=float(tracker_ik_target_smoothing),
                )
                if tracker_ik
                else tracker_pos_world
            )
            predicted = apply_tracker_position_ik(
                predicted_frame_raw=predicted,
                root_pos_world=root_pos,
                root_yaw=self.current_root_yaw,
                tracker_pos_world=ik_tracker_pos_world,
                sensor_valid=sensor_valid,
                joint_offsets_parent=joint_offsets_parent,
                schema_name=self.schema.name,
                joint_rest_local_rotations_6d=joint_rest_local_rotations_6d,
                enabled=tracker_ik,
                iterations=tracker_ik_iterations,
                lr=tracker_ik_lr,
                initial_body_pose_parent_6d=previous_body_pose,
                previous_body_pose_parent_6d=previous_body_pose,
                blend=tracker_ik_blend,
                delta_limit=tracker_ik_delta_limit,
            )
            if tracker_ik and joint_offsets_parent is not None:
                self.final_joints_world = fk_joints_from_target(
                    target_raw=predicted,
                    root_pos_world=root_pos,
                    root_yaw=self.current_root_yaw,
                    joint_offsets_parent=joint_offsets_parent,
                    schema_name=self.schema.name,
                    joint_rest_local_rotations_6d=joint_rest_local_rotations_6d,
                )
                if self.root_resolver is not None and self.root_resolver.state.initialized:
                    self.root_resolver.state.final_joints_world = self.final_joints_world.copy()

        self.current_root_pos_world = root_pos.astype(np.float32)
        predicted_frame_raw[...] = predicted
        self.append_output_frame(
            tracker_feature_raw,
            predicted,
            root_pos_world=self.current_root_pos_world,
            history_target_raw=None,
        )
        return predicted.copy(), self.current_root_pos_world.copy()

    def hold_output(self, tracker_feature_raw: np.ndarray, root_pos_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.last_output_raw is None:
            held = self.make_initial_history_frame(
                tracker_feature_raw,
                root_height=float(root_pos_world[1]) if root_pos_world.shape == (3,) else 0.0,
            )
        else:
            held = self.last_output_raw.copy()
        root_pos = (
            np.asarray(root_pos_world, dtype=np.float32).copy()
            if self.current_root_pos_world is None
            else self.current_root_pos_world.copy()
        )
        self.append_output_frame(tracker_feature_raw, held, root_pos_world=root_pos)
        return held.copy(), root_pos.copy()


def simulate_unity_stream(
    model,
    diffusion,
    tracker_pos_world: np.ndarray,
    tracker_rot_world_6d: np.ndarray,
    sensor_valid: np.ndarray,
    device: torch.device,
    use_ddim: bool,
    schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    normalizer: RealtimePoseNormalizer | None = None,
    initial_root_yaw: float = 0.0,
    invalid_frame_policy: str = INVALID_FRAME_POLICY_HOLD,
    warmup_target_raw: np.ndarray | None = None,
    history_features_raw: np.ndarray | None = None,
    history_features_until_frame: int | None = None,
    reference_root_yaw: np.ndarray | None = None,
    reference_root_pos_world: np.ndarray | None = None,
    joint_offsets_parent: np.ndarray | None = None,
    joint_rest_local_rotations_6d: np.ndarray | None = None,
    timestamp_seconds: np.ndarray | None = None,
    floor_y: np.ndarray | None = None,
    tracking_origin_revision: np.ndarray | None = None,
    previous_to_current_world: np.ndarray | None = None,
    root_correction: bool = True,
    tracker_ik: bool = True,
    tracker_ik_iterations: int = DEFAULT_TRACKER_IK_ITERATIONS,
    tracker_ik_lr: float = DEFAULT_TRACKER_IK_LR,
    tracker_ik_blend: float = DEFAULT_TRACKER_IK_BLEND,
    tracker_ik_target_smoothing: float = DEFAULT_TRACKER_IK_TARGET_SMOOTHING,
    tracker_ik_delta_limit: float = DEFAULT_TRACKER_IK_DELTA_LIMIT,
    ik_init_mode: str = "random",
    ik_init_timestep: int = -1,
    ik_init_iterations: int = 16,
    ik_init_lr: float = 0.03,
    ik_init_pos_weight: float = 1.0,
    ik_init_rot_weight: float = 0.2,
    ik_init_reg_weight: float = 0.01,
    ik_init_delta_limit: float = 0.15,
) -> dict[str, np.ndarray]:
    schema = get_schema_spec(schema_name)
    ik_init_mode = validate_ik_init_mode(ik_init_mode)
    resolved_ik_init_timestep = (
        resolve_ik_init_timestep(diffusion, ik_init_timestep)
        if ik_init_mode == IK_INIT_MODE_TRACKER_POSE
        else -1
    )
    frame_count = int(tracker_pos_world.shape[0])
    timestamps = (
        np.arange(frame_count, dtype=np.float64) / 60.0
        if timestamp_seconds is None
        else np.asarray(timestamp_seconds, dtype=np.float64)
    )
    floor_values = (
        np.zeros(frame_count, dtype=np.float32)
        if floor_y is None
        else np.asarray(floor_y, dtype=np.float32)
    )
    origin_revisions = (
        np.zeros(frame_count, dtype=np.int64)
        if tracking_origin_revision is None
        else np.asarray(tracking_origin_revision, dtype=np.int64)
    )
    origin_transforms = None if previous_to_current_world is None else np.asarray(previous_to_current_world, dtype=np.float32)
    if timestamps.shape != (frame_count,):
        raise ValueError(f"timestamp_seconds must be [{frame_count}], got {timestamps.shape}")
    if floor_values.shape != (frame_count,):
        raise ValueError(f"floor_y must be [{frame_count}], got {floor_values.shape}")
    if origin_revisions.shape != (frame_count,):
        raise ValueError(f"tracking_origin_revision must be [{frame_count}], got {origin_revisions.shape}")
    if origin_transforms is not None and origin_transforms.shape != (frame_count, 4, 4):
        raise ValueError(f"previous_to_current_world must be [{frame_count},4,4], got {origin_transforms.shape}")
    state = UnityStreamState(
        schema_name=schema.name,
        initial_root_yaw=initial_root_yaw,
        invalid_frame_policy=invalid_frame_policy,
    )
    warmup_target = None if warmup_target_raw is None else np.asarray(warmup_target_raw, dtype=np.float32)
    if warmup_target is not None and warmup_target.shape != (schema.feature_dim,):
        raise ValueError(f"warmup_target_raw 应为 [{schema.feature_dim}]，实际为 {warmup_target.shape}")
    history_features = None if history_features_raw is None else np.asarray(history_features_raw, dtype=np.float32)
    if history_features is not None and history_features.shape != (frame_count, schema.feature_dim):
        raise ValueError(
            f"history_features_raw 应为 [{frame_count},{schema.feature_dim}]，实际为 {history_features.shape}"
        )
    history_until = (
        frame_count
        if history_features is not None and history_features_until_frame is None
        else max(0, min(frame_count, int(history_features_until_frame or 0)))
    )
    reference_yaw = None if reference_root_yaw is None else np.asarray(reference_root_yaw, dtype=np.float32)
    if reference_yaw is not None and reference_yaw.shape != (frame_count,):
        raise ValueError(f"reference_root_yaw 应为 [{frame_count}]，实际为 {reference_yaw.shape}")
    reference_root_pos = (
        None if reference_root_pos_world is None else np.asarray(reference_root_pos_world, dtype=np.float32)
    )
    if reference_root_pos is not None and reference_root_pos.shape != (frame_count, 3):
        raise ValueError(f"reference_root_pos_world 应为 [{frame_count},3]，实际为 {reference_root_pos.shape}")
    joint_offsets = None if joint_offsets_parent is None else np.asarray(joint_offsets_parent, dtype=np.float32)
    if joint_offsets is not None and joint_offsets.shape != (SMPL_JOINT_COUNT, 3):
        raise ValueError(f"joint_offsets_parent 应为 [{SMPL_JOINT_COUNT},3]，实际为 {joint_offsets.shape}")
    joint_rest_rotations = (
        None
        if joint_rest_local_rotations_6d is None
        else np.asarray(joint_rest_local_rotations_6d, dtype=np.float32)
    )
    if joint_rest_rotations is not None and joint_rest_rotations.shape != (SMPL_JOINT_COUNT, 6):
        raise ValueError(f"joint_rest_local_rotations_6d 应为 [{SMPL_JOINT_COUNT},6]，实际为 {joint_rest_rotations.shape}")

    predicted_features = []
    conditioned_features = []
    root_yaw_predicted = []
    preliminary_joints_world = []
    preliminary_root_yaw = []
    root_pos_predicted = []
    validity_flags = []
    is_predicted = []
    tracker_ref_root_pos = []
    tracker_ref_root_yaw = []
    tracker_ref_source = []
    root_sources = []
    reconnect_alphas = []
    codec_fk_elapsed_ms = []
    ddim_elapsed_ms = []
    resolver_elapsed_ms = []
    end_to_end_elapsed_ms = []

    for frame_index in range(frame_count):
        frame_started = time.perf_counter()
        state.last_preliminary_joints_world = None
        state.last_preliminary_root_yaw = None
        frame_valid = np.asarray(sensor_valid[frame_index], dtype=bool)
        validity_ok = sensor_validity_ok(frame_valid)
        if not validity_ok and invalid_frame_policy == INVALID_FRAME_POLICY_RAISE:
            raise ValueError(f"第 {frame_index} 帧 tracker 有效性不满足运行时合约：{frame_valid.astype(int).tolist()}")

        use_reference_history_frame = history_features is not None and frame_index < history_until
        use_reference_state_frame = (
            frame_index < history_until if history_features_until_frame is not None else True
        )

        if reference_yaw is not None and use_reference_state_frame:
            state.current_root_yaw = float(reference_yaw[max(frame_index - 1, 0)])
        if reference_root_pos is not None and use_reference_state_frame:
            state.current_root_pos_world = reference_root_pos[max(frame_index - 1, 0)].copy()

        codec_started = time.perf_counter()
        previous_final_root = (
            np.asarray([tracker_pos_world[frame_index, HEAD_TRACKER_INDEX, 0], floor_values[frame_index], tracker_pos_world[frame_index, HEAD_TRACKER_INDEX, 2]], dtype=np.float32)
            if state.current_root_pos_world is None
            else state.current_root_pos_world
        )
        pelvis_offset = np.zeros(3, dtype=np.float32) if joint_offsets is None else joint_offsets[0]
        root_pos, reference_yaw_value, reference_source_value = build_tracker_reference_np(
            tracker_pos_world=tracker_pos_world[frame_index],
            tracker_rot_world_6d=tracker_rot_world_6d[frame_index],
            sensor_valid=frame_valid,
            previous_final_root_pos_world=previous_final_root,
            previous_final_root_yaw=np.asarray(state.current_root_yaw, dtype=np.float32),
            pelvis_offset_parent=pelvis_offset,
            floor_y=float(floor_values[frame_index]),
        )
        tracker_feature_raw = encode_unity_tracker_frame(
            tracker_pos_world=tracker_pos_world[frame_index],
            tracker_rot_world_6d=tracker_rot_world_6d[frame_index],
            sensor_valid=frame_valid,
            reference_root_yaw=float(reference_yaw_value),
            schema_name=schema.name,
            root_pos_world=root_pos,
            joint_offsets_parent=joint_offsets,
        )
        codec_fk_elapsed_ms.append((time.perf_counter() - codec_started) * 1000.0)
        tracker_ref_root_pos.append(np.asarray(root_pos, dtype=np.float32).copy())
        tracker_ref_root_yaw.append(float(reference_yaw_value))
        tracker_ref_source.append(int(reference_source_value))
        history_target_raw = history_features[frame_index] if use_reference_history_frame else None

        if not state.has_full_history():
            warmup_root_height = (
                float(tracker_pos_world[frame_index, HIP_TRACKER_INDEX, 1])
                if frame_valid[HIP_TRACKER_INDEX]
                else float(joint_offsets[0, 1]) if joint_offsets is not None else 0.0
            )
            predicted_frame_raw, output_root_pos = state.append_warmup_frame(
                tracker_feature_raw=tracker_feature_raw,
                root_pos_world=root_pos,
                root_height=warmup_root_height,
                target_feature_raw=history_target_raw if history_target_raw is not None else warmup_target,
            )
            state.current_root_pos_world = np.asarray(root_pos, dtype=np.float32).copy()
            state.current_root_yaw = float(reference_yaw_value)
            state.current_pelvis_height = float(warmup_root_height)
            output_root_pos = state.current_root_pos_world.copy()
            if reference_yaw is not None and use_reference_state_frame:
                state.current_root_yaw = float(reference_yaw[frame_index])
            if reference_root_pos is not None and use_reference_state_frame:
                output_root_pos = reference_root_pos[frame_index].copy()
                state.current_root_pos_world = output_root_pos.copy()
            conditioned_frame_raw = tracker_feature_raw.copy()
            conditioned_frame_raw[schema.target_slice()] = 0.0
            frame_predicted = False
            ddim_elapsed_ms.append(0.0)
            state.last_resolver_elapsed_ms = 0.0
        elif not validity_ok:
            if history_target_raw is None:
                predicted_frame_raw, output_root_pos = state.hold_output(tracker_feature_raw, root_pos_world=root_pos)
            else:
                predicted_frame_raw = tracker_feature_raw.copy()
                predicted_frame_raw[schema.target_slice()] = history_target_raw[schema.target_slice()]
                output_root_pos = root_pos.copy()
                state.append_output_frame(
                    tracker_feature_raw=tracker_feature_raw,
                    output_frame_raw=predicted_frame_raw,
                    root_pos_world=output_root_pos,
                    history_target_raw=history_target_raw,
                )
                if reference_yaw is not None and use_reference_state_frame:
                    state.current_root_yaw = float(reference_yaw[frame_index])
                if reference_root_pos is not None and use_reference_state_frame:
                    state.current_root_pos_world = output_root_pos.copy()
            conditioned_frame_raw = tracker_feature_raw.copy()
            conditioned_frame_raw[schema.target_slice()] = 0.0
            frame_predicted = False
            ddim_elapsed_ms.append(0.0)
            state.last_resolver_elapsed_ms = 0.0
        else:
            window_raw = state.build_window_raw(tracker_feature_raw)
            conditioned_raw = normalize_conditioned_window(window_raw, normalizer=normalizer, schema_name=schema.name)
            batch = {
                "conditioned_x": torch.from_numpy(conditioned_raw.T).unsqueeze(0).float().to(device),
                "valid_frame_mask": torch.ones(1, REALTIME_POSE_SEQ_LEN, dtype=torch.bool, device=device),
            }
            if joint_offsets is not None:
                batch["joint_offsets_parent"] = torch.from_numpy(joint_offsets).unsqueeze(0).float().to(device)
            if joint_rest_rotations is not None:
                batch["joint_rest_local_rotations_6d"] = (
                    torch.from_numpy(joint_rest_rotations).unsqueeze(0).float().to(device)
                )
            ik_init_image = build_ik_init_image_for_batch(
                batch,
                device=device,
                schema_name=schema.name,
                normalizer=normalizer,
                ik_init_mode=ik_init_mode,
                ik_init_iterations=ik_init_iterations,
                ik_init_lr=ik_init_lr,
                ik_init_pos_weight=ik_init_pos_weight,
                ik_init_rot_weight=ik_init_rot_weight,
                ik_init_reg_weight=ik_init_reg_weight,
                ik_init_delta_limit=ik_init_delta_limit,
            )
            ddim_started = time.perf_counter()
            reconstructed = reconstruct_batch(
                model=model,
                diffusion=diffusion,
                batch=batch,
                device=device,
                use_ddim=use_ddim,
                schema_name=schema.name,
                init_image=ik_init_image,
                start_timestep=resolved_ik_init_timestep if ik_init_image is not None else None,
            )
            ddim_elapsed_ms.append((time.perf_counter() - ddim_started) * 1000.0)
            reconstructed_np = tensor_bct_to_numpy_btc(reconstructed)[0]
            reconstructed_raw = inverse_feature_window(reconstructed_np, normalizer=normalizer)
            predicted_frame_raw = reconstructed_raw[REALTIME_POSE_TARGET_START].copy()
            conditioned_frame_raw = inverse_feature_window(conditioned_raw, normalizer=normalizer)[REALTIME_POSE_TARGET_START]
            predicted_frame_raw, output_root_pos = state.accept_prediction(
                tracker_feature_raw=tracker_feature_raw,
                predicted_frame_raw=predicted_frame_raw,
                fallback_root_pos_world=root_pos,
                history_target_raw=history_target_raw,
                tracker_pos_world=tracker_pos_world[frame_index],
                tracker_rot_world_6d=tracker_rot_world_6d[frame_index],
                sensor_valid=frame_valid,
                joint_offsets_parent=joint_offsets,
                joint_rest_local_rotations_6d=joint_rest_rotations,
                root_correction=bool(root_correction),
                tracker_ik=bool(tracker_ik),
                tracker_ik_iterations=int(tracker_ik_iterations),
                tracker_ik_lr=float(tracker_ik_lr),
                tracker_ik_blend=float(tracker_ik_blend),
                tracker_ik_target_smoothing=float(tracker_ik_target_smoothing),
                tracker_ik_delta_limit=float(tracker_ik_delta_limit),
                timestamp=float(timestamps[frame_index]),
                floor_y=float(floor_values[frame_index]),
                tracking_origin_revision=int(origin_revisions[frame_index]),
                previous_to_current_world=(
                    None if origin_transforms is None else origin_transforms[frame_index]
                ),
            )
            frame_predicted = True

        if state.last_preliminary_joints_world is None and joint_offsets is not None:
            # warmup/invalid 帧不参与正式指标，但仍保存有限的 FK，避免结果文件出现 NaN 占位。
            state.last_preliminary_root_yaw = float(state.current_root_yaw)
            state.last_preliminary_joints_world = fk_joints_from_target(
                target_raw=predicted_frame_raw,
                root_pos_world=output_root_pos,
                root_yaw=state.current_root_yaw,
                joint_offsets_parent=joint_offsets,
                schema_name=schema.name,
                joint_rest_local_rotations_6d=joint_rest_rotations,
            ).astype(np.float32, copy=True)

        predicted_features.append(predicted_frame_raw.astype(np.float32))
        conditioned_features.append(conditioned_frame_raw.astype(np.float32))
        root_yaw_predicted.append(float(state.current_root_yaw))
        preliminary_joints_world.append(
            np.full((SMPL_JOINT_COUNT, 3), np.nan, dtype=np.float32)
            if state.last_preliminary_joints_world is None
            else state.last_preliminary_joints_world.astype(np.float32, copy=True)
        )
        preliminary_root_yaw.append(
            np.nan if state.last_preliminary_root_yaw is None else float(state.last_preliminary_root_yaw)
        )
        root_pos_predicted.append(output_root_pos.astype(np.float32))
        validity_flags.append(bool(validity_ok))
        is_predicted.append(bool(frame_predicted))
        root_sources.append(int(state.last_root_source))
        reconnect_alphas.append(float(state.last_reconnect_alpha))
        resolver_elapsed_ms.append(float(state.last_resolver_elapsed_ms))
        frame_elapsed_ms = (time.perf_counter() - frame_started) * 1000.0
        end_to_end_elapsed_ms.append(frame_elapsed_ms)
        # Include encoding, history assembly and all FK work, while keeping DDIM
        # and Resolver timings as non-overlapping headline buckets.
        codec_fk_elapsed_ms[-1] = max(
            0.0,
            frame_elapsed_ms - float(ddim_elapsed_ms[-1]) - float(resolver_elapsed_ms[-1]),
        )
        state.last_validity_ok = bool(validity_ok)

    predicted_mask = np.asarray(is_predicted, dtype=bool)
    validity_mask = np.asarray(validity_flags, dtype=bool)
    return {
        "schema_name": np.asarray(schema.name),
        "feature_space": np.asarray("raw"),
        "input_feature_space": np.asarray("normalized" if normalizer is not None else "raw"),
        "conditioned_features_raw": np.asarray(conditioned_features, dtype=np.float32)[None],
        "predicted_features_raw": np.asarray(predicted_features, dtype=np.float32)[None],
        "root_yaw_predicted": np.asarray(root_yaw_predicted, dtype=np.float32)[None],
        "preliminary_joints_world": np.asarray(preliminary_joints_world, dtype=np.float32)[None],
        "preliminary_root_yaw": np.asarray(preliminary_root_yaw, dtype=np.float32)[None],
        "root_pos_world_predicted": np.asarray(root_pos_predicted, dtype=np.float32)[None],
        "root_pos_world_estimated": np.asarray(root_pos_predicted, dtype=np.float32)[None],
        "tracker_pos_world": np.asarray(tracker_pos_world, dtype=np.float32)[None],
        "tracker_rot_world_6d": np.asarray(tracker_rot_world_6d, dtype=np.float32)[None],
        "sensor_valid": np.asarray(sensor_valid, dtype=bool)[None],
        "timestamp_seconds": timestamps[None],
        "floor_y": floor_values[None],
        "tracking_origin_revision": origin_revisions[None],
        "tracker_ref_root_pos_world": np.asarray(tracker_ref_root_pos, dtype=np.float32)[None],
        "tracker_ref_root_yaw": np.asarray(tracker_ref_root_yaw, dtype=np.float32)[None],
        "tracker_ref_source": np.asarray(tracker_ref_source, dtype=np.int8)[None],
        "root_source": np.asarray(root_sources, dtype=np.int8)[None],
        "reconnect_alpha": np.asarray(reconnect_alphas, dtype=np.float32)[None],
        "codec_fk_elapsed_ms": np.asarray(codec_fk_elapsed_ms, dtype=np.float32)[None],
        "ddim_elapsed_ms": np.asarray(ddim_elapsed_ms, dtype=np.float32)[None],
        "resolver_elapsed_ms": np.asarray(resolver_elapsed_ms, dtype=np.float32)[None],
        "end_to_end_elapsed_ms": np.asarray(end_to_end_elapsed_ms, dtype=np.float32)[None],
        "validity_ok": validity_mask[None],
        "is_predicted": predicted_mask[None],
        "eval_frame_mask": (predicted_mask & validity_mask)[None],
        "warmup_frames": np.asarray(REALTIME_POSE_TARGET_START, dtype=np.int64),
        "feature_contract_version": np.asarray(FEATURE_CONTRACT_VERSION, dtype=np.int64),
        "tracker_codec_version": np.asarray(TRACKER_CODEC_VERSION),
        "reference_policy_version": np.asarray(REFERENCE_POLICY_VERSION),
        "resolver_contract_version": np.asarray(RESOLVER_CONTRACT_VERSION),
        "root_correction": np.asarray(bool(root_correction)),
        "tracker_ik": np.asarray(bool(tracker_ik)),
        "tracker_ik_iterations": np.asarray(int(tracker_ik_iterations), dtype=np.int64),
        "tracker_ik_lr": np.asarray(float(tracker_ik_lr), dtype=np.float32),
        "tracker_ik_blend": np.asarray(float(tracker_ik_blend), dtype=np.float32),
        "tracker_ik_target_smoothing": np.asarray(float(tracker_ik_target_smoothing), dtype=np.float32),
        "tracker_ik_delta_limit": np.asarray(float(tracker_ik_delta_limit), dtype=np.float32),
        "ik_init_mode": np.asarray(ik_init_mode),
        "ik_init_timestep": np.asarray(int(resolved_ik_init_timestep), dtype=np.int64),
        "ik_init_iterations": np.asarray(int(ik_init_iterations), dtype=np.int64),
        "ik_init_lr": np.asarray(float(ik_init_lr), dtype=np.float32),
        "ik_init_pos_weight": np.asarray(float(ik_init_pos_weight), dtype=np.float32),
        "ik_init_rot_weight": np.asarray(float(ik_init_rot_weight), dtype=np.float32),
        "ik_init_reg_weight": np.asarray(float(ik_init_reg_weight), dtype=np.float32),
        "ik_init_delta_limit": np.asarray(float(ik_init_delta_limit), dtype=np.float32),
        "inpaint_mask": build_realtime_inpaint_mask(1, torch.device("cpu"), schema_name=schema.name)
        .cpu()
        .numpy()
        .transpose(0, 2, 1),
    }


def save_simulation(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def main(argv: list[str] | None = None) -> dict[str, Path]:
    parser = build_arg_parser()
    cli_schema_explicit = has_explicit_schema_arg(argv)
    args = parse_and_load_from_model(parser, argv=argv)
    checkpoint_args = load_args_json(Path(args.model_path))
    args.schema = resolve_runtime_schema(
        cli_schema=args.schema,
        checkpoint_args=checkpoint_args,
        cli_schema_explicit=cli_schema_explicit,
    )
    if int(args.seq_len) != REALTIME_POSE_SEQ_LEN:
        raise ValueError(f"Unity realtime stream 固定使用 {REALTIME_POSE_SEQ_LEN} 帧窗口，实际为 {args.seq_len}")
    schema = get_schema_spec(args.schema)
    if int(args.input_feats) != schema.feature_dim:
        raise ValueError(f"{schema.name} input_feats 应为 {schema.feature_dim}，实际为 {args.input_feats}")

    stream = load_tracker_stream(
        Path(args.tracker_stream_path).resolve(),
        assume_identity_tracker_rot=bool(args.assume_identity_tracker_rot),
        limit=int(args.limit),
    )
    normalizer = (
        RealtimePoseNormalizer(args.normalizer_dir, schema_name=schema.name)
        if bool(args.normalize_input)
        else None
    )

    dist_util.setup_dist(args.device if args.cuda else -1)
    device = dist_util.dev()
    model, diffusion = create_model_and_diffusion(args)
    model, source = load_checkpoint_model(model, args.model_path, device=device, use_ema=args.use_ema)
    payload = simulate_unity_stream(
        model=model,
        diffusion=diffusion,
        tracker_pos_world=stream["tracker_pos_world"],
        tracker_rot_world_6d=stream["tracker_rot_world_6d"],
        sensor_valid=stream["sensor_valid"],
        device=device,
        use_ddim=str(args.ts_respace).startswith("ddim"),
        schema_name=schema.name,
        normalizer=normalizer,
        initial_root_yaw=float(args.initial_root_yaw),
        invalid_frame_policy=args.invalid_frame_policy,
        joint_offsets_parent=stream.get("joint_offsets_parent"),
        joint_rest_local_rotations_6d=stream.get("joint_rest_local_rotations_6d"),
        timestamp_seconds=stream["timestamp_seconds"],
        floor_y=stream["floor_y"],
        tracking_origin_revision=stream["tracking_origin_revision"],
        previous_to_current_world=stream.get("previous_to_current_world"),
        root_correction=bool(args.root_correction),
        tracker_ik=bool(args.tracker_ik),
        tracker_ik_iterations=int(args.tracker_ik_iterations),
        tracker_ik_lr=float(args.tracker_ik_lr),
        tracker_ik_blend=float(args.tracker_ik_blend),
        tracker_ik_target_smoothing=float(args.tracker_ik_target_smoothing),
        tracker_ik_delta_limit=float(args.tracker_ik_delta_limit),
        ik_init_mode=args.ik_init_mode,
        ik_init_timestep=int(args.ik_init_timestep),
        ik_init_iterations=int(args.ik_init_iterations),
        ik_init_lr=float(args.ik_init_lr),
        ik_init_pos_weight=float(args.ik_init_pos_weight),
        ik_init_rot_weight=float(args.ik_init_rot_weight),
        ik_init_reg_weight=float(args.ik_init_reg_weight),
        ik_init_delta_limit=float(args.ik_init_delta_limit),
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (
            load_artifact_roots(args.artifact_roots_config or None).outputs_root
            / "unity_stream_simulation"
        )
    )
    output_path = output_dir / "unity_stream_simulation.npz"
    save_simulation(output_path, payload)
    print(f"[simulate_unity_stream] weights={source} output={output_path}")
    return {"output_path": output_path}


if __name__ == "__main__":
    main()
