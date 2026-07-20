from __future__ import annotations

from dataclasses import dataclass

import torch

from data_loaders.realtime_pose_kinematics import (
    fk_body_fbx_local_torch,
    make_yaw_rotation_torch,
    rotation_6d_to_matrix_torch,
)
from data_loaders.sensor_masking import HEAD_TRACKER_INDEX, HIP_TRACKER_INDEX
from data_loaders.tracker_codec import (
    decode_tracker_positions_torch,
    decode_tracker_rotations_torch,
)
from sample.runtime_root_resolver import RuntimeRootResolverConfig

@dataclass(frozen=True)
class DifferentiableResolverResult:
    final_root_pos_world: torch.Tensor
    final_root_yaw: torch.Tensor
    final_pelvis_height: torch.Tensor
    final_joints_world: torch.Tensor
    final_joint_rot_world: torch.Tensor
    preliminary_joints_world: torch.Tensor
    preliminary_root_yaw: torch.Tensor
    tracker_pos_world: torch.Tensor
    tracker_rot_world: torch.Tensor


def wrap_angle_torch(angle: torch.Tensor) -> torch.Tensor:
    """使用 atan2 保持角度可微，并映射到 [-pi, pi]。"""

    return torch.atan2(torch.sin(angle), torch.cos(angle))


def rotation_matrix_log_vector(rotation: torch.Tensor) -> torch.Tensor:
    """把 [...,3,3] 旋转矩阵转换为稳定的 axis-angle 向量。"""

    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix 应以 [3,3] 结尾，实际为 {tuple(rotation.shape)}")
    rotation = rotation.float()
    trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    angle = torch.acos(cosine)
    vee = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    sine = torch.sin(angle)
    scale = angle / (2.0 * sine).clamp_min(1e-6)
    scale = torch.where(angle < 1e-4, torch.full_like(scale, 0.5), scale)
    return vee * scale.unsqueeze(-1)


def _require_tensor(y: dict, name: str, *, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    if name not in y:
        raise KeyError(f"differentiable Resolver 缺少 batch 字段：{name}")
    value = y[name].to(device=device)
    return value if dtype is None else value.to(dtype=dtype)


def _run_body_fbx_fk(
    *,
    pose: torch.Tensor,
    root: torch.Tensor,
    yaw: torch.Tensor,
    pelvis_height: torch.Tensor,
    offsets: torch.Tensor,
    rest_local_rotations_6d: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    fk_offsets = offsets.clone()
    fk_offsets[:, 0, 1] = pelvis_height
    return fk_body_fbx_local_torch(
        body_pose_local_delta_6d=pose,
        actor_root_pos_world=root,
        root_heading=yaw,
        rest_local_positions=fk_offsets,
        rest_local_rotations_6d=rest_local_rotations_6d,
        return_global_rot=True,
    )


def resolve_realtime_pose_frame_torch(
    *,
    pred_pose: torch.Tensor,
    pred_root_delta_xz_ref: torch.Tensor,
    pred_yaw_delta_sincos: torch.Tensor,
    pred_pelvis_height: torch.Tensor,
    y: dict,
    config: RuntimeRootResolverConfig | None = None,
) -> DifferentiableResolverResult:
    """可微地复现 RuntimeRootResolver 的单帧 Hip/no-Hip/reconnect/filter 路径。"""

    if pred_pose.ndim != 2 or pred_pose.shape[-1] != 24 * 6:
        raise ValueError(f"pred_pose 应为 [B,144]，实际为 {tuple(pred_pose.shape)}")
    batch_size = pred_pose.shape[0]
    device = pred_pose.device
    dtype = pred_pose.dtype
    cfg = config or RuntimeRootResolverConfig()

    ref_root_pos = _require_tensor(y, "tracker_ref_root_pos_world", device=device, dtype=dtype)
    ref_root_yaw = _require_tensor(y, "tracker_ref_root_yaw", device=device, dtype=dtype).view(-1)
    tracker_pos_ref = _require_tensor(y, "target_tracker_pos_ref", device=device, dtype=dtype)
    tracker_rot_ref = _require_tensor(y, "target_tracker_rot_ref_6d", device=device, dtype=dtype)
    sensor_valid = _require_tensor(y, "target_sensor_valid", device=device).bool()
    floor_y = _require_tensor(y, "target_floor_y", device=device, dtype=dtype).view(-1)
    offsets = _require_tensor(y, "joint_offsets_parent", device=device, dtype=dtype)
    if sensor_valid.shape != (batch_size, 6):
        raise ValueError(f"target_sensor_valid 应为 [B,6]，实际为 {tuple(sensor_valid.shape)}")

    tracker_pos_world = decode_tracker_positions_torch(tracker_pos_ref, ref_root_pos, ref_root_yaw)
    tracker_rot_world_6d = decode_tracker_rotations_torch(tracker_rot_ref, ref_root_yaw)
    tracker_rot_world = rotation_6d_to_matrix_torch(tracker_rot_world_6d)

    previous_root = _require_tensor(y, "resolver_before_target_root_pos_world", device=device, dtype=dtype)
    previous_yaw = _require_tensor(y, "resolver_before_target_root_yaw", device=device, dtype=dtype).view(-1)
    previous_height = _require_tensor(
        y, "resolver_before_target_pelvis_height", device=device, dtype=dtype
    ).view(-1)
    previous_hip_valid = _require_tensor(y, "resolver_before_target_hip_valid", device=device).bool().view(-1)
    reconnect_start_root = _require_tensor(
        y, "resolver_before_target_reconnect_start_root_pos_world", device=device, dtype=dtype
    )
    reconnect_start_yaw = _require_tensor(
        y, "resolver_before_target_reconnect_start_root_yaw", device=device, dtype=dtype
    ).view(-1)
    reconnect_start_height = _require_tensor(
        y, "resolver_before_target_reconnect_start_pelvis_height", device=device, dtype=dtype
    ).view(-1)
    reconnect_elapsed = _require_tensor(
        y, "resolver_before_target_reconnect_elapsed_seconds", device=device, dtype=dtype
    ).view(-1)
    previous_timestamp = _require_tensor(
        y, "resolver_before_target_last_timestamp_seconds", device=device, dtype=dtype
    ).view(-1)
    previous_revision = _require_tensor(
        y, "resolver_before_target_tracking_origin_revision", device=device
    ).long().view(-1)
    timestamp = _require_tensor(y, "target_timestamp_seconds", device=device, dtype=dtype).view(-1)
    revision = _require_tensor(y, "target_tracking_origin_revision", device=device).long().view(-1)

    delta_seconds_raw = timestamp - previous_timestamp
    reset_boundary = (
        (revision != previous_revision)
        | (delta_seconds_raw < 0.0)
        | (delta_seconds_raw > float(cfg.timestamp_reset_threshold_seconds))
    )
    state_valid = ~reset_boundary
    delta_seconds = torch.where(state_valid, delta_seconds_raw.clamp_min(0.0), torch.zeros_like(delta_seconds_raw))

    head = tracker_pos_world[:, HEAD_TRACKER_INDEX]
    fallback_root = torch.stack((head[:, 0], floor_y, head[:, 2]), dim=-1)
    model_previous_root = torch.where(state_valid[:, None], previous_root, fallback_root)
    model_previous_yaw = torch.where(state_valid, previous_yaw, torch.zeros_like(previous_yaw))
    delta_3d = torch.zeros_like(model_previous_root)
    delta_3d[:, 0] = pred_root_delta_xz_ref[:, 0]
    delta_3d[:, 2] = pred_root_delta_xz_ref[:, 1]
    model_root = model_previous_root + torch.einsum(
        "bij,bj->bi", make_yaw_rotation_torch(model_previous_yaw), delta_3d
    )
    model_root = torch.stack((model_root[:, 0], floor_y, model_root[:, 2]), dim=-1)
    yaw_pair = pred_yaw_delta_sincos.float()
    yaw_pair_norm = torch.linalg.norm(yaw_pair, dim=-1)
    safe_yaw_sin = yaw_pair[:, 0]
    safe_yaw_cos = torch.where(
        yaw_pair_norm > 1e-6,
        yaw_pair[:, 1],
        torch.ones_like(yaw_pair[:, 1]),
    )
    model_yaw = wrap_angle_torch(
        model_previous_yaw + torch.atan2(safe_yaw_sin, safe_yaw_cos)
    )
    model_height = pred_pelvis_height.view(-1)

    rest = y.get("joint_rest_local_rotations_6d")
    rest = None if rest is None else rest.to(device=device, dtype=dtype)
    preliminary_joints, _ = _run_body_fbx_fk(
        pose=pred_pose,
        root=model_root,
        yaw=model_yaw,
        pelvis_height=model_height,
        offsets=offsets,
        rest_local_rotations_6d=rest,
    )

    hip_rotation = tracker_rot_world[:, HIP_TRACKER_INDEX]
    hip_yaw = torch.atan2(hip_rotation[:, 0, 2], hip_rotation[:, 2, 2])
    pelvis_offset = offsets[:, 0]
    hip_root = tracker_pos_world[:, HIP_TRACKER_INDEX] - torch.einsum(
        "bij,bj->bi", make_yaw_rotation_torch(hip_yaw), pelvis_offset
    )
    hip_root = torch.stack((hip_root[:, 0], floor_y, hip_root[:, 2]), dim=-1)
    hip_height = tracker_pos_world[:, HIP_TRACKER_INDEX, 1] - floor_y

    duration = max(float(cfg.reconnect_duration_seconds), 1e-6)
    reconnect_active = (reconnect_elapsed > 0.0) & (reconnect_elapsed < duration)
    reconnect_needed = state_valid & ((~previous_hip_valid) | reconnect_active)
    start_root = torch.where(reconnect_active[:, None], reconnect_start_root, previous_root)
    start_yaw = torch.where(reconnect_active, reconnect_start_yaw, previous_yaw)
    start_height = torch.where(reconnect_active, reconnect_start_height, previous_height)
    reconnect_alpha = ((reconnect_elapsed + delta_seconds) / duration).clamp(0.0, 1.0)
    blend = reconnect_alpha.square() * (3.0 - 2.0 * reconnect_alpha)
    reconnect_root = (1.0 - blend[:, None]) * start_root + blend[:, None] * hip_root
    reconnect_yaw = wrap_angle_torch(start_yaw + wrap_angle_torch(hip_yaw - start_yaw) * blend)
    reconnect_height = (1.0 - blend) * start_height + blend * hip_height

    tau = float(cfg.hip_filter_time_constant_seconds)
    if tau <= 0.0:
        filter_alpha = torch.ones_like(delta_seconds)
    else:
        filter_alpha = torch.where(
            delta_seconds > 0.0,
            1.0 - torch.exp(-delta_seconds / tau),
            torch.ones_like(delta_seconds),
        )
    filter_alpha = torch.where(state_valid, filter_alpha, torch.ones_like(filter_alpha))
    filtered_root = (1.0 - filter_alpha[:, None]) * previous_root + filter_alpha[:, None] * hip_root
    filtered_yaw = wrap_angle_torch(
        previous_yaw + wrap_angle_torch(hip_yaw - previous_yaw) * filter_alpha
    )
    filtered_height = (1.0 - filter_alpha) * previous_height + filter_alpha * hip_height

    hip_final_root = torch.where(reconnect_needed[:, None], reconnect_root, filtered_root)
    hip_final_yaw = torch.where(reconnect_needed, reconnect_yaw, filtered_yaw)
    hip_final_height = torch.where(reconnect_needed, reconnect_height, filtered_height)

    predicted_head = preliminary_joints[:, 15]
    root_to_head = predicted_head - model_root
    head_candidate_root = torch.stack(
        (
            head[:, 0] - root_to_head[:, 0],
            floor_y,
            head[:, 2] - root_to_head[:, 2],
        ),
        dim=-1,
    )
    head_weight = float(cfg.nohip_head_anchor_weight)
    head_final_root = (1.0 - head_weight) * model_root + head_weight * head_candidate_root
    head_final_root = torch.stack((head_final_root[:, 0], floor_y, head_final_root[:, 2]), dim=-1)
    height_correction = (head[:, 1] - predicted_head[:, 1]).clamp(
        -float(cfg.max_head_height_correction_m),
        float(cfg.max_head_height_correction_m),
    )
    head_final_height = model_height + height_correction

    hip_valid = sensor_valid[:, HIP_TRACKER_INDEX]
    final_root = torch.where(hip_valid[:, None], hip_final_root, head_final_root)
    final_yaw = torch.where(hip_valid, hip_final_yaw, model_yaw)
    final_height = torch.where(hip_valid, hip_final_height, head_final_height)
    final_root = torch.stack((final_root[:, 0], floor_y, final_root[:, 2]), dim=-1)
    final_joints, final_rotations = _run_body_fbx_fk(
        pose=pred_pose,
        root=final_root,
        yaw=final_yaw,
        pelvis_height=final_height,
        offsets=offsets,
        rest_local_rotations_6d=rest,
    )
    return DifferentiableResolverResult(
        final_root_pos_world=final_root,
        final_root_yaw=final_yaw,
        final_pelvis_height=final_height,
        final_joints_world=final_joints,
        final_joint_rot_world=final_rotations,
        preliminary_joints_world=preliminary_joints,
        preliminary_root_yaw=model_yaw,
        tracker_pos_world=tracker_pos_world,
        tracker_rot_world=tracker_rot_world,
    )
