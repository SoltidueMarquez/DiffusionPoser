from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from data_loaders.realtime_pose_config import TrackerReliabilityConfig
from data_loaders.realtime_pose_geometry import (
    assemble_tracker_features_np,
    build_head_path_window_np,
    build_pose_target_np,
    build_tracker_measurements_np,
    decode_target_head_rotations_np,
    extract_forward_yaw_np,
    global_head_rotations_to_local_delta_6d_np,
    resolve_root_head_reference_np,
)
from data_loaders.realtime_pose_kinematics import make_yaw_rotation_np, rotation_6d_to_matrix_np
from data_loaders.sensor_masking import (
    HEAD_TRACKER_INDEX,
    REALTIME_POSE_FRAME_OFFSETS,
    REALTIME_POSE_HISTORY_ANCHOR_COUNT,
    REALTIME_POSE_HISTORY_ANCHOR_INDICES,
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_TARGET_DIM,
    REALTIME_POSE_WINDOW_LENGTH,
    TRACKER_COUNT,
    TRACKER_FEATURE_DIM,
    TRACKER_TO_JOINT,
)
from data_loaders.tracker_reliability import (
    compute_hard_rotation_state_np,
    compute_region_coverage_np,
    compute_tracker_reliability_np,
)
from diffusion.realtime_pose_projection import project_realtime_pose_xstart


@dataclass(frozen=True)
class WorldPoseState:
    joint_rotations_world: np.ndarray
    root_yaw_world: float
    hip_height: float
    root_position_world: np.ndarray


@dataclass(frozen=True)
class ResolvedPose:
    target_raw: np.ndarray
    joint_rotations_world: np.ndarray
    body_local_delta_6d: np.ndarray
    root_yaw_world: float
    hip_height: float
    root_position_world: np.ndarray
    joints_world: np.ndarray
    hard_rotation_max_error: float

    def as_world_state(self) -> WorldPoseState:
        return WorldPoseState(
            joint_rotations_world=self.joint_rotations_world.copy(),
            root_yaw_world=float(self.root_yaw_world),
            hip_height=float(self.hip_height),
            root_position_world=self.root_position_world.copy(),
        )


@dataclass(frozen=True)
class RuntimeStepResult:
    resolved_pose: ResolvedPose
    raw_pred_xstart: np.ndarray
    deployed_pred_xstart: np.ndarray
    kappa_position: np.ndarray
    kappa_rotation: np.ndarray
    hard_rotation_state: np.ndarray
    current_tracker_raw: np.ndarray
    future_leg_prediction: np.ndarray | None = None
    contact_logits: np.ndarray | None = None


@dataclass(frozen=True)
class _TrackerFrameState:
    tracker_pos_world: np.ndarray
    tracker_rot_world_6d: np.ndarray
    configured: np.ndarray
    measured_valid: np.ndarray
    d_off: np.ndarray
    d_on: np.ndarray
    head_yaw_world: float
    floor_y: float


@dataclass(frozen=True)
class _PreparedRuntimeStep:
    position: np.ndarray
    rotation_6d: np.ndarray
    configured: np.ndarray
    measured: np.ndarray
    d_off: np.ndarray
    d_on: np.ndarray
    kappa_position: np.ndarray
    kappa_rotation: np.ndarray
    head_yaw: float
    floor_y: float
    tracker_window_raw: np.ndarray
    hard_rotation_state_window: np.ndarray
    conditioning: dict[str, torch.Tensor]


class RealtimePoseRuntime:
    """无 GT warmup 的状态化 Python runtime。"""

    def __init__(
        self,
        model,
        diffusion,
        device: torch.device,
        joint_offsets_parent: np.ndarray,
        joint_rest_local_rotations_6d: np.ndarray,
        normalizer=None,
        projected_ddim_mode: str = "all_steps",
        projected_ddim_late_steps: int = 5,
    ):
        self.model = model
        self.diffusion = diffusion
        self.device = device
        self.normalizer = normalizer
        reliability_config = getattr(model, "reliability_config", None)
        if not isinstance(reliability_config, TrackerReliabilityConfig):
            raise TypeError("RealtimePoseRuntime 要求模型公开 TrackerReliabilityConfig。")
        self.reliability_config = reliability_config.validate()
        self.projected_ddim_mode = str(projected_ddim_mode)
        self.projected_ddim_late_steps = int(projected_ddim_late_steps)
        self.joint_offsets_parent = np.asarray(joint_offsets_parent, dtype=np.float32).reshape(24, 3)
        self.joint_rest_local_rotations_6d = np.asarray(
            joint_rest_local_rotations_6d, dtype=np.float32
        ).reshape(24, 6)
        self.pose_history: list[WorldPoseState] = []
        self.tracker_history: list[_TrackerFrameState] = []
        self.previous_d_off = np.zeros(TRACKER_COUNT, dtype=np.int64)
        self.previous_d_on = np.zeros(TRACKER_COUNT, dtype=np.int64)
        self.previous_head_yaw: float | None = None

    def step(
        self,
        tracker_pos_world: np.ndarray,
        tracker_rot_world_6d: np.ndarray,
        configured: np.ndarray,
        measured_valid: np.ndarray,
        floor_y: float,
    ) -> RuntimeStepResult:
        return step_realtime_pose_batch(
            [self],
            np.asarray(tracker_pos_world, dtype=np.float32)[None],
            np.asarray(tracker_rot_world_6d, dtype=np.float32)[None],
            np.asarray(configured, dtype=bool)[None],
            np.asarray(measured_valid, dtype=bool)[None],
            np.asarray([floor_y], dtype=np.float32),
        )[0]

    def _build_conditioning(
        self,
        position: np.ndarray,
        rotation_6d: np.ndarray,
        configured: np.ndarray,
        measured: np.ndarray,
        d_off: np.ndarray,
        d_on: np.ndarray,
        head_yaw: float,
        floor_y: float,
    ) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        history_count = min(len(self.pose_history), REALTIME_POSE_HISTORY_LENGTH)
        if len(self.tracker_history) != len(self.pose_history):
            raise RuntimeError("Pose 与 Tracker 密集历史长度不一致。")
        session_start = REALTIME_POSE_HISTORY_LENGTH - history_count
        anchor_indices = np.asarray(REALTIME_POSE_HISTORY_ANCHOR_INDICES, dtype=np.int64)
        history_valid = anchor_indices >= session_start
        window_valid = np.concatenate([history_valid, np.asarray([True])])

        pose_raw = np.zeros(
            (REALTIME_POSE_HISTORY_ANCHOR_COUNT, REALTIME_POSE_TARGET_DIM),
            dtype=np.float32,
        )
        tracker_positions = np.zeros(
            (REALTIME_POSE_WINDOW_LENGTH, TRACKER_COUNT, 3), dtype=np.float32
        )
        tracker_rotations = np.zeros(
            (REALTIME_POSE_WINDOW_LENGTH, TRACKER_COUNT, 6), dtype=np.float32
        )
        configured_window = np.zeros(
            (REALTIME_POSE_WINDOW_LENGTH, TRACKER_COUNT), dtype=bool
        )
        measured_window = np.zeros_like(configured_window)
        d_off_window = np.zeros_like(configured_window, dtype=np.int64)
        d_on_window = np.zeros_like(configured_window, dtype=np.int64)
        head_positions = np.zeros((REALTIME_POSE_WINDOW_LENGTH, 3), dtype=np.float32)
        head_yaws = np.zeros(REALTIME_POSE_WINDOW_LENGTH, dtype=np.float32)

        selected_pose_rotations: list[np.ndarray] = []
        selected_pose_slots: list[int] = []
        for slot, dense_index in enumerate(anchor_indices.tolist()):
            if dense_index < session_start:
                continue
            history_index = dense_index - session_start
            pose_state = self.pose_history[history_index]
            tracker_state = self.tracker_history[history_index]
            selected_pose_slots.append(slot)
            selected_pose_rotations.append(pose_state.joint_rotations_world)
            tracker_positions[slot] = tracker_state.tracker_pos_world
            tracker_rotations[slot] = tracker_state.tracker_rot_world_6d
            configured_window[slot] = tracker_state.configured
            measured_window[slot] = tracker_state.measured_valid
            d_off_window[slot] = tracker_state.d_off
            d_on_window[slot] = tracker_state.d_on
            head_positions[slot] = tracker_state.tracker_pos_world[HEAD_TRACKER_INDEX]
            head_yaws[slot] = tracker_state.head_yaw_world
        if selected_pose_rotations:
            pose_raw[np.asarray(selected_pose_slots, dtype=np.int64)] = build_pose_target_np(
                np.stack(selected_pose_rotations, axis=0), head_yaw
            )

        tracker_positions[-1] = position
        tracker_rotations[-1] = rotation_6d
        configured_window[-1] = configured
        measured_window[-1] = measured
        d_off_window[-1] = d_off
        d_on_window[-1] = d_on
        head_positions[-1] = position[HEAD_TRACKER_INDEX]
        head_yaws[-1] = head_yaw
        tracker_measurements = build_tracker_measurements_np(
            tracker_positions,
            tracker_rotations,
            position[HEAD_TRACKER_INDEX],
            floor_y,
            head_yaw,
        )
        tracker_raw = np.zeros(
            (REALTIME_POSE_WINDOW_LENGTH, TRACKER_COUNT, TRACKER_FEATURE_DIM),
            dtype=np.float32,
        )
        tracker_raw[window_valid] = assemble_tracker_features_np(
            tracker_measurements[window_valid],
            configured_window[window_valid],
            measured_window[window_valid],
            d_off_window[window_valid],
            d_on_window[window_valid],
            duration_cap=self.reliability_config.duration_cap,
        )
        hard_rotation_window = np.zeros_like(configured_window)
        hard_rotation_window[window_valid] = compute_hard_rotation_state_np(
            configured_window[window_valid],
            measured_window[window_valid],
            d_on_window[window_valid],
            self.reliability_config,
        )
        head_path_raw = build_head_path_window_np(
            head_positions,
            head_yaws,
            position[HEAD_TRACKER_INDEX],
            floor_y,
            head_yaw,
        )
        pose = self._normalize_pose_numpy(pose_raw)
        tracker_window = self._normalize_tracker_numpy(tracker_raw)
        head_path = (
            head_path_raw
            if self.normalizer is None
            else np.asarray(
                self.normalizer.normalize_head_path(head_path_raw), dtype=np.float32
            )
        )
        pose[~history_valid] = 0.0
        tracker_window[~window_valid] = 0.0
        head_path[~window_valid] = 0.0
        kappa_position, kappa_rotation = compute_tracker_reliability_np(
            configured_window[:-1], measured_window[:-1], d_on_window[:-1],
            config=self.reliability_config,
        )
        rho_position, rho_rotation = compute_region_coverage_np(
            kappa_position, kappa_rotation
        )
        history_confidence = 0.5 * (rho_position + rho_rotation)
        history_confidence *= history_valid[:, None]
        conditioning = {
            "history_pose_observation": torch.as_tensor(
                pose, device=self.device, dtype=torch.float32
            ).unsqueeze(0),
            "tracker_window": torch.as_tensor(
                tracker_window, device=self.device, dtype=torch.float32
            ).unsqueeze(0),
            "head_path_window": torch.as_tensor(
                head_path, device=self.device, dtype=torch.float32
            ).unsqueeze(0),
            "history_region_confidence": torch.as_tensor(
                history_confidence, device=self.device, dtype=torch.float32
            ).unsqueeze(0),
            "window_valid_mask": torch.as_tensor(
                window_valid, device=self.device, dtype=torch.bool
            ).unsqueeze(0),
            "frame_offsets": torch.tensor(
                REALTIME_POSE_FRAME_OFFSETS, device=self.device, dtype=torch.long
            ).unsqueeze(0),
        }
        return conditioning, tracker_raw, hard_rotation_window


    def _normalizer_pose_stats(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.normalizer is None:
            return None, None
        return (
            self.normalizer.pose_mean.to(self.device),
            self.normalizer.pose_scale.to(self.device),
        )

    def _normalize_pose_numpy(self, value: np.ndarray) -> np.ndarray:
        return value if self.normalizer is None else np.asarray(self.normalizer.normalize_pose(value), dtype=np.float32)

    def _normalize_tracker_numpy(self, value: np.ndarray) -> np.ndarray:
        return value if self.normalizer is None else np.asarray(self.normalizer.normalize_tracker(value), dtype=np.float32)


    def _inverse_pose(self, value: torch.Tensor) -> np.ndarray:
        cpu = value.detach().cpu()
        if self.normalizer is not None:
            cpu = self.normalizer.inverse_pose(cpu)
        return np.asarray(cpu, dtype=np.float32)

    def _prepare_step(
        self,
        tracker_pos_world: np.ndarray,
        tracker_rot_world_6d: np.ndarray,
        configured: np.ndarray,
        measured_valid: np.ndarray,
        floor_y: float,
    ) -> _PreparedRuntimeStep:
        """只构建当前帧条件，不采样也不推进该序列的历史。"""

        position = np.asarray(tracker_pos_world, dtype=np.float32).reshape(TRACKER_COUNT, 3)
        rotation_6d = np.asarray(tracker_rot_world_6d, dtype=np.float32).reshape(TRACKER_COUNT, 6)
        configured = np.asarray(configured, dtype=bool).reshape(TRACKER_COUNT)
        measured = np.asarray(measured_valid, dtype=bool).reshape(TRACKER_COUNT)
        if np.any(measured & ~configured):
            raise ValueError("measured_valid 必须是 configured 的子集。")
        if not configured[HEAD_TRACKER_INDEX] or not measured[HEAD_TRACKER_INDEX]:
            raise ValueError("Head 必须始终为 configured 且 measured_valid。")

        head_rotation = rotation_6d_to_matrix_np(
            rotation_6d[HEAD_TRACKER_INDEX : HEAD_TRACKER_INDEX + 1]
        )
        head_yaw = float(
            extract_forward_yaw_np(
                head_rotation,
                initial_yaw=0.0 if self.previous_head_yaw is None else self.previous_head_yaw,
            )[0]
        )
        d_off, d_on = _advance_durations(
            self.previous_d_off,
            self.previous_d_on,
            configured,
            measured,
            self.reliability_config.duration_cap,
        )
        kappa_pos, kappa_rot = compute_tracker_reliability_np(
            configured,
            measured,
            d_on,
            config=self.reliability_config,
        )
        conditioning, tracker_window_raw, hard_rotation_state_window = self._build_conditioning(
            position,
            rotation_6d,
            configured,
            measured,
            d_off,
            d_on,
            head_yaw,
            float(floor_y),
        )
        return _PreparedRuntimeStep(
            position=position,
            rotation_6d=rotation_6d,
            configured=configured,
            measured=measured,
            d_off=d_off,
            d_on=d_on,
            kappa_position=kappa_pos,
            kappa_rotation=kappa_rot,
            head_yaw=head_yaw,
            floor_y=float(floor_y),
            tracker_window_raw=tracker_window_raw,
            hard_rotation_state_window=hard_rotation_state_window,
            conditioning=conditioning,
        )

    def _finish_step(
        self,
        prepared: _PreparedRuntimeStep,
        raw_target: np.ndarray,
        deployed_target: np.ndarray,
        future_leg: np.ndarray | None,
        contact_logits: np.ndarray | None,
    ) -> RuntimeStepResult:
        """用采样结果推进一条序列，不影响批内其他序列。"""

        resolved = decode_and_resolve_pose(
            deployed_target,
            prepared.tracker_window_raw[-1],
            prepared.hard_rotation_state_window[-1],
            prepared.head_yaw,
            prepared.position[HEAD_TRACKER_INDEX],
            prepared.floor_y,
            self.joint_offsets_parent,
            self.joint_rest_local_rotations_6d,
        )
        frame_state = _TrackerFrameState(
            prepared.position.copy(),
            prepared.rotation_6d.copy(),
            prepared.configured.copy(),
            prepared.measured.copy(),
            prepared.d_off.copy(),
            prepared.d_on.copy(),
            prepared.head_yaw,
            prepared.floor_y,
        )
        self.pose_history.append(resolved.as_world_state())
        self.tracker_history.append(frame_state)
        self.pose_history = self.pose_history[-REALTIME_POSE_HISTORY_LENGTH:]
        self.tracker_history = self.tracker_history[-REALTIME_POSE_HISTORY_LENGTH:]
        self.previous_d_off = prepared.d_off.copy()
        self.previous_d_on = prepared.d_on.copy()
        self.previous_head_yaw = prepared.head_yaw
        return RuntimeStepResult(
            resolved_pose=resolved,
            raw_pred_xstart=np.asarray(raw_target, dtype=np.float32),
            deployed_pred_xstart=np.asarray(deployed_target, dtype=np.float32),
            kappa_position=prepared.kappa_position.copy(),
            kappa_rotation=prepared.kappa_rotation.copy(),
            hard_rotation_state=prepared.hard_rotation_state_window[-1].copy(),
            current_tracker_raw=prepared.tracker_window_raw[-1].copy(),
            future_leg_prediction=(
                None if future_leg is None else np.asarray(future_leg, dtype=np.float32)
            ),
            contact_logits=(
                None if contact_logits is None else np.asarray(contact_logits, dtype=np.float32)
            ),
        )


def step_realtime_pose_batch(
    runtimes: list[RealtimePoseRuntime],
    tracker_pos_world: np.ndarray,
    tracker_rot_world_6d: np.ndarray,
    configured: np.ndarray,
    measured_valid: np.ndarray,
    floor_y: np.ndarray,
    noise: torch.Tensor | None = None,
) -> list[RuntimeStepResult]:
    """将多条独立序列的当前帧合成一次模型采样。

    批内条件张量为 ``[B, ...]``；每个 runtime 仍各自维护历史、
    tracker 持续时间和头部轨迹，因此可以在序列结束后缩小活跃批。
    """

    if not runtimes:
        raise ValueError("跨序列批处理至少需要一个 runtime。")
    batch_size = len(runtimes)
    first = runtimes[0]
    for runtime in runtimes[1:]:
        if runtime.model is not first.model or runtime.diffusion is not first.diffusion:
            raise ValueError("批内 runtime 必须共享同一个 model 和 diffusion。")
        if runtime.device != first.device or runtime.normalizer is not first.normalizer:
            raise ValueError("批内 runtime 必须共享 device 和 normalizer。")
        if (
            runtime.projected_ddim_mode != first.projected_ddim_mode
            or runtime.projected_ddim_late_steps != first.projected_ddim_late_steps
        ):
            raise ValueError("批内 runtime 必须使用相同的 Projected DDIM 设置。")

    positions = np.asarray(tracker_pos_world, dtype=np.float32).reshape(
        batch_size, TRACKER_COUNT, 3
    )
    rotations = np.asarray(tracker_rot_world_6d, dtype=np.float32).reshape(
        batch_size, TRACKER_COUNT, 6
    )
    configured_batch = np.asarray(configured, dtype=bool).reshape(batch_size, TRACKER_COUNT)
    measured_batch = np.asarray(measured_valid, dtype=bool).reshape(batch_size, TRACKER_COUNT)
    floor_batch = np.asarray(floor_y, dtype=np.float32).reshape(batch_size)
    prepared_steps = [
        runtime._prepare_step(
            positions[index],
            rotations[index],
            configured_batch[index],
            measured_batch[index],
            float(floor_batch[index]),
        )
        for index, runtime in enumerate(runtimes)
    ]
    conditioning = {
        name: torch.cat([step.conditioning[name] for step in prepared_steps], dim=0)
        for name in prepared_steps[0].conditioning
    }
    model_impl = getattr(first.model, "module", first.model)
    # 历史编码在整个 DDIM 采样中复用，且评测时不保留计算图。
    with torch.no_grad():
        prepared_conditioning = model_impl.prepare_conditioning(
            conditioning["history_pose_observation"],
            conditioning["tracker_window"],
            conditioning["head_path_window"],
            conditioning["history_region_confidence"],
            conditioning["window_valid_mask"],
            conditioning["frame_offsets"],
        )
    tracker_window_tensor = torch.as_tensor(
        np.stack([step.tracker_window_raw for step in prepared_steps]),
        device=first.device,
        dtype=torch.float32,
    )
    hard_window_tensor = torch.as_tensor(
        np.stack([step.hard_rotation_state_window for step in prepared_steps]),
        device=first.device,
        dtype=torch.bool,
    )
    if noise is not None:
        if tuple(noise.shape) != (batch_size, REALTIME_POSE_TARGET_DIM):
            raise ValueError(
                f"noise 应为 [{batch_size},{REALTIME_POSE_TARGET_DIM}]，实际为 {tuple(noise.shape)}"
            )
        noise = noise.to(device=first.device, dtype=torch.float32)
    mean, scale = first._normalizer_pose_stats()
    sample = first.diffusion.projected_ddim_sample_loop(
        first.model,
        shape=(batch_size, REALTIME_POSE_TARGET_DIM),
        projection_fn=lambda value: project_realtime_pose_xstart(
            value,
            tracker_window_tensor[:, -1],
            hard_window_tensor[:, -1],
            mean,
            scale,
        ),
        clip_denoised=False,
        model_kwargs={"prepared_conditioning": prepared_conditioning},
        device=first.device,
        noise=noise,
        projection_mode=first.projected_ddim_mode,
        late_steps=first.projected_ddim_late_steps,
    )
    raw_targets = first._inverse_pose(sample["raw_pred_xstart"])
    deployed_targets = first._inverse_pose(sample["deployed_pred_xstart"])
    auxiliary = sample.get("auxiliary_outputs", {})
    future_leg = _batch_auxiliary_numpy(auxiliary.get("future_leg"), batch_size)
    contact_logits = _batch_auxiliary_numpy(auxiliary.get("contact_logits"), batch_size)
    return [
        runtime._finish_step(
            prepared_steps[index],
            raw_targets[index],
            deployed_targets[index],
            None if future_leg is None else future_leg[index],
            None if contact_logits is None else contact_logits[index],
        )
        for index, runtime in enumerate(runtimes)
    ]


def _batch_auxiliary_numpy(value, batch_size: int) -> np.ndarray | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        result = value.detach().cpu().numpy()
    else:
        result = np.asarray(value)
    if result.shape[0] != batch_size:
        raise ValueError(f"辅助输出批维不匹配：{result.shape[0]} != {batch_size}")
    return np.asarray(result, dtype=np.float32)


def decode_and_resolve_pose(
    target_raw: np.ndarray,
    current_tracker_raw: np.ndarray,
    hard_rotation_state: np.ndarray,
    current_head_yaw_world: float,
    current_head_position_world: np.ndarray,
    floor_y: float,
    joint_offsets_parent: np.ndarray,
    joint_rest_local_rotations_6d: np.ndarray,
) -> ResolvedPose:
    """执行一次 Head-Anchored Resolver，只验证 hard Tracker rotation。"""

    target = np.asarray(target_raw, dtype=np.float32).reshape(REALTIME_POSE_TARGET_DIM)
    tracker = np.asarray(current_tracker_raw, dtype=np.float32).reshape(TRACKER_COUNT, TRACKER_FEATURE_DIM)
    hard = np.asarray(hard_rotation_state, dtype=bool).reshape(TRACKER_COUNT)
    rotations_head, root_yaw_head = decode_target_head_rotations_np(target)
    root_head, hip_height, joints_head = resolve_root_head_reference_np(
        rotations_head,
        float(root_yaw_head),
        joint_offsets_parent,
        observed_head_height=float(tracker[HEAD_TRACKER_INDEX, 1]),
    )
    yaw_rotation = make_yaw_rotation_np(np.asarray([current_head_yaw_world], dtype=np.float64))[0]
    rotations_world = np.einsum("ij,ajk->aik", yaw_rotation, rotations_head)
    origin = np.asarray(
        [current_head_position_world[0], float(floor_y), current_head_position_world[2]],
        dtype=np.float64,
    )
    root_world = origin + yaw_rotation @ root_head.astype(np.float64)
    root_world[1] = float(floor_y)
    joints_world = origin[None] + np.einsum("ij,aj->ai", yaw_rotation, joints_head)
    rest_rotation = rotation_6d_to_matrix_np(joint_rest_local_rotations_6d)
    local_delta = global_head_rotations_to_local_delta_6d_np(
        rotations_head,
        root_heading_head=root_yaw_head,
        rest_local_rotations=rest_rotation,
    )
    tracker_rot_head = rotation_6d_to_matrix_np(tracker[:, 3:9])
    hard_error = 0.0
    for tracker_index in np.flatnonzero(hard):
        joint_index = TRACKER_TO_JOINT[int(tracker_index)]
        difference = rotations_head[joint_index].T @ tracker_rot_head[tracker_index]
        cosine = np.clip((np.trace(difference) - 1.0) * 0.5, -1.0, 1.0)
        hard_error = max(hard_error, float(np.arccos(cosine)))
    if hard_error > 1e-5:
        raise ValueError(f"Projected pose 的 hard Tracker rotation 不一致：{hard_error:.6g} rad")
    return ResolvedPose(
        target_raw=target,
        joint_rotations_world=rotations_world.astype(np.float32),
        body_local_delta_6d=local_delta.astype(np.float32),
        root_yaw_world=float(current_head_yaw_world + float(root_yaw_head)),
        hip_height=float(hip_height),
        root_position_world=root_world.astype(np.float32),
        joints_world=joints_world.astype(np.float32),
        hard_rotation_max_error=hard_error,
    )


def _advance_durations(
    previous_d_off: np.ndarray,
    previous_d_on: np.ndarray,
    configured: np.ndarray,
    measured_valid: np.ndarray,
    cap: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid = configured & measured_valid
    missing = configured & ~measured_valid
    d_off = np.zeros(TRACKER_COUNT, dtype=np.int64)
    d_on = np.zeros(TRACKER_COUNT, dtype=np.int64)
    d_off[missing] = np.minimum(previous_d_off[missing] + 1, int(cap))
    d_on[valid] = np.minimum(previous_d_on[valid] + 1, int(cap))
    return d_off, d_on

def _left_pad(value: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    result = np.zeros(target_shape, dtype=np.float32)
    if value.shape[0]:
        result[-value.shape[0] :] = value
    return result
