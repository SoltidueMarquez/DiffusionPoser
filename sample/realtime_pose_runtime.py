from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from data_loaders.realtime_pose_config import TrackerReliabilityConfig
from data_loaders.realtime_pose_geometry import (
    assemble_tracker_features_np,
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
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_TARGET_DIM,
    TRACKER_COUNT,
    TRACKER_FEATURE_DIM,
    TRACKER_TO_JOINT,
)
from data_loaders.tracker_reliability import (
    compute_hard_rotation_state_np,
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
        reliability_config: TrackerReliabilityConfig | None = None,
        projected_ddim_mode: str = "all_steps",
        projected_ddim_late_steps: int = 5,
    ):
        self.model = model
        self.diffusion = diffusion
        self.device = device
        self.normalizer = normalizer
        self.reliability_config = (reliability_config or TrackerReliabilityConfig()).validate()
        self.projected_ddim_mode = str(projected_ddim_mode)
        self.projected_ddim_late_steps = int(projected_ddim_late_steps)
        self.joint_offsets_parent = np.asarray(joint_offsets_parent, dtype=np.float32).reshape(24, 3)
        self.joint_rest_local_rotations_6d = np.asarray(
            joint_rest_local_rotations_6d, dtype=np.float32
        ).reshape(24, 6)
        self.pose_history: list[WorldPoseState] = []
        self.tracker_history: list[_TrackerFrameState] = []
        self.trajectory_history: list[np.ndarray] = []
        self.previous_d_off = np.zeros(TRACKER_COUNT, dtype=np.int64)
        self.previous_d_on = np.zeros(TRACKER_COUNT, dtype=np.int64)
        self.previous_head_yaw: float | None = None
        self.previous_head_position: np.ndarray | None = None

    def step(
        self,
        tracker_pos_world: np.ndarray,
        tracker_rot_world_6d: np.ndarray,
        configured: np.ndarray,
        measured_valid: np.ndarray,
        floor_y: float,
    ) -> RuntimeStepResult:
        position = np.asarray(tracker_pos_world, dtype=np.float32).reshape(TRACKER_COUNT, 3)
        rotation_6d = np.asarray(tracker_rot_world_6d, dtype=np.float32).reshape(TRACKER_COUNT, 6)
        configured = np.asarray(configured, dtype=bool).reshape(TRACKER_COUNT)
        measured = np.asarray(measured_valid, dtype=bool).reshape(TRACKER_COUNT)
        if np.any(measured & ~configured):
            raise ValueError("measured_valid 必须是 configured 子集。")
        if not configured[HEAD_TRACKER_INDEX] or not measured[HEAD_TRACKER_INDEX]:
            raise ValueError("Head 必须始终 configured 且 measured_valid。")

        head_rotation = rotation_6d_to_matrix_np(rotation_6d[HEAD_TRACKER_INDEX : HEAD_TRACKER_INDEX + 1])
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
        hard = compute_hard_rotation_state_np(
            configured,
            measured,
            d_on,
            self.reliability_config,
        )
        current_trajectory = self._build_current_trajectory(position[HEAD_TRACKER_INDEX], head_yaw, floor_y)
        conditioning, current_tracker_raw = self._build_conditioning(
            position,
            rotation_6d,
            configured,
            measured,
            d_off,
            d_on,
            head_yaw,
            float(floor_y),
            current_trajectory,
        )
        model_impl = getattr(self.model, "module", self.model)
        # 历史条件在同一帧的全部 DDIM step 间复用；在线推理不保留其计算图。
        with torch.no_grad():
            prepared = model_impl.prepare_conditioning(
                conditioning["pose_history"],
                conditioning["tracker_history"],
                conditioning["current_tracker"],
                conditioning["trajectory_history"],
                conditioning["current_trajectory"],
                conditioning["valid_frame_mask"],
            )
        model_kwargs = {"prepared_conditioning": prepared}
        current_tracker_tensor = torch.as_tensor(
            current_tracker_raw, device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        hard_tensor = torch.as_tensor(hard, device=self.device, dtype=torch.bool).unsqueeze(0)
        mean, std = self._normalizer_pose_stats()
        result = self.diffusion.projected_ddim_sample_loop(
            self.model,
            shape=(1, REALTIME_POSE_TARGET_DIM),
            projection_fn=lambda value: project_realtime_pose_xstart(
                value,
                current_tracker_tensor,
                hard_tensor,
                mean,
                std,
            ),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            device=self.device,
            projection_mode=self.projected_ddim_mode,
            late_steps=self.projected_ddim_late_steps,
        )
        raw_target = self._inverse_pose(result["raw_pred_xstart"])[0]
        deployed_target = self._inverse_pose(result["deployed_pred_xstart"])[0]
        resolved = decode_and_resolve_pose(
            deployed_target,
            current_tracker_raw,
            hard,
            head_yaw,
            position[HEAD_TRACKER_INDEX],
            float(floor_y),
            self.joint_offsets_parent,
            self.joint_rest_local_rotations_6d,
        )
        frame_state = _TrackerFrameState(
            position.copy(),
            rotation_6d.copy(),
            configured.copy(),
            measured.copy(),
            d_off.copy(),
            d_on.copy(),
            head_yaw,
            float(floor_y),
        )
        self.pose_history.append(resolved.as_world_state())
        self.tracker_history.append(frame_state)
        self.trajectory_history.append(current_trajectory.copy())
        self.pose_history = self.pose_history[-REALTIME_POSE_HISTORY_LENGTH:]
        self.tracker_history = self.tracker_history[-REALTIME_POSE_HISTORY_LENGTH:]
        self.trajectory_history = self.trajectory_history[-REALTIME_POSE_HISTORY_LENGTH:]
        self.previous_d_off, self.previous_d_on = d_off, d_on
        self.previous_head_yaw = head_yaw
        self.previous_head_position = position[HEAD_TRACKER_INDEX].copy()
        auxiliary = result.get("auxiliary_outputs", {})
        future_leg = auxiliary.get("future_leg")
        contact_logits = auxiliary.get("contact_logits")
        return RuntimeStepResult(
            resolved,
            raw_target,
            deployed_target,
            kappa_pos,
            kappa_rot,
            hard.copy(),
            current_tracker_raw.copy(),
            None if future_leg is None else np.asarray(future_leg.detach().cpu()[0], dtype=np.float32),
            None if contact_logits is None else np.asarray(contact_logits.detach().cpu()[0], dtype=np.float32),
        )

    def _build_current_trajectory(
        self,
        head_position: np.ndarray,
        head_yaw: float,
        floor_y: float,
    ) -> np.ndarray:
        delta_world = np.zeros(3, dtype=np.float64)
        delta_yaw = 0.0
        if self.previous_head_position is not None and self.previous_head_yaw is not None:
            delta_world = head_position.astype(np.float64) - self.previous_head_position.astype(np.float64)
            delta_yaw = (head_yaw - self.previous_head_yaw + np.pi) % (2.0 * np.pi) - np.pi
            reference_yaw = self.previous_head_yaw
        else:
            reference_yaw = head_yaw
        yaw_inv = make_yaw_rotation_np(np.asarray([reference_yaw], dtype=np.float64))[0].T
        delta_ref = yaw_inv @ delta_world
        return np.asarray(
            [delta_ref[0], delta_ref[2], head_position[1] - float(floor_y), np.sin(delta_yaw), np.cos(delta_yaw)],
            dtype=np.float32,
        )

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
        current_trajectory: np.ndarray,
    ) -> tuple[dict[str, torch.Tensor], np.ndarray]:
        history_count = len(self.pose_history)
        if history_count:
            reference = self.tracker_history[-1]
            pose_raw = build_pose_target_np(
                np.stack([state.joint_rotations_world for state in self.pose_history], axis=0),
                reference.head_yaw_world,
            )
            tracker_measurement = build_tracker_measurements_np(
                np.stack([state.tracker_pos_world for state in self.tracker_history], axis=0),
                np.stack([state.tracker_rot_world_6d for state in self.tracker_history], axis=0),
                reference.tracker_pos_world[HEAD_TRACKER_INDEX],
                reference.floor_y,
                reference.head_yaw_world,
            )
            tracker_raw = assemble_tracker_features_np(
                tracker_measurement,
                np.stack([state.configured for state in self.tracker_history]),
                np.stack([state.measured_valid for state in self.tracker_history]),
                np.stack([state.d_off for state in self.tracker_history]),
                np.stack([state.d_on for state in self.tracker_history]),
                duration_cap=self.reliability_config.duration_cap,
            )
            trajectory_raw = np.stack(self.trajectory_history, axis=0)
        else:
            pose_raw = np.empty((0, REALTIME_POSE_TARGET_DIM), dtype=np.float32)
            tracker_raw = np.empty((0, TRACKER_COUNT, TRACKER_FEATURE_DIM), dtype=np.float32)
            trajectory_raw = np.empty((0, 5), dtype=np.float32)

        current_measurement = build_tracker_measurements_np(
            position[None],
            rotation_6d[None],
            position[HEAD_TRACKER_INDEX],
            floor_y,
            head_yaw,
        )
        current_tracker_raw = assemble_tracker_features_np(
            current_measurement,
            configured[None],
            measured[None],
            d_off[None],
            d_on[None],
            duration_cap=self.reliability_config.duration_cap,
        )[0]
        pose = self._normalize_pose_numpy(pose_raw)
        tracker = self._normalize_tracker_numpy(tracker_raw)
        current_tracker = self._normalize_tracker_numpy(current_tracker_raw)
        trajectory = self._normalize_trajectory_numpy(trajectory_raw)
        current_traj = self._normalize_trajectory_numpy(current_trajectory[None])
        valid = np.zeros(REALTIME_POSE_HISTORY_LENGTH, dtype=bool)
        valid[-history_count:] = True if history_count else False
        return {
            "pose_history": torch.as_tensor(
                _left_pad(pose, (REALTIME_POSE_HISTORY_LENGTH, REALTIME_POSE_TARGET_DIM)),
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(0),
            "tracker_history": torch.as_tensor(
                _left_pad(tracker, (REALTIME_POSE_HISTORY_LENGTH, TRACKER_COUNT, TRACKER_FEATURE_DIM)),
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(0),
            "current_tracker": torch.as_tensor(current_tracker, device=self.device, dtype=torch.float32).unsqueeze(0),
            "trajectory_history": torch.as_tensor(
                _left_pad(trajectory, (REALTIME_POSE_HISTORY_LENGTH, 5)),
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(0),
            "current_trajectory": torch.as_tensor(current_traj, device=self.device, dtype=torch.float32).unsqueeze(0),
            "valid_frame_mask": torch.as_tensor(valid, device=self.device, dtype=torch.bool).unsqueeze(0),
        }, current_tracker_raw

    def _normalizer_pose_stats(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.normalizer is None:
            return None, None
        return (
            self.normalizer.pose_mean.to(self.device),
            self.normalizer.pose_std.to(self.device),
        )

    def _normalize_pose_numpy(self, value: np.ndarray) -> np.ndarray:
        return value if self.normalizer is None else np.asarray(self.normalizer.normalize_pose(value), dtype=np.float32)

    def _normalize_tracker_numpy(self, value: np.ndarray) -> np.ndarray:
        return value if self.normalizer is None else np.asarray(self.normalizer.normalize_tracker(value), dtype=np.float32)

    def _normalize_trajectory_numpy(self, value: np.ndarray) -> np.ndarray:
        result = np.asarray(value, dtype=np.float32).copy()
        if self.normalizer is not None and result.size:
            result[..., 2] = self.normalizer.normalize_head_height(result[..., 2])
        return result

    def _inverse_pose(self, value: torch.Tensor) -> np.ndarray:
        cpu = value.detach().cpu()
        if self.normalizer is not None:
            cpu = self.normalizer.inverse_pose(cpu)
        return np.asarray(cpu, dtype=np.float32)


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
