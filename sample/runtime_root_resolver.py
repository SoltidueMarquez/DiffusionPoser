from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable

import numpy as np

from data_loaders.realtime_pose_kinematics import integrate_root_delta_xz_ref, make_yaw_rotation_np
from data_loaders.sensor_masking import HEAD_TRACKER_INDEX, HIP_TRACKER_INDEX, TRACKER_COUNT, validate_sensor_valid
from data_loaders.tracker_codec import yaw_from_rotation_6d_np


RESOLVER_CONTRACT_VERSION = "runtime_root_resolver_v1"


class RootSource(IntEnum):
    HIP = 0
    HEAD_FK = 1
    RECONNECT = 2
    RESET = 3


@dataclass(frozen=True)
class RuntimeRootResolverConfig:
    reconnect_duration_seconds: float = 0.1
    hip_filter_time_constant_seconds: float = 0.03
    max_head_height_correction_m: float = 0.10
    timestamp_reset_threshold_seconds: float = 0.25


@dataclass
class RuntimeRootResolverState:
    initialized: bool = False
    final_root_pos_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    final_root_yaw: float = 0.0
    final_pelvis_height: float = 0.0
    final_joints_world: np.ndarray = field(default_factory=lambda: np.zeros((24, 3), dtype=np.float32))
    hip_was_valid: bool = False
    reconnect_active: bool = False
    reconnect_elapsed_seconds: float = 0.0
    reconnect_start_root_pos_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    reconnect_start_root_yaw: float = 0.0
    reconnect_start_pelvis_height: float = 0.0
    last_timestamp: float | None = None
    floor_y: float = 0.0
    tracking_origin_revision: int = 0

    def copy(self) -> "RuntimeRootResolverState":
        return RuntimeRootResolverState(
            initialized=bool(self.initialized),
            final_root_pos_world=np.asarray(self.final_root_pos_world, dtype=np.float32).copy(),
            final_root_yaw=float(self.final_root_yaw),
            final_pelvis_height=float(self.final_pelvis_height),
            final_joints_world=np.asarray(self.final_joints_world, dtype=np.float32).copy(),
            hip_was_valid=bool(self.hip_was_valid),
            reconnect_active=bool(self.reconnect_active),
            reconnect_elapsed_seconds=float(self.reconnect_elapsed_seconds),
            reconnect_start_root_pos_world=np.asarray(self.reconnect_start_root_pos_world, dtype=np.float32).copy(),
            reconnect_start_root_yaw=float(self.reconnect_start_root_yaw),
            reconnect_start_pelvis_height=float(self.reconnect_start_pelvis_height),
            last_timestamp=None if self.last_timestamp is None else float(self.last_timestamp),
            floor_y=float(self.floor_y),
            tracking_origin_revision=int(self.tracking_origin_revision),
        )


@dataclass(frozen=True)
class RuntimeRootResolverResult:
    final_root_pos_world: np.ndarray
    final_root_yaw: float
    final_pelvis_height: float
    final_joints_world: np.ndarray
    final_root_delta_xz_ref: np.ndarray
    final_yaw_delta_sincos: np.ndarray
    root_source: RootSource
    reconnect_alpha: float
    state: RuntimeRootResolverState


FkCallback = Callable[[np.ndarray, float, float], np.ndarray]


def wrap_angle(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def shortest_angle_lerp(start: float, target: float, alpha: float) -> float:
    return wrap_angle(float(start) + wrap_angle(float(target) - float(start)) * float(alpha))


def smoothstep(alpha: float) -> float:
    value = float(np.clip(alpha, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def encode_single_root_delta_xz_ref(
    previous_root_pos_world: np.ndarray,
    previous_root_yaw: float,
    current_root_pos_world: np.ndarray,
) -> np.ndarray:
    delta_world = np.asarray(current_root_pos_world, dtype=np.float64) - np.asarray(
        previous_root_pos_world, dtype=np.float64
    )
    rotation = make_yaw_rotation_np(np.asarray([float(previous_root_yaw)], dtype=np.float64))[0]
    delta_ref = rotation.T @ delta_world
    return np.asarray([delta_ref[0], delta_ref[2]], dtype=np.float32)


class RuntimeRootResolver:
    """Python 侧 Root 状态机；C# 侧按同一契约独立实现。"""

    def __init__(
        self,
        pelvis_offset_parent: np.ndarray,
        config: RuntimeRootResolverConfig | None = None,
        state: RuntimeRootResolverState | None = None,
    ) -> None:
        self.pelvis_offset_parent = np.asarray(pelvis_offset_parent, dtype=np.float32).reshape(3)
        self.config = config or RuntimeRootResolverConfig()
        self.state = state.copy() if state is not None else RuntimeRootResolverState()

    def reset(self, tracking_origin_revision: int | None = None) -> None:
        revision = self.state.tracking_origin_revision if tracking_origin_revision is None else int(tracking_origin_revision)
        self.state = RuntimeRootResolverState(tracking_origin_revision=revision)

    def resolve(
        self,
        *,
        tracker_pos_world: np.ndarray,
        tracker_rot_world_6d: np.ndarray,
        sensor_valid: np.ndarray,
        timestamp: float,
        floor_y: float,
        tracking_origin_revision: int,
        model_root_delta_xz_ref: np.ndarray,
        model_yaw_delta_sincos: np.ndarray,
        model_pelvis_height: float,
        fk_callback: FkCallback,
        preliminary_joints_world: np.ndarray | None = None,
        previous_to_current_world: np.ndarray | None = None,
    ) -> RuntimeRootResolverResult:
        tracker_pos = np.asarray(tracker_pos_world, dtype=np.float32).reshape(TRACKER_COUNT, 3)
        tracker_rot = np.asarray(tracker_rot_world_6d, dtype=np.float32).reshape(TRACKER_COUNT, 6)
        valid = np.asarray(sensor_valid, dtype=bool).reshape(TRACKER_COUNT)
        validate_sensor_valid(valid[None])

        timestamp = float(timestamp)
        revision = int(tracking_origin_revision)
        reset_boundary = self._prepare_time_and_origin(
            timestamp=timestamp,
            tracking_origin_revision=revision,
            previous_to_current_world=previous_to_current_world,
        )
        previous = self.state.copy()
        delta_seconds = self._accepted_delta_seconds(timestamp)

        model_delta = np.asarray(model_root_delta_xz_ref, dtype=np.float32).reshape(1, 2)
        model_yaw_pair = np.asarray(model_yaw_delta_sincos, dtype=np.float32).reshape(2)
        model_yaw_delta = float(np.arctan2(model_yaw_pair[0], model_yaw_pair[1]))
        previous_root = previous.final_root_pos_world if previous.initialized else np.asarray(
            [tracker_pos[HEAD_TRACKER_INDEX, 0], float(floor_y), tracker_pos[HEAD_TRACKER_INDEX, 2]],
            dtype=np.float32,
        )
        previous_yaw = previous.final_root_yaw if previous.initialized else 0.0
        model_root = integrate_root_delta_xz_ref(
            previous_root[None],
            np.asarray([previous_yaw], dtype=np.float32),
            model_delta,
        )[0]
        model_root[1] = float(floor_y)
        model_yaw = wrap_angle(previous_yaw + model_yaw_delta)
        model_height = float(model_pelvis_height)
        if preliminary_joints_world is None:
            preliminary_joints = self._run_fk(fk_callback, model_root, model_yaw, model_height)
        else:
            preliminary_joints = np.asarray(preliminary_joints_world, dtype=np.float32).reshape(24, 3)

        hip_valid = bool(valid[HIP_TRACKER_INDEX])
        reconnect_alpha = 0.0
        if hip_valid:
            hip_root, hip_yaw, hip_height = self._hip_target(
                tracker_pos_world=tracker_pos,
                tracker_rot_world_6d=tracker_rot,
                floor_y=float(floor_y),
            )
            if previous.initialized and (not previous.hip_was_valid or previous.reconnect_active):
                if not previous.reconnect_active:
                    self.state.reconnect_start_root_pos_world = previous.final_root_pos_world.copy()
                    self.state.reconnect_start_root_yaw = previous.final_root_yaw
                    self.state.reconnect_start_pelvis_height = previous.final_pelvis_height
                elapsed = previous.reconnect_elapsed_seconds + delta_seconds
                duration = max(float(self.config.reconnect_duration_seconds), 1e-6)
                reconnect_alpha = float(np.clip(elapsed / duration, 0.0, 1.0))
                blend = smoothstep(reconnect_alpha)
                final_root = (1.0 - blend) * self.state.reconnect_start_root_pos_world + blend * hip_root
                final_yaw = shortest_angle_lerp(self.state.reconnect_start_root_yaw, hip_yaw, blend)
                final_height = (1.0 - blend) * self.state.reconnect_start_pelvis_height + blend * hip_height
                source = RootSource.RECONNECT if reconnect_alpha < 1.0 else RootSource.HIP
                self.state.reconnect_active = reconnect_alpha < 1.0
                self.state.reconnect_elapsed_seconds = 0.0 if reconnect_alpha >= 1.0 else elapsed
            else:
                final_root, final_yaw, final_height = self._filter_continuous_hip(
                    previous=previous,
                    target_root=hip_root,
                    target_yaw=hip_yaw,
                    target_height=hip_height,
                    delta_seconds=delta_seconds,
                )
                source = RootSource.HIP
                self.state.reconnect_active = False
                self.state.reconnect_elapsed_seconds = 0.0
        else:
            predicted_head = preliminary_joints[15]
            root_to_head = predicted_head - model_root
            final_root = model_root.copy()
            final_root[[0, 2]] = tracker_pos[HEAD_TRACKER_INDEX, [0, 2]] - root_to_head[[0, 2]]
            final_root[1] = float(floor_y)
            head_residual_y = float(tracker_pos[HEAD_TRACKER_INDEX, 1] - predicted_head[1])
            correction = float(
                np.clip(
                    head_residual_y,
                    -float(self.config.max_head_height_correction_m),
                    float(self.config.max_head_height_correction_m),
                )
            )
            final_yaw = model_yaw
            final_height = model_height + correction
            source = RootSource.HEAD_FK
            self.state.reconnect_active = False
            self.state.reconnect_elapsed_seconds = 0.0

        final_root = np.asarray(final_root, dtype=np.float32)
        final_root[1] = float(floor_y)
        final_joints = self._run_fk(fk_callback, final_root, float(final_yaw), float(final_height))
        if previous.initialized and not reset_boundary:
            root_delta = encode_single_root_delta_xz_ref(
                previous.final_root_pos_world,
                previous.final_root_yaw,
                final_root,
            )
            yaw_delta = wrap_angle(float(final_yaw) - previous.final_root_yaw)
            yaw_delta_pair = np.asarray([np.sin(yaw_delta), np.cos(yaw_delta)], dtype=np.float32)
        else:
            root_delta = np.zeros(2, dtype=np.float32)
            yaw_delta_pair = np.asarray([0.0, 1.0], dtype=np.float32)
            source = RootSource.RESET

        self.state.initialized = True
        self.state.final_root_pos_world = final_root.copy()
        self.state.final_root_yaw = float(final_yaw)
        self.state.final_pelvis_height = float(final_height)
        self.state.final_joints_world = final_joints.copy()
        self.state.hip_was_valid = hip_valid
        self.state.last_timestamp = timestamp
        self.state.floor_y = float(floor_y)
        self.state.tracking_origin_revision = revision
        result_state = self.state.copy()
        return RuntimeRootResolverResult(
            final_root_pos_world=final_root,
            final_root_yaw=float(final_yaw),
            final_pelvis_height=float(final_height),
            final_joints_world=final_joints,
            final_root_delta_xz_ref=root_delta,
            final_yaw_delta_sincos=yaw_delta_pair,
            root_source=source,
            reconnect_alpha=float(reconnect_alpha),
            state=result_state,
        )

    def _prepare_time_and_origin(
        self,
        *,
        timestamp: float,
        tracking_origin_revision: int,
        previous_to_current_world: np.ndarray | None,
    ) -> bool:
        reset_boundary = False
        if self.state.initialized and tracking_origin_revision != self.state.tracking_origin_revision:
            if previous_to_current_world is None:
                self.reset(tracking_origin_revision)
                reset_boundary = True
            else:
                self._transform_state(np.asarray(previous_to_current_world, dtype=np.float64).reshape(4, 4))
                self.state.tracking_origin_revision = tracking_origin_revision
        if self.state.last_timestamp is not None:
            delta = timestamp - float(self.state.last_timestamp)
            if delta < 0.0 or delta > float(self.config.timestamp_reset_threshold_seconds):
                self.reset(tracking_origin_revision)
                reset_boundary = True
        return reset_boundary

    def _accepted_delta_seconds(self, timestamp: float) -> float:
        if self.state.last_timestamp is None:
            return 0.0
        return max(0.0, float(timestamp) - float(self.state.last_timestamp))

    def _transform_state(self, transform: np.ndarray) -> None:
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        for field_name in ("final_root_pos_world", "reconnect_start_root_pos_world"):
            value = np.asarray(getattr(self.state, field_name), dtype=np.float64)
            setattr(self.state, field_name, (rotation @ value + translation).astype(np.float32))
        if self.state.final_joints_world.size:
            self.state.final_joints_world = (
                np.einsum("ij,sj->si", rotation, self.state.final_joints_world) + translation[None]
            ).astype(np.float32)
        yaw_offset = float(np.arctan2(rotation[0, 2], rotation[2, 2]))
        self.state.final_root_yaw = wrap_angle(self.state.final_root_yaw + yaw_offset)
        self.state.reconnect_start_root_yaw = wrap_angle(self.state.reconnect_start_root_yaw + yaw_offset)
        floor_point = rotation @ np.asarray([0.0, self.state.floor_y, 0.0], dtype=np.float64) + translation
        self.state.floor_y = float(floor_point[1])

    def _hip_target(
        self,
        *,
        tracker_pos_world: np.ndarray,
        tracker_rot_world_6d: np.ndarray,
        floor_y: float,
    ) -> tuple[np.ndarray, float, float]:
        hip_yaw = float(yaw_from_rotation_6d_np(tracker_rot_world_6d[HIP_TRACKER_INDEX]))
        rotation = make_yaw_rotation_np(np.asarray([hip_yaw], dtype=np.float64))[0]
        root = tracker_pos_world[HIP_TRACKER_INDEX].astype(np.float64) - rotation @ self.pelvis_offset_parent.astype(
            np.float64
        )
        root[1] = floor_y
        height = float(tracker_pos_world[HIP_TRACKER_INDEX, 1] - floor_y)
        return root.astype(np.float32), hip_yaw, height

    def _filter_continuous_hip(
        self,
        *,
        previous: RuntimeRootResolverState,
        target_root: np.ndarray,
        target_yaw: float,
        target_height: float,
        delta_seconds: float,
    ) -> tuple[np.ndarray, float, float]:
        if not previous.initialized or float(self.config.hip_filter_time_constant_seconds) <= 0.0:
            return target_root, target_yaw, target_height
        tau = float(self.config.hip_filter_time_constant_seconds)
        alpha = 1.0 - float(np.exp(-max(delta_seconds, 0.0) / tau)) if delta_seconds > 0.0 else 1.0
        root = (1.0 - alpha) * previous.final_root_pos_world + alpha * target_root
        yaw = shortest_angle_lerp(previous.final_root_yaw, target_yaw, alpha)
        height = (1.0 - alpha) * previous.final_pelvis_height + alpha * target_height
        return root.astype(np.float32), yaw, float(height)

    @staticmethod
    def _run_fk(callback: FkCallback, root: np.ndarray, yaw: float, height: float) -> np.ndarray:
        joints = np.asarray(callback(np.asarray(root, dtype=np.float32), float(yaw), float(height)), dtype=np.float32)
        if joints.shape != (24, 3):
            raise ValueError(f"fk_callback 应返回 [24,3]，实际为 {joints.shape}")
        return joints
