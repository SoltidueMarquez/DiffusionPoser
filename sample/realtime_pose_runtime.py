from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from data_loaders.realtime_pose_config import IKInpaintingConfig
from data_loaders.realtime_pose_geometry import (
    assemble_current_tracker_features_np,
    build_tracker_measurements_np,
    decode_target_head_rotations_np,
    extract_rotation_heading_components_np,
    global_head_rotations_to_local_delta_6d_np,
    ROOT_HEADING_OBSERVABILITY_EPS,
    resolve_root_head_reference_np,
)
from data_loaders.realtime_pose_kinematics import (
    make_yaw_rotation_np,
    rotation_6d_to_matrix_np,
)
from data_loaders.realtime_pose_predictor_features import build_predictor_step_features_np
from data_loaders.sensor_masking import (
    CORE_TRACKER_INDICES,
    HEAD_TRACKER_INDEX,
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_TARGET_DIM,
    PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH,
    TRACKER_COUNT,
    TRACKER_FEATURE_DIM,
    validate_tracker_available,
)
from diffusion.realtime_pose_inpainting import build_current_realtime_pose_conditions
from diffusion.realtime_pose_projection import project_realtime_pose_xstart


@dataclass(frozen=True)
class WorldPoseState:
    joint_rotations_world: np.ndarray  # [24,3,3]
    root_yaw_world: float
    hip_height: float
    root_position_world: np.ndarray  # [3]


@dataclass(frozen=True)
class ResolvedPose:
    target_raw: np.ndarray
    joint_rotations_world: np.ndarray
    body_local_delta_6d: np.ndarray
    root_yaw_world: float
    hip_height: float
    root_position_world: np.ndarray
    joints_world: np.ndarray

    def as_world_state(self) -> WorldPoseState:
        return WorldPoseState(
            joint_rotations_world=self.joint_rotations_world.copy(),
            root_yaw_world=float(self.root_yaw_world),
            hip_height=float(self.hip_height),
            root_position_world=self.root_position_world.copy(),
        )


@dataclass(frozen=True)
class RuntimeStepResult:
    predictor_pose_horizon: np.ndarray  # [11,144]，原始物理值
    raw_pred_pose: np.ndarray  # [144]
    deployed_pred_pose: np.ndarray  # [144]
    ik_gap: np.ndarray  # [24]，弧度
    ik_confidence: np.ndarray  # [24]
    denoise_strength: np.ndarray  # [24]
    resolved_pose: ResolvedPose
    current_tracker_raw: np.ndarray  # [6,10]
    current_head_yaw_world: float


@dataclass(frozen=True)
class _TrackerWorldState:
    positions: np.ndarray
    rotations_6d: np.ndarray
    floor_y: float


@dataclass(frozen=True)
class _PreparedStep:
    motion_context: np.ndarray
    core_tracker_context: np.ndarray
    current_tracker_raw: np.ndarray
    tracker_positions: np.ndarray
    tracker_rotations_6d: np.ndarray
    head_yaw_world: float
    floor_y: float
    consumes_preloaded_current: bool


class RealtimePoseRuntime:
    """30Hz Predictor + 单帧 DiT runtime；调用前必须显式提供完整历史。

    `step()` 一次对应一个 30Hz 模型帧。更高刷新率的显示插值属于 Python
    模型输出之后的客户端职责，不在此处降采样或补帧。
    """

    def __init__(
        self,
        predictor_model,
        dit_model,
        diffusion,
        device: torch.device,
        joint_offsets_parent: np.ndarray,
        joint_rest_local_rotations_6d: np.ndarray,
        normalizer,
        fabrik_iterations: int = 2,
        ik_direction_only_quality: float | None = None,
        ik_residual_scale: float | None = None,
        ik_position_solved_quality: float | None = None,
        ik_gap_low: float | None = None,
        ik_gap_high: float | None = None,
        ik_direction_support: float = 0.35,
        ik_untracked_strength: float = 0.05,
    ):
        self.predictor_model = predictor_model.eval().requires_grad_(False)
        self.dit_model = dit_model.eval().requires_grad_(False)
        self.diffusion = diffusion
        self.device = torch.device(device)
        self.normalizer = normalizer
        self.ik_inpainting_config = IKInpaintingConfig(
            fabrik_iterations=int(fabrik_iterations),
            direction_only_quality=ik_direction_only_quality,
            residual_scale=ik_residual_scale,
            position_solved_quality=ik_position_solved_quality,
            gap_low=ik_gap_low,
            gap_high=ik_gap_high,
            direction_support=ik_direction_support,
            untracked_strength=ik_untracked_strength,
        ).validate()
        self.joint_offsets_parent = np.asarray(
            joint_offsets_parent, dtype=np.float32
        ).reshape(24, 3)
        self.joint_rest_local_rotations_6d = np.asarray(
            joint_rest_local_rotations_6d, dtype=np.float32
        ).reshape(24, 6)
        self.pose_history: list[WorldPoseState] = []
        self.tracker_history: list[_TrackerWorldState] = []
        self._preloaded_current_pending = False

    def initialize_history(
        self,
        pose_history: list[WorldPoseState],
        tracker_positions_world: np.ndarray,
        tracker_rotations_world_6d: np.ndarray,
        floor_y: np.ndarray | float,
    ) -> None:
        """设置历史。

        Pose 必须恰好 10 帧。Tracker 可传 offset `-11..-1` 的 11 帧，随后由
        `step()` 追加当前帧；也可传 `-11..0` 的 12 帧，此时第一次 step 的
        当前测量必须与最后一帧一致。
        """

        if len(pose_history) != REALTIME_POSE_HISTORY_LENGTH:
            raise ValueError("runtime 初始化必须提供恰好 10 帧 Pose history。")
        positions = np.asarray(tracker_positions_world, dtype=np.float32)
        rotations = np.asarray(tracker_rotations_world_6d, dtype=np.float32)
        if positions.ndim != 3 or positions.shape[1:] != (TRACKER_COUNT, 3):
            raise ValueError("tracker_positions_world 必须为 [11或12,6,3]。")
        if rotations.shape != (positions.shape[0], TRACKER_COUNT, 6):
            raise ValueError("tracker_rotations_world_6d 必须为 [11或12,6,6]。")
        if positions.shape[0] not in (
            PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH,
            PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH + 1,
        ):
            raise ValueError("Tracker history 必须为 11 或 12 帧。")
        floors = np.asarray(floor_y, dtype=np.float32)
        if floors.ndim == 0:
            floors = np.full(positions.shape[0], float(floors), dtype=np.float32)
        if floors.shape != (positions.shape[0],):
            raise ValueError("floor_y 必须为标量或与 Tracker history 等长。")
        if not all(
            np.asarray(state.joint_rotations_world).shape == (24, 3, 3)
            for state in pose_history
        ):
            raise ValueError("每个 WorldPoseState 必须包含 [24,3,3] 旋转。")
        core = list(CORE_TRACKER_INDICES)
        if not (
            np.isfinite(positions[:, core]).all()
            and np.isfinite(rotations[:, core]).all()
        ):
            raise ValueError("核心三点 Tracker history 必须全部有效。")
        self.pose_history = [
            WorldPoseState(
                np.asarray(state.joint_rotations_world, dtype=np.float32).copy(),
                float(state.root_yaw_world),
                float(state.hip_height),
                np.asarray(state.root_position_world, dtype=np.float32).reshape(3).copy(),
            )
            for state in pose_history
        ]
        self.tracker_history = [
            _TrackerWorldState(positions[index].copy(), rotations[index].copy(), float(floors[index]))
            for index in range(positions.shape[0])
        ]
        self._preloaded_current_pending = positions.shape[0] == (
            PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH + 1
        )

    def step(
        self,
        tracker_pos_world: np.ndarray,
        tracker_rot_world_6d: np.ndarray,
        tracker_available: np.ndarray,
        floor_y: float,
        noise: torch.Tensor | None = None,
    ) -> RuntimeStepResult:
        return step_realtime_pose_batch(
            [self],
            np.asarray(tracker_pos_world, dtype=np.float32)[None],
            np.asarray(tracker_rot_world_6d, dtype=np.float32)[None],
            np.asarray(tracker_available, dtype=bool)[None],
            np.asarray([floor_y], dtype=np.float32),
            noise=noise,
        )[0]

    def _prepare_step(
        self,
        tracker_positions: np.ndarray,
        tracker_rotations_6d: np.ndarray,
        tracker_available: np.ndarray,
        floor_y: float,
    ) -> _PreparedStep:
        if len(self.pose_history) != REALTIME_POSE_HISTORY_LENGTH:
            raise RuntimeError("runtime 尚未初始化完整 10 帧 Pose history。")
        if len(self.tracker_history) not in (11, 12):
            raise RuntimeError("runtime 尚未初始化 11/12 帧 Tracker history。")
        positions = np.asarray(tracker_positions, dtype=np.float32).reshape(TRACKER_COUNT, 3)
        rotations = np.asarray(tracker_rotations_6d, dtype=np.float32).reshape(TRACKER_COUNT, 6)
        available = validate_tracker_available(tracker_available).reshape(TRACKER_COUNT)
        consumes_preloaded_current = bool(self._preloaded_current_pending)
        if consumes_preloaded_current:
            current_matches_last = np.allclose(
                self.tracker_history[-1].positions[list(CORE_TRACKER_INDICES)],
                positions[list(CORE_TRACKER_INDICES)],
                atol=1e-6,
            ) and np.allclose(
                self.tracker_history[-1].rotations_6d[list(CORE_TRACKER_INDICES)],
                rotations[list(CORE_TRACKER_INDICES)],
                atol=1e-6,
            )
            if not current_matches_last:
                raise ValueError(
                    "初始化已包含 offset 0 时，第一次 step 的核心 Tracker 必须与该帧一致。"
                )
        current_state = _TrackerWorldState(
            positions.copy(), rotations.copy(), float(floor_y)
        )
        tracker_window = (
            self.tracker_history[-12:]
            if consumes_preloaded_current
            else [*self.tracker_history, current_state][-12:]
        )
        if len(tracker_window) != 12:
            raise RuntimeError("Predictor 当前 step 必须具有 offset -11..0 的 12 帧 Tracker。")
        predictor_features = build_predictor_step_features_np(
            motion_rotations_world=np.stack(
                [state.joint_rotations_world for state in self.pose_history], axis=0
            ),
            tracker_positions_world_with_previous=np.stack(
                [state.positions for state in tracker_window], axis=0
            ),
            tracker_rotations_world_6d_with_previous=np.stack(
                [state.rotations_6d for state in tracker_window], axis=0
            ),
            floor_y=float(floor_y),
        )
        continuous = build_tracker_measurements_np(
            positions[None],
            rotations[None],
            positions[HEAD_TRACKER_INDEX],
            float(floor_y),
            predictor_features.current_head_yaw_world,
        )[0]
        current_tracker_raw = assemble_current_tracker_features_np(
            continuous, available
        )
        return _PreparedStep(
            motion_context=predictor_features.motion_context,
            core_tracker_context=predictor_features.core_tracker_context,
            current_tracker_raw=current_tracker_raw,
            tracker_positions=positions,
            tracker_rotations_6d=rotations,
            head_yaw_world=predictor_features.current_head_yaw_world,
            floor_y=float(floor_y),
            consumes_preloaded_current=consumes_preloaded_current,
        )

    def _finish_step(
        self,
        prepared: _PreparedStep,
        predictor_pose_horizon: np.ndarray,
        raw_pred_pose: np.ndarray,
        deployed_pred_pose: np.ndarray,
        ik_gap: np.ndarray,
        ik_confidence: np.ndarray,
        denoise_strength: np.ndarray,
    ) -> RuntimeStepResult:
        resolved = decode_and_resolve_pose(
            deployed_pred_pose,
            prepared.current_tracker_raw,
            prepared.head_yaw_world,
            prepared.tracker_positions[HEAD_TRACKER_INDEX],
            prepared.floor_y,
            self.joint_offsets_parent,
            self.joint_rest_local_rotations_6d,
            previous_root_yaw_world=self.pose_history[-1].root_yaw_world,
        )
        self.pose_history.append(resolved.as_world_state())
        self.pose_history = self.pose_history[-REALTIME_POSE_HISTORY_LENGTH:]
        if prepared.consumes_preloaded_current:
            self._preloaded_current_pending = False
        else:
            self.tracker_history.append(
                _TrackerWorldState(
                    prepared.tracker_positions.copy(),
                    prepared.tracker_rotations_6d.copy(),
                    prepared.floor_y,
                )
            )
        self.tracker_history = self.tracker_history[-12:]
        return RuntimeStepResult(
            predictor_pose_horizon=np.asarray(predictor_pose_horizon, dtype=np.float32).reshape(11, 144),
            raw_pred_pose=np.asarray(raw_pred_pose, dtype=np.float32).reshape(144),
            deployed_pred_pose=np.asarray(deployed_pred_pose, dtype=np.float32).reshape(144),
            ik_gap=np.asarray(ik_gap, dtype=np.float32).reshape(24),
            ik_confidence=np.asarray(ik_confidence, dtype=np.float32).reshape(24),
            denoise_strength=np.asarray(denoise_strength, dtype=np.float32).reshape(24),
            resolved_pose=resolved,
            current_tracker_raw=prepared.current_tracker_raw.copy(),
            current_head_yaw_world=float(prepared.head_yaw_world),
        )


def step_realtime_pose_batch(
    runtimes: list[RealtimePoseRuntime],
    tracker_pos_world: np.ndarray,
    tracker_rot_world_6d: np.ndarray,
    tracker_available: np.ndarray,
    floor_y: np.ndarray,
    noise: torch.Tensor | None = None,
) -> list[RuntimeStepResult]:
    if not runtimes:
        raise ValueError("批处理至少需要一个 runtime。")
    batch_size = len(runtimes)
    first = runtimes[0]
    for runtime in runtimes[1:]:
        if runtime.predictor_model is not first.predictor_model or runtime.dit_model is not first.dit_model:
            raise ValueError("批内 runtime 必须共享 Predictor 与 DiT。")
        if runtime.diffusion is not first.diffusion or runtime.device != first.device:
            raise ValueError("批内 runtime 必须共享 diffusion 与 device。")
    positions = np.asarray(tracker_pos_world, dtype=np.float32).reshape(batch_size, 6, 3)
    rotations = np.asarray(tracker_rot_world_6d, dtype=np.float32).reshape(batch_size, 6, 6)
    available = np.asarray(tracker_available, dtype=bool).reshape(batch_size, 6)
    floors = np.asarray(floor_y, dtype=np.float32).reshape(batch_size)
    prepared = [
        runtime._prepare_step(positions[index], rotations[index], available[index], floors[index])
        for index, runtime in enumerate(runtimes)
    ]
    motion_raw = np.stack([value.motion_context for value in prepared])
    sparse_raw = np.stack([value.core_tracker_context for value in prepared])
    motion = _normalize_pose_numpy(first.normalizer, motion_raw)
    sparse = _normalize_sparse_numpy(first.normalizer, sparse_raw)
    motion_tensor = torch.as_tensor(motion, device=first.device, dtype=torch.float32)
    sparse_tensor = torch.as_tensor(sparse, device=first.device, dtype=torch.float32)
    tracker_tensor = torch.as_tensor(
        np.stack([value.current_tracker_raw for value in prepared]),
        device=first.device,
        dtype=torch.float32,
    )
    offsets_tensor = torch.as_tensor(
        np.stack([runtime.joint_offsets_parent for runtime in runtimes]),
        device=first.device,
        dtype=torch.float32,
    )
    mean, scale = _normalizer_pose_stats(first)
    tracker_mean, tracker_scale = _normalizer_tracker_stats(first)
    with torch.no_grad():
        predictor_normalized = first.predictor_model(motion_tensor, sparse_tensor)
        predictor_current_raw = _inverse_pose_tensor(first.normalizer, predictor_normalized[:, 0])
        _, ik_condition, tracker_geometry = build_current_realtime_pose_conditions(
            initial_pose_raw=predictor_current_raw,
            current_tracker_raw=tracker_tensor,
            joint_offsets_parent=offsets_tensor,
            pose_mean=mean,
            pose_scale=scale,
            tracker_mean=tracker_mean,
            tracker_scale=tracker_scale,
            config=first.ik_inpainting_config,
        )
        tracker_available_tensor = tracker_tensor[..., 9] > 0.5
        dit_impl = getattr(first.dit_model, "module", first.dit_model)
        prepared_conditioning = dit_impl.prepare_conditioning(
            motion_context=motion_tensor,
            predictor_pose_horizon=predictor_normalized,
            tracker_geometry=tracker_geometry,
            tracker_available=tracker_available_tensor,
            ik_residual=ik_condition.ik_residual,
            ik_gap=ik_condition.ik_gap,
            ik_confidence=ik_condition.ik_confidence,
            denoise_strength=ik_condition.denoise_strength,
            constraint_type=ik_condition.constraint_type,
        )
    sampling_shape = (batch_size, REALTIME_POSE_TARGET_DIM)
    if noise is not None:
        if tuple(noise.shape) != sampling_shape:
            raise ValueError("noise 必须为 [B,144]。")
        noise = noise.to(first.device, dtype=torch.float32)
    sample = first.diffusion.projected_ddim_sample_loop(
        first.dit_model,
        shape=sampling_shape,
        predictor_current=predictor_normalized[:, 0],
        projection_fn=lambda value: project_realtime_pose_xstart(
            value,
            tracker_tensor,
            pose_mean=mean,
            pose_scale=scale,
        ),
        noise=noise,
        clip_denoised=False,
        model_kwargs={"prepared_conditioning": prepared_conditioning},
        device=first.device,
        eta=0.0,
    )
    predictor_raw = _inverse_pose_tensor(first.normalizer, predictor_normalized).cpu().numpy()
    raw = _inverse_pose_tensor(first.normalizer, sample["raw_pred_pose"]).cpu().numpy()
    deployed = _inverse_pose_tensor(
        first.normalizer, sample["deployed_pred_pose"]
    ).cpu().numpy()
    gaps = ik_condition.ik_gap.detach().cpu().numpy()
    confidence = ik_condition.ik_confidence.detach().cpu().numpy()
    strengths = ik_condition.denoise_strength.detach().cpu().numpy()
    return [
        runtime._finish_step(
            prepared[index],
            predictor_raw[index],
            raw[index],
            deployed[index],
            gaps[index],
            confidence[index],
            strengths[index],
        )
        for index, runtime in enumerate(runtimes)
    ]


def _normalizer_pose_stats(runtime: RealtimePoseRuntime):
    if runtime.normalizer is None or runtime.normalizer.disable:
        return None, None
    return (
        runtime.normalizer.pose_mean.to(runtime.device),
        runtime.normalizer.pose_scale.to(runtime.device),
    )


def _normalizer_tracker_stats(runtime: RealtimePoseRuntime):
    if runtime.normalizer is None or runtime.normalizer.disable:
        return None, None
    return (
        runtime.normalizer.tracker_mean.to(runtime.device),
        (runtime.normalizer.tracker_std + runtime.normalizer.eps).to(runtime.device),
    )


def _normalize_pose_numpy(normalizer, value: np.ndarray) -> np.ndarray:
    return value if normalizer is None else np.asarray(normalizer.normalize_pose(value), dtype=np.float32)


def _normalize_sparse_numpy(normalizer, value: np.ndarray) -> np.ndarray:
    return value if normalizer is None else np.asarray(normalizer.normalize_predictor_sparse(value), dtype=np.float32)


def _inverse_pose_tensor(normalizer, value: torch.Tensor) -> torch.Tensor:
    return value if normalizer is None else normalizer.inverse_pose(value)


def decode_and_resolve_pose(
    target_raw: np.ndarray,
    current_tracker_raw: np.ndarray,
    current_head_yaw_world: float,
    current_head_position_world: np.ndarray,
    floor_y: float,
    joint_offsets_parent: np.ndarray,
    joint_rest_local_rotations_6d: np.ndarray,
    previous_root_yaw_world: float | None = None,
) -> ResolvedPose:
    """从当前 Cn 的 144D Pose 恢复 Head-anchored 世界状态。"""

    target = np.asarray(target_raw, dtype=np.float32).reshape(REALTIME_POSE_TARGET_DIM)
    tracker = np.asarray(current_tracker_raw, dtype=np.float32).reshape(
        TRACKER_COUNT, TRACKER_FEATURE_DIM
    )
    rotations_head, measured_root_yaw_head = decode_target_head_rotations_np(target)
    _, _, root_heading_observability = extract_rotation_heading_components_np(
        rotations_head[0]
    )
    candidate_root_yaw_world = float(current_head_yaw_world + measured_root_yaw_head)
    if float(root_heading_observability) < ROOT_HEADING_OBSERVABILITY_EPS:
        root_yaw_world = (
            float(current_head_yaw_world)
            if previous_root_yaw_world is None
            else float(previous_root_yaw_world)
        )
    elif previous_root_yaw_world is None:
        root_yaw_world = candidate_root_yaw_world
    else:
        delta = (candidate_root_yaw_world - previous_root_yaw_world + np.pi) % (
            2.0 * np.pi
        ) - np.pi
        root_yaw_world = float(previous_root_yaw_world + delta)
    root_yaw_head = root_yaw_world - float(current_head_yaw_world)
    root_head, hip_height, joints_head = resolve_root_head_reference_np(
        rotations_head,
        root_yaw_head,
        joint_offsets_parent,
        observed_head_height=float(tracker[HEAD_TRACKER_INDEX, 1]),
    )
    yaw_rotation = make_yaw_rotation_np(
        np.asarray([current_head_yaw_world], dtype=np.float64)
    )[0]
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
    return ResolvedPose(
        target_raw=target.copy(),
        joint_rotations_world=rotations_world.astype(np.float32),
        body_local_delta_6d=local_delta.astype(np.float32),
        root_yaw_world=root_yaw_world,
        hip_height=float(hip_height),
        root_position_world=root_world.astype(np.float32),
        joints_world=joints_world.astype(np.float32),
    )
