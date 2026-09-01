from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import torch

from data_loaders.build_realtime_longseq_eval_set import resolve_source_entry_path
from data_loaders.generate_realtime_pose_tasks import (
    compute_source_joint_rotations_world,
    load_realtime_source,
)
from data_loaders.realtime_pose_kinematics import rotation_6d_to_matrix_np
from data_loaders.sensor_masking import (
    CORE_TRACKER_INDICES,
    REALTIME_POSE_EVAL_METRICS_START_FRAME,
    REALTIME_POSE_FPS,
    REALTIME_POSE_TARGET_DIM,
    TRACKER_COUNT,
    TRACKER_TO_JOINT,
    validate_tracker_available,
)
from eval.evaluate_realtime_pose_predictor import (
    PREDICTOR_EVAL_FIRST_GENERATED_FRAME,
    evaluation_last_frame_exclusive,
)
from eval.realtime_pose_metrics import (
    aggregate_rpm_p2_mc_metrics,
    compute_rpm_p2_mc_metrics,
)
from sample.realtime_pose_runtime import (
    RealtimePoseRuntime,
    WorldPoseState,
    decode_and_resolve_pose,
)


@dataclass(frozen=True)
class TrackerStageDefinition:
    """一个计分阶段的 Tracker 语义。"""

    index: int
    name: str
    tracker_available: tuple[bool, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TrackerSequenceSchedule:
    """一条序列正式计分区间内的逐帧 Tracker schedule。"""

    tracker_available: np.ndarray  # [T,6] bool
    stage_indices: np.ndarray  # [T] int64


ScheduleBuilder = Callable[[int], TrackerSequenceSchedule]


@dataclass(frozen=True)
class TrackerEvaluationProtocol:
    """静态配置和动态退化实验共享的长序列评估协议。"""

    name: str
    warmup_tracker_available: tuple[bool, ...]
    stages: tuple[TrackerStageDefinition, ...]
    schedule_builder: ScheduleBuilder
    include_stage_metrics: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)


def create_eval_noise_generator(seed: int, device: torch.device) -> torch.Generator:
    """每个协议从同一 seed 开始，保证逐帧扩散噪声可直接比较。"""

    return torch.Generator(device=device).manual_seed(int(seed))


def build_static_tracker_protocol(
    name: str,
    tracker_available: np.ndarray | tuple[bool, ...],
) -> TrackerEvaluationProtocol:
    """把旧静态 Tracker 配置包装成共享评估协议。"""

    available = validate_tracker_available(np.asarray(tracker_available, dtype=bool))
    if available.shape != (TRACKER_COUNT,):
        raise ValueError("静态 tracker_available 必须为 [6]。")
    available_tuple = tuple(bool(value) for value in available.tolist())
    stage = TrackerStageDefinition(
        index=0,
        name=str(name),
        tracker_available=available_tuple,
    )

    def build_schedule(scored_frame_count: int) -> TrackerSequenceSchedule:
        frame_count = int(scored_frame_count)
        if frame_count <= 0:
            raise ValueError("静态协议至少需要一个计分帧。")
        return TrackerSequenceSchedule(
            tracker_available=np.broadcast_to(
                available[None], (frame_count, TRACKER_COUNT)
            ).copy(),
            stage_indices=np.zeros((frame_count,), dtype=np.int64),
        )

    return TrackerEvaluationProtocol(
        name=str(name),
        warmup_tracker_available=available_tuple,
        stages=(stage,),
        schedule_builder=build_schedule,
        include_stage_metrics=False,
        metadata={
            "tracker_available": list(available_tuple),
            "seen_during_training": True,
        },
    )


def evaluate_longseq_protocols(
    *,
    protocols: list[TrackerEvaluationProtocol]
    | tuple[TrackerEvaluationProtocol, ...],
    entries: list[dict],
    predictor,
    dit,
    diffusion,
    device: torch.device,
    normalizer,
    args,
) -> list[dict]:
    """依次评估多个协议；每个协议会重置相同的扩散噪声序列。"""

    if not protocols:
        raise ValueError("至少需要一个 Tracker 评估协议。")
    return [
        evaluate_longseq_protocol(
            protocol=protocol,
            entries=entries,
            predictor=predictor,
            dit=dit,
            diffusion=diffusion,
            device=device,
            normalizer=normalizer,
            args=args,
        )
        for protocol in protocols
    ]


def evaluate_longseq_protocol(
    *,
    protocol: TrackerEvaluationProtocol,
    entries: list[dict],
    predictor,
    dit,
    diffusion,
    device: torch.device,
    normalizer,
    args,
) -> dict:
    """对一个 Tracker 协议运行完整闭环，并返回整体与可选阶段指标。"""

    if not entries:
        raise RuntimeError("没有可用于长序列评估的 source sequence。")
    _validate_protocol(protocol)
    generated_frame_count = 0
    evaluated_frame_count = 0
    raw_sequence_metrics: list[dict[str, float | None]] = []
    deployed_sequence_metrics: list[dict[str, float | None]] = []
    stage_raw_metrics: list[list[dict[str, float | None]]] = [
        [] for _ in protocol.stages
    ]
    stage_deployed_metrics: list[list[dict[str, float | None]]] = [
        [] for _ in protocol.stages
    ]
    stage_frame_counts = np.zeros((len(protocol.stages),), dtype=np.int64)
    tracker_position_errors: list[float] = []
    tracker_rotation_errors: list[float] = []
    runtime_latencies_ms: list[float] = []
    noise_generator = create_eval_noise_generator(args.seed, device)

    for entry in entries:
        source = load_realtime_source(resolve_source_entry_path(entry))
        world_rotations = compute_source_joint_rotations_world(source)
        last = evaluation_last_frame_exclusive(
            len(world_rotations), int(args.max_frames)
        )
        scored_frame_count = last - REALTIME_POSE_EVAL_METRICS_START_FRAME
        if scored_frame_count <= 0:
            continue
        schedule = protocol.schedule_builder(scored_frame_count)
        _validate_sequence_schedule(protocol, schedule, scored_frame_count)

        scored_raw_rotations: list[np.ndarray] = []
        scored_deployed_rotations: list[np.ndarray] = []
        scored_target_rotations: list[np.ndarray] = []
        scored_raw_positions: list[np.ndarray] = []
        scored_deployed_positions: list[np.ndarray] = []
        scored_target_positions: list[np.ndarray] = []
        pose_history = [
            WorldPoseState(
                joint_rotations_world=world_rotations[index],
                root_yaw_world=float(source["root_yaw"][index]),
                hip_height=float(source["pelvis_height"][index, 0]),
                root_position_world=source["root_pos_world"][index],
            )
            for index in range(1, 11)
        ]
        runtime = create_longseq_runtime(
            source=source,
            predictor=predictor,
            dit=dit,
            diffusion=diffusion,
            device=device,
            normalizer=normalizer,
            args=args,
        )
        runtime.initialize_history(
            pose_history,
            source["tracker_pos_world"][:11],
            source["tracker_rot_world_6d"][:11],
            source["root_pos_world"][:11, 1],
        )

        for current in range(PREDICTOR_EVAL_FIRST_GENERATED_FRAME, last):
            scored = current >= REALTIME_POSE_EVAL_METRICS_START_FRAME
            if scored:
                scored_offset = current - REALTIME_POSE_EVAL_METRICS_START_FRAME
                frame_tracker_available = schedule.tracker_available[scored_offset]
            else:
                frame_tracker_available = np.asarray(
                    protocol.warmup_tracker_available, dtype=bool
                )
            previous_root_yaw = runtime.pose_history[-1].root_yaw_world
            noise = torch.randn(
                (1, REALTIME_POSE_TARGET_DIM),
                generator=noise_generator,
                device=device,
            )
            if scored and device.type == "cuda":
                torch.cuda.synchronize(device)
            started_at = time.perf_counter() if scored else 0.0
            result = runtime.step(
                source["tracker_pos_world"][current],
                source["tracker_rot_world_6d"][current],
                frame_tracker_available,
                float(source["root_pos_world"][current, 1]),
                noise=noise,
            )
            generated_frame_count += 1
            if not scored:
                continue
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            runtime_latencies_ms.append((time.perf_counter() - started_at) * 1000.0)
            raw_resolved = decode_and_resolve_pose(
                result.raw_pred_pose,
                result.current_tracker_raw,
                result.current_head_yaw_world,
                source["tracker_pos_world"][current, 0],
                float(source["root_pos_world"][current, 1]),
                source["joint_offsets_parent"],
                source["joint_rest_local_rotations_6d"],
                previous_root_yaw,
            )
            scored_raw_rotations.append(raw_resolved.joint_rotations_world)
            scored_deployed_rotations.append(
                result.resolved_pose.joint_rotations_world
            )
            scored_target_rotations.append(world_rotations[current])
            scored_raw_positions.append(raw_resolved.joints_world)
            scored_deployed_positions.append(result.resolved_pose.joints_world)
            scored_target_positions.append(source["joints_world"][current])
            _append_tracker_errors(
                tracker_available=frame_tracker_available,
                predicted_joint_positions=result.resolved_pose.joints_world,
                predicted_joint_rotations=result.resolved_pose.joint_rotations_world,
                measured_tracker_positions=source["tracker_pos_world"][current],
                measured_tracker_rotations_6d=source["tracker_rot_world_6d"][current],
                position_errors=tracker_position_errors,
                rotation_errors=tracker_rotation_errors,
            )
            evaluated_frame_count += 1

        sequence_values = {
            "raw_rotations": np.stack(scored_raw_rotations, axis=0),
            "deployed_rotations": np.stack(scored_deployed_rotations, axis=0),
            "target_rotations": np.stack(scored_target_rotations, axis=0),
            "raw_positions": np.stack(scored_raw_positions, axis=0),
            "deployed_positions": np.stack(scored_deployed_positions, axis=0),
            "target_positions": np.stack(scored_target_positions, axis=0),
        }
        raw_overall, raw_stages = compute_sequence_metrics_by_stage(
            predicted_global_rotations=sequence_values["raw_rotations"],
            target_global_rotations=sequence_values["target_rotations"],
            predicted_joint_positions=sequence_values["raw_positions"],
            target_joint_positions=sequence_values["target_positions"],
            stage_indices=schedule.stage_indices,
            stage_count=len(protocol.stages),
            fps=REALTIME_POSE_FPS,
        )
        deployed_overall, deployed_stages = compute_sequence_metrics_by_stage(
            predicted_global_rotations=sequence_values["deployed_rotations"],
            target_global_rotations=sequence_values["target_rotations"],
            predicted_joint_positions=sequence_values["deployed_positions"],
            target_joint_positions=sequence_values["target_positions"],
            stage_indices=schedule.stage_indices,
            stage_count=len(protocol.stages),
            fps=REALTIME_POSE_FPS,
        )
        raw_sequence_metrics.append(raw_overall)
        deployed_sequence_metrics.append(deployed_overall)
        for stage_index in range(len(protocol.stages)):
            stage_raw_metrics[stage_index].append(raw_stages[stage_index])
            stage_deployed_metrics[stage_index].append(
                deployed_stages[stage_index]
            )
            stage_frame_counts[stage_index] += int(
                np.count_nonzero(schedule.stage_indices == stage_index)
            )

    if evaluated_frame_count <= 0:
        raise RuntimeError(f"Tracker 协议 {protocol.name} 没有可计分帧。")
    report = {
        "name": protocol.name,
        **dict(protocol.metadata),
        "generated_frames": generated_frame_count,
        "evaluated_frames": evaluated_frame_count,
        "dit_raw": aggregate_rpm_p2_mc_metrics(raw_sequence_metrics),
        "dit_deployed": aggregate_rpm_p2_mc_metrics(deployed_sequence_metrics),
        "tracker_error": {
            "position_cm": float(np.mean(tracker_position_errors) * 100.0),
            "rotation_deg": float(np.degrees(np.mean(tracker_rotation_errors))),
        },
        "runtime_latency_ms": {
            "mean": float(np.mean(runtime_latencies_ms)),
            "p50": float(np.percentile(runtime_latencies_ms, 50.0)),
            "p90": float(np.percentile(runtime_latencies_ms, 90.0)),
        },
    }
    if protocol.include_stage_metrics:
        report["stages"] = [
            {
                "stage_index": definition.index,
                "name": definition.name,
                "tracker_count": int(sum(definition.tracker_available)),
                "tracker_available": list(definition.tracker_available),
                **dict(definition.metadata),
                "evaluated_frames": int(stage_frame_counts[definition.index]),
                "dit_raw": aggregate_rpm_p2_mc_metrics(
                    stage_raw_metrics[definition.index]
                ),
                "dit_deployed": aggregate_rpm_p2_mc_metrics(
                    stage_deployed_metrics[definition.index]
                ),
            }
            for definition in protocol.stages
        ]
    print(f"[longseq] {protocol.name}: {report['dit_deployed']}", flush=True)
    return report


def compute_sequence_metrics_by_stage(
    *,
    predicted_global_rotations: np.ndarray,
    target_global_rotations: np.ndarray,
    predicted_joint_positions: np.ndarray,
    target_joint_positions: np.ndarray,
    stage_indices: np.ndarray,
    stage_count: int,
    fps: float,
) -> tuple[dict[str, float | None], list[dict[str, float | None]]]:
    """先计算整条拼接序列，再计算阶段切片，避免漏掉切换边界导数。"""

    indices = np.asarray(stage_indices, dtype=np.int64)
    frame_count = int(np.asarray(predicted_global_rotations).shape[0])
    if indices.shape != (frame_count,):
        raise ValueError("stage_indices 必须与预测序列等长。")
    if int(stage_count) <= 0 or not np.isin(
        indices, np.arange(int(stage_count), dtype=np.int64)
    ).all():
        raise ValueError("stage_indices 包含未声明的阶段。")
    common = {
        "target_global_rotations": target_global_rotations,
        "target_joint_positions": target_joint_positions,
        "fps": float(fps),
    }
    overall = compute_rpm_p2_mc_metrics(
        predicted_global_rotations=predicted_global_rotations,
        predicted_joint_positions=predicted_joint_positions,
        **common,
    )
    stages: list[dict[str, float | None]] = []
    for stage_index in range(int(stage_count)):
        selected = indices == stage_index
        if not np.any(selected):
            raise ValueError(f"阶段 {stage_index} 没有计分帧。")
        stages.append(
            compute_rpm_p2_mc_metrics(
                predicted_global_rotations=predicted_global_rotations[selected],
                target_global_rotations=target_global_rotations[selected],
                predicted_joint_positions=predicted_joint_positions[selected],
                target_joint_positions=target_joint_positions[selected],
                fps=float(fps),
            )
        )
    return overall, stages


def create_longseq_runtime(
    *, source: dict, predictor, dit, diffusion, device, normalizer, args
) -> RealtimePoseRuntime:
    """按长序列正式评估参数创建闭环 runtime，供评估和可视化共用。"""

    return RealtimePoseRuntime(
        predictor,
        dit,
        diffusion,
        device,
        source["joint_offsets_parent"],
        source["joint_rest_local_rotations_6d"],
        normalizer,
        fabrik_iterations=args.fabrik_iterations,
        ik_direction_only_quality=args.ik_direction_only_quality,
        ik_residual_scale=args.ik_residual_scale,
        ik_position_solved_quality=args.ik_position_solved_quality,
        ik_gap_low=args.ik_gap_low,
        ik_gap_high=args.ik_gap_high,
        ik_direction_support=args.ik_direction_support,
        ik_untracked_strength=args.ik_untracked_strength,
        allow_missing_core_trackers=bool(
            getattr(args, "allow_missing_core_trackers", False)
        ),
    )


def _append_tracker_errors(
    *,
    tracker_available: np.ndarray,
    predicted_joint_positions: np.ndarray,
    predicted_joint_rotations: np.ndarray,
    measured_tracker_positions: np.ndarray,
    measured_tracker_rotations_6d: np.ndarray,
    position_errors: list[float],
    rotation_errors: list[float],
) -> None:
    tracker_indices = np.flatnonzero(tracker_available)
    joint_indices = np.asarray(TRACKER_TO_JOINT, dtype=np.int64)[tracker_indices]
    position_errors.extend(
        np.linalg.norm(
            predicted_joint_positions[joint_indices]
            - measured_tracker_positions[tracker_indices],
            axis=-1,
        ).tolist()
    )
    predicted_tracker_rotations = predicted_joint_rotations[joint_indices]
    measured_tracker_rotations = rotation_6d_to_matrix_np(
        measured_tracker_rotations_6d[tracker_indices]
    )
    relative = predicted_tracker_rotations.transpose(0, 2, 1) @ measured_tracker_rotations
    cosine = np.clip(
        (np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    rotation_errors.extend(np.arccos(cosine).tolist())


def _validate_protocol(protocol: TrackerEvaluationProtocol) -> None:
    warmup = validate_tracker_available(
        np.asarray(protocol.warmup_tracker_available, dtype=bool)
    )
    if warmup.shape != (TRACKER_COUNT,):
        raise ValueError("warmup_tracker_available 必须为 [6]。")
    if not protocol.stages:
        raise ValueError("Tracker 协议至少需要一个计分阶段。")
    for expected_index, stage in enumerate(protocol.stages):
        if stage.index != expected_index:
            raise ValueError("Tracker stage index 必须从 0 连续递增。")
        available = validate_tracker_available(
            np.asarray(stage.tracker_available, dtype=bool)
        )
        if available.shape != (TRACKER_COUNT,):
            raise ValueError("阶段 tracker_available 必须为 [6]。")


def _validate_sequence_schedule(
    protocol: TrackerEvaluationProtocol,
    schedule: TrackerSequenceSchedule,
    scored_frame_count: int,
) -> None:
    available = np.asarray(schedule.tracker_available, dtype=bool)
    stage_indices = np.asarray(schedule.stage_indices, dtype=np.int64)
    expected = (int(scored_frame_count), TRACKER_COUNT)
    if available.shape != expected:
        raise ValueError(
            f"Tracker schedule 必须为 {expected}，实际为 {available.shape}。"
        )
    validate_tracker_available(available)
    if stage_indices.shape != (int(scored_frame_count),):
        raise ValueError("Tracker schedule 的 stage_indices 必须为 [T]。")
    for definition in protocol.stages:
        selected = stage_indices == definition.index
        if not np.any(selected):
            raise ValueError(f"阶段 {definition.index} 没有计分帧。")
        expected_available = np.asarray(definition.tracker_available, dtype=bool)
        if not np.array_equal(
            available[selected],
            np.broadcast_to(expected_available, available[selected].shape),
        ):
            raise ValueError(
                f"阶段 {definition.index} 的逐帧 mask 与阶段定义不一致。"
            )
    valid_indices = np.arange(len(protocol.stages), dtype=np.int64)
    if not np.isin(stage_indices, valid_indices).all():
        raise ValueError("Tracker schedule 包含未声明的阶段。")
    if not available[:, list(CORE_TRACKER_INDICES)].all():
        raise ValueError("Tracker schedule 的核心三点必须始终 available。")
