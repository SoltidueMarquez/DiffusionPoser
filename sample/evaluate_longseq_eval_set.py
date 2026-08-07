from __future__ import annotations

import argparse
import hashlib
import json
import time
from argparse import BooleanOptionalAction
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from data_loaders.build_realtime_longseq_eval_set import (
    DEFAULT_LONGSEQ_EVAL_ROOT,
    build_sequence_output_dir_name,
    read_longseq_manifest,
    resolve_longseq_eval_dir,
    resolve_manifest_source_path,
)
from data_loaders.generate_realtime_pose_tasks import (
    compute_source_joint_rotations_world,
    load_realtime_source,
)
from data_loaders.realtime_pose_geometry import build_pose_target_np, extract_rotation_heading_np
from data_loaders.sensor_masking import (
    BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY,
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_TARGET_DIM,
    TRACKER_PATTERN_CATEGORIES,
)
from data_loaders.tracker_timeline import (
    TrackerTimeline,
    build_isolated_condition_timeline,
    classify_tracker_frame,
    isolated_condition_eval_mask,
    stable_context_seed,
)
from eval.evaluate_realtime_pose import public_result
from eval.evaluate_realtime_pose_rollout import evaluate_rollout_file, summarize_rollouts
from sample.realtime_pose_runtime import (
    RealtimePoseRuntime,
    RuntimeStepResult,
    step_realtime_pose_batch,
)
from sample.render_realtime_pose_comparison import render_realtime_pose_comparison
from sample.utils import load_checkpoint_model
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion
from utils.normalizer import RealtimePoseNormalizer
from utils.parser_util import (
    add_base_options,
    add_diffusion_options,
    add_model_options,
    add_sampling_options,
    parse_and_load_from_model,
    str2bool,
)


CONDITION_OUTPUT_TAGS = {
    "fixed_six": "f6",
    "fixed_three": "f3",
    "three_to_six": "36",
    "six_to_three": "63",
    "two_point_dropout_reconnect": "dr",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="在固定长序列集合上执行 144D 自回归评估。")
    add_base_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sampling_options(parser)

    longseq = parser.add_argument_group("longseq_eval")
    longseq.add_argument("--eval_root", default=DEFAULT_LONGSEQ_EVAL_ROOT, type=str)
    longseq.add_argument("--eval_set", default="latest", type=str)
    longseq.add_argument(
        "--normalizer_dir",
        default="dataset/meta_AMASS_realtime_pose_144d_pelvis_residual_root_y0_stationary5_60hz",
        type=str,
    )
    longseq.add_argument("--normalize_input", default=True, type=str2bool)
    longseq.add_argument("--input_feats", default=REALTIME_POSE_TARGET_DIM, type=int)
    longseq.add_argument("--limit", default=0, type=int)
    longseq.add_argument("--sequence_batch_size", default=4, type=int)
    longseq.add_argument(
        "--conditions",
        nargs="+",
        choices=TRACKER_PATTERN_CATEGORIES,
        default=list(TRACKER_PATTERN_CATEGORIES),
    )
    longseq.add_argument("--timeline_seed", default=10, type=int)
    longseq.add_argument(
        "--diffusion_noise_mode",
        default="per_frame",
        choices=("per_frame", "fixed_sequence", "correlated"),
    )
    longseq.add_argument("--diffusion_noise_rho", default=0.95, type=float)
    longseq.add_argument("--inference_steps", default=5, type=int)
    longseq.add_argument("--latency_warmup_frames", default=20, type=int)
    longseq.add_argument("--require_cuda", default=True, action=BooleanOptionalAction)
    longseq.add_argument("--show_progress", default=True, action=BooleanOptionalAction)

    render = parser.add_argument_group("render")
    render.add_argument("--render_mp4", default=False, action=BooleanOptionalAction)
    render.add_argument("--render_fps", default=30, type=int)
    render.add_argument("--render_stride", default=1, type=int)
    render.add_argument("--render_camera_mode", default="follow", choices=["global", "follow"], type=str)
    render.add_argument("--render_layout", default="overlay", choices=["split", "overlay"], type=str)
    render.add_argument("--render_local_radius", default=1.25, type=float)
    return parser


def rollout_long_sequence_source(
    model,
    diffusion,
    source: dict[str, np.ndarray],
    timeline: TrackerTimeline,
    device: torch.device,
    normalizer: RealtimePoseNormalizer | None,
    projected_ddim_mode: str = "all_steps",
    projected_ddim_late_steps: int = 5,
    measure_latency: bool = False,
    show_progress: bool = False,
    progress_desc: str = "",
) -> dict[str, np.ndarray]:
    """从首帧冷启动逐帧闭环采样，不使用任何 GT pose warmup。"""

    frame_count = int(source[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY].shape[0])
    if frame_count <= 0:
        raise ValueError("长序列 source 不能为空。")
    joint_rotations_world = compute_source_joint_rotations_world(source)
    runtime = RealtimePoseRuntime(
        model,
        diffusion,
        device,
        source["joint_offsets_parent"],
        source["joint_rest_local_rotations_6d"],
        normalizer=normalizer,
        projected_ddim_mode=projected_ddim_mode,
        projected_ddim_late_steps=projected_ddim_late_steps,
    )
    values: dict[str, list] = {
        name: []
        for name in (
            "reference_target_raw",
            "raw_pred_target_raw",
            "deployed_pred_target_raw",
            "reference_body_local_delta_6d",
            "predicted_body_local_delta_6d",
            "reference_joints_world",
            "predicted_joints_world",
            "reference_root_position_world",
            "predicted_root_position_world",
            "reference_root_yaw_world",
            "predicted_root_yaw_world",
            "reference_hip_height",
            "predicted_hip_height",
            "tracker_pos_world",
            "tracker_rot_world_6d",
            "current_tracker_raw",
            "configured",
            "measured_valid",
            "d_off",
            "d_on",
            "hard_rotation_state",
            "history_length",
            "contact_target",
            "contact_logits",
            "future_leg_prediction",
            "future_leg_target",
            "scenario",
            "hard_rotation_max_error",
            "sampling_latency_ms",
            "e2e_latency_ms",
        )
    }
    frame_indices = range(frame_count)
    if show_progress:
        frame_indices = tqdm(
            frame_indices,
            total=frame_count,
            desc=progress_desc or "longseq",
            unit="frame",
            dynamic_ncols=True,
        )
    for frame_index in frame_indices:
        frame_started = time.perf_counter()
        history_length = len(runtime.pose_history)
        if measure_latency:
            torch.cuda.synchronize(device)
            sampling_started = time.perf_counter()
        step = runtime.step(
            source["tracker_pos_world"][frame_index],
            source["tracker_rot_world_6d"][frame_index],
            timeline.configured[frame_index],
            timeline.measured_valid[frame_index],
            float(source["root_pos_world"][frame_index, 1]),
        )
        if measure_latency:
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - sampling_started) * 1000.0
            total_elapsed = (time.perf_counter() - frame_started) * 1000.0
        else:
            elapsed = total_elapsed = float("nan")
        resolved = step.resolved_pose
        reference_target = build_pose_target_np(
            joint_rotations_world[frame_index : frame_index + 1],
            runtime.previous_head_yaw,
        )[0]
        values["reference_target_raw"].append(reference_target)
        values["raw_pred_target_raw"].append(step.raw_pred_xstart)
        values["deployed_pred_target_raw"].append(step.deployed_pred_xstart)
        values["reference_body_local_delta_6d"].append(
            source[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY][frame_index]
        )
        values["predicted_body_local_delta_6d"].append(resolved.body_local_delta_6d)
        values["reference_joints_world"].append(source["joints_world"][frame_index])
        values["predicted_joints_world"].append(resolved.joints_world)
        values["reference_root_position_world"].append(source["root_pos_world"][frame_index])
        values["predicted_root_position_world"].append(resolved.root_position_world)
        values["reference_root_yaw_world"].append(
            float(extract_rotation_heading_np(joint_rotations_world[frame_index, 0]))
        )
        values["predicted_root_yaw_world"].append(resolved.root_yaw_world)
        values["reference_hip_height"].append(float(source["pelvis_height"][frame_index, 0]))
        values["predicted_hip_height"].append(resolved.hip_height)
        for name in ("tracker_pos_world", "tracker_rot_world_6d"):
            values[name].append(source[name][frame_index])
        values["current_tracker_raw"].append(step.current_tracker_raw)
        for name in ("configured", "measured_valid"):
            values[name].append(getattr(timeline, name)[frame_index])
        values["d_off"].append(runtime.previous_d_off.copy())
        values["d_on"].append(runtime.previous_d_on.copy())
        values["hard_rotation_state"].append(step.hard_rotation_state)
        values["history_length"].append(history_length)
        values["contact_target"].append(source["stationary_prob_5"][frame_index, 1:3])
        values["contact_logits"].append(
            np.full(2, np.nan, dtype=np.float32)
            if step.contact_logits is None
            else step.contact_logits
        )
        values["future_leg_prediction"].append(
            np.full((3, 8, 6), np.nan, dtype=np.float32)
            if step.future_leg_prediction is None
            else step.future_leg_prediction
        )
        if frame_index + 3 < frame_count:
            future_pose = build_pose_target_np(
                joint_rotations_world[frame_index + 1 : frame_index + 4],
                runtime.previous_head_yaw,
            ).reshape(3, 24, 6)
            values["future_leg_target"].append(
                future_pose[:, np.asarray([1, 4, 7, 10, 2, 5, 8, 11])]
            )
        else:
            values["future_leg_target"].append(np.full((3, 8, 6), np.nan, dtype=np.float32))
        values["scenario"].append(_classify_timeline_frame(timeline, frame_index))
        values["hard_rotation_max_error"].append(resolved.hard_rotation_max_error)
        values["sampling_latency_ms"].append(elapsed)
        values["e2e_latency_ms"].append(total_elapsed)

    payload = {
        name: np.asarray(items, dtype=np.float32 if name not in {
            "configured", "measured_valid", "hard_rotation_state", "scenario"
        } else None)[None]
        for name, items in values.items()
    }
    payload["configured"] = payload["configured"].astype(bool)
    payload["measured_valid"] = payload["measured_valid"].astype(bool)
    payload["hard_rotation_state"] = payload["hard_rotation_state"].astype(bool)
    payload["d_off"] = payload["d_off"].astype(np.int64)
    payload["d_on"] = payload["d_on"].astype(np.int64)
    payload["history_length"] = payload["history_length"].astype(np.int64)
    payload["scenario"] = np.asarray(values["scenario"])[None]
    payload["fps"] = np.float32(60.0)
    payload["absolute_frame_index"] = np.arange(frame_count, dtype=np.int64)
    payload["eval_frame_mask"] = np.ones((1, frame_count), dtype=bool)
    return payload


def rollout_long_sequence_sources(
    model,
    diffusion,
    sources: list[dict[str, np.ndarray]],
    timelines: list[TrackerTimeline],
    device: torch.device,
    normalizer: RealtimePoseNormalizer | None,
    projected_ddim_mode: str = "all_steps",
    projected_ddim_late_steps: int = 5,
    measure_latency: bool = False,
    show_progress: bool = False,
    progress_desc: str = "",
    diffusion_seeds: list[int] | None = None,
    diffusion_noise_mode: str = "per_frame",
    diffusion_noise_rho: float = 0.95,
) -> list[dict[str, np.ndarray]]:
    """跨序列逐帧批处理，序列结束后仅保留其余活跃 runtime。"""

    if not sources or len(sources) != len(timelines):
        raise ValueError("sources 与 timelines 必须非空且数量相同。")
    if diffusion_seeds is not None and len(diffusion_seeds) != len(sources):
        raise ValueError("diffusion_seeds 必须与 sources 数量相同。")
    if diffusion_noise_mode not in {"per_frame", "fixed_sequence", "correlated"}:
        raise ValueError("diffusion_noise_mode 必须为 per_frame/fixed_sequence/correlated。")
    if not -1.0 <= float(diffusion_noise_rho) <= 1.0:
        raise ValueError("diffusion_noise_rho 必须位于 [-1,1]。")
    frame_counts = [
        int(source[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY].shape[0]) for source in sources
    ]
    if any(frame_count <= 0 for frame_count in frame_counts):
        raise ValueError("长序列 source 不能为空。")
    if any(len(timeline.configured) != frame_count for timeline, frame_count in zip(timelines, frame_counts)):
        raise ValueError("timeline 帧数必须与对应 source 一致。")

    joint_rotations = [compute_source_joint_rotations_world(source) for source in sources]
    runtimes = [
        RealtimePoseRuntime(
            model,
            diffusion,
            device,
            source["joint_offsets_parent"],
            source["joint_rest_local_rotations_6d"],
            normalizer=normalizer,
            projected_ddim_mode=projected_ddim_mode,
            projected_ddim_late_steps=projected_ddim_late_steps,
        )
        for source in sources
    ]
    values = [_new_rollout_values() for _ in sources]
    noise_generators = None
    if diffusion_seeds is not None:
        noise_generators = []
        for seed in diffusion_seeds:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            noise_generators.append(generator)
    fixed_noise: list[torch.Tensor | None] = [None] * len(sources)
    previous_noise: list[torch.Tensor | None] = [None] * len(sources)
    progress = None
    if show_progress:
        progress = tqdm(
            total=sum(frame_counts),
            desc=progress_desc or "longseq batch",
            unit="frame",
            dynamic_ncols=True,
        )
    for frame_index in range(max(frame_counts)):
        active_indices = [
            index for index, frame_count in enumerate(frame_counts) if frame_index < frame_count
        ]
        active_runtimes = [runtimes[index] for index in active_indices]
        history_lengths = [len(runtime.pose_history) for runtime in active_runtimes]
        frame_started = time.perf_counter()
        if measure_latency:
            torch.cuda.synchronize(device)
            sampling_started = time.perf_counter()
        frame_noise = None
        if noise_generators is not None:
            noise_values = []
            for index in active_indices:
                if diffusion_noise_mode == "fixed_sequence":
                    if fixed_noise[index] is None:
                        fixed_noise[index] = torch.randn(
                            REALTIME_POSE_TARGET_DIM,
                            generator=noise_generators[index],
                            device=device,
                        )
                    value = fixed_noise[index]
                else:
                    innovation = torch.randn(
                        REALTIME_POSE_TARGET_DIM,
                        generator=noise_generators[index],
                        device=device,
                    )
                    if (
                        diffusion_noise_mode == "correlated"
                        and previous_noise[index] is not None
                    ):
                        rho = float(diffusion_noise_rho)
                        value = (
                            rho * previous_noise[index]
                            + np.sqrt(1.0 - rho**2) * innovation
                        )
                    else:
                        value = innovation
                    previous_noise[index] = value
                noise_values.append(value)
            frame_noise = torch.stack(noise_values)
        steps = step_realtime_pose_batch(
            active_runtimes,
            np.stack([sources[index]["tracker_pos_world"][frame_index] for index in active_indices]),
            np.stack(
                [sources[index]["tracker_rot_world_6d"][frame_index] for index in active_indices]
            ),
            np.stack([timelines[index].configured[frame_index] for index in active_indices]),
            np.stack([timelines[index].measured_valid[frame_index] for index in active_indices]),
            np.asarray(
                [sources[index]["root_pos_world"][frame_index, 1] for index in active_indices],
                dtype=np.float32,
            ),
            noise=frame_noise,
        )
        if measure_latency:
            torch.cuda.synchronize(device)
            sampling_elapsed = (time.perf_counter() - sampling_started) * 1000.0
            e2e_elapsed = (time.perf_counter() - frame_started) * 1000.0
        else:
            sampling_elapsed = e2e_elapsed = float("nan")
        for active_offset, sequence_index in enumerate(active_indices):
            _append_rollout_frame(
                values=values[sequence_index],
                source=sources[sequence_index],
                timeline=timelines[sequence_index],
                runtime=runtimes[sequence_index],
                joint_rotations_world=joint_rotations[sequence_index],
                frame_index=frame_index,
                history_length=history_lengths[active_offset],
                step=steps[active_offset],
                sampling_latency_ms=sampling_elapsed,
                e2e_latency_ms=e2e_elapsed,
            )
        if progress is not None:
            progress.update(len(active_indices))
    if progress is not None:
        progress.close()
    return [
        _finalize_rollout_values(sequence_values, frame_count)
        for sequence_values, frame_count in zip(values, frame_counts)
    ]


def _new_rollout_values() -> dict[str, list]:
    return {
        name: []
        for name in (
            "reference_target_raw",
            "raw_pred_target_raw",
            "deployed_pred_target_raw",
            "reference_body_local_delta_6d",
            "predicted_body_local_delta_6d",
            "reference_joints_world",
            "predicted_joints_world",
            "reference_root_position_world",
            "predicted_root_position_world",
            "reference_root_yaw_world",
            "predicted_root_yaw_world",
            "reference_hip_height",
            "predicted_hip_height",
            "tracker_pos_world",
            "tracker_rot_world_6d",
            "current_tracker_raw",
            "configured",
            "measured_valid",
            "d_off",
            "d_on",
            "hard_rotation_state",
            "history_length",
            "contact_target",
            "contact_logits",
            "future_leg_prediction",
            "future_leg_target",
            "scenario",
            "hard_rotation_max_error",
            "sampling_latency_ms",
            "e2e_latency_ms",
        )
    }


def _append_rollout_frame(
    values: dict[str, list],
    source: dict[str, np.ndarray],
    timeline: TrackerTimeline,
    runtime: RealtimePoseRuntime,
    joint_rotations_world: np.ndarray,
    frame_index: int,
    history_length: int,
    step: RuntimeStepResult,
    sampling_latency_ms: float,
    e2e_latency_ms: float,
) -> None:
    resolved = step.resolved_pose
    reference_target = build_pose_target_np(
        joint_rotations_world[frame_index : frame_index + 1],
        runtime.previous_head_yaw,
    )[0]
    values["reference_target_raw"].append(reference_target)
    values["raw_pred_target_raw"].append(step.raw_pred_xstart)
    values["deployed_pred_target_raw"].append(step.deployed_pred_xstart)
    values["reference_body_local_delta_6d"].append(
        source[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY][frame_index]
    )
    values["predicted_body_local_delta_6d"].append(resolved.body_local_delta_6d)
    values["reference_joints_world"].append(source["joints_world"][frame_index])
    values["predicted_joints_world"].append(resolved.joints_world)
    values["reference_root_position_world"].append(source["root_pos_world"][frame_index])
    values["predicted_root_position_world"].append(resolved.root_position_world)
    values["reference_root_yaw_world"].append(
        float(extract_rotation_heading_np(joint_rotations_world[frame_index, 0]))
    )
    values["predicted_root_yaw_world"].append(resolved.root_yaw_world)
    values["reference_hip_height"].append(float(source["pelvis_height"][frame_index, 0]))
    values["predicted_hip_height"].append(resolved.hip_height)
    for name in ("tracker_pos_world", "tracker_rot_world_6d"):
        values[name].append(source[name][frame_index])
    values["current_tracker_raw"].append(step.current_tracker_raw)
    for name in ("configured", "measured_valid"):
        values[name].append(getattr(timeline, name)[frame_index])
    values["d_off"].append(runtime.previous_d_off.copy())
    values["d_on"].append(runtime.previous_d_on.copy())
    values["hard_rotation_state"].append(step.hard_rotation_state)
    values["history_length"].append(history_length)
    values["contact_target"].append(source["stationary_prob_5"][frame_index, 1:3])
    values["contact_logits"].append(
        np.full(2, np.nan, dtype=np.float32)
        if step.contact_logits is None
        else step.contact_logits
    )
    values["future_leg_prediction"].append(
        np.full((3, 8, 6), np.nan, dtype=np.float32)
        if step.future_leg_prediction is None
        else step.future_leg_prediction
    )
    frame_count = int(source[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY].shape[0])
    if frame_index + 3 < frame_count:
        future_pose = build_pose_target_np(
            joint_rotations_world[frame_index + 1 : frame_index + 4],
            runtime.previous_head_yaw,
        ).reshape(3, 24, 6)
        values["future_leg_target"].append(
            future_pose[:, np.asarray([1, 4, 7, 10, 2, 5, 8, 11])]
        )
    else:
        values["future_leg_target"].append(np.full((3, 8, 6), np.nan, dtype=np.float32))
    values["scenario"].append(_classify_timeline_frame(timeline, frame_index))
    values["hard_rotation_max_error"].append(resolved.hard_rotation_max_error)
    values["sampling_latency_ms"].append(sampling_latency_ms)
    values["e2e_latency_ms"].append(e2e_latency_ms)


def _finalize_rollout_values(values: dict[str, list], frame_count: int) -> dict[str, np.ndarray]:
    payload = {
        name: np.asarray(
            items,
            dtype=(
                np.float32
                if name not in {"configured", "measured_valid", "hard_rotation_state", "scenario"}
                else None
            ),
        )[None]
        for name, items in values.items()
    }
    for name in ("configured", "measured_valid", "hard_rotation_state"):
        payload[name] = payload[name].astype(bool)
    payload["d_off"] = payload["d_off"].astype(np.int64)
    payload["d_on"] = payload["d_on"].astype(np.int64)
    payload["history_length"] = payload["history_length"].astype(np.int64)
    payload["scenario"] = np.asarray(values["scenario"])[None]
    payload["fps"] = np.float32(60.0)
    payload["absolute_frame_index"] = np.arange(frame_count, dtype=np.int64)
    payload["eval_frame_mask"] = np.ones((1, frame_count), dtype=bool)
    return payload


def _classify_timeline_frame(timeline: TrackerTimeline, frame_index: int) -> str:
    """复用数据管线的事件语义，避免评估脚本维护另一套分类规则。"""

    return classify_tracker_frame(timeline, frame_index)


def summarize_latency(values_ms: np.ndarray, warmup_frames: int = 0) -> dict[str, float | int | None]:
    """汇总逐帧延迟；预热帧仍参与质量评估，但不参与性能统计。"""

    values = np.asarray(values_ms, dtype=np.float64).reshape(-1)
    warmup = min(max(int(warmup_frames), 0), values.size)
    measured = values[warmup:]
    measured = measured[np.isfinite(measured)]
    if measured.size == 0:
        return {
            "frames": 0,
            "warmup_frames": warmup,
            "mean_ms": None,
            "p50_ms": None,
            "p90_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
            "effective_fps": None,
            "frames_under_16_67ms_ratio": None,
        }
    mean_ms = float(measured.mean())
    return {
        "frames": int(measured.size),
        "warmup_frames": warmup,
        "mean_ms": mean_ms,
        "p50_ms": float(np.percentile(measured, 50)),
        "p90_ms": float(np.percentile(measured, 90)),
        "p95_ms": float(np.percentile(measured, 95)),
        "p99_ms": float(np.percentile(measured, 99)),
        "max_ms": float(measured.max()),
        "effective_fps": 1000.0 / mean_ms if mean_ms > 0.0 else None,
        "frames_under_16_67ms_ratio": float(np.mean(measured <= (1000.0 / 60.0))),
    }


def evaluate_longseq_entries(
    entries: list[dict[str, Any]],
    eval_set_dir: Path,
    output_dir: Path,
    model,
    diffusion,
    device: torch.device,
    normalizer: RealtimePoseNormalizer | None,
    projected_ddim_mode: str = "all_steps",
    projected_ddim_late_steps: int = 5,
    model_path: str | Path = "",
    weights: str = "",
    limit: int = 0,
    sequence_batch_size: int = 1,
    conditions: list[str] | tuple[str, ...] = TRACKER_PATTERN_CATEGORIES,
    timeline_seed: int = 10,
    diffusion_noise_mode: str = "per_frame",
    diffusion_noise_rho: float = 0.95,
    render_mp4: bool = False,
    render_fps: int = 30,
    render_stride: int = 1,
    render_camera_mode: str = "follow",
    render_layout: str = "overlay",
    render_local_radius: float = 1.25,
    latency_warmup_frames: int = 20,
    runtime_metadata: dict[str, Any] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    eval_set_dir = Path(eval_set_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_entries = entries[: int(limit)] if int(limit) > 0 else entries
    if not selected_entries:
        raise RuntimeError("长序列评估集合为空。")
    selected_conditions = tuple(dict.fromkeys(str(value) for value in conditions))
    if not selected_conditions or any(
        condition not in TRACKER_PATTERN_CATEGORIES for condition in selected_conditions
    ):
        raise ValueError(f"条件必须来自 {TRACKER_PATTERN_CATEGORIES}。")
    # 按条件展开后再分批，批内条件一致，便于解读性能与显存数据。
    jobs = [
        (entry, condition)
        for condition in selected_conditions
        for entry in selected_entries
    ]

    results = []
    sampling_latency_values: list[np.ndarray] = []
    e2e_latency_values: list[np.ndarray] = []
    batch_size = max(1, int(sequence_batch_size))
    for batch_start in range(0, len(jobs), batch_size):
        batch_jobs = jobs[batch_start : batch_start + batch_size]
        batch_sources: list[dict[str, np.ndarray]] = []
        batch_timelines: list[TrackerTimeline] = []
        batch_eval_masks: list[np.ndarray] = []
        batch_diffusion_seeds: list[int] = []
        for entry, condition in batch_jobs:
            sequence_id = str(entry["sequence_id"])
            source_path = resolve_manifest_source_path(eval_set_dir=eval_set_dir, entry=entry)
            source = load_realtime_source(source_path)
            frame_count = int(source[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY].shape[0])
            timeline = build_isolated_condition_timeline(
                source_id=sequence_id,
                frame_count=frame_count,
                condition=condition,
                global_seed=int(timeline_seed),
            )
            batch_sources.append(source)
            batch_timelines.append(timeline)
            batch_eval_masks.append(isolated_condition_eval_mask(timeline, condition))
            # 同一源序列在不同条件下使用相同的逐帧初始噪声，避免扩散随机性混入对照。
            batch_diffusion_seeds.append(
                int(stable_context_seed(timeline_seed, sequence_id, "longseq_diffusion") % (2**63))
            )
            print(f"[longseq] {condition}/{sequence_id}: {frame_count} frames")
        payloads = rollout_long_sequence_sources(
            model=model,
            diffusion=diffusion,
            sources=batch_sources,
            timelines=batch_timelines,
            device=device,
            normalizer=normalizer,
            projected_ddim_mode=projected_ddim_mode,
            projected_ddim_late_steps=projected_ddim_late_steps,
            measure_latency=device.type == "cuda",
            show_progress=bool(show_progress),
            progress_desc=(
                f"{batch_start + 1}-{batch_start + len(batch_jobs)}/"
                f"{len(jobs)} batch"
            ),
            diffusion_seeds=batch_diffusion_seeds,
            diffusion_noise_mode=diffusion_noise_mode,
            diffusion_noise_rho=diffusion_noise_rho,
        )

        for (entry, condition), source, payload, eval_mask in zip(
            batch_jobs,
            batch_sources,
            payloads,
            batch_eval_masks,
        ):
            payload["eval_frame_mask"] = eval_mask[None]
            sequence_id = str(entry["sequence_id"])
            frame_count = int(source[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY].shape[0])
            condition_dir = CONDITION_OUTPUT_TAGS[condition]
            sequence_dir = output_dir / condition_dir / build_sequence_output_dir_name(entry)
            sequence_dir.mkdir(parents=True, exist_ok=True)
            result_path = sequence_dir / "rollout_result.npz"
            np.savez(result_path, **payload)
            result = evaluate_rollout_file(result_path)
            # 第一个批的每条流共享 GPU 预热阶段，应统一排除。
            sequence_warmup = int(latency_warmup_frames) if batch_start == 0 else 0
            result["latency"] = {
                "sampling": summarize_latency(payload["sampling_latency_ms"], sequence_warmup),
                "e2e": summarize_latency(payload["e2e_latency_ms"], sequence_warmup),
            }
            sampling_latency_values.append(
                payload["sampling_latency_ms"].reshape(-1)[sequence_warmup:]
            )
            e2e_latency_values.append(payload["e2e_latency_ms"].reshape(-1)[sequence_warmup:])
            result.update(
                {
                    "sequence_id": sequence_id,
                    "condition": condition,
                    "source_relative_path": str(entry.get("source_relative_path", "")),
                    "num_frames": frame_count,
                    "evaluated_frames": int(eval_mask.sum()),
                    "result_path": str(result_path),
                }
            )

            if render_mp4:
                mp4_path = sequence_dir / "comparison.mp4"
                render_realtime_pose_comparison(
                    output_path=mp4_path,
                    reference_joints=payload["reference_joints_world"],
                    predicted_joints=payload["predicted_joints_world"],
                    tracker_pos_world=payload["tracker_pos_world"],
                    sensor_valid=payload["measured_valid"],
                    eval_frame_mask=payload["eval_frame_mask"],
                    root_yaw_reference=payload["reference_root_yaw_world"],
                    root_yaw_predicted=payload["predicted_root_yaw_world"],
                    fps=int(render_fps),
                    stride=int(render_stride),
                    camera_mode=str(render_camera_mode),
                    layout=str(render_layout),
                    local_radius=float(render_local_radius),
                )
                result["mp4_path"] = str(mp4_path)

            summary_path = sequence_dir / "rollout_eval_summary.json"
            with summary_path.open("w", encoding="utf-8") as file:
                json.dump({"summary": public_result(result)}, file, ensure_ascii=False, indent=2)
            result["summary_path"] = str(summary_path)
            results.append(result)

    aggregate = summarize_rollouts(results)
    aggregate["by_condition"] = {
        condition: summarize_rollouts(
            [result for result in results if result["condition"] == condition]
        )
        for condition in selected_conditions
    }
    aggregate["latency"] = {
        "sampling": summarize_latency(np.concatenate(sampling_latency_values)),
        "e2e": summarize_latency(np.concatenate(e2e_latency_values)),
    }
    metadata = dict(runtime_metadata or {})
    metadata["projected_ddim_mode"] = str(projected_ddim_mode)
    metadata["projected_ddim_late_steps"] = int(projected_ddim_late_steps)
    metadata["sequence_batch_size"] = int(batch_size)
    metadata["latency_scope"] = "active_batch_wall_time_per_stream"
    metadata["evaluation_protocol"] = "isolated_condition_cold_start"
    metadata["conditions"] = list(selected_conditions)
    metadata["shared_diffusion_noise_across_conditions"] = True
    metadata["diffusion_noise_mode"] = str(diffusion_noise_mode)
    metadata["diffusion_noise_rho"] = float(diffusion_noise_rho)
    if device.type == "cuda":
        metadata["peak_cuda_memory_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
    summary_payload = {
        "summary": aggregate,
        "files": [public_result(result) for result in results],
        "metadata": {
            "kind": "realtime_pose_144d_longseq_isolated_conditions",
            "eval_set_dir": str(eval_set_dir),
            "output_dir": str(output_dir),
            "model_path": str(model_path),
            "weights": str(weights),
            "timeline_seed": int(timeline_seed),
            "source_sequence_count": len(selected_entries),
            "condition_count": len(selected_conditions),
            "rollout_count": len(results),
            **metadata,
        },
    }
    aggregate_path = output_dir / "longseq_eval_summary.json"
    with aggregate_path.open("w", encoding="utf-8") as file:
        json.dump(summary_payload, file, ensure_ascii=False, indent=2)
    summary_payload["summary_path"] = str(aggregate_path)
    return summary_payload


def build_default_output_dir(
    eval_set_dir: Path,
    model_path: str | Path,
    weights: str,
    projected_ddim_mode: str = "all_steps",
    projected_ddim_late_steps: int = 5,
    sequence_batch_size: int = 1,
    conditions: list[str] | tuple[str, ...] = TRACKER_PATTERN_CATEGORIES,
) -> Path:
    """构造短路径，并用稳定摘要隔离不同评测集与训练 run。"""

    model_path = Path(model_path)
    model_stem = model_path.stem
    step_text = model_stem.removeprefix("model")
    step_tag = str(int(step_text)) if step_text.isdigit() else _short_digest(model_stem, 6)
    weight_tag = {"ema": "e", "model": "m", "": "m"}.get(str(weights), "w")
    projection_tag = {
        "all_steps": "a",
        "final_step": "f",
    }.get(str(projected_ddim_mode))
    if projection_tag is None:
        projection_tag = (
            f"l{int(projected_ddim_late_steps)}"
            if str(projected_ddim_mode) == "late_steps"
            else "p"
        )
    # 摘要纳入评测集和 checkpoint 父目录，相同 step 的不同 run 不会覆盖。
    identity = _short_digest(
        f"{Path(eval_set_dir).resolve()}\n{model_path.resolve().parent}",
        10,
    )
    condition_values = tuple(dict.fromkeys(str(value) for value in conditions))
    condition_tag = f"c{len(condition_values)}"
    if condition_values != TRACKER_PATTERN_CATEGORIES:
        condition_tag += _short_digest("\n".join(condition_values), 4)
    leaf = (
        f"{step_tag}{weight_tag}-{projection_tag}-b{int(sequence_batch_size)}-"
        f"{condition_tag}-{identity}"
    )
    return Path("output") / "l" / leaf


def _short_digest(value: str, length: int) -> str:
    digest_size = max(1, (int(length) + 1) // 2)
    return hashlib.blake2s(str(value).encode("utf-8"), digest_size=digest_size).hexdigest()[:length]


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_and_load_from_model(
        build_arg_parser(),
        argv=argv,
        ignore_keys={"ts_respace"},
    )
    if int(args.inference_steps) <= 0:
        raise ValueError("inference_steps 必须大于 0。")
    if int(args.projected_ddim_late_steps) <= 0:
        raise ValueError("projected_ddim_late_steps 必须大于 0。")
    if int(args.sequence_batch_size) <= 0:
        raise ValueError("sequence_batch_size 必须大于 0。")
    if not -1.0 <= float(args.diffusion_noise_rho) <= 1.0:
        raise ValueError("diffusion_noise_rho 必须位于 [-1,1]。")
    selected_conditions = list(dict.fromkeys(str(value) for value in args.conditions))
    args.ts_respace = f"ddim{int(args.inference_steps)}"
    fixseed(int(args.seed))
    eval_set_dir = resolve_longseq_eval_dir(eval_root=args.eval_root, eval_set=args.eval_set)
    entries = read_longseq_manifest(eval_set_dir)
    normalizer = (
        RealtimePoseNormalizer(args.normalizer_dir)
        if bool(args.normalize_input)
        else None
    )

    if bool(args.require_cuda) and (not bool(args.cuda) or not torch.cuda.is_available()):
        raise RuntimeError("长序列实时评测要求使用可用的 CUDA GPU。")
    dist_util.setup_dist(args.device if args.cuda else -1)
    device = dist_util.dev()
    if bool(args.require_cuda) and device.type != "cuda":
        raise RuntimeError(f"长序列实时评测期望 CUDA，实际设备为 {device}。")
    model, diffusion = create_model_and_diffusion(args)
    if int(diffusion.num_timesteps) != int(args.inference_steps):
        raise RuntimeError(
            f"期望 {args.inference_steps} 个 DDIM 推理步，实际为 {diffusion.num_timesteps}。"
        )
    model, weights = load_checkpoint_model(model, args.model_path, device=device, use_ema=args.use_ema)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timestep_map = [int(value) for value in diffusion.timestep_map]
    print(
        f"[longseq] device={device}, train_steps={args.diffusion_steps}, "
        f"inference_steps={diffusion.num_timesteps}, "
        f"sequence_batch_size={args.sequence_batch_size}, conditions={selected_conditions}, "
        f"timestep_map={timestep_map}"
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if str(args.output_dir).strip()
        else build_default_output_dir(
            eval_set_dir,
            args.model_path,
            weights,
            projected_ddim_mode=args.projected_ddim_mode,
            projected_ddim_late_steps=args.projected_ddim_late_steps,
            sequence_batch_size=args.sequence_batch_size,
            conditions=selected_conditions,
        ).resolve()
    )
    summary = evaluate_longseq_entries(
        entries=entries,
        eval_set_dir=eval_set_dir,
        output_dir=output_dir,
        model=model,
        diffusion=diffusion,
        device=device,
        normalizer=normalizer,
        projected_ddim_mode=str(args.projected_ddim_mode),
        projected_ddim_late_steps=int(args.projected_ddim_late_steps),
        model_path=args.model_path,
        weights=weights,
        limit=int(args.limit),
        sequence_batch_size=int(args.sequence_batch_size),
        conditions=selected_conditions,
        timeline_seed=int(args.timeline_seed),
        diffusion_noise_mode=str(args.diffusion_noise_mode),
        diffusion_noise_rho=float(args.diffusion_noise_rho),
        render_mp4=bool(args.render_mp4),
        render_fps=int(args.render_fps),
        render_stride=int(args.render_stride),
        render_camera_mode=str(args.render_camera_mode),
        render_layout=str(args.render_layout),
        render_local_radius=float(args.render_local_radius),
        latency_warmup_frames=int(args.latency_warmup_frames),
        runtime_metadata={
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
            "batch_size": int(args.sequence_batch_size),
            "sequence_batch_size": int(args.sequence_batch_size),
            "training_diffusion_steps": int(args.diffusion_steps),
            "inference_steps": int(diffusion.num_timesteps),
            "timestep_map": timestep_map,
            "use_ema": bool(args.use_ema),
            "diffusion_seed": int(args.seed),
            "diffusion_noise_mode": str(args.diffusion_noise_mode),
            "diffusion_noise_rho": float(args.diffusion_noise_rho),
            "projected_ddim_mode": str(args.projected_ddim_mode),
            "projected_ddim_late_steps": int(args.projected_ddim_late_steps),
            "latency_warmup_frames": int(args.latency_warmup_frames),
            "history_initialization": "cold_start_zero_padding",
        },
        show_progress=bool(args.show_progress),
    )
    print(f"[evaluate_longseq_eval_set] wrote {summary['summary_path']}")
    return summary


if __name__ == "__main__":
    main()
