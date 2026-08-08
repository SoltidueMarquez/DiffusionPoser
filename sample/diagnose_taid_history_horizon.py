from __future__ import annotations

import argparse
import hashlib
import json
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
from data_loaders.realtime_pose_geometry import extract_rotation_heading_np
from data_loaders.realtime_pose_kinematics import rotation_6d_to_matrix_np
from data_loaders.sensor_masking import (
    BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY,
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_TARGET_DIM,
    TRACKER_PATTERN_CATEGORIES,
)
from data_loaders.tracker_timeline import (
    TrackerTimeline,
    build_isolated_condition_timeline,
    isolated_condition_eval_mask,
    stable_context_seed,
)
from eval.evaluate_realtime_pose import evaluate_file, public_result, summarize
from eval.evaluate_realtime_pose_rollout import evaluate_rollout_file, summarize_rollouts
from sample.evaluate_longseq_eval_set import (
    CONDITION_OUTPUT_TAGS,
    _append_rollout_frame,
    _finalize_rollout_values,
    _new_rollout_values,
)
from sample.realtime_pose_runtime import (
    RealtimePoseRuntime,
    WorldPoseState,
    step_realtime_pose_batch,
)
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


TEACHER_FORCED_PROTOCOL = "teacher_forced"
CLOSED_LOOP_PROTOCOL = "closed_loop"
DIAGNOSTIC_PROTOCOLS = (TEACHER_FORCED_PROTOCOL, CLOSED_LOOP_PROTOCOL)
DEFAULT_HORIZONS = (1, 4, 15, 30, 60)
PAIR_MATCH_TOLERANCE = 1e-6


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用 GT 60 帧历史和多 horizon 闭环定位 TAID B1 的 history exposure bias。"
    )
    add_base_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sampling_options(parser)

    diagnostic = parser.add_argument_group("history_horizon_diagnostic")
    diagnostic.add_argument("--eval_root", default=DEFAULT_LONGSEQ_EVAL_ROOT, type=str)
    diagnostic.add_argument("--eval_set", default="latest", type=str)
    diagnostic.add_argument(
        "--normalizer_dir",
        default="dataset/meta_AMASS_realtime_pose_144d_pelvis_residual_root_y0_stationary5_60hz",
        type=str,
    )
    diagnostic.add_argument("--normalize_input", default=True, type=str2bool)
    diagnostic.add_argument("--input_feats", default=REALTIME_POSE_TARGET_DIM, type=int)
    diagnostic.add_argument("--limit", default=0, type=int)
    diagnostic.add_argument("--sequence_batch_size", default=2, type=int)
    diagnostic.add_argument(
        "--conditions",
        nargs="+",
        choices=TRACKER_PATTERN_CATEGORIES,
        default=["fixed_three", "fixed_six"],
    )
    diagnostic.add_argument("--timeline_seed", default=10, type=int)
    diagnostic.add_argument("--inference_steps", default=5, type=int)
    diagnostic.add_argument("--history_length", default=REALTIME_POSE_HISTORY_LENGTH, type=int)
    diagnostic.add_argument("--horizons", nargs="+", default=list(DEFAULT_HORIZONS), type=int)
    diagnostic.add_argument("--require_cuda", default=True, action=BooleanOptionalAction)
    diagnostic.add_argument("--show_progress", default=True, action=BooleanOptionalAction)
    return parser


def validate_diagnostic_options(
    history_length: int,
    horizons: list[int] | tuple[int, ...],
    sequence_batch_size: int,
) -> tuple[int, ...]:
    if int(history_length) != REALTIME_POSE_HISTORY_LENGTH:
        raise ValueError(
            f"第一版 history_length 只允许 {REALTIME_POSE_HISTORY_LENGTH}，"
            f"实际为 {history_length}。"
        )
    values = tuple(int(value) for value in horizons)
    if not values:
        raise ValueError("horizons 不能为空。")
    if values != tuple(sorted(set(values))):
        raise ValueError("horizons 必须严格升序且不能重复。")
    if values[0] < 1 or values[-1] > REALTIME_POSE_HISTORY_LENGTH:
        raise ValueError(
            f"horizons 必须位于 [1,{REALTIME_POSE_HISTORY_LENGTH}]。"
        )
    if int(sequence_batch_size) <= 0:
        raise ValueError("sequence_batch_size 必须大于 0。")
    return values


def build_ground_truth_world_pose_states(
    source: dict[str, np.ndarray],
) -> tuple[WorldPoseState, ...]:
    """从 source 构造诊断专用 GT world state，不读取 Actor Root yaw 分解量。"""

    rotations_world = compute_source_joint_rotations_world(source)
    pelvis_heading = extract_rotation_heading_np(rotations_world[:, 0])
    frame_count = int(rotations_world.shape[0])
    return tuple(
        WorldPoseState(
            joint_rotations_world=rotations_world[index].astype(np.float32, copy=True),
            root_yaw_world=float(pelvis_heading[index]),
            hip_height=float(source["pelvis_height"][index, 0]),
            root_position_world=source["root_pos_world"][index].astype(np.float32, copy=True),
        )
        for index in range(frame_count)
    )


def build_protocol_frame_metadata(
    frame_count: int,
    protocol: str,
    history_length: int = REALTIME_POSE_HISTORY_LENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """返回协议 eval mask、horizon、GT reset mask、完整块数和尾帧数。"""

    if protocol not in DIAGNOSTIC_PROTOCOLS:
        raise ValueError(f"未知 history protocol：{protocol}")
    if int(history_length) != REALTIME_POSE_HISTORY_LENGTH:
        raise ValueError("history_length 必须保持 60。")
    indices = np.arange(int(frame_count), dtype=np.int64)
    after_history = indices >= int(history_length)
    horizon = np.zeros(int(frame_count), dtype=np.int16)
    reset = np.zeros(int(frame_count), dtype=bool)
    if protocol == TEACHER_FORCED_PROTOCOL:
        eval_mask = after_history
        reset[after_history] = True
        return eval_mask, horizon, reset, 0, 0

    remaining = max(0, int(frame_count) - int(history_length))
    complete_blocks = remaining // int(history_length)
    evaluated_end = int(history_length) + complete_blocks * int(history_length)
    eval_mask = after_history & (indices < evaluated_end)
    horizon[eval_mask] = (
        (indices[eval_mask] - int(history_length)) % int(history_length) + 1
    ).astype(np.int16)
    reset_indices = indices[eval_mask & (horizon == 1)]
    reset[reset_indices] = True
    excluded_tail = remaining - complete_blocks * int(history_length)
    return eval_mask, horizon, reset, complete_blocks, excluded_tail


def rollout_history_protocol_sources(
    model,
    diffusion,
    sources: list[dict[str, np.ndarray]],
    timelines: list[TrackerTimeline],
    protocol: str,
    device: torch.device,
    normalizer: RealtimePoseNormalizer | None,
    projected_ddim_mode: str = "all_steps",
    projected_ddim_late_steps: int = 5,
    history_length: int = REALTIME_POSE_HISTORY_LENGTH,
    show_progress: bool = False,
    progress_desc: str = "",
    diffusion_seeds: list[int] | None = None,
) -> list[dict[str, np.ndarray]]:
    """批量执行 teacher-forced 或定期 GT reset 的 60 帧闭环诊断。"""

    if protocol not in DIAGNOSTIC_PROTOCOLS:
        raise ValueError(f"未知 history protocol：{protocol}")
    if not sources or len(sources) != len(timelines):
        raise ValueError("sources 与 timelines 必须非空且数量相同。")
    if diffusion_seeds is not None and len(diffusion_seeds) != len(sources):
        raise ValueError("diffusion_seeds 必须与 sources 数量相同。")
    frame_counts = [
        int(source[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY].shape[0]) for source in sources
    ]
    if any(frame_count <= int(history_length) for frame_count in frame_counts):
        raise ValueError("history horizon 诊断要求每条 source 至少超过 60 帧。")
    if any(
        len(timeline.configured) != frame_count
        for timeline, frame_count in zip(timelines, frame_counts)
    ):
        raise ValueError("timeline 帧数必须与对应 source 一致。")

    joint_rotations = [compute_source_joint_rotations_world(source) for source in sources]
    gt_states = [build_ground_truth_world_pose_states(source) for source in sources]
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
    generators = None
    if diffusion_seeds is not None:
        generators = []
        for seed in diffusion_seeds:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            generators.append(generator)

    progress = None
    if show_progress:
        progress = tqdm(
            total=sum(frame_counts),
            desc=progress_desc or protocol,
            unit="frame",
            dynamic_ncols=True,
        )
    for frame_index in range(max(frame_counts)):
        active_indices = [
            index for index, frame_count in enumerate(frame_counts) if frame_index < frame_count
        ]
        for sequence_index in active_indices:
            should_reset = frame_index >= int(history_length) and (
                protocol == TEACHER_FORCED_PROTOCOL
                or (
                    (frame_index - int(history_length)) % int(history_length) == 0
                    and frame_index + int(history_length) <= frame_counts[sequence_index]
                )
            )
            if should_reset:
                runtimes[sequence_index].replace_pose_history_for_diagnostic(
                    gt_states[sequence_index][frame_index - int(history_length) : frame_index]
                )
        active_runtimes = [runtimes[index] for index in active_indices]
        history_lengths = [len(runtime.pose_history) for runtime in active_runtimes]
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
            noise=(
                None
                if generators is None
                else torch.stack(
                    [
                        torch.randn(
                            REALTIME_POSE_TARGET_DIM,
                            generator=generators[index],
                            device=device,
                        )
                        for index in active_indices
                    ]
                )
            ),
        )
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
                sampling_latency_ms=float("nan"),
                e2e_latency_ms=float("nan"),
            )
        if progress is not None:
            progress.update(len(active_indices))
    if progress is not None:
        progress.close()

    payloads = []
    for sequence_values, frame_count in zip(values, frame_counts):
        payload = _finalize_rollout_values(sequence_values, frame_count)
        eval_mask, horizon, reset, complete_blocks, excluded_tail = build_protocol_frame_metadata(
            frame_count,
            protocol,
            history_length=history_length,
        )
        payload["eval_frame_mask"] = eval_mask[None]
        payload["diagnostic_horizon_frame"] = horizon[None]
        payload["gt_pose_history_reset"] = reset[None]
        payload["history_protocol"] = np.asarray(protocol)
        payload["complete_block_count"] = np.asarray(complete_blocks, dtype=np.int64)
        payload["excluded_tail_frames"] = np.asarray(excluded_tail, dtype=np.int64)
        payloads.append(payload)
    return payloads


def _append_curve_samples(
    accumulator: dict[int, dict[str, list[Any]]],
    payload: dict[str, np.ndarray],
) -> None:
    horizon = np.asarray(payload["diagnostic_horizon_frame"], dtype=np.int64).reshape(-1)
    eval_mask = np.asarray(payload["eval_frame_mask"], dtype=bool).reshape(-1)
    reference_joints = np.asarray(payload["reference_joints_world"], dtype=np.float64)[0]
    predicted_joints = np.asarray(payload["predicted_joints_world"], dtype=np.float64)[0]
    mpjpe_cm = np.linalg.norm(
        predicted_joints[:, :22] - reference_joints[:, :22], axis=-1
    ).mean(axis=-1) * 100.0
    reference_local = rotation_6d_to_matrix_np(
        np.asarray(payload["reference_body_local_delta_6d"], dtype=np.float64)[0].reshape(-1, 24, 6)
    )
    predicted_local = rotation_6d_to_matrix_np(
        np.asarray(payload["predicted_body_local_delta_6d"], dtype=np.float64)[0].reshape(-1, 24, 6)
    )
    relative = np.swapaxes(predicted_local[:, 1:22], -1, -2) @ reference_local[:, 1:22]
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    mpjre_deg = np.degrees(np.arccos(cosine)).mean(axis=-1)
    reference_root = np.asarray(payload["reference_root_position_world"], dtype=np.float64)[0]
    predicted_root = np.asarray(payload["predicted_root_position_world"], dtype=np.float64)[0]
    root_xz_m = np.linalg.norm(
        predicted_root[:, [0, 2]] - reference_root[:, [0, 2]], axis=-1
    )
    root_yaw_ref = np.asarray(payload["reference_root_yaw_world"], dtype=np.float64).reshape(-1)
    root_yaw_pred = np.asarray(payload["predicted_root_yaw_world"], dtype=np.float64).reshape(-1)
    root_yaw_deg = np.degrees(
        np.abs(
            np.arctan2(
                np.sin(root_yaw_pred - root_yaw_ref),
                np.cos(root_yaw_pred - root_yaw_ref),
            )
        )
    )
    for value in range(1, REALTIME_POSE_HISTORY_LENGTH + 1):
        mask = eval_mask & (horizon == value)
        item = accumulator[value]
        item["mpjpe_cm"].append(mpjpe_cm[mask])
        item["mpjre_deg"].append(mpjre_deg[mask])
        item["root_xz_error_m"].append(root_xz_m[mask])
        item["root_yaw_error_deg"].append(root_yaw_deg[mask])
        selected_yaw = root_yaw_deg[mask]
        item["pi_majority"].append(
            bool(selected_yaw.size and np.mean(selected_yaw > 150.0) > 0.5)
        )


def _new_curve_accumulator() -> dict[int, dict[str, list[Any]]]:
    return {
        horizon: {
            "mpjpe_cm": [],
            "mpjre_deg": [],
            "root_xz_error_m": [],
            "root_yaw_error_deg": [],
            "pi_majority": [],
        }
        for horizon in range(1, REALTIME_POSE_HISTORY_LENGTH + 1)
    }


def _finite_concat(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.empty(0, dtype=np.float64)
    result = np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1) for value in values])
    return result[np.isfinite(result)]


def summarize_horizon_curve(
    accumulator: dict[int, dict[str, list[Any]]],
) -> dict[str, dict[str, float | int | None | dict[str, float | int | None]]]:
    curve = {}
    for horizon, item in accumulator.items():
        mpjpe = _finite_concat(item["mpjpe_cm"])
        mpjre = _finite_concat(item["mpjre_deg"])
        root_xz = _finite_concat(item["root_xz_error_m"])
        root_yaw = _finite_concat(item["root_yaw_error_deg"])
        curve[str(horizon)] = {
            "samples": int(mpjpe.size),
            "mpjpe_cm": float(mpjpe.mean()) if mpjpe.size else None,
            "mpjre_deg": float(mpjre.mean()) if mpjre.size else None,
            "root_xz_error_m": float(root_xz.mean()) if root_xz.size else None,
            "root_yaw_diagnostics": {
                "samples": int(root_yaw.size),
                "median_deg": float(np.median(root_yaw)) if root_yaw.size else None,
                "p90_deg": float(np.percentile(root_yaw, 90)) if root_yaw.size else None,
                "p95_deg": float(np.percentile(root_yaw, 95)) if root_yaw.size else None,
                "error_over_90_ratio": float(np.mean(root_yaw > 90.0)) if root_yaw.size else None,
                "error_over_150_ratio": float(np.mean(root_yaw > 150.0)) if root_yaw.size else None,
                "pi_majority_sequence_count": int(sum(bool(value) for value in item["pi_majority"])),
                "sequence_count": int(len(item["pi_majority"])),
            },
        }
    return curve


def compare_teacher_forced_h1(
    teacher_payload: dict[str, np.ndarray],
    closed_payload: dict[str, np.ndarray],
) -> dict[str, float | int | bool | None]:
    horizon = np.asarray(closed_payload["diagnostic_horizon_frame"], dtype=np.int64)
    mask = (
        np.asarray(teacher_payload["eval_frame_mask"], dtype=bool)
        & np.asarray(closed_payload["eval_frame_mask"], dtype=bool)
        & (horizon == 1)
    )
    frame_mask = mask.reshape(-1)
    if not frame_mask.any():
        return {
            "samples": 0,
            "deployed_pose_max_abs": None,
            "joint_max_gap_m": None,
            "root_max_gap_m": None,
            "matches": False,
        }
    teacher_pose = np.asarray(teacher_payload["deployed_pred_target_raw"], dtype=np.float64)[0]
    closed_pose = np.asarray(closed_payload["deployed_pred_target_raw"], dtype=np.float64)[0]
    teacher_joints = np.asarray(teacher_payload["predicted_joints_world"], dtype=np.float64)[0]
    closed_joints = np.asarray(closed_payload["predicted_joints_world"], dtype=np.float64)[0]
    teacher_root = np.asarray(teacher_payload["predicted_root_position_world"], dtype=np.float64)[0]
    closed_root = np.asarray(closed_payload["predicted_root_position_world"], dtype=np.float64)[0]
    pose_gap = float(np.max(np.abs(teacher_pose[frame_mask] - closed_pose[frame_mask])))
    joint_gap = float(
        np.max(
            np.linalg.norm(
                teacher_joints[frame_mask] - closed_joints[frame_mask], axis=-1
            )
        )
    )
    root_gap = float(
        np.max(
            np.linalg.norm(teacher_root[frame_mask] - closed_root[frame_mask], axis=-1)
        )
    )
    return {
        "samples": int(frame_mask.sum()),
        "deployed_pose_max_abs": pose_gap,
        "joint_max_gap_m": joint_gap,
        "root_max_gap_m": root_gap,
        "matches": bool(max(pose_gap, joint_gap, root_gap) <= PAIR_MATCH_TOLERANCE),
    }


def summarize_pair_checks(
    checks: list[dict[str, float | int | bool | None]],
) -> dict[str, float | int | bool | None]:
    def maximum(name: str) -> float | None:
        values = [float(item[name]) for item in checks if item.get(name) is not None]
        return max(values) if values else None

    return {
        "samples": sum(int(item.get("samples", 0)) for item in checks),
        "deployed_pose_max_abs": maximum("deployed_pose_max_abs"),
        "joint_max_gap_m": maximum("joint_max_gap_m"),
        "root_max_gap_m": maximum("root_max_gap_m"),
        "tolerance": PAIR_MATCH_TOLERANCE,
        "matches": bool(checks and all(bool(item.get("matches")) for item in checks)),
    }


def decide_next_branch(
    teacher_summary: dict[str, Any],
    endpoint_summary: dict[str, dict[str, Any]],
    pair_summary: dict[str, Any],
) -> dict[str, Any]:
    thresholds = {
        "teacher_fixed_six_mpjpe_cm_max": 10.0,
        "closed_fixed_six_h1_mpjpe_cm_max": 10.0,
        "hard_tracker_rotation_mean_deg_max": 1e-5,
        "teacher_fixed_three_error_over_150_ratio_max": 0.3389,
        "teacher_fixed_three_pi_majority_sequence_count_max": 0,
        "h15_relative_to_h1_min": 1.5,
        "h30_h60_mean_relative_to_h15_min": 1.10,
    }
    by_condition = teacher_summary.get("by_condition", {})
    fixed_six = by_condition.get("fixed_six")
    fixed_three = by_condition.get("fixed_three")
    fixed_six_endpoints = endpoint_summary.get("fixed_six", {})
    required_endpoints = [fixed_six_endpoints.get(str(value)) for value in (1, 15, 30, 60)]
    if fixed_six is None or fixed_three is None or any(value is None for value in required_endpoints):
        return {
            "branch": "insufficient_conditions_or_horizons",
            "eligible_for_15_frame_experiment": False,
            "thresholds": thresholds,
            "gates": {},
        }

    h1, h15, h30, h60 = required_endpoints
    yaw = fixed_three["root_yaw_diagnostics"]
    teacher_six_pass = float(fixed_six["mpjpe_cm"]) <= thresholds[
        "teacher_fixed_six_mpjpe_cm_max"
    ]
    h1_pass = float(h1["mpjpe_cm"]) <= thresholds["closed_fixed_six_h1_mpjpe_cm_max"]
    hard_pass = float(fixed_six["deployed_hard_tracker_rotation_deg"]) <= thresholds[
        "hard_tracker_rotation_mean_deg_max"
    ]
    teacher_three_pass = (
        float(yaw["error_over_150_ratio"])
        <= thresholds["teacher_fixed_three_error_over_150_ratio_max"]
        and int(yaw["pi_majority_sequence_count"])
        <= thresholds["teacher_fixed_three_pi_majority_sequence_count_max"]
    )
    h1_value = float(h1["mpjpe_cm"])
    h15_value = float(h15["mpjpe_cm"])
    h30_value = float(h30["mpjpe_cm"])
    h60_value = float(h60["mpjpe_cm"])
    h15_degraded = h15_value > 10.0 or h15_value >= 1.5 * h1_value
    continued_degradation = (h30_value + h60_value) * 0.5 >= 1.10 * h15_value
    pair_pass = bool(pair_summary.get("matches"))
    exposure_confirmed = h15_degraded and continued_degradation
    eligible = all(
        (
            teacher_six_pass,
            h1_pass,
            hard_pass,
            teacher_three_pass,
            pair_pass,
            exposure_confirmed,
        )
    )
    if not pair_pass:
        branch = "diagnostic_contract_mismatch"
    elif not teacher_six_pass:
        branch = "audit_prior_supervision_capacity"
    elif not teacher_three_pass:
        branch = "audit_fixed_three_prior_yaw"
    elif eligible:
        branch = "plan_15_frame_rollout_experiment"
    else:
        branch = "retain_diagnostic_no_training_change"
    return {
        "branch": branch,
        "eligible_for_15_frame_experiment": bool(eligible),
        "exposure_bias_confirmed": bool(exposure_confirmed),
        "thresholds": thresholds,
        "gates": {
            "teacher_fixed_six_capacity": teacher_six_pass,
            "closed_fixed_six_h1": h1_pass,
            "hard_tracker_rotation": hard_pass,
            "teacher_fixed_three_pi_mode": teacher_three_pass,
            "teacher_vs_h1_pair_match": pair_pass,
            "h15_degraded": h15_degraded,
            "h30_h60_continue_degrading": continued_degradation,
        },
        "values": {
            "teacher_fixed_six_mpjpe_cm": float(fixed_six["mpjpe_cm"]),
            "closed_fixed_six_h1_mpjpe_cm": h1_value,
            "closed_fixed_six_h15_mpjpe_cm": h15_value,
            "closed_fixed_six_h30_mpjpe_cm": h30_value,
            "closed_fixed_six_h60_mpjpe_cm": h60_value,
            "teacher_fixed_six_hard_rotation_mean_deg": float(
                fixed_six["deployed_hard_tracker_rotation_deg"]
            ),
            "teacher_fixed_three_error_over_150_ratio": float(
                yaw["error_over_150_ratio"]
            ),
            "teacher_fixed_three_pi_majority_sequence_count": int(
                yaw["pi_majority_sequence_count"]
            ),
        },
    }


def evaluate_history_diagnostic_entries(
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
    sequence_batch_size: int = 2,
    conditions: list[str] | tuple[str, ...] = ("fixed_three", "fixed_six"),
    timeline_seed: int = 10,
    horizons: list[int] | tuple[int, ...] = DEFAULT_HORIZONS,
    history_length: int = REALTIME_POSE_HISTORY_LENGTH,
    runtime_metadata: dict[str, Any] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    eval_set_dir = Path(eval_set_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_entries = entries[: int(limit)] if int(limit) > 0 else entries
    if not selected_entries:
        raise RuntimeError("history horizon 诊断集合为空。")
    selected_conditions = tuple(dict.fromkeys(str(value) for value in conditions))
    if not selected_conditions or any(
        condition not in TRACKER_PATTERN_CATEGORIES for condition in selected_conditions
    ):
        raise ValueError(f"条件必须来自 {TRACKER_PATTERN_CATEGORIES}。")
    selected_horizons = validate_diagnostic_options(
        history_length,
        horizons,
        sequence_batch_size,
    )
    jobs = [
        (entry, condition)
        for condition in selected_conditions
        for entry in selected_entries
    ]

    protocol_results: dict[str, list[dict[str, Any]]] = {
        protocol: [] for protocol in DIAGNOSTIC_PROTOCOLS
    }
    endpoint_results: dict[str, dict[int, list[dict[str, Any]]]] = {
        condition: {horizon: [] for horizon in selected_horizons}
        for condition in selected_conditions
    }
    curve_accumulators = {
        condition: _new_curve_accumulator() for condition in selected_conditions
    }
    pair_checks: dict[str, list[dict[str, float | int | bool | None]]] = {
        condition: [] for condition in selected_conditions
    }
    batch_size = max(1, int(sequence_batch_size))
    for batch_start in range(0, len(jobs), batch_size):
        batch_jobs = jobs[batch_start : batch_start + batch_size]
        batch_sources: list[dict[str, np.ndarray]] = []
        batch_timelines: list[TrackerTimeline] = []
        batch_condition_masks: list[np.ndarray] = []
        batch_diffusion_seeds: list[int] = []
        for entry, condition in batch_jobs:
            sequence_id = str(entry["sequence_id"])
            source = load_realtime_source(
                resolve_manifest_source_path(eval_set_dir=eval_set_dir, entry=entry)
            )
            frame_count = int(source[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY].shape[0])
            if frame_count < int(history_length) * 2:
                raise ValueError(
                    f"{condition}/{sequence_id} 至少需要 {int(history_length) * 2} 帧，"
                    f"实际为 {frame_count}。"
                )
            timeline = build_isolated_condition_timeline(
                source_id=sequence_id,
                frame_count=frame_count,
                condition=condition,
                global_seed=int(timeline_seed),
            )
            batch_sources.append(source)
            batch_timelines.append(timeline)
            batch_condition_masks.append(isolated_condition_eval_mask(timeline, condition))
            batch_diffusion_seeds.append(
                int(
                    stable_context_seed(
                        timeline_seed,
                        sequence_id,
                        "taid_history_horizon_diffusion",
                    )
                    % (2**63)
                )
            )
            print(f"[history-diagnostic] {condition}/{sequence_id}: {frame_count} frames")

        protocol_payloads = {
            protocol: rollout_history_protocol_sources(
                model=model,
                diffusion=diffusion,
                sources=batch_sources,
                timelines=batch_timelines,
                protocol=protocol,
                device=device,
                normalizer=normalizer,
                projected_ddim_mode=projected_ddim_mode,
                projected_ddim_late_steps=projected_ddim_late_steps,
                history_length=history_length,
                show_progress=bool(show_progress),
                progress_desc=(
                    f"{protocol} {batch_start + 1}-{batch_start + len(batch_jobs)}/"
                    f"{len(jobs)}"
                ),
                diffusion_seeds=batch_diffusion_seeds,
            )
            for protocol in DIAGNOSTIC_PROTOCOLS
        }

        for item_index, ((entry, condition), condition_mask) in enumerate(
            zip(batch_jobs, batch_condition_masks)
        ):
            sequence_id = str(entry["sequence_id"])
            teacher_payload = protocol_payloads[TEACHER_FORCED_PROTOCOL][item_index]
            closed_payload = protocol_payloads[CLOSED_LOOP_PROTOCOL][item_index]
            for payload in (teacher_payload, closed_payload):
                payload["eval_frame_mask"] &= condition_mask[None]
            pair_check = compare_teacher_forced_h1(teacher_payload, closed_payload)
            pair_check.update(sequence_id=sequence_id, condition=condition)
            pair_checks[condition].append(pair_check)

            for protocol, payload in (
                (TEACHER_FORCED_PROTOCOL, teacher_payload),
                (CLOSED_LOOP_PROTOCOL, closed_payload),
            ):
                sequence_dir = (
                    output_dir
                    / protocol
                    / CONDITION_OUTPUT_TAGS[condition]
                    / build_sequence_output_dir_name(entry)
                )
                sequence_dir.mkdir(parents=True, exist_ok=True)
                result_path = sequence_dir / "diagnostic_result.npz"
                np.savez_compressed(result_path, **payload)
                result = evaluate_rollout_file(result_path)
                result.update(
                    {
                        "sequence_id": sequence_id,
                        "condition": condition,
                        "history_protocol": protocol,
                        "source_relative_path": str(entry.get("source_relative_path", "")),
                        "num_frames": int(payload["reference_target_raw"].shape[1]),
                        "evaluated_frames": int(payload["eval_frame_mask"].sum()),
                        "complete_block_count": int(payload["complete_block_count"]),
                        "excluded_tail_frames": int(payload["excluded_tail_frames"]),
                        "result_path": str(result_path),
                        "result_sha256": sha256_file(result_path),
                    }
                )
                protocol_results[protocol].append(result)
                if protocol == CLOSED_LOOP_PROTOCOL:
                    _append_curve_samples(curve_accumulators[condition], payload)
                    for horizon in selected_horizons:
                        endpoint = evaluate_file(
                            result_path,
                            eval_frame_mask_override=(
                                payload["diagnostic_horizon_frame"] == int(horizon)
                            ),
                        )
                        endpoint.update(
                            sequence_id=sequence_id,
                            condition=condition,
                            horizon_frame=int(horizon),
                        )
                        endpoint_results[condition][int(horizon)].append(endpoint)

    teacher_summary = summarize_rollouts(protocol_results[TEACHER_FORCED_PROTOCOL])
    teacher_summary["by_condition"] = {
        condition: summarize_rollouts(
            [
                result
                for result in protocol_results[TEACHER_FORCED_PROTOCOL]
                if result["condition"] == condition
            ]
        )
        for condition in selected_conditions
    }
    closed_summary = summarize_rollouts(protocol_results[CLOSED_LOOP_PROTOCOL])
    closed_summary["by_condition"] = {
        condition: summarize_rollouts(
            [
                result
                for result in protocol_results[CLOSED_LOOP_PROTOCOL]
                if result["condition"] == condition
            ]
        )
        for condition in selected_conditions
    }
    endpoint_summary = {
        condition: {
            str(horizon): public_result(summarize(endpoint_results[condition][horizon]))
            for horizon in selected_horizons
        }
        for condition in selected_conditions
    }
    curve_summary = {
        condition: summarize_horizon_curve(curve_accumulators[condition])
        for condition in selected_conditions
    }
    pair_summary = {
        "by_condition": {
            condition: summarize_pair_checks(pair_checks[condition])
            for condition in selected_conditions
        },
        "overall": summarize_pair_checks(
            [item for condition in selected_conditions for item in pair_checks[condition]]
        ),
        "files": [
            item for condition in selected_conditions for item in pair_checks[condition]
        ],
    }
    decision = decide_next_branch(
        teacher_summary,
        endpoint_summary,
        pair_summary["overall"],
    )

    model_file = Path(model_path).resolve() if str(model_path).strip() else None
    metadata = {
        "kind": "taid_b1_teacher_forced_history_horizon_diagnostic",
        "evaluation_protocol": "offline_gt_pose_history_diagnostic_only",
        "eval_set_dir": str(eval_set_dir),
        "eval_manifest_sha256": sha256_file(eval_set_dir / "manifest.jsonl"),
        "output_dir": str(output_dir),
        "model_path": "" if model_file is None else str(model_file),
        "checkpoint_sha256": (
            sha256_file(model_file) if model_file is not None and model_file.is_file() else None
        ),
        "weights": str(weights),
        "timeline_seed": int(timeline_seed),
        "history_length": int(history_length),
        "horizons": [int(value) for value in selected_horizons],
        "conditions": list(selected_conditions),
        "source_sequence_count": len(selected_entries),
        "sequence_batch_size": int(batch_size),
        "shared_diffusion_noise_across_protocols_and_conditions": True,
        "gt_pose_history_is_diagnostic_only": True,
        **dict(runtime_metadata or {}),
    }
    if str(weights) == "ema" and model_file is not None:
        ema_file = model_file.with_name(model_file.name.replace("model", "ema", 1))
        metadata["ema_path"] = str(ema_file)
        metadata["ema_sha256"] = sha256_file(ema_file) if ema_file.is_file() else None
    if normalizer is not None:
        metadata["normalizer_dir"] = str(normalizer.base_dir.resolve())
        metadata["normalizer_hashes"] = {
            path.name: sha256_file(path)
            for path in sorted(normalizer.base_dir.iterdir())
            if path.is_file()
        }

    curve_path = output_dir / "history_horizon_curve.json"
    curve_payload = {
        "curve": curve_summary,
        "endpoints": endpoint_summary,
        "metadata": metadata,
    }
    curve_path.write_text(
        json.dumps(curve_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_payload = {
        "summary": {
            "teacher_forced": public_result(teacher_summary),
            "closed_loop": public_result(closed_summary),
            "closed_loop_endpoints": endpoint_summary,
            "teacher_vs_closed_h1": pair_summary,
            "decision": decision,
        },
        "files": [
            public_result(result)
            for protocol in DIAGNOSTIC_PROTOCOLS
            for result in protocol_results[protocol]
        ],
        "metadata": {
            **metadata,
            "curve_path": str(curve_path),
            "curve_sha256": sha256_file(curve_path),
        },
    }
    summary_path = output_dir / "history_horizon_diagnostic_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_payload["summary_path"] = str(summary_path)
    return summary_payload


def build_default_output_dir(
    eval_set_dir: Path,
    model_path: str | Path,
    weights: str,
    history_length: int,
    horizons: list[int] | tuple[int, ...],
    conditions: list[str] | tuple[str, ...],
    sequence_batch_size: int,
) -> Path:
    model = Path(model_path)
    step_text = model.stem.removeprefix("model")
    step_tag = str(int(step_text)) if step_text.isdigit() else short_digest(model.stem, 6)
    weight_tag = "e" if str(weights) == "ema" else "m"
    identity = short_digest(
        "\n".join(
            (
                str(Path(eval_set_dir).resolve()),
                str(model.resolve().parent),
                ",".join(str(value) for value in horizons),
                ",".join(str(value) for value in conditions),
                str(sequence_batch_size),
            )
        ),
        10,
    )
    leaf = f"{step_tag}{weight_tag}-h{int(history_length)}-b{int(sequence_batch_size)}-{identity}"
    return Path("output") / "diagnostics" / "taid_history_horizon" / leaf


def short_digest(value: str, length: int) -> str:
    size = max(1, (int(length) + 1) // 2)
    return hashlib.blake2s(str(value).encode("utf-8"), digest_size=size).hexdigest()[:length]


def sha256_file(path: str | Path) -> str:
    file_path = Path(path).resolve()
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_and_load_from_model(
        build_arg_parser(),
        argv=argv,
        ignore_keys={"ts_respace"},
    )
    horizons = validate_diagnostic_options(
        args.history_length,
        args.horizons,
        args.sequence_batch_size,
    )
    if int(args.inference_steps) <= 0:
        raise ValueError("inference_steps 必须大于 0。")
    if int(args.projected_ddim_late_steps) <= 0:
        raise ValueError("projected_ddim_late_steps 必须大于 0。")
    selected_conditions = list(dict.fromkeys(str(value) for value in args.conditions))
    args.ts_respace = f"ddim{int(args.inference_steps)}"
    fixseed(int(args.seed))
    eval_set_dir = resolve_longseq_eval_dir(eval_root=args.eval_root, eval_set=args.eval_set)
    entries = read_longseq_manifest(eval_set_dir)
    normalizer = RealtimePoseNormalizer(args.normalizer_dir) if args.normalize_input else None

    if bool(args.require_cuda) and (not bool(args.cuda) or not torch.cuda.is_available()):
        raise RuntimeError("history horizon 真实诊断要求使用可用的 CUDA GPU。")
    dist_util.setup_dist(args.device if args.cuda else -1)
    device = dist_util.dev()
    if bool(args.require_cuda) and device.type != "cuda":
        raise RuntimeError(f"history horizon 诊断期望 CUDA，实际设备为 {device}。")
    model, diffusion = create_model_and_diffusion(args)
    if int(diffusion.num_timesteps) != int(args.inference_steps):
        raise RuntimeError(
            f"期望 {args.inference_steps} 个 DDIM 推理步，实际为 {diffusion.num_timesteps}。"
        )
    model, weights = load_checkpoint_model(
        model,
        args.model_path,
        device=device,
        use_ema=args.use_ema,
    )
    model.eval()
    timestep_map = [int(value) for value in diffusion.timestep_map]
    print(
        f"[history-diagnostic] device={device}, inference_steps={diffusion.num_timesteps}, "
        f"batch={args.sequence_batch_size}, conditions={selected_conditions}, "
        f"horizons={list(horizons)}, timestep_map={timestep_map}"
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if str(args.output_dir).strip()
        else build_default_output_dir(
            eval_set_dir,
            args.model_path,
            weights,
            history_length=int(args.history_length),
            horizons=horizons,
            conditions=selected_conditions,
            sequence_batch_size=int(args.sequence_batch_size),
        ).resolve()
    )
    summary = evaluate_history_diagnostic_entries(
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
        horizons=horizons,
        history_length=int(args.history_length),
        runtime_metadata={
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
            "training_diffusion_steps": int(args.diffusion_steps),
            "inference_steps": int(diffusion.num_timesteps),
            "timestep_map": timestep_map,
            "use_ema": bool(args.use_ema),
            "diffusion_seed": int(args.seed),
            "projected_ddim_mode": str(args.projected_ddim_mode),
            "projected_ddim_late_steps": int(args.projected_ddim_late_steps),
        },
        show_progress=bool(args.show_progress),
    )
    print(f"[diagnose_taid_history_horizon] wrote {summary['summary_path']}")
    return summary


if __name__ == "__main__":
    main()
