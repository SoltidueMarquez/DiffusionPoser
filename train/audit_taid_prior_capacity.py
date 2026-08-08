from __future__ import annotations

import hashlib
import json
import subprocess
from argparse import Namespace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from data_loaders.get_data import get_dataset_loader
from data_loaders.build_realtime_longseq_eval_set import (
    read_longseq_manifest,
    resolve_manifest_source_path,
)
from data_loaders.generate_realtime_pose_tasks import load_realtime_source
from data_loaders.realtime_pose_config import TARGET_JOINT_REGIONS
from data_loaders.realtime_pose_geometry import (
    decode_target_head_rotations_torch,
    resolve_root_head_reference_torch,
)
from data_loaders.realtime_pose_kinematics import SMPL_JOINT_NAMES, rotation_6d_to_matrix_np
from data_loaders.sensor_masking import (
    REALTIME_POSE_MAX_ROLLOUT_STEPS,
    TRACKER_CONFIGURED_OFFSET,
    TRACKER_COUNT,
    TRACKER_D_OFF_OFFSET,
    TRACKER_D_ON_OFFSET,
    TRACKER_MEASURED_VALID_OFFSET,
)
from diffusion.realtime_pose_projection import project_realtime_pose_xstart
from sample.diagnose_taid_history_horizon import sha256_file
from sample.realtime_pose_runtime import decode_and_resolve_pose
from sample.utils import load_checkpoint_model
from train.train_platforms import NoPlatform
from train.training_loop import TrainLoop, move_batch_to_device
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion
from utils.parser_util import build_train_arg_parser, parse_and_load_from_model, str2bool
from utils.run_dirs import resolve_latest_or_self


PRIOR_PREFIX = "taid_conditioner.prior."
REGION_NAMES = ("torso", "left_arm", "right_arm", "left_leg", "right_leg")
HORIZON_ENDPOINTS = (1, 4, 15, 30, 60)
GRADIENT_LOSS_KEYS = (
    "prior_rotation_loss",
    "prior_fk_loss",
    "prior_root_loss",
    "prior_velocity_loss",
    "prior_contact_loss",
    "rollout_loss",
    "rollout_joint_vel_loss",
    "rollout_rotation_vel_loss",
)
GRADIENT_MODULE_PREFIXES = {
    "pose_history_encoder_gru": (
        f"{PRIOR_PREFIX}history_frame_encoder.",
        f"{PRIOR_PREFIX}history_gru.",
    ),
    "current_tracker_fusion": (f"{PRIOR_PREFIX}tracker_fusion.",),
    "shared_fusion": (f"{PRIOR_PREFIX}fusion.",),
    "pose_head": (f"{PRIOR_PREFIX}pose_head.",),
    "root_xyz_head": (f"{PRIOR_PREFIX}root_head.",),
    "velocity_head": (f"{PRIOR_PREFIX}joint_velocity_head.",),
    "contact_head": (f"{PRIOR_PREFIX}contact_head.",),
}


def build_argument_parser():
    parser = build_train_arg_parser()
    parser.description = "只读审计 TAID B1 Prior 的输入敏感度、监督梯度与区域闭环误差。"
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--use_ema", default=True, type=str2bool)
    parser.add_argument("--history_diagnostic_dir", required=True, type=str)
    parser.add_argument(
        "--output_dir",
        default="output/diagnostics/taid_prior_capacity",
        type=str,
    )
    return parser


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _short_digest(*values: str, length: int = 12) -> str:
    digest = hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()
    return digest[: int(length)]


def _git_diff_sha256(root: Path) -> str | None:
    """把已跟踪和未跟踪源码路径都纳入只读代码状态指纹。"""

    try:
        diff = subprocess.run(
            ["git", "diff", "--binary", "--", "."],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256(diff + b"\n--untracked--\n" + untracked).hexdigest()


def _swap_batch(value: torch.Tensor) -> torch.Tensor:
    if value.shape[0] < 2:
        raise ValueError("输入敏感度审计要求 batch_size 至少为2。")
    return value.flip(0)


def _clone_model_kwargs(model_kwargs: dict[str, Any]) -> dict[str, Any]:
    copied = dict(model_kwargs)
    copied["y"] = dict(model_kwargs["y"])
    return copied


def _prepare_from_kwargs(model: torch.nn.Module, model_kwargs: dict[str, Any]):
    y = model_kwargs["y"]
    return model.prepare_conditioning(
        y["pose_history"],
        y["tracker_history"],
        y["current_tracker"],
        model_kwargs["trajectory_history"],
        y["current_trajectory"],
        y["valid_frame_mask"],
        current_tracker_raw=y["current_tracker_raw"],
        joint_offsets_parent=y["joint_offsets_parent"],
        pose_mean=y.get("normalizer_mean"),
        pose_std=y.get("normalizer_std"),
        tracker_mean=y.get("tracker_normalizer_mean"),
        tracker_std=y.get("tracker_normalizer_std"),
    )


def _deployed_prior_values(prepared, model_kwargs: dict[str, Any]) -> dict[str, torch.Tensor]:
    y = model_kwargs["y"]
    prior = prepared.taid.prior
    deployed_model = project_realtime_pose_xstart(
        prior.pose_model,
        y["current_tracker_raw"],
        y["hard_rotation_state"].bool(),
        y.get("normalizer_mean"),
        y.get("normalizer_std"),
    )
    deployed_raw = deployed_model
    if y.get("normalizer_mean") is not None and y.get("normalizer_std") is not None:
        deployed_raw = (
            deployed_model * y["normalizer_std"].to(deployed_model)
            + y["normalizer_mean"].to(deployed_model)
        )
    rotations, root_yaw = decode_target_head_rotations_torch(deployed_raw)
    root, _, joints = resolve_root_head_reference_torch(
        rotations,
        root_yaw,
        y["joint_offsets_parent"],
        y["current_tracker_raw"][:, 0, 1],
    )
    return {
        "pose_raw": prior.pose_raw,
        "deployed_joints": joints,
        "deployed_root": root,
        "root_xyz": prior.root_head[:, :3],
        "root_yaw": prior.root_head[:, 3],
        "contact": prior.contact_probability,
        "velocity": prior.joint_velocity_head,
        "alpha": prepared.taid.role_state.alpha,
    }


def _circular_max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    delta = torch.atan2(torch.sin(left - right), torch.cos(left - right))
    return float(delta.detach().abs().max().cpu().item())


def compare_prior_values(
    baseline: dict[str, torch.Tensor],
    perturbed: dict[str, torch.Tensor],
) -> dict[str, float]:
    def mean_abs(key: str) -> float:
        return float(
            (perturbed[key] - baseline[key]).detach().abs().mean().cpu().item()
        )

    def max_abs(key: str) -> float:
        return float(
            (perturbed[key] - baseline[key]).detach().abs().max().cpu().item()
        )

    return {
        "prior_pose_raw_mean_abs": mean_abs("pose_raw"),
        "prior_pose_raw_max_abs": max_abs("pose_raw"),
        "deployed_joint_mean_gap_m": float(
            torch.linalg.norm(
                perturbed["deployed_joints"] - baseline["deployed_joints"], dim=-1
            )
            .detach()
            .mean()
            .cpu()
            .item()
        ),
        "pose_root_yaw_max_abs_rad": _circular_max_abs(
            perturbed["root_yaw"], baseline["root_yaw"]
        ),
        "root_xyz_mean_abs_m": mean_abs("root_xyz"),
        "contact_mean_abs": mean_abs("contact"),
        "velocity_mean_abs_m_s": mean_abs("velocity"),
    }


def _zero_alpha_history_pair(
    model_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """构造同一 alpha=0 Tracker，仅历史值不同的严格泄漏对照。"""

    baseline = _clone_model_kwargs(model_kwargs)
    perturbed = _clone_model_kwargs(model_kwargs)
    tracker_index = TRACKER_COUNT - 1
    for values in (baseline, perturbed):
        current = values["y"]["current_tracker"].clone()
        current[:, tracker_index, TRACKER_CONFIGURED_OFFSET] = 0.0
        current[:, tracker_index, TRACKER_MEASURED_VALID_OFFSET] = 0.0
        current[:, tracker_index, TRACKER_D_OFF_OFFSET] = 0.0
        current[:, tracker_index, TRACKER_D_ON_OFFSET] = 0.0
        values["y"]["current_tracker"] = current
        values["current_tracker"] = current
        current_raw = values["y"]["current_tracker_raw"].clone()
        current_raw[:, tracker_index, TRACKER_CONFIGURED_OFFSET] = 0.0
        current_raw[:, tracker_index, TRACKER_MEASURED_VALID_OFFSET] = 0.0
        current_raw[:, tracker_index, TRACKER_D_OFF_OFFSET] = 0.0
        current_raw[:, tracker_index, TRACKER_D_ON_OFFSET] = 0.0
        values["y"]["current_tracker_raw"] = current_raw
    changed = perturbed["y"]["tracker_history"].clone()
    changed[:, :, tracker_index] = changed[:, :, tracker_index] + 17.0
    perturbed["y"]["tracker_history"] = changed
    perturbed["tracker_history"] = changed
    return baseline, perturbed, tracker_index


@torch.no_grad()
def audit_input_sensitivity(
    model: torch.nn.Module,
    model_kwargs: dict[str, Any],
) -> dict[str, Any]:
    baseline_prepared = _prepare_from_kwargs(model, model_kwargs)
    baseline = _deployed_prior_values(baseline_prepared, model_kwargs)
    results: dict[str, Any] = {}
    field_map = {
        "pose_history": ("y", "pose_history"),
        "tracker_history": ("y", "tracker_history"),
        "current_tracker": ("y", "current_tracker"),
        "current_trajectory": ("y", "current_trajectory"),
    }
    for name, (_, field) in field_map.items():
        changed = _clone_model_kwargs(model_kwargs)
        changed_value = _swap_batch(changed["y"][field])
        changed["y"][field] = changed_value
        if field in changed:
            changed[field] = changed_value
        prepared = _prepare_from_kwargs(model, changed)
        results[name] = compare_prior_values(
            baseline,
            _deployed_prior_values(prepared, changed),
        )

    zero_base_kwargs, zero_changed_kwargs, tracker_index = _zero_alpha_history_pair(
        model_kwargs
    )
    zero_base_prepared = _prepare_from_kwargs(model, zero_base_kwargs)
    zero_changed_prepared = _prepare_from_kwargs(model, zero_changed_kwargs)
    zero_base = _deployed_prior_values(zero_base_prepared, zero_base_kwargs)
    zero_changed = _deployed_prior_values(zero_changed_prepared, zero_changed_kwargs)
    zero_result = compare_prior_values(zero_base, zero_changed)
    zero_result.update(
        tracker_index=int(tracker_index),
        baseline_alpha_max=float(
            zero_base["alpha"][:, tracker_index].max().cpu().item()
        ),
    )
    results["alpha_zero_tracker_history"] = zero_result
    results["contract"] = {
        "tracker_history_strictly_unused": bool(
            results["tracker_history"]["prior_pose_raw_max_abs"] == 0.0
        ),
        "pose_history_is_used": bool(
            results["pose_history"]["prior_pose_raw_max_abs"] > 0.0
        ),
        "alpha_zero_history_no_leak": bool(
            zero_result["prior_pose_raw_max_abs"] == 0.0
            and zero_result["baseline_alpha_max"] == 0.0
        ),
    }
    return results


def _parameter_groups(
    model: torch.nn.Module,
) -> tuple[list[tuple[str, torch.nn.Parameter]], dict[str, list[int]]]:
    parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    groups: dict[str, list[int]] = {}
    for group_name, prefixes in GRADIENT_MODULE_PREFIXES.items():
        groups[group_name] = [
            index
            for index, (name, _) in enumerate(parameters)
            if any(name.startswith(prefix) for prefix in prefixes)
        ]
    return parameters, groups


def _group_gradient_summary(
    gradients: tuple[torch.Tensor | None, ...],
    indices: Iterable[int],
) -> dict[str, Any]:
    selected = [gradients[index] for index in indices if gradients[index] is not None]
    if not selected:
        return {"parameter_tensors": 0, "norm": 0.0, "finite": True, "nonzero": False}
    squared = sum(
        float(gradient.detach().float().square().sum().cpu().item())
        for gradient in selected
    )
    finite = all(bool(torch.isfinite(gradient).all().item()) for gradient in selected)
    return {
        "parameter_tensors": len(selected),
        "norm": float(squared**0.5),
        "finite": bool(finite),
        "nonzero": bool(squared > 0.0),
    }


def _shared_gradient_cosine(
    left: tuple[torch.Tensor | None, ...],
    right: tuple[torch.Tensor | None, ...],
    indices: Iterable[int],
) -> float | None:
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    for index in indices:
        left_value = left[index]
        right_value = right[index]
        if left_value is None or right_value is None:
            continue
        left_float = left_value.detach().float()
        right_float = right_value.detach().float()
        dot += float((left_float * right_float).sum().cpu().item())
        left_sq += float(left_float.square().sum().cpu().item())
        right_sq += float(right_float.square().sum().cpu().item())
    if left_sq <= 0.0 or right_sq <= 0.0:
        return None
    return float(dot / ((left_sq * right_sq) ** 0.5))


def audit_gradient_flow(
    model: torch.nn.Module,
    loop: TrainLoop,
    batch: dict[str, Any],
    timesteps: torch.Tensor,
) -> dict[str, Any]:
    model.train()
    losses = loop.compute_losses(batch=batch, timesteps=timesteps)
    parameters, group_indices = _parameter_groups(model)
    parameter_values = tuple(parameter for _, parameter in parameters)
    gradients_by_loss: dict[str, tuple[torch.Tensor | None, ...]] = {}
    report: dict[str, Any] = {}
    for loss_key in GRADIENT_LOSS_KEYS:
        if loss_key not in losses:
            report[loss_key] = {"available": False}
            continue
        scalar = losses[loss_key].mean()
        gradients = torch.autograd.grad(
            scalar,
            parameter_values,
            retain_graph=True,
            allow_unused=True,
        )
        gradients_by_loss[loss_key] = gradients
        report[loss_key] = {
            "available": True,
            "value": float(scalar.detach().float().cpu().item()),
            "modules": {
                group_name: _group_gradient_summary(gradients, indices)
                for group_name, indices in group_indices.items()
            },
        }

    shared_indices = sorted(
        set(
            group_indices["pose_history_encoder_gru"]
            + group_indices["current_tracker_fusion"]
            + group_indices["shared_fusion"]
        )
    )
    cosine: dict[str, float | None] = {}
    available_keys = list(gradients_by_loss)
    for left_index, left_key in enumerate(available_keys):
        for right_key in available_keys[left_index + 1 :]:
            cosine[f"{left_key}__{right_key}"] = _shared_gradient_cosine(
                gradients_by_loss[left_key],
                gradients_by_loss[right_key],
                shared_indices,
            )

    trainable_outside_prior = [
        name for name, _ in parameters if not name.startswith(PRIOR_PREFIX)
    ]
    checks = {
        "prior_velocity_reaches_velocity_head": bool(
            report.get("prior_velocity_loss", {})
            .get("modules", {})
            .get("velocity_head", {})
            .get("nonzero", False)
        ),
        "prior_velocity_reaches_shared_latent": bool(
            report.get("prior_velocity_loss", {})
            .get("modules", {})
            .get("shared_fusion", {})
            .get("nonzero", False)
        ),
        "prior_velocity_does_not_reach_pose_head": not bool(
            report.get("prior_velocity_loss", {})
            .get("modules", {})
            .get("pose_head", {})
            .get("nonzero", False)
        ),
        "rollout_joint_velocity_reaches_pose_head": bool(
            report.get("rollout_joint_vel_loss", {})
            .get("modules", {})
            .get("pose_head", {})
            .get("nonzero", False)
        ),
        "rollout_rotation_velocity_reaches_pose_head": bool(
            report.get("rollout_rotation_vel_loss", {})
            .get("modules", {})
            .get("pose_head", {})
            .get("nonzero", False)
        ),
        "b1_prior_only_trainable": not trainable_outside_prior and bool(parameters),
        "observation_encoder_frozen": not any(
            parameter.requires_grad
            for parameter in model.observation_encoder.parameters()
        ),
    }
    model.eval()
    return {
        "losses": report,
        "shared_prior_gradient_cosine": cosine,
        "checks": checks,
        "trainable_parameter_tensors": len(parameters),
        "trainable_outside_prior": trainable_outside_prior,
    }


def _masked_finite(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = np.asarray(value)[mask]
    return selected[np.isfinite(selected)]


def _mean_or_none(value: np.ndarray) -> float | None:
    finite = np.asarray(value, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else None


def _rotation_error_deg_per_joint(
    reference_6d: np.ndarray,
    predicted_6d: np.ndarray,
) -> np.ndarray:
    reference = rotation_6d_to_matrix_np(reference_6d.reshape(-1, 24, 6))
    predicted = rotation_6d_to_matrix_np(predicted_6d.reshape(-1, 24, 6))
    relative = np.swapaxes(predicted, -1, -2) @ reference
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _regional_metrics(payload: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    reference_joints = np.asarray(payload["reference_joints_world"], dtype=np.float64)[0]
    predicted_joints = np.asarray(payload["predicted_joints_world"], dtype=np.float64)[0]
    joint_position_cm = np.linalg.norm(predicted_joints - reference_joints, axis=-1) * 100.0
    joint_rotation_deg = _rotation_error_deg_per_joint(
        np.asarray(payload["reference_body_local_delta_6d"], dtype=np.float64)[0],
        np.asarray(payload["predicted_body_local_delta_6d"], dtype=np.float64)[0],
    )
    per_joint = {}
    for joint_index, joint_name in enumerate(SMPL_JOINT_NAMES):
        per_joint[joint_name] = {
            "mpjpe_cm": _mean_or_none(joint_position_cm[mask, joint_index]),
            "mpjre_deg": _mean_or_none(joint_rotation_deg[mask, joint_index]),
        }
    by_region = {}
    for region_index, region_name in enumerate(REGION_NAMES):
        joint_mask = np.asarray(TARGET_JOINT_REGIONS) == region_index
        by_region[region_name] = {
            "mpjpe_cm": _mean_or_none(joint_position_cm[mask][:, joint_mask]),
            "mpjre_deg": _mean_or_none(joint_rotation_deg[mask][:, joint_mask]),
        }
    return {"samples": int(mask.sum()), "by_region": by_region, "per_joint": per_joint}


def _feedback_amplification(payload: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(payload["reference_joints_world"], dtype=np.float64)[0]
    predicted = np.asarray(payload["predicted_joints_world"], dtype=np.float64)[0]
    frame_error = np.linalg.norm(predicted[:, :22] - reference[:, :22], axis=-1).mean(axis=-1) * 100.0
    valid = mask.copy()
    valid[0] = False
    previous_error = frame_error[:-1]
    current_error = frame_error[1:]
    paired = valid[1:] & mask[:-1]
    if not paired.any():
        return {"samples": 0, "correlation": None, "mean_gain": None}
    previous = previous_error[paired]
    current = current_error[paired]
    correlation = None
    if np.std(previous) > 0.0 and np.std(current) > 0.0:
        correlation = float(np.corrcoef(previous, current)[0, 1])
    gain = current / np.clip(previous, 1e-6, None)
    return {
        "samples": int(previous.size),
        "correlation": correlation,
        "mean_gain": float(np.mean(gain)),
        "median_gain": float(np.median(gain)),
    }


def _raw_resolver_metrics(
    payload: dict[str, np.ndarray],
    mask: np.ndarray,
    source: dict[str, np.ndarray],
) -> dict[str, Any]:
    """让 raw Prior pose 走与runtime相同的Resolver；hard mask置零以保留投影前姿态。"""

    frame_indices = np.flatnonzero(mask)
    reference_joints = np.asarray(payload["reference_joints_world"], dtype=np.float64)[0]
    deployed_joints = np.asarray(payload["predicted_joints_world"], dtype=np.float64)[0]
    raw_targets = np.asarray(payload["raw_pred_target_raw"], dtype=np.float32)[0]
    current_tracker_raw = np.asarray(payload["current_tracker_raw"], dtype=np.float32)[0]
    head_yaw = np.asarray(payload["current_head_yaw_world"], dtype=np.float64).reshape(-1)
    tracker_pos_world = np.asarray(payload["tracker_pos_world"], dtype=np.float32)[0]
    floor_y = np.asarray(payload["reference_root_position_world"], dtype=np.float64)[0, :, 1]
    raw_mpjpe_cm = []
    deployed_mpjpe_cm = []
    for frame_index in frame_indices:
        resolved = decode_and_resolve_pose(
            target_raw=raw_targets[frame_index],
            current_tracker_raw=current_tracker_raw[frame_index],
            hard_rotation_state=np.zeros(TRACKER_COUNT, dtype=bool),
            current_head_yaw_world=float(head_yaw[frame_index]),
            current_head_position_world=tracker_pos_world[frame_index, 0],
            floor_y=float(floor_y[frame_index]),
            joint_offsets_parent=source["joint_offsets_parent"],
            joint_rest_local_rotations_6d=source["joint_rest_local_rotations_6d"],
        )
        raw_mpjpe_cm.append(
            float(
                np.linalg.norm(
                    resolved.joints_world[:22] - reference_joints[frame_index, :22],
                    axis=-1,
                ).mean()
                * 100.0
            )
        )
        deployed_mpjpe_cm.append(
            float(
                np.linalg.norm(
                    deployed_joints[frame_index, :22]
                    - reference_joints[frame_index, :22],
                    axis=-1,
                ).mean()
                * 100.0
            )
        )
    raw_mean = _mean_or_none(np.asarray(raw_mpjpe_cm))
    deployed_mean = _mean_or_none(np.asarray(deployed_mpjpe_cm))
    return {
        "samples": int(frame_indices.size),
        "raw_prior_resolver_mpjpe_cm": raw_mean,
        "deployed_resolver_mpjpe_cm": deployed_mean,
        "hard_projection_mpjpe_gain_cm": (
            None
            if raw_mean is None or deployed_mean is None
            else float(raw_mean - deployed_mean)
        ),
    }


def audit_regional_horizon(history_diagnostic_dir: Path) -> dict[str, Any]:
    summary_path = history_diagnostic_dir / "history_horizon_diagnostic_summary.json"
    curve_path = history_diagnostic_dir / "history_horizon_curve.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    curve = json.loads(curve_path.read_text(encoding="utf-8"))
    eval_set_dir = Path(summary["metadata"]["eval_set_dir"])
    manifest_entries = {
        str(entry["sequence_id"]): entry
        for entry in read_longseq_manifest(eval_set_dir)
    }
    source_cache: dict[str, dict[str, np.ndarray]] = {}
    result: dict[str, Any] = {
        "source_summary_path": str(summary_path.resolve()),
        "source_summary_sha256": sha256_file(summary_path),
        "source_curve_path": str(curve_path.resolve()),
        "source_curve_sha256": sha256_file(curve_path),
        "curve_endpoints": {
            condition: {
                str(horizon): values[str(horizon)]
                for horizon in HORIZON_ENDPOINTS
            }
            for condition, values in curve["curve"].items()
        },
        "full_endpoints": curve["endpoints"],
        "files": [],
    }
    for file_result in summary["files"]:
        result_path = Path(file_result["result_path"])
        with np.load(result_path, allow_pickle=False) as data:
            payload = {key: np.asarray(data[key]) for key in data.files}
        eval_mask = np.asarray(payload["eval_frame_mask"], dtype=bool).reshape(-1)
        horizon = np.asarray(payload["diagnostic_horizon_frame"], dtype=np.int64).reshape(-1)
        protocol = str(np.asarray(payload["history_protocol"]).item())
        sequence_id = str(file_result["sequence_id"])
        if sequence_id not in source_cache:
            entry = manifest_entries[sequence_id]
            source_cache[sequence_id] = load_realtime_source(
                resolve_manifest_source_path(eval_set_dir=eval_set_dir, entry=entry)
            )
        item: dict[str, Any] = {
            "condition": str(file_result["condition"]),
            "sequence_id": sequence_id,
            "protocol": protocol,
            "result_path": str(result_path.resolve()),
            "result_sha256": sha256_file(result_path),
            "overall": _regional_metrics(payload, eval_mask),
            "raw_projection_resolver": _raw_resolver_metrics(
                payload,
                eval_mask,
                source_cache[sequence_id],
            ),
            "raw_rotation_deg": file_result.get("raw_rotation_deg"),
            "deployed_rotation_deg": file_result.get("deployed_rotation_deg"),
            "hard_projection_rotation_gain_deg": (
                None
                if file_result.get("raw_rotation_deg") is None
                or file_result.get("deployed_rotation_deg") is None
                else float(
                    file_result["raw_rotation_deg"]
                    - file_result["deployed_rotation_deg"]
                )
            ),
            "taid_internal_resolver": {
                key: file_result.get(key)
                for key in (
                    "taid_prior_root_xz_error_m",
                    "taid_prior_root_y_error_m",
                    "taid_prior_vs_deployed_root_gap_m",
                    "taid_prior_vs_deployed_yaw_deg",
                    "taid_prior_internal_mpjpe_cm",
                    "taid_prior_vs_deployed_joint_gap_cm",
                    "taid_prior_available_ratio",
                )
            },
            "feedback_amplification": _feedback_amplification(payload, eval_mask),
        }
        if protocol == "closed_loop":
            item["by_horizon"] = {
                str(value): _regional_metrics(payload, eval_mask & (horizon == value))
                for value in HORIZON_ENDPOINTS
            }
        result["files"].append(item)
    return result


def _hash_directory_files(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {
        str(file.relative_to(path)).replace("\\", "/"): sha256_file(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def _validate_old_structure_audit(sensitivity: dict[str, Any]) -> None:
    contract = sensitivity["contract"]
    if not contract["tracker_history_strictly_unused"]:
        raise RuntimeError(
            "旧结构审计发现 tracker_history 已影响 Prior；这与待修契约前提冲突，必须停止结构修改。"
        )
    if not contract["alpha_zero_history_no_leak"]:
        raise RuntimeError("alpha=0 Tracker 历史发生泄漏，审计未通过。")


def audit_taid_prior_capacity(args: Namespace) -> dict[str, Any]:
    if str(args.taid_ablation) != "B1":
        raise ValueError("29X 只允许审计 TAID B1。")
    if int(args.batch_size) != 2:
        raise ValueError("29X 固定 batch_size=2。")
    if int(args.rollout_steps) != REALTIME_POSE_MAX_ROLLOUT_STEPS:
        raise ValueError(
            f"29X 固定使用 K{REALTIME_POSE_MAX_ROLLOUT_STEPS} task/rollout。"
        )
    fixseed(int(args.seed))
    args.data_dir = str(resolve_latest_or_self(args.data_dir, kind="tasks"))
    args.normalizer_dir = str(
        resolve_latest_or_self(args.normalizer_dir, kind="normalizer")
    )
    model_path = Path(args.model_path).resolve()
    history_dir = Path(args.history_diagnostic_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    identity = _short_digest(str(model_path), str(history_dir), str(args.seed))
    output_dir = output_root / f"{model_path.stem}-{identity}"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = get_dataset_loader(
        data_dir=args.data_dir,
        batch_size=2,
        input_feats=args.input_feats,
        seq_len=args.seq_len,
        split=args.data_split,
        normalizer_dir=args.normalizer_dir,
        normalize_input=args.normalize_input,
        num_workers=0,
        pin_memory=args.cuda,
        enable_rollout=True,
        rollout_steps=args.rollout_steps,
        rollout_prob=1.0,
        cold_start_prob=args.cold_start_prob,
        cold_start_history_weights=args.cold_start_history_weights,
        cold_start_scenario_weights=args.cold_start_scenario_weights,
        scenario_weights=args.scenario_weights,
        seed=args.seed,
    )
    model, diffusion = create_model_and_diffusion(args)
    model.to(dist_util.dev())
    model, weights = load_checkpoint_model(
        model,
        model_path,
        dist_util.dev(),
        use_ema=bool(args.use_ema),
    )
    # ema_pytorch 的推理副本会冻结所有参数；审计需要 autograd，但仍严格恢复 B1 的
    # Prior-only 边界，不改变任何权重值。
    model._configure_taid_trainable_parameters()

    # TrainLoop 只复用现有 mask/loss/rollout 实现；禁止再次加载 B0、创建EMA或写checkpoint。
    loop_args = Namespace(**vars(args))
    loop_args.init_checkpoint = ""
    loop_args.resume_checkpoint = ""
    loop_args.model_ema = False
    loop_args.save_dir = str(output_dir / "_unused_train_loop")
    loop_args.rollout_prob = 1.0
    platform = NoPlatform(loop_args.save_dir)
    loop = TrainLoop(loop_args, platform, model, diffusion, data)
    batch = move_batch_to_device(next(iter(data)), dist_util.dev())
    model_kwargs = loop.mask_manager(batch, batch["x"])
    sensitivity = audit_input_sensitivity(model, model_kwargs)
    _validate_old_structure_audit(sensitivity)

    generator = torch.Generator(device=dist_util.dev())
    generator.manual_seed(int(args.seed) + 29024)
    timesteps = torch.randint(
        low=0,
        high=diffusion.num_timesteps,
        size=(batch["x"].shape[0],),
        device=dist_util.dev(),
        generator=generator,
    )
    gradients = audit_gradient_flow(model, loop, batch, timesteps)
    regional = audit_regional_horizon(history_dir)

    sensitivity_path = output_dir / "prior_input_sensitivity.json"
    gradient_path = output_dir / "prior_gradient_flow.json"
    regional_path = output_dir / "prior_regional_horizon.json"
    _write_json(sensitivity_path, sensitivity)
    _write_json(gradient_path, gradients)
    _write_json(regional_path, regional)

    repo_root = Path(__file__).resolve().parents[1]
    ema_path = model_path.with_name(model_path.name.replace("model", "ema", 1))
    args_path = model_path.parent / "args.json"
    metadata = {
        "kind": "taid_b1_prior_capacity_audit_before_tracker_history",
        "read_only": True,
        "optimizer_step_executed": False,
        "checkpoint_written": False,
        "model_path": str(model_path),
        "checkpoint_sha256": sha256_file(model_path),
        "weights": weights,
        "ema_path": str(ema_path) if ema_path.is_file() else None,
        "ema_sha256": sha256_file(ema_path) if ema_path.is_file() else None,
        "args_path": str(args_path) if args_path.is_file() else None,
        "args_sha256": sha256_file(args_path) if args_path.is_file() else None,
        "data_dir": str(Path(args.data_dir).resolve()),
        "task_hashes": _hash_directory_files(Path(args.data_dir)),
        "normalizer_dir": str(Path(args.normalizer_dir).resolve()),
        "normalizer_hashes": _hash_directory_files(Path(args.normalizer_dir)),
        "history_diagnostic_dir": str(history_dir),
        "git_diff_sha256": _git_diff_sha256(repo_root),
        "seed": int(args.seed),
        "batch_size": 2,
        "rollout_steps": int(args.rollout_steps),
    }
    summary = {
        "metadata": metadata,
        "contract": sensitivity["contract"],
        "gradient_checks": gradients["checks"],
        "outputs": {
            "prior_input_sensitivity": {
                "path": str(sensitivity_path),
                "sha256": sha256_file(sensitivity_path),
            },
            "prior_gradient_flow": {
                "path": str(gradient_path),
                "sha256": sha256_file(gradient_path),
            },
            "prior_regional_horizon": {
                "path": str(regional_path),
                "sha256": sha256_file(regional_path),
            },
        },
    }
    summary_path = output_dir / "prior_capacity_audit_summary.json"
    _write_json(summary_path, summary)
    platform.close()
    summary["summary_path"] = str(summary_path)
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_argument_parser()
    args = parse_and_load_from_model(
        parser,
        argv=argv,
        ignore_keys={
            "data_dir",
            "normalizer_dir",
            "batch_size",
            "rollout_steps",
            "rollout_prob",
            "num_workers",
            "save_dir",
        },
    )
    dist_util.setup_dist(args.device if args.cuda else -1)
    result = audit_taid_prior_capacity(args)
    print(
        "[audit_taid_prior_capacity] wrote "
        f"{result['summary_path']}"
    )
    return result


if __name__ == "__main__":
    main()
