from __future__ import annotations

import json
from argparse import BooleanOptionalAction, Namespace
from contextlib import nullcontext
from typing import Any

import torch
from torch.amp import autocast

from data_loaders.get_data import get_dataset_loader
from data_loaders.sensor_masking import (
    REALTIME_POSE_MAX_ROLLOUT_STEPS,
    TRACKER_COUNT,
    TRACKER_NAMES,
)
from train.train_platforms import NoPlatform
from train.training_loop import (
    TrainLoop,
    build_rollout_frame_weights,
    move_batch_to_device,
    validate_finite_losses,
)
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion
from utils.parser_util import build_train_arg_parser
from utils.run_dirs import resolve_latest_or_self


DEFAULT_BATCH_CANDIDATES = (32, 16, 8, 4)
PRIOR_PARAMETER_PREFIX = "taid_conditioner.prior."


def build_argument_parser():
    parser = build_train_arg_parser()
    parser.description = "只读验证一个真实 K15 batch 的 loss、梯度边界与显存。"
    parser.add_argument(
        "--batch_candidates",
        nargs="+",
        default=list(DEFAULT_BATCH_CANDIDATES),
        type=int,
    )
    parser.add_argument("--max_reserved_gib", default=14.0, type=float)
    parser.add_argument(
        "--require_tracker_history_prior",
        default=False,
        action=BooleanOptionalAction,
        help="要求 B1 Prior 的 active Tracker history 路径非零且 alpha=0 路径严格为零。",
    )
    parser.add_argument(
        "--require_fixed_slot_prior",
        default=False,
        action=BooleanOptionalAction,
        help="要求 Prior 使用固定六槽投影，并核对形状、初始等价性和槽位梯度。",
    )
    return parser


def normalize_batch_candidates(values: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    candidates = tuple(int(value) for value in values)
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("batch_candidates 必须是非空正整数列表。")
    if len(set(candidates)) != len(candidates):
        raise ValueError("batch_candidates 不能重复。")
    if any(left <= right for left, right in zip(candidates, candidates[1:])):
        raise ValueError("batch_candidates 必须严格降序。")
    return candidates


def summarize_gradient_boundary(model: torch.nn.Module) -> dict[str, Any]:
    trainable = []
    with_gradient = []
    finite = True
    nonzero = False
    unexpected = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable.append(name)
        if parameter.grad is None:
            continue
        with_gradient.append(name)
        finite = finite and bool(torch.isfinite(parameter.grad).all().item())
        nonzero = nonzero or bool(torch.count_nonzero(parameter.grad).item())
        if not name.startswith(PRIOR_PARAMETER_PREFIX):
            unexpected.append(name)
    trainable_outside_prior = [
        name for name in trainable if not name.startswith(PRIOR_PARAMETER_PREFIX)
    ]
    return {
        "trainable_parameter_count": len(trainable),
        "gradient_parameter_count": len(with_gradient),
        "all_gradients_finite": finite,
        "has_nonzero_gradient": nonzero,
        "trainable_outside_prior": trainable_outside_prior,
        "gradients_outside_prior": unexpected,
        "prior_only": bool(trainable)
        and not trainable_outside_prior
        and not unexpected
        and finite
        and nonzero,
    }


def losses_to_scalars(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        key: float(value.detach().float().mean().cpu().item())
        for key, value in sorted(losses.items())
        if torch.is_tensor(value)
    }


def summarize_rollout_weight_contract(
    scalar_losses: dict[str, float],
    *,
    rollout_steps: int,
    weighting: str,
) -> dict[str, Any]:
    """核对预检日志中的逐步权重，避免 Launch 参数存在但训练聚合未生效。"""

    expected = build_rollout_frame_weights(
        rollout_steps,
        weighting,
        dtype=torch.float64,
    ).tolist()
    actual = [
        scalar_losses.get(f"rollout_step_{step_index}_weight")
        for step_index in range(1, rollout_steps)
    ]
    available = all(value is not None for value in actual)
    tolerance = 1e-6
    matches = available and all(
        abs(float(value) - float(target)) <= tolerance
        for value, target in zip(actual, expected)
    )
    return {
        "weighting": weighting,
        "expected_step_1": float(expected[0]),
        "expected_step_last": float(expected[-1]),
        "actual_step_1": None if not actual else actual[0],
        "actual_step_last": None if not actual else actual[-1],
        "actual_sum": None if not available else float(sum(float(value) for value in actual)),
        "matches": bool(matches),
    }


def _cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def summarize_tracker_history_prior_contract(
    model: torch.nn.Module,
    loop: TrainLoop,
    batch: dict[str, Any],
) -> dict[str, Any]:
    """直接探测 history_summary→Tracker token 的梯度，不受零初始化 pose head 干扰。"""

    model_kwargs = loop.mask_manager(batch, batch["x"])
    y = dict(model_kwargs["y"])
    tracker_history = y["tracker_history"].detach().clone().requires_grad_(True)
    y["tracker_history"] = tracker_history
    model_kwargs = dict(model_kwargs)
    model_kwargs["tracker_history"] = tracker_history
    model_kwargs["y"] = y

    # Head 永远是 active Anchor；最后一个 Tracker 被显式置为 alpha=0，形成严格泄漏对照。
    inactive_index = 5
    current_tracker = y["current_tracker"].clone()
    current_tracker[:, inactive_index, 9:13] = 0.0
    current_tracker_raw = y["current_tracker_raw"].clone()
    current_tracker_raw[:, inactive_index, 9:13] = 0.0
    y["current_tracker"] = current_tracker
    y["current_tracker_raw"] = current_tracker_raw
    model_kwargs["current_tracker"] = current_tracker

    prepared = model.prepare_conditioning(
        y["pose_history"],
        tracker_history,
        current_tracker,
        model_kwargs["trajectory_history"],
        y["current_trajectory"],
        y["valid_frame_mask"],
        current_tracker_raw=current_tracker_raw,
        joint_offsets_parent=y["joint_offsets_parent"],
        pose_mean=y.get("normalizer_mean"),
        pose_std=y.get("normalizer_std"),
        tracker_mean=y.get("tracker_normalizer_mean"),
        tracker_std=y.get("tracker_normalizer_std"),
    )
    if prepared.taid is None:
        raise RuntimeError("Tracker-history preflight 要求启用 TAID。")
    prior = prepared.taid.prior
    alpha = prepared.taid.role_state.alpha.to(prior.tracker_tokens.dtype)
    weighted_tokens = prior.tracker_tokens * alpha[..., None]
    denominator = alpha.sum(dim=1, keepdim=True).clamp_min(1.0)
    normalized_slots = weighted_tokens / denominator[..., None]
    projection = model.taid_conditioner.prior.anchor_slot_projection
    slot_summary = projection(normalized_slots)
    history_gradient, slot_weight_gradient = torch.autograd.grad(
        slot_summary.square().mean(),
        (tracker_history, projection.weight),
        allow_unused=True,
    )
    active_norm = (
        0.0
        if history_gradient is None
        else float(torch.linalg.norm(history_gradient[:, :, 0]).detach().cpu().item())
    )
    inactive_norm = (
        0.0
        if history_gradient is None
        else float(
            torch.linalg.norm(history_gradient[:, :, inactive_index]).detach().cpu().item()
        )
    )
    first_linear = model.taid_conditioner.prior.tracker_fusion[0]
    expected_in_features = int(model.latent_dim) * 4
    expected_projection_shape = (
        int(model.latent_dim),
        TRACKER_COUNT * int(model.latent_dim),
    )
    slot_gradient_norms: list[float] = []
    active_slot_mask = [
        bool(value)
        for value in (alpha > 0).any(dim=0).detach().cpu().tolist()
    ]
    slot_gradients_finite = slot_weight_gradient is not None
    if slot_weight_gradient is not None:
        slot_gradients_finite = bool(torch.isfinite(slot_weight_gradient).all().item())
        for tracker_index in range(TRACKER_COUNT):
            start = tracker_index * int(model.latent_dim)
            end = start + int(model.latent_dim)
            slot_gradient_norms.append(
                float(
                    torch.linalg.norm(slot_weight_gradient[:, start:end])
                    .detach()
                    .cpu()
                    .item()
                )
            )
    legacy_summary = normalized_slots.sum(dim=1)
    initialization_max_abs_gap = float(
        (slot_summary - legacy_summary).abs().max().detach().cpu().item()
    )
    active_slot_gradients_nonzero = bool(
        len(slot_gradient_norms) == TRACKER_COUNT
        and all(
            (not is_active) or slot_gradient_norms[index] > 0.0
            for index, is_active in enumerate(active_slot_mask)
        )
    )
    fixed_slot_contract = bool(
        str(model.taid_config.prior_tracker_aggregation) == "fixed_slots"
        and tuple(projection.weight.shape) == expected_projection_shape
        and slot_gradients_finite
        and len(slot_gradient_norms) == TRACKER_COUNT
        and active_slot_gradients_nonzero
        and initialization_max_abs_gap <= 1e-6
    )
    return {
        "history_summary_shape": list(prepared.observation.history_summary.shape),
        "tracker_fusion_in_features": int(first_linear.in_features),
        "expected_tracker_fusion_in_features": expected_in_features,
        "active_tracker_index": 0,
        "active_alpha_min": float(alpha[:, 0].min().detach().cpu().item()),
        "active_history_gradient_norm": active_norm,
        "inactive_tracker_index": inactive_index,
        "inactive_alpha_max": float(alpha[:, inactive_index].max().detach().cpu().item()),
        "inactive_history_gradient_norm": inactive_norm,
        "tracker_slot_order": list(TRACKER_NAMES),
        "prior_tracker_aggregation": str(model.taid_config.prior_tracker_aggregation),
        "slot_projection_shape": list(projection.weight.shape),
        "expected_slot_projection_shape": list(expected_projection_shape),
        "active_slot_mask": active_slot_mask,
        "slot_block_gradient_norms": slot_gradient_norms,
        "slot_gradients_finite": bool(slot_gradients_finite),
        "active_slot_gradients_nonzero": active_slot_gradients_nonzero,
        "initial_projection_vs_weighted_mean_max_abs": initialization_max_abs_gap,
        "fixed_slot_contract_passes": fixed_slot_contract,
        "observation_encoder_frozen": not any(
            parameter.requires_grad for parameter in model.observation_encoder.parameters()
        ),
        "passes": bool(
            tuple(prepared.observation.history_summary.shape[1:])
            == (6, int(model.latent_dim))
            and int(first_linear.in_features) == expected_in_features
            and active_norm > 0.0
            and inactive_norm == 0.0
            and fixed_slot_contract
            and float(alpha[:, inactive_index].max().detach().cpu().item()) == 0.0
            and not any(
                parameter.requires_grad
                for parameter in model.observation_encoder.parameters()
            )
        ),
    }


def run_preflight_attempt(base_args: Namespace, batch_size: int) -> dict[str, Any]:
    args = Namespace(**vars(base_args))
    args.batch_size = int(batch_size)
    # 预检必须物化完整 15 帧，而不是让 rollout_prob 随机跳过该 batch。
    args.rollout_prob = 1.0
    fixseed(int(args.seed))
    data = get_dataset_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        input_feats=args.input_feats,
        seq_len=args.seq_len,
        split=args.data_split,
        normalizer_dir=args.normalizer_dir,
        normalize_input=args.normalize_input,
        num_workers=args.num_workers,
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
    platform = NoPlatform(args.save_dir)
    loop = TrainLoop(args, platform, model, diffusion, data)
    model.train()
    device = dist_util.dev()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    batch = move_batch_to_device(next(iter(data)), device)
    loop.opt.zero_grad(set_to_none=True)
    timesteps = torch.randint(
        low=0,
        high=diffusion.num_timesteps,
        size=(batch["x"].shape[0],),
        device=device,
    )
    amp_context = (
        autocast(device_type="cuda", dtype=loop.amp_dtype)
        if device.type == "cuda"
        else nullcontext()
    )
    with amp_context:
        losses = loop.compute_losses(batch=batch, timesteps=timesteps)
        loss = losses["loss"].mean()
    validate_finite_losses(losses=losses, loss=loss, batch=batch)
    loop.scaler.scale(loss).backward()
    if loop.scaler.is_enabled():
        loop.scaler.unscale_(loop.opt)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_reserved_bytes = 0

    scalar_losses = losses_to_scalars(losses)
    required_rollout_diagnostics = (
        "rollout_uniform_frame_loss",
        "rollout_step_1_weight",
        "rollout_step_1_weighted_loss",
        "rollout_step_14_loss",
        "rollout_step_14_weight",
        "rollout_step_14_weighted_loss",
        "rollout_step_14_prior_fk_loss",
        "rollout_step_14_prior_internal_fk_loss",
        "rollout_step_14_prior_root_pose_gap_m",
        "rollout_step_14_prior_root_pose_gap_xz_m",
        "rollout_step_14_prior_joint_resolver_gap_m",
    )
    missing_rollout_diagnostics = [
        name for name in required_rollout_diagnostics if name not in scalar_losses
    ]
    weight_contract = summarize_rollout_weight_contract(
        scalar_losses,
        rollout_steps=int(args.rollout_steps),
        weighting=str(args.rollout_frame_weighting),
    )
    gradient_boundary = summarize_gradient_boundary(model)
    tracker_history_contract = summarize_tracker_history_prior_contract(
        model,
        loop,
        batch,
    )
    result = {
        "batch_size": int(batch_size),
        "device": str(device),
        "peak_reserved_bytes": peak_reserved_bytes,
        "peak_reserved_gib": peak_reserved_bytes / float(1024**3),
        "losses": scalar_losses,
        "rollout_frame_weighting": str(args.rollout_frame_weighting),
        "rollout_diagnostic_missing": missing_rollout_diagnostics,
        "weight_contract": weight_contract,
        "gradient_boundary": gradient_boundary,
        "tracker_history_contract": tracker_history_contract,
    }
    result["finite"] = all(torch.isfinite(value).all().item() for value in losses.values())
    platform.close()
    del loop, model, diffusion, data, batch, losses, loss
    _cleanup_cuda()
    return result


def validate_realtime_pose_rollout(args: Namespace) -> dict[str, Any]:
    if int(args.rollout_steps) != REALTIME_POSE_MAX_ROLLOUT_STEPS:
        raise ValueError(
            f"显存预检固定要求 rollout_steps={REALTIME_POSE_MAX_ROLLOUT_STEPS}。"
        )
    if str(args.taid_ablation) != "B1":
        raise ValueError("显存预检固定构建 TAID B1。")
    if not str(args.init_checkpoint):
        raise ValueError("显存预检必须通过 init_checkpoint 从正式 B0 初始化。")
    if float(args.max_reserved_gib) <= 0:
        raise ValueError("max_reserved_gib 必须大于0。")
    candidates = normalize_batch_candidates(args.batch_candidates)
    args.data_dir = str(resolve_latest_or_self(args.data_dir, kind="tasks"))
    args.normalizer_dir = str(resolve_latest_or_self(args.normalizer_dir, kind="normalizer"))
    attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for batch_size in candidates:
        try:
            attempt = run_preflight_attempt(args, batch_size)
        except torch.cuda.OutOfMemoryError as exc:
            _cleanup_cuda()
            attempts.append(
                {"batch_size": batch_size, "status": "oom", "error": str(exc)}
            )
            continue
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            _cleanup_cuda()
            attempts.append(
                {"batch_size": batch_size, "status": "oom", "error": str(exc)}
            )
            continue
        attempt["status"] = "measured"
        attempt["within_memory_limit"] = (
            float(attempt["peak_reserved_gib"]) <= float(args.max_reserved_gib)
        )
        attempt["accepted"] = bool(
            attempt["finite"]
            and not attempt["rollout_diagnostic_missing"]
            and attempt["weight_contract"]["matches"]
            and attempt["gradient_boundary"]["prior_only"]
            and (
                not bool(args.require_tracker_history_prior)
                or attempt["tracker_history_contract"]["passes"]
            )
            and (
                not bool(args.require_fixed_slot_prior)
                or attempt["tracker_history_contract"]["fixed_slot_contract_passes"]
            )
            and attempt["within_memory_limit"]
        )
        attempts.append(attempt)
        if attempt["accepted"]:
            selected = attempt
            break

    result = {
        "selected_batch_size": None if selected is None else selected["batch_size"],
        "max_reserved_gib": float(args.max_reserved_gib),
        "rollout_steps": int(args.rollout_steps),
        "rollout_frame_weighting": str(args.rollout_frame_weighting),
        "require_tracker_history_prior": bool(args.require_tracker_history_prior),
        "require_fixed_slot_prior": bool(args.require_fixed_slot_prior),
        "optimizer_step_executed": False,
        "checkpoint_written": False,
        "attempts": attempts,
    }
    if selected is None:
        raise RuntimeError(
            "没有 batch candidate 同时满足 OOM、有限性、14 GiB 和 Prior 梯度边界门槛。\n"
            + json.dumps(result, ensure_ascii=False, indent=2)
        )
    return result


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_argument_parser().parse_args(argv)
    dist_util.setup_dist(args.device if args.cuda else -1)
    result = validate_realtime_pose_rollout(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
