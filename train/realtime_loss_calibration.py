from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import torch

from diffusion.realtime_pose import (
    REALTIME_POSE_LOSS_GRADIENT_TARGET_RATIOS,
    REALTIME_POSE_LOSS_TERM_TO_WEIGHT,
)
from train.training_loop import move_batch_to_device


CALIBRATION_MASK_CATEGORIES = (
    "full_six",
    "standard_three",
    "static_sparse",
    "dynamic_dropout",
)


def expected_active_loss_terms(losses: Mapping[str, torch.Tensor]) -> dict[str, bool]:
    hip_missing = bool(losses["hip_missing_fraction"].detach().bool().any().item())
    temporal = bool(losses["temporal_sample_fraction"].detach().bool().any().item())
    contact = bool((losses["contact_active_foot_count"].detach() > 0).any().item())
    stationary_margin = bool(
        (losses["stationary_margin_loss"].detach().abs() > 1e-12).any().item()
    )
    return {
        "local_rotation_loss": True,
        "body_geometry_loss": True,
        "tracker_relative_pos_loss": True,
        "tracker_relative_rot_loss": True,
        "nohip_yaw_loss": hip_missing,
        "nohip_root_xz_loss": hip_missing,
        "nohip_height_loss": hip_missing,
        "stationary_margin_loss": stationary_margin,
        "contact_height_loss": contact,
        "contact_velocity_loss": contact and temporal,
        "joint_velocity_loss": temporal,
        "rotation_velocity_loss": temporal,
        "yaw_velocity_loss": hip_missing and temporal,
    }


def gradient_norm(
    loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> float:
    params = tuple(parameter for parameter in parameters if parameter.requires_grad)
    gradients = torch.autograd.grad(
        loss.mean(),
        params,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squared = torch.zeros((), device=loss.device, dtype=torch.float64)
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.detach().double().square().sum()
    return float(torch.sqrt(squared).cpu().item())


def measure_loss_gradient_norms(
    *,
    losses: Mapping[str, torch.Tensor],
    parameters: Iterable[torch.nn.Parameter],
) -> dict[str, float]:
    names = ("simple_loss", *REALTIME_POSE_LOSS_TERM_TO_WEIGHT.keys())
    missing = [name for name in names if name not in losses]
    if missing:
        raise KeyError(f"loss gradient calibration 缺少项：{missing}")
    params = tuple(parameters)
    result = {}
    for index, name in enumerate(names):
        result[name] = gradient_norm(
            losses[name],
            params,
            retain_graph=index < len(names) - 1,
        )
    return result


def calibrate_realtime_loss_weights(
    gradient_norm_samples: Iterable[Mapping[str, float]],
    *,
    target_ratios: Mapping[str, float] = REALTIME_POSE_LOSS_GRADIENT_TARGET_RATIOS,
    minimum: float = 1e-6,
    maximum: float = 100.0,
    zero_tolerance: float = 1e-12,
) -> dict[str, object]:
    samples = list(gradient_norm_samples)
    if not samples:
        raise ValueError("至少需要一个 gradient norm sample。")
    simple_values = np.asarray([float(sample["simple_loss"]) for sample in samples], dtype=np.float64)
    simple_values = simple_values[np.isfinite(simple_values) & (simple_values > zero_tolerance)]
    if simple_values.size == 0:
        raise RuntimeError("L_x0 在全部 calibration batch 上都没有有效梯度。")
    simple_median = float(np.median(simple_values))
    weights: dict[str, float] = {}
    medians: dict[str, float] = {"simple_loss": simple_median}
    measured_ratios: dict[str, float] = {}
    for loss_name, target_ratio in target_ratios.items():
        values = np.asarray([float(sample[loss_name]) for sample in samples], dtype=np.float64)
        values = values[np.isfinite(values) & (values > zero_tolerance)]
        if values.size == 0:
            raise RuntimeError(f"预期激活的 {loss_name} 在全部 calibration batch 上都是零梯度。")
        median = float(np.median(values))
        weight = float(np.clip(float(target_ratio) * simple_median / median, minimum, maximum))
        weight_name = REALTIME_POSE_LOSS_TERM_TO_WEIGHT[loss_name]
        weights[weight_name] = weight
        medians[loss_name] = median
        measured_ratios[loss_name] = weight * median / simple_median
    return {
        "weights": weights,
        "gradient_norm_medians": medians,
        "target_ratios": {name: float(value) for name, value in target_ratios.items()},
        "measured_ratios": measured_ratios,
        "sample_count": len(samples),
        "clamp": [float(minimum), float(maximum)],
    }


def collect_fixed_calibration_norms(
    *,
    loop,
    batches_by_category: Mapping[str, dict],
    seed: int = 10,
) -> list[dict[str, object]]:
    """在四种 mask × 四个 timestep 区间上收集固定 16 组 Loss v3 梯度范数。"""

    missing = [name for name in CALIBRATION_MASK_CATEGORIES if name not in batches_by_category]
    if missing:
        raise KeyError(f"calibration 缺少 mask category batch：{missing}")
    timestep_centers = np.linspace(0, loop.diffusion.num_timesteps - 1, 4).round().astype(np.int64)
    results: list[dict[str, object]] = []
    was_training = loop.model.training
    loop.model.train(True)
    try:
        for category_index, category in enumerate(CALIBRATION_MASK_CATEGORIES):
            source_batch = move_batch_to_device(batches_by_category[category], loop.device)
            for timestep_index, timestep in enumerate(timestep_centers.tolist()):
                batch = loop.prepare_teacher_forced_temporal_state(source_batch)
                # temporal 项只在 predicted history 上激活；标定时使用可复现的 teacher 值作为预测前态。
                batch = dict(batch)
                batch["previous_state_is_predicted"] = torch.ones_like(
                    batch["previous_state_is_predicted"], dtype=torch.bool
                )
                sample = batch["x"]
                timesteps = torch.full(
                    (sample.shape[0],),
                    int(timestep),
                    device=loop.device,
                    dtype=torch.long,
                )
                generator = torch.Generator(device=loop.device)
                generator.manual_seed(int(seed) + category_index * 100 + timestep_index)
                noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
                losses = loop.diffusion.training_losses(
                    loop.model,
                    sample,
                    timesteps,
                    model_kwargs=loop.mask_manager(batch, sample),
                    noise=noise,
                    feature_w=loop._feature_weights_for_batch(sample.shape[0], sample.shape[2]),
                    snr_gamma=0.0,
                    use_l1=loop.use_l1,
                )
                norms = measure_loss_gradient_norms(losses=losses, parameters=loop.model.parameters())
                expected_active = expected_active_loss_terms(losses)
                zero_when_active = [
                    name
                    for name, active in expected_active.items()
                    if active and (not np.isfinite(norms[name]) or norms[name] <= 1e-12)
                ]
                if zero_when_active:
                    raise RuntimeError(
                        f"{category}/t={timestep} 预期激活却得到零梯度：{zero_when_active}"
                    )
                results.append(
                    {
                        "category": category,
                        "timestep": int(timestep),
                        "expected_active_terms": [
                            name for name, active in expected_active.items() if active
                        ],
                        **norms,
                    }
                )
    finally:
        loop.model.train(was_training)
    return results


def write_calibration_report(path: str | Path, report: Mapping[str, object]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _build_category_loader(args, category: str):
    from data_loaders.get_data import get_dataset_loader

    return get_dataset_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        input_feats=args.input_feats,
        seq_len=args.seq_len,
        split=args.data_split,
        normalizer_dir=args.normalizer_dir,
        normalize_input=args.normalize_input,
        preload_data=args.preload_data,
        num_workers=args.num_workers,
        pin_memory=args.cuda,
        schema_name=args.schema,
        tracker_pos_noise_std=args.tracker_pos_noise_std,
        tracker_rot_noise_std=args.tracker_rot_noise_std,
        non_head_tracker_dropout_prob=args.non_head_tracker_dropout_prob,
        history_pose_noise_std=args.history_pose_noise_std,
        history_yaw_noise_std=args.history_yaw_noise_std,
        root_yaw_ref_noise_std=args.root_yaw_ref_noise_std,
        history_pose_dropout_prob=args.history_pose_dropout_prob,
        history_pose_replace_prob=args.history_pose_replace_prob,
        history_yaw_replace_prob=args.history_yaw_replace_prob,
        history_root_yaw_drift_std=args.history_root_yaw_drift_std,
        tracker_latency_max_frames=args.tracker_latency_max_frames,
        tracker_burst_dropout_prob=args.tracker_burst_dropout_prob,
        tracker_outlier_prob=args.tracker_outlier_prob,
        enable_rollout=False,
        rollout_steps=1,
        tracker_mask_policy="fixed_categories",
        tracker_mask_seed=args.tracker_mask_seed,
        tracker_mask_fill=args.tracker_mask_fill,
        tracker_mask_categories=[category],
    )


def main(argv: list[str] | None = None) -> Path:
    """加载同一 Stage A checkpoint，生成四类 mask × 四段 timestep 的 Loss v3 标定报告。"""

    from diffusion import logger
    from train.train_diffusionposer import resolve_input_artifact_dirs
    from train.train_platforms import NoPlatform
    from train.training_loop import TrainLoop
    from utils import dist_util
    from utils.fixseed import fixseed
    from utils.model_util import create_model_and_diffusion
    from utils.parser_util import train_args

    argv_values = sys.argv[1:] if argv is None else list(argv)
    calibration_parser = ArgumentParser(add_help=False)
    calibration_parser.add_argument("--calibration_output", required=True, type=str)
    calibration_args, train_argv = calibration_parser.parse_known_args(argv_values)
    args = train_args(train_argv)
    if not str(args.init_checkpoint).strip():
        raise ValueError("梯度标定必须通过 --init_checkpoint 指定统一的 Stage A checkpoint。")
    if str(args.resume_checkpoint).strip():
        raise ValueError("梯度标定不能使用 --resume_checkpoint；必须重置 optimizer/global step/EMA。")

    # 标定只需要单步 Loss v3，不创建 rollout sampler 或 EMA 副本。
    args.rollout_steps = 1
    args.short_rollout_prob = 0.0
    args.short_rollout_loss_weight = 0.0
    args.long_rollout_prob = 0.0
    args.long_rollout_loss_weight = 0.0
    args.eval_during_training = False
    args.model_ema = False
    args.num_steps = 1
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    fixseed(args.seed)
    resolve_input_artifact_dirs(args)
    dist_util.setup_dist(args.device if args.cuda else -1)
    logger.configure(dir=args.save_dir)

    loaders = {
        category: _build_category_loader(args, category)
        for category in CALIBRATION_MASK_CATEGORIES
    }
    batches = {category: next(iter(loader)) for category, loader in loaders.items()}
    batch_keyids = {
        category: list(batch.get("keyid", []))
        for category, batch in batches.items()
    }

    model, diffusion = create_model_and_diffusion(args)
    model.to(dist_util.dev())
    platform = NoPlatform(args.save_dir)
    try:
        loop = TrainLoop(args, platform, model, diffusion, loaders[CALIBRATION_MASK_CATEGORIES[0]])
        samples = collect_fixed_calibration_norms(
            loop=loop,
            batches_by_category=batches,
            seed=args.seed,
        )
        report = calibrate_realtime_loss_weights(samples)
        report.update(
            {
                "seed": int(args.seed),
                "batch_size": int(args.batch_size),
                "schema": str(args.schema),
                "data_dir": str(args.data_dir),
                "normalizer_dir": str(args.normalizer_dir),
                "init_checkpoint": str(Path(args.init_checkpoint).resolve()),
                "batch_keyids": batch_keyids,
                "gradient_norm_samples": samples,
            }
        )
        output = write_calibration_report(calibration_args.calibration_output, report)
        print(json.dumps(report["weights"], ensure_ascii=False, indent=2), flush=True)
        print(f"calibration report: {output.resolve()}", flush=True)
        return output
    finally:
        platform.close()


if __name__ == "__main__":
    main()
