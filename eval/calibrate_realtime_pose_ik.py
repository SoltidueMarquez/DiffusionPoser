from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data_loaders.realtime_pose_config import IKInpaintingConfig
from data_loaders.realtime_pose_dataset import RealtimePoseTaskDataset, TaskRequest
from data_loaders.realtime_pose_ik import (
    DIRECT_ROTATION,
    DIRECTION_ONLY,
    POSITION_SOLVED,
    build_current_ik,
    build_ik_joint_chain_length,
    build_ik_joint_source_reliability,
)
from data_loaders.realtime_pose_kinematics import JOINT_INDEX, rotation_6d_to_matrix_torch
from data_loaders.tracker_reliability import compute_tracker_online_confidence_torch


_CHAIN_JOINTS = {
    "torso": ("spine1", "spine2", "spine3", "neck"),
    "left_arm": ("left_shoulder", "left_elbow"),
    "right_arm": ("right_shoulder", "right_elbow"),
    "left_leg": ("left_hip", "left_knee", "left_ankle"),
    "right_leg": ("right_hip", "right_knee", "right_ankle"),
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用 materialized train task 离线校准 IK direction quality 与 residual scale。"
    )
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--max_samples", default=20_000, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--tracker_confidence_warmup", default=15, type=int)
    parser.add_argument("--fabrik_iterations", default=2, type=int)
    return parser


def fit_direction_confidence_parameters(
    source_reliability: np.ndarray,
    residual_ratio: np.ndarray,
    rotation_error_rad: np.ndarray,
) -> dict[str, float]:
    """拟合 `source * quality * exp(-ratio/scale)` 到旋转相似度。

    旋转监督使用 `cos²(theta/2)`，它由 SO(3) geodesic error 唯一决定，范围
    自然落在 `[0,1]`，因此不需要人为指定“多少度算正确”的阈值。
    """

    source = np.asarray(source_reliability, dtype=np.float64).reshape(-1)
    ratio = np.asarray(residual_ratio, dtype=np.float64).reshape(-1)
    error = np.asarray(rotation_error_rad, dtype=np.float64).reshape(-1)
    if not (source.shape == ratio.shape == error.shape) or source.size == 0:
        raise ValueError("校准输入必须是同形非空一维数组。")
    finite = np.isfinite(source) & np.isfinite(ratio) & np.isfinite(error)
    finite &= (source > 0.0) & (ratio >= 0.0) & (error >= 0.0)
    source, ratio, error = source[finite], ratio[finite], error[finite]
    if source.size < 2:
        raise ValueError("有效 DIRECTION_ONLY 校准样本不足。")

    positive_ratio = ratio[ratio > 0.0]
    if positive_ratio.size == 0:
        raise ValueError("所有 endpoint residual ratio 均为零，无法辨识 residual_scale。")
    lower = max(float(np.percentile(positive_ratio, 1)) * 0.1, 1e-6)
    upper = max(float(np.percentile(positive_ratio, 99)) * 10.0, lower * 10.0)
    scales = np.geomspace(lower, upper, num=512)
    target = np.square(np.cos(0.5 * np.clip(error, 0.0, np.pi)))

    best: tuple[float, float, float] | None = None
    for scale in scales:
        base = source * np.exp(-ratio / scale)
        denominator = float(np.dot(base, base))
        if denominator <= 0.0:
            continue
        quality = float(np.clip(np.dot(base, target) / denominator, 1e-4, 0.9999))
        mse = float(np.mean(np.square(base * quality - target)))
        if best is None or mse < best[0]:
            best = (mse, quality, float(scale))
    if best is None:
        raise RuntimeError("无法拟合 IK confidence 参数。")
    return {
        "direction_only_quality": best[1],
        "residual_scale": best[2],
        "fit_mse": best[0],
    }


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "p50": None, "p90": None, "p95": None}
    return {
        "count": int(array.size),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def collect_calibration_samples(
    dataset: RealtimePoseTaskDataset,
    max_samples: int,
    batch_size: int,
    seed: int,
    tracker_confidence_warmup: int,
    fabrik_iterations: int,
) -> dict[str, np.ndarray]:
    """重放 train task 的五种 Tracker timeline，收集真实 IK 误差。"""

    if int(max_samples) <= 0 or int(batch_size) <= 0:
        raise ValueError("max_samples 和 batch_size 必须大于 0。")
    generator = np.random.default_rng(int(seed))
    sample_count = min(int(max_samples), len(dataset) * 5)
    task_indices = generator.permutation(len(dataset))
    requests = [
        TaskRequest(
            task_index=int(task_indices[(index // 5) % len(task_indices)]),
            config_index=index % 5,
        )
        for index in range(sample_count)
    ]
    collected = {
        "constraint_type": [],
        "source_reliability": [],
        "residual_ratio": [],
        "rotation_error_rad": [],
    }
    # 这些临时数值只满足 build_current_ik 的 confidence 接口；IK pose、mask、
    # constraint type 与 residual 都不依赖它们，因此不会污染待校准样本。
    geometry_config = IKInpaintingConfig(
        tracker_confidence_warmup=tracker_confidence_warmup,
        fabrik_iterations=fabrik_iterations,
        direction_only_quality=0.5,
        residual_scale=0.1,
    ).validate()

    for start in range(0, len(requests), int(batch_size)):
        items = [dataset[request] for request in requests[start : start + int(batch_size)]]
        previous_pose = torch.stack([item["history_pose_observation"][-1] for item in items])
        previous_valid = torch.stack([item["window_valid_mask"][-2] for item in items])
        tracker = torch.stack([item["tracker_window_raw"][-1] for item in items])
        configured = torch.stack([item["configured"][-1] for item in items])
        measured = torch.stack([item["measured_valid"][-1] for item in items])
        d_on = torch.stack([item["d_on"][-1] for item in items]).float()
        offsets = torch.stack([item["joint_offsets_parent"] for item in items]).float()
        rest = torch.stack([item["joint_rest_local_rotations_6d"] for item in items]).float()
        target_pose = torch.stack([item["x"][0] for item in items]).reshape(-1, 24, 6)
        tracker_source = compute_tracker_online_confidence_torch(
            configured & measured,
            d_on,
            warmup_frames=tracker_confidence_warmup,
        )
        result = build_current_ik(
            previous_pose_raw=previous_pose.float(),
            previous_pose_valid=previous_valid.bool(),
            current_tracker_raw=tracker.float(),
            tracker_source_reliability=tracker_source,
            joint_offsets_parent=offsets,
            joint_rest_local_rotations_6d=rest,
            config=geometry_config,
        )
        predicted = rotation_6d_to_matrix_torch(result.pose)
        target = rotation_6d_to_matrix_torch(target_pose)
        relative = predicted.transpose(-1, -2) @ target
        cosine = ((torch.diagonal(relative, dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(
            -1.0, 1.0
        )
        error = torch.acos(cosine)
        joint_chain_length = build_ik_joint_chain_length(offsets)
        ratio = result.position_residual / joint_chain_length.clamp_min(1e-8)

        joint_source = build_ik_joint_source_reliability(
            tracker_source_reliability=tracker_source,
            constraint_type=result.constraint_type,
        )
        for name, tensor in (
            ("constraint_type", result.constraint_type),
            ("source_reliability", joint_source),
            ("residual_ratio", ratio),
            ("rotation_error_rad", error),
        ):
            collected[name].append(tensor.detach().cpu().numpy())

    return {name: np.concatenate(values, axis=0) for name, values in collected.items()}


def build_calibration_report(samples: dict[str, np.ndarray]) -> dict:
    constraint = samples["constraint_type"]
    direction = constraint == DIRECTION_ONLY
    fitted = fit_direction_confidence_parameters(
        samples["source_reliability"][direction],
        samples["residual_ratio"][direction],
        samples["rotation_error_rad"][direction],
    )
    by_type = {}
    for name, value in (
        ("direct_rotation", DIRECT_ROTATION),
        ("position_solved", POSITION_SOLVED),
        ("direction_only", DIRECTION_ONLY),
    ):
        mask = constraint == value
        by_type[name] = {
            "rotation_error_rad": _summary(samples["rotation_error_rad"][mask]),
            "residual_ratio": _summary(samples["residual_ratio"][mask]),
        }
    by_chain = {}
    for name, joints in _CHAIN_JOINTS.items():
        indices = [JOINT_INDEX[joint] for joint in joints]
        mask = direction[:, indices]
        by_chain[name] = {
            "rotation_error_rad": _summary(samples["rotation_error_rad"][:, indices][mask]),
            "residual_ratio": _summary(samples["residual_ratio"][:, indices][mask]),
        }
    return {
        "recommended_parameters": {
            "ik_direction_only_quality": fitted["direction_only_quality"],
            "ik_residual_scale": fitted["residual_scale"],
        },
        "fit_mse": fitted["fit_mse"],
        "position_solved": "N/A: current shortest-arc FABRIK emits no POSITION_SOLVED",
        "by_constraint_type": by_type,
        "by_chain": by_chain,
    }


def main(argv: list[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    dataset = RealtimePoseTaskDataset(
        data_dir=args.data_dir,
        split=args.split,
        normalize_input=False,
        normalizer_dir=None,
    )
    try:
        samples = collect_calibration_samples(
            dataset=dataset,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            tracker_confidence_warmup=args.tracker_confidence_warmup,
            fabrik_iterations=args.fabrik_iterations,
        )
    finally:
        dataset.close()
    report = build_calibration_report(samples)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["recommended_parameters"], indent=2, ensure_ascii=False))
    return output


if __name__ == "__main__":
    main()
