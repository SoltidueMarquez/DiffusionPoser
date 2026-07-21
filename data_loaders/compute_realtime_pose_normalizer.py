from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from data_loaders.realtime_pose_dataset import (
    RealtimePoseTaskDataset,
    find_manifest_path,
    read_task_manifest,
)
from data_loaders.realtime_pose_contract import (
    RUNTIME_CONTRACT_METADATA_FIELDS,
    validate_stationary_label_metadata,
)
from data_loaders.sensor_masking import (
    DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SCHEMA_NAMES,
    TRACKER_COUNT,
    get_schema_spec,
)
from schemas.realtime_pose_stationary5_v1.contract import (
    SMPL_JOINT_NAMES,
    STATIONARY_JOINT_NAMES,
    TRACKER_NAMES,
)
from data_loaders.stationary_label_config import STATIONARY_LABEL_METADATA_FIELDS
from utils.artifact_paths import normalizer_root, task_root
from utils.artifact_roots import load_artifact_roots
from utils.normalizer import RealtimePoseNormalizer
from utils.run_dirs import resolve_latest_or_self, timestamped_child_dir, write_latest_pointer


DEFAULT_TASK_SET_NAME = "amass_60hz_tasks"
DEFAULT_NORMALIZER_NAME = "amass_60hz_train"
CONVERGENCE_DIAGNOSTIC_TOP_K = 10


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute realtime_pose mean/std normalizer from source references.")
    group = parser.add_argument_group("paths")
    group.add_argument("--artifact_roots_config", default="", type=str)
    group.add_argument("--task_set_name", default=DEFAULT_TASK_SET_NAME, type=str)
    group.add_argument("--normalizer_name", default=DEFAULT_NORMALIZER_NAME, type=str)
    group.add_argument("--task_dir", default="", type=str)
    group.add_argument("--output_dir", default="", type=str)
    group.add_argument("--direct_output", action="store_true")

    group = parser.add_argument_group("statistics")
    group.add_argument("--schema", default=DEFAULT_REALTIME_POSE_SCHEMA_NAME, choices=REALTIME_POSE_SCHEMA_NAMES, type=str)
    group.add_argument("--split", default="train", type=str)
    group.add_argument("--eps", default=1e-8, type=float)
    group.add_argument("--windows_per_source", default=2, type=int)
    group.add_argument("--convergence_windows_per_source", default=4, type=int)
    group.add_argument("--check_convergence", default=True, type=str2bool)
    group.add_argument("--tracker_mask_seed", default=10, type=int)
    group.add_argument("--run_name", default="auto", type=str)
    group.add_argument("--overwrite", action="store_true")
    return parser


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def resolve_normalizer_paths(args: argparse.Namespace) -> argparse.Namespace:
    """把空路径参数解析到 schema-aware task/normalizer 根目录。"""

    roots = None

    def get_roots():
        nonlocal roots
        if roots is None:
            roots = load_artifact_roots(getattr(args, "artifact_roots_config", "") or None)
        return roots

    schema_name = str(getattr(args, "schema", DEFAULT_REALTIME_POSE_SCHEMA_NAME))
    task_set_name = str(getattr(args, "task_set_name", DEFAULT_TASK_SET_NAME))
    normalizer_name = str(getattr(args, "normalizer_name", DEFAULT_NORMALIZER_NAME))

    if _path_arg_is_empty(getattr(args, "task_dir", "")):
        args.task_dir = task_root(get_roots(), schema_name=schema_name, task_set_name=task_set_name)
    else:
        args.task_dir = Path(args.task_dir)

    if _path_arg_is_empty(getattr(args, "output_dir", "")):
        args.output_dir = normalizer_root(get_roots(), schema_name=schema_name, normalizer_name=normalizer_name)
    else:
        args.output_dir = Path(args.output_dir)

    args.generated_root = get_roots().generated_root
    return args


def _path_arg_is_empty(value: object) -> bool:
    if value is None:
        return True
    return not str(value).strip()


def compute_realtime_pose_normalizer(args: argparse.Namespace) -> dict[str, object]:
    args = resolve_normalizer_paths(args)
    task_dir = resolve_latest_or_self(Path(args.task_dir), kind="tasks")
    output_root = Path(args.output_dir).resolve()
    output_dir = (
        output_root
        if bool(getattr(args, "direct_output", False))
        else timestamped_child_dir(output_root, resolve_normalizer_run_label(args))
    )
    args.output_dir = str(output_dir)
    schema = get_schema_spec(getattr(args, "schema", DEFAULT_REALTIME_POSE_SCHEMA_NAME))
    if str(args.split).lower() != "train":
        raise ValueError("Normalizer 只能从 train source-reference manifest 统计。")
    if not task_dir.exists():
        raise FileNotFoundError(f"{schema.name} task 目录不存在：{task_dir}")
    ensure_output_dir(output_dir=output_dir, overwrite=bool(args.overwrite))

    manifest_path = find_manifest_path(task_dir, args.split)
    task_entries = read_task_manifest(manifest_path)
    if not task_entries:
        raise RuntimeError(f"split={args.split} 没有匹配到 {schema.name} task。")

    windows_per_source = int(getattr(args, "windows_per_source", 2))
    convergence_windows = int(getattr(args, "convergence_windows_per_source", 4))
    if windows_per_source <= 0:
        raise ValueError("windows_per_source must be positive")
    if convergence_windows < windows_per_source:
        raise ValueError("convergence_windows_per_source must be >= windows_per_source")

    official = collect_normalizer_stats(
        task_dir=task_dir,
        split=str(args.split),
        schema_name=schema.name,
        tracker_mask_seed=int(getattr(args, "tracker_mask_seed", 10)),
        tracker_mask_epoch=0,
        windows_per_source=windows_per_source,
        label=f"K{windows_per_source}",
        eps=float(args.eps),
    )
    comparison = collect_normalizer_stats(
        task_dir=task_dir,
        split=str(args.split),
        schema_name=schema.name,
        tracker_mask_seed=int(getattr(args, "tracker_mask_seed", 10)),
        tracker_mask_epoch=0,
        windows_per_source=convergence_windows,
        label=f"K{convergence_windows}",
        eps=float(args.eps),
    )
    mean = official["mean"]
    std = official["std"]
    convergence = build_convergence_report(
        official_mean=mean,
        official_std=std,
        comparison_mean=comparison["mean"],
        comparison_std=comparison["std"],
        schema_name=schema.name,
        official_windows=windows_per_source,
        comparison_windows=convergence_windows,
        official_valid_counts=official["feature_valid_counts"],
        comparison_valid_counts=comparison["feature_valid_counts"],
        official_zero_count_channels=official["zero_count_channels"],
        comparison_zero_count_channels=comparison["zero_count_channels"],
    )
    convergence_path = output_dir / "normalizer_convergence.json"
    atomic_write_json(convergence_path, convergence)
    if bool(getattr(args, "check_convergence", True)) and not bool(convergence["passed"]):
        raise RuntimeError(f"Normalizer K{windows_per_source}/K{convergence_windows} 收敛门禁失败：{convergence_path}")

    normalizer = RealtimePoseNormalizer(base_dir=output_dir, eps=float(args.eps), disable=True, schema_name=schema.name)
    normalizer.save(mean=mean, std=std)

    meta = {
        "schema_name": schema.name,
        "schema_canonical_name": str(schema.canonical_name),
        "pose_representation": schema.pose_representation,
        "root_y_policy": schema.root_y_policy,
        "pelvis_height_mode": schema.pelvis_height_mode,
        "generated_root": str(Path(getattr(args, "generated_root", output_root))),
        "task_set_name": str(getattr(args, "task_set_name", DEFAULT_TASK_SET_NAME)),
        "normalizer_name": str(getattr(args, "normalizer_name", DEFAULT_NORMALIZER_NAME)),
        "task_dir": str(task_dir),
        "normalizer_root": str(output_root),
        "output_dir": str(output_dir),
        "split": args.split,
        "matched_sources": int(official["matched_sources"]),
        "windows_per_source": windows_per_source,
        "convergence_windows_per_source": convergence_windows,
        "normalizer_samples": int(official["sample_count"]),
        "tracker_mask_seed": int(getattr(args, "tracker_mask_seed", 10)),
        "sampling_epoch": 0,
        "source_cache_max_mib": 512,
        "total_frames": int(official["total_frames"]),
        "tracker_valid_observation_counts": official["tracker_valid_observation_counts"],
        "tracker_pattern_counts": official["tracker_pattern_counts"],
        "tracker_pattern_distribution": {
            key: float(value) / float(official["sample_count"])
            for key, value in sorted(official["tracker_pattern_counts"].items())
        },
        "feature_dim": schema.feature_dim,
        "eps": float(args.eps),
        "std_definition": "population",
        "task_manifest_path": str(manifest_path.resolve()),
        "task_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_manifest_path": str(official["source_manifest_path"]),
        "source_manifest_sha256": str(official["source_manifest_sha256"]),
        "normalizer_convergence_report": str(convergence_path),
        "normalizer_convergence_passed": bool(convergence["passed"]),
    }
    for key in RUNTIME_CONTRACT_METADATA_FIELDS:
        if key not in task_entries[0]:
            raise ValueError(f"task manifest entry missing runtime contract field {key!r}")
        meta[key] = task_entries[0][key]
    if schema.supports_stationary_prob:
        for key in STATIONARY_LABEL_METADATA_FIELDS:
            if key not in task_entries[0]:
                raise ValueError(f"task manifest entry missing stationary label field {key!r}")
            meta[key] = task_entries[0][key]
        validate_stationary_label_metadata(meta, source=str(manifest_path))
    meta["codec_reference_policy_hash"] = hashlib.sha256(
        (
            str(meta["tracker_codec_version"])
            + "|"
            + str(meta["reference_policy_version"])
            + "|"
            + str(meta["resolver_contract_version"])
        ).encode("utf-8")
    ).hexdigest()
    save_meta(output_dir=output_dir, meta=meta)
    latest_metadata = {
        "output_dir": str(output_dir),
        "normalizer_dir": str(output_dir),
        "normalizer_root": str(output_root),
        "task_dir": str(task_dir),
        "schema_name": schema.name,
        "schema_canonical_name": str(schema.canonical_name),
        "pose_representation": schema.pose_representation,
        "root_y_policy": schema.root_y_policy,
        "pelvis_height_mode": schema.pelvis_height_mode,
        "generated_root": str(Path(getattr(args, "generated_root", output_root))),
        "task_set_name": str(getattr(args, "task_set_name", DEFAULT_TASK_SET_NAME)),
        "normalizer_name": str(getattr(args, "normalizer_name", DEFAULT_NORMALIZER_NAME)),
        "split": args.split,
        "matched_sources": int(official["matched_sources"]),
    }
    if schema.supports_stationary_prob:
        latest_metadata.update(
            {key: meta[key] for key in STATIONARY_LABEL_METADATA_FIELDS}
        )
    write_latest_pointer(
        root_dir=output_root,
        kind="normalizer",
        output_dir=output_dir,
        metadata=latest_metadata,
    )
    return meta


def resolve_normalizer_run_label(args: argparse.Namespace) -> str:
    run_name = str(getattr(args, "run_name", "auto") or "auto").strip()
    if run_name.lower() in {"", "auto"}:
        return f"{getattr(args, 'schema', DEFAULT_REALTIME_POSE_SCHEMA_NAME)}_normalizer_{getattr(args, 'split', 'train')}"
    return run_name


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    mean_path = output_dir / "mean.pt"
    std_path = output_dir / "std.pt"
    meta_path = output_dir / "normalizer_meta.json"
    convergence_path = output_dir / "normalizer_convergence.json"
    existing = [path for path in (mean_path, std_path, meta_path, convergence_path) if path.exists()]
    if existing and not overwrite:
        existing_text = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"normalizer 输出已存在：{existing_text}。如需重算，请添加 --overwrite。")
    output_dir.mkdir(parents=True, exist_ok=True)


def collect_normalizer_stats(
    *,
    task_dir: Path,
    split: str,
    schema_name: str,
    tracker_mask_seed: int,
    tracker_mask_epoch: int,
    windows_per_source: int,
    label: str,
    eps: float,
) -> dict[str, object]:
    """按 source 等权采样固定 slot，并关闭训练数值增强后累计特征统计。"""

    schema = get_schema_spec(schema_name)
    dataset = RealtimePoseTaskDataset(
        data_dir=task_dir,
        split=split,
        normalize_input=False,
        preload_data=False,
        source_cache_max_mib=512,
        tracker_mask_policy="dynamic_categories",
        tracker_mask_seed=int(tracker_mask_seed),
        schema_name=schema.name,
        enable_rollout=False,
        rollout_steps=1,
        samples_per_source_override=int(windows_per_source),
    )
    # Normalizer 永远固定在 sampling epoch 0；该值只控制同一套确定性 mask。
    dataset.set_epoch(int(tracker_mask_epoch))
    if dataset.effective_sampling_epoch() != 0:
        raise ValueError("Normalizer 只能使用 sampling epoch 0。")

    running_sum = np.zeros(schema.feature_dim, dtype=np.float64)
    running_sumsq = np.zeros(schema.feature_dim, dtype=np.float64)
    running_count = np.zeros(schema.feature_dim, dtype=np.float64)
    tracker_valid_counts = np.zeros(TRACKER_COUNT, dtype=np.int64)
    pattern_counts: dict[str, int] = {}
    total_frames = 0
    for index in tqdm(range(len(dataset)), desc=f"normalizer {label}"):
        item = dataset[index]
        features = item["x"].detach().cpu().numpy().T
        sensor_valid = item["sensor_valid"].detach().cpu().numpy().T.astype(bool, copy=False)
        task_sum, task_sumsq, task_count = masked_task_feature_stats(
            features=features,
            sensor_valid=sensor_valid,
            schema_name=schema.name,
        )
        running_sum += task_sum
        running_sumsq += task_sumsq
        running_count += task_count
        tracker_valid_counts += sensor_valid.sum(axis=0, dtype=np.int64)
        total_frames += int(features.shape[0])
        pattern = str(item["tracker_pattern"])
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    mean, std = finalize_mean_std(
        running_sum=running_sum,
        running_sumsq=running_sumsq,
        running_count=running_count,
        eps=eps,
    )
    source_keys = {str(entry["stablemotion_split_key"]) for entry in dataset.entries}
    return {
        "mean": mean,
        "std": std,
        "feature_valid_counts": running_count,
        "zero_count_channels": np.flatnonzero(running_count <= 0).astype(int).tolist(),
        "finite": bool(np.isfinite(mean).all() and np.isfinite(std).all()),
        "sample_count": len(dataset),
        "matched_sources": len(source_keys),
        "total_frames": total_frames,
        "tracker_valid_observation_counts": tracker_valid_counts.astype(int).tolist(),
        "tracker_pattern_counts": dict(sorted(pattern_counts.items())),
        "source_manifest_path": str(dataset.task_marker["source_manifest_path"]),
        "source_manifest_sha256": str(dataset.task_marker["source_manifest_sha256"]),
    }


def build_convergence_report(
    *,
    official_mean: np.ndarray,
    official_std: np.ndarray,
    comparison_mean: np.ndarray,
    comparison_std: np.ndarray,
    schema_name: str,
    official_windows: int,
    comparison_windows: int,
    official_valid_counts: np.ndarray | None = None,
    comparison_valid_counts: np.ndarray | None = None,
    official_zero_count_channels: list[int] | None = None,
    comparison_zero_count_channels: list[int] | None = None,
) -> dict[str, object]:
    """比较 K2/K4；近常量通道使用绝对差，避免除以接近零的 std。"""

    schema = get_schema_spec(schema_name)
    official_mean64 = np.asarray(official_mean, dtype=np.float64)
    official_std64 = np.asarray(official_std, dtype=np.float64)
    comparison_mean64 = np.asarray(comparison_mean, dtype=np.float64)
    comparison_std64 = np.asarray(comparison_std, dtype=np.float64)
    arrays_finite = bool(
        np.isfinite(official_mean64).all()
        and np.isfinite(official_std64).all()
        and np.isfinite(comparison_mean64).all()
        and np.isfinite(comparison_std64).all()
    )

    special_mask = np.zeros(schema.feature_dim, dtype=bool)
    special_mask[schema.sensor_valid_slice()] = True
    if schema.supports_stationary_prob:
        special_mask[schema.stationary_prob_slice()] = True
    low_variance_mask = (comparison_std64 < 1e-4) & ~special_mask
    scaled_mask = ~special_mask & ~low_variance_mask

    normalized_mean_shift = np.abs(official_mean64 - comparison_mean64) / np.maximum(
        comparison_std64, 1e-4
    )
    relative_std_shift = np.abs(official_std64 - comparison_std64) / np.maximum(
        comparison_std64, 1e-4
    )
    absolute_mean_shift = np.abs(official_mean64 - comparison_mean64)
    absolute_std_shift = np.abs(official_std64 - comparison_std64)

    official_counts64 = _optional_feature_vector(
        official_valid_counts,
        feature_dim=schema.feature_dim,
        name="official_valid_counts",
    )
    comparison_counts64 = _optional_feature_vector(
        comparison_valid_counts,
        feature_dim=schema.feature_dim,
        name="comparison_valid_counts",
    )

    def masked_percentile(values: np.ndarray, mask: np.ndarray, percentile: float) -> float:
        selected = values[mask]
        return float(np.percentile(selected, percentile)) if selected.size else 0.0

    def masked_max(values: np.ndarray, mask: np.ndarray) -> float:
        selected = values[mask]
        return float(selected.max(initial=0.0)) if selected.size else 0.0

    metrics = {
        "normalized_mean_shift_p95": masked_percentile(normalized_mean_shift, scaled_mask, 95.0),
        "relative_std_shift_p95": masked_percentile(relative_std_shift, scaled_mask, 95.0),
        "normalized_mean_shift_max": masked_max(normalized_mean_shift, scaled_mask),
        "relative_std_shift_max": masked_max(relative_std_shift, scaled_mask),
        "low_variance_absolute_mean_shift_max": masked_max(absolute_mean_shift, low_variance_mask),
        "low_variance_absolute_std_shift_max": masked_max(absolute_std_shift, low_variance_mask),
        "stationary_sensor_valid_absolute_mean_shift_max": masked_max(absolute_mean_shift, special_mask),
    }
    thresholds = {
        "normalized_mean_shift_p95": 0.01,
        "relative_std_shift_p95": 0.01,
        "normalized_mean_std_shift_max": 0.05,
        "low_variance_absolute_mean_std_shift_max": 0.05,
        "stationary_sensor_valid_absolute_mean_shift_max": 0.01,
    }
    diagnostic_specs = {
        "normalized_mean_shift": (
            normalized_mean_shift,
            scaled_mask,
            "normalized_mean_shift",
            thresholds["normalized_mean_std_shift_max"],
        ),
        "relative_std_shift": (
            relative_std_shift,
            scaled_mask,
            "relative_std_shift",
            thresholds["normalized_mean_std_shift_max"],
        ),
        "low_variance_absolute_mean_shift": (
            absolute_mean_shift,
            low_variance_mask,
            "absolute_mean_shift",
            thresholds["low_variance_absolute_mean_std_shift_max"],
        ),
        "low_variance_absolute_std_shift": (
            absolute_std_shift,
            low_variance_mask,
            "absolute_std_shift",
            thresholds["low_variance_absolute_mean_std_shift_max"],
        ),
        "stationary_sensor_valid_absolute_mean_shift": (
            absolute_mean_shift,
            special_mask,
            "absolute_mean_shift",
            thresholds["stationary_sensor_valid_absolute_mean_shift_max"],
        ),
    }
    top_channel_diagnostics: dict[str, object] = {"top_k": CONVERGENCE_DIAGNOSTIC_TOP_K}
    for key, (values, mask, metric_name, max_threshold) in diagnostic_specs.items():
        top_channel_diagnostics[key] = _top_channel_shift_records(
            values=values,
            mask=mask,
            metric_name=metric_name,
            max_threshold=max_threshold,
            schema_name=schema.name,
            official_mean=official_mean64,
            official_std=official_std64,
            comparison_mean=comparison_mean64,
            comparison_std=comparison_std64,
            absolute_mean_shift=absolute_mean_shift,
            absolute_std_shift=absolute_std_shift,
            normalized_mean_shift=normalized_mean_shift,
            relative_std_shift=relative_std_shift,
            official_valid_counts=official_counts64,
            comparison_valid_counts=comparison_counts64,
            special_mask=special_mask,
            low_variance_mask=low_variance_mask,
        )
    failed_conditions: list[str] = []
    if not arrays_finite:
        failed_conditions.append("mean/std contains NaN or Inf")
    zero_count_channels = sorted(
        set(official_zero_count_channels or []) | set(comparison_zero_count_channels or [])
    )
    if zero_count_channels:
        failed_conditions.append(f"feature valid count is zero: {zero_count_channels}")
    if metrics["normalized_mean_shift_p95"] > thresholds["normalized_mean_shift_p95"]:
        failed_conditions.append("normalized mean shift P95 exceeds 0.01")
    if metrics["relative_std_shift_p95"] > thresholds["relative_std_shift_p95"]:
        failed_conditions.append("relative std shift P95 exceeds 0.01")
    if max(metrics["normalized_mean_shift_max"], metrics["relative_std_shift_max"]) > thresholds[
        "normalized_mean_std_shift_max"
    ]:
        failed_conditions.append("normalized mean/std shift max exceeds 0.05")
    if max(
        metrics["low_variance_absolute_mean_shift_max"],
        metrics["low_variance_absolute_std_shift_max"],
    ) > thresholds["low_variance_absolute_mean_std_shift_max"]:
        failed_conditions.append("low-variance absolute mean/std shift max exceeds 0.05")
    if metrics["stationary_sensor_valid_absolute_mean_shift_max"] > thresholds[
        "stationary_sensor_valid_absolute_mean_shift_max"
    ]:
        failed_conditions.append("stationary/sensor_valid absolute mean shift max exceeds 0.01")

    return {
        "schema_name": schema.name,
        "official_windows_per_source": int(official_windows),
        "comparison_windows_per_source": int(comparison_windows),
        "sampling_epoch": 0,
        "low_variance_std_threshold": 1e-4,
        "thresholds": thresholds,
        "metrics": metrics,
        "top_channel_diagnostics": top_channel_diagnostics,
        "zero_count_channels": zero_count_channels,
        "finite": arrays_finite,
        "failed_conditions": failed_conditions,
        "passed": not failed_conditions,
    }


def _optional_feature_vector(
    values: np.ndarray | None,
    *,
    feature_dim: int,
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (feature_dim,):
        raise ValueError(f"{name} shape 必须为 ({feature_dim},)，当前为 {array.shape}")
    return array


def _top_channel_shift_records(
    *,
    values: np.ndarray,
    mask: np.ndarray,
    metric_name: str,
    max_threshold: float,
    schema_name: str,
    official_mean: np.ndarray,
    official_std: np.ndarray,
    comparison_mean: np.ndarray,
    comparison_std: np.ndarray,
    absolute_mean_shift: np.ndarray,
    absolute_std_shift: np.ndarray,
    normalized_mean_shift: np.ndarray,
    relative_std_shift: np.ndarray,
    official_valid_counts: np.ndarray | None,
    comparison_valid_counts: np.ndarray | None,
    special_mask: np.ndarray,
    low_variance_mask: np.ndarray,
) -> list[dict[str, object]]:
    indices = np.flatnonzero(mask)
    ranked = sorted(
        (int(index) for index in indices),
        key=lambda index: (
            0 if np.isfinite(values[index]) else 1,
            float(values[index]) if np.isfinite(values[index]) else float("inf"),
            -index,
        ),
        reverse=True,
    )[:CONVERGENCE_DIAGNOSTIC_TOP_K]
    official_total = float(official_valid_counts.max(initial=0.0)) if official_valid_counts is not None else 0.0
    comparison_total = (
        float(comparison_valid_counts.max(initial=0.0)) if comparison_valid_counts is not None else 0.0
    )
    records: list[dict[str, object]] = []
    for index in ranked:
        feature_group, channel_name = _describe_feature_channel(schema_name, index)
        official_count = None if official_valid_counts is None else int(official_valid_counts[index])
        comparison_count = None if comparison_valid_counts is None else int(comparison_valid_counts[index])
        records.append(
            {
                "channel_index": index,
                "channel_name": channel_name,
                "feature_group": feature_group,
                "ranking_metric": metric_name,
                "ranking_value": _finite_float_or_none(values[index]),
                "exceeds_max_threshold": bool(
                    not np.isfinite(values[index]) or float(values[index]) > float(max_threshold)
                ),
                "official_mean": _finite_float_or_none(official_mean[index]),
                "comparison_mean": _finite_float_or_none(comparison_mean[index]),
                "official_std": _finite_float_or_none(official_std[index]),
                "comparison_std": _finite_float_or_none(comparison_std[index]),
                "absolute_mean_shift": _finite_float_or_none(absolute_mean_shift[index]),
                "normalized_mean_shift": _finite_float_or_none(normalized_mean_shift[index]),
                "absolute_std_shift": _finite_float_or_none(absolute_std_shift[index]),
                "relative_std_shift": _finite_float_or_none(relative_std_shift[index]),
                "official_valid_count": official_count,
                "comparison_valid_count": comparison_count,
                "official_valid_fraction": (
                    None if official_count is None or official_total <= 0.0 else float(official_count) / official_total
                ),
                "comparison_valid_fraction": (
                    None
                    if comparison_count is None or comparison_total <= 0.0
                    else float(comparison_count) / comparison_total
                ),
                "gate_class": (
                    "stationary_sensor_valid"
                    if bool(special_mask[index])
                    else "low_variance"
                    if bool(low_variance_mask[index])
                    else "scaled"
                ),
            }
        )
    return records


def _finite_float_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _describe_feature_channel(schema_name: str, channel_index: int) -> tuple[str, str]:
    schema = get_schema_spec(schema_name)
    index = int(channel_index)
    if not 0 <= index < schema.feature_dim:
        raise IndexError(f"feature channel 越界：{index}")

    if index in range(schema.body_pose_slice().start, schema.body_pose_slice().stop):
        offset = index - schema.body_pose_slice().start
        joint_index, component = divmod(offset, 6)
        joint_name = SMPL_JOINT_NAMES[joint_index]
        return schema.body_pose_key, f"{schema.body_pose_key}.{joint_name}.rot6d_{component}"
    if index in range(schema.root_heading_delta_slice().start, schema.root_heading_delta_slice().stop):
        component = ("sin", "cos")[index - schema.root_heading_delta_slice().start]
        return schema.root_heading_delta_key, f"{schema.root_heading_delta_key}.{component}"
    if schema.supports_root_motion and index in range(schema.root_delta_xz_slice().start, schema.root_delta_xz_slice().stop):
        component = ("x", "z")[index - schema.root_delta_xz_slice().start]
        return "root_delta_xz_ref", f"root_delta_xz_ref.{component}"
    if schema.supports_root_motion and index in range(schema.pelvis_height_slice().start, schema.pelvis_height_slice().stop):
        return schema.pelvis_height_key, f"{schema.pelvis_height_key}.y"
    if schema.supports_stationary_prob and index in range(
        schema.stationary_prob_slice().start,
        schema.stationary_prob_slice().stop,
    ):
        component = STATIONARY_JOINT_NAMES[index - schema.stationary_prob_slice().start]
        return "stationary_prob_5", f"stationary_prob_5.{component}"
    if index in range(schema.tracker_pos_slice().start, schema.tracker_pos_slice().stop):
        offset = index - schema.tracker_pos_slice().start
        tracker_index, component_index = divmod(offset, 3)
        component = ("x", "y", "z")[component_index]
        return "tracker_pos_ref", f"tracker_pos_ref.{TRACKER_NAMES[tracker_index]}.{component}"
    if index in range(schema.tracker_rot_slice().start, schema.tracker_rot_slice().stop):
        offset = index - schema.tracker_rot_slice().start
        tracker_index, component = divmod(offset, 6)
        return "tracker_rot_ref_6d", f"tracker_rot_ref_6d.{TRACKER_NAMES[tracker_index]}.rot6d_{component}"
    if index in range(schema.sensor_valid_slice().start, schema.sensor_valid_slice().stop):
        tracker_index = index - schema.sensor_valid_slice().start
        return "sensor_valid", f"sensor_valid.{TRACKER_NAMES[tracker_index]}"
    return "unknown", f"feature_{index}"


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    # Windows 长路径边界下不要把 `.json.tmp` 叠加到最终文件名。
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def masked_task_feature_stats(
    features: np.ndarray,
    sensor_valid: np.ndarray,
    schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """统计单个 task window；invalid tracker 的零填充值不进入 tracker pos/rot 的均值方差。"""

    schema = get_schema_spec(schema_name)
    mask = np.ones_like(features, dtype=bool)
    valid = np.asarray(sensor_valid, dtype=bool)
    for tracker_index in range(TRACKER_COUNT):
        missing = ~valid[:, tracker_index]
        if not missing.any():
            continue
        mask[missing, schema.tracker_pos_slice(tracker_index)] = False
        mask[missing, schema.tracker_rot_slice(tracker_index)] = False
    masked = features.astype(np.float64, copy=False) * mask.astype(np.float64)
    return (
        masked.sum(axis=0, dtype=np.float64),
        np.square(masked).sum(axis=0, dtype=np.float64),
        mask.sum(axis=0, dtype=np.float64),
    )


def finalize_mean_std(
    running_sum: np.ndarray,
    running_sumsq: np.ndarray,
    running_count: int | np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    count = np.asarray(running_count, dtype=np.float64)
    safe_count = np.maximum(count, 1.0)
    mean = running_sum / safe_count
    second_moment = running_sumsq / safe_count
    variance = np.maximum(second_moment - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.clip(std, a_min=eps, a_max=None)
    if count.shape:
        empty = count <= 0
        mean = np.where(empty, 0.0, mean)
        std = np.where(empty, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def save_meta(output_dir: Path, meta: dict[str, object]) -> None:
    meta_path = output_dir / "normalizer_meta.json"
    atomic_write_json(meta_path, meta)


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    meta = compute_realtime_pose_normalizer(args)
    print("[compute_realtime_pose_normalizer] 统计完成。")
    print(f"- 匹配 source 数：{meta['matched_sources']}")
    print(f"- 采样窗口数：{meta['normalizer_samples']}")
    print(f"- 累计有效帧数：{meta['total_frames']}")
    print(f"- 输出目录：{meta['output_dir']}")
    return meta


if __name__ == "__main__":
    main()
