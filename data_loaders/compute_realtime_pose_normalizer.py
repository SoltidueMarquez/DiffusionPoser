from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from data_loaders.realtime_pose_dataset import (
    encode_realtime_pose_features,
    find_manifest_path,
    load_materialized_task_npz,
    load_realtime_task_arrays,
    read_task_manifest,
)
from data_loaders.sensor_masking import (
    DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    POSE_REPRESENTATION_KEY,
    REALTIME_POSE_SCHEMA_NAMES,
    REALTIME_POSE_SEQ_LEN,
    TRACKER_COUNT,
    get_schema_spec,
    validate_pose_representation,
)
from utils.artifact_paths import normalizer_root, task_root
from utils.data_roots import load_data_roots
from utils.normalizer import RealtimePoseNormalizer
from utils.run_dirs import resolve_latest_or_self, timestamped_child_dir, write_latest_pointer


DEFAULT_TASK_SET_NAME = "amass_60hz_tasks"
DEFAULT_NORMALIZER_NAME = "amass_60hz_train"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute realtime_pose mean/std normalizer from materialized tasks.")
    group = parser.add_argument_group("paths")
    group.add_argument("--data_roots_config", default="", type=str)
    group.add_argument("--task_set_name", default=DEFAULT_TASK_SET_NAME, type=str)
    group.add_argument("--normalizer_name", default=DEFAULT_NORMALIZER_NAME, type=str)
    group.add_argument("--task_dir", default="", type=str)
    group.add_argument("--output_dir", default="", type=str)

    group = parser.add_argument_group("statistics")
    group.add_argument("--schema", default=DEFAULT_REALTIME_POSE_SCHEMA_NAME, choices=REALTIME_POSE_SCHEMA_NAMES, type=str)
    group.add_argument("--split", default="train", type=str)
    group.add_argument("--eps", default=1e-8, type=float)
    group.add_argument("--run_name", default="auto", type=str)
    group.add_argument("--overwrite", action="store_true")
    return parser


def resolve_normalizer_paths(args: argparse.Namespace) -> argparse.Namespace:
    """把空路径参数解析到 schema-aware task/normalizer 根目录。"""

    roots = None

    def get_roots():
        nonlocal roots
        if roots is None:
            roots = load_data_roots(getattr(args, "data_roots_config", "") or None)
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

    if roots is not None or not _path_arg_is_empty(getattr(args, "data_roots_config", "")):
        args.generated_root = get_roots().generated_root
    else:
        # 旧式显式路径没有 data_roots 来源，用 normalizer root 作为可追踪的产物根记录。
        args.generated_root = Path(args.output_dir)
    return args


def _path_arg_is_empty(value: object) -> bool:
    if value is None:
        return True
    return not str(value).strip()


def compute_realtime_pose_normalizer(args: argparse.Namespace) -> dict[str, object]:
    args = resolve_normalizer_paths(args)
    task_dir = resolve_latest_or_self(Path(args.task_dir), kind="tasks")
    output_root = Path(args.output_dir).resolve()
    output_dir = timestamped_child_dir(output_root, resolve_normalizer_run_label(args))
    args.output_dir = str(output_dir)
    schema = get_schema_spec(getattr(args, "schema", DEFAULT_REALTIME_POSE_SCHEMA_NAME))
    if not task_dir.exists():
        raise FileNotFoundError(f"{schema.name} task 目录不存在：{task_dir}")
    ensure_output_dir(output_dir=output_dir, overwrite=bool(args.overwrite))

    manifest_path = find_manifest_path(task_dir, args.split)
    task_entries = read_task_manifest(manifest_path)
    if not task_entries:
        raise RuntimeError(f"split={args.split} 没有匹配到 {schema.name} task。")

    running_sum: np.ndarray | None = None
    running_sumsq: np.ndarray | None = None
    running_count: np.ndarray | None = None
    total_frames = 0
    tracker_valid_observation_counts = np.zeros(TRACKER_COUNT, dtype=np.int64)

    for entry in tqdm(task_entries, desc=f"统计 split={args.split} realtime normalizer", unit="task"):
        entry_schema = str(entry.get("schema_name", schema.name))
        if entry_schema != schema.name:
            raise ValueError(f"task {entry.get('task_id', '<unknown>')} schema_name={entry_schema}，期望 {schema.name}")
        validate_pose_representation(
            entry.get(POSE_REPRESENTATION_KEY),
            schema_name=schema.name,
            source=f"{manifest_path}:{entry.get('task_id', '<unknown>')}",
        )
        task = load_materialized_task_npz(manifest_dir=manifest_path.parent, task_path=entry["task_path"], schema_name=schema.name)
        arrays = load_realtime_task_arrays(task=task, seq_len=REALTIME_POSE_SEQ_LEN, schema_name=schema.name)
        features = encode_realtime_pose_features(arrays, schema_name=schema.name)
        seq_sum, seq_sumsq, seq_count = masked_task_feature_stats(features=features, sensor_valid=arrays["sensor_valid"], schema_name=schema.name)
        running_sum = seq_sum if running_sum is None else running_sum + seq_sum
        running_sumsq = seq_sumsq if running_sumsq is None else running_sumsq + seq_sumsq
        running_count = seq_count if running_count is None else running_count + seq_count
        total_frames += int(features.shape[0])
        tracker_valid_observation_counts += arrays["sensor_valid"].sum(axis=0).astype(np.int64)

    if running_sum is None or running_sumsq is None or running_count is None or total_frames <= 0:
        raise RuntimeError("没有成功统计到有效帧，无法生成 realtime_pose normalizer。")

    mean, std = finalize_mean_std(running_sum, running_sumsq, running_count, eps=float(args.eps))
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
        "matched_tasks": len(task_entries),
        "total_frames": total_frames,
        "tracker_valid_observation_counts": tracker_valid_observation_counts.astype(int).tolist(),
        "feature_dim": schema.feature_dim,
        "eps": float(args.eps),
        "std_definition": "population",
    }
    save_meta(output_dir=output_dir, meta=meta)
    write_latest_pointer(
        root_dir=output_root,
        kind="normalizer",
        output_dir=output_dir,
        metadata={
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
            "matched_tasks": len(task_entries),
        },
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
    existing = [path for path in (mean_path, std_path, meta_path) if path.exists()]
    if existing and not overwrite:
        existing_text = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"normalizer 输出已存在：{existing_text}。如需重算，请添加 --overwrite。")
    output_dir.mkdir(parents=True, exist_ok=True)


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
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2, ensure_ascii=False, sort_keys=True)


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    meta = compute_realtime_pose_normalizer(args)
    print("[compute_realtime_pose_normalizer] 统计完成。")
    print(f"- 匹配 task 数：{meta['matched_tasks']}")
    print(f"- 累计有效帧数：{meta['total_frames']}")
    print(f"- 输出目录：{meta['output_dir']}")
    return meta


if __name__ == "__main__":
    main()
