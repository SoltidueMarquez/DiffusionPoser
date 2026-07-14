"""
把 GlobalPose 官方 test dataset 转成 DiffusionPoser realtime_pose source。

这个转换器只使用 GlobalPose `.pt` 里的 GT `pose/tran`，通过本仓库 SMPL FK
生成 oracle tracker 条件，用于验证 DiffusionPoser 在 GlobalPose/TotalCapture
动作分布上的流程和重建能力。它不是基于 IMU 观测的公平复现输入。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from data_converter.amass_smpl_utils import (
    MotionSource,
    SmplModelCache,
    SmplMotion,
    run_smpl_forward,
    write_manifest_record,
)
from data_converter.amass_to_realtime_pose import (
    DEFAULT_SMPL_MODEL_DIR,
    build_realtime_pose_features,
    resolve_body_fbx_rest_for_schema,
)
from data_loaders.realtime_pose_contract import (
    runtime_contract_metadata,
    validate_realtime_source_contract,
    validate_root_y0_invariants,
)
from data_loaders.sensor_masking import (
    DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    POSE_REPRESENTATION_KEY,
    REALTIME_POSE_SCHEMA_NAMES,
    get_schema_spec,
)
from data_loaders.stationary_label_config import stationary_label_metadata
from data_loaders.sensor_masking import STATIONARY_JOINT_INDICES, STATIONARY_JOINT_NAMES
from utils.artifact_paths import source_root
from utils.data_roots import load_data_roots


DEFAULT_TARGET_FPS = 60.0
DEFAULT_TRACKER_SOURCE = "oracle_gt_pose_tran"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert GlobalPose test dataset to realtime_pose oracle sources.")
    parser.add_argument("--data_roots_config", default="", type=str)
    parser.add_argument("--globalpose_dataset", required=True, type=str)
    parser.add_argument("--dataset_name", default="", type=str)
    parser.add_argument("--source_set_name", default="", type=str)
    parser.add_argument("--smpl_model_dir", default="", type=str)
    parser.add_argument("--body_fbx_rest_json", default="", type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--schema", default=DEFAULT_REALTIME_POSE_SCHEMA_NAME, choices=REALTIME_POSE_SCHEMA_NAMES, type=str)
    parser.add_argument("--target_fps", default=DEFAULT_TARGET_FPS, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--allow_partial", action="store_true")
    return parser.parse_args(argv)


def resolve_converter_paths(args: argparse.Namespace) -> argparse.Namespace:
    roots = None

    def get_roots():
        nonlocal roots
        if roots is None:
            roots = load_data_roots(getattr(args, "data_roots_config", "") or None)
        return roots

    args.globalpose_dataset = Path(args.globalpose_dataset)
    dataset_name = str(getattr(args, "dataset_name", "") or "").strip()
    if not dataset_name:
        dataset_name = args.globalpose_dataset.stem
    args.dataset_name = sanitize_path_stem(dataset_name)

    source_set_name = str(getattr(args, "source_set_name", "") or "").strip()
    if not source_set_name:
        source_set_name = f"globalpose_{args.dataset_name}_oracle_tracker"
    args.source_set_name = source_set_name

    if _path_arg_is_empty(getattr(args, "smpl_model_dir", "")):
        if _path_arg_is_empty(getattr(args, "data_roots_config", "")):
            args.smpl_model_dir = DEFAULT_SMPL_MODEL_DIR
        else:
            roots_value = get_roots().smpl_model_dir
            args.smpl_model_dir = roots_value if roots_value is not None else DEFAULT_SMPL_MODEL_DIR
    else:
        args.smpl_model_dir = Path(args.smpl_model_dir)

    if _path_arg_is_empty(getattr(args, "body_fbx_rest_json", "")):
        if _path_arg_is_empty(getattr(args, "data_roots_config", "")):
            args.body_fbx_rest_json = ""
        else:
            roots_value = get_roots().body_fbx_rest_json
            args.body_fbx_rest_json = roots_value if roots_value is not None else ""
    else:
        args.body_fbx_rest_json = Path(args.body_fbx_rest_json)

    if _path_arg_is_empty(getattr(args, "output_dir", "")):
        args.output_dir = source_root(
            get_roots(),
            schema_name=str(args.schema),
            source_set_name=str(args.source_set_name),
        )
    else:
        args.output_dir = Path(args.output_dir)
    return args


def _path_arg_is_empty(value: object) -> bool:
    if value is None:
        return True
    return not str(value).strip()


def validate_converter_args(args: argparse.Namespace) -> None:
    if not Path(args.globalpose_dataset).exists():
        raise FileNotFoundError(f"GlobalPose dataset 不存在: {args.globalpose_dataset}")
    if not Path(args.smpl_model_dir).exists():
        raise FileNotFoundError(f"SMPL 模型目录不存在: {args.smpl_model_dir}")
    if float(args.target_fps) <= 0.0:
        raise ValueError("--target_fps 必须为正数")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch_size 必须为正整数")
    if bool(args.overwrite) and bool(args.skip_existing):
        raise ValueError("--overwrite 和 --skip_existing 不能同时启用")


def convert_globalpose_dataset(
    args: argparse.Namespace,
    body_fbx_rest=None,
    model_cache: SmplModelCache | None = None,
) -> dict[str, int]:
    args = resolve_converter_paths(args)
    validate_converter_args(args)
    schema = get_schema_spec(str(args.schema))
    body_fbx_rest = resolve_body_fbx_rest_for_schema(args) if body_fbx_rest is None else body_fbx_rest
    model_cache = SmplModelCache(model_dir=Path(args.smpl_model_dir)) if model_cache is None else model_cache
    dataset = load_globalpose_dataset(Path(args.globalpose_dataset))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    counts = {"converted": 0, "skipped_existing": 0, "failed": 0}
    sequence_total = len(dataset["pose"])
    limit = int(getattr(args, "limit", 0) or 0)
    if limit > 0:
        sequence_total = min(sequence_total, limit)

    for sequence_index in range(sequence_total):
        try:
            source = build_sequence_source(dataset=dataset, sequence_index=sequence_index, args=args)
            output_path = output_dir / source.relative_path
            if output_path.exists() and bool(args.skip_existing):
                record = build_manifest_record(
                    args=args,
                    source=source,
                    output_path=output_path,
                    status="skipped_existing",
                    frames=source.poses.shape[0],
                )
                write_manifest_record(manifest_path, record)
                counts["skipped_existing"] += 1
                continue
            if output_path.exists() and not bool(args.overwrite):
                raise FileExistsError(f"输出文件已存在: {output_path}")

            smpl_motion = build_smpl_motion_from_globalpose_sequence(
                pose_axis_angle=source.poses,
                tran=source.trans,
                source=source,
                model_cache=model_cache,
                batch_size=int(args.batch_size),
            )
            features = build_realtime_pose_features(
                smpl_motion=smpl_motion,
                schema_name=schema.name,
                target_fps=float(args.target_fps),
                body_fbx_rest=body_fbx_rest,
            )
            save_globalpose_realtime_source(
                output_path=output_path,
                features=features,
                source=source,
                args=args,
            )
            record = build_manifest_record(
                args=args,
                source=source,
                output_path=output_path,
                status="converted",
                frames=features[schema.body_pose_key].shape[0],
            )
            write_manifest_record(manifest_path, record)
            counts["converted"] += 1
        except Exception:
            counts["failed"] += 1
            if not bool(args.allow_partial):
                raise
    return counts


def load_globalpose_dataset(path: Path) -> dict[str, Any]:
    import torch

    dataset = torch.load(path, map_location="cpu")
    if not isinstance(dataset, dict):
        raise ValueError(f"GlobalPose dataset 必须是 dict: {path}")
    missing = [key for key in ("pose", "tran") if key not in dataset]
    if missing:
        raise KeyError(f"{path} 缺少字段: {missing}")
    if len(dataset["pose"]) != len(dataset["tran"]):
        raise ValueError(f"{path} pose/tran 序列数量不一致")
    return dataset


def build_sequence_source(dataset: dict[str, Any], sequence_index: int, args: argparse.Namespace) -> MotionSource:
    pose = tensor_to_numpy(dataset["pose"][sequence_index]).astype(np.float64)
    tran = tensor_to_numpy(dataset["tran"][sequence_index]).astype(np.float64)
    if pose.ndim != 2 or pose.shape[1] != 72:
        raise ValueError(f"GlobalPose pose 应为 [T,72]，实际为 {pose.shape}")
    if tran.shape != (pose.shape[0], 3):
        raise ValueError(f"GlobalPose tran 应为 [T,3] 且与 pose 同帧数，实际为 {tran.shape}")
    sequence_name = sequence_name_at(dataset, sequence_index)
    sequence_stem = sanitize_path_stem(sequence_name)
    relative_path = Path(str(args.dataset_name)) / f"{sequence_stem}.npz"
    source = MotionSource(
        path=Path(args.globalpose_dataset),
        relative_path=relative_path,
        poses=pose,
        trans=tran,
        betas=np.zeros(10, dtype=np.float64),
        gender="male",
        source_fps=float(args.target_fps),
        is_mirrored=False,
        original_relative_path=Path(str(args.globalpose_dataset).replace("\\", "/")) / sequence_name,
    )
    return source


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def sequence_name_at(dataset: dict[str, Any], sequence_index: int) -> str:
    names = dataset.get("name")
    if names is None:
        return f"seq_{sequence_index:04d}"
    value = names[sequence_index]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        value = value.item()
    return str(value)


def sanitize_path_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    stem = stem.strip("._-")
    return stem or "sequence"


def build_smpl_motion_from_globalpose_sequence(
    *,
    pose_axis_angle: np.ndarray,
    tran: np.ndarray,
    source: MotionSource,
    model_cache: SmplModelCache,
    batch_size: int,
) -> SmplMotion:
    motion_source = MotionSource(
        path=source.path,
        relative_path=source.relative_path,
        poses=pose_axis_angle,
        trans=tran,
        betas=source.betas,
        gender=source.gender,
        source_fps=source.source_fps,
        is_mirrored=source.is_mirrored,
        original_relative_path=source.original_relative_path,
    )
    return run_smpl_forward(motion_source, model_cache=model_cache, batch_size=batch_size)


def save_globalpose_realtime_source(
    output_path: Path,
    features: dict[str, np.ndarray],
    source: MotionSource,
    args: argparse.Namespace,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = get_schema_spec(str(args.schema))
    metadata = build_source_metadata(args=args, source=source, features=features)
    validate_root_y0_invariants(features, schema=schema, source=str(output_path))
    payload = dict(features)
    payload["metadata"] = json.dumps(metadata, ensure_ascii=False)
    np.savez(output_path, **payload)
    with np.load(output_path, allow_pickle=False) as data:
        validate_realtime_source_contract(data, schema=schema, source=str(output_path))


def build_source_metadata(args: argparse.Namespace, source: MotionSource, features: dict[str, np.ndarray]) -> dict[str, Any]:
    schema = get_schema_spec(str(args.schema))
    sequence_name = source.relative_path.stem
    raw_relative_path = f"{Path(args.globalpose_dataset).name}:{sequence_name}"
    stablemotion_split_key = stablemotion_key(args=args, source=source)
    metadata: dict[str, Any] = {
        "schema_name": schema.name,
        "schema_canonical_name": str(schema.canonical_name),
        "pose_representation": schema.pose_representation,
        "root_y_policy": schema.root_y_policy,
        "pelvis_height_mode": schema.pelvis_height_mode,
        "source_path": str(args.globalpose_dataset),
        "source_relative_path": source.relative_path.as_posix(),
        "original_source_relative_path": raw_relative_path,
        "stablemotion_split_key": stablemotion_split_key,
        "is_mirrored": False,
        "source_fps": float(args.target_fps),
        "target_fps": float(args.target_fps),
        "frames": int(features[schema.body_pose_key].shape[0]),
        "tracker_order": ["head", "left_wrist", "right_wrist", "waist", "left_foot", "right_foot"],
        "tracker_source": DEFAULT_TRACKER_SOURCE,
        "raw_dataset": "GlobalPose",
        "raw_root_key": "globalpose_dataset",
        "raw_relative_path": raw_relative_path,
        "source_set_name": str(args.source_set_name),
        "converter_args": build_converter_args_metadata(args),
        "globalpose_dataset_path": str(args.globalpose_dataset),
        "globalpose_dataset_name": str(args.dataset_name),
        "globalpose_sequence_name": sequence_name,
    }
    metadata.update(runtime_contract_metadata())
    if schema.supports_stationary_prob:
        metadata["stationary_joint_indices"] = [int(index) for index in STATIONARY_JOINT_INDICES]
        metadata["stationary_joint_names"] = list(STATIONARY_JOINT_NAMES)
        metadata.update(stationary_label_metadata())
    if "body_fbx_rest_json" in features:
        metadata["body_fbx_rest_json"] = str(features["body_fbx_rest_json"].item())
    return metadata


def build_converter_args_metadata(args: argparse.Namespace) -> dict[str, object]:
    return {
        "target_fps": float(args.target_fps),
        "schema": str(args.schema),
        "source_set_name": str(args.source_set_name),
        "dataset_name": str(args.dataset_name),
        "tracker_source": DEFAULT_TRACKER_SOURCE,
    }


def build_manifest_record(
    args: argparse.Namespace,
    source: MotionSource,
    output_path: Path,
    status: str,
    frames: int,
) -> dict[str, Any]:
    schema = get_schema_spec(str(args.schema))
    sequence_name = source.relative_path.stem
    raw_relative_path = f"{Path(args.globalpose_dataset).name}:{sequence_name}"
    record = {
        "status": status,
        "schema_name": schema.name,
        "schema_canonical_name": str(schema.canonical_name),
        POSE_REPRESENTATION_KEY: schema.pose_representation,
        "root_y_policy": schema.root_y_policy,
        "pelvis_height_mode": schema.pelvis_height_mode,
        "source_path": str(args.globalpose_dataset),
        "source_relative_path": source.relative_path.as_posix(),
        "original_source_relative_path": raw_relative_path,
        "is_mirrored": False,
        "stablemotion_split_key": stablemotion_key(args=args, source=source),
        "output_path": str(output_path),
        "frames": int(frames),
        "tracker_source": DEFAULT_TRACKER_SOURCE,
        "raw_dataset": "GlobalPose",
        "raw_root_key": "globalpose_dataset",
        "raw_relative_path": raw_relative_path,
        "source_set_name": str(args.source_set_name),
        "converter_args": build_converter_args_metadata(args),
        "globalpose_dataset_path": str(args.globalpose_dataset),
        "globalpose_dataset_name": str(args.dataset_name),
        "globalpose_sequence_name": sequence_name,
    }
    if schema.supports_stationary_prob:
        record.update(stationary_label_metadata())
    return record


def stablemotion_key(args: argparse.Namespace, source: MotionSource) -> str:
    return f"GlobalPose/{args.dataset_name}/{source.relative_path.stem}"


def main(argv: list[str] | None = None) -> dict[str, int]:
    args = parse_args(argv)
    return convert_globalpose_dataset(args)


if __name__ == "__main__":
    counts = main()
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))
