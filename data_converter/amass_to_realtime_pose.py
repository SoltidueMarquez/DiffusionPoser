"""
把 AMASS SMPL/SMPL-H 动作转换为 realtime_pose 源数据。

默认生成当前主链路 `realtime_pose_v2_contact`。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from data_converter.amass_smpl_utils import (
    MIRROR_DIR_NAME,
    MotionSource,
    SmplModelCache,
    SmplMotion,
    iter_amass_motion_files,
    load_motion_source,
    mirror_motion_source,
    run_smpl_forward,
    validate_args as validate_shared_args,
    write_manifest_record,
)
from data_loaders.realtime_pose_kinematics import (
    JOINT_INDEX,
    TRACKER_JOINT_INDICES,
    build_body_pose_parent_6d,
    derive_foot_contact,
    encode_root_delta_xz_ref,
    estimate_parent_offsets,
    extract_yaw_from_rotations,
    rotation_6d_forward_up_np,
    wrap_radians,
)
from data_loaders.sensor_masking import (
    DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SCHEMA_NAMES,
    get_schema_spec,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AMASS SMPL motions to realtime_pose files.")
    parser.add_argument("--amass_dir", default="dataset/AMASS", type=Path)
    parser.add_argument("--smpl_model_dir", default="dataset/body_models", type=Path)
    parser.add_argument("--output_dir", default="dataset/AMASS_realtime_pose_v2_60hz", type=Path)
    parser.add_argument("--target_fps", default=60.0, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--rebuild_manifest", action="store_true")
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument("--reuse_source_dir", default="", type=Path)
    parser.add_argument("--schema", default=DEFAULT_REALTIME_POSE_SCHEMA_NAME, choices=REALTIME_POSE_SCHEMA_NAMES, type=str)
    # 复用 shared validate_args 所需字段；当前转换链路不实际使用这些可视化参数。
    parser.add_argument("--height_threshold", default=0.04, type=float)
    parser.add_argument("--speed_threshold", default=0.15, type=float)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualize_limit", default=0, type=int)
    parser.add_argument("--visualize_dir", default=Path("output/realtime_pose_visualization"), type=Path)
    parser.add_argument("--visualize_fps", default=20.0, type=float)
    return parser.parse_args(argv)


def output_path_for(source: MotionSource, output_dir: Path) -> Path:
    return output_dir / source.relative_path.with_suffix(".npz")


def build_realtime_pose_features(
    smpl_motion: SmplMotion,
    schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    target_fps: float = 60.0,
) -> dict[str, np.ndarray]:
    """构建 realtime_pose_v2 源字段。"""

    schema = get_schema_spec(schema_name)
    joint_positions = smpl_motion.joint_positions.astype(np.float64)
    joint_rotations = smpl_motion.joint_rotations.astype(np.float64)
    pelvis_world = joint_positions[:, JOINT_INDEX["pelvis"]].copy()
    root_pos_world = pelvis_world.copy()
    root_pos_world[:, 1] = 0.0
    root_yaw = extract_yaw_from_rotations(joint_rotations[:, JOINT_INDEX["pelvis"]]).astype(np.float32)
    yaw_delta = np.zeros_like(root_yaw, dtype=np.float32)
    if root_yaw.shape[0] > 1:
        yaw_delta[1:] = wrap_radians(root_yaw[1:].astype(np.float64) - root_yaw[:-1].astype(np.float64)).astype(np.float32)
    root_yaw_delta_sincos = np.stack([np.sin(yaw_delta), np.cos(yaw_delta)], axis=-1).astype(np.float32)

    body_pose_parent_6d = build_body_pose_parent_6d(
        global_rotations=joint_rotations,
        root_yaws=root_yaw.astype(np.float64),
    )
    tracker_pos_world = joint_positions[:, TRACKER_JOINT_INDICES].astype(np.float32)
    tracker_rot_world_6d = rotation_6d_forward_up_np(joint_rotations[:, TRACKER_JOINT_INDICES]).astype(np.float32)
    joints_world = joint_positions.astype(np.float32)
    joint_offsets_parent = estimate_parent_offsets(
        joints_world=joints_world,
        body_pose_parent_6d=body_pose_parent_6d,
        root_yaws=root_yaw,
        root_pos_world=root_pos_world,
    )

    features = {
        "body_pose_parent_6d": body_pose_parent_6d,
        "root_pos_world": root_pos_world.astype(np.float32),
        "root_yaw": root_yaw.astype(np.float32),
        "root_yaw_delta_sincos": root_yaw_delta_sincos,
        "tracker_pos_world": tracker_pos_world,
        "tracker_rot_world_6d": tracker_rot_world_6d,
        "joints_world": joints_world,
        "joint_offsets_parent": joint_offsets_parent,
    }
    if schema.supports_root_motion:
        features["root_delta_xz_ref"] = encode_root_delta_xz_ref(
            root_pos_world=root_pos_world.astype(np.float32),
            root_yaw=root_yaw.astype(np.float32),
        )
        features["root_height"] = pelvis_world[:, 1:2].astype(np.float32)
    if schema.supports_contact:
        features["foot_contact"] = derive_foot_contact(
            joints_world=joints_world,
            fps=float(target_fps),
            height_threshold=0.05,
            speed_threshold=0.05,
        )
    return features


def save_realtime_pose_motion(
    output_path: Path,
    features: dict[str, np.ndarray],
    source: MotionSource,
    target_fps: float,
    schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_name": schema_name,
        "source_path": str(source.path),
        "source_relative_path": str(source.relative_path),
        "original_source_relative_path": str(source.original_relative_path or source.relative_path),
        "stablemotion_split_key": str(source.relative_path.with_suffix(".npy")).replace("\\", "/"),
        "is_mirrored": bool(source.is_mirrored),
        "source_fps": float(source.source_fps),
        "target_fps": float(target_fps),
        "frames": int(features["body_pose_parent_6d"].shape[0]),
        "tracker_order": ["head", "left_wrist", "right_wrist", "waist", "left_foot", "right_foot"],
    }
    np.savez(output_path, **features, metadata=json.dumps(metadata, ensure_ascii=False))


def load_metadata_from_npz(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata" not in data.files:
        return {}
    value = data["metadata"]
    try:
        text = str(value.item())
    except Exception:
        text = str(value)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def source_relative_path_for(path: Path, args: argparse.Namespace, mirror_variant: bool) -> Path:
    relative_path = path.relative_to(args.amass_dir)
    return Path(MIRROR_DIR_NAME) / relative_path if mirror_variant else relative_path


def record_for_output(
    path: Path,
    output_path: Path,
    args: argparse.Namespace,
    mirror_variant: bool,
    status: str,
) -> dict[str, Any]:
    relative_path = source_relative_path_for(path=path, args=args, mirror_variant=mirror_variant)
    metadata: dict[str, Any] = {}
    frames = 0
    if output_path.exists():
        with np.load(output_path, allow_pickle=False) as data:
            metadata = load_metadata_from_npz(data)
            if "body_pose_parent_6d" in data.files:
                frames = int(data["body_pose_parent_6d"].shape[0])
    source_relative_path = Path(str(metadata.get("source_relative_path", relative_path)))
    stablemotion_key = str(metadata.get("stablemotion_split_key", source_relative_path.with_suffix(".npy"))).replace("\\", "/")
    return {
        "status": status,
        "schema_name": str(getattr(args, "schema", DEFAULT_REALTIME_POSE_SCHEMA_NAME)),
        "source_path": str(metadata.get("source_path", path)),
        "source_relative_path": str(source_relative_path),
        "original_source_relative_path": str(metadata.get("original_source_relative_path", path.relative_to(args.amass_dir))),
        "is_mirrored": bool(metadata.get("is_mirrored", mirror_variant)),
        "stablemotion_split_key": stablemotion_key,
        "output_path": str(output_path),
        "frames": int(metadata.get("frames", frames)),
    }


def reusable_source_path_for(path: Path, args: argparse.Namespace, mirror_variant: bool) -> Path | None:
    reuse_source_dir = Path(args.reuse_source_dir) if getattr(args, "reuse_source_dir", "") else None
    if reuse_source_dir is None:
        return None
    relative_path = source_relative_path_for(path=path, args=args, mirror_variant=mirror_variant)
    return reuse_source_dir / relative_path.with_suffix(".npz")


def load_reusable_realtime_features(
    reuse_path: Path,
    schema_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(reuse_path, allow_pickle=False) as data:
        required = required_source_fields(schema_name)
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"{reuse_path} 不能作为 realtime source 复用，缺少字段：{missing}")
        features = {key: np.asarray(data[key]).astype(np.float32, copy=True) for key in required}
        metadata = load_metadata_from_npz(data)
    return features, metadata


def required_source_fields(schema_name: str) -> set[str]:
    schema = get_schema_spec(schema_name)
    required = {
        "body_pose_parent_6d",
        "root_pos_world",
        "root_yaw",
        "root_yaw_delta_sincos",
        "tracker_pos_world",
        "tracker_rot_world_6d",
        "joints_world",
        "joint_offsets_parent",
    }
    if schema.supports_root_motion:
        required.update({"root_delta_xz_ref", "root_height"})
    if schema.supports_contact:
        required.add("foot_contact")
    return required


def realtime_source_has_schema(path: Path, schema_name: str) -> bool:
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as data:
        return required_source_fields(schema_name).issubset(data.files)


def save_reused_realtime_source(
    output_path: Path,
    features: dict[str, np.ndarray],
    metadata: dict[str, Any],
    path: Path,
    args: argparse.Namespace,
    mirror_variant: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    relative_path = source_relative_path_for(path=path, args=args, mirror_variant=mirror_variant)
    next_metadata = dict(metadata)
    next_metadata.update(
        {
            "schema_name": str(args.schema),
            "source_path": str(next_metadata.get("source_path", path)),
            "source_relative_path": str(next_metadata.get("source_relative_path", relative_path)),
            "original_source_relative_path": str(next_metadata.get("original_source_relative_path", path.relative_to(args.amass_dir))),
            "stablemotion_split_key": str(next_metadata.get("stablemotion_split_key", relative_path.with_suffix(".npy"))).replace("\\", "/"),
            "is_mirrored": bool(next_metadata.get("is_mirrored", mirror_variant)),
            "target_fps": float(args.target_fps),
            "frames": int(features["body_pose_parent_6d"].shape[0]),
            "tracker_order": ["head", "left_wrist", "right_wrist", "waist", "left_foot", "right_foot"],
        }
    )
    np.savez(output_path, **features, metadata=json.dumps(next_metadata, ensure_ascii=False))


def try_reuse_existing_realtime_source(
    path: Path,
    output_path: Path,
    args: argparse.Namespace,
    mirror_variant: bool,
) -> dict[str, Any] | None:
    reuse_path = reusable_source_path_for(path=path, args=args, mirror_variant=mirror_variant)
    if reuse_path is None or not reuse_path.exists():
        return None
    return reuse_realtime_source_file(
        reuse_path=reuse_path,
        output_path=output_path,
        path=path,
        args=args,
        mirror_variant=mirror_variant,
        status="reused_source",
    )


def reuse_realtime_source_file(
    reuse_path: Path,
    output_path: Path,
    path: Path,
    args: argparse.Namespace,
    mirror_variant: bool,
    status: str,
) -> dict[str, Any]:
    features, metadata = load_reusable_realtime_features(
        reuse_path=reuse_path,
        schema_name=str(args.schema),
    )
    save_reused_realtime_source(
        output_path=output_path,
        features=features,
        metadata=metadata,
        path=path,
        args=args,
        mirror_variant=mirror_variant,
    )
    return record_for_output(
        path=path,
        output_path=output_path,
        args=args,
        mirror_variant=mirror_variant,
        status=status,
    )


def convert_one_motion(
    path: Path,
    args: argparse.Namespace,
    model_cache: SmplModelCache,
    mirror_variant: bool = False,
) -> dict[str, Any]:
    relative_path = source_relative_path_for(path=path, args=args, mirror_variant=mirror_variant)
    output_path = args.output_dir / relative_path.with_suffix(".npz")

    if output_path.exists() and args.skip_existing:
        if realtime_source_has_schema(output_path, str(args.schema)):
            return record_for_output(
                path=path,
                output_path=output_path,
                args=args,
                mirror_variant=mirror_variant,
                status="skipped_existing",
            )
        if args.overwrite or args.rebuild_manifest:
            return reuse_realtime_source_file(
                reuse_path=output_path,
                output_path=output_path,
                path=path,
                args=args,
                mirror_variant=mirror_variant,
                status="upgraded_existing_source",
            )
        raise ValueError(f"已有 source 不满足 {args.schema}：{output_path}，请使用 --overwrite 或 --rebuild_manifest。")
    elif output_path.exists() and not args.overwrite:
        raise FileExistsError(f"输出文件已存在：{output_path}，请使用 --overwrite 或 --skip_existing。")

    reused_record = try_reuse_existing_realtime_source(
        path=path,
        output_path=output_path,
        args=args,
        mirror_variant=mirror_variant,
    )
    if reused_record is not None:
        return reused_record

    source = load_motion_source(path=path, amass_dir=args.amass_dir, target_fps=args.target_fps)
    if mirror_variant:
        source = mirror_motion_source(source)
    smpl_motion = run_smpl_forward(source=source, model_cache=model_cache, batch_size=args.batch_size)
    features = build_realtime_pose_features(
        smpl_motion,
        schema_name=str(getattr(args, "schema", DEFAULT_REALTIME_POSE_SCHEMA_NAME)),
        target_fps=float(args.target_fps),
    )
    save_realtime_pose_motion(
        output_path=output_path,
        features=features,
        source=source,
        target_fps=args.target_fps,
        schema_name=str(getattr(args, "schema", DEFAULT_REALTIME_POSE_SCHEMA_NAME)),
    )
    return record_for_output(
        path=path,
        output_path=output_path,
        args=args,
        mirror_variant=mirror_variant,
        status="converted",
    )


def main(argv: list[str] | None = None) -> dict[str, int]:
    args = parse_args(argv)
    validate_shared_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    if (args.overwrite or args.rebuild_manifest) and manifest_path.exists():
        manifest_path.unlink()

    motion_files = iter_amass_motion_files(args.amass_dir)
    if args.limit:
        motion_files = motion_files[: args.limit]
    model_cache = SmplModelCache(model_dir=args.smpl_model_dir)

    converted = reused = skipped = failed = 0
    failed_records: list[dict[str, Any]] = []
    for path in tqdm(motion_files, desc=f"Converting AMASS to {args.schema}"):
        mirror_variants = (False, True) if args.mirror else (False,)
        for mirror_variant in mirror_variants:
            try:
                record = convert_one_motion(path=path, args=args, model_cache=model_cache, mirror_variant=mirror_variant)
            except Exception as exc:
                failed += 1
                source_relative_path = path.relative_to(args.amass_dir)
                if mirror_variant:
                    source_relative_path = Path(MIRROR_DIR_NAME) / source_relative_path
                record = {
                    "status": "failed",
                    "schema_name": str(getattr(args, "schema", DEFAULT_REALTIME_POSE_SCHEMA_NAME)),
                    "source_path": str(path),
                    "source_relative_path": str(source_relative_path),
                    "stablemotion_split_key": str(source_relative_path.with_suffix(".npy")).replace("\\", "/"),
                    "is_mirrored": bool(mirror_variant),
                    "error": repr(exc),
                }
                failed_records.append(record)
            else:
                if record["status"] == "converted":
                    converted += 1
                elif record["status"] in {"reused_source", "upgraded_existing_source"}:
                    reused += 1
                else:
                    skipped += 1
            write_manifest_record(manifest_path, record)

    print(
        f"完成 AMASS -> {args.schema} 转换：converted={converted}, "
        f"reused={reused}, skipped={skipped}, failed={failed}, manifest={manifest_path}"
    )
    if failed_records and not args.allow_partial:
        preview = "; ".join(
            f"{record['source_relative_path']}: {record['error']}"
            for record in failed_records[:3]
        )
        raise RuntimeError(
            f"AMASS 转换存在 {len(failed_records)} 个失败样本，默认停止以避免下游使用部分数据。"
            f"示例：{preview}。如确认可接受部分数据，请添加 --allow_partial。"
        )
    return {"converted": converted, "reused": reused, "skipped": skipped, "failed": failed}


if __name__ == "__main__":
    main()
