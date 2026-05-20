"""
把 AMASS SMPL/SMPL-H 动作转换为 `realtime_pose_v1` 源数据。

输出不包含旧 277 维格式中的 body velocity / contact 通道；contact 在训练和运行时由
`joints_world` 动态派生。
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
    estimate_parent_offsets,
    extract_yaw_from_rotations,
    rotation_6d_forward_up_np,
    wrap_radians,
)
from data_loaders.sensor_masking import REALTIME_POSE_SCHEMA_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AMASS SMPL motions to realtime_pose_v1 files.")
    parser.add_argument("--amass_dir", default="dataset/AMASS", type=Path)
    parser.add_argument("--smpl_model_dir", default="dataset/body_models", type=Path)
    parser.add_argument("--output_dir", default="dataset/AMASS_realtime_pose_60hz", type=Path)
    parser.add_argument("--target_fps", default=60.0, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--allow_partial", action="store_true")
    # 复用旧 validate_args 所需字段；新 schema 不实际使用 contact/visualize 参数。
    parser.add_argument("--height_threshold", default=0.04, type=float)
    parser.add_argument("--speed_threshold", default=0.15, type=float)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualize_limit", default=0, type=int)
    parser.add_argument("--visualize_dir", default=Path("output/realtime_pose_visualization"), type=Path)
    parser.add_argument("--visualize_fps", default=20.0, type=float)
    return parser.parse_args()


def output_path_for(source: MotionSource, output_dir: Path) -> Path:
    return output_dir / source.relative_path.with_suffix(".npz")


def build_realtime_pose_features(smpl_motion: SmplMotion) -> dict[str, np.ndarray]:
    joint_positions = smpl_motion.joint_positions.astype(np.float64)
    joint_rotations = smpl_motion.joint_rotations.astype(np.float64)
    root_pos_world = joint_positions[:, JOINT_INDEX["pelvis"]].copy()
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
    )

    return {
        "body_pose_parent_6d": body_pose_parent_6d,
        "root_pos_world": root_pos_world.astype(np.float32),
        "root_yaw": root_yaw.astype(np.float32),
        "root_yaw_delta_sincos": root_yaw_delta_sincos,
        "tracker_pos_world": tracker_pos_world,
        "tracker_rot_world_6d": tracker_rot_world_6d,
        "joints_world": joints_world,
        "joint_offsets_parent": joint_offsets_parent,
    }


def save_realtime_pose_motion(output_path: Path, features: dict[str, np.ndarray], source: MotionSource, target_fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_name": REALTIME_POSE_SCHEMA_NAME,
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


def convert_one_motion(
    path: Path,
    args: argparse.Namespace,
    model_cache: SmplModelCache,
    mirror_variant: bool = False,
) -> dict[str, Any]:
    source = load_motion_source(path=path, amass_dir=args.amass_dir, target_fps=args.target_fps)
    if mirror_variant:
        source = mirror_motion_source(source)
    output_path = output_path_for(source, args.output_dir)

    if output_path.exists() and args.skip_existing:
        status = "skipped_existing"
        with np.load(output_path, allow_pickle=False) as data:
            frames = int(data["body_pose_parent_6d"].shape[0])
    elif output_path.exists() and not args.overwrite:
        raise FileExistsError(f"输出文件已存在：{output_path}，请使用 --overwrite 或 --skip_existing。")
    else:
        status = "converted"
        smpl_motion = run_smpl_forward(source=source, model_cache=model_cache, batch_size=args.batch_size)
        features = build_realtime_pose_features(smpl_motion)
        save_realtime_pose_motion(
            output_path=output_path,
            features=features,
            source=source,
            target_fps=args.target_fps,
        )
        frames = int(features["body_pose_parent_6d"].shape[0])

    return {
        "status": status,
        "schema_name": REALTIME_POSE_SCHEMA_NAME,
        "source_path": str(path),
        "source_relative_path": str(source.relative_path),
        "original_source_relative_path": str(source.original_relative_path or source.relative_path),
        "is_mirrored": source.is_mirrored,
        "stablemotion_split_key": str(source.relative_path.with_suffix(".npy")).replace("\\", "/"),
        "output_path": str(output_path),
        "frames": frames,
    }


def main() -> None:
    args = parse_args()
    validate_shared_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    if args.overwrite and manifest_path.exists():
        manifest_path.unlink()

    motion_files = iter_amass_motion_files(args.amass_dir)
    if args.limit:
        motion_files = motion_files[: args.limit]
    model_cache = SmplModelCache(model_dir=args.smpl_model_dir)

    converted = skipped = failed = 0
    failed_records: list[dict[str, Any]] = []
    for path in tqdm(motion_files, desc="Converting AMASS to realtime_pose_v1"):
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
                    "schema_name": REALTIME_POSE_SCHEMA_NAME,
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
                else:
                    skipped += 1
            write_manifest_record(manifest_path, record)

    print(
        f"完成 AMASS -> realtime_pose_v1 转换：converted={converted}, "
        f"skipped={skipped}, failed={failed}, manifest={manifest_path}"
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


if __name__ == "__main__":
    main()
