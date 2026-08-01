"""
把 AMASS SMPL/SMPL-H 动作转换为 realtime_pose 源数据。
输出字段与维度以仓库根目录的 `contract.md` 为准。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from data_converter.amass_smpl_utils import (
    MIRROR_DIR_NAME,
    MotionSource,
    ShortMotionError,
    SmplModelCache,
    SmplMotion,
    iter_amass_motion_files,
    load_motion_source,
    mirror_motion_source,
    run_smpl_forward,
    validate_args as validate_shared_args,
    write_manifest_record,
)
from data_loaders.body_fbx_kinematics import (
    BodyFbxRest,
    actor_root_positions_from_pelvis,
    extract_root_heading_from_source_pelvis_up,
    fk_body_fbx_local_delta_root_y0,
    load_body_fbx_rest,
    source_global_rotations_to_body_fbx_local_delta_6d,
)
from data_loaders.realtime_pose_kinematics import (
    JOINT_INDEX,
    derive_stationary_prob_5,
    encode_root_delta_xz_ref,
    rotation_6d_forward_up_np,
    wrap_radians,
)
from data_loaders.realtime_pose_validation import (
    load_realtime_metadata,
    validate_realtime_source_arrays,
)
from data_loaders.sensor_masking import (
    BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY,
    STATIONARY_JOINT_INDICES,
    STATIONARY_JOINT_NAMES,
)


REUSABLE_SOURCE_FIELDS = {
    BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY,
    "root_pos_world",
    "root_yaw",
    "root_heading_delta_sincos",
    "root_delta_xz_ref",
    "pelvis_height",
    "tracker_pos_world",
    "tracker_rot_world_6d",
    "joints_world",
    "joint_offsets_parent",
    "joint_rest_local_rotations_6d",
    "stationary_prob_5",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AMASS SMPL motions to realtime_pose files.")
    parser.add_argument("--amass_dir", default="dataset/AMASS", type=Path)
    parser.add_argument("--smpl_model_dir", default="dataset/body_models", type=Path)
    parser.add_argument(
        "--output_dir",
        default="dataset/AMASS_realtime_pose_body_fbx_local_pelvis_residual_root_y0_stationary5_60hz",
        type=Path,
    )
    parser.add_argument("--target_fps", default=60.0, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--worker_torch_threads", default=1, type=int)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--rebuild_manifest", action="store_true")
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument("--reuse_source_dir", default="", type=Path)
    parser.add_argument("--body_fbx_rest_json", default="", type=Path)
    # 复用 shared validate_args 所需字段；当前转换链路不实际使用这些可视化参数。
    parser.add_argument("--height_threshold", default=0.04, type=float)
    parser.add_argument("--speed_threshold", default=0.15, type=float)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualize_limit", default=0, type=int)
    parser.add_argument("--visualize_dir", default=Path("output/realtime_pose_visualization"), type=Path)
    parser.add_argument("--visualize_fps", default=20.0, type=float)
    return parser.parse_args(argv)


_WORKER_ARGS: argparse.Namespace | None = None
_WORKER_MODEL_CACHE: SmplModelCache | None = None


def validate_converter_args(args: argparse.Namespace) -> None:
    validate_shared_args(args)
    if int(args.num_workers) <= 0:
        raise ValueError("--num_workers 必须为正整数")
    if int(args.worker_torch_threads) < 0:
        raise ValueError("--worker_torch_threads 必须大于等于 0")


def build_conversion_work_items(args: argparse.Namespace, motion_files: list[Path]) -> list[tuple[Path, bool]]:
    mirror_variants = (False, True) if args.mirror else (False,)
    return [(path, mirror_variant) for path in motion_files for mirror_variant in mirror_variants]


def copy_args_for_worker(args: argparse.Namespace) -> argparse.Namespace:
    payload = dict(vars(args))
    payload.pop("_body_fbx_rest", None)
    return argparse.Namespace(**payload)


def init_conversion_worker(args: argparse.Namespace, torch_threads: int) -> None:
    global _WORKER_ARGS, _WORKER_MODEL_CACHE
    _WORKER_ARGS = args
    _WORKER_MODEL_CACHE = SmplModelCache(model_dir=args.smpl_model_dir)
    configure_worker_torch_threads(torch_threads)


def configure_worker_torch_threads(torch_threads: int) -> None:
    if int(torch_threads) <= 0:
        return
    import torch

    # Limit per-process Torch threads to avoid CPU oversubscription.
    torch.set_num_threads(int(torch_threads))
    try:
        torch.set_num_interop_threads(max(1, int(torch_threads)))
    except RuntimeError:
        pass


def convert_work_item_in_worker(item: tuple[Path, bool]) -> dict[str, Any]:
    global _WORKER_MODEL_CACHE
    if _WORKER_ARGS is None:
        raise RuntimeError("converter worker 尚未初始化 args")
    if _WORKER_MODEL_CACHE is None:
        _WORKER_MODEL_CACHE = SmplModelCache(model_dir=_WORKER_ARGS.smpl_model_dir)
    path, mirror_variant = item
    return convert_one_motion_safely(
        path=path,
        args=_WORKER_ARGS,
        model_cache=_WORKER_MODEL_CACHE,
        mirror_variant=mirror_variant,
    )


def output_path_for(source: MotionSource, output_dir: Path) -> Path:
    return output_dir / source.relative_path.with_suffix(".npz")


def build_realtime_pose_features(
    smpl_motion: SmplMotion,
    target_fps: float = 60.0,
    body_fbx_rest: BodyFbxRest | None = None,
) -> dict[str, np.ndarray]:
    """构建当前 realtime_pose source 字段。"""

    if body_fbx_rest is None:
        raise ValueError("生成 source 必须提供 body_fbx_rest。")
    joint_positions = smpl_motion.joint_positions.astype(np.float64)
    joint_rotations = smpl_motion.joint_rotations.astype(np.float64)
    pelvis_world = joint_positions[:, JOINT_INDEX["pelvis"]].copy()
    root_yaw = extract_root_heading_from_source_pelvis_up(
        joint_rotations[:, JOINT_INDEX["pelvis"]],
    ).astype(np.float32)
    body_pose_6d = source_global_rotations_to_body_fbx_local_delta_6d(
        joint_rotations,
        root_heading=root_yaw,
    )
    root_pos_world = actor_root_positions_from_pelvis(
        pelvis_world=pelvis_world,
        root_heading=root_yaw,
        pelvis_rest_local_position=body_fbx_rest.pelvis_local_position,
    )
    root_pos_world[:, 1] = 0.0
    pelvis_height = pelvis_world[:, 1:2].astype(np.float32)
    joints_world, joint_rotations_world = fk_body_fbx_local_delta_root_y0(
        body_pose_local_delta_6d=body_pose_6d,
        actor_root_pos_world=root_pos_world,
        root_heading=root_yaw,
        pelvis_height=pelvis_height,
        rest=body_fbx_rest,
    )
    tracker_pos_world = joints_world[:, body_fbx_rest.tracker_joint_indices].astype(np.float32)
    tracker_rot_world_6d = rotation_6d_forward_up_np(
        joint_rotations_world[:, body_fbx_rest.tracker_joint_indices],
    ).astype(np.float32)
    joint_offsets_parent = body_fbx_rest.rest_local_positions.astype(np.float32)
    joint_rest_local_rotations_6d = rotation_6d_forward_up_np(
        body_fbx_rest.rest_local_rotations,
    ).astype(np.float32)
    yaw_delta = np.zeros_like(root_yaw, dtype=np.float32)
    if root_yaw.shape[0] > 1:
        yaw_delta[1:] = wrap_radians(root_yaw[1:].astype(np.float64) - root_yaw[:-1].astype(np.float64)).astype(np.float32)
    root_heading_delta_sincos = np.stack([np.sin(yaw_delta), np.cos(yaw_delta)], axis=-1).astype(np.float32)

    features = {
        BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY: body_pose_6d,
        "root_pos_world": root_pos_world.astype(np.float32),
        "root_yaw": root_yaw.astype(np.float32),
        "root_heading_delta_sincos": root_heading_delta_sincos,
        "tracker_pos_world": tracker_pos_world,
        "tracker_rot_world_6d": tracker_rot_world_6d,
        "joints_world": joints_world,
        "joint_offsets_parent": joint_offsets_parent,
    }
    features["joint_rest_local_rotations_6d"] = joint_rest_local_rotations_6d
    if body_fbx_rest is not None and body_fbx_rest.source_path is not None:
        features["body_fbx_rest_json"] = np.asarray(str(body_fbx_rest.source_path))
    # source 是可复用的世界运动缓存，不等同于 144 维扩散 target。Root 轨迹、
    # Pelvis 高度和 stationary 标签继续离线保存，但主 task 不会读取这些目标通道。
    features["root_delta_xz_ref"] = encode_root_delta_xz_ref(
        root_pos_world=root_pos_world.astype(np.float32),
        root_yaw=root_yaw.astype(np.float32),
    )
    features["pelvis_height"] = pelvis_world[:, 1:2].astype(np.float32)
    features["stationary_prob_5"] = derive_stationary_prob_5(
        joints_world=joints_world,
        fps=float(target_fps),
    )
    return features


def save_realtime_pose_motion(
    output_path: Path,
    features: dict[str, np.ndarray],
    source: MotionSource,
    target_fps: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_path": str(source.path),
        "source_relative_path": str(source.relative_path),
        "original_source_relative_path": str(source.original_relative_path or source.relative_path),
        "stablemotion_split_key": str(source.relative_path.with_suffix(".npy")).replace("\\", "/"),
        "is_mirrored": bool(source.is_mirrored),
        "source_fps": float(source.source_fps),
        "target_fps": float(target_fps),
        "frames": int(features[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY].shape[0]),
        "tracker_order": ["head", "left_wrist", "right_wrist", "waist", "left_foot", "right_foot"],
    }
    metadata["stationary_joint_indices"] = [int(index) for index in STATIONARY_JOINT_INDICES]
    metadata["stationary_joint_names"] = list(STATIONARY_JOINT_NAMES)
    if "body_fbx_rest_json" in features:
        metadata["body_fbx_rest_json"] = str(features["body_fbx_rest_json"].item())
    validate_realtime_source_arrays(features, metadata=metadata, expected_fps=target_fps, path=output_path)
    np.savez(output_path, **features, metadata=json.dumps(metadata, ensure_ascii=False))


def load_metadata_from_npz(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    return load_realtime_metadata(data)


def resolve_body_fbx_rest(args: argparse.Namespace) -> BodyFbxRest:
    cached = getattr(args, "_body_fbx_rest", None)
    if cached is not None:
        return cached
    rest_arg = getattr(args, "body_fbx_rest_json", "")
    rest_path = rest_arg if str(rest_arg).strip() else None
    rest = load_body_fbx_rest(rest_path)
    setattr(args, "_body_fbx_rest", rest)
    return rest


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
            frames = validate_realtime_source_arrays(
                data,
                metadata=metadata,
                expected_fps=float(args.target_fps),
                path=output_path,
            )
    source_relative_path = Path(str(metadata.get("source_relative_path", relative_path)))
    stablemotion_key = str(metadata.get("stablemotion_split_key", source_relative_path.with_suffix(".npy"))).replace("\\", "/")
    return {
        "status": status,
        "source_path": str(metadata.get("source_path", path)),
        "source_relative_path": str(source_relative_path),
        "original_source_relative_path": str(metadata.get("original_source_relative_path", path.relative_to(args.amass_dir))),
        "is_mirrored": bool(metadata.get("is_mirrored", mirror_variant)),
        "stablemotion_split_key": stablemotion_key,
        "output_path": str(output_path),
        "frames": int(metadata.get("frames", frames)),
        "target_fps": float(metadata["target_fps"]),
    }


def reusable_source_path_for(path: Path, args: argparse.Namespace, mirror_variant: bool) -> Path | None:
    reuse_source_dir = Path(args.reuse_source_dir) if getattr(args, "reuse_source_dir", "") else None
    if reuse_source_dir is None:
        return None
    relative_path = source_relative_path_for(path=path, args=args, mirror_variant=mirror_variant)
    return reuse_source_dir / relative_path.with_suffix(".npz")


def load_reusable_realtime_features(
    reuse_path: Path,
    expected_target_fps: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(reuse_path, allow_pickle=False) as data:
        metadata = load_metadata_from_npz(data)
        validate_realtime_source_arrays(
            data,
            metadata=metadata,
            expected_fps=expected_target_fps,
            path=reuse_path,
        )
        features = {
            key: np.asarray(data[key]).astype(np.float32, copy=True)
            for key in REUSABLE_SOURCE_FIELDS
        }
    return features, metadata


def realtime_source_is_reusable(path: Path, expected_target_fps: float) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            if not REUSABLE_SOURCE_FIELDS.issubset(data.files):
                return False
            validate_realtime_source_arrays(data, expected_fps=expected_target_fps, path=path)
    except (KeyError, TypeError, ValueError, OSError):
        return False
    return True


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
            "source_path": str(next_metadata.get("source_path", path)),
            "source_relative_path": str(next_metadata.get("source_relative_path", relative_path)),
            "original_source_relative_path": str(next_metadata.get("original_source_relative_path", path.relative_to(args.amass_dir))),
            "stablemotion_split_key": str(next_metadata.get("stablemotion_split_key", relative_path.with_suffix(".npy"))).replace("\\", "/"),
            "is_mirrored": bool(next_metadata.get("is_mirrored", mirror_variant)),
            "target_fps": float(args.target_fps),
            "frames": int(features[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY].shape[0]),
            "tracker_order": ["head", "left_wrist", "right_wrist", "waist", "left_foot", "right_foot"],
        }
    )
    next_metadata["stationary_joint_indices"] = [int(index) for index in STATIONARY_JOINT_INDICES]
    next_metadata["stationary_joint_names"] = list(STATIONARY_JOINT_NAMES)
    validate_realtime_source_arrays(
        features,
        metadata=next_metadata,
        expected_fps=float(args.target_fps),
        path=output_path,
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
    if not realtime_source_is_reusable(reuse_path, expected_target_fps=float(args.target_fps)):
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
        reuse_path,
        expected_target_fps=float(args.target_fps),
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
        if realtime_source_is_reusable(output_path, expected_target_fps=float(args.target_fps)):
            return record_for_output(
                path=path,
                output_path=output_path,
                args=args,
                mirror_variant=mirror_variant,
                status="skipped_existing",
            )
        if not (args.overwrite or args.rebuild_manifest):
            raise ValueError(
                f"已有 source 与当前字段或 target_fps={float(args.target_fps):g} 不兼容: {output_path}，"
                "请使用 --overwrite 或 --rebuild_manifest 重新转换。"
            )
    elif output_path.exists() and not args.overwrite:
        raise FileExistsError(f"输出文件已存在: {output_path}，请使用 --overwrite 或 --skip_existing。")

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
    body_fbx_rest = resolve_body_fbx_rest(args)
    features = build_realtime_pose_features(
        smpl_motion,
        target_fps=float(args.target_fps),
        body_fbx_rest=body_fbx_rest,
    )
    save_realtime_pose_motion(
        output_path=output_path,
        features=features,
        source=source,
        target_fps=args.target_fps,
    )
    return record_for_output(
        path=path,
        output_path=output_path,
        args=args,
        mirror_variant=mirror_variant,
        status="converted",
    )


def failed_record_for_exception(
    path: Path,
    args: argparse.Namespace,
    mirror_variant: bool,
    exc: Exception,
) -> dict[str, Any]:
    source_relative_path = path.relative_to(args.amass_dir)
    if mirror_variant:
        source_relative_path = Path(MIRROR_DIR_NAME) / source_relative_path
    return {
        "status": "failed",
        "source_path": str(path),
        "source_relative_path": str(source_relative_path),
        "stablemotion_split_key": str(source_relative_path.with_suffix(".npy")).replace("\\", "/"),
        "is_mirrored": bool(mirror_variant),
        "error": repr(exc),
    }


def skipped_short_record_for_exception(
    path: Path,
    args: argparse.Namespace,
    mirror_variant: bool,
    exc: ShortMotionError,
) -> dict[str, Any]:
    """短动作属于预期的数据筛除，不应掩盖真正的转换异常。"""

    source_relative_path = source_relative_path_for(
        path=path,
        args=args,
        mirror_variant=mirror_variant,
    )
    return {
        "status": "skipped_short",
        "source_path": str(path),
        "source_relative_path": str(source_relative_path),
        "stablemotion_split_key": str(source_relative_path.with_suffix(".npy")).replace("\\", "/"),
        "is_mirrored": bool(mirror_variant),
        "reason": str(exc),
    }


def convert_one_motion_safely(
    path: Path,
    args: argparse.Namespace,
    model_cache: SmplModelCache,
    mirror_variant: bool = False,
) -> dict[str, Any]:
    try:
        return convert_one_motion(
            path=path,
            args=args,
            model_cache=model_cache,
            mirror_variant=mirror_variant,
        )
    except ShortMotionError as exc:
        return skipped_short_record_for_exception(
            path=path,
            args=args,
            mirror_variant=mirror_variant,
            exc=exc,
        )
    except Exception as exc:
        return failed_record_for_exception(
            path=path,
            args=args,
            mirror_variant=mirror_variant,
            exc=exc,
        )


def iter_conversion_records(
    args: argparse.Namespace,
    work_items: list[tuple[Path, bool]],
):
    progress = tqdm(
        total=len(work_items),
        desc="Converting AMASS to realtime pose source",
    )
    try:
        if int(args.num_workers) <= 1:
            model_cache = SmplModelCache(model_dir=args.smpl_model_dir)
            for path, mirror_variant in work_items:
                yield convert_one_motion_safely(
                    path=path,
                    args=args,
                    model_cache=model_cache,
                    mirror_variant=mirror_variant,
                )
                progress.update(1)
            return

        worker_args = copy_args_for_worker(args)
        with ProcessPoolExecutor(
            max_workers=int(args.num_workers),
            initializer=init_conversion_worker,
            initargs=(worker_args, int(args.worker_torch_threads)),
        ) as executor:
            for record in executor.map(convert_work_item_in_worker, work_items, chunksize=1):
                yield record
                progress.update(1)
    finally:
        progress.close()


def main(argv: list[str] | None = None) -> dict[str, int]:
    args = parse_args(argv)
    validate_converter_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    if (args.overwrite or args.rebuild_manifest) and manifest_path.exists():
        manifest_path.unlink()

    motion_files = iter_amass_motion_files(args.amass_dir)
    if args.limit:
        motion_files = motion_files[: args.limit]
    work_items = build_conversion_work_items(args=args, motion_files=motion_files)

    converted = reused = skipped = skipped_short = failed = 0
    failed_records: list[dict[str, Any]] = []
    for record in iter_conversion_records(args=args, work_items=work_items):
        if record["status"] == "converted":
            converted += 1
        elif record["status"] in {"reused_source", "upgraded_existing_source"}:
            reused += 1
        elif record["status"] == "skipped_short":
            skipped_short += 1
        elif record["status"] == "failed":
            failed += 1
            failed_records.append(record)
        else:
            skipped += 1
        write_manifest_record(manifest_path, record)

    print(
        f"完成 AMASS 转换: converted={converted}, "
        f"reused={reused}, skipped={skipped}, skipped_short={skipped_short}, "
        f"failed={failed}, manifest={manifest_path}"
    )
    if failed_records and not args.allow_partial:
        preview = "; ".join(
            f"{record['source_relative_path']}: {record['error']}"
            for record in failed_records[:3]
        )
        raise RuntimeError(
            f"AMASS 转换存在 {len(failed_records)} 个失败样本，默认停止以避免下游使用部分数据。"
            f"示例: {preview}。如确认可接受部分数据，请添加 --allow_partial。"
        )
    return {
        "converted": converted,
        "reused": reused,
        "skipped": skipped,
        "skipped_short": skipped_short,
        "failed": failed,
    }


if __name__ == "__main__":
    main()
