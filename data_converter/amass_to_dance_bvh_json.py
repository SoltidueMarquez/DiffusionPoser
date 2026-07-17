"""
把 AMASS SMPL/SMPL-H 动作转换成 dance_bvh Unity 工程可直接读取的 JSON。

输出格式保持和 `D:/Projects/Other/dance_bvh` 当前 `JsonReader` 兼容：

    dance_array[sequence][frame] = 24 * 9 个 parent-local 旋转矩阵值 + 3 个 root 平移值

默认先把 AMASS pelvis/root 的 Z-up 世界朝向转成 SMPL Unity prefab 更匹配的 Y-up 朝向；
dance_bvh 工程的 `RotateTest` 再在运行时做右手系 -> Unity 左手系转换。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from data_converter.amass_smpl_utils import (
    MotionSource,
    build_smpl_local_rotations,
    iter_amass_motion_files,
    load_motion_source,
    mirror_motion_source,
)
from data_loaders.realtime_pose_kinematics import SMPL_JOINT_NAMES
from utils.artifact_roots import load_artifact_roots


DANCE_BVH_JOINT_COUNT = 24
DANCE_BVH_FRAME_DIM = DANCE_BVH_JOINT_COUNT * 9 + 3
DEFAULT_OUTPUT_FILENAME = "dance_output.json"
RAW_AMASS_BASIS = "raw_amass"
AMASS_ZUP_TO_SMPL_YUP_BASIS = "amass_zup_to_smpl_yup"
COORDINATE_BASIS_CHOICES = (AMASS_ZUP_TO_SMPL_YUP_BASIS, RAW_AMASS_BASIS)
AMASS_ZUP_TO_SMPL_YUP_ROOT = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AMASS motions to dance_bvh playable JSON.")
    parser.add_argument("--artifact_roots_config", default="", type=str)
    parser.add_argument(
        "--amass_path",
        default=None,
        type=Path,
        help="AMASS .npz file or directory. Directory mode converts motions recursively.",
    )
    parser.add_argument(
        "--amass_dir",
        default=None,
        type=Path,
        help="AMASS root used for relative paths. Defaults to amass_path for directories or parent for files.",
    )
    parser.add_argument("--output_json", default=None, type=Path)
    parser.add_argument("--target_fps", default=60.0, type=float)
    parser.add_argument("--coordinate_basis", default=AMASS_ZUP_TO_SMPL_YUP_BASIS, choices=COORDINATE_BASIS_CHOICES)
    parser.add_argument("--translation_scale", default=100.0, type=float)
    parser.add_argument("--keep_world_translation", action="store_true")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--indent", default=None, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve_converter_paths(args: argparse.Namespace) -> argparse.Namespace:
    roots = load_artifact_roots(getattr(args, "artifact_roots_config", "") or None)
    if args.amass_path is None:
        args.amass_path = roots.amass_root
    if args.output_json is None:
        args.output_json = roots.outputs_root / "dance_bvh" / DEFAULT_OUTPUT_FILENAME
    return args


def resolve_motion_inputs(amass_path: Path, amass_dir: Path | None) -> tuple[list[Path], Path]:
    source_path = amass_path.expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"AMASS 输入不存在：{source_path}")

    if amass_dir is None:
        source_root = source_path if source_path.is_dir() else source_path.parent
    else:
        source_root = amass_dir.expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"AMASS 根目录不存在：{source_root}")

    motion_paths = iter_amass_motion_files(source_path) if source_path.is_dir() else [source_path]
    for path in motion_paths:
        try:
            path.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(f"{path} 不在 AMASS 根目录 {source_root} 下，无法记录相对路径。") from exc
    return motion_paths, source_root


def build_dance_bvh_frames(
    source: MotionSource,
    coordinate_basis: str = AMASS_ZUP_TO_SMPL_YUP_BASIS,
    translation_scale: float = 100.0,
    zero_origin_xz: bool = True,
) -> np.ndarray:
    """把单段 AMASS motion 转成 `[T,219]`，其中旋转为 SMPL parent-local 3x3 矩阵。"""

    local_rotations = convert_rotations_to_coordinate_basis(
        build_smpl_local_rotations(source.poses),
        coordinate_basis=coordinate_basis,
    )
    if local_rotations.shape[1:] != (DANCE_BVH_JOINT_COUNT, 3, 3):
        raise ValueError(f"local_rotations 应为 [T,24,3,3]，实际为 {local_rotations.shape}")

    root_translation = convert_translation_to_coordinate_basis(
        source.trans.astype(np.float64, copy=True),
        coordinate_basis=coordinate_basis,
    )
    if zero_origin_xz and root_translation.shape[0] > 0:
        # Unity 示例读取器只使用 X/Z 平移；默认把第一帧放到场景原点附近，保留 Y 高度语义。
        root_translation[:, [0, 2]] -= root_translation[:1, [0, 2]]
    root_translation *= float(translation_scale)

    frames = np.concatenate(
        [
            local_rotations.reshape(local_rotations.shape[0], DANCE_BVH_JOINT_COUNT * 9),
            root_translation,
        ],
        axis=1,
    )
    if frames.shape[1] != DANCE_BVH_FRAME_DIM:
        raise ValueError(f"dance_bvh 每帧应为 {DANCE_BVH_FRAME_DIM} 维，实际为 {frames.shape[1]}")
    return frames.astype(np.float32)


def convert_rotations_to_coordinate_basis(rotations: np.ndarray, coordinate_basis: str) -> np.ndarray:
    if coordinate_basis == RAW_AMASS_BASIS:
        return rotations
    if coordinate_basis == AMASS_ZUP_TO_SMPL_YUP_BASIS:
        converted = rotations.copy()
        # AMASS 的非根关节已经是 SMPL parent-local pose；只需要把 pelvis/root 的世界朝向换到
        # dance_bvh 读取器期望的 Y-up 右手坐标，再交给 Unity 端做 handedness conversion。
        converted[:, 0] = AMASS_ZUP_TO_SMPL_YUP_ROOT @ converted[:, 0]
        return converted
    raise ValueError(f"unsupported coordinate_basis: {coordinate_basis}")


def convert_translation_to_coordinate_basis(translation: np.ndarray, coordinate_basis: str) -> np.ndarray:
    if coordinate_basis == RAW_AMASS_BASIS:
        return translation
    if coordinate_basis == AMASS_ZUP_TO_SMPL_YUP_BASIS:
        return translation @ AMASS_ZUP_TO_SMPL_YUP_ROOT.T
    raise ValueError(f"unsupported coordinate_basis: {coordinate_basis}")


def sequence_metadata_for(
    source: MotionSource,
    frames: np.ndarray,
    coordinate_basis: str,
    translation_scale: float,
    zero_origin_xz: bool,
) -> dict[str, Any]:
    return {
        "source_path": str(source.path),
        "source_relative_path": str(source.relative_path).replace("\\", "/"),
        "original_source_relative_path": str(source.original_relative_path or source.relative_path).replace("\\", "/"),
        "is_mirrored": bool(source.is_mirrored),
        "source_fps": float(source.source_fps),
        "frames": int(frames.shape[0]),
        "frame_dim": int(frames.shape[1]),
        "coordinate_basis": coordinate_basis,
        "translation_scale": float(translation_scale),
        "translation_zero_origin_xz": bool(zero_origin_xz),
    }


def convert_motion_to_sequence(
    path: Path,
    amass_dir: Path,
    target_fps: float,
    coordinate_basis: str,
    translation_scale: float,
    zero_origin_xz: bool,
    mirror_variant: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = load_motion_source(path=path, amass_dir=amass_dir, target_fps=target_fps)
    if mirror_variant:
        source = mirror_motion_source(source)
    frames = build_dance_bvh_frames(
        source=source,
        coordinate_basis=coordinate_basis,
        translation_scale=translation_scale,
        zero_origin_xz=zero_origin_xz,
    )
    return frames, sequence_metadata_for(
        source=source,
        frames=frames,
        coordinate_basis=coordinate_basis,
        translation_scale=translation_scale,
        zero_origin_xz=zero_origin_xz,
    )


def write_dance_bvh_json(
    output_json: Path,
    sequences: list[np.ndarray],
    sequence_metadata: list[dict[str, Any]],
    target_fps: float,
    coordinate_basis: str,
    indent: int | None = None,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dance_array": [sequence.astype(float).tolist() for sequence in sequences],
        "metadata": {
            "format": "dance_bvh_json",
            "frame_dim": DANCE_BVH_FRAME_DIM,
            "joint_order": list(SMPL_JOINT_NAMES),
            "coordinate_basis": coordinate_basis,
            "rotation_space": "SMPL parent-local; pelvis/root AMASS Z-up converted to Y-up before Unity handedness conversion",
            "rotation_layout": "24 row-major 3x3 matrices followed by root translation xyz",
            "target_fps": float(target_fps),
            "sequence_count": len(sequences),
            "sequences": sequence_metadata,
        },
    }
    separators = None if indent is not None else (",", ":")
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=indent, separators=separators)


def main(argv: list[str] | None = None) -> dict[str, int]:
    args = resolve_converter_paths(parse_args(argv))
    if args.target_fps <= 0:
        raise ValueError("--target_fps 必须为正数")
    if args.translation_scale <= 0:
        raise ValueError("--translation_scale 必须为正数")
    if args.limit < 0:
        raise ValueError("--limit 不能为负数")
    if args.output_json.exists() and not args.overwrite:
        raise FileExistsError(f"输出文件已存在：{args.output_json}，请添加 --overwrite。")

    motion_paths, amass_dir = resolve_motion_inputs(args.amass_path, args.amass_dir)
    if args.limit:
        motion_paths = motion_paths[: args.limit]
    if not motion_paths:
        raise RuntimeError(f"没有找到可转换的 AMASS .npz：{args.amass_path}")

    sequences: list[np.ndarray] = []
    sequence_metadata: list[dict[str, Any]] = []
    zero_origin_xz = not args.keep_world_translation

    for path in tqdm(motion_paths, desc="Converting AMASS to dance_bvh JSON"):
        mirror_variants = (False, True) if args.mirror else (False,)
        for mirror_variant in mirror_variants:
            frames, metadata = convert_motion_to_sequence(
                path=path,
                amass_dir=amass_dir,
                target_fps=args.target_fps,
                coordinate_basis=args.coordinate_basis,
                translation_scale=args.translation_scale,
                zero_origin_xz=zero_origin_xz,
                mirror_variant=mirror_variant,
            )
            sequences.append(frames)
            sequence_metadata.append(metadata)

    write_dance_bvh_json(
        output_json=args.output_json,
        sequences=sequences,
        sequence_metadata=sequence_metadata,
        target_fps=args.target_fps,
        coordinate_basis=args.coordinate_basis,
        indent=args.indent,
    )
    frame_count = sum(int(sequence.shape[0]) for sequence in sequences)
    print(
        f"完成 AMASS -> dance_bvh JSON：sequences={len(sequences)}, "
        f"frames={frame_count}, output={args.output_json}"
    )
    return {"sequences": len(sequences), "frames": frame_count}


if __name__ == "__main__":
    main()
