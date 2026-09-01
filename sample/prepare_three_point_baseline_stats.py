from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sample.three_point_baseline_data import compute_baseline_pose_stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="计算 RPM-P2 与 AGRoL-P1 官方 checkpoint 所需的 pose mean/std。"
    )
    parser.add_argument("--amass_dir", required=True, type=Path)
    parser.add_argument("--rpm_p1_train_split", required=True, type=Path)
    parser.add_argument("--rpm_p2_train_split", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def write_stats(
    *,
    output_path: Path,
    amass_dir: Path,
    split_file: Path,
    protocol: str,
    min_feature_frames: int,
    target_fps: float,
    overwrite: bool,
) -> Path:
    if output_path.exists() and not overwrite:
        print(f"[baseline-stats] reuse: {output_path}", flush=True)
        return output_path
    mean, std, frame_count, sequence_count = compute_baseline_pose_stats(
        amass_dir=amass_dir,
        split_file=split_file,
        protocol=protocol,
        min_feature_frames=min_feature_frames,
        target_fps=target_fps,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        mean=mean,
        std=std,
        frame_count=np.asarray(frame_count, dtype=np.int64),
        sequence_count=np.asarray(sequence_count, dtype=np.int64),
        protocol=np.asarray(protocol),
        min_feature_frames=np.asarray(min_feature_frames, dtype=np.int64),
        target_fps=np.asarray(target_fps, dtype=np.float32),
    )
    print(f"[baseline-stats] wrote: {output_path}", flush=True)
    return output_path


def main(argv: list[str] | None = None) -> tuple[Path, Path]:
    args = build_arg_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    # AGRoL 训练只保留至少 196 个有效特征帧的 P1 序列。
    agrol_path = write_stats(
        output_path=output_dir / "agrol_p1_pose_stats.npz",
        amass_dir=args.amass_dir.expanduser().resolve(),
        split_file=args.rpm_p1_train_split.expanduser().resolve(),
        protocol="agrol_p1",
        min_feature_frames=196,
        target_fps=60.0,
        overwrite=bool(args.overwrite),
    )
    # RPM-P2 Reactive 的训练展开长度为 W=5、context=10、free-running=30。
    rpm_path = write_stats(
        output_path=output_dir / "rpm_p2_pose_stats.npz",
        amass_dir=args.amass_dir.expanduser().resolve(),
        split_file=args.rpm_p2_train_split.expanduser().resolve(),
        protocol="rpm_p2",
        min_feature_frames=45,
        target_fps=30.0,
        overwrite=bool(args.overwrite),
    )
    return rpm_path, agrol_path


if __name__ == "__main__":
    main()
