"""为掉线诊断准备与 source 时间轴一致的官方 RPM 输入。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data_converter.amass_smpl_utils import (
    AMASS_TO_UNITY, SmplModelCache, load_motion_source, run_smpl_forward,
)
from data_loaders.build_realtime_longseq_eval_set import (
    read_longseq_source_entries, resolve_source_entry_path,
)
from data_loaders.generate_realtime_pose_tasks import normalize_split_key
from sample.evaluate_hand_signal_recovery import (
    build_gap_key_by_source, build_hand_tracker_availability, load_rpm_p2_hand_gap_config,
)
from sample.three_point_baseline_data import (
    axis_angle_to_baseline_motion_6d, build_sparse_features_from_trackers,
)


def mask_rpm_hand_features(sparse: np.ndarray, gaps) -> np.ndarray:
    """复用官方协议：在 gap 内将该手的全部 18 个输入通道置零。"""
    result = np.asarray(sparse, dtype=np.float32).copy()
    for hand, intervals in enumerate(gaps, start=1):
        channels = [
            *range(hand * 6, hand * 6 + 6),
            *range(18 + hand * 6, 24 + hand * 6),
            *range(36 + hand * 3, 39 + hand * 3),
            *range(45 + hand * 3, 48 + hand * 3),
        ]
        for start, end in intervals:
            result[int(start):int(end), channels] = 0.0
    return result


def prepare_inputs(args) -> None:
    torch.set_num_threads(2)
    _, gaps = load_rpm_p2_hand_gap_config(args.gap_config)
    mapping = build_gap_key_by_source(args.split_dir / "test.txt", set(gaps))
    entries = read_longseq_source_entries(
        args.source_dir, args.split_dir, split="test", min_frames=31, include_mirror=False,
    )
    cache = SmplModelCache(args.smpl_model_dir)
    mean = torch.load(args.rpm_stats_dir / "amass_p2_mean.pt", weights_only=True).numpy()
    std = torch.load(args.rpm_stats_dir / "amass_p2_std.pt", weights_only=True).numpy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, entry in enumerate(entries):
        relative = Path(entry["source_relative_path"]).with_suffix(".npz")
        raw_path = args.amass_dir / relative
        source_path = resolve_source_entry_path(entry)
        motion = load_motion_source(raw_path, args.amass_dir, target_fps=30.0)
        with np.load(source_path, allow_pickle=False) as data:
            frame_count = len(data["joints_world"])
        if frame_count != len(motion.poses):
            raise ValueError(f"source 与 AMASS 长度不一致：{relative}")
        smpl = run_smpl_forward(motion, cache, batch_size=256)
        # RPM 输入使用原始 SMPL 关节坐标与基底，不能直接输入 body.fbx tracker。
        tracker_indices = [15, 20, 21]
        rotations = AMASS_TO_UNITY.T @ smpl.joint_rotations[:, tracker_indices] @ AMASS_TO_UNITY
        sparse = build_sparse_features_from_trackers(
            smpl.raw_joint_positions[:, tracker_indices], rotations,
        )
        key = mapping[normalize_split_key(str(relative))]
        masked = mask_rpm_hand_features(sparse, gaps[key])
        path = args.output_dir / f"{index:04d}.npz"
        np.savez_compressed(
            path, rpm_motion_6d=axis_angle_to_baseline_motion_6d(motion.poses)[1:],
            rpm_sparse_54d=masked, rpm_pose_mean=mean, rpm_pose_std=std,
            source_feature_frame_offset=np.asarray(1), source_frame_count=np.asarray(frame_count),
            source_path=np.asarray(str(source_path)), source_relative_path=np.asarray(str(relative)),
            amass_path=np.asarray(str(raw_path)), gap_key=np.asarray(key),
            tracker_available=build_hand_tracker_availability(
                source_frame_count=frame_count, hand_gap_intervals=gaps[key], gap_frame_offset=1,
            ),
        )
        manifest.append({"file": path.name, "source": str(relative), "gap_key": key})
        print(f"[dropout-input] {index + 1}/{len(entries)} {relative}", flush=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source_dir", "split_dir", "amass_dir", "smpl_model_dir", "rpm_stats_dir", "gap_config", "output_dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    prepare_inputs(parser.parse_args(argv))


if __name__ == "__main__":
    main()
