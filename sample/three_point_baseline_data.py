from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from data_converter.amass_smpl_utils import (
    SOURCE_BODY_JOINT_COUNT,
    MotionSource,
    SmplModelCache,
    build_smpl_local_rotations,
    load_motion_source,
    local_to_global_rotations,
    run_smpl_forward,
)
from data_loaders.realtime_pose_kinematics import TRACKER_JOINT_INDICES


BASELINE_TRACKER_COUNT = 3
BASELINE_MOTION_DIM = SOURCE_BODY_JOINT_COUNT * 6
BASELINE_SPARSE_DIM = BASELINE_TRACKER_COUNT * (6 + 6 + 3 + 3)


@dataclass(frozen=True)
class StreamingMoments:
    """按批次累计 `[N,D]` 特征的均值与无偏标准差。"""

    count: int
    mean: np.ndarray
    m2: np.ndarray

    @classmethod
    def empty(cls, feature_dim: int) -> "StreamingMoments":
        return cls(
            count=0,
            mean=np.zeros((int(feature_dim),), dtype=np.float64),
            m2=np.zeros((int(feature_dim),), dtype=np.float64),
        )

    def update(self, values: np.ndarray) -> "StreamingMoments":
        batch = np.asarray(values, dtype=np.float64)
        if batch.ndim != 2 or batch.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"统计输入应为 [N,{self.mean.shape[0]}]，实际为 {batch.shape}"
            )
        if batch.shape[0] == 0:
            return self
        batch_count = int(batch.shape[0])
        batch_mean = batch.mean(axis=0)
        centered = batch - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0)
        if self.count == 0:
            return StreamingMoments(batch_count, batch_mean, batch_m2)

        combined_count = self.count + batch_count
        delta = batch_mean - self.mean
        combined_mean = self.mean + delta * (batch_count / combined_count)
        combined_m2 = (
            self.m2
            + batch_m2
            + delta * delta * (self.count * batch_count / combined_count)
        )
        return StreamingMoments(combined_count, combined_mean, combined_m2)

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            raise ValueError(f"至少需要 2 帧才能计算无偏标准差，实际为 {self.count}")
        variance = np.maximum(self.m2 / float(self.count - 1), 0.0)
        return self.mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def matrix_to_first_two_columns_6d(rotations: np.ndarray) -> np.ndarray:
    """使用 RPM/AGRoL 的 6D 约定编码旋转矩阵的前两列。"""

    matrices = np.asarray(rotations, dtype=np.float64)
    if matrices.shape[-2:] != (3, 3):
        raise ValueError(f"rotations 最后两维应为 [3,3]，实际为 {matrices.shape}")
    return np.concatenate(
        [matrices[..., :, 0], matrices[..., :, 1]], axis=-1
    ).astype(np.float32)


def axis_angle_to_baseline_motion_6d(axis_angle: np.ndarray) -> np.ndarray:
    """把 SMPL22 axis-angle `[T,66]` 转为官方基线使用的 `[T,132]`。"""

    poses = np.asarray(axis_angle, dtype=np.float64)
    expected_dim = SOURCE_BODY_JOINT_COUNT * 3
    if poses.ndim != 2 or poses.shape[1] < expected_dim:
        raise ValueError(f"axis_angle 至少应为 [T,{expected_dim}]，实际为 {poses.shape}")
    matrices = Rotation.from_rotvec(
        poses[:, :expected_dim].reshape(-1, 3)
    ).as_matrix()
    encoded = matrix_to_first_two_columns_6d(matrices)
    return encoded.reshape(poses.shape[0], BASELINE_MOTION_DIM)


def baseline_motion_6d_to_rotation_matrices(motion_6d: np.ndarray) -> np.ndarray:
    """用 Gram-Schmidt 把官方基线 `[T,132]` 输出投影回 SMPL22 旋转矩阵。"""

    motion = np.asarray(motion_6d, dtype=np.float64)
    if motion.ndim != 2 or motion.shape[1] != BASELINE_MOTION_DIM:
        raise ValueError(
            f"baseline motion 应为 [T,{BASELINE_MOTION_DIM}]，实际为 {motion.shape}"
        )
    vectors = motion.reshape(motion.shape[0], SOURCE_BODY_JOINT_COUNT, 2, 3)
    first = vectors[..., 0, :]
    second_raw = vectors[..., 1, :]
    first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    second = second_raw - np.sum(first * second_raw, axis=-1, keepdims=True) * first
    second /= np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
    third = np.cross(first, second)
    matrices = np.stack([first, second, third], axis=-1)
    if not np.isfinite(matrices).all():
        raise ValueError("baseline motion 投影后含 NaN/Inf。")
    return matrices.astype(np.float32)


def official_agrol_resample_poses(raw_npz: Path) -> np.ndarray:
    """复现 AGRoL 对 P1 的 60/120 Hz 整帧抽样规则。"""

    with np.load(raw_npz, allow_pickle=True) as payload:
        poses = np.asarray(payload["poses"], dtype=np.float64)
        fps_key = (
            "mocap_framerate"
            if "mocap_framerate" in payload.files
            else "mocap_frame_rate"
        )
        fps = float(np.asarray(payload[fps_key]).item())
    if np.isclose(fps, 120.0):
        stride = 2
    elif np.isclose(fps, 60.0):
        stride = 1
    else:
        raise ValueError(f"AGRoL P1 只接受 60/120 Hz，{raw_npz} 实际为 {fps:g} Hz")
    return poses[::stride]


def official_rpm_resample_poses(raw_npz: Path, target_fps: float) -> np.ndarray:
    """复现 RPM `prepare_data.py` 的整数索引重采样。"""

    with np.load(raw_npz, allow_pickle=True) as payload:
        poses = np.asarray(payload["poses"], dtype=np.float64)
        fps_key = (
            "mocap_framerate"
            if "mocap_framerate" in payload.files
            else "mocap_frame_rate"
        )
        fps = float(np.asarray(payload[fps_key]).item())
    if float(target_fps) > fps:
        raise ValueError(f"RPM 不允许升采样：{raw_npz} 为 {fps:g} Hz")
    new_frame_count = int(float(target_fps) / fps * poses.shape[0])
    last_frame = poses.shape[0] - 2 if poses.shape[0] % 2 == 0 else poses.shape[0] - 1
    indices = np.linspace(0, last_frame, num=new_frame_count, dtype=np.int64)
    return poses[indices]


def split_entry_to_amass_path(amass_dir: Path, entry: str) -> Path:
    """把 RPM research-kit 的 `.npy` 名单映射回本机 Stage-II `.npz`。"""

    relative = Path(str(entry).strip()).with_suffix(".npz")
    path = Path(amass_dir) / relative
    if not path.is_file():
        raise FileNotFoundError(f"split 条目找不到原始 AMASS 文件：{path}")
    return path


def compute_baseline_pose_stats(
    *,
    amass_dir: Path,
    split_file: Path,
    protocol: str,
    min_feature_frames: int,
    target_fps: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """按官方训练过滤规则计算 pose 6D mean/std。"""

    entries = [
        line.strip()
        for line in Path(split_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    moments = StreamingMoments.empty(BASELINE_MOTION_DIM)
    used_sequences = 0
    for index, entry in enumerate(entries):
        raw_path = split_entry_to_amass_path(amass_dir, entry)
        if protocol == "agrol_p1":
            poses = official_agrol_resample_poses(raw_path)
        elif protocol == "rpm_p2":
            poses = official_rpm_resample_poses(raw_path, target_fps=target_fps)
        else:
            raise ValueError(f"不支持的基线统计协议：{protocol}")

        # 两个官方预处理都丢弃第 0 帧，使 rotation/position velocity 与当前帧对齐。
        feature_frame_count = int(poses.shape[0] - 1)
        if feature_frame_count < int(min_feature_frames):
            continue
        motion_6d = axis_angle_to_baseline_motion_6d(poses)[1:]
        moments = moments.update(motion_6d)
        used_sequences += 1
        if (index + 1) % 500 == 0 or index + 1 == len(entries):
            print(
                f"[baseline-stats] {protocol}: {index + 1}/{len(entries)} "
                f"files, used={used_sequences}, frames={moments.count}",
                flush=True,
            )
    mean, std = moments.finalize()
    return mean, std, int(moments.count), int(used_sequences)


def build_sparse_features_from_trackers(
    tracker_positions_amass: np.ndarray,
    tracker_rotations_amass: np.ndarray,
) -> np.ndarray:
    """构造 RPM/AGRoL 54D 条件；第 0 帧因缺少速度而不输出。"""

    positions = np.asarray(tracker_positions_amass, dtype=np.float64)
    rotations = np.asarray(tracker_rotations_amass, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != (BASELINE_TRACKER_COUNT, 3):
        raise ValueError(f"tracker positions 应为 [T,3,3]，实际为 {positions.shape}")
    if rotations.shape != positions.shape[:2] + (3, 3):
        raise ValueError(f"tracker rotations 应为 [T,3,3,3]，实际为 {rotations.shape}")
    if positions.shape[0] < 2:
        raise ValueError("至少需要 2 帧 tracker 才能构造速度特征。")

    rotation_6d = matrix_to_first_two_columns_6d(rotations)
    relative_rotation = np.swapaxes(rotations[:-1], -1, -2) @ rotations[1:]
    rotation_velocity_6d = matrix_to_first_two_columns_6d(relative_rotation)
    position_delta = positions[1:] - positions[:-1]
    return np.concatenate(
        [
            rotation_6d[1:].reshape(positions.shape[0] - 1, -1),
            rotation_velocity_6d.reshape(positions.shape[0] - 1, -1),
            positions[1:].reshape(positions.shape[0] - 1, -1),
            position_delta.reshape(positions.shape[0] - 1, -1),
        ],
        axis=-1,
    ).astype(np.float32)


def build_official_tracker_signals_from_amass(
    source: MotionSource,
    smpl_model_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """按官方预处理语义生成 AMASS 世界系 Head/双腕信号。

    项目 source 中的 tracker 已经过 SMPL→body.fbx rest/basis 重定向，不能只做
    Unity↔AMASS 轴交换后交给官方基线，否则三枚 tracker 的旋转会恒定偏 180°。
    这里直接从同一 AMASS 动作和 SMPL-H 骨架构造 `[T,3,3]` position 与
    `[T,3,3,3]` rotation，关节顺序严格保持 Head、Left Wrist、Right Wrist。
    """

    model_cache = SmplModelCache(model_dir=Path(smpl_model_dir))
    smpl_motion = run_smpl_forward(
        source=source,
        model_cache=model_cache,
        batch_size=256,
    )
    tracker_indices = TRACKER_JOINT_INDICES[:BASELINE_TRACKER_COUNT]
    local_rotations = build_smpl_local_rotations(source.poses)
    global_rotations_amass = local_to_global_rotations(
        local_rotations,
        smpl_motion.parents,
    )
    positions_amass = smpl_motion.raw_joint_positions[:, tracker_indices]
    rotations_amass = global_rotations_amass[:, tracker_indices]
    return (
        np.asarray(positions_amass, dtype=np.float32),
        np.asarray(rotations_amass, dtype=np.float32),
    )


def resample_tracker_signals(
    positions: np.ndarray,
    rotations: np.ndarray,
    source_fps: float,
    target_fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """线性插值 position、Slerp rotation，保持首尾时间戳一致。"""

    position_values = np.asarray(positions, dtype=np.float64)
    rotation_values = np.asarray(rotations, dtype=np.float64)
    if position_values.ndim != 3 or position_values.shape[1:] != (3, 3):
        raise ValueError(f"positions 应为 [T,3,3]，实际为 {position_values.shape}")
    if rotation_values.shape != position_values.shape[:2] + (3, 3):
        raise ValueError(f"rotations 应为 [T,3,3,3]，实际为 {rotation_values.shape}")
    source_times = np.arange(position_values.shape[0], dtype=np.float64) / float(source_fps)
    duration = source_times[-1]
    target_times = np.arange(0.0, duration + 1e-8, 1.0 / float(target_fps))
    target_times[-1] = min(target_times[-1], duration)

    target_positions = np.empty((target_times.shape[0], 3, 3), dtype=np.float64)
    target_rotations = np.empty((target_times.shape[0], 3, 3, 3), dtype=np.float64)
    for tracker_index in range(3):
        for axis in range(3):
            target_positions[:, tracker_index, axis] = np.interp(
                target_times,
                source_times,
                position_values[:, tracker_index, axis],
            )
        slerp = Slerp(
            source_times,
            Rotation.from_matrix(rotation_values[:, tracker_index]),
        )
        target_rotations[:, tracker_index] = slerp(target_times).as_matrix()
    return target_positions.astype(np.float32), target_rotations.astype(np.float32)


def prepare_baseline_sequence_input(
    *,
    source_npz: Path,
    amass_npz: Path,
    smpl_model_dir: Path,
    rpm_stats_npz: Path,
    agrol_stats_npz: Path,
    source_fps: float = 30.0,
    agrol_fps: float = 60.0,
) -> dict[str, np.ndarray]:
    """把同一条项目 source 物化为两个官方基线可直接消费的输入。"""

    with np.load(source_npz, allow_pickle=False) as source:
        source_frame_count = int(source["tracker_pos_world"].shape[0])
    rpm_source = load_motion_source(
        path=Path(amass_npz),
        amass_dir=Path(amass_npz).parent,
        target_fps=float(source_fps),
    )
    if rpm_source.poses.shape[0] != source_frame_count:
        raise ValueError(
            "原始 AMASS 与项目 source 重采样长度不一致："
            f"{rpm_source.poses.shape[0]} != {source_frame_count}"
        )
    rpm_positions, rpm_rotations = build_official_tracker_signals_from_amass(
        source=rpm_source,
        smpl_model_dir=smpl_model_dir,
    )
    rpm_motion = axis_angle_to_baseline_motion_6d(rpm_source.poses)[1:]
    rpm_sparse = build_sparse_features_from_trackers(
        rpm_positions,
        rpm_rotations,
    )

    agrol_source = load_motion_source(
        path=Path(amass_npz),
        amass_dir=Path(amass_npz).parent,
        target_fps=float(agrol_fps),
    )
    minimum_agrol_frames = 2 * source_frame_count - 1
    if agrol_source.poses.shape[0] < minimum_agrol_frames:
        raise ValueError(
            "AGRoL 60 Hz 时间轴短于 30 Hz source 对应范围："
            f"{agrol_source.poses.shape[0]} < {minimum_agrol_frames}"
        )
    agrol_positions, agrol_rotations = build_official_tracker_signals_from_amass(
        source=agrol_source,
        smpl_model_dir=smpl_model_dir,
    )
    agrol_sparse = build_sparse_features_from_trackers(
        agrol_positions,
        agrol_rotations,
    )
    with np.load(rpm_stats_npz, allow_pickle=False) as stats:
        rpm_mean = np.asarray(stats["mean"], dtype=np.float32)
        rpm_std = np.asarray(stats["std"], dtype=np.float32)
    with np.load(agrol_stats_npz, allow_pickle=False) as stats:
        agrol_mean = np.asarray(stats["mean"], dtype=np.float32)
        agrol_std = np.asarray(stats["std"], dtype=np.float32)

    return {
        "rpm_motion_6d": rpm_motion,
        "rpm_sparse_54d": rpm_sparse,
        "rpm_pose_mean": rpm_mean,
        "rpm_pose_std": rpm_std,
        "agrol_sparse_60hz_54d": agrol_sparse,
        "agrol_pose_mean": agrol_mean,
        "agrol_pose_std": agrol_std,
        "source_frame_count": np.asarray(source_frame_count, dtype=np.int64),
        "source_feature_frame_offset": np.asarray(1, dtype=np.int64),
        "source_fps": np.asarray(source_fps, dtype=np.float32),
        "agrol_fps": np.asarray(agrol_fps, dtype=np.float32),
        "tracker_coordinate_system": np.asarray("official_amass_smpl"),
    }
