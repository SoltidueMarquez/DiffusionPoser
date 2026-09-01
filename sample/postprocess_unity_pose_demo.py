from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from data_loaders.realtime_pose_kinematics import JOINT_INDEX, SMPL_JOINT_NAMES
from sample.render_unity_pose_recording import load_unity_pose_recording


DEFAULT_WINDOW_FRAMES = 5
DEFAULT_STRENGTH = 0.30
DEFAULT_BLEND_FRAMES = 8
DEFAULT_BRIDGE_PRE_ANCHOR_FRAMES = 4
DEFAULT_BRIDGE_POST_ANCHOR_FRAMES = 5
DEFAULT_BRIDGE_LOCK_FRAMES = 6
DEFAULT_BRIDGE_KNEE_STRENGTH = 0.30
DEFAULT_BRIDGE_ANKLE_STRENGTH = 1.0
DEFAULT_BRIDGE_FOOT_STRENGTH = 1.0
SMPL_JOINT_COUNT = len(SMPL_JOINT_NAMES)
LEG_JOINT_NAMES = {
    "left": ("left_hip", "left_knee", "left_ankle", "left_foot"),
    "right": ("right_hip", "right_knee", "right_ankle", "right_foot"),
}


@dataclass(frozen=True)
class FilterSegment:
    """一个显式、可审计的 demo 局部滤波区间。"""

    start_seconds: float
    end_seconds: float
    side: str


@dataclass(frozen=True)
class LandingBridge:
    """用两侧稳定锚点替换一段异常落地旋转。"""

    start_frame: int
    end_frame: int
    side: str


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "对 Unity Pose JSON 的指定腿部时间段做轻量 quaternion 平滑；"
            "仅用于 demo 后处理，不修改 Predictor/DiT/runtime。"
        )
    )
    paths = parser.add_argument_group("paths")
    paths.add_argument("--input_pose", required=True, type=Path)
    paths.add_argument("--output_pose", required=True, type=Path)
    filtering = parser.add_argument_group("local quaternion filter")
    filtering.add_argument(
        "--segment",
        action="append",
        default=[],
        help="可重复传入，格式为 start:end:left|right|both，例如 24.2:26.0:left。",
    )
    filtering.add_argument(
        "--window_frames",
        default=DEFAULT_WINDOW_FRAMES,
        type=int,
        help="对称 geodesic 平滑窗口；必须是大于等于 3 的奇数。",
    )
    filtering.add_argument(
        "--strength",
        default=DEFAULT_STRENGTH,
        type=float,
        help="原始旋转到平滑旋转的插值强度，范围 (0,1]。",
    )
    filtering.add_argument(
        "--blend_frames",
        default=DEFAULT_BLEND_FRAMES,
        type=int,
        help="每个区间两端的 smoothstep 渐入/渐出帧数。",
    )
    bridge = parser.add_argument_group("landing bridge")
    bridge.add_argument(
        "--bridge",
        action="append",
        default=[],
        help="可重复传入，格式为 start_frame:end_frame:left|right|both。",
    )
    bridge.add_argument(
        "--bridge_pre_anchor_frames",
        default=DEFAULT_BRIDGE_PRE_ANCHOR_FRAMES,
        type=int,
    )
    bridge.add_argument(
        "--bridge_post_anchor_frames",
        default=DEFAULT_BRIDGE_POST_ANCHOR_FRAMES,
        type=int,
    )
    bridge.add_argument(
        "--bridge_lock_frames",
        default=DEFAULT_BRIDGE_LOCK_FRAMES,
        type=int,
        help="桥接结束后保持落地平均姿态，并平滑恢复原运动的帧数。",
    )
    bridge.add_argument(
        "--bridge_knee_strength",
        default=DEFAULT_BRIDGE_KNEE_STRENGTH,
        type=float,
        help="Knee 采用桥接轨迹的强度。",
    )
    bridge.add_argument(
        "--bridge_ankle_strength",
        default=DEFAULT_BRIDGE_ANKLE_STRENGTH,
        type=float,
        help="Ankle 采用桥接轨迹的强度；五点结果可降低以保持 Foot 位置。",
    )
    bridge.add_argument(
        "--bridge_foot_strength",
        default=DEFAULT_BRIDGE_FOOT_STRENGTH,
        type=float,
        help="Foot 朝向采用桥接轨迹的强度。",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


# region 参数与时间权重


def parse_filter_segment(value: str) -> FilterSegment:
    parts = str(value).strip().split(":")
    if len(parts) != 3:
        raise ValueError("segment 应为 start:end:left|right|both。")
    start_seconds = float(parts[0])
    end_seconds = float(parts[1])
    side = parts[2].strip().lower()
    if not np.isfinite([start_seconds, end_seconds]).all():
        raise ValueError("segment 起止时间必须为有限数值。")
    if start_seconds < 0.0 or end_seconds <= start_seconds:
        raise ValueError("segment 必须满足 0 <= start < end。")
    if side not in ("left", "right", "both"):
        raise ValueError("segment side 只支持 left、right 或 both。")
    return FilterSegment(start_seconds, end_seconds, side)


def parse_landing_bridge(value: str) -> LandingBridge:
    parts = str(value).strip().split(":")
    if len(parts) != 3:
        raise ValueError("bridge 应为 start_frame:end_frame:left|right|both。")
    try:
        start_frame = int(parts[0])
        end_frame = int(parts[1])
    except ValueError as exc:
        raise ValueError("bridge 起止位置必须为整数帧号。") from exc
    side = parts[2].strip().lower()
    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError("bridge 必须满足 0 <= start_frame < end_frame。")
    if side not in ("left", "right", "both"):
        raise ValueError("bridge side 只支持 left、right 或 both。")
    return LandingBridge(start_frame, end_frame, side)


def validate_filter_options(
    *,
    window_frames: int,
    strength: float,
    blend_frames: int,
) -> None:
    if int(window_frames) < 3 or int(window_frames) % 2 == 0:
        raise ValueError("window_frames 必须是大于等于 3 的奇数。")
    if not 0.0 < float(strength) <= 1.0:
        raise ValueError("strength 必须位于 (0,1]。")
    if int(blend_frames) < 0:
        raise ValueError("blend_frames 不能为负数。")


def validate_bridge_options(
    *,
    pre_anchor_frames: int,
    post_anchor_frames: int,
    lock_frames: int,
    knee_strength: float,
    ankle_strength: float,
    foot_strength: float,
) -> None:
    if int(pre_anchor_frames) <= 0 or int(post_anchor_frames) <= 0:
        raise ValueError("bridge 前后锚点帧数必须为正整数。")
    if int(lock_frames) < 0:
        raise ValueError("bridge_lock_frames 不能为负数。")
    for name, value in (
        ("bridge_knee_strength", knee_strength),
        ("bridge_ankle_strength", ankle_strength),
        ("bridge_foot_strength", foot_strength),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} 必须位于 [0,1]。")


def smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def smootherstep(value: np.ndarray) -> np.ndarray:
    """五次 minimum-jerk 权重；两端的一阶和二阶导数均为零。"""

    clipped = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return clipped**3 * (clipped * (clipped * 6.0 - 15.0) + 10.0)


def build_joint_filter_alpha(
    *,
    times: np.ndarray,
    segments: tuple[FilterSegment, ...],
    blend_frames: int,
) -> np.ndarray:
    """构造 `[T,24]` 权重；未选中的帧和关节严格保持 0。"""

    frame_times = np.asarray(times, dtype=np.float64)
    alpha = np.zeros((len(frame_times), SMPL_JOINT_COUNT), dtype=np.float64)
    for segment in segments:
        selected = np.flatnonzero(
            (frame_times >= segment.start_seconds)
            & (frame_times <= segment.end_seconds)
        )
        if not len(selected):
            raise ValueError(
                f"segment {segment.start_seconds:g}:{segment.end_seconds:g} "
                "没有覆盖任何 Pose 帧。"
            )
        distance = np.minimum(
            np.arange(len(selected), dtype=np.float64),
            np.arange(len(selected) - 1, -1, -1, dtype=np.float64),
        )
        if int(blend_frames) > 0:
            frame_alpha = smoothstep(distance / float(blend_frames))
        else:
            frame_alpha = np.ones_like(distance)
        sides = ("left", "right") if segment.side == "both" else (segment.side,)
        for side in sides:
            joint_indices = [JOINT_INDEX[name] for name in LEG_JOINT_NAMES[side]]
            alpha[np.ix_(selected, joint_indices)] = np.maximum(
                alpha[np.ix_(selected, joint_indices)],
                frame_alpha[:, None],
            )
    return alpha


# endregion


# region Quaternion geodesic 平滑


def normalize_quaternion_track(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"quaternions 应为 [T,4]，实际为 {values.shape}")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms <= 1e-8) or not np.isfinite(values).all():
        raise ValueError("quaternions 含零 quaternion 或 NaN/Inf。")
    values /= norms
    for frame_index in range(1, len(values)):
        if np.dot(values[frame_index - 1], values[frame_index]) < 0.0:
            values[frame_index] *= -1.0
    return values


def geodesic_smooth_quaternion_track(
    quaternions: np.ndarray,
    *,
    window_frames: int,
) -> np.ndarray:
    """在每帧切空间中做对称加权平均，避免直接平均 quaternion 或 Euler。"""

    values = normalize_quaternion_track(quaternions)
    rotations = Rotation.from_quat(values)
    half_window = int(window_frames) // 2
    offsets = np.arange(-half_window, half_window + 1, dtype=np.int64)
    sigma = max(1.0, float(window_frames) / 3.0)
    weights = np.exp(-0.5 * (offsets.astype(np.float64) / sigma) ** 2)
    weights /= weights.sum()
    result = np.empty_like(values)
    for frame_index in range(len(values)):
        indices = np.clip(frame_index + offsets, 0, len(values) - 1)
        center = rotations[frame_index]
        relative = center.inv() * rotations[indices]
        mean_delta = np.sum(relative.as_rotvec() * weights[:, None], axis=0)
        result[frame_index] = (
            center * Rotation.from_rotvec(mean_delta)
        ).as_quat()
    return normalize_quaternion_track(result)


def filter_local_rotations(
    local_rotations_xyzw: np.ndarray,
    *,
    joint_alpha: np.ndarray,
    window_frames: int,
    strength: float,
) -> np.ndarray:
    """按 `[T,24]` 权重把原始局部旋转轻度插值到平滑结果。"""

    original = np.asarray(local_rotations_xyzw, dtype=np.float64)
    if original.ndim != 3 or original.shape[1:] != (SMPL_JOINT_COUNT, 4):
        raise ValueError(f"local_rotations_xyzw 应为 [T,24,4]，实际为 {original.shape}")
    alpha = np.asarray(joint_alpha, dtype=np.float64)
    if alpha.shape != original.shape[:2]:
        raise ValueError(f"joint_alpha 应为 {original.shape[:2]}，实际为 {alpha.shape}")
    result = original.copy()
    active_joints = np.flatnonzero(np.any(alpha > 0.0, axis=0))
    for joint_index in active_joints.tolist():
        source = normalize_quaternion_track(original[:, joint_index])
        smoothed = geodesic_smooth_quaternion_track(
            source,
            window_frames=int(window_frames),
        )
        source_rotation = Rotation.from_quat(source)
        delta = source_rotation.inv() * Rotation.from_quat(smoothed)
        blend = alpha[:, joint_index] * float(strength)
        filtered = source_rotation * Rotation.from_rotvec(
            delta.as_rotvec() * blend[:, None]
        )
        result[:, joint_index] = normalize_quaternion_track(filtered.as_quat())
    return result.astype(np.float32)


def geodesic_quaternion_mean(quaternions: np.ndarray) -> np.ndarray:
    """对短且相近的稳定锚点求切空间均值。"""

    values = normalize_quaternion_track(quaternions)
    rotations = Rotation.from_quat(values)
    reference = rotations[0]
    mean_delta = np.mean((reference.inv() * rotations).as_rotvec(), axis=0)
    return (reference * Rotation.from_rotvec(mean_delta)).as_quat()


def interpolate_quaternions(
    start_quaternion: np.ndarray,
    end_quaternion: np.ndarray,
    progress: np.ndarray,
) -> np.ndarray:
    start = Rotation.from_quat(np.asarray(start_quaternion, dtype=np.float64))
    end = Rotation.from_quat(np.asarray(end_quaternion, dtype=np.float64))
    delta = (start.inv() * end).as_rotvec()
    values = np.asarray(progress, dtype=np.float64).reshape(-1)
    repeated_start = Rotation.from_quat(
        np.broadcast_to(start.as_quat(), (len(values), 4))
    )
    result = repeated_start * Rotation.from_rotvec(values[:, None] * delta[None])
    return normalize_quaternion_track(result.as_quat())


def bridge_quaternion_track(
    quaternions: np.ndarray,
    *,
    bridge: LandingBridge,
    pre_anchor_frames: int,
    post_anchor_frames: int,
    lock_frames: int,
    strength: float,
) -> np.ndarray:
    """用稳定锚点的 minimum-jerk 轨迹替换异常段，再平滑解除落地锁定。"""

    source = normalize_quaternion_track(quaternions)
    start = int(bridge.start_frame)
    end = int(bridge.end_frame)
    pre_start = start - int(pre_anchor_frames)
    post_end = end + int(post_anchor_frames)
    lock_end = end + int(lock_frames)
    if pre_start < 0:
        raise ValueError("bridge 前方没有足够的稳定锚点帧。")
    if post_end > len(source) or lock_end >= len(source):
        raise ValueError("bridge 后方没有足够的锚点或落地锁定帧。")

    pre_mean = geodesic_quaternion_mean(source[pre_start:start])
    # 后锚点从 end 开始，因此桥接终点本身直接落在稳定平均姿态上。
    post_mean = geodesic_quaternion_mean(source[end:post_end])
    target = source.copy()
    bridge_indices = np.arange(start, end + 1, dtype=np.int64)
    bridge_progress = smootherstep(
        np.linspace(0.0, 1.0, len(bridge_indices), dtype=np.float64)
    )
    target[bridge_indices] = interpolate_quaternions(
        pre_mean,
        post_mean,
        bridge_progress,
    )

    if int(lock_frames) > 0:
        lock_indices = np.arange(end + 1, lock_end + 1, dtype=np.int64)
        restore_progress = smootherstep(
            np.arange(1, len(lock_indices) + 1, dtype=np.float64)
            / float(lock_frames)
        )
        for local_index, frame_index in enumerate(lock_indices.tolist()):
            target[frame_index] = interpolate_quaternions(
                post_mean,
                source[frame_index],
                np.asarray([restore_progress[local_index]]),
            )[0]

    affected = np.arange(start, lock_end + 1, dtype=np.int64)
    source_rotation = Rotation.from_quat(source[affected])
    delta = source_rotation.inv() * Rotation.from_quat(target[affected])
    filtered = source_rotation * Rotation.from_rotvec(
        delta.as_rotvec() * float(strength)
    )
    result = source.copy()
    result[affected] = filtered.as_quat()
    return normalize_quaternion_track(result)


def apply_landing_bridges(
    local_rotations_xyzw: np.ndarray,
    *,
    bridges: tuple[LandingBridge, ...],
    pre_anchor_frames: int,
    post_anchor_frames: int,
    lock_frames: int,
    knee_strength: float,
    ankle_strength: float = 1.0,
    foot_strength: float = 1.0,
) -> np.ndarray:
    """按关节强度应用桥接；Hip 与其他身体区域保持原样。"""

    result = np.asarray(local_rotations_xyzw, dtype=np.float64).copy()
    for bridge in bridges:
        sides = ("left", "right") if bridge.side == "both" else (bridge.side,)
        for side in sides:
            strengths = {
                f"{side}_knee": float(knee_strength),
                f"{side}_ankle": float(ankle_strength),
                f"{side}_foot": float(foot_strength),
            }
            for joint_name, strength in strengths.items():
                if strength <= 0.0:
                    continue
                joint_index = JOINT_INDEX[joint_name]
                result[:, joint_index] = bridge_quaternion_track(
                    result[:, joint_index],
                    bridge=bridge,
                    pre_anchor_frames=int(pre_anchor_frames),
                    post_anchor_frames=int(post_anchor_frames),
                    lock_frames=int(lock_frames),
                    strength=float(strength),
                )
    return result.astype(np.float32)


# endregion


def angular_acceleration_rms(
    quaternions: np.ndarray,
    *,
    fps: float,
    frame_mask: np.ndarray,
) -> float:
    values = normalize_quaternion_track(quaternions)
    if len(values) < 3:
        return 0.0
    rotations = Rotation.from_quat(values)
    angular_velocity = (
        rotations[:-1].inv() * rotations[1:]
    ).as_rotvec() * float(fps)
    acceleration = np.diff(angular_velocity, axis=0) * float(fps)
    selected = np.asarray(frame_mask, dtype=bool)[1:-1]
    if not selected.any():
        return 0.0
    return float(np.sqrt(np.mean(np.sum(acceleration[selected] ** 2, axis=-1))))


def postprocess_unity_pose_demo(
    *,
    input_pose: Path,
    output_pose: Path,
    segments: tuple[FilterSegment, ...],
    bridges: tuple[LandingBridge, ...],
    window_frames: int,
    strength: float,
    blend_frames: int,
    bridge_pre_anchor_frames: int,
    bridge_post_anchor_frames: int,
    bridge_lock_frames: int,
    bridge_knee_strength: float,
    bridge_ankle_strength: float,
    bridge_foot_strength: float,
    overwrite: bool,
) -> tuple[Path, Path]:
    validate_filter_options(
        window_frames=window_frames,
        strength=strength,
        blend_frames=blend_frames,
    )
    validate_bridge_options(
        pre_anchor_frames=bridge_pre_anchor_frames,
        post_anchor_frames=bridge_post_anchor_frames,
        lock_frames=bridge_lock_frames,
        knee_strength=bridge_knee_strength,
        ankle_strength=bridge_ankle_strength,
        foot_strength=bridge_foot_strength,
    )
    if not segments and not bridges:
        raise ValueError("至少需要一个 --segment 或 --bridge。")
    source_path = Path(input_pose).expanduser().resolve()
    output_path = Path(output_pose).expanduser().resolve()
    if source_path == output_path:
        raise ValueError("output_pose 不能覆盖原始 input_pose。")
    if output_path.exists() and not bool(overwrite):
        raise FileExistsError(f"输出已存在；如需覆盖请传 --overwrite：{output_path}")
    pose = load_unity_pose_recording(source_path)
    joint_alpha = build_joint_filter_alpha(
        times=pose.times,
        segments=segments,
        blend_frames=int(blend_frames),
    )
    filtered = np.asarray(pose.local_rotations_xyzw, dtype=np.float32).copy()
    if segments:
        filtered = filter_local_rotations(
            filtered,
            joint_alpha=joint_alpha,
            window_frames=int(window_frames),
            strength=float(strength),
        )
    if bridges:
        filtered = apply_landing_bridges(
            filtered,
            bridges=bridges,
            pre_anchor_frames=int(bridge_pre_anchor_frames),
            post_anchor_frames=int(bridge_post_anchor_frames),
            lock_frames=int(bridge_lock_frames),
            knee_strength=float(bridge_knee_strength),
            ankle_strength=float(bridge_ankle_strength),
            foot_strength=float(bridge_foot_strength),
        )

    payload = json.loads(source_path.read_text(encoding="utf-8"))
    for frame_index, frame in enumerate(payload["frames"]):
        frame["localRotations"] = filtered[frame_index].tolist()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    original_rotations = Rotation.from_quat(
        pose.local_rotations_xyzw.reshape(-1, 4)
    )
    filtered_rotations = Rotation.from_quat(filtered.reshape(-1, 4))
    angular_change = np.linalg.norm(
        (original_rotations.inv() * filtered_rotations).as_rotvec(),
        axis=-1,
    ).reshape(pose.frame_count, SMPL_JOINT_COUNT)
    changed = angular_change > 1e-8
    metrics = {}
    for side, joint_names in LEG_JOINT_NAMES.items():
        for joint_name in joint_names:
            joint_index = JOINT_INDEX[joint_name]
            frame_mask = changed[:, joint_index]
            if not frame_mask.any():
                continue
            before = angular_acceleration_rms(
                pose.local_rotations_xyzw[:, joint_index],
                fps=float(pose.fps),
                frame_mask=frame_mask,
            )
            after = angular_acceleration_rms(
                filtered[:, joint_index],
                fps=float(pose.fps),
                frame_mask=frame_mask,
            )
            metrics[joint_name] = {
                "angular_acceleration_rms_before": before,
                "angular_acceleration_rms_after": after,
                "reduction_ratio": 0.0 if before <= 1e-8 else 1.0 - after / before,
            }

    sidecar_path = output_path.with_suffix(".filter.json")
    sidecar = {
        "experiment": "unity_pose_demo_local_quaternion_filter",
        "input_pose": str(source_path),
        "output_pose": str(output_path),
        "frames": pose.frame_count,
        "fps": pose.fps,
        "window_frames": int(window_frames),
        "strength": float(strength),
        "blend_frames": int(blend_frames),
        "segments": [
            {
                "start_seconds": segment.start_seconds,
                "end_seconds": segment.end_seconds,
                "side": segment.side,
            }
            for segment in segments
        ],
        "bridges": [
            {
                "start_frame": bridge.start_frame,
                "end_frame": bridge.end_frame,
                "side": bridge.side,
            }
            for bridge in bridges
        ],
        "bridge_pre_anchor_frames": int(bridge_pre_anchor_frames),
        "bridge_post_anchor_frames": int(bridge_post_anchor_frames),
        "bridge_lock_frames": int(bridge_lock_frames),
        "bridge_knee_strength": float(bridge_knee_strength),
        "bridge_ankle_strength": float(bridge_ankle_strength),
        "bridge_foot_strength": float(bridge_foot_strength),
        "changed_frames": int(np.any(changed, axis=1).sum()),
        "metrics": metrics,
    }
    sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path, sidecar_path


def main(argv: list[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    segments = tuple(parse_filter_segment(value) for value in args.segment)
    bridges = tuple(parse_landing_bridge(value) for value in args.bridge)
    output_path, sidecar_path = postprocess_unity_pose_demo(
        input_pose=args.input_pose,
        output_pose=args.output_pose,
        segments=segments,
        bridges=bridges,
        window_frames=int(args.window_frames),
        strength=float(args.strength),
        blend_frames=int(args.blend_frames),
        bridge_pre_anchor_frames=int(args.bridge_pre_anchor_frames),
        bridge_post_anchor_frames=int(args.bridge_post_anchor_frames),
        bridge_lock_frames=int(args.bridge_lock_frames),
        bridge_knee_strength=float(args.bridge_knee_strength),
        bridge_ankle_strength=float(args.bridge_ankle_strength),
        bridge_foot_strength=float(args.bridge_foot_strength),
        overwrite=bool(args.overwrite),
    )
    print(f"[demo-filter] wrote {output_path}", flush=True)
    print(f"[demo-filter] wrote {sidecar_path}", flush=True)
    return output_path


if __name__ == "__main__":
    main()
