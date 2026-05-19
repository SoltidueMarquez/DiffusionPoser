from __future__ import annotations

import importlib.util
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from data_loaders.sensor_masking import (
    BODY_VEL_DIM,
    BODY_VEL_START,
    MODEL_INPUT_DIM,
    SENSOR_NAMES,
    TRACKER_POS_DIM,
    TRACKER_POS_START,
    X277_FEATURE_DIM,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    Line2D = None
    HAS_MATPLOTLIB = False


def _has_video_export_backend() -> bool:
    return (
        importlib.util.find_spec("moviepy") is not None
        or importlib.util.find_spec("imageio_ffmpeg") is not None
        or shutil.which("ffmpeg") is not None
    )


HAS_VISUALIZATION_BACKEND = HAS_MATPLOTLIB and _has_video_export_backend()


# region X277 骨架约定

SMPL_JOINT_COUNT = 24
SMPL_JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)
JOINT_INDEX = {name: index for index, name in enumerate(SMPL_JOINT_NAMES)}
TRACKER_JOINT_INDICES = np.array(
    [
        JOINT_INDEX["head"],
        JOINT_INDEX["left_wrist"],
        JOINT_INDEX["right_wrist"],
        JOINT_INDEX["pelvis"],
        JOINT_INDEX["left_foot"],
        JOINT_INDEX["right_foot"],
    ],
    dtype=np.int64,
)
FULL_RECONSTRUCTION_VISUALIZATION_NOTE = (
    "Approximate full reconstruction: 24 joints are decoded from X277 body velocity/root delta; "
    "offline trackers are shown as overlaid markers."
)
KINEMATIC_CHAINS = (
    (0, 3, 6, 9, 12, 15),
    (9, 13, 16, 18, 20, 22),
    (9, 14, 17, 19, 21, 23),
    (0, 1, 4, 7, 10),
    (0, 2, 5, 8, 11),
)

# AMASS -> Unity 的坐标轴交换矩阵在转换器里同名使用。
# 这里反向用于把 Unity Y-up 轨迹转成 Matplotlib 更直观的 Z-up 显示。
AMASS_TO_UNITY = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

# 测试任务文件只保存 X277 clip，不保存 SMPL forward 产生的原始 joints。
# 因此这里保留一套稳定的 SMPL 风格静态骨架，只用于提供默认身体比例。
DEFAULT_SMPL_JOINTS_UNITY = np.array(
    [
        [0.00, 0.95, 0.00],
        [-0.12, 0.90, 0.00],
        [0.12, 0.90, 0.00],
        [0.00, 1.12, 0.00],
        [-0.13, 0.52, 0.02],
        [0.13, 0.52, 0.02],
        [0.00, 1.28, 0.00],
        [-0.13, 0.10, 0.03],
        [0.13, 0.10, 0.03],
        [0.00, 1.45, 0.00],
        [-0.13, 0.04, 0.18],
        [0.13, 0.04, 0.18],
        [0.00, 1.60, 0.00],
        [-0.14, 1.52, 0.00],
        [0.14, 1.52, 0.00],
        [0.00, 1.78, 0.03],
        [-0.30, 1.50, 0.00],
        [0.30, 1.50, 0.00],
        [-0.55, 1.30, 0.00],
        [0.55, 1.30, 0.00],
        [-0.75, 1.12, 0.00],
        [0.75, 1.12, 0.00],
        [-0.84, 1.08, 0.02],
        [0.84, 1.08, 0.02],
    ],
    dtype=np.float64,
)


def make_yaw_rotation(yaw: np.ndarray) -> np.ndarray:
    """构造 Unity Y 轴 yaw 的 local-to-world 旋转矩阵。"""

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotations = np.zeros((yaw.shape[0], 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = cos_yaw
    rotations[:, 0, 2] = sin_yaw
    rotations[:, 1, 1] = 1.0
    rotations[:, 2, 0] = -sin_yaw
    rotations[:, 2, 2] = cos_yaw
    return rotations


def unity_to_z_up_display(points: np.ndarray) -> np.ndarray:
    """把 Unity Y-up 点云转成 Matplotlib 使用的 Z-up 显示坐标。"""

    return points @ AMASS_TO_UNITY.T


def normalize_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """安全归一化单个 3D 向量，避免 tracker 重合时出现 NaN。"""

    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-8:
        return np.asarray(fallback, dtype=np.float64)
    return value / norm


def build_smpl_like_joints_from_tracker_points(tracker_points: np.ndarray) -> np.ndarray:
    """
    用 6 个 tracker 点搭一个 SMPL 拓扑的人体脚手架。

    data_converter 的高质量 overlay 有真实 SMPL joints 作为 seed；测试任务没有这个
    seed，只能从 X277 自身恢复。直接用默认站姿积分速度会让爬行/下蹲动作严重失真，
    所以这里优先信任 head / wrists / waist / feet 六个当前帧 tracker 点，再用稳定的
    人体比例补出 spine、elbow、knee 等中间关节。这个骨架用于“看修复效果”，不作为指标。
    """

    trackers = np.asarray(tracker_points, dtype=np.float64)
    if trackers.shape != (len(SENSOR_NAMES), 3):
        raise ValueError(f"tracker_points 应为 [6,3]，实际为 {trackers.shape}")

    head, left_wrist, right_wrist, pelvis, left_foot, right_foot = trackers
    side_axis = normalize_vector(right_wrist - left_wrist, fallback=np.asarray([1.0, 0.0, 0.0]))
    body_axis = normalize_vector(head - pelvis, fallback=np.asarray([0.0, 1.0, 0.0]))

    joints = DEFAULT_SMPL_JOINTS_UNITY.copy()
    joints[JOINT_INDEX["pelvis"]] = pelvis
    joints[JOINT_INDEX["head"]] = head
    joints[JOINT_INDEX["left_wrist"]] = left_wrist
    joints[JOINT_INDEX["right_wrist"]] = right_wrist
    joints[JOINT_INDEX["left_foot"]] = left_foot
    joints[JOINT_INDEX["right_foot"]] = right_foot

    # 脊柱沿 waist->head 插值，可以自然覆盖站立、下蹲、爬行这几类姿态。
    joints[JOINT_INDEX["spine1"]] = pelvis + body_axis * np.linalg.norm(head - pelvis) * 0.25
    joints[JOINT_INDEX["spine2"]] = pelvis + body_axis * np.linalg.norm(head - pelvis) * 0.45
    joints[JOINT_INDEX["spine3"]] = pelvis + body_axis * np.linalg.norm(head - pelvis) * 0.62
    joints[JOINT_INDEX["neck"]] = pelvis + body_axis * np.linalg.norm(head - pelvis) * 0.78

    shoulder_center = pelvis * 0.35 + head * 0.65
    shoulder_width = float(np.clip(np.linalg.norm(right_wrist - left_wrist) * 0.35, 0.26, 0.55))
    hip_width = float(np.clip(np.linalg.norm(right_foot - left_foot) * 0.18, 0.18, 0.32))

    joints[JOINT_INDEX["left_collar"]] = shoulder_center - side_axis * shoulder_width * 0.35
    joints[JOINT_INDEX["right_collar"]] = shoulder_center + side_axis * shoulder_width * 0.35
    joints[JOINT_INDEX["left_shoulder"]] = shoulder_center - side_axis * shoulder_width * 0.50
    joints[JOINT_INDEX["right_shoulder"]] = shoulder_center + side_axis * shoulder_width * 0.50
    joints[JOINT_INDEX["left_elbow"]] = joints[JOINT_INDEX["left_shoulder"]] * 0.5 + left_wrist * 0.5
    joints[JOINT_INDEX["right_elbow"]] = joints[JOINT_INDEX["right_shoulder"]] * 0.5 + right_wrist * 0.5
    joints[JOINT_INDEX["left_hand"]] = left_wrist + (left_wrist - joints[JOINT_INDEX["left_elbow"]]) * 0.2
    joints[JOINT_INDEX["right_hand"]] = right_wrist + (right_wrist - joints[JOINT_INDEX["right_elbow"]]) * 0.2

    joints[JOINT_INDEX["left_hip"]] = pelvis - side_axis * hip_width * 0.5
    joints[JOINT_INDEX["right_hip"]] = pelvis + side_axis * hip_width * 0.5
    joints[JOINT_INDEX["left_knee"]] = joints[JOINT_INDEX["left_hip"]] * 0.55 + left_foot * 0.45
    joints[JOINT_INDEX["right_knee"]] = joints[JOINT_INDEX["right_hip"]] * 0.55 + right_foot * 0.45
    joints[JOINT_INDEX["left_ankle"]] = joints[JOINT_INDEX["left_hip"]] * 0.25 + left_foot * 0.75
    joints[JOINT_INDEX["right_ankle"]] = joints[JOINT_INDEX["right_hip"]] * 0.25 + right_foot * 0.75
    return joints


# endregion


# region X277 解码

def extract_tracker_positions(features: np.ndarray) -> np.ndarray:
    """从 `[T, 277]` 或 `[B, T, 277]` 特征中提取 tracker root-local 位置。"""

    array = np.asarray(features, dtype=np.float32)
    if array.ndim == 2:
        if array.shape[1] < TRACKER_POS_START + 6 * TRACKER_POS_DIM:
            raise ValueError(f"features 特征维不够，期望至少 {TRACKER_POS_START + 6 * TRACKER_POS_DIM}，实际为 {array.shape}")
        return array[:, TRACKER_POS_START : TRACKER_POS_START + 6 * TRACKER_POS_DIM].reshape(array.shape[0], 6, 3)
    if array.ndim == 3:
        return np.stack([extract_tracker_positions(sample) for sample in array], axis=0)
    raise ValueError(f"features 只能是 `[T, 277]` 或 `[B, T, 277]`，实际为 {array.shape}")


def decode_x277_tracker_positions(features: np.ndarray) -> np.ndarray:
    """
    把 X277 的 6 个 tracker root-local 位置转成 Unity 世界空间 `[T,6,3]`。

    tracker_pos_root_now 属于当前帧 root 坐标，所以和 body_rot 不同，必须先把
    root delta / yaw delta 推进到当前帧，再把 tracker 点从 root-local 放回世界。
    """

    x277 = np.asarray(features, dtype=np.float64)
    if x277.ndim != 2 or x277.shape[1] < 277:
        raise ValueError(f"features 应为 `[T, 277]`，实际为 {x277.shape}")

    root_position = np.zeros(3, dtype=np.float64)
    root_yaw = 0.0
    tracker_frames: list[np.ndarray] = []
    for row in x277:
        prev_root_rotation = make_yaw_rotation(np.asarray([root_yaw], dtype=np.float64))[0]
        delta_xz = row[270:272].astype(np.float64)
        delta_world = np.asarray([delta_xz[0], 0.0, delta_xz[1]], dtype=np.float64) @ prev_root_rotation.T
        root_position = root_position + delta_world
        root_yaw = root_yaw + math.radians(float(row[272]))

        current_root_rotation = make_yaw_rotation(np.asarray([root_yaw], dtype=np.float64))[0]
        tracker_root_positions = row[216:234].reshape(len(SENSOR_NAMES), 3)
        tracker_frames.append(tracker_root_positions @ current_root_rotation.T + root_position)
    return np.stack(tracker_frames, axis=0).astype(np.float32)


def decode_x277_joint_positions_from_body_velocity(
    features: np.ndarray,
    target_fps: float = 60.0,
    seed_joint_positions: np.ndarray | None = None,
) -> np.ndarray:
    """
    用 X277 的 body velocity 和 root delta 近似恢复完整 24 关节轨迹。

    full_reconstruction_current 会补全 body rotation、body velocity、root delta、yaw、contact，
    以及断线 tracker 的 pos/rot。测试 task 里没有保存原始 SMPL joints，所以这里不能做精确 FK；
    采用第一帧 tracker 搭出的近似骨架作为 seed，然后沿时间积分 body velocity，并用 root delta
    约束 pelvis 的水平轨迹。这样视频里看到的是“完整补全后的身体运动”，而不是只看 6 个 tracker 点。
    """

    x277 = np.asarray(features, dtype=np.float64)
    if x277.ndim != 2 or x277.shape[1] < X277_FEATURE_DIM:
        raise ValueError(f"features 应为 `[T, 277]`，实际为 {x277.shape}")
    if target_fps <= 0:
        raise ValueError("target_fps 必须为正数。")
    if x277.shape[0] == 0:
        raise ValueError("features 至少需要 1 帧。")

    if seed_joint_positions is None:
        tracker_positions = decode_x277_tracker_positions(x277)
        joint_positions = build_smpl_like_joints_from_tracker_points(tracker_positions[0]).astype(np.float64)
    else:
        seed = np.asarray(seed_joint_positions, dtype=np.float64)
        if seed.shape != (SMPL_JOINT_COUNT, 3):
            raise ValueError(f"seed_joint_positions 应为 [24, 3]，实际为 {seed.shape}")
        joint_positions = seed.copy()

    decoded_frames = [joint_positions.copy()]
    root_position = joint_positions[JOINT_INDEX["pelvis"]].copy()
    root_yaw = math.radians(float(x277[0, 272]))

    for frame_index in range(1, x277.shape[0]):
        row = x277[frame_index]
        prev_root_rotation = make_yaw_rotation(np.asarray([root_yaw], dtype=np.float64))[0]
        delta_xz = row[270:272].astype(np.float64)
        delta_world = np.asarray([delta_xz[0], 0.0, delta_xz[1]], dtype=np.float64) @ prev_root_rotation.T
        root_position = root_position + delta_world
        root_yaw = root_yaw + math.radians(float(row[272]))

        current_root_rotation = make_yaw_rotation(np.asarray([root_yaw], dtype=np.float64))[0]
        joint_vel_root_now = row[BODY_VEL_START : BODY_VEL_START + BODY_VEL_DIM].reshape(SMPL_JOINT_COUNT, 3)
        joint_vel_world_now = joint_vel_root_now @ current_root_rotation.T
        joint_positions = joint_positions + joint_vel_world_now / float(target_fps)

        # body velocity 控制垂直和局部姿态变化，root delta 控制水平位移；二者同时使用能减少长期漂移。
        pelvis = joint_positions[JOINT_INDEX["pelvis"]]
        horizontal_correction = np.asarray(
            [root_position[0] - pelvis[0], 0.0, root_position[2] - pelvis[2]],
            dtype=np.float64,
        )
        joint_positions = joint_positions + horizontal_correction[None]
        decoded_frames.append(joint_positions.copy())

    return np.stack(decoded_frames, axis=0).astype(np.float32)


# endregion


# region 渲染

def render_full_reconstruction_visualization(
    *,
    reference_motion: np.ndarray,
    conditioned_motion: np.ndarray,
    reconstructed_motion: np.ndarray,
    sensor_missing_labels: np.ndarray,
    inpaint_mask: np.ndarray,
    output_path: Path,
    fps: float,
    title: str,
    valid_length: int | None = None,
    x277_fps: float = 60.0,
) -> dict:
    """
    生成 current277 完整补全视频，对比条件输入、模型补全和 GT。

    这里主体骨架来自 body velocity/root delta 的近似解码，
    因此可以观察 body/root/contact 这类完整补全目标造成的整体姿态和轨迹变化。
    """

    if not HAS_VISUALIZATION_BACKEND:
        raise ImportError("缺少 matplotlib 或可用的 ffmpeg 导出后端，无法生成 mp4 可视化。")
    if fps <= 0:
        raise ValueError("fps 必须为正数。")

    labels = np.asarray(sensor_missing_labels, dtype=bool)
    if labels.ndim != 2 or labels.shape[1] != len(SENSOR_NAMES):
        raise ValueError(f"sensor_missing_labels 应为 [T, 6]，实际为 {labels.shape}")

    mask = np.asarray(inpaint_mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[1] not in {X277_FEATURE_DIM, MODEL_INPUT_DIM}:
        raise ValueError(f"inpaint_mask 应为 [T, 277] 或 [T, 283]，实际为 {mask.shape}")
    target_frame_mask = mask[:, :X277_FEATURE_DIM].any(axis=1)

    # 三条轨迹共享同一个 seed，避免条件输入里目标帧被清零后导致第一帧骨架塌缩，影响横向比较。
    seed_trackers = decode_x277_tracker_positions(reference_motion)
    seed_joints = build_smpl_like_joints_from_tracker_points(seed_trackers[0])
    reference_joints = decode_x277_joint_positions_from_body_velocity(
        reference_motion,
        target_fps=x277_fps,
        seed_joint_positions=seed_joints,
    )
    conditioned_joints = decode_x277_joint_positions_from_body_velocity(
        conditioned_motion,
        target_fps=x277_fps,
        seed_joint_positions=seed_joints,
    )
    reconstructed_joints = decode_x277_joint_positions_from_body_velocity(
        reconstructed_motion,
        target_fps=x277_fps,
        seed_joint_positions=seed_joints,
    )

    reference_trackers = decode_x277_tracker_positions(reference_motion)
    conditioned_trackers = decode_x277_tracker_positions(conditioned_motion)
    reconstructed_trackers = decode_x277_tracker_positions(reconstructed_motion)

    frame_count = min(
        reference_joints.shape[0],
        conditioned_joints.shape[0],
        reconstructed_joints.shape[0],
        reference_trackers.shape[0],
        conditioned_trackers.shape[0],
        reconstructed_trackers.shape[0],
        labels.shape[0],
        target_frame_mask.shape[0],
    )
    if valid_length is not None:
        frame_count = min(frame_count, int(valid_length))
    if frame_count <= 0:
        raise ValueError("没有可视化帧。")

    tracks = [
        unity_to_z_up_display(conditioned_joints[:frame_count]).copy(),
        unity_to_z_up_display(reconstructed_joints[:frame_count]).copy(),
        unity_to_z_up_display(reference_joints[:frame_count]).copy(),
    ]
    tracker_tracks = [
        unity_to_z_up_display(conditioned_trackers[:frame_count]).copy(),
        unity_to_z_up_display(reconstructed_trackers[:frame_count]).copy(),
        unity_to_z_up_display(reference_trackers[:frame_count]).copy(),
    ]

    min_z = min(
        min(float(track[:, :, 2].min()) for track in tracks),
        min(float(track[:, :, 2].min()) for track in tracker_tracks),
    )
    for track in tracks + tracker_tracks:
        track[:, :, 2] -= min_z
    all_points = np.concatenate([track.reshape(-1, 3) for track in tracks + tracker_tracks], axis=0)
    center_xy = all_points[:, :2].mean(axis=0)
    xy_span = np.ptp(all_points[:, :2], axis=0)
    radius = max(float(xy_span.max()) * 0.55, 1.5)
    z_max = max(float(all_points[:, 2].max()), 1.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 6), dpi=120)
    captured_frames: list[np.ndarray] = []
    frame_meta: list[dict] = []

    for frame_index in tqdm(range(frame_count), desc=f"Rendering {title}", leave=False, unit="frame"):
        frame_labels = labels[frame_index]
        missing_sensor_indices = np.flatnonzero(frame_labels).tolist()
        missing_sensor_names = [SENSOR_NAMES[index] for index in missing_sensor_indices]
        is_target_frame = bool(target_frame_mask[frame_index])
        highlighted_joints = (
            np.arange(SMPL_JOINT_COUNT, dtype=np.int64)
            if is_target_frame
            else TRACKER_JOINT_INDICES[missing_sensor_indices]
            if missing_sensor_indices
            else np.asarray([], dtype=np.int64)
        )

        fig.clf()
        axes = [
            fig.add_subplot(1, 3, 1, projection="3d"),
            fig.add_subplot(1, 3, 2, projection="3d"),
            fig.add_subplot(1, 3, 3, projection="3d"),
        ]
        panel_specs = [
            ("Conditioned input", tracks[0], tracker_tracks[0], "#6B7280", "#F97316"),
            ("Reconstructed output", tracks[1], tracker_tracks[1], "#2563EB", "#22D3EE"),
            ("Ground truth", tracks[2], tracker_tracks[2], "#059669", "#059669"),
        ]
        for ax, (panel_title, track, tracker_track, color, target_color) in zip(axes, panel_specs):
            configure_visualization_axis(ax=ax, center_xy=center_xy, radius=radius, z_max=z_max, title=panel_title)
            draw_ground_grid(ax=ax, center_xy=center_xy, radius=radius)
            draw_skeleton(
                ax=ax,
                joints=track[frame_index],
                color=color,
                damaged_color=target_color,
                highlighted_joint_indices=highlighted_joints,
                alpha=0.72,
                linewidth=2.8,
            )
            draw_tracker_points(
                ax=ax,
                tracker_points=tracker_track[frame_index],
                color=color,
                damaged_color=target_color,
                damaged_sensor_indices=missing_sensor_indices,
            )

        target_text = "yes" if is_target_frame else "no"
        missing_text = ", ".join(missing_sensor_names) if missing_sensor_names else "none"
        headline = (
            f"{title}\nFrame {frame_index + 1}/{frame_count} | "
            f"Full reconstruction target: {target_text} | Offline sensors: {missing_text}"
        )
        fig.text(0.02, 0.97, headline, va="top", ha="left", fontsize=10, color="black")
        fig.text(0.02, 0.025, FULL_RECONSTRUCTION_VISUALIZATION_NOTE, va="bottom", ha="left", fontsize=8, color="#4B5563")
        legend_handles = [
            Line2D([0], [0], color="#6B7280", lw=4, alpha=0.8, label="Conditioned"),
            Line2D([0], [0], color="#2563EB", lw=4, alpha=0.8, label="Reconstructed"),
            Line2D([0], [0], color="#059669", lw=4, alpha=0.8, label="Ground truth"),
            Line2D([0], [0], color="#F97316", lw=4, alpha=0.9, label="Target / offline"),
        ]
        axes[1].legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=4, frameon=False)
        draw_timeline(fig=fig, frame_index=frame_index, frame_count=frame_count)

        fig.canvas.draw()
        frame_rgba = np.asarray(fig.canvas.buffer_rgba())
        captured_frames.append(frame_rgba[..., :3].copy())
        frame_meta.append(
            {
                "frame_index": frame_index,
                "is_reconstruction_target": is_target_frame,
                "missing_sensor_indices": missing_sensor_indices,
                "missing_sensor_names": missing_sensor_names,
            }
        )

    plt.close(fig)
    export_visualization_frames(frames=captured_frames, output_path=output_path, fps=fps)
    return {
        "output_path": str(output_path),
        "fps": float(fps),
        "x277_fps": float(x277_fps),
        "frame_count": frame_count,
        "frames": frame_meta,
        "visualization": "current277_full_reconstruction",
        "note": FULL_RECONSTRUCTION_VISUALIZATION_NOTE,
    }


def configure_visualization_axis(ax: Any, center_xy: np.ndarray, radius: float, z_max: float, title: str) -> None:
    ax.view_init(elev=22.0, azim=-60.0)
    ax.set_xlim(center_xy[0] - radius, center_xy[0] + radius)
    ax.set_ylim(center_xy[1] - radius, center_xy[1] + radius)
    ax.set_zlim(0.0, z_max + 0.2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z / Up")
    ax.set_title(title, pad=10)
    ax.set_box_aspect((1, 1, 0.8))


def draw_ground_grid(ax: Any, center_xy: np.ndarray, radius: float, divisions: int = 10) -> None:
    min_x, max_x = center_xy[0] - radius, center_xy[0] + radius
    min_y, max_y = center_xy[1] - radius, center_xy[1] + radius
    xs = np.linspace(min_x, max_x, divisions + 1)
    ys = np.linspace(min_y, max_y, divisions + 1)
    for x_value in xs:
        ax.plot([x_value, x_value], [min_y, max_y], [0.0, 0.0], color="lightgray", linewidth=0.6, alpha=0.45)
    for y_value in ys:
        ax.plot([min_x, max_x], [y_value, y_value], [0.0, 0.0], color="lightgray", linewidth=0.6, alpha=0.45)


def draw_skeleton(
    *,
    ax: Any,
    joints: np.ndarray,
    color: str,
    damaged_color: str,
    highlighted_joint_indices: np.ndarray,
    alpha: float,
    linewidth: float,
) -> None:
    highlighted = set(int(index) for index in highlighted_joint_indices)
    for chain in KINEMATIC_CHAINS:
        chain_indices = np.asarray(chain, dtype=np.int64)
        chain_points = joints[chain_indices]
        for start in range(len(chain_indices) - 1):
            segment_indices = chain_indices[start : start + 2]
            segment_points = chain_points[start : start + 2]
            is_damaged_segment = any(int(index) in highlighted for index in segment_indices)
            ax.plot(
                segment_points[:, 0],
                segment_points[:, 1],
                segment_points[:, 2],
                color=damaged_color if is_damaged_segment else color,
                alpha=0.95 if is_damaged_segment else alpha,
                linewidth=linewidth + 0.9 if is_damaged_segment else linewidth,
            )

    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color=color, alpha=min(alpha + 0.15, 1.0), s=9)
    if highlighted:
        highlighted_indices = np.asarray(sorted(highlighted), dtype=np.int64)
        highlighted_points = joints[highlighted_indices]
        ax.scatter(
            highlighted_points[:, 0],
            highlighted_points[:, 1],
            highlighted_points[:, 2],
            color=damaged_color,
            alpha=1.0,
            s=42,
            depthshade=False,
        )


def draw_tracker_points(
    *,
    ax: Any,
    tracker_points: np.ndarray,
    color: str,
    damaged_color: str,
    damaged_sensor_indices: list[int],
) -> None:
    """
    单独绘制 6 个 tracker，而不是把它们覆盖进人体关节。

    这样即使 corrupted input 里某个 tracker 被清零，人体骨架仍然保持 body pose
    的人形结构；清零后的 tracker 点会落到 root 附近，作为“传感器输入已损坏”
    的可视化证据出现。
    """

    points = np.asarray(tracker_points, dtype=np.float64)
    damaged = set(int(index) for index in damaged_sensor_indices)
    normal_indices = [index for index in range(len(SENSOR_NAMES)) if index not in damaged]
    if normal_indices:
        normal_points = points[np.asarray(normal_indices, dtype=np.int64)]
        ax.scatter(
            normal_points[:, 0],
            normal_points[:, 1],
            normal_points[:, 2],
            color=color,
            marker="^",
            alpha=0.78,
            s=34,
            depthshade=False,
        )
    if damaged:
        damaged_points = points[np.asarray(sorted(damaged), dtype=np.int64)]
        ax.scatter(
            damaged_points[:, 0],
            damaged_points[:, 1],
            damaged_points[:, 2],
            color=damaged_color,
            marker="X",
            alpha=1.0,
            s=76,
            depthshade=False,
        )


def draw_timeline(fig: Any, frame_index: int, frame_count: int) -> None:
    progress = (frame_index + 1) / max(frame_count, 1)
    timeline = fig.add_axes([0.08, 0.035, 0.84, 0.025])
    timeline.set_xlim(0.0, 1.0)
    timeline.set_ylim(0.0, 1.0)
    timeline.axis("off")
    timeline.barh([0.5], [1.0], height=0.45, color="#DDDDDD")
    timeline.barh([0.5], [progress], height=0.45, color="#2F80ED")
    timeline.text(0.0, -0.8, "start", fontsize=8, ha="left")
    timeline.text(1.0, -0.8, "end", fontsize=8, ha="right")


def export_visualization_frames(frames: list[np.ndarray], output_path: Path, fps: float) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    try:
        import moviepy.editor as mp

        clip = mp.ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(str(output_path), codec="libx264", audio=False, verbose=False, logger=None)
        clip.close()
        return output_path
    except Exception as exc:
        errors.append(f"moviepy: {exc!r}")

    try:
        import imageio.v2 as imageio

        with imageio.get_writer(str(output_path), fps=fps, codec="libx264", macro_block_size=16) as writer:
            for frame in frames:
                writer.append_data(frame)
        return output_path
    except Exception as exc:
        errors.append(f"imageio_ffmpeg: {exc!r}")

    try:
        from matplotlib.animation import FFMpegWriter

        height, width, _ = frames[0].shape
        dpi = 120
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        image = ax.imshow(frames[0])
        writer = FFMpegWriter(fps=fps, codec="libx264")
        with writer.saving(fig, str(output_path), dpi=dpi):
            for frame in frames:
                image.set_data(frame)
                writer.grab_frame()
        plt.close(fig)
        return output_path
    except Exception as exc:
        errors.append(f"matplotlib_ffmpeg: {exc!r}")

    raise RuntimeError(
        "MP4 可视化导出失败。请安装 moviepy 和 imageio-ffmpeg，或把 ffmpeg 加入 PATH。"
        f" 详细错误: {' | '.join(errors)}"
    )


# endregion
