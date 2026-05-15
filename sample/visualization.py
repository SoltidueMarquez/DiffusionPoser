from __future__ import annotations

import importlib.util
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from data_loaders.sensor_masking import SENSOR_NAMES, TRACKER_POS_DIM, TRACKER_POS_START

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
VISUALIZATION_LIMITATION_NOTE = (
    "Approximate X277 skeleton: body pose is scaffolded from six tracker points; "
    "damaged/repaired trackers are overlaid as markers."
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


def decode_x277_joint_positions(
    features: np.ndarray,
    target_fps: float = 60.0,
    seed_joint_positions: np.ndarray | None = None,
) -> np.ndarray:
    """
    从 X277 近似解码 `[T, 24, 3]` 的 Unity 世界空间人体骨架。

    转换器的 overlay 可以拿到真实 SMPL joints，所以蓝色 decoded X277 看起来很稳。
    测试任务只有 X277 特征，缺少那个 seed；这里改用 6 个 tracker 点搭一个
    SMPL 拓扑脚手架，优先保证“看起来是人”，再把损坏/修复 tracker 作为点叠加。
    """

    x277 = np.asarray(features, dtype=np.float64)
    if x277.ndim != 2 or x277.shape[1] < 277:
        raise ValueError(f"features 应为 `[T, 277]`，实际为 {x277.shape}")
    if target_fps <= 0:
        raise ValueError("target_fps 必须为正数。")
    if seed_joint_positions is not None:
        seed = np.asarray(seed_joint_positions, dtype=np.float64)
        if seed.shape != (SMPL_JOINT_COUNT, 3):
            raise ValueError(f"seed_joint_positions 应为 [24, 3]，实际为 {seed.shape}")

    tracker_positions = decode_x277_tracker_positions(x277)
    decoded_frames = [build_smpl_like_joints_from_tracker_points(frame_trackers) for frame_trackers in tracker_positions]
    return np.stack(decoded_frames, axis=0).astype(np.float32)


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


# endregion


# region 渲染

def render_fix_visualization(
    *,
    reference_motion: np.ndarray,
    corrupted_motion: np.ndarray,
    repaired_motion: np.ndarray,
    sensor_missing_labels: np.ndarray,
    output_path: Path,
    fps: float,
    title: str,
    valid_length: int | None = None,
    x277_fps: float = 60.0,
) -> dict:
    """
    生成修复结果视频，三栏对比损坏输入、修复输出和 GT。

    视频强调的是损坏 tracker 在近似人体骨架端点上的变化；它不是正式评估指标，
    后续如果要做定量结果，仍应直接基于 X277 或反解后的 SMPL 数据计算。
    """

    if not HAS_VISUALIZATION_BACKEND:
        raise ImportError("缺少 matplotlib 或可用的 ffmpeg 导出后端，无法生成 mp4 可视化。")
    if fps <= 0:
        raise ValueError("fps 必须为正数。")

    ref_joints = decode_x277_joint_positions(reference_motion, target_fps=x277_fps)
    # 人体骨架只作为姿态上下文：fix-only 任务真正被改写的是 tracker position/rotation。
    # 如果对 corrupted input 也用被清零的 tracker 重搭骨架，骨架会被缺失值拉塌，反而掩盖修复效果。
    # 因此三栏共用 GT/reference 的人体脚手架，各自只叠加自己的 tracker 点。
    corrupted_joints = ref_joints.copy()
    repaired_joints = ref_joints.copy()
    ref_trackers = decode_x277_tracker_positions(reference_motion)
    corrupted_trackers = decode_x277_tracker_positions(corrupted_motion)
    repaired_trackers = decode_x277_tracker_positions(repaired_motion)
    labels = np.asarray(sensor_missing_labels, dtype=bool)
    if labels.ndim != 2 or labels.shape[1] != len(SENSOR_NAMES):
        raise ValueError(f"sensor_missing_labels 应为 [T, 6]，实际为 {labels.shape}")

    frame_count = min(
        ref_joints.shape[0],
        corrupted_joints.shape[0],
        repaired_joints.shape[0],
        ref_trackers.shape[0],
        corrupted_trackers.shape[0],
        repaired_trackers.shape[0],
        labels.shape[0],
    )
    if valid_length is not None:
        frame_count = min(frame_count, int(valid_length))
    if frame_count <= 0:
        raise ValueError("没有可视化帧。")

    tracks = [
        unity_to_z_up_display(corrupted_joints[:frame_count]).copy(),
        unity_to_z_up_display(repaired_joints[:frame_count]).copy(),
        unity_to_z_up_display(ref_joints[:frame_count]).copy(),
    ]
    tracker_tracks = [
        unity_to_z_up_display(corrupted_trackers[:frame_count]).copy(),
        unity_to_z_up_display(repaired_trackers[:frame_count]).copy(),
        unity_to_z_up_display(ref_trackers[:frame_count]).copy(),
    ]

    # 和转换器保持一致：三条轨迹使用同一块地面和同一套空间尺度，避免视觉比较被坐标轴缩放误导。
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
        highlighted_joints = TRACKER_JOINT_INDICES[missing_sensor_indices] if missing_sensor_indices else np.asarray([], dtype=np.int64)

        fig.clf()
        axes = [
            fig.add_subplot(1, 3, 1, projection="3d"),
            fig.add_subplot(1, 3, 2, projection="3d"),
            fig.add_subplot(1, 3, 3, projection="3d"),
        ]
        panel_specs = [
            ("Corrupted input", tracks[0], tracker_tracks[0], "#6B7280", "#F97316"),
            ("Repaired output", tracks[1], tracker_tracks[1], "#2563EB", "#22D3EE"),
            ("Ground truth", tracks[2], tracker_tracks[2], "#059669", "#059669"),
        ]
        for ax, (panel_title, track, tracker_track, color, damaged_color) in zip(axes, panel_specs):
            configure_visualization_axis(ax=ax, center_xy=center_xy, radius=radius, z_max=z_max, title=panel_title)
            draw_ground_grid(ax=ax, center_xy=center_xy, radius=radius)
            draw_skeleton(
                ax=ax,
                joints=track[frame_index],
                color=color,
                damaged_color=damaged_color,
                highlighted_joint_indices=highlighted_joints,
                alpha=0.72,
                linewidth=2.8,
            )
            draw_tracker_points(
                ax=ax,
                tracker_points=tracker_track[frame_index],
                color=color,
                damaged_color=damaged_color,
                damaged_sensor_indices=missing_sensor_indices,
            )

        headline = f"{title}\nFrame {frame_index + 1}/{frame_count} | Damaged sensors: {', '.join(missing_sensor_names) if missing_sensor_names else 'none'}"
        fig.text(0.02, 0.97, headline, va="top", ha="left", fontsize=10, color="black")
        fig.text(0.02, 0.025, VISUALIZATION_LIMITATION_NOTE, va="bottom", ha="left", fontsize=8, color="#4B5563")
        legend_handles = [
            Line2D([0], [0], color="#6B7280", lw=4, alpha=0.8, label="Corrupted"),
            Line2D([0], [0], color="#2563EB", lw=4, alpha=0.8, label="Repaired"),
            Line2D([0], [0], color="#059669", lw=4, alpha=0.8, label="Ground truth"),
            Line2D([0], [0], color="#F97316", lw=4, alpha=0.9, label="Damaged part"),
        ]
        axes[1].legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=4, frameon=False)
        draw_timeline(fig=fig, frame_index=frame_index, frame_count=frame_count)

        fig.canvas.draw()
        frame_rgba = np.asarray(fig.canvas.buffer_rgba())
        captured_frames.append(frame_rgba[..., :3].copy())
        frame_meta.append(
            {
                "frame_index": frame_index,
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
        "visualization": "x277_smpl_skeleton",
        "note": VISUALIZATION_LIMITATION_NOTE,
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
