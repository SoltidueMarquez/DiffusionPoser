"""
把 AMASS SMPL/SMPL-H 动作转换为 DiffusionPoser 项目使用的 X277 帧特征。

输出的每一行对应目标时间轴中的当前帧 `t`，其中 body 旋转和 body 速度来自上一帧，
tracker/root delta/contact 来自当前帧或当前帧相邻关系。这样可以和 Unity
tracking_net 的“上一帧身体状态 + 当前稀疏 tracker 条件”读取习惯保持一致。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm


FEATURE_DIM = 277
SCHEMA_NAME = "current277_v1"
SMPL_JOINT_COUNT = 24
SOURCE_BODY_JOINT_COUNT = 22
SMPLH_JOINT_COUNT = 52
TARGET_FPS_DEFAULT = 60.0

TRACKER_NAMES = (
    "head",
    "left_wrist",
    "right_wrist",
    "waist",
    "left_foot",
    "right_foot",
)
CONTACT_NAMES = ("left_heel", "left_toe", "right_heel", "right_toe")
KINEMATIC_CHAINS = (
    (0, 3, 6, 9, 12, 15),
    (9, 13, 16, 18, 20, 22),
    (9, 14, 17, 19, 21, 23),
    (0, 1, 4, 7, 10),
    (0, 2, 5, 8, 11),
)

# SMPL 24 关节顺序使用常见 LBS kinematic tree。这里显式写出名字，是为了让
# tracker 映射和 contact 调试时不需要反查 SMPL 论文或模型文件。
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

# StableMotion 提供的是 SMPL-H 模型。SMPL-H 的 body joint 顺序前 22 个和 AMASS
# body pose 对齐，左右 hand leaf 需要从手部 index joint 中抽取出来，才能还原成
# 本项目需要的 24 个 SMPL 风格 body segment。
SMPLH_TO_SMPL_JOINT_INDICES = np.array(
    list(range(23)) + [37],
    dtype=np.int64,
)
SMPLH_LEFT_HAND_START = 22
SMPLH_RIGHT_HAND_START = 37
MIRROR_DIR_NAME = "M"

# 与 StableMotion/data_loaders/amasstools/smpl_mirroring.py 保持一致：
# 镜像时先交换 SMPL body 的左右关节，再对 axis-angle 的 y/z 分量取负。
SMPL_BODY_JOINTS_FLIP_PERM = np.array(
    [
        0,
        2,
        1,
        3,
        5,
        4,
        6,
        8,
        7,
        9,
        11,
        10,
        12,
        14,
        13,
        15,
        17,
        16,
        19,
        18,
        21,
        20,
    ],
    dtype=np.int64,
)

# 如果 smplx 模型对象没有暴露 parents，则使用标准 SMPL 24 关节父子关系。
DEFAULT_SMPL_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)

FEATURE_SLICES = {
    "body_rot_root_now": [0, 144],
    "body_vel_root_now": [144, 216],
    "tracker_pos_root_now": [216, 234],
    "tracker_rot_root_now": [234, 270],
    "root_delta_xz": [270, 272],
    "root_yaw_delta_degree": [272, 273],
    "contact_cur": [273, 277],
}

# AMASS -> Unity 使用 (x, y, z)_unity = (x, z, y)_amass。这个矩阵包含一次轴交换，
# 对位置、速度和旋转都使用同一个基变换，避免“位置已转、朝向未转”的隐性错误。
AMASS_TO_UNITY = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class MotionSource:
    """一个 AMASS 动作文件在转换流程中的最小输入表示。"""

    path: Path
    relative_path: Path
    poses: np.ndarray
    trans: np.ndarray
    betas: np.ndarray
    gender: str
    source_fps: float
    is_mirrored: bool = False
    original_relative_path: Path | None = None


@dataclass(frozen=True)
class SmplMotion:
    """SMPL 前向推理后的世界空间数据，全部已经转换到 Unity 坐标系。"""

    raw_joint_positions: np.ndarray
    joint_positions: np.ndarray
    joint_rotations: np.ndarray
    vertices: np.ndarray
    rest_joints: np.ndarray
    rest_vertices: np.ndarray
    parents: np.ndarray


@dataclass(frozen=True)
class RootFrames:
    """只保留水平位移和 yaw 的 root frame，用于把世界量转为 Root 局部量。"""

    positions: np.ndarray
    rotations: np.ndarray
    yaws: np.ndarray


# region 参数解析


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AMASS SMPL motions to DiffusionPoser current277 files.")

    group = parser.add_argument_group("paths")
    group.add_argument("--amass_dir", default="dataset/AMASS", type=Path, help="AMASS npz root directory.")
    group.add_argument(
        "--output_dir",
        default="dataset/AMASS_current277_60hz",
        type=Path,
        help="Directory that mirrors AMASS relative paths and stores converted npz files.",
    )
    group.add_argument(
        "--smpl_model_dir",
        default="dataset/body_models",
        type=Path,
        help="Directory containing SMPL model files used by smplx.",
    )

    group = parser.add_argument_group("conversion")
    group.add_argument("--target_fps", default=TARGET_FPS_DEFAULT, type=float, help="Unified output frame rate.")
    group.add_argument("--batch_size", default=256, type=int, help="SMPL forward batch size.")
    group.add_argument("--limit", default=0, type=int, help="Convert at most this many valid AMASS motion files.")
    group.add_argument("--overwrite", action="store_true", help="Overwrite existing converted npz files.")
    group.add_argument("--skip_existing", action="store_true", help="Skip output files that already exist.")
    group.add_argument(
        "--mirror",
        dest="mirror",
        action="store_true",
        default=True,
        help="同时生成 StableMotion 风格的 M/ 镜像样本，默认开启。",
    )
    group.add_argument(
        "--no_mirror",
        dest="mirror",
        action="store_false",
        help="只转换原始动作，不生成 M/ 镜像样本。",
    )

    group = parser.add_argument_group("visualization")
    group.add_argument(
        "--visualize_num",
        default=0,
        type=int,
        help="渲染前 N 个动作的三轨叠加视频；0 表示不渲染，<0 表示渲染全部成功样本。",
    )
    group.add_argument(
        "--visualize_dir",
        default=None,
        type=Path,
        help="可视化视频输出目录；默认使用 <output_dir>/visualizations。",
    )
    group.add_argument("--visualize_fps", default=20.0, type=float, help="可视化视频帧率。")

    group = parser.add_argument_group("contact")
    group.add_argument(
        "--height_threshold",
        default=0.04,
        type=float,
        help="Foot contact height threshold in meters after ground normalization.",
    )
    group.add_argument(
        "--speed_threshold",
        default=0.20,
        type=float,
        help="Foot contact horizontal speed threshold in meters/second.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.amass_dir.exists():
        raise FileNotFoundError(f"AMASS 数据目录不存在: {args.amass_dir}")
    if not args.smpl_model_dir.exists():
        raise FileNotFoundError(
            f"SMPL 模型目录不存在: {args.smpl_model_dir}。请下载 SMPL 模型后用 --smpl_model_dir 指定。"
        )
    if args.target_fps <= 0:
        raise ValueError("--target_fps 必须为正数。")
    if args.batch_size <= 0:
        raise ValueError("--batch_size 必须为正整数。")
    if args.overwrite and args.skip_existing:
        raise ValueError("--overwrite 和 --skip_existing 不能同时启用。")
    if args.visualize_fps <= 0:
        raise ValueError("--visualize_fps 必须为正数。")
    if args.visualize_dir is None:
        args.visualize_dir = args.output_dir / "visualizations"


# endregion


# region 文件读取与重采样


def iter_amass_motion_files(amass_dir: Path) -> list[Path]:
    """收集真正包含 motion 的 AMASS 文件，跳过 DFaust 等只有 shape/betas 的 npz。"""

    return sorted(
        path
        for path in amass_dir.rglob("*.npz")
        if path.name != "shape.npz" and not is_mirror_relative_path(path.relative_to(amass_dir))
    )


def is_mirror_relative_path(relative_path: Path) -> bool:
    """StableMotion 把镜像数据放在 M/ 下；这里避免把镜像样本再次镜像。"""

    return bool(relative_path.parts) and relative_path.parts[0] == MIRROR_DIR_NAME


def load_motion_source(path: Path, amass_dir: Path, target_fps: float) -> MotionSource:
    with np.load(path, allow_pickle=True) as data:
        missing_keys = [key for key in ("poses", "trans", "mocap_framerate") if key not in data.files]
        if missing_keys:
            raise ValueError(f"缺少必要字段: {missing_keys}")

        poses = np.asarray(data["poses"], dtype=np.float64)
        trans = np.asarray(data["trans"], dtype=np.float64)
        source_fps = float(np.asarray(data["mocap_framerate"]).item())

        if poses.ndim != 2 or poses.shape[1] < SOURCE_BODY_JOINT_COUNT * 3:
            raise ValueError(f"poses 应至少为 [T,66]，实际为 {poses.shape}")
        if trans.shape != (poses.shape[0], 3):
            raise ValueError(f"trans 应为 [T,3] 且与 poses 同帧数，实际为 {trans.shape}")
        if poses.shape[0] < 3:
            raise ValueError("原始帧数少于 3，无法构造上一帧速度。")

        betas = np.asarray(data["betas"], dtype=np.float64) if "betas" in data.files else np.zeros(10)
        gender = normalize_gender(data["gender"] if "gender" in data.files else "neutral")

    poses_resampled, trans_60hz = resample_motion_to_target_fps(
        poses=poses,
        trans=trans,
        source_fps=source_fps,
        target_fps=target_fps,
    )
    if poses_resampled.shape[0] < 3:
        raise ValueError(f"重采样到 {target_fps:g}Hz 后少于 3 帧，无法构造 X277。")

    return MotionSource(
        path=path,
        relative_path=path.relative_to(amass_dir),
        poses=poses_resampled,
        trans=trans_60hz,
        betas=betas,
        gender=gender,
        source_fps=source_fps,
        original_relative_path=path.relative_to(amass_dir),
    )


def mirror_motion_source(source: MotionSource) -> MotionSource:
    """
    生成 StableMotion 风格的镜像动作，不写回 AMASS 原始文件。

    输出相对路径加上 `M/` 前缀，和 StableMotion split.txt 中的镜像命名保持一致：
    `BMLmovi/a/b.npy` -> `M/BMLmovi/a/b.npz`。
    """

    return MotionSource(
        path=source.path,
        relative_path=Path(MIRROR_DIR_NAME) / source.relative_path,
        poses=mirror_axis_angle_poses(source.poses),
        trans=mirror_translations(source.trans),
        betas=source.betas.copy(),
        gender=source.gender,
        source_fps=source.source_fps,
        is_mirrored=True,
        original_relative_path=source.original_relative_path or source.relative_path,
    )


def build_pose_flip_permutation(joint_count: int) -> np.ndarray:
    if joint_count < SOURCE_BODY_JOINT_COUNT:
        raise ValueError(f"镜像 pose 至少需要 {SOURCE_BODY_JOINT_COUNT} 个 joint，实际为 {joint_count}")

    joint_permutation = list(SMPL_BODY_JOINTS_FLIP_PERM)
    if joint_count >= SMPLH_JOINT_COUNT:
        # SMPL-H 的手部顺序为 left hand 15 joints 后接 right hand 15 joints；
        # 镜像时左右手整体交换，手内部的 MANO joint 顺序保持不变。
        joint_permutation.extend(range(SMPLH_RIGHT_HAND_START, SMPLH_JOINT_COUNT))
        joint_permutation.extend(range(SMPLH_LEFT_HAND_START, SMPLH_RIGHT_HAND_START))
        joint_permutation.extend(range(SMPLH_JOINT_COUNT, joint_count))
    else:
        joint_permutation.extend(range(SOURCE_BODY_JOINT_COUNT, joint_count))

    pose_permutation: list[int] = []
    for joint_index in joint_permutation:
        pose_permutation.extend([3 * joint_index, 3 * joint_index + 1, 3 * joint_index + 2])
    return np.asarray(pose_permutation, dtype=np.int64)


def mirror_axis_angle_poses(poses: np.ndarray) -> np.ndarray:
    """
    镜像 AMASS/SMPL-H axis-angle pose。

    这里复用 StableMotion 的核心数学约定：先按左右关节 permutation 重排，再把每个
    axis-angle 的 y/z 分量取负。这个操作对应 AMASS 右手 Z-up 中绕 yz 平面的左右镜像。
    """

    joint_count = poses.shape[-1] // 3
    pose_permutation = build_pose_flip_permutation(joint_count)
    mirrored = poses[..., pose_permutation].copy()
    mirrored[..., 1::3] *= -1.0
    mirrored[..., 2::3] *= -1.0
    return mirrored


def mirror_translations(trans: np.ndarray) -> np.ndarray:
    mirrored = trans.copy()
    mirrored[..., 0] *= -1.0
    return mirrored


def normalize_gender(value: Any) -> str:
    """AMASS gender 可能是 numpy 标量、bytes 或普通字符串，这里统一成 smplx 可识别的值。"""

    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    gender = str(value).lower()
    if gender in {"male", "female", "neutral"}:
        return gender
    return "neutral"


def resample_motion_to_target_fps(
    poses: np.ndarray,
    trans: np.ndarray,
    source_fps: float,
    target_fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    把 AMASS 原始时间轴重采样到统一 fps。

    poses 是 AMASS/SMPL-H axis-angle 局部旋转，常见为 `[T,156]`。旋转不能直接线性插值，
    所以逐关节使用 Slerp；trans 是根节点世界平移，可以在线性速度假设下插值。
    """

    if poses.shape[1] % 3 != 0:
        raise ValueError(f"poses 最后一维必须能被 3 整除，实际为 {poses.shape}")
    frame_count = poses.shape[0]
    joint_count = poses.shape[1] // 3
    duration = (frame_count - 1) / source_fps
    source_times = np.arange(frame_count, dtype=np.float64) / source_fps
    target_times = np.arange(0.0, duration + 1e-8, 1.0 / target_fps, dtype=np.float64)
    target_times[-1] = min(target_times[-1], source_times[-1])

    if len(target_times) == frame_count and np.allclose(target_times, source_times):
        return poses.copy(), trans.copy()

    target_trans = np.stack(
        [np.interp(target_times, source_times, trans[:, axis]) for axis in range(3)],
        axis=-1,
    )

    source_rotvec = poses.reshape(frame_count, joint_count, 3)
    target_rotvec = np.empty((len(target_times), joint_count, 3), dtype=np.float64)
    for joint_index in range(joint_count):
        # Slerp 保持旋转在 SO(3) 流形上，比逐维线性插 axis-angle 更不容易产生抖动。
        rotations = Rotation.from_rotvec(source_rotvec[:, joint_index])
        slerp = Slerp(source_times, rotations)
        target_rotvec[:, joint_index] = slerp(target_times).as_rotvec()

    return target_rotvec.reshape(len(target_times), -1), target_trans


# endregion


# region SMPL 前向与几何工具


class SmplModelCache:
    """按 gender 缓存 smplx 模型，避免每个动作重复加载大模型文件。"""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self._models: dict[str, Any] = {}

    def get(self, gender: str) -> Any:
        if gender in self._models:
            return self._models[gender]

        try:
            model = create_smpl_model(model_dir=self.model_dir, gender=gender)
        except Exception:
            if gender == "neutral":
                raise
            model = create_smpl_model(model_dir=self.model_dir, gender="neutral")
            gender = "neutral"

        model.eval()
        self._models[gender] = model
        return model


def create_smpl_model(model_dir: Path, gender: str) -> Any:
    try:
        import smplx
    except ImportError as exc:
        raise ImportError("缺少 smplx 依赖，请先安装 requirements.txt。") from exc

    if has_model_files(model_dir=model_dir, model_prefix="SMPL"):
        model = smplx.SMPL(
            model_path=str(model_dir),
            gender=gender,
            batch_size=1,
            num_betas=10,
        )
        model.diffusionposer_model_type = "smpl"
        return model

    if has_model_files(model_dir=model_dir, model_prefix="SMPLH"):
        model = smplx.SMPLH(
            model_path=str(model_dir),
            gender=gender,
            batch_size=1,
            num_betas=16,
            use_pca=False,
            flat_hand_mean=True,
            ext="npz",
        )
        model.diffusionposer_model_type = "smplh"
        return model

    raise FileNotFoundError(
        f"在 {model_dir} 中没有找到 SMPL_*.pkl/npz 或 SMPLH_*.npz 模型文件。"
    )


def has_model_files(model_dir: Path, model_prefix: str) -> bool:
    return any(
        (model_dir / f"{model_prefix}_{gender}.{ext}").exists()
        for gender in ("MALE", "FEMALE", "NEUTRAL")
        for ext in ("pkl", "npz")
    )


def run_smpl_forward(source: MotionSource, model_cache: SmplModelCache, batch_size: int) -> SmplMotion:
    """
    使用 SMPL 得到 24 joint 和 vertices 的世界空间轨迹。

    AMASS 的前 66 维是 root + 21 个身体关节。若使用标准 SMPL，需要补两个 hand
    leaf 旋转；若使用 StableMotion 迁移来的 SMPL-H，则左右手由独立 hand pose 控制。
    """

    try:
        import torch
    except ImportError as exc:
        raise ImportError("缺少 torch 依赖，请先安装 requirements.txt。") from exc

    model = model_cache.get(source.gender)
    model_type = getattr(model, "diffusionposer_model_type", "smpl")
    device = next(model.parameters()).device
    frame_count = source.poses.shape[0]
    num_betas = get_model_num_betas(model)
    betas = fit_betas(source.betas, num_betas)

    global_orient = source.poses[:, :3]
    body_pose = build_body_pose_for_model(source.poses, model_type=model_type)
    left_hand_pose, right_hand_pose = build_hand_poses_for_model(source.poses)

    joints_list: list[np.ndarray] = []
    vertices_list: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, frame_count, batch_size):
            end = min(start + batch_size, frame_count)
            batch_len = end - start
            parameters = {
                "global_orient": torch.as_tensor(global_orient[start:end], dtype=torch.float32, device=device),
                "body_pose": torch.as_tensor(body_pose[start:end], dtype=torch.float32, device=device),
                "betas": torch.as_tensor(np.repeat(betas[None], batch_len, axis=0), dtype=torch.float32, device=device),
                "transl": torch.as_tensor(source.trans[start:end], dtype=torch.float32, device=device),
                "return_verts": True,
            }
            if model_type == "smplh":
                parameters.update(
                    {
                        "left_hand_pose": torch.as_tensor(
                            left_hand_pose[start:end], dtype=torch.float32, device=device
                        ),
                        "right_hand_pose": torch.as_tensor(
                            right_hand_pose[start:end], dtype=torch.float32, device=device
                        ),
                    }
                )
            output = model(**parameters)
            joints_list.append(extract_smpl_style_joints(output.joints, model_type=model_type))
            vertices_list.append(output.vertices.detach().cpu().numpy())

        # 用同一个 body shape 的零姿态 mesh 来选 heel/toe 顶点簇。这样顶点索引只取决于
        # SMPL 拓扑和体型，不会被第一帧动作姿态影响，更贴近“rest mesh 自动选点”的约定。
        rest_parameters = {
            "global_orient": torch.zeros((1, 3), dtype=torch.float32, device=device),
            "body_pose": torch.zeros((1, body_pose.shape[1]), dtype=torch.float32, device=device),
            "betas": torch.as_tensor(betas[None], dtype=torch.float32, device=device),
            "transl": torch.zeros((1, 3), dtype=torch.float32, device=device),
            "return_verts": True,
        }
        if model_type == "smplh":
            rest_parameters.update(
                left_hand_pose=torch.zeros((1, 45), dtype=torch.float32, device=device),
                right_hand_pose=torch.zeros((1, 45), dtype=torch.float32, device=device),
            )
        rest_output = model(**rest_parameters)
        rest_joints_amass = extract_smpl_style_joints(rest_output.joints, model_type=model_type)[0].astype(np.float64)
        rest_vertices_amass = rest_output.vertices[0].detach().cpu().numpy().astype(np.float64)

    joint_positions_amass = np.concatenate(joints_list, axis=0).astype(np.float64)
    vertices_amass = np.concatenate(vertices_list, axis=0).astype(np.float64)
    local_rotations_amass = build_smpl_local_rotations(source.poses)
    parents = get_smpl_parents(model)
    joint_rotations_amass = local_to_global_rotations(local_rotations_amass, parents)

    return SmplMotion(
        raw_joint_positions=joint_positions_amass,
        joint_positions=transform_points_to_unity(joint_positions_amass),
        joint_rotations=transform_rotations_to_unity(joint_rotations_amass),
        vertices=transform_points_to_unity(vertices_amass),
        rest_joints=transform_points_to_unity(rest_joints_amass),
        rest_vertices=transform_points_to_unity(rest_vertices_amass),
        parents=parents,
    )


def build_body_pose_for_model(poses: np.ndarray, model_type: str) -> np.ndarray:
    if model_type == "smplh":
        # SMPL-H 的 body_pose 只有 root 之外的 21 个身体关节；左右手由独立参数控制。
        return poses[:, 3 : SOURCE_BODY_JOINT_COUNT * 3]

    left_hand_leaf, right_hand_leaf = build_smpl_hand_leaf_rotations(poses)
    return np.concatenate(
        [poses[:, 3 : SOURCE_BODY_JOINT_COUNT * 3], left_hand_leaf, right_hand_leaf],
        axis=1,
    )


def build_hand_poses_for_model(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame_count = poses.shape[0]
    if poses.shape[1] >= SMPLH_JOINT_COUNT * 3:
        return (
            poses[:, SMPLH_LEFT_HAND_START * 3 : SMPLH_RIGHT_HAND_START * 3],
            poses[:, SMPLH_RIGHT_HAND_START * 3 : SMPLH_JOINT_COUNT * 3],
        )
    return (
        np.zeros((frame_count, 45), dtype=np.float64),
        np.zeros((frame_count, 45), dtype=np.float64),
    )


def build_smpl_hand_leaf_rotations(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame_count = poses.shape[0]
    if poses.shape[1] >= SMPLH_JOINT_COUNT * 3:
        return (
            poses[:, SMPLH_LEFT_HAND_START * 3 : SMPLH_LEFT_HAND_START * 3 + 3],
            poses[:, SMPLH_RIGHT_HAND_START * 3 : SMPLH_RIGHT_HAND_START * 3 + 3],
        )
    return (
        np.zeros((frame_count, 3), dtype=np.float64),
        np.zeros((frame_count, 3), dtype=np.float64),
    )


def extract_smpl_style_joints(joints: Any, model_type: str) -> np.ndarray:
    if model_type == "smplh":
        return joints[:, SMPLH_TO_SMPL_JOINT_INDICES].detach().cpu().numpy()
    return joints[:, :SMPL_JOINT_COUNT].detach().cpu().numpy()


def get_model_num_betas(model: Any) -> int:
    if hasattr(model, "num_betas"):
        return int(model.num_betas)
    if hasattr(model, "betas"):
        return int(model.betas.shape[-1])
    return 10


def fit_betas(source_betas: np.ndarray, num_betas: int) -> np.ndarray:
    fitted = np.zeros(num_betas, dtype=np.float64)
    used = min(num_betas, source_betas.shape[0])
    fitted[:used] = source_betas[:used]
    return fitted


def get_smpl_parents(model: Any) -> np.ndarray:
    if getattr(model, "diffusionposer_model_type", "smpl") == "smplh":
        # SMPL-H 的 kinematic tree 在手部展开成 52 个 joint；这里输出仍是项目约定的
        # 24 个 SMPL 风格 segment，因此旋转递推使用标准 24J 父子关系。
        return DEFAULT_SMPL_PARENTS.copy()
    if hasattr(model, "parents"):
        parents = np.asarray(model.parents.detach().cpu().numpy(), dtype=np.int64)
        return parents[:SMPL_JOINT_COUNT]
    return DEFAULT_SMPL_PARENTS.copy()


def build_smpl_local_rotations(poses: np.ndarray) -> np.ndarray:
    frame_count = poses.shape[0]
    local_rotvec = np.zeros((frame_count, SMPL_JOINT_COUNT, 3), dtype=np.float64)
    local_rotvec[:, :SOURCE_BODY_JOINT_COUNT] = poses[:, : SOURCE_BODY_JOINT_COUNT * 3].reshape(
        frame_count,
        SOURCE_BODY_JOINT_COUNT,
        3,
    )
    left_hand_leaf, right_hand_leaf = build_smpl_hand_leaf_rotations(poses)
    local_rotvec[:, JOINT_INDEX["left_hand"]] = left_hand_leaf
    local_rotvec[:, JOINT_INDEX["right_hand"]] = right_hand_leaf
    return Rotation.from_rotvec(local_rotvec.reshape(-1, 3)).as_matrix().reshape(frame_count, SMPL_JOINT_COUNT, 3, 3)


def local_to_global_rotations(local_rotations: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """把 SMPL 局部关节旋转沿 kinematic tree 递推为世界旋转 `[T,24,3,3]`。"""

    global_rotations = np.empty_like(local_rotations)
    for joint_index, parent_index in enumerate(parents):
        if parent_index < 0:
            global_rotations[:, joint_index] = local_rotations[:, joint_index]
        else:
            global_rotations[:, joint_index] = global_rotations[:, parent_index] @ local_rotations[:, joint_index]
    return global_rotations


def transform_points_to_unity(points: np.ndarray) -> np.ndarray:
    return points @ AMASS_TO_UNITY.T


def transform_rotations_to_unity(rotations: np.ndarray) -> np.ndarray:
    return AMASS_TO_UNITY @ rotations @ AMASS_TO_UNITY.T


def make_yaw_rotation(yaw: np.ndarray) -> np.ndarray:
    """
    构造只含 Unity Y 轴 yaw 的 local-to-world 旋转矩阵。

    yaw=0 时 forward 是 +Z；正 yaw 让 forward 朝 +X 转动。后续转 Root 局部时使用
    `root_rot.T @ world_vector`，因此这个矩阵必须表示 Root 局部坐标轴在世界中的方向。
    """

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotations = np.zeros((yaw.shape[0], 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = cos_yaw
    rotations[:, 0, 2] = sin_yaw
    rotations[:, 1, 1] = 1.0
    rotations[:, 2, 0] = -sin_yaw
    rotations[:, 2, 2] = cos_yaw
    return rotations


def extract_yaw_from_rotations(rotations: np.ndarray) -> np.ndarray:
    """从 joint 世界旋转中提取水平 forward 的 yaw，忽略 roll/pitch。"""

    forward = rotations[..., 2]
    horizontal_norm = np.linalg.norm(forward[:, [0, 2]], axis=-1)
    yaws = np.arctan2(forward[:, 0], forward[:, 2])

    # 极端姿态下 forward 可能接近竖直，yaw 会不稳定；这里沿用上一帧 yaw，
    # 让 root frame 连续，不因为一个翻滚姿态产生巨大的水平朝向跳变。
    unstable = horizontal_norm < 1e-6
    for index in range(1, len(yaws)):
        if unstable[index]:
            yaws[index] = yaws[index - 1]
    if len(yaws) and unstable[0]:
        yaws[0] = 0.0
    return yaws


def build_root_frames(joint_positions: np.ndarray, joint_rotations: np.ndarray) -> RootFrames:
    pelvis_positions = joint_positions[:, JOINT_INDEX["pelvis"]]
    pelvis_rotations = joint_rotations[:, JOINT_INDEX["pelvis"]]
    yaws = extract_yaw_from_rotations(pelvis_rotations)
    positions = pelvis_positions.copy()
    positions[:, 1] = 0.0
    rotations = make_yaw_rotation(yaws)
    return RootFrames(positions=positions, rotations=rotations, yaws=yaws)


def rotation_6d_forward_up(rotations: np.ndarray) -> np.ndarray:
    """
    读取 Unity 对齐的 6D 旋转：forward(+Z) 后接 up(+Y)。

    输入可以是 `[T,J,3,3]` 或 `[T,K,3,3]`，输出会保留前两维并展开成最后 6 维。
    """

    forward = rotations[..., :, 2]
    up = rotations[..., :, 1]
    return np.concatenate([forward, up], axis=-1)


def wrap_degrees(angle_degree: np.ndarray) -> np.ndarray:
    return (angle_degree + 180.0) % 360.0 - 180.0


# endregion


# region tracker 与 contact


def infer_foot_vertex_groups(
    rest_vertices: np.ndarray,
    rest_joints: np.ndarray,
    cluster_size: int = 20,
) -> dict[str, np.ndarray]:
    """
    从 rest mesh 自动选 heel/toe 顶点簇。

    不硬编码 vertex id 的好处是兼容不同 SMPL 文件版本；做法是先按左右脚 joint 附近筛选
    顶点，再按局部前后方向分成 toe/heel。每个 contact 点用一个小簇均值，抗 mesh 噪声更稳。
    """

    groups: dict[str, np.ndarray] = {}
    for side, foot_joint_name in (("left", "left_foot"), ("right", "right_foot")):
        foot_center = rest_joints[JOINT_INDEX[foot_joint_name]]
        ankle_center = rest_joints[JOINT_INDEX[f"{side}_ankle"]]
        side_mask = rest_vertices[:, 0] <= foot_center[0] if side == "left" else rest_vertices[:, 0] >= foot_center[0]
        low_mask = rest_vertices[:, 1] <= foot_center[1] + 0.15
        near_mask = np.linalg.norm(rest_vertices - foot_center, axis=-1) <= 0.35
        candidates = np.where(side_mask & low_mask & near_mask)[0]
        if candidates.size < cluster_size * 2:
            candidates = np.argsort(np.linalg.norm(rest_vertices - foot_center, axis=-1))[: cluster_size * 4]

        # 用 ankle->foot 的水平向量作为脚尖方向；如果模型处在 T-pose 且该向量退化，
        # 则回退到 Unity +Z，使 toe/heel 选择仍然可运行。
        toe_direction = foot_center - ankle_center
        toe_direction[1] = 0.0
        direction_norm = np.linalg.norm(toe_direction)
        if direction_norm < 1e-6:
            toe_direction = np.array([0.0, 0.0, 1.0])
        else:
            toe_direction = toe_direction / direction_norm

        scores = (rest_vertices[candidates] - foot_center) @ toe_direction
        toe_indices = candidates[np.argsort(scores)[-cluster_size:]]
        heel_indices = candidates[np.argsort(scores)[:cluster_size]]
        groups[f"{side}_toe"] = toe_indices
        groups[f"{side}_heel"] = heel_indices
    return groups


def compute_contact(
    vertices: np.ndarray,
    vertex_groups: dict[str, np.ndarray],
    target_fps: float,
    height_threshold: float,
    speed_threshold: float,
) -> np.ndarray:
    """根据 foot vertex 簇的高度和水平速度得到 `[T,4]` 当前帧接触标签。"""

    foot_points = np.stack([vertices[:, vertex_groups[name]].mean(axis=1) for name in CONTACT_NAMES], axis=1)
    ground_height = float(np.percentile(foot_points[..., 1], 1.0))
    heights = foot_points[..., 1] - ground_height
    velocities = np.zeros_like(foot_points)
    velocities[1:] = (foot_points[1:] - foot_points[:-1]) * target_fps
    horizontal_speed = np.linalg.norm(velocities[..., [0, 2]], axis=-1)
    return ((heights <= height_threshold) & (horizontal_speed <= speed_threshold)).astype(np.float32)


def gather_tracker_rotations(joint_rotations: np.ndarray) -> np.ndarray:
    return joint_rotations[:, TRACKER_JOINT_INDICES]


def gather_tracker_positions(joint_positions: np.ndarray) -> np.ndarray:
    return joint_positions[:, TRACKER_JOINT_INDICES]


# endregion


# region X277 组装与保存


def build_x277_features(
    smpl_motion: SmplMotion,
    target_fps: float,
    height_threshold: float,
    speed_threshold: float,
) -> np.ndarray:
    """
    组装 current277_v1 特征，返回 `[T-1,277]`。

    第 0 行对应原目标序列的 t=1。除速度/root delta 依赖上一帧外，
    body、tracker、root delta、contact 都严格使用当前帧 t。
    """

    joint_positions = smpl_motion.joint_positions
    joint_rotations = smpl_motion.joint_rotations
    root_frames = build_root_frames(joint_positions, joint_rotations)
    tracker_positions = gather_tracker_positions(joint_positions)
    tracker_rotations = gather_tracker_rotations(joint_rotations)
    vertex_groups = infer_foot_vertex_groups(
        rest_vertices=smpl_motion.rest_vertices,
        rest_joints=smpl_motion.rest_joints,
    )
    contacts = compute_contact(
        vertices=smpl_motion.vertices,
        vertex_groups=vertex_groups,
        target_fps=target_fps,
        height_threshold=height_threshold,
        speed_threshold=speed_threshold,
    )

    rows: list[np.ndarray] = []
    for current_index in range(1, joint_positions.shape[0]):
        prev_index = current_index - 1
        row = np.empty(FEATURE_DIM, dtype=np.float64)

        current_root_rot_inv = root_frames.rotations[current_index].T

        body_rot_root_now = current_root_rot_inv[None] @ joint_rotations[current_index]
        row[0:144] = rotation_6d_forward_up(body_rot_root_now[None]).reshape(-1)

        joint_vel_world_now = (joint_positions[current_index] - joint_positions[prev_index]) * target_fps
        joint_vel_root_now = joint_vel_world_now @ root_frames.rotations[current_index]
        row[144:216] = joint_vel_root_now.reshape(-1)

        tracker_pos_root_now = (tracker_positions[current_index] - root_frames.positions[current_index]) @ root_frames.rotations[
            current_index
        ]
        row[216:234] = tracker_pos_root_now.reshape(-1)

        tracker_rot_root_now = current_root_rot_inv[None] @ tracker_rotations[current_index]
        row[234:270] = rotation_6d_forward_up(tracker_rot_root_now[None]).reshape(-1)

        waist_delta_root = (joint_positions[current_index, JOINT_INDEX["pelvis"]] - joint_positions[prev_index, JOINT_INDEX["pelvis"]]) @ root_frames.rotations[
            prev_index
        ]
        row[270:272] = waist_delta_root[[0, 2]]

        yaw_delta_degree = math.degrees(root_frames.yaws[current_index] - root_frames.yaws[prev_index])
        row[272] = wrap_degrees(np.asarray(yaw_delta_degree))
        row[273:277] = contacts[current_index]
        rows.append(row)

    x = np.stack(rows, axis=0).astype(np.float32)
    validate_x277(x)
    return x


def validate_x277(x: np.ndarray) -> None:
    if x.ndim != 2 or x.shape[1] != FEATURE_DIM:
        raise ValueError(f"输出应为 [T,{FEATURE_DIM}]，实际为 {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("输出包含 NaN 或 Inf。")
    contact = x[:, 273:277]
    if not np.isin(contact, [0.0, 1.0]).all():
        raise ValueError("contact_cur 必须只包含 0/1。")
    for start, end in ((0, 144), (234, 270)):
        rotation_6d = x[:, start:end].reshape(x.shape[0], -1, 6)
        forward_norm = np.linalg.norm(rotation_6d[..., :3], axis=-1)
        up_norm = np.linalg.norm(rotation_6d[..., 3:], axis=-1)
        if np.any(forward_norm < 0.5) or np.any(up_norm < 0.5):
            raise ValueError("rotation 6D 中存在异常短的 forward/up 向量。")


def output_path_for(source: MotionSource, output_dir: Path) -> Path:
    return output_dir / source.relative_path.with_suffix(".npz")


def save_converted_motion(
    output_path: Path,
    x: np.ndarray,
    source: MotionSource,
    target_fps: float,
    height_threshold: float,
    speed_threshold: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_path": str(source.path),
        "source_relative_path": str(source.relative_path),
        "original_source_relative_path": str(source.original_relative_path or source.relative_path),
        "is_mirrored": source.is_mirrored,
        "stablemotion_split_key": str(source.relative_path.with_suffix(".npy")).replace("\\", "/"),
        "source_fps": source.source_fps,
        "target_fps": target_fps,
        "feature_dim": FEATURE_DIM,
        "schema_name": SCHEMA_NAME,
        "feature_slices": FEATURE_SLICES,
        "tracker_order": TRACKER_NAMES,
        "contact_order": CONTACT_NAMES,
        "contact_time": "current_frame_t",
        "coordinate_system": "unity_y_up_left_handed_root_local",
        "rotation_6d": "forward_xyz_then_up_xyz",
        "waist_yaw_delta_unit": "degree",
        "height_threshold": height_threshold,
        "speed_threshold": speed_threshold,
    }
    np.savez_compressed(
        output_path,
        x=x,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )


def write_manifest_record(manifest_path: Path, record: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


# endregion


# region 可视化


def load_converted_x277(output_path: Path) -> np.ndarray:
    with np.load(output_path, allow_pickle=True) as data:
        if "x" not in data.files:
            raise ValueError(f"已存在的输出文件缺少 x 字段: {output_path}")
        x = np.asarray(data["x"], dtype=np.float32)
    validate_x277(x)
    return x


def should_visualize(args: argparse.Namespace, attempted_count: int) -> bool:
    if args.visualize_num == 0:
        return False
    if args.visualize_num < 0:
        return True
    return attempted_count < args.visualize_num


def visualization_path_for(source: MotionSource, visualize_dir: Path) -> Path:
    relative_stem = source.relative_path.with_suffix("")
    return visualize_dir / relative_stem.parent / f"{relative_stem.name}_x277_overlay.mp4"


def unity_to_z_up_display(points: np.ndarray) -> np.ndarray:
    """把 Unity Y-up 轨迹转回 Matplotlib 使用的 Z-up 显示坐标。"""

    return points @ AMASS_TO_UNITY.T


def decode_x277_joint_positions(
    x: np.ndarray,
    seed_joint_positions: np.ndarray,
    seed_root_position: np.ndarray,
    seed_root_yaw: float,
    target_fps: float,
) -> np.ndarray:
    """
    从 X277 的 root delta 和 root-local 线速度近似反推 24 joint 世界位置。

    这里的目的不是做正式反解，而是把特征中的速度/Root 坐标语义可视化出来：
    如果蓝色轨迹和红色轨迹明显分离，通常说明坐标系、速度帧索引或 yaw_delta 单位有问题。
    """

    joint_positions = seed_joint_positions.astype(np.float64).copy()
    root_position = seed_root_position.astype(np.float64).copy()
    root_yaw = float(seed_root_yaw)
    decoded_frames: list[np.ndarray] = []

    for row in x:
        prev_root_rotation = make_yaw_rotation(np.asarray([root_yaw], dtype=np.float64))[0]
        delta_xz = row[270:272].astype(np.float64)
        delta_world = np.asarray([delta_xz[0], 0.0, delta_xz[1]], dtype=np.float64) @ prev_root_rotation.T
        root_position = root_position + delta_world
        root_yaw = root_yaw + math.radians(float(row[272]))
        current_root_rotation = make_yaw_rotation(np.asarray([root_yaw], dtype=np.float64))[0]

        joint_vel_root_now = row[144:216].reshape(SMPL_JOINT_COUNT, 3).astype(np.float64)
        joint_vel_world_now = joint_vel_root_now @ current_root_rotation.T
        joint_positions = joint_positions + joint_vel_world_now / target_fps
        decoded_frames.append(joint_positions.copy())

    return np.stack(decoded_frames, axis=0)


def build_visualization_tracks(
    smpl_motion: SmplMotion,
    x: np.ndarray,
    target_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root_frames = build_root_frames(smpl_motion.joint_positions, smpl_motion.joint_rotations)
    decoded_unity = decode_x277_joint_positions(
        x=x,
        seed_joint_positions=smpl_motion.joint_positions[0],
        seed_root_position=root_frames.positions[0],
        seed_root_yaw=float(root_frames.yaws[0]),
        target_fps=target_fps,
    )

    frame_count = min(decoded_unity.shape[0], smpl_motion.joint_positions.shape[0] - 1)
    raw_display = smpl_motion.raw_joint_positions[1 : 1 + frame_count]
    unity_display = unity_to_z_up_display(smpl_motion.joint_positions[1 : 1 + frame_count])
    decoded_display = unity_to_z_up_display(decoded_unity[:frame_count])
    return raw_display, unity_display, decoded_display


def render_x277_overlay_visualization(
    raw_display: np.ndarray,
    unity_display: np.ndarray,
    decoded_display: np.ndarray,
    output_path: Path,
    fps: float,
    title: str,
) -> Path:
    """
    渲染三轨叠加动画：
    绿=原始 AMASS Z-up，红=Unity 坐标转换后再转回 Z-up 显示，蓝=X277 近似解码。
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise ImportError("缺少 matplotlib，可视化需要先安装 requirements.txt。") from exc

    frames = min(raw_display.shape[0], unity_display.shape[0], decoded_display.shape[0])
    if frames <= 0:
        raise ValueError("没有可渲染的可视化帧。")

    tracks = [
        raw_display[:frames].copy(),
        unity_display[:frames].copy(),
        decoded_display[:frames].copy(),
    ]

    # 统一落地，保证三条轨迹在同一个地面网格上比较，而不是被绝对高度差干扰。
    min_z = min(float(track[:, :, 2].min()) for track in tracks)
    for track in tracks:
        track[:, :, 2] -= min_z

    all_points = np.concatenate([track.reshape(-1, 3) for track in tracks], axis=0)
    center_xy = all_points[:, :2].mean(axis=0)
    xy_span = np.ptp(all_points[:, :2], axis=0)
    radius = max(float(xy_span.max()) * 0.55, 1.5)
    z_max = max(float(all_points[:, 2].max()), 1.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 8), dpi=120)
    captured_frames: list[np.ndarray] = []

    for frame_index in tqdm(range(frames), desc="Rendering X277 overlay", leave=False):
        fig.clf()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        configure_visualization_axis(ax=ax, center_xy=center_xy, radius=radius, z_max=z_max)
        draw_ground_grid(ax=ax, center_xy=center_xy, radius=radius)
        draw_skeleton(ax=ax, joints=tracks[0][frame_index], color="green", alpha=0.48, linewidth=3.5)
        draw_skeleton(ax=ax, joints=tracks[1][frame_index], color="red", alpha=0.62, linewidth=3.0)
        draw_skeleton(ax=ax, joints=tracks[2][frame_index], color="blue", alpha=0.70, linewidth=2.5)

        second = frame_index / fps
        fig.text(
            0.02,
            0.97,
            f"{title}\nFrame {frame_index + 1}/{frames} | Time {second:.2f}s | display=Z-up",
            va="top",
            ha="left",
            fontsize=10,
            color="black",
        )
        legend_handles = [
            Line2D([0], [0], color="green", lw=4, alpha=0.7, label="Raw AMASS Z-up"),
            Line2D([0], [0], color="red", lw=4, alpha=0.7, label="Unity Y-up -> display Z-up"),
            Line2D([0], [0], color="blue", lw=4, alpha=0.8, label="Decoded X277"),
        ]
        ax.legend(handles=legend_handles, loc="upper right")
        draw_timeline(fig=fig, frame_index=frame_index, frame_count=frames)

        fig.canvas.draw()
        frame_rgba = np.asarray(fig.canvas.buffer_rgba())
        captured_frames.append(frame_rgba[..., :3].copy())

    plt.close(fig)
    return export_visualization_frames(frames=captured_frames, output_path=output_path, fps=fps)


def configure_visualization_axis(ax: Any, center_xy: np.ndarray, radius: float, z_max: float) -> None:
    ax.view_init(elev=22.0, azim=-60.0)
    ax.set_xlim(center_xy[0] - radius, center_xy[0] + radius)
    ax.set_ylim(center_xy[1] - radius, center_xy[1] + radius)
    ax.set_zlim(0.0, z_max + 0.2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z / Up")
    ax.set_title("AMASS -> Unity -> X277 Overlay", pad=12)


def draw_ground_grid(ax: Any, center_xy: np.ndarray, radius: float, divisions: int = 10) -> None:
    min_x, max_x = center_xy[0] - radius, center_xy[0] + radius
    min_y, max_y = center_xy[1] - radius, center_xy[1] + radius
    xs = np.linspace(min_x, max_x, divisions + 1)
    ys = np.linspace(min_y, max_y, divisions + 1)
    for x_value in xs:
        ax.plot([x_value, x_value], [min_y, max_y], [0.0, 0.0], color="lightgray", linewidth=0.6, alpha=0.55)
    for y_value in ys:
        ax.plot([min_x, max_x], [y_value, y_value], [0.0, 0.0], color="lightgray", linewidth=0.6, alpha=0.55)


def draw_skeleton(ax: Any, joints: np.ndarray, color: str, alpha: float, linewidth: float) -> None:
    for chain in KINEMATIC_CHAINS:
        chain_points = joints[np.asarray(chain, dtype=np.int64)]
        ax.plot(
            chain_points[:, 0],
            chain_points[:, 1],
            chain_points[:, 2],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color=color, alpha=min(alpha + 0.15, 1.0), s=8)


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

    # 路径 1：优先复用 StableMotion-FSQ 的 MoviePy/libx264 方案。MoviePy 通常会通过
    # imageio-ffmpeg 找到可用 ffmpeg，比 Matplotlib 自己查找系统 ffmpeg 更稳。
    try:
        import moviepy.editor as mp

        clip = mp.ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(
            str(output_path),
            codec="libx264",
            audio=False,
            verbose=False,
            logger=None,
        )
        clip.close()
        return output_path
    except Exception as exc:
        errors.append(f"moviepy: {exc!r}")

    # 路径 2：直接使用 imageio 的 ffmpeg writer。它和 MoviePy 一样可以借助
    # imageio-ffmpeg 的 bundled ffmpeg，作为 Matplotlib 之前的轻量兜底。
    try:
        import imageio.v2 as imageio

        with imageio.get_writer(
            str(output_path),
            fps=fps,
            codec="libx264",
            macro_block_size=16,
        ) as writer:
            for frame in frames:
                writer.append_data(frame)
        return output_path
    except Exception as exc:
        errors.append(f"imageio_ffmpeg: {exc!r}")

    # 路径 3：Matplotlib FFMpegWriter，适合系统 PATH 中已经配置 ffmpeg 的机器。
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
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


def visualize_converted_motion(
    source: MotionSource,
    smpl_motion: SmplMotion,
    x: np.ndarray,
    args: argparse.Namespace,
) -> Path:
    raw_display, unity_display, decoded_display = build_visualization_tracks(
        smpl_motion=smpl_motion,
        x=x,
        target_fps=args.target_fps,
    )
    output_path = visualization_path_for(source=source, visualize_dir=args.visualize_dir)
    return render_x277_overlay_visualization(
        raw_display=raw_display,
        unity_display=unity_display,
        decoded_display=decoded_display,
        output_path=output_path,
        fps=args.visualize_fps,
        title=str(source.relative_path),
    )


# endregion


# region 主流程


def convert_one_motion(
    path: Path,
    args: argparse.Namespace,
    model_cache: SmplModelCache,
    enable_visualization: bool,
    mirror_variant: bool = False,
) -> dict[str, Any]:
    source = load_motion_source(path=path, amass_dir=args.amass_dir, target_fps=args.target_fps)
    if mirror_variant:
        source = mirror_motion_source(source)
    output_path = output_path_for(source, args.output_dir)
    smpl_motion: SmplMotion | None = None
    x: np.ndarray | None = None

    if output_path.exists() and args.skip_existing:
        status = "skipped_existing"
        if enable_visualization:
            x = load_converted_x277(output_path)
    elif output_path.exists() and not args.overwrite:
        raise FileExistsError(f"输出文件已存在: {output_path}。请使用 --overwrite 或 --skip_existing。")
    else:
        status = "converted"
        smpl_motion = run_smpl_forward(source=source, model_cache=model_cache, batch_size=args.batch_size)
        x = build_x277_features(
            smpl_motion=smpl_motion,
            target_fps=args.target_fps,
            height_threshold=args.height_threshold,
            speed_threshold=args.speed_threshold,
        )
        save_converted_motion(
            output_path=output_path,
            x=x,
            source=source,
            target_fps=args.target_fps,
            height_threshold=args.height_threshold,
            speed_threshold=args.speed_threshold,
        )

    record = {
        "status": status,
        "source_path": str(path),
        "source_relative_path": str(source.relative_path),
        "original_source_relative_path": str(source.original_relative_path or source.relative_path),
        "is_mirrored": source.is_mirrored,
        "stablemotion_split_key": str(source.relative_path.with_suffix(".npy")).replace("\\", "/"),
        "output_path": str(output_path),
        "visualization_attempted": bool(enable_visualization),
        "visualized": False,
    }
    if x is not None:
        record["frames"] = int(x.shape[0])
        record["feature_dim"] = int(x.shape[1])

    if enable_visualization:
        try:
            if smpl_motion is None:
                smpl_motion = run_smpl_forward(source=source, model_cache=model_cache, batch_size=args.batch_size)
            if x is None:
                x = load_converted_x277(output_path)
            visualization_path = visualize_converted_motion(
                source=source,
                smpl_motion=smpl_motion,
                x=x,
                args=args,
            )
            record["visualized"] = True
            record["visualization_path"] = str(visualization_path)
        except Exception as exc:
            record["visualization_error"] = repr(exc)

    return record


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    if args.overwrite and manifest_path.exists():
        manifest_path.unlink()

    motion_files = iter_amass_motion_files(args.amass_dir)
    if args.limit:
        motion_files = motion_files[: args.limit]

    model_cache = SmplModelCache(model_dir=args.smpl_model_dir)
    converted = 0
    skipped = 0
    failed = 0
    visualization_attempted = 0
    visualized = 0

    for path in tqdm(motion_files, desc="Converting AMASS to X277"):
        mirror_variants = (False, True) if args.mirror else (False,)
        for mirror_variant in mirror_variants:
            enable_visualization = should_visualize(args=args, attempted_count=visualization_attempted)
            try:
                record = convert_one_motion(
                    path=path,
                    args=args,
                    model_cache=model_cache,
                    enable_visualization=enable_visualization,
                    mirror_variant=mirror_variant,
                )
            except Exception as exc:
                failed += 1
                source_relative_path = path.relative_to(args.amass_dir)
                if mirror_variant:
                    source_relative_path = Path(MIRROR_DIR_NAME) / source_relative_path
                record = {
                    "status": "failed",
                    "source_path": str(path),
                    "source_relative_path": str(source_relative_path),
                    "is_mirrored": mirror_variant,
                    "stablemotion_split_key": str(source_relative_path.with_suffix(".npy")).replace("\\", "/"),
                    "error": repr(exc),
                    "visualization_attempted": False,
                    "visualized": False,
                }
            else:
                if record["status"] == "converted":
                    converted += 1
                else:
                    skipped += 1
                if record.get("visualization_attempted"):
                    visualization_attempted += 1
                if record.get("visualized"):
                    visualized += 1
            write_manifest_record(manifest_path, record)

    print(
        f"完成 AMASS -> X277 转换：converted={converted}, skipped={skipped}, failed={failed}, "
        f"visualized={visualized}, manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()


# endregion
