from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from data_loaders.realtime_pose_kinematics import JOINT_INDEX, SMPL_JOINT_NAMES, SMPL_PARENTS


SOURCE_BODY_JOINT_COUNT = 22
SMPL_JOINT_COUNT = len(SMPL_JOINT_NAMES)
SMPLH_JOINT_COUNT = 52
SMPLH_LEFT_HAND_START = 22
SMPLH_RIGHT_HAND_START = 37
SMPLH_TO_SMPL_JOINT_INDICES = np.array(list(range(23)) + [37], dtype=np.int64)
MIRROR_DIR_NAME = "M"

SMPL_BODY_JOINTS_FLIP_PERM = np.array(
    [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20],
    dtype=np.int64,
)
AMASS_TO_UNITY = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


class ShortMotionError(ValueError):
    """动作帧数不足以完成稳定重采样或 source 转换。"""


@dataclass(frozen=True)
class MotionSource:
    """AMASS motion 在转换流程中的最小输入表示。"""

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
    """SMPL 前向后的世界空间数据，坐标已经转换到项目的 Unity 约定。"""

    raw_joint_positions: np.ndarray
    joint_positions: np.ndarray
    joint_rotations: np.ndarray
    rest_joints: np.ndarray
    parents: np.ndarray


def validate_args(args) -> None:
    if not args.amass_dir.exists():
        raise FileNotFoundError(f"AMASS 数据目录不存在：{args.amass_dir}")
    if not args.smpl_model_dir.exists():
        raise FileNotFoundError(f"SMPL 模型目录不存在：{args.smpl_model_dir}")
    if args.target_fps <= 0:
        raise ValueError("--target_fps 必须为正数")
    if args.batch_size <= 0:
        raise ValueError("--batch_size 必须为正整数")
    if args.overwrite and args.skip_existing and not getattr(args, "rebuild_manifest", False):
        raise ValueError("--overwrite 和 --skip_existing 不能同时启用；如只想重写 manifest，请使用 --rebuild_manifest")


def iter_amass_motion_files(amass_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in amass_dir.rglob("*.npz")
        if path.name != "shape.npz" and not is_mirror_relative_path(path.relative_to(amass_dir))
    )


def is_mirror_relative_path(relative_path: Path) -> bool:
    return bool(relative_path.parts) and relative_path.parts[0] == MIRROR_DIR_NAME


def load_motion_source(path: Path, amass_dir: Path, target_fps: float) -> MotionSource:
    with np.load(path, allow_pickle=True) as data:
        missing_keys = [key for key in ("poses", "trans", "mocap_framerate") if key not in data.files]
        if missing_keys:
            raise ValueError(f"缺少必要字段：{missing_keys}")
        poses = np.asarray(data["poses"], dtype=np.float64)
        trans = np.asarray(data["trans"], dtype=np.float64)
        source_fps = float(np.asarray(data["mocap_framerate"]).item())
        if poses.ndim != 2 or poses.shape[1] < SOURCE_BODY_JOINT_COUNT * 3:
            raise ValueError(f"poses 至少应为 [T,66]，实际为 {poses.shape}")
        if trans.shape != (poses.shape[0], 3):
            raise ValueError(f"trans 应为 [T,3] 且与 poses 同帧数，实际为 {trans.shape}")
        if poses.shape[0] < 3:
            raise ShortMotionError("原始帧数少于 3，跳过该动作。")
        betas = np.asarray(data["betas"], dtype=np.float64) if "betas" in data.files else np.zeros(10)
        gender = normalize_gender(data["gender"] if "gender" in data.files else "neutral")

    poses_resampled, trans_resampled = resample_motion_to_target_fps(
        poses=poses,
        trans=trans,
        source_fps=source_fps,
        target_fps=target_fps,
    )
    if poses_resampled.shape[0] < 3:
        raise ShortMotionError(f"重采样到 {target_fps:g}Hz 后少于 3 帧，跳过该动作。")
    relative_path = path.relative_to(amass_dir)
    return MotionSource(
        path=path,
        relative_path=relative_path,
        poses=poses_resampled,
        trans=trans_resampled,
        betas=betas,
        gender=gender,
        source_fps=source_fps,
        original_relative_path=relative_path,
    )


def mirror_motion_source(source: MotionSource) -> MotionSource:
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
    joint_count = poses.shape[-1] // 3
    mirrored = poses[..., build_pose_flip_permutation(joint_count)].copy()
    mirrored[..., 1::3] *= -1.0
    mirrored[..., 2::3] *= -1.0
    return mirrored


def mirror_translations(trans: np.ndarray) -> np.ndarray:
    mirrored = trans.copy()
    mirrored[..., 0] *= -1.0
    return mirrored


def normalize_gender(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    gender = str(value).lower()
    return gender if gender in {"male", "female", "neutral"} else "neutral"


def resample_motion_to_target_fps(
    poses: np.ndarray,
    trans: np.ndarray,
    source_fps: float,
    target_fps: float,
) -> tuple[np.ndarray, np.ndarray]:
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
        rotations = Rotation.from_rotvec(source_rotvec[:, joint_index])
        slerp = Slerp(source_times, rotations)
        target_rotvec[:, joint_index] = slerp(target_times).as_rotvec()
    return target_rotvec.reshape(len(target_times), -1), target_trans


class SmplModelCache:
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
        model = smplx.SMPL(model_path=str(model_dir), gender=gender, batch_size=1, num_betas=10)
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
    raise FileNotFoundError(f"在 {model_dir} 中没有找到 SMPL/SMPLH 模型文件。")


def has_model_files(model_dir: Path, model_prefix: str) -> bool:
    return any(
        (model_dir / f"{model_prefix}_{gender}.{ext}").exists()
        for gender in ("MALE", "FEMALE", "NEUTRAL")
        for ext in ("pkl", "npz")
    )


def run_smpl_forward(source: MotionSource, model_cache: SmplModelCache, batch_size: int) -> SmplMotion:
    import torch

    model = model_cache.get(source.gender)
    model_type = getattr(model, "diffusionposer_model_type", "smpl")
    device = next(model.parameters()).device
    frame_count = source.poses.shape[0]
    betas = fit_betas(source.betas, get_model_num_betas(model))

    global_orient = source.poses[:, :3]
    body_pose = build_body_pose_for_model(source.poses, model_type=model_type)
    left_hand_pose, right_hand_pose = build_hand_poses_for_model(source.poses)
    joints_list: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, frame_count, batch_size):
            end = min(start + batch_size, frame_count)
            batch_len = end - start
            parameters = {
                "global_orient": torch.as_tensor(global_orient[start:end], dtype=torch.float32, device=device),
                "body_pose": torch.as_tensor(body_pose[start:end], dtype=torch.float32, device=device),
                "betas": torch.as_tensor(np.repeat(betas[None], batch_len, axis=0), dtype=torch.float32, device=device),
                "transl": torch.as_tensor(source.trans[start:end], dtype=torch.float32, device=device),
                "return_verts": False,
            }
            if model_type == "smplh":
                parameters.update(
                    left_hand_pose=torch.as_tensor(left_hand_pose[start:end], dtype=torch.float32, device=device),
                    right_hand_pose=torch.as_tensor(right_hand_pose[start:end], dtype=torch.float32, device=device),
                )
            output = model(**parameters)
            joints_list.append(extract_smpl_style_joints(output.joints, model_type=model_type))

        rest_parameters = {
            "global_orient": torch.zeros((1, 3), dtype=torch.float32, device=device),
            "body_pose": torch.zeros((1, body_pose.shape[1]), dtype=torch.float32, device=device),
            "betas": torch.as_tensor(betas[None], dtype=torch.float32, device=device),
            "transl": torch.zeros((1, 3), dtype=torch.float32, device=device),
            "return_verts": False,
        }
        if model_type == "smplh":
            rest_parameters.update(
                left_hand_pose=torch.zeros((1, 45), dtype=torch.float32, device=device),
                right_hand_pose=torch.zeros((1, 45), dtype=torch.float32, device=device),
            )
        rest_output = model(**rest_parameters)
        rest_joints = extract_smpl_style_joints(rest_output.joints, model_type=model_type)[0].astype(np.float64)

    joint_positions_amass = np.concatenate(joints_list, axis=0).astype(np.float64)
    local_rotations = build_smpl_local_rotations(source.poses)
    parents = get_smpl_parents(model)
    joint_rotations_amass = local_to_global_rotations(local_rotations, parents)
    return SmplMotion(
        raw_joint_positions=joint_positions_amass,
        joint_positions=transform_points_to_unity(joint_positions_amass),
        joint_rotations=transform_rotations_to_unity(joint_rotations_amass),
        rest_joints=transform_points_to_unity(rest_joints),
        parents=parents,
    )


def build_body_pose_for_model(poses: np.ndarray, model_type: str) -> np.ndarray:
    if model_type == "smplh":
        return poses[:, 3 : SOURCE_BODY_JOINT_COUNT * 3]
    left_hand_leaf, right_hand_leaf = build_smpl_hand_leaf_rotations(poses)
    return np.concatenate([poses[:, 3 : SOURCE_BODY_JOINT_COUNT * 3], left_hand_leaf, right_hand_leaf], axis=1)


def build_hand_poses_for_model(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame_count = poses.shape[0]
    if poses.shape[1] >= SMPLH_JOINT_COUNT * 3:
        return (
            poses[:, SMPLH_LEFT_HAND_START * 3 : SMPLH_RIGHT_HAND_START * 3],
            poses[:, SMPLH_RIGHT_HAND_START * 3 : SMPLH_JOINT_COUNT * 3],
        )
    return np.zeros((frame_count, 45), dtype=np.float64), np.zeros((frame_count, 45), dtype=np.float64)


def build_smpl_hand_leaf_rotations(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame_count = poses.shape[0]
    if poses.shape[1] >= SMPLH_JOINT_COUNT * 3:
        return (
            poses[:, SMPLH_LEFT_HAND_START * 3 : SMPLH_LEFT_HAND_START * 3 + 3],
            poses[:, SMPLH_RIGHT_HAND_START * 3 : SMPLH_RIGHT_HAND_START * 3 + 3],
        )
    return np.zeros((frame_count, 3), dtype=np.float64), np.zeros((frame_count, 3), dtype=np.float64)


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
        return SMPL_PARENTS.copy()
    if hasattr(model, "parents"):
        parents = np.asarray(model.parents.detach().cpu().numpy(), dtype=np.int64)
        return parents[:SMPL_JOINT_COUNT]
    return SMPL_PARENTS.copy()


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


def write_manifest_record(manifest_path: Path, record: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
