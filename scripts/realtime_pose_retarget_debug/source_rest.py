from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from data_converter.amass_smpl_utils import (
    AMASS_TO_UNITY,
    create_smpl_model,
    extract_smpl_style_joints,
    fit_betas,
    get_model_num_betas,
    normalize_gender,
)
from data_loaders.realtime_pose_kinematics import SMPL_JOINT_NAMES, SMPL_PARENTS
from data_loaders.sensor_masking import (
    POSE_REPRESENTATION_KEY,
    REALTIME_POSE_V2_CONTACT_SCHEMA_NAME,
    SMPL_JOINT_COUNT,
    get_schema_spec,
    validate_pose_representation,
)

from .config import DEFAULT_REPLAY_JSON, DEFAULT_SOURCE_REST_JSON, DIFFUSIONPOSER_ROOT
from .replay_io import load_json, resolve_source_npz


IDENTITY_QUATERNION = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
DEFAULT_SMPL_MODEL_DIR = DIFFUSIONPOSER_ROOT / "dataset" / "body_models"


def vector3_payload(value: np.ndarray) -> dict[str, float]:
    return {
        "x": float(value[0]),
        "y": float(value[1]),
        "z": float(value[2]),
    }


def quaternion_payload() -> dict[str, float]:
    return dict(IDENTITY_QUATERNION)


def load_converted_source_metadata(source_npz: Path) -> dict[str, Any]:
    """读取 converted source npz 的 metadata，并校验当前 root-global 6D schema。"""

    schema = get_schema_spec(REALTIME_POSE_V2_CONTACT_SCHEMA_NAME)
    with np.load(source_npz, allow_pickle=True) as data:
        if POSE_REPRESENTATION_KEY not in data.files:
            raise KeyError(f"{source_npz} missing `{POSE_REPRESENTATION_KEY}`")
        validate_pose_representation(data[POSE_REPRESENTATION_KEY], schema_name=schema.name, source=str(source_npz))

        if "metadata" not in data.files:
            raise KeyError(f"{source_npz} missing `metadata`; cannot resolve original AMASS source_path for SMPL T-pose rest.")
        metadata_text = str(data["metadata"].item())

    try:
        metadata = json.loads(metadata_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source_npz} metadata is not valid JSON.") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{source_npz} metadata must be a JSON object.")

    metadata_pose_representation = metadata.get("pose_representation")
    if metadata_pose_representation is not None:
        validate_pose_representation(metadata_pose_representation, schema_name=schema.name, source=f"{source_npz}:metadata")
    return metadata


def resolve_original_amass_path(source_npz: Path, metadata: dict[str, Any]) -> Path:
    """从 converted metadata 定位原始 AMASS 文件，镜像样本也复用同一份 body shape。"""

    source_path = metadata.get("source_path")
    if not source_path:
        raise KeyError(f"{source_npz} metadata missing `source_path`; cannot build SMPL zero-pose rest.")

    path = Path(str(source_path))
    candidates = [path] if path.is_absolute() else [DIFFUSIONPOSER_ROOT / path, source_npz.parent / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"original AMASS source file not found from metadata source_path: {source_path}")


def load_amass_body_shape(amass_path: Path) -> tuple[np.ndarray, str]:
    with np.load(amass_path, allow_pickle=True) as data:
        if "betas" not in data.files:
            raise KeyError(f"{amass_path} missing `betas`; cannot build subject-specific SMPL T-pose rest.")
        if "gender" not in data.files:
            raise KeyError(f"{amass_path} missing `gender`; cannot select SMPL body model.")
        betas = np.asarray(data["betas"], dtype=np.float64).reshape(-1)
        gender = normalize_gender(data["gender"])

    if betas.size == 0:
        raise ValueError(f"{amass_path} `betas` is empty.")
    return betas, gender


def load_smpl_zero_pose_joints(amass_path: Path, smpl_model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """用原始 AMASS 的 betas/gender 运行一次 SMPL/SMPL-H zero-pose forward。

    返回值保持 SMPL/AMASS 原始坐标，不套 AMASS_TO_UNITY，这样 identity skeleton
    在调试可视化中就是 Y-up 的 SMPL zero-pose/T-pose。
    """

    import torch

    smpl_model_dir = smpl_model_dir.resolve()
    if not smpl_model_dir.exists():
        raise FileNotFoundError(f"SMPL model directory not found: {smpl_model_dir}")

    betas, gender = load_amass_body_shape(amass_path)
    try:
        model = create_smpl_model(model_dir=smpl_model_dir, gender=gender)
    except Exception as exc:
        raise RuntimeError(f"failed to load SMPL/SMPL-H model for gender={gender!r} from {smpl_model_dir}") from exc
    model.eval()
    model_type = getattr(model, "diffusionposer_model_type", "smpl")
    device = next(model.parameters()).device
    fitted_betas = fit_betas(betas, get_model_num_betas(model))

    body_pose_dim = 63 if model_type == "smplh" else 69
    parameters = {
        "global_orient": torch.zeros((1, 3), dtype=torch.float32, device=device),
        "body_pose": torch.zeros((1, body_pose_dim), dtype=torch.float32, device=device),
        "betas": torch.as_tensor(fitted_betas[None], dtype=torch.float32, device=device),
        "transl": torch.zeros((1, 3), dtype=torch.float32, device=device),
        "return_verts": True,
    }
    if model_type == "smplh":
        parameters.update(
            left_hand_pose=torch.zeros((1, 45), dtype=torch.float32, device=device),
            right_hand_pose=torch.zeros((1, 45), dtype=torch.float32, device=device),
        )

    with torch.no_grad():
        output = model(**parameters)
        rest_joints = extract_smpl_style_joints(output.joints, model_type=model_type)[0].astype(np.float64)
        rest_vertices = output.vertices[0].detach().cpu().numpy().astype(np.float64)
    if rest_joints.shape != (SMPL_JOINT_COUNT, 3):
        raise ValueError(f"SMPL zero-pose joints must be [24,3], got {rest_joints.shape}")
    return rest_joints, rest_vertices


def build_tpose_rest_local_offsets(rest_joints: np.ndarray, rest_vertices: np.ndarray | None = None) -> np.ndarray:
    """从 SMPL zero-pose joints 生成 Unity 可见 source skeleton 的 local offsets。

    root offset 表示 grounded pelvis height；其他关节保存 `J[i] - J[parent]`。
    这个结果只服务 source rest JSON 的 T-pose 可视化，不替代训练/FK 的 joint_offsets_parent。
    """

    joints = np.asarray(rest_joints, dtype=np.float64)
    if joints.shape != (SMPL_JOINT_COUNT, 3):
        raise ValueError(f"rest_joints must be [24,3], got {joints.shape}")

    if rest_vertices is not None:
        vertices = np.asarray(rest_vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"rest_vertices must be [V,3], got {vertices.shape}")
        ground_y = float(np.min(vertices[:, 1]))
    else:
        ground_y = float(np.min(joints[:, 1]))

    offsets = np.zeros((SMPL_JOINT_COUNT, 3), dtype=np.float64)
    offsets[0] = np.asarray([0.0, joints[0, 1] - ground_y, 0.0], dtype=np.float64)
    for joint_index in range(1, SMPL_JOINT_COUNT):
        parent_index = int(SMPL_PARENTS[joint_index])
        offsets[joint_index] = joints[joint_index] - joints[parent_index]

    if offsets[0, 1] <= 0.0:
        raise ValueError(f"grounded pelvis height must be positive, got {offsets[0, 1]}")
    return offsets.astype(np.float32)


def build_source_fk_local_offsets(rest_joints: np.ndarray, rest_vertices: np.ndarray | None = None) -> np.ndarray:
    """生成与 `body_pose_root_global_6d` 当前旋转坐标兼容的 source FK offsets。

    `restLocalOffsets` 保持 raw SMPL zero-pose，可用于 identity T-pose 可视化；
    SourceSkeletonFK 播放时会套用经过 AMASS_TO_UNITY 共轭后的当前帧旋转，所以子骨骼
    offset 也必须进入同一坐标约定。root 单独保留 grounded pelvis height。
    """

    rest_offsets = build_tpose_rest_local_offsets(rest_joints=rest_joints, rest_vertices=rest_vertices)
    fk_offsets = rest_offsets.copy()
    fk_offsets[1:] = fk_offsets[1:] @ AMASS_TO_UNITY.T
    fk_offsets[0] = rest_offsets[0]
    return fk_offsets.astype(np.float32)


def build_source_rest_pose_payload(
    source_npz: Path,
    smpl_model_dir: Path = DEFAULT_SMPL_MODEL_DIR,
) -> dict[str, Any]:
    """导出真正的 SMPL zero-pose/T-pose source rest payload。"""

    schema = get_schema_spec(REALTIME_POSE_V2_CONTACT_SCHEMA_NAME)
    metadata = load_converted_source_metadata(source_npz)
    amass_path = resolve_original_amass_path(source_npz=source_npz, metadata=metadata)
    rest_joints, rest_vertices = load_smpl_zero_pose_joints(amass_path=amass_path, smpl_model_dir=smpl_model_dir)
    rest_local_offsets = build_tpose_rest_local_offsets(rest_joints=rest_joints, rest_vertices=rest_vertices)
    source_fk_local_offsets = build_source_fk_local_offsets(rest_joints=rest_joints, rest_vertices=rest_vertices)

    rotations = [quaternion_payload() for _ in range(SMPL_JOINT_COUNT)]
    return {
        "debugMarker": "DEBUG_RETARGET_SOURCE_REST",
        "schemaName": schema.name,
        "poseRepresentation": schema.pose_representation,
        "sourceNpz": str(source_npz.resolve()),
        "sourceAmass": str(amass_path),
        "bodyModelDir": str(smpl_model_dir.resolve()),
        "restPoseSource": "smpl_zero_pose_tpose",
        "isMirrored": bool(metadata.get("is_mirrored", False)),
        "boneCount": SMPL_JOINT_COUNT,
        "boneNames": list(SMPL_JOINT_NAMES),
        "parentIndices": [int(value) for value in SMPL_PARENTS.tolist()],
        "restLocalOffsets": [vector3_payload(offset) for offset in rest_local_offsets],
        "sourceFkLocalOffsets": [vector3_payload(offset) for offset in source_fk_local_offsets],
        "restLocalRotations": rotations,
        "restWorldRotations": [quaternion_payload() for _ in range(SMPL_JOINT_COUNT)],
        "rotationConvention": "smpl_zero_pose_tpose offsets; restWorldRotations are identity SMPL zero-pose joint orientations; body.fbx playback preserves bind local positions and applies parent-local source rotations onto target rest local rotations",
    }


def export_source_rest_pose_json(
    source_npz: Path,
    output_json: Path,
    smpl_model_dir: Path = DEFAULT_SMPL_MODEL_DIR,
) -> Path:
    payload = build_source_rest_pose_payload(source_npz=source_npz, smpl_model_dir=smpl_model_dir)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    return output_json


def export_source_rest_pose_from_replay(
    replay_json: Path = DEFAULT_REPLAY_JSON,
    output_json: Path = DEFAULT_SOURCE_REST_JSON,
    smpl_model_dir: Path = DEFAULT_SMPL_MODEL_DIR,
) -> Path:
    payload = load_json(replay_json)
    source_npz = resolve_source_npz(payload, replay_json)
    return export_source_rest_pose_json(source_npz=source_npz, output_json=output_json, smpl_model_dir=smpl_model_dir)
