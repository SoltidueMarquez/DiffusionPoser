from __future__ import annotations

import argparse
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
from data_loaders.realtime_pose_validation import load_realtime_metadata
from data_loaders.sensor_masking import BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY, SMPL_JOINT_COUNT


DIFFUSIONPOSER_ROOT = Path(__file__).resolve().parents[1]
UNITY_ROOT = DIFFUSIONPOSER_ROOT.parent / "SIGGRAPH2024Unity"
DEFAULT_REPLAY_JSON = (
    UNITY_ROOT
    / "Assets"
    / "Projects"
    / "RealtimePose"
    / "TestData"
    / "Generated"
    / "realtime_pose_replay.json"
)
DEFAULT_SOURCE_REST_JSON = DEFAULT_REPLAY_JSON.with_name("smpl_source_rest.json")
DEFAULT_SMPL_MODEL_DIR = DIFFUSIONPOSER_ROOT / "dataset" / "body_models"
IDENTITY_QUATERNION = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export subject-specific SMPL zero-pose rest JSON."
    )
    parser.add_argument("--replay_json", default=str(DEFAULT_REPLAY_JSON), type=str)
    parser.add_argument(
        "--source_npz",
        default="",
        type=str,
        help="显式指定 converted source npz；提供后不读取 replay metadata。",
    )
    parser.add_argument("--output_json", default=str(DEFAULT_SOURCE_REST_JSON), type=str)
    parser.add_argument("--smpl_model_dir", default=str(DEFAULT_SMPL_MODEL_DIR), type=str)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 必须包含 JSON object。")
    return payload


def _resolve_existing_path(raw_path: str, anchors: tuple[Path, ...], label: str) -> Path:
    path = Path(raw_path)
    candidates = (path,) if path.is_absolute() else tuple(anchor / path for anchor in anchors)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"无法从 {label} 定位文件：{raw_path}")


def resolve_source_npz_from_replay(replay_json: Path) -> Path:
    payload = _load_json(replay_json)
    metadata = payload.get("metadata") or {}
    source_path = metadata.get("sourcePath") or (
        metadata.get("sourceMetadata") or {}
    ).get("source_path")
    if not source_path:
        raise KeyError(f"{replay_json} metadata 缺少 sourcePath。")
    return _resolve_existing_path(
        str(source_path),
        (DIFFUSIONPOSER_ROOT, replay_json.parent, Path.cwd()),
        "replay metadata",
    )


def load_converted_source_metadata(source_npz: Path) -> dict[str, Any]:
    with np.load(source_npz, allow_pickle=False) as payload:
        if BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY not in payload.files:
            raise KeyError(
                f"{source_npz} 缺少 `{BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY}`，"
                "不是当前 converted source。"
            )
        return load_realtime_metadata(payload, path=source_npz)


def resolve_original_amass_path(source_npz: Path, metadata: dict[str, Any]) -> Path:
    source_path = metadata.get("source_path")
    if not source_path:
        raise KeyError(f"{source_npz} metadata 缺少 source_path。")
    return _resolve_existing_path(
        str(source_path),
        (DIFFUSIONPOSER_ROOT, source_npz.parent, Path.cwd()),
        "converted source metadata",
    )


def load_amass_body_shape(amass_path: Path) -> tuple[np.ndarray, str]:
    with np.load(amass_path, allow_pickle=True) as payload:
        if "betas" not in payload.files or "gender" not in payload.files:
            raise KeyError(f"{amass_path} 必须包含 betas 和 gender。")
        betas = np.asarray(payload["betas"], dtype=np.float64).reshape(-1)
        gender = normalize_gender(payload["gender"])
    if betas.size == 0:
        raise ValueError(f"{amass_path} 的 betas 不能为空。")
    return betas, gender


def load_smpl_zero_pose(
    amass_path: Path,
    smpl_model_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """按原始 AMASS 身形运行一次 zero-pose forward，得到个体化 T-pose。"""

    import torch

    model_dir = smpl_model_dir.resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"SMPL model 目录不存在：{model_dir}")
    betas, gender = load_amass_body_shape(amass_path)
    model = create_smpl_model(model_dir=model_dir, gender=gender)
    model.eval()
    model_type = getattr(model, "diffusionposer_model_type", "smpl")
    device = next(model.parameters()).device
    fitted_betas = fit_betas(betas, get_model_num_betas(model))
    body_pose_dim = 63 if model_type == "smplh" else 69
    parameters: dict[str, Any] = {
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
        joints = extract_smpl_style_joints(output.joints, model_type=model_type)[0]
        vertices = output.vertices[0].detach().cpu().numpy()
    joints = np.asarray(joints, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    if joints.shape != (SMPL_JOINT_COUNT, 3):
        raise ValueError(f"SMPL zero-pose joints 应为 [24,3]，实际为 {joints.shape}")
    return joints, vertices


def build_rest_local_offsets(
    rest_joints: np.ndarray,
    rest_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """同时生成原始 T-pose offset 和与当前 Unity 坐标约定一致的 FK offset。"""

    ground_y = float(np.min(rest_vertices[:, 1]))
    offsets = np.zeros((SMPL_JOINT_COUNT, 3), dtype=np.float64)
    offsets[0] = np.asarray([0.0, rest_joints[0, 1] - ground_y, 0.0])
    for joint_index in range(1, SMPL_JOINT_COUNT):
        parent_index = int(SMPL_PARENTS[joint_index])
        offsets[joint_index] = rest_joints[joint_index] - rest_joints[parent_index]
    if offsets[0, 1] <= 0.0:
        raise ValueError("grounded pelvis height 必须大于零。")
    source_fk_offsets = offsets.copy()
    source_fk_offsets[1:] = source_fk_offsets[1:] @ AMASS_TO_UNITY.T
    return offsets.astype(np.float32), source_fk_offsets.astype(np.float32)


def _vector3(value: np.ndarray) -> dict[str, float]:
    return {"x": float(value[0]), "y": float(value[1]), "z": float(value[2])}


def build_source_rest_payload(source_npz: Path, smpl_model_dir: Path) -> dict[str, Any]:
    metadata = load_converted_source_metadata(source_npz)
    amass_path = resolve_original_amass_path(source_npz, metadata)
    joints, vertices = load_smpl_zero_pose(amass_path, smpl_model_dir)
    rest_offsets, source_fk_offsets = build_rest_local_offsets(joints, vertices)
    rotations = [dict(IDENTITY_QUATERNION) for _ in range(SMPL_JOINT_COUNT)]
    return {
        "debugMarker": "DEBUG_RETARGET_SOURCE_REST",
        "schemaName": "realtime_pose_source",
        "poseRepresentation": "body_fbx_local_delta_6d",
        "sourceNpz": str(source_npz.resolve()),
        "sourceAmass": str(amass_path),
        "bodyModelDir": str(smpl_model_dir.resolve()),
        "restPoseSource": "smpl_zero_pose_tpose",
        "isMirrored": bool(metadata.get("is_mirrored", False)),
        "boneCount": SMPL_JOINT_COUNT,
        "boneNames": list(SMPL_JOINT_NAMES),
        "parentIndices": [int(value) for value in SMPL_PARENTS.tolist()],
        "restLocalOffsets": [_vector3(value) for value in rest_offsets],
        "sourceFkLocalOffsets": [_vector3(value) for value in source_fk_offsets],
        "restLocalRotations": rotations,
        "restWorldRotations": [dict(IDENTITY_QUATERNION) for _ in range(SMPL_JOINT_COUNT)],
        "rotationConvention": "SMPL zero-pose offsets; child FK offsets use AMASS-to-Unity coordinates.",
    }


def export_source_rest_json(
    source_npz: Path,
    output_json: Path,
    smpl_model_dir: Path,
) -> Path:
    payload = build_source_rest_payload(source_npz, smpl_model_dir)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    return output_json


def main(argv: list[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    source_npz = (
        Path(args.source_npz).resolve()
        if args.source_npz
        else resolve_source_npz_from_replay(Path(args.replay_json).resolve())
    )
    path = export_source_rest_json(
        source_npz=source_npz,
        output_json=Path(args.output_json).resolve(),
        smpl_model_dir=Path(args.smpl_model_dir).resolve(),
    )
    print(f"[export_smpl_source_rest] wrote: {path}")
    return path


if __name__ == "__main__":
    main()
