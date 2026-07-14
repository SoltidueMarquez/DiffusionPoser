"""
把 AMASS SMPL/SMPL-H 动作转换为 realtime_pose 源数据。
默认生成 `realtime_pose_stationary5_v1`；
`realtime_pose_body_fbx_local_root_y0_v1` 仅作为 legacy alias 保留。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from data_converter.amass_smpl_utils import (
    MIRROR_DIR_NAME,
    MotionSource,
    SmplModelCache,
    SmplMotion,
    iter_amass_motion_files,
    load_motion_source,
    mirror_motion_source,
    run_smpl_forward,
    validate_args as validate_shared_args,
    write_manifest_record,
)
from data_loaders.body_fbx_kinematics import (
    BodyFbxRest,
    actor_root_positions_from_pelvis,
    extract_root_heading_from_source_pelvis_up,
    fk_body_fbx_local_delta_root_y0,
    load_body_fbx_rest,
    source_global_rotations_to_body_fbx_local_delta_6d,
)
from data_loaders.realtime_pose_kinematics import (
    JOINT_INDEX,
    TRACKER_JOINT_INDICES,
    build_body_pose_root_global_6d,
    derive_stationary_prob_5,
    encode_root_delta_xz_ref,
    estimate_root_global_offsets,
    extract_yaw_from_rotations,
    rotation_6d_forward_up_np,
    wrap_radians,
)
from data_loaders.realtime_pose_contract import (
    RUNTIME_CONTRACT_METADATA_FIELDS,
    load_source_metadata,
    required_realtime_source_fields,
    runtime_contract_metadata,
    validate_realtime_source_contract,
    validate_root_y0_invariants,
    validate_schema_metadata,
)
from data_loaders.sensor_masking import (
    POSE_REPRESENTATION_KEY,
    POSE_REPRESENTATION_BODY_FBX_LOCAL_DELTA_6D,
    DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SCHEMA_NAMES,
    STATIONARY_JOINT_INDICES,
    STATIONARY_JOINT_NAMES,
    get_schema_spec,
)
from data_loaders.stationary_label_config import stationary_label_metadata
from utils.artifact_paths import source_root
from utils.data_roots import load_data_roots


DEFAULT_SOURCE_SET_NAME = "amass_60hz"
DEFAULT_SMPL_MODEL_DIR = Path("dataset/body_models")
SOURCE_PROVENANCE_FIELDS = (
    "schema_canonical_name",
    "raw_dataset",
    "raw_root_key",
    "raw_relative_path",
    "source_set_name",
    "converter_args",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AMASS SMPL motions to realtime_pose files.")
    parser.add_argument("--data_roots_config", default="", type=str)
    parser.add_argument("--source_set_name", default=DEFAULT_SOURCE_SET_NAME, type=str)
    parser.add_argument("--amass_dir", default="", type=str)
    parser.add_argument("--smpl_model_dir", default="", type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--target_fps", default=60.0, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--worker_torch_threads", default=1, type=int)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--rebuild_manifest", action="store_true")
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument("--reuse_source_dir", default="", type=Path)
    parser.add_argument("--schema", default=DEFAULT_REALTIME_POSE_SCHEMA_NAME, choices=REALTIME_POSE_SCHEMA_NAMES, type=str)
    parser.add_argument("--body_fbx_rest_json", default="", type=str)
    # 复用 shared validate_args 所需字段；当前转换链路不实际使用这些可视化参数。
    parser.add_argument("--height_threshold", default=0.04, type=float)
    parser.add_argument("--speed_threshold", default=0.15, type=float)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualize_limit", default=0, type=int)
    parser.add_argument("--visualize_dir", default=Path("output/realtime_pose_visualization"), type=Path)
    parser.add_argument("--visualize_fps", default=20.0, type=float)
    return parser.parse_args(argv)


def resolve_converter_paths(args: argparse.Namespace) -> argparse.Namespace:
    """根据显式 CLI 参数或 data_roots 配置补齐转换入口路径。

    argparse 无法可靠区分“用户显式传了旧默认值”和“使用默认值”，所以路径参数先保留为空字符串；
    这里再把空值视为需要从 schema-aware 数据根推导的路径。
    """

    roots = None

    def get_roots():
        nonlocal roots
        if roots is None:
            roots = load_data_roots(getattr(args, "data_roots_config", "") or None)
        return roots

    schema_name = str(getattr(args, "schema", DEFAULT_REALTIME_POSE_SCHEMA_NAME))
    source_set_name = str(getattr(args, "source_set_name", DEFAULT_SOURCE_SET_NAME))

    if _path_arg_is_empty(getattr(args, "amass_dir", "")):
        args.amass_dir = get_roots().amass_root
    else:
        args.amass_dir = Path(args.amass_dir)

    if _path_arg_is_empty(getattr(args, "smpl_model_dir", "")):
        roots_value = get_roots().smpl_model_dir
        args.smpl_model_dir = roots_value if roots_value is not None else DEFAULT_SMPL_MODEL_DIR
    else:
        args.smpl_model_dir = Path(args.smpl_model_dir)

    if _path_arg_is_empty(getattr(args, "output_dir", "")):
        args.output_dir = source_root(get_roots(), schema_name=schema_name, source_set_name=source_set_name)
    else:
        args.output_dir = Path(args.output_dir)

    if _path_arg_is_empty(getattr(args, "body_fbx_rest_json", "")):
        roots_value = get_roots().body_fbx_rest_json
        args.body_fbx_rest_json = roots_value if roots_value is not None else ""
    else:
        args.body_fbx_rest_json = Path(args.body_fbx_rest_json)

    return args


def _path_arg_is_empty(value: object) -> bool:
    if value is None:
        return True
    return not str(value).strip()


def build_converter_args_metadata(args: argparse.Namespace) -> dict[str, object]:
    return {
        "target_fps": float(getattr(args, "target_fps", 60.0)),
        "mirror": bool(getattr(args, "mirror", False)),
        "schema": str(getattr(args, "schema", DEFAULT_REALTIME_POSE_SCHEMA_NAME)),
        "source_set_name": str(getattr(args, "source_set_name", DEFAULT_SOURCE_SET_NAME)),
    }


def build_source_provenance(source: MotionSource, args: argparse.Namespace, schema) -> dict[str, object]:
    raw_relative_path = source.original_relative_path or source.relative_path
    return build_path_provenance(raw_relative_path=raw_relative_path, args=args, schema=schema)


def build_path_provenance(raw_relative_path: Path | str, args: argparse.Namespace, schema) -> dict[str, object]:
    return {
        "schema_name": schema.name,
        "schema_canonical_name": str(schema.canonical_name),
        "raw_dataset": "AMASS",
        "raw_root_key": "amass_root",
        "raw_relative_path": Path(raw_relative_path).as_posix(),
        "source_set_name": str(getattr(args, "source_set_name", DEFAULT_SOURCE_SET_NAME)),
        "converter_args": build_converter_args_metadata(args),
    }


def build_source_provenance_from_metadata(
    metadata: dict[str, Any],
    raw_relative_path: Path | str,
    args: argparse.Namespace,
    schema,
) -> dict[str, object]:
    fallback = build_path_provenance(raw_relative_path=raw_relative_path, args=args, schema=schema)
    # 已有 source 的来源记录代表当时生成数据的真实输入；重建 manifest 时不能用本次 CLI 覆盖。
    return {
        field: metadata[field] if field in metadata else fallback[field]
        for field in SOURCE_PROVENANCE_FIELDS
    }


_WORKER_ARGS: argparse.Namespace | None = None
_WORKER_MODEL_CACHE: SmplModelCache | None = None


def validate_converter_args(args: argparse.Namespace) -> None:
    validate_shared_args(args)
    if int(args.num_workers) <= 0:
        raise ValueError("--num_workers 必须为正整数")
    if int(args.worker_torch_threads) < 0:
        raise ValueError("--worker_torch_threads 必须大于等于 0")


def build_conversion_work_items(args: argparse.Namespace, motion_files: list[Path]) -> list[tuple[Path, bool]]:
    mirror_variants = (False, True) if args.mirror else (False,)
    return [(path, mirror_variant) for path in motion_files for mirror_variant in mirror_variants]


def copy_args_for_worker(args: argparse.Namespace) -> argparse.Namespace:
    payload = dict(vars(args))
    payload.pop("_body_fbx_rest", None)
    return argparse.Namespace(**payload)


def init_conversion_worker(args: argparse.Namespace, torch_threads: int) -> None:
    global _WORKER_ARGS, _WORKER_MODEL_CACHE
    _WORKER_ARGS = args
    _WORKER_MODEL_CACHE = SmplModelCache(model_dir=args.smpl_model_dir)
    configure_worker_torch_threads(torch_threads)


def configure_worker_torch_threads(torch_threads: int) -> None:
    if int(torch_threads) <= 0:
        return
    import torch

    # Limit per-process Torch threads to avoid CPU oversubscription.
    torch.set_num_threads(int(torch_threads))
    try:
        torch.set_num_interop_threads(max(1, int(torch_threads)))
    except RuntimeError:
        pass


def convert_work_item_in_worker(item: tuple[Path, bool]) -> dict[str, Any]:
    global _WORKER_MODEL_CACHE
    if _WORKER_ARGS is None:
        raise RuntimeError("converter worker 尚未初始化 args")
    if _WORKER_MODEL_CACHE is None:
        _WORKER_MODEL_CACHE = SmplModelCache(model_dir=_WORKER_ARGS.smpl_model_dir)
    path, mirror_variant = item
    return convert_one_motion_safely(
        path=path,
        args=_WORKER_ARGS,
        model_cache=_WORKER_MODEL_CACHE,
        mirror_variant=mirror_variant,
    )


def output_path_for(source: MotionSource, output_dir: Path) -> Path:
    return output_dir / source.relative_path.with_suffix(".npz")


def build_realtime_pose_features(
    smpl_motion: SmplMotion,
    schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    target_fps: float = 60.0,
    body_fbx_rest: BodyFbxRest | None = None,
) -> dict[str, np.ndarray]:
    """构建当前 realtime_pose source 字段。"""

    schema = get_schema_spec(schema_name)
    joint_positions = smpl_motion.joint_positions.astype(np.float64)
    joint_rotations = smpl_motion.joint_rotations.astype(np.float64)
    pelvis_world = joint_positions[:, JOINT_INDEX["pelvis"]].copy()
    if schema.pose_representation == POSE_REPRESENTATION_BODY_FBX_LOCAL_DELTA_6D:
        if body_fbx_rest is None:
            raise ValueError("生成 body_fbx_local_delta_6d source 必须提供 body_fbx_rest。")
        root_yaw = extract_root_heading_from_source_pelvis_up(
            joint_rotations[:, JOINT_INDEX["pelvis"]],
        ).astype(np.float32)
        body_pose_6d = source_global_rotations_to_body_fbx_local_delta_6d(joint_rotations)
        root_pos_world = actor_root_positions_from_pelvis(
            pelvis_world=pelvis_world,
            root_heading=root_yaw,
            pelvis_rest_local_position=body_fbx_rest.pelvis_local_position,
        )
        root_pos_world[:, 1] = 0.0
        pelvis_height = pelvis_world[:, 1:2].astype(np.float32)
        joints_world, joint_rotations_world = fk_body_fbx_local_delta_root_y0(
            body_pose_local_delta_6d=body_pose_6d,
            actor_root_pos_world=root_pos_world,
            root_heading=root_yaw,
            pelvis_height=pelvis_height,
            rest=body_fbx_rest,
        )
        tracker_pos_world = joints_world[:, body_fbx_rest.tracker_joint_indices].astype(np.float32)
        tracker_rot_world_6d = rotation_6d_forward_up_np(
            joint_rotations_world[:, body_fbx_rest.tracker_joint_indices],
        ).astype(np.float32)
        joint_offsets_parent = body_fbx_rest.rest_local_positions.astype(np.float32)
        joint_rest_local_rotations_6d = rotation_6d_forward_up_np(
            body_fbx_rest.rest_local_rotations,
        ).astype(np.float32)
    else:
        root_pos_world = pelvis_world.copy()
        root_pos_world[:, 1] = 0.0
        root_yaw = extract_yaw_from_rotations(joint_rotations[:, JOINT_INDEX["pelvis"]]).astype(np.float32)
        body_pose_6d = build_body_pose_root_global_6d(
            global_rotations=joint_rotations,
            root_yaws=root_yaw.astype(np.float64),
        )
        tracker_pos_world = joint_positions[:, TRACKER_JOINT_INDICES].astype(np.float32)
        tracker_rot_world_6d = rotation_6d_forward_up_np(joint_rotations[:, TRACKER_JOINT_INDICES]).astype(np.float32)
        joints_world = joint_positions.astype(np.float32)
        joint_rotations_world = joint_rotations.astype(np.float32)
        joint_offsets_parent = estimate_root_global_offsets(
            joints_world=joints_world,
            body_pose_root_global_6d=body_pose_6d,
            root_yaws=root_yaw,
            root_pos_world=root_pos_world,
        )
        joint_rest_local_rotations_6d = None
    yaw_delta = np.zeros_like(root_yaw, dtype=np.float32)
    if root_yaw.shape[0] > 1:
        yaw_delta[1:] = wrap_radians(root_yaw[1:].astype(np.float64) - root_yaw[:-1].astype(np.float64)).astype(np.float32)
    root_heading_delta_sincos = np.stack([np.sin(yaw_delta), np.cos(yaw_delta)], axis=-1).astype(np.float32)

    features = {
        schema.body_pose_key: body_pose_6d,
        POSE_REPRESENTATION_KEY: np.asarray(schema.pose_representation),
        "root_pos_world": root_pos_world.astype(np.float32),
        "root_yaw": root_yaw.astype(np.float32),
        schema.root_heading_delta_key: root_heading_delta_sincos,
        "tracker_pos_world": tracker_pos_world,
        "tracker_rot_world_6d": tracker_rot_world_6d,
        "joints_world": joints_world,
        "joint_offsets_parent": joint_offsets_parent,
    }
    if joint_rest_local_rotations_6d is not None:
        features["joint_rest_local_rotations_6d"] = joint_rest_local_rotations_6d
    if body_fbx_rest is not None and body_fbx_rest.source_path is not None:
        features["body_fbx_rest_json"] = np.asarray(str(body_fbx_rest.source_path))
    if schema.supports_root_motion:
        features["root_delta_xz_ref"] = encode_root_delta_xz_ref(
            root_pos_world=root_pos_world.astype(np.float32),
            root_yaw=root_yaw.astype(np.float32),
        )
        features[schema.pelvis_height_key] = pelvis_world[:, 1:2].astype(np.float32)
    if schema.supports_stationary_prob:
        features["stationary_prob_5"] = derive_stationary_prob_5(
            joints_world=joints_world,
            joint_rotations_world=joint_rotations_world,
            fps=float(target_fps),
        )
    return features


def save_realtime_pose_motion(
    output_path: Path,
    features: dict[str, np.ndarray],
    source: MotionSource,
    target_fps: float,
    schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    args: argparse.Namespace | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = get_schema_spec(schema_name)
    provenance_args = argparse.Namespace(
        target_fps=target_fps,
        mirror=source.is_mirrored,
        schema=schema_name,
        source_set_name=DEFAULT_SOURCE_SET_NAME,
    ) if args is None else args
    metadata = {
        "schema_name": schema_name,
        "schema_canonical_name": str(schema.canonical_name),
        "pose_representation": schema.pose_representation,
        "root_y_policy": schema.root_y_policy,
        "pelvis_height_mode": schema.pelvis_height_mode,
        "source_path": str(source.path),
        "source_relative_path": str(source.relative_path),
        "original_source_relative_path": str(source.original_relative_path or source.relative_path),
        "stablemotion_split_key": str(source.relative_path.with_suffix(".npy")).replace("\\", "/"),
        "is_mirrored": bool(source.is_mirrored),
        "source_fps": float(source.source_fps),
        "target_fps": float(target_fps),
        "frames": int(features[schema.body_pose_key].shape[0]),
        "tracker_order": ["head", "left_wrist", "right_wrist", "waist", "left_foot", "right_foot"],
    }
    metadata.update(runtime_contract_metadata())
    metadata.update(build_source_provenance(source=source, args=provenance_args, schema=schema))
    if schema.supports_stationary_prob:
        metadata["stationary_joint_indices"] = [int(index) for index in STATIONARY_JOINT_INDICES]
        metadata["stationary_joint_names"] = list(STATIONARY_JOINT_NAMES)
        metadata.update(stationary_label_metadata())
    if schema.pose_representation == POSE_REPRESENTATION_BODY_FBX_LOCAL_DELTA_6D and "body_fbx_rest_json" in features:
        metadata["body_fbx_rest_json"] = str(features["body_fbx_rest_json"].item())
    np.savez(output_path, **features, metadata=json.dumps(metadata, ensure_ascii=False))


def load_metadata_from_npz(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata" not in data.files:
        return {}
    value = data["metadata"]
    try:
        text = str(value.item())
    except Exception:
        text = str(value)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def resolve_body_fbx_rest_for_schema(args: argparse.Namespace) -> BodyFbxRest | None:
    schema = get_schema_spec(str(getattr(args, "schema", DEFAULT_REALTIME_POSE_SCHEMA_NAME)))
    if schema.pose_representation != POSE_REPRESENTATION_BODY_FBX_LOCAL_DELTA_6D:
        return None
    cached = getattr(args, "_body_fbx_rest", None)
    if cached is not None:
        return cached
    rest_arg = getattr(args, "body_fbx_rest_json", "")
    rest_text = str(rest_arg).strip()
    # argparse 会把 default="" + type=Path 解析成 Path(".")；这里必须把空路径和当前目录都
    # 视作“未显式提供”，否则会尝试把仓库目录当 JSON 文件打开并触发 PermissionError。
    rest_path = None if rest_text in {"", "."} else rest_arg
    rest = load_body_fbx_rest(rest_path)
    setattr(args, "_body_fbx_rest", rest)
    return rest


def source_relative_path_for(path: Path, args: argparse.Namespace, mirror_variant: bool) -> Path:
    relative_path = path.relative_to(args.amass_dir)
    return Path(MIRROR_DIR_NAME) / relative_path if mirror_variant else relative_path


def record_for_output(
    path: Path,
    output_path: Path,
    args: argparse.Namespace,
    mirror_variant: bool,
    status: str,
) -> dict[str, Any]:
    relative_path = source_relative_path_for(path=path, args=args, mirror_variant=mirror_variant)
    metadata: dict[str, Any] = {}
    frames = 0
    schema = get_schema_spec(str(getattr(args, "schema", DEFAULT_REALTIME_POSE_SCHEMA_NAME)))
    if output_path.exists():
        with np.load(output_path, allow_pickle=False) as data:
            metadata = load_metadata_from_npz(data)
            if schema.body_pose_key in data.files:
                frames = int(data[schema.body_pose_key].shape[0])
    source_relative_path = Path(str(metadata.get("source_relative_path", relative_path)))
    raw_relative_path = metadata.get(
        "raw_relative_path",
        metadata.get("original_source_relative_path", path.relative_to(args.amass_dir)),
    )
    provenance = build_source_provenance_from_metadata(
        metadata=metadata,
        raw_relative_path=raw_relative_path,
        args=args,
        schema=schema,
    )
    stablemotion_key = str(metadata.get("stablemotion_split_key", source_relative_path.with_suffix(".npy"))).replace("\\", "/")
    record = {
        "status": status,
        "schema_name": schema.name,
        "schema_canonical_name": str(schema.canonical_name),
        "pose_representation": schema.pose_representation,
        "root_y_policy": schema.root_y_policy,
        "pelvis_height_mode": schema.pelvis_height_mode,
        "source_path": str(metadata.get("source_path", path)),
        "source_relative_path": str(source_relative_path),
        "original_source_relative_path": str(metadata.get("original_source_relative_path", path.relative_to(args.amass_dir))),
        "is_mirrored": bool(metadata.get("is_mirrored", mirror_variant)),
        "stablemotion_split_key": stablemotion_key,
        "output_path": str(output_path),
        "frames": int(metadata.get("frames", frames)),
        **provenance,
    }
    if schema.supports_stationary_prob:
        record.update(stationary_label_metadata())
    return record


def reusable_source_path_for(path: Path, args: argparse.Namespace, mirror_variant: bool) -> Path | None:
    reuse_source_dir = Path(args.reuse_source_dir) if getattr(args, "reuse_source_dir", "") else None
    if reuse_source_dir is None:
        return None
    relative_path = source_relative_path_for(path=path, args=args, mirror_variant=mirror_variant)
    return reuse_source_dir / relative_path.with_suffix(".npz")


def load_reusable_realtime_features(
    reuse_path: Path,
    schema_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    schema = get_schema_spec(schema_name)
    with np.load(reuse_path, allow_pickle=False) as data:
        metadata = load_source_metadata(data, source=str(reuse_path))
        missing_runtime_fields = [
            key for key in RUNTIME_CONTRACT_METADATA_FIELDS if key not in metadata
        ]
        if missing_runtime_fields:
            if len(missing_runtime_fields) != len(RUNTIME_CONTRACT_METADATA_FIELDS):
                raise ValueError(
                    f"{reuse_path} metadata 部分缺少 v2 runtime 字段: "
                    f"{missing_runtime_fields}; 拒绝猜测或覆盖已有版本。"
                )
            metadata = {**metadata, **runtime_contract_metadata()}

        validation_payload = {
            key: np.asarray(data[key])
            for key in data.files
            if key != "metadata"
        }
        validation_payload["metadata"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
        validate_realtime_source_contract(validation_payload, schema=schema, source=str(reuse_path))
        validate_schema_metadata(metadata, schema=schema, source=str(reuse_path))
        required = required_source_fields(schema_name)
        features = {
            key: np.asarray(data[key]).astype(np.float32, copy=True)
            for key in required
            if key != POSE_REPRESENTATION_KEY
        }
        features[POSE_REPRESENTATION_KEY] = np.asarray(schema.pose_representation)
        validate_root_y0_invariants(features, schema=schema, source=str(reuse_path))
    return features, metadata


def required_source_fields(schema_name: str) -> set[str]:
    return required_realtime_source_fields(schema_name)


def realtime_source_has_schema(path: Path, schema_name: str) -> bool:
    if not path.exists():
        return False
    schema = get_schema_spec(schema_name)
    with np.load(path, allow_pickle=False) as data:
        try:
            validate_realtime_source_contract(data, schema=schema, source=str(path))
        except (KeyError, ValueError):
            return False
        return True


def save_reused_realtime_source(
    output_path: Path,
    features: dict[str, np.ndarray],
    metadata: dict[str, Any],
    path: Path,
    args: argparse.Namespace,
    mirror_variant: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    relative_path = source_relative_path_for(path=path, args=args, mirror_variant=mirror_variant)
    schema = get_schema_spec(str(args.schema))
    validate_schema_metadata(metadata, schema=schema, source=str(output_path))
    validate_root_y0_invariants(features, schema=schema, source=str(output_path))
    next_metadata = dict(metadata)
    raw_relative_path = next_metadata.get(
        "raw_relative_path",
        next_metadata.get("original_source_relative_path", path.relative_to(args.amass_dir)),
    )
    next_metadata.update(
        {
            "schema_name": schema.name,
            "schema_canonical_name": str(schema.canonical_name),
            "pose_representation": schema.pose_representation,
            "root_y_policy": schema.root_y_policy,
            "pelvis_height_mode": schema.pelvis_height_mode,
            "source_path": str(next_metadata.get("source_path", path)),
            "source_relative_path": str(next_metadata.get("source_relative_path", relative_path)),
            "original_source_relative_path": str(next_metadata.get("original_source_relative_path", path.relative_to(args.amass_dir))),
            "stablemotion_split_key": str(next_metadata.get("stablemotion_split_key", relative_path.with_suffix(".npy"))).replace("\\", "/"),
            "is_mirrored": bool(next_metadata.get("is_mirrored", mirror_variant)),
            "target_fps": float(args.target_fps),
            "frames": int(features[schema.body_pose_key].shape[0]),
            "tracker_order": ["head", "left_wrist", "right_wrist", "waist", "left_foot", "right_foot"],
        }
    )
    next_metadata.update(runtime_contract_metadata())
    next_metadata.update(
        build_source_provenance_from_metadata(
            metadata=next_metadata,
            raw_relative_path=raw_relative_path,
            args=args,
            schema=schema,
        )
    )
    if schema.supports_stationary_prob:
        next_metadata["stationary_joint_indices"] = [int(index) for index in STATIONARY_JOINT_INDICES]
        next_metadata["stationary_joint_names"] = list(STATIONARY_JOINT_NAMES)
        next_metadata.update(stationary_label_metadata())
    np.savez(output_path, **features, metadata=json.dumps(next_metadata, ensure_ascii=False))


def try_reuse_existing_realtime_source(
    path: Path,
    output_path: Path,
    args: argparse.Namespace,
    mirror_variant: bool,
) -> dict[str, Any] | None:
    reuse_path = reusable_source_path_for(path=path, args=args, mirror_variant=mirror_variant)
    if reuse_path is None or not reuse_path.exists():
        return None
    return reuse_realtime_source_file(
        reuse_path=reuse_path,
        output_path=output_path,
        path=path,
        args=args,
        mirror_variant=mirror_variant,
        status="reused_source",
    )


def reuse_realtime_source_file(
    reuse_path: Path,
    output_path: Path,
    path: Path,
    args: argparse.Namespace,
    mirror_variant: bool,
    status: str,
) -> dict[str, Any]:
    features, metadata = load_reusable_realtime_features(
        reuse_path=reuse_path,
        schema_name=str(args.schema),
    )
    save_reused_realtime_source(
        output_path=output_path,
        features=features,
        metadata=metadata,
        path=path,
        args=args,
        mirror_variant=mirror_variant,
    )
    return record_for_output(
        path=path,
        output_path=output_path,
        args=args,
        mirror_variant=mirror_variant,
        status=status,
    )


def convert_one_motion(
    path: Path,
    args: argparse.Namespace,
    model_cache: SmplModelCache,
    mirror_variant: bool = False,
) -> dict[str, Any]:
    relative_path = source_relative_path_for(path=path, args=args, mirror_variant=mirror_variant)
    output_path = args.output_dir / relative_path.with_suffix(".npz")

    if output_path.exists() and args.skip_existing:
        if realtime_source_has_schema(output_path, str(args.schema)):
            return record_for_output(
                path=path,
                output_path=output_path,
                args=args,
                mirror_variant=mirror_variant,
                status="skipped_existing",
            )
        if args.overwrite or args.rebuild_manifest:
            return reuse_realtime_source_file(
                reuse_path=output_path,
                output_path=output_path,
                path=path,
                args=args,
                mirror_variant=mirror_variant,
                status="upgraded_existing_source",
            )
        raise ValueError(f"已有 source 不满足 {args.schema}: {output_path}，请使用 --overwrite 或 --rebuild_manifest。")
    elif output_path.exists() and not args.overwrite:
        raise FileExistsError(f"输出文件已存在: {output_path}，请使用 --overwrite 或 --skip_existing。")

    reused_record = try_reuse_existing_realtime_source(
        path=path,
        output_path=output_path,
        args=args,
        mirror_variant=mirror_variant,
    )
    if reused_record is not None:
        return reused_record

    source = load_motion_source(path=path, amass_dir=args.amass_dir, target_fps=args.target_fps)
    if mirror_variant:
        source = mirror_motion_source(source)
    smpl_motion = run_smpl_forward(source=source, model_cache=model_cache, batch_size=args.batch_size)
    body_fbx_rest = resolve_body_fbx_rest_for_schema(args)
    features = build_realtime_pose_features(
        smpl_motion,
        schema_name=str(getattr(args, "schema", DEFAULT_REALTIME_POSE_SCHEMA_NAME)),
        target_fps=float(args.target_fps),
        body_fbx_rest=body_fbx_rest,
    )
    save_realtime_pose_motion(
        output_path=output_path,
        features=features,
        source=source,
        target_fps=args.target_fps,
        schema_name=str(getattr(args, "schema", DEFAULT_REALTIME_POSE_SCHEMA_NAME)),
        args=args,
    )
    return record_for_output(
        path=path,
        output_path=output_path,
        args=args,
        mirror_variant=mirror_variant,
        status="converted",
    )


def failed_record_for_exception(
    path: Path,
    args: argparse.Namespace,
    mirror_variant: bool,
    exc: Exception,
) -> dict[str, Any]:
    source_relative_path = path.relative_to(args.amass_dir)
    if mirror_variant:
        source_relative_path = Path(MIRROR_DIR_NAME) / source_relative_path
    schema = get_schema_spec(str(getattr(args, "schema", DEFAULT_REALTIME_POSE_SCHEMA_NAME)))
    provenance = build_path_provenance(raw_relative_path=path.relative_to(args.amass_dir), args=args, schema=schema)
    return {
        "status": "failed",
        "schema_name": schema.name,
        "schema_canonical_name": str(schema.canonical_name),
        "pose_representation": schema.pose_representation,
        "root_y_policy": schema.root_y_policy,
        "pelvis_height_mode": schema.pelvis_height_mode,
        "source_path": str(path),
        "source_relative_path": str(source_relative_path),
        "stablemotion_split_key": str(source_relative_path.with_suffix(".npy")).replace("\\", "/"),
        "is_mirrored": bool(mirror_variant),
        "error": repr(exc),
        **provenance,
    }


def convert_one_motion_safely(
    path: Path,
    args: argparse.Namespace,
    model_cache: SmplModelCache,
    mirror_variant: bool = False,
) -> dict[str, Any]:
    try:
        return convert_one_motion(
            path=path,
            args=args,
            model_cache=model_cache,
            mirror_variant=mirror_variant,
        )
    except Exception as exc:
        return failed_record_for_exception(
            path=path,
            args=args,
            mirror_variant=mirror_variant,
            exc=exc,
        )


def iter_conversion_records(
    args: argparse.Namespace,
    work_items: list[tuple[Path, bool]],
):
    progress = tqdm(
        total=len(work_items),
        desc=f"Converting AMASS to {args.schema}",
    )
    try:
        if int(args.num_workers) <= 1:
            model_cache = SmplModelCache(model_dir=args.smpl_model_dir)
            for path, mirror_variant in work_items:
                yield convert_one_motion_safely(
                    path=path,
                    args=args,
                    model_cache=model_cache,
                    mirror_variant=mirror_variant,
                )
                progress.update(1)
            return

        worker_args = copy_args_for_worker(args)
        with ProcessPoolExecutor(
            max_workers=int(args.num_workers),
            initializer=init_conversion_worker,
            initargs=(worker_args, int(args.worker_torch_threads)),
        ) as executor:
            for record in executor.map(convert_work_item_in_worker, work_items, chunksize=1):
                yield record
                progress.update(1)
    finally:
        progress.close()


def main(argv: list[str] | None = None) -> dict[str, int]:
    args = resolve_converter_paths(parse_args(argv))
    validate_converter_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    if (args.overwrite or args.rebuild_manifest) and manifest_path.exists():
        manifest_path.unlink()

    motion_files = iter_amass_motion_files(args.amass_dir)
    if args.limit:
        motion_files = motion_files[: args.limit]
    work_items = build_conversion_work_items(args=args, motion_files=motion_files)

    converted = reused = skipped = failed = 0
    failed_records: list[dict[str, Any]] = []
    for record in iter_conversion_records(args=args, work_items=work_items):
        if record["status"] == "converted":
            converted += 1
        elif record["status"] in {"reused_source", "upgraded_existing_source"}:
            reused += 1
        elif record["status"] == "failed":
            failed += 1
            failed_records.append(record)
        else:
            skipped += 1
        write_manifest_record(manifest_path, record)

    print(
        f"完成 AMASS -> {args.schema} 转换: converted={converted}, "
        f"reused={reused}, skipped={skipped}, failed={failed}, manifest={manifest_path}"
    )
    if failed_records and not args.allow_partial:
        preview = "; ".join(
            f"{record['source_relative_path']}: {record['error']}"
            for record in failed_records[:3]
        )
        raise RuntimeError(
            f"AMASS 转换存在 {len(failed_records)} 个失败样本，默认停止以避免下游使用部分数据。"
            f"示例: {preview}。如确认可接受部分数据，请添加 --allow_partial。"
        )
    return {"converted": converted, "reused": reused, "skipped": skipped, "failed": failed}


if __name__ == "__main__":
    main()
