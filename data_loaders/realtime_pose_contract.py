from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from data_loaders.sensor_masking import (
    BODY_POSE_DIM,
    LEGACY_BODY_POSE_PARENT_KEY,
    POSE_REPRESENTATION_BODY_FBX_LOCAL_DELTA_6D,
    POSE_REPRESENTATION_KEY,
    ROOT_DELTA_XZ_DIM,
    ROOT_HEIGHT_DIM,
    ROOT_Y_POLICY_FIXED_ZERO,
    ROOT_YAW_DELTA_DIM,
    STATIONARY_PROB_DIM,
    TRACKER_COUNT,
    SchemaSpec,
    get_schema_spec,
    scalar_string,
    validate_pose_representation,
)
from data_loaders.stationary_label_config import STATIONARY_LABEL_METADATA_FIELDS, stationary_label_metadata
from data_loaders.tracker_codec import REFERENCE_POLICY_VERSION, TRACKER_CODEC_VERSION
from schemas.realtime_pose_stationary5_v1.contract import RESOLVER_CONTRACT_VERSION


FEATURE_CONTRACT_VERSION = 2
JOINT_MAPPING_VERSION = "smpl24_tracker6_v1"
COORDINATE_CONVENTION_VERSION = "realtime_pose_y_up_xright_zforward_v1"
RESOLVER_CONTEXT_FRAMES = 32
TRACKER_SPACE_SYNTHETIC_JOINT_WORLD = "synthetic_joint_world"
TRACKER_SPACE_CALIBRATED_JOINT_WORLD = "calibrated_joint_world"
ALLOWED_TRACKER_SPACES = {
    TRACKER_SPACE_SYNTHETIC_JOINT_WORLD,
    TRACKER_SPACE_CALIBRATED_JOINT_WORLD,
}
RUNTIME_CONTRACT_METADATA_FIELDS = (
    "feature_contract_version",
    "tracker_space",
    "calibration_version",
    "joint_mapping_version",
    "coordinate_convention_version",
    "tracker_codec_version",
    "reference_policy_version",
    "resolver_contract_version",
)


SOURCE_STATIC_FIELDS = {
    POSE_REPRESENTATION_KEY,
    "root_pos_world",
    "root_yaw",
    "tracker_pos_world",
    "tracker_rot_world_6d",
    "joints_world",
    "joint_offsets_parent",
}
SCHEMA_METADATA_FIELDS = (
    "schema_name",
    POSE_REPRESENTATION_KEY,
    "root_y_policy",
    "pelvis_height_mode",
    *RUNTIME_CONTRACT_METADATA_FIELDS,
)
ROOT_Y0_ATOL = 1e-6


def validate_schema_metadata(metadata: Mapping[str, Any], schema: SchemaSpec, source: str) -> None:
    base_fields = ("schema_name", POSE_REPRESENTATION_KEY, "root_y_policy", "pelvis_height_mode")
    missing = [key for key in base_fields if key not in metadata]
    if missing:
        raise ValueError(f"{source} metadata 缺少字段: {missing}")

    schema_name = scalar_string(metadata["schema_name"], "schema_name")
    if schema_name != schema.name:
        raise ValueError(f"{source} schema_name={schema_name!r}, expected {schema.name!r}.")
    validate_pose_representation(metadata[POSE_REPRESENTATION_KEY], schema_name=schema.name, source=source)

    root_y_policy = scalar_string(metadata["root_y_policy"], "root_y_policy")
    if root_y_policy != schema.root_y_policy:
        raise ValueError(f"{source} root_y_policy={root_y_policy!r}, expected {schema.root_y_policy!r}.")
    pelvis_height_mode = scalar_string(metadata["pelvis_height_mode"], "pelvis_height_mode")
    if pelvis_height_mode != schema.pelvis_height_mode:
        raise ValueError(
            f"{source} pelvis_height_mode={pelvis_height_mode!r}, expected {schema.pelvis_height_mode!r}."
        )
    missing_runtime = [key for key in RUNTIME_CONTRACT_METADATA_FIELDS if key not in metadata]
    if missing_runtime:
        raise ValueError(f"{source} metadata 缺少 v2 runtime 字段: {missing_runtime}")
    feature_contract_version = _scalar_int(metadata["feature_contract_version"], "feature_contract_version")
    if feature_contract_version != FEATURE_CONTRACT_VERSION:
        raise ValueError(
            f"{source} feature_contract_version={feature_contract_version}, expected {FEATURE_CONTRACT_VERSION}."
        )
    tracker_space = scalar_string(metadata["tracker_space"], "tracker_space")
    if tracker_space not in ALLOWED_TRACKER_SPACES:
        raise ValueError(f"{source} tracker_space={tracker_space!r} 未标定或不受支持。")
    expected_values = {
        "joint_mapping_version": JOINT_MAPPING_VERSION,
        "coordinate_convention_version": COORDINATE_CONVENTION_VERSION,
        "tracker_codec_version": TRACKER_CODEC_VERSION,
        "reference_policy_version": REFERENCE_POLICY_VERSION,
        "resolver_contract_version": RESOLVER_CONTRACT_VERSION,
    }
    for key, expected in expected_values.items():
        actual = scalar_string(metadata[key], key)
        if actual != expected:
            raise ValueError(f"{source} {key}={actual!r}, expected {expected!r}.")
    if not scalar_string(metadata["calibration_version"], "calibration_version").strip():
        raise ValueError(f"{source} calibration_version 不能为空。")


def runtime_contract_metadata(
    *,
    tracker_space: str = TRACKER_SPACE_SYNTHETIC_JOINT_WORLD,
    calibration_version: str = "synthetic_identity_v1",
) -> dict[str, Any]:
    if tracker_space not in ALLOWED_TRACKER_SPACES:
        raise ValueError(f"不支持 tracker_space={tracker_space!r}")
    return {
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "tracker_space": tracker_space,
        "calibration_version": str(calibration_version),
        "joint_mapping_version": JOINT_MAPPING_VERSION,
        "coordinate_convention_version": COORDINATE_CONVENTION_VERSION,
        "tracker_codec_version": TRACKER_CODEC_VERSION,
        "reference_policy_version": REFERENCE_POLICY_VERSION,
        "resolver_contract_version": RESOLVER_CONTRACT_VERSION,
    }


def validate_stationary_label_metadata(metadata: Mapping[str, Any], source: str) -> None:
    """校验因果 stationary v2 配置，防止旧的居中标签数据混入训练链路。"""

    expected = stationary_label_metadata()
    missing = [key for key in STATIONARY_LABEL_METADATA_FIELDS if key not in metadata]
    if missing:
        raise ValueError(f"{source} stationary label metadata 缺少字段: {missing}")
    for key, expected_value in expected.items():
        value = metadata[key]
        if isinstance(expected_value, str):
            actual = scalar_string(value, key)
            if actual != expected_value:
                raise ValueError(f"{source} {key}={actual!r}, expected {expected_value!r}.")
        elif isinstance(expected_value, int):
            actual = _scalar_int(value, key)
            if actual != int(expected_value):
                raise ValueError(f"{source} {key}={actual!r}, expected {int(expected_value)!r}.")
        else:
            actual = _scalar_float(value, key)
            if not np.isclose(actual, float(expected_value), rtol=0.0, atol=1e-8):
                raise ValueError(f"{source} {key}={actual!r}, expected {float(expected_value)!r}.")


def validate_root_y0_invariants(arrays: Mapping[str, Any], schema: SchemaSpec, source: str) -> None:
    if schema.root_y_policy != ROOT_Y_POLICY_FIXED_ZERO:
        return
    root_pos_world = np.asarray(arrays["root_pos_world"])
    if root_pos_world.ndim != 2 or root_pos_world.shape[1] != 3:
        raise ValueError(f"{source} root_pos_world 应为 [T,3]，实际为 {tuple(root_pos_world.shape)}")
    if not np.allclose(root_pos_world[:, 1], 0.0, atol=ROOT_Y0_ATOL):
        max_abs = float(np.max(np.abs(root_pos_world[:, 1])))
        raise ValueError(f"{source} root-y0 schema 要求 root_pos_world[:,1] 全为 0，最大绝对值为 {max_abs}.")

    pelvis_height = np.asarray(arrays[schema.pelvis_height_key])
    joints_world = np.asarray(arrays["joints_world"])
    if pelvis_height.ndim != 2 or pelvis_height.shape[1] != 1:
        raise ValueError(f"{source} {schema.pelvis_height_key} 应为 [T,1]，实际为 {tuple(pelvis_height.shape)}")
    if joints_world.ndim != 3 or joints_world.shape[1:] != (24, 3):
        raise ValueError(f"{source} joints_world 应为 [T,24,3]，实际为 {tuple(joints_world.shape)}")
    if not np.allclose(pelvis_height[:, 0], joints_world[:, 0, 1], atol=ROOT_Y0_ATOL):
        max_abs = float(np.max(np.abs(pelvis_height[:, 0] - joints_world[:, 0, 1])))
        raise ValueError(
            f"{source} root-y0 schema 要求 {schema.pelvis_height_key} 等于 joints_world[:,0,1]，最大误差为 {max_abs}."
        )


def validate_realtime_source_contract(data: Mapping[str, Any] | Any, schema: SchemaSpec, source: str) -> None:
    keys = _payload_keys(data)
    if LEGACY_BODY_POSE_PARENT_KEY in keys:
        raise ValueError(f"{source} contains legacy {LEGACY_BODY_POSE_PARENT_KEY}; regenerate source data.")

    required = required_realtime_source_fields(schema)
    missing = sorted(required.difference(keys))
    if missing:
        raise KeyError(f"{source} 缺少 {schema.name} 源字段: {missing}")

    metadata = load_source_metadata(data, source=source)
    validate_schema_metadata(metadata, schema=schema, source=source)
    if schema.supports_stationary_prob:
        validate_stationary_label_metadata(metadata, source=source)
    validate_pose_representation(_payload_value(data, POSE_REPRESENTATION_KEY), schema_name=schema.name, source=source)

    frame_count = int(np.asarray(_payload_value(data, schema.body_pose_key)).shape[0])
    _validate_shapes(
        payload=data,
        expected=expected_realtime_source_shapes(schema=schema, frame_count=frame_count),
        source=source,
    )
    validate_root_y0_invariants(data, schema=schema, source=source)


def required_realtime_source_fields(schema: SchemaSpec | str) -> set[str]:
    schema = get_schema_spec(schema if isinstance(schema, str) else schema.name)
    required = set(SOURCE_STATIC_FIELDS)
    required.add(schema.body_pose_key)
    required.add(schema.root_heading_delta_key)
    if schema.supports_root_motion:
        required.update({"root_delta_xz_ref", schema.pelvis_height_key})
    if schema.supports_stationary_prob:
        required.add("stationary_prob_5")
    if schema.pose_representation == POSE_REPRESENTATION_BODY_FBX_LOCAL_DELTA_6D:
        required.add("joint_rest_local_rotations_6d")
    return required


def expected_realtime_source_shapes(schema: SchemaSpec, frame_count: int) -> dict[str, tuple[int, ...]]:
    shapes = {
        schema.body_pose_key: (frame_count, BODY_POSE_DIM),
        "root_pos_world": (frame_count, 3),
        "root_yaw": (frame_count,),
        schema.root_heading_delta_key: (frame_count, ROOT_YAW_DELTA_DIM),
        "tracker_pos_world": (frame_count, TRACKER_COUNT, 3),
        "tracker_rot_world_6d": (frame_count, TRACKER_COUNT, 6),
        "joints_world": (frame_count, 24, 3),
        "joint_offsets_parent": (24, 3),
    }
    if schema.supports_root_motion:
        shapes["root_delta_xz_ref"] = (frame_count, ROOT_DELTA_XZ_DIM)
        shapes[schema.pelvis_height_key] = (frame_count, ROOT_HEIGHT_DIM)
    if schema.supports_stationary_prob:
        shapes["stationary_prob_5"] = (frame_count, STATIONARY_PROB_DIM)
    if schema.pose_representation == POSE_REPRESENTATION_BODY_FBX_LOCAL_DELTA_6D:
        shapes["joint_rest_local_rotations_6d"] = (24, 6)
    return shapes


def load_source_metadata(data: Mapping[str, Any] | Any, source: str) -> dict[str, Any]:
    if "metadata" not in _payload_keys(data):
        raise ValueError(f"{source} 缺少 metadata；旧 source 不能复用到当前 schema。")
    value = _payload_value(data, "metadata")
    try:
        text = str(np.asarray(value).item())
    except Exception:
        text = str(value)
    try:
        metadata = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} metadata 不是合法 JSON。") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{source} metadata 必须是 JSON object。")
    return metadata


def _payload_keys(payload: Mapping[str, Any] | Any) -> set[str]:
    if hasattr(payload, "files"):
        return set(payload.files)
    return set(payload.keys())


def _payload_value(payload: Mapping[str, Any] | Any, key: str) -> Any:
    return payload[key]


def _validate_shapes(payload: Mapping[str, Any] | Any, expected: Mapping[str, tuple[int, ...]], source: str) -> None:
    for key, shape in expected.items():
        actual = tuple(np.asarray(_payload_value(payload, key)).shape)
        if actual != shape:
            raise ValueError(f"{source} 字段 {key} 应为 {shape}，实际为 {actual}")


def _scalar_int(value: Any, name: str) -> int:
    array = np.asarray(value)
    if array.shape == ():
        return int(array.item())
    if array.size == 1:
        return int(array.reshape(()).item())
    raise ValueError(f"{name} must be a scalar int, got shape={array.shape}")


def _scalar_float(value: Any, name: str) -> float:
    array = np.asarray(value)
    if array.shape == ():
        return float(array.item())
    if array.size == 1:
        return float(array.reshape(()).item())
    raise ValueError(f"{name} must be a scalar float, got shape={array.shape}")
