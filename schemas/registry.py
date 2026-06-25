from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from schemas.base import SchemaAdapter, SchemaSpec


DEFAULT_SCHEMA_NAME = "realtime_pose_stationary5_v1"
LEGACY_BODY_FBX_LOCAL_ROOT_Y0_SCHEMA_NAME = "realtime_pose_body_fbx_local_root_y0_v1"

POSE_REPRESENTATION_BODY_FBX_LOCAL_DELTA_6D = "body_fbx_local_delta_6d"
BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY = "body_pose_body_fbx_local_delta_6d"
ROOT_Y_POLICY_FIXED_ZERO = "fixed_zero"
PELVIS_HEIGHT_MODE_PELVIS_LOCAL_OFFSET_Y = "pelvis_local_offset_y"

BODY_POSE_START = 0
ROOT_YAW_DELTA_START = 144
ROOT_DELTA_XZ_START = 146
PELVIS_HEIGHT_START = 148
STATIONARY_PROB_START = 149
TRACKER_POS_REF_START = 154
TRACKER_ROT_REF_START = 172
SENSOR_VALID_START = 208

FEATURE_DIM = 214
TARGET_DIM = 154

_ADAPTERS: dict[str, SchemaAdapter] = {}


def register_schema(adapter: SchemaAdapter) -> SchemaAdapter:
    name = adapter.spec.name
    if not name:
        raise ValueError("schema adapter 必须提供非空 spec.name。")
    if name in _ADAPTERS:
        raise ValueError(f"schema 已注册: {name}")
    _ADAPTERS[name] = adapter
    return adapter


def get_schema_adapter(schema_name: str | None) -> SchemaAdapter:
    name = str(schema_name or DEFAULT_SCHEMA_NAME)
    try:
        return _ADAPTERS[name]
    except KeyError as exc:
        choices = tuple(_ADAPTERS.keys())
        raise ValueError(f"未知 schema: {name}，可选值为 {choices}") from exc


def get_schema_spec(schema_name: str | None) -> SchemaSpec:
    return get_schema_adapter(schema_name).spec


def list_schema_names(trainable_only: bool = False, exportable_only: bool = False) -> list[str]:
    names: list[str] = []
    for name, adapter in _ADAPTERS.items():
        spec = adapter.spec
        if trainable_only and not spec.trainable:
            continue
        if exportable_only and not spec.exportable:
            continue
        names.append(name)
    return names


def get_default_schema_name() -> str:
    return DEFAULT_SCHEMA_NAME


@dataclass(frozen=True)
class _Stationary5Adapter:
    spec: SchemaSpec

    def validate_source(self, payload: Mapping[str, Any]) -> None:
        self._validate_payload_schema_name(payload)

    def validate_task(self, payload: Mapping[str, Any]) -> None:
        self._validate_payload_schema_name(payload)

    def build_inpaint_mask(self, seq_len: int | None = None) -> np.ndarray:
        actual_seq_len = self.spec.seq_len if seq_len is None else int(seq_len)
        if actual_seq_len != self.spec.seq_len:
            raise ValueError(f"{self.spec.name} 固定使用 {self.spec.seq_len} 帧窗口。")
        mask = np.zeros((actual_seq_len, self.spec.feature_dim), dtype=bool)
        mask[self.spec.target_start, self.spec.target_slice()] = True
        return mask

    def build_unity_feature_schema(self) -> Mapping[str, Any]:
        return {
            "schema_name": self.spec.name,
            "canonical_name": self.spec.canonical_name,
            "feature_dim": self.spec.feature_dim,
            "target_dim": self.spec.target_dim,
            "seq_len": self.spec.seq_len,
            "target_start": self.spec.target_start,
            "target_length": self.spec.target_length,
            "pose_representation": self.spec.pose_representation,
            "root_y_policy": self.spec.root_y_policy,
            "pelvis_height_mode": self.spec.pelvis_height_mode,
        }

    def _validate_payload_schema_name(self, payload: Mapping[str, Any]) -> None:
        value = payload.get("schema_name")
        if value is None:
            return
        if str(value) != self.spec.name:
            raise ValueError(f"schema_name={value!r} 与 adapter spec.name={self.spec.name!r} 不一致。")


def _make_stationary5_spec(name: str, task_format: str, one_line: str) -> SchemaSpec:
    return SchemaSpec(
        name=name,
        canonical_name=DEFAULT_SCHEMA_NAME,
        one_line=one_line,
        task_format=task_format,
        feature_dim=FEATURE_DIM,
        target_dim=TARGET_DIM,
        body_pose_start=BODY_POSE_START,
        root_yaw_delta_start=ROOT_YAW_DELTA_START,
        root_delta_xz_start=ROOT_DELTA_XZ_START,
        root_height_start=PELVIS_HEIGHT_START,
        stationary_prob_start=STATIONARY_PROB_START,
        tracker_pos_ref_start=TRACKER_POS_REF_START,
        tracker_rot_ref_start=TRACKER_ROT_REF_START,
        sensor_valid_start=SENSOR_VALID_START,
        pose_representation=POSE_REPRESENTATION_BODY_FBX_LOCAL_DELTA_6D,
        body_pose_key=BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY,
        root_heading_delta_key="root_heading_delta_sincos",
        pelvis_height_key="pelvis_height",
        root_y_policy=ROOT_Y_POLICY_FIXED_ZERO,
        pelvis_height_mode=PELVIS_HEIGHT_MODE_PELVIS_LOCAL_OFFSET_Y,
    )


register_schema(
    _Stationary5Adapter(
        _make_stationary5_spec(
            name=DEFAULT_SCHEMA_NAME,
            task_format="materialized_realtime_pose_stationary5_v1",
            one_line="Realtime pose stationary5 canonical schema.",
        )
    )
)
register_schema(
    _Stationary5Adapter(
        _make_stationary5_spec(
            name=LEGACY_BODY_FBX_LOCAL_ROOT_Y0_SCHEMA_NAME,
            task_format="materialized_realtime_pose_body_fbx_local_root_y0_v1",
            one_line="Legacy exact name for the stationary5 canonical schema.",
        )
    )
)
