from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from schemas.base import SchemaSpec
from schemas.realtime_pose_stationary5_v1.contract import (
    LEGACY_SCHEMA_NAME,
    LEGACY_TASK_FORMAT,
    SCHEMA_NAME,
    TASK_FORMAT,
    build_stationary5_spec,
)
from schemas.realtime_pose_stationary5_v1.unity import build_stationary5_unity_feature_schema


@dataclass(frozen=True)
class Stationary5Adapter:
    spec: SchemaSpec

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def canonical_name(self) -> str:
        return str(self.spec.canonical_name)

    def validate_source(self, payload: Mapping[str, Any]) -> None:
        self._validate_schema_metadata(payload)

    def validate_task(self, payload: Mapping[str, Any]) -> None:
        self._validate_schema_metadata(payload)

    def build_inpaint_mask(self, seq_len: int | None = None) -> np.ndarray:
        actual_seq_len = self.spec.seq_len if seq_len is None else int(seq_len)
        if actual_seq_len != self.spec.seq_len:
            raise ValueError(f"{self.spec.name} 固定使用 {self.spec.seq_len} 帧窗口，实际为 {actual_seq_len}")
        mask = np.zeros((actual_seq_len, self.spec.feature_dim), dtype=bool)
        # inpaint_mask 只监督第 61 帧的目标通道，历史条件和 tracker 观测通道保持可见。
        mask[self.spec.target_start, self.spec.target_slice()] = True
        return mask

    def build_unity_feature_schema(self) -> Mapping[str, Any]:
        return build_stationary5_unity_feature_schema(self.spec)

    def _validate_schema_metadata(self, payload: Mapping[str, Any]) -> None:
        # Task 2 只校验 schema 元信息，source/task 数组契约会在后续任务接入。
        self._expect_if_present(payload, "schema_name", self.spec.name)
        self._expect_if_present(payload, "schemaName", self.spec.name)
        self._expect_if_present(payload, "pose_representation", self.spec.pose_representation)
        self._expect_if_present(payload, "poseRepresentation", self.spec.pose_representation)
        self._expect_if_present(payload, "root_y_policy", self.spec.root_y_policy)
        self._expect_if_present(payload, "rootYPolicy", self.spec.root_y_policy)
        self._expect_if_present(payload, "pelvis_height_mode", self.spec.pelvis_height_mode)
        self._expect_if_present(payload, "pelvisHeightMode", self.spec.pelvis_height_mode)

    def _expect_if_present(self, payload: Mapping[str, Any], key: str, expected: object) -> None:
        value = payload.get(key)
        if value is not None and str(value) != str(expected):
            raise ValueError(f"{key}={value!r} 与 {self.spec.name} 期望值 {expected!r} 不一致")


def build_stationary5_adapter(
    name: str = SCHEMA_NAME,
    canonical_name: str = SCHEMA_NAME,
    task_format: str = TASK_FORMAT,
    one_line: str = "Realtime pose stationary5 canonical schema.",
) -> Stationary5Adapter:
    return Stationary5Adapter(
        build_stationary5_spec(
            name=name,
            canonical_name=canonical_name,
            task_format=task_format,
            one_line=one_line,
        )
    )


realtime_pose_stationary5_v1 = build_stationary5_adapter()
realtime_pose_body_fbx_local_root_y0_v1 = build_stationary5_adapter(
    name=LEGACY_SCHEMA_NAME,
    canonical_name=SCHEMA_NAME,
    task_format=LEGACY_TASK_FORMAT,
    one_line="Legacy exact name for the stationary5 canonical schema.",
)

STATIONARY5_ADAPTERS = (realtime_pose_stationary5_v1, realtime_pose_body_fbx_local_root_y0_v1)
