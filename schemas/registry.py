from __future__ import annotations

from schemas.base import SchemaAdapter, SchemaSpec
from schemas.realtime_pose_stationary5_v1.adapter import STATIONARY5_ADAPTERS
from schemas.realtime_pose_stationary5_v1.contract import LEGACY_SCHEMA_NAME, SCHEMA_NAME


DEFAULT_SCHEMA_NAME = SCHEMA_NAME
LEGACY_BODY_FBX_LOCAL_ROOT_Y0_SCHEMA_NAME = LEGACY_SCHEMA_NAME

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


for _adapter in STATIONARY5_ADAPTERS:
    register_schema(_adapter)
