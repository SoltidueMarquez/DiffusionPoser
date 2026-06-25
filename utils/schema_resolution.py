from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence

from data_loaders.sensor_masking import DEFAULT_REALTIME_POSE_SCHEMA_NAME


def resolve_runtime_schema(
    cli_schema: str | None,
    checkpoint_args: Mapping[str, object] | None,
    *,
    cli_schema_explicit: bool = False,
) -> str:
    checkpoint_schema = _checkpoint_exact_schema(checkpoint_args)
    cli_schema_value = _non_empty_string(cli_schema)
    if checkpoint_schema is not None:
        if cli_schema_explicit and cli_schema_value is not None and cli_schema_value != checkpoint_schema:
            raise ValueError(
                f"checkpoint schema {checkpoint_schema!r} 与显式 CLI schema {cli_schema_value!r} 不一致；"
                "不允许只因 canonical 相同而混用 exact schema。"
            )
        return checkpoint_schema
    if cli_schema_value is not None:
        return cli_schema_value
    return DEFAULT_REALTIME_POSE_SCHEMA_NAME


def has_explicit_schema_arg(argv: Sequence[str] | None) -> bool:
    values = sys.argv[1:] if argv is None else list(argv)
    return any(value == "--schema" or value.startswith("--schema=") for value in values)


def _checkpoint_exact_schema(checkpoint_args: Mapping[str, object] | None) -> str | None:
    if not checkpoint_args:
        return None
    schema = _non_empty_string(checkpoint_args.get("schema"))
    schema_name = _non_empty_string(checkpoint_args.get("schema_name"))
    if schema is not None and schema_name is not None and schema != schema_name:
        raise ValueError(
            f"checkpoint args schema={schema!r} 与 schema_name={schema_name!r} 不一致；"
            "必须使用同一个 exact schema。"
        )
    return schema or schema_name


def _non_empty_string(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
