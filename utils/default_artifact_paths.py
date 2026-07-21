from __future__ import annotations

from pathlib import Path

from data_loaders.sensor_masking import DEFAULT_REALTIME_POSE_SCHEMA_NAME
from utils.artifact_paths import longseq_eval_root, normalizer_root, source_root, task_root
from utils.artifact_roots import load_artifact_roots

# 保留名称是为了让已有调用方和测试的 patch 入口保持稳定。


DEFAULT_REALTIME_POSE_SOURCE_SET_NAME = "amass_60hz"
DEFAULT_REALTIME_POSE_TASK_SET_NAME = "amass_60hz_tasks"
DEFAULT_REALTIME_POSE_NORMALIZER_NAME = "amass_60hz_train"
DEFAULT_REALTIME_POSE_LONGSEQ_EVAL_NAME = "amass_60hz_test_stress_long"


def default_amass_root(artifact_roots_config: str | Path | None = None) -> Path:
    return _load_roots(artifact_roots_config).amass_root


def default_realtime_pose_source_root(
    schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    artifact_roots_config: str | Path | None = None,
) -> Path:
    return source_root(
        _load_roots(artifact_roots_config),
        schema_name=schema_name,
        source_set_name=DEFAULT_REALTIME_POSE_SOURCE_SET_NAME,
    )


def default_realtime_pose_task_root(
    schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    artifact_roots_config: str | Path | None = None,
) -> Path:
    return task_root(
        _load_roots(artifact_roots_config),
        schema_name=schema_name,
        task_set_name=DEFAULT_REALTIME_POSE_TASK_SET_NAME,
    )


def default_realtime_pose_normalizer_root(
    schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    artifact_roots_config: str | Path | None = None,
) -> Path:
    return normalizer_root(
        _load_roots(artifact_roots_config),
        schema_name=schema_name,
        normalizer_name=DEFAULT_REALTIME_POSE_NORMALIZER_NAME,
    )


def default_realtime_pose_longseq_eval_root(
    schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    artifact_roots_config: str | Path | None = None,
) -> Path:
    return longseq_eval_root(
        _load_roots(artifact_roots_config),
        schema_name=schema_name,
        eval_set_name=DEFAULT_REALTIME_POSE_LONGSEQ_EVAL_NAME,
    )


def _load_roots(artifact_roots_config: str | Path | None):
    """未显式指定配置时沿用 loader 的无参数入口，保留稳定的默认配置语义。"""
    if artifact_roots_config is None:
        return load_artifact_roots()
    return load_artifact_roots(artifact_roots_config)
