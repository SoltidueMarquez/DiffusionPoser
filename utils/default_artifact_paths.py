from __future__ import annotations

from pathlib import Path

from data_loaders.sensor_masking import DEFAULT_REALTIME_POSE_SCHEMA_NAME
from utils.artifact_paths import longseq_eval_root, normalizer_root, source_root, task_root
from utils.data_roots import load_data_roots


DEFAULT_REALTIME_POSE_SOURCE_SET_NAME = "amass_60hz"
DEFAULT_REALTIME_POSE_TASK_SET_NAME = "amass_60hz_tasks"
DEFAULT_REALTIME_POSE_NORMALIZER_NAME = "amass_60hz_train"
DEFAULT_REALTIME_POSE_LONGSEQ_EVAL_NAME = "amass_60hz_test_stress_long"


def default_realtime_pose_source_root(schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME) -> Path:
    return source_root(
        load_data_roots(),
        schema_name=schema_name,
        source_set_name=DEFAULT_REALTIME_POSE_SOURCE_SET_NAME,
    )


def default_realtime_pose_task_root(schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME) -> Path:
    return task_root(
        load_data_roots(),
        schema_name=schema_name,
        task_set_name=DEFAULT_REALTIME_POSE_TASK_SET_NAME,
    )


def default_realtime_pose_normalizer_root(schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME) -> Path:
    return normalizer_root(
        load_data_roots(),
        schema_name=schema_name,
        normalizer_name=DEFAULT_REALTIME_POSE_NORMALIZER_NAME,
    )


def default_realtime_pose_longseq_eval_root(schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME) -> Path:
    return longseq_eval_root(
        load_data_roots(),
        schema_name=schema_name,
        eval_set_name=DEFAULT_REALTIME_POSE_LONGSEQ_EVAL_NAME,
    )
