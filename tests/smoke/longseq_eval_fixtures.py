from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data_loaders.generate_realtime_pose_tasks import write_task_output_marker
from data_loaders.realtime_pose_contract import runtime_contract_metadata
from data_loaders.sensor_masking import (
    REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_LENGTH,
    REALTIME_POSE_TARGET_START,
    TASK_MODE_REALTIME_POSE,
    get_schema_spec,
)
from data_loaders.stationary_label_config import stationary_label_metadata
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source
from utils.run_dirs import write_latest_pointer


def write_toy_longseq_task_run(tmp_path: Path) -> tuple[Path, Path]:
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    source_root = tmp_path / "sources"
    task_root = tmp_path / "tasks"
    task_run = task_root / "manual_run"
    manifest_path = task_run / "test" / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    records = [
        ("CMU/55/long_a_poses.npz", 70, False),
        ("CMU/55/long_a_poses.npz", 70, False),
        ("KIT/442/long_b_poses.npz", 75, False),
        ("M/CMU/55/long_a_poses.npz", 70, True),
    ]
    source_manifest_entries = {}
    with manifest_path.open("w", encoding="utf-8", newline="\n") as file:
        for index, (relative_path, frame_count, is_mirrored) in enumerate(records):
            source_path = source_root / relative_path
            write_toy_source_npz(source_path=source_path, frame_count=frame_count)
            entry = {
                "task_id": f"task_{index:03d}",
                "task_format": schema.task_format,
                "split": "test",
                "source_path": str(source_path),
                "source_relative_path": relative_path,
                "stablemotion_split_key": str(Path(relative_path).with_suffix(".npy")).replace("\\", "/"),
                "source_frames": frame_count,
                "samples_per_source": 1,
                "sampling_seed": 10,
                "max_rollout_steps": 1,
                "seq_len": REALTIME_POSE_SEQ_LEN,
                "feature_dim": schema.feature_dim,
                "target_start": REALTIME_POSE_TARGET_START,
                "target_length": REALTIME_POSE_TARGET_LENGTH,
                "schema_name": schema.name,
                "schema_canonical_name": str(schema.canonical_name),
                "pose_representation": schema.pose_representation,
                "root_y_policy": schema.root_y_policy,
                "pelvis_height_mode": schema.pelvis_height_mode,
                "task_mode": TASK_MODE_REALTIME_POSE,
                "is_mirrored": is_mirrored,
                "tracker_pattern": "full_six",
                "tracker_pattern_detail": {"category": "full_six", "sensor_valid": [True] * 6},
            }
            entry.update(runtime_contract_metadata())
            entry.update(stationary_label_metadata())
            file.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
            source_manifest_entries[relative_path] = {
                **entry,
                "status": "converted",
                "frames": frame_count,
                "output_path": relative_path,
            }

    source_root.mkdir(parents=True, exist_ok=True)
    with (source_root / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for entry in source_manifest_entries.values():
            file.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    write_task_output_marker(
        source_dir=source_root,
        output_dir=task_run,
        split_dir=None,
        schema_name=schema.name,
        generated_root=tmp_path,
        source_set_name="toy_longseq",
        task_set_name="toy_longseq",
    )

    write_latest_pointer(
        root_dir=task_root,
        kind="tasks",
        output_dir=task_run,
        metadata={"schema_name": schema.name, "splits": ["test"], "counts": {"test": len(records)}},
    )
    return task_root, task_run


def write_toy_source_npz(source_path: Path, frame_count: int) -> None:
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source = build_toy_realtime_source(frame_count=frame_count)
    np.savez(source_path, **source)
