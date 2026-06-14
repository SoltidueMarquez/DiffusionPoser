from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data_loaders.sensor_masking import REALTIME_POSE_SCHEMA_NAME, get_schema_spec
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
        ("ACCAD/short_poses.npz", 30, False),
    ]
    with manifest_path.open("w", encoding="utf-8", newline="\n") as file:
        for index, (relative_path, frame_count, is_mirrored) in enumerate(records):
            source_path = source_root / relative_path
            write_toy_source_npz(source_path=source_path, frame_count=frame_count)
            entry = {
                "task_id": f"task_{index:03d}",
                "task_path": f"tasks/task_{index:03d}.npz",
                "split": "test",
                "source_path": str(source_path),
                "source_relative_path": relative_path,
                "stablemotion_split_key": str(Path(relative_path).with_suffix(".npy")).replace("\\", "/"),
                "source_frames": frame_count,
                "schema_name": schema.name,
                "pose_representation": schema.pose_representation,
                "root_y_policy": schema.root_y_policy,
                "pelvis_height_mode": schema.pelvis_height_mode,
                "is_mirrored": is_mirrored,
            }
            file.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

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
