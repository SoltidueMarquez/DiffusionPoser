from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data_loaders.sensor_masking import TRACKER_FEATURE_DIM, TRACKER_PATTERN_CATEGORIES
from data_loaders.generate_realtime_pose_tasks import shard_fields
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source
from utils.run_dirs import write_latest_pointer


def write_toy_longseq_task_run(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "sources"
    task_root = tmp_path / "tasks"
    task_run = task_root / "manual_run"
    manifest_path = task_run / "test" / "sources.jsonl"
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
                "source_id": str(Path(relative_path).with_suffix(".npy")).replace("\\", "/"),
                "source_index": index,
                "source_path": str(source_path),
                "source_relative_path": relative_path,
                "source_frames": frame_count,
                "target_fps": 30.0,
                "is_mirrored": is_mirrored,
            }
            file.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    (manifest_path.parent / "task_store.json").write_text(
        json.dumps(
            {
                "generation_plan_hash": "toy_plan_hash",
                "split": "test",
                "sample_count": len(records),
                "source_count": len(records),
                "two_point_phase_counts": {
                    "dropout": (len(records) + 1) // 2,
                    "reconnect": len(records) // 2,
                },
                "config_names": list(TRACKER_PATTERN_CATEGORIES),
                "tracker_feature_dim": TRACKER_FEATURE_DIM,
                "schema_fields": sorted(shard_fields()),
                "shards": [],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    write_latest_pointer(
        root_dir=task_root,
        kind="tasks",
        output_dir=task_run,
        metadata={"splits": ["test"], "counts": {"test": len(records)}},
    )
    return task_root, task_run


def write_toy_source_npz(source_path: Path, frame_count: int) -> None:
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source = build_toy_realtime_source(frame_count=frame_count)
    np.savez(source_path, **source)
