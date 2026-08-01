from __future__ import annotations

import argparse
import json

from data_loaders.build_realtime_longseq_eval_set import (
    build_realtime_longseq_eval_set,
    read_longseq_manifest,
    resolve_manifest_source_path,
)
from data_loaders.generate_realtime_pose_tasks import load_realtime_source
from data_loaders.sensor_masking import BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY
from tests.smoke.longseq_eval_fixtures import write_toy_longseq_task_run
from utils.run_dirs import read_latest_pointer


def test_longseq_eval_builder_selects_test_non_mirror_long_sources(tmp_path):
    task_root, _task_run = write_toy_longseq_task_run(tmp_path)
    output_root = tmp_path / "longseq_eval"

    output_dir = build_realtime_longseq_eval_set(
        argparse.Namespace(
            task_dir=str(task_root),
            task_run="latest",
            output_root=str(output_root),
            run_name="v1_test_stress_long_seed10",
            preset="stress_long",
            split="test",
            min_frames=60,
            include_mirror=False,
            overwrite=True,
        )
    )

    entries = read_longseq_manifest(output_dir)
    assert [entry["source_relative_path"] for entry in entries] == [
        "KIT/442/long_b_poses.npz",
        "CMU/55/long_a_poses.npz",
    ]
    assert {entry["preset"] for entry in entries} == {"stress_long"}
    assert {entry["is_mirrored"] for entry in entries} == {False}
    assert all(entry["split"] == "test" for entry in entries)

    for entry in entries:
        copied_path = resolve_manifest_source_path(output_dir, entry)
        assert copied_path.exists()
        source = load_realtime_source(copied_path)
        assert source[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY].shape[0] == entry["num_frames"]

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["sequence_count"] == 2
    assert summary["total_frames"] == 145
    assert read_latest_pointer(output_root, "longseq_eval") == output_dir
