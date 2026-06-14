from __future__ import annotations

import argparse
import json

import numpy as np

from data_loaders.build_realtime_longseq_eval_set import build_realtime_longseq_eval_set
from data_loaders.longseq_eval_dropout import LongseqDropoutConfig
from data_loaders.sensor_masking import REALTIME_POSE_SCHEMA_NAME, TRACKER_COUNT, get_schema_spec
from export.write_longseq_eval_unity_replays import export_longseq_eval_unity_replays
from tests.smoke.longseq_eval_fixtures import write_toy_longseq_task_run


def test_longseq_eval_unity_replay_export_writes_one_json_per_sequence(tmp_path):
    task_root, _task_run = write_toy_longseq_task_run(tmp_path)
    eval_set_dir = build_realtime_longseq_eval_set(
        argparse.Namespace(
            task_dir=str(task_root),
            task_run="latest",
            output_root=str(tmp_path / "longseq_eval"),
            run_name="stress",
            preset="stress_long",
            split="test",
            min_frames=60,
            include_mirror=False,
            schema=REALTIME_POSE_SCHEMA_NAME,
            overwrite=True,
        )
    )

    summary = export_longseq_eval_unity_replays(eval_set_dir=eval_set_dir)

    assert summary["file_count"] == 2
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    for item in summary["files"]:
        payload = json.loads(open(item["output_json"], "r", encoding="utf-8").read())
        assert payload["schemaName"] == schema.name
        assert payload["frameCount"] == item["num_frames"]
        assert payload["targetFeatureLength"] == schema.target_dim
        assert payload["trackerCount"] == TRACKER_COUNT
    assert json.loads(open(summary["summary_path"], "r", encoding="utf-8").read())["file_count"] == 2


def test_longseq_eval_unity_replay_export_can_apply_dropout(tmp_path):
    task_root, _task_run = write_toy_longseq_task_run(tmp_path)
    eval_set_dir = build_realtime_longseq_eval_set(
        argparse.Namespace(
            task_dir=str(task_root),
            task_run="latest",
            output_root=str(tmp_path / "longseq_eval"),
            run_name="stress",
            preset="stress_long",
            split="test",
            min_frames=60,
            include_mirror=False,
            schema=REALTIME_POSE_SCHEMA_NAME,
            overwrite=True,
        )
    )

    summary = export_longseq_eval_unity_replays(
        eval_set_dir=eval_set_dir,
        dropout_config=LongseqDropoutConfig(
            preset="tracker_mask_train",
            tracker_mask_policy="fixed_categories",
            tracker_mask_categories=("upper-body",),
        ),
    )

    payload = json.loads(open(summary["files"][0]["output_json"], "r", encoding="utf-8").read())
    valid = np.asarray(payload["sensorValid"], dtype=np.int32).reshape(payload["frameCount"], TRACKER_COUNT)
    assert not valid.all()
    assert valid[:, 3].all()
    assert valid.sum(axis=1).min() >= 3
    assert summary["files"][0]["valid_tracker_ratio"] < 1.0
