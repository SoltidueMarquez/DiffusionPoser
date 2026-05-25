from __future__ import annotations

import json

import numpy as np

from data_loaders.realtime_pose_dataset import encode_realtime_pose_features
from data_loaders.sensor_masking import (
    REALTIME_POSE_INPUT_DIM,
    REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_START,
)
from eval.evaluate_realtime_pose_rollout import main as evaluate_rollout_main
from sample.reconstruct_rollout import save_rollout
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source


def test_rollout_eval_writes_finite_metrics(tmp_path):
    source = build_toy_realtime_source(frame_count=REALTIME_POSE_SEQ_LEN)
    source["sensor_valid"] = np.ones((REALTIME_POSE_SEQ_LEN, 6), dtype=bool)
    features = encode_realtime_pose_features(source, schema_name=REALTIME_POSE_SCHEMA_NAME)[None]
    joints = source["joints_world"][None]
    output_path = tmp_path / "rollout_result.npz"
    save_rollout(
        output_path,
        {
            "reference_features_raw": features,
            "predicted_features_raw": features.copy(),
            "reference_joints_world": joints,
            "predicted_joints_world": joints.copy(),
            "root_yaw_reference": source["root_yaw"][None],
            "root_yaw_predicted": source["root_yaw"][None].copy(),
            "tracker_pos_ref": np.zeros((1, REALTIME_POSE_SEQ_LEN, 6, 3), dtype=np.float32),
            "sensor_valid": np.ones((1, REALTIME_POSE_SEQ_LEN, 6), dtype=np.float32),
            "metadata": np.asarray({"schema_name": REALTIME_POSE_SCHEMA_NAME}, dtype=object),
        },
    )
    output_json = tmp_path / "summary.json"
    summary = evaluate_rollout_main(["--input_dir", str(tmp_path), "--output_json", str(output_json)])
    assert summary["file_count"] == 1
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert np.isfinite(payload["summary"]["mpjpe_mean"])


def test_rollout_eval_reports_nonzero_mpjpe_for_changed_joints(tmp_path):
    source = build_toy_realtime_source(frame_count=REALTIME_POSE_SEQ_LEN)
    source["sensor_valid"] = np.ones((REALTIME_POSE_SEQ_LEN, 6), dtype=bool)
    features = encode_realtime_pose_features(source, schema_name=REALTIME_POSE_SCHEMA_NAME)[None]
    reference_joints = source["joints_world"][None]
    predicted_joints = reference_joints.copy()
    predicted_joints[:, :, 0, 0] += 0.1
    output_path = tmp_path / "rollout_result.npz"
    save_rollout(
        output_path,
        {
            "reference_features_raw": features,
            "predicted_features_raw": features.copy(),
            "reference_joints_world": reference_joints,
            "predicted_joints_world": predicted_joints,
            "root_yaw_reference": source["root_yaw"][None],
            "root_yaw_predicted": source["root_yaw"][None].copy(),
            "tracker_pos_ref": np.zeros((1, REALTIME_POSE_SEQ_LEN, 6, 3), dtype=np.float32),
            "sensor_valid": np.ones((1, REALTIME_POSE_SEQ_LEN, 6), dtype=np.float32),
            "metadata": np.asarray({"schema_name": REALTIME_POSE_SCHEMA_NAME}, dtype=object),
        },
    )
    summary = evaluate_rollout_main(["--input_dir", str(tmp_path)])
    assert summary["mpjpe_mean"] > 0.0


def test_rollout_eval_respects_eval_frame_mask(tmp_path):
    source = build_toy_realtime_source(frame_count=REALTIME_POSE_SEQ_LEN)
    source["sensor_valid"] = np.ones((REALTIME_POSE_SEQ_LEN, 6), dtype=bool)
    features = encode_realtime_pose_features(source, schema_name=REALTIME_POSE_SCHEMA_NAME)[None]
    reference_joints = source["joints_world"][None]
    predicted_joints = reference_joints.copy()
    predicted_joints[:, 0] += 100.0
    eval_frame_mask = np.zeros((1, REALTIME_POSE_SEQ_LEN), dtype=bool)
    eval_frame_mask[:, -1] = True
    output_path = tmp_path / "rollout_result.npz"
    save_rollout(
        output_path,
        {
            "reference_features_raw": features,
            "predicted_features_raw": features.copy(),
            "reference_joints_world": reference_joints,
            "predicted_joints_world": predicted_joints,
            "root_yaw_reference": source["root_yaw"][None],
            "root_yaw_predicted": source["root_yaw"][None].copy(),
            "tracker_pos_ref": np.zeros((1, REALTIME_POSE_SEQ_LEN, 6, 3), dtype=np.float32),
            "sensor_valid": np.ones((1, REALTIME_POSE_SEQ_LEN, 6), dtype=np.float32),
            "eval_frame_mask": eval_frame_mask,
            "warmup_frames": np.asarray(REALTIME_POSE_TARGET_START, dtype=np.int64),
            "metadata": np.asarray({"schema_name": REALTIME_POSE_SCHEMA_NAME}, dtype=object),
        },
    )

    summary = evaluate_rollout_main(["--input_dir", str(tmp_path)])
    assert summary["evaluated_frames"] == 1
    assert summary["warmup_frames"] == REALTIME_POSE_TARGET_START
    assert summary["mpjpe_mean"] == 0.0
