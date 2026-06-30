from __future__ import annotations

import csv
import json

import numpy as np

from data_loaders.sensor_masking import (
    REALTIME_POSE_INPUT_DIM,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_START,
    get_schema_spec,
)
from eval.evaluate_stationary_signal_source import evaluate_directory, main


def _write_prediction_npz(path, *, feature_pred, head_pred=None):
    schema = get_schema_spec("realtime_pose_stationary5_v1")
    reference = np.zeros((1, 4, REALTIME_POSE_INPUT_DIM), dtype=np.float32)
    reconstructed = reference.copy()
    target = np.asarray(
        [
            [1, 1, 0, 0, 1],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
        ],
        dtype=np.float32,
    )
    reference[0, :, schema.stationary_prob_slice()] = target
    reconstructed[0, :, schema.stationary_prob_slice()] = np.asarray(feature_pred, dtype=np.float32)
    payload = {
        "reference_features_raw": reference,
        "reconstructed_features_raw": reconstructed,
        "reference_stationary_prob_5": target,
        "feature_stationary_prob_5": np.asarray(feature_pred, dtype=np.float32),
    }
    if head_pred is not None:
        payload["head_stationary_prob_5"] = np.asarray(head_pred, dtype=np.float32)
    np.savez(path, **payload)


def test_evaluate_directory_writes_summary_and_csv_reports(tmp_path):
    feature_pred = np.zeros((4, 5), dtype=np.float32)
    head_pred = np.ones((4, 5), dtype=np.float32)
    _write_prediction_npz(tmp_path / "clip_a.npz", feature_pred=feature_pred, head_pred=head_pred)

    output_dir = tmp_path / "report"
    result = evaluate_directory(input_dir=tmp_path, output_dir=output_dir, thresholds=(0.5,))

    assert result["summary_path"] == output_dir / "metrics_summary.json"
    assert result["per_clip_path"].exists()
    assert result["per_joint_path"].exists()
    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    assert set(summary["signals"]) == {"feature_channel", "stationary_head"}
    assert summary["signals"]["feature_channel"]["thresholds"]["0.5"]["aggregate"]["missed_lock_rate"] > 0.0
    assert summary["signals"]["stationary_head"]["thresholds"]["0.5"]["aggregate"]["false_lock_rate"] > 0.0

    with result["per_clip_path"].open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert {row["signal_source"] for row in rows} == {"feature_channel", "stationary_head"}


def test_stationary_signal_cli_main_returns_paths(tmp_path):
    _write_prediction_npz(tmp_path / "clip_a.npz", feature_pred=np.zeros((4, 5), dtype=np.float32))
    output_dir = tmp_path / "cli_report"

    result = main(["--input_dir", str(tmp_path), "--output_dir", str(output_dir), "--thresholds", "0.5,0.7"])

    assert result["summary_path"].exists()


def test_evaluate_directory_uses_target_frame_for_realtime_windows(tmp_path):
    schema = get_schema_spec("realtime_pose_stationary5_v1")
    reference = np.zeros((1, REALTIME_POSE_SEQ_LEN, REALTIME_POSE_INPUT_DIM), dtype=np.float32)
    reconstructed = reference.copy()
    reference_stationary = np.zeros((1, REALTIME_POSE_SEQ_LEN, 5), dtype=np.float32)
    feature_stationary = np.ones((1, REALTIME_POSE_SEQ_LEN, 5), dtype=np.float32)
    feature_stationary[:, REALTIME_POSE_TARGET_START, :] = 0.0
    reference[0, :, schema.stationary_prob_slice()] = reference_stationary[0]
    reconstructed[0, :, schema.stationary_prob_slice()] = feature_stationary[0]
    np.savez(
        tmp_path / "clip_target_frame.npz",
        reference_features_raw=reference,
        reconstructed_features_raw=reconstructed,
        reference_stationary_prob_5=reference_stationary,
        feature_stationary_prob_5=feature_stationary,
    )

    result = evaluate_directory(input_dir=tmp_path, output_dir=tmp_path / "report", thresholds=(0.5,))

    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    aggregate = summary["signals"]["feature_channel"]["thresholds"]["0.5"]["aggregate"]
    assert aggregate["false_lock_rate"] == 0.0
