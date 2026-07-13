from __future__ import annotations

import csv
import json

import numpy as np

from eval import globalpose_stationary_compare as compare
from eval.globalpose_stationary_compare import (
    build_sequence_report_payload,
    compute_compare_metrics,
    compute_window_metrics,
    evaluate_cached_sequences,
    render_sequence_html,
    sigmoid,
)


def test_compute_compare_metrics_reports_neutral_disagreement_rates():
    gt = np.asarray(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    pred = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    metrics = compute_compare_metrics(gt_stationary_prob_5=gt, globalpose_stationary_prob_5=pred)

    assert metrics["foot_source_low_globalpose_high_rate"] > 0.0
    assert metrics["foot_source_high_globalpose_low_rate"] > 0.0
    assert "foot_false_lock_rate" not in metrics
    assert "foot_missed_lock_rate" not in metrics
    assert metrics["foot_bce"] > metrics["foot_mae"]


def test_compute_window_metrics_uses_last_partial_position():
    gt = np.zeros((7, 5), dtype=np.float32)
    pred = np.zeros((7, 5), dtype=np.float32)
    pred[3:, 1] = 1.0

    rows = compute_window_metrics(
        sequence_name="seq_a",
        gt_stationary_prob_5=gt,
        globalpose_stationary_prob_5=pred,
        window_size=4,
        window_stride=3,
    )

    assert [(row["frame_start"], row["frame_end"]) for row in rows] == [(0, 3), (3, 6)]
    assert rows[-1]["foot_source_low_globalpose_high_rate"] > 0.0
    assert "foot_false_lock_rate" not in rows[-1]


def test_evaluate_cached_sequences_writes_html_report(tmp_path):
    gt_source_dir = tmp_path / "gt"
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "out"
    gt_source_dir.mkdir()
    cache_dir.mkdir()
    gt = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.2, 0.2],
            [0.0, 0.0, 0.0, 0.2, 0.2],
            [0.0, 1.0, 1.0, 0.2, 0.2],
            [0.0, 1.0, 1.0, 0.2, 0.2],
            [0.0, 0.0, 0.0, 0.2, 0.2],
            [0.0, 0.0, 0.0, 0.2, 0.2],
        ],
        dtype=np.float32,
    )
    pred = np.asarray(
        [
            [0.0, 1.0, 1.0, 0.1, 0.1],
            [0.0, 1.0, 1.0, 0.1, 0.1],
            [0.0, 1.0, 1.0, 0.1, 0.1],
            [0.0, 0.0, 0.0, 0.1, 0.1],
            [0.0, 0.0, 0.0, 0.1, 0.1],
            [0.0, 0.0, 0.0, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    np.savez(gt_source_dir / "seq_a.npz", stationary_prob_5=gt)
    np.savez(cache_dir / "seq_a.npz", globalpose_stationary_prob_5=pred)

    result = evaluate_cached_sequences(
        gt_source_dir=gt_source_dir,
        cache_dir=cache_dir,
        output_dir=output_dir,
        dataset_name="dummy",
        window_size=3,
        window_stride=2,
        sequence_names=["seq_a"],
    )

    assert result["summary"]["sequence_count"] == 1
    index_html = output_dir / "report" / "index.html"
    detail_html = output_dir / "report" / "sequences" / "seq_a.html"
    data_json = output_dir / "report" / "data" / "seq_a.json"
    assert index_html.exists()
    assert detail_html.exists()
    assert data_json.exists()
    assert "frameSlider" in detail_html.read_text(encoding="utf-8")
    payload = json.loads(data_json.read_text(encoding="utf-8"))
    assert payload["sequence"] == "seq_a"
    assert payload["bad_windows"]

    with (output_dir / "metrics" / "per_sequence.csv").open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["sequence"] == "seq_a"
    assert "foot_source_low_globalpose_high_rate" in rows[0]
    assert "foot_source_high_globalpose_low_rate" in rows[0]
    assert "foot_false_lock_rate" not in rows[0]


def test_build_skeleton_payload_serializes_gt_and_globalpose_joints():
    gt_joints = np.zeros((2, 24, 3), dtype=np.float32)
    gp_joints = np.ones((2, 24, 3), dtype=np.float32)
    gt_joints[:, 10, 1] = 0.25
    gp_joints[:, 11, 1] = 0.75

    assert hasattr(compare, "build_skeleton_payload")
    payload = compare.build_skeleton_payload(gt_joints_world=gt_joints, globalpose_joints_world=gp_joints, decimals=3)

    assert payload["frame_count"] == 2
    assert payload["joint_names"][0] == "pelvis"
    assert payload["parents"][0] == -1
    assert payload["gt_joints_world"][0][10][1] == 0.25
    assert payload["globalpose_joints_world"][0][11][1] == 0.75
    assert payload["camera_bounds"]["sequence"]["front"]["min"][0] == 0.0
    assert payload["camera_bounds"]["sequence"]["front"]["max"][0] == 1.0
    assert "pelvis" in payload["camera_bounds"]


def test_build_skeleton_camera_bounds_use_sequence_not_current_frame():
    gt_joints = np.zeros((3, 24, 3), dtype=np.float32)
    gp_joints = np.zeros((3, 24, 3), dtype=np.float32)
    gt_joints[:, 15, 1] = 1.0
    gp_joints[:, 15, 1] = 1.0
    gp_joints[2, 10, 0] = 3.0
    gp_joints[2, 10, 1] = -1.0

    bounds = compare.build_skeleton_camera_bounds(
        gt_joints_world=gt_joints,
        globalpose_joints_world=gp_joints,
        windows=[{"frame_start": 1, "frame_end": 2}],
        decimals=3,
    )

    assert bounds["sequence"]["front"]["max"][0] == 3.0
    assert bounds["sequence"]["front"]["min"][1] == -1.0
    assert bounds["windows"][0]["front"]["max"][0] == 3.0
    assert bounds["pelvis"]["front"]["max"][0] == 3.0


def test_globalpose_fk_joints_keep_y_up_for_report_world():
    joints = np.zeros((1, 24, 3), dtype=np.float32)
    joints[0, 15] = [0.0, 1.5, 0.1]
    joints[0, 10] = [0.2, -0.9, 0.0]

    assert hasattr(compare, "globalpose_fk_joints_to_report_world")
    converted = compare.globalpose_fk_joints_to_report_world(joints)

    assert converted[0, 15, 1] == 1.5
    assert converted[0, 15, 2] == 0.1
    assert converted[0, 10, 1] == -0.9


def test_build_skeleton_payloads_use_official_gt_fk_not_source_joints(monkeypatch, tmp_path):
    official_gt = np.zeros((2, 24, 3), dtype=np.float32)
    globalpose_pred = np.ones((2, 24, 3), dtype=np.float32)
    official_gt[:, 15, 1] = 1.5
    globalpose_pred[:, 15, 1] = 1.25

    def fake_load_official_gt(**_kwargs):
        return {"seq_a": official_gt}

    def fake_load_globalpose_pred(**_kwargs):
        return {"seq_a": globalpose_pred}

    def fail_if_source_joints_are_used(**_kwargs):
        raise AssertionError("source joints_world must not be used for GlobalPose skeleton GT")

    monkeypatch.setattr(compare, "load_globalpose_dataset_joints_world", fake_load_official_gt, raising=False)
    monkeypatch.setattr(compare, "load_globalpose_result_joints_world", fake_load_globalpose_pred)
    monkeypatch.setattr(compare, "read_gt_joints_world", fail_if_source_joints_are_used)

    payloads = compare.build_skeleton_payloads_for_sequences(
        globalpose_repo=tmp_path,
        globalpose_dataset=tmp_path / "dataset.pt",
        globalpose_result=tmp_path / "result.pt",
        gt_source_dir=tmp_path / "gt_source",
        sequence_names=["seq_a"],
        decimals=3,
    )

    payload = payloads["seq_a"]
    assert payload["gt_joints_world"][0][15][1] == 1.5
    assert payload["globalpose_joints_world"][0][15][1] == 1.25


def test_evaluate_cached_sequences_embeds_requested_skeleton_payload(tmp_path):
    gt_source_dir = tmp_path / "gt"
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "out"
    gt_source_dir.mkdir()
    cache_dir.mkdir()
    gt = np.zeros((2, 5), dtype=np.float32)
    pred = np.zeros((2, 5), dtype=np.float32)
    skeleton_payload = {
        "frame_count": 2,
        "joint_names": ["pelvis"] * 24,
        "parents": [-1] + [0] * 23,
        "stationary_joint_indices": [0, 10, 11, 22, 23],
        "gt_joints_world": np.zeros((2, 24, 3), dtype=np.float32).tolist(),
        "globalpose_joints_world": np.ones((2, 24, 3), dtype=np.float32).tolist(),
    }
    np.savez(gt_source_dir / "seq_a.npz", stationary_prob_5=gt)
    np.savez(cache_dir / "seq_a.npz", globalpose_stationary_prob_5=pred)

    evaluate_cached_sequences(
        gt_source_dir=gt_source_dir,
        cache_dir=cache_dir,
        output_dir=output_dir,
        dataset_name="dummy",
        window_size=2,
        window_stride=2,
        sequence_names=["seq_a"],
        skeleton_payloads_by_sequence={"seq_a": skeleton_payload},
    )

    payload = json.loads((output_dir / "report" / "data" / "seq_a.json").read_text(encoding="utf-8"))
    assert payload["skeleton"]["frame_count"] == 2
    assert payload["skeleton"]["globalpose_joints_world"][0][0][0] == 1.0


def test_render_sequence_html_includes_skeleton_viewer_when_payload_has_skeleton():
    gt = np.zeros((2, 5), dtype=np.float32)
    pred = np.zeros((2, 5), dtype=np.float32)
    payload = build_sequence_report_payload(
        sequence_name="seq_a",
        fps=60.0,
        gt_stationary_prob_5=gt,
        globalpose_stationary_prob_5=pred,
        metrics=compute_compare_metrics(gt_stationary_prob_5=gt, globalpose_stationary_prob_5=pred),
        bad_windows=[],
        thresholds={"gt_low": 0.3, "gt_high": 0.7, "pred": 0.5},
        skeleton_payload={
            "frame_count": 2,
            "joint_names": ["pelvis"] * 24,
            "parents": [-1] + [0] * 23,
            "stationary_joint_indices": [0, 10, 11, 22, 23],
            "gt_joints_world": np.zeros((2, 24, 3), dtype=np.float32).tolist(),
            "globalpose_joints_world": np.ones((2, 24, 3), dtype=np.float32).tolist(),
        },
    )

    html = render_sequence_html(payload)

    assert 'id="skeletonCanvas"' in html
    assert 'id="skeletonMode"' in html
    assert 'id="skeletonView"' in html
    assert 'id="skeletonCamera"' in html
    assert 'id="skeletonZoom"' in html
    assert "cameraBoundsFor" in html
    assert "drawSkeletonViewer" in html
    assert "stationaryProbabilityText" in html
    assert "drawStationaryProbabilityLabel" in html
    assert 'labelSource: "source"' in html
    assert 'labelSource: "globalpose"' in html
    assert "sourceRow[probIndex])}/${formatProbabilityValue(globalposeRow[probIndex])" in html
    assert "fillText(`${name}: ${state}`" not in html
    assert "Source label" in html
    assert "Pose GT" in html
    assert "source low / GlobalPose high" in html
    assert "source high / GlobalPose low" in html
    assert "false lock" not in html
    assert "missed lock" not in html
    assert "<th>GT</th>" not in html


def test_sigmoid_matches_probability_range():
    values = np.asarray([-100.0, 0.0, 100.0], dtype=np.float32)

    probs = sigmoid(values)

    assert float(probs[0]) < 1e-6
    assert float(probs[1]) == 0.5
    assert float(probs[2]) > 1.0 - 1e-6
