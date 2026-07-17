from __future__ import annotations

import json

from tests.smoke.realtime_pose_fixtures import write_toy_source_dataset
from tests.tools.stationary5_source_motion_review import build_motion_review_for_sources, main


def test_stationary5_source_motion_review_writes_interactive_html_and_summary(tmp_path):
    source_dir = tmp_path / "sources"
    output_dir = tmp_path / "motion_review"
    write_toy_source_dataset(source_dir, frame_count=96)

    result = build_motion_review_for_sources(
        source_root=source_dir,
        output_dir=output_dir,
        max_sources=1,
        seed=123,
        fps=60.0,
        clip_frames=32,
        clip_start=10,
    )

    summary_path = output_dir / "summary.json"
    index_path = output_dir / "index.html"
    assert result["summary_path"] == str(summary_path)
    assert result["index_path"] == str(index_path)
    assert summary_path.exists()
    assert index_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["selected_source_count"] == 1
    assert summary["clip_frames"] == 32
    assert summary["sources"][0]["frame_count"] == 96
    assert summary["sources"][0]["clip_start_frame"] == 10
    assert summary["sources"][0]["clip_frame_count"] == 32
    assert len(summary["sources"][0]["joint_speed_mps_p95"]) == 5

    detail_path = output_dir / summary["sources"][0]["detail_html"]
    assert detail_path.exists()
    detail_html = detail_path.read_text(encoding="utf-8")
    assert "motionCanvas" in detail_html
    assert "frameSlider" in detail_html
    assert "playButton" in detail_html
    assert "reviewData" in detail_html
    assert "skeleton_edges" in detail_html
    assert "stationary_prob_5" in detail_html
    assert "joint_center_speed_5" in detail_html
    assert "root speed" in detail_html
    assert "pelvis height" in detail_html
    assert "left_foot height" in detail_html
    assert "right_foot height" in detail_html


def test_stationary5_source_motion_review_cli(tmp_path):
    source_dir = tmp_path / "sources"
    output_dir = tmp_path / "motion_review"
    write_toy_source_dataset(source_dir, frame_count=64)

    exit_code = main(
        [
            "--source_dir",
            str(source_dir),
            "--output_dir",
            str(output_dir),
            "--max_sources",
            "1",
            "--seed",
            "7",
            "--clip_frames",
            "24",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "index.html").exists()
