from __future__ import annotations

import numpy as np

from data_loaders.generate_realtime_pose_tasks import main as generate_realtime_pose_tasks_main
from data_loaders.realtime_pose_dataset import RealtimePoseTaskDataset
from data_loaders.sensor_masking import REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SCHEMA_NAME, REALTIME_POSE_SEQ_LEN
from tests.smoke.realtime_pose_fixtures import write_toy_source_dataset
from visual_editor.models import StudioConfig
from visual_editor.services import MotionStudioService


def test_realtime_pose_studio_scans_frames_and_exports_tasks(tmp_path):
    source_dir = tmp_path / "sources"
    task_dir = tmp_path / "tasks"
    result_dir = tmp_path / "results"
    export_dir = tmp_path / "exports"
    runtime_dir = tmp_path / "runtime"
    write_toy_source_dataset(source_dir)
    generate_realtime_pose_tasks_main(
        [
            "--source_dir",
            str(source_dir),
            "--output_dir",
            str(task_dir),
            "--splits",
            "train",
            "--samples_per_file",
            "1",
            "--schema",
            REALTIME_POSE_SCHEMA_NAME,
            "--split_dir",
            "",
            "--overwrite",
        ]
    )
    result_dir.mkdir()
    np.savez(
        result_dir / "toy_result.npz",
        schema_name=np.asarray(REALTIME_POSE_SCHEMA_NAME),
        reference_features=np.zeros((1, REALTIME_POSE_SEQ_LEN, REALTIME_POSE_INPUT_DIM), dtype=np.float32),
        reconstructed_features=np.zeros((1, REALTIME_POSE_SEQ_LEN, REALTIME_POSE_INPUT_DIM), dtype=np.float32),
    )
    service = MotionStudioService(
        StudioConfig.from_paths(
            amass_dir=tmp_path / "amass",
            source_dir=source_dir,
            data_dir=task_dir,
            result_dir=result_dir,
            output_dir=export_dir,
            runtime_dir=runtime_dir,
            realtime_pose_fps=60.0,
        )
    )
    payload = service.library_payload()
    kinds = {asset["kind"] for asset in payload["assets"]}
    assert {"source", "task", "result"}.issubset(kinds)
    result_asset = next(asset for asset in payload["assets"] if asset["kind"] == "result")
    assert result_asset["frame_count"] == REALTIME_POSE_SEQ_LEN
    source_asset = next(asset for asset in payload["assets"] if asset["kind"] == "source")
    frames = service.frames_payload(asset_id=source_asset["asset_id"], track_id="realtime_source", start=0, count=2)
    assert frames["count"] == 2
    assert len(frames["frames"][0]["trackers"]) == 6
    assert len(frames["frames"][0]["sensor_valid"]) == 6
    assert "contact" not in frames["frames"][0]
    assert "sensor_missing_labels" not in frames["frames"][0]

    project = service.edit.create_project(asset_id=source_asset["asset_id"], track_id="realtime_source")
    exported = service.edit.export(
        project_id=project["project_id"],
        request={
            "output_dir": str(export_dir),
            "frame_start": 60,
            "frame_end": 60,
            "tracker_patterns": ["full-trackers", "mixed-sparse"],
            "split": "train",
            "export_name": "studio_export",
        },
    )
    dataset = RealtimePoseTaskDataset(exported["export_dir"], split="train", normalize_input=False)
    assert len(dataset) == 2
    assert exported["mask_policy"] == "fixed_patterns"
    assert tuple(dataset[0]["x"].shape) == (REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SEQ_LEN)
