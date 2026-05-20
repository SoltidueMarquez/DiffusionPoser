from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data_loaders.sensor_masking import (
    BODY_VEL_DIM,
    BODY_VEL_START,
    MODEL_INPUT_DIM,
    SENSOR_LABEL_DIM,
    TRACKER_POS_DIM,
    TRACKER_POS_START,
    X277_FEATURE_DIM,
)
from data_loaders.x277_dataset import X277MissingTaskDataset
from sample.visualization import SMPL_JOINT_COUNT
import visual_editor.library as library_module
from visual_editor.decoder import MotionDecoder
from visual_editor.models import ComparePane, StudioConfig
from visual_editor.services import MotionStudioService


def test_smpl_runtime_status_reports_missing_editor_dependencies(tmp_path: Path, monkeypatch):
    model_dir = tmp_path / "body_models"
    model_dir.mkdir()
    (model_dir / "SMPLH_NEUTRAL.npz").write_bytes(b"")

    monkeypatch.setattr(library_module.importlib.util, "find_spec", lambda name: None)

    status = library_module.smpl_runtime_status(model_dir)
    assert status["available"] is False
    assert status["has_model_files"] is True
    assert set(status["missing_packages"]) == {"torch", "smplx", "scipy"}
    assert "requirements-smpl.txt" in status["reason"]


def test_motion_studio_indexes_compares_edits_and_exports(tmp_path: Path, monkeypatch):
    paths = build_toy_studio_files(tmp_path)

    def fake_amass_joints(self, path: Path) -> np.ndarray:
        joints = np.zeros((18, SMPL_JOINT_COUNT, 3), dtype=np.float32)
        joints[:, :, 1] = np.linspace(0.0, 1.0, SMPL_JOINT_COUNT, dtype=np.float32)[None]
        joints[:, 0, 0] = np.arange(18, dtype=np.float32) * 0.02
        return joints

    monkeypatch.setattr(
        library_module,
        "smpl_runtime_status",
        lambda model_dir: {"available": True, "reason": "", "missing_packages": [], "has_model_files": True},
    )
    monkeypatch.setattr(MotionDecoder, "build_amass_joint_cache", fake_amass_joints)
    service = MotionStudioService(
        StudioConfig.from_paths(
            amass_dir=paths["amass_dir"],
            source_dir=paths["source_dir"],
            data_dir=paths["data_dir"],
            result_dir=paths["result_dir"],
            output_dir=tmp_path / "exports",
            runtime_dir=tmp_path / "runtime",
            smpl_model_dir=paths["smpl_model_dir"],
        )
    )

    library = service.library_payload()
    kinds = {asset["kind"] for asset in library["assets"]}
    assert {"amass", "x277", "task", "repair"}.issubset(kinds)

    assets = {asset.kind: asset for asset in service.assets.values()}
    amass_asset = assets["amass"]
    repair_asset = assets["repair"]
    x277_asset = assets["x277"]

    amass_frames = service.frames_payload(
        asset_id=amass_asset.asset_id,
        track_id="amass_raw",
        start=0,
        count=4,
    )
    assert amass_frames["count"] == 4
    assert len(amass_frames["frames"][0]["joints"]) == SMPL_JOINT_COUNT

    compare = service.compare_payload(
        panes=[
            ComparePane(asset_id=repair_asset.asset_id, track_id="ground_truth", label="GT"),
            ComparePane(asset_id=repair_asset.asset_id, track_id="reconstructed", label="Repair"),
        ],
        start=0,
        count=6,
    )
    assert len(compare["panes"]) == 2
    assert compare["panes"][0]["frames"][0]["track_id"] == "ground_truth"
    assert compare["panes"][1]["frames"][0]["track_id"] == "reconstructed"

    quad = service.compare_payload(
        panes=[
            ComparePane(asset_id=amass_asset.asset_id, track_id="amass_raw", label="AMASS"),
            ComparePane(asset_id=amass_asset.asset_id, track_id="x277_converted", label="X277"),
            ComparePane(asset_id=repair_asset.asset_id, track_id="ground_truth", label="GT"),
            ComparePane(asset_id=repair_asset.asset_id, track_id="reconstructed", label="Repair"),
        ],
        start=0,
        count=5,
    )
    assert len(quad["panes"]) == 4
    assert all(pane["count"] == 5 for pane in quad["panes"])

    project = service.edit.create_project(
        asset_id=x277_asset.asset_id,
        track_id="x277_converted",
        name="toy edit",
    )
    updated = service.edit.patch_keyframe(
        project_id=project["project_id"],
        target="head",
        frame=10,
        position=[0.25, 1.8, 0.1],
        action="upsert",
    )
    preview = service.edit.preview(project_id=updated["project_id"], start=8, count=4)
    assert preview["count"] == 4
    assert np.isfinite(np.asarray(preview["frames"][2]["trackers"], dtype=np.float32)).all()

    export_result = service.edit.export(
        project_id=updated["project_id"],
        request={
            "frame_start": 10,
            "frame_end": 11,
            "stride": 1,
            "split": "train",
            "missing_sensors": ["left_wrist"],
            "export_name": "toy_export",
        },
    )
    export_dir = Path(export_result["export_dir"])
    dataset = X277MissingTaskDataset(data_dir=export_dir, split="train", seq_len=11, normalize_input=False)
    item = dataset[0]
    assert tuple(item["x"].shape) == (MODEL_INPUT_DIM, 11)
    assert tuple(item["sensor_missing_labels"].shape) == (SENSOR_LABEL_DIM, 11)
    assert int(item["target_start"]) == 10
    assert int(item["target_length"]) == 1


def build_toy_studio_files(tmp_path: Path) -> dict[str, Path]:
    amass_dir = tmp_path / "AMASS"
    source_dir = tmp_path / "AMASS_current277_60hz"
    data_dir = tmp_path / "missing_tasks"
    result_dir = tmp_path / "output"
    smpl_model_dir = tmp_path / "body_models"
    for path in (amass_dir, source_dir, data_dir / "train" / "tasks", result_dir / "run" / "sample0", smpl_model_dir):
        path.mkdir(parents=True, exist_ok=True)

    raw_path = amass_dir / "ACCAD" / "toy_poses.npz"
    raw_path.parent.mkdir(parents=True)
    np.savez(
        raw_path,
        poses=np.zeros((18, 156), dtype=np.float32),
        trans=np.zeros((18, 3), dtype=np.float32),
        betas=np.zeros((16,), dtype=np.float32),
        gender=np.asarray("neutral"),
        mocap_framerate=np.float32(60.0),
    )

    x277 = build_toy_x277(frame_count=18)
    x277_path = source_dir / "ACCAD" / "toy_poses.npz"
    x277_path.parent.mkdir(parents=True)
    np.savez(x277_path, x=x277)
    with (source_dir / "manifest.jsonl").open("w", encoding="utf-8") as file:
        file.write(
            json.dumps(
                {
                    "feature_dim": X277_FEATURE_DIM,
                    "frames": 18,
                    "is_mirrored": False,
                    "output_path": str(x277_path),
                    "source_path": str(raw_path),
                    "source_relative_path": "ACCAD/toy_poses.npz",
                    "original_source_relative_path": "ACCAD/toy_poses.npz",
                    "stablemotion_split_key": "ACCAD/toy_poses.npy",
                    "status": "converted",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )

    task_path = data_dir / "train" / "tasks" / "toy_task.npz"
    clip = x277[:11].copy()
    sensor_missing_labels = np.zeros((11, SENSOR_LABEL_DIM), dtype=bool)
    inpaint_mask = np.zeros((11, MODEL_INPUT_DIM), dtype=bool)
    np.savez(
        task_path,
        x277=clip,
        sensor_missing_labels=sensor_missing_labels,
        inpaint_mask=inpaint_mask,
        start_frame=np.int64(3),
        valid_length=np.int64(11),
        source_frames=np.int64(18),
        seq_len=np.int64(11),
    )
    with (data_dir / "train" / "manifest.jsonl").open("w", encoding="utf-8") as file:
        file.write(
            json.dumps(
                {
                    "task_id": "toy_task",
                    "task_path": "tasks/toy_task.npz",
                    "source_path": str(x277_path),
                    "source_relative_path": "ACCAD/toy_poses.npz",
                    "split": "train",
                    "start_frame": 3,
                    "valid_length": 11,
                    "seq_len": 11,
                    "feature_dim": MODEL_INPUT_DIM,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )

    stream_dir = result_dir / "run" / "sample0"
    np.savez(
        stream_dir / "stream_outputs.npz",
        reference_motion=x277.copy(),
        conditioned_motion=x277.copy(),
        reconstructed_motion=x277 * 0.98,
        sensor_missing_labels=np.zeros((18, SENSOR_LABEL_DIM), dtype=bool),
        inpaint_mask=np.zeros((18, MODEL_INPUT_DIM), dtype=bool),
        valid_frame_mask=np.ones((18,), dtype=bool),
    )
    with (stream_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "sample_name": "sample0",
                "task_id": "toy_task",
                "source_path": str(x277_path),
                "valid_length": 18,
                "x277_fps": 60.0,
            },
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    return {
        "amass_dir": amass_dir,
        "source_dir": source_dir,
        "data_dir": data_dir,
        "result_dir": result_dir,
        "smpl_model_dir": smpl_model_dir,
    }


def build_toy_x277(frame_count: int) -> np.ndarray:
    motion = np.zeros((frame_count, X277_FEATURE_DIM), dtype=np.float32)
    local_trackers = np.asarray(
        [
            [0.00, 1.72, 0.05],
            [-0.45, 1.18, 0.02],
            [0.45, 1.18, 0.02],
            [0.00, 0.95, 0.00],
            [-0.16, 0.05, 0.12],
            [0.16, 0.05, 0.12],
        ],
        dtype=np.float32,
    )
    for frame_index in range(frame_count):
        motion[
            frame_index,
            TRACKER_POS_START : TRACKER_POS_START + SENSOR_LABEL_DIM * TRACKER_POS_DIM,
        ] = local_trackers.reshape(-1)
        body_velocity = np.zeros((SMPL_JOINT_COUNT, 3), dtype=np.float32)
        body_velocity[:, 0] = 0.03 * 60.0
        motion[frame_index, BODY_VEL_START : BODY_VEL_START + BODY_VEL_DIM] = body_velocity.reshape(-1)
        motion[frame_index, 270:272] = np.asarray([0.03, 0.0], dtype=np.float32)
        motion[frame_index, 272] = 0.0
        motion[frame_index, 273:277] = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    return motion
