from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from data_loaders.generate_realtime_pose_tasks import load_realtime_source
from data_loaders.sensor_masking import REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SCHEMA_NAME
from visual_editor.models import ComparePreset, MotionAsset, MotionTrack, StudioConfig
from visual_editor.realtime_pose import iter_jsonl, load_task_npz, stable_id


class MotionLibrary:
    """扫描 realtime_pose_v2 source、task 和 reconstruction result。"""

    def __init__(self, config: StudioConfig):
        self.config = config
        self.assets: dict[str, MotionAsset] = {}
        self.index_meta: dict[str, Any] = {}

    def load(self, force: bool = False) -> dict[str, MotionAsset]:
        del force
        assets: dict[str, MotionAsset] = {}
        assets.update(self.scan_sources())
        assets.update(self.scan_tasks())
        assets.update(self.scan_results())
        self.assets = assets
        self.index_meta = {
            "schema_name": "realtime_pose_studio_library_v2",
            "source_dir": str(self.config.source_dir),
            "data_dir": str(self.config.data_dir),
            "result_dir": str(self.config.result_dir),
            "asset_count": len(assets),
        }
        return assets

    def scan_sources(self) -> dict[str, MotionAsset]:
        assets: dict[str, MotionAsset] = {}
        source_dir = self.config.source_dir
        if not source_dir.exists():
            return assets
        for path in sorted(source_dir.rglob("*.npz")):
            if "tasks" in path.parts:
                continue
            try:
                source = load_realtime_source(path)
            except Exception:
                continue
            relative = path.relative_to(source_dir).as_posix()
            frame_count = int(source["body_pose_parent_6d"].shape[0])
            asset_id = stable_id("source", str(path.resolve()))
            track = MotionTrack(
                track_id="realtime_source",
                label="Realtime Source",
                data_key="realtime_source",
                frame_count=frame_count,
                fps=float(self.config.realtime_pose_fps),
                source_path=path,
                compatible_realtime_pose=True,
                meta={"source_relative_path": relative, "schema_name": REALTIME_POSE_SCHEMA_NAME},
            )
            assets[asset_id] = MotionAsset(
                asset_id=asset_id,
                kind="source",
                label=Path(relative).stem,
                tracks={track.track_id: track},
                source_path=path,
                frame_count=frame_count,
                fps=float(self.config.realtime_pose_fps),
                group=relative.split("/")[0] if "/" in relative else "Realtime Source",
                meta={"schema_name": REALTIME_POSE_SCHEMA_NAME, "source_relative_path": relative},
            )
        return assets

    def scan_tasks(self) -> dict[str, MotionAsset]:
        assets: dict[str, MotionAsset] = {}
        data_dir = self.config.data_dir
        if not data_dir.exists():
            return assets
        manifest_paths = sorted(data_dir.glob("*/manifest.jsonl"))
        root_manifest = data_dir / "manifest.jsonl"
        if root_manifest.exists():
            manifest_paths.append(root_manifest)
        for manifest_path in manifest_paths:
            for entry in iter_jsonl(manifest_path):
                if entry.get("schema_name") != REALTIME_POSE_SCHEMA_NAME:
                    continue
                task_path = manifest_path.parent / str(entry["task_path"])
                try:
                    task = load_task_npz(task_path)
                except Exception:
                    continue
                frame_count = int(task["body_pose_parent_6d"].shape[0])
                asset_id = stable_id("task", str(task_path.resolve()))
                track = MotionTrack(
                    track_id="task_reference",
                    label=f"Task {entry.get('tracker_pattern', '')}".strip(),
                    data_key="task",
                    frame_count=frame_count,
                    fps=float(self.config.realtime_pose_fps),
                    source_path=task_path,
                    compatible_realtime_pose=True,
                    meta=dict(entry),
                )
                assets[asset_id] = MotionAsset(
                    asset_id=asset_id,
                    kind="task",
                    label=str(entry.get("task_id") or task_path.stem),
                    tracks={track.track_id: track},
                    source_path=task_path,
                    frame_count=frame_count,
                    fps=float(self.config.realtime_pose_fps),
                    group=str(entry.get("split") or manifest_path.parent.name),
                    meta=dict(entry),
                )
        return assets

    def scan_results(self) -> dict[str, MotionAsset]:
        assets: dict[str, MotionAsset] = {}
        result_dir = self.config.result_dir
        if not result_dir.exists():
            return assets
        for path in sorted(result_dir.rglob("*.npz")):
            try:
                with np.load(path, allow_pickle=False) as data:
                    files = set(data.files)
                    if not ({"reference_features", "reconstructed_features"} & files):
                        continue
                    feature_key = "reference_features" if "reference_features" in files else "reconstructed_features"
                    feature_array = np.asarray(data[feature_key])
                    if feature_array.ndim == 2 and feature_array.shape[1] == REALTIME_POSE_INPUT_DIM:
                        frame_count = int(feature_array.shape[0])
                    elif feature_array.ndim == 3 and feature_array.shape[2] == REALTIME_POSE_INPUT_DIM:
                        frame_count = int(feature_array.shape[1])
                    elif feature_array.ndim == 3 and feature_array.shape[1] == REALTIME_POSE_INPUT_DIM:
                        frame_count = int(feature_array.shape[2])
                    else:
                        continue
            except Exception:
                continue
            asset_id = stable_id("result", str(path.resolve()))
            tracks = {}
            for key, label in (
                ("reference_features", "Reference"),
                ("conditioned_features", "Conditioned"),
                ("reconstructed_features", "Reconstructed"),
            ):
                tracks[key] = MotionTrack(
                    track_id=key,
                    label=label,
                    data_key=key,
                    frame_count=frame_count,
                    fps=float(self.config.realtime_pose_fps),
                    source_path=path,
                    compatible_realtime_pose=False,
                    meta={"schema_name": REALTIME_POSE_SCHEMA_NAME},
                )
            assets[asset_id] = MotionAsset(
                asset_id=asset_id,
                kind="result",
                label=path.stem,
                tracks=tracks,
                source_path=path,
                frame_count=frame_count,
                fps=float(self.config.realtime_pose_fps),
                group="Reconstruction Result",
                meta={"schema_name": REALTIME_POSE_SCHEMA_NAME},
            )
        return assets

    def payload(self) -> dict[str, Any]:
        return {
            "schema_name": "realtime_pose_studio_library_v2",
            "index": self.index_meta,
            "assets": [asset.to_dict() for asset in self.assets.values()],
            "stats": {
                "source_count": sum(1 for item in self.assets.values() if item.kind == "source"),
                "task_count": sum(1 for item in self.assets.values() if item.kind == "task"),
                "result_count": sum(1 for item in self.assets.values() if item.kind == "result"),
            },
            "compare_presets": [preset.to_dict() for preset in default_compare_presets()],
        }


def default_compare_presets() -> list[ComparePreset]:
    return [
        ComparePreset("realtime_source", "Realtime Source", 1, "Show converted realtime_pose_v2 source motion."),
        ComparePreset("task_reference", "Task Reference", 1, "Show realtime_pose_v2 materialized task."),
        ComparePreset("result", "Reconstruction Result", 3, "Compare reference, conditioned, and reconstructed tracks."),
    ]
