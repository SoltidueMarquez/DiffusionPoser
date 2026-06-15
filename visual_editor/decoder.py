from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from data_loaders.generate_realtime_pose_tasks import load_realtime_source
from data_loaders.sensor_masking import POSE_REPRESENTATION_KEY, REALTIME_POSE_SCHEMA_NAME, get_schema_spec
from visual_editor.models import MotionAsset, MotionTrack, StudioConfig
from visual_editor.realtime_pose import build_realtime_pose_frames, load_stream_npz, load_task_npz


class MotionDecoder:
    def __init__(self, config: StudioConfig, assets: dict[str, MotionAsset]):
        self.config = config
        self.assets = assets

    def set_assets(self, assets: dict[str, MotionAsset]) -> None:
        self.assets = assets

    def get_asset(self, asset_id: str) -> MotionAsset:
        if asset_id not in self.assets:
            raise KeyError(f"asset not found: {asset_id}")
        return self.assets[asset_id]

    def get_track(self, asset_id: str, track_id: str) -> MotionTrack:
        asset = self.get_asset(asset_id)
        if track_id not in asset.tracks:
            raise KeyError(f"track not found: {asset_id}/{track_id}")
        return asset.tracks[track_id]

    def frames(self, *, asset_id: str, track_id: str, start: int, count: int, frame_offset: int = 0) -> dict[str, Any]:
        track = self.get_track(asset_id, track_id)
        motion, task = self.load_realtime_track(track)
        return build_realtime_pose_frames(
            asset_id=asset_id,
            track_id=track_id,
            motion=motion,
            task=task,
            start=start,
            count=count,
            fps=track.fps,
            frame_offset=frame_offset,
        )

    def load_realtime_track(self, track: MotionTrack) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
        if track.data_key == "realtime_source":
            return load_realtime_source(track.source_path), None
        if track.data_key == "task":
            task = load_task_npz(track.source_path)
            schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
            motion = {key: task[key] for key in (
                schema.body_pose_key,
                POSE_REPRESENTATION_KEY,
                "root_pos_world",
                "root_yaw",
                schema.root_heading_delta_key,
                "root_delta_xz_ref",
                schema.pelvis_height_key,
                "stationary_prob_5",
                "tracker_pos_world",
                "tracker_rot_world_6d",
                "joints_world",
                "joint_offsets_parent",
            ) if key in task}
            if "joint_rest_local_rotations_6d" in task:
                motion["joint_rest_local_rotations_6d"] = task["joint_rest_local_rotations_6d"]
            return motion, task
        if track.data_key in {"reference_features", "conditioned_features", "reconstructed_features"}:
            raise ValueError("result feature tracks do not contain root/joint arrays for studio FK display yet")
        raise ValueError(f"track is not realtime_pose_v2-compatible: {track.track_id}")

    def load_result_track(self, track: MotionTrack) -> np.ndarray:
        payload = load_stream_npz(track.source_path)
        if track.data_key not in payload:
            raise KeyError(f"{track.source_path} missing `{track.data_key}`")
        return np.asarray(payload[track.data_key], dtype=np.float32)

    def mesh(self, *, asset_id: str, track_id: str, frame: int) -> dict[str, Any]:
        del frame
        self.get_track(asset_id, track_id)
        raise ValueError("realtime_pose_v2 Studio 当前只提供 joints/tracker 帧数据，SMPL mesh 预览尚未接入。")
