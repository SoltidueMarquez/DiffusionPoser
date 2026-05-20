from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from visual_editor.models import MotionAsset, MotionTrack, StudioConfig
from visual_editor.x277 import (
    build_amass_frames,
    build_x277_frames,
    file_stat_payload,
    load_stream_npz,
    load_task_npz,
    load_x277_from_npz,
    stable_id,
)


class MotionDecoder:
    def __init__(self, config: StudioConfig, assets: dict[str, MotionAsset]):
        self.config = config
        self.assets = assets
        self.frame_cache_dir = self.config.runtime_dir / "frame_cache"
        self.frame_cache_dir.mkdir(parents=True, exist_ok=True)
        self._smpl_model_cache = None

    def set_assets(self, assets: dict[str, MotionAsset]) -> None:
        self.assets = assets

    def get_asset(self, asset_id: str) -> MotionAsset:
        if asset_id not in self.assets:
            raise KeyError(f"unknown asset_id: {asset_id}")
        return self.assets[asset_id]

    def get_track(self, asset_id: str, track_id: str) -> MotionTrack:
        asset = self.get_asset(asset_id)
        if track_id not in asset.tracks:
            raise KeyError(f"unknown track_id for {asset_id}: {track_id}")
        return asset.tracks[track_id]

    def frames(
        self,
        *,
        asset_id: str,
        track_id: str,
        start: int,
        count: int,
        frame_offset: int = 0,
    ) -> dict[str, Any]:
        track = self.get_track(asset_id, track_id)
        if not track.available:
            raise ValueError(track.unavailable_reason or f"track unavailable: {track_id}")
        if track.track_id == "amass_raw":
            joints = self.load_amass_joints(track)
            return build_amass_frames(
                asset_id=asset_id,
                track_id=track_id,
                joints=joints,
                start=start,
                count=count,
                fps=track.fps,
                frame_offset=frame_offset,
            )
        x277, task = self.load_x277_track(track)
        return build_x277_frames(
            asset_id=asset_id,
            track_id=track_id,
            x277=x277,
            task=task,
            start=start,
            count=count,
            fps=track.fps,
            frame_offset=frame_offset,
        )

    def load_x277_track(self, track: MotionTrack) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
        if track.data_key == "x":
            return load_x277_from_npz(track.source_path, key="x"), None
        if track.data_key == "x277":
            task = load_task_npz(track.source_path)
            return task["x277"].astype(np.float32, copy=True), task
        stream_keys = {"reference_motion", "conditioned_motion", "reconstructed_motion"}
        if track.data_key in stream_keys:
            payload = load_stream_npz(track.source_path)
            task = {
                key: payload[key]
                for key in ("sensor_missing_labels", "inpaint_mask", "valid_frame_mask")
                if key in payload
            }
            return payload[track.data_key].astype(np.float32, copy=True), task
        raise ValueError(f"track is not X277-compatible: {track.track_id}")

    def load_amass_joints(self, track: MotionTrack) -> np.ndarray:
        if not self.config.smpl_model_dir:
            raise ValueError("AMASS raw track requires smpl_model_dir")
        cache_path = self.amass_cache_path(track.source_path)
        if cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as data:
                    return data["joints"].astype(np.float32)
            except Exception:
                pass
        joints = self.build_amass_joint_cache(track.source_path)
        np.savez(cache_path, joints=joints.astype(np.float32))
        return joints.astype(np.float32)

    def amass_cache_path(self, path: Path) -> Path:
        stat = file_stat_payload(path)
        token = stable_id("amass_joints", f"{stat['path']}|{stat['size']}|{stat['mtime_ns']}|{self.config.x277_fps}")
        return self.frame_cache_dir / f"{token}.npz"

    def build_amass_joint_cache(self, path: Path) -> np.ndarray:
        try:
            from data_converter.amass_to_x277 import SmplModelCache, load_motion_source, run_smpl_forward
        except Exception as exc:
            raise RuntimeError("AMASS raw visualization requires torch/smplx and converter dependencies") from exc
        if self.config.smpl_model_dir is None or not self.config.smpl_model_dir.exists():
            raise FileNotFoundError(f"smpl_model_dir not found: {self.config.smpl_model_dir}")
        if self._smpl_model_cache is None:
            self._smpl_model_cache = SmplModelCache(model_dir=self.config.smpl_model_dir)
        source = load_motion_source(path=path, amass_dir=self.config.amass_dir, target_fps=float(self.config.x277_fps))
        motion = run_smpl_forward(source=source, model_cache=self._smpl_model_cache, batch_size=256)
        return motion.joint_positions.astype(np.float32)

    def mesh(self, *, asset_id: str, track_id: str, frame: int) -> dict[str, Any]:
        track = self.get_track(asset_id, track_id)
        if not track.available:
            return {"available": False, "reason": track.unavailable_reason or "track unavailable"}
        if track.track_id == "amass_raw":
            return {"available": False, "reason": "AMASS raw mesh preview is not cached in v1; joints are available."}
        if not self.config.smpl_model_dir:
            return {"available": False, "reason": "smpl_model_dir is not configured."}
        from visual_editor.smpl_preview import build_smpl_mesh_payload

        x277, _ = self.load_x277_track(track)
        return build_smpl_mesh_payload(smpl_model_dir=self.config.smpl_model_dir, x277=x277, frame=frame)
