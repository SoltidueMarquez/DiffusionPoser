from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_REALTIME_POSE_FPS = 60.0
LIBRARY_CACHE_SCHEMA = "realtime_pose_studio_library_cache_v1"
LIBRARY_CACHE_VERSION = 2
EDIT_PROJECT_SCHEMA = "realtime_pose_studio_edit_project_v1"
EDITED_DATASET_SCHEMA = "realtime_pose_studio_edited_dataset_v1"


@dataclass
class StudioConfig:
    amass_dir: Path
    source_dir: Path
    data_dir: Path
    result_dir: Path
    output_dir: Path
    smpl_model_dir: Path | None = None
    runtime_dir: Path = Path("visual_editor/.runtime")
    realtime_pose_fps: float = DEFAULT_REALTIME_POSE_FPS

    @classmethod
    def from_paths(
        cls,
        *,
        amass_dir: str | Path,
        source_dir: str | Path,
        data_dir: str | Path,
        result_dir: str | Path,
        output_dir: str | Path,
        smpl_model_dir: str | Path | None = None,
        runtime_dir: str | Path = Path("visual_editor/.runtime"),
        realtime_pose_fps: float | None = None,
    ) -> "StudioConfig":
        smpl_path = Path(smpl_model_dir) if smpl_model_dir else None
        return cls(
            amass_dir=Path(amass_dir),
            source_dir=Path(source_dir),
            data_dir=Path(data_dir),
            result_dir=Path(result_dir),
            output_dir=Path(output_dir),
            smpl_model_dir=smpl_path,
            runtime_dir=Path(runtime_dir),
            realtime_pose_fps=float(realtime_pose_fps if realtime_pose_fps is not None else DEFAULT_REALTIME_POSE_FPS),
        )


@dataclass
class MotionTrack:
    track_id: str
    label: str
    data_key: str
    frame_count: int
    fps: float
    source_path: Path
    compatible_realtime_pose: bool
    available: bool = True
    unavailable_reason: str = ""
    frame_offset: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "data_key": self.data_key,
            "frame_count": int(self.frame_count),
            "fps": float(self.fps),
            "source_path": str(self.source_path),
            "compatible_realtime_pose": bool(self.compatible_realtime_pose),
            "available": bool(self.available),
            "unavailable_reason": self.unavailable_reason,
            "frame_offset": int(self.frame_offset),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MotionTrack":
        return cls(
            track_id=str(payload["track_id"]),
            label=str(payload["label"]),
            data_key=str(payload["data_key"]),
            frame_count=int(payload["frame_count"]),
            fps=float(payload["fps"]),
            source_path=Path(str(payload["source_path"])),
            compatible_realtime_pose=bool(payload.get("compatible_realtime_pose", False)),
            available=bool(payload.get("available", True)),
            unavailable_reason=str(payload.get("unavailable_reason", "")),
            frame_offset=int(payload.get("frame_offset", 0)),
            meta=dict(payload.get("meta", {})),
        )


@dataclass
class MotionAsset:
    asset_id: str
    kind: str
    label: str
    tracks: dict[str, MotionTrack]
    source_path: Path
    frame_count: int
    fps: float
    group: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "kind": self.kind,
            "label": self.label,
            "source_path": str(self.source_path),
            "frame_count": int(self.frame_count),
            "fps": float(self.fps),
            "group": self.group,
            "meta": self.meta,
            "tracks": [track.to_dict() for track in self.tracks.values()],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MotionAsset":
        tracks = {track.track_id: track for track in (MotionTrack.from_dict(item) for item in payload.get("tracks", []))}
        return cls(
            asset_id=str(payload["asset_id"]),
            kind=str(payload["kind"]),
            label=str(payload["label"]),
            tracks=tracks,
            source_path=Path(str(payload["source_path"])),
            frame_count=int(payload["frame_count"]),
            fps=float(payload["fps"]),
            group=str(payload.get("group", "")),
            meta=dict(payload.get("meta", {})),
        )


@dataclass
class ComparePane:
    asset_id: str
    track_id: str
    label: str = ""
    frame_offset: int = 0


@dataclass
class ComparePreset:
    preset_id: str
    label: str
    pane_count: int
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset_id": self.preset_id,
            "label": self.label,
            "pane_count": int(self.pane_count),
            "description": self.description,
        }
