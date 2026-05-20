from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from visual_editor.decoder import MotionDecoder
from visual_editor.library import MotionLibrary
from visual_editor.models import ComparePane, EDIT_PROJECT_SCHEMA, StudioConfig
from visual_editor.x277 import (
    KEYFRAME_TARGETS,
    SENSOR_NAMES,
    build_x277_frames,
    decode_root_trajectory,
    encode_body_velocity_from_skeletons,
    encode_root_deltas,
    interpolate_track,
    safe_token,
    skeletons_from_trackers,
    stable_id,
    tracker_world_to_root_local,
    utc_now,
    write_current277_task_dataset,
    write_json,
    load_json,
)
from sample.visualization import decode_x277_tracker_positions


class CompareService:
    def __init__(self, decoder: MotionDecoder):
        self.decoder = decoder

    def frames(self, *, panes: list[ComparePane], start: int, count: int) -> dict[str, Any]:
        if not panes:
            raise ValueError("compare requires at least one pane")
        if len(panes) > 4:
            raise ValueError("compare supports at most 4 panes")
        payload_panes = []
        warnings = []
        fps = 60.0
        for index, pane in enumerate(panes):
            asset = self.decoder.get_asset(pane.asset_id)
            track = self.decoder.get_track(pane.asset_id, pane.track_id)
            label = pane.label or f"{asset.label} / {track.label}"
            try:
                frame_payload = self.decoder.frames(
                    asset_id=pane.asset_id,
                    track_id=pane.track_id,
                    start=start,
                    count=count,
                    frame_offset=pane.frame_offset,
                )
                fps = float(track.fps)
            except Exception as exc:
                frame_payload = {
                    "asset_id": pane.asset_id,
                    "track_id": pane.track_id,
                    "start": int(start),
                    "frame_offset": int(pane.frame_offset),
                    "count": 0,
                    "frame_count": int(track.frame_count),
                    "fps": float(track.fps),
                    "frames": [],
                }
                warnings.append(f"{label}: {type(exc).__name__}: {exc}")
            payload_panes.append(
                {
                    "pane_index": index,
                    "label": label,
                    "asset": asset.to_dict(),
                    "track": track.to_dict(),
                    **frame_payload,
                }
            )
        return {
            "schema_name": "x277_motion_studio_compare_frames_v1",
            "start": int(start),
            "count": int(count),
            "fps": fps,
            "panes": payload_panes,
            "warnings": warnings,
        }


class EditService:
    def __init__(self, config: StudioConfig, decoder: MotionDecoder):
        self.config = config
        self.decoder = decoder
        self.projects_dir = self.config.runtime_dir / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def project_path(self, project_id: str) -> Path:
        return self.projects_dir / f"{safe_token(project_id)}.x277edit.json"

    def create_project(self, *, asset_id: str, track_id: str, name: str | None = None) -> dict[str, Any]:
        asset = self.decoder.get_asset(asset_id)
        track = self.decoder.get_track(asset_id, track_id)
        if not track.compatible_x277:
            raise ValueError(f"track is not editable/exportable: {track_id}")
        project_id = stable_id("edit", f"{asset_id}|{track_id}|{utc_now()}")
        project = {
            "schema_name": EDIT_PROJECT_SCHEMA,
            "project_id": project_id,
            "name": name or f"{asset.label} / {track.label} edit",
            "asset_id": asset_id,
            "track_id": track_id,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "source": {"asset": asset.to_dict(), "track": track.to_dict()},
            "keyframes": {target: [] for target in KEYFRAME_TARGETS},
        }
        write_json(self.project_path(project_id), project)
        return project

    def load_project(self, project_id: str) -> dict[str, Any]:
        path = self.project_path(project_id)
        if not path.exists():
            raise FileNotFoundError(f"edit project not found: {path}")
        project = load_json(path)
        if project.get("schema_name") != EDIT_PROJECT_SCHEMA:
            raise ValueError(f"{path} is not an edit project")
        return project

    def save_project(self, project: dict[str, Any]) -> dict[str, Any]:
        project["updated_at"] = utc_now()
        write_json(self.project_path(str(project["project_id"])), project)
        return project

    def patch_keyframe(
        self,
        *,
        project_id: str,
        target: str,
        frame: int,
        position: list[float] | tuple[float, float, float] | None,
        action: str,
    ) -> dict[str, Any]:
        if target not in KEYFRAME_TARGETS:
            raise ValueError(f"target must be one of {KEYFRAME_TARGETS}")
        project = self.load_project(project_id)
        x277, _ = self.load_project_motion(project)
        frame_index = int(frame)
        if frame_index < 0 or frame_index >= x277.shape[0]:
            raise ValueError(f"frame out of range: {frame_index}, T={x277.shape[0]}")
        keyframes = list(project.setdefault("keyframes", {}).setdefault(target, []))
        keyframes = [item for item in keyframes if int(item.get("frame", -1)) != frame_index]
        if action == "upsert":
            if position is None:
                raise ValueError("upsert requires position")
            point = np.asarray(position, dtype=np.float32)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError(f"position must be finite [3], got {position}")
            keyframes.append({"frame": frame_index, "position": point.astype(float).tolist()})
        elif action != "delete":
            raise ValueError(f"unknown keyframe action: {action}")
        project["keyframes"][target] = sorted(keyframes, key=lambda item: int(item["frame"]))
        return self.save_project(project)

    def load_project_motion(self, project: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
        track = self.decoder.get_track(str(project["asset_id"]), str(project["track_id"]))
        return self.decoder.load_x277_track(track)

    def apply_project_motion(self, project_id: str) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray] | None]:
        project = self.load_project(project_id)
        x277, task = self.load_project_motion(project)
        edited = x277.copy()
        root_positions, root_yaws = decode_root_trajectory(edited)
        tracker_world = decode_x277_tracker_positions(edited)
        keyframes = project.get("keyframes", {})

        root_positions = interpolate_track(root_positions, keyframes.get("root", []))
        for sensor_index, sensor_name in enumerate(SENSOR_NAMES):
            tracker_world[:, sensor_index, :] = interpolate_track(
                tracker_world[:, sensor_index, :],
                keyframes.get(sensor_name, []),
            )

        tracker_local = tracker_world_to_root_local(
            tracker_world=tracker_world,
            root_positions=root_positions,
            root_yaws=root_yaws,
        )
        edited[:, 216:234] = tracker_local.reshape(edited.shape[0], -1)
        encode_root_deltas(x277=edited, root_positions=root_positions, root_yaws=root_yaws)
        joints_world = skeletons_from_trackers(tracker_world)
        encode_body_velocity_from_skeletons(
            x277=edited,
            joints_world=joints_world,
            root_yaws=root_yaws,
            fps=self.config.x277_fps,
        )
        return edited, project, task

    def preview(self, *, project_id: str, start: int = 0, count: int = 60) -> dict[str, Any]:
        edited, project, task = self.apply_project_motion(project_id)
        return build_x277_frames(
            asset_id=str(project["asset_id"]),
            track_id=str(project["track_id"]),
            x277=edited,
            task=task,
            start=start,
            count=count,
            fps=self.config.x277_fps,
        )

    def export(self, *, project_id: str, request: dict[str, Any]) -> dict[str, Any]:
        edited, project, _ = self.apply_project_motion(project_id)
        asset = self.decoder.get_asset(str(project["asset_id"]))
        track = self.decoder.get_track(str(project["asset_id"]), str(project["track_id"]))
        export_root = Path(request.get("output_dir") or self.config.output_dir)
        return write_current277_task_dataset(
            edited=edited,
            source_label=f"{asset.label} / {track.label}",
            source_path=track.source_path,
            source_relative_path=str(track.meta.get("source_relative_path") or asset.label),
            project_id=project_id,
            output_dir=export_root,
            request=request,
        )


class MotionStudioService:
    def __init__(self, config: StudioConfig):
        self.config = config
        self.config.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.library = MotionLibrary(config)
        self.assets = self.library.load()
        self.decoder = MotionDecoder(config=config, assets=self.assets)
        self.compare = CompareService(self.decoder)
        self.edit = EditService(config=config, decoder=self.decoder)

    def refresh_library(self) -> dict[str, Any]:
        self.assets = self.library.load(force=True)
        self.decoder.set_assets(self.assets)
        return self.library_payload()

    def library_payload(self) -> dict[str, Any]:
        return self.library.payload()

    def asset_payload(self, asset_id: str) -> dict[str, Any]:
        return self.decoder.get_asset(asset_id).to_dict()

    def frames_payload(self, *, asset_id: str, track_id: str, start: int, count: int, frame_offset: int = 0) -> dict[str, Any]:
        return self.decoder.frames(
            asset_id=asset_id,
            track_id=track_id,
            start=start,
            count=count,
            frame_offset=frame_offset,
        )

    def compare_payload(self, *, panes: list[ComparePane], start: int, count: int) -> dict[str, Any]:
        return self.compare.frames(panes=panes, start=start, count=count)

    def mesh_payload(self, *, asset_id: str, track_id: str, frame: int) -> dict[str, Any]:
        return self.decoder.mesh(asset_id=asset_id, track_id=track_id, frame=frame)

    def ai_index_payload(self) -> dict[str, Any]:
        ai_index_path = Path(__file__).resolve().parent / "ai_index.json"
        static_index = load_json(ai_index_path) if ai_index_path.exists() else {}
        return {
            "static": static_index,
            "runtime": {
                "amass_dir": str(self.config.amass_dir),
                "source_dir": str(self.config.source_dir),
                "data_dir": str(self.config.data_dir),
                "result_dir": str(self.config.result_dir),
                "output_dir": str(self.config.output_dir),
                "asset_count": len(self.assets),
                "index": self.library.index_meta,
            },
        }
