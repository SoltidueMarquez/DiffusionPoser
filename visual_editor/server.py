from __future__ import annotations

import argparse
import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from visual_editor.models import ComparePane, StudioConfig
from visual_editor.services import MotionStudioService


class PaneRequest(BaseModel):
    asset_id: str
    track_id: str
    label: str = ""
    frame_offset: int = 0


class CompareRequest(BaseModel):
    start: int = 0
    count: int = 120
    panes: list[PaneRequest] = Field(default_factory=list, min_length=1, max_length=4)


class CreateEditProjectRequest(BaseModel):
    asset_id: str
    track_id: str
    name: str | None = None


class KeyframePatchRequest(BaseModel):
    target: str
    frame: int
    position: list[float] | None = Field(default=None, min_length=3, max_length=3)
    action: Literal["upsert", "delete"] = "upsert"


class PreviewRequest(BaseModel):
    start: int = 0
    count: int = 60


class ExportRequest(BaseModel):
    frame_start: int | None = None
    frame_end: int | None = None
    stride: int = 1
    target_frames: list[int] | None = None
    split: str = "train"
    missing_sensors: list[int | str] = Field(default_factory=list)
    tracker_pattern: str | None = None
    tracker_patterns: list[str] | None = None
    seed: int = 10
    export_name: str | None = None
    output_dir: str | None = None


def build_app(service: MotionStudioService) -> FastAPI:
    app = FastAPI(
        title="RealtimePose Studio API",
        version="1.0.0",
        description="Local realtime_pose_v2 source/task/result viewer and task export API.",
    )
    app.state.service = service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5177", "http://localhost:5177", "null"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def run_or_404(fn, *args, **kwargs) -> Any:
        try:
            return fn(*args, **kwargs)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, FileExistsError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/library")
    def library() -> dict[str, Any]:
        return service.library_payload()

    @app.post("/api/library/refresh")
    def refresh_library() -> dict[str, Any]:
        return service.refresh_library()

    @app.get("/api/ai-index")
    def ai_index() -> dict[str, Any]:
        return service.ai_index_payload()

    @app.get("/api/assets/{asset_id}")
    def asset(asset_id: str) -> dict[str, Any]:
        return run_or_404(service.asset_payload, asset_id)

    @app.get("/api/assets/{asset_id}/frames")
    def frames(asset_id: str, track_id: str, start: int = 0, count: int = 60, frame_offset: int = 0) -> dict[str, Any]:
        return run_or_404(
            service.frames_payload,
            asset_id=asset_id,
            track_id=track_id,
            start=start,
            count=count,
            frame_offset=frame_offset,
        )

    @app.post("/api/compare/frames")
    def compare_frames(request: CompareRequest) -> dict[str, Any]:
        panes = [ComparePane(**pane.model_dump()) for pane in request.panes]
        return run_or_404(service.compare_payload, panes=panes, start=request.start, count=request.count)

    @app.get("/api/assets/{asset_id}/mesh")
    def mesh(asset_id: str, track_id: str, frame: int = 0) -> dict[str, Any]:
        return run_or_404(service.mesh_payload, asset_id=asset_id, track_id=track_id, frame=frame)

    @app.post("/api/edit/projects")
    def create_edit_project(request: CreateEditProjectRequest) -> dict[str, Any]:
        return run_or_404(
            service.edit.create_project,
            asset_id=request.asset_id,
            track_id=request.track_id,
            name=request.name,
        )

    @app.get("/api/edit/projects/{project_id}")
    def get_edit_project(project_id: str) -> dict[str, Any]:
        return run_or_404(service.edit.load_project, project_id)

    @app.patch("/api/edit/projects/{project_id}/keyframes")
    def patch_keyframes(project_id: str, request: KeyframePatchRequest) -> dict[str, Any]:
        return run_or_404(
            service.edit.patch_keyframe,
            project_id=project_id,
            target=request.target,
            frame=request.frame,
            position=request.position,
            action=request.action,
        )

    @app.post("/api/edit/projects/{project_id}/preview")
    def preview(project_id: str, request: PreviewRequest) -> dict[str, Any]:
        return run_or_404(service.edit.preview, project_id=project_id, start=request.start, count=request.count)

    @app.post("/api/edit/projects/{project_id}/export")
    def export_project(project_id: str, request: ExportRequest) -> dict[str, Any]:
        return run_or_404(service.edit.export, project_id=project_id, request=request.model_dump(exclude_none=True))

    return app


def config_from_env() -> StudioConfig:
    return StudioConfig.from_paths(
        amass_dir=os.environ.get("REALTIME_POSE_EDITOR_AMASS_DIR", "dataset/AMASS"),
        source_dir=os.environ.get("REALTIME_POSE_EDITOR_SOURCE_DIR", "dataset/AMASS_realtime_pose_body_fbx_local_root_y0_60hz"),
        data_dir=os.environ.get("REALTIME_POSE_EDITOR_DATA_DIR", "dataset/AMASS_realtime_pose_body_fbx_local_root_y0_60hz_tasks"),
        result_dir=os.environ.get("REALTIME_POSE_EDITOR_RESULT_DIR", "output"),
        output_dir=os.environ.get("REALTIME_POSE_EDITOR_OUTPUT_DIR", "visual_editor/.runtime/exports"),
        smpl_model_dir=os.environ.get("REALTIME_POSE_EDITOR_SMPL_MODEL_DIR", "dataset/body_models"),
        runtime_dir=os.environ.get("REALTIME_POSE_EDITOR_RUNTIME_DIR", "visual_editor/.runtime"),
        realtime_pose_fps=float(os.environ.get("REALTIME_POSE_EDITOR_FPS", "60.0")),
    )


app = build_app(MotionStudioService(config_from_env())) if __name__ != "__main__" else None


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local RealtimePose Studio API server.")
    parser.add_argument("--amass_dir", default=os.environ.get("REALTIME_POSE_EDITOR_AMASS_DIR", "dataset/AMASS"))
    parser.add_argument("--source_dir", default=os.environ.get("REALTIME_POSE_EDITOR_SOURCE_DIR", "dataset/AMASS_realtime_pose_body_fbx_local_root_y0_60hz"))
    parser.add_argument("--data_dir", default=os.environ.get("REALTIME_POSE_EDITOR_DATA_DIR", "dataset/AMASS_realtime_pose_body_fbx_local_root_y0_60hz_tasks"))
    parser.add_argument("--result_dir", default=os.environ.get("REALTIME_POSE_EDITOR_RESULT_DIR", "output"))
    parser.add_argument("--output_dir", default=os.environ.get("REALTIME_POSE_EDITOR_OUTPUT_DIR", "visual_editor/.runtime/exports"))
    parser.add_argument("--smpl_model_dir", default=os.environ.get("REALTIME_POSE_EDITOR_SMPL_MODEL_DIR", "dataset/body_models"))
    parser.add_argument("--runtime_dir", default=os.environ.get("REALTIME_POSE_EDITOR_RUNTIME_DIR", "visual_editor/.runtime"))
    parser.add_argument("--realtime_pose_fps", default=60.0, type=float)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--reload", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    service = MotionStudioService(
        StudioConfig.from_paths(
            amass_dir=args.amass_dir,
            source_dir=args.source_dir,
            data_dir=args.data_dir,
            result_dir=args.result_dir,
            output_dir=args.output_dir,
            smpl_model_dir=args.smpl_model_dir or None,
            runtime_dir=args.runtime_dir,
            realtime_pose_fps=args.realtime_pose_fps,
        )
    )
    uvicorn.run(build_app(service), host=args.host, port=args.port, reload=bool(args.reload))


if __name__ == "__main__":
    main()
