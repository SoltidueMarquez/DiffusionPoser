# X277/AMASS Motion Studio

`visual_editor/` is a local 3D viewer/editor for AMASS, converted X277, materialized current277 tasks, and streaming repair outputs.

## Environment

All editor dependencies and runtime files stay under this directory:

- Python: `visual_editor/.venv/`
- Node: `visual_editor/node_modules/`
- Caches: `visual_editor/.cache/`
- Runtime/index/frame caches: `visual_editor/.runtime/`
- Edited exports: `visual_editor/.runtime/exports/`

Bootstrap:

```powershell
powershell -ExecutionPolicy Bypass -File visual_editor/scripts/bootstrap.ps1
```

## Launch

One-click Electron shell:

```powershell
start_visual_editor.cmd
```

Web safe mode:

```powershell
start_visual_editor_web.cmd
```

By default the launcher uses `dataset/body_models` as `smpl_model_dir`, so raw AMASS rendering is enabled automatically when the model files exist.

Explicit paths:

```powershell
start_visual_editor.cmd -AmassDir dataset/AMASS -SourceDir dataset/AMASS_current277_60hz -DataDir dataset/AMASS_current277_60hz_missing_tasks -ResultDir output -OutputDir visual_editor/.runtime/exports -SmplModelDir dataset/body_models
```

API only:

```powershell
visual_editor/.venv/Scripts/python -m visual_editor.server --amass_dir dataset/AMASS --source_dir dataset/AMASS_current277_60hz --data_dir dataset/AMASS_current277_60hz_missing_tasks --result_dir output --output_dir visual_editor/.runtime/exports --smpl_model_dir dataset/body_models
```

Frontend development:

```powershell
npm.cmd --prefix visual_editor run viewer:dev
```

## Data Sources

- AMASS raw: `poses/trans/betas/gender/mocap_framerate`; direct rendering needs local `smpl_model_dir`.
- Converted X277: `.npz` with `x: [T, 277]`.
- Task windows: `.npz` with `x277`, masks, and current277 metadata.
- Repair outputs: `stream_outputs.npz` with `reference_motion`, `conditioned_motion`, and `reconstructed_motion`.

The library index is cached in `visual_editor/.runtime/library_cache.json`. SMPL frame results are cached in `visual_editor/.runtime/frame_cache/`.

## Optional SMPL

SMPL/SMPL-H assets are not downloaded automatically. The default local model directory is `dataset/body_models`. To enable raw AMASS rendering or mesh preview, install optional dependencies:

```powershell
visual_editor/.venv/Scripts/python -m pip install -r visual_editor/requirements-smpl.txt --cache-dir visual_editor/.cache/pip
start_visual_editor.cmd
```

## Current Interfaces

- `GET /api/library`
- `POST /api/library/refresh`
- `GET /api/assets/{asset_id}/frames?track_id=&start=&count=&frame_offset=`
- `POST /api/compare/frames`
- `POST /api/edit/projects`
- `PATCH /api/edit/projects/{project_id}/keyframes`
- `POST /api/edit/projects/{project_id}/preview`
- `POST /api/edit/projects/{project_id}/export`

Exported edited datasets remain current277 loader-compatible: 10 history frames plus the 11th target frame.
