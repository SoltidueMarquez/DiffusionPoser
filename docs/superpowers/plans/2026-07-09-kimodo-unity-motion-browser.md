# Kimodo Unity Motion Browser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Unity Editor asset-browser-style Kimodo motion preview tool that imports Kimodo raw NPZ into per-motion Unity assets, previews motions inside the tool window, and exposes DiffusionPoser conversion actions.

**Architecture:** Kimodo Python code converts raw `.npz/.bvh` files into a Unity-friendly motion package: one `KimodoMotionAsset` plus JSON `TextAsset`s and binary `.bytes` streams. Unity owns the browser, preview scene, Actor template binding, and command buttons. DiffusionPoser remains a command-line dependency for the second-layer pipeline; RealtimePose schema is used only after conversion, not for raw Kimodo preview.

**Tech Stack:** Python 3.11, numpy, pytest, PowerShell, Unity 2022.3.17f1c1, Unity Editor C#, `AI4Animation.Actor`, `PreviewRenderUtility` or hidden preview scene, Unity `ScriptableObject`, `TextAsset.bytes`.

## Global Constraints

- Default communication and code comments for project-specific logic should use Chinese when explanation is needed.
- Kimodo project root is `D:\Projects\SchoolWorkProjects\kimodo`.
- Unity project root is `D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity`.
- DiffusionPoser project root is `D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\DiffusionPoser`.
- Unity Editor version is `2022.3.17f1c1`.
- Kimodo raw files stay outside Unity under `D:\Projects\SchoolWorkProjects\kimodo\artifacts\kimodo_raw`.
- Unity imports only preview cache files under `Assets/Projects/RealtimePose/KimodoBrowser/Motions/<motion_name>/`.
- Do not create `KimodoMotionLibrary.asset`; the browser scans `t:KimodoMotionAsset`.
- Do not store per-frame motion arrays directly in ScriptableObject fields.
- Do not modify the current Unity scene during preview.
- First implementation supports SMPL/body.fbx Actor templates first; Humanoid retarget is detected but not implemented.
- DiffusionPoser command lists must begin with `["conda", "run", "--no-capture-output", "-n", "diffusionposer5070", "python", "-m"]`.
- Kimodo GUI command lists should begin with `["conda", "run", "--no-capture-output", "--prefix", "D:\\Anaconda\\envs\\kimodo_gui", "python", "-m"]` when run from Unity.
- RealtimePose conversion outputs must keep exact `schema_name`, default `realtime_pose_stationary5_v1`.

---

## Scope Check

This feature spans two repositories, but it is one vertical workflow:

- `D:\Projects\SchoolWorkProjects\kimodo` produces preview packages from raw Kimodo NPZ files.
- `SIGGRAPH2024Unity` consumes those packages and owns the EditorWindow.

Keep the tasks split by repository and commit after each independently testable slice. The DiffusionPoser repository only stores this plan and continues to provide existing command-line conversion scripts.

## File Structure

Kimodo project:

- Create `D:/Projects/SchoolWorkProjects/kimodo/kimodo_gui/export_unity_preview.py`: CLI and pure functions that convert Kimodo raw NPZ to a Unity preview package.
- Create `D:/Projects/SchoolWorkProjects/kimodo/tests/test_export_unity_preview.py`: pytest coverage for manifest, skeleton, stream sizes, quaternion conversion, and source hash.

Unity project runtime/editor files:

- Create `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/KimodoMotionAsset.cs`: per-motion ScriptableObject.
- Create `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/KimodoMotionManifest.cs`: serializable manifest/skeleton DTOs.
- Create `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/KimodoMotionReader.cs`: validates manifest and decodes `.bytes` streams.
- Create `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/KimodoMotionFrame.cs`: in-memory frame view for current frame positions/rotations/contacts.
- Create `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/KimodoMotionApplier.cs`: applies mapped motion to an Actor clone or reports mapping failure.
- Create `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/KimodoDebugSkeletonRenderer.cs`: draws fallback 77-joint preview.
- Create `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/Editor/KimodoPreviewImporter.cs`: runs Python exporter and creates `KimodoMotionAsset`.
- Create `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/Editor/KimodoPreviewSession.cs`: owns hidden preview context, camera, Actor clone, and debug renderer.
- Create `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/Editor/KimodoPipelineRunner.cs`: builds and runs second-layer Python/conda commands.
- Create `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/Editor/KimodoMotionBrowserWindow.cs`: browser UI.
- Create `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/Editor/KimodoMotionBrowserSmoke.cs`: batchmode validation entry points.

Generated Unity asset folder:

- Create on demand `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Motions/<motion_name>/`.

## Task 1: Kimodo Unity Preview Exporter

**Files:**
- Create: `D:/Projects/SchoolWorkProjects/kimodo/kimodo_gui/export_unity_preview.py`
- Create: `D:/Projects/SchoolWorkProjects/kimodo/tests/test_export_unity_preview.py`

**Interfaces:**
- Produces: `export_unity_preview(input_npz: Path, output_dir: Path, source_bvh: Path | None = None, motion_name: str | None = None) -> dict[str, object]`
- Produces CLI: `python -m kimodo_gui.export_unity_preview --input_npz <path> --output_dir <dir> [--source_bvh <path>] [--motion_name <name>]`
- Produces package files: `manifest.json`, `skeleton.json`, `joint_positions_world.bytes`, `local_rotations_xyzw.bytes`, `global_rotations_xyzw.bytes`, `root_positions.bytes`, `foot_contacts.bytes`

- [ ] **Step 1: Write failing exporter tests**

Create `D:/Projects/SchoolWorkProjects/kimodo/tests/test_export_unity_preview.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kimodo_gui.export_unity_preview import export_unity_preview


def write_mock_kimodo_npz(path: Path, frames: int = 4, joints: int = 77) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    local = np.tile(np.eye(3, dtype=np.float32), (frames, joints, 1, 1))
    global_rot = np.tile(np.eye(3, dtype=np.float32), (frames, joints, 1, 1))
    positions = np.zeros((frames, joints, 3), dtype=np.float32)
    positions[:, :, 1] = np.linspace(0.0, 1.0, joints, dtype=np.float32)
    root_positions = np.zeros((frames, 3), dtype=np.float32)
    root_positions[:, 2] = np.arange(frames, dtype=np.float32)
    foot_contacts = np.zeros((frames, 6), dtype=np.bool_)
    foot_contacts[:, 0] = True
    np.savez(
        path,
        local_rot_mats=local,
        global_rot_mats=global_rot,
        posed_joints=positions,
        root_positions=root_positions,
        foot_contacts=foot_contacts,
        global_root_heading=np.zeros((frames, 2), dtype=np.float32),
    )


def test_export_unity_preview_writes_manifest_skeleton_and_streams(tmp_path):
    input_npz = tmp_path / "raw" / "walk_turn_wave_01.npz"
    output_dir = tmp_path / "unity" / "walk_turn_wave_01"
    write_mock_kimodo_npz(input_npz)

    summary = export_unity_preview(input_npz=input_npz, output_dir=output_dir)

    assert summary["frameCount"] == 4
    assert summary["jointCount"] == 77
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    skeleton = json.loads((output_dir / "skeleton.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "kimodo_motion_preview"
    assert manifest["version"] == 1
    assert manifest["sourceNpzPath"] == str(input_npz.resolve())
    assert manifest["streams"]["jointPositionsWorld"]["shape"] == [4, 77, 3]
    assert skeleton["jointCount"] == 77
    assert len(skeleton["jointNames"]) == 77
    assert len(skeleton["parentIndices"]) == 77
    assert (output_dir / "joint_positions_world.bytes").stat().st_size == 4 * 77 * 3 * 4
    assert (output_dir / "local_rotations_xyzw.bytes").stat().st_size == 4 * 77 * 4 * 4
    assert (output_dir / "global_rotations_xyzw.bytes").stat().st_size == 4 * 77 * 4 * 4
    assert (output_dir / "root_positions.bytes").stat().st_size == 4 * 3 * 4
    assert (output_dir / "foot_contacts.bytes").stat().st_size == 4 * 6


def test_export_unity_preview_cli(tmp_path, capsys):
    input_npz = tmp_path / "raw" / "clip.npz"
    output_dir = tmp_path / "preview" / "clip"
    write_mock_kimodo_npz(input_npz, frames=2)

    from kimodo_gui.export_unity_preview import main

    main(["--input_npz", str(input_npz), "--output_dir", str(output_dir)])
    captured = capsys.readouterr()
    assert "kimodo_motion_preview" in captured.out
    assert (output_dir / "manifest.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
conda run --prefix D:\Anaconda\envs\kimodo_gui --no-capture-output pytest tests/test_export_unity_preview.py -q
```

Expected: FAIL because `kimodo_gui.export_unity_preview` does not exist.

- [ ] **Step 3: Implement exporter**

Create `D:/Projects/SchoolWorkProjects/kimodo/kimodo_gui/export_unity_preview.py` with these concrete behaviors:

```python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


KIMODO_PREVIEW_FORMAT = "kimodo_motion_preview"
KIMODO_PREVIEW_VERSION = 1


def kimodo_joint_names() -> list[str]:
    names = [
        "Root", "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd",
        "Jaw", "LeftEye", "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumbEnd",
        "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4", "LeftHandIndexEnd",
        "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandMiddleEnd",
        "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4", "LeftHandRingEnd",
        "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4", "LeftHandPinkyEnd",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumbEnd",
        "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4", "RightHandIndexEnd",
        "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4", "RightHandMiddleEnd",
        "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4", "RightHandRingEnd",
        "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4", "RightHandPinkyEnd",
        "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "LeftToeEnd",
        "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    ]
    if len(names) < 77:
        names.extend(f"Joint{i:02d}" for i in range(len(names), 77))
    return names[:77]


def kimodo_parent_indices() -> list[int]:
    parents = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 4, 12, 13, 14, 15]
    parents.extend([15, 16, 17, 18, 15, 20, 21, 22, 23, 15, 25, 26, 27, 28, 15, 30, 31, 32, 33])
    parents.extend([15, 35, 36, 37, 38, 4, 40, 41, 42])
    parents.extend([42, 43, 44, 45, 42, 47, 48, 49, 50, 42, 52, 53, 54, 55, 42, 57, 58, 59, 60])
    parents.extend([42, 62, 63, 64, 65, 1, 67, 68, 69, 70, 1, 72, 73, 74])
    if len(parents) < 77:
        parents.extend([0] * (77 - len(parents)))
    return parents[:77]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rotation_matrices_to_xyzw(mats: np.ndarray) -> np.ndarray:
    mats = np.asarray(mats, dtype=np.float32)
    flat = mats.reshape(-1, 3, 3)
    q = np.empty((flat.shape[0], 4), dtype=np.float32)
    for i, m in enumerate(flat):
        trace = float(m[0, 0] + m[1, 1] + m[2, 2])
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            q[i, 3] = 0.25 * s
            q[i, 0] = (m[2, 1] - m[1, 2]) / s
            q[i, 1] = (m[0, 2] - m[2, 0]) / s
            q[i, 2] = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q[i, 3] = (m[2, 1] - m[1, 2]) / s
            q[i, 0] = 0.25 * s
            q[i, 1] = (m[0, 1] + m[1, 0]) / s
            q[i, 2] = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q[i, 3] = (m[0, 2] - m[2, 0]) / s
            q[i, 0] = (m[0, 1] + m[1, 0]) / s
            q[i, 1] = 0.25 * s
            q[i, 2] = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q[i, 3] = (m[1, 0] - m[0, 1]) / s
            q[i, 0] = (m[0, 2] + m[2, 0]) / s
            q[i, 1] = (m[1, 2] + m[2, 1]) / s
            q[i, 2] = 0.25 * s
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(norm, 1e-8)
    return q.reshape(*mats.shape[:2], 4).astype(np.float32)


def write_stream(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    contiguous = np.ascontiguousarray(array)
    path.write_bytes(contiguous.tobytes(order="C"))
    return sha256_file(path)


def export_unity_preview(
    input_npz: Path,
    output_dir: Path,
    source_bvh: Path | None = None,
    motion_name: str | None = None,
) -> dict[str, object]:
    input_npz = Path(input_npz).resolve()
    output_dir = Path(output_dir).resolve()
    source_bvh = Path(source_bvh).resolve() if source_bvh else input_npz.with_suffix(".bvh")
    motion_name = motion_name or input_npz.stem
    with np.load(input_npz, allow_pickle=True) as data:
        positions = np.asarray(data["posed_joints"], dtype=np.float32)
        local_mats = np.asarray(data["local_rot_mats"], dtype=np.float32)
        global_mats = np.asarray(data["global_rot_mats"], dtype=np.float32)
        root_positions = np.asarray(data["root_positions"], dtype=np.float32)
        foot_contacts = np.asarray(data["foot_contacts"], dtype=np.uint8)

    frame_count, joint_count, dim = positions.shape
    if dim != 3 or joint_count != 77:
        raise ValueError(f"posed_joints must be [T,77,3], got {positions.shape}")
    if local_mats.shape != (frame_count, joint_count, 3, 3):
        raise ValueError(f"local_rot_mats shape mismatch: {local_mats.shape}")
    if global_mats.shape != (frame_count, joint_count, 3, 3):
        raise ValueError(f"global_rot_mats shape mismatch: {global_mats.shape}")
    if root_positions.shape != (frame_count, 3):
        raise ValueError(f"root_positions must be [T,3], got {root_positions.shape}")
    if foot_contacts.shape != (frame_count, 6):
        raise ValueError(f"foot_contacts must be [T,6], got {foot_contacts.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)
    local_quats = rotation_matrices_to_xyzw(local_mats)
    global_quats = rotation_matrices_to_xyzw(global_mats)
    stream_defs: dict[str, dict[str, Any]] = {}
    for name, filename, dtype, shape, values in [
        ("jointPositionsWorld", "joint_positions_world.bytes", "float32", [frame_count, joint_count, 3], positions),
        ("localRotations", "local_rotations_xyzw.bytes", "float32", [frame_count, joint_count, 4], local_quats),
        ("globalRotations", "global_rotations_xyzw.bytes", "float32", [frame_count, joint_count, 4], global_quats),
        ("rootPositions", "root_positions.bytes", "float32", [frame_count, 3], root_positions),
        ("footContacts", "foot_contacts.bytes", "uint8", [frame_count, 6], foot_contacts),
    ]:
        stream_defs[name] = {
            "path": filename,
            "dtype": dtype,
            "shape": shape,
            "sha256": write_stream(output_dir / filename, values),
        }
        if name.endswith("Rotations"):
            stream_defs[name]["quaternionOrder"] = "xyzw"

    skeleton = {
        "format": "kimodo_skeleton",
        "version": 1,
        "jointCount": joint_count,
        "jointNames": kimodo_joint_names(),
        "parentIndices": kimodo_parent_indices(),
        "smpl24Mapping": {},
        "humanoidMappingCandidates": {},
    }
    (output_dir / "skeleton.json").write_text(json.dumps(skeleton, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "format": KIMODO_PREVIEW_FORMAT,
        "version": KIMODO_PREVIEW_VERSION,
        "motionName": motion_name,
        "sourceKind": "kimodo_raw_npz",
        "sourceNpzPath": str(input_npz),
        "sourceBvhPath": str(source_bvh) if source_bvh.exists() else "",
        "sourceSha256": sha256_file(input_npz),
        "fps": 30.0,
        "frameCount": int(frame_count),
        "jointCount": int(joint_count),
        "unit": "meter",
        "coordinateSystem": "unity_world",
        "rootJointIndex": 0,
        "streams": stream_defs,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"format": KIMODO_PREVIEW_FORMAT, "frameCount": int(frame_count), "jointCount": int(joint_count), "outputDir": str(output_dir)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a Kimodo raw NPZ as a Unity preview motion package.")
    parser.add_argument("--input_npz", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--source_bvh", default=None, type=Path)
    parser.add_argument("--motion_name", default=None, type=str)
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = build_arg_parser().parse_args(argv)
    summary = export_unity_preview(args.input_npz, args.output_dir, args.source_bvh, args.motion_name)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run exporter tests**

Run:

```powershell
conda run --prefix D:\Anaconda\envs\kimodo_gui --no-capture-output pytest tests/test_export_unity_preview.py -q
```

Expected: PASS.

- [ ] **Step 5: Test against the real sample**

Run:

```powershell
conda run --prefix D:\Anaconda\envs\kimodo_gui --no-capture-output python -m kimodo_gui.export_unity_preview `
  --input_npz D:\Projects\SchoolWorkProjects\kimodo\artifacts\kimodo_raw\walk_turn_wave_01.npz `
  --source_bvh D:\Projects\SchoolWorkProjects\kimodo\artifacts\kimodo_raw\walk_turn_wave_01.bvh `
  --output_dir D:\Projects\SchoolWorkProjects\kimodo\artifacts\unity_preview\walk_turn_wave_01
```

Expected: JSON summary contains `"frameCount": 180` and output package files exist.

- [ ] **Step 6: Commit Kimodo exporter**

Run:

```powershell
git -C D:\Projects\SchoolWorkProjects\kimodo add kimodo_gui/export_unity_preview.py tests/test_export_unity_preview.py
git -C D:\Projects\SchoolWorkProjects\kimodo commit -m "feat: export kimodo unity preview packages"
```

Expected: commit succeeds in the Kimodo repository.

## Task 2: Unity Motion Asset and Reader

**Files:**
- Create: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/KimodoMotionAsset.cs`
- Create: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/KimodoMotionManifest.cs`
- Create: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/KimodoMotionFrame.cs`
- Create: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/KimodoMotionReader.cs`
- Create: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/Editor/KimodoMotionBrowserSmoke.cs`

**Interfaces:**
- Produces: `KimodoMotionAsset : ScriptableObject`
- Produces: `KimodoMotionReader.TryCreate(KimodoMotionAsset asset, out KimodoMotionReader reader, out string error) -> bool`
- Produces: `KimodoMotionReader.TryReadFrame(int frameIndex, KimodoMotionFrame frame, out string error) -> bool`
- Produces batch validation: `RealtimePose.EditorTools.KimodoMotionBrowserSmoke.ValidateReader()`

- [ ] **Step 1: Add smoke method that fails before runtime classes exist**

Create `KimodoMotionBrowserSmoke.cs` with a method that references the planned types:

```csharp
using System;
using UnityEditor;
using UnityEngine;

namespace RealtimePose.EditorTools
{
    public static class KimodoMotionBrowserSmoke
    {
        public static void ValidateReader()
        {
            KimodoMotionAsset asset = ScriptableObject.CreateInstance<KimodoMotionAsset>();
            asset.DisplayName = "empty";
            if (KimodoMotionReader.TryCreate(asset, out KimodoMotionReader _, out string error))
            {
                throw new InvalidOperationException("Empty asset should not create a reader.");
            }
            if (string.IsNullOrEmpty(error))
            {
                throw new InvalidOperationException("Reader failure should explain the missing manifest.");
            }
            Debug.Log("[KimodoMotionBrowser] ValidateReader passed.");
        }
    }
}
```

- [ ] **Step 2: Compile to verify failure**

Run:

```powershell
$UnityExe = @(
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1c1\Editor\Unity.exe',
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1\Editor\Unity.exe',
  'C:\Program Files\Unity 2022.3.17f1c1\Editor\Unity.exe'
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $UnityExe) { throw 'Unity executable not found.' }
& $UnityExe -batchmode -quit -projectPath 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' -logFile 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-kimodo-reader-red.log'
if ($LASTEXITCODE -eq 0) { throw 'Expected compile failure before KimodoMotionAsset exists.' }
```

Expected: compile fails because `KimodoMotionAsset` and `KimodoMotionReader` do not exist.

- [ ] **Step 3: Implement asset and DTO classes**

Create `KimodoMotionAsset.cs`:

```csharp
using UnityEngine;

namespace RealtimePose
{
    public enum KimodoMotionStatus
    {
        New,
        Approved,
        Rejected,
        Converted,
        Broken
    }

    [CreateAssetMenu(menuName = "RealtimePose/Kimodo Motion Asset", fileName = "KimodoMotionAsset")]
    public sealed class KimodoMotionAsset : ScriptableObject
    {
        public string DisplayName;
        public string SourceNpzPath;
        public string SourceBvhPath;
        public string SourceSha256;
        public float Fps = 30f;
        public int FrameCount;
        public int JointCount = 77;
        public TextAsset Manifest;
        public TextAsset Skeleton;
        public TextAsset JointPositionsWorld;
        public TextAsset LocalRotationsXyzw;
        public TextAsset GlobalRotationsXyzw;
        public TextAsset RootPositions;
        public TextAsset FootContacts;
        public KimodoMotionStatus Status = KimodoMotionStatus.New;
        public string[] Tags = new string[0];
        [TextArea(2, 6)]
        public string Notes;
        public Object DefaultBindingProfile;
        public TextAsset ConvertedReplayJson;
        public string ConvertedSourcePath;
    }
}
```

Create `KimodoMotionManifest.cs` with `[Serializable]` DTOs matching `manifest.json` and `skeleton.json`. Use explicit fields: `format`, `version`, `motionName`, `sourceNpzPath`, `fps`, `frameCount`, `jointCount`, `streams`, `jointNames`, `parentIndices`.

Create `KimodoMotionFrame.cs`:

```csharp
using UnityEngine;

namespace RealtimePose
{
    public sealed class KimodoMotionFrame
    {
        public readonly Vector3[] JointPositions;
        public readonly Quaternion[] LocalRotations;
        public readonly Quaternion[] GlobalRotations;
        public readonly Vector3[] RootPositions;
        public readonly byte[] FootContacts;

        public KimodoMotionFrame(int jointCount)
        {
            JointPositions = new Vector3[jointCount];
            LocalRotations = new Quaternion[jointCount];
            GlobalRotations = new Quaternion[jointCount];
            RootPositions = new Vector3[1];
            FootContacts = new byte[6];
        }
    }
}
```

- [ ] **Step 4: Implement reader**

Create `KimodoMotionReader.cs` with:

- `TryCreate` rejects missing manifest/skeleton/streams with a concrete `error`.
- It uses `JsonUtility.FromJson<KimodoMotionManifest>()`.
- It validates `FrameCount > 0`, `JointCount == 77`, byte lengths, and `TextAsset.bytes`.
- It decodes little-endian float32 streams with `BitConverter.ToSingle`.
- It reads quaternion order `xyzw` into `new Quaternion(x, y, z, w)`.

- [ ] **Step 5: Compile and run smoke method**

Run:

```powershell
$UnityExe = @(
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1c1\Editor\Unity.exe',
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1\Editor\Unity.exe',
  'C:\Program Files\Unity 2022.3.17f1c1\Editor\Unity.exe'
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $UnityExe) { throw 'Unity executable not found.' }
& $UnityExe -batchmode -quit -projectPath 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' -executeMethod RealtimePose.EditorTools.KimodoMotionBrowserSmoke.ValidateReader -logFile 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-kimodo-reader.log'
if ($LASTEXITCODE -ne 0) { Get-Content -Tail 200 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-kimodo-reader.log'; exit $LASTEXITCODE }
```

Expected: exit code 0 and log contains `ValidateReader passed`.

- [ ] **Step 6: Commit Unity reader slice**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' add -- `
  'Assets/Projects/RealtimePose/KimodoBrowser'
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' commit -m 'feat(realtimepose): add kimodo motion asset reader'
```

Expected: commit succeeds in the Unity repository.

## Task 3: Unity Preview Importer

**Files:**
- Create: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/Editor/KimodoPreviewImporter.cs`
- Modify: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/Editor/KimodoMotionBrowserSmoke.cs`

**Interfaces:**
- Produces: `KimodoPreviewImporter.ImportPreview(string inputNpzPath, string sourceBvhPath, string unityMotionRoot, out KimodoMotionAsset asset, out string error) -> bool`
- Produces menu utility internally used by Browser.

- [ ] **Step 1: Extend smoke with importer symbol reference**

Add this method to `KimodoMotionBrowserSmoke.cs`:

```csharp
public static void ValidateImporterSymbols()
{
    if (typeof(KimodoPreviewImporter).Name != "KimodoPreviewImporter")
    {
        throw new InvalidOperationException("Importer type missing.");
    }
    Debug.Log("[KimodoMotionBrowser] ValidateImporterSymbols passed.");
}
```

- [ ] **Step 2: Compile to verify failure**

Run Unity batchmode with `-executeMethod RealtimePose.EditorTools.KimodoMotionBrowserSmoke.ValidateImporterSymbols`.

Expected: compile fails because `KimodoPreviewImporter` does not exist.

- [ ] **Step 3: Implement importer**

Create `KimodoPreviewImporter.cs`:

- Resolve package name from NPZ filename.
- Create target folder under `Assets/Projects/RealtimePose/KimodoBrowser/Motions/<name>`.
- Run:

```powershell
conda run --no-capture-output --prefix D:\Anaconda\envs\kimodo_gui python -m kimodo_gui.export_unity_preview --input_npz <npz> --source_bvh <bvh> --output_dir <temp>
```

- Copy `manifest.json`, `skeleton.json`, and `.bytes` files from temp package into the Unity asset folder.
- `AssetDatabase.ImportAsset` the folder.
- Create `<motion_name>.asset` as `KimodoMotionAsset`.
- Assign all TextAsset references with `AssetDatabase.LoadAssetAtPath<TextAsset>()`.
- Set `DisplayName`, `SourceNpzPath`, `SourceBvhPath`, `Fps`, `FrameCount`, `JointCount`, and `Status`.

- [ ] **Step 4: Compile importer**

Run:

```powershell
$UnityExe = @(
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1c1\Editor\Unity.exe',
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1\Editor\Unity.exe',
  'C:\Program Files\Unity 2022.3.17f1c1\Editor\Unity.exe'
) | Where-Object { Test-Path $_ } | Select-Object -First 1
& $UnityExe -batchmode -quit -projectPath 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' -executeMethod RealtimePose.EditorTools.KimodoMotionBrowserSmoke.ValidateImporterSymbols -logFile 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-kimodo-importer.log'
if ($LASTEXITCODE -ne 0) { Get-Content -Tail 200 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-kimodo-importer.log'; exit $LASTEXITCODE }
```

Expected: exit code 0.

- [ ] **Step 5: Manually import real sample through importer**

Add a temporary one-shot smoke method only if needed:

```csharp
public static void ImportWalkTurnWaveSample()
{
    string npz = @"D:\Projects\SchoolWorkProjects\kimodo\artifacts\kimodo_raw\walk_turn_wave_01.npz";
    string bvh = @"D:\Projects\SchoolWorkProjects\kimodo\artifacts\kimodo_raw\walk_turn_wave_01.bvh";
    string root = "Assets/Projects/RealtimePose/KimodoBrowser/Motions";
    if (!KimodoPreviewImporter.ImportPreview(npz, bvh, root, out KimodoMotionAsset asset, out string error))
    {
        throw new InvalidOperationException(error);
    }
    if (asset.FrameCount != 180 || asset.JointCount != 77)
    {
        throw new InvalidOperationException("Imported sample metadata mismatch.");
    }
}
```

Run it once in batchmode, then keep the method if it is useful as a repeatable smoke; otherwise remove it before commit.

- [ ] **Step 6: Commit importer slice**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' add -- 'Assets/Projects/RealtimePose/KimodoBrowser'
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' commit -m 'feat(realtimepose): import kimodo preview assets'
```

## Task 4: Preview Session and Debug Skeleton

**Files:**
- Create: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/KimodoDebugSkeletonRenderer.cs`
- Create: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/Editor/KimodoPreviewSession.cs`
- Modify: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/Editor/KimodoMotionBrowserSmoke.cs`

**Interfaces:**
- Produces: `KimodoPreviewSession.LoadMotion(KimodoMotionAsset asset) -> bool`
- Produces: `KimodoPreviewSession.Seek(int frameIndex) -> void`
- Produces: `KimodoPreviewSession.Render(Rect rect, bool showRootTrail, bool showFootContacts) -> void`

- [ ] **Step 1: Add session smoke method**

Add:

```csharp
public static void ValidatePreviewSessionSymbols()
{
    using (KimodoPreviewSession session = new KimodoPreviewSession())
    {
        session.Seek(0);
    }
    Debug.Log("[KimodoMotionBrowser] ValidatePreviewSessionSymbols passed.");
}
```

Expected first compile: FAIL because `KimodoPreviewSession` does not exist.

- [ ] **Step 2: Implement debug skeleton renderer**

`KimodoDebugSkeletonRenderer` should:

- Own line material and sphere/capsule primitive meshes created in preview context.
- Accept `KimodoMotionFrame` plus `parentIndices`.
- Update joint transforms and bone line transforms per frame.
- Color foot contact markers using `FootContacts`.

- [ ] **Step 3: Implement preview session**

`KimodoPreviewSession` should:

- Implement `IDisposable`.
- Own a hidden root `GameObject`, camera, light, and renderer objects.
- Load a `KimodoMotionReader`.
- Maintain `CurrentFrameIndex`, `IsPlaying`, `PlaybackSpeed`, `Loop`.
- In `Render(Rect rect, bool showRootTrail, bool showFootContacts)`, advance time only when playing and repaint via caller.
- Use `PreviewRenderUtility` for first implementation to avoid touching scene objects.

- [ ] **Step 4: Compile session**

Run Unity batchmode `-executeMethod RealtimePose.EditorTools.KimodoMotionBrowserSmoke.ValidatePreviewSessionSymbols`.

Expected: exit code 0.

- [ ] **Step 5: Commit preview session**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' add -- 'Assets/Projects/RealtimePose/KimodoBrowser'
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' commit -m 'feat(realtimepose): preview kimodo debug skeletons'
```

## Task 5: SMPL Actor Template Preview

**Files:**
- Create: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/KimodoMotionApplier.cs`
- Modify: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/Editor/KimodoPreviewSession.cs`
- Modify: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/Editor/KimodoMotionBrowserSmoke.cs`

**Interfaces:**
- Produces: `KimodoMotionApplier.Bind(Actor actor, KimodoMotionSkeleton skeleton, out string report) -> bool`
- Produces: `KimodoMotionApplier.Apply(KimodoMotionFrame frame, float deltaTime) -> bool`
- Consumes: `AI4Animation.Actor`, `DefaultPoseSkeletons`, `KimodoMotionFrame`

- [ ] **Step 1: Add applier smoke method**

Add:

```csharp
public static void ValidateApplierSymbols()
{
    KimodoMotionApplier applier = new KimodoMotionApplier();
    if (applier == null)
    {
        throw new InvalidOperationException("Applier missing.");
    }
    Debug.Log("[KimodoMotionBrowser] ValidateApplierSymbols passed.");
}
```

Expected first compile: FAIL because `KimodoMotionApplier` does not exist.

- [ ] **Step 2: Implement mapping table**

Inside `KimodoMotionApplier`, define explicit first-version mapping:

```csharp
private static readonly (string targetBone, string[] kimodoCandidates)[] Smpl24Mapping =
{
    ("pelvis", new[] {"Hips", "Root"}),
    ("left_hip", new[] {"LeftUpLeg"}),
    ("right_hip", new[] {"RightUpLeg"}),
    ("spine1", new[] {"Spine1"}),
    ("left_knee", new[] {"LeftLeg"}),
    ("right_knee", new[] {"RightLeg"}),
    ("spine2", new[] {"Spine2"}),
    ("left_ankle", new[] {"LeftFoot"}),
    ("right_ankle", new[] {"RightFoot"}),
    ("spine3", new[] {"Chest"}),
    ("left_foot", new[] {"LeftToeBase", "LeftFoot"}),
    ("right_foot", new[] {"RightToeBase", "RightFoot"}),
    ("neck", new[] {"Neck1", "Neck2"}),
    ("left_collar", new[] {"LeftShoulder"}),
    ("right_collar", new[] {"RightShoulder"}),
    ("head", new[] {"Head"}),
    ("left_shoulder", new[] {"LeftArm"}),
    ("right_shoulder", new[] {"RightArm"}),
    ("left_elbow", new[] {"LeftForeArm"}),
    ("right_elbow", new[] {"RightForeArm"}),
    ("left_wrist", new[] {"LeftHand"}),
    ("right_wrist", new[] {"RightHand"}),
    ("left_hand", new[] {"LeftHand"}),
    ("right_hand", new[] {"RightHand"})
};
```

- [ ] **Step 3: Implement Actor clone binding in preview session**

Add to `KimodoPreviewSession`:

```csharp
public void SetActorTemplate(AI4Animation.Actor actorTemplate)
```

Behavior:

- If `actorTemplate == null`, destroy existing clone and use debug skeleton.
- Instantiate a hidden clone in the preview utility root.
- Bind `KimodoMotionApplier`.
- If binding reports fewer than 12 mapped bones, destroy clone and fallback to debug skeleton.

- [ ] **Step 4: Compile applier**

Run Unity batchmode `-executeMethod RealtimePose.EditorTools.KimodoMotionBrowserSmoke.ValidateApplierSymbols`.

Expected: exit code 0.

- [ ] **Step 5: Commit actor preview slice**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' add -- 'Assets/Projects/RealtimePose/KimodoBrowser'
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' commit -m 'feat(realtimepose): preview kimodo motions on actor templates'
```

## Task 6: Browser Window UI

**Files:**
- Create: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/Editor/KimodoMotionBrowserWindow.cs`
- Modify: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/Editor/KimodoMotionBrowserSmoke.cs`

**Interfaces:**
- Produces menu: `RealtimePose/Kimodo Motion Browser`
- Consumes: `KimodoPreviewImporter`, `KimodoPreviewSession`, `KimodoMotionReader`

- [ ] **Step 1: Add browser window smoke symbol**

Add:

```csharp
public static void ValidateBrowserWindowSymbols()
{
    EditorWindow window = EditorWindow.CreateInstance<KimodoMotionBrowserWindow>();
    window.Close();
    Debug.Log("[KimodoMotionBrowser] ValidateBrowserWindowSymbols passed.");
}
```

Expected first compile: FAIL because `KimodoMotionBrowserWindow` does not exist.

- [ ] **Step 2: Implement window layout**

`KimodoMotionBrowserWindow` should:

- Add `[MenuItem("RealtimePose/Kimodo Motion Browser")]`.
- Store `searchFolder = "Assets/Projects/RealtimePose/KimodoBrowser/Motions"`.
- On refresh, call `AssetDatabase.FindAssets("t:KimodoMotionAsset", new[] { searchFolder })`.
- Left panel: motion list and `Refresh`.
- Center panel: preview rect and status.
- Bottom: play/pause button, frame slider, speed field, loop toggle, root trail/contact toggles.
- Right panel: raw NPZ path selector, Actor template object field, `Import Preview`, status/tags/notes fields, pipeline placeholder buttons.

- [ ] **Step 3: Compile window**

Run Unity batchmode `-executeMethod RealtimePose.EditorTools.KimodoMotionBrowserSmoke.ValidateBrowserWindowSymbols`.

Expected: exit code 0.

- [ ] **Step 4: Manual UI verification**

Open Unity Editor:

```powershell
$UnityExe = @(
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1c1\Editor\Unity.exe',
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1\Editor\Unity.exe',
  'C:\Program Files\Unity 2022.3.17f1c1\Editor\Unity.exe'
) | Where-Object { Test-Path $_ } | Select-Object -First 1
Start-Process -FilePath $UnityExe -ArgumentList @('-projectPath', 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity') -WindowStyle Hidden
```

Expected:

- Menu `RealtimePose/Kimodo Motion Browser` opens the window.
- `Refresh` lists imported `KimodoMotionAsset`.
- Selecting `walk_turn_wave_01` shows preview controls.
- Play/pause changes frame index without entering Play Mode.

- [ ] **Step 5: Commit UI slice**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' add -- 'Assets/Projects/RealtimePose/KimodoBrowser'
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' commit -m 'feat(realtimepose): add kimodo motion browser window'
```

## Task 7: DiffusionPoser Pipeline Buttons

**Files:**
- Create: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/Editor/KimodoPipelineRunner.cs`
- Modify: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/Editor/KimodoMotionBrowserWindow.cs`
- Modify: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/KimodoMotionAsset.cs`
- Modify: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/Scripts/Editor/KimodoMotionBrowserSmoke.cs`

**Interfaces:**
- Produces: `KimodoPipelineRunner.TryBuildPseudoAmassCommand(KimodoMotionAsset asset, out string[] command, out string error) -> bool`
- Produces: `KimodoPipelineRunner.TryBuildRealtimePoseReplayCommand(KimodoMotionAsset asset, out string[] command, out string error) -> bool`
- Produces: `KimodoPipelineRunner.RunAsync(string[] command, Action<string> onLine, Action<int> onExit) -> void`

- [ ] **Step 1: Add command builder smoke**

Add:

```csharp
public static void ValidatePipelineCommandBuilders()
{
    KimodoMotionAsset asset = ScriptableObject.CreateInstance<KimodoMotionAsset>();
    asset.DisplayName = "walk_turn_wave_01";
    asset.SourceNpzPath = @"D:\Projects\SchoolWorkProjects\kimodo\artifacts\kimodo_raw\walk_turn_wave_01.npz";
    if (!KimodoPipelineRunner.TryBuildPseudoAmassCommand(asset, out string[] command, out string error))
    {
        throw new InvalidOperationException(error);
    }
    string joined = string.Join(" ", command);
    if (!joined.Contains("--no-capture-output") || !joined.Contains("kimodo_gui.pseudo_amass"))
    {
        throw new InvalidOperationException("Pseudo-AMASS command must call kimodo_gui.pseudo_amass through conda run --no-capture-output.");
    }
    Debug.Log("[KimodoMotionBrowser] ValidatePipelineCommandBuilders passed.");
}
```

Expected first compile: FAIL because `KimodoPipelineRunner` does not exist.

- [ ] **Step 2: Implement command builder**

Create `KimodoPipelineRunner.cs`:

- Use `ProcessStartInfo` with `UseShellExecute = false`, redirected stdout/stderr.
- Do not shell-join commands for execution.
- `TryBuildPseudoAmassCommand` should reject assets with an empty `SourceNpzPath`. For valid assets it should produce:

```csharp
new[]
{
    "conda", "run", "--no-capture-output", "--prefix", @"D:\Anaconda\envs\kimodo_gui",
    "python", "-m", "kimodo_gui.pseudo_amass",
    "--input_dir", @"D:\Projects\SchoolWorkProjects\kimodo\artifacts\kimodo_raw",
    "--output_dir", @"D:\Projects\SchoolWorkProjects\kimodo\artifacts\pseudo_amass\walk_turn_wave_01"
}
```

- `TryBuildRealtimePoseReplayCommand` should reject assets with an empty `ConvertedSourcePath`. For valid converted source assets it should produce:

```csharp
new[]
{
    "conda", "run", "--no-capture-output", "-n", "diffusionposer5070",
    "python", "-m", "export.write_unity_replay_stream",
    "--source_npz", asset.ConvertedSourcePath,
    "--schema", "realtime_pose_stationary5_v1"
}
```

If the current Kimodo raw NPZ cannot be converted by the existing pseudo-AMASS converter, the button must show the returned `error` string and must not mark the asset as converted.

- [ ] **Step 3: Wire window buttons**

In `KimodoMotionBrowserWindow` right panel:

- `Convert to pseudo-AMASS`
- `Export RealtimePose Replay`
- `Build Source/Task/Normalizer`

Each button:

- Disables while a command is running.
- Streams logs into a scroll view.
- Updates `KimodoMotionAsset.Status` only when exit code is 0.

- [ ] **Step 4: Compile pipeline**

Run Unity batchmode `-executeMethod RealtimePose.EditorTools.KimodoMotionBrowserSmoke.ValidatePipelineCommandBuilders`.

Expected: exit code 0.

- [ ] **Step 5: Commit pipeline slice**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' add -- 'Assets/Projects/RealtimePose/KimodoBrowser'
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' commit -m 'feat(realtimepose): add kimodo pipeline actions'
```

## Task 8: End-to-End Verification and Docs

**Files:**
- Create: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/KimodoBrowser/README.md`
- Modify if needed: `D:/Projects/SchoolWorkProjects/kimodo/README.md`

**Interfaces:**
- Verifies the full workflow from raw sample to Unity preview asset and Browser playback.

- [ ] **Step 1: Write Unity browser README**

Create `KimodoBrowser/README.md` with:

```markdown
# Kimodo Motion Browser

Menu: `RealtimePose/Kimodo Motion Browser`

Default raw directory:
`D:\Projects\SchoolWorkProjects\kimodo\artifacts\kimodo_raw`

Imported preview assets are stored under:
`Assets/Projects/RealtimePose/KimodoBrowser/Motions`

The tool imports preview caches only. It does not copy raw `.npz` or `.bvh` files into Unity.

First version supports SMPL/body.fbx `AI4Animation.Actor` templates. If no Actor template is assigned, the preview uses the Kimodo 77-joint debug skeleton.
```

- [ ] **Step 2: Run Kimodo Python tests**

Run:

```powershell
conda run --prefix D:\Anaconda\envs\kimodo_gui --no-capture-output pytest tests/test_export_unity_preview.py -q
```

Expected: PASS.

- [ ] **Step 3: Run Kimodo full tests if the environment is available**

Run:

```powershell
conda run --prefix D:\Anaconda\envs\kimodo_gui --no-capture-output pytest -q
```

Expected: PASS. If this fails due unrelated existing GUI/Docker environment assumptions, record failing tests in final notes and keep targeted exporter tests as the required gate for this feature.

- [ ] **Step 4: Run Unity compile smoke**

Run:

```powershell
$UnityExe = @(
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1c1\Editor\Unity.exe',
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1\Editor\Unity.exe',
  'C:\Program Files\Unity 2022.3.17f1c1\Editor\Unity.exe'
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $UnityExe) { throw 'Unity executable not found.' }
& $UnityExe -batchmode -quit -projectPath 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' -logFile 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-kimodo-browser-final.log'
if ($LASTEXITCODE -ne 0) { Get-Content -Tail 260 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-kimodo-browser-final.log'; exit $LASTEXITCODE }
```

Expected: exit code 0.

- [ ] **Step 5: Manual end-to-end check**

In Unity:

1. Open `RealtimePose/Kimodo Motion Browser`.
2. Import `D:\Projects\SchoolWorkProjects\kimodo\artifacts\kimodo_raw\walk_turn_wave_01.npz`.
3. Confirm `Assets/Projects/RealtimePose/KimodoBrowser/Motions/walk_turn_wave_01/walk_turn_wave_01.asset` exists.
4. Select the imported asset.
5. Press Play in the window timeline.
6. Confirm the preview animates without entering Play Mode.
7. Assign a SMPL/body.fbx Actor template.
8. Confirm the preview switches from debug skeleton to Actor clone or reports missing mappings clearly.

- [ ] **Step 6: Commit final docs**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' add -- 'Assets/Projects/RealtimePose/KimodoBrowser/README.md'
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' commit -m 'docs(realtimepose): document kimodo motion browser'
```

Expected: Unity docs commit succeeds.

## Self-Review

- Spec coverage: Tasks cover Python preview package export, per-motion SO assets, no global library SO, Unity asset scanning, scene-independent preview, Actor template fallback behavior, second-layer pipeline command buttons, and end-to-end verification.
- Placeholder scan: The only intentionally deferred area is full Humanoid retarget, matching the approved spec. Pipeline module names may require existing converter extension; Task 7 requires disabled button text if the runnable converter is not ready, so it must not fake success.
- Type consistency: `KimodoMotionAsset`, `KimodoMotionReader`, `KimodoMotionFrame`, `KimodoPreviewImporter`, `KimodoPreviewSession`, `KimodoMotionApplier`, `KimodoPipelineRunner`, and `KimodoMotionBrowserWindow` are introduced before later tasks consume them.
