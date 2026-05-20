from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from data_loaders.sensor_masking import X277_FEATURE_DIM
from visual_editor.models import (
    LIBRARY_CACHE_SCHEMA,
    LIBRARY_CACHE_VERSION,
    ComparePreset,
    MotionAsset,
    MotionTrack,
    StudioConfig,
)
from visual_editor.x277 import (
    file_stat_payload,
    iter_jsonl,
    load_json,
    normalize_relative_path,
    path_identity,
    resolve_manifest_file,
    stable_id,
    write_json,
    utc_now,
)


SMPL_RUNTIME_PACKAGES = ("torch", "smplx", "scipy")


def has_smpl_model_files(model_dir: Path | None) -> bool:
    if model_dir is None or not model_dir.exists():
        return False
    smpl_files = any(
        (model_dir / f"SMPL_{gender}.{ext}").exists()
        for gender in ("MALE", "FEMALE", "NEUTRAL")
        for ext in ("pkl", "npz")
    )
    smplh_files = any((model_dir / f"SMPLH_{gender}.npz").exists() for gender in ("MALE", "FEMALE", "NEUTRAL"))
    return bool(smpl_files or smplh_files)


def smpl_runtime_status(model_dir: Path | None) -> dict[str, Any]:
    if model_dir is None:
        return {
            "available": False,
            "reason": "未配置 smpl_model_dir；AMASS Raw 需要本地 SMPL/SMPL-H 模型。",
            "missing_packages": [],
            "has_model_files": False,
        }
    if not model_dir.exists():
        return {
            "available": False,
            "reason": f"smpl_model_dir 不存在: {model_dir}",
            "missing_packages": [],
            "has_model_files": False,
        }
    if not has_smpl_model_files(model_dir):
        return {
            "available": False,
            "reason": f"在 {model_dir} 中没有找到 SMPL_*.pkl/npz 或 SMPLH_*.npz 模型文件。",
            "missing_packages": [],
            "has_model_files": False,
        }

    missing_packages = [name for name in SMPL_RUNTIME_PACKAGES if importlib.util.find_spec(name) is None]
    if missing_packages:
        install_command = (
            "visual_editor/.venv/Scripts/python -m pip install "
            "-r visual_editor/requirements-smpl.txt --cache-dir visual_editor/.cache/pip"
        )
        return {
            "available": False,
            "reason": f"缺少可选依赖: {', '.join(missing_packages)}。运行 {install_command} 后刷新 Library。",
            "missing_packages": missing_packages,
            "has_model_files": True,
        }
    return {
        "available": True,
        "reason": "",
        "missing_packages": [],
        "has_model_files": True,
    }


def source_manifest_path(source_dir: Path) -> Path | None:
    manifest_path = source_dir / "manifest.jsonl"
    return manifest_path if manifest_path.exists() else None


def task_manifest_paths(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    manifest_paths: list[Path] = []
    root_manifest = data_dir / "manifest.jsonl"
    if root_manifest.exists():
        manifest_paths.append(root_manifest)
    manifest_paths.extend(sorted(path for path in data_dir.glob("*/manifest.jsonl") if path not in manifest_paths))
    return manifest_paths


def resolve_source_motion_path(source_dir: Path, entry: dict[str, Any]) -> Path | None:
    candidates: list[Path] = []
    output_path = resolve_manifest_file(source_dir, entry.get("output_path"))
    if output_path is not None:
        candidates.append(output_path)
    relative_text = entry.get("source_relative_path") or entry.get("stablemotion_split_key")
    if relative_text:
        relative = normalize_relative_path(str(relative_text))
        if relative.endswith(".npy"):
            relative = f"{relative[:-4]}.npz"
        candidates.append(source_dir / relative)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def load_amass_header(path: Path) -> tuple[int, float]:
    with np.load(path, allow_pickle=True) as data:
        if "poses" not in data.files:
            raise KeyError(f"{path} missing poses")
        frame_count = int(data["poses"].shape[0])
        fps = float(data["mocap_framerate"]) if "mocap_framerate" in data.files else 60.0
    return frame_count, fps


class MotionLibrary:
    def __init__(self, config: StudioConfig):
        self.config = config
        self.cache_path = self.config.runtime_dir / "library_cache.json"
        self.assets: dict[str, MotionAsset] = {}
        self.index_meta: dict[str, Any] = {}

    def load(self, *, force: bool = False) -> dict[str, MotionAsset]:
        self.config.runtime_dir.mkdir(parents=True, exist_ok=True)
        signature, cacheable = self.index_signature()
        if not force and cacheable:
            cached = self.load_cache(signature)
            if cached is not None:
                self.assets = cached
                return self.assets

        started_at = time.perf_counter()
        assets: dict[str, MotionAsset] = {}
        source_entries = list(self.iter_source_entries())
        task_entries = list(self.iter_task_entries())
        task_start_by_id = {str(entry.get("task_id")): int(entry.get("start_frame") or 0) for _, _, entry in task_entries}

        assets.update(self.scan_amass_assets(source_entries=source_entries))
        assets.update(self.scan_x277_assets(source_entries=source_entries))
        assets.update(self.scan_task_assets(task_entries=task_entries))
        assets.update(self.scan_repair_assets(task_start_by_id=task_start_by_id))
        self.assets = dict(sorted(assets.items(), key=lambda item: (item[1].kind, item[1].label)))
        self.index_meta = {
            "cache_hit": False,
            "cacheable": bool(cacheable),
            "cache_path": str(self.cache_path),
            "scanned_at": utc_now(),
            "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
            "asset_count": len(self.assets),
            "amass_count": sum(1 for item in self.assets.values() if item.kind == "amass"),
            "x277_count": sum(1 for item in self.assets.values() if item.kind == "x277"),
            "task_count": sum(1 for item in self.assets.values() if item.kind == "task"),
            "repair_count": sum(1 for item in self.assets.values() if item.kind == "repair"),
        }
        if cacheable:
            self.write_cache(signature)
        return self.assets

    def index_signature(self) -> tuple[dict[str, Any], bool]:
        manifest_paths = []
        source_manifest = source_manifest_path(self.config.source_dir)
        if source_manifest is not None:
            manifest_paths.append(source_manifest)
        manifest_paths.extend(task_manifest_paths(self.config.data_dir))
        result_streams = sorted(self.config.result_dir.rglob("stream_outputs.npz")) if self.config.result_dir.exists() else []
        result_metadata = [path.with_name("metadata.json") for path in result_streams if path.with_name("metadata.json").exists()]
        result_files = [*result_streams[:5000], *result_metadata[:5000]]
        source_needs_scan = self.config.source_dir.exists() and source_manifest is None
        amass_needs_scan = self.config.amass_dir.exists() and source_manifest is None
        signature = {
            "version": LIBRARY_CACHE_VERSION,
            "amass_dir": path_identity(self.config.amass_dir),
            "source_dir": path_identity(self.config.source_dir),
            "data_dir": path_identity(self.config.data_dir),
            "result_dir": path_identity(self.config.result_dir),
            "smpl_model_dir": str(self.config.smpl_model_dir) if self.config.smpl_model_dir else "",
            "manifests": [file_stat_payload(path) for path in manifest_paths],
            "result_metadata": [file_stat_payload(path) for path in result_files],
            "source_needs_scan": source_needs_scan,
            "amass_needs_scan": amass_needs_scan,
            "smpl_runtime": smpl_runtime_status(self.config.smpl_model_dir),
        }
        cacheable = bool(manifest_paths or result_files) and not source_needs_scan and not amass_needs_scan
        return signature, cacheable

    def load_cache(self, signature: dict[str, Any]) -> dict[str, MotionAsset] | None:
        if not self.cache_path.exists():
            return None
        try:
            payload = load_json(self.cache_path)
            if payload.get("schema_name") != LIBRARY_CACHE_SCHEMA or payload.get("signature") != signature:
                return None
            assets = {
                asset.asset_id: asset
                for asset in (MotionAsset.from_dict(item) for item in payload.get("assets", []))
            }
        except Exception:
            return None
        self.index_meta = {
            **dict(payload.get("index_meta", {})),
            "cache_hit": True,
            "cache_path": str(self.cache_path),
            "loaded_at": utc_now(),
        }
        return dict(sorted(assets.items(), key=lambda item: (item[1].kind, item[1].label)))

    def write_cache(self, signature: dict[str, Any]) -> None:
        payload = {
            "schema_name": LIBRARY_CACHE_SCHEMA,
            "created_at": utc_now(),
            "signature": signature,
            "index_meta": self.index_meta,
            "assets": [asset.to_dict() for asset in self.assets.values()],
        }
        try:
            write_json(self.cache_path, payload)
        except OSError:
            pass

    def iter_source_entries(self):
        manifest_path = source_manifest_path(self.config.source_dir)
        if manifest_path is not None:
            for entry in iter_jsonl(manifest_path):
                yield entry

    def iter_task_entries(self):
        for manifest_path in task_manifest_paths(self.config.data_dir):
            split = manifest_path.parent.name if manifest_path.parent != self.config.data_dir else ""
            for entry in iter_jsonl(manifest_path):
                yield manifest_path, split, entry

    def scan_amass_assets(self, *, source_entries: list[dict[str, Any]]) -> dict[str, MotionAsset]:
        records: dict[str, MotionAsset] = {}
        smpl_status = smpl_runtime_status(self.config.smpl_model_dir)
        smpl_available = bool(smpl_status["available"])
        smpl_reason = str(smpl_status["reason"])
        if source_entries:
            for entry in source_entries:
                raw_path = resolve_manifest_file(self.config.amass_dir, entry.get("source_path"))
                x277_path = resolve_source_motion_path(self.config.source_dir, entry)
                if raw_path is None or not raw_path.exists():
                    continue
                relative = normalize_relative_path(str(entry.get("original_source_relative_path") or entry.get("source_relative_path") or raw_path.name))
                is_mirrored = bool(entry.get("is_mirrored", False))
                asset_id = stable_id("amass", f"{raw_path}|{is_mirrored}|{x277_path}")
                frame_count = int(entry.get("frames") or 0)
                tracks: dict[str, MotionTrack] = {}
                tracks["amass_raw"] = MotionTrack(
                    track_id="amass_raw",
                    label="AMASS Raw",
                    data_key="amass_raw",
                    frame_count=frame_count,
                    fps=float(self.config.x277_fps),
                    source_path=raw_path,
                    compatible_x277=False,
                    available=bool(smpl_available and not is_mirrored),
                    unavailable_reason="" if smpl_available and not is_mirrored else ("镜像 AMASS Raw 不直接渲染，请查看 X277 Converted。" if is_mirrored else smpl_reason),
                    meta={"is_mirrored": is_mirrored, "smpl_status": smpl_status},
                )
                if x277_path is not None and x277_path.exists():
                    tracks["x277_converted"] = MotionTrack(
                        track_id="x277_converted",
                        label="X277 Converted",
                        data_key="x",
                        frame_count=frame_count,
                        fps=float(self.config.x277_fps),
                        source_path=x277_path,
                        compatible_x277=True,
                        meta={"source_relative_path": normalize_relative_path(str(entry.get("source_relative_path", "")))},
                    )
                records[asset_id] = MotionAsset(
                    asset_id=asset_id,
                    kind="amass",
                    label=relative,
                    tracks=tracks,
                    source_path=raw_path,
                    frame_count=frame_count,
                    fps=float(self.config.x277_fps),
                    group=relative.split("/")[0] if "/" in relative else "AMASS",
                    meta={"is_mirrored": is_mirrored, "x277_path": str(x277_path or "")},
                )
            return records

        if not self.config.amass_dir.exists():
            return records
        for raw_path in sorted(self.config.amass_dir.rglob("*.npz"))[:20000]:
            try:
                frame_count, fps = load_amass_header(raw_path)
            except Exception:
                continue
            relative = normalize_relative_path(raw_path.relative_to(self.config.amass_dir).as_posix())
            asset_id = stable_id("amass", str(raw_path.resolve()))
            records[asset_id] = MotionAsset(
                asset_id=asset_id,
                kind="amass",
                label=relative,
                source_path=raw_path,
                frame_count=frame_count,
                fps=fps,
                group=relative.split("/")[0] if "/" in relative else "AMASS",
                tracks={
                    "amass_raw": MotionTrack(
                        track_id="amass_raw",
                        label="AMASS Raw",
                        data_key="amass_raw",
                        frame_count=frame_count,
                        fps=fps,
                        source_path=raw_path,
                        compatible_x277=False,
                        available=smpl_available,
                        unavailable_reason="" if smpl_available else smpl_reason,
                        meta={"smpl_status": smpl_status},
                    )
                },
            )
        return records

    def scan_x277_assets(self, *, source_entries: list[dict[str, Any]]) -> dict[str, MotionAsset]:
        records: dict[str, MotionAsset] = {}
        if source_entries:
            for entry in source_entries:
                x277_path = resolve_source_motion_path(self.config.source_dir, entry)
                if x277_path is None or not x277_path.exists():
                    continue
                frame_count = int(entry.get("frames") or 0)
                feature_dim = int(entry.get("feature_dim") or X277_FEATURE_DIM)
                if frame_count <= 0 or feature_dim != X277_FEATURE_DIM:
                    continue
                relative = normalize_relative_path(str(entry.get("source_relative_path", ""))) or x277_path.name
                asset_id = stable_id("x277", str(x277_path.resolve()))
                track = MotionTrack(
                    track_id="x277_converted",
                    label="X277 Converted",
                    data_key="x",
                    frame_count=frame_count,
                    fps=float(self.config.x277_fps),
                    source_path=x277_path,
                    compatible_x277=True,
                    meta={"source_relative_path": relative},
                )
                records[asset_id] = MotionAsset(
                    asset_id=asset_id,
                    kind="x277",
                    label=relative,
                    source_path=x277_path,
                    frame_count=frame_count,
                    fps=float(self.config.x277_fps),
                    group=relative.split("/")[0] if "/" in relative else "X277",
                    tracks={track.track_id: track},
                    meta={"source_relative_path": relative},
                )
            return records

        if not self.config.source_dir.exists():
            return records
        for x277_path in sorted(self.config.source_dir.rglob("*.npz")):
            if "tasks" in x277_path.parts:
                continue
            try:
                with np.load(x277_path, allow_pickle=False) as data:
                    if "x" not in data.files or data["x"].ndim != 2 or data["x"].shape[1] != X277_FEATURE_DIM:
                        continue
                    frame_count = int(data["x"].shape[0])
            except Exception:
                continue
            relative = normalize_relative_path(x277_path.relative_to(self.config.source_dir).as_posix())
            asset_id = stable_id("x277", str(x277_path.resolve()))
            track = MotionTrack(
                track_id="x277_converted",
                label="X277 Converted",
                data_key="x",
                frame_count=frame_count,
                fps=float(self.config.x277_fps),
                source_path=x277_path,
                compatible_x277=True,
                meta={"source_relative_path": relative},
            )
            records[asset_id] = MotionAsset(
                asset_id=asset_id,
                kind="x277",
                label=relative,
                source_path=x277_path,
                frame_count=frame_count,
                fps=float(self.config.x277_fps),
                group=relative.split("/")[0] if "/" in relative else "X277",
                tracks={track.track_id: track},
            )
        return records

    def scan_task_assets(self, *, task_entries: list[tuple[Path, str, dict[str, Any]]]) -> dict[str, MotionAsset]:
        records: dict[str, MotionAsset] = {}
        for manifest_path, split, entry in task_entries:
            task_path = resolve_manifest_file(manifest_path.parent, entry.get("task_path"))
            if task_path is None or not task_path.exists():
                continue
            frame_count = int(entry.get("valid_length") or entry.get("seq_len") or 0)
            if frame_count <= 0:
                continue
            task_id = str(entry.get("task_id") or task_path.stem)
            asset_id = stable_id("task", f"{manifest_path}|{task_id}|{task_path}")
            track = MotionTrack(
                track_id="task_reference",
                label="Task Reference",
                data_key="x277",
                frame_count=frame_count,
                fps=float(self.config.x277_fps),
                source_path=task_path,
                compatible_x277=True,
                meta={
                    "task_id": task_id,
                    "split": split,
                    "start_frame": int(entry.get("start_frame") or 0),
                    "source_path": str(entry.get("source_path", "")),
                    "source_relative_path": normalize_relative_path(str(entry.get("source_relative_path", ""))),
                },
            )
            records[asset_id] = MotionAsset(
                asset_id=asset_id,
                kind="task",
                label=f"{split or 'tasks'} / {task_id}",
                source_path=task_path,
                frame_count=frame_count,
                fps=float(self.config.x277_fps),
                group=split or "tasks",
                tracks={track.track_id: track},
                meta=track.meta,
            )
        return records

    def scan_repair_assets(self, *, task_start_by_id: dict[str, int]) -> dict[str, MotionAsset]:
        records: dict[str, MotionAsset] = {}
        if not self.config.result_dir.exists():
            return records
        for stream_path in sorted(self.config.result_dir.rglob("stream_outputs.npz")):
            metadata_path = stream_path.with_name("metadata.json")
            metadata = {}
            if metadata_path.exists():
                try:
                    metadata = load_json(metadata_path)
                except Exception:
                    metadata = {}
            task_id = str(metadata.get("task_id") or stream_path.parent.name)
            sample_name = str(metadata.get("sample_name") or stream_path.parent.name)
            source_start = int(metadata.get("start_frame") or task_start_by_id.get(task_id, 0))
            valid_length = int(metadata.get("valid_length") or 0)
            if valid_length <= 0:
                try:
                    with np.load(stream_path, allow_pickle=False) as data:
                        valid_length = int(data["reference_motion"].shape[0])
                except Exception:
                    continue
            asset_id = stable_id("repair", str(stream_path.resolve()))
            tracks = {}
            for track_id, label, data_key in (
                ("ground_truth", "Ground Truth", "reference_motion"),
                ("conditioned", "Conditioned", "conditioned_motion"),
                ("reconstructed", "Repair", "reconstructed_motion"),
            ):
                tracks[track_id] = MotionTrack(
                    track_id=track_id,
                    label=label,
                    data_key=data_key,
                    frame_count=valid_length,
                    fps=float(metadata.get("x277_fps") or self.config.x277_fps),
                    source_path=stream_path,
                    compatible_x277=True,
                    meta={
                        "task_id": task_id,
                        "sample_name": sample_name,
                        "source_path": str(metadata.get("source_path", "")),
                        "source_start_frame": source_start,
                    },
                )
            records[asset_id] = MotionAsset(
                asset_id=asset_id,
                kind="repair",
                label=sample_name,
                source_path=stream_path,
                frame_count=valid_length,
                fps=float(metadata.get("x277_fps") or self.config.x277_fps),
                group=stream_path.parent.parent.name,
                tracks=tracks,
                meta={
                    **metadata,
                    "source_start_frame": source_start,
                    "stream_path": str(stream_path),
                },
            )
        return records

    def payload(self) -> dict[str, Any]:
        smpl_status = smpl_runtime_status(self.config.smpl_model_dir)
        return {
            "schema_name": "x277_motion_studio_library_v1",
            "config": {
                "amass_dir": str(self.config.amass_dir),
                "source_dir": str(self.config.source_dir),
                "data_dir": str(self.config.data_dir),
                "result_dir": str(self.config.result_dir),
                "output_dir": str(self.config.output_dir),
                "runtime_dir": str(self.config.runtime_dir),
                "smpl_model_dir": str(self.config.smpl_model_dir) if self.config.smpl_model_dir else "",
                "smpl_available": bool(smpl_status["available"]),
                "smpl_unavailable_reason": str(smpl_status["reason"]),
                "smpl_status": smpl_status,
                "x277_fps": float(self.config.x277_fps),
            },
            "index": self.index_meta,
            "assets": [asset.to_dict() for asset in self.assets.values()],
            "presets": [preset.to_dict() for preset in default_presets()],
        }


def default_presets() -> list[ComparePreset]:
    return [
        ComparePreset("amass_x277", "AMASS vs X277", 2, "Compare raw AMASS with converted X277 fallback."),
        ComparePreset("gt_repair", "GT vs Repair", 2, "Compare reference_motion and reconstructed_motion."),
        ComparePreset("conditioned_repair", "Conditioned vs Repair", 2, "Compare model input condition and repair result."),
        ComparePreset("quad_full", "AMASS / X277 / GT / Repair", 4, "Four-pane source and repair overview."),
    ]
