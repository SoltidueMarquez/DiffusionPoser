from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from data_loaders.realtime_pose_kinematics import SMPL_JOINT_NAMES

from .body_fbx import parse_body_fbx_offsets_from_meta
from .metrics import (
    angle_stats,
    compute_6d_validity,
    compute_bone_direction_angles_degrees,
    compute_fk_joints,
    compute_joint_errors,
    compute_root_variant_stats,
    error_stats,
)
from .replay_io import load_replay_arrays, load_source_offsets, resolve_source_npz
from .unity_dump import compare_unity_dump


def build_debug_report(
    replay_json: Path,
    body_fbx_meta: Path,
    frame_start: int = 0,
    frame_count: int = 0,
    unity_dump_json: Path | None = None,
) -> dict[str, Any]:
    replay = load_replay_arrays(replay_json, frame_start=frame_start, frame_count=frame_count)
    source_npz = resolve_source_npz(replay.payload, replay_json)
    source_offsets = load_source_offsets(source_npz)
    body_offsets = parse_body_fbx_offsets_from_meta(body_fbx_meta)

    source_joints = compute_fk_joints(replay.target_features_raw, replay.root_pos_world, replay.root_yaw, source_offsets)
    body_joints = compute_fk_joints(replay.target_features_raw, replay.root_pos_world, replay.root_yaw, body_offsets)
    source_errors = compute_joint_errors(source_joints, replay.reference_joints_world)
    body_errors = compute_joint_errors(body_joints, replay.reference_joints_world)
    source_direction_angles = compute_bone_direction_angles_degrees(source_joints, replay.reference_joints_world)
    body_direction_angles = compute_bone_direction_angles_degrees(body_joints, replay.reference_joints_world)

    source_stats = error_stats(source_errors)
    body_stats = error_stats(body_errors)
    unity_comparison = compare_unity_dump(unity_dump_json, replay) if unity_dump_json is not None and str(unity_dump_json) else None

    source_ok = source_stats["mean_m"] < 1e-4
    body_bad = body_stats["mean_m"] > max(0.05, source_stats["mean_m"] * 1000.0)
    unity_decoder_ok = None if unity_comparison is None else bool(unity_comparison["decoderLooksAligned"])
    if source_ok and body_bad and (unity_decoder_ok is not False):
        likely_cause = "SMPL local rotations require retarget/bind-pose conversion before driving body.fbx."
    elif not source_ok:
        likely_cause = "Source feature/source offsets FK roundtrip failed; inspect JSON export, root convention, or 6D feature construction first."
    elif unity_decoder_ok is False:
        likely_cause = "Unity 6D decoder differs from Python decoder; fix cross-runtime rotation decoding before retargeting."
    else:
        likely_cause = "Inconclusive; inspect per-frame/per-bone diagnostics."

    return {
        "debugMarker": "DEBUG_RETARGET_PROBE",
        "replayJson": str(replay_json.resolve()),
        "sourceNpz": str(source_npz),
        "bodyFbxMeta": str(body_fbx_meta.resolve()),
        "frameStart": int(replay.frame_indices[0]),
        "frameCount": int(replay.frame_indices.shape[0]),
        "frameIndices": [int(value) for value in replay.frame_indices.tolist()],
        "thresholds": {
            "sourceRoundtripMeanM": 1e-4,
            "unityDecoderMaxAngleDeg": 0.1,
        },
        "sourceRoundtrip": source_stats,
        "bodyFbxOffsetReplay": body_stats,
        "sourceBoneDirectionAngles": angle_stats(source_direction_angles),
        "bodyFbxBoneDirectionAngles": angle_stats(body_direction_angles),
        "rootVariantStats": compute_root_variant_stats(replay, source_offsets),
        "rotation6dValidity": compute_6d_validity(replay.target_features_raw),
        "offsetDeltaNormCm": [
            float(value) for value in (np.linalg.norm(body_offsets - source_offsets, axis=-1) * 100.0).tolist()
        ],
        "classification": {
            "sourceRoundtripOk": bool(source_ok),
            "bodyFbxOffsetsFail": bool(body_bad),
            "unityDecoderOk": unity_decoder_ok,
            "likelyCause": likely_cause,
        },
        "perJointErrorsCm": build_per_joint_error_rows(
            replay.frame_indices,
            source_errors,
            body_errors,
            source_direction_angles,
            body_direction_angles,
        ),
        "unityDumpComparison": unity_comparison,
    }


def build_per_joint_error_rows(
    frame_indices: np.ndarray,
    source_errors_m: np.ndarray,
    body_errors_m: np.ndarray,
    source_direction_angles_deg: np.ndarray,
    body_direction_angles_deg: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for local_frame, frame_index in enumerate(frame_indices.tolist()):
        for bone_index, bone_name in enumerate(SMPL_JOINT_NAMES):
            rows.append(
                {
                    "frameIndex": int(frame_index),
                    "boneIndex": int(bone_index),
                    "boneName": str(bone_name),
                    "sourceErrorCm": float(source_errors_m[local_frame, bone_index] * 100.0),
                    "bodyFbxOffsetErrorCm": float(body_errors_m[local_frame, bone_index] * 100.0),
                    "sourceBoneDirectionAngleDeg": float(source_direction_angles_deg[local_frame, bone_index]),
                    "bodyFbxBoneDirectionAngleDeg": float(body_direction_angles_deg[local_frame, bone_index]),
                }
            )
    return rows


def write_report(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "retarget_debug_summary.json"
    per_joint_path = output_dir / "per_joint_errors_cm.csv"
    with summary_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write("\n")

    with per_joint_path.open("w", encoding="utf-8", newline="\n") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "frameIndex",
                "boneIndex",
                "boneName",
                "sourceErrorCm",
                "bodyFbxOffsetErrorCm",
                "sourceBoneDirectionAngleDeg",
                "bodyFbxBoneDirectionAngleDeg",
            ],
        )
        writer.writeheader()
        writer.writerows(report["perJointErrorsCm"])

    paths = {"summary": summary_path, "per_joint_errors": per_joint_path}
    if report.get("unityDumpComparison") is not None:
        unity_path = output_dir / "unity_cross_runtime_summary.json"
        unity_payload = {key: value for key, value in report["unityDumpComparison"].items() if key != "rows"}
        with unity_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(unity_payload, file, indent=2, ensure_ascii=False)
            file.write("\n")
        paths["unity_cross_runtime"] = unity_path
    return paths
