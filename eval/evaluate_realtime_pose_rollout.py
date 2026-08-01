from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_loaders.realtime_pose_kinematics import rotation_6d_to_matrix_np
from eval.evaluate_realtime_pose import evaluate_file, public_result, summarize


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评估 140D rollout 与 Tracker 重连突变。")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_json", default="")
    return parser


def evaluate_rollout_file(path: Path) -> dict[str, object]:
    result = evaluate_file(path)
    with np.load(path, allow_pickle=False) as data:
        joints = np.asarray(data["predicted_joints_world"], dtype=np.float64)
        local_delta = np.asarray(data["predicted_body_local_delta_6d"], dtype=np.float64)
        measured = np.asarray(data["measured_valid"], dtype=bool)
        eval_mask = np.asarray(data["eval_frame_mask"], dtype=bool)

    sequence_count, steps = joints.shape[:2]
    if steps < 2:
        reconnect = np.zeros((sequence_count, 0), dtype=bool)
        velocity = np.zeros((sequence_count, 0), dtype=np.float64)
        angular = np.zeros((sequence_count, 0), dtype=np.float64)
    else:
        reconnect = (measured[:, 1:] & ~measured[:, :-1])[:, :, 1:].any(axis=-1)
        reconnect &= eval_mask[:, 1:] & eval_mask[:, :-1]
        velocity = np.linalg.norm(joints[:, 1:, :22] - joints[:, :-1, :22], axis=-1).mean(axis=-1)
        rotations = rotation_6d_to_matrix_np(local_delta.reshape(sequence_count, steps, 24, 6))
        relative = np.swapaxes(rotations[:, :-1, 1:22], -1, -2) @ rotations[:, 1:, 1:22]
        cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
        angular = np.arccos(cosine).mean(axis=-1)

    reconnect_count = int(reconnect.sum())
    velocity_sum = float(velocity[reconnect].sum()) if reconnect_count else 0.0
    angular_sum_deg = float(np.degrees(angular[reconnect]).sum()) if reconnect_count else 0.0
    result["reconnect_frames"] = reconnect_count
    result["reconnect_velocity_jump_m"] = velocity_sum / reconnect_count if reconnect_count else 0.0
    result["reconnect_angular_jump_deg"] = angular_sum_deg / reconnect_count if reconnect_count else 0.0
    result["_reconnect_stats"] = {
        "count": reconnect_count,
        "velocity_sum": velocity_sum,
        "angular_sum_deg": angular_sum_deg,
    }
    return result


def summarize_rollouts(results: list[dict[str, object]]) -> dict[str, object]:
    summary = summarize(results)
    reconnect_count = sum(int(result["_reconnect_stats"]["count"]) for result in results)
    velocity_sum = sum(float(result["_reconnect_stats"]["velocity_sum"]) for result in results)
    angular_sum = sum(float(result["_reconnect_stats"]["angular_sum_deg"]) for result in results)
    summary["reconnect_frames"] = reconnect_count
    summary["reconnect_velocity_jump_m"] = velocity_sum / reconnect_count if reconnect_count else 0.0
    summary["reconnect_angular_jump_deg"] = angular_sum / reconnect_count if reconnect_count else 0.0
    return summary


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = build_arg_parser().parse_args(argv)
    input_dir = Path(args.input_dir).resolve()
    paths = sorted(input_dir.rglob("rollout_result*.npz")) or sorted(input_dir.rglob("*.npz"))
    results = [evaluate_rollout_file(path) for path in paths]
    summary = summarize_rollouts(results)
    output_json = Path(args.output_json).resolve() if args.output_json else input_dir / "rollout_eval_summary.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(
            {"summary": summary, "files": [public_result(result) for result in results]},
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[evaluate_realtime_pose_rollout] wrote {output_json}")
    return summary


if __name__ == "__main__":
    main()
