from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from data_loaders.sensor_masking import TRACKER_COUNT, TRACKER_NAMES


@dataclass(frozen=True)
class TrackerGap:
    """原始 Unity Tracker 序列中需要替换的半开帧区间。"""

    start_frame: int
    end_frame: int
    tracker_index: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "在原始 Unity Tracker JSON 上修复短时离群点；"
            "原文件保持不变，输出独立 cleaned JSON 和诊断 sidecar。"
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--gap",
        action="append",
        default=[],
        help=(
            "可重复传入，格式为 start_frame:end_frame:tracker_index；"
            "区间采用 Python 半开语义 [start_frame,end_frame)。"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


# region 参数与数据读取


def parse_tracker_gap(value: str) -> TrackerGap:
    parts = str(value).strip().split(":")
    if len(parts) != 3:
        raise ValueError("gap 应为 start_frame:end_frame:tracker_index。")
    try:
        start_frame, end_frame, tracker_index = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("gap 的三个字段必须都是整数。") from exc
    if start_frame < 1 or end_frame <= start_frame:
        raise ValueError("gap 必须满足 1 <= start_frame < end_frame。")
    if not 0 <= tracker_index < TRACKER_COUNT:
        raise ValueError(f"tracker_index 必须位于 [0,{TRACKER_COUNT - 1}]。")
    return TrackerGap(start_frame, end_frame, tracker_index)


def load_tracker_payload(
    path: Path,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    input_path = path.resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    if len(frames) < 3:
        raise ValueError("Tracker JSON 至少需要 3 帧。")

    times = np.asarray([frame["time"] for frame in frames], dtype=np.float64)
    positions = np.asarray([frame["positions"] for frame in frames], dtype=np.float64)
    rotations = np.asarray([frame["rotations"] for frame in frames], dtype=np.float64)
    available = np.asarray([frame["available"] for frame in frames], dtype=bool)
    if positions.shape != (len(frames), TRACKER_COUNT, 3):
        raise ValueError(f"positions 应为 [T,{TRACKER_COUNT},3]，实际为 {positions.shape}。")
    if rotations.shape != (len(frames), TRACKER_COUNT, 4):
        raise ValueError(f"rotations 应为 [T,{TRACKER_COUNT},4]，实际为 {rotations.shape}。")
    if available.shape != (len(frames), TRACKER_COUNT):
        raise ValueError(f"available 应为 [T,{TRACKER_COUNT}]，实际为 {available.shape}。")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
        raise ValueError("Tracker JSON 的 time 必须严格递增且均为有限数值。")
    if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
        raise ValueError("Tracker JSON 的位置或旋转含 NaN/Inf。")
    rotation_norms = np.linalg.norm(rotations, axis=-1)
    if np.any(rotation_norms <= 1e-8):
        raise ValueError("Tracker JSON 含零 quaternion。")
    rotations /= rotation_norms[..., None]
    return payload, times, positions, rotations, available


def validate_gaps(gaps: tuple[TrackerGap, ...], frame_count: int) -> None:
    if not gaps:
        raise ValueError("至少需要一个 --gap。")
    occupied: set[tuple[int, int]] = set()
    for gap in gaps:
        if gap.end_frame >= frame_count:
            raise ValueError("gap 右侧必须保留一帧真实锚点。")
        for frame_index in range(gap.start_frame, gap.end_frame):
            key = (gap.tracker_index, frame_index)
            if key in occupied:
                raise ValueError("同一 Tracker 的 gap 区间不能重叠。")
            occupied.add(key)


def smootherstep(values: np.ndarray) -> np.ndarray:
    """五次 minimum-jerk 权重，两端速度和加速度均为零。"""

    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return clipped**3 * (clipped * (clipped * 6.0 - 15.0) + 10.0)


# endregion


# region Tracker gap 修复


def interpolate_quaternion_gap(
    start_quaternion: np.ndarray,
    end_quaternion: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    start = np.asarray(start_quaternion, dtype=np.float64)
    end = np.asarray(end_quaternion, dtype=np.float64)
    start /= np.linalg.norm(start)
    end /= np.linalg.norm(end)
    # q 与 -q 是同一个姿态；统一符号后沿 SO(3) 最短弧插值。
    if float(np.dot(start, end)) < 0.0:
        end *= -1.0
    start_rotation = Rotation.from_quat(start)
    delta = (start_rotation.inv() * Rotation.from_quat(end)).as_rotvec()
    repeated_start = Rotation.from_quat(
        np.broadcast_to(start, (len(weights), 4))
    )
    return (
        repeated_start
        * Rotation.from_rotvec(np.asarray(weights)[:, None] * delta[None])
    ).as_quat()


def repair_tracker_gaps(
    *,
    times: np.ndarray,
    positions: np.ndarray,
    rotations_xyzw: np.ndarray,
    gaps: tuple[TrackerGap, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在 `[T,6,3/4]` 原始采样上，用两侧真实锚点替换短时坏点。"""

    frame_times = np.asarray(times, dtype=np.float64)
    repaired_positions = np.asarray(positions, dtype=np.float64).copy()
    repaired_rotations = np.asarray(rotations_xyzw, dtype=np.float64).copy()
    changed = np.zeros(repaired_positions.shape[:2], dtype=bool)
    validate_gaps(gaps, len(frame_times))

    for gap in gaps:
        left_anchor = gap.start_frame - 1
        right_anchor = gap.end_frame
        selected = np.arange(gap.start_frame, gap.end_frame, dtype=np.int64)
        duration = frame_times[right_anchor] - frame_times[left_anchor]
        if duration <= 0.0:
            raise ValueError("gap 两侧锚点时间必须严格递增。")
        progress = (
            frame_times[selected] - frame_times[left_anchor]
        ) / duration
        weights = smootherstep(progress)

        start_position = repaired_positions[left_anchor, gap.tracker_index]
        end_position = repaired_positions[right_anchor, gap.tracker_index]
        repaired_positions[selected, gap.tracker_index] = (
            (1.0 - weights[:, None]) * start_position[None]
            + weights[:, None] * end_position[None]
        )
        repaired_rotations[selected, gap.tracker_index] = (
            interpolate_quaternion_gap(
                repaired_rotations[left_anchor, gap.tracker_index],
                repaired_rotations[right_anchor, gap.tracker_index],
                weights,
            )
        )
        changed[selected, gap.tracker_index] = True

    return repaired_positions, repaired_rotations, changed


def tracker_step_metrics(
    positions: np.ndarray,
    rotations_xyzw: np.ndarray,
    *,
    gap: TrackerGap,
) -> dict[str, float]:
    """统计 gap 及两侧锚点间的最大单帧位移和旋转步长。"""

    selected = slice(gap.start_frame - 1, gap.end_frame + 1)
    tracker_positions = np.asarray(positions[selected, gap.tracker_index], dtype=np.float64)
    tracker_rotations = np.asarray(
        rotations_xyzw[selected, gap.tracker_index], dtype=np.float64
    )
    tracker_rotations /= np.linalg.norm(tracker_rotations, axis=-1, keepdims=True)
    position_steps_cm = np.linalg.norm(
        np.diff(tracker_positions, axis=0), axis=-1
    ) * 100.0
    rotations = Rotation.from_quat(tracker_rotations)
    rotation_steps_deg = np.linalg.norm(
        (rotations[:-1].inv() * rotations[1:]).as_rotvec(), axis=-1
    ) * (180.0 / np.pi)
    return {
        "max_position_step_cm": float(position_steps_cm.max(initial=0.0)),
        "max_rotation_step_deg": float(rotation_steps_deg.max(initial=0.0)),
    }


# endregion


def clean_tracker_recording(
    *,
    input_path: Path,
    output_path: Path,
    gaps: tuple[TrackerGap, ...],
    overwrite: bool,
) -> tuple[Path, Path]:
    source_path = input_path.resolve()
    target_path = output_path.resolve()
    sidecar_path = target_path.with_suffix(".cleaning.json")
    for path in (target_path, sidecar_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"输出已存在，请传入 --overwrite：{path}")

    payload, times, positions, rotations, available = load_tracker_payload(source_path)
    repaired_positions, repaired_rotations, changed = repair_tracker_gaps(
        times=times,
        positions=positions,
        rotations_xyzw=rotations,
        gaps=gaps,
    )
    frames = payload["frames"]
    for frame_index, tracker_index in np.argwhere(changed).tolist():
        # 只写回明确修复的 Tracker 槽位，避免 quaternion 归一化让同帧其他
        # Tracker 产生无语义的浮点变化，便于逐字段审计 cleaned JSON。
        frames[frame_index]["positions"][tracker_index] = (
            repaired_positions[frame_index, tracker_index].tolist()
        )
        frames[frame_index]["rotations"][tracker_index] = (
            repaired_rotations[frame_index, tracker_index].tolist()
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    diagnostics = []
    for gap in gaps:
        diagnostics.append(
            {
                "start_frame": gap.start_frame,
                "end_frame_exclusive": gap.end_frame,
                "start_time": float(times[gap.start_frame]),
                "end_time": float(times[gap.end_frame - 1]),
                "tracker_index": gap.tracker_index,
                "schema_tracker_name": TRACKER_NAMES[gap.tracker_index],
                "available_during_gap": bool(
                    available[
                        gap.start_frame : gap.end_frame,
                        gap.tracker_index,
                    ].all()
                ),
                "before": tracker_step_metrics(positions, rotations, gap=gap),
                "after": tracker_step_metrics(
                    repaired_positions,
                    repaired_rotations,
                    gap=gap,
                ),
            }
        )
    sidecar = {
        "experiment": "unity_tracker_short_gap_cleanup",
        "input": str(source_path),
        "output": str(target_path),
        "frames": len(frames),
        "method": "minimum_jerk_position_and_so3_rotation_interpolation",
        "changed_values": int(changed.sum()),
        "availability_modified": False,
        "gaps": diagnostics,
    }
    sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target_path, sidecar_path


def main() -> None:
    args = build_arg_parser().parse_args()
    gaps = tuple(parse_tracker_gap(value) for value in args.gap)
    output_path, sidecar_path = clean_tracker_recording(
        input_path=args.input,
        output_path=args.output,
        gaps=gaps,
        overwrite=bool(args.overwrite),
    )
    print(f"[tracker-cleanup] wrote {output_path}", flush=True)
    print(f"[tracker-cleanup] wrote {sidecar_path}", flush=True)


if __name__ == "__main__":
    main()
