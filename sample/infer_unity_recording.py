from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from data_loaders.body_fbx_kinematics import (
    BodyFbxRest,
    load_body_fbx_rest,
    quaternion_xyzw_to_matrix_np,
)
from data_loaders.realtime_pose_geometry import (
    extract_forward_yaw_np,
    resolve_root_head_reference_np,
)
from data_loaders.realtime_pose_kinematics import (
    make_yaw_rotation_np,
    rotation_6d_forward_up_np,
    rotation_6d_to_matrix_np,
)
from data_loaders.sensor_masking import (
    CORE_TRACKER_INDICES,
    FOOT_TRACKER_INDICES,
    HIP_TRACKER_INDEX,
    REALTIME_POSE_FPS,
    REALTIME_POSE_TARGET_DIM,
    TRACKER_COUNT,
    TRACKER_TO_JOINT,
)
from sample.realtime_pose_longseq_evaluator import create_eval_noise_generator
from sample.realtime_pose_runtime import RealtimePoseRuntime, WorldPoseState
from sample.utils import load_checkpoint_model
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_realtime_pose_predictor
from utils.normalizer import RealtimePoseNormalizer
from utils.parser_util import (
    add_diffusion_options,
    add_ik_inpainting_options,
    add_model_options,
    parse_and_load_from_model,
    str2bool,
)


BOOTSTRAP_TRACKER_FRAMES = 12
POSE_HISTORY_FRAMES = 10
DEFAULT_WARMUP_FRAMES = 30


# region 数据结构


@dataclass(frozen=True)
class UnityTrackerRecording:
    """Unity 录制的物理数据；时间轴尚未固定为模型的 30Hz。"""

    times: np.ndarray  # [T]
    positions: np.ndarray  # [T,6,3]
    rotations_xyzw: np.ndarray  # [T,6,4]
    available: np.ndarray  # [T,6]
    floor_y: float


@dataclass(frozen=True)
class ResampledTrackerRecording:
    """供 runtime 使用的 30Hz 六点 Tracker 序列。"""

    times: np.ndarray  # [T]
    positions: np.ndarray  # [T,6,3]
    rotations_xyzw: np.ndarray  # [T,6,4]
    rotations_6d: np.ndarray  # [T,6,6]
    rotations_world: np.ndarray  # [T,6,3,3]
    available: np.ndarray  # [T,6]
    floor_y: float


@dataclass(frozen=True)
class UnityPoseFrame:
    """Python runtime 结果转换后的 Unity Transform 数据。"""

    root_position: np.ndarray  # [3]
    root_rotation_xyzw: np.ndarray  # [4]
    pelvis_local_position: np.ndarray  # [3]
    local_rotations_xyzw: np.ndarray  # [24,4]


# endregion


# region CLI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="读取 FLUIDUnity Tracker JSON，运行 Predictor + 单帧 DiT，并写回 Unity Pose JSON。"
    )
    add_model_options(parser)
    add_diffusion_options(parser)
    add_ik_inpainting_options(parser)
    paths = parser.add_argument_group("FLUIDUnity bridge")
    paths.add_argument("--input", required=True, type=Path)
    paths.add_argument("--output", required=True, type=Path)
    paths.add_argument("--predictor_model_path", required=True, type=Path)
    paths.add_argument("--model_path", required=True, type=Path)
    paths.add_argument("--normalizer_dir", required=True, type=Path)
    paths.add_argument("--body_fbx_rest_json", required=True, type=Path)
    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", default="cuda")
    runtime.add_argument("--seed", default=10, type=int)
    runtime.add_argument("--use_ema", default=True, type=str2bool)
    runtime.add_argument("--warmup_frames", default=DEFAULT_WARMUP_FRAMES, type=int)
    runtime.add_argument(
        "--ignore_hip",
        action="store_true",
        help="强制把 Hip availability 设为 false，适合 Hip Tracker 异常的录制。",
    )
    runtime.add_argument(
        "--ignore_feet",
        action="store_true",
        help="强制把左右脚 availability 设为 false，用三点模式执行完整推理。",
    )
    # 训练 checkpoint 保存 50 个基础 timestep；Demo 固定用当前正式 10-step DDIM。
    parser.set_defaults(ts_respace="10")
    return parser


def resolve_device(value: str) -> torch.device:
    requested = str(value).strip().lower()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        requested = "cuda:0"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[FLUID] CUDA 不可用，改用 CPU。", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


# endregion


# region Unity Tracker 读取与重采样


def load_unity_tracker_recording(path: str | Path) -> UnityTrackerRecording:
    recording_path = Path(path).resolve()
    payload = json.loads(recording_path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    if len(frames) < BOOTSTRAP_TRACKER_FRAMES:
        raise ValueError("Unity 录制至少需要 12 帧。")

    times = np.asarray([frame["time"] for frame in frames], dtype=np.float64)
    positions = np.asarray([frame["positions"] for frame in frames], dtype=np.float32)
    rotations = np.asarray([frame["rotations"] for frame in frames], dtype=np.float64)
    available = np.asarray([frame["available"] for frame in frames], dtype=bool)
    if positions.shape[1:] != (TRACKER_COUNT, 3):
        raise ValueError(f"positions 应为 [T,6,3]，实际为 {positions.shape}")
    if rotations.shape != (len(times), TRACKER_COUNT, 4):
        raise ValueError(f"rotations 应为 [T,6,4]，实际为 {rotations.shape}")
    if available.shape != (len(times), TRACKER_COUNT):
        raise ValueError(f"available 应为 [T,6]，实际为 {available.shape}")

    # Unity 的渲染循环偶尔可能在同一时间戳写两帧；只保留最后一帧，避免 Slerp 要求失效。
    order = np.argsort(times, kind="stable")
    times = times[order]
    positions = positions[order]
    rotations = rotations[order]
    available = available[order]
    keep = np.concatenate([times[1:] > times[:-1], np.asarray([True])])
    times = times[keep]
    positions = positions[keep]
    rotations = rotations[keep]
    available = available[keep]
    times = times - times[0]

    rotation_norm = np.linalg.norm(rotations, axis=-1, keepdims=True)
    rotations = rotations / np.maximum(rotation_norm, 1e-8)
    return UnityTrackerRecording(
        times=times,
        positions=positions,
        rotations_xyzw=rotations.astype(np.float32),
        available=available,
        floor_y=float(payload.get("floorY", 0.0)),
    )


def apply_tracker_availability_overrides(
    recording: UnityTrackerRecording,
    *,
    ignore_hip: bool,
    ignore_feet: bool,
) -> UnityTrackerRecording:
    """只覆盖本次推理使用的 availability，不修改录制的测量值或原始 JSON。"""

    available = np.asarray(recording.available, dtype=bool).copy()
    if bool(ignore_hip):
        available[:, HIP_TRACKER_INDEX] = False
    if bool(ignore_feet):
        available[:, list(FOOT_TRACKER_INDICES)] = False
    return UnityTrackerRecording(
        times=recording.times,
        positions=recording.positions,
        rotations_xyzw=recording.rotations_xyzw,
        available=available,
        floor_y=recording.floor_y,
    )


def resample_tracker_recording(
    recording: UnityTrackerRecording,
    fps: float = REALTIME_POSE_FPS,
) -> ResampledTrackerRecording:
    """按 Unity 时间戳重采样；模型时序从此处开始固定为 30Hz。"""

    frame_interval = 1.0 / float(fps)
    target_frame_count = int(np.floor(float(recording.times[-1]) * float(fps))) + 1
    target_times = np.arange(target_frame_count, dtype=np.float64) * frame_interval
    positions = np.empty((len(target_times), TRACKER_COUNT, 3), dtype=np.float32)
    rotations = np.empty((len(target_times), TRACKER_COUNT, 4), dtype=np.float32)
    for tracker_index in range(TRACKER_COUNT):
        for axis in range(3):
            positions[:, tracker_index, axis] = np.interp(
                target_times,
                recording.times,
                recording.positions[:, tracker_index, axis],
            )

        tracker_quaternions = recording.rotations_xyzw[:, tracker_index].astype(np.float64).copy()
        # q 与 -q 表示同一旋转；先统一相邻符号，确保插值走最短弧。
        for frame_index in range(1, len(tracker_quaternions)):
            if np.dot(tracker_quaternions[frame_index - 1], tracker_quaternions[frame_index]) < 0.0:
                tracker_quaternions[frame_index] *= -1.0
        slerp = Slerp(recording.times, Rotation.from_quat(tracker_quaternions))
        rotations[:, tracker_index] = slerp(target_times).as_quat().astype(np.float32)

    insertion = np.searchsorted(recording.times, target_times, side="left")
    right = np.clip(insertion, 0, len(recording.times) - 1)
    left = np.clip(right - 1, 0, len(recording.times) - 1)
    choose_left = np.abs(target_times - recording.times[left]) <= np.abs(
        recording.times[right] - target_times
    )
    nearest = np.where(choose_left, left, right)
    available = recording.available[nearest].copy()
    rotation_matrices = quaternion_xyzw_to_matrix_np(rotations).astype(np.float32)
    rotations_6d = rotation_6d_forward_up_np(rotation_matrices).astype(np.float32)
    return ResampledTrackerRecording(
        times=target_times,
        positions=positions,
        rotations_xyzw=rotations,
        rotations_6d=rotations_6d,
        rotations_world=rotation_matrices,
        available=available,
        floor_y=float(recording.floor_y),
    )


def find_first_core_window(available: np.ndarray) -> int:
    """返回第一段连续 12 帧核心三点有效窗口的起点。"""

    core = np.asarray(available, dtype=bool)[:, list(CORE_TRACKER_INDICES)]
    for start in range(0, len(core) - BOOTSTRAP_TRACKER_FRAMES + 1):
        if core[start : start + BOOTSTRAP_TRACKER_FRAMES].all():
            return start
    raise RuntimeError("录制中没有连续 12 帧有效的 Head、左手和右手。")


# endregion


# region Runtime 初始化与 Unity 输出


def build_bootstrap_pose_history(
    recording: ResampledTrackerRecording,
    rest: BodyFbxRest,
    start: int,
) -> list[WorldPoseState]:
    """用 rest skeleton 和真实 Tracker 构造 runtime 所需的 10 帧完整历史。"""

    indices = np.arange(start + 1, start + 1 + POSE_HISTORY_FRAMES, dtype=np.int64)
    head_yaws = extract_forward_yaw_np(recording.rotations_world[indices, 0])
    states: list[WorldPoseState] = []
    for local_index, frame_index in enumerate(indices.tolist()):
        root_yaw = float(head_yaws[local_index])
        heading = make_yaw_rotation_np(np.asarray([root_yaw], dtype=np.float64))[0]
        world_rotations = np.empty((24, 3, 3), dtype=np.float64)
        for joint_index, parent_index in enumerate(rest.parents.tolist()):
            if parent_index < 0:
                world_rotations[joint_index] = heading @ rest.rest_local_rotations[joint_index]
            else:
                world_rotations[joint_index] = (
                    world_rotations[parent_index] @ rest.rest_local_rotations[joint_index]
                )

        # Tracker 已经过 Unity 设备到骨骼校准，初始化时可直接覆盖对应关节的世界旋转。
        for tracker_index, joint_index in enumerate(TRACKER_TO_JOINT):
            if recording.available[frame_index, tracker_index]:
                world_rotations[joint_index] = recording.rotations_world[frame_index, tracker_index]

        rotations_head = np.einsum("ij,ajk->aik", heading.T, world_rotations)
        observed_head_height = float(
            recording.positions[frame_index, 0, 1] - recording.floor_y
        )
        root_head, hip_height, _ = resolve_root_head_reference_np(
            rotations_head,
            root_yaw_head=0.0,
            rest_local_positions=rest.rest_local_positions,
            observed_head_height=observed_head_height,
        )
        head_position = recording.positions[frame_index, 0]
        origin = np.asarray(
            [head_position[0], recording.floor_y, head_position[2]],
            dtype=np.float64,
        )
        root_world = origin + heading @ root_head.astype(np.float64)
        root_world[1] = recording.floor_y
        states.append(
            WorldPoseState(
                joint_rotations_world=world_rotations.astype(np.float32),
                root_yaw_world=root_yaw,
                hip_height=float(hip_height),
                root_position_world=root_world.astype(np.float32),
            )
        )
    return states


def resolved_pose_to_unity_frame(
    resolved_pose,
    rest: BodyFbxRest,
    previous: UnityPoseFrame | None,
) -> UnityPoseFrame:
    delta = rotation_6d_to_matrix_np(
        np.asarray(resolved_pose.body_local_delta_6d, dtype=np.float32).reshape(24, 6)
    )
    local_rotations = rest.rest_local_rotations.astype(np.float64) @ delta
    local_quaternions = Rotation.from_matrix(local_rotations).as_quat().astype(np.float32)
    root_rotation = Rotation.from_matrix(
        make_yaw_rotation_np(np.asarray([resolved_pose.root_yaw_world], dtype=np.float64))[0]
    ).as_quat().astype(np.float32)

    # 连续帧统一 quaternion 符号，Unity Slerp 就不会因为 q/-q 表示等价旋转而跳变。
    if previous is not None:
        if np.dot(previous.root_rotation_xyzw, root_rotation) < 0.0:
            root_rotation *= -1.0
        signs = np.sum(previous.local_rotations_xyzw * local_quaternions, axis=-1) < 0.0
        local_quaternions[signs] *= -1.0

    pelvis_local = np.asarray(rest.pelvis_local_position, dtype=np.float32).copy()
    pelvis_local[1] = float(resolved_pose.hip_height)
    return UnityPoseFrame(
        root_position=np.asarray(resolved_pose.root_position_world, dtype=np.float32).copy(),
        root_rotation_xyzw=root_rotation,
        pelvis_local_position=pelvis_local,
        local_rotations_xyzw=local_quaternions,
    )


def run_unity_recording_inference(
    *,
    recording: ResampledTrackerRecording,
    rest: BodyFbxRest,
    predictor,
    dit,
    diffusion,
    device: torch.device,
    normalizer: RealtimePoseNormalizer,
    args,
) -> tuple[np.ndarray, list[UnityPoseFrame]]:
    start = find_first_core_window(recording.available)
    runtime = RealtimePoseRuntime(
        predictor,
        dit,
        diffusion,
        device,
        rest.rest_local_positions,
        rotation_6d_forward_up_np(rest.rest_local_rotations),
        normalizer,
        fabrik_iterations=args.fabrik_iterations,
        ik_direction_only_quality=args.ik_direction_only_quality,
        ik_residual_scale=args.ik_residual_scale,
        ik_position_solved_quality=args.ik_position_solved_quality,
        ik_gap_low=args.ik_gap_low,
        ik_gap_high=args.ik_gap_high,
        ik_direction_support=args.ik_direction_support,
        ik_untracked_strength=args.ik_untracked_strength,
    )
    runtime.initialize_history(
        build_bootstrap_pose_history(recording, rest, start),
        recording.positions[start : start + 11],
        recording.rotations_6d[start : start + 11],
        recording.floor_y,
    )

    output_times: list[float] = []
    output_frames: list[UnityPoseFrame] = []
    previous_frame: UnityPoseFrame | None = None
    noise_generator = create_eval_noise_generator(args.seed, device)
    for current in range(start + 11, len(recording.times)):
        core_available = recording.available[current, list(CORE_TRACKER_INDICES)].all()
        if not core_available:
            # Demo 中不重建 runtime 状态机；短时掉线只让画面保持上一姿态。
            if previous_frame is not None:
                output_times.append(float(recording.times[current]))
                output_frames.append(previous_frame)
            continue

        noise = torch.randn(
            (1, REALTIME_POSE_TARGET_DIM),
            generator=noise_generator,
            device=device,
        )
        result = runtime.step(
            recording.positions[current],
            recording.rotations_6d[current],
            recording.available[current],
            recording.floor_y,
            noise=noise,
        )
        previous_frame = resolved_pose_to_unity_frame(
            result.resolved_pose,
            rest,
            previous_frame,
        )
        output_times.append(float(recording.times[current]))
        output_frames.append(previous_frame)

    warmup_frames = max(0, int(args.warmup_frames))
    if len(output_frames) <= warmup_frames:
        raise RuntimeError("录制过短，没有超过初始化与 warmup 区间。")
    selected_times = np.asarray(output_times[warmup_frames:], dtype=np.float64)
    selected_times -= selected_times[0]
    return selected_times, output_frames[warmup_frames:]


def write_unity_pose_result(
    path: str | Path,
    times: np.ndarray,
    frames: list[UnityPoseFrame],
) -> Path:
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fps": int(REALTIME_POSE_FPS),
        "frames": [
            {
                "time": float(time),
                "rootPosition": frame.root_position.tolist(),
                "rootRotation": frame.root_rotation_xyzw.tolist(),
                "pelvisLocalPosition": frame.pelvis_local_position.tolist(),
                "localRotations": frame.local_rotations_xyzw.tolist(),
            }
            for time, frame in zip(times.tolist(), frames, strict=True)
        ],
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


# endregion


def main(argv: list[str] | None = None) -> Path:
    parser = build_arg_parser()
    args = parse_and_load_from_model(
        parser,
        argv,
        ignore_keys={
            "input",
            "output",
            "normalizer_dir",
            "body_fbx_rest_json",
            "device",
            "seed",
            "use_ema",
            "warmup_frames",
            "ignore_hip",
            "ignore_feet",
        },
    )
    fixseed(int(args.seed))
    device = resolve_device(args.device)
    print(f"[FLUID] device: {device}", flush=True)

    raw_recording = apply_tracker_availability_overrides(
        load_unity_tracker_recording(args.input),
        ignore_hip=bool(args.ignore_hip),
        ignore_feet=bool(args.ignore_feet),
    )
    if args.ignore_hip:
        print("[FLUID] Hip Tracker ignored for the full recording.", flush=True)
    if args.ignore_feet:
        print("[FLUID] Foot Trackers ignored for the full recording.", flush=True)
    recording = resample_tracker_recording(raw_recording)
    rest = load_body_fbx_rest(args.body_fbx_rest_json)
    dit, diffusion = create_model_and_diffusion(args)
    dit, weight_source = load_checkpoint_model(
        dit,
        args.model_path,
        device,
        use_ema=bool(args.use_ema),
    )
    predictor = load_realtime_pose_predictor(args.predictor_model_path, device)
    normalizer = RealtimePoseNormalizer(args.normalizer_dir)
    print(
        f"[FLUID] recording frames: {len(raw_recording.times)} -> {len(recording.times)} @ 30Hz",
        flush=True,
    )
    times, frames = run_unity_recording_inference(
        recording=recording,
        rest=rest,
        predictor=predictor,
        dit=dit,
        diffusion=diffusion,
        device=device,
        normalizer=normalizer,
        args=args,
    )
    output_path = write_unity_pose_result(args.output, times, frames)
    print(
        f"[FLUID] wrote {len(frames)} frames ({weight_source}): {output_path}",
        flush=True,
    )
    return output_path


if __name__ == "__main__":
    main()
