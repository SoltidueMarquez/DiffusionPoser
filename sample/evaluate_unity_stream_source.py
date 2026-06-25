from __future__ import annotations

import argparse
import json
from argparse import BooleanOptionalAction
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_loaders.generate_realtime_pose_tasks import load_realtime_source
from data_loaders.realtime_pose_dataset import encode_realtime_pose_features
from data_loaders.sensor_masking import (
    REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SCHEMA_NAMES,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_START,
    TRACKER_COUNT,
    get_schema_spec,
)
from eval.evaluate_realtime_pose_rollout import evaluate_rollout_file
from sample.render_realtime_pose_comparison import render_realtime_pose_comparison
from sample.simulate_unity_stream import (
    DEFAULT_TRACKER_IK_BLEND,
    DEFAULT_TRACKER_IK_DELTA_LIMIT,
    DEFAULT_TRACKER_IK_ITERATIONS,
    DEFAULT_TRACKER_IK_LR,
    DEFAULT_TRACKER_IK_TARGET_SMOOTHING,
    fk_joints_from_target,
    full_valid_sensor_mask,
    simulate_unity_stream,
)
from sample.utils import load_checkpoint_model
from utils import dist_util
from utils.model_util import create_model_and_diffusion
from utils.normalizer import RealtimePoseNormalizer
from utils.parser_util import (
    add_base_options,
    add_diffusion_options,
    add_model_options,
    add_sampling_options,
    load_args_json,
    parse_and_load_from_model,
    str2bool,
)
from utils.schema_resolution import has_explicit_schema_arg, resolve_runtime_schema


HISTORY_POSE_SOURCE_PREDICTED = "predicted"
HISTORY_POSE_SOURCE_REFERENCE = "reference"
HISTORY_POSE_SOURCE_CHOICES = (HISTORY_POSE_SOURCE_REFERENCE, HISTORY_POSE_SOURCE_PREDICTED)
WARMUP_TARGET_SOURCE_FIRST_FRAME = "first_frame"
WARMUP_TARGET_SOURCE_IDENTITY = "identity"
WARMUP_TARGET_SOURCE_CHOICES = (WARMUP_TARGET_SOURCE_FIRST_FRAME, WARMUP_TARGET_SOURCE_IDENTITY)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a realtime_pose_stationary5_v1 Unity stream against a long GT source.",
        allow_abbrev=False,
    )
    add_base_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sampling_options(parser)
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    group = parser.add_argument_group("long_sequence")
    group.add_argument("--source_path", required=True, type=str)
    group.add_argument("--schema", default=None, choices=REALTIME_POSE_SCHEMA_NAMES, type=str)
    group.add_argument("--normalizer_dir", required=True, type=str)
    group.add_argument("--normalize_input", default=True, type=str2bool)
    group.add_argument("--input_feats", default=schema.feature_dim, type=int)
    group.add_argument("--seq_len", default=REALTIME_POSE_SEQ_LEN, type=int)
    group.add_argument("--loop_count", default=1, type=int)
    group.add_argument("--initial_root_yaw", default=None, type=float)
    group.add_argument(
        "--history_pose_source",
        default=HISTORY_POSE_SOURCE_REFERENCE,
        choices=HISTORY_POSE_SOURCE_CHOICES,
        type=str,
        help="reference 表示只用长序列 GT 初始化前 60 帧 history，之后自回归；predicted 表示 warm-up 后全程自回归。",
    )
    group.add_argument(
        "--warmup_target_source",
        default=WARMUP_TARGET_SOURCE_FIRST_FRAME,
        choices=WARMUP_TARGET_SOURCE_CHOICES,
        type=str,
        help="predicted 历史模式下 warm-up target 的来源；identity 为运行时零/identity 初始化。",
    )
    group.add_argument("--render_mp4", default=True, type=str2bool)
    group.add_argument("--render_fps", default=30, type=int)
    group.add_argument("--render_stride", default=1, type=int)
    group.add_argument("--render_camera_mode", default="follow", choices=["global", "follow"], type=str)
    group.add_argument("--render_layout", default="overlay", choices=["split", "overlay"], type=str)
    group.add_argument("--render_local_radius", default=1.25, type=float)
    group.add_argument(
        "--root_correction",
        default=True,
        action=BooleanOptionalAction,
        help="用真实 waist tracker 修正在线 root yaw/root xz/root height。",
    )
    group.add_argument(
        "--tracker_ik",
        default=True,
        action=BooleanOptionalAction,
        help="推理后用已知 head/hands/feet tracker 做 position IK。",
    )
    group.add_argument("--tracker_ik_iterations", default=DEFAULT_TRACKER_IK_ITERATIONS, type=int)
    group.add_argument("--tracker_ik_lr", default=DEFAULT_TRACKER_IK_LR, type=float)
    group.add_argument("--tracker_ik_blend", default=DEFAULT_TRACKER_IK_BLEND, type=float)
    group.add_argument("--tracker_ik_target_smoothing", default=DEFAULT_TRACKER_IK_TARGET_SMOOTHING, type=float)
    group.add_argument("--tracker_ik_delta_limit", default=DEFAULT_TRACKER_IK_DELTA_LIMIT, type=float)
    return parser


def read_source_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        if "metadata" not in data.files:
            return {}
        value = data["metadata"]
    try:
        text = str(value.item())
    except Exception:
        text = str(value)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def load_v2_source_with_sensor_valid(path: Path, schema_name: str = REALTIME_POSE_SCHEMA_NAME) -> dict[str, np.ndarray]:
    metadata = read_source_metadata(path)
    schema = get_schema_spec(schema_name)
    source_schema_name = str(metadata.get("schema_name", schema.name))
    if source_schema_name != schema.name:
        raise ValueError(f"{path} schema_name 必须是 {schema.name}，实际为 {source_schema_name}")

    source = load_realtime_source(path, schema_name=schema.name)
    frame_count = int(source["tracker_pos_world"].shape[0])
    with np.load(path, allow_pickle=False) as data:
        if "sensor_valid" in data.files:
            sensor_valid = np.asarray(data["sensor_valid"], dtype=bool)
        else:
            sensor_valid = full_valid_sensor_mask(frame_count)
    if sensor_valid.shape != (frame_count, TRACKER_COUNT):
        raise ValueError(f"sensor_valid 应为 [{frame_count},{TRACKER_COUNT}]，实际为 {sensor_valid.shape}")
    source["sensor_valid"] = sensor_valid
    return source


def repeat_source_sequence(source: dict[str, np.ndarray], loop_count: int) -> dict[str, np.ndarray]:
    loops = max(1, int(loop_count))
    frame_count = int(source["tracker_pos_world"].shape[0])
    repeated: dict[str, np.ndarray] = {}
    for key, value in source.items():
        array = np.asarray(value)
        if key in {"joint_offsets_parent", "joint_rest_local_rotations_6d"}:
            repeated[key] = array.astype(np.float32, copy=True)
        elif array.ndim > 0 and array.shape[0] == frame_count:
            repeated[key] = np.concatenate([array] * loops, axis=0).astype(array.dtype, copy=False)
        else:
            repeated[key] = array.copy()
    return repeated


def decode_features_to_joints(
    features: np.ndarray,
    root_pos_world: np.ndarray,
    root_yaw: np.ndarray,
    joint_offsets_parent: np.ndarray,
    schema_name: str = REALTIME_POSE_SCHEMA_NAME,
    joint_rest_local_rotations_6d: np.ndarray | None = None,
) -> np.ndarray:
    schema = get_schema_spec(schema_name)
    values = np.asarray(features, dtype=np.float32)
    roots = np.asarray(root_pos_world, dtype=np.float32).copy()
    yaw = np.asarray(root_yaw, dtype=np.float32)
    joints = []
    for frame_index in range(values.shape[0]):
        joints.append(
            fk_joints_from_target(
                target_raw=values[frame_index],
                root_pos_world=roots[frame_index],
                root_yaw=float(yaw[frame_index]),
                joint_offsets_parent=joint_offsets_parent,
                schema_name=schema.name,
                joint_rest_local_rotations_6d=joint_rest_local_rotations_6d,
            )
        )
    return np.stack(joints, axis=0).astype(np.float32)


def build_long_sequence_payload(
    model,
    diffusion,
    source: dict[str, np.ndarray],
    device: torch.device,
    use_ddim: bool,
    normalizer: RealtimePoseNormalizer | None,
    initial_root_yaw: float | None = None,
    history_pose_source: str = HISTORY_POSE_SOURCE_REFERENCE,
    warmup_target_source: str = WARMUP_TARGET_SOURCE_FIRST_FRAME,
    root_correction: bool = True,
    tracker_ik: bool = True,
    tracker_ik_iterations: int = DEFAULT_TRACKER_IK_ITERATIONS,
    tracker_ik_lr: float = DEFAULT_TRACKER_IK_LR,
    tracker_ik_blend: float = DEFAULT_TRACKER_IK_BLEND,
    tracker_ik_target_smoothing: float = DEFAULT_TRACKER_IK_TARGET_SMOOTHING,
    tracker_ik_delta_limit: float = DEFAULT_TRACKER_IK_DELTA_LIMIT,
    ik_init_mode: str = "random",
    ik_init_timestep: int = -1,
    ik_init_iterations: int = 16,
    ik_init_lr: float = 0.03,
    ik_init_pos_weight: float = 1.0,
    ik_init_rot_weight: float = 0.2,
    ik_init_reg_weight: float = 0.01,
    ik_init_delta_limit: float = 0.15,
    schema_name: str = REALTIME_POSE_SCHEMA_NAME,
) -> dict[str, np.ndarray]:
    frame_count = int(source["tracker_pos_world"].shape[0])
    if frame_count <= REALTIME_POSE_TARGET_START:
        raise ValueError(f"长序列至少需要 {REALTIME_POSE_TARGET_START + 1} 帧，实际为 {frame_count}")
    if history_pose_source not in HISTORY_POSE_SOURCE_CHOICES:
        raise ValueError(f"history_pose_source 必须是 {HISTORY_POSE_SOURCE_CHOICES} 之一，实际为 {history_pose_source}")
    if warmup_target_source not in WARMUP_TARGET_SOURCE_CHOICES:
        raise ValueError(f"warmup_target_source 必须是 {WARMUP_TARGET_SOURCE_CHOICES} 之一，实际为 {warmup_target_source}")

    schema = get_schema_spec(schema_name)
    reference_features = encode_realtime_pose_features(source, schema_name=schema.name)
    use_reference_history = history_pose_source == HISTORY_POSE_SOURCE_REFERENCE
    warmup_target_raw = (
        None
        if use_reference_history or warmup_target_source == WARMUP_TARGET_SOURCE_IDENTITY
        else reference_features[0].copy()
    )
    stream_payload = simulate_unity_stream(
        model=model,
        diffusion=diffusion,
        tracker_pos_world=source["tracker_pos_world"],
        tracker_rot_world_6d=source["tracker_rot_world_6d"],
        sensor_valid=source["sensor_valid"],
        device=device,
        use_ddim=use_ddim,
        schema_name=schema.name,
        normalizer=normalizer,
        initial_root_yaw=float(source["root_yaw"][0] if initial_root_yaw is None else initial_root_yaw),
        warmup_target_raw=warmup_target_raw,
        history_features_raw=reference_features if use_reference_history else None,
        history_features_until_frame=REALTIME_POSE_TARGET_START if use_reference_history else None,
        reference_root_yaw=np.asarray(source["root_yaw"], dtype=np.float32) if use_reference_history else None,
        reference_root_pos_world=np.asarray(source["root_pos_world"], dtype=np.float32) if use_reference_history else None,
        joint_offsets_parent=np.asarray(source["joint_offsets_parent"], dtype=np.float32),
        joint_rest_local_rotations_6d=np.asarray(source["joint_rest_local_rotations_6d"], dtype=np.float32)
        if "joint_rest_local_rotations_6d" in source
        else None,
        root_correction=bool(root_correction),
        tracker_ik=bool(tracker_ik),
        tracker_ik_iterations=int(tracker_ik_iterations),
        tracker_ik_lr=float(tracker_ik_lr),
        tracker_ik_blend=float(tracker_ik_blend),
        tracker_ik_target_smoothing=float(tracker_ik_target_smoothing),
        tracker_ik_delta_limit=float(tracker_ik_delta_limit),
        ik_init_mode=ik_init_mode,
        ik_init_timestep=int(ik_init_timestep),
        ik_init_iterations=int(ik_init_iterations),
        ik_init_lr=float(ik_init_lr),
        ik_init_pos_weight=float(ik_init_pos_weight),
        ik_init_rot_weight=float(ik_init_rot_weight),
        ik_init_reg_weight=float(ik_init_reg_weight),
        ik_init_delta_limit=float(ik_init_delta_limit),
    )

    predicted_features = np.asarray(stream_payload["predicted_features_raw"][0], dtype=np.float32)
    root_yaw_predicted = np.asarray(stream_payload["root_yaw_predicted"][0], dtype=np.float32)
    root_pos_predicted = np.asarray(stream_payload["root_pos_world_predicted"][0], dtype=np.float32)
    predicted_joints = decode_features_to_joints(
        features=predicted_features,
        root_pos_world=root_pos_predicted,
        root_yaw=root_yaw_predicted,
        joint_offsets_parent=source["joint_offsets_parent"],
        schema_name=schema.name,
        joint_rest_local_rotations_6d=source.get("joint_rest_local_rotations_6d"),
    )

    return {
        "schema_name": np.asarray(schema.name),
        "feature_space": np.asarray("raw"),
        "input_feature_space": stream_payload["input_feature_space"],
        "reference_features_raw": reference_features[None],
        "predicted_features_raw": predicted_features[None],
        "conditioned_features_raw": stream_payload["conditioned_features_raw"],
        "reference_joints_world": np.asarray(source["joints_world"], dtype=np.float32)[None],
        "predicted_joints_world": predicted_joints[None],
        "root_yaw_reference": np.asarray(source["root_yaw"], dtype=np.float32)[None],
        "root_yaw_predicted": root_yaw_predicted[None],
        "root_pos_world_reference": np.asarray(source["root_pos_world"], dtype=np.float32)[None],
        "root_pos_world_predicted": root_pos_predicted[None],
        "tracker_pos_world": np.asarray(source["tracker_pos_world"], dtype=np.float32)[None],
        "tracker_rot_world_6d": np.asarray(source["tracker_rot_world_6d"], dtype=np.float32)[None],
        "sensor_valid": np.asarray(source["sensor_valid"], dtype=bool)[None],
        "validity_ok": stream_payload["validity_ok"],
        "is_predicted": stream_payload["is_predicted"],
        "eval_frame_mask": stream_payload["eval_frame_mask"],
        "warmup_frames": np.asarray(REALTIME_POSE_TARGET_START, dtype=np.int64),
        "root_correction": stream_payload["root_correction"],
        "tracker_ik": stream_payload["tracker_ik"],
        "tracker_ik_iterations": stream_payload["tracker_ik_iterations"],
        "tracker_ik_lr": stream_payload["tracker_ik_lr"],
        "tracker_ik_blend": stream_payload["tracker_ik_blend"],
        "tracker_ik_target_smoothing": stream_payload["tracker_ik_target_smoothing"],
        "tracker_ik_delta_limit": stream_payload["tracker_ik_delta_limit"],
        "ik_init_mode": stream_payload["ik_init_mode"],
        "ik_init_timestep": stream_payload["ik_init_timestep"],
        "ik_init_iterations": stream_payload["ik_init_iterations"],
        "ik_init_lr": stream_payload["ik_init_lr"],
        "ik_init_pos_weight": stream_payload["ik_init_pos_weight"],
        "ik_init_rot_weight": stream_payload["ik_init_rot_weight"],
        "ik_init_reg_weight": stream_payload["ik_init_reg_weight"],
        "ik_init_delta_limit": stream_payload["ik_init_delta_limit"],
        "metadata": np.asarray(
            {
                "schema_name": schema.name,
                "frames": frame_count,
                "warmup_frames": REALTIME_POSE_TARGET_START,
                "history_pose_source": history_pose_source,
                "warmup_target_source": warmup_target_source,
                "reference_history_frames": REALTIME_POSE_TARGET_START if use_reference_history else 0,
                "autoregressive_after_warmup": True,
                "root_correction": bool(root_correction),
                "tracker_ik": bool(tracker_ik),
                "tracker_ik_iterations": int(tracker_ik_iterations),
                "tracker_ik_lr": float(tracker_ik_lr),
                "tracker_ik_blend": float(tracker_ik_blend),
                "tracker_ik_target_smoothing": float(tracker_ik_target_smoothing),
                "tracker_ik_delta_limit": float(tracker_ik_delta_limit),
                "ik_init_mode": str(ik_init_mode),
                "ik_init_timestep": int(stream_payload["ik_init_timestep"]),
                "ik_init_iterations": int(ik_init_iterations),
                "ik_init_lr": float(ik_init_lr),
                "ik_init_pos_weight": float(ik_init_pos_weight),
                "ik_init_rot_weight": float(ik_init_rot_weight),
                "ik_init_reg_weight": float(ik_init_reg_weight),
                "ik_init_delta_limit": float(ik_init_delta_limit),
            },
            dtype=object,
        ),
    }


def save_long_sequence_result(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def write_eval_summary(result_path: Path, output_json: Path) -> dict[str, Any]:
    result = evaluate_rollout_file(result_path)
    summary = {key: value for key, value in result.items() if key != "path" and not isinstance(value, list)}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump({"summary": summary, "files": [result]}, file, indent=2, ensure_ascii=False)
    return summary


def validate_v2_runtime_args(args: argparse.Namespace, cli_schema_explicit: bool = False) -> None:
    checkpoint_args = load_args_json(Path(args.model_path))
    args.schema = resolve_runtime_schema(
        cli_schema=getattr(args, "schema", None),
        checkpoint_args=checkpoint_args,
        cli_schema_explicit=cli_schema_explicit,
    )
    schema = get_schema_spec(args.schema)
    if int(args.seq_len) != REALTIME_POSE_SEQ_LEN:
        raise ValueError(f"{schema.name} 固定使用 {REALTIME_POSE_SEQ_LEN} 帧窗口，实际为 {args.seq_len}")
    if int(args.input_feats) != schema.feature_dim:
        raise ValueError(f"{schema.name} input_feats 应为 {schema.feature_dim}，实际为 {args.input_feats}")


def main(argv: list[str] | None = None) -> dict[str, Path]:
    parser = build_arg_parser()
    cli_schema_explicit = has_explicit_schema_arg(argv)
    args = parse_and_load_from_model(parser, argv=argv)
    validate_v2_runtime_args(args, cli_schema_explicit=cli_schema_explicit)

    source = load_v2_source_with_sensor_valid(Path(args.source_path).resolve(), schema_name=args.schema)
    source = repeat_source_sequence(source, loop_count=int(args.loop_count))
    normalizer = RealtimePoseNormalizer(args.normalizer_dir, schema_name=args.schema)
    if not bool(args.normalize_input):
        normalizer = None

    dist_util.setup_dist(args.device if args.cuda else -1)
    device = dist_util.dev()
    model, diffusion = create_model_and_diffusion(args)
    model, source_name = load_checkpoint_model(model, args.model_path, device=device, use_ema=args.use_ema)
    payload = build_long_sequence_payload(
        model=model,
        diffusion=diffusion,
        source=source,
        device=device,
        use_ddim=str(args.ts_respace).startswith("ddim"),
        normalizer=normalizer,
        initial_root_yaw=args.initial_root_yaw,
        history_pose_source=str(args.history_pose_source),
        warmup_target_source=str(args.warmup_target_source),
        root_correction=bool(args.root_correction),
        tracker_ik=bool(args.tracker_ik),
        tracker_ik_iterations=int(args.tracker_ik_iterations),
        tracker_ik_lr=float(args.tracker_ik_lr),
        tracker_ik_blend=float(args.tracker_ik_blend),
        tracker_ik_target_smoothing=float(args.tracker_ik_target_smoothing),
        tracker_ik_delta_limit=float(args.tracker_ik_delta_limit),
        ik_init_mode=args.ik_init_mode,
        ik_init_timestep=int(args.ik_init_timestep),
        ik_init_iterations=int(args.ik_init_iterations),
        ik_init_lr=float(args.ik_init_lr),
        ik_init_pos_weight=float(args.ik_init_pos_weight),
        ik_init_rot_weight=float(args.ik_init_rot_weight),
        ik_init_reg_weight=float(args.ik_init_reg_weight),
        ik_init_delta_limit=float(args.ik_init_delta_limit),
        schema_name=args.schema,
    )
    metadata = dict(payload["metadata"].item())
    metadata.update({"weights": source_name, "loop_count": int(args.loop_count)})
    payload["metadata"] = np.asarray(metadata, dtype=object)

    output_dir = Path(args.output_dir or "output/unity_stream_long_sequence").resolve()
    result_path = output_dir / "unity_stream_long_sequence_result.npz"
    summary_path = output_dir / "unity_stream_eval_summary.json"
    save_long_sequence_result(result_path, payload)
    write_eval_summary(result_path, summary_path)

    outputs = {"result_path": result_path, "summary_path": summary_path}
    if bool(args.render_mp4):
        mp4_path = output_dir / "unity_stream_comparison.mp4"
        render_realtime_pose_comparison(
            output_path=mp4_path,
            reference_joints=payload["reference_joints_world"],
            predicted_joints=payload["predicted_joints_world"],
            tracker_pos_world=payload["tracker_pos_world"],
            sensor_valid=payload["sensor_valid"],
            eval_frame_mask=payload["eval_frame_mask"],
            root_yaw_reference=payload["root_yaw_reference"],
            root_yaw_predicted=payload["root_yaw_predicted"],
            fps=int(args.render_fps),
            stride=int(args.render_stride),
            camera_mode=str(args.render_camera_mode),
            layout=str(args.render_layout),
            local_radius=float(args.render_local_radius),
        )
        outputs["mp4_path"] = mp4_path

    print(f"[evaluate_unity_stream_source] result={result_path} summary={summary_path}")
    return outputs


if __name__ == "__main__":
    main()
