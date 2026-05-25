from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from data_loaders.realtime_pose_dataset import zero_missing_tracker_channels
from data_loaders.realtime_pose_kinematics import (
    integrate_root_delta_xz_ref,
    make_yaw_rotation_np,
    rotation_6d_forward_up_np,
    rotation_6d_to_matrix_np,
)
from data_loaders.sensor_masking import (
    DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    FOOT_CONTACT_DIM,
    HIP_TRACKER_INDEX,
    MIN_VALID_TRACKERS,
    REALTIME_POSE_SCHEMA_NAMES,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_START,
    ROOT_DELTA_XZ_DIM,
    SMPL_JOINT_COUNT,
    TRACKER_COUNT,
    get_schema_spec,
)
from sample.reconstruct_stream import build_realtime_inpaint_mask, reconstruct_batch, tensor_bct_to_numpy_btc
from sample.utils import load_checkpoint_model
from utils import dist_util
from utils.model_util import create_model_and_diffusion
from utils.normalizer import RealtimePoseNormalizer
from utils.parser_util import (
    add_base_options,
    add_diffusion_options,
    add_model_options,
    add_sampling_options,
    parse_and_load_from_model,
    str2bool,
)


IDENTITY_6D = np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
INVALID_FRAME_POLICY_HOLD = "hold"
INVALID_FRAME_POLICY_RAISE = "raise"
INVALID_FRAME_POLICIES = (INVALID_FRAME_POLICY_HOLD, INVALID_FRAME_POLICY_RAISE)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate Unity realtime tracker-only input in Python.")
    add_base_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sampling_options(parser)
    default_schema = get_schema_spec(DEFAULT_REALTIME_POSE_SCHEMA_NAME)
    stream = parser.add_argument_group("unity_stream")
    stream.add_argument("--tracker_stream_path", required=True, type=str, help="包含 tracker_pos_world/sensor_valid 的 npz。")
    stream.add_argument("--schema", default=DEFAULT_REALTIME_POSE_SCHEMA_NAME, choices=REALTIME_POSE_SCHEMA_NAMES, type=str)
    stream.add_argument("--input_feats", default=default_schema.feature_dim, type=int)
    stream.add_argument("--seq_len", default=REALTIME_POSE_SEQ_LEN, type=int)
    stream.add_argument("--normalizer_dir", default="", type=str)
    stream.add_argument("--normalize_input", default=True, type=str2bool)
    stream.add_argument("--initial_root_yaw", default=0.0, type=float)
    stream.add_argument(
        "--invalid_frame_policy",
        default=INVALID_FRAME_POLICY_HOLD,
        choices=INVALID_FRAME_POLICIES,
        type=str,
        help="tracker 有效性不满足运行时合约时，hold 表示沿用上一帧输出。",
    )
    stream.add_argument("--assume_identity_tracker_rot", action="store_true")
    stream.add_argument("--limit", default=0, type=int)
    return parser


def full_valid_sensor_mask(frame_count: int) -> np.ndarray:
    return np.ones((int(frame_count), TRACKER_COUNT), dtype=bool)


def identity_tracker_rotations(frame_count: int) -> np.ndarray:
    return np.tile(IDENTITY_6D, (int(frame_count), TRACKER_COUNT, 1)).astype(np.float32)


def load_tracker_stream(
    path: Path,
    assume_identity_tracker_rot: bool = False,
    limit: int = 0,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if "tracker_pos_world" not in data.files:
            raise KeyError(f"{path} 缺少 tracker_pos_world 字段。")
        tracker_pos_world = np.asarray(data["tracker_pos_world"], dtype=np.float32)
        if tracker_pos_world.ndim != 3 or tracker_pos_world.shape[1:] != (TRACKER_COUNT, 3):
            raise ValueError(f"tracker_pos_world 应为 [T,{TRACKER_COUNT},3]，实际为 {tracker_pos_world.shape}")

        if "tracker_rot_world_6d" in data.files:
            tracker_rot_world_6d = np.asarray(data["tracker_rot_world_6d"], dtype=np.float32)
        elif assume_identity_tracker_rot:
            tracker_rot_world_6d = identity_tracker_rotations(tracker_pos_world.shape[0])
        else:
            raise KeyError(
                f"{path} 缺少 tracker_rot_world_6d；如果只想调试位置流，请显式传 --assume_identity_tracker_rot。"
            )
        if tracker_rot_world_6d.shape != (tracker_pos_world.shape[0], TRACKER_COUNT, 6):
            raise ValueError(
                f"tracker_rot_world_6d 应为 [T,{TRACKER_COUNT},6]，实际为 {tracker_rot_world_6d.shape}"
            )

        sensor_valid = (
            np.asarray(data["sensor_valid"], dtype=bool)
            if "sensor_valid" in data.files
            else full_valid_sensor_mask(tracker_pos_world.shape[0])
        )
        if sensor_valid.shape != (tracker_pos_world.shape[0], TRACKER_COUNT):
            raise ValueError(f"sensor_valid 应为 [T,{TRACKER_COUNT}]，实际为 {sensor_valid.shape}")

    frame_count = tracker_pos_world.shape[0]
    if int(limit) > 0:
        frame_count = min(frame_count, int(limit))
    return {
        "tracker_pos_world": tracker_pos_world[:frame_count],
        "tracker_rot_world_6d": tracker_rot_world_6d[:frame_count],
        "sensor_valid": sensor_valid[:frame_count],
    }


def sensor_validity_ok(sensor_valid: np.ndarray) -> bool:
    valid = np.asarray(sensor_valid, dtype=bool)
    return bool(valid.shape == (TRACKER_COUNT,) and valid[HIP_TRACKER_INDEX] and valid.sum() >= MIN_VALID_TRACKERS)


def estimate_root_pos_from_hip_tracker(tracker_pos_world: np.ndarray) -> np.ndarray:
    root_pos = np.asarray(tracker_pos_world[HIP_TRACKER_INDEX], dtype=np.float32).copy()
    root_pos[1] = 0.0
    return root_pos


def encode_unity_tracker_frame(
    tracker_pos_world: np.ndarray,
    tracker_rot_world_6d: np.ndarray,
    sensor_valid: np.ndarray,
    reference_root_yaw: float,
    schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    root_pos_world: np.ndarray | None = None,
) -> np.ndarray:
    """
    把 Unity 当前帧 tracker transform 编成模型 raw feature。

    运行时只能使用上一帧预测 root_yaw 作为 reference yaw；当前帧 target
    通道会在采样前置零，所以这里仅填 tracker 条件和 sensor_valid。
    """

    schema = get_schema_spec(schema_name)
    tracker_pos = np.asarray(tracker_pos_world, dtype=np.float32)
    tracker_rot = np.asarray(tracker_rot_world_6d, dtype=np.float32)
    valid = np.asarray(sensor_valid, dtype=bool)
    if tracker_pos.shape != (TRACKER_COUNT, 3):
        raise ValueError(f"单帧 tracker_pos_world 应为 [{TRACKER_COUNT},3]，实际为 {tracker_pos.shape}")
    if tracker_rot.shape != (TRACKER_COUNT, 6):
        raise ValueError(f"单帧 tracker_rot_world_6d 应为 [{TRACKER_COUNT},6]，实际为 {tracker_rot.shape}")
    if valid.shape != (TRACKER_COUNT,):
        raise ValueError(f"单帧 sensor_valid 应为 [{TRACKER_COUNT}]，实际为 {valid.shape}")

    root_pos = (
        estimate_root_pos_from_hip_tracker(tracker_pos)
        if root_pos_world is None
        else np.asarray(root_pos_world, dtype=np.float32).copy()
    )
    root_pos[1] = 0.0

    yaw_rotation = make_yaw_rotation_np(np.asarray([float(reference_root_yaw)], dtype=np.float64))[0]
    tracker_pos_ref = (tracker_pos.astype(np.float64) - root_pos.astype(np.float64)[None]) @ yaw_rotation

    tracker_rot_world = rotation_6d_to_matrix_np(tracker_rot)
    tracker_rot_ref = yaw_rotation.T[None] @ tracker_rot_world
    tracker_rot_ref_6d = rotation_6d_forward_up_np(tracker_rot_ref)

    features = np.zeros((schema.feature_dim,), dtype=np.float32)
    features[schema.tracker_pos_slice()] = tracker_pos_ref.reshape(-1).astype(np.float32)
    features[schema.tracker_rot_slice()] = tracker_rot_ref_6d.reshape(-1).astype(np.float32)
    features[schema.sensor_valid_slice()] = valid.astype(np.float32)
    for tracker_index in range(TRACKER_COUNT):
        if valid[tracker_index]:
            continue
        features[schema.tracker_pos_slice(tracker_index)] = 0.0
        features[schema.tracker_rot_slice(tracker_index)] = 0.0
    return features


def initial_target_feature(schema_name: str, root_height: float = 0.0) -> np.ndarray:
    schema = get_schema_spec(schema_name)
    features = np.zeros((schema.feature_dim,), dtype=np.float32)
    features[schema.body_pose_slice()] = np.tile(IDENTITY_6D, SMPL_JOINT_COUNT)
    features[schema.root_yaw_delta_slice()] = np.asarray([0.0, 1.0], dtype=np.float32)
    if schema.supports_root_motion:
        features[schema.root_delta_xz_slice()] = np.zeros((ROOT_DELTA_XZ_DIM,), dtype=np.float32)
        features[schema.root_height_slice()] = np.asarray([float(root_height)], dtype=np.float32)
    if schema.supports_contact:
        features[schema.foot_contact_slice()] = np.zeros((FOOT_CONTACT_DIM,), dtype=np.float32)
    return features


def normalize_conditioned_window(
    window_raw: np.ndarray,
    normalizer: RealtimePoseNormalizer | None,
    schema_name: str,
) -> np.ndarray:
    schema = get_schema_spec(schema_name)
    conditioned = window_raw.copy()
    if normalizer is not None:
        conditioned = np.asarray(normalizer.normalize(conditioned), dtype=np.float32)
        sensor_valid = np.asarray(window_raw[:, schema.sensor_valid_slice()], dtype=bool)
        zero_missing_tracker_channels(features=conditioned, sensor_valid=sensor_valid, schema_name=schema.name)
    conditioned[REALTIME_POSE_TARGET_START, schema.target_slice()] = 0.0
    return conditioned.astype(np.float32, copy=False)


def inverse_feature_window(
    window: np.ndarray,
    normalizer: RealtimePoseNormalizer | None,
) -> np.ndarray:
    if normalizer is None:
        return window.astype(np.float32, copy=False)
    return np.asarray(normalizer.inverse(window), dtype=np.float32)


@dataclass
class UnityStreamState:
    schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME
    initial_root_yaw: float = 0.0
    invalid_frame_policy: str = INVALID_FRAME_POLICY_HOLD
    history_raw: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=REALTIME_POSE_TARGET_START))
    current_root_yaw: float = field(init=False)
    current_root_pos_world: np.ndarray | None = None
    last_output_raw: np.ndarray | None = None
    last_root_pos_world: np.ndarray | None = None
    last_validity_ok: bool = True

    def __post_init__(self) -> None:
        if self.invalid_frame_policy not in INVALID_FRAME_POLICIES:
            raise ValueError(f"未知 invalid_frame_policy={self.invalid_frame_policy}")
        self.schema = get_schema_spec(self.schema_name)
        self.current_root_yaw = float(self.initial_root_yaw)

    def has_full_history(self) -> bool:
        return len(self.history_raw) == REALTIME_POSE_TARGET_START

    def make_initial_history_frame(self, tracker_feature_raw: np.ndarray, root_height: float = 0.0) -> np.ndarray:
        frame = np.asarray(tracker_feature_raw, dtype=np.float32).copy()
        frame[self.schema.target_slice()] = initial_target_feature(self.schema.name, root_height=root_height)[
            self.schema.target_slice()
        ]
        return frame

    def append_warmup_frame(
        self,
        tracker_feature_raw: np.ndarray,
        root_pos_world: np.ndarray,
        root_height: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.current_root_pos_world is None:
            self.current_root_pos_world = np.asarray(root_pos_world, dtype=np.float32).copy()
        history_frame = self.make_initial_history_frame(tracker_feature_raw, root_height=root_height)
        self.history_raw.append(history_frame)
        self.last_output_raw = history_frame.copy()
        self.last_root_pos_world = self.current_root_pos_world.copy()
        return history_frame.copy(), self.current_root_pos_world.copy()

    def build_window_raw(self, tracker_feature_raw: np.ndarray) -> np.ndarray:
        if not self.has_full_history():
            raise RuntimeError("Unity stream history 尚未填满 60 帧。")
        current = np.asarray(tracker_feature_raw, dtype=np.float32).copy()
        current[self.schema.target_slice()] = 0.0
        return np.concatenate([np.stack(list(self.history_raw), axis=0), current[None]], axis=0)

    def append_output_frame(
        self,
        tracker_feature_raw: np.ndarray,
        output_frame_raw: np.ndarray,
        root_pos_world: np.ndarray,
    ) -> None:
        history_frame = np.asarray(tracker_feature_raw, dtype=np.float32).copy()
        history_frame[self.schema.target_slice()] = np.asarray(output_frame_raw, dtype=np.float32)[self.schema.target_slice()]
        self.history_raw.append(history_frame)
        self.last_output_raw = history_frame.copy()
        self.last_root_pos_world = np.asarray(root_pos_world, dtype=np.float32).copy()

    def accept_prediction(
        self,
        tracker_feature_raw: np.ndarray,
        predicted_frame_raw: np.ndarray,
        fallback_root_pos_world: np.ndarray,
    ) -> np.ndarray:
        predicted = np.asarray(predicted_frame_raw, dtype=np.float32)
        prev_root_yaw = float(self.current_root_yaw)
        prev_root_pos = (
            np.asarray(fallback_root_pos_world, dtype=np.float32).copy()
            if self.current_root_pos_world is None
            else self.current_root_pos_world.copy()
        )
        yaw_delta = predicted[self.schema.root_yaw_delta_slice()]
        self.current_root_yaw = float(prev_root_yaw + np.arctan2(float(yaw_delta[0]), float(yaw_delta[1])))

        if self.schema.supports_root_motion:
            root_pos = integrate_root_delta_xz_ref(
                prev_root_pos_world=prev_root_pos[None],
                prev_root_yaw=np.asarray([prev_root_yaw], dtype=np.float32),
                root_delta_xz_ref=predicted[self.schema.root_delta_xz_slice()][None],
            )[0]
            root_pos[1] = 0.0
        else:
            root_pos = np.asarray(fallback_root_pos_world, dtype=np.float32).copy()

        self.current_root_pos_world = root_pos.astype(np.float32)
        self.append_output_frame(tracker_feature_raw, predicted, root_pos_world=self.current_root_pos_world)
        return self.current_root_pos_world.copy()

    def hold_output(self, tracker_feature_raw: np.ndarray, root_pos_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.last_output_raw is None:
            held = self.make_initial_history_frame(
                tracker_feature_raw,
                root_height=float(root_pos_world[1]) if root_pos_world.shape == (3,) else 0.0,
            )
        else:
            held = self.last_output_raw.copy()
        root_pos = (
            np.asarray(root_pos_world, dtype=np.float32).copy()
            if self.current_root_pos_world is None
            else self.current_root_pos_world.copy()
        )
        self.append_output_frame(tracker_feature_raw, held, root_pos_world=root_pos)
        return held.copy(), root_pos.copy()


def simulate_unity_stream(
    model,
    diffusion,
    tracker_pos_world: np.ndarray,
    tracker_rot_world_6d: np.ndarray,
    sensor_valid: np.ndarray,
    device: torch.device,
    use_ddim: bool,
    schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    normalizer: RealtimePoseNormalizer | None = None,
    initial_root_yaw: float = 0.0,
    invalid_frame_policy: str = INVALID_FRAME_POLICY_HOLD,
    warmup_target_raw: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    schema = get_schema_spec(schema_name)
    state = UnityStreamState(
        schema_name=schema.name,
        initial_root_yaw=initial_root_yaw,
        invalid_frame_policy=invalid_frame_policy,
    )
    warmup_target = None if warmup_target_raw is None else np.asarray(warmup_target_raw, dtype=np.float32)
    if warmup_target is not None and warmup_target.shape != (schema.feature_dim,):
        raise ValueError(f"warmup_target_raw 应为 [{schema.feature_dim}]，实际为 {warmup_target.shape}")

    predicted_features = []
    conditioned_features = []
    root_yaw_predicted = []
    root_pos_predicted = []
    validity_flags = []
    is_predicted = []

    for frame_index in range(int(tracker_pos_world.shape[0])):
        frame_valid = np.asarray(sensor_valid[frame_index], dtype=bool)
        validity_ok = sensor_validity_ok(frame_valid)
        if not validity_ok and invalid_frame_policy == INVALID_FRAME_POLICY_RAISE:
            raise ValueError(f"第 {frame_index} 帧 tracker 有效性不满足运行时合约：{frame_valid.astype(int).tolist()}")

        root_pos = estimate_root_pos_from_hip_tracker(tracker_pos_world[frame_index])
        tracker_feature_raw = encode_unity_tracker_frame(
            tracker_pos_world=tracker_pos_world[frame_index],
            tracker_rot_world_6d=tracker_rot_world_6d[frame_index],
            sensor_valid=frame_valid,
            reference_root_yaw=state.current_root_yaw,
            schema_name=schema.name,
            root_pos_world=root_pos,
        )

        if not state.has_full_history():
            predicted_frame_raw, output_root_pos = state.append_warmup_frame(
                tracker_feature_raw=tracker_feature_raw,
                root_pos_world=root_pos,
                root_height=float(tracker_pos_world[frame_index, HIP_TRACKER_INDEX, 1]),
            )
            if warmup_target is not None:
                predicted_frame_raw[schema.target_slice()] = warmup_target[schema.target_slice()]
                state.history_raw[-1][schema.target_slice()] = warmup_target[schema.target_slice()]
                state.last_output_raw = state.history_raw[-1].copy()
            conditioned_frame_raw = tracker_feature_raw.copy()
            conditioned_frame_raw[schema.target_slice()] = 0.0
            frame_predicted = False
        elif not validity_ok:
            predicted_frame_raw, output_root_pos = state.hold_output(tracker_feature_raw, root_pos_world=root_pos)
            conditioned_frame_raw = tracker_feature_raw.copy()
            conditioned_frame_raw[schema.target_slice()] = 0.0
            frame_predicted = False
        else:
            window_raw = state.build_window_raw(tracker_feature_raw)
            conditioned_raw = normalize_conditioned_window(window_raw, normalizer=normalizer, schema_name=schema.name)
            batch = {
                "conditioned_x": torch.from_numpy(conditioned_raw.T).unsqueeze(0).float().to(device),
                "valid_frame_mask": torch.ones(1, REALTIME_POSE_SEQ_LEN, dtype=torch.bool, device=device),
            }
            reconstructed = reconstruct_batch(
                model=model,
                diffusion=diffusion,
                batch=batch,
                device=device,
                use_ddim=use_ddim,
                schema_name=schema.name,
            )
            reconstructed_np = tensor_bct_to_numpy_btc(reconstructed)[0]
            reconstructed_raw = inverse_feature_window(reconstructed_np, normalizer=normalizer)
            predicted_frame_raw = reconstructed_raw[REALTIME_POSE_TARGET_START].copy()
            conditioned_frame_raw = inverse_feature_window(conditioned_raw, normalizer=normalizer)[REALTIME_POSE_TARGET_START]
            output_root_pos = state.accept_prediction(
                tracker_feature_raw=tracker_feature_raw,
                predicted_frame_raw=predicted_frame_raw,
                fallback_root_pos_world=root_pos,
            )
            frame_predicted = True

        predicted_features.append(predicted_frame_raw.astype(np.float32))
        conditioned_features.append(conditioned_frame_raw.astype(np.float32))
        root_yaw_predicted.append(float(state.current_root_yaw))
        root_pos_predicted.append(output_root_pos.astype(np.float32))
        validity_flags.append(bool(validity_ok))
        is_predicted.append(bool(frame_predicted))
        state.last_validity_ok = bool(validity_ok)

    predicted_mask = np.asarray(is_predicted, dtype=bool)
    validity_mask = np.asarray(validity_flags, dtype=bool)
    return {
        "schema_name": np.asarray(schema.name),
        "feature_space": np.asarray("raw"),
        "input_feature_space": np.asarray("normalized" if normalizer is not None else "raw"),
        "conditioned_features_raw": np.asarray(conditioned_features, dtype=np.float32)[None],
        "predicted_features_raw": np.asarray(predicted_features, dtype=np.float32)[None],
        "root_yaw_predicted": np.asarray(root_yaw_predicted, dtype=np.float32)[None],
        "root_pos_world_predicted": np.asarray(root_pos_predicted, dtype=np.float32)[None],
        "root_pos_world_estimated": np.asarray(root_pos_predicted, dtype=np.float32)[None],
        "tracker_pos_world": np.asarray(tracker_pos_world, dtype=np.float32)[None],
        "tracker_rot_world_6d": np.asarray(tracker_rot_world_6d, dtype=np.float32)[None],
        "sensor_valid": np.asarray(sensor_valid, dtype=bool)[None],
        "validity_ok": validity_mask[None],
        "is_predicted": predicted_mask[None],
        "eval_frame_mask": (predicted_mask & validity_mask)[None],
        "warmup_frames": np.asarray(REALTIME_POSE_TARGET_START, dtype=np.int64),
        "inpaint_mask": build_realtime_inpaint_mask(1, torch.device("cpu"), schema_name=schema.name)
        .cpu()
        .numpy()
        .transpose(0, 2, 1),
    }


def save_simulation(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def main(argv: list[str] | None = None) -> dict[str, Path]:
    parser = build_arg_parser()
    args = parse_and_load_from_model(parser, argv=argv)
    if int(args.seq_len) != REALTIME_POSE_SEQ_LEN:
        raise ValueError(f"Unity realtime stream 固定使用 {REALTIME_POSE_SEQ_LEN} 帧窗口，实际为 {args.seq_len}")
    schema = get_schema_spec(args.schema)
    if int(args.input_feats) != schema.feature_dim:
        raise ValueError(f"{schema.name} input_feats 应为 {schema.feature_dim}，实际为 {args.input_feats}")

    stream = load_tracker_stream(
        Path(args.tracker_stream_path).resolve(),
        assume_identity_tracker_rot=bool(args.assume_identity_tracker_rot),
        limit=int(args.limit),
    )
    normalizer = (
        RealtimePoseNormalizer(args.normalizer_dir, schema_name=schema.name)
        if bool(args.normalize_input)
        else None
    )

    dist_util.setup_dist(args.device if args.cuda else -1)
    device = dist_util.dev()
    model, diffusion = create_model_and_diffusion(args)
    model, source = load_checkpoint_model(model, args.model_path, device=device, use_ema=args.use_ema)
    payload = simulate_unity_stream(
        model=model,
        diffusion=diffusion,
        tracker_pos_world=stream["tracker_pos_world"],
        tracker_rot_world_6d=stream["tracker_rot_world_6d"],
        sensor_valid=stream["sensor_valid"],
        device=device,
        use_ddim=str(args.ts_respace).startswith("ddim"),
        schema_name=schema.name,
        normalizer=normalizer,
        initial_root_yaw=float(args.initial_root_yaw),
        invalid_frame_policy=args.invalid_frame_policy,
    )
    output_dir = Path(args.output_dir or "output/unity_stream_simulation").resolve()
    output_path = output_dir / "unity_stream_simulation.npz"
    save_simulation(output_path, payload)
    print(f"[simulate_unity_stream] weights={source} output={output_path}")
    return {"output_path": output_path}


if __name__ == "__main__":
    main()
