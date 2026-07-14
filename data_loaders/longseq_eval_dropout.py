from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from data_loaders.realtime_pose_dataset import dropout_non_head_trackers
from data_loaders.sensor_masking import (
    HEAD_TRACKER_INDEX,
    TRACKER_COUNT,
    TRACKER_MASK_POLICY_DYNAMIC_CATEGORIES,
    TRACKER_MASK_POLICY_FIXED_CATEGORIES,
    TRACKER_MASK_POLICY_TASK,
    make_tracker_pattern,
    normalize_tracker_pattern_categories,
    validate_sensor_valid,
)
from utils.parser_util import str2bool


DROPOUT_PRESET_NONE = "none"
DROPOUT_PRESET_TRACKER_MASK_TRAIN = "tracker_mask_train"
DROPOUT_PRESET_TRAIN_DEFAULT = "train_default"
DROPOUT_PRESETS = (DROPOUT_PRESET_NONE, DROPOUT_PRESET_TRACKER_MASK_TRAIN, DROPOUT_PRESET_TRAIN_DEFAULT)


@dataclass(frozen=True)
class LongseqDropoutConfig:
    preset: str = DROPOUT_PRESET_NONE
    tracker_mask_policy: str = TRACKER_MASK_POLICY_TASK
    tracker_mask_seed: int = 10
    tracker_mask_categories: tuple[str, ...] = ("full_six",)
    tracker_mask_segment_frames: int = 61
    non_head_tracker_dropout_prob: float = 0.0
    tracker_burst_dropout_prob: float = 0.0
    tracker_latency_max_frames: int = 0
    tracker_outlier_prob: float = 0.0
    tracker_pos_noise_std: float = 0.0
    tracker_rot_noise_std: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tracker_mask_categories"] = list(self.tracker_mask_categories)
        return payload


def add_longseq_dropout_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("longseq_dropout")
    group.add_argument("--dropout_preset", default=DROPOUT_PRESET_NONE, choices=DROPOUT_PRESETS, type=str)
    group.add_argument(
        "--tracker_mask_policy",
        default="",
        choices=("", TRACKER_MASK_POLICY_TASK, TRACKER_MASK_POLICY_DYNAMIC_CATEGORIES, TRACKER_MASK_POLICY_FIXED_CATEGORIES),
        type=str,
    )
    group.add_argument("--tracker_mask_seed", default=10, type=int)
    group.add_argument("--tracker_mask_categories", nargs="+", default=["all"], type=str)
    group.add_argument("--tracker_mask_segment_frames", default=61, type=int)
    group.add_argument(
        "--non_head_tracker_dropout_prob",
        "--non_hip_tracker_dropout_prob",
        dest="non_head_tracker_dropout_prob",
        default=0.0,
        type=float,
    )
    group.add_argument("--tracker_burst_dropout_prob", default=0.0, type=float)
    group.add_argument("--tracker_latency_max_frames", default=0, type=int)
    group.add_argument("--tracker_outlier_prob", default=0.0, type=float)
    group.add_argument("--tracker_pos_noise_std", default=0.0, type=float)
    group.add_argument("--tracker_rot_noise_std", default=0.0, type=float)
    group.add_argument("--disable_dropout", default=False, type=str2bool)


def build_longseq_dropout_config(args: argparse.Namespace) -> LongseqDropoutConfig:
    if bool(getattr(args, "disable_dropout", False)):
        return LongseqDropoutConfig()

    preset = str(getattr(args, "dropout_preset", DROPOUT_PRESET_NONE) or DROPOUT_PRESET_NONE)
    if preset not in DROPOUT_PRESETS:
        raise ValueError(f"unknown dropout_preset={preset}, choices={DROPOUT_PRESETS}")

    policy = str(getattr(args, "tracker_mask_policy", "") or "")
    categories = normalize_tracker_pattern_categories(tuple(getattr(args, "tracker_mask_categories", ["all"]) or ["all"]))
    config = LongseqDropoutConfig(
        preset=preset,
        tracker_mask_policy=policy or TRACKER_MASK_POLICY_TASK,
        tracker_mask_seed=int(getattr(args, "tracker_mask_seed", 10)),
        tracker_mask_categories=categories,
        tracker_mask_segment_frames=int(getattr(args, "tracker_mask_segment_frames", 61)),
        non_head_tracker_dropout_prob=float(getattr(args, "non_head_tracker_dropout_prob", 0.0)),
        tracker_burst_dropout_prob=float(getattr(args, "tracker_burst_dropout_prob", 0.0)),
        tracker_latency_max_frames=int(getattr(args, "tracker_latency_max_frames", 0)),
        tracker_outlier_prob=float(getattr(args, "tracker_outlier_prob", 0.0)),
        tracker_pos_noise_std=float(getattr(args, "tracker_pos_noise_std", 0.0)),
        tracker_rot_noise_std=float(getattr(args, "tracker_rot_noise_std", 0.0)),
    )
    if preset == DROPOUT_PRESET_TRACKER_MASK_TRAIN:
        config = replace_config(
            config,
            tracker_mask_policy=policy or TRACKER_MASK_POLICY_DYNAMIC_CATEGORIES,
            tracker_mask_categories=categories,
        )
    elif preset == DROPOUT_PRESET_TRAIN_DEFAULT:
        config = replace_config(
            config,
            tracker_mask_policy=policy or TRACKER_MASK_POLICY_DYNAMIC_CATEGORIES,
            tracker_mask_categories=categories,
            tracker_latency_max_frames=2 if int(getattr(args, "tracker_latency_max_frames", 0)) == 0 else config.tracker_latency_max_frames,
            tracker_burst_dropout_prob=0.05
            if float(getattr(args, "tracker_burst_dropout_prob", 0.0)) == 0.0
            else config.tracker_burst_dropout_prob,
            tracker_outlier_prob=0.01
            if float(getattr(args, "tracker_outlier_prob", 0.0)) == 0.0
            else config.tracker_outlier_prob,
        )
    return config


def replace_config(config: LongseqDropoutConfig, **updates) -> LongseqDropoutConfig:
    values = config.to_dict()
    values.update(updates)
    values["tracker_mask_categories"] = tuple(values["tracker_mask_categories"])
    return LongseqDropoutConfig(**values)


def apply_longseq_dropout_to_source(
    source: dict[str, np.ndarray],
    sequence_id: str,
    config: LongseqDropoutConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if config.preset == DROPOUT_PRESET_NONE and config.tracker_mask_policy == TRACKER_MASK_POLICY_TASK:
        return {key: value.copy() for key, value in source.items()}, build_dropout_metadata(source, config)

    result = {key: value.copy() for key, value in source.items()}
    frame_count = int(result["tracker_pos_world"].shape[0])
    base_valid = np.asarray(result.get("sensor_valid", np.ones((frame_count, TRACKER_COUNT), dtype=bool)), dtype=bool)
    sensor_valid = build_longseq_sensor_valid(
        frame_count=frame_count,
        sequence_id=sequence_id,
        base_sensor_valid=base_valid,
        config=config,
    )
    rng = stable_rng(sequence_id=sequence_id, seed=config.tracker_mask_seed, salt="longseq_augment")

    if config.tracker_latency_max_frames > 0:
        delay = int(rng.integers(0, int(config.tracker_latency_max_frames) + 1))
        if delay > 0:
            for key in ("tracker_pos_world", "tracker_rot_world_6d"):
                delayed = result[key].copy()
                delayed[delay:] = result[key][:-delay]
                delayed[:delay] = result[key][:1]
                result[key] = delayed

    if config.tracker_pos_noise_std > 0:
        noise = rng.normal(0.0, float(config.tracker_pos_noise_std), size=result["tracker_pos_world"].shape).astype(np.float32)
        result["tracker_pos_world"] = result["tracker_pos_world"] + noise * sensor_valid[:, :, None].astype(np.float32)

    if config.tracker_rot_noise_std > 0:
        noise = rng.normal(0.0, float(config.tracker_rot_noise_std), size=result["tracker_rot_world_6d"].shape).astype(np.float32)
        result["tracker_rot_world_6d"] = result["tracker_rot_world_6d"] + noise * sensor_valid[:, :, None].astype(np.float32)

    if config.tracker_outlier_prob > 0:
        outlier_mask = (rng.random(result["tracker_pos_world"].shape[:2]) < float(config.tracker_outlier_prob)) & sensor_valid
        if outlier_mask.any():
            pos_outlier = rng.normal(0.0, 0.15, size=result["tracker_pos_world"].shape).astype(np.float32)
            rot_outlier = rng.normal(0.0, 0.20, size=result["tracker_rot_world_6d"].shape).astype(np.float32)
            result["tracker_pos_world"] = result["tracker_pos_world"] + pos_outlier * outlier_mask[:, :, None].astype(np.float32)
            result["tracker_rot_world_6d"] = result["tracker_rot_world_6d"] + rot_outlier * outlier_mask[:, :, None].astype(np.float32)

    result["sensor_valid"] = sensor_valid
    return result, build_dropout_metadata(result, config)


def build_longseq_sensor_valid(
    frame_count: int,
    sequence_id: str,
    base_sensor_valid: np.ndarray,
    config: LongseqDropoutConfig,
) -> np.ndarray:
    valid = np.asarray(base_sensor_valid, dtype=bool).copy()
    if valid.shape != (int(frame_count), TRACKER_COUNT):
        raise ValueError(f"base_sensor_valid should be [{frame_count},{TRACKER_COUNT}], got {valid.shape}")

    if config.tracker_mask_policy != TRACKER_MASK_POLICY_TASK:
        valid = apply_segmented_tracker_patterns(
            frame_count=frame_count,
            sequence_id=sequence_id,
            base_sensor_valid=valid,
            config=config,
        )

    dropout_prob = max(float(config.non_head_tracker_dropout_prob), float(config.tracker_burst_dropout_prob))
    if dropout_prob > 0:
        rng = stable_rng(sequence_id=sequence_id, seed=config.tracker_mask_seed, salt="frame_dropout")
        valid = dropout_non_head_trackers(sensor_valid=valid, rng=rng, dropout_prob=dropout_prob)

    validate_sensor_valid(valid)
    return valid


def apply_segmented_tracker_patterns(
    frame_count: int,
    sequence_id: str,
    base_sensor_valid: np.ndarray,
    config: LongseqDropoutConfig,
) -> np.ndarray:
    segment_frames = max(1, int(config.tracker_mask_segment_frames))
    categories = tuple(config.tracker_mask_categories)
    valid = np.zeros((int(frame_count), TRACKER_COUNT), dtype=bool)
    segment_count = int(np.ceil(int(frame_count) / float(segment_frames)))
    fixed_category = None
    if config.tracker_mask_policy == TRACKER_MASK_POLICY_FIXED_CATEGORIES:
        digest = stable_digest(sequence_id=sequence_id, seed=config.tracker_mask_seed, salt="fixed_category")
        fixed_category = categories[int(digest[:8], 16) % len(categories)]

    for segment_index in range(segment_count):
        start = segment_index * segment_frames
        end = min(int(frame_count), start + segment_frames)
        if config.tracker_mask_policy == TRACKER_MASK_POLICY_FIXED_CATEGORIES:
            category = str(fixed_category)
        elif config.tracker_mask_policy == TRACKER_MASK_POLICY_DYNAMIC_CATEGORIES:
            category = dynamic_segment_category(
                categories=categories,
                sequence_id=sequence_id,
                seed=config.tracker_mask_seed,
                segment_index=segment_index,
            )
        else:
            raise ValueError(f"unknown tracker_mask_policy={config.tracker_mask_policy}")
        rng = stable_rng(
            sequence_id=sequence_id,
            seed=config.tracker_mask_seed,
            salt=f"segment_pattern:{segment_index}:{category}",
        )
        pattern = make_tracker_pattern(category, rng)
        valid[start:end] = np.asarray(pattern.sensor_valid, dtype=bool)[None, :]

    valid &= base_sensor_valid
    validate_sensor_valid(valid)
    return valid


def dynamic_segment_category(
    categories: tuple[str, ...],
    sequence_id: str,
    seed: int,
    segment_index: int,
) -> str:
    cycle_index = int(segment_index) // len(categories)
    position = int(segment_index) % len(categories)
    values = list(categories)
    rng = stable_rng(sequence_id=sequence_id, seed=seed, salt=f"dynamic_category_cycle:{cycle_index}")
    rng.shuffle(values)
    return str(values[position])


def build_dropout_metadata(source: dict[str, np.ndarray], config: LongseqDropoutConfig) -> dict[str, Any]:
    sensor_valid = np.asarray(source.get("sensor_valid"), dtype=bool)
    return {
        "dropout_config": config.to_dict(),
        "valid_tracker_ratio": float(sensor_valid.mean()) if sensor_valid.size else 1.0,
        "min_valid_trackers": int(sensor_valid.sum(axis=1).min()) if sensor_valid.size else TRACKER_COUNT,
        "max_valid_trackers": int(sensor_valid.sum(axis=1).max()) if sensor_valid.size else TRACKER_COUNT,
        "head_valid_all": bool(sensor_valid[:, HEAD_TRACKER_INDEX].all()) if sensor_valid.size else True,
    }


def stable_rng(sequence_id: str, seed: int, salt: str) -> np.random.Generator:
    digest = stable_digest(sequence_id=sequence_id, seed=seed, salt=salt)
    return np.random.default_rng(int(digest[:16], 16) % (2**32))


def stable_digest(sequence_id: str, seed: int, salt: str) -> str:
    payload = f"{int(seed)}:{sequence_id}:{salt}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()
