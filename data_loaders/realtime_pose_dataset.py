from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from data_loaders.manifest_utils import filter_entries_by_folder_path
from data_loaders.realtime_pose_geometry import assemble_tracker_features_np
from data_loaders.realtime_pose_task_store import ShardReader, read_store_metadata
from data_loaders.sensor_masking import (
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_SEQ_LEN,
    TRACKER_COUNT,
    TRACKER_FEATURE_DIM,
    TRACKER_PATTERN_CATEGORIES,
    validate_realtime_seq_len,
)
from data_loaders.tracker_reliability import compute_hard_rotation_state_np
from data_loaders.tracker_timeline import compute_tracker_durations, stable_context_seed
from utils.normalizer import RealtimePoseNormalizer
from utils.run_dirs import resolve_latest_or_self


COLD_START_HISTORY_BUCKETS = (
    (0, 0),
    (1, 4),
    (5, 14),
    (15, 29),
    (30, REALTIME_POSE_HISTORY_LENGTH - 1),
)


def _normalize_optional_sampling_weights(
    values: list[float] | tuple[float, ...] | None,
    *,
    name: str,
) -> np.ndarray | None:
    """校验并归一化可选的五分类采样权重。"""

    if values is None:
        return None
    weights = np.asarray(values, dtype=np.float64)
    if (
        weights.shape != (5,)
        or not np.isfinite(weights).all()
        or np.any(weights < 0.0)
        or not np.any(weights > 0.0)
    ):
        raise ValueError(f"{name} 必须是五个有限非负数，且至少一项大于零。")
    return weights / weights.sum()


def sample_cold_start_history_length(
    rng: np.random.Generator,
    bucket_weights: np.ndarray,
) -> int:
    """先选历史阶段桶，再在桶内均匀采样整数历史长度。"""

    bucket_index = int(rng.choice(len(COLD_START_HISTORY_BUCKETS), p=bucket_weights))
    lower, upper = COLD_START_HISTORY_BUCKETS[bucket_index]
    return int(rng.integers(lower, upper + 1))


@dataclass(frozen=True)
class TaskRequest:
    task_index: int
    config_index: int
    rollout_steps: int
    history_length: int = REALTIME_POSE_HISTORY_LENGTH


@dataclass(frozen=True)
class _RequestTrackerTimeline:
    """一个请求从基础历史到 rollout 末尾的 Tracker 状态线，形状均为 `[60+K,6]`。"""

    configured: np.ndarray
    measured_valid: np.ndarray
    d_off: np.ndarray
    d_on: np.ndarray
    hard_rotation_state: np.ndarray


class RealtimePoseTaskDataset(Dataset):
    """按请求从 mmap shard 读取一个基础窗口及实际需要的 rollout step。"""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        seq_len: int = REALTIME_POSE_SEQ_LEN,
        normalizer_dir: str | Path | None = None,
        normalize_input: bool = True,
        folder_path: str | Path | None = None,
    ):
        validate_realtime_seq_len(seq_len)
        self.data_dir = resolve_latest_or_self(data_dir, kind="tasks")
        self.split = str(split)
        self.split_dir = self.data_dir / self.split
        self.metadata = read_store_metadata(self.split_dir)
        if int(self.metadata.get("tracker_feature_dim", -1)) != TRACKER_FEATURE_DIM:
            raise ValueError(
                f"task store Tracker 维度不是当前要求的 {TRACKER_FEATURE_DIM}；旧 task 不可复用。"
            )
        if tuple(self.metadata.get("config_names", ())) != TRACKER_PATTERN_CATEGORIES:
            raise ValueError("task store 五类场景名称与当前契约不一致；请重新生成 task。")
        self.plan_hash = str(self.metadata["generation_plan_hash"])
        self.max_rollout_steps = int(self.metadata["max_rollout_steps"])
        self.shards = list(self.metadata["shards"])
        self.normalizer = create_normalizer(normalizer_dir, bool(normalize_input))
        if self.normalizer is not None:
            normalizer_hash = str(self.normalizer.metadata.get("generation_plan_hash", ""))
            if normalizer_hash != self.plan_hash:
                raise ValueError(
                    "task 与 normalizer 的 generation-plan hash 不一致："
                    f"{self.plan_hash} != {normalizer_hash}"
                )

        self.sources = read_jsonl(self.split_dir / "sources.jsonl")
        self.sources.sort(key=lambda value: int(value["source_index"]))
        self.joint_offsets_parent = np.load(
            self.split_dir / "source_joint_offsets_parent.npy", mmap_mode="r", allow_pickle=False
        )
        self.joint_rest_local_rotations_6d = np.load(
            self.split_dir / "source_joint_rest_local_rotations_6d.npy", mmap_mode="r", allow_pickle=False
        )
        allowed_sources = set(range(len(self.sources)))
        if folder_path:
            filtered = filter_entries_by_folder_path(self.sources, folder_path)
            allowed_sources = {int(source["source_index"]) for source in filtered}

        self.locations: list[tuple[int, int]] = []
        self.indices_by_shard: list[list[int]] = [[] for _ in self.shards]
        for shard_index, shard in enumerate(self.shards):
            source_indices = np.load(
                self.split_dir / shard["path"] / "source_index.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            for row_index, source_index in enumerate(source_indices.tolist()):
                if int(source_index) in allowed_sources:
                    dataset_index = len(self.locations)
                    self.locations.append((shard_index, row_index))
                    self.indices_by_shard[shard_index].append(dataset_index)
        if not self.locations:
            raise RuntimeError(f"{self.split_dir} 中没有可用 task。")
        self.reader = ShardReader(self.split_dir, self.shards, max_open_shards=2)

    def __len__(self) -> int:
        return len(self.locations)

    def close(self) -> None:
        self.reader.close()
        for array in (self.joint_offsets_parent, self.joint_rest_local_rotations_6d):
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __del__(self) -> None:
        reader = getattr(self, "reader", None)
        if reader is not None:
            reader.close()
        for name in ("joint_offsets_parent", "joint_rest_local_rotations_6d"):
            array = getattr(self, name, None)
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __getitem__(self, request: int | TaskRequest) -> dict[str, Any]:
        if isinstance(request, TaskRequest):
            task_index = int(request.task_index)
            config_index = int(request.config_index)
            rollout_steps = int(request.rollout_steps)
            initial_history_length = int(request.history_length)
        else:
            task_index = int(request)
            config_index = 0
            rollout_steps = 1
            initial_history_length = REALTIME_POSE_HISTORY_LENGTH
        if not 0 <= config_index < 5:
            raise IndexError(f"config_index 必须在 [0,4]，实际为 {config_index}")
        if not 1 <= rollout_steps <= self.max_rollout_steps:
            raise ValueError(
                f"rollout_steps 必须在 [1,{self.max_rollout_steps}]，实际为 {rollout_steps}"
            )
        if not 0 <= initial_history_length <= REALTIME_POSE_HISTORY_LENGTH:
            raise ValueError(
                f"history_length 必须在 [0,{REALTIME_POSE_HISTORY_LENGTH}]，"
                f"实际为 {initial_history_length}"
            )

        shard_index, row_index = self.locations[task_index]
        shard = self.reader.get(shard_index)
        tracker_timeline = self._build_request_tracker_timeline(
            shard=shard,
            row_index=row_index,
            config_index=config_index,
            rollout_steps=rollout_steps,
            initial_history_length=initial_history_length,
        )
        base = self._step_to_item(
            shard,
            row_index,
            config_index,
            step=0,
            include_history=True,
            initial_history_length=initial_history_length,
            tracker_timeline=tracker_timeline,
        )
        if rollout_steps > 1:
            base["rollout"] = [
                self._step_to_item(
                    shard,
                    row_index,
                    config_index,
                    step=step,
                    include_history=False,
                    initial_history_length=initial_history_length,
                    tracker_timeline=tracker_timeline,
                )
                for step in range(1, rollout_steps)
            ]
        return base

    def _build_request_tracker_timeline(
        self,
        shard: dict[str, np.ndarray],
        row_index: int,
        config_index: int,
        rollout_steps: int,
        initial_history_length: int,
    ) -> _RequestTrackerTimeline:
        """按虚拟会话起点一次性重放状态，避免 rollout 每一步独立重置 duration。"""

        state_length = REALTIME_POSE_HISTORY_LENGTH + int(rollout_steps)
        original = {
            "configured": np.asarray(
                shard["configured"][row_index, config_index, :state_length], dtype=bool
            ).copy(),
            "measured_valid": np.asarray(
                shard["measured_valid"][row_index, config_index, :state_length], dtype=bool
            ).copy(),
            "d_off": np.asarray(
                shard["d_off"][row_index, config_index, :state_length], dtype=np.uint8
            ).copy(),
            "d_on": np.asarray(
                shard["d_on"][row_index, config_index, :state_length], dtype=np.uint8
            ).copy(),
            "hard_rotation_state": np.asarray(
                shard["hard_rotation_state"][row_index, config_index, :state_length], dtype=bool
            ).copy(),
        }
        _validate_tracker_state_features(**original)
        if initial_history_length == REALTIME_POSE_HISTORY_LENGTH:
            return _RequestTrackerTimeline(**original)

        session_start = REALTIME_POSE_HISTORY_LENGTH - int(initial_history_length)
        configured_visible = original["configured"][session_start:]
        measured_visible = original["measured_valid"][session_start:]
        d_off_visible, d_on_visible = compute_tracker_durations(
            configured_visible,
            measured_visible,
        )
        hard_visible = compute_hard_rotation_state_np(
            configured_visible,
            measured_visible,
            d_on_visible,
        )

        # 补零区域表示会话尚未开始，不能保留离线 source 在这些帧积累的可靠度状态。
        configured = np.zeros_like(original["configured"], dtype=bool)
        measured_valid = np.zeros_like(original["measured_valid"], dtype=bool)
        d_off = np.zeros_like(original["d_off"], dtype=np.uint8)
        d_on = np.zeros_like(original["d_on"], dtype=np.uint8)
        hard_rotation_state = np.zeros_like(original["hard_rotation_state"], dtype=bool)
        configured[session_start:] = configured_visible
        measured_valid[session_start:] = measured_visible
        d_off[session_start:] = d_off_visible
        d_on[session_start:] = d_on_visible
        hard_rotation_state[session_start:] = hard_visible
        return _RequestTrackerTimeline(
            configured=configured,
            measured_valid=measured_valid,
            d_off=d_off,
            d_on=d_on,
            hard_rotation_state=hard_rotation_state,
        )

    def task_id_at(self, task_index: int) -> str:
        shard_index, row_index = self.locations[int(task_index)]
        shard = self.reader.get(shard_index)
        source = self.sources[int(shard["source_index"][row_index])]
        start_frame = int(shard["start_frame"][row_index])
        from data_loaders.generate_realtime_pose_tasks import make_task_id

        return make_task_id(self.split, str(source["source_id"]), start_frame)

    def _step_to_item(
        self,
        shard: dict[str, np.ndarray],
        row_index: int,
        config_index: int,
        step: int,
        include_history: bool,
        initial_history_length: int,
        tracker_timeline: _RequestTrackerTimeline,
    ) -> dict[str, Any]:
        source_index = int(shard["source_index"][row_index])
        source = self.sources[source_index]
        state_slice = slice(step, step + REALTIME_POSE_SEQ_LEN)
        configured = tracker_timeline.configured[state_slice].copy()
        measured_valid = tracker_timeline.measured_valid[state_slice].copy()
        d_off = tracker_timeline.d_off[state_slice].copy()
        d_on = tracker_timeline.d_on[state_slice].copy()
        hard_rotation = tracker_timeline.hard_rotation_state[state_slice].copy()
        history_length = min(
            REALTIME_POSE_HISTORY_LENGTH,
            int(initial_history_length) + int(step),
        )
        invalid_history_length = REALTIME_POSE_HISTORY_LENGTH - history_length

        tracker_history_raw = np.zeros(
            (REALTIME_POSE_HISTORY_LENGTH, TRACKER_COUNT, TRACKER_FEATURE_DIM),
            dtype=np.float32,
        )
        if history_length:
            tracker_history_raw[-history_length:] = assemble_tracker_features_np(
                np.asarray(
                    shard["tracker_history_continuous"][row_index, step, -history_length:],
                    dtype=np.float32,
                ).copy(),
                configured[invalid_history_length:-1],
                measured_valid[invalid_history_length:-1],
                d_off[invalid_history_length:-1],
                d_on[invalid_history_length:-1],
            )
        current_tracker_raw = assemble_tracker_features_np(
            np.asarray(shard["current_tracker_continuous"][row_index, step], dtype=np.float32)[None].copy(),
            configured[-1:], measured_valid[-1:], d_off[-1:], d_on[-1:],
        )[0]

        current_target_raw = np.asarray(shard["current_target"][row_index, step], dtype=np.float32).copy()
        trajectory_history = np.asarray(shard["trajectory_history"][row_index, step], dtype=np.float32).copy()
        current_trajectory = np.asarray(shard["current_trajectory"][row_index, step], dtype=np.float32).copy()
        session_start_in_history = (
            REALTIME_POSE_HISTORY_LENGTH - int(initial_history_length) - int(step)
        )
        if 0 <= session_start_in_history < REALTIME_POSE_HISTORY_LENGTH:
            trajectory_history[session_start_in_history, :2] = 0.0
            trajectory_history[session_start_in_history, 3:] = (0.0, 1.0)
        elif initial_history_length == 0 and step == 0:
            current_trajectory[0, :2] = 0.0
            current_trajectory[0, 3:] = (0.0, 1.0)
        if self.normalizer is None:
            current_target = current_target_raw
            tracker_history = tracker_history_raw
            current_tracker = current_tracker_raw
        else:
            current_target = self.normalizer.normalize_pose(current_target_raw)
            tracker_history = self.normalizer.normalize_tracker(tracker_history_raw)
            current_tracker = self.normalizer.normalize_tracker(current_tracker_raw)
            trajectory_history[:, 2] = self.normalizer.normalize_head_height(trajectory_history[:, 2])
            current_trajectory[:, 2] = self.normalizer.normalize_head_height(current_trajectory[:, 2])
        # padding 是模型输入空间中的字面量零；不能把零高度再送入 normalizer。
        trajectory_history[:invalid_history_length] = 0.0
        start_frame = int(shard["start_frame"][row_index]) + int(step)
        task_id = self.task_id_from_values(source, int(shard["start_frame"][row_index]))
        valid_frame_mask = np.zeros(REALTIME_POSE_HISTORY_LENGTH, dtype=bool)
        if history_length:
            valid_frame_mask[-history_length:] = True
        item: dict[str, Any] = {
            "x": torch.from_numpy(current_target).float(),
            "current_target": torch.from_numpy(current_target).float(),
            "tracker_history": torch.from_numpy(tracker_history).float(),
            "current_tracker": torch.from_numpy(current_tracker).float(),
            "current_tracker_raw": torch.from_numpy(current_tracker_raw).float(),
            "trajectory_history": torch.from_numpy(trajectory_history).float(),
            "current_trajectory": torch.from_numpy(current_trajectory).float(),
            "valid_frame_mask": torch.from_numpy(valid_frame_mask).bool(),
            "history_length": torch.tensor(history_length, dtype=torch.long),
            "configured": torch.from_numpy(configured).bool(),
            "measured_valid": torch.from_numpy(measured_valid).bool(),
            "d_off": torch.from_numpy(d_off.astype(np.int64)).long(),
            "d_on": torch.from_numpy(d_on.astype(np.int64)).long(),
            "hard_rotation_state": torch.from_numpy(hard_rotation[-1]).bool(),
            "current_tracker_pos_head_ref": torch.from_numpy(current_tracker_raw[:, :3]).float(),
            "current_tracker_rot_head_ref_6d": torch.from_numpy(current_tracker_raw[:, 3:9]).float(),
            "target_joints_head_ref": torch.from_numpy(
                np.asarray(shard["target_joints_head_ref"][row_index, step], dtype=np.float32).copy()
            ).float(),
            "prev_joints_head_ref": torch.from_numpy(
                np.asarray(shard["prev_joints_head_ref"][row_index, step], dtype=np.float32).copy()
            ).float(),
            "target_root_position_head_ref": torch.from_numpy(
                np.asarray(shard["target_root_position_head_ref"][row_index, step], dtype=np.float32).copy()
            ).float(),
            "target_root_yaw_world": torch.tensor(
                float(shard["target_root_yaw_world"][row_index, step]), dtype=torch.float32
            ),
            "target_hip_height": torch.tensor(
                float(shard["target_hip_height"][row_index, step]), dtype=torch.float32
            ),
            "history_head_yaw_world": torch.tensor(
                float(shard["history_head_yaw_world"][row_index, step]), dtype=torch.float32
            ),
            "current_head_yaw_world": torch.tensor(
                float(shard["current_head_yaw_world"][row_index, step]), dtype=torch.float32
            ),
            "current_head_position_world": torch.from_numpy(
                np.asarray(shard["current_head_position_world"][row_index, step], dtype=np.float32).copy()
            ).float(),
            "floor_y": torch.tensor(float(shard["floor_y"][row_index, step]), dtype=torch.float32),
            "future_leg_target": torch.from_numpy(
                np.asarray(shard["future_leg_target"][row_index, step], dtype=np.float32).copy()
            ).float(),
            "contact_target": torch.from_numpy(
                np.asarray(shard["contact_target"][row_index, step], dtype=np.float32).copy()
            ).float(),
            "joint_offsets_parent": torch.from_numpy(
                np.asarray(self.joint_offsets_parent[source_index], dtype=np.float32).copy()
            ).float(),
            "joint_rest_local_rotations_6d": torch.from_numpy(
                np.asarray(self.joint_rest_local_rotations_6d[source_index], dtype=np.float32).copy()
            ).float(),
            "scenario_id": torch.tensor(config_index, dtype=torch.long),
            "scenario": str(self.metadata["config_names"][config_index]),
            "start_frame": torch.tensor(start_frame, dtype=torch.long),
            "task_id": task_id,
            "source_path": str(source["source_path"]),
        }
        if include_history:
            pose_history_raw = np.asarray(shard["pose_history"][row_index], dtype=np.float32).copy()
            pose_history = (
                pose_history_raw
                if self.normalizer is None
                else self.normalizer.normalize_pose(pose_history_raw)
            )
            pose_history[:invalid_history_length] = 0.0
            item["pose_history"] = torch.from_numpy(pose_history).float()
        return item

    def task_id_from_values(self, source: dict[str, Any], start_frame: int) -> str:
        from data_loaders.generate_realtime_pose_tasks import make_task_id

        return make_task_id(self.split, str(source["source_id"]), start_frame)


class RealtimePoseBatchSampler(Sampler[list[TaskRequest]]):
    """先打乱 shard/窗口，再为每个 batch 决定统一 rollout 长度。"""

    def __init__(
        self,
        dataset: RealtimePoseTaskDataset,
        batch_size: int,
        seed: int,
        scenario_weights: list[float] | tuple[float, ...],
        rollout_steps: int,
        rollout_prob: float,
        cold_start_prob: float,
        shuffle: bool,
        drop_last: bool,
        cold_start_history_weights: list[float] | tuple[float, ...] | None = None,
        cold_start_scenario_weights: list[float] | tuple[float, ...] | None = None,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rollout_steps = int(rollout_steps)
        self.rollout_prob = float(rollout_prob)
        self.cold_start_prob = float(cold_start_prob)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.scenario_weights = _normalize_optional_sampling_weights(
            scenario_weights,
            name="scenario_weights",
        )
        assert self.scenario_weights is not None
        self.cold_start_history_weights = _normalize_optional_sampling_weights(
            cold_start_history_weights,
            name="cold_start_history_weights",
        )
        self.cold_start_scenario_weights = _normalize_optional_sampling_weights(
            cold_start_scenario_weights,
            name="cold_start_scenario_weights",
        )
        if not 1 <= self.rollout_steps <= dataset.max_rollout_steps:
            raise ValueError("rollout_steps 超出 task store 可用范围。")
        if not 0.0 <= self.rollout_prob <= 1.0:
            raise ValueError("rollout_prob 必须在 [0,1]。")
        if not 0.0 <= self.cold_start_prob <= 1.0:
            raise ValueError("cold_start_prob 必须在 [0,1]。")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[TaskRequest]]:
        shard_order = [index for index, values in enumerate(self.dataset.indices_by_shard) if values]
        rng = np.random.Generator(np.random.PCG64(stable_context_seed(self.seed, self.epoch, "shards")))
        if self.shuffle:
            rng.shuffle(shard_order)
        ordered_indices: list[int] = []
        for shard_index in shard_order:
            values = np.asarray(self.dataset.indices_by_shard[shard_index], dtype=np.int64)
            if self.shuffle:
                local_rng = np.random.Generator(
                    np.random.PCG64(stable_context_seed(self.seed, self.epoch, shard_index, "rows"))
                )
                local_rng.shuffle(values)
            ordered_indices.extend(int(value) for value in values.tolist())

        weight_token = ",".join(f"{value:.17g}" for value in self.scenario_weights.tolist())
        cold_scenario_token = (
            ""
            if self.cold_start_scenario_weights is None
            else "|cold="
            + ",".join(
                f"{value:.17g}" for value in self.cold_start_scenario_weights.tolist()
            )
        )
        history_weight_token = (
            ""
            if self.cold_start_history_weights is None
            else ",".join(
                f"{value:.17g}" for value in self.cold_start_history_weights.tolist()
            )
        )
        for batch_index, first in enumerate(range(0, len(ordered_indices), self.batch_size)):
            indices = ordered_indices[first : first + self.batch_size]
            if len(indices) < self.batch_size and self.drop_last:
                break
            rollout_rng = np.random.Generator(
                np.random.PCG64(stable_context_seed(self.seed, self.epoch, batch_index, "rollout"))
            )
            use_rollout = self.rollout_steps > 1 and float(rollout_rng.random()) < self.rollout_prob
            batch_rollout_steps = self.rollout_steps if use_rollout else 1
            requests: list[TaskRequest] = []
            for task_index in indices:
                task_id = self.dataset.task_id_at(task_index)
                history_seed_parts: tuple[object, ...] = (
                    (task_id, self.epoch, self.seed, "cold_start")
                    if self.cold_start_history_weights is None
                    else (
                        task_id,
                        self.epoch,
                        self.seed,
                        history_weight_token,
                        "cold_start_stratified",
                    )
                )
                history_rng = np.random.Generator(
                    np.random.PCG64(stable_context_seed(*history_seed_parts))
                )
                use_cold_start = float(history_rng.random()) < self.cold_start_prob
                if not use_cold_start:
                    history_length = REALTIME_POSE_HISTORY_LENGTH
                elif self.cold_start_history_weights is None:
                    # 未启用分层配置时严格保留旧版 0～59 均匀采样行为。
                    history_length = int(
                        history_rng.integers(0, REALTIME_POSE_HISTORY_LENGTH)
                    )
                else:
                    history_length = sample_cold_start_history_length(
                        history_rng,
                        self.cold_start_history_weights,
                    )

                active_scenario_weights = (
                    self.cold_start_scenario_weights
                    if use_cold_start and self.cold_start_scenario_weights is not None
                    else self.scenario_weights
                )
                active_scenario_token = (
                    weight_token + cold_scenario_token
                    if use_cold_start and self.cold_start_scenario_weights is not None
                    else weight_token
                )
                config_rng = np.random.Generator(
                    np.random.PCG64(
                        stable_context_seed(
                            task_id,
                            self.epoch,
                            self.seed,
                            active_scenario_token,
                            "scenario",
                        )
                    )
                )
                config_index = int(config_rng.choice(5, p=active_scenario_weights))
                requests.append(
                    TaskRequest(
                        task_index=task_index,
                        config_index=config_index,
                        rollout_steps=batch_rollout_steps,
                        history_length=history_length,
                    )
                )
            yield requests

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def _validate_tracker_state_features(
    configured: np.ndarray,
    measured_valid: np.ndarray,
    d_off: np.ndarray,
    d_on: np.ndarray,
    hard_rotation_state: np.ndarray,
) -> None:
    if np.any(measured_valid & ~configured):
        raise ValueError("measured_valid 必须是 configured 的子集。")
    if not configured[:, 0].all() or not measured_valid[:, 0].all():
        raise ValueError("Head 必须始终 configured 且 measured_valid。")
    if any(value.shape != configured.shape for value in (d_off, d_on, hard_rotation_state)):
        raise ValueError("Tracker state arrays 必须同为 [T,6]。")
    if np.any((d_off < 0) | (d_off > 60)) or np.any((d_on < 0) | (d_on > 60)):
        raise ValueError("d_off/d_on 必须位于 [0,60]。")
    if not np.all(d_off[~configured | measured_valid] == 0):
        raise ValueError("未配置或有效 Tracker 的 d_off 必须为零。")
    if not np.all(d_on[~configured | ~measured_valid] == 0):
        raise ValueError("未配置或掉线 Tracker 的 d_on 必须为零。")
    if not hard_rotation_state[:, 0].all():
        raise ValueError("Head rotation 必须始终 hard。")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                values.append(json.loads(line))
    return values


def create_normalizer(
    normalizer_dir: str | Path | None,
    normalize_input: bool,
) -> RealtimePoseNormalizer | None:
    if not normalize_input:
        return None
    if normalizer_dir is None:
        raise ValueError("normalize_input=True 时必须提供 normalizer_dir。")
    return RealtimePoseNormalizer(normalizer_dir)
