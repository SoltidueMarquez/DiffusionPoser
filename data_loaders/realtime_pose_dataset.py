from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from data_loaders.manifest_utils import filter_entries_by_folder_path
from data_loaders.realtime_pose_geometry import build_known_mask_from_measured_np
from data_loaders.realtime_pose_task_store import ShardReader, read_store_metadata
from data_loaders.sensor_masking import (
    MISSING_AGE_CAP,
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_SEQ_LEN,
    TRACKER_CONFIGURED_OFFSET,
    TRACKER_COUNT,
    TRACKER_FEATURE_DIM,
    TRACKER_MEASURED_VALID_OFFSET,
    TRACKER_MISSING_AGE_OFFSET,
    validate_realtime_seq_len,
)
from data_loaders.tracker_timeline import stable_context_seed
from utils.normalizer import RealtimePoseNormalizer
from utils.run_dirs import resolve_latest_or_self


@dataclass(frozen=True)
class TaskRequest:
    task_index: int
    config_index: int
    rollout_steps: int


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
        else:
            task_index = int(request)
            config_index = 0
            rollout_steps = 1
        if not 0 <= config_index < 5:
            raise IndexError(f"config_index 必须在 [0,4]，实际为 {config_index}")
        if not 1 <= rollout_steps <= self.max_rollout_steps:
            raise ValueError(
                f"rollout_steps 必须在 [1,{self.max_rollout_steps}]，实际为 {rollout_steps}"
            )

        shard_index, row_index = self.locations[task_index]
        shard = self.reader.get(shard_index)
        base = self._step_to_item(shard, row_index, config_index, step=0, include_history=True)
        if rollout_steps > 1:
            base["rollout"] = [
                self._step_to_item(shard, row_index, config_index, step=step, include_history=False)
                for step in range(1, rollout_steps)
            ]
        return base

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
    ) -> dict[str, Any]:
        source_index = int(shard["source_index"][row_index])
        source = self.sources[source_index]
        state_slice = slice(step, step + REALTIME_POSE_SEQ_LEN)
        configured = np.asarray(shard["configured"][row_index, config_index, state_slice], dtype=bool)
        measured_valid = np.asarray(shard["measured_valid"][row_index, config_index, state_slice], dtype=bool)
        missing_age = np.asarray(shard["missing_age"][row_index, config_index, state_slice], dtype=np.uint8)
        missing_age_norm = missing_age.astype(np.float32) / float(MISSING_AGE_CAP)
        _validate_tracker_state_features(configured, measured_valid, missing_age_norm)

        tracker_continuous = np.asarray(
            shard["tracker_continuous"][row_index, step], dtype=np.float32
        ).copy()
        tracker_continuous *= measured_valid[..., None]
        tracker_raw = np.empty(
            (REALTIME_POSE_SEQ_LEN, TRACKER_COUNT, TRACKER_FEATURE_DIM), dtype=np.float32
        )
        tracker_raw[..., :9] = tracker_continuous
        tracker_raw[..., TRACKER_CONFIGURED_OFFSET] = configured
        tracker_raw[..., TRACKER_MEASURED_VALID_OFFSET] = measured_valid
        tracker_raw[..., TRACKER_MISSING_AGE_OFFSET] = missing_age_norm

        current_target_raw = np.asarray(shard["current_target"][row_index, step], dtype=np.float32).copy()
        full_known_raw = np.asarray(shard["full_known_target"][row_index, step], dtype=np.float32).copy()
        known_mask = build_known_mask_from_measured_np(measured_valid[-1])
        if self.normalizer is None:
            current_target = current_target_raw
            known_target = full_known_raw
            tracker_window = tracker_raw
        else:
            current_target = self.normalizer.normalize_pose(current_target_raw)
            known_target = self.normalizer.normalize_pose(full_known_raw)
            tracker_window = self.normalizer.normalize_tracker(tracker_raw)
        known_target = np.where(known_mask, known_target, np.zeros_like(known_target))
        current_tracker = tracker_raw[-1]
        start_frame = int(shard["start_frame"][row_index]) + int(step)
        task_id = self.task_id_from_values(source, int(shard["start_frame"][row_index]))
        valid_frame_mask = np.ones(REALTIME_POSE_HISTORY_LENGTH, dtype=bool)
        item: dict[str, Any] = {
            "x": torch.from_numpy(current_target).float(),
            "current_target": torch.from_numpy(current_target).float(),
            "tracker_window": torch.from_numpy(tracker_window).float(),
            "known_target": torch.from_numpy(known_target).float(),
            "known_mask": torch.from_numpy(known_mask).bool(),
            "unknown_mask": torch.from_numpy(~known_mask).bool(),
            "inpaint_mask": torch.from_numpy(~known_mask).bool(),
            "conditioned_x": torch.from_numpy(known_target).float(),
            "valid_frame_mask": torch.from_numpy(valid_frame_mask).bool(),
            "attention_mask": torch.from_numpy(valid_frame_mask).bool(),
            "configured": torch.from_numpy(configured).bool(),
            "measured_valid": torch.from_numpy(measured_valid).bool(),
            "missing_age": torch.from_numpy(missing_age.astype(np.int64)).long(),
            "missing_age_norm": torch.from_numpy(missing_age_norm).float(),
            "current_tracker_pos_head_ref": torch.from_numpy(current_tracker[:, :3]).float(),
            "current_tracker_rot_head_ref_6d": torch.from_numpy(current_tracker[:, 3:9]).float(),
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
            "current_head_yaw_world": torch.tensor(
                float(shard["current_head_yaw_world"][row_index, step]), dtype=torch.float32
            ),
            "current_head_position_world": torch.from_numpy(
                np.asarray(shard["current_head_position_world"][row_index, step], dtype=np.float32).copy()
            ).float(),
            "floor_y": torch.tensor(float(shard["floor_y"][row_index, step]), dtype=torch.float32),
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
        shuffle: bool,
        drop_last: bool,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rollout_steps = int(rollout_steps)
        self.rollout_prob = float(rollout_prob)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        weights = np.asarray(scenario_weights, dtype=np.float64)
        if weights.shape != (5,) or np.any(weights < 0.0) or not np.any(weights > 0.0):
            raise ValueError("scenario_weights 必须是五个非负数，且至少一项大于零。")
        self.scenario_weights = weights / weights.sum()
        if not 1 <= self.rollout_steps <= dataset.max_rollout_steps:
            raise ValueError("rollout_steps 超出 task store 可用范围。")
        if not 0.0 <= self.rollout_prob <= 1.0:
            raise ValueError("rollout_prob 必须在 [0,1]。")

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
                config_rng = np.random.Generator(
                    np.random.PCG64(
                        stable_context_seed(task_id, self.epoch, self.seed, weight_token, "scenario")
                    )
                )
                config_index = int(config_rng.choice(5, p=self.scenario_weights))
                requests.append(TaskRequest(task_index, config_index, batch_rollout_steps))
            yield requests

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def _validate_tracker_state_features(
    configured: np.ndarray,
    measured_valid: np.ndarray,
    missing_age_norm: np.ndarray,
) -> None:
    if np.any(measured_valid & ~configured):
        raise ValueError("measured_valid 必须是 configured 的子集。")
    if not configured[:, 0].all() or not measured_valid[:, 0].all():
        raise ValueError("Head 必须始终 configured 且 measured_valid。")
    if np.any(missing_age_norm < 0.0) or np.any(missing_age_norm > 1.0):
        raise ValueError("missing_age_norm 必须在 [0,1]。")
    should_zero = ~configured | measured_valid
    if not np.allclose(missing_age_norm[should_zero], 0.0):
        raise ValueError("未配置或已重连 Tracker 的 missing_age_norm 必须为零。")


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
