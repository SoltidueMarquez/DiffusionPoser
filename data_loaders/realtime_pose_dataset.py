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
    REALTIME_POSE_FRAME_OFFSETS,
    REALTIME_POSE_HISTORY_ANCHOR_INDICES,
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_WINDOW_LENGTH,
    TRACKER_COUNT,
    TRACKER_FEATURE_DIM,
    TRACKER_PATTERN_CATEGORIES,
    validate_realtime_seq_len,
)
from data_loaders.tracker_reliability import (
    compute_hard_rotation_state_np,
    compute_region_coverage_np,
    compute_tracker_reliability_np,
)
from data_loaders.tracker_timeline import compute_tracker_durations, stable_context_seed
from utils.normalizer import RealtimePoseNormalizer
from utils.run_dirs import resolve_latest_or_self


@dataclass(frozen=True)
class TaskRequest:
    task_index: int
    config_index: int
    history_length: int = REALTIME_POSE_HISTORY_LENGTH


class RealtimePoseTaskDataset(Dataset):
    """从 mmap task store 读取同步的 10 帧历史和 1 帧当前目标。"""

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
                f"task store 的 Tracker 特征维度必须为 {TRACKER_FEATURE_DIM}，请重建 task。"
            )
        if tuple(self.metadata.get("config_names", ())) != TRACKER_PATTERN_CATEGORIES:
            raise ValueError("task store 的 Tracker 场景配置与当前代码不兼容，请重建 task。")
        self.plan_hash = str(self.metadata["generation_plan_hash"])
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
            self.split_dir / "source_joint_offsets_parent.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        self.joint_rest_local_rotations_6d = np.load(
            self.split_dir / "source_joint_rest_local_rotations_6d.npy",
            mmap_mode="r",
            allow_pickle=False,
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
            initial_history_length = int(request.history_length)
        else:
            task_index = int(request)
            config_index = 0
            initial_history_length = REALTIME_POSE_HISTORY_LENGTH
        if not 0 <= config_index < len(TRACKER_PATTERN_CATEGORIES):
            raise IndexError(f"config_index 必须位于 [0,4]，得到 {config_index}。")
        if not 0 <= initial_history_length <= REALTIME_POSE_HISTORY_LENGTH:
            raise ValueError(
                f"history_length 必须位于 [0,{REALTIME_POSE_HISTORY_LENGTH}]，"
                f"得到 {initial_history_length}。"
            )
        shard_index, row_index = self.locations[task_index]
        return self._window_item(
            shard=self.reader.get(shard_index),
            row_index=row_index,
            config_index=config_index,
            initial_history_length=initial_history_length,
        )

    def _window_item(
        self,
        shard: dict[str, np.ndarray],
        row_index: int,
        config_index: int,
        initial_history_length: int,
    ) -> dict[str, Any]:
        """先重放 61 帧密集 Tracker 状态，再同步抽取 10+1 个锚点。"""

        configured_source = np.asarray(
            shard["configured"][row_index, config_index], dtype=bool
        ).copy()
        measured_source = np.asarray(
            shard["measured_valid"][row_index, config_index], dtype=bool
        ).copy()
        expected_dense_shape = (REALTIME_POSE_SEQ_LEN, TRACKER_COUNT)
        if configured_source.shape != expected_dense_shape:
            raise ValueError(f"configured 必须是 {expected_dense_shape}，得到 {configured_source.shape}。")

        # 冷启动发生在虚拟会话起点，持续时间和 hard state 必须从该点重新累计。
        session_start = REALTIME_POSE_HISTORY_LENGTH - int(initial_history_length)
        configured_dense = np.zeros_like(configured_source)
        measured_dense = np.zeros_like(measured_source)
        d_off_dense = np.zeros_like(configured_source, dtype=np.uint8)
        d_on_dense = np.zeros_like(configured_source, dtype=np.uint8)
        hard_dense = np.zeros_like(configured_source)
        configured_visible = configured_source[session_start:]
        measured_visible = measured_source[session_start:]
        d_off_visible, d_on_visible = compute_tracker_durations(
            configured_visible, measured_visible
        )
        hard_visible = compute_hard_rotation_state_np(
            configured_visible, measured_visible, d_on_visible
        )
        configured_dense[session_start:] = configured_visible
        measured_dense[session_start:] = measured_visible
        d_off_dense[session_start:] = d_off_visible
        d_on_dense[session_start:] = d_on_visible
        hard_dense[session_start:] = hard_visible

        dense_indices = np.asarray(
            (*REALTIME_POSE_HISTORY_ANCHOR_INDICES, REALTIME_POSE_HISTORY_LENGTH),
            dtype=np.int64,
        )
        window_valid = dense_indices >= session_start
        configured = configured_dense[dense_indices]
        measured_valid = measured_dense[dense_indices]
        d_off = d_off_dense[dense_indices]
        d_on = d_on_dense[dense_indices]
        hard_rotation = hard_dense[dense_indices]

        tracker_continuous = np.asarray(
            shard["tracker_window_continuous"][row_index], dtype=np.float32
        ).copy()
        tracker_raw = np.zeros(
            (REALTIME_POSE_WINDOW_LENGTH, TRACKER_COUNT, TRACKER_FEATURE_DIM),
            dtype=np.float32,
        )
        tracker_raw[window_valid] = assemble_tracker_features_np(
            tracker_continuous[window_valid],
            configured[window_valid],
            measured_valid[window_valid],
            d_off[window_valid],
            d_on[window_valid],
        )
        pose_window_raw = np.asarray(
            shard["pose_window_clean"][row_index], dtype=np.float32
        ).copy()
        head_path_raw = np.asarray(
            shard["head_path_window"][row_index], dtype=np.float32
        ).copy()
        if self.normalizer is None:
            pose_window = pose_window_raw
            tracker_window = tracker_raw
            head_path_window = head_path_raw
        else:
            pose_window = self.normalizer.normalize_pose(pose_window_raw)
            tracker_window = self.normalizer.normalize_tracker(tracker_raw)
            head_path_window = self.normalizer.normalize_head_path(head_path_raw)

        # 归一化后再清零，确保 padding 在模型输入中仍是字面零。
        pose_window[~window_valid] = 0.0
        tracker_window[~window_valid] = 0.0
        head_path_window[~window_valid] = 0.0
        kappa_pos, kappa_rot = compute_tracker_reliability_np(
            configured[:-1], measured_valid[:-1], d_on[:-1]
        )
        rho_pos, rho_rot = compute_region_coverage_np(kappa_pos, kappa_rot)
        history_confidence = 0.5 * (rho_pos + rho_rot)
        history_confidence *= window_valid[:-1, None]

        source_index = int(shard["source_index"][row_index])
        source = self.sources[source_index]
        start_frame = int(shard["start_frame"][row_index])
        result: dict[str, Any] = {
            "x": torch.from_numpy(pose_window).float(),
            "history_pose_observation": torch.from_numpy(pose_window[:-1].copy()).float(),
            "tracker_window": torch.from_numpy(tracker_window).float(),
            "head_path_window": torch.from_numpy(head_path_window).float(),
            "history_region_confidence": torch.from_numpy(history_confidence).float(),
            "window_valid_mask": torch.from_numpy(window_valid).bool(),
            "frame_offsets": torch.tensor(REALTIME_POSE_FRAME_OFFSETS, dtype=torch.long),
            "history_length": torch.tensor(initial_history_length, dtype=torch.long),
            "configured": torch.from_numpy(configured).bool(),
            "measured_valid": torch.from_numpy(measured_valid).bool(),
            "d_off": torch.from_numpy(d_off.astype(np.int64)).long(),
            "d_on": torch.from_numpy(d_on.astype(np.int64)).long(),
            "tracker_window_raw": torch.from_numpy(tracker_raw).float(),
            "hard_rotation_state_window": torch.from_numpy(hard_rotation).bool(),
            "target_joints_head_ref": torch.from_numpy(
                np.asarray(shard["target_joints_head_ref"][row_index], dtype=np.float32).copy()
            ).float(),
            "target_root_position_head_ref": torch.from_numpy(
                np.asarray(
                    shard["target_root_position_head_ref"][row_index], dtype=np.float32
                ).copy()
            ).float(),
            "target_root_yaw_world": torch.tensor(
                float(shard["target_root_yaw_world"][row_index]), dtype=torch.float32
            ),
            "target_hip_height": torch.tensor(
                float(shard["target_hip_height"][row_index]), dtype=torch.float32
            ),
            "current_head_yaw_world": torch.tensor(
                float(shard["current_head_yaw_world"][row_index]), dtype=torch.float32
            ),
            "current_head_position_world": torch.from_numpy(
                np.asarray(
                    shard["current_head_position_world"][row_index], dtype=np.float32
                ).copy()
            ).float(),
            "floor_y": torch.tensor(float(shard["floor_y"][row_index]), dtype=torch.float32),
            "future_leg_target": torch.from_numpy(
                np.asarray(shard["future_leg_target"][row_index], dtype=np.float32).copy()
            ).float(),
            "contact_target": torch.from_numpy(
                np.asarray(shard["contact_target"][row_index], dtype=np.float32).copy()
            ).float(),
            "joint_offsets_parent": torch.from_numpy(
                np.asarray(self.joint_offsets_parent[source_index], dtype=np.float32).copy()
            ).float(),
            "joint_rest_local_rotations_6d": torch.from_numpy(
                np.asarray(
                    self.joint_rest_local_rotations_6d[source_index], dtype=np.float32
                ).copy()
            ).float(),
            "scenario_id": torch.tensor(config_index, dtype=torch.long),
            "scenario": str(self.metadata["config_names"][config_index]),
            "start_frame": torch.tensor(start_frame, dtype=torch.long),
            "task_id": self.task_id_from_values(source, start_frame),
            "source_path": str(source["source_path"]),
        }
        return result

    def task_id_at(self, task_index: int) -> str:
        shard_index, row_index = self.locations[int(task_index)]
        shard = self.reader.get(shard_index)
        source = self.sources[int(shard["source_index"][row_index])]
        start_frame = int(shard["start_frame"][row_index])
        return self.task_id_from_values(source, start_frame)

    def task_id_from_values(self, source: dict[str, Any], start_frame: int) -> str:
        from data_loaders.generate_realtime_pose_tasks import make_task_id

        return make_task_id(self.split, str(source["source_id"]), start_frame)


class RealtimePoseBatchSampler(Sampler[list[TaskRequest]]):
    """按 shard 打乱任务，并确定每个样本的场景与冷启动历史长度。"""

    def __init__(
        self,
        dataset: RealtimePoseTaskDataset,
        batch_size: int,
        seed: int,
        scenario_weights: list[float] | tuple[float, ...],
        cold_start_prob: float,
        shuffle: bool,
        drop_last: bool,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.cold_start_prob = float(cold_start_prob)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        weights = np.asarray(scenario_weights, dtype=np.float64)
        if weights.shape != (5,) or np.any(weights < 0.0) or not np.any(weights > 0.0):
            raise ValueError("scenario_weights 必须是五个非负数，且至少一项大于零。")
        self.scenario_weights = weights / weights.sum()
        if not 0.0 <= self.cold_start_prob <= 1.0:
            raise ValueError("cold_start_prob 必须位于 [0,1]。")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[TaskRequest]]:
        shard_order = [
            index for index, values in enumerate(self.dataset.indices_by_shard) if values
        ]
        rng = np.random.Generator(
            np.random.PCG64(stable_context_seed(self.seed, self.epoch, "shards"))
        )
        if self.shuffle:
            rng.shuffle(shard_order)
        ordered_indices: list[int] = []
        for shard_index in shard_order:
            values = np.asarray(self.dataset.indices_by_shard[shard_index], dtype=np.int64)
            if self.shuffle:
                local_rng = np.random.Generator(
                    np.random.PCG64(
                        stable_context_seed(self.seed, self.epoch, shard_index, "rows")
                    )
                )
                local_rng.shuffle(values)
            ordered_indices.extend(int(value) for value in values.tolist())

        weight_token = ",".join(
            f"{value:.17g}" for value in self.scenario_weights.tolist()
        )
        for first in range(0, len(ordered_indices), self.batch_size):
            indices = ordered_indices[first : first + self.batch_size]
            if len(indices) < self.batch_size and self.drop_last:
                break
            requests: list[TaskRequest] = []
            for task_index in indices:
                task_id = self.dataset.task_id_at(task_index)
                config_rng = np.random.Generator(
                    np.random.PCG64(
                        stable_context_seed(
                            task_id, self.epoch, self.seed, weight_token, "scenario"
                        )
                    )
                )
                config_index = int(config_rng.choice(5, p=self.scenario_weights))
                history_rng = np.random.Generator(
                    np.random.PCG64(
                        stable_context_seed(
                            task_id, self.epoch, self.seed, "cold_start"
                        )
                    )
                )
                use_cold_start = float(history_rng.random()) < self.cold_start_prob
                history_length = (
                    int(history_rng.integers(0, REALTIME_POSE_HISTORY_LENGTH))
                    if use_cold_start
                    else REALTIME_POSE_HISTORY_LENGTH
                )
                requests.append(
                    TaskRequest(
                        task_index=task_index,
                        config_index=config_index,
                        history_length=history_length,
                    )
                )
            yield requests

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


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
