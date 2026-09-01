from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from data_loaders.realtime_pose_task_store import ShardReader, discover_shards
from data_loaders.realtime_pose_predictor_features import (
    build_predictor_sparse_availability_mask_np,
)
from data_loaders.rpm_hand_dropout import (
    RPM_HAND_DROPOUT_TRAIN_SEED,
    build_rpm_dit_training_availability,
    rpm_hand_dropout_sample_key,
    stable_rpm_hand_dropout_seed,
)
from data_loaders.sensor_masking import (
    CORE_TRACKER_INDICES,
    HAND_TRACKER_INDICES,
    HEAD_TRACKER_INDEX,
    REALTIME_POSE_SEQ_LEN,
    TRACKER_AVAILABLE_OFFSET,
    TRACKER_CONTINUOUS_DIM,
    TRACKER_COUNT,
    TRACKER_FEATURE_DIM,
    TRAIN_TRACKER_ENDPOINTS,
    validate_realtime_seq_len,
    validate_tracker_available,
)
from utils.normalizer import RealtimePoseNormalizer


TASK_SHARD_FIELDS = (
    "motion_context_clean",
    "core_tracker_context_clean",
    "current_pose_target_clean",
    "current_tracker_continuous",
    "previous_pose_target_clean",
    "target_joints_head_ref",
    "target_root_position_head_ref",
    "target_root_yaw_world",
    "target_hip_height",
    "current_head_yaw_world",
    "current_head_position_world",
    "floor_y",
    "joint_offsets_parent",
    "joint_rest_local_rotations_6d",
    "task_seed",
    "current_frame",
)


@dataclass(frozen=True)
class TaskRequest:
    task_index: int
    config_index: int


class RealtimePoseTaskDataset(Dataset):
    """读取 Predictor 条件与单帧 DiT 监督；Tracker 场景不写入 Task Store。"""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        seq_len: int = REALTIME_POSE_SEQ_LEN,
        normalizer_dir: str | Path | None = None,
        normalize_input: bool = True,
        rpm_hand_dropout: bool = False,
        rpm_hand_dropout_seed: int = RPM_HAND_DROPOUT_TRAIN_SEED,
    ):
        validate_realtime_seq_len(seq_len)
        self.data_dir = Path(data_dir).resolve()
        self.split = str(split)
        self.split_dir = self.data_dir / self.split
        self.shards = discover_shards(self.split_dir, TASK_SHARD_FIELDS)
        self.normalizer = create_normalizer(normalizer_dir, bool(normalize_input))
        self.rpm_hand_dropout = bool(rpm_hand_dropout)
        self.rpm_hand_dropout_seed = int(rpm_hand_dropout_seed)
        self.locations: list[tuple[int, int]] = []
        self.indices_by_shard: list[list[int]] = [[] for _ in self.shards]
        for shard_index, shard in enumerate(self.shards):
            for row_index in range(int(shard["row_count"])):
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

    def __del__(self) -> None:
        reader = getattr(self, "reader", None)
        if reader is not None:
            reader.close()

    def __getitem__(self, request: int | TaskRequest) -> dict[str, Any]:
        if isinstance(request, TaskRequest):
            task_index = int(request.task_index)
            config_index = int(request.config_index)
        else:
            task_index = int(request)
            config_index = 0
        if not 0 <= config_index < len(TRAIN_TRACKER_ENDPOINTS):
            raise IndexError(
                f"config_index 必须位于 [0,{len(TRAIN_TRACKER_ENDPOINTS) - 1}]。"
            )
        shard_index, row_index = self.locations[task_index]
        return self._item(
            shard=self.reader.get(shard_index),
            row_index=row_index,
            config_index=config_index,
        )

    def _item(
        self,
        shard: dict[str, np.ndarray],
        row_index: int,
        config_index: int,
    ) -> dict[str, Any]:
        motion_raw = np.asarray(
            shard["motion_context_clean"][row_index], dtype=np.float32
        ).copy()
        sparse_raw = np.asarray(
            shard["core_tracker_context_clean"][row_index], dtype=np.float32
        ).copy()
        target_raw = np.asarray(
            shard["current_pose_target_clean"][row_index], dtype=np.float32
        ).copy()
        previous_raw = np.asarray(
            shard["previous_pose_target_clean"][row_index], dtype=np.float32
        ).copy()
        task_seed = int(shard["task_seed"][row_index])
        hand_available_with_previous = np.ones(
            (12, TRACKER_COUNT), dtype=bool
        )
        if self.rpm_hand_dropout:
            dropout_seed = stable_rpm_hand_dropout_seed(
                self.rpm_hand_dropout_seed,
                self.split,
                rpm_hand_dropout_sample_key(task_seed),
            )
            hand_available_with_previous = build_rpm_dit_training_availability(
                seed=dropout_seed
            )
        sparse_available = build_predictor_sparse_availability_mask_np(
            hand_available_with_previous
        )
        if self.normalizer is None:
            motion = motion_raw
            sparse = sparse_raw
            target = target_raw
            previous = previous_raw
        else:
            motion = self.normalizer.normalize_pose(motion_raw)
            sparse = self.normalizer.normalize_predictor_sparse(sparse_raw)
            target = self.normalizer.normalize_pose(target_raw)
            previous = self.normalizer.normalize_pose(previous_raw)
        if self.rpm_hand_dropout:
            # Null token 与 runtime 保持一致，定义在归一化域零点，而不是把
            # 原始零值送入旧 normalizer 后产生异常大的伪观测。
            sparse = np.where(sparse_available, sparse, 0.0).astype(np.float32)

        tracker_available = np.asarray(
            TRAIN_TRACKER_ENDPOINTS[config_index], dtype=bool
        ).copy()
        if self.rpm_hand_dropout:
            tracker_available[list(HAND_TRACKER_INDICES)] = (
                hand_available_with_previous[-1, list(HAND_TRACKER_INDICES)]
            )
        validate_tracker_available(
            tracker_available,
            required_tracker_indices=(
                (HEAD_TRACKER_INDEX,)
                if self.rpm_hand_dropout
                else tuple(CORE_TRACKER_INDICES)
            ),
        )
        tracker_continuous = np.asarray(
            shard["current_tracker_continuous"][row_index], dtype=np.float32
        ).copy()
        tracker_raw = np.zeros(
            (TRACKER_COUNT, TRACKER_FEATURE_DIM), dtype=np.float32
        )
        tracker_raw[:, :TRACKER_CONTINUOUS_DIM] = np.where(
            tracker_available[:, None], tracker_continuous, 0.0
        )
        tracker_raw[:, TRACKER_AVAILABLE_OFFSET] = tracker_available

        result: dict[str, Any] = {
            "x": torch.from_numpy(np.asarray(target, dtype=np.float32)).float(),
            "motion_context": torch.from_numpy(
                np.asarray(motion, dtype=np.float32)
            ).float(),
            "core_tracker_context": torch.from_numpy(
                np.asarray(sparse, dtype=np.float32)
            ).float(),
            "current_tracker_raw": torch.from_numpy(tracker_raw).float(),
            "tracker_available": torch.from_numpy(tracker_available).bool(),
            "previous_pose_target": torch.from_numpy(
                np.asarray(previous, dtype=np.float32)
            ).float(),
            "target_joints_head_ref": _tensor_field(
                shard, "target_joints_head_ref", row_index
            ),
            "target_root_position_head_ref": _tensor_field(
                shard, "target_root_position_head_ref", row_index
            ),
            "target_root_yaw_world": torch.tensor(
                float(shard["target_root_yaw_world"][row_index]),
                dtype=torch.float32,
            ),
            "target_hip_height": torch.tensor(
                float(shard["target_hip_height"][row_index]),
                dtype=torch.float32,
            ),
            "current_head_yaw_world": torch.tensor(
                float(shard["current_head_yaw_world"][row_index]),
                dtype=torch.float32,
            ),
            "current_head_position_world": _tensor_field(
                shard, "current_head_position_world", row_index
            ),
            "floor_y": torch.tensor(
                float(shard["floor_y"][row_index]), dtype=torch.float32
            ),
            "joint_offsets_parent": _tensor_field(
                shard, "joint_offsets_parent", row_index
            ),
            "joint_rest_local_rotations_6d": _tensor_field(
                shard, "joint_rest_local_rotations_6d", row_index
            ),
            "current_frame": torch.tensor(
                int(shard["current_frame"][row_index]), dtype=torch.long
            ),
            "task_id": self.task_id_from_seed(task_seed),
        }
        return result

    def task_id_at(self, task_index: int) -> str:
        shard_index, row_index = self.locations[int(task_index)]
        shard = self.reader.get(shard_index)
        return self.task_id_from_seed(int(shard["task_seed"][row_index]))

    @staticmethod
    def task_id_from_seed(task_seed: int) -> str:
        return rpm_hand_dropout_sample_key(task_seed)


class RealtimePoseBatchSampler(Sampler[list[TaskRequest]]):
    """按 shard 打乱任务，并等概率轮换全部 8 种 Tracker 配置。"""

    def __init__(
        self,
        dataset: RealtimePoseTaskDataset,
        batch_size: int,
        seed: int,
        shuffle: bool,
        drop_last: bool,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[TaskRequest]]:
        shard_order = [
            index
            for index, values in enumerate(self.dataset.indices_by_shard)
            if values
        ]
        rng = np.random.Generator(
            np.random.PCG64(_stable_seed(self.seed, self.epoch, "shards"))
        )
        if self.shuffle:
            rng.shuffle(shard_order)
        ordered_indices: list[int] = []
        for shard_index in shard_order:
            values = np.asarray(
                self.dataset.indices_by_shard[shard_index], dtype=np.int64
            )
            if self.shuffle:
                local_rng = np.random.Generator(
                    np.random.PCG64(
                        _stable_seed(self.seed, self.epoch, shard_index, "rows")
                    )
                )
                local_rng.shuffle(values)
            ordered_indices.extend(int(value) for value in values.tolist())

        for first in range(0, len(ordered_indices), self.batch_size):
            indices = ordered_indices[first : first + self.batch_size]
            if len(indices) < self.batch_size and self.drop_last:
                break
            requests: list[TaskRequest] = []
            for local_index, task_index in enumerate(indices):
                # 用全局样本序号与 epoch 共同轮换配置；无需额外随机状态即可在
                # 长期训练中得到确定、均匀且可复现的 8 种 availability 分布。
                config_index = int(
                    (first + local_index + self.epoch) % len(TRAIN_TRACKER_ENDPOINTS)
                )
                requests.append(TaskRequest(task_index, config_index))
            yield requests

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def create_normalizer(
    normalizer_dir: str | Path | None,
    normalize_input: bool,
) -> RealtimePoseNormalizer | None:
    if not normalize_input:
        return None
    if normalizer_dir is None:
        raise ValueError("normalize_input=True 时必须提供 normalizer_dir。")
    return RealtimePoseNormalizer(normalizer_dir)


def _tensor_field(
    shard: dict[str, np.ndarray], name: str, row_index: int
) -> torch.Tensor:
    return torch.from_numpy(
        np.asarray(shard[name][row_index], dtype=np.float32).copy()
    ).float()


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:8], byteorder="little", signed=False
    )
