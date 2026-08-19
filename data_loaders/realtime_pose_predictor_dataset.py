from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from data_loaders.generate_realtime_pose_tasks import (
    compute_source_joint_rotations_world,
    filter_entries_by_split,
    make_task_id,
    read_source_entries,
    read_split_keys,
    select_window_starts,
)
from data_loaders.realtime_pose_kinematics import rotation_6d_forward_up_np
from data_loaders.sensor_masking import (
    BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY,
    PREDICTOR_TRAINING_FIRST_OFFSET,
    PREDICTOR_TRAINING_LAST_OFFSET,
    PREDICTOR_TRAINING_SEQUENCE_LENGTH,
)


_PREDICTOR_FK_SOURCE_FIELDS = (
    BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY,
    "root_pos_world",
    "root_yaw",
    "pelvis_height",
    "tracker_pos_world",
    "tracker_rot_world_6d",
    "joint_offsets_parent",
    "joint_rest_local_rotations_6d",
)


@dataclass(frozen=True)
class PredictorResidentSequence:
    """一条 source 的 Predictor 常驻字段；不保留可重建的 3x3 旋转矩阵。"""

    joint_rotations_world_6d: np.ndarray  # [T,24,6]
    tracker_positions_world: np.ndarray  # [T,6,3]
    tracker_rotations_world_6d: np.ndarray  # [T,6,6]
    floor_y: np.ndarray  # [T]
    joint_offsets_parent: np.ndarray  # [24,3]

    @property
    def nbytes(self) -> int:
        return sum(value.nbytes for value in self.__dict__.values())


class RealtimePosePredictorSequenceDataset(Dataset):
    """启动时预载 source，训练时只从内存切出 `-11～+40` 的 Predictor 序列。"""

    def __init__(
        self,
        source_dir: str | Path,
        split_dir: str | Path | None,
        split: str,
        windows_per_source: int,
        seed: int,
        limit: int = 0,
    ):
        self.source_dir = Path(source_dir).resolve()
        self.split = str(split)
        self.seed = int(seed)
        entries = read_source_entries(self.source_dir)
        split_keys = (
            read_split_keys(Path(split_dir).resolve(), self.split)
            if split_dir
            else None
        )
        entries = filter_entries_by_split(entries, split_keys)
        if int(limit) > 0:
            entries = entries[: int(limit)]

        self.entries: list[dict[str, Any]] = []
        self.sequences: list[PredictorResidentSequence] = []
        self.windows: list[tuple[int, int]] = []
        preload_started = perf_counter()
        for entry in tqdm(
            entries,
            desc=f"Predictor preload [{self.split}]",
            unit="source",
        ):
            path = Path(entry["source_path"])
            source, frame_count = _load_predictor_fk_source(path)
            current_frames = select_window_starts(
                frame_count=frame_count,
                count=int(windows_per_source),
                max_rollout_steps=PREDICTOR_TRAINING_LAST_OFFSET,
                global_seed=self.seed,
                split=self.split,
                source_id=str(entry["stablemotion_split_key"]),
            )
            if not current_frames:
                continue

            sequence = _build_resident_sequence(source)
            entry_index = len(self.entries)
            self.entries.append({**entry, "frame_count": frame_count})
            self.sequences.append(sequence)
            self.windows.extend(
                (entry_index, int(current_frame))
                for current_frame in current_frames
            )
        if not self.windows:
            raise RuntimeError(
                f"split={self.split} 没有满足 Predictor -11～+40 窗口的 source。"
            )
        self.resident_bytes = sum(sequence.nbytes for sequence in self.sequences)
        self.preload_elapsed_seconds = perf_counter() - preload_started
        print(
            f"Predictor dataset [{self.split}] 已预载 {len(self.sequences)} 条 source、"
            f"{len(self.windows)} 个窗口，常驻 {self.resident_bytes / 2**30:.2f} GiB，"
            f"耗时 {self.preload_elapsed_seconds:.1f}s。",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry_index, current_frame = self.windows[int(index)]
        sequence = self.sequences[entry_index]
        first = current_frame + PREDICTOR_TRAINING_FIRST_OFFSET
        last = current_frame + PREDICTOR_TRAINING_LAST_OFFSET + 1
        if last - first != PREDICTOR_TRAINING_SEQUENCE_LENGTH:
            raise RuntimeError("Predictor raw sequence 长度计算错误。")
        entry = self.entries[entry_index]
        return {
            "joint_rotations_world_6d": torch.from_numpy(
                sequence.joint_rotations_world_6d[first:last].copy()
            ),
            "tracker_positions_world": torch.from_numpy(
                sequence.tracker_positions_world[first:last].copy()
            ),
            "tracker_rotations_world_6d": torch.from_numpy(
                sequence.tracker_rotations_world_6d[first:last].copy()
            ),
            "floor_y": torch.from_numpy(
                sequence.floor_y[first:last].copy()
            ),
            "joint_offsets_parent": torch.from_numpy(
                sequence.joint_offsets_parent.copy()
            ),
            "current_frame": torch.tensor(current_frame, dtype=torch.long),
            "task_id": make_task_id(
                self.split,
                str(entry["stablemotion_split_key"]),
                current_frame,
            ),
        }


def _load_predictor_fk_source(path: Path) -> tuple[dict[str, np.ndarray], int]:
    """每个所需字段只解压一次，不读取 Predictor 训练未使用的 joints 等字段。"""

    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in _PREDICTOR_FK_SOURCE_FIELDS if key not in payload.files]
        if missing:
            raise KeyError(f"{path} 缺少 Predictor source 字段: {missing}")
        source = {
            key: np.ascontiguousarray(
                np.asarray(payload[key], dtype=np.float32)
            )
            for key in _PREDICTOR_FK_SOURCE_FIELDS
        }

    body_pose = source[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY]
    if body_pose.ndim != 2 or body_pose.shape[1] != 144:
        raise ValueError(
            f"{path} {BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY} 必须为 [T,144]，"
            f"实际为 {body_pose.shape}"
        )
    frame_count = int(body_pose.shape[0])
    if frame_count <= 0:
        raise ValueError(f"{path} Predictor source 帧数必须大于 0。")
    expected_shapes = {
        "root_pos_world": (frame_count, 3),
        "root_yaw": (frame_count,),
        "pelvis_height": (frame_count, 1),
        "tracker_pos_world": (frame_count, 6, 3),
        "tracker_rot_world_6d": (frame_count, 6, 6),
        "joint_offsets_parent": (24, 3),
        "joint_rest_local_rotations_6d": (24, 6),
    }
    for key, expected_shape in expected_shapes.items():
        value = source[key]
        if value.shape != expected_shape:
            raise ValueError(
                f"{path} {key} 必须为 {expected_shape}，实际为 {value.shape}"
            )
    if any(not np.isfinite(value).all() for value in source.values()):
        raise ValueError(f"{path} Predictor source 字段包含 NaN 或 Inf。")
    return source, frame_count


def _build_resident_sequence(
    source: dict[str, np.ndarray],
) -> PredictorResidentSequence:
    """整段 FK 只执行一次，随后丢弃原始 source 与 3x3 世界旋转。"""

    joint_rotations_world = compute_source_joint_rotations_world(source)
    sequence = PredictorResidentSequence(
        joint_rotations_world_6d=np.ascontiguousarray(
            rotation_6d_forward_up_np(joint_rotations_world), dtype=np.float32
        ),
        tracker_positions_world=np.ascontiguousarray(
            source["tracker_pos_world"], dtype=np.float32
        ),
        tracker_rotations_world_6d=np.ascontiguousarray(
            source["tracker_rot_world_6d"], dtype=np.float32
        ),
        floor_y=np.ascontiguousarray(source["root_pos_world"][:, 1], dtype=np.float32),
        joint_offsets_parent=np.ascontiguousarray(
            source["joint_offsets_parent"], dtype=np.float32
        ),
    )
    # DataLoader worker 只读共享这些数组，避免无意写入触发 fork copy-on-write。
    for value in sequence.__dict__.values():
        value.setflags(write=False)
    return sequence


def get_predictor_dataset_loader(
    *,
    source_dir: str | Path,
    split_dir: str | Path | None,
    split: str,
    windows_per_source: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    limit: int = 0,
) -> DataLoader:
    dataset = RealtimePosePredictorSequenceDataset(
        source_dir=source_dir,
        split_dir=split_dir,
        split=split,
        windows_per_source=windows_per_source,
        seed=seed,
        limit=limit,
    )
    worker_kwargs: dict[str, Any] = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory) or int(num_workers) > 0,
    }
    if int(num_workers) > 0:
        worker_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=bool(shuffle),
        **worker_kwargs,
    )
