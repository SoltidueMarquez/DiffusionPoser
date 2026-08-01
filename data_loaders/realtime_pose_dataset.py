from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from data_loaders.manifest_utils import filter_entries_by_folder_path
from data_loaders.realtime_pose_validation import (
    validate_realtime_task_arrays,
    validate_realtime_task_manifest_entry,
)
from data_loaders.sensor_masking import (
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_SEQ_LEN,
    TRACKER_CONFIGURED_OFFSET,
    TRACKER_MEASURED_VALID_OFFSET,
    TRACKER_MISSING_AGE_OFFSET,
    validate_realtime_seq_len,
)
from utils.normalizer import RealtimePoseNormalizer
from utils.run_dirs import resolve_latest_or_self


_SCENARIO_TO_ID = {
    "fixed_six": 0,
    "fixed_three": 1,
    "three_to_six": 2,
    "six_to_three": 3,
    "dropout": 4,
}


class RealtimePoseTaskDataset(Dataset):
    """读取预生成的 140 维姿态任务；Tracker 时间线不会在这里重新随机。"""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        seq_len: int = REALTIME_POSE_SEQ_LEN,
        normalizer_dir: str | Path | None = None,
        normalize_input: bool = True,
        preload_data: bool = False,
        folder_path: str | Path | None = None,
        enable_rollout: bool = False,
        rollout_steps: int = 1,
    ):
        validate_realtime_seq_len(seq_len)
        self.data_dir = resolve_latest_or_self(data_dir, kind="tasks")
        self.split = str(split)
        self.seq_len = int(seq_len)
        self.normalize_input = bool(normalize_input)
        self.preload_data = bool(preload_data)
        self.rollout_steps = int(rollout_steps)
        self.enable_rollout = bool(enable_rollout) and self.rollout_steps > 1
        if self.rollout_steps < 1:
            raise ValueError("rollout_steps 必须至少为 1。")

        self.normalizer = create_normalizer(
            normalizer_dir=normalizer_dir,
            normalize_input=self.normalize_input,
        )
        self.manifest_path = find_manifest_path(self.data_dir, self.split)
        self.manifest_dir = self.manifest_path.parent
        self.entries = read_task_manifest(self.manifest_path)
        if folder_path:
            self.entries = filter_entries_by_folder_path(self.entries, folder_path)
        if not self.entries:
            raise RuntimeError(f"{self.manifest_path} 中没有可用 task。")
        for entry in self.entries:
            if self.enable_rollout:
                paths = entry.get("rollout_task_paths", [])
                if not isinstance(paths, list) or len(paths) < self.rollout_steps - 1:
                    raise ValueError(
                        f"task {entry.get('task_id')} 不足以提供 rollout_steps={self.rollout_steps}。"
                    )

        self.task_cache: list[dict[str, np.ndarray]] | None = None
        if self.preload_data:
            self.task_cache = [self._load_entry(entry) for entry in self.entries]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        task = self.task_cache[index] if self.task_cache is not None else self._load_entry(entry)
        item = self._task_to_item(task, entry)
        if self.enable_rollout:
            item["rollout"] = [
                self._task_to_item(
                    load_materialized_task_npz(
                        self.manifest_dir,
                        path,
                    ),
                    entry,
                )
                for path in entry["rollout_task_paths"][: self.rollout_steps - 1]
            ]
        return item

    def set_epoch(self, epoch: int) -> None:
        # 时间线已由 source ID 和绝对帧固定；保留入口供训练循环统一调用。
        del epoch

    def _load_entry(self, entry: dict[str, Any]) -> dict[str, np.ndarray]:
        return load_materialized_task_npz(
            self.manifest_dir,
            entry["task_path"],
        )

    def _task_to_item(self, task: dict[str, np.ndarray], entry: dict[str, Any]) -> dict[str, Any]:
        arrays = load_realtime_task_arrays(task, seq_len=self.seq_len)
        for key in ("source_frames", "seq_len", "max_rollout_steps"):
            task_value = int(np.asarray(arrays[key]).item())
            manifest_value = int(entry[key])
            if task_value != manifest_value:
                raise ValueError(
                    f"task {entry.get('task_id', '<unknown>')} 的 {key}={task_value}，"
                    f"与 manifest 的 {manifest_value} 不一致"
                )
        pose_history_raw = arrays["pose_history"].astype(np.float32, copy=True)
        current_target_raw = arrays["current_target"].astype(np.float32, copy=True)
        known_target_raw = arrays["known_target"].astype(np.float32, copy=True)
        known_mask = arrays["known_mask"].astype(bool, copy=True)
        tracker_raw = arrays["tracker_window"].astype(np.float32, copy=True)

        if self.normalizer is None:
            pose_history = pose_history_raw
            current_target = current_target_raw
            known_target = known_target_raw
            tracker_window = tracker_raw
        else:
            pose_history = self.normalizer.normalize_pose(pose_history_raw)
            current_target = self.normalizer.normalize_pose(current_target_raw)
            known_target = self.normalizer.normalize_pose(known_target_raw)
            tracker_window = self.normalizer.normalize_tracker(tracker_raw)
        # unknown 位置不携带任何伪条件；known_mask 才是唯一写回依据。
        known_target = np.where(known_mask, known_target, np.zeros_like(known_target))

        configured = tracker_raw[..., TRACKER_CONFIGURED_OFFSET] > 0.5
        measured_valid = tracker_raw[..., TRACKER_MEASURED_VALID_OFFSET] > 0.5
        missing_age_norm = tracker_raw[..., TRACKER_MISSING_AGE_OFFSET]
        _validate_tracker_state_features(configured, measured_valid, missing_age_norm)
        current_tracker = tracker_raw[-1]
        scenario = str(np.asarray(arrays["scenario"]).reshape(()).item())

        item: dict[str, Any] = {
            "x": torch.from_numpy(current_target).float(),
            "current_target": torch.from_numpy(current_target).float(),
            "pose_history": torch.from_numpy(pose_history).float(),
            "tracker_window": torch.from_numpy(tracker_window).float(),
            "known_target": torch.from_numpy(known_target).float(),
            "known_mask": torch.from_numpy(known_mask).bool(),
            "unknown_mask": torch.from_numpy(~known_mask).bool(),
            # 旧训练循环名仅作为同义入口；形状和语义均已是 [140] 的 unknown mask。
            "inpaint_mask": torch.from_numpy(~known_mask).bool(),
            "conditioned_x": torch.from_numpy(known_target).float(),
            "valid_frame_mask": torch.from_numpy(arrays["valid_frame_mask"].astype(bool)).bool(),
            "attention_mask": torch.from_numpy(arrays["valid_frame_mask"].astype(bool)).bool(),
            "configured": torch.from_numpy(configured).bool(),
            "measured_valid": torch.from_numpy(measured_valid).bool(),
            "missing_age": torch.from_numpy(arrays["missing_age"].astype(np.int64)).long(),
            "missing_age_norm": torch.from_numpy(missing_age_norm.astype(np.float32)).float(),
            "current_tracker_pos_head_ref": torch.from_numpy(current_tracker[:, :3]).float(),
            "current_tracker_rot_head_ref_6d": torch.from_numpy(current_tracker[:, 3:9]).float(),
            "target_joints_head_ref": torch.from_numpy(arrays["target_joints_head_ref"].astype(np.float32)).float(),
            "prev_joints_head_ref": torch.from_numpy(arrays["prev_joints_head_ref"].astype(np.float32)).float(),
            "target_root_position_head_ref": torch.from_numpy(
                arrays["target_root_position_head_ref"].astype(np.float32)
            ).float(),
            "target_root_yaw_world": torch.as_tensor(arrays["target_root_yaw_world"], dtype=torch.float32),
            "target_hip_height": torch.as_tensor(arrays["target_hip_height"], dtype=torch.float32),
            "current_head_yaw_world": torch.as_tensor(arrays["current_head_yaw_world"], dtype=torch.float32),
            "current_head_position_world": torch.from_numpy(
                arrays["current_head_position_world"].astype(np.float32)
            ).float(),
            "floor_y": torch.as_tensor(arrays["floor_y"], dtype=torch.float32),
            "joint_offsets_parent": torch.from_numpy(arrays["joint_offsets_parent"].astype(np.float32)).float(),
            "joint_rest_local_rotations_6d": torch.from_numpy(
                arrays["joint_rest_local_rotations_6d"].astype(np.float32)
            ).float(),
            "scenario_id": torch.tensor(_SCENARIO_TO_ID.get(scenario, -1), dtype=torch.long),
            "scenario": scenario,
            "start_frame": torch.as_tensor(arrays["start_frame"], dtype=torch.long),
            "task_id": str(entry.get("task_id", "")),
            "source_path": str(np.asarray(arrays["source_path"]).reshape(()).item()),
        }
        return item


def _validate_tracker_state_features(
    configured: np.ndarray,
    measured_valid: np.ndarray,
    missing_age_norm: np.ndarray,
) -> None:
    if np.any(measured_valid & ~configured):
        raise ValueError("measured_valid 必须是 configured 的子集。")
    if not configured[:, 0].all() or not measured_valid[:, 0].all():
        raise ValueError("Head 必须始终 configured 且 measured_valid。")
    if np.any(missing_age_norm < -1e-7) or np.any(missing_age_norm > 1.0 + 1e-7):
        raise ValueError("missing_age_norm 必须在 [0,1]。")
    should_zero = ~configured | measured_valid
    if not np.allclose(missing_age_norm[should_zero], 0.0, atol=1e-7):
        raise ValueError("未配置或已重连 Tracker 的 missing_age_norm 必须为零。")


def find_manifest_path(data_dir: str | Path, split: str) -> Path:
    root = resolve_latest_or_self(data_dir, kind="tasks")
    candidates = (root / str(split) / "manifest.jsonl", root / f"manifest_{split}.jsonl", root / "manifest.jsonl")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"找不到 split={split} 的 task manifest：{root}")


def read_task_manifest(path: str | Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 必须是 JSON object。")
            validate_realtime_task_manifest_entry(value, label=f"{path}:{line_number}")
            entries.append(value)
    return entries


def load_materialized_task_npz(
    manifest_dir: str | Path,
    task_path: str | Path,
) -> dict[str, np.ndarray]:
    path = Path(task_path)
    if not path.is_absolute():
        path = Path(manifest_dir) / path
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        task = {key: np.asarray(data[key]).copy() for key in data.files}
    return task


def load_realtime_task_arrays(
    task: dict[str, np.ndarray],
    seq_len: int = REALTIME_POSE_SEQ_LEN,
) -> dict[str, np.ndarray]:
    validate_realtime_seq_len(seq_len)
    validate_realtime_task_arrays(task, seq_len=seq_len)
    return task


def create_normalizer(
    normalizer_dir: str | Path | None,
    normalize_input: bool,
) -> RealtimePoseNormalizer | None:
    if not normalize_input:
        return None
    if normalizer_dir is None:
        raise ValueError("normalize_input=True 时必须提供 normalizer_dir。")
    return RealtimePoseNormalizer(normalizer_dir)
