from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data_loaders.manifest_utils import filter_entries_by_folder_path
from data_loaders.generate_realtime_pose_tasks import (
    TASK_OUTPUT_MARKER,
    add_runtime_task_arrays,
    clip_source,
    load_realtime_source,
)
from data_loaders.realtime_pose_contract import (
    RESOLVER_CONTEXT_FRAMES,
    RUNTIME_CONTRACT_METADATA_FIELDS,
    validate_schema_metadata,
    validate_stationary_label_metadata,
)
from data_loaders.realtime_pose_kinematics import (
    make_yaw_rotation_np,
)
from data_loaders.tracker_codec import build_tracker_reference_np, encode_tracker_positions_np, encode_tracker_rotations_np
from data_loaders.sensor_masking import (
    BODY_POSE_DIM,
    POSE_REPRESENTATION_KEY,
    POSE_REPRESENTATION_BODY_FBX_LOCAL_DELTA_6D,
    HEAD_TRACKER_INDEX,
    HIP_TRACKER_INDEX,
    REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_LENGTH,
    REALTIME_POSE_TARGET_START,
    ROOT_DELTA_XZ_DIM,
    ROOT_HEIGHT_DIM,
    ROOT_YAW_DELTA_DIM,
    SENSOR_VALID_DIM,
    SchemaSpec,
    STATIONARY_PROB_DIM,
    TRACKER_COUNT,
    TRACKER_PATTERN_CATEGORIES,
    TRACKER_MASK_FILL_MODES,
    TRACKER_MASK_FILL_ZERO,
    TRACKER_MASK_POLICIES,
    TRACKER_MASK_POLICY_AUTO,
    TRACKER_MASK_POLICY_DYNAMIC_CATEGORIES,
    TRACKER_MASK_POLICY_FIXED_CATEGORIES,
    TRACKER_MASK_POLICY_TASK,
    TASK_MODE_REALTIME_POSE,
    TrackerPattern,
    create_realtime_inpaint_mask,
    make_tracker_pattern,
    make_dynamic_dropout_sensor_valid,
    normalize_tracker_pattern_categories,
    repeat_pattern_sensor_valid,
    validate_realtime_seq_len,
    validate_realtime_target,
    validate_pose_representation,
    validate_sensor_valid,
    get_schema_spec,
)
from data_loaders.stationary_label_config import STATIONARY_LABEL_METADATA_FIELDS
from utils.normalizer import RealtimePoseNormalizer
from utils.run_dirs import resolve_latest_or_self


@dataclass(frozen=True)
class RandomContext:
    worker_id: int
    access_index: int
    epoch: int = 0


class RealtimePoseTaskDataset(Dataset):
    """从 source-reference task 在线构造 `[C,T]` 训练样本。"""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        seq_len: int = REALTIME_POSE_SEQ_LEN,
        normalizer_dir: str | Path | None = None,
        normalize_input: bool = True,
        preload_data: bool = False,
        source_cache_max_mib: int = 512,
        folder_path: str | Path | None = None,
        tracker_pos_noise_std: float = 0.0,
        tracker_rot_noise_std: float = 0.0,
        non_head_tracker_dropout_prob: float = 0.0,
        history_pose_noise_std: float = 0.0,
        history_yaw_noise_std: float = 0.0,
        root_yaw_ref_noise_std: float = 0.0,
        tracker_mask_policy: str = TRACKER_MASK_POLICY_AUTO,
        tracker_mask_seed: int = 0,
        tracker_mask_fill: str = TRACKER_MASK_FILL_ZERO,
        tracker_mask_categories: list[str] | tuple[str, ...] | None = None,
        schema_name: str = REALTIME_POSE_SCHEMA_NAME,
        history_pose_dropout_prob: float = 0.0,
        history_pose_replace_prob: float = 0.0,
        history_yaw_replace_prob: float = 0.0,
        history_root_yaw_drift_std: float = 0.0,
        tracker_latency_max_frames: int = 0,
        tracker_burst_dropout_prob: float = 0.0,
        tracker_outlier_prob: float = 0.0,
        enable_rollout: bool = False,
        rollout_steps: int = 1,
        samples_per_source_override: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = int(seq_len)
        validate_realtime_seq_len(self.seq_len)
        self.schema = get_schema_spec(schema_name)
        self.normalize_input = bool(normalize_input)
        self.preload_data = bool(preload_data)
        if self.preload_data:
            raise ValueError("source-reference task 不支持 preload_data=True；请使用按需 source 读取。")
        self.source_cache_max_bytes = max(int(source_cache_max_mib), 0) * 1024 * 1024
        self.is_train_split = "train" in str(split).lower()
        self.tracker_pos_noise_std = float(tracker_pos_noise_std)
        self.tracker_rot_noise_std = float(tracker_rot_noise_std)
        self.non_head_tracker_dropout_prob = float(non_head_tracker_dropout_prob)
        self.history_pose_noise_std = float(history_pose_noise_std)
        self.history_yaw_noise_std = float(history_yaw_noise_std)
        self.root_yaw_ref_noise_std = float(root_yaw_ref_noise_std)
        self.history_pose_dropout_prob = float(history_pose_dropout_prob)
        self.history_pose_replace_prob = float(history_pose_replace_prob)
        self.history_yaw_replace_prob = float(history_yaw_replace_prob)
        self.history_root_yaw_drift_std = float(history_root_yaw_drift_std)
        self.tracker_latency_max_frames = int(tracker_latency_max_frames)
        self.tracker_burst_dropout_prob = float(tracker_burst_dropout_prob)
        self.tracker_outlier_prob = float(tracker_outlier_prob)
        self.rollout_steps = int(rollout_steps)
        if self.rollout_steps < 1:
            raise ValueError(f"rollout_steps must be >= 1, got {rollout_steps}")
        self.enable_rollout = bool(enable_rollout) and self.rollout_steps > 1
        self.samples_per_source_override = (
            None if samples_per_source_override is None else int(samples_per_source_override)
        )
        if self.samples_per_source_override is not None and self.samples_per_source_override <= 0:
            raise ValueError("samples_per_source_override must be positive")
        self.tracker_mask_policy = self.resolve_tracker_mask_policy(tracker_mask_policy)
        self.tracker_mask_seed = int(tracker_mask_seed)
        self.tracker_mask_fill = str(tracker_mask_fill)
        if self.tracker_mask_fill not in TRACKER_MASK_FILL_MODES:
            raise ValueError(f"tracker_mask_fill 当前只支持 {TRACKER_MASK_FILL_MODES}，实际为 {tracker_mask_fill}")
        self.tracker_mask_categories = normalize_tracker_pattern_categories(tracker_mask_categories)
        self.epoch = 0
        self.access_index = 0
        self.normalizer = create_normalizer(
            normalizer_dir=normalizer_dir,
            normalize_input=self.normalize_input,
            schema_name=self.schema.name,
        )

        self.manifest_path = find_manifest_path(data_dir=self.data_dir, split=split)
        self.manifest_dir = self.manifest_path.parent
        self.task_marker = load_source_reference_task_marker(self.manifest_path, schema_name=self.schema.name)
        self.entries = read_task_manifest(self.manifest_path)
        if folder_path:
            self.entries = filter_entries_by_folder_path(self.entries, folder_path=folder_path)
        if not self.entries:
            raise RuntimeError(f"{self.manifest_path} 中没有可用 realtime_pose task。")

        for entry in self.entries:
            reject_materialized_entry(entry, source=str(self.manifest_path))
            if str(entry.get("schema_name", "")) != self.schema.name:
                raise ValueError(f"任务 {entry.get('task_id')} 不是 {self.schema.name}。")
            if str(entry.get("task_format", "")) != self.schema.task_format:
                raise ValueError(f"任务 {entry.get('task_id')} 的 task_format 不匹配。")
            validate_pose_representation(
                entry.get(POSE_REPRESENTATION_KEY),
                schema_name=self.schema.name,
                source=f"{self.manifest_path}:{entry.get('task_id', '<unknown>')}",
            )
            if int(entry.get("feature_dim", -1)) != self.schema.feature_dim:
                raise ValueError(f"任务 {entry.get('task_id')} feature_dim 不等于 {self.schema.feature_dim}。")
            if int(entry.get("seq_len", -1)) != self.seq_len:
                raise ValueError(f"任务 {entry.get('task_id')} 的 seq_len 不等于 {self.seq_len}。")
            validate_realtime_target(int(entry.get("target_start", -1)), int(entry.get("target_length", -1)))
            validate_source_reference_entry(entry, schema=self.schema, required_rollout_steps=self.rollout_steps)

        self.sample_refs = [
            (entry_index, sample_slot)
            for entry_index, entry in enumerate(self.entries)
            for sample_slot in range(
                self.samples_per_source_override
                if self.samples_per_source_override is not None
                else int(entry["samples_per_source"])
            )
        ]
        self._source_cache: OrderedDict[Path, tuple[dict[str, np.ndarray], int]] = OrderedDict()
        self._source_cache_bytes = 0
        self._last_chain_key: tuple[int, int] | None = None
        self._last_chain_tasks: list[dict[str, np.ndarray]] | None = None

    def __len__(self) -> int:
        return len(self.sample_refs)

    def __getitem__(self, index: int) -> dict:
        entry_index, _sample_slot = self.sample_refs[index]
        entry = self.entries[entry_index]
        random_context = self.next_random_context()
        task = self.load_task(index=index, entry=entry)
        arrays = load_realtime_task_arrays(task=task, seq_len=self.seq_len, schema_name=self.schema.name)
        arrays, applied_tracker_pattern = self.apply_tracker_mask_policy(
            arrays=arrays,
            entry=entry,
            index=index,
            random_context=random_context,
        )
        if self.is_train_split:
            rng = self.stable_rng(
                entry=entry,
                index=index,
                salt=f"augment:e{random_context.epoch}:w{random_context.worker_id}:a{random_context.access_index}",
            )
            dropout_timeline_seed = self.stable_seed(
                entry=entry,
                index=index,
                salt=(
                    f"augment_dropout:e{random_context.epoch}:"
                    f"w{random_context.worker_id}:a{random_context.access_index}"
                ),
            )
            tracker_latency_frames = sample_chain_tracker_latency(
                max_frames=self.tracker_latency_max_frames,
                chain_seed=dropout_timeline_seed,
            )
            sensor_valid_before_augmentation = arrays["sensor_valid"].copy()
            arrays = augment_realtime_arrays(
                arrays=arrays,
                rng=rng,
                tracker_pos_noise_std=self.tracker_pos_noise_std,
                tracker_rot_noise_std=self.tracker_rot_noise_std,
                non_head_tracker_dropout_prob=(
                    0.0
                    if applied_tracker_pattern == "standard_three" or self.tracker_mask_policy != TRACKER_MASK_POLICY_TASK
                    else self.non_head_tracker_dropout_prob
                ),
                tracker_latency_max_frames=self.tracker_latency_max_frames,
                tracker_latency_frames=tracker_latency_frames,
                tracker_burst_dropout_prob=(
                    0.0
                    if applied_tracker_pattern == "standard_three" or self.tracker_mask_policy != TRACKER_MASK_POLICY_TASK
                    else self.tracker_burst_dropout_prob
                ),
                tracker_outlier_prob=self.tracker_outlier_prob,
                history_pose_noise_std=self.history_pose_noise_std,
                history_yaw_noise_std=self.history_yaw_noise_std,
                root_yaw_ref_noise_std=self.root_yaw_ref_noise_std,
                dropout_timeline_seed=dropout_timeline_seed,
            )
            refresh_tracker_reference(
                arrays,
                refresh_resolver_state=not np.array_equal(
                    sensor_valid_before_augmentation,
                    arrays["sensor_valid"],
                ),
            )

        sensor_valid = arrays["sensor_valid"]
        raw_features = encode_realtime_pose_features(arrays, schema_name=self.schema.name)
        features = raw_features.copy()
        if self.normalizer is not None:
            features = self.normalizer.normalize(features)
            zero_missing_tracker_channels(features=features, sensor_valid=sensor_valid, schema_name=self.schema.name)

        conditioned = features.copy()
        if self.is_train_split:
            history_rng = self.stable_rng(
                entry=entry,
                index=index,
                salt=f"history_condition:e{random_context.epoch}:w{random_context.worker_id}:a{random_context.access_index}",
            )
            apply_history_condition_corruption(
                conditioned=conditioned,
                schema=self.schema,
                rng=history_rng,
                history_pose_noise_std=self.history_pose_noise_std,
                history_yaw_noise_std=self.history_yaw_noise_std,
                history_pose_dropout_prob=self.history_pose_dropout_prob,
                history_pose_replace_prob=self.history_pose_replace_prob,
                history_yaw_replace_prob=self.history_yaw_replace_prob,
                history_root_yaw_drift_std=self.history_root_yaw_drift_std,
            )
        conditioned[REALTIME_POSE_TARGET_START, self.schema.target_slice()] = 0.0
        inpaint_mask = np.asarray(arrays["inpaint_mask"], dtype=bool)
        valid_frame_mask = np.ones(self.seq_len, dtype=bool)

        item = {
            "x": torch.from_numpy(features.T).float(),
            "conditioned_x": torch.from_numpy(conditioned.T).float(),
            "valid_frame_mask": torch.from_numpy(valid_frame_mask).bool(),
            "attention_mask": torch.from_numpy(valid_frame_mask).bool(),
            "sensor_valid": torch.from_numpy(sensor_valid.T).bool(),
            "inpaint_mask": torch.from_numpy(inpaint_mask.T).bool(),
            "target_joints_world": torch.from_numpy(arrays["joints_world"][REALTIME_POSE_TARGET_START]).float(),
            "gt_prev_joints_world": torch.from_numpy(
                arrays["joints_world"][REALTIME_POSE_TARGET_START - 1]
            ).float(),
            "gt_prev_local_pose_6d": torch.from_numpy(
                raw_features[REALTIME_POSE_TARGET_START - 1, self.schema.body_pose_slice()]
            ).float(),
            "target_root_pos_world": torch.from_numpy(arrays["root_pos_world"][REALTIME_POSE_TARGET_START]).float(),
            "prev_root_pos_world": torch.from_numpy(arrays["root_pos_world"][REALTIME_POSE_TARGET_START - 1]).float(),
            "prev_root_yaw": torch.tensor(float(arrays["root_yaw"][REALTIME_POSE_TARGET_START - 1])).float(),
            "gt_prev_root_yaw": torch.tensor(
                float(arrays["root_yaw"][REALTIME_POSE_TARGET_START - 1])
            ).float(),
            "target_root_yaw": torch.tensor(float(arrays["root_yaw"][REALTIME_POSE_TARGET_START])).float(),
            "tracker_ref_root_pos_world": torch.from_numpy(
                arrays["tracker_ref_root_pos_world"][REALTIME_POSE_TARGET_START]
            ).float(),
            "tracker_ref_root_yaw": torch.tensor(
                float(arrays["tracker_ref_root_yaw"][REALTIME_POSE_TARGET_START])
            ).float(),
            "tracker_ref_source": torch.tensor(
                int(arrays["tracker_ref_source"][REALTIME_POSE_TARGET_START]), dtype=torch.int64
            ),
            "target_timestamp_seconds": torch.tensor(
                float(arrays["timestamp_seconds"][REALTIME_POSE_TARGET_START]), dtype=torch.float64
            ),
            "target_frame_dt_seconds": torch.tensor(
                float(
                    arrays["timestamp_seconds"][REALTIME_POSE_TARGET_START]
                    - arrays["timestamp_seconds"][REALTIME_POSE_TARGET_START - 1]
                ),
                dtype=torch.float64,
            ),
            "target_floor_y": torch.tensor(float(arrays["floor_y"][REALTIME_POSE_TARGET_START])).float(),
            "target_tracking_origin_revision": torch.tensor(
                int(arrays["tracking_origin_revision"][REALTIME_POSE_TARGET_START]), dtype=torch.int64
            ),
            **resolver_state_tensors(arrays, prefix="resolver_window_start"),
            "resolver_before_target_root_pos_world": torch.from_numpy(
                arrays["resolver_before_target_root_pos_world"]
            ).float(),
            "resolver_before_target_root_yaw": torch.tensor(
                float(arrays["resolver_before_target_root_yaw"])
            ).float(),
            "resolver_before_target_pelvis_height": torch.tensor(
                float(arrays["resolver_before_target_pelvis_height"])
            ).float(),
            "resolver_before_target_joints_world": torch.from_numpy(
                arrays["resolver_before_target_joints_world"]
            ).float(),
            "resolver_before_target_hip_valid": torch.tensor(
                bool(arrays["resolver_before_target_hip_valid"]), dtype=torch.bool
            ),
            **resolver_state_tensors(arrays, prefix="resolver_before_target"),
            "target_tracker_pos_ref": torch.from_numpy(
                raw_features[
                    REALTIME_POSE_TARGET_START,
                    self.schema.tracker_pos_slice(),
                ].reshape(TRACKER_COUNT, 3)
            ).float(),
            "target_tracker_rot_ref_6d": torch.from_numpy(
                raw_features[
                    REALTIME_POSE_TARGET_START,
                    self.schema.tracker_rot_slice(),
                ].reshape(TRACKER_COUNT, 6)
            ).float(),
            "target_sensor_valid": torch.from_numpy(sensor_valid[REALTIME_POSE_TARGET_START]).bool(),
            "joint_offsets_parent": torch.from_numpy(arrays["joint_offsets_parent"]).float(),
            "length": self.seq_len,
            "keyid": entry.get("task_id", ""),
            "source_path": entry.get("source_path", ""),
            "task_mode": entry.get("task_mode", ""),
            "schema_name": self.schema.name,
            "target_start": REALTIME_POSE_TARGET_START,
            "target_length": REALTIME_POSE_TARGET_LENGTH,
            "tracker_pattern": applied_tracker_pattern,
            "tracker_mask_policy": self.tracker_mask_policy,
        }
        item.update(
            {
                "pred_prev_joints_world": torch.from_numpy(
                    arrays["joints_world"][REALTIME_POSE_TARGET_START - 1]
                ).float(),
                "pred_prev_local_pose_6d": torch.from_numpy(
                    raw_features[
                        REALTIME_POSE_TARGET_START - 1,
                        self.schema.body_pose_slice(),
                    ]
                ).float(),
                "previous_state_is_predicted": torch.tensor(False, dtype=torch.bool),
            }
        )
        if "joint_rest_local_rotations_6d" in arrays:
            item["joint_rest_local_rotations_6d"] = torch.from_numpy(arrays["joint_rest_local_rotations_6d"]).float()
        if self.schema.supports_root_motion:
            item["target_root_delta_xz_ref"] = torch.from_numpy(
                arrays["root_delta_xz_ref"][REALTIME_POSE_TARGET_START]
            ).float()
            item["target_root_height"] = torch.tensor(
                float(arrays[self.schema.pelvis_height_key][REALTIME_POSE_TARGET_START, 0])
            ).float()
        if self.schema.supports_stationary_prob:
            item["target_stationary_prob_5"] = torch.from_numpy(
                arrays["stationary_prob_5"][REALTIME_POSE_TARGET_START]
            ).float()
        if self.enable_rollout:
            item["rollout"] = self.build_rollout_items(
                entry=entry,
                index=index,
                random_context=random_context,
                base_item=item,
            )
        return item

    def build_rollout_items(
        self,
        entry: dict,
        index: int,
        random_context: RandomContext,
        base_item: dict,
    ) -> list[dict]:
        if self._last_chain_key != (int(index), int(self.effective_sampling_epoch())) or self._last_chain_tasks is None:
            self.load_task(index=index, entry=entry)
        assert self._last_chain_tasks is not None
        items = []
        previous_item = base_item
        for rollout_step in range(1, self.rollout_steps):
            task = self._last_chain_tasks[rollout_step]
            next_item = self.materialize_item_from_task(
                task=task,
                entry=entry,
                index=index,
                random_context=random_context,
                keyid_suffix=f":rollout{rollout_step}",
            )
            # source-reference task 先构造整条连续链。这里显式传播重叠的 60 帧，
            # 让 tracker/history 的随机增强与真实滑窗一样只发生一次；最后一帧
            # 随后会由训练循环用上一窗口的预测完整回灌。
            next_item["x"][:, :REALTIME_POSE_TARGET_START] = previous_item["x"][:, 1:]
            next_item["conditioned_x"][:, :REALTIME_POSE_TARGET_START] = previous_item["conditioned_x"][:, 1:]
            next_item["sensor_valid"][:, :REALTIME_POSE_TARGET_START] = previous_item["sensor_valid"][:, 1:]
            items.append(next_item)
            previous_item = next_item
        return items

    def materialize_item_from_task(
        self,
        task: dict[str, np.ndarray],
        entry: dict,
        index: int,
        random_context: RandomContext,
        keyid_suffix: str = "",
    ) -> dict:
        """把相邻 rollout 窗口转换成和主窗口一致的训练 batch 字段。"""

        arrays = load_realtime_task_arrays(task=task, seq_len=self.seq_len, schema_name=self.schema.name)
        arrays, applied_tracker_pattern = self.apply_tracker_mask_policy(
            arrays=arrays,
            entry=entry,
            index=index,
            random_context=random_context,
        )
        if self.is_train_split:
            rng = self.stable_rng(
                entry=entry,
                index=index,
                salt=f"augment:{keyid_suffix}:e{random_context.epoch}:w{random_context.worker_id}:a{random_context.access_index}",
            )
            # mask 增强和主窗口共享同一 timeline seed；窗口内的局部索引不能决定掉线状态。
            dropout_timeline_seed = self.stable_seed(
                entry=entry,
                index=index,
                salt=(
                    f"augment_dropout:e{random_context.epoch}:"
                    f"w{random_context.worker_id}:a{random_context.access_index}"
                ),
            )
            tracker_latency_frames = sample_chain_tracker_latency(
                max_frames=self.tracker_latency_max_frames,
                chain_seed=dropout_timeline_seed,
            )
            sensor_valid_before_augmentation = arrays["sensor_valid"].copy()
            arrays = augment_realtime_arrays(
                arrays=arrays,
                rng=rng,
                tracker_pos_noise_std=self.tracker_pos_noise_std,
                tracker_rot_noise_std=self.tracker_rot_noise_std,
                non_head_tracker_dropout_prob=(
                    0.0
                    if applied_tracker_pattern == "standard_three" or self.tracker_mask_policy != TRACKER_MASK_POLICY_TASK
                    else self.non_head_tracker_dropout_prob
                ),
                tracker_latency_max_frames=self.tracker_latency_max_frames,
                tracker_latency_frames=tracker_latency_frames,
                tracker_burst_dropout_prob=(
                    0.0
                    if applied_tracker_pattern == "standard_three" or self.tracker_mask_policy != TRACKER_MASK_POLICY_TASK
                    else self.tracker_burst_dropout_prob
                ),
                tracker_outlier_prob=self.tracker_outlier_prob,
                history_pose_noise_std=self.history_pose_noise_std,
                history_yaw_noise_std=self.history_yaw_noise_std,
                root_yaw_ref_noise_std=self.root_yaw_ref_noise_std,
                dropout_timeline_seed=dropout_timeline_seed,
            )
            refresh_tracker_reference(
                arrays,
                refresh_resolver_state=not np.array_equal(
                    sensor_valid_before_augmentation,
                    arrays["sensor_valid"],
                ),
            )

        sensor_valid = arrays["sensor_valid"]
        raw_features = encode_realtime_pose_features(arrays, schema_name=self.schema.name)
        features = raw_features.copy()
        if self.normalizer is not None:
            features = self.normalizer.normalize(features)
            zero_missing_tracker_channels(features=features, sensor_valid=sensor_valid, schema_name=self.schema.name)

        conditioned = features.copy()
        if self.is_train_split:
            rng = self.stable_rng(
                entry=entry,
                index=index,
                salt=f"history_condition:{keyid_suffix}:e{random_context.epoch}:w{random_context.worker_id}:a{random_context.access_index}",
            )
            # rollout 子窗口的预测历史由训练循环实时回灌，避免复用离线 cache 泄漏到相邻窗口。
            apply_history_condition_corruption(
                conditioned=conditioned,
                schema=self.schema,
                rng=rng,
                history_pose_noise_std=self.history_pose_noise_std,
                history_yaw_noise_std=self.history_yaw_noise_std,
                history_pose_dropout_prob=self.history_pose_dropout_prob,
                history_pose_replace_prob=self.history_pose_replace_prob,
                history_yaw_replace_prob=self.history_yaw_replace_prob,
                history_root_yaw_drift_std=self.history_root_yaw_drift_std,
            )
        conditioned[REALTIME_POSE_TARGET_START, self.schema.target_slice()] = 0.0
        inpaint_mask = np.asarray(arrays["inpaint_mask"], dtype=bool)
        valid_frame_mask = np.ones(self.seq_len, dtype=bool)

        item = {
            "x": torch.from_numpy(features.T).float(),
            "conditioned_x": torch.from_numpy(conditioned.T).float(),
            "valid_frame_mask": torch.from_numpy(valid_frame_mask).bool(),
            "attention_mask": torch.from_numpy(valid_frame_mask).bool(),
            "sensor_valid": torch.from_numpy(sensor_valid.T).bool(),
            "inpaint_mask": torch.from_numpy(inpaint_mask.T).bool(),
            "target_joints_world": torch.from_numpy(arrays["joints_world"][REALTIME_POSE_TARGET_START]).float(),
            "gt_prev_joints_world": torch.from_numpy(
                arrays["joints_world"][REALTIME_POSE_TARGET_START - 1]
            ).float(),
            "gt_prev_local_pose_6d": torch.from_numpy(
                raw_features[REALTIME_POSE_TARGET_START - 1, self.schema.body_pose_slice()]
            ).float(),
            "target_root_pos_world": torch.from_numpy(arrays["root_pos_world"][REALTIME_POSE_TARGET_START]).float(),
            "prev_root_pos_world": torch.from_numpy(arrays["root_pos_world"][REALTIME_POSE_TARGET_START - 1]).float(),
            "prev_root_yaw": torch.tensor(float(arrays["root_yaw"][REALTIME_POSE_TARGET_START - 1])).float(),
            "gt_prev_root_yaw": torch.tensor(
                float(arrays["root_yaw"][REALTIME_POSE_TARGET_START - 1])
            ).float(),
            "target_root_yaw": torch.tensor(float(arrays["root_yaw"][REALTIME_POSE_TARGET_START])).float(),
            "tracker_ref_root_pos_world": torch.from_numpy(
                arrays["tracker_ref_root_pos_world"][REALTIME_POSE_TARGET_START]
            ).float(),
            "tracker_ref_root_yaw": torch.tensor(
                float(arrays["tracker_ref_root_yaw"][REALTIME_POSE_TARGET_START])
            ).float(),
            "tracker_ref_source": torch.tensor(
                int(arrays["tracker_ref_source"][REALTIME_POSE_TARGET_START]), dtype=torch.int64
            ),
            "target_timestamp_seconds": torch.tensor(
                float(arrays["timestamp_seconds"][REALTIME_POSE_TARGET_START]), dtype=torch.float64
            ),
            "target_frame_dt_seconds": torch.tensor(
                float(
                    arrays["timestamp_seconds"][REALTIME_POSE_TARGET_START]
                    - arrays["timestamp_seconds"][REALTIME_POSE_TARGET_START - 1]
                ),
                dtype=torch.float64,
            ),
            "target_floor_y": torch.tensor(float(arrays["floor_y"][REALTIME_POSE_TARGET_START])).float(),
            "target_tracking_origin_revision": torch.tensor(
                int(arrays["tracking_origin_revision"][REALTIME_POSE_TARGET_START]), dtype=torch.int64
            ),
            **resolver_state_tensors(arrays, prefix="resolver_window_start"),
            "resolver_before_target_root_pos_world": torch.from_numpy(
                arrays["resolver_before_target_root_pos_world"]
            ).float(),
            "resolver_before_target_root_yaw": torch.tensor(
                float(arrays["resolver_before_target_root_yaw"])
            ).float(),
            "resolver_before_target_pelvis_height": torch.tensor(
                float(arrays["resolver_before_target_pelvis_height"])
            ).float(),
            "resolver_before_target_joints_world": torch.from_numpy(
                arrays["resolver_before_target_joints_world"]
            ).float(),
            "resolver_before_target_hip_valid": torch.tensor(
                bool(arrays["resolver_before_target_hip_valid"]), dtype=torch.bool
            ),
            **resolver_state_tensors(arrays, prefix="resolver_before_target"),
            "target_tracker_pos_ref": torch.from_numpy(
                raw_features[REALTIME_POSE_TARGET_START, self.schema.tracker_pos_slice()].reshape(TRACKER_COUNT, 3)
            ).float(),
            "target_tracker_rot_ref_6d": torch.from_numpy(
                raw_features[REALTIME_POSE_TARGET_START, self.schema.tracker_rot_slice()].reshape(TRACKER_COUNT, 6)
            ).float(),
            "target_sensor_valid": torch.from_numpy(sensor_valid[REALTIME_POSE_TARGET_START]).bool(),
            "joint_offsets_parent": torch.from_numpy(arrays["joint_offsets_parent"]).float(),
            "length": self.seq_len,
            "keyid": f"{entry.get('task_id', '')}{keyid_suffix}",
            "source_path": entry.get("source_path", ""),
            "task_mode": entry.get("task_mode", ""),
            "schema_name": self.schema.name,
            "target_start": REALTIME_POSE_TARGET_START,
            "target_length": REALTIME_POSE_TARGET_LENGTH,
            "tracker_pattern": applied_tracker_pattern,
            "tracker_mask_policy": self.tracker_mask_policy,
        }
        if "joint_rest_local_rotations_6d" in arrays:
            item["joint_rest_local_rotations_6d"] = torch.from_numpy(arrays["joint_rest_local_rotations_6d"]).float()
        if self.schema.supports_root_motion:
            item["target_root_delta_xz_ref"] = torch.from_numpy(
                arrays["root_delta_xz_ref"][REALTIME_POSE_TARGET_START]
            ).float()
            item["target_root_height"] = torch.tensor(
                float(arrays[self.schema.pelvis_height_key][REALTIME_POSE_TARGET_START, 0])
            ).float()
        if self.schema.supports_stationary_prob:
            item["target_stationary_prob_5"] = torch.from_numpy(
                arrays["stationary_prob_5"][REALTIME_POSE_TARGET_START]
            ).float()
        return item

    def load_task(self, index: int, entry: dict) -> dict[str, np.ndarray]:
        chain_key = (int(index), int(self.effective_sampling_epoch()))
        if self._last_chain_key == chain_key and self._last_chain_tasks is not None:
            return self._last_chain_tasks[0]
        _entry_index, sample_slot = self.sample_refs[index]
        source_path = resolve_source_reference_path(entry=entry, marker=self.task_marker)
        source = self.load_source(source_path)
        start_frame = self.sample_start_frame(entry=entry, sample_slot=sample_slot)
        full_sensor_valid = self.build_task_sensor_timeline(entry=entry, source_frames=int(entry["source_frames"]))
        chain_steps = self.rollout_steps if self.enable_rollout else 1
        tasks = [
            build_online_task_window(
                source=source,
                entry=entry,
                schema=self.schema,
                start_frame=start_frame + rollout_step,
                rollout_step=rollout_step,
                full_sensor_valid=full_sensor_valid,
            )
            for rollout_step in range(chain_steps)
        ]
        self._last_chain_key = chain_key
        self._last_chain_tasks = tasks
        return tasks[0]

    def effective_sampling_epoch(self) -> int:
        return int(self.epoch) if self.is_train_split else 0

    def sample_start_frame(self, entry: dict, sample_slot: int) -> int:
        return select_source_window_start(
            entry=entry,
            sample_slot=sample_slot,
            sampling_epoch=self.effective_sampling_epoch(),
        )

    def load_source(self, source_path: Path) -> dict[str, np.ndarray]:
        resolved = source_path.resolve()
        cached = self._source_cache.pop(resolved, None)
        if cached is not None:
            self._source_cache[resolved] = cached
            return cached[0]
        source = load_realtime_source(resolved, schema_name=self.schema.name)
        source_bytes = sum(value.nbytes for value in source.values() if isinstance(value, np.ndarray))
        if 0 < source_bytes <= self.source_cache_max_bytes:
            while self._source_cache and self._source_cache_bytes + source_bytes > self.source_cache_max_bytes:
                _evicted_path, (_evicted_source, evicted_bytes) = self._source_cache.popitem(last=False)
                self._source_cache_bytes -= evicted_bytes
            self._source_cache[resolved] = (source, source_bytes)
            self._source_cache_bytes += source_bytes
        return source

    def build_task_sensor_timeline(self, entry: dict, source_frames: int) -> np.ndarray:
        return build_source_reference_sensor_timeline(
            entry=entry,
            source_frames=source_frames,
            sampling_epoch=self.effective_sampling_epoch(),
            tracker_mask_seed=self.tracker_mask_seed,
        )

    def set_epoch(self, epoch: int) -> None:
        """训练循环在 epoch 开头调用，使动态 mask 和增强能随 epoch 可复现地变化。"""

        self.epoch = int(epoch)
        self.access_index = 0

    def next_random_context(self) -> RandomContext:
        """返回当前 Dataset 实例内的随机上下文。

        DataLoader 多 worker 会复制 Dataset 实例，因此把 worker_id 放进随机种子里，
        避免每个 worker 生成完全相同的动态遮盖和增强序列。
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_id = int(worker_info.id) if worker_info is not None else 0
        access_index = self.access_index
        self.access_index += 1
        return RandomContext(worker_id=worker_id, access_index=access_index, epoch=self.epoch)

    def resolve_tracker_mask_policy(self, policy: str) -> str:
        policy = str(policy or TRACKER_MASK_POLICY_AUTO)
        if policy not in TRACKER_MASK_POLICIES:
            raise ValueError(f"未知 tracker_mask_policy={policy}，可选值为 {TRACKER_MASK_POLICIES}")
        if policy == TRACKER_MASK_POLICY_AUTO:
            return TRACKER_MASK_POLICY_DYNAMIC_CATEGORIES if self.is_train_split else TRACKER_MASK_POLICY_TASK
        return policy

    def apply_tracker_mask_policy(
        self,
        arrays: dict[str, np.ndarray],
        entry: dict,
        index: int,
        random_context: RandomContext,
    ) -> tuple[dict[str, np.ndarray], str]:
        if self.tracker_mask_policy == TRACKER_MASK_POLICY_TASK:
            return arrays, str(entry.get("tracker_pattern", "task"))

        result = {key: value.copy() for key, value in arrays.items()}
        if self.tracker_mask_policy == TRACKER_MASK_POLICY_DYNAMIC_CATEGORIES:
            category = self.dynamic_mask_category(entry=entry, index=index, random_context=random_context)
            rng = self.stable_rng(
                entry=entry,
                index=index,
                salt=(
                    f"dynamic_pattern:{category}:e{random_context.epoch}:"
                    f"w{random_context.worker_id}:a{random_context.access_index}"
                ),
            )
        elif self.tracker_mask_policy == TRACKER_MASK_POLICY_FIXED_CATEGORIES:
            category = self.fixed_mask_category(entry=entry, index=index)
            rng = self.fixed_mask_rng(entry=entry, index=index, category=category)
        else:
            raise ValueError(f"未知 tracker_mask_policy={self.tracker_mask_policy}")

        pattern = make_tracker_pattern(category, rng)
        source_frames = int(np.asarray(result["source_frames"]).item())
        absolute_start_frame = int(np.asarray(result["start_frame"]).item())
        full_sensor_valid = (
            make_dynamic_dropout_sensor_valid(rng=rng, seq_len=source_frames)
            if category == "dynamic_dropout"
            else repeat_pattern_sensor_valid(pattern, seq_len=source_frames)
        )
        result["sensor_valid"] = full_sensor_valid[
            absolute_start_frame : absolute_start_frame + self.seq_len
        ].copy()
        refresh_tracker_reference(
            result,
            full_sensor_valid=full_sensor_valid,
            absolute_start_frame=absolute_start_frame,
        )
        return result, pattern.category

    def dynamic_mask_category(self, entry: dict, index: int, random_context: RandomContext) -> str:
        # 动态遮盖按“洗牌后的类别轮转”采样，既覆盖所有类别，又能用 seed 复现实验。
        categories = tuple(self.tracker_mask_categories)
        if set(categories) == set(TRACKER_PATTERN_CATEGORIES) and len(categories) == len(TRACKER_PATTERN_CATEGORIES):
            schedule = np.asarray(
                [
                    "full_six", "full_six", "full_six",
                    "standard_three", "standard_three", "standard_three",
                    "static_sparse", "static_sparse",
                    "dynamic_dropout", "dynamic_dropout",
                ],
                dtype=object,
            )
        else:
            schedule = np.asarray(categories, dtype=object)
        cycle_index = random_context.access_index // len(schedule)
        position = random_context.access_index % len(schedule)
        rng = self.stable_rng(
            entry=entry,
            index=index,
            salt=f"dynamic_category:e{random_context.epoch}:w{random_context.worker_id}:c{cycle_index}",
        )
        rng.shuffle(schedule)
        return str(schedule[position])

    def fixed_mask_category(self, entry: dict, index: int) -> str:
        categories = self.tracker_mask_categories
        digest = self.stable_mask_digest(entry=entry, index=index, salt="category")
        category_index = int(digest[:8], 16) % len(categories)
        return categories[category_index]

    def fixed_mask_rng(self, entry: dict, index: int, category: str) -> np.random.Generator:
        return self.stable_rng(entry=entry, index=index, salt=category)

    def stable_rng(self, entry: dict, index: int, salt: str) -> np.random.Generator:
        return np.random.default_rng(self.stable_seed(entry=entry, index=index, salt=salt))

    def stable_seed(self, entry: dict, index: int, salt: str) -> int:
        digest = self.stable_mask_digest(entry=entry, index=index, salt=salt)
        return int(digest[:16], 16) % (2**32)

    def stable_mask_digest(self, entry: dict, index: int, salt: str) -> str:
        task_id = str(entry.get("task_id", index))
        payload = f"{self.tracker_mask_seed}:{task_id}:{index}:{salt}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def create_normalizer(
    normalizer_dir: str | Path | None,
    normalize_input: bool,
    schema_name: str = REALTIME_POSE_SCHEMA_NAME,
) -> RealtimePoseNormalizer | None:
    if not normalize_input:
        return None
    if normalizer_dir is None or str(normalizer_dir).strip() == "":
        raise ValueError("开启 normalize_input 时必须提供 normalizer_dir。")
    return RealtimePoseNormalizer(base_dir=normalizer_dir, schema_name=schema_name)


def find_manifest_path(data_dir: Path, split: str) -> Path:
    data_dir = resolve_latest_or_self(data_dir, kind="tasks")
    candidates = [data_dir / split / "manifest.jsonl", data_dir / "manifest.jsonl"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"找不到 realtime_pose manifest，已尝试: {tried}")


def read_task_manifest(manifest_path: Path) -> list[dict]:
    entries = []
    with manifest_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def load_source_reference_task_marker(manifest_path: Path, schema_name: str) -> dict:
    marker_candidates = (manifest_path.parent / TASK_OUTPUT_MARKER, manifest_path.parent.parent / TASK_OUTPUT_MARKER)
    marker_path = next((path for path in marker_candidates if path.exists()), None)
    if marker_path is None:
        raise FileNotFoundError(f"source-reference task 缺少 {TASK_OUTPUT_MARKER}：{manifest_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    schema = get_schema_spec(schema_name)
    if str(marker.get("task_format", "")) != schema.task_format:
        raise ValueError(
            f"{marker_path} task_format={marker.get('task_format')!r}，期望 {schema.task_format!r}；"
            "materialized task 已失效。"
        )
    if str(marker.get("schema_name", "")) != schema.name:
        raise ValueError(f"{marker_path} schema_name 与 {schema.name} 不匹配。")
    if str(marker.get("schema_canonical_name", "")) != str(schema.canonical_name):
        raise ValueError(f"{marker_path} schema_canonical_name 不匹配。")
    validate_schema_metadata(marker, schema=schema, source=str(marker_path))
    if schema.supports_stationary_prob:
        validate_stationary_label_metadata(marker, source=str(marker_path))
    source_manifest_path = Path(str(marker.get("source_manifest_path", "")))
    expected_sha = str(marker.get("source_manifest_sha256", ""))
    if not source_manifest_path.is_file() or not expected_sha:
        raise ValueError(f"{marker_path} 缺少有效 source manifest 绑定。")
    actual_sha = hashlib.sha256(source_manifest_path.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(f"source manifest SHA256 已变化：{source_manifest_path}")
    marker["marker_path"] = str(marker_path)
    return marker


def reject_materialized_entry(entry: dict, source: str) -> None:
    forbidden = [key for key in ("task_path", "rollout_task_paths") if key in entry]
    if forbidden:
        raise ValueError(f"{source}:{entry.get('task_id', '<unknown>')} 包含失效 materialized 字段 {forbidden}")


def validate_source_reference_entry(entry: dict, schema: SchemaSpec, required_rollout_steps: int) -> None:
    required = (
        "task_id",
        "task_format",
        "schema_name",
        "schema_canonical_name",
        POSE_REPRESENTATION_KEY,
        "root_y_policy",
        "pelvis_height_mode",
        "source_path",
        "source_relative_path",
        "stablemotion_split_key",
        "source_frames",
        "samples_per_source",
        "sampling_seed",
        "max_rollout_steps",
        "seq_len",
        "feature_dim",
        "target_start",
        "target_length",
        "task_mode",
        *RUNTIME_CONTRACT_METADATA_FIELDS,
    )
    if schema.supports_stationary_prob:
        required = (*required, *STATIONARY_LABEL_METADATA_FIELDS)
    missing = [key for key in required if key not in entry]
    if missing:
        raise ValueError(f"source-reference task {entry.get('task_id', '<unknown>')} 缺少字段 {missing}")
    if str(entry["task_format"]) != schema.task_format:
        raise ValueError(f"任务 {entry['task_id']} task_format={entry['task_format']!r}，期望 {schema.task_format!r}")
    if str(entry["schema_name"]) != schema.name:
        raise ValueError(f"任务 {entry['task_id']} schema_name={entry['schema_name']!r}，期望 {schema.name!r}")
    if str(entry["schema_canonical_name"]) != str(schema.canonical_name):
        raise ValueError(f"任务 {entry['task_id']} schema_canonical_name 不匹配。")
    validate_schema_metadata(entry, schema=schema, source=str(entry["task_id"]))
    validate_pose_representation(entry[POSE_REPRESENTATION_KEY], schema_name=schema.name, source=str(entry["task_id"]))
    if str(entry["root_y_policy"]) != schema.root_y_policy:
        raise ValueError(f"任务 {entry['task_id']} root_y_policy 不匹配。")
    if str(entry["pelvis_height_mode"]) != schema.pelvis_height_mode:
        raise ValueError(f"任务 {entry['task_id']} pelvis_height_mode 不匹配。")
    if str(entry["task_mode"]) != TASK_MODE_REALTIME_POSE:
        raise ValueError(f"任务 {entry['task_id']} task_mode 不匹配。")
    if int(entry["feature_dim"]) != schema.feature_dim or int(entry["seq_len"]) != schema.seq_len:
        raise ValueError(f"任务 {entry['task_id']} feature/sequence contract 不匹配。")
    validate_realtime_target(int(entry["target_start"]), int(entry["target_length"]))
    if schema.supports_stationary_prob:
        validate_stationary_label_metadata(entry, source=str(entry["task_id"]))
    if int(entry["samples_per_source"]) <= 0:
        raise ValueError(f"任务 {entry['task_id']} samples_per_source 必须为正数。")
    if int(entry["max_rollout_steps"]) < int(required_rollout_steps):
        raise ValueError(
            f"任务 {entry['task_id']} max_rollout_steps={entry['max_rollout_steps']}，"
            f"不足以返回 rollout_steps={required_rollout_steps}"
        )
    required_frames = REALTIME_POSE_SEQ_LEN + int(entry["max_rollout_steps"]) - 1
    if int(entry["source_frames"]) < required_frames:
        raise ValueError(f"任务 {entry['task_id']} source_frames 不足 {required_frames}。")


def resolve_source_reference_path(entry: dict, marker: dict) -> Path:
    relative = Path(str(entry["source_relative_path"]).replace("\\", "/"))
    relative_candidate = Path(str(marker["source_dir"])) / relative
    if relative_candidate.is_file():
        return relative_candidate
    absolute_candidate = Path(str(entry["source_path"]))
    if absolute_candidate.is_file():
        return absolute_candidate
    raise FileNotFoundError(f"source-reference task 的 source 不存在：{relative_candidate}")


def select_source_window_start(entry: dict, sample_slot: int, sampling_epoch: int) -> int:
    """从 source、epoch 与 slot 稳定派生窗口起点，并在空间足够时消除 slot 冲突。"""

    source_frames = int(entry["source_frames"])
    max_rollout_steps = int(entry["max_rollout_steps"])
    valid_start_count = source_frames - REALTIME_POSE_SEQ_LEN - max_rollout_steps + 2
    if valid_start_count <= 0:
        raise ValueError(f"任务 {entry.get('task_id')} 没有合法在线窗口起点。")
    selected: list[int] = []
    for slot in range(int(sample_slot) + 1):
        payload = (
            f"{int(entry['sampling_seed'])}:{entry['stablemotion_split_key']}:"
            f"{int(sampling_epoch)}:{slot}"
        )
        candidate = int(hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16], 16) % valid_start_count
        if len(selected) < valid_start_count:
            while candidate in selected:
                candidate = (candidate + 1) % valid_start_count
        selected.append(candidate)
    return int(selected[-1])


def build_source_reference_sensor_timeline(
    *,
    entry: dict,
    source_frames: int,
    sampling_epoch: int,
    tracker_mask_seed: int,
) -> np.ndarray:
    detail = entry.get("tracker_pattern_detail") or {}
    category = str(entry.get("tracker_pattern", "full_six"))
    task_id = str(entry.get("task_id", entry.get("stablemotion_split_key", "source")))
    payload = f"{int(tracker_mask_seed)}:{task_id}:0:task_pattern:e{int(sampling_epoch)}"
    seed = int(hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    if category == "dynamic_dropout":
        return make_dynamic_dropout_sensor_valid(rng=rng, seq_len=source_frames)
    values = detail.get("sensor_valid")
    pattern = (
        TrackerPattern(category=category, sensor_valid=tuple(bool(value) for value in values))
        if isinstance(values, list) and len(values) == TRACKER_COUNT
        else make_tracker_pattern(category, rng)
    )
    return repeat_pattern_sensor_valid(pattern, seq_len=source_frames)


def load_source_reference_window(
    *,
    entry: dict,
    source_path: Path,
    sample_slot: int = 0,
    sampling_epoch: int = 0,
    tracker_mask_seed: int = 0,
) -> dict[str, np.ndarray]:
    schema = get_schema_spec(str(entry["schema_name"]))
    validate_source_reference_entry(entry, schema=schema, required_rollout_steps=1)
    source = load_realtime_source(source_path, schema_name=schema.name)
    start_frame = select_source_window_start(
        entry=entry,
        sample_slot=sample_slot,
        sampling_epoch=sampling_epoch,
    )
    full_sensor_valid = build_source_reference_sensor_timeline(
        entry=entry,
        source_frames=int(entry["source_frames"]),
        sampling_epoch=sampling_epoch,
        tracker_mask_seed=tracker_mask_seed,
    )
    return build_online_task_window(
        source=source,
        entry=entry,
        schema=schema,
        start_frame=start_frame,
        rollout_step=0,
        full_sensor_valid=full_sensor_valid,
    )


def build_online_task_window(
    *,
    source: dict[str, np.ndarray],
    entry: dict,
    schema: SchemaSpec,
    start_frame: int,
    rollout_step: int,
    full_sensor_valid: np.ndarray,
) -> dict[str, np.ndarray]:
    actual_frames = int(np.asarray(source[schema.body_pose_key]).shape[0])
    if actual_frames != int(entry["source_frames"]):
        raise ValueError(
            f"source {entry['source_relative_path']} 帧数已变化：manifest={entry['source_frames']} actual={actual_frames}"
        )
    task = dict(clip_source(source, start_frame=start_frame, seq_len=REALTIME_POSE_SEQ_LEN))
    task.update(
        {
            "schema_name": np.asarray(schema.name),
            "task_format": np.asarray(schema.task_format),
            POSE_REPRESENTATION_KEY: np.asarray(schema.pose_representation),
            "root_y_policy": np.asarray(schema.root_y_policy),
            "pelvis_height_mode": np.asarray(schema.pelvis_height_mode),
        }
    )
    sensor_valid = np.asarray(
        full_sensor_valid[start_frame : start_frame + REALTIME_POSE_SEQ_LEN], dtype=bool
    ).copy()
    task = add_runtime_task_arrays(
        task_arrays=task,
        sensor_valid=sensor_valid,
        source=source,
        absolute_start_frame=start_frame,
        full_sensor_valid=full_sensor_valid,
    )
    task.update(
        {
            "source_path": np.asarray(str(entry["source_path"])),
            "sensor_valid": sensor_valid,
            "inpaint_mask": create_realtime_inpaint_mask(schema_name=schema.name),
            "start_frame": np.int64(start_frame),
            "target_start": np.int64(REALTIME_POSE_TARGET_START),
            "target_length": np.int64(REALTIME_POSE_TARGET_LENGTH),
            "valid_length": np.int64(REALTIME_POSE_SEQ_LEN),
            "source_frames": np.int64(actual_frames),
            "seq_len": np.int64(REALTIME_POSE_SEQ_LEN),
            "rollout_step": np.int64(rollout_step),
            "max_rollout_steps": np.int64(entry["max_rollout_steps"]),
        }
    )
    return task

def resolver_state_tensors(arrays: dict[str, np.ndarray], prefix: str) -> dict[str, torch.Tensor]:
    """Expose the complete serialized Resolver boundary state without changing the 214-D feature tensor."""

    return {
        f"{prefix}_root_pos_world": torch.from_numpy(arrays[f"{prefix}_root_pos_world"]).float(),
        f"{prefix}_root_yaw": torch.tensor(float(arrays[f"{prefix}_root_yaw"])).float(),
        f"{prefix}_pelvis_height": torch.tensor(float(arrays[f"{prefix}_pelvis_height"])).float(),
        f"{prefix}_joints_world": torch.from_numpy(arrays[f"{prefix}_joints_world"]).float(),
        f"{prefix}_hip_valid": torch.tensor(bool(arrays[f"{prefix}_hip_valid"]), dtype=torch.bool),
        f"{prefix}_reconnect_start_root_pos_world": torch.from_numpy(
            arrays[f"{prefix}_reconnect_start_root_pos_world"]
        ).float(),
        f"{prefix}_reconnect_start_root_yaw": torch.tensor(
            float(arrays[f"{prefix}_reconnect_start_root_yaw"])
        ).float(),
        f"{prefix}_reconnect_start_pelvis_height": torch.tensor(
            float(arrays[f"{prefix}_reconnect_start_pelvis_height"])
        ).float(),
        f"{prefix}_reconnect_elapsed_seconds": torch.tensor(
            float(arrays[f"{prefix}_reconnect_elapsed_seconds"])
        ).float(),
        f"{prefix}_last_timestamp_seconds": torch.tensor(
            float(arrays[f"{prefix}_last_timestamp_seconds"]), dtype=torch.float64
        ),
        f"{prefix}_floor_y": torch.tensor(float(arrays[f"{prefix}_floor_y"])).float(),
        f"{prefix}_tracking_origin_revision": torch.tensor(
            int(arrays[f"{prefix}_tracking_origin_revision"]), dtype=torch.int64
        ),
    }


def load_realtime_task_arrays(
    task: dict[str, np.ndarray],
    seq_len: int,
    schema_name: str = REALTIME_POSE_SCHEMA_NAME,
) -> dict[str, np.ndarray]:
    validate_realtime_seq_len(seq_len)
    schema = get_schema_spec(schema_name)
    task_seq_len = scalar_int(task, "seq_len")
    validate_realtime_seq_len(task_seq_len)
    valid_length = scalar_int(task, "valid_length")
    validate_realtime_seq_len(valid_length)
    validate_pose_representation(task[POSE_REPRESENTATION_KEY], schema_name=schema.name, source="task")

    arrays = {
        schema.body_pose_key: array_shape(task[schema.body_pose_key], (seq_len, BODY_POSE_DIM), schema.body_pose_key).astype(np.float32),
        "root_pos_world": array_shape(task["root_pos_world"], (seq_len, 3), "root_pos_world").astype(np.float32),
        "root_yaw": array_shape(task["root_yaw"], (seq_len,), "root_yaw").astype(np.float32),
        schema.root_heading_delta_key: array_shape(
            task[schema.root_heading_delta_key],
            (seq_len, ROOT_YAW_DELTA_DIM),
            schema.root_heading_delta_key,
        ).astype(np.float32),
        "tracker_pos_world": array_shape(task["tracker_pos_world"], (seq_len, TRACKER_COUNT, 3), "tracker_pos_world").astype(np.float32),
        "tracker_rot_world_6d": array_shape(task["tracker_rot_world_6d"], (seq_len, TRACKER_COUNT, 6), "tracker_rot_world_6d").astype(np.float32),
        "joints_world": array_shape(task["joints_world"], (seq_len, 24, 3), "joints_world").astype(np.float32),
        "joint_offsets_parent": array_shape(task["joint_offsets_parent"], (24, 3), "joint_offsets_parent").astype(np.float32),
        "sensor_valid": array_shape(task["sensor_valid"], (seq_len, SENSOR_VALID_DIM), "sensor_valid").astype(bool),
        "inpaint_mask": array_shape(task["inpaint_mask"], (seq_len, schema.feature_dim), "inpaint_mask").astype(bool),
        "tracker_ref_root_pos_world": array_shape(
            task["tracker_ref_root_pos_world"], (seq_len, 3), "tracker_ref_root_pos_world"
        ).astype(np.float32),
        "tracker_ref_root_yaw": array_shape(
            task["tracker_ref_root_yaw"], (seq_len,), "tracker_ref_root_yaw"
        ).astype(np.float32),
        "tracker_ref_source": array_shape(task["tracker_ref_source"], (seq_len,), "tracker_ref_source").astype(np.int8),
        "timestamp_seconds": array_shape(task["timestamp_seconds"], (seq_len,), "timestamp_seconds").astype(np.float64),
        "floor_y": array_shape(task["floor_y"], (seq_len,), "floor_y").astype(np.float32),
        "tracking_origin_revision": array_shape(
            task["tracking_origin_revision"], (seq_len,), "tracking_origin_revision"
        ).astype(np.int64),
        "start_frame": np.asarray(task["start_frame"], dtype=np.int64).reshape(()),
        "source_frames": np.asarray(task["source_frames"], dtype=np.int64).reshape(()),
        "resolver_context_frame_indices": array_shape(
            task["resolver_context_frame_indices"],
            (RESOLVER_CONTEXT_FRAMES,),
            "resolver_context_frame_indices",
        ).astype(np.int64),
        "resolver_context_root_pos_world": array_shape(
            task["resolver_context_root_pos_world"],
            (RESOLVER_CONTEXT_FRAMES, 3),
            "resolver_context_root_pos_world",
        ).astype(np.float32),
        "resolver_context_root_yaw": array_shape(
            task["resolver_context_root_yaw"],
            (RESOLVER_CONTEXT_FRAMES,),
            "resolver_context_root_yaw",
        ).astype(np.float32),
        "resolver_context_pelvis_height": array_shape(
            task["resolver_context_pelvis_height"],
            (RESOLVER_CONTEXT_FRAMES, 1),
            "resolver_context_pelvis_height",
        ).astype(np.float32),
        "resolver_context_joints_world": array_shape(
            task["resolver_context_joints_world"],
            (RESOLVER_CONTEXT_FRAMES, 24, 3),
            "resolver_context_joints_world",
        ).astype(np.float32),
        "resolver_context_timestamp_seconds": array_shape(
            task["resolver_context_timestamp_seconds"],
            (RESOLVER_CONTEXT_FRAMES,),
            "resolver_context_timestamp_seconds",
        ).astype(np.float64),
        "resolver_context_floor_y": array_shape(
            task["resolver_context_floor_y"],
            (RESOLVER_CONTEXT_FRAMES,),
            "resolver_context_floor_y",
        ).astype(np.float32),
        "resolver_context_tracking_origin_revision": array_shape(
            task["resolver_context_tracking_origin_revision"],
            (RESOLVER_CONTEXT_FRAMES,),
            "resolver_context_tracking_origin_revision",
        ).astype(np.int64),
        "resolver_window_start_root_pos_world": array_shape(
            task["resolver_window_start_root_pos_world"], (3,), "resolver_window_start_root_pos_world"
        ).astype(np.float32),
        "resolver_window_start_joints_world": array_shape(
            task["resolver_window_start_joints_world"], (24, 3), "resolver_window_start_joints_world"
        ).astype(np.float32),
        "resolver_window_start_reconnect_start_root_pos_world": array_shape(
            task["resolver_window_start_reconnect_start_root_pos_world"],
            (3,),
            "resolver_window_start_reconnect_start_root_pos_world",
        ).astype(np.float32),
        "resolver_before_target_root_pos_world": array_shape(
            task["resolver_before_target_root_pos_world"], (3,), "resolver_before_target_root_pos_world"
        ).astype(np.float32),
        "resolver_before_target_joints_world": array_shape(
            task["resolver_before_target_joints_world"], (24, 3), "resolver_before_target_joints_world"
        ).astype(np.float32),
        "resolver_before_target_reconnect_start_root_pos_world": array_shape(
            task["resolver_before_target_reconnect_start_root_pos_world"],
            (3,),
            "resolver_before_target_reconnect_start_root_pos_world",
        ).astype(np.float32),
    }
    for key in (
        "resolver_window_start_root_yaw",
        "resolver_window_start_pelvis_height",
        "resolver_window_start_reconnect_start_root_yaw",
        "resolver_window_start_reconnect_start_pelvis_height",
        "resolver_window_start_reconnect_elapsed_seconds",
        "resolver_window_start_floor_y",
        "resolver_before_target_root_yaw",
        "resolver_before_target_pelvis_height",
        "resolver_before_target_reconnect_start_root_yaw",
        "resolver_before_target_reconnect_start_pelvis_height",
        "resolver_before_target_reconnect_elapsed_seconds",
        "resolver_before_target_floor_y",
    ):
        arrays[key] = np.asarray(task[key], dtype=np.float32).reshape(())
    for key in (
        "resolver_window_start_last_timestamp_seconds",
        "resolver_before_target_last_timestamp_seconds",
    ):
        arrays[key] = np.asarray(task[key], dtype=np.float64).reshape(())
    arrays["resolver_window_start_hip_valid"] = np.asarray(task["resolver_window_start_hip_valid"], dtype=bool).reshape(())
    arrays["resolver_window_start_tracking_origin_revision"] = np.asarray(
        task["resolver_window_start_tracking_origin_revision"], dtype=np.int64
    ).reshape(())
    arrays["resolver_before_target_hip_valid"] = np.asarray(task["resolver_before_target_hip_valid"], dtype=bool).reshape(())
    arrays["resolver_before_target_tracking_origin_revision"] = np.asarray(
        task["resolver_before_target_tracking_origin_revision"], dtype=np.int64
    ).reshape(())
    if schema.supports_root_motion:
        arrays["root_delta_xz_ref"] = array_shape(
            task["root_delta_xz_ref"],
            (seq_len, ROOT_DELTA_XZ_DIM),
            "root_delta_xz_ref",
        ).astype(np.float32)
        arrays[schema.pelvis_height_key] = array_shape(
            task[schema.pelvis_height_key],
            (seq_len, ROOT_HEIGHT_DIM),
            schema.pelvis_height_key,
        ).astype(np.float32)
    if schema.supports_stationary_prob:
        arrays["stationary_prob_5"] = array_shape(
            task["stationary_prob_5"],
            (seq_len, STATIONARY_PROB_DIM),
            "stationary_prob_5",
        ).astype(np.float32)
    if schema.pose_representation == POSE_REPRESENTATION_BODY_FBX_LOCAL_DELTA_6D:
        arrays["joint_rest_local_rotations_6d"] = array_shape(
            task["joint_rest_local_rotations_6d"],
            (24, 6),
            "joint_rest_local_rotations_6d",
        ).astype(np.float32)
    validate_sensor_valid(arrays["sensor_valid"])
    expected_mask = np.zeros((seq_len, schema.feature_dim), dtype=bool)
    expected_mask[REALTIME_POSE_TARGET_START, schema.target_slice()] = True
    if not np.array_equal(arrays["inpaint_mask"], expected_mask):
        raise ValueError(f"inpaint_mask 必须只覆盖第 61 帧的 {schema.target_dim} 维 target。")
    return arrays


def encode_realtime_pose_features(
    arrays: dict[str, np.ndarray],
    schema_name: str = REALTIME_POSE_SCHEMA_NAME,
) -> np.ndarray:
    schema = get_schema_spec(schema_name)
    if "tracker_ref_root_pos_world" not in arrays:
        # Feature encoding only needs the inference-time tracker reference.  Resolver
        # boundary snapshots belong to serialized training tasks and must not be
        # synthesized here: evaluation helpers and legacy in-memory fixtures may
        # intentionally provide shorter masks or omit timestamp/origin metadata.
        refresh_tracker_reference(arrays, refresh_resolver_state=False)
    seq_len = arrays[schema.body_pose_key].shape[0]
    features = np.zeros((seq_len, schema.feature_dim), dtype=np.float32)
    features[:, schema.body_pose_slice()] = arrays[schema.body_pose_key]
    features[:, schema.root_yaw_delta_slice()] = arrays[schema.root_heading_delta_key]
    if schema.supports_root_motion:
        features[:, schema.root_delta_xz_slice()] = arrays["root_delta_xz_ref"]
        features[:, schema.root_height_slice()] = arrays[schema.pelvis_height_key]
    if schema.supports_stationary_prob:
        features[:, schema.stationary_prob_slice()] = arrays["stationary_prob_5"]
    features[:, schema.tracker_pos_slice()] = encode_tracker_pos_ref(arrays).reshape(seq_len, -1)
    features[:, schema.tracker_rot_slice()] = encode_tracker_rot_ref(arrays).reshape(seq_len, -1)
    features[:, schema.sensor_valid_slice()] = arrays["sensor_valid"].astype(np.float32)
    zero_missing_tracker_channels(features=features, sensor_valid=arrays["sensor_valid"], schema_name=schema.name)
    return features


def refresh_tracker_reference(
    arrays: dict[str, np.ndarray],
    *,
    refresh_resolver_state: bool = True,
    full_sensor_valid: np.ndarray | None = None,
    absolute_start_frame: int | None = None,
) -> None:
    """mask 或 observation 改变后，按推理前可计算策略重建 Tracker reference。"""

    first_previous_root = np.asarray(
        arrays.get("resolver_window_start_root_pos_world", arrays["root_pos_world"][0]),
        dtype=np.float32,
    )[None]
    first_previous_yaw = np.asarray(
        arrays.get("resolver_window_start_root_yaw", arrays["root_yaw"][0]),
        dtype=np.float32,
    ).reshape(1)
    previous_root = np.concatenate([first_previous_root, arrays["root_pos_world"][:-1]], axis=0)
    previous_yaw = np.concatenate([first_previous_yaw, arrays["root_yaw"][:-1]], axis=0)
    ref_pos, ref_yaw, ref_source = build_tracker_reference_np(
        tracker_pos_world=arrays["tracker_pos_world"],
        tracker_rot_world_6d=arrays["tracker_rot_world_6d"],
        sensor_valid=arrays["sensor_valid"],
        previous_final_root_pos_world=previous_root,
        previous_final_root_yaw=previous_yaw,
        pelvis_offset_parent=arrays["joint_offsets_parent"][0],
        floor_y=arrays.get("floor_y", 0.0),
    )
    arrays["tracker_ref_root_pos_world"] = ref_pos
    arrays["tracker_ref_root_yaw"] = ref_yaw
    arrays["tracker_ref_source"] = ref_source
    if refresh_resolver_state:
        if full_sensor_valid is not None:
            if absolute_start_frame is None:
                raise ValueError("online mask refresh requires absolute_start_frame")
            refresh_resolver_states_from_timeline(
                arrays,
                full_sensor_valid=full_sensor_valid,
                absolute_start_frame=int(absolute_start_frame),
            )
        else:
            refresh_resolver_before_target_state(arrays)


def refresh_resolver_states_from_timeline(
    arrays: dict[str, np.ndarray],
    *,
    full_sensor_valid: np.ndarray,
    absolute_start_frame: int,
) -> None:
    valid = validate_sensor_valid(np.asarray(full_sensor_valid, dtype=bool))
    source_frames = int(np.asarray(arrays["source_frames"]).item())
    if valid.shape[0] != source_frames:
        raise ValueError(f"full_sensor_valid length={valid.shape[0]}, expected source_frames={source_frames}")
    _write_online_resolver_snapshot(
        arrays,
        full_sensor_valid=valid,
        absolute_start_frame=absolute_start_frame,
        frame_index=max(absolute_start_frame - 1, 0),
        prefix="resolver_window_start",
    )
    _write_online_resolver_snapshot(
        arrays,
        full_sensor_valid=valid,
        absolute_start_frame=absolute_start_frame,
        frame_index=absolute_start_frame + REALTIME_POSE_TARGET_START - 1,
        prefix="resolver_before_target",
    )


def _write_online_resolver_snapshot(
    arrays: dict[str, np.ndarray],
    *,
    full_sensor_valid: np.ndarray,
    absolute_start_frame: int,
    frame_index: int,
    prefix: str,
) -> None:
    index = int(np.clip(frame_index, 0, full_sensor_valid.shape[0] - 1))
    hip_valid = bool(full_sensor_valid[index, HIP_TRACKER_INDEX])
    reconnect_start_index = index
    reconnect_elapsed = 0.0
    if hip_valid:
        first_valid = index
        while first_valid > 0 and full_sensor_valid[first_valid - 1, HIP_TRACKER_INDEX]:
            first_valid -= 1
        if first_valid > 0:
            candidate_start_index = first_valid - 1
            timestamps = np.asarray(arrays["timestamp_seconds"], dtype=np.float64)
            frame_dt = float(np.median(np.diff(timestamps))) if timestamps.size > 1 else 1.0 / 60.0
            elapsed = float(index - candidate_start_index) * max(frame_dt, 0.0)
            if elapsed < 0.1:
                reconnect_start_index = candidate_start_index
                reconnect_elapsed = elapsed

    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    root = _task_value_at_absolute_frame(
        arrays,
        key="root_pos_world",
        context_key="resolver_context_root_pos_world",
        absolute_start_frame=absolute_start_frame,
        frame_index=index,
    )
    yaw = _task_value_at_absolute_frame(
        arrays,
        key="root_yaw",
        context_key="resolver_context_root_yaw",
        absolute_start_frame=absolute_start_frame,
        frame_index=index,
    )
    height = _task_value_at_absolute_frame(
        arrays,
        key=schema.pelvis_height_key,
        context_key="resolver_context_pelvis_height",
        absolute_start_frame=absolute_start_frame,
        frame_index=index,
    )
    joints = _task_value_at_absolute_frame(
        arrays,
        key="joints_world",
        context_key="resolver_context_joints_world",
        absolute_start_frame=absolute_start_frame,
        frame_index=index,
    )
    reconnect_root = _task_value_at_absolute_frame(
        arrays,
        key="root_pos_world",
        context_key="resolver_context_root_pos_world",
        absolute_start_frame=absolute_start_frame,
        frame_index=reconnect_start_index,
    )
    reconnect_yaw = _task_value_at_absolute_frame(
        arrays,
        key="root_yaw",
        context_key="resolver_context_root_yaw",
        absolute_start_frame=absolute_start_frame,
        frame_index=reconnect_start_index,
    )
    reconnect_height = _task_value_at_absolute_frame(
        arrays,
        key=schema.pelvis_height_key,
        context_key="resolver_context_pelvis_height",
        absolute_start_frame=absolute_start_frame,
        frame_index=reconnect_start_index,
    )
    timestamp = _task_value_at_absolute_frame(
        arrays,
        key="timestamp_seconds",
        context_key="resolver_context_timestamp_seconds",
        absolute_start_frame=absolute_start_frame,
        frame_index=index,
    )
    floor_y = _task_value_at_absolute_frame(
        arrays,
        key="floor_y",
        context_key="resolver_context_floor_y",
        absolute_start_frame=absolute_start_frame,
        frame_index=index,
    )
    origin_revision = _task_value_at_absolute_frame(
        arrays,
        key="tracking_origin_revision",
        context_key="resolver_context_tracking_origin_revision",
        absolute_start_frame=absolute_start_frame,
        frame_index=index,
    )
    arrays[f"{prefix}_root_pos_world"] = np.asarray(root, dtype=np.float32)
    arrays[f"{prefix}_root_yaw"] = np.float32(yaw)
    arrays[f"{prefix}_pelvis_height"] = np.float32(np.asarray(height).reshape(-1)[0])
    arrays[f"{prefix}_joints_world"] = np.asarray(joints, dtype=np.float32)
    arrays[f"{prefix}_hip_valid"] = np.asarray(hip_valid)
    arrays[f"{prefix}_reconnect_start_root_pos_world"] = np.asarray(reconnect_root, dtype=np.float32)
    arrays[f"{prefix}_reconnect_start_root_yaw"] = np.float32(reconnect_yaw)
    arrays[f"{prefix}_reconnect_start_pelvis_height"] = np.float32(
        np.asarray(reconnect_height).reshape(-1)[0]
    )
    arrays[f"{prefix}_reconnect_elapsed_seconds"] = np.float32(reconnect_elapsed)
    arrays[f"{prefix}_last_timestamp_seconds"] = np.float64(timestamp)
    arrays[f"{prefix}_floor_y"] = np.float32(floor_y)
    arrays[f"{prefix}_tracking_origin_revision"] = np.int64(origin_revision)


def _task_value_at_absolute_frame(
    arrays: dict[str, np.ndarray],
    *,
    key: str,
    context_key: str,
    absolute_start_frame: int,
    frame_index: int,
):
    local_index = int(frame_index) - int(absolute_start_frame)
    values = np.asarray(arrays[key])
    if 0 <= local_index < values.shape[0]:
        return values[local_index]
    context_indices = np.asarray(arrays["resolver_context_frame_indices"], dtype=np.int64)
    matches = np.flatnonzero(context_indices == int(frame_index))
    if matches.size == 0:
        raise ValueError(
            f"Resolver context does not cover absolute frame {frame_index} for start={absolute_start_frame}"
        )
    return np.asarray(arrays[context_key])[int(matches[-1])]


def refresh_resolver_before_target_state(arrays: dict[str, np.ndarray]) -> None:
    """Keep rollout boundary state consistent when training-time masking changes."""

    index = REALTIME_POSE_TARGET_START - 1
    valid = np.asarray(arrays["sensor_valid"], dtype=bool)
    hip_valid = bool(valid[index, HIP_TRACKER_INDEX])
    reconnect_start_index = index
    reconnect_elapsed = 0.0
    if hip_valid:
        first_valid = index
        while first_valid > 0 and valid[first_valid - 1, HIP_TRACKER_INDEX]:
            first_valid -= 1
        if first_valid > 0:
            reconnect_start_index = first_valid - 1
            elapsed = float(arrays["timestamp_seconds"][index] - arrays["timestamp_seconds"][first_valid - 1])
            reconnect_elapsed = elapsed if elapsed < 0.1 else 0.0
    height_values = arrays.get("pelvis_height")
    if height_values is None:
        height = float(arrays["resolver_before_target_pelvis_height"])
        reconnect_height = height
    else:
        height = float(np.asarray(height_values)[index].reshape(-1)[0])
        reconnect_height = float(np.asarray(height_values)[reconnect_start_index].reshape(-1)[0])
    arrays["resolver_before_target_root_pos_world"] = np.asarray(arrays["root_pos_world"][index], dtype=np.float32)
    arrays["resolver_before_target_root_yaw"] = np.float32(arrays["root_yaw"][index])
    arrays["resolver_before_target_pelvis_height"] = np.float32(height)
    arrays["resolver_before_target_joints_world"] = np.asarray(arrays["joints_world"][index], dtype=np.float32)
    arrays["resolver_before_target_hip_valid"] = np.asarray(hip_valid)
    arrays["resolver_before_target_reconnect_start_root_pos_world"] = np.asarray(
        arrays["root_pos_world"][reconnect_start_index], dtype=np.float32
    )
    arrays["resolver_before_target_reconnect_start_root_yaw"] = np.float32(
        arrays["root_yaw"][reconnect_start_index]
    )
    arrays["resolver_before_target_reconnect_start_pelvis_height"] = np.float32(reconnect_height)
    arrays["resolver_before_target_reconnect_elapsed_seconds"] = np.float32(reconnect_elapsed)
    arrays["resolver_before_target_last_timestamp_seconds"] = np.float64(arrays["timestamp_seconds"][index])
    arrays["resolver_before_target_floor_y"] = np.float32(arrays["floor_y"][index])
    arrays["resolver_before_target_tracking_origin_revision"] = np.int64(
        arrays["tracking_origin_revision"][index]
    )


def encode_tracker_pos_ref(arrays: dict[str, np.ndarray]) -> np.ndarray:
    tracker_world = arrays["tracker_pos_world"].astype(np.float64)
    roots = arrays["tracker_ref_root_pos_world"].astype(np.float64)
    ref_yaw = arrays["tracker_ref_root_yaw"].astype(np.float64)
    if "root_yaw_ref_noise" in arrays:
        ref_yaw = ref_yaw + np.asarray(arrays["root_yaw_ref_noise"], dtype=np.float64)
    return encode_tracker_positions_np(tracker_world, roots, ref_yaw)


def encode_tracker_rot_ref(arrays: dict[str, np.ndarray]) -> np.ndarray:
    ref_yaw = arrays["tracker_ref_root_yaw"].astype(np.float64)
    if "root_yaw_ref_noise" in arrays:
        ref_yaw = ref_yaw + np.asarray(arrays["root_yaw_ref_noise"], dtype=np.float64)
    return encode_tracker_rotations_np(arrays["tracker_rot_world_6d"], ref_yaw)


def zero_missing_tracker_channels(
    features: np.ndarray,
    sensor_valid: np.ndarray,
    schema_name: str = REALTIME_POSE_SCHEMA_NAME,
) -> None:
    schema = get_schema_spec(schema_name)
    valid = np.asarray(sensor_valid, dtype=bool)
    for sensor_index in range(TRACKER_COUNT):
        missing = ~valid[:, sensor_index]
        if not missing.any():
            continue
        features[missing, schema.tracker_pos_slice(sensor_index)] = 0.0
        features[missing, schema.tracker_rot_slice(sensor_index)] = 0.0


def augment_realtime_arrays(
    arrays: dict[str, np.ndarray],
    rng: np.random.Generator,
    tracker_pos_noise_std: float = 0.0,
    tracker_rot_noise_std: float = 0.0,
    non_head_tracker_dropout_prob: float = 0.0,
    tracker_latency_max_frames: int = 0,
    tracker_latency_frames: int | None = None,
    tracker_burst_dropout_prob: float = 0.0,
    tracker_outlier_prob: float = 0.0,
    history_pose_noise_std: float = 0.0,
    history_yaw_noise_std: float = 0.0,
    root_yaw_ref_noise_std: float = 0.0,
    dropout_timeline_seed: int | None = None,
) -> dict[str, np.ndarray]:
    """训练增强只改 tracker 条件，history pose/yaw 污染在 `conditioned_x` 上单独做。"""

    result = {key: value.copy() for key, value in arrays.items()}
    sensor_valid = result["sensor_valid"].copy()
    dropout_prob = max(float(non_head_tracker_dropout_prob), float(tracker_burst_dropout_prob))
    if dropout_prob > 0:
        sensor_valid = dropout_non_head_trackers(
            sensor_valid=sensor_valid,
            rng=rng,
            dropout_prob=dropout_prob,
            absolute_start_frame=int(np.asarray(result.get("start_frame", 0)).item()),
            timeline_seed=dropout_timeline_seed,
        )
        result["sensor_valid"] = sensor_valid

    if tracker_latency_max_frames > 0:
        delay = (
            int(tracker_latency_frames)
            if tracker_latency_frames is not None
            else int(rng.integers(0, int(tracker_latency_max_frames) + 1))
        )
        if not 0 <= delay <= int(tracker_latency_max_frames):
            raise ValueError("tracker_latency_frames 超出 tracker_latency_max_frames")
        if delay > 0:
            for key in ("tracker_pos_world", "tracker_rot_world_6d"):
                delayed = result[key].copy()
                delayed[delay:] = result[key][:-delay]
                delayed[:delay] = result[key][:1]
                result[key] = delayed

    if tracker_pos_noise_std > 0:
        noise = rng.normal(0.0, tracker_pos_noise_std, size=result["tracker_pos_world"].shape).astype(np.float32)
        result["tracker_pos_world"] = result["tracker_pos_world"] + noise * sensor_valid[:, :, None].astype(np.float32)

    if tracker_rot_noise_std > 0:
        noise = rng.normal(0.0, tracker_rot_noise_std, size=result["tracker_rot_world_6d"].shape).astype(np.float32)
        result["tracker_rot_world_6d"] = result["tracker_rot_world_6d"] + noise * sensor_valid[:, :, None].astype(np.float32)

    if tracker_outlier_prob > 0:
        outlier_mask = (rng.random(result["tracker_pos_world"].shape[:2]) < float(tracker_outlier_prob)) & sensor_valid
        if outlier_mask.any():
            pos_outlier = rng.normal(0.0, 0.15, size=result["tracker_pos_world"].shape).astype(np.float32)
            rot_outlier = rng.normal(0.0, 0.20, size=result["tracker_rot_world_6d"].shape).astype(np.float32)
            result["tracker_pos_world"] = result["tracker_pos_world"] + pos_outlier * outlier_mask[:, :, None].astype(np.float32)
            result["tracker_rot_world_6d"] = result["tracker_rot_world_6d"] + rot_outlier * outlier_mask[:, :, None].astype(np.float32)

    if root_yaw_ref_noise_std > 0:
        result["root_yaw_ref_noise"] = rng.normal(
            0.0,
            root_yaw_ref_noise_std,
            size=(REALTIME_POSE_SEQ_LEN,),
        ).astype(np.float32)
    return result


def sample_chain_tracker_latency(max_frames: int, chain_seed: int) -> int:
    if int(max_frames) <= 0:
        return 0
    return int(np.random.default_rng(int(chain_seed)).integers(0, int(max_frames) + 1))


def apply_history_condition_corruption(
    conditioned: np.ndarray,
    schema: SchemaSpec,
    rng: np.random.Generator,
    history_pose_noise_std: float = 0.0,
    history_yaw_noise_std: float = 0.0,
    history_pose_dropout_prob: float = 0.0,
    history_pose_replace_prob: float = 0.0,
    history_yaw_replace_prob: float = 0.0,
    history_root_yaw_drift_std: float = 0.0,
) -> None:
    """只污染 history 条件，模拟 Unity 中预测 history 写回后的分布偏移。"""

    history = slice(0, REALTIME_POSE_TARGET_START)
    pose_slice = schema.body_pose_slice()
    yaw_slice = schema.root_yaw_delta_slice()
    history_pose = conditioned[history, pose_slice]
    history_yaw = conditioned[history, yaw_slice]
    if history_pose_noise_std > 0:
        history_pose += rng.normal(
            0.0,
            float(history_pose_noise_std),
            size=history_pose.shape,
        ).astype(np.float32)
    if history_pose_dropout_prob > 0:
        drop_frames = rng.random(history_pose.shape[0]) < float(history_pose_dropout_prob)
        history_pose[drop_frames] = 0.0
    if history_pose_replace_prob > 0:
        replace_frames = rng.random(history_pose.shape[0]) < float(history_pose_replace_prob)
        for frame_index in np.where(replace_frames)[0]:
            if frame_index > 0:
                history_pose[frame_index] = history_pose[frame_index - 1]
    if history_yaw_noise_std > 0:
        history_yaw += rng.normal(
            0.0,
            float(history_yaw_noise_std),
            size=history_yaw.shape,
        ).astype(np.float32)
    if history_root_yaw_drift_std > 0:
        drift = rng.normal(0.0, float(history_root_yaw_drift_std), size=(history_yaw.shape[0], 1)).astype(np.float32)
        history_yaw += np.cumsum(drift, axis=0)
    if history_yaw_replace_prob > 0:
        replace_frames = rng.random(history_yaw.shape[0]) < float(history_yaw_replace_prob)
        for frame_index in np.where(replace_frames)[0]:
            if frame_index > 0:
                history_yaw[frame_index] = history_yaw[frame_index - 1]
    yaw_norm = np.linalg.norm(history_yaw, axis=-1, keepdims=True)
    history_yaw[:] = history_yaw / np.maximum(yaw_norm, 1e-8)


def dropout_non_head_trackers(
    sensor_valid: np.ndarray,
    rng: np.random.Generator,
    dropout_prob: float,
    absolute_start_frame: int = 0,
    timeline_seed: int | None = None,
) -> np.ndarray:
    valid = np.asarray(sensor_valid, dtype=bool).copy()
    valid[:, HEAD_TRACKER_INDEX] = True
    non_head_indices = [index for index in range(TRACKER_COUNT) if index != HEAD_TRACKER_INDEX]
    for frame_index in range(valid.shape[0]):
        # rollout 相邻窗口共享绝对帧。按绝对帧派生 RNG，保证重叠 tracker mask 完全一致。
        frame_rng = rng
        if timeline_seed is not None:
            frame_rng = np.random.default_rng(
                np.random.SeedSequence([int(timeline_seed), int(absolute_start_frame) + frame_index])
            )
        original = valid[frame_index].copy()
        for _attempt in range(100):
            candidate = original.copy()
            for tracker_index in non_head_indices:
                if candidate[tracker_index] and frame_rng.random() < dropout_prob:
                    candidate[tracker_index] = False
            candidate[HEAD_TRACKER_INDEX] = True
            if candidate.sum() >= 3:
                valid[frame_index] = candidate
                break
        else:
            valid[frame_index] = original
    validate_sensor_valid(valid)
    return valid


def scalar_int(task: dict[str, np.ndarray], name: str) -> int:
    value = np.asarray(task[name])
    if value.shape != ():
        raise ValueError(f"{name} 应为标量，实际 shape={value.shape}")
    return int(value.item())


def array_shape(array: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} 应为 {shape}，实际为 {tuple(array.shape)}")
    return array
