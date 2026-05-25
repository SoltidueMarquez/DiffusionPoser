from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data_loaders.manifest_utils import filter_entries_by_folder_path
from data_loaders.realtime_pose_kinematics import (
    make_yaw_rotation_np,
    rotation_6d_forward_up_np,
    rotation_6d_to_matrix_np,
)
from data_loaders.sensor_masking import (
    BODY_POSE_DIM,
    HIP_TRACKER_INDEX,
    FOOT_CONTACT_DIM,
    REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_LENGTH,
    REALTIME_POSE_TARGET_START,
    ROOT_DELTA_XZ_DIM,
    ROOT_HEIGHT_DIM,
    ROOT_YAW_DELTA_DIM,
    SENSOR_VALID_DIM,
    SchemaSpec,
    TRACKER_COUNT,
    TRACKER_MASK_FILL_MODES,
    TRACKER_MASK_FILL_ZERO,
    TRACKER_MASK_POLICIES,
    TRACKER_MASK_POLICY_AUTO,
    TRACKER_MASK_POLICY_DYNAMIC_CATEGORIES,
    TRACKER_MASK_POLICY_FIXED_CATEGORIES,
    TRACKER_MASK_POLICY_TASK,
    make_tracker_pattern,
    normalize_tracker_pattern_categories,
    repeat_pattern_sensor_valid,
    validate_realtime_seq_len,
    validate_realtime_target,
    validate_sensor_valid,
    get_schema_spec,
)
from utils.normalizer import RealtimePoseNormalizer


@dataclass(frozen=True)
class RandomContext:
    worker_id: int
    access_index: int
    epoch: int = 0


class RealtimePoseTaskDataset(Dataset):
    """读取 realtime_pose materialized task 并输出 `[C,T]` 训练样本。"""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        seq_len: int = REALTIME_POSE_SEQ_LEN,
        normalizer_dir: str | Path | None = None,
        normalize_input: bool = True,
        preload_data: bool = False,
        folder_path: str | Path | None = None,
        tracker_pos_noise_std: float = 0.0,
        tracker_rot_noise_std: float = 0.0,
        non_hip_tracker_dropout_prob: float = 0.0,
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
        predicted_history_cache_dir: str | Path | None = None,
        predicted_history_prob: float = 0.0,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = int(seq_len)
        validate_realtime_seq_len(self.seq_len)
        self.schema = get_schema_spec(schema_name)
        self.normalize_input = bool(normalize_input)
        self.preload_data = bool(preload_data)
        self.is_train_split = "train" in str(split).lower()
        self.tracker_pos_noise_std = float(tracker_pos_noise_std)
        self.tracker_rot_noise_std = float(tracker_rot_noise_std)
        self.non_hip_tracker_dropout_prob = float(non_hip_tracker_dropout_prob)
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
        self.predicted_history_cache_dir = Path(predicted_history_cache_dir) if predicted_history_cache_dir else None
        self.predicted_history_prob = float(predicted_history_prob)
        self.tracker_mask_policy = self.resolve_tracker_mask_policy(tracker_mask_policy)
        self.tracker_mask_seed = int(tracker_mask_seed)
        self.tracker_mask_fill = str(tracker_mask_fill)
        if self.tracker_mask_fill not in TRACKER_MASK_FILL_MODES:
            raise ValueError(f"tracker_mask_fill 目前只支持 {TRACKER_MASK_FILL_MODES}，实际为 {tracker_mask_fill}")
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
        self.entries = read_task_manifest(self.manifest_path)
        if folder_path:
            self.entries = filter_entries_by_folder_path(self.entries, folder_path=folder_path)
        if not self.entries:
            raise RuntimeError(f"{self.manifest_path} 中没有可用 realtime_pose task。")

        for entry in self.entries:
            if str(entry.get("schema_name", "")) != self.schema.name:
                raise ValueError(f"任务 {entry.get('task_id')} 不是 {self.schema.name}。")
            if str(entry.get("task_format", "")) != self.schema.task_format:
                raise ValueError(f"任务 {entry.get('task_id')} 的 task_format 不匹配。")
            if int(entry.get("feature_dim", -1)) != self.schema.feature_dim:
                raise ValueError(f"任务 {entry.get('task_id')} feature_dim 不等于 {self.schema.feature_dim}。")
            if int(entry.get("seq_len", -1)) != self.seq_len:
                raise ValueError(f"任务 {entry.get('task_id')} 的 seq_len 不等于 {self.seq_len}。")
            validate_realtime_target(int(entry.get("target_start", -1)), int(entry.get("target_length", -1)))

        self.task_cache = None
        if self.preload_data:
            self.task_cache = [
                load_materialized_task_npz(
                    manifest_dir=self.manifest_dir,
                    task_path=entry["task_path"],
                    schema_name=self.schema.name,
                )
                for entry in self.entries
            ]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]
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
            arrays = augment_realtime_arrays(
                arrays=arrays,
                rng=rng,
                tracker_pos_noise_std=self.tracker_pos_noise_std,
                tracker_rot_noise_std=self.tracker_rot_noise_std,
                non_hip_tracker_dropout_prob=self.non_hip_tracker_dropout_prob,
                tracker_latency_max_frames=self.tracker_latency_max_frames,
                tracker_burst_dropout_prob=self.tracker_burst_dropout_prob,
                tracker_outlier_prob=self.tracker_outlier_prob,
                history_pose_noise_std=self.history_pose_noise_std,
                history_yaw_noise_std=self.history_yaw_noise_std,
                root_yaw_ref_noise_std=self.root_yaw_ref_noise_std,
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
                salt=f"history_condition:e{random_context.epoch}:w{random_context.worker_id}:a{random_context.access_index}",
            )
            conditioned = self.apply_predicted_history_cache(
                conditioned=conditioned,
                entry=entry,
                rng=rng,
            )
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
            "prev_joints_world": torch.from_numpy(arrays["joints_world"][REALTIME_POSE_TARGET_START - 1]).float(),
            "target_root_pos_world": torch.from_numpy(arrays["root_pos_world"][REALTIME_POSE_TARGET_START]).float(),
            "prev_root_pos_world": torch.from_numpy(arrays["root_pos_world"][REALTIME_POSE_TARGET_START - 1]).float(),
            "prev_root_yaw": torch.tensor(float(arrays["root_yaw"][REALTIME_POSE_TARGET_START - 1])).float(),
            "target_root_yaw": torch.tensor(float(arrays["root_yaw"][REALTIME_POSE_TARGET_START])).float(),
            "target_tracker_pos_ref": torch.from_numpy(
                raw_features[REALTIME_POSE_TARGET_START, self.schema.tracker_pos_slice()].reshape(TRACKER_COUNT, 3)
            ).float(),
            "target_tracker_rot_ref_6d": torch.from_numpy(
                raw_features[REALTIME_POSE_TARGET_START, self.schema.tracker_rot_slice()].reshape(TRACKER_COUNT, 6)
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
        if self.schema.supports_root_motion:
            item["target_root_delta_xz_ref"] = torch.from_numpy(
                arrays["root_delta_xz_ref"][REALTIME_POSE_TARGET_START]
            ).float()
            item["target_root_height"] = torch.tensor(float(arrays["root_height"][REALTIME_POSE_TARGET_START, 0])).float()
        if self.schema.supports_contact:
            item["target_foot_contact"] = torch.from_numpy(arrays["foot_contact"][REALTIME_POSE_TARGET_START]).float()
        return item

    def load_task(self, index: int, entry: dict) -> dict[str, np.ndarray]:
        if self.task_cache is not None:
            return self.task_cache[index]
        return load_materialized_task_npz(
            manifest_dir=self.manifest_dir,
            task_path=entry["task_path"],
            schema_name=self.schema.name,
        )

    def apply_predicted_history_cache(
        self,
        conditioned: np.ndarray,
        entry: dict,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        用离线 rollout 预测 history 替换 GT history。

        cache 文件以 `task_id.npz` 命名，只接受归一化空间的
        `predicted_features_normalized`。这里故意不自动兼容 raw/features 字段，
        避免把不同特征空间混进 `conditioned_x`。
        """

        if self.predicted_history_cache_dir is None or self.predicted_history_prob <= 0:
            return conditioned
        if rng.random() >= self.predicted_history_prob:
            return conditioned
        task_id = entry.get("task_id")
        if not task_id:
            raise KeyError("predicted history cache 需要 manifest entry 包含 task_id。")
        cache_path = self.predicted_history_cache_dir / f"{task_id}.npz"
        if not cache_path.exists():
            raise FileNotFoundError(f"predicted history cache 不存在：{cache_path}")
        with np.load(cache_path, allow_pickle=False) as data:
            if "predicted_features_normalized" not in data.files:
                raise KeyError(f"{cache_path} 缺少 predicted_features_normalized 字段。")
            if "schema_name" not in data.files or str(np.asarray(data["schema_name"]).item()) != self.schema.name:
                raise ValueError(f"{cache_path} schema_name 必须为 {self.schema.name}。")
            if "feature_space" not in data.files or str(np.asarray(data["feature_space"]).item()) != "normalized":
                raise ValueError(f"{cache_path} feature_space 必须为 normalized。")
            cached = np.asarray(data["predicted_features_normalized"], dtype=np.float32)
        if cached.ndim == 3:
            cached = cached[0]
        if cached.shape != conditioned.shape:
            raise ValueError(f"{cache_path} cache shape 应为 {conditioned.shape}，实际为 {cached.shape}")
        conditioned[:REALTIME_POSE_TARGET_START, self.schema.target_slice()] = cached[
            :REALTIME_POSE_TARGET_START,
            self.schema.target_slice(),
        ]
        return conditioned

    def set_epoch(self, epoch: int) -> None:
        """训练循环在 epoch 开头调用，使动态 mask 和增强能随 epoch 可复现地变化。"""

        self.epoch = int(epoch)
        self.access_index = 0

    def next_random_context(self) -> RandomContext:
        """
        返回当前 Dataset 实例内的随机上下文。

        DataLoader 多 worker 会复制 Dataset 实例，因此把 worker_id 放进随机种子里，
        可以避免每个 worker 生成完全相同的动态遮盖和增强序列。
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
        result["sensor_valid"] = repeat_pattern_sensor_valid(pattern, seq_len=self.seq_len)
        return result, pattern.category

    def dynamic_mask_category(self, entry: dict, index: int, random_context: RandomContext) -> str:
        # 动态遮盖按“洗牌后的类别轮转”采样，既覆盖所有类别，又能由 seed 复现实验。
        categories = np.asarray(self.tracker_mask_categories, dtype=object)
        cycle_index = random_context.access_index // len(categories)
        position = random_context.access_index % len(categories)
        rng = self.stable_rng(
            entry=entry,
            index=index,
            salt=f"dynamic_category:e{random_context.epoch}:w{random_context.worker_id}:c{cycle_index}",
        )
        rng.shuffle(categories)
        return str(categories[position])

    def fixed_mask_category(self, entry: dict, index: int) -> str:
        categories = self.tracker_mask_categories
        digest = self.stable_mask_digest(entry=entry, index=index, salt="category")
        category_index = int(digest[:8], 16) % len(categories)
        return categories[category_index]

    def fixed_mask_rng(self, entry: dict, index: int, category: str) -> np.random.Generator:
        return self.stable_rng(entry=entry, index=index, salt=category)

    def stable_rng(self, entry: dict, index: int, salt: str) -> np.random.Generator:
        digest = self.stable_mask_digest(entry=entry, index=index, salt=salt)
        seed = int(digest[:16], 16) % (2**32)
        return np.random.default_rng(seed)

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
    candidates = [data_dir / split / "manifest.jsonl", data_dir / "manifest.jsonl"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"找不到 realtime_pose manifest，已尝试：{', '.join(str(path) for path in candidates)}")


def read_task_manifest(manifest_path: Path) -> list[dict]:
    entries = []
    with manifest_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def load_materialized_task_npz(
    manifest_dir: Path,
    task_path: str,
    schema_name: str = REALTIME_POSE_SCHEMA_NAME,
) -> dict[str, np.ndarray]:
    path = manifest_dir / task_path
    if not path.exists():
        raise FileNotFoundError(f"realtime_pose task 文件不存在：{path}")
    with np.load(path, allow_pickle=False) as data:
        task = {key: data[key].copy() for key in data.files}
    required = {
        "body_pose_parent_6d",
        "root_pos_world",
        "root_yaw",
        "root_yaw_delta_sincos",
        "tracker_pos_world",
        "tracker_rot_world_6d",
        "joints_world",
        "joint_offsets_parent",
        "sensor_valid",
        "inpaint_mask",
        "start_frame",
        "valid_length",
        "source_frames",
        "seq_len",
    }
    schema = get_schema_spec(schema_name)
    if schema.supports_root_motion:
        required.update({"root_delta_xz_ref", "root_height"})
    if schema.supports_contact:
        required.add("foot_contact")
    missing = sorted(required.difference(task))
    if missing:
        raise KeyError(f"{path} 缺少 {schema.name} 字段：{missing}")
    return task


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

    arrays = {
        "body_pose_parent_6d": array_shape(task["body_pose_parent_6d"], (seq_len, BODY_POSE_DIM), "body_pose_parent_6d").astype(np.float32),
        "root_pos_world": array_shape(task["root_pos_world"], (seq_len, 3), "root_pos_world").astype(np.float32),
        "root_yaw": array_shape(task["root_yaw"], (seq_len,), "root_yaw").astype(np.float32),
        "root_yaw_delta_sincos": array_shape(task["root_yaw_delta_sincos"], (seq_len, ROOT_YAW_DELTA_DIM), "root_yaw_delta_sincos").astype(np.float32),
        "tracker_pos_world": array_shape(task["tracker_pos_world"], (seq_len, TRACKER_COUNT, 3), "tracker_pos_world").astype(np.float32),
        "tracker_rot_world_6d": array_shape(task["tracker_rot_world_6d"], (seq_len, TRACKER_COUNT, 6), "tracker_rot_world_6d").astype(np.float32),
        "joints_world": array_shape(task["joints_world"], (seq_len, 24, 3), "joints_world").astype(np.float32),
        "joint_offsets_parent": array_shape(task["joint_offsets_parent"], (24, 3), "joint_offsets_parent").astype(np.float32),
        "sensor_valid": array_shape(task["sensor_valid"], (seq_len, SENSOR_VALID_DIM), "sensor_valid").astype(bool),
        "inpaint_mask": array_shape(task["inpaint_mask"], (seq_len, schema.feature_dim), "inpaint_mask").astype(bool),
    }
    if schema.supports_root_motion:
        arrays["root_delta_xz_ref"] = array_shape(
            task["root_delta_xz_ref"],
            (seq_len, ROOT_DELTA_XZ_DIM),
            "root_delta_xz_ref",
        ).astype(np.float32)
        arrays["root_height"] = array_shape(task["root_height"], (seq_len, ROOT_HEIGHT_DIM), "root_height").astype(np.float32)
    if schema.supports_contact:
        arrays["foot_contact"] = array_shape(task["foot_contact"], (seq_len, FOOT_CONTACT_DIM), "foot_contact").astype(np.float32)
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
    seq_len = arrays["body_pose_parent_6d"].shape[0]
    features = np.zeros((seq_len, schema.feature_dim), dtype=np.float32)
    features[:, schema.body_pose_slice()] = arrays["body_pose_parent_6d"]
    features[:, schema.root_yaw_delta_slice()] = arrays["root_yaw_delta_sincos"]
    if schema.supports_root_motion:
        features[:, schema.root_delta_xz_slice()] = arrays["root_delta_xz_ref"]
        features[:, schema.root_height_slice()] = arrays["root_height"]
    if schema.supports_contact:
        features[:, schema.foot_contact_slice()] = arrays["foot_contact"]
    features[:, schema.tracker_pos_slice()] = encode_tracker_pos_ref(arrays).reshape(seq_len, -1)
    features[:, schema.tracker_rot_slice()] = encode_tracker_rot_ref(arrays).reshape(seq_len, -1)
    features[:, schema.sensor_valid_slice()] = arrays["sensor_valid"].astype(np.float32)
    zero_missing_tracker_channels(features=features, sensor_valid=arrays["sensor_valid"], schema_name=schema.name)
    return features


def encode_tracker_pos_ref(arrays: dict[str, np.ndarray]) -> np.ndarray:
    tracker_world = arrays["tracker_pos_world"].astype(np.float64)
    roots = arrays["root_pos_world"].astype(np.float64)
    root_yaw = arrays["root_yaw"].astype(np.float64)
    ref_yaw = np.concatenate([root_yaw[:1], root_yaw[:-1]], axis=0)
    if "root_yaw_ref_noise" in arrays:
        ref_yaw = ref_yaw + np.asarray(arrays["root_yaw_ref_noise"], dtype=np.float64)
    result = np.zeros_like(tracker_world)
    for frame_index in range(tracker_world.shape[0]):
        rotation = make_yaw_rotation_np(np.asarray([ref_yaw[frame_index]], dtype=np.float64))[0]
        result[frame_index] = (tracker_world[frame_index] - roots[frame_index][None]) @ rotation
    return result.astype(np.float32)


def encode_tracker_rot_ref(arrays: dict[str, np.ndarray]) -> np.ndarray:
    tracker_world_rot = rotation_6d_to_matrix_np(arrays["tracker_rot_world_6d"])
    root_yaw = arrays["root_yaw"].astype(np.float64)
    ref_yaw = np.concatenate([root_yaw[:1], root_yaw[:-1]], axis=0)
    if "root_yaw_ref_noise" in arrays:
        ref_yaw = ref_yaw + np.asarray(arrays["root_yaw_ref_noise"], dtype=np.float64)
    result = np.zeros_like(tracker_world_rot)
    for frame_index in range(tracker_world_rot.shape[0]):
        rotation_inv = make_yaw_rotation_np(np.asarray([ref_yaw[frame_index]], dtype=np.float64))[0].T
        result[frame_index] = rotation_inv[None] @ tracker_world_rot[frame_index]
    return rotation_6d_forward_up_np(result).astype(np.float32)


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
    non_hip_tracker_dropout_prob: float = 0.0,
    tracker_latency_max_frames: int = 0,
    tracker_burst_dropout_prob: float = 0.0,
    tracker_outlier_prob: float = 0.0,
    history_pose_noise_std: float = 0.0,
    history_yaw_noise_std: float = 0.0,
    root_yaw_ref_noise_std: float = 0.0,
) -> dict[str, np.ndarray]:
    """训练增强只改 tracker 条件；history pose/yaw 污染在 `conditioned_x` 上单独做。"""

    result = {key: value.copy() for key, value in arrays.items()}
    sensor_valid = result["sensor_valid"].copy()
    dropout_prob = max(float(non_hip_tracker_dropout_prob), float(tracker_burst_dropout_prob))
    if dropout_prob > 0:
        sensor_valid = dropout_non_hip_trackers(
            sensor_valid=sensor_valid,
            rng=rng,
            dropout_prob=dropout_prob,
        )
        result["sensor_valid"] = sensor_valid

    if tracker_latency_max_frames > 0:
        delay = int(rng.integers(0, int(tracker_latency_max_frames) + 1))
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


def dropout_non_hip_trackers(
    sensor_valid: np.ndarray,
    rng: np.random.Generator,
    dropout_prob: float,
) -> np.ndarray:
    valid = np.asarray(sensor_valid, dtype=bool).copy()
    valid[:, HIP_TRACKER_INDEX] = True
    non_hip_indices = [index for index in range(TRACKER_COUNT) if index != HIP_TRACKER_INDEX]
    for frame_index in range(valid.shape[0]):
        original = valid[frame_index].copy()
        for _attempt in range(100):
            candidate = original.copy()
            for tracker_index in non_hip_indices:
                if candidate[tracker_index] and rng.random() < dropout_prob:
                    candidate[tracker_index] = False
            candidate[HIP_TRACKER_INDEX] = True
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
