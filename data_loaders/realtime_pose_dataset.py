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
    BODY_POSE_START,
    HIP_TRACKER_INDEX,
    REALTIME_POSE_INPUT_DIM,
    REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_DIM,
    REALTIME_POSE_TARGET_LENGTH,
    REALTIME_POSE_TARGET_START,
    ROOT_YAW_DELTA_DIM,
    ROOT_YAW_DELTA_START,
    SENSOR_VALID_DIM,
    SENSOR_VALID_START,
    TASK_FORMAT_REALTIME_POSE_V1,
    TRACKER_COUNT,
    TRACKER_MASK_FILL_MODES,
    TRACKER_MASK_FILL_ZERO,
    TRACKER_MASK_POLICIES,
    TRACKER_MASK_POLICY_AUTO,
    TRACKER_MASK_POLICY_DYNAMIC_CATEGORIES,
    TRACKER_MASK_POLICY_FIXED_CATEGORIES,
    TRACKER_MASK_POLICY_TASK,
    TRACKER_POS_DIM,
    TRACKER_POS_REF_START,
    TRACKER_ROT_DIM,
    TRACKER_ROT_REF_START,
    make_tracker_pattern,
    normalize_tracker_pattern_categories,
    repeat_pattern_sensor_valid,
    validate_realtime_seq_len,
    validate_realtime_target,
    validate_sensor_valid,
)
from utils.normalizer import RealtimePoseNormalizer


@dataclass(frozen=True)
class RandomContext:
    worker_id: int
    access_index: int
    epoch: int = 0


class RealtimePoseTaskDataset(Dataset):
    """读取 `realtime_pose_v1` materialized task 并输出 `[C,T]` 训练样本。"""

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
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = int(seq_len)
        validate_realtime_seq_len(self.seq_len)
        self.normalize_input = bool(normalize_input)
        self.preload_data = bool(preload_data)
        self.is_train_split = "train" in str(split).lower()
        self.tracker_pos_noise_std = float(tracker_pos_noise_std)
        self.tracker_rot_noise_std = float(tracker_rot_noise_std)
        self.non_hip_tracker_dropout_prob = float(non_hip_tracker_dropout_prob)
        self.history_pose_noise_std = float(history_pose_noise_std)
        self.history_yaw_noise_std = float(history_yaw_noise_std)
        self.root_yaw_ref_noise_std = float(root_yaw_ref_noise_std)
        self.tracker_mask_policy = self.resolve_tracker_mask_policy(tracker_mask_policy)
        self.tracker_mask_seed = int(tracker_mask_seed)
        self.tracker_mask_fill = str(tracker_mask_fill)
        if self.tracker_mask_fill not in TRACKER_MASK_FILL_MODES:
            raise ValueError(f"tracker_mask_fill 目前只支持 {TRACKER_MASK_FILL_MODES}，实际为 {tracker_mask_fill}")
        self.tracker_mask_categories = normalize_tracker_pattern_categories(tracker_mask_categories)
        self.epoch = 0
        self.access_index = 0
        self.normalizer = create_normalizer(normalizer_dir=normalizer_dir, normalize_input=self.normalize_input)

        self.manifest_path = find_manifest_path(data_dir=self.data_dir, split=split)
        self.manifest_dir = self.manifest_path.parent
        self.entries = read_task_manifest(self.manifest_path)
        if folder_path:
            self.entries = filter_entries_by_folder_path(self.entries, folder_path=folder_path)
        if not self.entries:
            raise RuntimeError(f"{self.manifest_path} 中没有可用 realtime_pose_v1 task。")

        for entry in self.entries:
            if str(entry.get("schema_name", "")) != REALTIME_POSE_SCHEMA_NAME:
                raise ValueError(f"任务 {entry.get('task_id')} 不是 {REALTIME_POSE_SCHEMA_NAME}。")
            if str(entry.get("task_format", "")) != TASK_FORMAT_REALTIME_POSE_V1:
                raise ValueError(f"任务 {entry.get('task_id')} 的 task_format 不匹配。")
            if int(entry.get("seq_len", -1)) != self.seq_len:
                raise ValueError(f"任务 {entry.get('task_id')} 的 seq_len 不等于 {self.seq_len}。")
            validate_realtime_target(int(entry.get("target_start", -1)), int(entry.get("target_length", -1)))

        self.task_cache = None
        if self.preload_data:
            self.task_cache = [
                load_materialized_task_npz(manifest_dir=self.manifest_dir, task_path=entry["task_path"])
                for entry in self.entries
            ]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]
        random_context = self.next_random_context()
        task = self.load_task(index=index, entry=entry)
        arrays = load_realtime_task_arrays(task=task, seq_len=self.seq_len)
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
                history_pose_noise_std=self.history_pose_noise_std,
                history_yaw_noise_std=self.history_yaw_noise_std,
                root_yaw_ref_noise_std=self.root_yaw_ref_noise_std,
            )

        sensor_valid = arrays["sensor_valid"]
        features = encode_realtime_pose_features(arrays)
        if self.normalizer is not None:
            features = self.normalizer.normalize(features)
            zero_missing_tracker_channels(features=features, sensor_valid=sensor_valid)

        conditioned = features.copy()
        conditioned[REALTIME_POSE_TARGET_START, BODY_POSE_START:REALTIME_POSE_TARGET_DIM] = 0.0
        inpaint_mask = np.asarray(arrays["inpaint_mask"], dtype=bool)
        valid_frame_mask = np.ones(self.seq_len, dtype=bool)

        return {
            "x": torch.from_numpy(features.T).float(),
            "conditioned_x": torch.from_numpy(conditioned.T).float(),
            "valid_frame_mask": torch.from_numpy(valid_frame_mask).bool(),
            "attention_mask": torch.from_numpy(valid_frame_mask).bool(),
            "sensor_valid": torch.from_numpy(sensor_valid.T).bool(),
            "inpaint_mask": torch.from_numpy(inpaint_mask.T).bool(),
            "target_joints_world": torch.from_numpy(arrays["joints_world"][REALTIME_POSE_TARGET_START]).float(),
            "prev_joints_world": torch.from_numpy(arrays["joints_world"][REALTIME_POSE_TARGET_START - 1]).float(),
            "target_root_pos_world": torch.from_numpy(arrays["root_pos_world"][REALTIME_POSE_TARGET_START]).float(),
            "prev_root_yaw": torch.tensor(float(arrays["root_yaw"][REALTIME_POSE_TARGET_START - 1])).float(),
            "target_root_yaw": torch.tensor(float(arrays["root_yaw"][REALTIME_POSE_TARGET_START])).float(),
            "joint_offsets_parent": torch.from_numpy(arrays["joint_offsets_parent"]).float(),
            "length": self.seq_len,
            "keyid": entry.get("task_id", ""),
            "source_path": entry.get("source_path", ""),
            "task_mode": entry.get("task_mode", ""),
            "schema_name": entry.get("schema_name", ""),
            "target_start": REALTIME_POSE_TARGET_START,
            "target_length": REALTIME_POSE_TARGET_LENGTH,
            "tracker_pattern": applied_tracker_pattern,
            "tracker_mask_policy": self.tracker_mask_policy,
        }

    def load_task(self, index: int, entry: dict) -> dict[str, np.ndarray]:
        if self.task_cache is not None:
            return self.task_cache[index]
        return load_materialized_task_npz(manifest_dir=self.manifest_dir, task_path=entry["task_path"])

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


def create_normalizer(normalizer_dir: str | Path | None, normalize_input: bool) -> RealtimePoseNormalizer | None:
    if not normalize_input:
        return None
    if normalizer_dir is None or str(normalizer_dir).strip() == "":
        raise ValueError("开启 normalize_input 时必须提供 normalizer_dir。")
    return RealtimePoseNormalizer(base_dir=normalizer_dir)


def find_manifest_path(data_dir: Path, split: str) -> Path:
    candidates = [data_dir / split / "manifest.jsonl", data_dir / "manifest.jsonl"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"找不到 realtime_pose_v1 manifest，已尝试：{', '.join(str(path) for path in candidates)}")


def read_task_manifest(manifest_path: Path) -> list[dict]:
    entries = []
    with manifest_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def load_materialized_task_npz(manifest_dir: Path, task_path: str) -> dict[str, np.ndarray]:
    path = manifest_dir / task_path
    if not path.exists():
        raise FileNotFoundError(f"realtime_pose_v1 task 文件不存在：{path}")
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
    missing = sorted(required.difference(task))
    if missing:
        raise KeyError(f"{path} 缺少 realtime_pose_v1 字段：{missing}")
    return task


def load_realtime_task_arrays(task: dict[str, np.ndarray], seq_len: int) -> dict[str, np.ndarray]:
    validate_realtime_seq_len(seq_len)
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
        "inpaint_mask": array_shape(task["inpaint_mask"], (seq_len, REALTIME_POSE_INPUT_DIM), "inpaint_mask").astype(bool),
    }
    validate_sensor_valid(arrays["sensor_valid"])
    expected_mask = np.zeros((seq_len, REALTIME_POSE_INPUT_DIM), dtype=bool)
    expected_mask[REALTIME_POSE_TARGET_START, BODY_POSE_START:REALTIME_POSE_TARGET_DIM] = True
    if not np.array_equal(arrays["inpaint_mask"], expected_mask):
        raise ValueError("inpaint_mask 必须只覆盖第 61 帧的 body_pose + root_yaw_delta_sincos。")
    return arrays


def encode_realtime_pose_features(arrays: dict[str, np.ndarray]) -> np.ndarray:
    seq_len = arrays["body_pose_parent_6d"].shape[0]
    features = np.zeros((seq_len, REALTIME_POSE_INPUT_DIM), dtype=np.float32)
    features[:, BODY_POSE_START:BODY_POSE_START + BODY_POSE_DIM] = arrays["body_pose_parent_6d"]
    features[:, ROOT_YAW_DELTA_START:ROOT_YAW_DELTA_START + ROOT_YAW_DELTA_DIM] = arrays["root_yaw_delta_sincos"]
    features[:, TRACKER_POS_REF_START:TRACKER_POS_REF_START + TRACKER_POS_DIM] = encode_tracker_pos_ref(arrays).reshape(seq_len, -1)
    features[:, TRACKER_ROT_REF_START:TRACKER_ROT_REF_START + TRACKER_ROT_DIM] = encode_tracker_rot_ref(arrays).reshape(seq_len, -1)
    features[:, SENSOR_VALID_START:SENSOR_VALID_START + SENSOR_VALID_DIM] = arrays["sensor_valid"].astype(np.float32)
    zero_missing_tracker_channels(features=features, sensor_valid=arrays["sensor_valid"])
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


def zero_missing_tracker_channels(features: np.ndarray, sensor_valid: np.ndarray) -> None:
    valid = np.asarray(sensor_valid, dtype=bool)
    for sensor_index in range(TRACKER_COUNT):
        missing = ~valid[:, sensor_index]
        if not missing.any():
            continue
        pos_start = TRACKER_POS_REF_START + sensor_index * 3
        rot_start = TRACKER_ROT_REF_START + sensor_index * 6
        features[missing, pos_start:pos_start + 3] = 0.0
        features[missing, rot_start:rot_start + 6] = 0.0


def augment_realtime_arrays(
    arrays: dict[str, np.ndarray],
    rng: np.random.Generator,
    tracker_pos_noise_std: float = 0.0,
    tracker_rot_noise_std: float = 0.0,
    non_hip_tracker_dropout_prob: float = 0.0,
    history_pose_noise_std: float = 0.0,
    history_yaw_noise_std: float = 0.0,
    root_yaw_ref_noise_std: float = 0.0,
) -> dict[str, np.ndarray]:
    """训练增强只改条件可见信息，hip 永远有效且每帧至少 3 个 tracker。"""

    result = {key: value.copy() for key, value in arrays.items()}
    sensor_valid = result["sensor_valid"].copy()
    if non_hip_tracker_dropout_prob > 0:
        sensor_valid = dropout_non_hip_trackers(
            sensor_valid=sensor_valid,
            rng=rng,
            dropout_prob=float(non_hip_tracker_dropout_prob),
        )
        result["sensor_valid"] = sensor_valid

    if tracker_pos_noise_std > 0:
        noise = rng.normal(0.0, tracker_pos_noise_std, size=result["tracker_pos_world"].shape).astype(np.float32)
        result["tracker_pos_world"] = result["tracker_pos_world"] + noise * sensor_valid[:, :, None].astype(np.float32)

    if tracker_rot_noise_std > 0:
        noise = rng.normal(0.0, tracker_rot_noise_std, size=result["tracker_rot_world_6d"].shape).astype(np.float32)
        result["tracker_rot_world_6d"] = result["tracker_rot_world_6d"] + noise * sensor_valid[:, :, None].astype(np.float32)

    history = slice(0, REALTIME_POSE_TARGET_START)
    if history_pose_noise_std > 0:
        result["body_pose_parent_6d"][history] += rng.normal(
            0.0,
            history_pose_noise_std,
            size=result["body_pose_parent_6d"][history].shape,
        ).astype(np.float32)

    if history_yaw_noise_std > 0:
        yaw_noise = rng.normal(
            0.0,
            history_yaw_noise_std,
            size=result["root_yaw_delta_sincos"][history].shape,
        ).astype(np.float32)
        result["root_yaw_delta_sincos"][history] += yaw_noise
        norms = np.linalg.norm(result["root_yaw_delta_sincos"], axis=-1, keepdims=True)
        result["root_yaw_delta_sincos"] = result["root_yaw_delta_sincos"] / np.maximum(norms, 1e-8)

    if root_yaw_ref_noise_std > 0:
        result["root_yaw_ref_noise"] = rng.normal(
            0.0,
            root_yaw_ref_noise_std,
            size=(REALTIME_POSE_SEQ_LEN,),
        ).astype(np.float32)
    return result


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
