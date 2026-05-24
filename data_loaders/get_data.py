from torch.utils.data import DataLoader

from data_loaders.realtime_pose_dataset import RealtimePoseTaskDataset
from data_loaders.sensor_masking import DEFAULT_REALTIME_POSE_SCHEMA_NAME, get_schema_spec


def get_dataset_loader(
    data_dir: str,
    batch_size: int,
    input_feats: int,
    seq_len: int,
    split: str = "train",
    normalizer_dir: str | None = None,
    normalize_input: bool = True,
    preload_data: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    folder_path: str | None = None,
    tracker_pos_noise_std: float = 0.0,
    tracker_rot_noise_std: float = 0.0,
    non_hip_tracker_dropout_prob: float = 0.0,
    history_pose_noise_std: float = 0.0,
    history_yaw_noise_std: float = 0.0,
    root_yaw_ref_noise_std: float = 0.0,
    tracker_mask_policy: str = "auto",
    tracker_mask_seed: int = 0,
    tracker_mask_fill: str = "zero",
    tracker_mask_categories: list[str] | tuple[str, ...] | None = None,
    schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    history_pose_dropout_prob: float = 0.0,
    history_pose_replace_prob: float = 0.0,
    history_yaw_replace_prob: float = 0.0,
    history_root_yaw_drift_std: float = 0.0,
    tracker_latency_max_frames: int = 0,
    tracker_burst_dropout_prob: float = 0.0,
    tracker_outlier_prob: float = 0.0,
    predicted_history_cache_dir: str | None = None,
    predicted_history_prob: float = 0.0,
):
    """返回 realtime_pose_v1 训练 / 测试 DataLoader。"""

    if not data_dir:
        raise ValueError("请提供 --data_dir，指向 data_loaders.generate_realtime_pose_tasks 生成的任务目录。")
    schema = get_schema_spec(schema_name)
    if int(input_feats) != schema.feature_dim:
        raise ValueError(f"{schema.name} 需要 input_feats={schema.feature_dim}，当前为 {input_feats}")

    dataset = RealtimePoseTaskDataset(
        data_dir=data_dir,
        split=split,
        seq_len=seq_len,
        normalizer_dir=normalizer_dir,
        normalize_input=normalize_input,
        preload_data=preload_data,
        folder_path=folder_path,
        tracker_pos_noise_std=tracker_pos_noise_std,
        tracker_rot_noise_std=tracker_rot_noise_std,
        non_hip_tracker_dropout_prob=non_hip_tracker_dropout_prob,
        history_pose_noise_std=history_pose_noise_std,
        history_yaw_noise_std=history_yaw_noise_std,
        root_yaw_ref_noise_std=root_yaw_ref_noise_std,
        tracker_mask_policy=tracker_mask_policy,
        tracker_mask_seed=tracker_mask_seed,
        tracker_mask_fill=tracker_mask_fill,
        tracker_mask_categories=tracker_mask_categories,
        schema_name=schema.name,
        history_pose_dropout_prob=history_pose_dropout_prob,
        history_pose_replace_prob=history_pose_replace_prob,
        history_yaw_replace_prob=history_yaw_replace_prob,
        history_root_yaw_drift_std=history_root_yaw_drift_std,
        tracker_latency_max_frames=tracker_latency_max_frames,
        tracker_burst_dropout_prob=tracker_burst_dropout_prob,
        tracker_outlier_prob=tracker_outlier_prob,
        predicted_history_cache_dir=predicted_history_cache_dir,
        predicted_history_prob=predicted_history_prob,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": "train" in split,
        "drop_last": "train" in split,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        # Dataset.set_epoch 会在每个 epoch 更新动态遮盖/增强种子；不常驻 worker，
        # 让下一轮 DataLoader 迭代重新接收主进程里的 epoch 状态。
        loader_kwargs["persistent_workers"] = False
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)
