from torch.utils.data import DataLoader

from data_loaders.realtime_pose_dataset import RealtimePoseTaskDataset
from data_loaders.sensor_masking import REALTIME_POSE_INPUT_DIM


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
):
    """返回 realtime_pose_v1 训练 / 测试 DataLoader。"""

    if not data_dir:
        raise ValueError("请提供 --data_dir，指向 data_loaders.generate_realtime_pose_tasks 生成的任务目录。")
    if int(input_feats) != REALTIME_POSE_INPUT_DIM:
        raise ValueError(f"realtime_pose_v1 需要 input_feats={REALTIME_POSE_INPUT_DIM}，当前为 {input_feats}")

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
