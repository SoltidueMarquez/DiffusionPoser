from torch.utils.data import DataLoader

from data_loaders.realtime_pose_dataset import RealtimePoseTaskDataset


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
    enable_rollout: bool = False,
    rollout_steps: int = 1,
):
    """返回当前 140 维 realtime pose 训练 / 测试 DataLoader。"""

    if not data_dir:
        raise ValueError("请提供 --data_dir，指向 data_loaders.generate_realtime_pose_tasks 生成的任务目录。")

    dataset = RealtimePoseTaskDataset(
        data_dir=data_dir,
        split=split,
        seq_len=seq_len,
        normalizer_dir=normalizer_dir,
        normalize_input=normalize_input,
        preload_data=preload_data,
        folder_path=folder_path,
        enable_rollout=enable_rollout,
        rollout_steps=rollout_steps,
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
