from torch.utils.data import DataLoader

from data_loaders.realtime_pose_dataset import RealtimePoseBatchSampler, RealtimePoseTaskDataset


def get_dataset_loader(
    data_dir: str,
    batch_size: int,
    input_feats: int,
    seq_len: int,
    split: str = "train",
    normalizer_dir: str | None = None,
    normalize_input: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    folder_path: str | None = None,
    enable_rollout: bool = False,
    rollout_steps: int = 4,
    rollout_prob: float = 0.5,
    cold_start_prob: float = 0.1,
    scenario_weights: list[float] | tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2),
    seed: int = 10,
):
    """返回 mmap task store 的训练/评估 DataLoader。"""

    del input_feats
    if not data_dir:
        raise ValueError("请提供 --data_dir，指向 task store 目录。")
    dataset = RealtimePoseTaskDataset(
        data_dir=data_dir,
        split=split,
        seq_len=seq_len,
        normalizer_dir=normalizer_dir,
        normalize_input=normalize_input,
        folder_path=folder_path,
    )
    worker_kwargs = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory) or int(num_workers) > 0,
    }
    if int(num_workers) > 0:
        worker_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})

    is_train_split = "train" in str(split).lower()
    # Dataset 的整数索引只表示 task_index，会固定读取 config_index=0。
    # 所有 split 都通过 TaskRequest 取样，验证集才能真正遵守 scenario_weights。
    sampler = RealtimePoseBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        seed=seed,
        scenario_weights=scenario_weights,
        rollout_steps=rollout_steps if is_train_split and enable_rollout else 1,
        rollout_prob=rollout_prob if is_train_split and enable_rollout else 0.0,
        cold_start_prob=cold_start_prob if is_train_split else 0.0,
        shuffle=is_train_split,
        drop_last=is_train_split,
    )
    return DataLoader(dataset, batch_sampler=sampler, **worker_kwargs)
