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
    rollout_steps: int = 1,
    rollout_prob: float = 0.0,
    scenario_weights: list[float] | tuple[float, ...] = (1, 1, 1, 1, 1),
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

    if "train" in str(split):
        sampler = RealtimePoseBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            seed=seed,
            scenario_weights=scenario_weights,
            rollout_steps=rollout_steps if enable_rollout else 1,
            rollout_prob=rollout_prob if enable_rollout else 0.0,
            shuffle=True,
            drop_last=True,
        )
        return DataLoader(dataset, batch_sampler=sampler, **worker_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **worker_kwargs,
    )
