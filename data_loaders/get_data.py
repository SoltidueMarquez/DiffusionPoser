from torch.utils.data import DataLoader

from data_loaders.realtime_pose_dataset import (
    RealtimePoseBatchSampler,
    RealtimePoseTaskDataset,
)
from data_loaders.rpm_hand_dropout import RPM_HAND_DROPOUT_TRAIN_SEED


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
    seed: int = 10,
    rpm_hand_dropout: bool = False,
    rpm_hand_dropout_seed: int = RPM_HAND_DROPOUT_TRAIN_SEED,
):
    """返回单帧 DiT mmap Task Store 的 DataLoader。"""

    del input_feats
    if not data_dir:
        raise ValueError("请提供 --data_dir，指向 task store 目录。")
    dataset = RealtimePoseTaskDataset(
        data_dir=data_dir,
        split=split,
        seq_len=seq_len,
        normalizer_dir=normalizer_dir,
        normalize_input=normalize_input,
        rpm_hand_dropout=rpm_hand_dropout,
        rpm_hand_dropout_seed=rpm_hand_dropout_seed,
    )
    worker_kwargs = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory) or int(num_workers) > 0,
    }
    if int(num_workers) > 0:
        worker_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})

    is_train_split = "train" in str(split).lower()
    sampler = RealtimePoseBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        seed=seed,
        shuffle=is_train_split,
        drop_last=is_train_split,
    )
    return DataLoader(dataset, batch_sampler=sampler, **worker_kwargs)
