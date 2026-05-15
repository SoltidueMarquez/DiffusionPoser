from torch.utils.data import DataLoader

from data_loaders.x277_dataset import X277MissingTaskDataset


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
):
    """
    返回训练 DataLoader。

    当前训练入口只接收离线生成的 X277 缺失任务，避免误用随机数据完成“假训练”。
    真实数据 batch 字段约定为：
    `x: [B, 283, T]`、`valid_frame_mask: [B, T]`、`inpaint_mask: [B, 283, T]`。
    """

    if not data_dir:
        raise ValueError("请提供 --data_dir，指向 data_loaders.generate_x277_missing_tasks 生成的任务目录。")
    if input_feats != 283:
        raise ValueError(f"真实 X277 缺失任务需要 input_feats=283，当前为 {input_feats}")

    dataset = X277MissingTaskDataset(
        data_dir=data_dir,
        split=split,
        seq_len=seq_len,
        normalizer_dir=normalizer_dir,
        normalize_input=normalize_input,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": "train" in split,
        "drop_last": "train" in split,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        # Windows 下 worker 启动成本较高，持久化 worker 可以避免每个 epoch 反复 spawn。
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(dataset, **loader_kwargs)
