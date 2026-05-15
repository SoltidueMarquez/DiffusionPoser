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
    preload_data: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    folder_path: str | None = None,
):
    """
    返回训练 / 测试 DataLoader。

    当前入口只接受离线生成的 X277 缺失任务，不再混用原始数据或在线随机遮挡，
    这样测试阶段看到的输入就和训练时的任务格式完全一致。
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
        preload_data=preload_data,
        folder_path=folder_path,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": "train" in split,
        "drop_last": "train" in split,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        # Windows 下 worker 频繁重启的开销比较大；开启 persistent_workers 可以明显减少 epoch 间抖动。
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(dataset, **loader_kwargs)
