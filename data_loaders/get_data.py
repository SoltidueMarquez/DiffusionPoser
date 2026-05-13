from torch.utils.data import DataLoader

from data_loaders.smoke_dataset import RandomSparseSensorDataset


def get_dataset_loader(
    data_dir: str,
    batch_size: int,
    input_feats: int,
    seq_len: int,
    mask_ratio: float,
    num_batches: int,
    split: str = "train",
):
    """
    返回训练 DataLoader。

    当前版本先提供 smoke dataset，用来验证扩散框架、模型和训练 loop 已经接通。
    后续接入 DiffusionPoser 论文数据时，只需要保持 batch 字段约定不变：
    `x: [B, C, T]`、`valid_frame_mask: [B, T]`、`sensor_mask: [B, C, T]`。
    """

    if data_dir:
        raise NotImplementedError("真实 DiffusionPoser 数据集尚未接入；请先留空 data_dir 运行 smoke training。")

    dataset = RandomSparseSensorDataset(
        num_samples=batch_size * num_batches,
        input_feats=input_feats,
        seq_len=seq_len,
        mask_ratio=mask_ratio,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
