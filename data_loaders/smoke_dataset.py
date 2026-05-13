import torch
from torch.utils.data import Dataset


class RandomSparseSensorDataset(Dataset):
    """用于打通训练链路的最小随机数据集，真实论文数据预处理接入后可直接替换。"""

    def __init__(self, num_samples: int, input_feats: int, seq_len: int, mask_ratio: float):
        self.num_samples = num_samples
        self.input_feats = input_feats
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        x = torch.randn(self.input_feats, self.seq_len)
        valid_frame_mask = torch.ones(self.seq_len, dtype=torch.bool)

        # sensor_mask=True 表示该特征位置是已观测条件；False 表示训练时需要补全。
        sensor_mask = torch.rand(self.input_feats, self.seq_len) > self.mask_ratio
        if sensor_mask.all():
            sensor_mask[0, 0] = False
        return {
            "x": x,
            "valid_frame_mask": valid_frame_mask,
            "sensor_mask": sensor_mask,
        }
