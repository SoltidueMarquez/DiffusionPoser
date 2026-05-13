import os
from importlib import import_module

class TrainPlatform:
    def __init__(self, save_dir):
        pass

    def report_scalar(self, name, value, iteration, group_name=None):
        pass

    def report_args(self, args, name):
        pass

    def close(self):
        pass


class ClearmlPlatform(TrainPlatform):
    def __init__(self, save_dir):
        # ClearML 是可选日志后端：默认训练入口不会注册它，因此这里用动态导入避免
        # Pylance 在未安装 clearml 的复现实验环境中报 “Import could not be resolved”。
        try:
            Task = import_module("clearml").Task
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ClearmlPlatform 需要安装可选依赖 clearml；"
                "如需使用该日志平台，请先运行 `pip install clearml`。"
            ) from exc
        path, name = os.path.split(save_dir)
        self.task = Task.init(project_name='motion_diffusion',
                              task_name=name,
                              output_uri=path)
        self.logger = self.task.get_logger()

    def report_scalar(self, name, value, iteration, group_name):
        self.logger.report_scalar(title=group_name, series=name, iteration=iteration, value=value)

    def report_args(self, args, name):
        self.task.connect(args, name=name)

    def close(self):
        self.task.close()


class TensorboardPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from torch.utils.tensorboard import SummaryWriter
        # from tensorboardX import SummaryWriter
        self.writer = SummaryWriter(log_dir=save_dir)

    def report_scalar(self, name, value, iteration, group_name=None):
        self.writer.add_scalar(f'{group_name}/{name}', value, iteration)

    def close(self):
        self.writer.close()


class NoPlatform(TrainPlatform):
    def __init__(self, save_dir):
        pass


