import hashlib
import os
import tempfile
from importlib import import_module
from pathlib import Path

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
        if should_use_tensorboard_fallback(save_dir):
            fallback_dir = prepare_tensorboard_fallback_dir(save_dir)
            print(
                "[TensorboardPlatform] 当前 run 路径包含 Windows TensorBoard 不稳定因素，"
                f"改写到短路径：{fallback_dir}",
                flush=True,
            )
            self.writer = SummaryWriter(log_dir=str(fallback_dir))
            return
        try:
            self.writer = SummaryWriter(log_dir=save_dir)
        except OSError as exc:
            fallback_dir = prepare_tensorboard_fallback_dir(save_dir)
            print(
                "[TensorboardPlatform] 当前 run 路径无法直接创建 TensorBoard event 文件，"
                f"改写到短路径：{fallback_dir}。原始错误：{exc}",
                flush=True,
            )
            self.writer = SummaryWriter(log_dir=str(fallback_dir))

    def report_scalar(self, name, value, iteration, group_name=None):
        self.writer.add_scalar(f'{group_name}/{name}', value, iteration)

    def close(self):
        self.writer.close()


class NoPlatform(TrainPlatform):
    def __init__(self, save_dir):
        pass


def tensorboard_fallback_log_dir(save_dir) -> Path:
    """Windows 下 TensorBoard 对中文或超长路径不稳定，因此必要时使用短 ASCII 日志目录。"""

    save_path = Path(save_dir)
    digest = hashlib.sha1(str(save_path).encode("utf-8")).hexdigest()[:12]
    safe_name = "".join(char if char.isascii() and (char.isalnum() or char in "._-") else "_" for char in save_path.name)
    safe_name = safe_name.strip("._-") or "run"
    return Path(tempfile.gettempdir()) / "diffusionposer_tensorboard" / f"{safe_name}_{digest}"


def prepare_tensorboard_fallback_dir(save_dir) -> Path:
    fallback_dir = tensorboard_fallback_log_dir(save_dir)
    fallback_dir.mkdir(parents=True, exist_ok=True)
    (Path(save_dir) / "tensorboard_log_dir.txt").write_text(str(fallback_dir), encoding="utf-8")
    return fallback_dir


def should_use_tensorboard_fallback(save_dir) -> bool:
    """提前避开 TensorBoard 在 Windows 上对非 ASCII 和接近 MAX_PATH 的路径处理问题。"""

    if os.name != "nt":
        return False
    path_text = str(save_dir)
    if not path_text.isascii():
        return True
    estimated_event_path = str(Path(save_dir) / "events.out.tfevents.0000000000.hostname.000000.0")
    return len(estimated_event_path) >= 240


