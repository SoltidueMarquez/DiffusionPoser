from __future__ import annotations

import os
from pathlib import Path


def filesystem_path(path: str | Path) -> str:
    """返回可用于 Windows 长路径文件 API 的绝对路径字符串。

    artifact store 比原始 dataset 根目录更深。任务文件本身已经接近传统
    ``MAX_PATH``，所以复制、哈希和删除必须走 extended-length 前缀，避免在
    迁移到一半时因单个深层样本失败。
    """

    resolved = str(Path(path).expanduser().resolve())
    if os.name != "nt" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved[2:]
    return "\\\\?\\" + resolved


def path_exists(path: str | Path) -> bool:
    return os.path.exists(filesystem_path(path))


def ensure_directory(path: str | Path) -> None:
    os.makedirs(filesystem_path(path), exist_ok=True)
