from __future__ import annotations

from pathlib import Path


def normalize_folder_token(folder_path: str | Path) -> str:
    """把 folder_path 规整成适合做前缀比较的 token。"""

    token = str(folder_path).strip().replace("\\", "/")
    while token.endswith("/"):
        token = token[:-1]
    return token


def filter_entries_by_folder_path(entries: list[dict], folder_path: str | Path) -> list[dict]:
    """
    按 `source_relative_path` 或 `source_path` 前缀过滤任务。

    这个 helper 独立出来有两个原因：
    1. 数据集类里只保留“怎么用”的逻辑，减少文件长度；
    2. 单元测试可以直接测路径过滤，而不必依赖 torch / Dataset。
    """

    folder = Path(folder_path)
    folder_token = normalize_folder_token(folder_path)
    filtered: list[dict] = []

    for entry in entries:
        source_relative_path = normalize_folder_token(entry.get("source_relative_path", ""))
        source_path = Path(entry.get("source_path", ""))

        if folder.is_absolute():
            # 如果给的是绝对路径，就按真实 source_path 所在目录匹配。
            try:
                if source_path.resolve().is_relative_to(folder.resolve()):
                    filtered.append(entry)
            except Exception:
                if str(source_path.resolve()).startswith(str(folder.resolve())):
                    filtered.append(entry)
            continue

        if not folder_token:
            # 空字符串表示“不额外过滤”，直接返回全部条目。
            filtered.append(entry)
            continue

        # 相对路径则按 source_relative_path 做前缀过滤，便于按子文件夹挑选测试样本。
        if source_relative_path == folder_token or source_relative_path.startswith(folder_token + "/"):
            filtered.append(entry)

    return filtered
