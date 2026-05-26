from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def sanitize_run_label(value: str) -> str:
    """把实验名压成适合目录名的短标签，避免空格和中文符号影响脚本读取。"""

    label = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._-")
    return label or "run"


def timestamped_child_dir(root_dir: str | Path, label: str, run_id: str | None = None) -> Path:
    """在根目录下生成不会和已有目录冲突的时间戳子目录路径。"""

    root = Path(root_dir).resolve()
    timestamp = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = sanitize_run_label(label)
    candidate = root / f"{timestamp}_{safe_label}"
    suffix = 2
    while candidate.exists():
        candidate = root / f"{timestamp}_{safe_label}_{suffix:02d}"
        suffix += 1
    return candidate


def write_latest_pointer(root_dir: str | Path, kind: str, output_dir: str | Path, metadata: dict[str, Any]) -> None:
    """在根目录写 latest 指针，供 AI、脚本和人工快速找到最近一次产物。"""

    root = Path(root_dir).resolve()
    output = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": kind,
        "output_dir": str(output),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **metadata,
    }
    # kind=run 保持已有 latest_run.* 命名；tasks/normalizer 使用同样模式。
    (root / f"latest_{kind}.txt").write_text(str(output), encoding="utf-8")
    with (root / f"latest_{kind}.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, ensure_ascii=False)


def read_latest_pointer(root_dir: str | Path, kind: str) -> Path | None:
    """读取 latest 指针并返回仍然存在的实际产物目录。"""

    root = Path(root_dir).resolve()
    json_path = root / f"latest_{kind}.json"
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            path_text = payload.get("output_dir") or payload.get("save_dir")
            latest_dir = Path(str(path_text)).expanduser()
            if latest_dir.exists():
                return latest_dir
        except (OSError, json.JSONDecodeError):
            return None

    text_path = root / f"latest_{kind}.txt"
    if text_path.exists():
        try:
            latest_dir = Path(text_path.read_text(encoding="utf-8").strip()).expanduser()
        except OSError:
            return None
        if latest_dir.exists():
            return latest_dir
    return None


def resolve_latest_or_self(path: str | Path, kind: str) -> Path:
    """如果 path 是产物根目录并带 latest 指针，返回最近一次实际目录；否则返回自身。"""

    path = Path(path).resolve()
    latest = read_latest_pointer(path, kind=kind)
    return latest if latest is not None else path
