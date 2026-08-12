from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from data_loaders.generate_realtime_pose_tasks import (
    normalize_split_key,
    read_source_entries,
)


PROTOCOL_DIR_NAMES = {"p1": "RPM-P1", "p2": "RPM-P2"}
EXPECTED_COUNTS = {
    "p1": {"train": 4725, "test": 526},
    "p2": {"train": 11634, "test": 138},
}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="把 RPM 官方 P1/P2 名单映射为本项目可读取的 split。")
    parser.add_argument(
        "--research_kit_dir",
        required=True,
        type=Path,
        help="解压后的 RPM 研究包根目录，内部必须包含 protocols/p1 和 protocols/p2。",
    )
    parser.add_argument(
        "--raw_amass_dir",
        required=True,
        type=Path,
        help="完整 AMASS 原始目录，用于把 stageii 名称映射到本机 poses 文件名。",
    )
    parser.add_argument(
        "--source_dir",
        required=True,
        type=Path,
        help="当前 realtime pose source 目录，仅用于审计哪些官方条目尚未转换。",
    )
    parser.add_argument(
        "--output_root",
        default=Path("data_loaders/splits"),
        type=Path,
        help="在此目录下新建 RPM-P1 和 RPM-P2。",
    )
    return parser


def canonical_motion_identity(raw_path: str) -> str:
    """消除 RPM stageii 与本机 poses 文件名之间的非语义格式差异。"""

    normalized = str(raw_path).strip().replace("\\", "/")
    if normalized.endswith((".npz", ".npy")):
        normalized = normalized[:-4]
    lowered = normalized.casefold()
    for suffix in ("_stageii", "_poses"):
        if lowered.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    # AMASS 的同一动作在 SMPL-H 与 SMPL-X N 下载包中可能分别用空格、
    # 下划线或连字符表示分词。逐路径段移除标点后仍保留目录层级，既能对齐
    # 这些命名差异，也避免不同 subject 下的同名动作相互碰撞。
    return "/".join(
        re.sub(r"[^0-9a-z]+", "", segment.casefold())
        for segment in normalized.split("/")
    )


def build_source_identity_index(source_dir: Path) -> tuple[dict[str, dict[str, Any]], str]:
    manifest_path = source_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到 source manifest: {manifest_path}")
    identity_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in read_source_entries(source_dir):
        if entry["is_mirrored"]:
            continue
        identity = canonical_motion_identity(str(entry["stablemotion_split_key"]))
        identity_groups[identity].append(entry)
    ambiguous = {
        identity: [str(entry["stablemotion_split_key"]) for entry in entries]
        for identity, entries in identity_groups.items()
        if len(entries) != 1
    }
    if ambiguous:
        first_identity, first_entries = next(iter(sorted(ambiguous.items())))
        raise ValueError(f"source 规范化后存在歧义: {first_identity} -> {first_entries}")
    return (
        {identity: entries[0] for identity, entries in identity_groups.items()},
        sha256_file(manifest_path),
    )


def build_raw_identity_index(raw_amass_dir: Path) -> tuple[dict[str, list[str]], str]:
    if not raw_amass_dir.is_dir():
        raise FileNotFoundError(f"找不到 AMASS 原始目录: {raw_amass_dir}")
    identity_groups: defaultdict[str, list[str]] = defaultdict(list)
    inventory_lines: list[str] = []
    for path in sorted(raw_amass_dir.rglob("*_poses.npz")):
        relative = path.relative_to(raw_amass_dir).as_posix()
        identity_groups[canonical_motion_identity(relative)].append(
            normalize_split_key(Path(relative).with_suffix(".npy").as_posix())
        )
        inventory_lines.append(f"{relative}\t{path.stat().st_size}\n")
    if not identity_groups:
        raise RuntimeError(f"AMASS 原始目录中没有 *_poses.npz: {raw_amass_dir}")
    inventory_sha256 = hashlib.sha256("".join(inventory_lines).encode("utf-8")).hexdigest()
    return dict(identity_groups), inventory_sha256


def read_and_map_protocol_split(
    protocol_split_dir: Path,
    split: str,
    raw_index: dict[str, list[str]],
    source_identities: set[str],
) -> tuple[list[str], dict[str, int], dict[str, str], list[str]]:
    mapped_keys: list[str] = []
    subset_counts: dict[str, int] = {}
    input_hashes: dict[str, str] = {}
    missing_source_keys: list[str] = []
    split_paths = sorted(protocol_split_dir.glob(f"*/{split}_split.txt"))
    if not split_paths:
        return mapped_keys, subset_counts, input_hashes, missing_source_keys
    seen_keys: set[str] = set()
    for split_path in split_paths:
        subset = split_path.parent.name
        raw_paths = [
            line.strip()
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        subset_counts[subset] = len(raw_paths)
        input_hashes[f"{subset}/{split_path.name}"] = sha256_file(split_path)
        for raw_path in raw_paths:
            identity = canonical_motion_identity(raw_path)
            candidates = raw_index.get(identity, [])
            if not candidates:
                raise KeyError(f"RPM {split} 路径无法映射到原始 AMASS: {raw_path}")
            if len(candidates) != 1:
                raise ValueError(f"RPM {split} 路径映射到多个原始动作: {raw_path} -> {candidates}")
            source_key = candidates[0]
            if source_key in seen_keys:
                raise ValueError(f"RPM {split} 映射后出现重复 source key: {source_key}")
            seen_keys.add(source_key)
            mapped_keys.append(source_key)
            if identity not in source_identities:
                missing_source_keys.append(source_key)
    return mapped_keys, subset_counts, input_hashes, missing_source_keys


def build_rpm_protocol_split(
    protocol: str,
    research_kit_dir: Path,
    output_root: Path,
    raw_index: dict[str, list[str]],
    raw_inventory_sha256: str,
    source_identities: set[str],
    source_manifest_sha256: str,
) -> dict[str, Any]:
    protocol = protocol.casefold()
    if protocol not in PROTOCOL_DIR_NAMES:
        raise ValueError(f"不支持的 RPM 协议: {protocol}")
    protocol_split_dir = research_kit_dir / "protocols" / protocol / "splits"
    if not protocol_split_dir.is_dir():
        raise FileNotFoundError(f"找不到 RPM 协议目录: {protocol_split_dir}")
    output_dir = output_root / PROTOCOL_DIR_NAMES[protocol]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"为避免覆盖已有 split，输出目录必须不存在: {output_dir}")

    mapped: dict[str, list[str]] = {}
    counts_by_subset: dict[str, dict[str, int]] = {}
    input_hashes: dict[str, str] = {}
    missing_source_by_split: dict[str, list[str]] = {}
    for split in ("train", "test"):
        keys, subset_counts, split_hashes, missing_source_keys = read_and_map_protocol_split(
            protocol_split_dir=protocol_split_dir,
            split=split,
            raw_index=raw_index,
            source_identities=source_identities,
        )
        expected = EXPECTED_COUNTS[protocol][split]
        if len(keys) != expected:
            raise ValueError(f"RPM {protocol.upper()} {split} 应有 {expected} 条，实际为 {len(keys)}。")
        mapped[split] = keys
        counts_by_subset[split] = subset_counts
        input_hashes.update(split_hashes)
        missing_source_by_split[split] = missing_source_keys

    overlap = set(mapped["train"]).intersection(mapped["test"])
    if overlap:
        raise ValueError(f"RPM {protocol.upper()} train/test 映射后有交集: {sorted(overlap)[:3]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}
    for split, keys in mapped.items():
        path = output_dir / f"{split}.txt"
        path.write_text(
            "".join(f"{key}.npy\n" for key in keys),
            encoding="utf-8",
            newline="\n",
        )
        output_paths[split] = path

    metadata = {
        "protocol": f"rpm_{protocol}",
        "split_origin": "RPM_Table2_Research_Kit_P1_P2_20260811",
        "mapping_policy": "casefold_remove_punctuation_stageii_to_poses",
        "mirror_policy": "disabled",
        "counts": {
            "train": len(mapped["train"]),
            "test": len(mapped["test"]),
            "by_subset": counts_by_subset,
        },
        "raw_amass_inventory_sha256": raw_inventory_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "input_split_sha256": dict(sorted(input_hashes.items())),
        "output_split_sha256": {
            f"{split}.txt": sha256_file(path)
            for split, path in output_paths.items()
        },
        "mapping_audit": {
            "unmatched": 0,
            "ambiguous": 0,
            "duplicates": 0,
            "train_test_overlap": 0,
        },
        "current_source_coverage": {
            split: {
                "covered": len(mapped[split]) - len(missing_source_by_split[split]),
                "missing": len(missing_source_by_split[split]),
                "missing_keys": missing_source_by_split[split],
            }
            for split in ("train", "test")
        },
    }
    (output_dir / "split_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return metadata


def build_all_rpm_protocol_splits(
    research_kit_dir: str | Path,
    raw_amass_dir: str | Path,
    source_dir: str | Path,
    output_root: str | Path,
) -> dict[str, dict[str, Any]]:
    kit_root = Path(research_kit_dir).resolve()
    raw_root = Path(raw_amass_dir).resolve()
    source_root = Path(source_dir).resolve()
    output_root = Path(output_root).resolve()
    source_index, manifest_sha256 = build_source_identity_index(source_root)
    raw_index, raw_inventory_sha256 = build_raw_identity_index(raw_root)
    return {
        protocol: build_rpm_protocol_split(
            protocol=protocol,
            research_kit_dir=kit_root,
            output_root=output_root,
            raw_index=raw_index,
            raw_inventory_sha256=raw_inventory_sha256,
            source_identities=set(source_index),
            source_manifest_sha256=manifest_sha256,
        )
        for protocol in ("p1", "p2")
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    metadata = build_all_rpm_protocol_splits(
        research_kit_dir=args.research_kit_dir,
        raw_amass_dir=args.raw_amass_dir,
        source_dir=args.source_dir,
        output_root=args.output_root,
    )
    for protocol, values in metadata.items():
        counts = values["counts"]
        print(
            f"{PROTOCOL_DIR_NAMES[protocol]} 生成完成: "
            f"train={counts['train']}, test={counts['test']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
