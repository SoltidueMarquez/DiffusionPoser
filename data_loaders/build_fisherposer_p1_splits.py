from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from data_loaders.generate_realtime_pose_tasks import (
    USABLE_SOURCE_STATUSES,
    normalize_split_key,
    read_source_entries,
)
from data_loaders.sensor_masking import REALTIME_POSE_HISTORY_LENGTH


FISHERPOSER_P1_SUBSETS = (
    "CMU",
    "BioMotionLab_NTroje",
    "MPI_HDM05",
)
FISHERPOSER_P1_TARGET_FPS = 60.0
# Task Store 还需要当前帧后的 3 帧 future-leg 监督，因此不能只检查 61 帧窗口。
FISHERPOSER_P1_MIN_SOURCE_FRAMES = REALTIME_POSE_HISTORY_LENGTH + 1 + 3
LOCAL_TEST_RATIO = 0.1


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 FisherPoser P1 的 train/test 序列划分。")
    paths = parser.add_argument_group("paths")
    paths.add_argument(
        "--source_dir",
        default="dataset/AMASS_realtime_pose_body_fbx_local_pelvis_residual_root_y0_stationary5_60hz",
        help="已转换 realtime pose source 目录，必须包含 manifest.jsonl。",
    )
    paths.add_argument(
        "--output_dir",
        default="data_loaders/splits/fisherposer_p1",
        help="写入 train.txt、test.txt 和 split_meta.json 的新目录。",
    )
    paths.add_argument(
        "--official_split_dir",
        default="",
        help="AvatarPoser data_split 目录；留空时按 seed 在每个子集内做确定性 90/10。",
    )

    split = parser.add_argument_group("split")
    split.add_argument("--seed", default=10, type=int)
    return parser


def seeded_order(source_key: str, seed: int) -> str:
    """用路径和 seed 建立稳定顺序，避免 manifest 重排后改变划分。"""

    value = f"{int(seed)}\x1f{normalize_split_key(source_key)}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def build_fisherposer_p1_splits(
    source_dir: str | Path,
    output_dir: str | Path,
    official_split_dir: str | Path | None = None,
    seed: int = 10,
) -> dict[str, Any]:
    source_root = Path(source_dir).resolve()
    output_root = Path(output_dir).resolve()
    manifest_path = source_root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"FisherPoser P1 必须从 source manifest 生成，找不到: {manifest_path}")
    if output_root.exists():
        raise FileExistsError(f"为避免覆盖已有协议，输出目录必须不存在: {output_root}")

    manifest_records = read_manifest_records(manifest_path)
    source_entries = read_source_entries(source_root)
    eligible_by_key, eligibility_reasons, exclusion_counts = select_eligible_sources(
        manifest_records=manifest_records,
        source_entries=source_entries,
    )

    official_root = Path(official_split_dir).resolve() if official_split_dir else None
    official_meta: dict[str, Any] | None = None
    if official_root is not None:
        train_keys, test_keys, official_meta = import_official_splits(
            official_split_dir=official_root,
            eligible_by_key=eligible_by_key,
            eligibility_reasons=eligibility_reasons,
        )
        split_origin = "avatarposer_official"
    else:
        train_keys, test_keys = deterministic_local_split(
            eligible_by_key=eligible_by_key,
            seed=int(seed),
        )
        split_origin = "deterministic_local_90_10"

    validate_partitions(train_keys=train_keys, test_keys=test_keys, eligible_by_key=eligible_by_key)
    output_root.mkdir(parents=True, exist_ok=False)
    train_path = output_root / "train.txt"
    test_path = output_root / "test.txt"
    write_split_file(train_path, train_keys)
    write_split_file(test_path, test_keys)

    metadata: dict[str, Any] = {
        "protocol": "fisherposer_p1",
        "split_origin": split_origin,
        "seed": int(seed),
        "subsets": list(FISHERPOSER_P1_SUBSETS),
        "mirror_policy": "disabled",
        "target_fps": FISHERPOSER_P1_TARGET_FPS,
        "min_source_frames": FISHERPOSER_P1_MIN_SOURCE_FRAMES,
        "local_split_policy": {
            "test_ratio": LOCAL_TEST_RATIO,
            "ordering": "sha256(seed\\x1fsource_key)",
            "test_count_rounding": "ceil",
        },
        # 对 manifest 记录做规范化后再 hash，使条目顺序不影响协议身份。
        "source_manifest_canonical_sha256": canonical_manifest_sha256(manifest_records),
        "eligibility_exclusion_counts": exclusion_counts,
        "counts": build_counts(
            eligible_by_key=eligible_by_key,
            train_keys=train_keys,
            test_keys=test_keys,
        ),
        "split_sha256": {
            "train.txt": sha256_file(train_path),
            "test.txt": sha256_file(test_path),
        },
    }
    if official_meta is not None:
        metadata["official_split"] = official_meta

    meta_path = output_root / "split_meta.json"
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return metadata


def read_manifest_records(manifest_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest_path}:{line_number} 不是有效 JSON。") from exc
            if not isinstance(record, dict):
                raise TypeError(f"{manifest_path}:{line_number} 必须是 JSON object。")
            records.append(record)
    if not records:
        raise RuntimeError(f"source manifest 为空: {manifest_path}")
    return records


def select_eligible_sources(
    manifest_records: list[dict[str, Any]],
    source_entries: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, int]]:
    reasons: dict[str, str] = {}
    counts: defaultdict[str, int] = defaultdict(int)

    # 先从原始 manifest 统计被 read_source_entries 过滤掉的失败 source。
    for record in manifest_records:
        key = manifest_record_key(record)
        if not key:
            counts["missing_split_key"] += 1
            continue
        subset, is_mirrored = source_subset_and_mirror(key, bool(record.get("is_mirrored", False)))
        if subset not in FISHERPOSER_P1_SUBSETS:
            reasons[key] = "outside_p1_subsets"
            counts["outside_p1_subsets"] += 1
        elif is_mirrored:
            reasons[key] = "mirrored"
            counts["mirrored"] += 1
        elif record.get("status", "converted") not in USABLE_SOURCE_STATUSES:
            reasons[key] = "unusable_status"
            counts["unusable_status"] += 1

    eligible: dict[str, dict[str, Any]] = {}
    for entry in source_entries:
        key = normalize_split_key(str(entry["stablemotion_split_key"]))
        subset, is_mirrored = source_subset_and_mirror(key, bool(entry["is_mirrored"]))
        reason = reasons.get(key)
        if reason is None and subset not in FISHERPOSER_P1_SUBSETS:
            reason = "outside_p1_subsets"
        elif reason is None and is_mirrored:
            reason = "mirrored"
        elif reason is None and not math.isclose(
            float(entry["target_fps"]),
            FISHERPOSER_P1_TARGET_FPS,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            reason = "target_fps_mismatch"
        elif reason is None and int(entry["source_frames"]) < FISHERPOSER_P1_MIN_SOURCE_FRAMES:
            reason = "too_short"

        if reason is not None:
            if key not in reasons:
                counts[reason] += 1
            reasons[key] = reason
            continue
        if key in eligible:
            raise ValueError(f"source manifest 出现重复 stablemotion_split_key: {key}")
        eligible[key] = entry

    for name in (
        "missing_split_key",
        "outside_p1_subsets",
        "mirrored",
        "unusable_status",
        "target_fps_mismatch",
        "too_short",
    ):
        counts[name] += 0
    return eligible, reasons, dict(sorted(counts.items()))


def deterministic_local_split(
    eligible_by_key: dict[str, dict[str, Any]],
    seed: int,
) -> tuple[set[str], set[str]]:
    train_keys: set[str] = set()
    test_keys: set[str] = set()
    grouped = group_keys_by_subset(eligible_by_key)
    for subset in FISHERPOSER_P1_SUBSETS:
        keys = grouped[subset]
        if len(keys) < 2:
            raise RuntimeError(f"P1 子集 {subset} 至少需要 2 条合格序列，实际为 {len(keys)}。")
        ordered = sorted(keys, key=lambda key: (seeded_order(key, seed), key))
        test_count = int(math.ceil(len(ordered) * LOCAL_TEST_RATIO))
        test_keys.update(ordered[:test_count])
        train_keys.update(ordered[test_count:])
    return train_keys, test_keys


def import_official_splits(
    official_split_dir: Path,
    eligible_by_key: dict[str, dict[str, Any]],
    eligibility_reasons: dict[str, str],
) -> tuple[set[str], set[str], dict[str, Any]]:
    if not official_split_dir.is_dir():
        raise FileNotFoundError(f"找不到 AvatarPoser data_split 目录: {official_split_dir}")

    listed_train: set[str] = set()
    listed_test: set[str] = set()
    input_hashes: dict[str, str] = {}
    for subset in FISHERPOSER_P1_SUBSETS:
        for split_name, destination in (("train", listed_train), ("test", listed_test)):
            relative_path = Path(subset) / f"{split_name}_split.txt"
            path = official_split_dir / relative_path
            keys = read_official_split_file(path=path, expected_subset=subset)
            duplicate = destination.intersection(keys)
            if duplicate:
                raise ValueError(f"官方 {split_name} 名单出现跨子集重复 key: {sorted(duplicate)[:3]}")
            destination.update(keys)
            input_hashes[relative_path.as_posix()] = canonical_keys_sha256(keys)

    overlap = listed_train.intersection(listed_test)
    if overlap:
        raise ValueError(f"AvatarPoser 官方 train/test 有交集: {sorted(overlap)[:3]}")

    official_exclusions: defaultdict[str, list[str]] = defaultdict(list)

    def project(keys: set[str]) -> set[str]:
        selected: set[str] = set()
        for key in keys:
            if key in eligible_by_key:
                selected.add(key)
                continue
            if key.rsplit("/", 1)[-1] == "shape":
                reason = "non_motion_shape_file"
            else:
                reason = eligibility_reasons.get(key, "not_in_usable_source_manifest")
            official_exclusions[reason].append(output_split_key(key))
        return selected

    train_keys = project(listed_train)
    test_keys = project(listed_test)
    unlisted = set(eligible_by_key).difference(listed_train, listed_test)
    if unlisted:
        official_exclusions["eligible_not_listed"].extend(output_split_key(key) for key in unlisted)

    return train_keys, test_keys, {
        "input_canonical_sha256": dict(sorted(input_hashes.items())),
        "listed_counts": {"train": len(listed_train), "test": len(listed_test)},
        "eligibility_projection": "realtime_pose_task_compatible",
        "exclusions": {
            reason: sorted(keys)
            for reason, keys in sorted(official_exclusions.items())
        },
    }


def read_official_split_file(path: Path, expected_subset: str) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 AvatarPoser 官方 split: {path}")
    raw_keys = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = [normalize_split_key(line) for line in raw_keys]
    if len(keys) != len(set(keys)):
        raise ValueError(f"AvatarPoser 官方 split 包含重复 key: {path}")
    for key in keys:
        subset, is_mirrored = source_subset_and_mirror(key, False)
        if is_mirrored or subset != expected_subset:
            raise ValueError(f"{path} 包含不属于 {expected_subset} 的 key: {key}")
    return set(keys)


def validate_partitions(
    train_keys: set[str],
    test_keys: set[str],
    eligible_by_key: dict[str, dict[str, Any]],
) -> None:
    overlap = train_keys.intersection(test_keys)
    if overlap:
        raise ValueError(f"FisherPoser P1 train/test 有交集: {sorted(overlap)[:3]}")
    for split_name, keys in (("train", train_keys), ("test", test_keys)):
        if not keys:
            raise RuntimeError(f"FisherPoser P1 {split_name} 为空。")
        missing = keys.difference(eligible_by_key)
        if missing:
            raise RuntimeError(f"{split_name} 包含不可用 source: {sorted(missing)[:3]}")
        present_subsets = {source_subset_and_mirror(key, False)[0] for key in keys}
        missing_subsets = set(FISHERPOSER_P1_SUBSETS).difference(present_subsets)
        if missing_subsets:
            raise RuntimeError(f"{split_name} 缺少 P1 子集: {sorted(missing_subsets)}")


def build_counts(
    eligible_by_key: dict[str, dict[str, Any]],
    train_keys: set[str],
    test_keys: set[str],
) -> dict[str, Any]:
    grouped_eligible = group_keys_by_subset(eligible_by_key)
    grouped_train = group_keys_by_subset({key: eligible_by_key[key] for key in train_keys})
    grouped_test = group_keys_by_subset({key: eligible_by_key[key] for key in test_keys})
    return {
        "eligible": len(eligible_by_key),
        "train": len(train_keys),
        "test": len(test_keys),
        "by_subset": {
            subset: {
                "eligible": len(grouped_eligible[subset]),
                "train": len(grouped_train[subset]),
                "test": len(grouped_test[subset]),
            }
            for subset in FISHERPOSER_P1_SUBSETS
        },
    }


def group_keys_by_subset(entries: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    grouped = {subset: [] for subset in FISHERPOSER_P1_SUBSETS}
    for key in entries:
        subset, is_mirrored = source_subset_and_mirror(key, False)
        if not is_mirrored and subset in grouped:
            grouped[subset].append(key)
    return grouped


def manifest_record_key(record: dict[str, Any]) -> str:
    raw_key = record.get("stablemotion_split_key") or record.get("source_relative_path")
    return normalize_split_key(str(raw_key)) if raw_key else ""


def source_subset_and_mirror(source_key: str, declared_mirror: bool) -> tuple[str, bool]:
    parts = normalize_split_key(source_key).split("/")
    path_mirror = bool(parts and parts[0] == "M")
    subset_index = 1 if path_mirror else 0
    subset = parts[subset_index] if len(parts) > subset_index else ""
    return subset, bool(declared_mirror) or path_mirror


def output_split_key(source_key: str) -> str:
    return f"{normalize_split_key(source_key)}.npy"


def write_split_file(path: Path, source_keys: Iterable[str]) -> None:
    lines = [output_split_key(key) for key in sorted(set(source_keys))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def canonical_manifest_sha256(records: list[dict[str, Any]]) -> str:
    canonical_records = []
    for record in records:
        canonical_records.append(
            {
                "key": manifest_record_key(record),
                "status": str(record.get("status", "converted")),
                "frames": record.get("frames"),
                "target_fps": record.get("target_fps"),
                "is_mirrored": bool(record.get("is_mirrored", False)),
            }
        )
    payload = json.dumps(
        sorted(canonical_records, key=lambda value: json.dumps(value, sort_keys=True)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_keys_sha256(keys: Iterable[str]) -> str:
    payload = "".join(f"{normalize_split_key(key)}\n" for key in sorted(set(keys))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    metadata = build_fisherposer_p1_splits(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        official_split_dir=args.official_split_dir or None,
        seed=int(args.seed),
    )
    counts = metadata["counts"]
    print(
        "FisherPoser P1 split 生成完成: "
        f"origin={metadata['split_origin']}, train={counts['train']}, test={counts['test']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
