"""把已有 realtime_pose source 的 stationary_prob_5 迁移到当前标签契约。"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from data_loaders.realtime_pose_contract import (
    RUNTIME_CONTRACT_METADATA_FIELDS,
    load_source_metadata,
    required_realtime_source_fields,
    validate_realtime_source_contract,
    validate_root_y0_invariants,
    validate_schema_metadata,
)
from data_loaders.realtime_pose_kinematics import derive_stationary_prob_from_joints_world
from data_loaders.sensor_masking import (
    DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SCHEMA_NAMES,
    STATIONARY_JOINT_INDICES,
    STATIONARY_JOINT_NAMES,
    get_schema_spec,
)
from data_loaders.stationary_label_config import (
    STATIONARY_LABEL_METADATA_FIELDS,
    stationary_label_metadata,
)


LEGACY_STATIONARY_METADATA = {
    "stationary_label_method": "joint_center_speed_only_v1",
    "stationary_speed_full_motion": 0.25,
    "stationary_median_window": 5,
}
LEGACY_STATIONARY_METADATA_FIELDS = tuple(LEGACY_STATIONARY_METADATA)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relabel realtime_pose source files with the current causal stationary contract."
    )
    parser.add_argument("--source_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--source_set_name", required=True, type=str)
    parser.add_argument(
        "--schema",
        default=DEFAULT_REALTIME_POSE_SCHEMA_NAME,
        choices=REALTIME_POSE_SCHEMA_NAMES,
        type=str,
    )
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rebuild_manifest", action="store_true")
    parser.add_argument("--allow_partial", action="store_true")
    return parser.parse_args(argv)


def _metadata_scalar(metadata: dict[str, Any], key: str) -> Any:
    value = metadata[key]
    if isinstance(value, np.ndarray):
        return value.item()
    return value


def validate_legacy_stationary_metadata(metadata: dict[str, Any], source: str) -> None:
    missing = [key for key in LEGACY_STATIONARY_METADATA_FIELDS if key not in metadata]
    if missing:
        raise ValueError(f"{source} 缺少旧 stationary metadata: {missing}")
    for key, expected in LEGACY_STATIONARY_METADATA.items():
        actual = _metadata_scalar(metadata, key)
        if isinstance(expected, str):
            if str(actual) != expected:
                raise ValueError(f"{source} {key}={actual!r}, expected {expected!r}")
        elif not np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-8):
            raise ValueError(f"{source} {key}={actual!r}, expected {expected!r}")


def discover_source_files(source_dir: Path, limit: int = 0) -> list[Path]:
    paths = sorted(path for path in source_dir.rglob("*.npz") if path.is_file())
    if int(limit) > 0:
        paths = paths[: int(limit)]
    if not paths:
        raise FileNotFoundError(f"{source_dir} 下没有 realtime_pose source npz")
    return paths


def read_source_manifest_entries(source_dir: Path) -> dict[str, dict[str, Any]]:
    manifest_path = source_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return {}
    entries: dict[str, dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            entry = json.loads(line)
            relative_path = str(entry.get("source_relative_path") or "").replace("\\", "/")
            if relative_path:
                entries[relative_path] = entry
    return entries


def _validate_relabel_input(
    *,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    schema_name: str,
    source: str,
) -> None:
    schema = get_schema_spec(schema_name)
    validate_schema_metadata(metadata, schema=schema, source=source)
    validate_legacy_stationary_metadata(metadata, source=source)
    missing_runtime = [key for key in RUNTIME_CONTRACT_METADATA_FIELDS if key not in metadata]
    if missing_runtime:
        raise ValueError(f"{source} 缺少 runtime metadata: {missing_runtime}")
    required = required_realtime_source_fields(schema.name)
    missing_arrays = sorted(key for key in required if key not in arrays)
    if missing_arrays:
        raise KeyError(f"{source} 缺少 realtime source arrays: {missing_arrays}")
    validate_root_y0_invariants(arrays, schema=schema, source=source)


def _next_metadata(
    *,
    metadata: dict[str, Any],
    source_path: Path,
    relative_path: Path,
    source_set_name: str,
) -> dict[str, Any]:
    result = dict(metadata)
    for key in (*LEGACY_STATIONARY_METADATA_FIELDS, *STATIONARY_LABEL_METADATA_FIELDS):
        result.pop(key, None)
    result.update(stationary_label_metadata())
    result["source_set_name"] = str(source_set_name)
    result["stationary_relabel_source_path"] = str(source_path)
    result["stationary_relabel_source_relative_path"] = relative_path.as_posix()
    converter_args = result.get("converter_args")
    if isinstance(converter_args, dict):
        converter_args = dict(converter_args)
        converter_args["source_set_name"] = str(source_set_name)
        result["converter_args"] = converter_args
    return result


def _write_npz_atomic(
    output_path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.npz")
    np.savez(
        temporary_path,
        **arrays,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    temporary_path.replace(output_path)


def relabel_source_file(
    source_path: Path,
    source_dir: Path,
    output_dir: Path,
    source_set_name: str,
    schema_name: str,
    overwrite: bool,
) -> dict[str, Any]:
    relative_path = source_path.relative_to(source_dir)
    output_path = output_dir / relative_path
    if output_path.exists() and not overwrite:
        with np.load(output_path, allow_pickle=False) as data:
            validate_realtime_source_contract(
                data,
                schema=get_schema_spec(schema_name),
                source=str(output_path),
            )
            frames = int(np.asarray(data["joints_world"]).shape[0])
        return {
            "status": "skipped_existing",
            "input_source_path": str(source_path),
            "source_relative_path": relative_path.as_posix(),
            "output_path": str(output_path),
            "source_set_name": str(source_set_name),
            "frames": frames,
            **stationary_label_metadata(),
        }

    with np.load(source_path, allow_pickle=False) as data:
        metadata = load_source_metadata(data, source=str(source_path))
        arrays = {
            key: np.asarray(data[key]).copy()
            for key in data.files
            if key != "metadata"
        }
    _validate_relabel_input(
        arrays=arrays,
        metadata=metadata,
        schema_name=schema_name,
        source=str(source_path),
    )
    target_fps = float(metadata.get("target_fps", 60.0))
    arrays["stationary_prob_5"] = derive_stationary_prob_from_joints_world(
        arrays["joints_world"],
        fps=target_fps,
    )
    next_metadata = _next_metadata(
        metadata=metadata,
        source_path=source_path,
        relative_path=relative_path,
        source_set_name=source_set_name,
    )
    validation_payload = dict(arrays)
    validation_payload["metadata"] = np.asarray(json.dumps(next_metadata, ensure_ascii=False))
    validate_realtime_source_contract(
        validation_payload,
        schema=get_schema_spec(schema_name),
        source=str(output_path),
    )
    _write_npz_atomic(output_path, arrays=arrays, metadata=next_metadata)
    return {
        "status": "relabelled",
        "input_source_path": str(source_path),
        "source_relative_path": relative_path.as_posix(),
        "output_path": str(output_path),
        "source_set_name": str(source_set_name),
        "frames": int(arrays["joints_world"].shape[0]),
        **stationary_label_metadata(),
    }


def _relabel_worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    return relabel_source_file(*payload)


def write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def build_manifest_record(
    *,
    record: dict[str, Any],
    original_entry: dict[str, Any] | None,
    output_dir: Path,
    source_set_name: str,
    schema_name: str,
) -> dict[str, Any]:
    if record.get("status") == "failed":
        return dict(record)
    schema = get_schema_spec(schema_name)
    relative_path = str(record["source_relative_path"]).replace("\\", "/")
    result = dict(original_entry or {})
    for key in (*LEGACY_STATIONARY_METADATA_FIELDS, *STATIONARY_LABEL_METADATA_FIELDS):
        result.pop(key, None)
    result.update(
        {
            "status": str(record["status"]),
            "input_source_path": str(record["input_source_path"]),
            "source_relative_path": relative_path,
            "output_path": str((output_dir / Path(relative_path)).resolve()),
            "source_set_name": str(source_set_name),
            "frames": int(record["frames"]),
            "schema_name": schema.name,
            "schema_canonical_name": str(schema.canonical_name),
            "pose_representation": schema.pose_representation,
            "root_y_policy": schema.root_y_policy,
            "pelvis_height_mode": schema.pelvis_height_mode,
            "stablemotion_split_key": str(
                result.get("stablemotion_split_key")
                or Path(relative_path).with_suffix(".npy").as_posix()
            ).replace("\\", "/"),
            **stationary_label_metadata(),
        }
    )
    converter_args = result.get("converter_args")
    if isinstance(converter_args, dict):
        converter_args = dict(converter_args)
        converter_args["source_set_name"] = str(source_set_name)
        result["converter_args"] = converter_args
    return result


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if source_dir == output_dir:
        raise ValueError("stationary relabel 必须写入新的 source_dir，禁止原地覆盖")
    if int(args.num_workers) <= 0:
        raise ValueError("--num_workers 必须为正整数")

    source_paths = discover_source_files(source_dir, limit=args.limit)
    manifest_path = output_dir / "manifest.jsonl"
    if manifest_path.exists() and not (args.overwrite or args.rebuild_manifest):
        raise FileExistsError(
            f"{manifest_path} 已存在；请使用 --rebuild_manifest 或 --overwrite 重建"
        )
    original_entries = read_source_manifest_entries(source_dir)
    payloads = [
        (
            source_path,
            source_dir,
            output_dir,
            str(args.source_set_name),
            str(args.schema),
            bool(args.overwrite),
        )
        for source_path in source_paths
    ]
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if int(args.num_workers) == 1:
        iterator = zip(source_paths, payloads)
        for source_path, worker_args in tqdm(
            iterator,
            total=len(payloads),
            desc="relabel stationary",
        ):
            try:
                records.append(_relabel_worker(worker_args))
            except Exception as error:
                failures.append({"input_source_path": str(source_path), "error": repr(error)})
                if not args.allow_partial:
                    raise
    else:
        with ProcessPoolExecutor(max_workers=int(args.num_workers)) as executor:
            futures = [
                (source_path, executor.submit(_relabel_worker, payload))
                for source_path, payload in zip(source_paths, payloads)
            ]
            for source_path, future in tqdm(futures, total=len(futures), desc="relabel stationary"):
                try:
                    records.append(future.result())
                except Exception as error:
                    failures.append({"input_source_path": str(source_path), "error": repr(error)})
                    if not args.allow_partial:
                        raise

    manifest_records = [
        build_manifest_record(
            record=record,
            original_entry=original_entries.get(
                str(record["source_relative_path"]).replace("\\", "/")
            ),
            output_dir=output_dir,
            source_set_name=str(args.source_set_name),
            schema_name=str(args.schema),
        )
        for record in records
    ]
    manifest_records.extend({"status": "failed", **failure} for failure in failures)
    write_manifest(manifest_path, manifest_records)
    print(
        "[relabel_realtime_pose_stationary] "
        f"total={len(source_paths)} success={len(records) - len(failures)} "
        f"failed={len(failures)} manifest={manifest_path}"
    )
    return manifest_path


if __name__ == "__main__":
    main()
