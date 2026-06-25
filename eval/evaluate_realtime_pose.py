from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_loaders.sensor_masking import (
    BODY_POSE_DIM,
    BODY_POSE_START,
    REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_TARGET_START,
    ROOT_YAW_DELTA_DIM,
    ROOT_YAW_DELTA_START,
    get_schema_spec,
)


FEATURE_ARRAY_GROUPS = (
    ("raw", ("reference_features_raw",), ("reconstructed_features_raw",)),
    ("normalized", ("reference_features_normalized",), ("reconstructed_features_normalized",)),
    ("legacy", ("reference_features", "reference_motion", "x"), ("reconstructed_features", "reconstructed_motion", "pred_x0")),
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate realtime_pose_stationary5_v1 reconstruction result npz files.")
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--output_json", default="", type=str)
    return parser


def evaluate_file(path: Path) -> dict[str, float | int | str]:
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    with np.load(path, allow_pickle=False) as data:
        reference, reconstructed, feature_space = read_feature_pair(data, path, feature_dim=schema.feature_dim)
        if reconstructed.shape != reference.shape:
            raise ValueError(f"{path} reconstructed shape 不匹配：{reconstructed.shape} vs {reference.shape}")
        inpaint_mask = read_mask(data, reference.shape)

    target_mask = inpaint_mask[:, :, :schema.target_dim].any(axis=2)
    if not target_mask.any():
        target_mask = np.zeros(reference.shape[:2], dtype=bool)
        target_mask[:, REALTIME_POSE_TARGET_START] = True

    pose_slice = slice(BODY_POSE_START, BODY_POSE_START + BODY_POSE_DIM)
    yaw_slice = slice(ROOT_YAW_DELTA_START, ROOT_YAW_DELTA_START + ROOT_YAW_DELTA_DIM)
    pose_mse = float(np.mean(np.square(reconstructed[target_mask, pose_slice] - reference[target_mask, pose_slice])))

    pred_yaw = normalize_vectors(reconstructed[target_mask, yaw_slice])
    ref_yaw = normalize_vectors(reference[target_mask, yaw_slice])
    yaw_cos_loss = float(np.mean(1.0 - np.sum(pred_yaw * ref_yaw, axis=-1)))
    target_mae = float(np.mean(np.abs(reconstructed[target_mask, :schema.target_dim] - reference[target_mask, :schema.target_dim])))
    return {
        "path": str(path),
        "batch_size": int(reference.shape[0]),
        "frames": int(reference.shape[1]),
        "target_frames": int(target_mask.sum()),
        "feature_space": feature_space,
        "pose_mse": pose_mse,
        "yaw_cos_loss": yaw_cos_loss,
        "target_mae": target_mae,
    }


def read_feature_pair(data, path: Path, feature_dim: int) -> tuple[np.ndarray, np.ndarray, str]:
    """
    优先读取 raw 成对字段。

    `save_reconstruction` 会同时保存 raw/normalized；评估指标默认应在 raw 特征空间计算，
    否则 yaw sin/cos 和不同通道的 MSE 会被 normalizer 改变量纲。
    """

    files = set(data.files)
    for feature_space, reference_keys, reconstructed_keys in FEATURE_ARRAY_GROUPS:
        if not set(reference_keys).isdisjoint(files) and not set(reconstructed_keys).isdisjoint(files):
            reference = read_feature_array(data, path, reference_keys, feature_dim=feature_dim)
            reconstructed = read_feature_array(data, path, reconstructed_keys, feature_dim=feature_dim)
            return reference, reconstructed, feature_space
    raise KeyError(f"{path} 缺少可配对的 reference/reconstructed 特征字段。")


def read_feature_array(data, path: Path, keys: tuple[str, ...], feature_dim: int) -> np.ndarray:
    for key in keys:
        if key in data.files:
            array = np.asarray(data[key], dtype=np.float32)
            break
    else:
        raise KeyError(f"{path} 缺少特征数组，已尝试字段：{keys}")
    if array.ndim == 2 and array.shape[1] == feature_dim:
        return array[None]
    if array.ndim == 3 and array.shape[2] == feature_dim:
        return array
    if array.ndim == 3 and array.shape[1] == feature_dim:
        return np.transpose(array, (0, 2, 1))
    raise ValueError(f"{path} 特征应为 [T,{feature_dim}] 或 [B,T,{feature_dim}]，实际为 {array.shape}")


def read_mask(data, reference_shape: tuple[int, int, int]) -> np.ndarray:
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    if "inpaint_mask" not in data.files:
        mask = np.zeros(reference_shape, dtype=bool)
        mask[:, REALTIME_POSE_TARGET_START, :schema.target_dim] = True
        return mask
    mask = np.asarray(data["inpaint_mask"], dtype=bool)
    if mask.shape == reference_shape:
        return mask
    if mask.shape == reference_shape[1:]:
        return mask[None]
    if mask.shape == (reference_shape[2], reference_shape[1]):
        return mask.T[None]
    if mask.shape == (reference_shape[0], reference_shape[2], reference_shape[1]):
        return np.transpose(mask, (0, 2, 1))
    raise ValueError(f"inpaint_mask shape 不匹配：{mask.shape} vs {reference_shape}")


def normalize_vectors(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def summarize(results: list[dict[str, float | int | str]]) -> dict[str, float | int | str]:
    if not results:
        raise RuntimeError("没有可评估的 realtime_pose 结果文件。")
    feature_spaces = sorted({str(item.get("feature_space", "unknown")) for item in results})
    if len(feature_spaces) != 1:
        raise ValueError(f"不能混合评估不同 feature_space: {feature_spaces}")
    return {
        "schema_name": REALTIME_POSE_SCHEMA_NAME,
        "file_count": len(results),
        "feature_space": feature_spaces[0],
        "pose_mse": float(np.mean([float(item["pose_mse"]) for item in results])),
        "yaw_cos_loss": float(np.mean([float(item["yaw_cos_loss"]) for item in results])),
        "target_mae": float(np.mean([float(item["target_mae"]) for item in results])),
    }


def main(argv: list[str] | None = None) -> dict[str, float | int | str]:
    args = build_arg_parser().parse_args(argv)
    input_dir = Path(args.input_dir).resolve()
    results = [evaluate_file(path) for path in sorted(input_dir.rglob("*.npz"))]
    summary = summarize(results)
    output_json = Path(args.output_json).resolve() if args.output_json else input_dir / "realtime_pose_eval_summary.json"
    with output_json.open("w", encoding="utf-8") as file:
        json.dump({"summary": summary, "files": results}, file, indent=2, ensure_ascii=False)
    print(f"[evaluate_realtime_pose] wrote {output_json}")
    return summary


if __name__ == "__main__":
    main()
