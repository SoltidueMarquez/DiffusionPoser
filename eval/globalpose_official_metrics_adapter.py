from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from data_converter.amass_smpl_utils import AMASS_TO_UNITY
from data_loaders.body_fbx_kinematics import SOURCE_FK_TO_BODY_FBX_BASIS
from data_loaders.realtime_pose_kinematics import make_yaw_rotation_np, rotation_6d_to_matrix_np
from data_loaders.sensor_masking import REALTIME_POSE_SCHEMA_NAME, REALTIME_POSE_SCHEMA_NAMES, get_schema_spec
from eval.globalpose_metrics import compute_translation_drift
from utils.parser_util import str2bool


MOTION_METRIC_NAMES = [
    "L SIP Err (deg)",
    "L Angle Err (deg)",
    "L Joint Err (cm)",
    "L Vertex Err (cm)",
    "G SIP Err (deg)",
    "G Angle Err (deg)",
    "G Joint Err (cm)",
    "G Vertex Err (cm)",
    "Root Jitter (km/s^3)",
    "Joint Jitter (km/s^3)",
]
GLOBALPOSE_SIP_JOINT_MASK = [1, 2, 16, 17]
GLOBALPOSE_IGNORED_JOINT_MASK = [7, 8, 10, 11, 20, 21, 22, 23]
DEFAULT_WINDOW_SIZES = tuple(range(1, 8))


@dataclass(frozen=True)
class GlobalPoseMotion:
    """GlobalPose 官方 evaluator 需要的最小 motion 表示。"""

    sequence_name: str
    pose: np.ndarray  # [T, 24, 3, 3], SMPL local rotations in GlobalPose/SMPL coordinates.
    tran: np.ndarray  # [T, 3], SMPL root translation in GlobalPose/SMPL coordinates.
    frame_mask: np.ndarray | None = None
    metadata: dict[str, Any] | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute GlobalPose-style metrics from DiffusionPoser longseq rollout results.",
        allow_abbrev=False,
    )
    parser.add_argument("--results_dir", required=True, type=str)
    parser.add_argument("--globalpose_dataset", required=True, type=str)
    parser.add_argument("--globalpose_repo", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--dataset_name", default="", type=str)
    parser.add_argument("--schema", default=REALTIME_POSE_SCHEMA_NAME, choices=REALTIME_POSE_SCHEMA_NAMES, type=str)
    parser.add_argument("--evaluate_pose", default=True, type=str2bool)
    parser.add_argument("--evaluate_tran", default=True, type=str2bool)
    parser.add_argument("--trim_eval_mask", default=True, type=str2bool)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    return parser


# region realtime_pose -> GlobalPose motion


def decode_body_fbx_delta_to_smpl_local_rotations(
    body_pose_6d: np.ndarray,
    root_yaw: np.ndarray | None = None,
) -> np.ndarray:
    """把 body.fbx local-delta 6D 反解成 GlobalPose/SMPL 坐标下的 parent-local rotation。

    当前 schema 在 body pose 中把 root local rotation 固定为 identity，root heading 单独存放在
    `root_yaw`。因此这里的 root rotation 是 yaw-only 近似；非 root joint 可以和转换阶段保持同一
    个 basis 反变换。
    """

    values = np.asarray(body_pose_6d, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 24 * 6:
        raise ValueError(f"body_pose_6d 应为 [T,144]，实际为 {values.shape}")
    body_delta = rotation_6d_to_matrix_np(values.reshape(values.shape[0], 24, 6))
    basis = SOURCE_FK_TO_BODY_FBX_BASIS.astype(np.float64)
    local_unity = basis.T[None, None] @ body_delta @ basis[None, None]
    smpl_local = AMASS_TO_UNITY.T[None, None] @ local_unity @ AMASS_TO_UNITY[None, None]
    smpl_local[:, 0] = np.eye(3, dtype=np.float64)

    if root_yaw is not None:
        yaw = np.asarray(root_yaw, dtype=np.float64).reshape(-1)
        if yaw.shape != (values.shape[0],):
            raise ValueError(f"root_yaw 应为 [T]，实际为 {np.asarray(root_yaw).shape}")
        unity_root = make_yaw_rotation_np(yaw)
        # 项目坐标满足 R_unity = A @ R_smpl @ A.T，因此反变换为 A.T @ R_unity @ A。
        smpl_local[:, 0] = AMASS_TO_UNITY.T[None] @ unity_root @ AMASS_TO_UNITY[None]
    return smpl_local.astype(np.float32)


def motion_from_rollout_payload(
    payload: Mapping[str, Any],
    kind: str = "predicted",
    schema_name: str = REALTIME_POSE_SCHEMA_NAME,
) -> GlobalPoseMotion:
    """从 `unity_stream_long_sequence_result.npz` payload 中取出 predicted/reference motion。

    `kind="predicted"` 读取模型输出；`kind="reference"` 读取 reference，用于 roundtrip 自检。
    """

    if kind not in {"predicted", "reference"}:
        raise ValueError(f"kind 只能是 predicted/reference，实际为 {kind}")
    schema = get_schema_spec(schema_name)
    metadata = read_payload_metadata(payload)
    sequence_name = sequence_name_from_metadata(metadata)
    feature_key = f"{kind}_features_raw"
    joint_key = f"{kind}_joints_world"
    yaw_key = "root_yaw_predicted" if kind == "predicted" else "root_yaw_reference"
    missing = [key for key in (feature_key, joint_key, yaw_key) if not payload_has_key(payload, key)]
    if missing:
        raise KeyError(f"rollout payload 缺少字段：{missing}")

    features = squeeze_single_batch(payload[feature_key], feature_key).astype(np.float32)
    root_yaw = squeeze_single_batch(payload[yaw_key], yaw_key).astype(np.float32)
    joints_world = squeeze_single_batch(payload[joint_key], joint_key).astype(np.float32)
    if features.ndim != 2 or features.shape[1] < schema.body_pose_slice().stop:
        raise ValueError(f"{feature_key} 应为 [T,D] 或 [1,T,D]，实际为 {features.shape}")
    if joints_world.ndim != 3 or joints_world.shape[1:] != (24, 3):
        raise ValueError(f"{joint_key} 应为 [T,24,3] 或 [1,T,24,3]，实际为 {joints_world.shape}")
    if root_yaw.shape != (features.shape[0],):
        raise ValueError(f"{yaw_key} 应为 [T] 或 [1,T]，实际为 {root_yaw.shape}")

    body_pose_6d = features[:, schema.body_pose_slice()]
    pose = decode_body_fbx_delta_to_smpl_local_rotations(body_pose_6d=body_pose_6d, root_yaw=root_yaw)
    tran = unity_points_to_globalpose_smpl(joints_world[:, 0])
    frame_mask = None
    if payload_has_key(payload, "eval_frame_mask"):
        frame_mask = squeeze_single_batch(payload["eval_frame_mask"], "eval_frame_mask").astype(bool)
    return GlobalPoseMotion(
        sequence_name=sequence_name,
        pose=pose,
        tran=tran,
        frame_mask=frame_mask,
        metadata=metadata,
    )


def unity_points_to_globalpose_smpl(points_unity: np.ndarray) -> np.ndarray:
    points = np.asarray(points_unity, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError(f"points_unity 最后一维应为 3，实际为 {points.shape}")
    return (points @ AMASS_TO_UNITY).astype(np.float32)


def squeeze_single_batch(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim >= 2 and array.shape[0] == 1:
        return array[0]
    if array.ndim >= 2 and array.shape[0] != 1 and name.endswith("_raw"):
        raise ValueError(f"{name} 当前只支持 batch=1，实际为 {array.shape}")
    return array


def read_payload_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not payload_has_key(payload, "metadata"):
        return {}
    value = np.asarray(payload["metadata"])
    try:
        item = value.item()
    except ValueError:
        item = value
    if isinstance(item, dict):
        return dict(item)
    if isinstance(item, str) and item.strip():
        try:
            parsed = json.loads(item)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def sequence_name_from_metadata(metadata: Mapping[str, Any]) -> str:
    for key in ("globalpose_sequence_name", "source_relative_path", "raw_relative_path", "original_source_relative_path"):
        value = str(metadata.get(key, "") or "").strip()
        if not value:
            continue
        if ":" in value:
            value = value.split(":", 1)[1]
        return Path(value.replace("\\", "/")).stem

    sequence_id = str(metadata.get("sequence_id", "") or "").strip()
    for prefix in ("totalcapture_officalib_", "totalcapture_dipcalib_", "dipimu_"):
        if sequence_id.startswith(prefix):
            return sequence_id[len(prefix) :]
    return sequence_id


def payload_has_key(payload: Mapping[str, Any], key: str) -> bool:
    if hasattr(payload, "files"):
        return key in getattr(payload, "files")
    return key in payload


# endregion


# region GlobalPose targets and metrics


def load_globalpose_target_motions(dataset_path: str | Path) -> dict[str, GlobalPoseMotion]:
    import torch

    dataset = torch.load(dataset_path, map_location="cpu")
    targets: dict[str, GlobalPoseMotion] = {}
    for index, pose_axis_angle in enumerate(dataset["pose"]):
        name = str(dataset["name"][index]) if "name" in dataset else f"seq_{index:04d}"
        pose_np = tensor_to_numpy(pose_axis_angle).astype(np.float64)
        tran_np = tensor_to_numpy(dataset["tran"][index]).astype(np.float64)
        pose_matrix = axis_angle_pose_to_matrix(pose_np)
        targets[name] = GlobalPoseMotion(sequence_name=name, pose=pose_matrix, tran=tran_np.astype(np.float32))
    return targets


def axis_angle_pose_to_matrix(pose_axis_angle: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_axis_angle, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 24 * 3:
        raise ValueError(f"GlobalPose pose 应为 [T,72]，实际为 {pose.shape}")
    return Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(pose.shape[0], 24, 3, 3).astype(np.float32)


def compute_motion_evaluator_metrics(
    pose_p: np.ndarray,
    pose_t: np.ndarray,
    tran_p: np.ndarray,
    tran_t: np.ndarray,
    globalpose_repo: str | Path,
    device: str = "cpu",
) -> np.ndarray:
    """复刻 GlobalPose `test.py::MotionEvaluator`，返回 `[10,2]` mean/std。"""

    import torch

    art = import_globalpose_articulate(globalpose_repo)
    torch_device = torch.device(device)
    model_path = Path(globalpose_repo) / "models" / "SMPL_male.pkl"
    base_evaluator = art.FullMotionEvaluator(
        str(model_path),
        joint_mask=torch.tensor(GLOBALPOSE_SIP_JOINT_MASK, device=torch_device),
        device=torch_device,
    )
    pred_pose = torch.as_tensor(pose_p, dtype=torch.float32, device=torch_device).clone().view(-1, 24, 3, 3)
    true_pose = torch.as_tensor(pose_t, dtype=torch.float32, device=torch_device).clone().view(-1, 24, 3, 3)
    pred_tran = torch.as_tensor(tran_p, dtype=torch.float32, device=torch_device)
    true_tran = torch.as_tensor(tran_t, dtype=torch.float32, device=torch_device)

    eye = torch.eye(3, dtype=torch.float32, device=torch_device)
    pred_pose[:, GLOBALPOSE_IGNORED_JOINT_MASK] = eye
    true_pose[:, GLOBALPOSE_IGNORED_JOINT_MASK] = eye
    global_errs = base_evaluator(pose_p=pred_pose, pose_t=true_pose, tran_p=pred_tran, tran_t=true_tran)

    pred_pose[:, 0] = eye
    true_pose[:, 0] = eye
    local_errs = base_evaluator(pose_p=pred_pose, pose_t=true_pose)
    root_jitter = ((pred_tran[3:] - 3 * pred_tran[2:-1] + 3 * pred_tran[1:-2] - pred_tran[:-3]) * (60**3)).norm(
        dim=1
    )
    root_jitter_stats = torch.stack([root_jitter.mean(), root_jitter.std()]) if root_jitter.numel() else torch.zeros(2)

    result = torch.stack(
        [
            local_errs[9],
            local_errs[3],
            local_errs[0] * 100,
            local_errs[1] * 100,
            global_errs[9],
            global_errs[3],
            global_errs[0] * 100,
            global_errs[1] * 100,
            root_jitter_stats.to(global_errs.device) / 1000,
            global_errs[4] / 1000,
        ]
    )
    return result.detach().cpu().numpy().astype(np.float64)


def import_globalpose_articulate(globalpose_repo: str | Path):
    ensure_numpy_legacy_aliases_for_chumpy()
    repo = str(Path(globalpose_repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import articulate as art

    return art


def ensure_numpy_legacy_aliases_for_chumpy() -> None:
    """给旧版 chumpy 补 NumPy 2.x 已删除的别名。

    GlobalPose 的官方 SMPL pickle 会在反序列化时导入 `chumpy`。旧版 chumpy 仍然执行
    `from numpy import int, float, object, ...`，在当前 NumPy 下会失败。这里把兼容限定在
    本 adapter 进程内，避免改 site-packages 或为整个项目降级 NumPy。
    """

    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def aggregate_pose_metric_arrays(metric_arrays: list[np.ndarray]) -> dict[str, dict[str, float]]:
    if not metric_arrays:
        return {}
    stacked = np.stack(metric_arrays, axis=0)
    mean_array = stacked.mean(axis=0)
    return {
        name: {"mean": float(mean_array[index, 0]), "std": float(mean_array[index, 1])}
        for index, name in enumerate(MOTION_METRIC_NAMES)
    }


def compute_roundtrip_diagnostics(
    recovered: GlobalPoseMotion,
    target: GlobalPoseMotion,
    trim_eval_mask: bool = True,
) -> dict[str, float | int]:
    """评估 reference 从 realtime_pose 表示反解回 GlobalPose 表示的误差。

    `translation_delta_rmse_m` 使用相邻帧位移差，而不是绝对位置差；这样可以忽略 SMPL 模型
    root joint 与 GlobalPose `tran` 之间的固定 offset，重点检查漂移指标会用到的运动增量。
    """

    recovered_pose, target_pose, recovered_tran, target_tran = align_prediction_and_target(
        prediction=recovered,
        target=target,
        trim_eval_mask=trim_eval_mask,
    )
    angles = rotation_angle_degrees(recovered_pose, target_pose)
    nonroot_angles = angles[:, 1:]
    metric_joint_indices = [
        joint_index
        for joint_index in range(1, 24)
        if joint_index not in set(GLOBALPOSE_IGNORED_JOINT_MASK)
    ]
    metric_nonroot_angles = angles[:, metric_joint_indices]
    translation_delta_rmse = 0.0
    if recovered_tran.shape[0] > 1:
        recovered_delta = recovered_tran[1:] - recovered_tran[:-1]
        target_delta = target_tran[1:] - target_tran[:-1]
        translation_delta_rmse = float(np.sqrt(np.mean(np.sum((recovered_delta - target_delta) ** 2, axis=-1))))
    offset = target_tran.mean(axis=0) - recovered_tran.mean(axis=0)
    translation_aligned = recovered_tran + offset[None]
    translation_aligned_rmse = float(np.sqrt(np.mean(np.sum((translation_aligned - target_tran) ** 2, axis=-1))))
    return {
        "frames": int(recovered_pose.shape[0]),
        "root_angle_deg": float(np.mean(angles[:, 0])),
        "nonroot_local_angle_deg": float(np.mean(nonroot_angles)),
        "metric_nonroot_local_angle_deg": float(np.mean(metric_nonroot_angles)),
        "translation_delta_rmse_m": translation_delta_rmse,
        "translation_aligned_rmse_m": translation_aligned_rmse,
    }


def rotation_angle_degrees(rot_a: np.ndarray, rot_b: np.ndarray) -> np.ndarray:
    a = np.asarray(rot_a, dtype=np.float64)
    b = np.asarray(rot_b, dtype=np.float64)
    if a.shape != b.shape or a.shape[-2:] != (3, 3):
        raise ValueError(f"rotation shape 不匹配：{a.shape} vs {b.shape}")
    relative = a @ np.swapaxes(b, -1, -2)
    trace = np.trace(relative, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine)).astype(np.float64)


def summarize_roundtrip_diagnostics(items: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not items:
        return {}
    keys = (
        "root_angle_deg",
        "nonroot_local_angle_deg",
        "metric_nonroot_local_angle_deg",
        "translation_delta_rmse_m",
        "translation_aligned_rmse_m",
    )
    summary: dict[str, float | int] = {"sequence_count": len(items), "frames": int(sum(int(item["frames"]) for item in items))}
    for key in keys:
        summary[key] = float(np.mean([float(item[key]) for item in items]))
    return summary


def is_official_compatible_roundtrip(roundtrip: Mapping[str, float | int]) -> bool:
    if not roundtrip:
        return False
    return (
        float(roundtrip.get("root_angle_deg", float("inf"))) < 1e-3
        and float(roundtrip.get("metric_nonroot_local_angle_deg", float("inf"))) < 1e-3
        and float(roundtrip.get("translation_delta_rmse_m", float("inf"))) < 1e-4
    )


def aggregate_translation_drift_by_sequence(
    sequence_stats: Iterable[dict[int, dict[str, float | int]]],
    window_sizes: Iterable[int] = DEFAULT_WINDOW_SIZES,
) -> dict[int, dict[str, float | int]]:
    """按 GlobalPose `test.py` 的方式，先取每条序列均值，再跨序列平均。"""

    stats_list = list(sequence_stats)
    aggregate: dict[int, dict[str, float | int]] = {}
    for window_size in window_sizes:
        window = int(window_size)
        valid = [item[window] for item in stats_list if window in item and int(item[window]["count"]) > 0]
        if not valid:
            aggregate[window] = {
                "mean_m": float("nan"),
                "std_m": float("nan"),
                "drift_percent": float("nan"),
                "sequence_count": 0,
                "pair_count": 0,
            }
            continue
        mean_m = float(np.mean([float(item["mean_m"]) for item in valid]))
        std_m = float(np.mean([float(item["std_m"]) for item in valid]))
        aggregate[window] = {
            "mean_m": mean_m,
            "std_m": std_m,
            "drift_percent": mean_m / float(window) * 100.0,
            "sequence_count": len(valid),
            "pair_count": int(sum(int(item["count"]) for item in valid)),
        }
    return aggregate


# endregion


# region CLI orchestration


def evaluate_results_dir(args: argparse.Namespace) -> dict[str, Any]:
    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    targets = load_globalpose_target_motions(args.globalpose_dataset)
    result_paths = collect_rollout_result_paths(results_dir)
    if int(args.limit) > 0:
        result_paths = result_paths[: int(args.limit)]
    if not result_paths:
        raise RuntimeError(f"没有找到 rollout result npz：{results_dir}")

    pose_metric_arrays: list[np.ndarray] = []
    translation_stats = []
    roundtrip_items: list[dict[str, float | int]] = []
    per_sequence: list[dict[str, Any]] = []
    for result_path in result_paths:
        payload = load_npz_payload(result_path)
        prediction = motion_from_rollout_payload(payload, kind="predicted", schema_name=args.schema)
        if prediction.sequence_name not in targets:
            raise KeyError(f"{result_path} 对应序列 {prediction.sequence_name!r} 不在 GlobalPose dataset 中")
        target = targets[prediction.sequence_name]
        pred_pose, target_pose, pred_tran, target_tran = align_prediction_and_target(
            prediction=prediction,
            target=target,
            trim_eval_mask=bool(args.trim_eval_mask),
        )
        sequence_record: dict[str, Any] = {
            "sequence_name": prediction.sequence_name,
            "result_path": str(result_path),
            "frames": int(pred_pose.shape[0]),
        }
        if payload_has_key(payload, "reference_features_raw") and payload_has_key(payload, "reference_joints_world"):
            reference = motion_from_rollout_payload(payload, kind="reference", schema_name=args.schema)
            roundtrip = compute_roundtrip_diagnostics(
                recovered=reference,
                target=target,
                trim_eval_mask=bool(args.trim_eval_mask),
            )
            roundtrip_items.append(roundtrip)
            sequence_record["roundtrip"] = roundtrip
        if bool(args.evaluate_pose):
            metric_array = compute_motion_evaluator_metrics(
                pose_p=pred_pose,
                pose_t=target_pose,
                tran_p=pred_tran,
                tran_t=target_tran,
                globalpose_repo=args.globalpose_repo,
                device=args.device,
            )
            pose_metric_arrays.append(metric_array)
            sequence_record["pose_metrics"] = metric_array_to_named_dict(metric_array)
        if bool(args.evaluate_tran):
            drift = compute_translation_drift(pred_tran, target_tran, window_sizes=DEFAULT_WINDOW_SIZES)
            translation_stats.append(drift)
            sequence_record["translation_drift"] = stringify_window_keys(
                aggregate_translation_drift_by_sequence([drift], window_sizes=DEFAULT_WINDOW_SIZES)
            )
        per_sequence.append(sequence_record)

    roundtrip_summary = summarize_roundtrip_diagnostics(roundtrip_items)
    summary = {
        "dataset_name": str(args.dataset_name or Path(args.globalpose_dataset).stem),
        "metric_protocol": "globalpose_official_style",
        "input_protocol": "gt_derived_oracle_tracker",
        "official_compatible": is_official_compatible_roundtrip(roundtrip_summary),
        "roundtrip": roundtrip_summary,
        "results_dir": str(results_dir),
        "globalpose_dataset": str(Path(args.globalpose_dataset).resolve()),
        "globalpose_repo": str(Path(args.globalpose_repo).resolve()),
        "sequence_count": len(per_sequence),
        "pose_metrics": aggregate_pose_metric_arrays(pose_metric_arrays),
        "translation_drift": stringify_window_keys(
            aggregate_translation_drift_by_sequence(translation_stats, window_sizes=DEFAULT_WINDOW_SIZES)
        )
        if bool(args.evaluate_tran)
        else {},
        "files": per_sequence,
    }
    write_outputs(summary=summary, output_dir=output_dir)
    return summary


def collect_rollout_result_paths(results_dir: Path) -> list[Path]:
    paths = sorted(results_dir.rglob("unity_stream_long_sequence_result.npz"))
    if paths:
        return paths
    return sorted(results_dir.rglob("*.npz"))


def load_npz_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def align_prediction_and_target(
    prediction: GlobalPoseMotion,
    target: GlobalPoseMotion,
    trim_eval_mask: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_count = min(prediction.pose.shape[0], target.pose.shape[0], prediction.tran.shape[0], target.tran.shape[0])
    pred_pose = prediction.pose[:frame_count]
    target_pose = target.pose[:frame_count]
    pred_tran = prediction.tran[:frame_count]
    target_tran = target.tran[:frame_count]
    if trim_eval_mask and prediction.frame_mask is not None:
        mask = np.asarray(prediction.frame_mask[:frame_count], dtype=bool)
        if mask.any():
            pred_pose = pred_pose[mask]
            target_pose = target_pose[mask]
            pred_tran = pred_tran[mask]
            target_tran = target_tran[mask]
    return pred_pose, target_pose, pred_tran, target_tran


def metric_array_to_named_dict(metric_array: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        name: {"mean": float(metric_array[index, 0]), "std": float(metric_array[index, 1])}
        for index, name in enumerate(MOTION_METRIC_NAMES)
    }


def stringify_window_keys(metrics: dict[int, dict[str, float | int]]) -> dict[str, dict[str, float | int]]:
    return {f"{int(window)}m": dict(value) for window, value in metrics.items()}


def write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "globalpose_metrics_summary.json"
    with summary_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
        file.write("\n")

    csv_path = output_dir / "globalpose_metrics_per_sequence.csv"
    rows = []
    for item in summary.get("files", []):
        row = {"sequence_name": item.get("sequence_name", ""), "frames": item.get("frames", 0)}
        for name, value in item.get("pose_metrics", {}).items():
            row[f"{name} mean"] = value["mean"]
            row[f"{name} std"] = value["std"]
        for window_name, value in item.get("translation_drift", {}).items():
            row[f"{window_name} mean_m"] = value["mean_m"]
            row[f"{window_name} drift_percent"] = value["drift_percent"]
        rows.append(row)
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    summary = evaluate_results_dir(args)
    print(
        "[globalpose_official_metrics_adapter] "
        f"sequences={summary['sequence_count']} output={Path(args.output_dir).resolve()}"
    )
    return summary


# endregion


if __name__ == "__main__":
    main()
