from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from data_loaders.realtime_pose_kinematics import SMPL_JOINT_NAMES, SMPL_PARENTS
from data_loaders.sensor_masking import STATIONARY_JOINT_INDICES


STATIONARY_JOINT_NAMES = ("pelvis", "left_foot", "right_foot", "left_hand", "right_hand")
STATIONARY_JOINT_COLORS = ("#64748b", "#dc2626", "#16a34a", "#7c3aed", "#ea580c")
STATIONARY_PROB_DIM = 5
SMPL_JOINT_COUNT = 24
DEFAULT_FPS = 60.0
DEFAULT_WINDOW_SIZE = 300
DEFAULT_WINDOW_STRIDE = 150
DEFAULT_GT_LOW_THRESHOLD = 0.3
DEFAULT_GT_HIGH_THRESHOLD = 0.7
DEFAULT_PRED_THRESHOLD = 0.5
DEFAULT_GLOBALPOSE_REPO = "dataset/external/globalpose/repo"
DEFAULT_GLOBALPOSE_DATASET = "dataset/external/globalpose/repo/data/test_datasets/totalcapture_officalib.pt"
DEFAULT_GLOBALPOSE_RESULT = (
    "dataset/external/globalpose/repo/data/temp/results/TotalCapture (Official Calibration)_GlobalPose.pt"
)
DEFAULT_GT_SOURCE_DIR = (
    "dataset/generated/sources/realtime_pose_stationary5_v1/"
    "globalpose_totalcapture_officalib_oracle_tracker/totalcapture_officalib"
)
DEFAULT_OUTPUT_DIR = "output/benchmark_globalpose/stationary_compare/totalcapture_officalib_globalpose_vs_gt"


# region CLI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare GlobalPose stationary probabilities against GT stationary_prob_5 and build an HTML report."
    )
    paths = parser.add_argument_group("paths")
    paths.add_argument("--globalpose_repo", default=DEFAULT_GLOBALPOSE_REPO, type=str)
    paths.add_argument("--globalpose_dataset", default=DEFAULT_GLOBALPOSE_DATASET, type=str)
    paths.add_argument("--globalpose_result", default=DEFAULT_GLOBALPOSE_RESULT, type=str)
    paths.add_argument("--gt_source_dir", default=DEFAULT_GT_SOURCE_DIR, type=str)
    paths.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, type=str)

    run = parser.add_argument_group("run")
    run.add_argument("--dataset_name", default="TotalCapture Official Calibration", type=str)
    run.add_argument("--device", default="", type=str, help="GlobalPose device. Empty means cuda if available, else cpu.")
    run.add_argument("--sequences", default="", type=str, help="Comma-separated sequence names. Empty means all.")
    run.add_argument("--limit", default=0, type=int)
    run.add_argument("--overwrite_cache", default=False, type=str2bool)
    run.add_argument(
        "--fast_stationary_only",
        default=True,
        type=str2bool,
        help="Stop each GlobalPose forward pass right after vrnet.linear2 to skip physics updates.",
    )
    run.add_argument("--skip_dump", default=False, type=str2bool, help="Only use existing cache files.")
    run.add_argument("--include_skeleton", default=False, type=str2bool, help="Embed GT/GlobalPose skeleton data in detail pages.")
    run.add_argument(
        "--skeleton_sequences",
        default="",
        type=str,
        help="Comma-separated sequence names that should embed skeleton data. Empty means all evaluated sequences when enabled.",
    )

    metrics = parser.add_argument_group("metrics")
    metrics.add_argument("--fps", default=DEFAULT_FPS, type=float)
    metrics.add_argument("--window_size", default=DEFAULT_WINDOW_SIZE, type=int)
    metrics.add_argument("--window_stride", default=DEFAULT_WINDOW_STRIDE, type=int)
    metrics.add_argument("--gt_low_threshold", default=DEFAULT_GT_LOW_THRESHOLD, type=float)
    metrics.add_argument("--gt_high_threshold", default=DEFAULT_GT_HIGH_THRESHOLD, type=float)
    metrics.add_argument("--pred_threshold", default=DEFAULT_PRED_THRESHOLD, type=float)
    metrics.add_argument("--top_windows_per_sequence", default=20, type=int)
    metrics.add_argument("--skeleton_decimals", default=4, type=int)
    return parser


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = evaluate_globalpose_stationary_compare(args)
    print(
        "[globalpose_stationary_compare] "
        f"sequences={result['summary']['sequence_count']} report={result['paths']['report_index']}"
    )
    return result


# endregion


# region Core pipeline


def evaluate_globalpose_stationary_compare(args: argparse.Namespace) -> dict[str, Any]:
    globalpose_repo = Path(args.globalpose_repo).resolve()
    globalpose_dataset = Path(args.globalpose_dataset).resolve()
    globalpose_result = Path(args.globalpose_result).resolve()
    gt_source_dir = Path(args.gt_source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    cache_dir = output_dir / "cache" / "globalpose_stationary_prob"
    metrics_dir = output_dir / "metrics"
    report_dir = output_dir / "report"

    sequence_names = resolve_sequence_names(
        gt_source_dir=gt_source_dir,
        requested_sequences=parse_sequence_arg(str(args.sequences)),
        limit=int(args.limit),
    )
    if not bool(args.skip_dump):
        dump_missing_globalpose_stationary_prob(
            globalpose_repo=globalpose_repo,
            globalpose_dataset=globalpose_dataset,
            cache_dir=cache_dir,
            sequence_names=sequence_names,
            overwrite_cache=bool(args.overwrite_cache),
            device=str(args.device).strip(),
            fast_stationary_only=bool(args.fast_stationary_only),
        )

    skeleton_payloads_by_sequence: dict[str, Any] = {}
    skeleton_sequence_names = parse_sequence_arg(str(args.skeleton_sequences))
    if bool(args.include_skeleton):
        if not skeleton_sequence_names:
            skeleton_sequence_names = list(sequence_names)
        missing_skeleton_names = sorted(set(skeleton_sequence_names) - set(sequence_names))
        if missing_skeleton_names:
            raise KeyError(f"Skeleton sequences are not part of this evaluation run: {missing_skeleton_names}")
        skeleton_payloads_by_sequence = build_skeleton_payloads_for_sequences(
            globalpose_repo=globalpose_repo,
            globalpose_dataset=globalpose_dataset,
            globalpose_result=globalpose_result,
            gt_source_dir=gt_source_dir,
            sequence_names=skeleton_sequence_names,
            device=str(args.device).strip(),
            decimals=int(args.skeleton_decimals),
        )

    result = evaluate_cached_sequences(
        gt_source_dir=gt_source_dir,
        cache_dir=cache_dir,
        output_dir=output_dir,
        dataset_name=str(args.dataset_name),
        fps=float(args.fps),
        window_size=int(args.window_size),
        window_stride=int(args.window_stride),
        gt_low_threshold=float(args.gt_low_threshold),
        gt_high_threshold=float(args.gt_high_threshold),
        pred_threshold=float(args.pred_threshold),
        top_windows_per_sequence=int(args.top_windows_per_sequence),
        sequence_names=sequence_names,
        skeleton_payloads_by_sequence=skeleton_payloads_by_sequence,
    )
    result["paths"].update(
        {
            "globalpose_repo": str(globalpose_repo),
            "globalpose_dataset": str(globalpose_dataset),
            "globalpose_result": str(globalpose_result),
            "gt_source_dir": str(gt_source_dir),
            "cache_dir": str(cache_dir),
            "metrics_dir": str(metrics_dir),
            "report_dir": str(report_dir),
        }
    )
    return result


def evaluate_cached_sequences(
    *,
    gt_source_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    dataset_name: str,
    fps: float = DEFAULT_FPS,
    window_size: int = DEFAULT_WINDOW_SIZE,
    window_stride: int = DEFAULT_WINDOW_STRIDE,
    gt_low_threshold: float = DEFAULT_GT_LOW_THRESHOLD,
    gt_high_threshold: float = DEFAULT_GT_HIGH_THRESHOLD,
    pred_threshold: float = DEFAULT_PRED_THRESHOLD,
    top_windows_per_sequence: int = 20,
    sequence_names: list[str] | None = None,
    skeleton_payloads_by_sequence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gt_source_dir = Path(gt_source_dir)
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    report_dir = output_dir / "report"
    thresholds = {
        "gt_low": float(gt_low_threshold),
        "gt_high": float(gt_high_threshold),
        "pred": float(pred_threshold),
    }
    names = sequence_names or resolve_sequence_names(gt_source_dir=gt_source_dir, requested_sequences=[], limit=0)
    sequence_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    detail_payloads: list[dict[str, Any]] = []
    all_gt: list[np.ndarray] = []
    all_gp: list[np.ndarray] = []
    skeleton_payloads_by_sequence = skeleton_payloads_by_sequence or {}

    for sequence_name in names:
        gt = read_gt_stationary_prob(gt_source_dir=gt_source_dir, sequence_name=sequence_name)
        gp = read_globalpose_stationary_prob(cache_dir=cache_dir, sequence_name=sequence_name)
        assert_prob_shapes_match(sequence_name, gt, gp)
        metrics = compute_compare_metrics(
            gt_stationary_prob_5=gt,
            globalpose_stationary_prob_5=gp,
            gt_low_threshold=gt_low_threshold,
            gt_high_threshold=gt_high_threshold,
            pred_threshold=pred_threshold,
        )
        windows = compute_window_metrics(
            sequence_name=sequence_name,
            gt_stationary_prob_5=gt,
            globalpose_stationary_prob_5=gp,
            window_size=window_size,
            window_stride=window_stride,
            gt_low_threshold=gt_low_threshold,
            gt_high_threshold=gt_high_threshold,
            pred_threshold=pred_threshold,
        )
        top_windows = sorted(windows, key=lambda row: float(row["foot_bce"]), reverse=True)[: max(0, top_windows_per_sequence)]
        sequence_row = {"sequence": sequence_name, **metrics}
        sequence_rows.append(sequence_row)
        window_rows.extend(windows)
        detail_payloads.append(
            build_sequence_report_payload(
                sequence_name=sequence_name,
                fps=fps,
                gt_stationary_prob_5=gt,
                globalpose_stationary_prob_5=gp,
                metrics=metrics,
                bad_windows=top_windows,
                thresholds=thresholds,
                skeleton_payload=skeleton_payloads_by_sequence.get(sequence_name),
            )
        )
        all_gt.append(gt)
        all_gp.append(gp)

    if not all_gt:
        raise RuntimeError("No sequences to evaluate.")

    gt_all = np.concatenate(all_gt, axis=0)
    gp_all = np.concatenate(all_gp, axis=0)
    overall_metrics = compute_compare_metrics(
        gt_stationary_prob_5=gt_all,
        globalpose_stationary_prob_5=gp_all,
        gt_low_threshold=gt_low_threshold,
        gt_high_threshold=gt_high_threshold,
        pred_threshold=pred_threshold,
    )
    summary = {
        "dataset_name": dataset_name,
        "sequence_count": len(sequence_rows),
        "frames": int(gt_all.shape[0]),
        "fps": float(fps),
        "window_size": int(window_size),
        "window_stride": int(window_stride),
        "thresholds": thresholds,
        "overall_metrics": overall_metrics,
        "globalpose_prob_stats": array_stats(gp_all),
        "gt_prob_stats": array_stats(gt_all),
        "top_by_foot_bce": sorted(sequence_rows, key=lambda row: float(row["foot_bce"]), reverse=True)[:15],
        "top_by_foot_source_low_globalpose_high": sorted(
            sequence_rows,
            key=lambda row: (float(row["foot_source_low_globalpose_high_rate"]), float(row["foot_bce"])),
            reverse=True,
        )[:15],
        "top_by_foot_source_high_globalpose_low": sorted(
            sequence_rows,
            key=lambda row: (float(row["foot_source_high_globalpose_low_rate"]), float(row["foot_bce"])),
            reverse=True,
        )[:15],
    }
    paths = write_outputs(
        output_dir=output_dir,
        metrics_dir=metrics_dir,
        report_dir=report_dir,
        summary=summary,
        sequence_rows=sequence_rows,
        window_rows=window_rows,
        detail_payloads=detail_payloads,
    )
    return {"summary": summary, "paths": paths}


# endregion


# region GlobalPose dump


class _StopAfterStationaryLogits(Exception):
    pass


def dump_missing_globalpose_stationary_prob(
    *,
    globalpose_repo: Path,
    globalpose_dataset: Path,
    cache_dir: Path,
    sequence_names: list[str],
    overwrite_cache: bool,
    device: str,
    fast_stationary_only: bool,
) -> None:
    missing = [
        sequence_name
        for sequence_name in sequence_names
        if overwrite_cache or not cache_path(cache_dir=cache_dir, sequence_name=sequence_name).exists()
    ]
    if not missing:
        return
    dump_globalpose_stationary_prob(
        globalpose_repo=globalpose_repo,
        globalpose_dataset=globalpose_dataset,
        cache_dir=cache_dir,
        sequence_names=missing,
        device=device,
        fast_stationary_only=fast_stationary_only,
    )


def dump_globalpose_stationary_prob(
    *,
    globalpose_repo: Path,
    globalpose_dataset: Path,
    cache_dir: Path,
    sequence_names: list[str],
    device: str = "",
    fast_stationary_only: bool = True,
) -> list[Path]:
    if not globalpose_repo.exists():
        raise FileNotFoundError(f"GlobalPose repo does not exist: {globalpose_repo}")
    if not globalpose_dataset.exists():
        raise FileNotFoundError(f"GlobalPose dataset does not exist: {globalpose_dataset}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # GlobalPose loads weights and SMPL assets through relative paths, so keep its repo as cwd while importing/running.
    with pushd(globalpose_repo), globalpose_import_context(globalpose_repo):
        try:
            import torch
            import articulate as art
            from net import GPNet
        except Exception as exc:
            raise RuntimeError(
                "Failed to import GlobalPose runtime. Run this dump step in the GlobalPose-compatible "
                "environment, e.g. conda run --no-capture-output -n globalpose5070 python -m "
                "eval.globalpose_stationary_compare ..."
            ) from exc

        try:
            dataset = torch.load(globalpose_dataset, map_location="cpu", weights_only=False)
        except TypeError:
            dataset = torch.load(globalpose_dataset, map_location="cpu")
        dataset_names = [str(value) for value in dataset["name"]]
        index_by_name = {name: index for index, name in enumerate(dataset_names)}
        missing_names = [name for name in sequence_names if name not in index_by_name]
        if missing_names:
            raise KeyError(f"GlobalPose dataset does not contain sequences: {missing_names}")

        run_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = GPNet().eval().to(run_device)
        captured_logits: list[Any] = []

        def hook(_module, _inputs, output):
            value = output.detach().cpu().reshape(-1)
            if value.numel() >= 9:
                captured_logits.append(value[4:9].clone())
            if fast_stationary_only:
                raise _StopAfterStationaryLogits()

        handle = net.vrnet.linear2.register_forward_hook(hook)
        written: list[Path] = []
        gravity = torch.tensor([0, -9.8, 0])
        try:
            with torch.no_grad():
                for sequence_name in sequence_names:
                    sequence_index = index_by_name[sequence_name]
                    frame_count = int(dataset["pose"][sequence_index].shape[0])
                    print(
                        "[globalpose_stationary_compare] dump "
                        f"{sequence_name} frames={frame_count} device={run_device}",
                        flush=True,
                    )
                    start_capture = len(captured_logits)
                    a_s = dataset["aS"][sequence_index]
                    w_s = dataset["wS"][sequence_index]
                    r_is = dataset["RIS"][sequence_index]
                    r_im = dataset["RIM"][sequence_index]
                    r_sb = dataset["RSB"][sequence_index]
                    pose = art.math.axis_angle_to_rotation_matrix(dataset["pose"][sequence_index]).view(-1, 24, 3, 3)
                    r_mb = r_im.transpose(1, 2).matmul(r_is).matmul(r_sb).to(run_device)
                    a_m = (r_im.transpose(1, 2).matmul(r_is).matmul(a_s.unsqueeze(-1)).squeeze(-1) + gravity).to(
                        run_device
                    )
                    w_m = r_im.transpose(1, 2).matmul(r_is).matmul(w_s.unsqueeze(-1)).squeeze(-1).to(run_device)

                    net.rnn_initialize(pose[0])
                    for frame_index in range(frame_count):
                        try:
                            net.forward_frame(a_m[frame_index], w_m[frame_index], r_mb[frame_index])
                        except _StopAfterStationaryLogits:
                            pass

                    logits = torch.stack(captured_logits[start_capture:]).numpy().astype(np.float32)
                    if logits.shape != (frame_count, STATIONARY_PROB_DIM):
                        raise ValueError(f"{sequence_name} logits shape mismatch: {logits.shape}")
                    prob = sigmoid(logits).astype(np.float32)
                    output_path = cache_path(cache_dir=cache_dir, sequence_name=sequence_name)
                    np.savez(
                        output_path,
                        sequence_name=np.asarray(sequence_name),
                        globalpose_stationary_logits_5=logits,
                        globalpose_stationary_prob_5=prob,
                    )
                    written.append(output_path)
        finally:
            handle.remove()
        return written


@contextmanager
def pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def globalpose_import_context(globalpose_repo: Path):
    added_dll_dirs = []
    for value in (globalpose_repo / "carticulate", globalpose_repo / "carticulate_py38_backup_20260701_164406"):
        if value.exists() and hasattr(os, "add_dll_directory"):
            added_dll_dirs.append(os.add_dll_directory(str(value)))
    inserted = False
    repo_string = str(globalpose_repo)
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(repo_string)
            except ValueError:
                pass
        for directory in added_dll_dirs:
            directory.close()


# endregion


# region Skeleton payloads


def build_skeleton_payloads_for_sequences(
    *,
    globalpose_repo: Path,
    globalpose_dataset: Path,
    globalpose_result: Path,
    gt_source_dir: Path,
    sequence_names: list[str],
    device: str = "",
    decimals: int = 4,
) -> dict[str, Any]:
    if not sequence_names:
        return {}
    gt_joints_by_sequence = load_globalpose_dataset_joints_world(
        globalpose_repo=globalpose_repo,
        globalpose_dataset=globalpose_dataset,
        sequence_names=sequence_names,
        device=device,
    )
    globalpose_joints_by_sequence = load_globalpose_result_joints_world(
        globalpose_repo=globalpose_repo,
        globalpose_dataset=globalpose_dataset,
        globalpose_result=globalpose_result,
        sequence_names=sequence_names,
        device=device,
    )
    payloads: dict[str, Any] = {}
    for sequence_name in sequence_names:
        gt_joints = gt_joints_by_sequence[sequence_name]
        globalpose_joints = globalpose_joints_by_sequence[sequence_name]
        payloads[sequence_name] = build_skeleton_payload(
            gt_joints_world=gt_joints,
            globalpose_joints_world=globalpose_joints,
            decimals=decimals,
        )
    return payloads


def load_globalpose_result_joints_world(
    *,
    globalpose_repo: Path,
    globalpose_dataset: Path,
    globalpose_result: Path,
    sequence_names: list[str],
    device: str = "",
    chunk_size: int = 2048,
) -> dict[str, np.ndarray]:
    if not globalpose_repo.exists():
        raise FileNotFoundError(f"GlobalPose repo does not exist: {globalpose_repo}")
    if not globalpose_dataset.exists():
        raise FileNotFoundError(f"GlobalPose dataset does not exist: {globalpose_dataset}")
    if not globalpose_result.exists():
        raise FileNotFoundError(f"GlobalPose result does not exist: {globalpose_result}")

    with pushd(globalpose_repo), globalpose_import_context(globalpose_repo):
        try:
            import torch
            import articulate as art
        except Exception as exc:
            raise RuntimeError(
                "Failed to import GlobalPose runtime for skeleton FK. Use the GlobalPose-compatible "
                "environment, e.g. conda run --no-capture-output -n globalpose5070 python -m "
                "eval.globalpose_stationary_compare --include_skeleton true ..."
            ) from exc

        dataset = torch_load_cpu(torch, globalpose_dataset)
        result = torch_load_cpu(torch, globalpose_result)
        dataset_names = [str(value) for value in dataset["name"]]
        index_by_name = {name: index for index, name in enumerate(dataset_names)}
        missing_names = [name for name in sequence_names if name not in index_by_name]
        if missing_names:
            raise KeyError(f"GlobalPose dataset does not contain sequences: {missing_names}")
        if "pose" not in result or "tran" not in result:
            raise KeyError(f"{globalpose_result} must contain pose and tran")

        run_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = globalpose_repo / "models" / "SMPL_male.pkl"
        model = art.ParametricModel(str(model_path), device=run_device)
        joints_by_sequence: dict[str, np.ndarray] = {}
        with torch.no_grad():
            for sequence_name in sequence_names:
                sequence_index = index_by_name[sequence_name]
                pose = to_globalpose_rotation_tensor(torch, art, result["pose"][sequence_index], run_device)
                tran = torch.as_tensor(result["tran"][sequence_index], dtype=torch.float32, device=run_device).view(-1, 3)
                if int(pose.shape[0]) != int(tran.shape[0]):
                    raise ValueError(f"{sequence_name}: GlobalPose result pose/tran frame mismatch {pose.shape} vs {tran.shape}")
                chunks: list[np.ndarray] = []
                for start in range(0, int(pose.shape[0]), max(1, int(chunk_size))):
                    end = min(start + max(1, int(chunk_size)), int(pose.shape[0]))
                    _, joints = model.forward_kinematics(pose[start:end], tran=tran[start:end], calc_mesh=False)
                    chunks.append(joints.detach().cpu().numpy().astype(np.float32))
                joints_smpl = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, SMPL_JOINT_COUNT, 3), dtype=np.float32)
                joints_by_sequence[sequence_name] = globalpose_fk_joints_to_report_world(joints_smpl)
        return joints_by_sequence


def load_globalpose_dataset_joints_world(
    *,
    globalpose_repo: Path,
    globalpose_dataset: Path,
    sequence_names: list[str],
    device: str = "",
    chunk_size: int = 2048,
) -> dict[str, np.ndarray]:
    if not globalpose_repo.exists():
        raise FileNotFoundError(f"GlobalPose repo does not exist: {globalpose_repo}")
    if not globalpose_dataset.exists():
        raise FileNotFoundError(f"GlobalPose dataset does not exist: {globalpose_dataset}")

    with pushd(globalpose_repo), globalpose_import_context(globalpose_repo):
        try:
            import torch
            import articulate as art
        except Exception as exc:
            raise RuntimeError(
                "Failed to import GlobalPose runtime for official GT skeleton FK. Use the GlobalPose-compatible "
                "environment, e.g. conda run --no-capture-output -n globalpose5070 python -m "
                "eval.globalpose_stationary_compare --include_skeleton true ..."
            ) from exc

        dataset = torch_load_cpu(torch, globalpose_dataset)
        dataset_names = [str(value) for value in dataset["name"]]
        index_by_name = {name: index for index, name in enumerate(dataset_names)}
        missing_names = [name for name in sequence_names if name not in index_by_name]
        if missing_names:
            raise KeyError(f"GlobalPose dataset does not contain sequences: {missing_names}")
        if "pose" not in dataset or "tran" not in dataset:
            raise KeyError(f"{globalpose_dataset} must contain pose and tran")

        run_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = globalpose_repo / "models" / "SMPL_male.pkl"
        model = art.ParametricModel(str(model_path), device=run_device)
        joints_by_sequence: dict[str, np.ndarray] = {}
        with torch.no_grad():
            for sequence_name in sequence_names:
                sequence_index = index_by_name[sequence_name]
                pose = to_globalpose_rotation_tensor(torch, art, dataset["pose"][sequence_index], run_device)
                tran = torch.as_tensor(dataset["tran"][sequence_index], dtype=torch.float32, device=run_device).view(-1, 3)
                if int(pose.shape[0]) != int(tran.shape[0]):
                    raise ValueError(f"{sequence_name}: GlobalPose dataset pose/tran frame mismatch {pose.shape} vs {tran.shape}")
                chunks: list[np.ndarray] = []
                for start in range(0, int(pose.shape[0]), max(1, int(chunk_size))):
                    end = min(start + max(1, int(chunk_size)), int(pose.shape[0]))
                    _, joints = model.forward_kinematics(pose[start:end], tran=tran[start:end], calc_mesh=False)
                    chunks.append(joints.detach().cpu().numpy().astype(np.float32))
                joints_smpl = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, SMPL_JOINT_COUNT, 3), dtype=np.float32)
                joints_by_sequence[sequence_name] = globalpose_fk_joints_to_report_world(joints_smpl)
        return joints_by_sequence


def globalpose_fk_joints_to_report_world(joints: np.ndarray) -> np.ndarray:
    """GlobalPose 官方 result 的 FK 输出已经是 y-up 坐标，报告里不能再做 AMASS y/z 交换。"""

    return as_joint_world_array("globalpose_fk_joints", joints).astype(np.float32, copy=True)


def torch_load_cpu(torch_module: Any, path: Path) -> Any:
    try:
        return torch_module.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location="cpu")


def to_globalpose_rotation_tensor(torch_module: Any, art_module: Any, value: Any, device: Any) -> Any:
    pose = torch_module.as_tensor(value, dtype=torch_module.float32, device=device)
    if pose.ndim == 4 and tuple(pose.shape[1:]) == (SMPL_JOINT_COUNT, 3, 3):
        return pose
    if pose.ndim == 3 and tuple(pose.shape[1:]) == (SMPL_JOINT_COUNT, 3):
        return art_module.math.axis_angle_to_rotation_matrix(pose).view(-1, SMPL_JOINT_COUNT, 3, 3)
    if pose.ndim == 2 and int(pose.shape[1]) == SMPL_JOINT_COUNT * 3:
        return art_module.math.axis_angle_to_rotation_matrix(pose.view(-1, SMPL_JOINT_COUNT, 3)).view(
            -1, SMPL_JOINT_COUNT, 3, 3
        )
    raise ValueError(f"GlobalPose result pose must be rotation matrices or axis-angle, got {tuple(pose.shape)}")


def build_skeleton_payload(
    *,
    gt_joints_world: np.ndarray,
    globalpose_joints_world: np.ndarray,
    decimals: int = 4,
    windows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    gt = as_joint_world_array("gt_joints_world", gt_joints_world)
    gp = as_joint_world_array("globalpose_joints_world", globalpose_joints_world)
    if gt.shape != gp.shape:
        raise ValueError(f"GT and GlobalPose skeleton arrays must match: {gt.shape} vs {gp.shape}")
    return {
        "frame_count": int(gt.shape[0]),
        "joint_names": list(SMPL_JOINT_NAMES),
        "parents": [int(value) for value in SMPL_PARENTS.tolist()],
        "skeleton_edges": [
            [int(parent_index), int(joint_index)]
            for joint_index, parent_index in enumerate(SMPL_PARENTS.tolist())
            if int(parent_index) >= 0
        ],
        "stationary_joint_indices": [int(index) for index in STATIONARY_JOINT_INDICES],
        "camera_bounds": build_skeleton_camera_bounds(
            gt_joints_world=gt,
            globalpose_joints_world=gp,
            windows=windows,
            decimals=decimals,
        ),
        "gt_joints_world": round_nested_array(gt, decimals=decimals),
        "globalpose_joints_world": round_nested_array(gp, decimals=decimals),
    }


def build_skeleton_camera_bounds(
    *,
    gt_joints_world: np.ndarray,
    globalpose_joints_world: np.ndarray,
    windows: list[dict[str, Any]] | None = None,
    decimals: int = 4,
) -> dict[str, Any]:
    gt = as_joint_world_array("gt_joints_world", gt_joints_world)
    gp = as_joint_world_array("globalpose_joints_world", globalpose_joints_world)
    if gt.shape != gp.shape:
        raise ValueError(f"GT and GlobalPose skeleton arrays must match: {gt.shape} vs {gp.shape}")

    camera_bounds: dict[str, Any] = {
        "sequence": skeleton_bounds_for_range(gt=gt, gp=gp, start=0, end=gt.shape[0], pelvis_aligned=False, decimals=decimals),
        "pelvis": skeleton_bounds_for_range(gt=gt, gp=gp, start=0, end=gt.shape[0], pelvis_aligned=True, decimals=decimals),
        "windows": [],
    }
    for row in windows or []:
        frame_start = max(0, int(row.get("frame_start", 0)))
        frame_end = min(int(row.get("frame_end", gt.shape[0] - 1)), gt.shape[0] - 1)
        if frame_start > frame_end:
            continue
        window_bounds = skeleton_bounds_for_range(
            gt=gt,
            gp=gp,
            start=frame_start,
            end=frame_end + 1,
            pelvis_aligned=False,
            decimals=decimals,
        )
        window_bounds["frame_start"] = int(frame_start)
        window_bounds["frame_end"] = int(frame_end)
        camera_bounds["windows"].append(window_bounds)
    return camera_bounds


def skeleton_bounds_for_range(
    *,
    gt: np.ndarray,
    gp: np.ndarray,
    start: int,
    end: int,
    pelvis_aligned: bool,
    decimals: int,
) -> dict[str, Any]:
    gt_slice = np.asarray(gt[start:end], dtype=np.float32)
    gp_slice = np.asarray(gp[start:end], dtype=np.float32)
    if pelvis_aligned:
        gt_slice = gt_slice - gt_slice[:, :1]
        gp_slice = gp_slice - gp_slice[:, :1]
    combined = np.concatenate([gt_slice.reshape(-1, 3), gp_slice.reshape(-1, 3)], axis=0)
    return {
        "front": skeleton_projected_bounds(combined[:, [0, 1]], decimals=decimals),
        "side": skeleton_projected_bounds(combined[:, [2, 1]], decimals=decimals),
        "top": skeleton_projected_bounds(combined[:, [0, 2]], decimals=decimals),
    }


def skeleton_projected_bounds(points_2d: np.ndarray, decimals: int) -> dict[str, list[float]]:
    points = np.asarray(points_2d, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError(f"points_2d must be [N,2], got {points.shape}")
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    return {
        "min": round_nested_array(minimum, decimals=decimals),
        "max": round_nested_array(maximum, decimals=decimals),
    }


def as_joint_world_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[1:] != (SMPL_JOINT_COUNT, 3):
        raise ValueError(f"{name} must be [T,24,3] or [1,T,24,3], got {array.shape}")
    return array


# endregion


# region Metrics


def compute_compare_metrics(
    *,
    gt_stationary_prob_5: np.ndarray,
    globalpose_stationary_prob_5: np.ndarray,
    gt_low_threshold: float = DEFAULT_GT_LOW_THRESHOLD,
    gt_high_threshold: float = DEFAULT_GT_HIGH_THRESHOLD,
    pred_threshold: float = DEFAULT_PRED_THRESHOLD,
) -> dict[str, float | int]:
    gt = as_prob_matrix("gt_stationary_prob_5", gt_stationary_prob_5)
    pred = as_prob_matrix("globalpose_stationary_prob_5", globalpose_stationary_prob_5)
    if gt.shape != pred.shape:
        raise ValueError(f"GT and GlobalPose stationary arrays must match: {gt.shape} vs {pred.shape}")
    abs_error = np.abs(pred - gt)
    foot_gt = gt[:, 1:3]
    foot_pred = pred[:, 1:3]
    source_low_globalpose_high = (gt < float(gt_low_threshold)) & (pred > float(pred_threshold))
    source_high_globalpose_low = (gt > float(gt_high_threshold)) & (pred < float(pred_threshold))
    foot_source_low_globalpose_high = (foot_gt < float(gt_low_threshold)) & (foot_pred > float(pred_threshold))
    foot_source_high_globalpose_low = (foot_gt > float(gt_high_threshold)) & (foot_pred < float(pred_threshold))
    return {
        "frames": int(gt.shape[0]),
        "mae": float(np.mean(abs_error)),
        "foot_mae": float(np.mean(np.abs(foot_pred - foot_gt))),
        "bce": float(np.mean(binary_cross_entropy(pred, gt))),
        "foot_bce": float(np.mean(binary_cross_entropy(foot_pred, foot_gt))),
        "source_low_globalpose_high_rate": float(np.mean(source_low_globalpose_high)),
        "source_high_globalpose_low_rate": float(np.mean(source_high_globalpose_low)),
        "foot_source_low_globalpose_high_rate": float(np.mean(foot_source_low_globalpose_high)),
        "foot_source_high_globalpose_low_rate": float(np.mean(foot_source_high_globalpose_low)),
        "gt_active_rate": float(np.mean(gt > gt_high_threshold)),
        "gp_active_rate": float(np.mean(pred > pred_threshold)),
        "foot_gt_active_rate": float(np.mean(foot_gt > gt_high_threshold)),
        "foot_gp_active_rate": float(np.mean(foot_pred > pred_threshold)),
        "gt_prob_mean": float(np.mean(gt)),
        "gp_prob_mean": float(np.mean(pred)),
        "gt_foot_mean": float(np.mean(foot_gt)),
        "gp_foot_mean": float(np.mean(foot_pred)),
    }


def compute_window_metrics(
    *,
    sequence_name: str,
    gt_stationary_prob_5: np.ndarray,
    globalpose_stationary_prob_5: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
    window_stride: int = DEFAULT_WINDOW_STRIDE,
    gt_low_threshold: float = DEFAULT_GT_LOW_THRESHOLD,
    gt_high_threshold: float = DEFAULT_GT_HIGH_THRESHOLD,
    pred_threshold: float = DEFAULT_PRED_THRESHOLD,
) -> list[dict[str, Any]]:
    gt = as_prob_matrix("gt_stationary_prob_5", gt_stationary_prob_5)
    pred = as_prob_matrix("globalpose_stationary_prob_5", globalpose_stationary_prob_5)
    if gt.shape != pred.shape:
        raise ValueError(f"GT and GlobalPose stationary arrays must match: {gt.shape} vs {pred.shape}")
    frame_count = int(gt.shape[0])
    size = max(1, int(window_size))
    stride = max(1, int(window_stride))
    starts = list(range(0, max(frame_count - size + 1, 1), stride))
    if not starts:
        starts = [0]
    last_start = max(0, frame_count - size)
    if starts[-1] != last_start:
        starts.append(last_start)

    rows: list[dict[str, Any]] = []
    for start in starts:
        end = min(start + size, frame_count)
        metrics = compute_compare_metrics(
            gt_stationary_prob_5=gt[start:end],
            globalpose_stationary_prob_5=pred[start:end],
            gt_low_threshold=gt_low_threshold,
            gt_high_threshold=gt_high_threshold,
            pred_threshold=pred_threshold,
        )
        rows.append(
            {
                "sequence": sequence_name,
                "frame_start": int(start),
                "frame_end": int(end - 1),
                **metrics,
            }
        )
    return rows


def as_prob_matrix(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] != STATIONARY_PROB_DIM:
        raise ValueError(f"{name} must be [T,5] or [1,T,5], got {array.shape}")
    return np.clip(array, 0.0, 1.0)


def assert_prob_shapes_match(sequence_name: str, gt: np.ndarray, pred: np.ndarray) -> None:
    if gt.shape != pred.shape:
        raise ValueError(f"{sequence_name}: GT and GlobalPose shapes differ: {gt.shape} vs {pred.shape}")


def binary_cross_entropy(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    pred_clipped = np.clip(np.asarray(pred, dtype=np.float32), eps, 1.0 - eps)
    target_clipped = np.clip(np.asarray(target, dtype=np.float32), 0.0, 1.0)
    return -(target_clipped * np.log(pred_clipped) + (1.0 - target_clipped) * np.log(1.0 - pred_clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    values64 = np.asarray(values, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-values64))).astype(np.float32)


def array_stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float32)
    return {
        "min": float(np.nanmin(array)),
        "mean": float(np.nanmean(array)),
        "p50": float(np.nanpercentile(array, 50)),
        "p95": float(np.nanpercentile(array, 95)),
        "p99": float(np.nanpercentile(array, 99)),
        "max": float(np.nanmax(array)),
    }


# endregion


# region IO


def resolve_sequence_names(gt_source_dir: Path, requested_sequences: list[str], limit: int) -> list[str]:
    if requested_sequences:
        names = list(requested_sequences)
    else:
        names = sorted(path.stem for path in Path(gt_source_dir).glob("*.npz"))
    if int(limit) > 0:
        names = names[: int(limit)]
    if not names:
        raise RuntimeError(f"No GT source npz files found under {gt_source_dir}")
    return names


def parse_sequence_arg(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def read_gt_stationary_prob(*, gt_source_dir: Path, sequence_name: str) -> np.ndarray:
    path = Path(gt_source_dir) / f"{sequence_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing GT stationary source: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "stationary_prob_5" not in data.files:
            raise KeyError(f"{path} does not contain stationary_prob_5")
        return np.asarray(data["stationary_prob_5"], dtype=np.float32)


def read_gt_joints_world(*, gt_source_dir: Path, sequence_name: str) -> np.ndarray:
    path = Path(gt_source_dir) / f"{sequence_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing GT source: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "joints_world" not in data.files:
            raise KeyError(f"{path} does not contain joints_world")
        return as_joint_world_array("joints_world", data["joints_world"])


def read_globalpose_stationary_prob(*, cache_dir: Path, sequence_name: str) -> np.ndarray:
    path = cache_path(cache_dir=cache_dir, sequence_name=sequence_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing GlobalPose stationary cache: {path}")
    with np.load(path, allow_pickle=False) as data:
        for key in ("globalpose_stationary_prob_5", "stationary_prob_5"):
            if key in data.files:
                return np.asarray(data[key], dtype=np.float32)
    raise KeyError(f"{path} does not contain globalpose_stationary_prob_5")


def cache_path(*, cache_dir: Path, sequence_name: str) -> Path:
    return Path(cache_dir) / f"{safe_file_stem(sequence_name)}.npz"


def safe_file_stem(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in str(value))
    return cleaned.strip("._-") or hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:10]


def write_outputs(
    *,
    output_dir: Path,
    metrics_dir: Path,
    report_dir: Path,
    summary: dict[str, Any],
    sequence_rows: list[dict[str, Any]],
    window_rows: list[dict[str, Any]],
    detail_payloads: list[dict[str, Any]],
) -> dict[str, str]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    data_dir = report_dir / "data"
    sequence_dir = report_dir / "sequences"
    data_dir.mkdir(parents=True, exist_ok=True)
    sequence_dir.mkdir(parents=True, exist_ok=True)

    summary_path = metrics_dir / "summary.json"
    per_sequence_path = metrics_dir / "per_sequence.csv"
    per_window_path = metrics_dir / "per_window.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    write_csv(per_sequence_path, sequence_rows, SEQUENCE_FIELDNAMES)
    write_csv(per_window_path, window_rows, WINDOW_FIELDNAMES)

    detail_links: list[dict[str, Any]] = []
    for payload in detail_payloads:
        sequence_name = str(payload["sequence"])
        data_path = data_dir / f"{safe_file_stem(sequence_name)}.json"
        detail_path = sequence_dir / f"{safe_file_stem(sequence_name)}.html"
        data_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8", newline="\n")
        detail_path.write_text(render_sequence_html(payload), encoding="utf-8", newline="\n")
        detail_links.append(
            {
                "sequence": sequence_name,
                "detail_html": f"sequences/{detail_path.name}",
                "data_json": f"data/{data_path.name}",
            }
        )

    index_path = report_dir / "index.html"
    index_payload = {
        "summary": summary,
        "sequence_rows": sorted(sequence_rows, key=lambda row: float(row["foot_bce"]), reverse=True),
        "window_rows": sorted(window_rows, key=lambda row: float(row["foot_bce"]), reverse=True)[:50],
        "detail_links": detail_links,
    }
    index_path.write_text(render_index_html(index_payload), encoding="utf-8", newline="\n")
    return {
        "output_dir": str(output_dir),
        "summary_json": str(summary_path),
        "per_sequence_csv": str(per_sequence_path),
        "per_window_csv": str(per_window_path),
        "report_index": str(index_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], preferred_fieldnames: list[str]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8", newline="")
        return
    extra = sorted({key for row in rows for key in row.keys()} - set(preferred_fieldnames))
    fieldnames = [name for name in preferred_fieldnames if any(name in row for row in rows)] + extra
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


SEQUENCE_FIELDNAMES = [
    "sequence",
    "frames",
    "foot_bce",
    "foot_mae",
    "foot_source_low_globalpose_high_rate",
    "foot_source_high_globalpose_low_rate",
    "foot_gt_active_rate",
    "foot_gp_active_rate",
    "bce",
    "mae",
    "source_low_globalpose_high_rate",
    "source_high_globalpose_low_rate",
    "gt_prob_mean",
    "gp_prob_mean",
    "gt_foot_mean",
    "gp_foot_mean",
]

WINDOW_FIELDNAMES = [
    "sequence",
    "frame_start",
    "frame_end",
    "frames",
    "foot_bce",
    "foot_mae",
    "foot_source_low_globalpose_high_rate",
    "foot_source_high_globalpose_low_rate",
    "foot_gt_active_rate",
    "foot_gp_active_rate",
    "bce",
    "mae",
    "source_low_globalpose_high_rate",
    "source_high_globalpose_low_rate",
]


# endregion


# region Report rendering


def build_sequence_report_payload(
    *,
    sequence_name: str,
    fps: float,
    gt_stationary_prob_5: np.ndarray,
    globalpose_stationary_prob_5: np.ndarray,
    metrics: dict[str, Any],
    bad_windows: list[dict[str, Any]],
    thresholds: dict[str, float],
    skeleton_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gt = as_prob_matrix("gt_stationary_prob_5", gt_stationary_prob_5)
    gp = as_prob_matrix("globalpose_stationary_prob_5", globalpose_stationary_prob_5)
    payload = {
        "sequence": sequence_name,
        "fps": float(fps),
        "frame_count": int(gt.shape[0]),
        "joint_names": list(STATIONARY_JOINT_NAMES),
        "joint_colors": list(STATIONARY_JOINT_COLORS),
        "thresholds": thresholds,
        "metrics": metrics,
        "bad_windows": bad_windows,
        "gt_stationary_prob_5": round_nested_array(gt, decimals=5),
        "globalpose_stationary_prob_5": round_nested_array(gp, decimals=5),
    }
    if skeleton_payload is not None:
        if int(skeleton_payload.get("frame_count", -1)) != int(gt.shape[0]):
            raise ValueError(
                f"{sequence_name}: skeleton frame_count {skeleton_payload.get('frame_count')} "
                f"does not match stationary frame_count {gt.shape[0]}"
            )
        skeleton = dict(skeleton_payload)
        if "gt_joints_world" in skeleton and "globalpose_joints_world" in skeleton:
            skeleton["camera_bounds"] = build_skeleton_camera_bounds(
                gt_joints_world=np.asarray(skeleton["gt_joints_world"], dtype=np.float32),
                globalpose_joints_world=np.asarray(skeleton["globalpose_joints_world"], dtype=np.float32),
                windows=bad_windows,
                decimals=4,
            )
        payload["skeleton"] = skeleton
    return payload


def render_index_html(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    links = {item["sequence"]: item["detail_html"] for item in payload["detail_links"]}
    sequence_rows = []
    for index, row in enumerate(payload["sequence_rows"], start=1):
        sequence_name = str(row["sequence"])
        link = links.get(sequence_name, "#")
        sequence_rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td><a href=\"{escape(link)}\">{escape(sequence_name)}</a></td>"
            f"<td>{format_float(row['foot_bce'])}</td>"
            f"<td>{format_float(row['foot_mae'])}</td>"
            f"<td>{format_float(row['foot_source_low_globalpose_high_rate'])}</td>"
            f"<td>{format_float(row['foot_source_high_globalpose_low_rate'])}</td>"
            f"<td>{format_float(row['foot_gt_active_rate'])}</td>"
            f"<td>{format_float(row['foot_gp_active_rate'])}</td>"
            "</tr>"
        )

    window_rows = []
    for index, row in enumerate(payload["window_rows"], start=1):
        sequence_name = str(row["sequence"])
        frame_start = int(row["frame_start"])
        link = f"{links.get(sequence_name, '#')}#frame={frame_start}"
        window_rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td><a href=\"{escape(link)}\">{escape(sequence_name)}</a></td>"
            f"<td>{frame_start}-{int(row['frame_end'])}</td>"
            f"<td>{format_float(row['foot_bce'])}</td>"
            f"<td>{format_float(row['foot_mae'])}</td>"
            f"<td>{format_float(row['foot_source_low_globalpose_high_rate'])}</td>"
            f"<td>{format_float(row['foot_source_high_globalpose_low_rate'])}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>GlobalPose stationary compare</title>
  <style>{base_css()}</style>
</head>
<body>
  <main>
    <header class="page-header">
      <h1>GlobalPose stationary compare</h1>
      <p class="muted">{escape(summary["dataset_name"])} · sequences={summary["sequence_count"]} · frames={summary["frames"]}</p>
    </header>
    <section class="metric-grid">
      {metric_card("foot BCE", summary["overall_metrics"]["foot_bce"])}
      {metric_card("foot MAE", summary["overall_metrics"]["foot_mae"])}
      {metric_card("foot source low / GlobalPose high", summary["overall_metrics"]["foot_source_low_globalpose_high_rate"])}
      {metric_card("foot source high / GlobalPose low", summary["overall_metrics"]["foot_source_high_globalpose_low_rate"])}
    </section>
    <section>
      <h2>Largest sequence differences by foot_bce</h2>
      <table>
        <thead>
          <tr><th>#</th><th>sequence</th><th>foot_bce</th><th>foot_mae</th><th>source low / GlobalPose high</th><th>source high / GlobalPose low</th><th>Label foot active</th><th>GlobalPose foot active</th></tr>
        </thead>
        <tbody>{''.join(sequence_rows)}</tbody>
      </table>
    </section>
    <section>
      <h2>Review windows by foot_bce</h2>
      <table>
        <thead>
          <tr><th>#</th><th>sequence</th><th>frames</th><th>foot_bce</th><th>foot_mae</th><th>source low / GlobalPose high</th><th>source high / GlobalPose low</th></tr>
        </thead>
        <tbody>{''.join(window_rows)}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def render_sequence_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(payload["sequence"])} stationary compare</title>
  <style>{base_css()}</style>
</head>
<body>
  <main>
    <a href="../index.html">Back to index</a>
    <header class="page-header">
      <h1>{escape(payload["sequence"])}</h1>
      <p class="muted">frames={payload["frame_count"]}, fps={payload["fps"]:.3f}</p>
    </header>
    <section class="metric-grid">
      {metric_card("foot BCE", payload["metrics"]["foot_bce"])}
      {metric_card("foot MAE", payload["metrics"]["foot_mae"])}
      {metric_card("foot source low / GlobalPose high", payload["metrics"]["foot_source_low_globalpose_high_rate"])}
      {metric_card("foot source high / GlobalPose low", payload["metrics"]["foot_source_high_globalpose_low_rate"])}
    </section>
    <section class="viewer">
      <div class="controls">
        <button id="playButton" type="button">Play</button>
        <input id="frameSlider" type="range" min="0" value="0">
        <span id="frameLabel"></span>
      </div>
      <div class="skeleton-panel">
        <div class="skeleton-toolbar">
          <label>mode
            <select id="skeletonMode">
              <option value="overlay">Overlay</option>
              <option value="split">Split</option>
              <option value="pelvis">Pelvis-aligned</option>
            </select>
          </label>
          <label>view
            <select id="skeletonView">
              <option value="front">Front</option>
              <option value="side">Side</option>
              <option value="top">Top</option>
            </select>
          </label>
          <label>camera
            <select id="skeletonCamera">
              <option value="sequence">Sequence</option>
              <option value="window">Window</option>
              <option value="pelvis">Pelvis-follow</option>
            </select>
          </label>
          <label>zoom
            <select id="skeletonZoom">
              <option value="0.75">0.75x</option>
              <option value="1" selected>1x</option>
              <option value="1.5">1.5x</option>
              <option value="2">2x</option>
            </select>
          </label>
          <label><input id="skeletonShowGt" type="checkbox" checked> Pose GT</label>
          <label><input id="skeletonShowGp" type="checkbox" checked> GlobalPose</label>
          <span id="skeletonStatus" class="muted"></span>
        </div>
        <canvas id="skeletonCanvas" width="1180" height="520"></canvas>
      </div>
      <div id="jointToggles" class="toggle-row"></div>
      <canvas id="chartCanvas" width="1180" height="460"></canvas>
      <div class="window-row" id="badWindowButtons"></div>
      <table>
        <thead><tr><th>joint</th><th>Source label</th><th>GlobalPose</th><th>abs diff</th><th>relation</th></tr></thead>
        <tbody id="frameTable"></tbody>
      </table>
    </section>
  </main>
  <script id="compareData" type="application/json">{escape_script_json(payload_json)}</script>
  <script>{sequence_viewer_js()}</script>
</body>
</html>
"""


def base_css() -> str:
    return """
:root { color-scheme: light; font-family: Segoe UI, Arial, sans-serif; }
body { margin: 0; background: #f8fafc; color: #0f172a; }
main { max-width: 1220px; margin: 0 auto; padding: 28px 24px 48px; }
a { color: #2563eb; text-decoration: none; }
a:hover { text-decoration: underline; }
.page-header { margin: 12px 0 20px; }
h1 { margin: 0 0 6px; font-size: 28px; }
h2 { margin: 28px 0 12px; font-size: 19px; }
.muted { color: #64748b; margin: 0; }
.metric-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 16px 0 18px; }
.metric-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 14px; }
.metric-label { color: #64748b; font-size: 12px; }
.metric-value { font-size: 22px; font-weight: 650; margin-top: 4px; }
table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }
th, td { padding: 8px 10px; border-bottom: 1px solid #e2e8f0; font-size: 13px; text-align: right; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
thead th { background: #f1f5f9; color: #334155; font-weight: 650; }
tr:last-child td { border-bottom: 0; }
.viewer { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; }
.controls { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
button { border: 1px solid #cbd5e1; background: #fff; color: #0f172a; border-radius: 7px; padding: 6px 10px; cursor: pointer; }
button:hover { background: #f1f5f9; }
input[type="range"] { flex: 1; }
canvas { display: block; width: 100%; border: 1px solid #e2e8f0; border-radius: 8px; background: #fff; }
#chartCanvas { height: 460px; }
.skeleton-panel { margin: 10px 0 14px; }
.skeleton-toolbar { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; font-size: 13px; }
.skeleton-toolbar label { display: inline-flex; align-items: center; gap: 6px; }
.skeleton-toolbar select { border: 1px solid #cbd5e1; border-radius: 7px; background: #fff; padding: 5px 8px; }
#skeletonCanvas { height: 520px; background: #f8fafc; }
.toggle-row, .window-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 12px; }
.toggle-row label { display: inline-flex; align-items: center; gap: 5px; padding: 5px 8px; border: 1px solid #e2e8f0; border-radius: 7px; font-size: 13px; }
.state-source-low-globalpose-high { color: #dc2626; font-weight: 650; }
.state-source-high-globalpose-low { color: #2563eb; font-weight: 650; }
.state-aligned { color: #475569; }
@media (max-width: 780px) { .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
"""


def sequence_viewer_js() -> str:
    return r"""
const data = JSON.parse(document.getElementById("compareData").textContent);
const playButton = document.getElementById("playButton");
const frameSlider = document.getElementById("frameSlider");
const frameLabel = document.getElementById("frameLabel");
const skeletonCanvas = document.getElementById("skeletonCanvas");
const skeletonCtx = skeletonCanvas.getContext("2d");
const skeletonMode = document.getElementById("skeletonMode");
const skeletonView = document.getElementById("skeletonView");
const skeletonCamera = document.getElementById("skeletonCamera");
const skeletonZoom = document.getElementById("skeletonZoom");
const skeletonShowGt = document.getElementById("skeletonShowGt");
const skeletonShowGp = document.getElementById("skeletonShowGp");
const skeletonStatus = document.getElementById("skeletonStatus");
const chartCanvas = document.getElementById("chartCanvas");
const chartCtx = chartCanvas.getContext("2d");
const frameTable = document.getElementById("frameTable");
const jointToggles = document.getElementById("jointToggles");
const badWindowButtons = document.getElementById("badWindowButtons");
let frame = 0;
let playing = false;
let lastTick = 0;
const frameCount = data.frame_count;
const selected = data.joint_names.map((name) => name === "left_foot" || name === "right_foot");
const skeleton = data.skeleton || null;
const hasSkeleton = Boolean(
  skeleton &&
  Array.isArray(skeleton.gt_joints_world) &&
  Array.isArray(skeleton.globalpose_joints_world) &&
  skeleton.gt_joints_world.length > 0
);
const skeletonEdges = hasSkeleton
  ? (Array.isArray(skeleton.skeleton_edges)
      ? skeleton.skeleton_edges
      : skeleton.parents.map((parent, joint) => [parent, joint]).filter((edge) => edge[0] >= 0))
  : [];
const skeletonStationaryIndexToProb = {};
if (hasSkeleton) {
  (skeleton.stationary_joint_indices || [0, 10, 11, 22, 23]).forEach((jointIndex, probIndex) => {
    skeletonStationaryIndexToProb[String(jointIndex)] = probIndex;
  });
}
frameSlider.max = Math.max(frameCount - 1, 0);

function resizeCanvas(canvas, ctx) {
  const ratio = window.devicePixelRatio || 1;
  const displayWidth = canvas.clientWidth || canvas.width;
  const displayHeight = canvas.clientHeight || canvas.height;
  if (canvas.width !== Math.round(displayWidth * ratio) || canvas.height !== Math.round(displayHeight * ratio)) {
    canvas.width = Math.round(displayWidth * ratio);
    canvas.height = Math.round(displayHeight * ratio);
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function initToggles() {
  jointToggles.innerHTML = "";
  data.joint_names.forEach((name, index) => {
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = selected[index];
    checkbox.addEventListener("change", () => {
      selected[index] = checkbox.checked;
      drawChart();
    });
    const span = document.createElement("span");
    span.textContent = name;
    span.style.color = data.joint_colors[index];
    label.appendChild(checkbox);
    label.appendChild(span);
    jointToggles.appendChild(label);
  });
}

function initBadWindows() {
  badWindowButtons.innerHTML = "";
  data.bad_windows.slice(0, 12).forEach((windowRow, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `#${index + 1} ${windowRow.frame_start}-${windowRow.frame_end} foot_bce=${Number(windowRow.foot_bce).toFixed(3)}`;
    button.addEventListener("click", () => setFrame(windowRow.frame_start));
    badWindowButtons.appendChild(button);
  });
}

function stateFor(gt, gp) {
  if (gt < data.thresholds.gt_low && gp > data.thresholds.pred) return "source low / GlobalPose high";
  if (gt > data.thresholds.gt_high && gp < data.thresholds.pred) return "source high / GlobalPose low";
  return "aligned";
}

function stateClass(state) {
  if (state === "source low / GlobalPose high") return "state-source-low-globalpose-high";
  if (state === "source high / GlobalPose low") return "state-source-high-globalpose-low";
  return "state-aligned";
}

function stateColor(state, fallback) {
  if (state === "source low / GlobalPose high") return "#dc2626";
  if (state === "source high / GlobalPose low") return "#2563eb";
  return fallback;
}

function stateForSkeletonJoint(jointIndex) {
  const probIndex = skeletonStationaryIndexToProb[String(jointIndex)];
  if (probIndex === undefined) return "aligned";
  return stateFor(data.gt_stationary_prob_5[frame][probIndex], data.globalpose_stationary_prob_5[frame][probIndex]);
}

function formatProbabilityValue(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "nan";
  return number.toFixed(2);
}

function stationaryProbabilityText(jointIndex, labelSource) {
  const probIndex = skeletonStationaryIndexToProb[String(jointIndex)];
  if (probIndex === undefined) return "";
  const sourceRow = data.gt_stationary_prob_5[frame] || [];
  const globalposeRow = data.globalpose_stationary_prob_5[frame] || [];
  if (labelSource === "source") return formatProbabilityValue(sourceRow[probIndex]);
  if (labelSource === "globalpose") return formatProbabilityValue(globalposeRow[probIndex]);
  return `${formatProbabilityValue(sourceRow[probIndex])}/${formatProbabilityValue(globalposeRow[probIndex])}`;
}

function transformSkeletonJoints(joints, mode) {
  const pelvis = joints[0] || [0, 0, 0];
  return joints.map((joint) => {
    if (mode !== "pelvis") return joint;
    return [joint[0] - pelvis[0], joint[1] - pelvis[1], joint[2] - pelvis[2]];
  });
}

function projectSkeletonJoint(joint, view) {
  if (view === "side") return { x: joint[2], y: joint[1] };
  if (view === "top") return { x: joint[0], y: joint[2] };
  return { x: joint[0], y: joint[1] };
}

function skeletonViewLabel(view) {
  if (view === "side") return "Side z/y";
  if (view === "top") return "Top x/z";
  return "Front x/y";
}

function boundsFromJointGroups(jointGroups, view) {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  jointGroups.forEach((joints) => {
    joints.forEach((joint) => {
      const projected = projectSkeletonJoint(joint, view);
      const x = projected.x;
      const y = projected.y;
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    });
  });
  if (!Number.isFinite(minX) || Math.abs(maxX - minX) < 1e-6) {
    minX = -1;
    maxX = 1;
  }
  if (!Number.isFinite(minY) || Math.abs(maxY - minY) < 1e-6) {
    minY = -0.2;
    maxY = 1.8;
  }
  return { min: [minX, minY], max: [maxX, maxY] };
}

function cameraBoundsFor(mode, cameraMode, view, jointGroups) {
  if (!skeleton.camera_bounds) return boundsFromJointGroups(jointGroups, view);
  if (mode === "pelvis") return skeleton.camera_bounds.pelvis[view];
  if (cameraMode === "pelvis") return skeleton.camera_bounds.pelvis[view];
  if (cameraMode === "window") {
    const windowBounds = (skeleton.camera_bounds.windows || []).find((row) => frame >= row.frame_start && frame <= row.frame_end);
    if (windowBounds && windowBounds[view]) return windowBounds[view];
  }
  return skeleton.camera_bounds.sequence[view];
}

function currentPelvisCenter(view, showGt, showGp) {
  const centers = [];
  if (showGt && skeleton.gt_joints_world[frame]) centers.push(projectSkeletonJoint(skeleton.gt_joints_world[frame][0], view));
  if (showGp && skeleton.globalpose_joints_world[frame]) centers.push(projectSkeletonJoint(skeleton.globalpose_joints_world[frame][0], view));
  if (centers.length === 0) return null;
  return {
    x: centers.reduce((sum, value) => sum + value.x, 0) / centers.length,
    y: centers.reduce((sum, value) => sum + value.y, 0) / centers.length,
  };
}

function jointGroupCenter(joints, view) {
  const bounds = boundsFromJointGroups([joints], view);
  return {
    x: (bounds.min[0] + bounds.max[0]) * 0.5,
    y: (bounds.min[1] + bounds.max[1]) * 0.5,
  };
}

function createProjector(bounds, rect, view, zoom, centerOverride) {
  let minX = Number(bounds.min[0]);
  let minY = Number(bounds.min[1]);
  let maxX = Number(bounds.max[0]);
  let maxY = Number(bounds.max[1]);
  if (!Number.isFinite(minX) || !Number.isFinite(maxX) || Math.abs(maxX - minX) < 1e-6) {
    minX = -1;
    maxX = 1;
  }
  if (!Number.isFinite(minY) || !Number.isFinite(maxY) || Math.abs(maxY - minY) < 1e-6) {
    minY = -0.2;
    maxY = 1.8;
  }
  const margin = 34;
  const scaleX = (rect.w - margin * 2) / Math.max(maxX - minX, 1e-6);
  const scaleY = (rect.h - margin * 2) / Math.max(maxY - minY, 1e-6);
  const scale = Math.min(scaleX, scaleY) * zoom;
  const centerX = centerOverride ? centerOverride.x : (minX + maxX) * 0.5;
  const centerY = centerOverride ? centerOverride.y : (minY + maxY) * 0.5;
  return (joint) => {
    const projected = projectSkeletonJoint(joint, view);
    return {
      x: rect.x + rect.w * 0.5 + (projected.x - centerX) * scale,
      y: rect.y + rect.h * 0.5 - (projected.y - centerY) * scale,
    };
  };
}

function drawPanelFrame(ctx, rect, label) {
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1;
  ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
  ctx.fillStyle = "#334155";
  ctx.font = "13px Segoe UI, Arial, sans-serif";
  ctx.fillText(label, rect.x + 12, rect.y + 22);
}

function clamp(value, minValue, maxValue) {
  return Math.min(Math.max(value, minValue), maxValue);
}

function drawStationaryProbabilityLabel(ctx, point, jointIndex, style, state) {
  const label = stationaryProbabilityText(jointIndex, style.labelSource || "both");
  if (!label) return;
  const color = stateColor(state, style.highlightColor);
  const bounds = style.labelBounds || { x: 0, y: 0, w: ctx.canvas.width, h: ctx.canvas.height };
  ctx.save();
  ctx.font = "11px Segoe UI, Arial, sans-serif";
  const padX = 5;
  const boxW = Math.ceil(ctx.measureText(label).width) + padX * 2;
  const boxH = 17;
  const placeLeft = point.x > bounds.x + bounds.w * 0.55;
  let x = point.x + (placeLeft ? -boxW - 10 : 10);
  let y = point.y - boxH - 6;
  x = clamp(x, bounds.x + 4, bounds.x + bounds.w - boxW - 4);
  y = clamp(y, bounds.y + 28, bounds.y + bounds.h - boxH - 4);
  ctx.fillStyle = "rgba(255,255,255,0.88)";
  ctx.fillRect(x, y, boxW, boxH);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, boxW, boxH);
  ctx.fillStyle = "#0f172a";
  ctx.fillText(label, x + padX, y + 12);
  ctx.restore();
}

function drawSkeleton(ctx, joints, projector, style) {
  const points = joints.map((joint) => projector(joint));
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = style.boneColor;
  ctx.lineWidth = style.lineWidth;
  ctx.setLineDash(style.dashed ? [8, 6] : []);
  skeletonEdges.forEach((edge) => {
    const a = points[edge[0]];
    const b = points[edge[1]];
    if (!a || !b) return;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  });
  ctx.setLineDash([]);
  points.forEach((point) => {
    ctx.fillStyle = style.jointColor;
    ctx.beginPath();
    ctx.arc(point.x, point.y, 2.6, 0, Math.PI * 2);
    ctx.fill();
  });
  (skeleton.stationary_joint_indices || [0, 10, 11, 22, 23]).forEach((jointIndex) => {
    const point = points[jointIndex];
    if (!point) return;
    const state = stateForSkeletonJoint(jointIndex);
    const color = stateColor(state, style.highlightColor);
    ctx.strokeStyle = color;
    ctx.lineWidth = state === "aligned" ? 2 : 3.5;
    ctx.beginPath();
    ctx.arc(point.x, point.y, state === "aligned" ? 6 : 8, 0, Math.PI * 2);
    ctx.stroke();
    if (style.showProbabilityLabels) {
      drawStationaryProbabilityLabel(ctx, point, jointIndex, style, state);
    }
  });
  ctx.restore();
}

function drawSkeletonViewer() {
  return drawSkeletonViewerFixed();
  resizeCanvas(skeletonCanvas, skeletonCtx);
  const width = skeletonCanvas.clientWidth || 1180;
  const height = skeletonCanvas.clientHeight || 520;
  skeletonCtx.clearRect(0, 0, width, height);
  skeletonCtx.fillStyle = "#f8fafc";
  skeletonCtx.fillRect(0, 0, width, height);
  if (!hasSkeleton) {
    skeletonStatus.textContent = "skeleton data not embedded";
    skeletonCtx.fillStyle = "#64748b";
    skeletonCtx.font = "14px Segoe UI, Arial, sans-serif";
    skeletonCtx.fillText("Skeleton payload unavailable.", 24, 36);
    return;
  }
  if (frame >= skeleton.frame_count) {
    skeletonStatus.textContent = "frame outside skeleton payload";
    return;
  }
  const showGt = skeletonShowGt.checked;
  const showGp = skeletonShowGp.checked;
  const mode = skeletonMode.value;
  const view = skeletonView.value;
  skeletonStatus.textContent = `skeleton frames ${skeleton.frame_count} | labels Source/GlobalPose`;
  if (!showGt && !showGp) {
    skeletonCtx.fillStyle = "#64748b";
    skeletonCtx.font = "14px Segoe UI, Arial, sans-serif";
    skeletonCtx.fillText("Enable Pose GT or GlobalPose to draw skeletons.", 24, 36);
    return;
  }
  const gt = transformSkeletonJoints(skeleton.gt_joints_world[frame], mode);
  const gp = transformSkeletonJoints(skeleton.globalpose_joints_world[frame], mode);
  if (mode === "split") {
    const gap = 16;
    const leftRect = { x: 0, y: 0, w: (width - gap) * 0.5, h: height };
    const rightRect = { x: leftRect.w + gap, y: 0, w: (width - gap) * 0.5, h: height };
    drawPanelFrame(skeletonCtx, leftRect, `Pose GT · ${skeletonViewLabel(view)}`);
    drawPanelFrame(skeletonCtx, rightRect, `GlobalPose · ${skeletonViewLabel(view)}`);
    if (showGt) drawSkeleton(skeletonCtx, gt, createProjector([gt], leftRect, view), {
      boneColor: "#2563eb",
      jointColor: "#1d4ed8",
      highlightColor: "#0f766e",
      lineWidth: 3,
      dashed: false,
      labelBounds: leftRect,
      showProbabilityLabels: true,
      labelSource: "source",
    });
    if (showGp) drawSkeleton(skeletonCtx, gp, createProjector([gp], rightRect, view), {
      boneColor: "#dc2626",
      jointColor: "#b91c1c",
      highlightColor: "#f97316",
      lineWidth: 3,
      dashed: true,
      labelBounds: rightRect,
      showProbabilityLabels: true,
      labelSource: "globalpose",
    });
    return;
  }
  const rect = { x: 0, y: 0, w: width, h: height };
  drawPanelFrame(skeletonCtx, rect, `${mode === "pelvis" ? "Pelvis-aligned overlay" : "World overlay"} · ${skeletonViewLabel(view)}`);
  const groups = [];
  if (showGt) groups.push(gt);
  if (showGp) groups.push(gp);
  const projector = createProjector(groups, rect, view);
  if (showGt) drawSkeleton(skeletonCtx, gt, projector, {
    boneColor: "#2563eb",
    jointColor: "#1d4ed8",
    highlightColor: "#0f766e",
    lineWidth: 3,
    dashed: false,
    labelBounds: rect,
    showProbabilityLabels: true,
    labelSource: "both",
  });
  if (showGp) drawSkeleton(skeletonCtx, gp, projector, {
    boneColor: "#dc2626",
    jointColor: "#b91c1c",
    highlightColor: "#f97316",
    lineWidth: 2.6,
    dashed: true,
    labelBounds: rect,
    showProbabilityLabels: !showGt,
    labelSource: "both",
  });
}

function drawSkeletonViewerFixed() {
  resizeCanvas(skeletonCanvas, skeletonCtx);
  const width = skeletonCanvas.clientWidth || 1180;
  const height = skeletonCanvas.clientHeight || 520;
  skeletonCtx.clearRect(0, 0, width, height);
  skeletonCtx.fillStyle = "#f8fafc";
  skeletonCtx.fillRect(0, 0, width, height);
  if (!hasSkeleton) {
    skeletonStatus.textContent = "skeleton data not embedded";
    skeletonCtx.fillStyle = "#64748b";
    skeletonCtx.font = "14px Segoe UI, Arial, sans-serif";
    skeletonCtx.fillText("Skeleton payload unavailable.", 24, 36);
    return;
  }
  if (frame >= skeleton.frame_count) {
    skeletonStatus.textContent = "frame outside skeleton payload";
    return;
  }
  const showGt = skeletonShowGt.checked;
  const showGp = skeletonShowGp.checked;
  const mode = skeletonMode.value;
  const view = skeletonView.value;
  const cameraMode = skeletonCamera.value;
  const zoom = Number(skeletonZoom.value) || 1;
  skeletonStatus.textContent = `skeleton frames ${skeleton.frame_count} | labels Source/GlobalPose`;
  if (!showGt && !showGp) {
    skeletonCtx.fillStyle = "#64748b";
    skeletonCtx.font = "14px Segoe UI, Arial, sans-serif";
    skeletonCtx.fillText("Enable Pose GT or GlobalPose to draw skeletons.", 24, 36);
    return;
  }
  const gt = transformSkeletonJoints(skeleton.gt_joints_world[frame], mode);
  const gp = transformSkeletonJoints(skeleton.globalpose_joints_world[frame], mode);
  const groups = [];
  if (showGt) groups.push(gt);
  if (showGp) groups.push(gp);
  const bounds = cameraBoundsFor(mode, cameraMode, view, groups);
  if (mode === "split") {
    const gap = 16;
    const leftRect = { x: 0, y: 0, w: (width - gap) * 0.5, h: height };
    const rightRect = { x: leftRect.w + gap, y: 0, w: (width - gap) * 0.5, h: height };
    drawPanelFrame(skeletonCtx, leftRect, `Pose GT · ${skeletonViewLabel(view)} · ${cameraMode}`);
    drawPanelFrame(skeletonCtx, rightRect, `GlobalPose · ${skeletonViewLabel(view)} · ${cameraMode}`);
    const gtCenter = mode === "pelvis" ? null : jointGroupCenter(gt, view);
    const gpCenter = mode === "pelvis" ? null : jointGroupCenter(gp, view);
    if (showGt) drawSkeleton(skeletonCtx, gt, createProjector(bounds, leftRect, view, zoom, gtCenter), {
      boneColor: "#2563eb",
      jointColor: "#1d4ed8",
      highlightColor: "#0f766e",
      lineWidth: 3,
      dashed: false,
      labelBounds: leftRect,
      showProbabilityLabels: true,
      labelSource: "source",
    });
    if (showGp) drawSkeleton(skeletonCtx, gp, createProjector(bounds, rightRect, view, zoom, gpCenter), {
      boneColor: "#dc2626",
      jointColor: "#b91c1c",
      highlightColor: "#f97316",
      lineWidth: 3,
      dashed: true,
      labelBounds: rightRect,
      showProbabilityLabels: true,
      labelSource: "globalpose",
    });
    return;
  }
  const rect = { x: 0, y: 0, w: width, h: height };
  drawPanelFrame(skeletonCtx, rect, `${mode === "pelvis" ? "Pelvis-aligned overlay" : "World overlay"} · ${skeletonViewLabel(view)} · ${cameraMode}`);
  const centerOverride = mode === "pelvis" || cameraMode !== "pelvis" ? null : currentPelvisCenter(view, showGt, showGp);
  const projector = createProjector(bounds, rect, view, zoom, centerOverride);
  if (showGt) drawSkeleton(skeletonCtx, gt, projector, {
    boneColor: "#2563eb",
    jointColor: "#1d4ed8",
    highlightColor: "#0f766e",
    lineWidth: 3,
    dashed: false,
    labelBounds: rect,
    showProbabilityLabels: true,
    labelSource: "both",
  });
  if (showGp) drawSkeleton(skeletonCtx, gp, projector, {
    boneColor: "#dc2626",
    jointColor: "#b91c1c",
    highlightColor: "#f97316",
    lineWidth: 2.6,
    dashed: true,
    labelBounds: rect,
    showProbabilityLabels: !showGt,
    labelSource: "both",
  });
}

function drawChart() {
  resizeCanvas(chartCanvas, chartCtx);
  const width = chartCanvas.clientWidth || 1180;
  const height = chartCanvas.clientHeight || 460;
  const left = 62;
  const right = 20;
  const top = 24;
  const bottom = 58;
  const plotW = width - left - right;
  const plotH = height - top - bottom;
  chartCtx.clearRect(0, 0, width, height);
  chartCtx.fillStyle = "#ffffff";
  chartCtx.fillRect(0, 0, width, height);
  chartCtx.fillStyle = "#f8fafc";
  chartCtx.fillRect(left, top, plotW, plotH);
  chartCtx.strokeStyle = "#cbd5e1";
  chartCtx.strokeRect(left, top, plotW, plotH);

  for (const threshold of [data.thresholds.gt_low, data.thresholds.pred, data.thresholds.gt_high]) {
    const y = top + (1 - threshold) * plotH;
    chartCtx.strokeStyle = threshold === data.thresholds.pred ? "#94a3b8" : "#cbd5e1";
    chartCtx.setLineDash([5, 5]);
    chartCtx.beginPath();
    chartCtx.moveTo(left, y);
    chartCtx.lineTo(left + plotW, y);
    chartCtx.stroke();
    chartCtx.setLineDash([]);
    chartCtx.fillStyle = "#64748b";
    chartCtx.font = "11px Segoe UI, Arial, sans-serif";
    chartCtx.fillText(String(threshold), 8, y + 4);
  }

  const footIndices = [1, 2];
  for (let i = 0; i < frameCount; i += 1) {
    const hasSourceLowGlobalPoseHigh = footIndices.some((joint) => stateFor(data.gt_stationary_prob_5[i][joint], data.globalpose_stationary_prob_5[i][joint]) === "source low / GlobalPose high");
    const hasSourceHighGlobalPoseLow = footIndices.some((joint) => stateFor(data.gt_stationary_prob_5[i][joint], data.globalpose_stationary_prob_5[i][joint]) === "source high / GlobalPose low");
    if (!hasSourceLowGlobalPoseHigh && !hasSourceHighGlobalPoseLow) continue;
    const x = left + (frameCount <= 1 ? 0 : (i / (frameCount - 1)) * plotW);
    chartCtx.fillStyle = hasSourceLowGlobalPoseHigh ? "rgba(220, 38, 38, 0.08)" : "rgba(37, 99, 235, 0.08)";
    chartCtx.fillRect(x, top, Math.max(1, plotW / Math.max(frameCount, 1)), plotH);
  }

  function drawSeries(values, color, dashed) {
    chartCtx.strokeStyle = color;
    chartCtx.lineWidth = dashed ? 1.8 : 2.2;
    chartCtx.setLineDash(dashed ? [7, 5] : []);
    chartCtx.beginPath();
    values.forEach((value, valueIndex) => {
      const x = left + (frameCount <= 1 ? 0 : (valueIndex / (frameCount - 1)) * plotW);
      const y = top + (1 - value) * plotH;
      if (valueIndex === 0) chartCtx.moveTo(x, y);
      else chartCtx.lineTo(x, y);
    });
    chartCtx.stroke();
    chartCtx.setLineDash([]);
  }

  data.joint_names.forEach((name, jointIndex) => {
    if (!selected[jointIndex]) return;
    const color = data.joint_colors[jointIndex];
    drawSeries(data.gt_stationary_prob_5.map((row) => row[jointIndex]), color, false);
    drawSeries(data.globalpose_stationary_prob_5.map((row) => row[jointIndex]), color, true);
  });

  const cursorX = left + (frameCount <= 1 ? 0 : (frame / (frameCount - 1)) * plotW);
  chartCtx.strokeStyle = "#0f172a";
  chartCtx.lineWidth = 1;
  chartCtx.beginPath();
  chartCtx.moveTo(cursorX, top);
  chartCtx.lineTo(cursorX, top + plotH);
  chartCtx.stroke();

  let legendX = left;
  const legendY = top + plotH + 28;
  data.joint_names.forEach((name, jointIndex) => {
    if (!selected[jointIndex]) return;
    chartCtx.strokeStyle = data.joint_colors[jointIndex];
    chartCtx.lineWidth = 3;
    chartCtx.beginPath();
    chartCtx.moveTo(legendX, legendY - 4);
    chartCtx.lineTo(legendX + 18, legendY - 4);
    chartCtx.stroke();
    chartCtx.fillStyle = "#334155";
    chartCtx.font = "12px Segoe UI, Arial, sans-serif";
    chartCtx.fillText(`${name} Source label solid / GlobalPose dashed`, legendX + 24, legendY);
    legendX += 190;
  });
}

function renderFrameTable() {
  frameTable.innerHTML = "";
  data.joint_names.forEach((name, index) => {
    const gt = data.gt_stationary_prob_5[frame][index];
    const gp = data.globalpose_stationary_prob_5[frame][index];
    const state = stateFor(gt, gp);
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${name}</td><td>${gt.toFixed(3)}</td><td>${gp.toFixed(3)}</td><td>${Math.abs(gp - gt).toFixed(3)}</td><td class="${stateClass(state)}">${state}</td>`;
    frameTable.appendChild(tr);
  });
}

function setFrame(nextFrame) {
  frame = Math.max(0, Math.min(frameCount - 1, Math.round(nextFrame)));
  frameSlider.value = String(frame);
  frameLabel.textContent = `frame ${frame} / ${frameCount - 1} · time ${(frame / Math.max(data.fps, 1)).toFixed(2)}s`;
  if (history.replaceState) history.replaceState(null, "", `#frame=${frame}`);
  drawSkeletonViewerFixed();
  drawChart();
  renderFrameTable();
}

function tick(timestamp) {
  if (!playing) return;
  const frameMs = 1000 / Math.max(data.fps, 1);
  if (!lastTick || timestamp - lastTick >= frameMs) {
    lastTick = timestamp;
    setFrame(frame + 1 >= frameCount ? 0 : frame + 1);
  }
  window.requestAnimationFrame(tick);
}

playButton.addEventListener("click", () => {
  playing = !playing;
  playButton.textContent = playing ? "Pause" : "Play";
  lastTick = 0;
  if (playing) window.requestAnimationFrame(tick);
});
frameSlider.addEventListener("input", () => {
  playing = false;
  playButton.textContent = "Play";
  setFrame(Number(frameSlider.value));
});
chartCanvas.addEventListener("click", (event) => {
  const rect = chartCanvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const width = chartCanvas.clientWidth || 1180;
  const left = 62;
  const right = 20;
  if (x < left || x > width - right) return;
  setFrame(((x - left) / (width - left - right)) * Math.max(frameCount - 1, 0));
});
skeletonMode.addEventListener("change", () => drawSkeletonViewerFixed());
skeletonView.addEventListener("change", () => drawSkeletonViewerFixed());
skeletonCamera.addEventListener("change", () => drawSkeletonViewerFixed());
skeletonZoom.addEventListener("change", () => drawSkeletonViewerFixed());
skeletonShowGt.addEventListener("change", () => drawSkeletonViewerFixed());
skeletonShowGp.addEventListener("change", () => drawSkeletonViewerFixed());
window.addEventListener("resize", () => setFrame(frame));
initToggles();
initBadWindows();
const hashFrame = Number(new URLSearchParams(window.location.hash.replace(/^#/, "")).get("frame"));
setFrame(Number.isFinite(hashFrame) ? hashFrame : 0);
"""


def metric_card(label: str, value: Any) -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{escape(label)}</div>'
        f'<div class="metric-value">{format_float(value)}</div>'
        "</div>"
    )


def format_float(value: Any) -> str:
    return f"{float(value):.3f}"


def round_nested_array(values: np.ndarray, decimals: int) -> list[Any]:
    return np.round(np.asarray(values, dtype=np.float32), decimals=decimals).tolist()


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def escape_script_json(value: str) -> str:
    return value.replace("</", "<\\/")


# endregion


if __name__ == "__main__":
    main()
