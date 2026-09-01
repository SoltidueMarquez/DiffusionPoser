from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from data_loaders.generate_realtime_pose_tasks import load_realtime_source
from data_loaders.realtime_pose_geometry import decode_target_head_rotations_np
from data_loaders.realtime_pose_kinematics import (
    SMPL_PARENTS,
    rotation_6d_to_matrix_np,
)
from data_loaders.realtime_pose_predictor_features import (
    build_predictor_step_features_np,
)
from sample.realtime_pose_smpl_rendering import (
    SmplMeshSequence,
    body_fbx_world_to_smpl_local_rotations,
    create_smplh_model,
    require_directory,
    require_file,
    rotation_matrices_to_axis_angle,
    run_smplh_forward,
    transform_faces_to_unity_winding,
)
from sample.render_rpm_past_motion import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    PastMotionRender,
    build_front_alignment_rotation,
    build_orthographic_layout,
    render_past_motion_image,
    validate_rendered_image,
)
from utils.model_util import load_realtime_pose_predictor
from utils.normalizer import RealtimePoseNormalizer


DEFAULT_CURRENT_FRAME = 180
# Predictor horizon 的 index 0 是当前帧，1～10 是未来十帧。每隔两帧取一帧，
# 得到均匀覆盖整个预测窗口的五个展示姿态。
DEFAULT_HORIZON_INDICES = (2, 4, 6, 8, 10)
DEFAULT_OUTPUT = Path(
    "output/主方法图所需材料与参考/"
    "PredictorOutput_当前帧加未来5帧_SMPL男性_正面.png"
)


# region CLI 与路径解析


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "在指定当前帧运行一次 Predictor，把未来十帧均匀降采样为五帧，"
            "并按 Past Motion 的男性 SMPL-H 风格渲染。"
        )
    )
    paths = parser.add_argument_group("paths")
    paths.add_argument("--history_npz", required=True, type=Path)
    paths.add_argument("--history_json", required=True, type=Path)
    paths.add_argument("--source_npz", default=None, type=Path)
    paths.add_argument("--predictor_model_path", default=None, type=Path)
    paths.add_argument("--normalizer_dir", default=None, type=Path)
    paths.add_argument("--smpl_model_dir", required=True, type=Path)
    paths.add_argument("--output_png", default=DEFAULT_OUTPUT, type=Path)
    inference = parser.add_argument_group("inference")
    inference.add_argument(
        "--current_frame", default=DEFAULT_CURRENT_FRAME, type=int
    )
    inference.add_argument(
        "--horizon_indices",
        nargs=5,
        default=DEFAULT_HORIZON_INDICES,
        type=int,
        metavar=("H0", "H1", "H2", "H3", "H4"),
    )
    inference.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    return parser


def load_json_object(path: Path, label: str) -> tuple[Path, dict]:
    resolved = require_file(path, label)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} 顶层必须为 object：{resolved}")
    return resolved, value


def resolve_optional_path(
    explicit: Path | None,
    fallback: object,
    *,
    label: str,
    directory: bool = False,
) -> Path:
    value = explicit if explicit is not None else Path(str(fallback))
    return (
        require_directory(value, label)
        if directory
        else require_file(value, label)
    )


def select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前环境中 CUDA 不可用。")
    return torch.device(value)


def validate_horizon_indices(values: np.ndarray) -> np.ndarray:
    indices = np.asarray(values, dtype=np.int64)
    if indices.shape != (5,):
        raise ValueError(f"horizon_indices 必须为五个索引：{indices.shape}。")
    if not np.all(np.diff(indices) > 0):
        raise ValueError(
            f"horizon_indices 必须严格递增：{indices.tolist()}。"
        )
    if np.any(indices < 1) or np.any(indices > 10):
        raise ValueError(
            "未来帧索引必须位于 [1,10]："
            f"{indices.tolist()}。"
        )
    return indices


# endregion


# region Predictor 推理


def infer_predictor_horizon(
    *,
    history_npz: Path,
    history_report: dict,
    source_npz: Path,
    predictor_model_path: Path,
    normalizer_dir: Path,
    current_frame: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """恢复十帧 deployed history，并返回 Predictor 的 `[11,144]` 输出。"""

    source = load_realtime_source(require_file(source_npz, "source_npz"))
    frame = int(current_frame)
    source_frame_count = int(source["tracker_pos_world"].shape[0])
    if not 12 <= frame < source_frame_count:
        raise ValueError(
            f"current_frame 必须位于 [12,{source_frame_count})，实际为 {frame}。"
        )

    report_start = int(history_report["frame_start"])
    report_end = int(history_report["frame_end_exclusive"])
    if not report_start + 10 <= frame < report_end:
        raise ValueError(
            f"history 只覆盖 [{report_start},{report_end})，"
            f"无法恢复 frame {frame} 的十帧历史。"
        )
    history_start = frame - 10 - report_start
    history_end = frame - report_start
    with np.load(require_file(history_npz, "history_npz"), allow_pickle=False) as payload:
        if "deployed_rotations_world" not in payload.files:
            raise KeyError("history_npz 缺少 deployed_rotations_world。")
        motion_rotations_world = np.asarray(
            payload["deployed_rotations_world"][history_start:history_end],
            dtype=np.float32,
        )
    if motion_rotations_world.shape != (10, 24, 3, 3):
        raise ValueError(
            "恢复出的 deployed history 应为 [10,24,3,3]，"
            f"实际为 {motion_rotations_world.shape}。"
        )

    features = build_predictor_step_features_np(
        motion_rotations_world=motion_rotations_world,
        tracker_positions_world_with_previous=source["tracker_pos_world"][
            frame - 11 : frame + 1
        ],
        tracker_rotations_world_6d_with_previous=source[
            "tracker_rot_world_6d"
        ][frame - 11 : frame + 1],
        floor_y=float(source["root_pos_world"][frame, 1]),
    )
    normalizer = RealtimePoseNormalizer(normalizer_dir, disable=False)
    predictor = load_realtime_pose_predictor(
        predictor_model_path,
        device=device,
    )
    motion_normalized = np.asarray(
        normalizer.normalize_pose(features.motion_context), dtype=np.float32
    )
    sparse_normalized = np.asarray(
        normalizer.normalize_predictor_sparse(features.core_tracker_context),
        dtype=np.float32,
    )
    with torch.no_grad():
        horizon_normalized = predictor(
            torch.as_tensor(
                motion_normalized[None], device=device, dtype=torch.float32
            ),
            torch.as_tensor(
                sparse_normalized[None], device=device, dtype=torch.float32
            ),
        )[0]
        predictor_horizon = np.asarray(
            normalizer.inverse_pose(horizon_normalized).detach().cpu(),
            dtype=np.float32,
        )
    if predictor_horizon.shape != (11, 144):
        raise RuntimeError(
            "Predictor horizon 应为 [11,144]，"
            f"实际为 {predictor_horizon.shape}。"
        )
    history_frames = np.arange(frame - 10, frame, dtype=np.int64)
    return predictor_horizon, source, history_frames


# endregion


# region 男性 SMPL-H 与当前加未来五帧成图


def build_future_render(
    *,
    predictor_horizon: np.ndarray,
    source: dict[str, np.ndarray],
    horizon_indices: np.ndarray,
    current_frame: int,
    smpl_model_dir: Path,
) -> tuple[PastMotionRender, np.ndarray, np.ndarray, np.ndarray]:
    """把当前帧与五个 Predictor future poses 转移到男性 SMPL-H。"""

    rotations_head, root_yaw_head = decode_target_head_rotations_np(
        predictor_horizon
    )
    display_indices = np.concatenate(
        [np.asarray([0], dtype=np.int64), horizon_indices]
    )
    selected_rotations = np.asarray(
        rotations_head[display_indices], dtype=np.float32
    )
    selected_root_yaw = np.asarray(
        root_yaw_head[display_indices], dtype=np.float32
    )
    rest_rotations = rotation_6d_to_matrix_np(
        source["joint_rest_local_rotations_6d"]
    )
    local_rotations = body_fbx_world_to_smpl_local_rotations(
        selected_rotations,
        selected_root_yaw,
        rest_rotations,
        SMPL_PARENTS,
    )
    pose_axis_angle = rotation_matrices_to_axis_angle(
        local_rotations[:, :22]
    )
    model = create_smplh_model(
        model_dir=require_directory(smpl_model_dir, "smpl_model_dir"),
        gender="male",
        batch_size=int(display_indices.shape[0]),
    )
    sequence = run_smplh_forward(
        model=model,
        pose_axis_angle=pose_axis_angle,
        betas=np.zeros((10,), dtype=np.float32),
        translation_amass=np.zeros(
            (display_indices.shape[0], 3), dtype=np.float32
        ),
    )

    # 五个 future poses 共用最后一帧确定的展示 yaw，因此仍保留 Predictor
    # 预测出的相对躯干转动，而不是逐帧强制正面化。
    front_rotation, presentation_yaw_deg = build_front_alignment_rotation(
        selected_rotations[-5:]
    )
    oriented_sequence = SmplMeshSequence(
        vertices_world=(
            np.asarray(sequence.vertices_world, dtype=np.float64)
            @ front_rotation.T
        ).astype(np.float32),
        joints_world=(
            np.asarray(sequence.joints_world, dtype=np.float64)
            @ front_rotation.T
        ).astype(np.float32),
    )
    display_source_frames = int(current_frame) + display_indices
    render = PastMotionRender(
        sequence=oriented_sequence,
        faces=transform_faces_to_unity_winding(model.faces),
        frame_indices=display_source_frames,
        current_frame=int(current_frame),
        source_fps=30.0,
        presentation_yaw_deg=float(presentation_yaw_deg),
    )
    return render, display_indices, selected_rotations, selected_root_yaw


# endregion


# region Sidecar 与入口


def write_sidecars(
    *,
    output_png: Path,
    predictor_horizon: np.ndarray,
    selected_rotations: np.ndarray,
    selected_root_yaw: np.ndarray,
    display_indices: np.ndarray,
    horizon_indices: np.ndarray,
    history_frames: np.ndarray,
    current_frame: int,
    history_npz: Path,
    source_npz: Path,
    predictor_model_path: Path,
    normalizer_dir: Path,
    presentation_yaw_deg: float,
) -> tuple[Path, Path]:
    output_npz = output_png.with_suffix(".npz")
    output_json = output_png.with_suffix(".json")
    np.savez_compressed(
        output_npz,
        current_frame=np.asarray(current_frame, dtype=np.int32),
        history_source_frames=history_frames,
        horizon_indices=horizon_indices,
        display_horizon_indices=display_indices,
        predictor_pose_horizon=predictor_horizon,
        display_rotations_head=selected_rotations,
        display_root_yaw_head=selected_root_yaw,
    )
    report = {
        "asset": "predictor_output_current_plus_five_future_frames",
        "current_frame": int(current_frame),
        "history_source_frames": history_frames.astype(int).tolist(),
        "predictor_future_horizon": list(range(1, 11)),
        "current_output_offset": 0,
        "selected_future_offsets": horizon_indices.astype(int).tolist(),
        "display_offsets": display_indices.astype(int).tolist(),
        "display_source_frames": (
            int(current_frame) + display_indices
        ).astype(int).tolist(),
        "inference": "Predictor only; no IK and no DiT",
        "body_model": "SMPL-H male, zeros(10) betas",
        "projection": "front orthographic",
        "presentation_yaw_deg": float(presentation_yaw_deg),
        "resolution": [DEFAULT_WIDTH, DEFAULT_HEIGHT],
        "background_rgb": [255, 255, 255],
        "overlays": "none",
        "inputs": {
            "history_npz": str(Path(history_npz).resolve()),
            "source_npz": str(Path(source_npz).resolve()),
            "predictor_model_path": str(Path(predictor_model_path).resolve()),
            "normalizer_dir": str(Path(normalizer_dir).resolve()),
        },
        "outputs": {
            "png": str(output_png),
            "npz": str(output_npz),
        },
    }
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_npz, output_json


def main(argv: list[str] | None = None) -> tuple[Path, Path, Path]:
    args = build_arg_parser().parse_args(argv)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    _, history_report = load_json_object(args.history_json, "history_json")
    source_path = resolve_optional_path(
        args.source_npz,
        history_report.get("source_path", ""),
        label="source_npz",
    )
    predictor_path = resolve_optional_path(
        args.predictor_model_path,
        history_report.get("predictor_model_path", ""),
        label="predictor_model_path",
    )
    dit_model_path = require_file(
        Path(str(history_report.get("dit_model_path", ""))),
        "dit_model_path",
    )
    _, dit_args = load_json_object(
        dit_model_path.with_name("args.json"), "DiT args.json"
    )
    normalizer_dir = resolve_optional_path(
        args.normalizer_dir,
        dit_args.get("normalizer_dir", ""),
        label="normalizer_dir",
        directory=True,
    )
    device = select_device(args.device)
    horizon_indices = validate_horizon_indices(args.horizon_indices)
    predictor_horizon, source, history_frames = infer_predictor_horizon(
        history_npz=args.history_npz,
        history_report=history_report,
        source_npz=source_path,
        predictor_model_path=predictor_path,
        normalizer_dir=normalizer_dir,
        current_frame=int(args.current_frame),
        device=device,
    )
    (
        render,
        display_indices,
        selected_rotations,
        selected_root_yaw,
    ) = build_future_render(
        predictor_horizon=predictor_horizon,
        source=source,
        horizon_indices=horizon_indices,
        current_frame=int(args.current_frame),
        smpl_model_dir=args.smpl_model_dir,
    )
    layout = build_orthographic_layout(
        render,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
    )
    image = render_past_motion_image(
        render=render,
        layout=layout,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
    )
    validate_rendered_image(
        image,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
    )

    output_png = Path(args.output_png).expanduser().resolve()
    if output_png.suffix.lower() != ".png":
        raise ValueError(f"output_png 必须使用 .png 后缀：{output_png}")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png)
    output_npz, output_json = write_sidecars(
        output_png=output_png,
        predictor_horizon=predictor_horizon,
        selected_rotations=selected_rotations,
        selected_root_yaw=selected_root_yaw,
        display_indices=display_indices,
        horizon_indices=horizon_indices,
        history_frames=history_frames,
        current_frame=int(args.current_frame),
        history_npz=args.history_npz,
        source_npz=source_path,
        predictor_model_path=predictor_path,
        normalizer_dir=normalizer_dir,
        presentation_yaw_deg=render.presentation_yaw_deg,
    )
    print(f"[predictor-future] device: {device}", flush=True)
    print(
        f"[predictor-future] history: {history_frames.tolist()} "
        f"-> current {int(args.current_frame)}",
        flush=True,
    )
    print(
        f"[predictor-future] display offsets: "
        f"{display_indices.tolist()}",
        flush=True,
    )
    print(f"[predictor-future] wrote: {output_png}", flush=True)
    print(f"[predictor-future] wrote: {output_npz}", flush=True)
    print(f"[predictor-future] wrote: {output_json}", flush=True)
    return output_png, output_npz, output_json


# endregion


if __name__ == "__main__":
    main()
