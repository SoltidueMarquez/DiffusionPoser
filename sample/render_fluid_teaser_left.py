from __future__ import annotations

import argparse
from contextlib import suppress
from dataclasses import dataclass
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from data_converter.amass_smpl_utils import (
    SOURCE_BODY_JOINT_COUNT,
    load_motion_source,
    normalize_gender,
)
from data_loaders.body_fbx_kinematics import BodyFbxRest, load_body_fbx_rest
from data_loaders.generate_realtime_pose_tasks import load_realtime_source
from data_loaders.realtime_pose_kinematics import JOINT_INDEX
from data_loaders.sensor_masking import (
    REALTIME_POSE_EVAL_METRICS_START_FRAME,
    REALTIME_POSE_FPS,
    SMPL_JOINT_COUNT,
)
from sample.realtime_pose_smpl_rendering import (
    SmplMeshSequence,
    body_fbx_world_to_smpl_local_rotations,
    create_smplh_model,
    create_sphere_cloud,
    create_static_scene,
    load_font,
    read_scalar_string,
    require_directory,
    require_file,
    rotation_matrices_to_axis_angle,
    run_smplh_forward,
    transform_faces_to_unity_winding,
)
from sample.render_realtime_pose_smpl_presentation import (
    CAMERA_AZIMUTH_DEG,
    CAMERA_ELEVATION_DEG,
    CAMERA_FIT_PADDING,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    build_presentation_layout,
    build_visible_tracker_glyph_points,
    create_material,
    draw_centered_text,
)
from sample.render_realtime_pose_smpl_tracker_counts import (
    METHOD_CONFIG_NAMES,
    METHOD_ORDER,
    TRACKER_AVAILABLE_BY_METHOD,
    build_active_tracker_points,
    run_tracker_count_inference,
)
from sample.utils import load_checkpoint_model
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_realtime_pose_predictor
from utils.normalizer import RealtimePoseNormalizer
from utils.parser_util import (
    add_base_options,
    add_diffusion_options,
    add_model_options,
    add_sampling_options,
    parse_and_load_from_model,
    str2bool,
)


DEFAULT_SOURCE_FRAME = 204
TEASER_METHOD_ORDER = tuple(reversed(METHOD_ORDER))
TEASER_CONFIG_NAMES = tuple(reversed(METHOD_CONFIG_NAMES))
TEASER_TRACKER_AVAILABLE = tuple(reversed(TRACKER_AVAILABLE_BY_METHOD))

# 置信度只编码为不透明表面的颜色深浅，避免地面透过网格后产生“高置信更浅”
# 的视觉错觉；Tracker 按论文图注固定为洋红色。
LOW_CONFIDENCE_COLOR = (0.53, 0.68, 0.70)
HIGH_CONFIDENCE_COLOR = (0.06, 0.42, 0.47)
# 仅在显示阶段压缩置信度动态范围，让手臂等间接受 Tracker
# 约束的低置信区域仍清晰可见；原始数值、0/1 端点和排序均不改变。
CONFIDENCE_DISPLAY_GAMMA = 0.13
TRACKER_COLOR = (1.0, 0.0, 0.72, 1.0)
TRACKER_RADIUS = 0.040


@dataclass(frozen=True)
class TeaserBodySource:
    """单帧 FLUID 输出恢复为 SMPL-H 网格所需的源数据。"""

    body_fbx_rest: BodyFbxRest
    betas: np.ndarray
    gender: str
    translation_amass: np.ndarray  # [1,3]


# region CLI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render the FLUID teaser left panel from one shared burpee frame "
            "under nested 6/5/4/3 tracker configurations."
        )
    )
    add_base_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sampling_options(parser)
    paths = parser.add_argument_group("FLUID teaser paths")
    paths.add_argument("--source_npz", required=True, type=Path)
    paths.add_argument("--amass_npz", required=True, type=Path)
    paths.add_argument("--normalizer_dir", required=True, type=Path)
    paths.add_argument("--normalize_input", default=True, type=str2bool)
    paths.add_argument("--smpl_model_dir", required=True, type=Path)
    paths.add_argument("--output_png", required=True, type=Path)
    frame = parser.add_argument_group("FLUID teaser frame")
    frame.add_argument("--source_frame", default=DEFAULT_SOURCE_FRAME, type=int)
    frame.add_argument(
        "--camera_fit_padding",
        default=CAMERA_FIT_PADDING,
        type=float,
        help="左图自动相机拟合留白；数值越大，人物在画面中越小。",
    )
    frame.add_argument(
        "--camera_azimuth_deg",
        default=CAMERA_AZIMUTH_DEG,
        type=float,
        help="左图观察方位角；默认保持原正面展示。",
    )
    return parser


# endregion


# region Source 与 SMPL-H


def load_teaser_body_source(
    *,
    source_npz: Path,
    amass_npz: Path,
    source_frame: int,
) -> TeaserBodySource:
    """加载同一 source frame 对应的 body.fbx rest、身形与 AMASS 根平移。"""

    source_path = require_file(source_npz, "source_npz")
    amass_path = require_file(amass_npz, "amass_npz")
    frame = int(source_frame)
    with np.load(source_path, allow_pickle=False) as payload:
        body_fbx_rest_path = require_file(
            Path(read_scalar_string(payload, "body_fbx_rest_json")),
            "body_fbx_rest_json",
        )
        source_frame_count = int(payload["tracker_pos_world"].shape[0])
    if frame < 0 or frame >= source_frame_count:
        raise ValueError(
            f"source_frame={frame} 超出 converted source 长度 {source_frame_count}。"
        )

    # 与 source converter 使用同一套 Slerp/translation 插值，保证所选帧的
    # SMPL-H 根平移和 Tracker、FLUID 输出处于完全相同的 30 Hz 时间点。
    amass_source = load_motion_source(
        path=amass_path,
        amass_dir=amass_path.parent,
        target_fps=float(REALTIME_POSE_FPS),
    )
    if frame >= int(amass_source.trans.shape[0]):
        raise ValueError(
            f"source_frame={frame} 超出重采样 AMASS 长度 "
            f"{amass_source.trans.shape[0]}。"
        )
    return TeaserBodySource(
        body_fbx_rest=load_body_fbx_rest(body_fbx_rest_path),
        betas=np.asarray(amass_source.betas, dtype=np.float32).reshape(-1),
        gender=normalize_gender(amass_source.gender),
        translation_amass=np.asarray(
            amass_source.trans[frame : frame + 1], dtype=np.float32
        ),
    )


def validate_teaser_results(
    results: dict[str, dict[str, np.ndarray]],
) -> None:
    """验证四路单帧姿态与逐关节置信度，避免渲染静默使用错位数据。"""

    if tuple(results.keys()) != METHOD_ORDER:
        raise ValueError(
            f"results 必须按 {METHOD_ORDER} 排列，实际为 {tuple(results.keys())}"
        )
    for method_name in METHOD_ORDER:
        result = results[method_name]
        expected_shapes = {
            "rotations": (1, SMPL_JOINT_COUNT, 3, 3),
            "positions": (1, SMPL_JOINT_COUNT, 3),
            "root_yaw": (1,),
            "ik_confidence": (1, SMPL_JOINT_COUNT),
        }
        for key, expected_shape in expected_shapes.items():
            value = np.asarray(result[key])
            if value.shape != expected_shape:
                raise ValueError(
                    f"{method_name}.{key} 应为 {expected_shape}，实际为 {value.shape}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"{method_name}.{key} 含 NaN/Inf。")
        confidence = np.asarray(result["ik_confidence"], dtype=np.float32)
        if np.any(confidence < 0.0) or np.any(confidence > 1.0):
            raise ValueError(f"{method_name}.ik_confidence 必须位于 [0,1]。")


def collapse_smplh_skinning_weights(model) -> np.ndarray:
    """把 SMPL-H 的 52 个 LBS joint 合并为 FLUID 使用的 24 个逻辑 joint。

    SMPL-H 的前 22 个 joint 是身体，22:37 和 37:52 分别是左右手链。
    FLUID 只保留一个 left_hand/right_hand joint，因此把两组手指权重分别求和。
    返回的逻辑权重为 `[V,24]`，每个顶点再次归一化到和为 1。
    """

    weights = np.asarray(model.lbs_weights.detach().cpu(), dtype=np.float32)
    if weights.ndim != 2 or weights.shape[1] != 52:
        raise ValueError(f"SMPL-H lbs_weights 应为 [V,52]，实际为 {weights.shape}")
    logical = np.zeros((weights.shape[0], SMPL_JOINT_COUNT), dtype=np.float32)
    logical[:, :SOURCE_BODY_JOINT_COUNT] = weights[:, :SOURCE_BODY_JOINT_COUNT]
    logical[:, 22] = np.sum(weights[:, 22:37], axis=1)
    logical[:, 23] = np.sum(weights[:, 37:52], axis=1)
    row_sum = np.sum(logical, axis=1, keepdims=True)
    if np.any(row_sum <= 1e-8):
        raise ValueError("SMPL-H skinning weights 存在总权重为零的顶点。")
    return (logical / row_sum).astype(np.float32)


def build_teaser_mesh_sequences(
    *,
    body_source: TeaserBodySource,
    results: dict[str, dict[str, np.ndarray]],
    smpl_model_dir: Path,
) -> tuple[dict[str, SmplMeshSequence], np.ndarray, np.ndarray]:
    """将四路同帧 FLUID 姿态转换为共享身形和根平移的 SMPL-H 网格。"""

    validate_teaser_results(results)
    model = create_smplh_model(
        model_dir=require_directory(smpl_model_dir, "smpl_model_dir"),
        gender=body_source.gender,
        batch_size=1,
    )
    sequences: dict[str, SmplMeshSequence] = {}
    for method_name in METHOD_ORDER:
        result = results[method_name]
        local_rotations = body_fbx_world_to_smpl_local_rotations(
            result["rotations"],
            result["root_yaw"],
            body_source.body_fbx_rest.rest_local_rotations,
            body_source.body_fbx_rest.parents,
        )
        pose_axis_angle = rotation_matrices_to_axis_angle(
            local_rotations[:, :SOURCE_BODY_JOINT_COUNT]
        )
        sequences[method_name] = run_smplh_forward(
            model=model,
            pose_axis_angle=pose_axis_angle,
            betas=body_source.betas,
            translation_amass=body_source.translation_amass,
        )
    return (
        sequences,
        transform_faces_to_unity_winding(model.faces),
        collapse_smplh_skinning_weights(model),
    )


def confidence_to_vertex_colors(
    confidence: np.ndarray,
    logical_skinning_weights: np.ndarray,
) -> np.ndarray:
    """把 `[24]` 置信度经 LBS 连续插值为不透明的 `[V,4]` RGBA。

    SMPL-H 手掌/手指顶点主要绑定到 hand/finger joint，但当前 Tracker 位于
    Wrist，因此左右手表面先继承对应 Wrist 的置信度。之后使用每个顶点的完整
    24-joint skinning weights 混合颜色，使相邻骨段之间连续过渡而不出现硬边。
    """

    joint_confidence = np.asarray(confidence, dtype=np.float32)
    weights = np.asarray(logical_skinning_weights, dtype=np.float32)
    if joint_confidence.shape != (SMPL_JOINT_COUNT,):
        raise ValueError(
            f"confidence 应为 {(SMPL_JOINT_COUNT,)}，实际为 {joint_confidence.shape}"
        )
    if weights.ndim != 2 or weights.shape[1] != SMPL_JOINT_COUNT:
        raise ValueError(f"logical_skinning_weights 应为 [V,24]，实际为 {weights.shape}")
    display_confidence = joint_confidence.copy()
    display_confidence[22] = display_confidence[20]
    display_confidence[23] = display_confidence[21]
    display_confidence = np.power(
        np.clip(display_confidence, 0.0, 1.0),
        CONFIDENCE_DISPLAY_GAMMA,
    )
    vertex_confidence = np.clip(weights @ display_confidence, 0.0, 1.0)
    low = np.asarray(LOW_CONFIDENCE_COLOR, dtype=np.float32)
    high = np.asarray(HIGH_CONFIDENCE_COLOR, dtype=np.float32)
    rgba = np.empty((weights.shape[0], 4), dtype=np.float32)
    rgba[:, :3] = low[None] + vertex_confidence[:, None] * (high - low)[None]
    rgba[:, 3] = 1.0
    return np.rint(rgba * 255.0).astype(np.uint8)


# endregion


# region PNG 与 sidecar


def orient_teaser_stage(
    sequences: dict[str, SmplMeshSequence],
    tracker_frame: np.ndarray,
    reference_rotations: np.ndarray,
) -> tuple[dict[str, SmplMeshSequence], np.ndarray]:
    """只旋转展示坐标，使四个站立人体统一正面朝向相机。

    旋转角由 6-tracker 输出的 Spine3 正面方向确定，并同样作用于四路网格和
    Tracker；模型输出、sidecar 与所有相对几何都保持原值。
    """

    trackers = np.asarray(tracker_frame, dtype=np.float64)
    if trackers.shape != (6, 3):
        raise ValueError(f"tracker_frame 应为 [6,3]，实际为 {trackers.shape}")
    rotations = np.asarray(reference_rotations, dtype=np.float64)
    if rotations.shape != (SMPL_JOINT_COUNT, 3, 3):
        raise ValueError(
            "reference_rotations 应为 [24,3,3]，"
            f"实际为 {rotations.shape}"
        )
    torso_reference = -rotations[JOINT_INDEX["spine3"], :, 2]
    torso_reference_xz = torso_reference[[0, 2]]
    if float(np.linalg.norm(torso_reference_xz)) <= 1e-8:
        raise ValueError("Spine3 参考方向的水平分量为零，无法确定展示朝向。")
    current_angle = math.atan2(
        float(torso_reference_xz[1]),
        float(torso_reference_xz[0]),
    )
    # body.fbx 的 Spine3 参考轴在逆变换到 SMPL-H 后对应可见背面方向，
    # 因此让该轴背向相机，胸腹面才会真正朝向观察者。
    desired_angle = math.radians(CAMERA_AZIMUTH_DEG + 180.0)
    presentation_yaw = desired_angle - current_angle
    cos_yaw = math.cos(presentation_yaw)
    sin_yaw = math.sin(presentation_yaw)
    rotation = np.asarray(
        [
            [cos_yaw, 0.0, -sin_yaw],
            [0.0, 1.0, 0.0],
            [sin_yaw, 0.0, cos_yaw],
        ],
        dtype=np.float64,
    )
    oriented_sequences = {
        method_name: SmplMeshSequence(
            vertices_world=(
                np.asarray(sequence.vertices_world, dtype=np.float64)
                @ rotation.T
            ).astype(np.float32),
            joints_world=(
                np.asarray(sequence.joints_world, dtype=np.float64)
                @ rotation.T
            ).astype(np.float32),
        )
        for method_name, sequence in sequences.items()
    }
    return (
        oriented_sequences,
        (trackers @ rotation.T).astype(np.float32),
    )


def compose_teaser_image(viewport_rgb: np.ndarray) -> Image.Image:
    """只叠加四个 tracker 数量标签，不引入指标或额外叙事文字。"""

    image = Image.fromarray(np.asarray(viewport_rgb, dtype=np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    centers = tuple(
        int(round(OUTPUT_WIDTH * (index + 0.5) / len(TEASER_METHOD_ORDER)))
        for index in range(len(TEASER_METHOD_ORDER))
    )
    for method_name, center_x in zip(TEASER_METHOD_ORDER, centers):
        label_width = 230
        box = (
            center_x - label_width // 2,
            14,
            center_x + label_width // 2,
            70,
        )
        draw.rounded_rectangle(
            box,
            radius=18,
            fill=(255, 255, 255, 232),
            outline=(205, 211, 219, 238),
            width=2,
        )
        draw_centered_text(
            draw,
            box,
            method_name,
            load_font(22),
            (31, 41, 55, 255),
        )
    return Image.alpha_composite(image, overlay).convert("RGB")


def render_teaser_view(
    *,
    renderer,
    scene,
    sequences: dict[str, SmplMeshSequence],
    faces: np.ndarray,
    results: dict[str, dict[str, np.ndarray]],
    logical_skinning_weights: np.ndarray,
    tracker_frame: np.ndarray,
    method_offsets: np.ndarray,
    camera_pose: np.ndarray,
) -> np.ndarray:
    """一次渲染四个置信度着色人体和各自真实启用的洋红 Tracker。"""

    import pyrender
    import trimesh

    dynamic_nodes = []
    try:
        for method_index, method_name in enumerate(TEASER_METHOD_ORDER):
            vertices = np.asarray(
                sequences[method_name].vertices_world[0], dtype=np.float64
            ) + np.asarray(method_offsets[method_index], dtype=np.float64)
            body = trimesh.Trimesh(
                vertices=vertices,
                faces=np.asarray(faces, dtype=np.int64),
                process=False,
            )
            body.visual.vertex_colors = confidence_to_vertex_colors(
                results[method_name]["ik_confidence"][0],
                logical_skinning_weights,
            )
            render_mesh = pyrender.Mesh.from_trimesh(body, smooth=True)
            # Mesh.from_trimesh 对 vertex color 默认创建 BLEND material；本图所有
            # 顶点均完全不透明，显式切回 OPAQUE 避免透明排序改变颜色观感。
            for primitive in render_mesh.primitives:
                primitive.material.alphaMode = "OPAQUE"
                primitive.material.metallicFactor = 0.0
                primitive.material.roughnessFactor = 0.82
            dynamic_nodes.append(scene.add(render_mesh))

        active_points = build_active_tracker_points(
            tracker_frame,
            method_offsets,
            tracker_available_by_method=TEASER_TRACKER_AVAILABLE,
        )
        visible_tracker_points = build_visible_tracker_glyph_points(
            np.concatenate(active_points, axis=0),
            np.asarray(camera_pose, dtype=np.float64)[:3, 3],
        )
        tracker_cloud = create_sphere_cloud(
            visible_tracker_points,
            radius=TRACKER_RADIUS,
        )
        tracker_mesh = pyrender.Mesh.from_trimesh(
            tracker_cloud,
            material=create_material(pyrender, TRACKER_COLOR, 0.45),
            smooth=True,
        )
        dynamic_nodes.append(scene.add(tracker_mesh))
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.NONE)
        return np.asarray(color[..., :3], dtype=np.uint8)
    finally:
        for node in dynamic_nodes:
            scene.remove_node(node)


def render_teaser_png(
    *,
    output_png: Path,
    sequences: dict[str, SmplMeshSequence],
    faces: np.ndarray,
    results: dict[str, dict[str, np.ndarray]],
    logical_skinning_weights: np.ndarray,
    tracker_frame: np.ndarray,
    camera_fit_padding: float = CAMERA_FIT_PADDING,
    camera_azimuth_deg: float = CAMERA_AZIMUTH_DEG,
) -> Path:
    """构造共享舞台并输出固定为 1920×1080 的左侧 teaser PNG。"""

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    try:
        import pyrender
    except ImportError as exc:
        raise ImportError("缺少 pyrender，无法执行 FLUID teaser 离屏渲染。") from exc

    oriented_sequences, oriented_tracker_frame = orient_teaser_stage(
        sequences,
        tracker_frame,
        results[TEASER_METHOD_ORDER[0]]["rotations"][0],
    )
    display_sequences = {
        method_name: oriented_sequences[method_name]
        for method_name in TEASER_METHOD_ORDER
    }
    tracker_sequence = oriented_tracker_frame[None]
    camera_elevation = math.radians(CAMERA_ELEVATION_DEG)
    camera_azimuth = math.radians(float(camera_azimuth_deg))
    camera_view_direction = np.asarray(
        [
            math.cos(camera_elevation) * math.cos(camera_azimuth),
            math.sin(camera_elevation),
            math.cos(camera_elevation) * math.sin(camera_azimuth),
        ],
        dtype=np.float64,
    )
    layout = build_presentation_layout(
        sequences=display_sequences,
        tracker_pos_world=tracker_sequence,
        method_order=TEASER_METHOD_ORDER,
        tracker_available_by_method=np.asarray(
            TEASER_TRACKER_AVAILABLE, dtype=bool
        ),
        follow_method_name=TEASER_METHOD_ORDER[0],
        camera_fit_padding=float(camera_fit_padding),
        camera_view_direction=camera_view_direction,
    )
    floor_y = min(
        float(np.min(sequence.vertices_world[..., 1]))
        for sequence in display_sequences.values()
    )
    scene, camera_node = create_static_scene(
        layout.base_camera,
        floor_y=floor_y,
        grid_size=layout.grid_size,
        grid_center=layout.grid_center,
    )
    renderer = pyrender.OffscreenRenderer(OUTPUT_WIDTH, OUTPUT_HEIGHT)
    try:
        camera_pose = layout.camera_poses[0]
        scene.set_pose(camera_node, pose=camera_pose)
        viewport = render_teaser_view(
            renderer=renderer,
            scene=scene,
            sequences=display_sequences,
            faces=faces,
            results=results,
            logical_skinning_weights=logical_skinning_weights,
            tracker_frame=oriented_tracker_frame,
            method_offsets=layout.method_offsets,
            camera_pose=camera_pose,
        )
        image = compose_teaser_image(viewport)
        output = Path(output_png).expanduser().resolve()
        if output.suffix.lower() != ".png":
            raise ValueError(f"output_png 必须使用 .png 后缀，实际为 {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
    finally:
        with suppress(Exception):
            renderer.delete()
    print(
        f"[fluid-teaser] stage spacing={layout.stage_spacing:.3f} m, "
        f"size={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}",
        flush=True,
    )
    print(f"[fluid-teaser] wrote: {output}", flush=True)
    return output


def write_teaser_sidecar(
    *,
    output_png: Path,
    source_npz: Path,
    source_frame: int,
    tracker_frame: np.ndarray,
    results: dict[str, dict[str, np.ndarray]],
    args,
    dit_weight_source: str,
    sampling_steps: int,
) -> Path:
    """保存精确姿态、置信度和 Tracker mask，供后续无推理重排版。"""

    output = Path(output_png).expanduser().resolve()
    sidecar = output.with_suffix(".npz")
    arrays: dict[str, np.ndarray] = {
        "source_path": np.asarray(str(Path(source_npz).resolve())),
        "source_frame": np.asarray(int(source_frame), dtype=np.int32),
        "method_order": np.asarray(TEASER_METHOD_ORDER),
        "method_config_names": np.asarray(TEASER_CONFIG_NAMES),
        "tracker_pos_world": np.asarray(tracker_frame, dtype=np.float32),
        "tracker_available_by_method": np.asarray(
            TEASER_TRACKER_AVAILABLE, dtype=bool
        ),
        "dit_model_path": np.asarray(str(Path(args.dit_model_path).resolve())),
        "predictor_model_path": np.asarray(
            str(Path(args.predictor_model_path).resolve())
        ),
        "dit_weight_source": np.asarray(str(dit_weight_source)),
        "sampling_steps": np.asarray(int(sampling_steps), dtype=np.int32),
        "sampling_noise_seed": np.asarray(int(args.seed), dtype=np.int32),
    }
    for method_name, config_name in zip(METHOD_ORDER, METHOD_CONFIG_NAMES):
        result = results[method_name]
        arrays[f"{config_name}_rotations_world"] = result["rotations"][0]
        arrays[f"{config_name}_joints_world"] = result["positions"][0]
        arrays[f"{config_name}_root_yaw"] = result["root_yaw"][0]
        arrays[f"{config_name}_ik_confidence"] = result["ik_confidence"][0]
    np.savez_compressed(sidecar, **arrays)
    print(f"[fluid-teaser] wrote: {sidecar}", flush=True)
    return sidecar


# endregion


def main(argv: list[str] | None = None) -> Path:
    parser = build_arg_parser()
    args = parse_and_load_from_model(parser, argv)
    fixseed(args.seed)
    source_path = require_file(args.source_npz, "source_npz")
    source_frame = int(args.source_frame)
    if source_frame < REALTIME_POSE_EVAL_METRICS_START_FRAME:
        parser.error(
            f"--source_frame 不能早于正式输出帧 "
            f"{REALTIME_POSE_EVAL_METRICS_START_FRAME}。"
        )

    device = torch.device(
        f"cuda:{args.device}" if args.cuda and torch.cuda.is_available() else "cpu"
    )
    source = load_realtime_source(source_path)
    dit, diffusion = create_model_and_diffusion(args)
    dit, dit_weight_source = load_checkpoint_model(
        dit,
        args.dit_model_path,
        device,
        use_ema=args.use_ema,
    )
    predictor = load_realtime_pose_predictor(args.predictor_model_path, device)
    normalizer = RealtimePoseNormalizer(
        args.normalizer_dir,
        disable=not bool(args.normalize_input),
    )
    results = run_tracker_count_inference(
        source=source,
        predictor=predictor,
        dit=dit,
        diffusion=diffusion,
        device=device,
        normalizer=normalizer,
        args=args,
        source_frame_start=source_frame,
        source_frame_end_exclusive=source_frame + 1,
    )
    validate_teaser_results(results)
    body_source = load_teaser_body_source(
        source_npz=source_path,
        amass_npz=args.amass_npz,
        source_frame=source_frame,
    )
    sequences, faces, logical_skinning_weights = build_teaser_mesh_sequences(
        body_source=body_source,
        results=results,
        smpl_model_dir=args.smpl_model_dir,
    )
    tracker_frame = np.asarray(
        source["tracker_pos_world"][source_frame], dtype=np.float32
    )
    output = render_teaser_png(
        output_png=args.output_png,
        sequences=sequences,
        faces=faces,
        results=results,
        logical_skinning_weights=logical_skinning_weights,
        tracker_frame=tracker_frame,
        camera_fit_padding=float(args.camera_fit_padding),
        camera_azimuth_deg=float(args.camera_azimuth_deg),
    )
    write_teaser_sidecar(
        output_png=output,
        source_npz=source_path,
        source_frame=source_frame,
        tracker_frame=tracker_frame,
        results=results,
        args=args,
        dit_weight_source=dit_weight_source,
        sampling_steps=int(diffusion.num_timesteps),
    )
    return output


if __name__ == "__main__":
    main()
