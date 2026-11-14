"""独立实验的文件校验、模型加载和共享运行设置。"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from eval.scene_guidance import GuidanceConfig, SceneGeometry
from export.residual_collision_postprocess import load_collision_profile
from export.unity_onnx_models import PoseSampler
from sample.utils import load_checkpoint_model
from utils.model_util import create_model_and_diffusion
from utils.normalizer import RealtimePoseNormalizer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NORMALIZER_FILES = ("pose_mean.pt", "pose_scale.pt", "tracker_mean.pt", "tracker_std.pt",
                    "predictor_sparse_mean.pt", "predictor_sparse_std.pt")


def file_hash(path):
    """文本统一 LF 后计算 SHA256，避免 Windows checkout 的换行转换影响复现校验。"""
    path = Path(path)
    if path.suffix in {".json", ".py"}:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_name(path):
    path = Path(path).resolve()
    return path.relative_to(REPOSITORY_ROOT).as_posix() if path.is_relative_to(REPOSITORY_ROOT) else str(path)


def guidance_configs(args):
    configs = [GuidanceConfig(**row) for row in args.guidance_configs]
    if not configs:
        raise ValueError("配置必须提供非空 guidance_configs。")
    for config in configs:
        if config.method not in {"none", "postedit", "lm", "pigdm"}:
            raise ValueError(f"未知方法：{config.method}")
        if type(config.iterations) is not int or config.iterations < 1:
            raise ValueError("iterations 必须是正整数。")
        if any(not math.isfinite(value) or value <= 0 for value in
               (config.step_size, config.damping, config.strength, config.covariance_scale)):
            raise ValueError("引导参数必须是有限正数。")
    if len({config.name for config in configs}) != len(configs):
        raise ValueError("实验配置名称重复。")
    if args.methods:
        configs = [config for config in configs if config.method in args.methods]
        missing = set(args.methods) - {config.method for config in configs}
        if missing:
            raise ValueError(f"配置未提供请求的方法：{sorted(missing)}")
    if args.protocol == "sequence" and len({c.method for c in configs}) != len(configs):
        raise ValueError("长序列每种方法只能提供一组配置。")
    if args.protocol == "snapshot":
        # 逐帧姿态改变量和额外耗时始终需要同噪声的无修正基线。
        configs = [GuidanceConfig()] + [config for config in configs if config.method != "none"]
    return configs


def source_paths(args):
    paths = [Path(args.config), Path(args.model_path), Path(args.model_path).with_name("args.json"),
             Path(args.body_fbx_rest_json), Path(args.collision_profile), Path(args.replay_inputs)]
    paths += [Path(args.normalizer_dir) / name for name in NORMALIZER_FILES]
    if args.protocol == "sequence":
        paths += [Path(args.recording), Path(args.predictor_model_path),
                  Path(args.predictor_model_path).with_name("args.json")]
    return paths


def validate_inputs(args):
    guidance_configs(args)
    if (not math.isfinite(args.measurement_scale) or args.measurement_scale <= 0
            or not math.isfinite(args.margin) or args.margin < 0):
        raise ValueError("观测尺度须为有限正数，安全间隙须为有限非负数。")
    paths = source_paths(args)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"实验缺少必要输入：{missing}")
    sources = {source_name(path): file_hash(path) for path in paths}
    manifest_path = REPOSITORY_ROOT / "experiments/scene_guidance/inputs_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["sha256"]
    changed = [name for name, digest in sources.items() if name in manifest and digest != manifest[name]]
    if changed:
        raise ValueError(f"随仓库发布的实验输入校验失败：{changed}")
    # 记录实际代码来源，包含现有共享 runtime，便于区分输入和实现的变动。
    code_names = ("eval/evaluate_scene_guidance.py", "eval/scene_guidance.py", "eval/scene_guidance_cache.py",
                  "eval/scene_guidance_setup.py", "eval/scene_guidance_benchmark.py", "eval/scene_guidance_sequence.py",
                  "export/unity_onnx_models.py", "export/residual_collision_postprocess.py",
                  "sample/realtime_pose_runtime.py", "sample/infer_unity_recording.py")
    sources.update({name: file_hash(REPOSITORY_ROOT / name) for name in code_names})
    return sources


def prepare_output_directory(args):
    directory = Path(args.output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"结果目录非空，请指定新的 --output_dir：{directory}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def create_scene_sampler(args):
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("正式效率测量要求 CUDA GPU；输入校验和轻量 smoke test 可使用 CPU。")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dit, diffusion = create_model_and_diffusion(args)
    dit, weight_kind = load_checkpoint_model(dit, args.model_path, device, use_ema=True)
    if diffusion.num_timesteps != 5:
        raise ValueError("本实验固定五步 DDIM。")
    normalizer = RealtimePoseNormalizer(args.normalizer_dir)
    sampler = PoseSampler(dit, diffusion, normalizer).to(device).eval().requires_grad_(False)
    rest = json.loads(Path(args.body_fbx_rest_json).read_text(encoding="utf-8-sig"))
    profile = load_collision_profile(Path(args.collision_profile), rest)
    scene = SceneGeometry(rest, profile, normalizer.pose_mean, normalizer.pose_scale, args.margin).to(device)
    return sampler, scene, normalizer, diffusion, weight_kind
