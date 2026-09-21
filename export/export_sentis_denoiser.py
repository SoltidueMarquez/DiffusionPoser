"""导出固定 batch=1 的 Unity Predictor 和完整 residual sampler。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from export.unity_onnx_models import PoseSampler, SAMPLER_INPUTS, example_inputs
from export.collision_compute_export import export_collision_chain
from export.residual_collision_postprocess import (
    COLLISION_INPUTS, CollisionSettings, ResidualCollisionPostprocess,
    collision_manifest, load_collision_profile,
)
from sample.utils import load_checkpoint_model
from utils.model_util import create_model_and_diffusion, load_realtime_pose_predictor
from utils.normalizer import RealtimePoseNormalizer
from utils.parser_util import (add_model_options, add_diffusion_options,
                               add_ik_inpainting_options, parse_and_load_from_model)

IK_KEYS = ('fabrik_iterations', 'ik_direction_only_quality', 'ik_residual_scale',
           'ik_position_solved_quality', 'ik_gap_low', 'ik_gap_high',
           'ik_direction_support', 'ik_untracked_strength')
STATS = ('pose_mean', 'pose_scale', 'tracker_mean', 'tracker_std',
         'predictor_sparse_mean', 'predictor_sparse_std')
COLLISION_KEYS = {'collision_postprocess', 'collision_profile', 'postedit_steps',
                  'postedit_h', 'postedit_m', 'postedit_difference_step',
                  'collision_margin', 'collision_tracking_tolerance'}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_ik_inpainting_options(parser)
    parser.add_argument('--predictor_model_path', type=Path, required=True)
    parser.add_argument('--model_path', type=Path, required=True)
    parser.add_argument('--normalizer_dir', type=Path, required=True)
    parser.add_argument('--body_fbx_rest_json', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, default=Path('output/unity_onnx'))
    parser.add_argument('--sampler_only', action='store_true',
                        help='仅导出 Sampler，不改 Predictor、配置或角色 rest 文件。')
    parser.add_argument('--sampler_filename', default=None,
                        help='普通 Sampler 文件名；碰撞命令链使用固定的三个资产名。')
    parser.add_argument('--collision_postprocess', action='store_true',
                        help='仅在部署 sampler 中加入可选的逐步残差碰撞修正。')
    parser.add_argument('--collision_profile', type=Path,
                        help='Unity 编辑器标定的角色碰撞代理 JSON；开启后处理时必须提供。')
    parser.add_argument('--postedit_steps', type=int, default=5)
    parser.add_argument('--postedit_h', type=float, default=1e-4)
    parser.add_argument('--postedit_m', type=float, default=.01)
    parser.add_argument('--postedit_difference_step', type=float, default=1e-3)
    parser.add_argument('--collision_margin', type=float, default=.002,
                        help='角色坐标米制下的安全间隙；随模型常量导出。')
    parser.add_argument('--collision_tracking_tolerance', type=float, default=.02)
    parser.set_defaults(ts_respace='10')
    return parser


def load_models(args):
    """复用训练参数和 EMA 加载，导出统一在 CPU FP32 下进行。"""
    predictor = None if getattr(args, 'sampler_only', False) else load_realtime_pose_predictor(
        args.predictor_model_path, torch.device('cpu'))
    dit, diffusion = create_model_and_diffusion(args)
    dit, _ = load_checkpoint_model(dit, args.model_path, torch.device('cpu'), use_ema=True)
    normalizer = RealtimePoseNormalizer(args.normalizer_dir)
    return predictor, dit, diffusion, normalizer


def main(argv=None):
    # 静态导出包含很多小型几何算子，避免 CPU 线程池调度盖过实际计算。
    torch.set_num_threads(1)
    args = parse_and_load_from_model(build_arg_parser(), argv,
        ignore_keys={'normalizer_dir', 'output_dir', 'body_fbx_rest_json',
                     'sampler_only', 'sampler_filename'} | COLLISION_KEYS)
    if args.diffusion_steps != 50 or str(args.ts_respace) not in {'5', '10'}:
        raise ValueError('本次部署使用 50 个基础时间步、ts_respace=5 或 10。')
    if args.collision_postprocess and (str(args.ts_respace) != '5' or args.sampler_filename is not None):
        raise ValueError('碰撞命令链要求 --ts_respace 5，且不设置 --sampler_filename。')
    predictor, dit, diffusion, normalizer = load_models(args)
    postprocess, manifest = build_collision_postprocess(args, normalizer)
    sampler = PoseSampler(dit, diffusion, normalizer, postprocess).eval().requires_grad_(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = example_inputs(args.collision_postprocess)
    filename = None if args.collision_postprocess else sampler_filename(args, diffusion.num_timesteps)
    input_names = SAMPLER_INPUTS + (COLLISION_INPUTS if args.collision_postprocess else [])
    previous_fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        # 防止 eval fastpath 导出为 ONNX 不支持的 native attention。
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.inference_mode():
            if not args.sampler_only:
                torch.onnx.export(predictor, (inputs[0], torch.zeros(1, 11, 54)),
                    str(args.output_dir / 'predictor.onnx'), opset_version=15, dynamo=False,
                    input_names=['motion_context', 'core_tracker_context'],
                    output_names=['predictor_pose_horizon'])
            if args.collision_postprocess:
                export_collision_chain(sampler, manifest, args.output_dir)
            else:
                torch.onnx.export(sampler, inputs, str(args.output_dir / filename),
                    opset_version=15, dynamo=False, input_names=input_names,
                    output_names=['deployed_pose'])
    finally:
        torch.backends.mha.set_fastpath_enabled(previous_fastpath)
    print(f'Unity Sampler exported: {(args.output_dir if args.collision_postprocess else args.output_dir / filename).resolve()}; '
          f'timesteps={list(reversed(diffusion.timestep_map))}')
    # 对比模型与正式十步模型同目录存放时，不能覆盖正式配置和 Predictor。
    if args.sampler_only:
        return
    config = {'fps': 30, 'diffusion_steps': 50, 'sampling_steps': diffusion.num_timesteps, 'eta': 0,
              'normalizer_eps': normalizer.eps,
              'normalizer': {key: getattr(normalizer, key).tolist() for key in STATS},
              'ik': {key: getattr(args, key) for key in IK_KEYS}}
    (args.output_dir / 'runtime_config.json').write_text(
        json.dumps(config, indent=2) + '\n', encoding='utf-8')
    shutil.copyfile(args.body_fbx_rest_json, args.output_dir / 'body_fbx_rest.json')
    print(f'Unity ONNX exported: {args.output_dir.resolve()}')


def sampler_filename(args, sampling_steps):
    if args.collision_postprocess:
        raise ValueError('碰撞命令链使用三个模型和 collision_compute.json，没有单个 sampler 文件名。')
    base = 'pose_sampler_5step' if sampling_steps == 5 else 'pose_sampler'
    return args.sampler_filename or base + '.onnx'


def build_collision_postprocess(args, normalizer):
    if not args.collision_postprocess:
        return None, None
    if args.collision_profile is None:
        raise ValueError('--collision_postprocess 必须配合 --collision_profile。')
    rest = json.loads(args.body_fbx_rest_json.read_text(encoding='utf-8-sig'))
    profile = load_collision_profile(args.collision_profile, rest)
    settings = CollisionSettings(steps=args.postedit_steps, h=args.postedit_h, m=args.postedit_m,
        difference_step=args.postedit_difference_step, margin=args.collision_margin,
        tracking_tolerance=args.collision_tracking_tolerance)
    module = ResidualCollisionPostprocess(rest, profile, normalizer.pose_mean, normalizer.pose_scale, settings)
    return module, collision_manifest(settings, profile, rest, '')


if __name__ == "__main__":
    main()
