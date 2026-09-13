"""导出固定 batch=1 的 Unity Predictor 和完整 residual sampler。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from export.unity_onnx_models import PoseSampler, SAMPLER_INPUTS, example_inputs
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
                        help='Sampler 文件名；五步默认 pose_sampler_5step.onnx，十步保持原名。')
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
    args = parse_and_load_from_model(build_arg_parser(), argv,
        ignore_keys={'normalizer_dir', 'output_dir', 'body_fbx_rest_json',
                     'sampler_only', 'sampler_filename'})
    if args.diffusion_steps != 50 or str(args.ts_respace) not in {'5', '10'}:
        raise ValueError('本次部署使用 50 个基础时间步、ts_respace=5 或 10。')
    predictor, dit, diffusion, normalizer = load_models(args)
    sampler = PoseSampler(dit, diffusion, normalizer).eval().requires_grad_(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = example_inputs()
    filename = args.sampler_filename or (
        'pose_sampler_5step.onnx' if diffusion.num_timesteps == 5 else 'pose_sampler.onnx')
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
            torch.onnx.export(sampler, inputs, str(args.output_dir / filename),
                opset_version=15, dynamo=False, input_names=SAMPLER_INPUTS,
                output_names=['deployed_pose'])
    finally:
        torch.backends.mha.set_fastpath_enabled(previous_fastpath)
    print(f'Unity Sampler exported: {(args.output_dir / filename).resolve()}; '
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


if __name__ == "__main__":
    main()
