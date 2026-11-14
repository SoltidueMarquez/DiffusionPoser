"""从仓库内配置独立运行 LM、ΠGDM 场景引导对比。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.scene_guidance_setup import REPOSITORY_ROOT, validate_inputs
from utils.parser_util import add_diffusion_options, add_model_options, parse_and_load_from_model


def repository_path(value):
    """配置和命令行中的相对路径均以仓库根目录为基准。"""
    path = Path(value)
    return str(path if path.is_absolute() else REPOSITORY_ROOT / path)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_options(parser)
    add_diffusion_options(parser)
    group = parser.add_argument_group("scene guidance experiment")
    group.add_argument("--config", type=repository_path, default="experiments/scene_guidance/snapshot.json")
    group.add_argument("--protocol", choices=("snapshot", "sequence"), default="snapshot")
    group.add_argument("--model_path", type=repository_path, default="artifacts/unity_demo/dit/ema000100000.pt")
    group.add_argument("--predictor_model_path", type=repository_path, default="artifacts/unity_demo/predictor/model_latest.pt")
    group.add_argument("--normalizer_dir", type=repository_path, default="artifacts/unity_demo/normalizer")
    group.add_argument("--body_fbx_rest_json", type=repository_path, default="experiments/scene_guidance/inputs/body_fbx_rest.json")
    group.add_argument("--collision_profile", type=repository_path, default="experiments/scene_guidance/inputs/collision_profile.json")
    group.add_argument("--replay_inputs", type=repository_path, default="experiments/scene_guidance/inputs/captured_replay_inputs.json")
    group.add_argument("--recording", type=repository_path, default="experiments/scene_guidance/inputs/chair_probe.json")
    group.add_argument("--output_dir", type=repository_path, default="output/scene_guidance_snapshot")
    group.add_argument("--device", default="cuda:0")
    group.add_argument("--methods", nargs="+", choices=("none", "postedit", "lm", "pigdm"), default=None)
    group.add_argument("--tune", action="store_true", help="快照实验先扫描参数，再仅用调参集筛选配置。")
    group.add_argument("--tune_frames", type=int, default=20)
    group.add_argument("--split_strategy", choices=("uniform", "chronological"), default="uniform")
    group.add_argument("--seeds", type=int, nargs="+", default=[10, 20, 30])
    group.add_argument("--warmup", type=int, default=20)
    group.add_argument("--repeats", type=int, default=5)
    group.add_argument("--rollout_warmup", type=int, default=30)
    group.add_argument("--max_frames", type=int, default=0,
                       help="0 表示完整实验；正数仅限制测试/计分帧，供短程验证。")
    group.add_argument("--measurement_scale", type=float, default=.0005)
    group.add_argument("--margin", type=float, default=.002)
    group.add_argument("--check_inputs", action="store_true", help="仅检查所需文件和随仓库发布的输入哈希，无需 GPU。")
    parser.set_defaults(ts_respace="5")
    return parser


def parse_args(argv=None):
    parser = build_arg_parser()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=repository_path, default=repository_path(parser.get_default("config")))
    config_args, _ = config_parser.parse_known_args(argv)
    config = json.loads(Path(config_args.config).read_text(encoding="utf-8-sig"))
    allowed = {action.dest for action in parser._actions} - {"help", "config", "check_inputs"}
    unknown = set(config) - allowed - {"guidance_configs"}
    if unknown:
        parser.error(f"未知配置字段：{sorted(unknown)}")
    parser.set_defaults(**{key: value for key, value in config.items() if key != "guidance_configs"})
    ignored = {action.dest for group in parser._action_groups if group.title == "scene guidance experiment"
               for action in group._group_actions}
    # JSON 中显式指定的模型参数与 CLI 一样优先，缺省结构才从 checkpoint 补齐。
    args = parse_and_load_from_model(parser, argv, ignore_keys=ignored | set(config))
    args.guidance_configs = config.get("guidance_configs", [])
    if args.repeats < 1 or min(args.warmup, args.rollout_warmup, args.max_frames) < 0:
        parser.error("重复次数须为正，预热和帧数限制须非负。")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("种子须非负且不重复。")
    if args.protocol == "sequence" and args.tune:
        parser.error("长序列仅使用固定配置；参数扫描请使用 snapshot 协议。")
    return args


def main(argv=None):
    args = parse_args(argv)
    sources = validate_inputs(args)
    if args.check_inputs:
        print(json.dumps({"passed": True, "sources": sources}, ensure_ascii=False, indent=2))
        return sources
    if args.protocol == "sequence":
        from eval.scene_guidance_sequence import run_sequence
        return run_sequence(args)
    from eval.scene_guidance_benchmark import run_experiment
    return run_experiment(args)


if __name__ == "__main__":
    main()
