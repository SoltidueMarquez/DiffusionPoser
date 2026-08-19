"""Unity/Sentis 导出占位入口。

当前 Python 主链路需要 Predictor 与单帧 DiT 两个模型，双模型运行时导出不在
本轮实现范围内。
"""

from __future__ import annotations

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predictor + 单帧 DiT 双模型 Unity/Sentis 导出（未实现）。"
    )
    parser.add_argument("--model_path", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    build_arg_parser().parse_args(argv)
    raise NotImplementedError("Predictor + 单帧 DiT 双模型的 Unity/Sentis 导出本轮未实现。")


if __name__ == "__main__":
    main()
