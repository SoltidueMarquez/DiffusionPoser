---
name: diffusionposer-repro
description: Coding rules and project style guide for the DiffusionPoser Python reproduction project, including Chinese communication and Chinese code-comment requirements. Use when editing or creating code for diffusion or transformer motion reconstruction, sparse sensor preprocessing, training scripts, sampling scripts, evaluation utilities, or StableMotion-inspired modules in this workspace.
---

# DiffusionPoser Python Style

## Overview

This skill records coding conventions for the DiffusionPoser reproduction project. It is not a task plan. Use it as the local style and engineering contract when writing Python code for the project.

## Environment

- For this project, default to the Anaconda environment `diffusionposer5070` before running Python, pip, tests, training, sampling, export, or evaluation commands.
- On Windows PowerShell, prefer `conda activate diffusionposer5070` for an interactive shell. For one-off commands where activation state is uncertain, prefer `conda run -n diffusionposer5070 <command>`.

## Communication

- 使用中文回答与解释，除非用户明确要求使用其他语言。
- 解释代码、论文概念、实验配置和报错时，优先使用清晰的中文说明。
- 保留必要的英文术语、变量名、论文方法名和库 API 名称，不要为了中文化而改变代码语义。

## Project Boundaries

- Treat the DiffusionPoser paper and related PDFs as research requirements, not as code style.
- Treat StableMotion as a reference implementation for diffusion mechanics, module layout, and training/sampling structure.
- Keep reproduction-specific code explicit. Do not hide paper assumptions inside generic helper names.
- Prefer a readable research-code style over framework-heavy abstraction.

## Python Style

- Use clear module-level functions for scripts and factories; avoid large anonymous blocks under `if __name__ == "__main__":`.
- Keep public function and class names descriptive enough for research readers: `create_model_and_diffusion`, `SparseSensorDataset`, `DiffusionPoserDiT`.
- Use `snake_case` for functions, variables, files, and CLI arguments.
- Use `PascalCase` for classes.
- Use explicit keyword arguments when constructing models, datasets, and diffusion objects.
- Keep imports grouped as standard library, third-party, then local modules.
- Avoid wildcard imports.
- Avoid global mutable configuration except for constants.
- 为生成或修改的代码添加详细中文注释，帮助同学理解实现意图、张量含义、论文假设和关键数学步骤。
- 注释要解释“为什么这样做”和“这个变量代表什么”，不要只复述显而易见的 Python 语法。
- 代码标识符继续使用英文，注释和面向读者的说明优先使用中文。
- 当文件、类、函数或脚本较长时，使用 `# region ...` 和 `# endregion` 分区整理代码，例如参数解析、数据加载、模型构建、训练循环、采样逻辑、评估逻辑。
- `# region` 分区名使用简洁中文或中英混合短语，保持结构稳定，避免过度切碎代码。

## Tensor and ML Conventions

- Document tensor shapes at function boundaries when a tensor has more than two semantic axes.
- Prefer shape comments like `[B, C, T]`, `[B, J, 3]`, or `[B, T, D]` near transformations.
- Keep diffusion targets explicit: name variables `x_start`, `x_t`, `noise`, `pred_xstart`, or `eps` according to their meaning.
- Do not conflate sensor masks, sequence masks, and inpainting masks. Use distinct names such as `sensor_mask`, `valid_frame_mask`, and `inpaint_mask`.
- Keep time dimension ordering consistent within a module. If transposing between `[B, C, T]` and `[B, T, C]`, keep the transpose close to the layer that requires it.
- Prefer PyTorch tensor operations over NumPy inside model, loss, and training code.
- Do not detach, clone, cast, or move tensors across devices unless the reason is local and clear.
- Avoid printing from model or loss functions during normal training. Use logger or caller-controlled debug flags.

## Module Organization

- Put model architectures in `model/`.
- Put diffusion schedules, objectives, and samplers in `diffusion/`.
- Put dataset loading, preprocessing adapters, and collate functions in `data_loaders/`.
- Put train entrypoints and loops in `train/`.
- Put reconstruction and sampling entrypoints in `sample/`.
- Put metrics and paper-aligned evaluation code in `eval/`.
- Put reusable environment, seeding, parsing, checkpoint, and normalizer helpers in `utils/`.
- Keep CLI scripts thin; move reusable behavior into importable functions.

## CLI and Configuration

- Use `argparse` groups for base, dataset, model, diffusion, training, and sampling options.
- Prefer checkpoint-local `args.json` for reproducibility.
- Keep default paths relative to the project root.
- Use boolean flags for optional features such as EMA, guidance, and debug modes.
- Preserve stable checkpoint naming and output directory conventions once introduced.

## Validation Style

- Add smoke tests or small runnable checks for new model, dataset, and sampling paths when practical.
- Validate tensor shapes before long training runs.
- Keep deterministic seeds available through CLI options.
- Fail early with specific error messages for missing dataset paths, normalizer files, checkpoint files, and incompatible tensor shapes.

## Reference

See [project-layout.md](references/project-layout.md) for the workspace file map and StableMotion style anchors.
