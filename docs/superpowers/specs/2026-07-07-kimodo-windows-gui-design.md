# Kimodo Windows GUI Dataset Console Design

## 背景

目标是为 NVIDIA Kimodo 增加一个简易 Windows GUI，优先服务 DiffusionPoser 数据集生成，同时保留后续导出 Unity 资产或动画预览的扩展空间。Kimodo 本体运行在 WSL2/Docker 中，Windows 侧只负责 GUI、任务编排、日志展示、输出目录挂载和 DiffusionPoser 数据桥接。

新项目独立放在 `D:\Projects\SchoolWorkProjects\kimodo`，使用 Git 管理。当前 DiffusionPoser 仓库不直接承载 GUI，只通过命令行脚本被调用。

## 范围

第一版交付一个 PySide6 桌面客户端和配套脚本，完成以下链路：

1. 检测 Windows 本机依赖：Docker、WSL2、NVIDIA GPU、Hugging Face token、Kimodo Docker image、DiffusionPoser 路径、`diffusionposer5070` conda 环境。
2. 录入或导入 prompt 批次，设置每条 prompt 的生成数量、时长、seed 和输出命名。
3. 调用 Docker/WSL 中的 Kimodo 生成 SMPL-X/AMASS-like 动作文件。
4. 将 Kimodo 输出转换为 DiffusionPoser 当前 AMASS converter 能读取的 pseudo-AMASS `.npz`。
5. 调用 DiffusionPoser 现有数据链路生成 `realtime_pose_stationary5_v1` source、task 和 normalizer。
6. 保留 Unity 导出入口，第一版显示禁用按钮并说明依赖后续导出模块；实际导出在后续版本接入。

不在第一版范围内：

- 不做 Kimodo 模型训练或微调。
- 不实现复杂动画编辑器、时间轴编辑、Blender 插件或 Unity 插件。
- 不把 Kimodo/PyTorch/CUDA 大依赖安装到 Windows conda GUI 环境。
- 不修改 DiffusionPoser 的 schema 契约。

## 环境与安装布局

项目目录：

```text
D:\Projects\SchoolWorkProjects\kimodo\
  app\
  kimodo_runner\
  diffusionposer_bridge\
  configs\
  scripts\
  docker\
  vendor\
    nv-tlabs-kimodo\
  artifacts\
    kimodo_raw\
    pseudo_amass\
    diffusionposer_sources\
  runs\
  .cache\
    huggingface\
  README.md
```

Windows GUI 使用独立 conda 环境：

```powershell
conda create --prefix D:\Anaconda\envs\kimodo_gui python=3.11
conda activate D:\Anaconda\envs\kimodo_gui
pip install PySide6 pydantic rich watchdog
```

该环境只负责 GUI 和本地桥接逻辑。Kimodo 运行时依赖由 Docker image 管理。DiffusionPoser 数据转换继续使用现有 `diffusionposer5070` 环境，并且执行命令时使用 `conda run --no-capture-output -n diffusionposer5070 ...`，以便面板实时显示日志。

安装位置遵循 D 盘集中、C 盘最小化原则：

- GUI 项目 Git 仓库固定在 `D:\Projects\SchoolWorkProjects\kimodo`。
- GUI conda 环境固定在 `D:\Anaconda\envs\kimodo_gui`，不使用默认用户目录环境。
- Kimodo 官方源码 clone 到 `D:\Projects\SchoolWorkProjects\kimodo\vendor\nv-tlabs-kimodo`，作为外部依赖管理。
- Hugging Face token 和模型缓存放在 `D:\Projects\SchoolWorkProjects\kimodo\.cache\huggingface`，不提交 Git。
- 生成产物放在 `D:\Projects\SchoolWorkProjects\kimodo\artifacts`，不提交 Git。
- 批次日志、manifest 和运行记录放在 `D:\Projects\SchoolWorkProjects\kimodo\runs`，不提交 Git。
- Docker Desktop 程序本体可使用默认安装位置；Docker image/container 数据优先配置到 `D:\DockerDesktop`，避免 Kimodo 镜像占用 C 盘。
- 现有 DiffusionPoser 仓库和 `diffusionposer5070` 环境不迁移，GUI 仅通过配置路径调用它们，避免破坏当前训练和评估环境。

`artifacts/`、`runs/`、`.cache/` 和本机私有配置必须加入 `.gitignore`。只有 GUI 源码、配置模板、Docker 配置模板和文档进入 Git。

## 架构

GUI 是薄客户端，核心逻辑拆成三个服务模块：

- `kimodo_runner`：封装 Docker Compose 命令、环境检测、容器日志流、Kimodo 输出目录管理。
- `diffusionposer_bridge`：负责 Kimodo 输出索引、SMPL-X/AMASS-like 字段检查、pseudo-AMASS 转换、DiffusionPoser pipeline 命令组装。
- `app`：PySide6 界面、任务状态机、配置读写、日志面板和用户操作入口。

所有长任务都通过子进程运行，GUI 不直接 import Kimodo 或 DiffusionPoser 模型代码。任务输出统一写入 `artifacts/` 下，运行日志和 manifest 写入 `runs/`，便于复现和排查。

## 数据流

```text
Prompt batch
  -> Kimodo Docker runner
  -> artifacts/kimodo_raw/<batch_id>/
  -> pseudo-AMASS converter
  -> artifacts/pseudo_amass/<batch_id>/
  -> DiffusionPoser amass_to_realtime_pose
  -> realtime_pose_stationary5_v1 source
  -> generate_realtime_pose_tasks
  -> compute_realtime_pose_normalizer
```

Kimodo 到 pseudo-AMASS 的最小字段映射：

- `poses = concat(root_orient, pose_body)`，形状至少为 `[T, 66]`。
- `trans` 保持为 `[T, 3]`。
- `mocap_framerate` 从 Kimodo 的 `mocap_frame_rate` 或等价字段映射。
- `betas`、`gender` 缺失时使用稳定默认值，并在 metadata 中标记来源。

转换后仍由 DiffusionPoser 现有 `amass_to_realtime_pose.py` 负责生成 `body_fbx_local_delta_6d`、`root_y0`、tracker、stationary labels 和 schema metadata。

## GUI 设计

主窗口分为五个区域：

1. 环境状态：显示 Docker、WSL2、GPU、HF token、Kimodo image、DiffusionPoser repo、conda 环境状态。
2. Prompt 批处理：多行文本输入、导入 `.txt/.csv/.jsonl`、seed、时长、每条生成数量、batch 名称。
3. 生成任务：启动、停止、重试、清理失败输出，显示 Kimodo 实时日志。
4. 数据转换：一键生成 pseudo-AMASS、source、task、normalizer，显示每阶段状态。
5. 后续出口：Unity 导出、FBX/BVH 预览和输出目录打开入口。第一版中 Unity 导出可先作为禁用功能展示。

界面不解释 DiffusionPoser schema 细节，只显示必要状态、路径和错误。详细日志写文件，面板展示关键行和当前阶段。

## 配置

`configs/app.example.json` 提供可复制的本机配置模板：

```json
{
  "project_root": "D:/Projects/SchoolWorkProjects/kimodo",
  "diffusionposer_root": "D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/DiffusionPoser",
  "anaconda_root": "D:/Anaconda",
  "gui_conda_prefix": "D:/Anaconda/envs/kimodo_gui",
  "diffusionposer_conda_env": "diffusionposer5070",
  "schema": "realtime_pose_stationary5_v1",
  "source_set_name": "kimodo_generated",
  "docker_compose_file": "docker/compose.yaml",
  "docker_data_root": "D:/DockerDesktop",
  "hf_token_env": "HF_TOKEN",
  "hf_cache_dir": "D:/Projects/SchoolWorkProjects/kimodo/.cache/huggingface",
  "artifact_root": "D:/Projects/SchoolWorkProjects/kimodo/artifacts",
  "run_root": "D:/Projects/SchoolWorkProjects/kimodo/runs"
}
```

本机私有配置写入 `configs/app.local.json`，不提交 Git。

## 安装流程

安装脚本分为 Windows 侧和 Docker 侧：

1. `scripts/setup_env.ps1`：创建 `D:\Anaconda\envs\kimodo_gui`，安装 GUI 依赖，生成 `configs/app.local.json` 初始模板，检查 DiffusionPoser 路径和 `diffusionposer5070` 环境。
2. `scripts/setup_docker.ps1`：检查 WSL2、Docker Desktop、NVIDIA GPU 可见性和 `HF_TOKEN`，clone 或更新 `vendor\nv-tlabs-kimodo`，准备 Docker Compose 所需目录挂载。
3. `docker/compose.yaml`：定义 Kimodo 运行容器，挂载 Hugging Face cache、Kimodo raw output、prompt batch 文件和 logs。
4. GUI 的环境检测页复用同一批检查函数，避免脚本与界面出现两套判断。

安装脚本不自动迁移 Docker Desktop 数据目录，因为该操作依赖 Docker Desktop 当前版本和用户设置。设计上先检测当前 Docker 数据位置；如果不在 D 盘，GUI 给出提示，由用户在 Docker Desktop 设置中调整。

## 错误处理

环境检测阶段给出明确缺失项和建议动作，例如 Docker 未运行、GPU 不可见、HF token 缺失、DiffusionPoser 路径不存在、conda 环境不存在。

长任务阶段按阶段记录状态：

- `pending`
- `running`
- `succeeded`
- `failed`
- `cancelled`

失败时保留输出目录和完整日志，不自动删除。重试时新建 batch run id，避免覆盖已有产物。

## 测试策略

第一版测试重点是任务编排和数据契约，不测试 Kimodo 模型质量：

- 配置解析单元测试。
- Docker/DiffusionPoser 命令组装测试。
- Kimodo mock 输出到 pseudo-AMASS 的转换测试。
- manifest 写入和失败状态测试。
- 至少一个 smoke test 使用极小 mock `.npz` 验证 pseudo-AMASS 字段满足 DiffusionPoser converter 的输入要求。

真实 Kimodo 运行作为手动验收项：生成少量动作，确认能完成 pseudo-AMASS、source、task、normalizer 链路，并人工检查可视化结果。

安装验收项：

- `D:\Projects\SchoolWorkProjects\kimodo` 是 Git 仓库。
- `D:\Anaconda\envs\kimodo_gui` 可以运行 PySide6 GUI 启动脚本。
- Docker 可以在 WSL2 backend 下看到 NVIDIA GPU。
- `HF_TOKEN` 可被 Docker runner 读取，但不会写入 Git。
- Kimodo 官方源码存在于 `vendor\nv-tlabs-kimodo`。
- mock Kimodo 输出可以完成 pseudo-AMASS 转换和 DiffusionPoser 输入字段校验。

## 后续扩展

后续版本可以增加：

- Unity runtime asset 导出按钮，调用 DiffusionPoser 现有 export 脚本。
- FBX/BVH 导出或 Blender 预览入口。
- prompt 模板库和批次历史。
- 任务队列和暂停/恢复。
- 生成质量标注，把人工筛选结果写回 manifest，用于构建训练子集。

## 审批状态

用户已确认采用 PySide6 Windows GUI 方案，并要求：

- 数据集生成优先。
- Kimodo 可以运行在 WSL2/Docker 中。
- GUI 项目放在 `D:\Projects\SchoolWorkProjects\kimodo`。
- 使用 Git 管理新项目。
- 如需 Windows 侧环境，使用单独 conda 环境，并由 `D:\Anaconda` 管理。
- 安装位置采用 D 盘集中策略：GUI 环境、源码、缓存、产物和运行日志都放在 `D:\Projects\SchoolWorkProjects\kimodo` 或 `D:\Anaconda`，Docker 大数据优先放到 `D:\DockerDesktop`。
