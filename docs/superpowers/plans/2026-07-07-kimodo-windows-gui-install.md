# Kimodo Windows GUI Install Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create and configure `D:\Projects\SchoolWorkProjects\kimodo` as a Git-managed PySide6 Windows dataset console for Kimodo-to-DiffusionPoser generation, with D-drive conda/cache/artifact layout and Docker/Kimodo setup checks.

**Architecture:** The new project is a thin Windows controller. PySide6 and local Python modules handle configuration, environment probes, command construction, pseudo-AMASS conversion, logs, and a minimal GUI; Kimodo itself remains isolated behind Docker/WSL. DiffusionPoser is treated as an external repository called through `conda run --no-capture-output -n diffusionposer5070`.

**Tech Stack:** Python 3.11, PySide6, pydantic, numpy, pytest, PowerShell, Docker Compose, Git.

## Global Constraints

- New project root is exactly `D:\Projects\SchoolWorkProjects\kimodo`.
- GUI conda environment is exactly `D:\Anaconda\envs\kimodo_gui`.
- Kimodo official source is cloned under `D:\Projects\SchoolWorkProjects\kimodo\vendor\nv-tlabs-kimodo`.
- Hugging Face cache is `D:\Projects\SchoolWorkProjects\kimodo\.cache\huggingface`.
- Generated artifacts are under `D:\Projects\SchoolWorkProjects\kimodo\artifacts`.
- Run logs and manifests are under `D:\Projects\SchoolWorkProjects\kimodo\runs`.
- Do not migrate the existing DiffusionPoser repository or `diffusionposer5070` environment.
- Do not install Kimodo/PyTorch/CUDA into the Windows GUI conda environment.
- All commands that run DiffusionPoser through conda must include `--no-capture-output`.
- Current observed host state: target project directory exists but is not a Git repository; `D:\Anaconda\envs\kimodo_gui` does not exist; Docker CLI is not available in the current PATH; `HF_TOKEN` is missing; GPU is `NVIDIA GeForce RTX 5070` with about 12GB VRAM.

---

## File Structure

- `D:\Projects\SchoolWorkProjects\kimodo\.gitignore`: excludes local config, caches, artifacts, runs, virtual files, and vendored Kimodo checkout.
- `D:\Projects\SchoolWorkProjects\kimodo\README.md`: documents setup, environment layout, and first-run commands.
- `D:\Projects\SchoolWorkProjects\kimodo\pyproject.toml`: project metadata, runtime dependencies, pytest config.
- `D:\Projects\SchoolWorkProjects\kimodo\configs\app.example.json`: committed template config.
- `D:\Projects\SchoolWorkProjects\kimodo\scripts\setup_env.ps1`: creates the D-drive GUI conda env and installs local project in editable mode.
- `D:\Projects\SchoolWorkProjects\kimodo\scripts\setup_docker.ps1`: checks WSL2, Docker, GPU, `HF_TOKEN`, and clones/updates Kimodo vendor source.
- `D:\Projects\SchoolWorkProjects\kimodo\scripts\run_gui.ps1`: launches the GUI through the D-drive conda env.
- `D:\Projects\SchoolWorkProjects\kimodo\docker\compose.yaml`: Kimodo container template with D-drive cache/artifact mounts.
- `D:\Projects\SchoolWorkProjects\kimodo\kimodo_gui\config.py`: typed app configuration and path defaults.
- `D:\Projects\SchoolWorkProjects\kimodo\kimodo_gui\environment.py`: environment probes for conda, Docker, WSL, GPU, paths, and tokens.
- `D:\Projects\SchoolWorkProjects\kimodo\kimodo_gui\commands.py`: Docker/Kimodo and DiffusionPoser command builders.
- `D:\Projects\SchoolWorkProjects\kimodo\kimodo_gui\pseudo_amass.py`: Kimodo SMPL-X/AMASS-like `.npz` to pseudo-AMASS conversion.
- `D:\Projects\SchoolWorkProjects\kimodo\kimodo_gui\manifest.py`: run manifest model and JSONL writer.
- `D:\Projects\SchoolWorkProjects\kimodo\kimodo_gui\process.py`: subprocess runner with line streaming.
- `D:\Projects\SchoolWorkProjects\kimodo\kimodo_gui\app.py`: minimal PySide6 GUI.
- `D:\Projects\SchoolWorkProjects\kimodo\kimodo_gui\__main__.py`: `python -m kimodo_gui` entrypoint.
- `D:\Projects\SchoolWorkProjects\kimodo\tests\`: unit tests for config, environment probes, command builders, pseudo-AMASS conversion, manifest writing, and GUI import.

## Task 1: Bootstrap New Repository

**Files:**
- Create: `D:\Projects\SchoolWorkProjects\kimodo\.gitignore`
- Create: `D:\Projects\SchoolWorkProjects\kimodo\README.md`
- Create: `D:\Projects\SchoolWorkProjects\kimodo\pyproject.toml`
- Create: `D:\Projects\SchoolWorkProjects\kimodo\configs\app.example.json`

**Interfaces:**
- Produces: Importable package name `kimodo_gui`.
- Produces: Config template consumed by `kimodo_gui.config.AppConfig`.

- [ ] **Step 1: Initialize Git if needed**

Run:

```powershell
if (-not (Test-Path "D:\Projects\SchoolWorkProjects\kimodo\.git")) {
  git -C "D:\Projects\SchoolWorkProjects\kimodo" init
}
```

Expected: Git repository exists at `D:\Projects\SchoolWorkProjects\kimodo\.git`.

- [ ] **Step 2: Create repository metadata files**

Create `.gitignore` with:

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
configs/app.local.json
.cache/
artifacts/
runs/
vendor/
*.log
```

Create `pyproject.toml` with:

```toml
[project]
name = "kimodo-windows-gui"
version = "0.1.0"
description = "Windows GUI and dataset bridge for Kimodo-generated DiffusionPoser data."
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26",
  "pydantic>=2.7",
  "PySide6>=6.7",
  "rich>=13.7",
  "watchdog>=4.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.2",
]

[project.scripts]
kimodo-gui = "kimodo_gui.app:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

Create `configs/app.example.json` with the exact paths from the approved spec.

- [ ] **Step 3: Commit bootstrap**

Run:

```powershell
git -C "D:\Projects\SchoolWorkProjects\kimodo" add .gitignore README.md pyproject.toml configs/app.example.json
git -C "D:\Projects\SchoolWorkProjects\kimodo" commit -m "chore: bootstrap kimodo gui project"
```

Expected: Commit succeeds.

## Task 2: Add Configuration and Environment Probes

**Files:**
- Create: `D:\Projects\SchoolWorkProjects\kimodo\kimodo_gui\__init__.py`
- Create: `D:\Projects\SchoolWorkProjects\kimodo\kimodo_gui\config.py`
- Create: `D:\Projects\SchoolWorkProjects\kimodo\kimodo_gui\environment.py`
- Create: `D:\Projects\SchoolWorkProjects\kimodo\tests\test_config.py`
- Create: `D:\Projects\SchoolWorkProjects\kimodo\tests\test_environment.py`

**Interfaces:**
- Produces: `AppConfig.load(path: Path | None = None) -> AppConfig`.
- Produces: `default_config() -> AppConfig`.
- Produces: `probe_environment(config: AppConfig) -> list[ProbeResult]`.
- Produces: `ProbeResult(name: str, ok: bool, detail: str)`.

- [ ] **Step 1: Write failing config tests**

Create tests asserting:

```python
from pathlib import Path
from kimodo_gui.config import AppConfig, default_config

def test_default_config_uses_d_drive_layout():
    cfg = default_config()
    assert cfg.project_root == Path("D:/Projects/SchoolWorkProjects/kimodo")
    assert cfg.gui_conda_prefix == Path("D:/Anaconda/envs/kimodo_gui")
    assert cfg.artifact_root == cfg.project_root / "artifacts"
    assert cfg.run_root == cfg.project_root / "runs"

def test_config_load_from_json(tmp_path):
    path = tmp_path / "app.json"
    path.write_text('{"project_root": "D:/Projects/SchoolWorkProjects/kimodo"}', encoding="utf-8")
    cfg = AppConfig.load(path)
    assert cfg.project_root == Path("D:/Projects/SchoolWorkProjects/kimodo")
```

- [ ] **Step 2: Implement config**

Implement `AppConfig` as a pydantic model with `Path` fields and defaults matching `configs/app.example.json`. `load()` reads JSON if the file exists and overlays defaults for missing keys.

- [ ] **Step 3: Write and implement environment probes**

Create `ProbeResult` dataclass and probes for:

```python
("project_root", config.project_root.exists())
("gui_conda_prefix", config.gui_conda_prefix.exists())
("diffusionposer_root", config.diffusionposer_root.exists())
("hf_token", bool(os.environ.get(config.hf_token_env)))
("docker_cli", shutil.which("docker") is not None)
("wsl_cli", shutil.which("wsl") is not None)
("nvidia_smi", shutil.which("nvidia-smi") is not None)
```

Tests monkeypatch `PATH` and `HF_TOKEN` to verify missing and present states without requiring Docker to be installed.

- [ ] **Step 4: Run tests and commit**

Run:

```powershell
conda run --prefix D:\Anaconda\envs\kimodo_gui --no-capture-output pytest tests/test_config.py tests/test_environment.py -q
```

Expected: PASS after the environment is created in Task 7.

Commit:

```powershell
git -C "D:\Projects\SchoolWorkProjects\kimodo" add kimodo_gui tests
git -C "D:\Projects\SchoolWorkProjects\kimodo" commit -m "feat: add config and environment probes"
```

## Task 3: Add Command Builders

**Files:**
- Create: `D:\Projects\SchoolWorkProjects\kimodo\kimodo_gui\commands.py`
- Create: `D:\Projects\SchoolWorkProjects\kimodo\tests\test_commands.py`

**Interfaces:**
- Produces: `build_diffusionposer_convert_command(config: AppConfig, pseudo_amass_dir: Path, source_dir: Path) -> list[str]`.
- Produces: `build_diffusionposer_task_command(config: AppConfig, source_dir: Path, task_dir: Path) -> list[str]`.
- Produces: `build_diffusionposer_normalizer_command(config: AppConfig, task_dir: Path, normalizer_dir: Path) -> list[str]`.
- Produces: `build_docker_compose_command(config: AppConfig, *args: str) -> list[str]`.

- [ ] **Step 1: Write failing command tests**

Tests assert every DiffusionPoser command starts with:

```python
["conda", "run", "--no-capture-output", "-n", "diffusionposer5070", "python", "-m"]
```

Tests assert convert command includes:

```python
"data_converter.amass_to_realtime_pose"
"--schema", "realtime_pose_stationary5_v1"
"--amass_dir", "<pseudo_amass_dir>"
"--output_dir", "<source_dir>"
"--source_set_name", "kimodo_generated"
```

- [ ] **Step 2: Implement command builders**

Use only `list[str]` commands and explicit string conversion. Do not shell-join commands in Python.

- [ ] **Step 3: Run tests and commit**

Run:

```powershell
conda run --prefix D:\Anaconda\envs\kimodo_gui --no-capture-output pytest tests/test_commands.py -q
```

Commit command files after PASS.

## Task 4: Add Pseudo-AMASS Conversion

**Files:**
- Create: `D:\Projects\SchoolWorkProjects\kimodo\kimodo_gui\pseudo_amass.py`
- Create: `D:\Projects\SchoolWorkProjects\kimodo\tests\test_pseudo_amass.py`

**Interfaces:**
- Produces: `convert_kimodo_npz_to_pseudo_amass(input_path: Path, output_path: Path) -> dict[str, object]`.
- Produces: `convert_batch(input_dir: Path, output_dir: Path) -> list[dict[str, object]]`.

- [ ] **Step 1: Write failing converter tests**

Create a mock Kimodo file with:

```python
np.savez(
    input_path,
    root_orient=np.zeros((4, 3), dtype=np.float32),
    pose_body=np.zeros((4, 63), dtype=np.float32),
    trans=np.zeros((4, 3), dtype=np.float32),
    mocap_frame_rate=np.asarray(30.0, dtype=np.float32),
)
```

Assert output `.npz` contains:

```python
poses.shape == (4, 66)
trans.shape == (4, 3)
float(mocap_framerate) == 30.0
betas.shape == (10,)
str(gender) == "neutral"
```

- [ ] **Step 2: Implement converter**

Implement strict validation:

- `root_orient` must be `[T, 3]`.
- `pose_body` must be `[T, 63]` or wider; if wider, keep first 63 body dimensions.
- `trans` must be `[T, 3]`.
- Frame counts must match.
- `mocap_framerate` uses `mocap_frame_rate`, `mocap_framerate`, or default `60.0`.

- [ ] **Step 3: Run tests and commit**

Run:

```powershell
conda run --prefix D:\Anaconda\envs\kimodo_gui --no-capture-output pytest tests/test_pseudo_amass.py -q
```

Commit converter after PASS.

## Task 5: Add Manifest and Process Runner

**Files:**
- Create: `D:\Projects\SchoolWorkProjects\kimodo\kimodo_gui\manifest.py`
- Create: `D:\Projects\SchoolWorkProjects\kimodo\kimodo_gui\process.py`
- Create: `D:\Projects\SchoolWorkProjects\kimodo\tests\test_manifest.py`
- Create: `D:\Projects\SchoolWorkProjects\kimodo\tests\test_process.py`

**Interfaces:**
- Produces: `RunRecord` pydantic model with `run_id`, `stage`, `status`, `command`, `started_at`, `finished_at`, `returncode`, `log_path`.
- Produces: `append_run_record(path: Path, record: RunRecord) -> None`.
- Produces: `run_streaming(command: list[str], cwd: Path | None, log_path: Path) -> int`.

- [ ] **Step 1: Write failing tests**

Test manifest writes one JSON object per line. Test `run_streaming(["python", "-c", "print('ok')"], None, log_path)` returns `0` and writes `ok` to the log.

- [ ] **Step 2: Implement manifest and process runner**

Use `subprocess.Popen(..., stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)` and write streamed lines to UTF-8 log files.

- [ ] **Step 3: Run tests and commit**

Run:

```powershell
conda run --prefix D:\Anaconda\envs\kimodo_gui --no-capture-output pytest tests/test_manifest.py tests/test_process.py -q
```

Commit after PASS.

## Task 6: Add Minimal PySide6 GUI

**Files:**
- Create: `D:\Projects\SchoolWorkProjects\kimodo\kimodo_gui\app.py`
- Create: `D:\Projects\SchoolWorkProjects\kimodo\kimodo_gui\__main__.py`
- Create: `D:\Projects\SchoolWorkProjects\kimodo\tests\test_app_import.py`
- Create: `D:\Projects\SchoolWorkProjects\kimodo\scripts\run_gui.ps1`

**Interfaces:**
- Produces: `main() -> int`.
- Produces: `python -m kimodo_gui` entrypoint.

- [ ] **Step 1: Write import smoke test**

Test:

```python
def test_app_main_imports():
    from kimodo_gui.app import main
    assert callable(main)
```

- [ ] **Step 2: Implement GUI**

Create a `QMainWindow` with tabs:

- Environment
- Prompts
- Generate
- Convert
- Logs

The Environment tab calls `probe_environment(default_config())` and renders status rows. Buttons for generate and convert can call command-builder functions and append the planned command text to the log panel.

- [ ] **Step 3: Run test and launch smoke**

Run:

```powershell
conda run --prefix D:\Anaconda\envs\kimodo_gui --no-capture-output pytest tests/test_app_import.py -q
conda run --prefix D:\Anaconda\envs\kimodo_gui --no-capture-output python -m kimodo_gui --smoke
```

Expected: import test passes; smoke command initializes and exits without opening a persistent window.

- [ ] **Step 4: Commit GUI**

Commit after PASS.

## Task 7: Add Setup Scripts and Docker Template

**Files:**
- Create: `D:\Projects\SchoolWorkProjects\kimodo\scripts\setup_env.ps1`
- Create: `D:\Projects\SchoolWorkProjects\kimodo\scripts\setup_docker.ps1`
- Create: `D:\Projects\SchoolWorkProjects\kimodo\docker\compose.yaml`
- Create: `D:\Projects\SchoolWorkProjects\kimodo\tests\test_project_files.py`

**Interfaces:**
- Produces: `setup_env.ps1` creates `D:\Anaconda\envs\kimodo_gui` and installs `.[dev]`.
- Produces: `setup_docker.ps1` checks Docker/WSL/GPU/HF token and clones Kimodo into `vendor\nv-tlabs-kimodo`.

- [ ] **Step 1: Write project file tests**

Test that setup scripts and compose template exist and contain required path strings:

```python
assert "D:\\Anaconda\\envs\\kimodo_gui" in setup_env_text
assert "vendor\\nv-tlabs-kimodo" in setup_docker_text
assert ".cache/huggingface" in compose_text
```

- [ ] **Step 2: Implement `setup_env.ps1`**

Script behavior:

```powershell
$ErrorActionPreference = "Stop"
$Root = "D:\Projects\SchoolWorkProjects\kimodo"
$EnvPrefix = "D:\Anaconda\envs\kimodo_gui"
if (-not (Test-Path $EnvPrefix)) {
  conda create --yes --prefix $EnvPrefix python=3.11
}
conda run --prefix $EnvPrefix --no-capture-output python -m pip install --upgrade pip
conda run --prefix $EnvPrefix --no-capture-output python -m pip install -e "$Root[dev]"
```

- [ ] **Step 3: Implement `setup_docker.ps1`**

Script behavior:

```powershell
$ErrorActionPreference = "Stop"
$Root = "D:\Projects\SchoolWorkProjects\kimodo"
$Vendor = Join-Path $Root "vendor\nv-tlabs-kimodo"
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw "Docker CLI not found." }
if (-not (Get-Command wsl -ErrorAction SilentlyContinue)) { throw "WSL CLI not found." }
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { throw "nvidia-smi not found." }
if (-not $env:HF_TOKEN) { throw "HF_TOKEN is not set." }
if (-not (Test-Path $Vendor)) {
  git clone https://github.com/nv-tlabs/kimodo.git $Vendor
} else {
  git -C $Vendor pull --ff-only
}
```

- [ ] **Step 4: Implement Docker compose template**

Compose service uses NVIDIA GPU reservation, mounts:

- `../.cache/huggingface:/root/.cache/huggingface`
- `../artifacts/kimodo_raw:/workspace/output`
- `../vendor/nv-tlabs-kimodo:/workspace/kimodo`

- [ ] **Step 5: Run tests and commit**

Run:

```powershell
conda run --prefix D:\Anaconda\envs\kimodo_gui --no-capture-output pytest tests/test_project_files.py -q
```

Commit after PASS.

## Task 8: Execute Local Setup and Verification

**Files:**
- Modify only generated local files under `D:\Projects\SchoolWorkProjects\kimodo\configs`, `.cache`, `artifacts`, `runs`, and `vendor` as needed; these are gitignored.

**Interfaces:**
- Consumes: setup scripts and Python package from previous tasks.
- Produces: working `kimodo_gui` environment and local verification report.

- [ ] **Step 1: Create GUI conda environment**

Run:

```powershell
& "D:\Projects\SchoolWorkProjects\kimodo\scripts\setup_env.ps1"
```

Expected: `D:\Anaconda\envs\kimodo_gui` exists and `python -m kimodo_gui --smoke` runs.

- [ ] **Step 2: Run full unit test suite**

Run:

```powershell
conda run --prefix D:\Anaconda\envs\kimodo_gui --no-capture-output pytest -q
```

Expected: PASS.

- [ ] **Step 3: Run environment probe**

Run:

```powershell
conda run --prefix D:\Anaconda\envs\kimodo_gui --no-capture-output python -m kimodo_gui --check-env
```

Expected with current host state: Docker and `HF_TOKEN` are reported missing; GPU is present.

- [ ] **Step 4: Run Docker setup if host prerequisites are available**

Run:

```powershell
& "D:\Projects\SchoolWorkProjects\kimodo\scripts\setup_docker.ps1"
```

Expected on current host before installing Docker/token: clear failure message for missing Docker CLI or `HF_TOKEN`. After Docker Desktop and `HF_TOKEN` are configured: Kimodo repository exists under `vendor\nv-tlabs-kimodo`.

- [ ] **Step 5: Commit final verified state**

Run:

```powershell
git -C "D:\Projects\SchoolWorkProjects\kimodo" status --short
git -C "D:\Projects\SchoolWorkProjects\kimodo" log --oneline -5
```

Expected: tracked source files are committed; gitignored local artifacts may exist but are not staged.

## Self-Review

- Spec coverage: the plan creates the new Git repository, D-drive conda environment, D-drive cache/artifact/run layout, setup scripts, Docker/Kimodo checks, pseudo-AMASS converter, DiffusionPoser command builders, and minimal PySide6 GUI.
- Known external blockers: Docker CLI and `HF_TOKEN` are currently missing. The plan handles them as environment probe failures and setup script failures with explicit messages, without hiding the missing prerequisites.
- Type consistency: `AppConfig`, `ProbeResult`, command-builder names, pseudo-AMASS converter names, and manifest/process interfaces are defined before later tasks consume them.
