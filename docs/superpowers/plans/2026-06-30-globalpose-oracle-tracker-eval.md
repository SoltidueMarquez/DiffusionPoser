# GlobalPose Oracle Tracker Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 复现 GlobalPose 官方测试，并把 GlobalPose 测试集转换成 `realtime_pose_stationary5_v1` 的 GT-derived oracle tracker 评估集，用当前 DiffusionPoser 方法跑同一批动作数据。

**Architecture:** GlobalPose 官方仓库、权重、测试集和 SMPL 模型作为外部依赖放在 `dataset/external/globalpose/repo/`，不进入本仓库主代码路径。DiffusionPoser 只新增可维护的转换、指标和实验记录入口，生成物继续走 `dataset/generated/...`、`output/...` 的现有 artifact 规范。

**Tech Stack:** Windows, conda, Python 3.8 for GlobalPose, `globalpose38`, Python 3.11 for DiffusionPoser, `diffusionposer5070`, PyTorch, NumPy, SMPL, GlobalPose `articulate/carticulate`.

---

## Scope And Naming

本计划只覆盖 oracle tracker 版本：

```text
GlobalPose GT pose/tran
-> forward kinematics
-> tracker_pos_world / tracker_rot_world_6d
-> realtime_pose_stationary5_v1 source/task/longseq eval
-> DiffusionPoser rollout evaluation
```

实验命名固定为：

```text
globalpose_totalcapture_dipcalib_oracle_tracker
globalpose_totalcapture_officalib_oracle_tracker
globalpose_dipimu_oracle_tracker
```

这些结果不能作为和 GlobalPose 原论文 IMU benchmark 的严格公平比较。实验文档必须明确标注：DiffusionPoser 输入来自 GT-derived oracle tracker，而 GlobalPose 官方结果输入是 IMU。

## Target Directory Layout

```text
dataset/
  external/
    globalpose/
      repo/
        models/SMPL_male.pkl
        data/weights.pt
        data/test_datasets/dipimu.pt
        data/test_datasets/totalcapture_dipcalib.pt
        data/test_datasets/totalcapture_officalib.pt

  generated/
    sources/realtime_pose_stationary5_v1/globalpose_totalcapture_dipcalib_oracle_tracker/
    sources/realtime_pose_stationary5_v1/globalpose_totalcapture_officalib_oracle_tracker/
    sources/realtime_pose_stationary5_v1/globalpose_dipimu_oracle_tracker/
    tasks/realtime_pose_stationary5_v1/globalpose_totalcapture_dipcalib_oracle_tracker_tasks/
    tasks/realtime_pose_stationary5_v1/globalpose_totalcapture_officalib_oracle_tracker_tasks/
    tasks/realtime_pose_stationary5_v1/globalpose_dipimu_oracle_tracker_tasks/
    longseq_eval/realtime_pose_stationary5_v1/globalpose_totalcapture_dipcalib_oracle_tracker/
    longseq_eval/realtime_pose_stationary5_v1/globalpose_totalcapture_officalib_oracle_tracker/
    longseq_eval/realtime_pose_stationary5_v1/globalpose_dipimu_oracle_tracker/

output/
  benchmark_globalpose/
    official/
    ours_oracle_tracker/
```

## Files To Create Or Modify During Implementation

- Create: `configs/globalpose_eval.example.json`
  - Records external GlobalPose repo path and expected dataset filenames.
- Modify: `.gitignore`
  - Ignore `configs/globalpose_eval.local.json` if a local config file is introduced.
- Create: `data_converter/globalpose_to_realtime_pose.py`
  - Converts GlobalPose `.pt` datasets to `realtime_pose_stationary5_v1` source `.npz`.
- Create: `tests/smoke/data_pipeline/test_globalpose_oracle_tracker_source.py`
  - Tests source conversion shape, metadata, root-y0 invariants, stationary labels, and manifest output on tiny synthetic data.
- Create: `eval/globalpose_metrics.py`
  - Adds GlobalPose-style translation drift and optional pose metric helpers for DiffusionPoser output.
- Create: `tests/smoke/eval/test_globalpose_metrics.py`
  - Tests 1m-7m travelled-distance drift calculation on controlled trajectories.
- Create: `documents/experiments/globalpose_oracle_tracker_eval.md`
  - Records environment, commands, paths, checkpoint, outputs, results, and limitations.

---

### Task 1: Create GlobalPose Python 3.8 Environment

**Files:**
- No repository file changes.

- [ ] **Step 1: Create the conda environment**

Run:

```powershell
conda create -n globalpose38 python=3.8 -y
```

Expected: conda creates `globalpose38`.

- [ ] **Step 2: Install PyTorch CUDA 11.8**

Run:

```powershell
conda run --no-capture-output -n globalpose38 python -m pip install `
  torch==2.0.1+cu118 `
  torchvision==0.15.2+cu118 `
  --index-url https://download.pytorch.org/whl/cu118
```

Expected: pip installs torch 2.0.1 CUDA 11.8 wheels.

- [ ] **Step 3: Install GlobalPose runtime dependencies**

Run:

```powershell
conda run --no-capture-output -n globalpose38 python -m pip install `
  chumpy `
  open3d `
  pybullet `
  qpsolvers[osqp] `
  numpy-quaternion `
  vctoolkit==0.1.5.39 `
  matplotlib `
  tqdm `
  scipy `
  keyboard `
  pygame
```

Expected: pip installs all dependencies. If `chumpy` fails on deprecated NumPy aliases, record the exact error in `documents/experiments/globalpose_oracle_tracker_eval.md` and pin NumPy or patch the installed `chumpy` import lines according to GlobalPose README.

- [ ] **Step 4: Verify environment basics**

Run:

```powershell
conda run --no-capture-output -n globalpose38 python --version
conda run --no-capture-output -n globalpose38 python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
conda run --no-capture-output -n globalpose38 python -c "import qpsolvers; print(qpsolvers.available_solvers)"
```

Expected:

```text
Python 3.8.x
2.0.1+cu118 True
```

`osqp` should appear in the solver list.

### Task 2: Prepare Official GlobalPose Repository And Assets

**Files:**
- External only: `dataset/external/globalpose/repo/`
- Documentation later: `documents/experiments/globalpose_oracle_tracker_eval.md`

- [ ] **Step 1: Clone GlobalPose**

Run:

```powershell
git clone https://github.com/Xinyu-Yi/GlobalPose.git dataset/external/globalpose/repo
```

Expected: `dataset/external/globalpose/repo/test.py` exists.

- [ ] **Step 2: Place required assets**

Ensure these files exist:

```text
dataset/external/globalpose/repo/models/SMPL_male.pkl
dataset/external/globalpose/repo/data/weights.pt
dataset/external/globalpose/repo/data/test_datasets/dipimu.pt
dataset/external/globalpose/repo/data/test_datasets/totalcapture_dipcalib.pt
dataset/external/globalpose/repo/data/test_datasets/totalcapture_officalib.pt
```

Expected: all five files exist locally. Do not commit these files.

- [ ] **Step 3: Verify GlobalPose local imports**

Run from `dataset/external/globalpose/repo`:

```powershell
conda run --no-capture-output -n globalpose38 python -c "import articulate; import carticulate; print('ok')"
```

Expected:

```text
ok
```

### Task 3: Reproduce Official GlobalPose Evaluation

**Files:**
- External outputs: `dataset/external/globalpose/repo/data/temp/results/`
- Create later: `documents/experiments/globalpose_oracle_tracker_eval.md`
- Output copy target: `output/benchmark_globalpose/official/`

- [ ] **Step 1: Run official test script**

Run from `dataset/external/globalpose/repo`:

```powershell
conda run --no-capture-output -n globalpose38 python test.py
```

Expected: console prints results for:

```text
TotalCapture (Official Calibration)
TotalCapture (DIP Calibration)
DIP-IMU
```

- [ ] **Step 2: Preserve official outputs**

Copy generated result files and console logs into:

```text
output/benchmark_globalpose/official/totalcapture_officalib/
output/benchmark_globalpose/official/totalcapture_dipcalib/
output/benchmark_globalpose/official/dipimu/
```

Expected: official outputs are isolated from later DiffusionPoser outputs.

### Task 4: Inspect GlobalPose Dataset Format

**Files:**
- Read external data only.
- Create later: `documents/experiments/globalpose_oracle_tracker_eval.md`

- [ ] **Step 1: Inspect dataset keys and shapes**

Run:

```powershell
conda run --no-capture-output -n globalpose38 python - <<'PY'
import torch
from pathlib import Path

root = Path("dataset/external/globalpose/repo/data/test_datasets")
for name in ["totalcapture_dipcalib.pt", "totalcapture_officalib.pt", "dipimu.pt"]:
    data = torch.load(root / name, map_location="cpu")
    print(name, sorted(data.keys()))
    for key in ["name", "pose", "tran", "aS", "wS", "mS", "RIS", "RIM", "RSB"]:
        if key in data:
            value = data[key]
            first = value[0] if isinstance(value, list) else value
            shape = tuple(first.shape) if hasattr(first, "shape") else type(first).__name__
            print(" ", key, len(value) if isinstance(value, list) else "", shape)
PY
```

Expected:

```text
pose first sequence shape: [T,72]
tran first sequence shape: [T,3]
IMU fields first sequence shapes include [T,6,3] or [T,6,3,3]
```

### Task 5: Implement GlobalPose-To-RealtimePose Converter

**Files:**
- Create: `data_converter/globalpose_to_realtime_pose.py`
- Test: `tests/smoke/data_pipeline/test_globalpose_oracle_tracker_source.py`

- [ ] **Step 1: Write failing converter smoke test**

Test should create a tiny synthetic GlobalPose-like `.pt` with one sequence of at least 61 frames, call the converter entry point, and assert:

```text
manifest.jsonl exists
one source .npz exists
validate_realtime_source_contract passes
schema_name == realtime_pose_stationary5_v1
root_pos_world[:,1] == 0
pelvis_height == joints_world[:,0,1]
tracker_pos_world shape == [T,6,3]
tracker_rot_world_6d shape == [T,6,6]
sensor_valid shape == [T,6] and all true
stationary_prob_5 shape == [T,5]
```

- [ ] **Step 2: Run failing test**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/data_pipeline/test_globalpose_oracle_tracker_source.py -q
```

Expected: FAIL because `data_converter.globalpose_to_realtime_pose` does not exist yet.

- [ ] **Step 3: Implement minimal converter**

Implement a CLI with:

```text
--globalpose_dataset_path
--dataset_name
--output_dir
--schema realtime_pose_stationary5_v1
--limit
--overwrite
```

Implementation requirements:

```text
load GlobalPose .pt with torch.load(..., map_location="cpu")
read pose/tran/name
convert SMPL axis-angle pose/tran into this repo's source contract
derive root_yaw
encode root_heading_delta_sincos
encode root_delta_xz_ref
derive pelvis_height
derive stationary_prob_5
derive tracker_pos_world and tracker_rot_world_6d
write source .npz files
write manifest.jsonl
include metadata required by realtime_pose_stationary5_v1
```

- [ ] **Step 4: Run converter smoke test**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/data_pipeline/test_globalpose_oracle_tracker_source.py -q
```

Expected: PASS.

### Task 6: Convert First TotalCapture-DIP Sequence As Smoke

**Files:**
- Generated output: `dataset/generated/sources/realtime_pose_stationary5_v1/globalpose_totalcapture_dipcalib_oracle_tracker/`

- [ ] **Step 1: Convert one sequence**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 python -m data_converter.globalpose_to_realtime_pose `
  --globalpose_dataset_path dataset/external/globalpose/repo/data/test_datasets/totalcapture_dipcalib.pt `
  --dataset_name totalcapture_dipcalib `
  --output_dir dataset/generated/sources/realtime_pose_stationary5_v1/globalpose_totalcapture_dipcalib_oracle_tracker `
  --schema realtime_pose_stationary5_v1 `
  --limit 1 `
  --overwrite
```

Expected: one source sequence and `manifest.jsonl` are written.

- [ ] **Step 2: Validate generated source through existing loader**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 python - <<'PY'
from pathlib import Path
from data_loaders.generate_realtime_pose_tasks import load_realtime_source

root = Path("dataset/generated/sources/realtime_pose_stationary5_v1/globalpose_totalcapture_dipcalib_oracle_tracker")
paths = sorted(root.rglob("*.npz"))
assert paths, "no source npz files found"
source = load_realtime_source(paths[0], schema_name="realtime_pose_stationary5_v1")
print(paths[0])
print(source["tracker_pos_world"].shape)
PY
```

Expected: prints source path and a `[T,6,3]` tracker shape.

### Task 7: Generate Tasks And Longseq Eval For Smoke Source

**Files:**
- Generated task root: `dataset/generated/tasks/realtime_pose_stationary5_v1/globalpose_totalcapture_dipcalib_oracle_tracker_tasks/`
- Generated eval root: `dataset/generated/longseq_eval/realtime_pose_stationary5_v1/globalpose_totalcapture_dipcalib_oracle_tracker/`

- [ ] **Step 1: Generate full-tracker test tasks**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 python -m data_loaders.generate_realtime_pose_tasks `
  --schema realtime_pose_stationary5_v1 `
  --source_dir dataset/generated/sources/realtime_pose_stationary5_v1/globalpose_totalcapture_dipcalib_oracle_tracker `
  --output_dir dataset/generated/tasks/realtime_pose_stationary5_v1/globalpose_totalcapture_dipcalib_oracle_tracker_tasks `
  --splits test `
  --split_dir "" `
  --samples_per_file 1 `
  --mask_policy full `
  --run_name test_full_tracker_seed10
```

Expected: task manifest exists under the generated task run.

- [ ] **Step 2: Build long-sequence eval set**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 python -m data_loaders.build_realtime_longseq_eval_set `
  --schema realtime_pose_stationary5_v1 `
  --task_dir dataset/generated/tasks/realtime_pose_stationary5_v1/globalpose_totalcapture_dipcalib_oracle_tracker_tasks `
  --output_root dataset/generated/longseq_eval/realtime_pose_stationary5_v1/globalpose_totalcapture_dipcalib_oracle_tracker `
  --split test `
  --min_frames 61 `
  --run_name v1_smoke_one_sequence `
  --overwrite
```

Expected: `manifest.jsonl`, `config.json`, and copied source sequence exist in the longseq eval directory.

### Task 8: Run DiffusionPoser On The Smoke Eval Set

**Files:**
- Output: `output/benchmark_globalpose/ours_oracle_tracker/totalcapture_dipcalib_smoke/`

- [ ] **Step 1: Select checkpoint and normalizer**

Record concrete values:

```text
model_path=<absolute or repo-relative checkpoint path>
normalizer_dir=<absolute or repo-relative normalizer artifact path>
```

Do not run this task until both paths are known.

- [ ] **Step 2: Run longseq evaluation**

Run with chosen paths:

```powershell
conda run --no-capture-output -n diffusionposer5070 python -m sample.evaluate_longseq_eval_set `
  --schema realtime_pose_stationary5_v1 `
  --eval_root dataset/generated/longseq_eval/realtime_pose_stationary5_v1/globalpose_totalcapture_dipcalib_oracle_tracker `
  --eval_set v1_smoke_one_sequence `
  --model_path <model_path> `
  --normalizer_dir <normalizer_dir> `
  --output_dir output/benchmark_globalpose/ours_oracle_tracker/totalcapture_dipcalib_smoke `
  --history_pose_source reference `
  --warmup_target_source first_frame `
  --root_correction `
  --tracker_ik `
  --render_mp4 false
```

Expected: rollout `.npz` and summary JSON are written.

### Task 9: Add GlobalPose-Style Metrics For DiffusionPoser Output

**Files:**
- Create: `eval/globalpose_metrics.py`
- Test: `tests/smoke/eval/test_globalpose_metrics.py`

- [ ] **Step 1: Write drift metric test**

Create test cases for:

```text
perfect prediction -> 7m drift is 0
prediction with constant x offset -> travelled-window drift is 0
prediction with 10% scale error -> 7m drift is 0.7m and 10%
short sequence without 7m travelled distance -> 7m result is absent or NaN by documented policy
```

- [ ] **Step 2: Run failing test**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/eval/test_globalpose_metrics.py -q
```

Expected: FAIL because metric module does not exist yet.

- [ ] **Step 3: Implement travelled-distance drift**

Implement a pure NumPy function that accepts:

```text
predicted_root_pos_world: [T,3]
reference_root_pos_world: [T,3]
window_sizes: default 1..7
```

Return:

```text
mean_error_m_by_window
drift_percent_by_window
frame_pair_count_by_window
```

Use the same travelled-distance window definition as GlobalPose `test.py`.

- [ ] **Step 4: Run metric tests**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/eval/test_globalpose_metrics.py -q
```

Expected: PASS.

### Task 10: Scale From Smoke To Full Datasets

**Files:**
- Generated source/task/longseq eval under `dataset/generated/...`
- Results under `output/benchmark_globalpose/ours_oracle_tracker/...`

- [ ] **Step 1: Convert all TotalCapture DIP calibration sequences**

Run converter without `--limit` for:

```text
totalcapture_dipcalib.pt
```

Expected: all sequences are listed in source manifest.

- [ ] **Step 2: Generate tasks and longseq eval for TotalCapture DIP calibration**

Use the same commands as smoke, changing run name to:

```text
v1_all_sequences
```

Expected: all converted sequences appear in longseq manifest.

- [ ] **Step 3: Run DiffusionPoser on full TotalCapture DIP calibration eval**

Output:

```text
output/benchmark_globalpose/ours_oracle_tracker/totalcapture_dipcalib/
```

Expected: summary JSON exists.

- [ ] **Step 4: Repeat for TotalCapture Official calibration**

Use:

```text
totalcapture_officalib.pt
globalpose_totalcapture_officalib_oracle_tracker
output/benchmark_globalpose/ours_oracle_tracker/totalcapture_officalib/
```

Expected: summary JSON exists.

- [ ] **Step 5: Repeat for DIP-IMU pose-only evaluation**

Use:

```text
dipimu.pt
globalpose_dipimu_oracle_tracker
output/benchmark_globalpose/ours_oracle_tracker/dipimu/
```

Expected: summary JSON exists. Translation metrics must be marked not comparable or disabled for DIP-IMU.

### Task 11: Write Experiment Record

**Files:**
- Create: `documents/experiments/globalpose_oracle_tracker_eval.md`

- [ ] **Step 1: Record environment and assets**

Include:

```text
GlobalPose repo path
GlobalPose commit hash
globalpose38 Python version
PyTorch version and CUDA availability
DiffusionPoser environment
SMPL file location
weights.pt location
test dataset file locations
```

- [ ] **Step 2: Record exact commands**

Include commands for:

```text
official GlobalPose test.py
dataset inspection
converter smoke
full conversion
task generation
longseq eval generation
DiffusionPoser evaluation
metric summary generation
```

- [ ] **Step 3: Record results and limitations**

Include two result sections:

```text
GlobalPose Official IMU Results
DiffusionPoser Oracle Tracker Results
```

Include this limitation statement:

```text
DiffusionPoser oracle tracker results use tracker observations derived from ground-truth pose/tran. They validate behavior on the GlobalPose test motion distribution, but they are not a strict fair comparison against GlobalPose's sparse-IMU benchmark.
```

### Task 12: Final Verification

**Files:**
- Tests only.

- [ ] **Step 1: Run relevant smoke tests**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/data_pipeline/test_globalpose_oracle_tracker_source.py tests/smoke/eval/test_globalpose_metrics.py -q
```

Expected: PASS.

- [ ] **Step 2: Run broader affected smoke tests**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/data_pipeline tests/smoke/eval -q
```

Expected: PASS, or document unrelated pre-existing failures in the experiment record.

- [ ] **Step 3: Inspect git changes**

Run:

```powershell
git status --short
```

Expected: only planned source, test, config example, docs, and experiment files are changed. Generated `dataset/`, `output/`, `runs/`, `.pt`, `.npz`, and `.mp4` artifacts remain untracked or ignored.
