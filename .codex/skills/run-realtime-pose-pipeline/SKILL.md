---
name: run-realtime-pose-pipeline
description: Run the DiffusionPoser realtime pose pipeline from the repository root in a visible Windows CMD window with live logs. Use when the user asks to run scripts.run_realtime_pose_pipeline, root_y0_main, the realtime pose pipeline, or wants this pipeline launched after confirming or overriding default parameters.
---

# Run Realtime Pose Pipeline

## Purpose

Use this skill to launch `scripts.run_realtime_pose_pipeline` from the DiffusionPoser repository root in a visible `cmd.exe` window. The command must stream logs into the CMD window, so use `conda run --no-capture-output`.

## Default Parameters

- `cwd`: current DiffusionPoser repository root
- `conda_env`: `diffusionposer5070`
- `module`: `scripts.run_realtime_pose_pipeline`
- `body_fbx_rest_json`: `..\SIGGRAPH2024Unity\Assets\Projects\RealtimePose\Models\DiffusionPoser\body_fbx_rest.json`
- `run_name`: `root_y0_main`
- `overwrite`: `true`
- `cmd_mode`: `/k` so the CMD window stays open after the command exits

## Confirmation Workflow

1. First present the effective parameters and the command that will run.
2. Ask the user to confirm, or to reply with changed parameters.
3. Do not open CMD before confirmation.
4. If the user replies with parameter changes, update the effective parameters and show the revised command for confirmation.
5. If the user reply includes both parameter changes and clear confirmation, use the updated parameters and start CMD immediately.

Use a compact confirmation message like:

```text
默认参数如下：
- conda_env: diffusionposer5070
- body_fbx_rest_json: ..\SIGGRAPH2024Unity\Assets\Projects\RealtimePose\Models\DiffusionPoser\body_fbx_rest.json
- run_name: root_y0_main
- overwrite: true

将打开 CMD 并执行：
conda run --no-capture-output -n diffusionposer5070 python -m scripts.run_realtime_pose_pipeline --body_fbx_rest_json "..\SIGGRAPH2024Unity\Assets\Projects\RealtimePose\Models\DiffusionPoser\body_fbx_rest.json" --run_name root_y0_main --overwrite

确认使用这些参数启动吗？也可以直接回复要覆盖的参数。
```

## Supported Overrides

Accept these user overrides when present:

- `conda_env=<name>`
- `body_fbx_rest_json=<path>`
- `run_name=<name>`
- `overwrite=true|false`
- `extra_args=<additional CLI args>`
- `cwd=<repo path>` only if the user explicitly wants a different repository root

If `overwrite=false`, omit `--overwrite`. Append `extra_args` at the end of the Python command without inventing new defaults.

## Launch Command

After confirmation, run a PowerShell `Start-Process` command from the repository root. Use a visible CMD window because the user explicitly wants logs redirected to the CMD window.

```powershell
$repo = (Resolve-Path ".").Path
$condaEnv = "diffusionposer5070"
$bodyFbxRestJson = "..\SIGGRAPH2024Unity\Assets\Projects\RealtimePose\Models\DiffusionPoser\body_fbx_rest.json"
$runName = "root_y0_main"
$overwriteArg = "--overwrite"
$extraArgs = ""

$pythonCmd = "conda run --no-capture-output -n $condaEnv python -m scripts.run_realtime_pose_pipeline --body_fbx_rest_json `"$bodyFbxRestJson`" --run_name `"$runName`" $overwriteArg $extraArgs"
$cmd = "cd /d `"$repo`" && $pythonCmd"
Start-Process -FilePath "cmd.exe" -ArgumentList "/k `"$cmd`"" -WorkingDirectory $repo
```

Adjust the variable values to match the confirmed parameters before executing. Keep `--no-capture-output`; it is required for realtime log output in the CMD window.

## After Launch

Tell the user that CMD was opened and the command was started. Mention that the CMD window remains open because `/k` was used.
