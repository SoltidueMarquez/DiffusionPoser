@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set /p NORMALIZER_DIR=<"dataset\meta_AMASS_realtime_pose_body_fbx_local_60hz\latest_normalizer.txt"

echo [DiffusionPoser] repo=%CD%
echo [DiffusionPoser] normalizer=%NORMALIZER_DIR%
echo [DiffusionPoser] training realtime_pose_body_fbx_local_v1 target_dit for 200000 steps

"D:\Anaconda\Scripts\conda.exe" run --no-capture-output -n diffusionposer5070 python -m train.train_diffusionposer ^
  --schema realtime_pose_body_fbx_local_v1 ^
  --model_arch target_dit ^
  --input_feats 211 ^
  --data_dir dataset/AMASS_realtime_pose_body_fbx_local_60hz_tasks ^
  --data_split train ^
  --normalizer_dir "%NORMALIZER_DIR%" ^
  --save_dir runs/realtime_pose_body_fbx_local_target_dit ^
  --num_steps 200000 ^
  --overwrite

echo.
echo [DiffusionPoser] training command exited with code %ERRORLEVEL%
