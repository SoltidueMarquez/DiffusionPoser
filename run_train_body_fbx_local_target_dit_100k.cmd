@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo [DiffusionPoser] repo=%CD%
echo [DiffusionPoser] training realtime_pose_body_fbx_local_v1 target_dit for 100000 steps
echo [DiffusionPoser] using parser defaults: EMA on, tracker_pos_loss_weight=10, history/tracker augmentation on

"D:\Anaconda\Scripts\conda.exe" run --no-capture-output -n diffusionposer5070 python -m train.train_diffusionposer ^
  --schema realtime_pose_body_fbx_local_v1 ^
  --model_arch target_dit ^
  --input_feats 211 ^
  --data_dir dataset/AMASS_realtime_pose_body_fbx_local_60hz_tasks ^
  --data_split train ^
  --normalizer_dir dataset/meta_AMASS_realtime_pose_body_fbx_local_60hz ^
  --save_dir runs/realtime_pose_body_fbx_local_target_dit ^
  --num_steps 100000

echo.
echo [DiffusionPoser] training command exited with code %ERRORLEVEL%
