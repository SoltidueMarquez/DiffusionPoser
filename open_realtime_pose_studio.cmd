@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0visual_editor\scripts\start.ps1" -DataDir "dataset\AMASS_realtime_pose_body_fbx_local_root_y0_60hz_tasks" -SourceDir "dataset\AMASS_realtime_pose_body_fbx_local_root_y0_60hz" -OutputDir "visual_editor\.runtime\exports" -Rebuild %*
