@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0visual_editor\scripts\start.ps1" -DataDir "dataset\AMASS_current277_60hz_missing_tasks" -SourceDir "dataset\AMASS_current277_60hz" -OutputDir "visual_editor\.runtime\exports" -Rebuild %*
