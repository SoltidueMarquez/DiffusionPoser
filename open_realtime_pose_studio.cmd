@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0visual_editor\scripts\start.ps1" -OutputDir "visual_editor\.runtime\exports" -Rebuild %*
