@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0visual_editor\scripts\start_web.ps1" %*
