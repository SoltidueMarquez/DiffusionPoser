param(
    [string]$AmassDir = "",
    [string]$DataDir = "",
    [string]$SourceDir = "",
    [string]$ResultDir = "",
    [string]$OutputDir = "",
    [string]$SmplModelDir = "",
    [switch]$SkipBootstrap,
    [switch]$Rebuild,
    [switch]$PrepareOnly
)

$ErrorActionPreference = "Stop"
$EditorRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$RepoRoot = Resolve-Path (Join-Path $EditorRoot "..")
$RuntimeDir = Join-Path $EditorRoot ".runtime"
$DefaultOutputDir = Join-Path $RuntimeDir "exports"
$PreferredAmassDir = Join-Path $RepoRoot "dataset\AMASS"
$PreferredResultDir = Join-Path $RepoRoot "output"
$PreferredSmplModelDir = Join-Path $RepoRoot "dataset\body_models"
$VenvPython = Join-Path $EditorRoot ".venv\Scripts\python.exe"
$NodeModules = Join-Path $EditorRoot "node_modules"
$ApiLog = Join-Path $RuntimeDir "api-web.log"
$ViteLog = Join-Path $RuntimeDir "vite-web.log"

function Resolve-LaunchPath {
    param([string]$Value)
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return $Value
    }
    return Join-Path $RepoRoot $Value
}

function Stop-ProcessTree {
    param([int]$TargetProcessId)
    if ($TargetProcessId -gt 0) {
        taskkill /PID $TargetProcessId /T /F | Out-Null
    }
}

function Wait-HttpReady {
    param(
        [string]$Url,
        [string]$Name,
        [int]$TimeoutSeconds = 30
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 1 | Out-Null
            Write-Host "[visual_editor_web] $Name is ready: $Url"
            return
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    Write-Host "[visual_editor_web] timed out waiting for $Name, opening UI anyway: $Url"
}

New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null

if (-not $SkipBootstrap) {
    if (-not (Test-Path $VenvPython) -or -not (Test-Path $NodeModules)) {
        & (Join-Path $EditorRoot "scripts\bootstrap.ps1")
    }
}
if (-not (Test-Path $VenvPython)) {
    throw "Missing Python environment: $VenvPython"
}
if (-not (Test-Path $NodeModules)) {
    throw "Missing Node dependencies: $NodeModules"
}

if ($DataDir.Trim() -ne "") {
    $DataDir = Resolve-LaunchPath $DataDir
}
if ($AmassDir.Trim() -eq "") {
    $AmassDir = $PreferredAmassDir
} else {
    $AmassDir = Resolve-LaunchPath $AmassDir
}
if ($SourceDir.Trim() -ne "") {
    $SourceDir = Resolve-LaunchPath $SourceDir
}
if ($ResultDir.Trim() -eq "") {
    $ResultDir = $PreferredResultDir
} else {
    $ResultDir = Resolve-LaunchPath $ResultDir
}
if ($OutputDir.Trim() -eq "") {
    $OutputDir = $DefaultOutputDir
} else {
    $OutputDir = Resolve-LaunchPath $OutputDir
}
if ($SmplModelDir.Trim() -eq "") {
    $SmplModelDir = $PreferredSmplModelDir
} else {
    $SmplModelDir = Resolve-LaunchPath $SmplModelDir
}

if ($Rebuild) {
    npm.cmd --prefix $EditorRoot run viewer:build
}

$env:PYTHONPATH = [string]$RepoRoot
$env:REALTIME_POSE_EDITOR_AMASS_DIR = [string]$AmassDir
if ($DataDir.Trim() -ne "") {
    $env:REALTIME_POSE_EDITOR_DATA_DIR = [string]$DataDir
}
if ($SourceDir.Trim() -ne "") {
    $env:REALTIME_POSE_EDITOR_SOURCE_DIR = [string]$SourceDir
}
$env:REALTIME_POSE_EDITOR_RESULT_DIR = [string]$ResultDir
$env:REALTIME_POSE_EDITOR_OUTPUT_DIR = [string]$OutputDir
$env:REALTIME_POSE_EDITOR_RUNTIME_DIR = [string]$RuntimeDir
$env:REALTIME_POSE_EDITOR_SMPL_MODEL_DIR = [string]$SmplModelDir

if ($DataDir.Trim() -ne "") {
    $DisplayDataDir = $DataDir
} else {
    $DisplayDataDir = "<data_roots default>"
}
if ($SourceDir.Trim() -ne "") {
    $DisplaySourceDir = $SourceDir
} else {
    $DisplaySourceDir = "<data_roots default>"
}
Write-Host "[visual_editor_web] data_dir=$DisplayDataDir"
Write-Host "[visual_editor_web] amass_dir=$AmassDir"
Write-Host "[visual_editor_web] source_dir=$DisplaySourceDir"
Write-Host "[visual_editor_web] result_dir=$ResultDir"
Write-Host "[visual_editor_web] output_dir=$OutputDir"
Write-Host "[visual_editor_web] smpl_model_dir=$SmplModelDir"
if ($PrepareOnly) {
    Write-Host "[visual_editor_web] PrepareOnly completed. Services were not started."
    exit 0
}
Write-Host "[visual_editor_web] starting local API and browser UI..."

$ApiArgs = @(
    "-m", "visual_editor.server",
    "--host", "127.0.0.1",
    "--port", "8765",
    "--runtime_dir", $RuntimeDir,
    "--amass_dir", $AmassDir,
    "--result_dir", $ResultDir,
    "--output_dir", $OutputDir,
    "--smpl_model_dir", $SmplModelDir
)
if ($DataDir.Trim() -ne "") {
    $ApiArgs += @("--data_dir", $DataDir)
}
if ($SourceDir.Trim() -ne "") {
    $ApiArgs += @("--source_dir", $SourceDir)
}
$api = Start-Process -FilePath $VenvPython -ArgumentList $ApiArgs -WorkingDirectory $RepoRoot -RedirectStandardOutput $ApiLog -RedirectStandardError (Join-Path $RuntimeDir "api-web.err.log") -WindowStyle Hidden -PassThru

$vite = Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", "npm.cmd --prefix `"$EditorRoot`" run viewer:dev > `"$ViteLog`" 2>&1") -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru

Wait-HttpReady -Url "http://127.0.0.1:8765/api/health" -Name "API"
Wait-HttpReady -Url "http://127.0.0.1:5177" -Name "Vite"
Start-Process "http://127.0.0.1:5177"

Write-Host "[visual_editor_web] opened http://127.0.0.1:5177"
Write-Host "[visual_editor_web] close the browser tab when done, then press Enter here to stop local services."
[Console]::ReadLine() | Out-Null

Stop-ProcessTree -Pid $api.Id
Stop-ProcessTree -Pid $vite.Id
Write-Host "[visual_editor_web] stopped."
