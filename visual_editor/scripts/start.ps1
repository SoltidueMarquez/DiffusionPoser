param(
    [string]$AmassDir = "",
    [string]$DataDir = "",
    [string]$SourceDir = "",
    [string]$ResultDir = "",
    [string]$OutputDir = "",
    [string]$SmplModelDir = "",
    [switch]$SkipBootstrap,
    [switch]$Rebuild,
    [switch]$SkipBuild,
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
$DistIndex = Join-Path $EditorRoot "dist\index.html"

New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
New-Item -ItemType Directory -Force -Path $DefaultOutputDir | Out-Null

function Resolve-LaunchPath {
    param([string]$Value)
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return $Value
    }
    return Join-Path $RepoRoot $Value
}

if (-not $SkipBootstrap) {
    if (-not (Test-Path $VenvPython) -or -not (Test-Path $NodeModules)) {
        Write-Host "[visual_editor] Local dependencies are missing. Installing into visual_editor/ ..."
        & (Join-Path $EditorRoot "scripts\bootstrap.ps1")
    }
}

if (-not (Test-Path $VenvPython)) {
    throw "Missing Python environment: $VenvPython. Run visual_editor/scripts/bootstrap.ps1."
}
if (-not (Test-Path $NodeModules)) {
    throw "Missing Node dependencies: $NodeModules. Run visual_editor/scripts/bootstrap.ps1."
}

if ($DataDir.Trim() -ne "") {
    $DataDir = Resolve-LaunchPath $DataDir
}
if ($AmassDir.Trim() -eq "") {
    if (Test-Path $PreferredAmassDir) {
        $AmassDir = $PreferredAmassDir
    } else {
        $AmassDir = Join-Path $RuntimeDir "amass"
    }
} else {
    $AmassDir = Resolve-LaunchPath $AmassDir
}
if ($SourceDir.Trim() -ne "") {
    $SourceDir = Resolve-LaunchPath $SourceDir
}
if ($ResultDir.Trim() -eq "") {
    if (Test-Path $PreferredResultDir) {
        $ResultDir = $PreferredResultDir
    } else {
        $ResultDir = Join-Path $RuntimeDir "results"
    }
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

$NeedsBuild = $Rebuild -or -not (Test-Path $DistIndex)
if ((-not $NeedsBuild) -and (Select-String -Path $DistIndex -Pattern '="/assets/' -Quiet)) {
    # 旧版构建使用绝对 /assets 路径，Electron loadFile 会显示黑屏；检测到就自动重建。
    $NeedsBuild = $true
}
if ((-not $SkipBuild) -and $NeedsBuild) {
    Write-Host "[visual_editor] Building local frontend..."
    npm.cmd --prefix $EditorRoot run viewer:build
}

$env:PYTHONPATH = [string]$RepoRoot
$env:REALTIME_POSE_EDITOR_AMASS_DIR = [string](Resolve-Path -LiteralPath $AmassDir -ErrorAction SilentlyContinue)
if (-not $env:REALTIME_POSE_EDITOR_AMASS_DIR) {
    $env:REALTIME_POSE_EDITOR_AMASS_DIR = [string]$AmassDir
}
if ($DataDir.Trim() -ne "") {
    $env:REALTIME_POSE_EDITOR_DATA_DIR = [string](Resolve-Path -LiteralPath $DataDir -ErrorAction SilentlyContinue)
    if (-not $env:REALTIME_POSE_EDITOR_DATA_DIR) {
        $env:REALTIME_POSE_EDITOR_DATA_DIR = [string]$DataDir
    }
}
if ($SourceDir.Trim() -ne "") {
    $env:REALTIME_POSE_EDITOR_SOURCE_DIR = [string](Resolve-Path -LiteralPath $SourceDir -ErrorAction SilentlyContinue)
    if (-not $env:REALTIME_POSE_EDITOR_SOURCE_DIR) {
        $env:REALTIME_POSE_EDITOR_SOURCE_DIR = [string]$SourceDir
    }
}
$env:REALTIME_POSE_EDITOR_RESULT_DIR = [string](Resolve-Path -LiteralPath $ResultDir -ErrorAction SilentlyContinue)
if (-not $env:REALTIME_POSE_EDITOR_RESULT_DIR) {
    $env:REALTIME_POSE_EDITOR_RESULT_DIR = [string]$ResultDir
}
$env:REALTIME_POSE_EDITOR_OUTPUT_DIR = [string](Resolve-Path -LiteralPath $OutputDir -ErrorAction SilentlyContinue)
if (-not $env:REALTIME_POSE_EDITOR_OUTPUT_DIR) {
    $env:REALTIME_POSE_EDITOR_OUTPUT_DIR = [string]$OutputDir
}
$env:REALTIME_POSE_EDITOR_RUNTIME_DIR = [string]$RuntimeDir
$env:REALTIME_POSE_EDITOR_SMPL_MODEL_DIR = [string](Resolve-Path -LiteralPath $SmplModelDir -ErrorAction SilentlyContinue)
if (-not $env:REALTIME_POSE_EDITOR_SMPL_MODEL_DIR) {
    $env:REALTIME_POSE_EDITOR_SMPL_MODEL_DIR = [string]$SmplModelDir
}

Write-Host "[visual_editor] Starting RealtimePose Studio..."
Write-Host "[visual_editor] amass_dir=$env:REALTIME_POSE_EDITOR_AMASS_DIR"
if ($env:REALTIME_POSE_EDITOR_DATA_DIR) {
    $DisplayDataDir = $env:REALTIME_POSE_EDITOR_DATA_DIR
} else {
    $DisplayDataDir = "<artifact_roots default>"
}
if ($env:REALTIME_POSE_EDITOR_SOURCE_DIR) {
    $DisplaySourceDir = $env:REALTIME_POSE_EDITOR_SOURCE_DIR
} else {
    $DisplaySourceDir = "<artifact_roots default>"
}
Write-Host "[visual_editor] data_dir=$DisplayDataDir"
Write-Host "[visual_editor] source_dir=$DisplaySourceDir"
Write-Host "[visual_editor] result_dir=$env:REALTIME_POSE_EDITOR_RESULT_DIR"
Write-Host "[visual_editor] output_dir=$env:REALTIME_POSE_EDITOR_OUTPUT_DIR"
Write-Host "[visual_editor] smpl_model_dir=$env:REALTIME_POSE_EDITOR_SMPL_MODEL_DIR"
if ($PrepareOnly) {
    Write-Host "[visual_editor] PrepareOnly completed. Electron was not started."
    exit 0
}
npm.cmd --prefix $EditorRoot run electron:dev
