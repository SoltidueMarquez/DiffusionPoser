param(
    [switch]$SkipPython,
    [switch]$SkipNode
)

$ErrorActionPreference = "Stop"
$EditorRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPath = Join-Path $EditorRoot ".venv"
$CachePath = Join-Path $EditorRoot ".cache"
$NpmCachePath = Join-Path $CachePath "npm"

New-Item -ItemType Directory -Force -Path $CachePath | Out-Null
New-Item -ItemType Directory -Force -Path $NpmCachePath | Out-Null

if (-not $SkipPython) {
    if (-not (Test-Path $VenvPath)) {
        python -m venv $VenvPath
    }
    & (Join-Path $VenvPath "Scripts\python.exe") -m pip install --upgrade pip
    & (Join-Path $VenvPath "Scripts\python.exe") -m pip install -r (Join-Path $EditorRoot "requirements.txt") --cache-dir (Join-Path $CachePath "pip")
}

if (-not $SkipNode) {
    npm.cmd --prefix $EditorRoot install --cache $NpmCachePath
}

Write-Host "visual_editor bootstrap complete."
