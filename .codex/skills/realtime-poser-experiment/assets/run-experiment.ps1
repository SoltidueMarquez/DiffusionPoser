[CmdletBinding()]
param(
    [switch]$DryRun,
    [string]$UnityRootOverride = ""
)

$ErrorActionPreference = "Stop"
$ExperimentId = "{{EXPERIMENT_ID}}"
$UnityParticipation = "{{UNITY_PARTICIPATION}}"
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$UnityRoot = if ($UnityRootOverride) {
    (Resolve-Path -LiteralPath $UnityRootOverride).Path
} else {
    (Resolve-Path -LiteralPath (Join-Path $RepoRoot "..\SIGGRAPH2024Unity")).Path
}
$RunDir = Join-Path $RepoRoot "runs\$ExperimentId"
$LogDir = Join-Path $RunDir "logs"
$OutputDir = Join-Path $RepoRoot "output\$ExperimentId"
$ManifestPath = Join-Path $RunDir "experiment_runtime.json"
$LogPath = Join-Path $LogDir "console.log"

# Replace only the experiment module and arguments. Keep conda and --no-capture-output.
$CommandArgs = @(
    "run", "--no-capture-output", "-n", "diffusionposer5070",
    "python", "-m", "{{PYTHON_MODULE}}"
    {{COMMAND_ARGUMENTS}}
)

if ($ExperimentId -match "\{\{" -or $CommandArgs -join " " -match "\{\{") {
    throw "The experiment script still contains template placeholders."
}

$DiffusionPoserCommit = (git -C $RepoRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Cannot read the DiffusionPoser commit."
}
$UnityCommit = (git -C $UnityRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Cannot read the Unity commit."
}

$CommandText = "conda " + ($CommandArgs -join " ")
Write-Host "Experiment: $ExperimentId" -ForegroundColor Green
Write-Host "DiffusionPoser commit: $DiffusionPoserCommit"
Write-Host "Unity commit:          $UnityCommit ($UnityParticipation)"
Write-Host "Command:               $CommandText"
Write-Host "Run directory:         $RunDir"

if ($DryRun) {
    Write-Host "Dry-run complete. The experiment was not started and no runtime manifest was written." -ForegroundColor Yellow
    exit 0
}

New-Item -ItemType Directory -Force -Path $RunDir, $LogDir, $OutputDir | Out-Null
$StartedAt = [DateTimeOffset]::Now.ToString("o")
$FinishedAt = $null
$Status = "running"
$ExitCode = 0
$FailureReason = $null

try {
    & conda @CommandArgs 2>&1 | Tee-Object -FilePath $LogPath
    $ExitCode = $LASTEXITCODE
    $Status = if ($ExitCode -eq 0) { "completed" } else { "failed" }
    if ($ExitCode -ne 0) {
        $FailureReason = "The experiment command exited with code $ExitCode."
    }
} catch {
    $ExitCode = 1
    $Status = "failed"
    $FailureReason = $_.Exception.Message
} finally {
    $FinishedAt = [DateTimeOffset]::Now.ToString("o")
    $Manifest = [ordered]@{
        schema_version = 1
        experiment_id = $ExperimentId
        started_at = $StartedAt
        finished_at = $FinishedAt
        status = $Status
        exit_code = $ExitCode
        failure_reason = $FailureReason
        diffusionposer_commit = $DiffusionPoserCommit
        unity_commit = $UnityCommit
        unity_participation = $UnityParticipation
        command = $CommandText
        command_args = $CommandArgs
        paths = [ordered]@{
            run_dir = $RunDir
            log_dir = $LogDir
            output_dir = $OutputDir
            console_log = $LogPath
        }
    }
    $Json = $Manifest | ConvertTo-Json -Depth 8
    [IO.File]::WriteAllText($ManifestPath, $Json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
}

if ($ExitCode -ne 0) {
    throw $FailureReason
}

Write-Host "Experiment complete. Runtime manifest: $ManifestPath" -ForegroundColor Green
