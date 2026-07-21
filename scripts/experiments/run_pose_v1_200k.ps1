[CmdletBinding()]
param(
    [string]$ExperimentName = "pose-v1-seed10-from-scratch-200k",
    [int]$Seed = 10,
    [int]$BatchSize = 16,
    [int]$Device = 0,
    [int]$MonitorIntervalSeconds = 30,
    [switch]$Resume,
    [switch]$SkipSmoke,
    [switch]$SkipEvaluation,
    [switch]$NoMonitor,
    [switch]$DryRun,
    [switch]$MonitorOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# region 路径与公共配置

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$ProfilePath = Join-Path $RepoRoot "configs/experiments/c04-loss-v3-stable-rollout.json"
$ArtifactRootsPath = Join-Path $RepoRoot "configs/artifact_roots.example.json"
$ActiveArtifactRoot = (Resolve-Path (Join-Path $RepoRoot "../artifactStore/DiffusionPoser/active")).Path
$RunRoot = Join-Path $ActiveArtifactRoot ("runs/experiments/{0}" -f $ExperimentName)
$OutputRoot = Join-Path $ActiveArtifactRoot ("output/experiments/{0}" -f $ExperimentName)
$RunName = $ExperimentName -replace "[^A-Za-z0-9._-]+", "_"

if ([string]::IsNullOrWhiteSpace($ExperimentName)) {
    throw "ExperimentName 不能为空。"
}
if (-not (Test-Path -LiteralPath $ProfilePath)) {
    throw "找不到实验 profile：$ProfilePath"
}
if (-not (Test-Path -LiteralPath $ArtifactRootsPath)) {
    throw "找不到 artifact roots 配置：$ArtifactRootsPath"
}

$CommonTrainingArgs = @(
    "--save_dir", $RunRoot,
    "--run_name", $RunName,
    "--seed", [string]$Seed,
    "--batch_size", [string]$BatchSize,
    "--lr", "0.0001",
    "--lr_warmup_start", "0.000001",
    "--lr_warmup_steps", "2000",
    "--lr_min", "0.00001",
    "--weight_decay", "0",
    "--log_interval", "100",
    "--save_interval", "5000",
    "--checkpoint_max_keep", "8",
    "--gradient_clip",
    "--eval_during_training",
    "--eval_num_batches", "4",
    "--cuda", "true",
    "--device", [string]$Device
)

# endregion

# region 通用命令

function Invoke-CondaCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    Write-Host ""
    Write-Host "[$Description] conda $($Arguments -join ' ')" -ForegroundColor Cyan
    & conda @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description 失败，exit_code=$LASTEXITCODE"
    }
}

function Invoke-ProfileValidation {
    $arguments = @(
        "run", "--no-capture-output", "-n", "diffusionposer5070",
        "python", "-m", "scripts.experiments.run_profile",
        "--experiment-config", $ProfilePath,
        "--artifact-roots-config", $ArtifactRootsPath,
        "--stage", "validate"
    )
    Invoke-CondaCommand -Arguments $arguments -Description "Artifact validation"
}

function Invoke-TrainingStage {
    param(
        [Parameter(Mandatory = $true)]
        [string]$StageName,
        [Parameter(Mandatory = $true)]
        [int]$TargetSteps,
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$ResumeCheckpoint,
        [Parameter(Mandatory = $true)]
        [string[]]$RolloutArgs
    )

    $arguments = @(
        "run", "--no-capture-output", "-n", "diffusionposer5070",
        "python", "-m", "scripts.experiments.run_profile",
        "--experiment-config", $ProfilePath,
        "--artifact-roots-config", $ArtifactRootsPath,
        "--stage", "train"
    )
    if ($DryRun) {
        $arguments += "--dry-run"
    }
    $arguments += "--"
    $arguments += $CommonTrainingArgs
    # Profile 中保留 C04 warm-start 路径；这里显式清空，保证首次运行是真正从零训练。
    $arguments += "--init_checkpoint="
    if ([string]::IsNullOrWhiteSpace($ResumeCheckpoint)) {
        $arguments += "--resume_checkpoint="
    }
    else {
        $arguments += @("--resume_checkpoint", $ResumeCheckpoint)
    }
    $arguments += @("--num_steps", [string]$TargetSteps)
    $arguments += $RolloutArgs

    Invoke-CondaCommand -Arguments $arguments -Description $StageName
}

# endregion

# region Run 与 checkpoint 定位

function Get-LatestRunDirectory {
    $pointerPath = Join-Path $RunRoot "latest_run.json"
    if (-not (Test-Path -LiteralPath $pointerPath)) {
        return $null
    }
    $payload = Get-Content -Raw -LiteralPath $pointerPath | ConvertFrom-Json
    if ($null -eq $payload.save_dir -or [string]::IsNullOrWhiteSpace([string]$payload.save_dir)) {
        throw "latest_run.json 缺少 save_dir：$pointerPath"
    }
    return [string]$payload.save_dir
}

function Get-LatestCheckpointInfo {
    $runDirectory = Get-LatestRunDirectory
    if ([string]::IsNullOrWhiteSpace($runDirectory) -or -not (Test-Path -LiteralPath $runDirectory)) {
        return $null
    }
    $checkpoint = Get-ChildItem -LiteralPath $runDirectory -Filter "model*.pt" -File |
        Sort-Object Name |
        Select-Object -Last 1
    if ($null -eq $checkpoint) {
        return $null
    }
    $match = [regex]::Match($checkpoint.BaseName, "^model(?<step>\d{9})$")
    if (-not $match.Success) {
        throw "无法从 checkpoint 名称解析 step：$($checkpoint.Name)"
    }
    return [pscustomobject]@{
        RunDirectory = $runDirectory
        Path = $checkpoint.FullName
        Step = [int]$match.Groups["step"].Value
    }
}

# endregion

# region 实时监控

function Convert-ToFiniteDouble {
    param($Value)

    $number = 0.0
    if ($null -eq $Value -or -not [double]::TryParse([string]$Value, [ref]$number)) {
        return [double]::NaN
    }
    return $number
}

function Get-CsvField {
    param(
        [Parameter(Mandatory = $true)]
        $Row,
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $property = $Row.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return ""
    }
    return $property.Value
}

function Start-LossMonitor {
    Write-Host "等待训练目录：$RunRoot" -ForegroundColor Cyan
    while ($true) {
        $runDirectory = Get-LatestRunDirectory
        if (-not [string]::IsNullOrWhiteSpace($runDirectory)) {
            $progressPath = Join-Path $runDirectory "progress.csv"
            if (Test-Path -LiteralPath $progressPath) {
                try {
                    $rows = @(Import-Csv -LiteralPath $progressPath)
                    if ($rows.Count -gt 0) {
                        $last = $rows[-1]
                        $recent = @($rows | Select-Object -Last 5)
                        $step = [int](Convert-ToFiniteDouble $last.step)
                        $simpleLoss = Convert-ToFiniteDouble $last.simple_loss
                        $auxLoss = Convert-ToFiniteDouble $last.aux_loss
                        $loss = Convert-ToFiniteDouble $last.loss
                        $gradNorm = Convert-ToFiniteDouble $last.grad_norm_pre_clip
                        $clipFraction = Convert-ToFiniteDouble $last.grad_clipped_fraction
                        $ratio = if ($simpleLoss -gt 0.0) { $auxLoss / $simpleLoss } else { [double]::NaN }
                        $stage = if ($step -lt 80000) { "A/base" } elseif ($step -lt 140000) { "B/H1" } else { "C/H1+H2-8" }

                        Clear-Host
                        Write-Host "DiffusionPoser Pose V1 200k Loss Monitor" -ForegroundColor Green
                        Write-Host "Run: $runDirectory"
                        [pscustomobject]@{
                            Stage = $stage
                            Step = $step
                            LR = $last.lr
                            Loss = $last.loss
                            SimpleLoss = $last.simple_loss
                            AuxLoss = $last.aux_loss
                            AuxSimpleRatio = if ([double]::IsNaN($ratio)) { "n/a" } else { "{0:F4}" -f $ratio }
                            GradNorm = $last.grad_norm_pre_clip
                            GradClipFraction = $last.grad_clipped_fraction
                            ShortRolloutLoss = Get-CsvField -Row $last -Name "short_rollout_loss"
                            LongRolloutLoss = Get-CsvField -Row $last -Name "long_rollout_loss"
                            EvalLoss = Get-CsvField -Row $last -Name "eval/loss"
                        } | Format-List

                        $warnings = [System.Collections.Generic.List[string]]::new()
                        if ([double]::IsNaN($loss) -or [double]::IsInfinity($loss)) {
                            $warnings.Add("loss 出现 NaN/Inf")
                        }
                        if ($step -ge 5000 -and $recent.Count -ge 5) {
                            $allClipHigh = @($recent | Where-Object {
                                (Convert-ToFiniteDouble $_.grad_clipped_fraction) -ge 0.8
                            }).Count -eq $recent.Count
                            $allGradHigh = @($recent | Where-Object {
                                (Convert-ToFiniteDouble $_.grad_norm_pre_clip) -gt 10.0
                            }).Count -eq $recent.Count
                            $allRatioHigh = @($recent | Where-Object {
                                $simple = Convert-ToFiniteDouble $_.simple_loss
                                $aux = Convert-ToFiniteDouble $_.aux_loss
                                $simple -gt 0.0 -and ($aux / $simple) -gt 1.0
                            }).Count -eq $recent.Count
                            if ($allClipHigh) {
                                $warnings.Add("最近 5 个日志窗口 grad_clipped_fraction 均 >= 0.8")
                            }
                            if ($allGradHigh) {
                                $warnings.Add("最近 5 个日志窗口 grad_norm_pre_clip 均 > 10")
                            }
                            if ($allRatioHigh) {
                                $warnings.Add("最近 5 个日志窗口 aux_loss/simple_loss 均 > 1")
                            }
                        }

                        if ($warnings.Count -gt 0) {
                            Write-Host "警告：" -ForegroundColor Red
                            foreach ($warning in $warnings) {
                                Write-Host "  - $warning" -ForegroundColor Red
                            }
                            Write-Host "确认异常后，请在训练终端按 Ctrl+C；监控窗口不会自动杀训练。" -ForegroundColor Yellow
                        }
                        else {
                            Write-Host "当前未触发硬报警。Stage 切换时总 loss 突升不等于训练发散。" -ForegroundColor DarkGreen
                        }

                        if ($step -ge 200000) {
                            Write-Host "训练已达到 200k。" -ForegroundColor Green
                            return
                        }
                    }
                }
                catch {
                    Write-Host "读取 progress.csv 时遇到临时写入竞争，将在下次刷新重试：$($_.Exception.Message)" -ForegroundColor Yellow
                }
            }
        }
        Start-Sleep -Seconds ([Math]::Max(5, $MonitorIntervalSeconds))
    }
}

function Open-LossMonitorWindow {
    if ($NoMonitor -or $DryRun) {
        return
    }
    $argumentLine = @(
        "-NoExit",
        "-ExecutionPolicy", "Bypass",
        "-File", ('"{0}"' -f $PSCommandPath),
        "-MonitorOnly",
        "-ExperimentName", ('"{0}"' -f $ExperimentName),
        "-MonitorIntervalSeconds", [string]$MonitorIntervalSeconds
    ) -join " "
    Start-Process -FilePath "powershell.exe" -ArgumentList $argumentLine | Out-Null
}

# endregion

# region 最终 EMA 评估

function Invoke-FinalEvaluation {
    if ($SkipEvaluation -or $DryRun) {
        return
    }
    $runDirectory = Get-LatestRunDirectory
    if ([string]::IsNullOrWhiteSpace($runDirectory)) {
        throw "找不到最终 run 目录，无法评估。"
    }
    $modelPath = Join-Path $runDirectory "model000200000.pt"
    $emaPath = Join-Path $runDirectory "ema000200000.pt"
    if (-not (Test-Path -LiteralPath $modelPath)) {
        throw "找不到最终模型：$modelPath"
    }
    if (-not (Test-Path -LiteralPath $emaPath)) {
        throw "找不到最终 EMA：$emaPath"
    }

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $evaluationOutput = Join-Path $OutputRoot ("{0}/ema_000200000" -f $timestamp)
    $arguments = @(
        "run", "--no-capture-output", "-n", "diffusionposer5070",
        "python", "-m", "scripts.experiments.run_profile",
        "--experiment-config", $ProfilePath,
        "--artifact-roots-config", $ArtifactRootsPath,
        "--stage", "evaluate",
        "--",
        "--model_path", $modelPath,
        "--output_dir", $evaluationOutput,
        "--use_ema", "true",
        "--render_mp4", "false",
        "--limit", "0"
    )
    Invoke-CondaCommand -Arguments $arguments -Description "Final 200k EMA evaluation"

    $summaryPath = Join-Path $evaluationOutput "longseq_eval_summary.json"
    if (-not (Test-Path -LiteralPath $summaryPath)) {
        throw "评估完成但找不到 summary：$summaryPath"
    }
    $summary = (Get-Content -Raw -LiteralPath $summaryPath | ConvertFrom-Json).summary
    Write-Host ""
    Write-Host "200k EMA 主要姿态指标" -ForegroundColor Green
    [pscustomobject]@{
        MPJPE_cm = [Math]::Round([double]$summary.mpjpe_mean * 100.0, 3)
        MPJRE_deg = [Math]::Round([double]$summary.mpjre_deg, 3)
        MPJVE_cmps = [Math]::Round([double]$summary.mpjve_cmps, 3)
        Jitter_mps3 = [Math]::Round([double]$summary.jitter_mps3, 3)
        FullSix_MPJPE_cm = [Math]::Round([double]$summary.full_six_mpjpe_cm, 3)
        StandardThree_MPJPE_cm = [Math]::Round([double]$summary.standard_three_mpjpe_cm, 3)
        Reconnect_MPJVE_cmps = [Math]::Round([double]$summary.transition_3_to_6_reconnect_mpjve_cmps, 3)
        Reconnect_Jitter_mps3 = [Math]::Round([double]$summary.transition_3_to_6_reconnect_jitter_mps3, 3)
        PJ = [Math]::Round([double]$summary.pj, 3)
        AUJ = [Math]::Round([double]$summary.auj, 3)
    } | Format-List
    Write-Host "完整结果：$summaryPath" -ForegroundColor Cyan
}

# endregion

# region 主流程

Set-Location $RepoRoot

if ($MonitorOnly) {
    Start-LossMonitor
    exit 0
}

if (-not $SkipSmoke) {
    Invoke-CondaCommand -Arguments @(
        "run", "--no-capture-output", "-n", "diffusionposer5070",
        "pytest", "tests/smoke", "-q"
    ) -Description "Smoke tests"
}
Invoke-ProfileValidation

$existingPointer = Join-Path $RunRoot "latest_run.json"
if (-not $Resume -and -not $DryRun -and (Test-Path -LiteralPath $existingPointer)) {
    throw "实验目录已有 latest_run.json。若确认继续该 run，请加 -Resume；若要重训，请换 -ExperimentName。"
}

$checkpointInfo = if ($Resume) { Get-LatestCheckpointInfo } else { $null }
if ($Resume -and $null -eq $checkpointInfo) {
    throw "指定了 -Resume，但 $RunRoot 中没有可恢复的 model checkpoint。"
}
$currentStep = if ($null -eq $checkpointInfo) { 0 } else { [int]$checkpointInfo.Step }
if ($currentStep -gt 200000) {
    throw "latest checkpoint step=$currentStep，超过本实验目标 200000。"
}

Write-Host ""
Write-Host "Experiment: $ExperimentName" -ForegroundColor Green
Write-Host "Run root:   $RunRoot"
Write-Host "Start step: $currentStep"
Write-Host "Target:     200000"
Open-LossMonitorWindow

if ($currentStep -lt 200000) {
    $resumeValue = if ($currentStep -gt 0) { "latest" } else { "" }
    Invoke-TrainingStage -StageName "Single-process rollout curriculum 0-200k" -TargetSteps 200000 -ResumeCheckpoint $resumeValue -RolloutArgs @(
        "--rollout_steps", "9",
        "--short_rollout_prob", "0.5",
        "--short_rollout_loss_weight", "0.5",
        "--long_rollout_prob", "0.125",
        "--long_rollout_loss_weight", "0.5",
        "--rollout_h1_start_step", "80000",
        "--rollout_h2_start_step", "140000",
        "--rollout_h4_start_step", "150000",
        "--rollout_h8_start_step", "170000",
        "--rollout_prob_ramp_steps", "10000",
        "--rollout_max_horizon_prob", "0.5",
        "--long_rollout_transition_prob", "0.5"
    )
}

if ($DryRun) {
    Write-Host "Dry-run 完成：没有启动训练、监控或评估。" -ForegroundColor Green
    exit 0
}

Invoke-FinalEvaluation
Write-Host "Pose V1 200k 流程完成。" -ForegroundColor Green

# endregion
