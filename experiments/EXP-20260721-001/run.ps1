[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$Resume,
    [switch]$SkipConversion,
    [switch]$SkipFinalEvaluationMatrix,
    [int]$Device = -1,
    [int]$BatchSize = 0,
    [int]$NumWorkers = -1,
    [int]$ConverterWorkers = 0,
    [string]$UnityRootOverride = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ExperimentId = "EXP-20260721-001"
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$ConfigPath = Join-Path $PSScriptRoot "experiment.json"
$RecordPath = Join-Path $PSScriptRoot "README.md"
$SkillValidatorPath = Join-Path $RepoRoot ".codex\skills\realtime-poser-experiment\scripts\experiment_record.py"
$UnityRoot = if ([string]::IsNullOrWhiteSpace($UnityRootOverride)) {
    (Resolve-Path -LiteralPath (Join-Path $RepoRoot "..\SIGGRAPH2024Unity")).Path
} else {
    (Resolve-Path -LiteralPath $UnityRootOverride).Path
}

if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
    throw "找不到实验配置：$ConfigPath"
}
if (-not (Test-Path -LiteralPath $SkillValidatorPath -PathType Leaf)) {
    throw "找不到实验记录校验器：$SkillValidatorPath"
}

$Config = Get-Content -Raw -LiteralPath $ConfigPath -Encoding UTF8 | ConvertFrom-Json
if ([string]$Config.experiment_id -ne $ExperimentId) {
    throw "实验配置 experiment_id=$($Config.experiment_id)，期望 $ExperimentId。"
}

function Resolve-ConfiguredPath {
    param([Parameter(Mandatory = $true)][string]$Value)

    if ([IO.Path]::IsPathRooted($Value)) {
        return [IO.Path]::GetFullPath($Value)
    }
    return [IO.Path]::GetFullPath((Join-Path $RepoRoot $Value))
}

function ConvertTo-InvariantString {
    param([Parameter(Mandatory = $true)]$Value)

    if ($Value -is [double] -or $Value -is [single] -or $Value -is [decimal]) {
        return [Convert]::ToString($Value, [Globalization.CultureInfo]::InvariantCulture)
    }
    return [string]$Value
}

function Format-CommandArgument {
    param([Parameter(Mandatory = $true)][string]$Value)

    if ($Value -notmatch '[\s"]') {
        return $Value
    }
    return '"' + ($Value -replace '"', '\"') + '"'
}

function Format-CondaCommand {
    param([Parameter(Mandatory = $true)][object[]]$Arguments)

    $formatted = foreach ($argument in $Arguments) {
        Format-CommandArgument -Value ([string]$argument)
    }
    return "conda " + ($formatted -join " ")
}

function New-CondaModuleCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Module,
        [object[]]$Arguments = @()
    )

    return @(
        "run", "--no-capture-output", "-n", "diffusionposer5070",
        "python", "-m", $Module
    ) + @($Arguments)
}

function Resolve-LatestArtifactDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Kind
    )

    $pointerPath = Join-Path $Root ("latest_{0}.json" -f $Kind)
    if (Test-Path -LiteralPath $pointerPath -PathType Leaf) {
        try {
            $payload = Get-Content -Raw -LiteralPath $pointerPath -Encoding UTF8 | ConvertFrom-Json
            $candidate = if ($null -ne $payload.output_dir) {
                [string]$payload.output_dir
            } elseif ($null -ne $payload.save_dir) {
                [string]$payload.save_dir
            } else {
                ""
            }
            if (-not [string]::IsNullOrWhiteSpace($candidate) -and (Test-Path -LiteralPath $candidate)) {
                return [IO.Path]::GetFullPath($candidate)
            }
        } catch {
            Write-Warning "无法读取 latest 指针 $pointerPath：$($_.Exception.Message)"
        }
    }
    return [IO.Path]::GetFullPath($Root)
}

function Resolve-LatestTrainingDirectory {
    param([Parameter(Mandatory = $true)][string]$Root)

    $pointerPath = Join-Path $Root "latest_run.json"
    if (Test-Path -LiteralPath $pointerPath -PathType Leaf) {
        $payload = Get-Content -Raw -LiteralPath $pointerPath -Encoding UTF8 | ConvertFrom-Json
        if ($null -ne $payload.save_dir -and (Test-Path -LiteralPath ([string]$payload.save_dir))) {
            return [IO.Path]::GetFullPath([string]$payload.save_dir)
        }
    }
    return [IO.Path]::GetFullPath($Root)
}

function Get-CheckpointPath {
    param([Parameter(Mandatory = $true)][int]$Step)

    $runDirectory = Resolve-LatestTrainingDirectory -Root $TrainingRunRoot
    return Join-Path $runDirectory ("model{0:D9}.pt" -f $Step)
}

function Test-StageCompleted {
    param([Parameter(Mandatory = $true)][string]$Name)

    foreach ($stage in @($script:StageResults)) {
        if ([string]$stage.name -eq $Name -and [int]$stage.exit_code -eq 0) {
            return $true
        }
    }
    return $false
}

function Set-StageResult {
    param([Parameter(Mandatory = $true)]$Result)

    $remaining = @($script:StageResults | Where-Object { [string]$_.name -ne [string]$Result.name })
    $script:StageResults = @($remaining) + @($Result)
}

function Set-EvaluationResult {
    param([Parameter(Mandatory = $true)]$Result)

    $remaining = @($script:EvaluationResults | Where-Object { [string]$_.name -ne [string]$Result.name })
    $script:EvaluationResults = @($remaining) + @($Result)
}

function Write-JsonAtomically {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Payload
    )

    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $temporary = Join-Path $parent (".{0}.tmp" -f [IO.Path]::GetFileName($Path))
    $json = $Payload | ConvertTo-Json -Depth 20
    [IO.File]::WriteAllText($temporary, $json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
    Move-Item -Force -LiteralPath $temporary -Destination $Path
}

function Write-RuntimeManifest {
    param(
        [Parameter(Mandatory = $true)][string]$Status,
        [int]$ExitCode = 0,
        [string]$FailureReason = ""
    )

    if ($DryRun) {
        return
    }

    $finishedAt = if ($Status -in @("completed", "failed")) {
        [DateTimeOffset]::Now.ToString("o")
    } else {
        $null
    }
    $finalCheckpoint = Get-CheckpointPath -Step 130000
    if (-not (Test-Path -LiteralPath $finalCheckpoint -PathType Leaf)) {
        $finalCheckpoint = $null
    }
    $manifest = [ordered]@{
        schema_version = 1
        experiment_id = $ExperimentId
        status = $Status
        exit_code = $ExitCode
        failure_reason = if ([string]::IsNullOrWhiteSpace($FailureReason)) { $null } else { $FailureReason }
        started_at = $script:StartedAt
        finished_at = $finishedAt
        diffusionposer_commit = $DiffusionPoserCommit
        unity_commit = $UnityCommit
        unity_participation = "reference_only"
        command = $LaunchCommand
        config = $ConfigPath
        paths = [ordered]@{
            coordination_run_dir = $RunDir
            log_dir = $LogDir
            source_dir = $SourceDir
            train_task_root = $TrainTaskRoot
            train_task_dir = Resolve-LatestArtifactDirectory -Root $TrainTaskRoot -Kind "tasks"
            eval_task_root = $EvalTaskRoot
            eval_task_dir = Resolve-LatestArtifactDirectory -Root $EvalTaskRoot -Kind "tasks"
            normalizer_root = $NormalizerRoot
            normalizer_dir = Resolve-LatestArtifactDirectory -Root $NormalizerRoot -Kind "normalizer"
            longseq_root = $LongseqRoot
            longseq_set = $LongseqSetDir
            calibration_run_root = $CalibrationRunRoot
            calibration_report = $CalibrationReport
            training_run_root = $TrainingRunRoot
            training_run_dir = Resolve-LatestTrainingDirectory -Root $TrainingRunRoot
            final_checkpoint = $finalCheckpoint
            evaluation_output_root = $EvaluationOutputRoot
            evaluation_index = $EvaluationIndexPath
        }
        calibrated_loss_weights = $script:CalibratedLossWeights
        resource_estimate = $Config.resource_estimate
        stages = @($script:StageResults)
        evaluations = @($script:EvaluationResults)
    }
    Write-JsonAtomically -Path $ManifestPath -Payload $manifest
}

function Invoke-Stage {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Description,
        [Parameter(Mandatory = $true)][object[]]$CommandArgs
    )

    $commandText = Format-CondaCommand -Arguments $CommandArgs
    if ($Resume -and (Test-StageCompleted -Name $Name)) {
        Write-Host "[skip] $Name：已在 runtime manifest 中完成。" -ForegroundColor DarkYellow
        return
    }

    Write-Host ""
    Write-Host "[$Name] $Description" -ForegroundColor Cyan
    Write-Host $commandText
    if ($DryRun) {
        return
    }

    New-Item -ItemType Directory -Force -Path $RunDir, $LogDir | Out-Null
    $logPath = Join-Path $LogDir ("{0}.log" -f $Name)
    $startedAt = [DateTimeOffset]::Now.ToString("o")
    Set-StageResult -Result ([pscustomobject][ordered]@{
        name = $Name
        description = $Description
        status = "running"
        started_at = $startedAt
        finished_at = $null
        exit_code = -1
        command = $commandText
        command_args = @($CommandArgs)
        log = $logPath
    })
    Write-RuntimeManifest -Status "running"

    $exitCode = 1
    $failureReason = ""
    try {
        & conda @CommandArgs 2>&1 | Tee-Object -FilePath $logPath
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            $failureReason = "$Description 失败，exit_code=$exitCode"
        }
    } catch {
        $exitCode = 1
        $failureReason = $_.Exception.Message
    }

    $finishedAt = [DateTimeOffset]::Now.ToString("o")
    Set-StageResult -Result ([pscustomobject][ordered]@{
        name = $Name
        description = $Description
        status = if ($exitCode -eq 0) { "completed" } else { "failed" }
        started_at = $startedAt
        finished_at = $finishedAt
        exit_code = $exitCode
        command = $commandText
        command_args = @($CommandArgs)
        log = $logPath
    })
    if ($exitCode -ne 0) {
        Write-RuntimeManifest -Status "failed" -ExitCode $exitCode -FailureReason $failureReason
        throw $failureReason
    }
    Write-RuntimeManifest -Status "running"
}

function Get-BaseModelArguments {
    return @(
        "--schema", [string]$Config.schema_name,
        "--model_arch", [string]$Config.model.model_arch,
        "--input_feats", (ConvertTo-InvariantString $Config.model.input_feats),
        "--seq_len", (ConvertTo-InvariantString $Config.model.seq_len),
        "--max_seq_len", (ConvertTo-InvariantString $Config.model.max_seq_len),
        "--latent_dim", (ConvertTo-InvariantString $Config.model.latent_dim),
        "--layers", (ConvertTo-InvariantString $Config.model.layers),
        "--heads", (ConvertTo-InvariantString $Config.model.heads),
        "--diffusion_steps", (ConvertTo-InvariantString $Config.model.diffusion_steps),
        "--noise_schedule", [string]$Config.model.noise_schedule,
        "--predict_xstart", (ConvertTo-InvariantString $Config.model.predict_xstart),
        "--sigma_small", ([string][bool]$Config.model.sigma_small).ToLowerInvariant()
    )
}

function Get-ConditioningArguments {
    param([Parameter(Mandatory = $true)][object[]]$TrackerCategories)

    $arguments = @(
        "--tracker_mask_policy", "dynamic_categories",
        "--tracker_mask_seed", (ConvertTo-InvariantString $Config.conditioning.tracker_mask_seed),
        "--tracker_mask_fill", [string]$Config.conditioning.tracker_mask_fill,
        "--tracker_pos_noise_std", (ConvertTo-InvariantString $Config.conditioning.tracker_pos_noise_std),
        "--tracker_rot_noise_std", (ConvertTo-InvariantString $Config.conditioning.tracker_rot_noise_std),
        "--non_head_tracker_dropout_prob", (ConvertTo-InvariantString $Config.conditioning.non_head_tracker_dropout_prob),
        "--tracker_latency_max_frames", (ConvertTo-InvariantString $Config.conditioning.tracker_latency_max_frames),
        "--tracker_burst_dropout_prob", (ConvertTo-InvariantString $Config.conditioning.tracker_burst_dropout_prob),
        "--tracker_outlier_prob", (ConvertTo-InvariantString $Config.conditioning.tracker_outlier_prob),
        "--history_pose_noise_std", (ConvertTo-InvariantString $Config.conditioning.history_pose_noise_std),
        "--history_yaw_noise_std", (ConvertTo-InvariantString $Config.conditioning.history_yaw_noise_std),
        "--history_pose_dropout_prob", (ConvertTo-InvariantString $Config.conditioning.history_pose_dropout_prob),
        "--history_pose_replace_prob", (ConvertTo-InvariantString $Config.conditioning.history_pose_replace_prob),
        "--history_yaw_replace_prob", (ConvertTo-InvariantString $Config.conditioning.history_yaw_replace_prob),
        "--history_root_yaw_drift_std", (ConvertTo-InvariantString $Config.conditioning.history_root_yaw_drift_std),
        "--root_yaw_ref_noise_std", (ConvertTo-InvariantString $Config.conditioning.root_yaw_ref_noise_std),
        "--tracker_mask_categories"
    )
    $arguments += @($TrackerCategories)
    return $arguments
}

function Get-CalibratedLossArguments {
    if ($DryRun -and -not (Test-Path -LiteralPath $CalibrationReport -PathType Leaf)) {
        Write-Host "[dry-run] 正式训练将在 calibration 阶段完成后注入报告中的全部 loss 权重。" -ForegroundColor DarkYellow
        return ,@()
    }
    if (-not (Test-Path -LiteralPath $CalibrationReport -PathType Leaf)) {
        throw "找不到 loss 标定报告：$CalibrationReport"
    }

    $report = Get-Content -Raw -LiteralPath $CalibrationReport -Encoding UTF8 | ConvertFrom-Json
    if ($null -eq $report.weights) {
        throw "标定报告缺少 weights：$CalibrationReport"
    }
    $required = @(
        "local_rotation_loss_weight",
        "body_geometry_loss_weight",
        "tracker_relative_pos_loss_weight",
        "tracker_relative_rot_loss_weight",
        "nohip_yaw_loss_weight",
        "nohip_root_xz_loss_weight",
        "nohip_height_loss_weight",
        "contact_height_loss_weight",
        "contact_velocity_loss_weight",
        "joint_velocity_loss_weight",
        "rotation_velocity_loss_weight",
        "yaw_velocity_loss_weight"
    )
    $arguments = @()
    $weights = [ordered]@{}
    foreach ($name in $required) {
        $property = $report.weights.PSObject.Properties[$name]
        if ($null -eq $property) {
            throw "标定报告缺少 $name。"
        }
        $value = [double]$property.Value
        $weights[$name] = $value
        $arguments += @("--$name", (ConvertTo-InvariantString $value))
    }
    # 本实验仅用主 MSE 监督 stationary_prob_5，不采用 runtime 阈值 margin 辅助项。
    $weights["stationary_margin_loss_weight"] = [double]$Config.calibration.stationary_margin_loss_weight
    $script:CalibratedLossWeights = $weights
    return $arguments
}

function Get-TrainingArguments {
    param(
        [Parameter(Mandatory = $true)]$Stage,
        [Parameter(Mandatory = $true)][bool]$ResumeStage,
        [object[]]$LossArguments = @()
    )

    $arguments = @(
        "--data_dir", $TrainTaskRoot,
        "--data_split", "train",
        "--normalizer_dir", $NormalizerRoot,
        "--normalize_input", "true",
        "--save_dir", $TrainingRunRoot,
        "--run_name", "$ExperimentId-direct-baseline",
        "--seed", (ConvertTo-InvariantString $Seed),
        "--batch_size", (ConvertTo-InvariantString $EffectiveBatchSize),
        "--num_workers", (ConvertTo-InvariantString $EffectiveNumWorkers),
        "--source_cache_max_mib", (ConvertTo-InvariantString $Config.source_cache_max_mib),
        "--cuda", "true",
        "--device", (ConvertTo-InvariantString $EffectiveDevice),
        "--lr", (ConvertTo-InvariantString $Stage.lr),
        "--lr_anneal_steps", (ConvertTo-InvariantString $Config.training.lr_anneal_steps),
        "--weight_decay", (ConvertTo-InvariantString $Config.training.weight_decay),
        "--num_steps", (ConvertTo-InvariantString $Stage.target_steps),
        "--log_interval", (ConvertTo-InvariantString $Config.training.log_interval),
        "--save_interval", (ConvertTo-InvariantString $Config.training.save_interval),
        "--checkpoint_max_keep", (ConvertTo-InvariantString $Config.training.checkpoint_max_keep),
        "--rollout_steps", (ConvertTo-InvariantString $Stage.rollout_steps),
        "--short_rollout_prob", (ConvertTo-InvariantString $Stage.short_rollout_prob),
        "--short_rollout_loss_weight", (ConvertTo-InvariantString $Stage.short_rollout_loss_weight),
        "--long_rollout_prob", (ConvertTo-InvariantString $Stage.long_rollout_prob),
        "--long_rollout_loss_weight", (ConvertTo-InvariantString $Stage.long_rollout_loss_weight),
        "--long_rollout_phase1_steps", (ConvertTo-InvariantString $Stage.long_rollout_phase1_steps),
        "--long_rollout_phase2_steps", (ConvertTo-InvariantString $Stage.long_rollout_phase2_steps),
        "--long_rollout_phase1_max_horizon", (ConvertTo-InvariantString $Stage.long_rollout_phase1_max_horizon),
        "--long_rollout_phase2_max_horizon", (ConvertTo-InvariantString $Stage.long_rollout_phase2_max_horizon),
        "--long_rollout_transition_prob", (ConvertTo-InvariantString $Config.training.long_rollout_transition_prob),
        "--long_rollout_smooth_l1_beta", (ConvertTo-InvariantString $Config.training.long_rollout_smooth_l1_beta),
        "--rollout_ddim_steps", (ConvertTo-InvariantString $Config.training.rollout_ddim_steps),
        "--stationary_simple_loss_channel_weight", (ConvertTo-InvariantString $Config.calibration.official_stationary_simple_loss_channel_weight),
        "--stationary_margin_loss_weight", (ConvertTo-InvariantString $Config.calibration.stationary_margin_loss_weight),
        "--aux_loss_weight", "1",
        "--model_ema_decay", (ConvertTo-InvariantString $Config.model.model_ema_decay),
        "--model_ema_steps", (ConvertTo-InvariantString $Config.model.model_ema_steps),
        "--model_ema_update_after", (ConvertTo-InvariantString $Config.model.model_ema_update_after),
        "--gradient_clip"
    )
    $arguments += Get-BaseModelArguments
    $arguments += Get-ConditioningArguments -TrackerCategories @($Stage.tracker_categories)
    $arguments += @($LossArguments)
    if ($ResumeStage) {
        $arguments += @("--resume_checkpoint", "latest")
    } else {
        $arguments += "--resume_checkpoint="
        $arguments += "--init_checkpoint="
    }
    return $arguments
}

function Invoke-CheckpointEvaluation {
    param(
        [Parameter(Mandatory = $true)][int]$Step,
        [Parameter(Mandatory = $true)][string]$Protocol,
        [Parameter(Mandatory = $true)][string]$HistorySource,
        [Parameter(Mandatory = $true)][int]$Limit,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $checkpoint = Get-CheckpointPath -Step $Step
    if ($DryRun -and -not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
        $checkpoint = Join-Path $TrainingRunRoot ("<latest>\model{0:D9}.pt" -f $Step)
    } elseif (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
        throw "评估 checkpoint 不存在：$checkpoint"
    }

    $protocolSlug = $Protocol -replace "_", "-"
    $historySlug = $HistorySource -replace "_", "-"
    $limitSlug = if ($Limit -eq 0) { "full" } else { "limit$Limit" }
    $evaluationName = "eval-{0:D6}-{1}-{2}-{3}" -f $Step, $historySlug, $protocolSlug, $limitSlug
    $outputDir = Join-Path $EvaluationOutputRoot ("checkpoints\{0:D9}\{1}-{2}-{3}" -f $Step, $historySlug, $protocolSlug, $limitSlug)
    $summaryPath = Join-Path $outputDir "longseq_eval_summary.json"

    $maskPolicy = if ($Protocol -eq "dynamic_all") { "dynamic_categories" } else { "fixed_categories" }
    $maskCategories = if ($Protocol -eq "dynamic_all") { @("all") } else { @($Protocol) }
    $arguments = @(
        "--model_path", $checkpoint,
        "--output_dir", $outputDir,
        "--eval_root", $LongseqRoot,
        "--eval_set", "latest",
        "--schema", [string]$Config.schema_name,
        "--normalizer_dir", $NormalizerRoot,
        "--normalize_input", "true",
        "--use_ema", "true",
        "--ts_respace", [string]$Config.evaluation.ts_respace,
        "--render_mp4", "false",
        "--seed", (ConvertTo-InvariantString $Seed),
        "--batch_size", "1",
        "--history_pose_source", $HistorySource,
        "--warmup_target_source", [string]$Config.evaluation.warmup_target_source,
        "--loop_count", "1",
        "--limit", (ConvertTo-InvariantString $Limit),
        "--dropout_preset", "tracker_mask_train",
        "--tracker_mask_policy", $maskPolicy,
        "--tracker_mask_categories"
    )
    $arguments += $maskCategories
    $arguments += @(
        "--tracker_mask_segment_frames", (ConvertTo-InvariantString $Config.evaluation.tracker_mask_segment_frames),
        "--tracker_mask_seed", (ConvertTo-InvariantString $Seed),
        "--non_head_tracker_dropout_prob", "0",
        "--tracker_burst_dropout_prob", "0",
        "--tracker_latency_max_frames", "0",
        "--tracker_outlier_prob", "0",
        "--tracker_pos_noise_std", "0",
        "--tracker_rot_noise_std", "0",
        "--hip_dropout_duration_frames", "0",
        "--cuda", "true",
        "--device", (ConvertTo-InvariantString $EffectiveDevice),
        "--root_correction"
    )
    Invoke-Stage -Name $evaluationName -Description $Label -CommandArgs (New-CondaModuleCommand -Module "sample.evaluate_longseq_eval_set" -Arguments $arguments)

    if (-not $DryRun) {
        if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
            throw "评估完成但缺少 summary：$summaryPath"
        }
        Set-EvaluationResult -Result ([pscustomobject][ordered]@{
            name = $evaluationName
            step = $Step
            protocol = $Protocol
            history_pose_source = $HistorySource
            limit = $Limit
            checkpoint = $checkpoint
            output_dir = $outputDir
            summary = $summaryPath
        })
        Write-RuntimeManifest -Status "running"
    }
}

function Write-EvaluationIndex {
    if ($DryRun) {
        return
    }

    $entries = @()
    foreach ($evaluation in @($script:EvaluationResults)) {
        if (-not (Test-Path -LiteralPath ([string]$evaluation.summary) -PathType Leaf)) {
            continue
        }
        $payload = Get-Content -Raw -LiteralPath ([string]$evaluation.summary) -Encoding UTF8 | ConvertFrom-Json
        $summary = $payload.summary
        $metrics = [ordered]@{}
        foreach ($metricName in @($Config.evaluation.metrics_for_analysis)) {
            $property = $summary.PSObject.Properties[[string]$metricName]
            if ($null -ne $property) {
                $metrics[[string]$metricName] = $property.Value
            }
        }
        $entries += [pscustomobject][ordered]@{
            name = [string]$evaluation.name
            step = [int]$evaluation.step
            protocol = [string]$evaluation.protocol
            history_pose_source = [string]$evaluation.history_pose_source
            limit = [int]$evaluation.limit
            checkpoint = [string]$evaluation.checkpoint
            summary = [string]$evaluation.summary
            metrics = $metrics
        }
    }
    $index = [ordered]@{
        schema_version = 1
        experiment_id = $ExperimentId
        generated_at = [DateTimeOffset]::Now.ToString("o")
        calibration_report = $CalibrationReport
        training_run_dir = Resolve-LatestTrainingDirectory -Root $TrainingRunRoot
        selection_rule = [string]$Config.acceptance.selection_rule
        evaluations = $entries
    }
    Write-JsonAtomically -Path $EvaluationIndexPath -Payload $index
}

$ArtifactRootsConfig = Resolve-ConfiguredPath -Value ([string]$Config.paths.artifact_roots_config)
$SourceDir = Resolve-ConfiguredPath -Value ([string]$Config.paths.source_dir)
$TrainTaskRoot = Resolve-ConfiguredPath -Value ([string]$Config.paths.train_task_root)
$EvalTaskRoot = Resolve-ConfiguredPath -Value ([string]$Config.paths.eval_task_root)
$NormalizerRoot = Resolve-ConfiguredPath -Value ([string]$Config.paths.normalizer_root)
$LongseqRoot = Resolve-ConfiguredPath -Value ([string]$Config.paths.longseq_root)
$LongseqSetDir = $LongseqRoot
$CalibrationRunRoot = Resolve-ConfiguredPath -Value ([string]$Config.paths.calibration_run_root)
$TrainingRunRoot = Resolve-ConfiguredPath -Value ([string]$Config.paths.training_run_root)
$EvaluationOutputRoot = Resolve-ConfiguredPath -Value ([string]$Config.paths.evaluation_output_root)
$RunDir = Join-Path $RepoRoot "runs\$ExperimentId"
$LogDir = Join-Path $RunDir "logs"
$ManifestPath = Join-Path $RunDir "experiment_runtime.json"
$CalibrationReport = Resolve-ConfiguredPath -Value ([string]$Config.calibration.report)
$CalibrationWorkDir = Join-Path $RunDir "calibration\work"
$EvaluationIndexPath = Join-Path $RunDir "evaluation_index.json"
$Seed = [int]$Config.seed
$EffectiveDevice = if ($Device -ge 0) { $Device } else { [int]$Config.device }
$EffectiveBatchSize = if ($BatchSize -gt 0) { $BatchSize } else { [int]$Config.batch_size }
$EffectiveNumWorkers = if ($NumWorkers -ge 0) { $NumWorkers } else { [int]$Config.num_workers }
$EffectiveConverterWorkers = if ($ConverterWorkers -gt 0) { $ConverterWorkers } else { [int]$Config.data.converter_num_workers }
$DiffusionPoserCommit = (git -c "safe.directory=$RepoRoot" -C $RepoRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "无法读取 DiffusionPoser commit。"
}
$UnityCommit = (git -c "safe.directory=$UnityRoot" -C $UnityRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "无法读取 Unity commit。"
}
$LaunchCommand = if ([string]::IsNullOrWhiteSpace($MyInvocation.Line)) {
    "powershell -ExecutionPolicy Bypass -File experiments/$ExperimentId/run.ps1"
} else {
    $MyInvocation.Line.Trim()
}
$script:StartedAt = [DateTimeOffset]::Now.ToString("o")
$script:StageResults = @()
$script:EvaluationResults = @()
$script:CalibratedLossWeights = [ordered]@{}

if (-not $DryRun -and (Test-Path -LiteralPath $RunDir) -and -not $Resume) {
    throw "实验运行目录已存在：$RunDir。继续已有实验请添加 -Resume。"
}
if ($Resume -and (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
    $existingManifest = Get-Content -Raw -LiteralPath $ManifestPath -Encoding UTF8 | ConvertFrom-Json
    if ([string]$existingManifest.experiment_id -ne $ExperimentId) {
        throw "runtime manifest experiment_id 不一致。"
    }
    if ($null -ne $existingManifest.started_at) {
        $script:StartedAt = [string]$existingManifest.started_at
    }
    $script:StageResults = @($existingManifest.stages)
    $script:EvaluationResults = @($existingManifest.evaluations)
    if ($null -ne $existingManifest.calibrated_loss_weights) {
        $script:CalibratedLossWeights = $existingManifest.calibrated_loss_weights
    }
}

Set-Location $RepoRoot
Write-Host "Experiment:             $ExperimentId" -ForegroundColor Green
Write-Host "DiffusionPoser commit:  $DiffusionPoserCommit"
Write-Host "Unity reference commit: $UnityCommit"
Write-Host "Config:                 $ConfigPath"
Write-Host "Runtime manifest:       $ManifestPath"
Write-Host "Training artifacts:     $TrainingRunRoot"
Write-Host "Evaluation outputs:     $EvaluationOutputRoot"
$artifactDrive = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($SourceDir))
$availableGiB = $artifactDrive.AvailableFreeSpace / 1GB
$minimumFreeGiB = [double]$Config.resource_estimate.minimum_free_space_gib
Write-Host ("Artifact disk free:     {0:N1} GiB（脚本阈值 {1:N1} GiB）" -f $availableGiB, $minimumFreeGiB)
if (-not $DryRun -and $availableGiB -lt $minimumFreeGiB) {
    throw ("artifact 盘剩余空间不足：{0:N1} GiB；本实验至少要求 {1:N1} GiB。" -f $availableGiB, $minimumFreeGiB)
}

try {
    $validateArguments = @(
        $SkillValidatorPath,
        "validate",
        "--record", $RecordPath,
        "--phase", "pre-run",
        "--unity-root", $UnityRoot
    )
    $validateCommand = @(
        "run", "--no-capture-output", "-n", "diffusionposer5070",
        "python"
    ) + $validateArguments
    Invoke-Stage -Name "validate-record" -Description "校验实验记录与双仓库版本" -CommandArgs $validateCommand

    $preflightCode = "import torch, numpy, yaml; print('torch=' + torch.__version__)"
    $preflightCommand = @(
        "run", "--no-capture-output", "-n", "diffusionposer5070",
        "python", "-c", $preflightCode
    )
    Invoke-Stage -Name "preflight-environment" -Description "检查训练环境核心依赖" -CommandArgs $preflightCommand

    if ([bool]$Config.data.reuse_existing_source) {
        $sourceManifest = Join-Path $SourceDir "manifest.jsonl"
        if (-not (Test-Path -LiteralPath $sourceManifest -PathType Leaf)) {
            throw "配置要求复用 source，但 manifest 不存在：$sourceManifest"
        }
        Write-Host ""
        Write-Host "[source-reuse] 复用实验目录中已完成的 exact-schema source：$SourceDir" -ForegroundColor Cyan
    } elseif (-not $SkipConversion) {
        $convertArguments = @(
            "--artifact_roots_config", $ArtifactRootsConfig,
            "--schema", [string]$Config.schema_name,
            "--source_set_name", $ExperimentId,
            "--output_dir", $SourceDir,
            "--target_fps", (ConvertTo-InvariantString $Config.data.target_fps),
            "--batch_size", (ConvertTo-InvariantString $Config.data.converter_batch_size),
            "--num_workers", (ConvertTo-InvariantString $EffectiveConverterWorkers),
            "--worker_torch_threads", (ConvertTo-InvariantString $Config.data.converter_worker_torch_threads),
            "--mirror",
            "--skip_existing",
            "--rebuild_manifest"
        )
        Invoke-Stage -Name "convert" -Description "从 AMASS 转换 exact-schema realtime pose source" -CommandArgs (New-CondaModuleCommand -Module "data_converter.amass_to_realtime_pose" -Arguments $convertArguments)
    } elseif (-not $DryRun -and -not (Test-Path -LiteralPath (Join-Path $SourceDir "manifest.jsonl") -PathType Leaf)) {
        throw "使用 -SkipConversion 时 source manifest 必须已经存在：$SourceDir"
    }

    $trainTaskArguments = @(
        "--artifact_roots_config", $ArtifactRootsConfig,
        "--schema", [string]$Config.schema_name,
        "--source_set_name", $ExperimentId,
        "--task_set_name", "$ExperimentId-train",
        "--source_dir", $SourceDir,
        "--output_dir", $TrainTaskRoot,
        "--output_split_name", [string]$Config.data.train_task_output_name,
        "--split_dir", (Join-Path $RepoRoot "data_loaders\splits"),
        "--splits", "train",
        "--samples_per_source", (ConvertTo-InvariantString $Config.data.train_samples_per_source),
        "--rollout_steps", (ConvertTo-InvariantString $Config.data.train_rollout_steps),
        "--mask_policy", [string]$Config.data.mask_policy,
        "--patterns_per_source", (ConvertTo-InvariantString $Config.data.patterns_per_source),
        "--short_source_policy", "skip",
        "--seed", (ConvertTo-InvariantString $Seed),
        "--run_name", "$ExperimentId-train"
    )
    if ([bool]$Config.data.direct_task_output) {
        $trainTaskArguments += "--direct_output"
    }
    Invoke-Stage -Name "tasks-train" -Description "生成 train source-reference manifest：每 source 两个在线窗口、支持 H8" -CommandArgs (New-CondaModuleCommand -Module "data_loaders.generate_realtime_pose_tasks" -Arguments $trainTaskArguments)

    $evalTaskArguments = @(
        "--artifact_roots_config", $ArtifactRootsConfig,
        "--schema", [string]$Config.schema_name,
        "--source_set_name", $ExperimentId,
        "--task_set_name", "$ExperimentId-eval",
        "--source_dir", $SourceDir,
        "--output_dir", $EvalTaskRoot,
        "--output_split_name", [string]$Config.data.eval_task_output_name,
        "--split_dir", (Join-Path $RepoRoot "data_loaders\splits"),
        "--splits", "test",
        "--samples_per_source", (ConvertTo-InvariantString $Config.data.eval_samples_per_source),
        "--rollout_steps", (ConvertTo-InvariantString $Config.data.eval_rollout_steps),
        "--mask_policy", [string]$Config.data.mask_policy,
        "--patterns_per_source", (ConvertTo-InvariantString $Config.data.patterns_per_source),
        "--short_source_policy", "skip",
        "--seed", (ConvertTo-InvariantString $Seed),
        "--run_name", "$ExperimentId-eval"
    )
    if ([bool]$Config.data.direct_task_output) {
        $evalTaskArguments += "--direct_output"
    }
    Invoke-Stage -Name "tasks-eval" -Description "生成固定 sampling epoch 0 的 test source-reference manifest" -CommandArgs (New-CondaModuleCommand -Module "data_loaders.generate_realtime_pose_tasks" -Arguments $evalTaskArguments)

    $normalizerArguments = @(
        "--artifact_roots_config", $ArtifactRootsConfig,
        "--schema", [string]$Config.schema_name,
        "--task_set_name", "$ExperimentId-train",
        "--normalizer_name", "$ExperimentId-train",
        "--task_dir", $TrainTaskRoot,
        "--output_dir", $NormalizerRoot,
        "--split", "train",
        "--windows_per_source", (ConvertTo-InvariantString $Config.data.normalizer_windows_per_source),
        "--convergence_windows_per_source", (ConvertTo-InvariantString $Config.data.normalizer_convergence_windows_per_source),
        "--check_convergence", (ConvertTo-InvariantString $Config.data.normalizer_check_convergence),
        "--tracker_mask_seed", (ConvertTo-InvariantString $Seed),
        "--run_name", $ExperimentId
    )
    if ([bool]$Config.data.direct_normalizer_output) {
        $normalizerArguments += "--direct_output"
    }
    Invoke-Stage -Name "normalizer" -Description "按 train source-reference 执行 K2 正式统计与 K4 收敛门禁" -CommandArgs (New-CondaModuleCommand -Module "data_loaders.compute_realtime_pose_normalizer" -Arguments $normalizerArguments)

    $longseqArguments = @(
        "--task_dir", $EvalTaskRoot,
        "--task_run", "latest",
        "--task_subdir", [string]$Config.data.eval_task_output_name,
        "--output_root", $LongseqRoot,
        "--run_name", "longseq",
        "--preset", [string]$Config.data.longseq_preset,
        "--split", "test",
        "--min_frames", (ConvertTo-InvariantString $Config.data.longseq_min_frames),
        "--schema", [string]$Config.schema_name
    )
    if ([bool]$Config.data.direct_longseq_output) {
        $longseqArguments += "--direct_output"
    }
    if ([bool]$Config.data.longseq_include_mirror) {
        $longseqArguments += "--include_mirror"
    }
    Invoke-Stage -Name "longseq-set" -Description "冻结 stress-long 长序列评估集合" -CommandArgs (New-CondaModuleCommand -Module "data_loaders.build_realtime_longseq_eval_set" -Arguments $longseqArguments)

    $warmupArguments = @(
        "--data_dir", $TrainTaskRoot,
        "--data_split", "train",
        "--normalizer_dir", $NormalizerRoot,
        "--normalize_input", "true",
        "--save_dir", $CalibrationRunRoot,
        "--run_name", "$ExperimentId-calibration-warmup",
        "--seed", (ConvertTo-InvariantString $Seed),
        "--batch_size", (ConvertTo-InvariantString $EffectiveBatchSize),
        "--num_workers", (ConvertTo-InvariantString $EffectiveNumWorkers),
        "--source_cache_max_mib", (ConvertTo-InvariantString $Config.source_cache_max_mib),
        "--cuda", "true",
        "--device", (ConvertTo-InvariantString $EffectiveDevice),
        "--lr", (ConvertTo-InvariantString $Config.calibration.warmup_lr),
        "--lr_anneal_steps", "0",
        "--weight_decay", "0",
        "--num_steps", (ConvertTo-InvariantString $Config.calibration.warmup_steps),
        "--log_interval", (ConvertTo-InvariantString $Config.training.log_interval),
        "--save_interval", "5000",
        "--checkpoint_max_keep", "2",
        "--aux_loss_weight", (ConvertTo-InvariantString $Config.calibration.warmup_aux_loss_weight),
        "--stationary_simple_loss_channel_weight", (ConvertTo-InvariantString $Config.calibration.warmup_stationary_simple_loss_channel_weight),
        "--rollout_steps", "1",
        "--short_rollout_prob", "0",
        "--short_rollout_loss_weight", "0",
        "--long_rollout_prob", "0",
        "--long_rollout_loss_weight", "0",
        "--gradient_clip",
        "--no-model_ema",
        "--resume_checkpoint=",
        "--init_checkpoint="
    )
    $warmupArguments += Get-BaseModelArguments
    $warmupArguments += Get-ConditioningArguments -TrackerCategories @($Config.calibration.warmup_tracker_categories)
    Invoke-Stage -Name "calibration-warmup" -Description "训练 10k 主目标 warm-up，供梯度标定使用" -CommandArgs (New-CondaModuleCommand -Module "train.train_diffusionposer" -Arguments $warmupArguments)

    $warmupRunDirectory = Resolve-LatestTrainingDirectory -Root $CalibrationRunRoot
    $warmupCheckpoint = Join-Path $warmupRunDirectory ("model{0:D9}.pt" -f [int]$Config.calibration.warmup_steps)
    if ($DryRun -and -not (Test-Path -LiteralPath $warmupCheckpoint -PathType Leaf)) {
        $warmupCheckpoint = Join-Path $CalibrationRunRoot ("<latest>\model{0:D9}.pt" -f [int]$Config.calibration.warmup_steps)
    } elseif (-not (Test-Path -LiteralPath $warmupCheckpoint -PathType Leaf)) {
        throw "标定 warm-up checkpoint 不存在：$warmupCheckpoint"
    }
    $calibrationArguments = @(
        "--calibration_output", $CalibrationReport,
        "--init_checkpoint", $warmupCheckpoint,
        "--data_dir", $TrainTaskRoot,
        "--data_split", "train",
        "--normalizer_dir", $NormalizerRoot,
        "--normalize_input", "true",
        "--save_dir", $CalibrationWorkDir,
        "--run_name", "$ExperimentId-loss-calibration",
        "--seed", (ConvertTo-InvariantString $Seed),
        "--batch_size", (ConvertTo-InvariantString $EffectiveBatchSize),
        "--num_workers", (ConvertTo-InvariantString $EffectiveNumWorkers),
        "--source_cache_max_mib", (ConvertTo-InvariantString $Config.source_cache_max_mib),
        "--cuda", "true",
        "--device", (ConvertTo-InvariantString $EffectiveDevice),
        "--stationary_simple_loss_channel_weight", (ConvertTo-InvariantString $Config.calibration.official_stationary_simple_loss_channel_weight)
    )
    $calibrationArguments += Get-BaseModelArguments
    $calibrationArguments += Get-ConditioningArguments -TrackerCategories @("all")
    Invoke-Stage -Name "loss-calibration" -Description "标定全部辅助 loss（含 no-Hip Root XZ）" -CommandArgs (New-CondaModuleCommand -Module "train.realtime_loss_calibration" -Arguments $calibrationArguments)

    $lossArguments = Get-CalibratedLossArguments
    $stageIndex = 0
    foreach ($trainingStage in @($Config.training.stages)) {
        $resumeStage = $stageIndex -gt 0
        $targetSteps = [int]$trainingStage.target_steps
        $description = "训练直接扩散基线至 {0:N0} step" -f $targetSteps
        $arguments = Get-TrainingArguments -Stage $trainingStage -ResumeStage $resumeStage -LossArguments $lossArguments
        Invoke-Stage -Name ([string]$trainingStage.name) -Description $description -CommandArgs (New-CondaModuleCommand -Module "train.train_diffusionposer" -Arguments $arguments)

        Invoke-CheckpointEvaluation `
            -Step $targetSteps `
            -Protocol "dynamic_all" `
            -HistorySource "predicted" `
            -Limit ([int]$Config.evaluation.stage_limit) `
            -Label "阶段 checkpoint 的 predicted-history 动态 mask 评估"

        if (@($Config.evaluation.reference_history_steps) -contains $targetSteps) {
            Invoke-CheckpointEvaluation `
                -Step $targetSteps `
                -Protocol "dynamic_all" `
                -HistorySource "reference" `
                -Limit ([int]$Config.evaluation.reference_history_limit) `
                -Label "阶段 checkpoint 的 reference-history 诊断评估"
        }
        $stageIndex += 1
    }

    if (-not $SkipFinalEvaluationMatrix) {
        foreach ($fullStep in @($Config.evaluation.full_matrix_steps)) {
            foreach ($protocol in @($Config.evaluation.final_full_protocols)) {
                Invoke-CheckpointEvaluation `
                    -Step ([int]$fullStep) `
                    -Protocol ([string]$protocol) `
                    -HistorySource "predicted" `
                    -Limit 0 `
                    -Label "$fullStep step checkpoint 的完整 $protocol 协议评估"
            }
        }
    }

    if ($DryRun) {
        Write-Host ""
        Write-Host "Dry-run 完成：未创建运行目录、manifest、数据或训练产物。" -ForegroundColor Yellow
        exit 0
    }

    Write-EvaluationIndex
    Write-RuntimeManifest -Status "completed" -ExitCode 0
    Write-Host ""
    Write-Host "实验执行完成。" -ForegroundColor Green
    Write-Host "Runtime manifest: $ManifestPath"
    Write-Host "Evaluation index: $EvaluationIndexPath"
} catch {
    if (-not $DryRun) {
        Write-RuntimeManifest -Status "failed" -ExitCode 1 -FailureReason $_.Exception.Message
    }
    throw
}
