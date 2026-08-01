param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir,

    [Parameter(Mandatory = $true)]
    [int]$TargetSteps,

    [int]$SaveInterval = 5000,
    [int]$LogInterval = 100,
    [string]$WindowTitle = "DiffusionPoser Stage B rollout"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = "C:\Users\Administrator\.conda\envs\diffusionposer5070\python.exe"
$dataDir = Join-Path $repo "dataset\generated\tasks\realtime_pose_stationary5_v1\amass_60hz_v2_base_tasks\20260714_001222_v2_base_seed10"
$normalizerDir = Join-Path $repo "dataset\generated\normalizers\realtime_pose_stationary5_v1\amass_60hz_v2_train\20260714_003252_v2_base_online_seed10"
$resumeCheckpoint = Join-Path $RunDir "model000020000.pt"
$logPath = Join-Path $RunDir "stage_b_rollout.stdout.log"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python executable not found: $python"
}
if (-not (Test-Path -LiteralPath $resumeCheckpoint)) {
    throw "Resume checkpoint not found: $resumeCheckpoint"
}

$Host.UI.RawUI.WindowTitle = $WindowTitle
Set-Location -LiteralPath $repo

$trainArgs = @(
    "-u", "-m", "train.train_diffusionposer",
    "--schema", "realtime_pose_stationary5_v1",
    "--model_arch", "target_dit",
    "--input_feats", "214",
    "--data_dir", $dataDir,
    "--data_split", "train",
    "--normalizer_dir", $normalizerDir,
    "--save_dir", $RunDir,
    "--run_name", "stage_b_rollout_probe_seed10",
    "--batch_size", "16",
    "--num_workers", "0",
    "--num_steps", "$TargetSteps",
    "--save_interval", "$SaveInterval",
    "--log_interval", "$LogInterval",
    "--checkpoint_max_keep", "5",
    "--lr", "5e-5",
    "--train_platform_type", "NoPlatform",
    "--layers", "8",
    "--heads", "8",
    "--latent_dim", "512",
    "--diffusion_steps", "50",
    "--history_pose_noise_std", "0.02",
    "--history_yaw_noise_std", "0.02",
    "--history_pose_dropout_prob", "0.05",
    "--history_pose_replace_prob", "0.05",
    "--history_yaw_replace_prob", "0.0",
    "--tracker_latency_max_frames", "0",
    "--tracker_burst_dropout_prob", "0.0",
    "--tracker_outlier_prob", "0.0",
    "--tracker_mask_policy", "dynamic_categories",
    "--tracker_mask_categories", "full_six", "standard_three",
    "--rollout_steps", "2",
    "--rollout_prob", "0.5",
    "--rollout_loss_weight", "0.5",
    "--rollout_ddim_steps", "10",
    "--eval_num_batches", "4",
    "--cuda", "true",
    "--device", "0",
    "--resume_checkpoint", $resumeCheckpoint,
    "--model_ema",
    "--gradient_clip"
)

Write-Host "Stage B rollout command:"
Write-Host "$python $($trainArgs -join ' ')"
Write-Host "Log: $logPath"
Write-Host "Run directory: $RunDir"

# Windows PowerShell 5.1 wraps native stderr (including harmless PyTorch
# warnings) as ErrorRecord objects.  Keep those visible without aborting.
$ErrorActionPreference = "Continue"
& $python @trainArgs 2>&1 | Tee-Object -FilePath $logPath
$exitCode = $LASTEXITCODE
Write-Host "Training process exited with code $exitCode"
exit $exitCode
