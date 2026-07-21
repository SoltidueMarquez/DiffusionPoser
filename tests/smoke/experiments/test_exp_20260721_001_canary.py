import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_SCRIPT = REPO_ROOT / "experiments" / "EXP-20260721-001" / "run.ps1"
EXPERIMENT_CONFIG = REPO_ROOT / "experiments" / "EXP-20260721-001" / "experiment.json"


def test_normalizer_uses_converged_k4_k8_sampling() -> None:
    config = json.loads(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))

    assert config["data"]["normalizer_windows_per_source"] == 4
    assert config["data"]["normalizer_convergence_windows_per_source"] == 8
    assert config["data"]["normalizer_check_convergence"] is True

    script = RUN_SCRIPT.read_text(encoding="utf-8")
    assert '"--windows_per_source", (ConvertTo-InvariantString $Config.data.normalizer_windows_per_source)' in script
    assert (
        '"--convergence_windows_per_source", '
        '(ConvertTo-InvariantString $Config.data.normalizer_convergence_windows_per_source)'
    ) in script
    assert "执行 K4 正式统计与 K8 收敛门禁" in script


def test_canary_mode_uses_isolated_h8_training_run() -> None:
    script = RUN_SCRIPT.read_text(encoding="utf-8")

    assert "[int]$CanarySteps = 0" in script
    assert '$env:GIT_CONFIG_COUNT = "2"' in script
    assert '$env:GIT_CONFIG_KEY_0 = "safe.directory"' in script
    assert '$env:GIT_CONFIG_KEY_1 = "safe.directory"' in script
    assert "不写入用户的 global Git config" in script
    assert '"canary-b{0}-h8-s{1}" -f $EffectiveBatchSize, $CanarySteps' in script
    assert "Canary 模式不允许与 -Resume 混用" in script
    assert "Join-Path (Split-Path -Parent $OfficialTrainingRunRoot) $CanaryName" in script
    assert "if (-not $CanaryMode)" in script

    # Canary 从零开始运行强制 long-rollout 压力配置，不复用正式 checkpoint。
    assert "long_rollout_prob = 1.0" in script
    assert "long_rollout_phase1_steps = 0" in script
    assert "long_rollout_phase2_steps = 0" in script
    assert "long_rollout_phase1_max_horizon = 8" in script
    assert "long_rollout_phase2_max_horizon = 8" in script
    assert "-ResumeStage $false" in script


def test_stage_failure_keeps_native_stderr_in_its_log() -> None:
    script = RUN_SCRIPT.read_text(encoding="utf-8")

    assert '$ErrorActionPreference = "Continue"' in script
    assert "& conda @CommandArgs 2>&1 | Tee-Object -FilePath $logPath" in script
    assert '$ErrorActionPreference = $previousErrorActionPreference' in script
    assert "详见 $logPath" in script


def test_canary_mode_has_an_early_exit_before_formal_training() -> None:
    script = RUN_SCRIPT.read_text(encoding="utf-8")
    canary_start = script.index("if ($CanaryMode) {", script.index("Invoke-Stage -Name \"normalizer\""))
    formal_warmup = script.index("$warmupArguments = @(")
    canary_block = script[canary_start:formal_warmup]

    assert "Write-RuntimeManifest -Status \"completed\"" in canary_block
    assert "exit 0" in canary_block
    assert "Canary artifacts:" in canary_block
