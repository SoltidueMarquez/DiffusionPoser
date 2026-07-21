from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from data_loaders.sensor_masking import (
    POSE_REPRESENTATION_KEY,
    REALTIME_POSE_INPUT_DIM,
    REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SEQ_LEN,
    get_schema_spec,
)
from export.export_sentis_denoiser import (
    DEFAULT_MODEL_CONFIG,
    UNITY_EXPORT_SOURCE_NAME,
    UNITY_MODEL_PACKAGE_MANIFEST_NAME,
    UNITY_MODEL_PROFILE_ASSET_NAME,
    build_unity_export_source,
    build_unity_model_package_manifest,
    build_model_config,
    resolve_unity_model_id,
    validate_normalizer_export_contract,
)
from export.write_unity_runtime_assets import (
    build_normalizer,
    build_realtime_pose_feature_schema,
    write_runtime_assets,
)
from schemas.registry import get_schema_adapter
from utils.normalizer import RealtimePoseNormalizer


CANONICAL_SCHEMA_NAME = "realtime_pose_stationary5_v1"
LEGACY_SCHEMA_NAME = "realtime_pose_body_fbx_local_root_y0_v1"


def write_legacy_policyless_normalizer(normalizer_dir, schema):
    normalizer_dir.mkdir()
    torch.save(torch.zeros(REALTIME_POSE_INPUT_DIM), normalizer_dir / "mean.pt")
    torch.save(torch.ones(REALTIME_POSE_INPUT_DIM), normalizer_dir / "std.pt")
    (normalizer_dir / "normalizer_meta.json").write_text(
        json.dumps(
            {
                "schema_name": schema.name,
                POSE_REPRESENTATION_KEY: schema.pose_representation,
                "feature_dim": schema.feature_dim,
                "eps": 1e-8,
            }
        ),
        encoding="utf-8",
    )


def make_sentis_cli_args(model_path, **overrides):
    values = {key: None for key in DEFAULT_MODEL_CONFIG}
    values.update({"model_path": str(model_path)})
    values.update(overrides)
    return SimpleNamespace(**values)


def test_runtime_assets_are_realtime_pose_only(tmp_path):
    assets = write_runtime_assets(output_dir=tmp_path, normalize_input=False, schema_name=REALTIME_POSE_SCHEMA_NAME)
    schema = json.loads(assets["feature_schema"].read_text(encoding="utf-8"))
    schema_spec = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    assert schema["schemaVersion"] == 4
    assert schema["schemaName"] == REALTIME_POSE_SCHEMA_NAME
    assert schema["poseRepresentation"] == schema_spec.pose_representation
    assert schema["featureDim"] == REALTIME_POSE_INPUT_DIM
    assert schema["sequenceLength"] == REALTIME_POSE_SEQ_LEN
    assert schema["targetStart"] == 60
    assert schema["targetLength"] == 1
    assert schema["targetFeatureLength"] == schema_spec.target_dim
    assert schema["bodyPoseBodyFbxLocalDelta6d"] == {"name": schema_spec.body_pose_key, "start": 0, "length": 144}
    assert schema["rootHeadingDeltaSinCos"] == {"name": "root_heading_delta_sincos", "start": 144, "length": 2}
    assert schema["rootDeltaXZReference"] == {"name": "root_delta_xz_ref", "start": 146, "length": 2}
    assert schema["pelvisHeight"] == {"name": "pelvis_height", "start": 148, "length": 1}
    assert schema["stationaryProb5"]["name"] == "stationary_prob_5"
    assert schema["stationaryProb5"]["start"] == 149
    assert schema["stationaryProb5"]["length"] == 5
    assert schema["stationaryProb5"]["jointIndices"] == [0, 10, 11, 22, 23]
    assert schema["trackerPositionReference"] == {"name": "tracker_pos_ref", "start": 154, "length": 18}
    assert schema["trackerRotation6dReference"] == {"name": "tracker_rot_ref_6d", "start": 172, "length": 36}
    assert schema["sensorValid"] == {"name": "sensor_valid", "start": 208, "length": 6}
    assert schema["runtimeRules"]["poseRepresentation"] == schema_spec.pose_representation
    assert schema["runtimeRules"]["rootPositionY"] == schema_spec.root_y_policy
    assert schema["runtimeRules"]["pelvisHeightApplication"] == schema_spec.pelvis_height_mode
    assert schema["runtimeRules"]["onnxDummyInputShape"] == [1, REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SEQ_LEN]


def test_runtime_assets_write_exact_schema_for_stationary5_aliases(tmp_path):
    for schema_name in (CANONICAL_SCHEMA_NAME, LEGACY_SCHEMA_NAME):
        assets = write_runtime_assets(
            output_dir=tmp_path / schema_name,
            normalize_input=False,
            schema_name=schema_name,
        )

        feature_schema = json.loads(assets["feature_schema"].read_text(encoding="utf-8"))

        assert feature_schema["schemaName"] == schema_name
        assert feature_schema["featureDim"] == 214
        assert feature_schema["runtimeRules"]["rootPositionY"] == "fixed_zero"


def test_sibling_unity_runtime_accepts_only_current_stationary5_schema_name():
    unity_schema_path = (
        __import__("pathlib").Path(__file__).resolve().parents[3].parent
        / "SIGGRAPH2024Unity"
        / "Assets"
        / "Projects"
        / "RealtimePose"
        / "Scripts"
        / "Features"
        / "PoseFeatureSchema.cs"
    )
    if not unity_schema_path.exists():
        pytest.skip("Sibling Unity project is not available in this workspace.")

    source = unity_schema_path.read_text(encoding="utf-8")

    assert 'RealtimePoseSchemaName = "realtime_pose_stationary5_v1"' in source
    assert "LegacyRealtimePoseSchemaName" not in source
    assert "return value == RealtimePoseSchemaName;" in source


def test_sibling_unity_auto_creates_one_profile_per_exported_model_package():
    unity_root = __import__("pathlib").Path(__file__).resolve().parents[3].parent / "SIGGRAPH2024Unity"
    scripts_root = unity_root / "Assets" / "Projects" / "RealtimePose" / "Scripts"
    profile_path = scripts_root / "Core" / "RealtimePoseModelProfile.cs"
    importer_path = scripts_root / "Editor" / "RealtimePoseModelProfileAutoImporter.cs"
    driver_path = scripts_root / "Core" / "DiffusionPoserRealtimeDriver.cs"
    if not profile_path.exists():
        pytest.skip("Sibling Unity project is not available in this workspace.")

    profile_source = profile_path.read_text(encoding="utf-8")
    importer_source = importer_path.read_text(encoding="utf-8")
    driver_source = driver_path.read_text(encoding="utf-8")

    assert "CreateAssetMenu" in profile_source
    assert "public ModelAsset DenoiserModel;" in profile_source
    assert "public TextAsset FeatureSchemaJson;" in profile_source
    assert "public TextAsset NormalizerJson;" in profile_source
    assert "public TextAsset DdimScheduleJson;" in profile_source
    assert "public TextAsset ExportSourceJson;" in profile_source
    assert "ValidateRuntimeAssetMetadata(NormalizerJson" in profile_source
    assert 'ManifestFileName = "realtime_pose_model_package.json"' in importer_source
    assert "OnPostprocessAllAssets" in importer_source
    assert "AssetDatabase.CreateAsset(profile, profilePath);" in importer_source
    assert "profile.ApplyImportedPackage(" in importer_source
    assert "public RealtimePoseModelProfile ModelProfile;" in driver_source
    assert "ModelProfile.LoadRuntimeAssets" in driver_source
    assert "TrySwitchModelProfile" in driver_source
    assert "public ModelAsset DenoiserModel;" not in driver_source
    assert "public TextAsset FeatureSchemaJson;" not in driver_source
    assert "public TextAsset NormalizerJson;" not in driver_source
    assert "public TextAsset DdimScheduleJson;" not in driver_source
    assert "ModelProfile != null ?" not in driver_source
    assert "GetEffectiveSampleSteps" not in driver_source
    assert "GetEffectiveClipDenoised" not in driver_source
    assert not list((scripts_root / "Editor").glob("RealtimePose*ModelProfileUpgrade.cs"))


def test_sibling_unity_replay_and_schedule_require_current_exact_schema_name():
    unity_root = __import__("pathlib").Path(__file__).resolve().parents[3].parent / "SIGGRAPH2024Unity"
    replay_path = unity_root / "Assets" / "Projects" / "RealtimePose" / "Scripts" / "Input" / "JsonReplayInput.cs"
    schedule_path = unity_root / "Assets" / "Projects" / "RealtimePose" / "Scripts" / "Inference" / "DdimSchedule.cs"
    if not replay_path.exists():
        pytest.skip("Sibling Unity project is not available in this workspace.")

    replay_source = replay_path.read_text(encoding="utf-8")
    schedule_source = schedule_path.read_text(encoding="utf-8")

    assert "stream.schemaName != PoseFeatureSchema.RealtimePoseSchemaName" in replay_source
    assert "schemaName != PoseFeatureSchema.RealtimePoseSchemaName" in schedule_source


def test_sibling_unity_replay_sensor_valid_static_mask_is_consistent_and_recorded():
    unity_root = __import__("pathlib").Path(__file__).resolve().parents[3].parent / "SIGGRAPH2024Unity"
    scripts_root = unity_root / "Assets" / "Projects" / "RealtimePose" / "Scripts"
    types_path = scripts_root / "Core" / "RealtimePoseTypes.cs"
    driver_path = scripts_root / "Core" / "DiffusionPoserRealtimeDriver.cs"
    replay_path = scripts_root / "Input" / "JsonReplayInput.cs"
    evaluator_path = scripts_root / "Debug" / "RealtimePoseReplayEvaluator.cs"
    if not types_path.exists():
        pytest.skip("Sibling Unity project is not available in this workspace.")

    types_source = types_path.read_text(encoding="utf-8")
    driver_source = driver_path.read_text(encoding="utf-8")
    replay_source = replay_path.read_text(encoding="utf-8")
    evaluator_source = evaluator_path.read_text(encoding="utf-8")

    assert "ApplyStaticMask" in types_source
    assert "[Flags]" in types_source
    assert "ReplayTrackerMask.Head | ReplayTrackerMask.LeftWrist | ReplayTrackerMask.RightWrist" in driver_source
    assert "jsonReplayInput.SensorValidOverride = ReplaySensorValidOverride" in driver_source
    assert "jsonReplayInput.SensorValidMask = ReplaySensorValidMask" in driver_source
    assert "IsSensorValid(frameIndex, (TrackerSensor)sensor)" in replay_source
    assert "bool valid = IsSensorValid(frameIndex, sensor);" in replay_source
    assert "if (!replayValid || SensorValidOverride == ReplaySensorValidOverrideMode.FromReplay)" in replay_source
    assert "Replay sensor_valid static mask must include Head." in replay_source
    assert "TrackerFrame.MinRealtimePoseValidSensors" in replay_source
    assert "public string sensorValidOverride;" in evaluator_source
    assert "ReplaySource.ReplaySensorValidOverrideDescription" in evaluator_source


def test_sibling_unity_runtime_uses_only_main_stationary_feature_channel():
    unity_root = __import__("pathlib").Path(__file__).resolve().parents[3].parent / "SIGGRAPH2024Unity"
    types_path = (
        unity_root
        / "Assets"
        / "Projects"
        / "RealtimePose"
        / "Scripts"
        / "Core"
        / "RealtimePoseTypes.cs"
    )
    prediction_path = (
        unity_root
        / "Assets"
        / "Projects"
        / "RealtimePose"
        / "Scripts"
        / "Output"
        / "PosePrediction.cs"
    )
    pipeline_path = (
        unity_root
        / "Assets"
        / "Projects"
        / "RealtimePose"
        / "Scripts"
        / "Core"
        / "RealtimePoseInferencePipeline.cs"
    )
    driver_path = (
        unity_root
        / "Assets"
        / "Projects"
        / "RealtimePose"
        / "Scripts"
        / "Core"
        / "DiffusionPoserRealtimeDriver.cs"
    )
    if not types_path.exists():
        pytest.skip("Sibling Unity project is not available in this workspace.")

    types_source = types_path.read_text(encoding="utf-8")
    prediction_source = prediction_path.read_text(encoding="utf-8")
    pipeline_source = pipeline_path.read_text(encoding="utf-8")
    driver_source = driver_path.read_text(encoding="utf-8")

    assert "StationarySignalSource" not in types_source
    assert "HasStationaryHead" not in prediction_source
    assert "ApplyStationarySignalSource" not in pipeline_source
    assert "StationarySignalSource" not in driver_source


def test_sibling_unity_replay_metrics_record_main_stationary_lock_rates():
    unity_root = __import__("pathlib").Path(__file__).resolve().parents[3].parent / "SIGGRAPH2024Unity"
    evaluator_path = (
        unity_root
        / "Assets"
        / "Projects"
        / "RealtimePose"
        / "Scripts"
        / "Debug"
        / "RealtimePoseReplayEvaluator.cs"
    )
    runner_path = (
        unity_root
        / "Assets"
        / "Projects"
        / "RealtimePose"
        / "Scripts"
        / "Editor"
        / "RealtimePoseAutomatedReplayTestRunner.cs"
    )
    if not evaluator_path.exists():
        pytest.skip("Sibling Unity project is not available in this workspace.")

    evaluator_source = evaluator_path.read_text(encoding="utf-8")
    runner_source = runner_path.read_text(encoding="utf-8")

    assert "stationarySignalSourceUsed" not in evaluator_source
    assert "falseLockRate" in evaluator_source
    assert "missedLockRate" in evaluator_source
    assert "stationaryProbJitter" in evaluator_source
    assert "StationarySignalSource" not in runner_source
    assert "playbackController.EvaluationWriteMetricsOnDisable = true" in runner_source


def test_export_feature_schema_matches_legacy_adapter_builder():
    assert build_realtime_pose_feature_schema(schema_name=LEGACY_SCHEMA_NAME) == get_schema_adapter(
        LEGACY_SCHEMA_NAME
    ).build_unity_feature_schema()


def test_realtime_normalizer_requires_schema_dimensions(tmp_path):
    normalizer_dir = tmp_path / "meta"
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    RealtimePoseNormalizer(normalizer_dir, disable=True, schema_name=schema.name).save(
        torch.zeros(REALTIME_POSE_INPUT_DIM),
        torch.ones(REALTIME_POSE_INPUT_DIM),
    )
    payload = build_normalizer(
        feature_dim=REALTIME_POSE_INPUT_DIM,
        normalizer_dir=normalizer_dir,
        normalize_input=True,
        strict=True,
        schema_name=REALTIME_POSE_SCHEMA_NAME,
    )
    assert payload["enabled"] is True
    assert payload["featureDim"] == REALTIME_POSE_INPUT_DIM
    assert payload["poseRepresentation"] == schema.pose_representation
    assert payload["rootYPolicy"] == schema.root_y_policy
    assert payload["pelvisHeightMode"] == schema.pelvis_height_mode
    assert len(payload["mean"]) == REALTIME_POSE_INPUT_DIM


def test_realtime_normalizer_rejects_missing_pose_metadata(tmp_path):
    normalizer_dir = tmp_path / "old_meta"
    normalizer_dir.mkdir()
    torch.save(torch.zeros(REALTIME_POSE_INPUT_DIM), normalizer_dir / "mean.pt")
    torch.save(torch.ones(REALTIME_POSE_INPUT_DIM), normalizer_dir / "std.pt")
    with pytest.raises(FileNotFoundError, match="normalizer_meta.json"):
        build_normalizer(
            feature_dim=REALTIME_POSE_INPUT_DIM,
            normalizer_dir=normalizer_dir,
            normalize_input=True,
            strict=True,
            schema_name=REALTIME_POSE_SCHEMA_NAME,
        )


def test_realtime_normalizer_rejects_missing_root_y_policy_metadata(tmp_path):
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    normalizer_dir = tmp_path / "old_meta"
    write_legacy_policyless_normalizer(normalizer_dir, schema)

    with pytest.raises(ValueError, match="root_y_policy"):
        RealtimePoseNormalizer(normalizer_dir, schema_name=schema.name)


def test_runtime_export_rejects_policyless_normalizer(tmp_path):
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    normalizer_dir = tmp_path / "old_meta"
    write_legacy_policyless_normalizer(normalizer_dir, schema)

    with pytest.raises(ValueError, match="root_y_policy"):
        write_runtime_assets(
            output_dir=tmp_path / "assets",
            normalize_input=True,
            normalizer_dir=normalizer_dir,
            schema_name=schema.name,
        )


def test_normalized_runtime_asset_export_requires_normalizer_dir():
    with pytest.raises(FileNotFoundError):
        build_normalizer(
            feature_dim=REALTIME_POSE_INPUT_DIM,
            normalizer_dir=None,
            normalize_input=True,
            strict=False,
            schema_name=REALTIME_POSE_SCHEMA_NAME,
        )


def test_runtime_assets_default_to_body_fbx_local(tmp_path):
    schema_spec = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    assets = write_runtime_assets(output_dir=tmp_path, normalize_input=False)
    schema = json.loads(assets["feature_schema"].read_text(encoding="utf-8"))
    assert schema["schemaName"] == REALTIME_POSE_SCHEMA_NAME
    assert schema["poseRepresentation"] == schema_spec.pose_representation
    assert schema["featureDim"] == schema_spec.feature_dim
    assert schema["targetFeatureLength"] == schema_spec.target_dim
    assert schema["rootDeltaXZReference"]["start"] == schema_spec.root_delta_xz_start
    assert schema["pelvisHeight"]["start"] == schema_spec.root_height_start
    assert schema["stationaryProb5"]["start"] == schema_spec.stationary_prob_start


def test_runtime_assets_have_only_main_stationary_feature_channel(tmp_path):
    assets = write_runtime_assets(
        output_dir=tmp_path,
        normalize_input=False,
        schema_name=REALTIME_POSE_SCHEMA_NAME,
    )
    schema = json.loads(assets["feature_schema"].read_text(encoding="utf-8"))
    assert "stationaryProb5Output" not in schema
    assert "stationaryHeadOutputEnabled" not in schema["runtimeRules"]


def test_sentis_export_writes_unity_profile_manifest_after_runtime_assets(tmp_path):
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    model_path = tmp_path / "model000000123.pt"
    onnx_path = tmp_path / "diffusionposer_denoiser.onnx"
    model_path.write_bytes(b"checkpoint")
    onnx_path.write_bytes(b"onnx")
    runtime_assets = {
        "feature_schema": tmp_path / "feature_schema.json",
        "normalizer": tmp_path / "normalizer.json",
        "ddim_schedule": tmp_path / "ddim_schedule.json",
    }
    for path in runtime_assets.values():
        path.write_text("{}", encoding="utf-8")

    model_id = resolve_unity_model_id("", tmp_path)
    export_source = build_unity_export_source(
        model_id=model_id,
        source_checkpoint_path=model_path,
        checkpoint_args={"training_schedule_signature": "toy-signature", "step": 123},
        normalizer_dir=tmp_path / "normalizer_source",
        onnx_path=onnx_path,
        model_source="ema",
        schema=schema,
    )
    manifest = build_unity_model_package_manifest(
        model_id=model_id,
        display_name="Toy Profile",
        default_sample_steps=10,
        onnx_path=onnx_path,
        runtime_assets=runtime_assets,
        export_source=export_source,
        schema=schema,
    )

    assert model_id == tmp_path.name
    assert manifest["packageFormatVersion"] == 1
    assert manifest["profileAssetFile"] == UNITY_MODEL_PROFILE_ASSET_NAME
    assert manifest["onnxFile"] == onnx_path.name
    assert manifest["featureSchemaFile"] == "feature_schema.json"
    assert manifest["normalizerFile"] == "normalizer.json"
    assert manifest["ddimScheduleFile"] == "ddim_schedule.json"
    assert manifest["exportSourceFile"] == UNITY_EXPORT_SOURCE_NAME
    assert manifest["schemaName"] == schema.name
    assert manifest["featureDim"] == schema.feature_dim
    assert manifest["sequenceLength"] == schema.seq_len
    assert export_source["onnxSha256"] == manifest["onnxSha256"]
    assert UNITY_MODEL_PACKAGE_MANIFEST_NAME == "realtime_pose_model_package.json"


def test_unity_model_id_rejects_a_path(tmp_path):
    with pytest.raises(ValueError, match="单一目录名"):
        resolve_unity_model_id("packages/model_a", Path(tmp_path))


def test_sentis_export_rejects_normalized_checkpoint_without_normalizer():
    args = SimpleNamespace(normalize_input=True, normalizer_dir="")
    with pytest.raises(FileNotFoundError):
        validate_normalizer_export_contract(args, {"normalize_input": True})


def test_sentis_export_rejects_cli_schema_when_checkpoint_exact_schema_differs(tmp_path):
    checkpoint = tmp_path / "model000000001.pt"
    checkpoint.write_bytes(b"")
    legacy = get_schema_spec(LEGACY_SCHEMA_NAME)
    (tmp_path / "args.json").write_text(
        json.dumps(
            {
                "schema": LEGACY_SCHEMA_NAME,
                "schema_name": LEGACY_SCHEMA_NAME,
                "schema_canonical_name": legacy.canonical_name,
                "input_feats": legacy.feature_dim,
                "seq_len": legacy.seq_len,
                "max_seq_len": legacy.seq_len,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    args = make_sentis_cli_args(checkpoint, schema=CANONICAL_SCHEMA_NAME)

    with pytest.raises(ValueError, match="schema"):
        build_model_config(args)
