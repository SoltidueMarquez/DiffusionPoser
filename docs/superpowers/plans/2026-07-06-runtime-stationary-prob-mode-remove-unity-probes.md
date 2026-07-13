# Runtime Stationary Prob Mode And Unity Probe Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a clean runtime stationary-probability source switch, then remove the Unity collider/probe-based contact detection path while keeping the original GlobalPose-style stationary + height + residual-QP contact physics.

**Architecture:** Keep model stationary output and runtime stationary input separate. `RealtimePoseInferencePipeline` continues to decode model stationary values from the feature channel or stationary head; a new resolver in the realtime driver derives the runtime stationary signal from model, tracker velocity, replay reference, or constants. GlobalPose consumes only the runtime stationary signal, and its contact detection keeps the StationaryJoint flow while all Unity `PhysicsSystem` probe/collider branches are removed.

**Tech Stack:** Unity 2022.3.17f1c1, C#, Unity Sentis, PowerShell, Git.

---

## Scope

This plan does **not** delete GlobalPose contact physics. It removes only the Unity collision/probe contact observation path:

```text
Unity SphereCast / OverlapSphere / ContactProbeResult
  -> probe hit
  -> PhysicsSystem contact mode
  -> probe normal friction cone / probe surface height
```

This plan keeps:

```text
runtime stationary_prob_5
  -> stationary active mask
  -> previous contact / near floor
  -> same-height expansion
  -> potential contact
  -> residual-force QP refinement
  -> contact force / contact torque
  -> LSQR #2
```

## Repositories And Paths

- DiffusionPoser repo: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/DiffusionPoser`
- Unity project: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity`
- RealtimePose core: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Core`
- RealtimePose input: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Input`
- RealtimePose physics: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics`
- GlobalPose flow: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow`

## File Structure

- Modify `Assets/Projects/RealtimePose/Scripts/Core/RealtimePoseTypes.cs`
  - Add `RuntimeStationaryProbMode`.
  - Keep `StationarySignalSource` unchanged; it remains model-output selection only.

- Create `Assets/Projects/RealtimePose/Scripts/Core/RuntimeStationaryProb5Resolver.cs`
  - Own the mapping from model/replay/tracker/constant sources to a five-channel runtime stationary signal.
  - Keep tracker velocity thresholds and channel-to-tracker mapping in one focused class.

- Modify `Assets/Projects/RealtimePose/Scripts/Core/DiffusionPoserRealtimeDriver.cs`
  - Add inspector fields for runtime stationary mode and tracker speed thresholds.
  - Cache both model stationary and runtime stationary.
  - Expose `TryGetRuntimeStationaryProb5` for physics and logging.

- Modify `Assets/Projects/RealtimePose/Scripts/Physics/RealtimePosePhysicsBridge.cs`
  - Read runtime stationary from `DiffusionPoserRealtimeDriver`.
  - Log runtime mode and runtime values.
  - Stop treating `PosePrediction.StationaryProb5` as the only physics signal.

- Modify `Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/RealtimePoseGlobalPosePhysicsSettings.cs`
  - Remove `PhysicsSystem` from the active behavior.
  - Remove serialized probe config usage from runtime settings.

- Modify `Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/RealtimePoseGlobalPoseFrameBuilder.cs`
  - Stop allocating and running probe configs/results.
  - Continue injecting `GpVrStationaryProb5`.
  - Continue building predicted contact points from reference joints.

- Modify `Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/CarticulateGpNetPySolver.cs`
  - Remove `ProbeResults` and `ProbeConfigs` from `FrameInput`.
  - Call GlobalPose contact evaluation without probe inputs.
  - Build default Y-up friction cones.

- Modify `Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/CarticulateGpNetPyPdTracking.cs`
  - Remove `PhysicsSystem` probe branches.
  - Track contact only from `stationaryActive5`.

- Modify `Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/CarticulateGpNetPyContactQp.cs`
  - Remove probe-based contact decisions, probe normal friction, and probe surface height.
  - Keep StationaryJoint contact logic, potential-contact refinement, QP solve, and contact torque.

- Modify `Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/CarticulateGpNetPyKinematicsNative.cs`
  - Replace probe-normal cone building with default floor-normal cone building.

- Delete `Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/CarticulateGpContactProbe.cs`
  - Delete its `.meta` after Unity confirms no references remain.

---

### Task 1: Preflight And Baseline

**Files:**
- Read only: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/ProjectSettings/ProjectVersion.txt`
- Read only: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts`

- [ ] **Step 1: Confirm Unity editor version**

Run:

```powershell
Get-Content -Raw -LiteralPath 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\ProjectSettings\ProjectVersion.txt'
```

Expected:

```text
m_EditorVersion: 2022.3.17f1c1
```

- [ ] **Step 2: Confirm dirty state before code changes**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\DiffusionPoser' status --short
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' status --short
```

Expected:

```text
DiffusionPoser may contain existing unrelated untracked files.
SIGGRAPH2024Unity should show no unrelated modified C# files before implementation starts.
```

- [ ] **Step 3: Capture current probe references**

Run:

```powershell
rg "PhysicsSystem|ContactProbe|ProbeResults|ProbeConfigs|contactProbeConfigs|BuildAllFrictionCones|ApplyGroundConstraint" 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Assets\Projects\RealtimePose\Scripts\Physics'
```

Expected:

```text
Matches appear in GlobalPoseFlow and physics settings. Use this output as the removal checklist for Tasks 4-6.
```

- [ ] **Step 4: Compile before changes**

Run:

```powershell
$UnityExe = @(
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1c1\Editor\Unity.exe',
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1\Editor\Unity.exe',
  'C:\Program Files\Unity 2022.3.17f1c1\Editor\Unity.exe'
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $UnityExe) { throw 'Unity 2022.3.17f1c1 executable was not found in the known install locations.' }
& $UnityExe -batchmode -quit -projectPath 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' -logFile 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-stationary-probe-preflight.log'
if ($LASTEXITCODE -ne 0) { Get-Content -Tail 160 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-stationary-probe-preflight.log'; exit $LASTEXITCODE }
```

Expected:

```text
Exit code 0. If Unity is not installed at those paths, stop and ask for the actual Unity.exe path.
```

- [ ] **Step 5: Commit preflight-only state if a branch is being used**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' status --short
```

Expected:

```text
No code changes from this task.
```

---

### Task 2: Add Runtime Stationary Mode Types

**Files:**
- Modify: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Core/RealtimePoseTypes.cs`

- [ ] **Step 1: Add the enum next to StationarySignalSource**

Add this code after `StationarySignalSource`:

```csharp
    public enum RuntimeStationaryProbMode
    {
        ModelPrediction,
        TrackerVelocity,
        TrackerVelocityWithModelFallback,
        ReplayReference,
        ConstantZero,
        ConstantOne
    }
```

- [ ] **Step 2: Add comments that separate model source from runtime source**

Replace the existing `StationarySignalSource` block with this commented version:

```csharp
    /// <summary>
    /// Selects where the model prediction reads stationary_prob_5 from.
    /// Runtime overrides such as tracker velocity are handled by RuntimeStationaryProbMode.
    /// </summary>
    public enum StationarySignalSource
    {
        Auto,
        FeatureChannel,
        StationaryHead
    }

    /// <summary>
    /// Selects the final stationary_prob_5 consumed by realtime logs and GlobalPose contact physics.
    /// This does not change the model output tensor or the schema contract.
    /// </summary>
    public enum RuntimeStationaryProbMode
    {
        ModelPrediction,
        TrackerVelocity,
        TrackerVelocityWithModelFallback,
        ReplayReference,
        ConstantZero,
        ConstantOne
    }
```

- [ ] **Step 3: Compile**

Run:

```powershell
$UnityExe = @(
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1c1\Editor\Unity.exe',
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1\Editor\Unity.exe',
  'C:\Program Files\Unity 2022.3.17f1c1\Editor\Unity.exe'
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $UnityExe) { throw 'Unity executable not found.' }
& $UnityExe -batchmode -quit -projectPath 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' -logFile 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-stationary-mode-types.log'
if ($LASTEXITCODE -ne 0) { Get-Content -Tail 160 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-stationary-mode-types.log'; exit $LASTEXITCODE }
```

Expected:

```text
Exit code 0.
```

- [ ] **Step 4: Commit**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' add 'Assets/Projects/RealtimePose/Scripts/Core/RealtimePoseTypes.cs'
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' commit -m 'feat(realtimepose): add runtime stationary mode enum'
```

Expected:

```text
Commit succeeds with only RealtimePoseTypes.cs staged.
```

---

### Task 3: Create RuntimeStationaryProb5Resolver

**Files:**
- Create: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Core/RuntimeStationaryProb5Resolver.cs`

- [ ] **Step 1: Create resolver file**

Create `RuntimeStationaryProb5Resolver.cs` with this implementation:

```csharp
using System;
using UnityEngine;

namespace RealtimePose
{
    internal sealed class RuntimeStationaryProb5Resolver
    {
        private static readonly TrackerSensor[] StationarySensors =
        {
            TrackerSensor.Waist,
            TrackerSensor.LeftFoot,
            TrackerSensor.RightFoot,
            TrackerSensor.LeftWrist,
            TrackerSensor.RightWrist
        };

        public float StaticSpeed = 0.03f;
        public float MovingSpeed = 0.25f;

        public bool Resolve(
            RuntimeStationaryProbMode mode,
            in TrackerFrame frame,
            PosePrediction modelPrediction,
            IReplayFrameSource replaySource,
            int replayFrameIndex,
            float[] output,
            out RuntimeStationaryProbMode modeUsed)
        {
            modeUsed = mode;
            if (output == null || output.Length < PoseFeatureSchema.StationaryProbDim)
            {
                return false;
            }

            switch (mode)
            {
                case RuntimeStationaryProbMode.ModelPrediction:
                    return CopyModel(modelPrediction, output);
                case RuntimeStationaryProbMode.TrackerVelocity:
                    FillTrackerVelocity(frame, output, fallbackModel: null);
                    return true;
                case RuntimeStationaryProbMode.TrackerVelocityWithModelFallback:
                    FillTrackerVelocity(frame, output, modelPrediction);
                    return true;
                case RuntimeStationaryProbMode.ReplayReference:
                    if (replaySource != null && replayFrameIndex >= 0 && replaySource.TryGetStationaryProb5(replayFrameIndex, output))
                    {
                        ClampOutput(output);
                        return true;
                    }
                    modeUsed = RuntimeStationaryProbMode.ModelPrediction;
                    return CopyModel(modelPrediction, output);
                case RuntimeStationaryProbMode.ConstantZero:
                    Array.Clear(output, 0, PoseFeatureSchema.StationaryProbDim);
                    return true;
                case RuntimeStationaryProbMode.ConstantOne:
                    for (int i = 0; i < PoseFeatureSchema.StationaryProbDim; i++)
                    {
                        output[i] = 1f;
                    }
                    return true;
                default:
                    modeUsed = RuntimeStationaryProbMode.ModelPrediction;
                    return CopyModel(modelPrediction, output);
            }
        }

        private static bool CopyModel(PosePrediction prediction, float[] output)
        {
            if (prediction == null ||
                prediction.StationaryProb5 == null ||
                prediction.StationaryProb5.Length < PoseFeatureSchema.StationaryProbDim)
            {
                return false;
            }

            for (int i = 0; i < PoseFeatureSchema.StationaryProbDim; i++)
            {
                output[i] = Mathf.Clamp01(prediction.StationaryProb5[i]);
            }

            return true;
        }

        private void FillTrackerVelocity(in TrackerFrame frame, float[] output, PosePrediction fallbackModel)
        {
            for (int i = 0; i < PoseFeatureSchema.StationaryProbDim; i++)
            {
                TrackerSample sample = frame.Get(StationarySensors[i]);
                if (sample.IsValid)
                {
                    output[i] = SpeedToProbability(sample.Velocity.magnitude);
                }
                else if (fallbackModel != null &&
                         fallbackModel.StationaryProb5 != null &&
                         fallbackModel.StationaryProb5.Length > i)
                {
                    output[i] = Mathf.Clamp01(fallbackModel.StationaryProb5[i]);
                }
                else
                {
                    output[i] = 0f;
                }
            }
        }

        private float SpeedToProbability(float speed)
        {
            float lo = Mathf.Max(0f, StaticSpeed);
            float hi = Mathf.Max(lo + 1e-5f, MovingSpeed);
            return 1f - Mathf.Clamp01((Mathf.Max(0f, speed) - lo) / (hi - lo));
        }

        private static void ClampOutput(float[] output)
        {
            for (int i = 0; i < PoseFeatureSchema.StationaryProbDim; i++)
            {
                output[i] = Mathf.Clamp01(output[i]);
            }
        }
    }
}
```

- [ ] **Step 2: Compile**

Run:

```powershell
$UnityExe = @(
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1c1\Editor\Unity.exe',
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1\Editor\Unity.exe',
  'C:\Program Files\Unity 2022.3.17f1c1\Editor\Unity.exe'
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $UnityExe) { throw 'Unity executable not found.' }
& $UnityExe -batchmode -quit -projectPath 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' -logFile 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-stationary-resolver.log'
if ($LASTEXITCODE -ne 0) { Get-Content -Tail 180 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-stationary-resolver.log'; exit $LASTEXITCODE }
```

Expected:

```text
Exit code 0.
```

- [ ] **Step 3: Commit**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' add 'Assets/Projects/RealtimePose/Scripts/Core/RuntimeStationaryProb5Resolver.cs' 'Assets/Projects/RealtimePose/Scripts/Core/RuntimeStationaryProb5Resolver.cs.meta'
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' commit -m 'feat(realtimepose): resolve runtime stationary probability'
```

Expected:

```text
Commit includes the new resolver file and Unity-generated .meta file.
```

---

### Task 4: Integrate Runtime Stationary Into DiffusionPoserRealtimeDriver

**Files:**
- Modify: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Core/DiffusionPoserRealtimeDriver.cs`

- [ ] **Step 1: Add inspector fields near StationarySignalSource**

Add these fields after the existing `StationarySignalSource` field:

```csharp
        [Header("Runtime Stationary")]
        [Tooltip("Final stationary_prob_5 source used by logs and GlobalPose contact physics.")]
        public RuntimeStationaryProbMode RuntimeStationaryProbMode = RuntimeStationaryProbMode.ModelPrediction;
        [Min(0f)]
        public float RuntimeStationaryStaticSpeed = 0.03f;
        [Min(1e-5f)]
        public float RuntimeStationaryMovingSpeed = 0.25f;
```

- [ ] **Step 2: Add buffers and resolver fields**

Add these private fields next to `latestStationaryProb5`:

```csharp
        private readonly float[] latestModelStationaryProb5 = new float[PoseFeatureSchema.StationaryProbDim];
        private readonly float[] latestRuntimeStationaryProb5 = new float[PoseFeatureSchema.StationaryProbDim];
        private readonly RuntimeStationaryProb5Resolver runtimeStationaryResolver = new RuntimeStationaryProb5Resolver();
        private bool hasLatestRuntimeStationaryProb5;
        private RuntimeStationaryProbMode latestRuntimeStationaryProbModeUsed = RuntimeStationaryProbMode.ModelPrediction;
```

Keep `latestStationaryProb5` during this task as a compatibility alias. Remove it only if all consumers are updated and compile cleanly.

- [ ] **Step 3: Add public accessors**

Add these properties/methods near `LatestStationaryProb5`:

```csharp
        public RuntimeStationaryProbMode LatestRuntimeStationaryProbModeUsed
        {
            get { return latestRuntimeStationaryProbModeUsed; }
        }

        public bool TryGetRuntimeStationaryProb5(float[] output)
        {
            if (!hasLatestRuntimeStationaryProb5 ||
                output == null ||
                output.Length < PoseFeatureSchema.StationaryProbDim)
            {
                return false;
            }

            Array.Copy(latestRuntimeStationaryProb5, output, PoseFeatureSchema.StationaryProbDim);
            return true;
        }

        public float[] LatestRuntimeStationaryProb5
        {
            get
            {
                float[] copy = new float[latestRuntimeStationaryProb5.Length];
                Array.Copy(latestRuntimeStationaryProb5, copy, latestRuntimeStationaryProb5.Length);
                return copy;
            }
        }
```

- [ ] **Step 4: Resolve runtime stationary after inference**

In `RunInferenceFrame`, replace the current `UpdateLatestStationaryProb5(result.Prediction);` line with:

```csharp
            UpdateLatestStationaryProb5(result.Prediction);
            ResolveRuntimeStationaryProb5(frame, replayFrameIndex, result.Prediction);
```

Add this method near `UpdateLatestStationaryProb5`:

```csharp
        private void ResolveRuntimeStationaryProb5(TrackerFrame frame, int replayFrameIndex, PosePrediction prediction)
        {
            runtimeStationaryResolver.StaticSpeed = RuntimeStationaryStaticSpeed;
            runtimeStationaryResolver.MovingSpeed = Mathf.Max(RuntimeStationaryStaticSpeed + 1e-5f, RuntimeStationaryMovingSpeed);

            IReplayFrameSource replaySource = activeTrackerSource as IReplayFrameSource;
            if (runtimeStationaryResolver.Resolve(
                    RuntimeStationaryProbMode,
                    frame,
                    prediction,
                    replaySource,
                    replayFrameIndex,
                    latestRuntimeStationaryProb5,
                    out latestRuntimeStationaryProbModeUsed))
            {
                hasLatestRuntimeStationaryProb5 = true;
            }
            else
            {
                Array.Clear(latestRuntimeStationaryProb5, 0, latestRuntimeStationaryProb5.Length);
                hasLatestRuntimeStationaryProb5 = false;
                latestRuntimeStationaryProbModeUsed = RuntimeStationaryProbMode.ModelPrediction;
            }
        }
```

- [ ] **Step 5: Keep model buffer explicitly named**

Update `UpdateLatestStationaryProb5` so it writes both the compatibility buffer and the model buffer:

```csharp
        private void UpdateLatestStationaryProb5(PosePrediction prediction)
        {
            Array.Clear(latestStationaryProb5, 0, latestStationaryProb5.Length);
            Array.Clear(latestModelStationaryProb5, 0, latestModelStationaryProb5.Length);
            if (prediction == null || prediction.StationaryProb5 == null)
            {
                return;
            }

            int count = Mathf.Min(latestStationaryProb5.Length, prediction.StationaryProb5.Length);
            Array.Copy(prediction.StationaryProb5, latestStationaryProb5, count);
            Array.Copy(prediction.StationaryProb5, latestModelStationaryProb5, count);
        }
```

- [ ] **Step 6: Compile**

Run:

```powershell
$UnityExe = @(
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1c1\Editor\Unity.exe',
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1\Editor\Unity.exe',
  'C:\Program Files\Unity 2022.3.17f1c1\Editor\Unity.exe'
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $UnityExe) { throw 'Unity executable not found.' }
& $UnityExe -batchmode -quit -projectPath 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' -logFile 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-driver-runtime-stationary.log'
if ($LASTEXITCODE -ne 0) { Get-Content -Tail 180 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-driver-runtime-stationary.log'; exit $LASTEXITCODE }
```

Expected:

```text
Exit code 0.
```

- [ ] **Step 7: Commit**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' add 'Assets/Projects/RealtimePose/Scripts/Core/DiffusionPoserRealtimeDriver.cs'
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' commit -m 'feat(realtimepose): cache runtime stationary signal'
```

Expected:

```text
Commit succeeds with only DiffusionPoserRealtimeDriver.cs staged.
```

---

### Task 5: Make PhysicsBridge Consume Runtime Stationary

**Files:**
- Modify: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/RealtimePosePhysicsBridge.cs`

- [ ] **Step 1: Add source tracking field**

Add this private field next to `hasStationaryProb5`:

```csharp
        private RuntimeStationaryProbMode latestRuntimeStationaryMode = RuntimeStationaryProbMode.ModelPrediction;
```

- [ ] **Step 2: Replace prediction cache with driver runtime cache**

Replace `CacheStationaryProb5(PosePrediction prediction)` with:

```csharp
        private void CacheStationaryProb5(RealtimePoseInferenceResult result)
        {
            if (InferenceDriver != null && InferenceDriver.TryGetRuntimeStationaryProb5(latestStationaryProb5))
            {
                latestRuntimeStationaryMode = InferenceDriver.LatestRuntimeStationaryProbModeUsed;
                hasStationaryProb5 = true;
                return;
            }

            PosePrediction prediction = result.Prediction;
            if (prediction == null ||
                prediction.StationaryProb5 == null ||
                prediction.StationaryProb5.Length < RealtimePosePhysicsSkeleton.StationaryProbDim)
            {
                hasStationaryProb5 = false;
                return;
            }

            for (int i = 0; i < RealtimePosePhysicsSkeleton.StationaryProbDim; i++)
            {
                latestStationaryProb5[i] = Mathf.Clamp01(prediction.StationaryProb5[i]);
            }

            latestRuntimeStationaryMode = RuntimeStationaryProbMode.ModelPrediction;
            hasStationaryProb5 = true;
        }
```

- [ ] **Step 3: Update callers**

In `OnInferenceCompleted`, replace:

```csharp
            CacheStationaryProb5(result.Prediction);
```

with:

```csharp
            CacheStationaryProb5(result);
```

In `TryGetStationaryProb5`, replace:

```csharp
                CacheStationaryProb5(InferenceDriver.LastInferenceResult.Prediction);
```

with:

```csharp
                CacheStationaryProb5(InferenceDriver.LastInferenceResult);
```

- [ ] **Step 4: Include runtime mode in stationary log**

In `LogStationaryFrame`, after appending `frame`, add:

```csharp
            sb.Append(" source=");
            sb.Append(latestRuntimeStationaryMode);
```

The resulting log prefix should read like:

```text
[RealtimePosePhysics][Stationary] frame=60 source=TrackerVelocity applied=True ...
```

- [ ] **Step 5: Compile**

Run:

```powershell
$UnityExe = @(
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1c1\Editor\Unity.exe',
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1\Editor\Unity.exe',
  'C:\Program Files\Unity 2022.3.17f1c1\Editor\Unity.exe'
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $UnityExe) { throw 'Unity executable not found.' }
& $UnityExe -batchmode -quit -projectPath 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' -logFile 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-physicsbridge-runtime-stationary.log'
if ($LASTEXITCODE -ne 0) { Get-Content -Tail 180 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-physicsbridge-runtime-stationary.log'; exit $LASTEXITCODE }
```

Expected:

```text
Exit code 0.
```

- [ ] **Step 6: Commit**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' add 'Assets/Projects/RealtimePose/Scripts/Physics/RealtimePosePhysicsBridge.cs'
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' commit -m 'feat(realtimepose): feed physics runtime stationary signal'
```

Expected:

```text
Commit succeeds with only RealtimePosePhysicsBridge.cs staged.
```

---

### Task 6: Remove Unity Probe Inputs From GlobalPose Frame Building

**Files:**
- Modify: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/RealtimePoseGlobalPoseFrameBuilder.cs`
- Modify: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/RealtimePoseGlobalPosePhysicsSettings.cs`

- [ ] **Step 1: Remove probe fields from frame builder**

In `RealtimePoseGlobalPoseFrameBuilder`, remove these fields:

```csharp
        private readonly ContactProbeResult[] probeResults;
        private readonly ContactProbeConfig[] probeConfigs;

        public ContactProbeResult[] ProbeResults => probeResults;
        public ContactProbeConfig[] ProbeConfigs => probeConfigs;
```

Remove constructor initialization:

```csharp
            probeConfigs = settingsAsset != null ? settingsAsset.GetProbeConfigs() : ContactProbeConfig.Default5();
            probeResults = new ContactProbeResult[solver.contactJoints.Length];
```

- [ ] **Step 2: Remove probe runtime branch**

In `BuildRuntimeFrameInput`, replace the block from `if (probeConfigs != null)` through `frame.ProbeResults = null;` with no code. The method should keep only:

```csharp
            CarticulateGpNetPySolver.FrameInput frame = BuildNetPyVrFrameInput(
                input.ReferenceRoot,
                input.ReferenceSmplJoints,
                input.ReferenceActor);
            InjectStationaryProb5(ref frame);

            if (solver != null && input.ReferenceSmplJoints != null)
            {
                frame.GpPredJointWorld15 = BuildPredictedContactPointsFromReferenceJoints(
                    input.ReferenceRoot,
                    input.ReferenceSmplJoints,
                    context.StateRoot,
                    snapshot,
                    settings.ContactMode);
            }

            return frame;
```

- [ ] **Step 3: Remove probe settings fields**

In `RealtimePoseGlobalPosePhysicsSettings`, remove:

```csharp
        [Header("Contact probes (Unity collision)")]
        [Tooltip("Per-contact-joint probe configs for PhysicsSystem SphereCast/OverlapSphere; StationaryJoint uses them only as static grasp metadata.")]
        public ContactProbeConfig[] contactProbeConfigs;

        public ContactProbeConfig[] GetProbeConfigs()
        {
            return contactProbeConfigs != null && contactProbeConfigs.Length >= 5
                ? contactProbeConfigs
                : ContactProbeConfig.Default5();
        }
```

- [ ] **Step 4: Keep contact mode but make PhysicsSystem unavailable at runtime**

Replace the enum with:

```csharp
    public enum RealtimePoseGlobalPoseContactMode
    {
        StationaryJoint = 0
    }
```

In `ToSolverSettings`, keep:

```csharp
                ContactMode = RealtimePoseGlobalPoseContactMode.StationaryJoint,
```

This deliberately ignores any old serialized integer value. The runtime now always uses GlobalPose stationary-joint contact logic.

- [ ] **Step 5: Compile**

Run:

```powershell
$UnityExe = @(
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1c1\Editor\Unity.exe',
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1\Editor\Unity.exe',
  'C:\Program Files\Unity 2022.3.17f1c1\Editor\Unity.exe'
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $UnityExe) { throw 'Unity executable not found.' }
& $UnityExe -batchmode -quit -projectPath 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' -logFile 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-remove-frame-probes.log'
if ($LASTEXITCODE -ne 0) { Get-Content -Tail 220 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-remove-frame-probes.log'; exit $LASTEXITCODE }
```

Expected:

```text
Exit code 0.
```

- [ ] **Step 6: Commit**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' add 'Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/RealtimePoseGlobalPoseFrameBuilder.cs' 'Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/RealtimePoseGlobalPosePhysicsSettings.cs'
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' commit -m 'refactor(globalpose): remove Unity probe frame inputs'
```

Expected:

```text
Commit succeeds with frame builder and settings only.
```

---

### Task 7: Remove Probe Branches From GlobalPose Solver

**Files:**
- Modify: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/CarticulateGpNetPySolver.cs`
- Modify: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/CarticulateGpNetPyPdTracking.cs`
- Modify: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/CarticulateGpNetPyContactQp.cs`
- Modify: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/CarticulateGpNetPyKinematicsNative.cs`

- [ ] **Step 1: Remove probe fields from FrameInput**

In `CarticulateGpNetPySolver.FrameInput`, remove:

```csharp
            public ContactProbeResult[] ProbeResults;
            public ContactProbeConfig[] ProbeConfigs;
```

- [ ] **Step 2: Build default friction cones**

In `CarticulateGpNetPySolver`, replace:

```csharp
            CarticulateGpNetPyKinematicsNative.BuildAllFrictionCones(this, settings.Mu, frame.ProbeResults);
```

with:

```csharp
            CarticulateGpNetPyKinematicsNative.BuildAllFrictionCones(this, settings.Mu);
```

In `CarticulateGpNetPyKinematicsNative`, replace `BuildAllFrictionCones` with:

```csharp
        internal static void BuildAllFrictionCones(CarticulateGpNetPySolver s, float mu)
        {
            for (int i = 0; i < s.nContact; i++)
            {
                BuildFrictionConeForNormal(Vector3.up, mu, s.frictionBPerContact, i * 12);
            }
        }
```

- [ ] **Step 3: Remove probe parameters from contact evaluation calls**

In `CarticulateGpNetPySolver`, replace:

```csharp
            CarticulateGpNetPyContactQp.EvaluateContacts(
                this, handle, segStationary, frame.ProbeResults, frame.ProbeConfigs,
                settings, contact, potential, contactConfidence);
```

with:

```csharp
            CarticulateGpNetPyContactQp.EvaluateContacts(
                this, handle, segStationary, settings, contact, potential, contactConfidence);
```

Replace:

```csharp
            float qpErr = CarticulateGpNetPyContactQp.SolveConeQp(this, settings, contact, residual6, frame.ProbeConfigs, out nVar);
            CarticulateGpNetPyContactQp.RunPotentialContactRefinement(this, settings, ref qpErr, contact, potential, residual6, frame.ProbeConfigs);
```

with:

```csharp
            float qpErr = CarticulateGpNetPyContactQp.SolveConeQp(this, settings, contact, residual6, out nVar);
            CarticulateGpNetPyContactQp.RunPotentialContactRefinement(this, settings, ref qpErr, contact, potential, residual6);
```

Replace:

```csharp
            CarticulateGpNetPyContactQp.ApplyGroundConstraint(this, settings, contact, frame.ProbeResults);
```

with:

```csharp
            CarticulateGpNetPyContactQp.ApplyGroundConstraint(this, settings, contact);
```

- [ ] **Step 4: Simplify ShouldTrackContact**

In `CarticulateGpNetPyPdTracking`, replace `ShouldTrackContact` with:

```csharp
        private static bool ShouldTrackContact(
            CarticulateGpNetPySolver s,
            int i)
        {
            return i < s.stationaryActive5.Length && s.stationaryActive5[i];
        }
```

Update all call sites:

```csharp
                if (!ShouldTrackContact(s, i))
```

Remove the tangent-only ground projection block that references `frame.ProbeConfigs`, `frame.ProbeResults`, and `probeHit` in `BuildRddotDes`.

- [ ] **Step 5: Replace EvaluateContacts with StationaryJoint-only logic**

In `CarticulateGpNetPyContactQp`, replace the `EvaluateContacts` signature and body with:

```csharp
        internal static void EvaluateContacts(
            CarticulateGpNetPySolver s,
            long handle,
            bool[] stationary,
            in CarticulateGpNetPySolver.Settings settings,
            bool[] outContact,
            bool[] outPotential,
            float[] outConfidence)
        {
            for (int i = 0; i < s.nContact; i++)
            {
                bool active = stationary != null && i < stationary.Length && stationary[i];
                bool previousContact = s.prevContactMask != null && i < s.prevContactMask.Length && s.prevContactMask[i];
                bool nearFloor = s.cjointCur[i].y < settings.FloorY + 0.05f;
                outContact[i] = active && (previousContact || nearFloor);
                outPotential[i] = false;
                outConfidence[i] = outContact[i] && i < s.stationaryWeight5.Length ? s.stationaryWeight5[i] : 0f;
            }

            ExpandContactByPairHeight(s, outContact, stationary);
            for (int i = 0; i < s.nContact; i++)
            {
                bool active = stationary != null && i < stationary.Length && stationary[i];
                outPotential[i] = active && !outContact[i];
                outConfidence[i] = outContact[i] && i < s.stationaryWeight5.Length ? s.stationaryWeight5[i] : 0f;
            }

            ApplyLegFilter(s, handle, outContact, outPotential, settings);
        }
```

- [ ] **Step 6: Remove probe config from QP helpers**

In `CarticulateGpNetPyContactQp`, update signatures:

```csharp
        internal static void RunPotentialContactRefinement(
            CarticulateGpNetPySolver s,
            in CarticulateGpNetPySolver.Settings settings,
            ref float err,
            bool[] contact,
            bool[] potential,
            float[] res6)
```

```csharp
        internal static float SolveConeQp(
            CarticulateGpNetPySolver s,
            in CarticulateGpNetPySolver.Settings settings,
            bool[] contact,
            float[] res6,
            out int nVar)
```

Inside both methods, remove `probeConfigs` from calls.

- [ ] **Step 7: Make grasp heuristic symmetric for both hands**

Replace `IsGrasp` with:

```csharp
        private static bool IsGrasp(
            int i,
            CarticulateGpNetPySolver s,
            in CarticulateGpNetPySolver.Settings settings)
        {
            return i >= 3 && s.cjointCur[i].y > settings.FloorY + 0.15f;
        }
```

Update callers:

```csharp
                nVar += IsGrasp(i, s, settings) ? 3 : 4;
```

and:

```csharp
                bool grasp = IsGrasp(i, s, settings);
```

This intentionally keeps both hands able to use the high-contact grasp force basis after probe configs are removed.

- [ ] **Step 8: Make ground constraint use floor only**

Replace `ApplyGroundConstraint` with:

```csharp
        internal static void ApplyGroundConstraint(
            CarticulateGpNetPySolver s,
            in CarticulateGpNetPySolver.Settings settings,
            bool[] contact)
        {
            for (int i = 0; i < s.nContact; i++)
            {
                float y = s.cjointPred[i * 3 + 1];
                float surfaceY = settings.FloorY;

                if (contact[i] && y < surfaceY + 0.15f)
                {
                    s.cjointPred[i * 3 + 1] = Mathf.Lerp(y, surfaceY, 0.1f);
                }

                if (s.cjointPred[i * 3 + 1] < surfaceY)
                {
                    s.cjointPred[i * 3 + 1] = surfaceY;
                }
            }
        }
```

- [ ] **Step 9: Compile**

Run:

```powershell
$UnityExe = @(
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1c1\Editor\Unity.exe',
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1\Editor\Unity.exe',
  'C:\Program Files\Unity 2022.3.17f1c1\Editor\Unity.exe'
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $UnityExe) { throw 'Unity executable not found.' }
& $UnityExe -batchmode -quit -projectPath 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' -logFile 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-remove-solver-probes.log'
if ($LASTEXITCODE -ne 0) { Get-Content -Tail 260 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-remove-solver-probes.log'; exit $LASTEXITCODE }
```

Expected:

```text
Exit code 0.
```

- [ ] **Step 10: Commit**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' add `
  'Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/CarticulateGpNetPySolver.cs' `
  'Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/CarticulateGpNetPyPdTracking.cs' `
  'Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/CarticulateGpNetPyContactQp.cs' `
  'Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/CarticulateGpNetPyKinematicsNative.cs'
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' commit -m 'refactor(globalpose): use stationary-only contact detection'
```

Expected:

```text
Commit succeeds with the four solver files staged.
```

---

### Task 8: Delete Unity Contact Probe Code

**Files:**
- Delete: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/CarticulateGpContactProbe.cs`
- Delete: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/CarticulateGpContactProbe.cs.meta`

- [ ] **Step 1: Verify no C# references remain**

Run:

```powershell
rg "CarticulateGpContactProbe|ContactProbeConfig|ContactProbeResult|ProbeResults|ProbeConfigs|contactProbeConfigs|PhysicsSystem" 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Assets\Projects\RealtimePose\Scripts'
```

Expected:

```text
No matches. If matches remain, finish Task 6 or Task 7 before deleting the file.
```

- [ ] **Step 2: Delete the probe file and meta**

Run:

```powershell
Remove-Item -LiteralPath 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Assets\Projects\RealtimePose\Scripts\Physics\GlobalPoseFlow\CarticulateGpContactProbe.cs'
Remove-Item -LiteralPath 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Assets\Projects\RealtimePose\Scripts\Physics\GlobalPoseFlow\CarticulateGpContactProbe.cs.meta'
```

Expected:

```text
Both files are removed.
```

- [ ] **Step 3: Compile**

Run:

```powershell
$UnityExe = @(
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1c1\Editor\Unity.exe',
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1\Editor\Unity.exe',
  'C:\Program Files\Unity 2022.3.17f1c1\Editor\Unity.exe'
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $UnityExe) { throw 'Unity executable not found.' }
& $UnityExe -batchmode -quit -projectPath 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' -logFile 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-delete-contact-probe.log'
if ($LASTEXITCODE -ne 0) { Get-Content -Tail 260 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-delete-contact-probe.log'; exit $LASTEXITCODE }
```

Expected:

```text
Exit code 0.
```

- [ ] **Step 4: Commit**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' add -A 'Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/CarticulateGpContactProbe.cs' 'Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/CarticulateGpContactProbe.cs.meta'
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' commit -m 'refactor(globalpose): delete Unity contact probe path'
```

Expected:

```text
Commit records the two file deletions.
```

---

### Task 9: Add Stationary Mode Debug Coverage

**Files:**
- Modify: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Debug/RealtimePoseReplayEvaluator.cs`
- Modify: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Editor/RealtimePoseAutomatedReplayTestRunner.cs`

- [ ] **Step 1: Record runtime mode in replay evaluator output**

In `RealtimePoseReplayEvaluator`, add a `runtimeStationaryProbModeUsed` string to the report data structure next to `stationarySignalSourceUsed`:

```csharp
            public string stationarySignalSourceUsed;
            public string runtimeStationaryProbModeUsed;
```

When building the report, set:

```csharp
                runtimeStationaryProbModeUsed = Driver != null
                    ? Driver.LatestRuntimeStationaryProbModeUsed.ToString()
                    : "unknown",
```

- [ ] **Step 2: Add runner cases for model, tracker, replay, zero, and one**

In `RealtimePoseAutomatedReplayTestRunner`, add run variants that set:

```csharp
driver.RuntimeStationaryProbMode = RuntimeStationaryProbMode.ModelPrediction;
driver.RuntimeStationaryProbMode = RuntimeStationaryProbMode.TrackerVelocity;
driver.RuntimeStationaryProbMode = RuntimeStationaryProbMode.ReplayReference;
driver.RuntimeStationaryProbMode = RuntimeStationaryProbMode.ConstantZero;
driver.RuntimeStationaryProbMode = RuntimeStationaryProbMode.ConstantOne;
```

Keep existing `StationarySignalSource.FeatureChannel` and `StationarySignalSource.StationaryHead` cases separate. This verifies model output source and runtime override source independently.

- [ ] **Step 3: Compile**

Run:

```powershell
$UnityExe = @(
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1c1\Editor\Unity.exe',
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1\Editor\Unity.exe',
  'C:\Program Files\Unity 2022.3.17f1c1\Editor\Unity.exe'
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $UnityExe) { throw 'Unity executable not found.' }
& $UnityExe -batchmode -quit -projectPath 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' -logFile 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-stationary-debug-coverage.log'
if ($LASTEXITCODE -ne 0) { Get-Content -Tail 220 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-stationary-debug-coverage.log'; exit $LASTEXITCODE }
```

Expected:

```text
Exit code 0.
```

- [ ] **Step 4: Commit**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' add 'Assets/Projects/RealtimePose/Scripts/Debug/RealtimePoseReplayEvaluator.cs' 'Assets/Projects/RealtimePose/Scripts/Editor/RealtimePoseAutomatedReplayTestRunner.cs'
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' commit -m 'test(realtimepose): cover runtime stationary modes'
```

Expected:

```text
Commit succeeds with evaluator and runner updates.
```

---

### Task 10: Final Verification

**Files:**
- Read only: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Logs`
- Read only: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts`

- [ ] **Step 1: Confirm Unity probe path is gone**

Run:

```powershell
rg "CarticulateGpContactProbe|ContactProbeConfig|ContactProbeResult|ProbeResults|ProbeConfigs|contactProbeConfigs|PhysicsSystem|SphereCast|OverlapSphere" 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Assets\Projects\RealtimePose\Scripts\Physics'
```

Expected:

```text
No matches in RealtimePose physics code.
```

- [ ] **Step 2: Confirm GlobalPose stationary contact path remains**

Run:

```powershell
rg "EvaluateContacts|RunPotentialContactRefinement|SolveConeQp|ComputeTorqueFromContacts|StationaryContactThreshold|stationaryActive5" 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Assets\Projects\RealtimePose\Scripts\Physics\GlobalPoseFlow'
```

Expected:

```text
Matches remain in CarticulateGpNetPySolver.cs, CarticulateGpNetPyContactQp.cs, CarticulateGpNetPyPdTracking.cs, and RealtimePoseGlobalPosePhysicsSettings.cs.
```

- [ ] **Step 3: Compile final Unity project**

Run:

```powershell
$UnityExe = @(
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1c1\Editor\Unity.exe',
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1\Editor\Unity.exe',
  'C:\Program Files\Unity 2022.3.17f1c1\Editor\Unity.exe'
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $UnityExe) { throw 'Unity executable not found.' }
& $UnityExe -batchmode -quit -projectPath 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' -logFile 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-final-stationary-probe-removal.log'
if ($LASTEXITCODE -ne 0) { Get-Content -Tail 260 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-final-stationary-probe-removal.log'; exit $LASTEXITCODE }
```

Expected:

```text
Exit code 0.
```

- [ ] **Step 4: Run DiffusionPoser smoke tests for exported/runtime contract**

Run:

```powershell
conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/export tests/smoke/schemas
```

Expected:

```text
Tests pass. These tests should remain unchanged because the schema contract is not changed.
```

- [ ] **Step 5: Manual scene checks**

Open `Assets/Projects/RealtimePose/Scenes/RealtimePose_DiffusionPoser_Test.unity` in Unity and check these modes:

```text
RuntimeStationaryProbMode = ModelPrediction
RuntimeStationaryProbMode = TrackerVelocity
RuntimeStationaryProbMode = ReplayReference
RuntimeStationaryProbMode = ConstantZero
RuntimeStationaryProbMode = ConstantOne
```

Expected observations:

```text
ModelPrediction: stationary log matches model prediction.
TrackerVelocity: static T-pose tracker values are close to [1,1,1,1,1].
ReplayReference: stationary log matches replay stationaryProb5 when replay data contains it.
ConstantZero: stationary active mask remains all 0 and no contact is confirmed.
ConstantOne: stationary active mask is all 1, while root/pelvis can still be filtered by leg-angle logic.
No log references PhysicsSystem, probe hit, SphereCast, or OverlapSphere.
```

- [ ] **Step 6: Final commit if verification required code fixes**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' status --short
```

Expected:

```text
Clean working tree after all implementation commits, or only known generated logs that are not staged.
```

---

## Self-Review Checklist

- Requirement coverage:
  - Runtime stationary source switching is covered by Tasks 2-5.
  - Tracker velocity stationary probability is covered by Tasks 3-5 and verified in Task 10.
  - Model stationary output remains separate from runtime stationary input in Tasks 4-5.
  - Unity collider/probe contact detection is removed in Tasks 6-8.
  - GlobalPose stationary + height + residual-QP contact physics remains in Tasks 7 and 10.

- Placeholder scan:
  - This plan contains only concrete implementation steps.

- Type consistency:
  - `RuntimeStationaryProbMode` is defined in Task 2 and used by Tasks 3-5 and Task 9.
  - `RuntimeStationaryProb5Resolver.Resolve` signature in Task 3 matches the driver call in Task 4.
  - Probe-related fields are removed from `FrameInput` before solver call sites are changed.

- Risk controls:
  - Defaults preserve existing behavior through `RuntimeStationaryProbMode.ModelPrediction`.
  - Probe removal is verified by `rg` before deleting `CarticulateGpContactProbe.cs`.
  - Schema, normalizer, and model tensor contracts are not changed.
