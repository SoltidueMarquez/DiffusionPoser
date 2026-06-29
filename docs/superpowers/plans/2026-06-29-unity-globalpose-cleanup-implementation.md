# Unity GlobalPose Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 深度清理 Unity 端 `GlobalPose` 物理链路，让通用 `RealtimePosePhysicsDriver` 不再硬编码 GlobalPose fallback，同时把 GlobalPose runtime 拆成职责清晰的小文件。

**Architecture:** 通用 physics 层只保留 method asset/runtime 抽象和 native dynamics 生命周期；GlobalPose 作为一个显式 `RealtimePosePhysicsMethodAsset` 实现存在于 `Physics/GlobalPoseFlow/`。先删除 legacy fallback 和通用 snapshot 污染，再按 frame builder、reference velocity sync、diagnostics builder、self-test helper 拆分大文件。

**Tech Stack:** Unity 2022.3.17f1c1, C#, UnityEditor batchmode, PowerShell, Git.

---

## Paths

- DiffusionPoser docs repo: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/DiffusionPoser`
- Unity project: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity`
- GlobalPose flow: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow`
- Physics driver: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/RealtimePosePhysicsDriver.cs`
- Physics debug types: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/RealtimePosePhysicsDebugging.cs`
- Physics runtime interfaces: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/RealtimePosePhysicsMethodRuntime.cs`
- Physics custom editor: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/Editor/RealtimePosePhysicsRuntimeEditor.cs`

## File Structure

- Modify `RealtimePosePhysicsDriver.cs`
  - Remove GlobalPose legacy serialized fields, obsolete methods, fallback runtime creation, and GlobalPose-specific snapshot population.
  - Keep only generic method asset/runtime invocation.

- Modify `RealtimePosePhysicsRuntimeEditor.cs`
  - Remove legacy fallback warning.
  - Show an error when no method asset is assigned.

- Modify `RealtimePosePhysicsDebugging.cs`
  - Remove GlobalPose-only scalar fields from `PhysicsStepSnapshot`.
  - Read GlobalPose metrics from `RealtimePoseGlobalPoseMethodDiagnostics` only when formatting fallback summaries.

- Keep `RealtimePosePhysicsMethodRuntime.cs`
  - Keep `RealtimePoseGlobalPoseMethodDiagnostics` here for now, because generic driver and debug code still need the type without adding a new dependency edge.

- Modify `RealtimePoseGlobalPoseMethodAsset.cs`
  - Reduce it to the `ScriptableObject` asset class plus a smaller `RealtimePoseGlobalPoseMethodRuntime`.
  - Remove helper methods after they are moved to new files.

- Create `RealtimePoseGlobalPoseFrameBuilder.cs`
  - Own the buffers and logic for building `CarticulateGpNetPySolver.FrameInput`, stationary signal injection, contact probes, and predicted contact points.

- Create `RealtimePoseGlobalPoseReferenceVelocitySync.cs`
  - Own previous-joint-rotation state and Actor angular velocity synchronization.

- Create `RealtimePoseGlobalPoseDiagnosticsBuilder.cs`
  - Own contact observations, tau csv, GRF json, virtual force override, and GlobalPose summary construction.

- Create `RealtimePoseGlobalPoseSelfTest.cs`
  - Own one-step self-test logic and self-test diagnostics summary.

---

### Task 1: Preflight And Baseline

**Files:**
- Read only: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/ProjectSettings/ProjectVersion.txt`
- Read only: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics`

- [ ] **Step 1: Confirm Unity project version**

Run:

```powershell
Get-Content -Raw -LiteralPath 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\ProjectSettings\ProjectVersion.txt'
```

Expected:

```text
m_EditorVersion: 2022.3.17f1c1
```

- [ ] **Step 2: Confirm dirty state before touching Unity code**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' status --short
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\DiffusionPoser' status --short
```

Expected:

```text
SIGGRAPH2024Unity may show user changes. Do not revert them.
DiffusionPoser may show unrelated stationary-signal work. Do not stage it.
```

- [ ] **Step 3: Capture current GlobalPose coupling list**

Run:

```powershell
rg -n "StartGlobalPosePhysics|StopGlobalPosePhysics|UsesLegacyGlobalPoseFallback|HasLegacyGlobalPoseSettings|legacyRealtimePoseGlobalPosePhysicsSettings|legacySyncReferenceAngularVelocityFromJointTransforms|ConeQpFitError|ResidualNorm|FirstLsqrResidualForce" -S 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Assets\Projects\RealtimePose\Scripts'
```

Expected:

```text
Matches in RealtimePosePhysicsDriver.cs, RealtimePosePhysicsRuntimeEditor.cs, RealtimePosePhysicsDebugging.cs, RealtimePosePhysicsMethodRuntime.cs, CarticulateGpNetPySolver.cs, and RealtimePoseGlobalPoseMethodAsset.cs.
```

- [ ] **Step 4: Verify Unity can compile before refactor**

Run:

```powershell
$UnityExe = @(
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1c1\Editor\Unity.exe',
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1\Editor\Unity.exe',
  'C:\Program Files\Unity 2022.3.17f1c1\Editor\Unity.exe'
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $UnityExe) { throw 'Unity 2022.3.17f1c1 executable was not found in the known install locations.' }
& $UnityExe -batchmode -quit -projectPath 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' -logFile 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-preflight-compile.log'
if ($LASTEXITCODE -ne 0) { Get-Content -Tail 120 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-preflight-compile.log'; exit $LASTEXITCODE }
```

Expected:

```text
Exit code 0. If Unity is not installed at those paths, stop and ask for the actual Unity.exe path.
```

- [ ] **Step 5: Commit nothing**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' status --short
```

Expected:

```text
No new Codex-authored files or edits from Task 1.
```

---

### Task 2: Remove Driver Legacy GlobalPose Fallback

**Files:**
- Modify: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/RealtimePosePhysicsDriver.cs`
- Modify: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/Editor/RealtimePosePhysicsRuntimeEditor.cs`

- [ ] **Step 1: Edit `RealtimePosePhysicsDriver.cs` imports and fields**

Remove this import:

```csharp
using UnityEngine.Serialization;
```

Remove these serialized fields and properties:

```csharp
[FormerlySerializedAs("globalPosePhysicsSettings")]
[SerializeField, HideInInspector] private RealtimePoseGlobalPosePhysicsSettings legacyRealtimePoseGlobalPosePhysicsSettings;
[FormerlySerializedAs("syncReferenceAngularVelocityFromJointTransforms")]
[SerializeField, HideInInspector] private bool legacySyncReferenceAngularVelocityFromJointTransforms = true;

[Obsolete("Use IsPhysicsSessionActive instead.")]
public bool IsGlobalPosePhysicsActive => IsPhysicsSessionActive;
public bool UsesLegacyGlobalPoseFallback => methodAsset == null;
public bool HasLegacyGlobalPoseSettings => legacyRealtimePoseGlobalPosePhysicsSettings != null;
```

Keep this `[FormerlySerializedAs]` on `startPhysicsSessionOnPlay` for scene migration:

```csharp
[FormerlySerializedAs("startGlobalPosePhysicsOnPlay")]
[SerializeField] private bool startPhysicsSessionOnPlay;
```

This means `UnityEngine.Serialization` stays required unless this attribute is also removed. For this task keep the import and only remove the GlobalPose settings fallback fields.

- [ ] **Step 2: Remove obsolete GlobalPose session wrappers**

Delete these methods from `RealtimePosePhysicsDriver.cs`:

```csharp
[Obsolete("Use StartPhysicsSession() instead.")]
public void StartGlobalPosePhysics()
{
    StartPhysicsSession();
}

[Obsolete("Use StopPhysicsSession() instead.")]
public void StopGlobalPosePhysics()
{
    StopPhysicsSession();
}
```

- [ ] **Step 3: Replace runtime creation and method name resolution**

Replace `CreateConfiguredRuntime()` with:

```csharp
private IRealtimePosePhysicsMethodRuntime CreateConfiguredRuntime()
{
    if (methodAsset == null)
    {
        PublishError(
            RealtimePosePhysicsDebugTopic.Validation,
            "RealtimePosePhysicsDriver requires an assigned RealtimePosePhysicsMethodAsset.");
        return null;
    }

    return methodAsset.CreateRuntime();
}
```

Replace `ResolveConfiguredMethodName()` with:

```csharp
private string ResolveConfiguredMethodName()
{
    return methodAsset != null ? methodAsset.MethodName : "<unconfigured>";
}
```

- [ ] **Step 4: Update custom inspector method section**

In `RealtimePosePhysicsRuntimeEditor.cs`, replace `DrawMethodSection` with:

```csharp
private static void DrawMethodSection(RealtimePosePhysicsDriver driver)
{
    EditorGUILayout.LabelField("Physics Method", EditorStyles.boldLabel);
    EditorGUILayout.LabelField("Configured Method", driver.CurrentMethodName);
    if (driver.AssignedMethodAsset == null)
    {
        EditorGUILayout.HelpBox(
            "No physics method asset is assigned. Assign a RealtimePosePhysicsMethodAsset before initializing the driver.",
            MessageType.Error);
    }

    EditorGUILayout.Space();
}
```

- [ ] **Step 5: Run source coupling check**

Run:

```powershell
rg -n "StartGlobalPosePhysics|StopGlobalPosePhysics|UsesLegacyGlobalPoseFallback|HasLegacyGlobalPoseSettings|legacyRealtimePoseGlobalPosePhysicsSettings|legacySyncReferenceAngularVelocityFromJointTransforms" -S 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Assets\Projects\RealtimePose\Scripts'
```

Expected:

```text
No matches.
```

- [ ] **Step 6: Compile in Unity batchmode**

Run:

```powershell
$UnityExe = @(
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1c1\Editor\Unity.exe',
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1\Editor\Unity.exe',
  'C:\Program Files\Unity 2022.3.17f1c1\Editor\Unity.exe'
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $UnityExe) { throw 'Unity 2022.3.17f1c1 executable was not found in the known install locations.' }
& $UnityExe -batchmode -quit -projectPath 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' -logFile 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-task2-compile.log'
if ($LASTEXITCODE -ne 0) { Get-Content -Tail 160 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-task2-compile.log'; exit $LASTEXITCODE }
```

Expected:

```text
Exit code 0.
```

- [ ] **Step 7: Commit Task 2**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' add -- 'Assets/Projects/RealtimePose/Scripts/Physics/RealtimePosePhysicsDriver.cs' 'Assets/Projects/RealtimePose/Scripts/Physics/Editor/RealtimePosePhysicsRuntimeEditor.cs'
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' commit -m "refactor(realtime-pose): require physics method asset"
```

Expected:

```text
Commit succeeds and stages only the two listed files.
```

---

### Task 3: Move GlobalPose Metrics Out Of Generic Snapshot

**Files:**
- Modify: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/RealtimePosePhysicsDebugging.cs`
- Modify: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/RealtimePosePhysicsDriver.cs`

- [ ] **Step 1: Remove GlobalPose-only fields from `PhysicsStepSnapshot`**

In `RealtimePosePhysicsDebugging.cs`, remove these fields from `PhysicsStepSnapshot`:

```csharp
public float ConeQpFitError;
public float ResidualNorm;
public int ConeQpVarCount;
public Vector3 FirstLsqrResidualForce;
public bool HasFirstLsqrResidualForce;
```

Keep these generic fields:

```csharp
public bool[] ActiveContactMask;
public int ActiveContactCount;
public PhysicsContactObservation[] Contacts;
public string TauCsv;
public string GrfJson;
public bool HasVirtualForceOverride;
public Vector3 VirtualForceOverride;
public RealtimePosePhysicsMethodDiagnostics MethodDiagnostics;
```

- [ ] **Step 2: Replace fallback summary formatting**

In `PhysicsDebugRouter.FormatStepSummary`, replace the method body with:

```csharp
private static string FormatStepSummary(PhysicsStepSnapshot snapshot)
{
    if (!string.IsNullOrEmpty(snapshot.MethodSummary))
    {
        return snapshot.MethodSummary;
    }

    StringBuilder sb = new StringBuilder(192);
    if (!string.IsNullOrEmpty(snapshot.MethodName))
    {
        sb.Append(snapshot.MethodName);
        sb.Append(" | ");
    }

    sb.Append("step=");
    sb.Append(snapshot.StepIndex);
    sb.Append(" activeContacts=");
    sb.Append(snapshot.ActiveContactCount);

    if (snapshot.MethodDiagnostics is RealtimePoseGlobalPoseMethodDiagnostics gpDiagnostics)
    {
        sb.Append(" qpErr=");
        sb.Append(gpDiagnostics.ConeQpFitError.ToString("F4"));
        sb.Append(" residual=");
        sb.Append(gpDiagnostics.ResidualNorm.ToString("F4"));
    }

    if (snapshot.Qddot != null && snapshot.Qddot.Length >= 3)
    {
        sb.Append(" qddotRoot=(");
        sb.Append(snapshot.Qddot[0].ToString("F4"));
        sb.Append(",");
        sb.Append(snapshot.Qddot[1].ToString("F4"));
        sb.Append(",");
        sb.Append(snapshot.Qddot[2].ToString("F4"));
        sb.Append(")");
    }

    return sb.ToString();
}
```

- [ ] **Step 3: Remove GlobalPose-specific snapshot assignment from driver**

In `RealtimePosePhysicsDriver.FinalizeStepSnapshot`, remove this entire branch:

```csharp
if (diagnostics is RealtimePoseGlobalPoseMethodDiagnostics gpDiagnostics)
{
    snapshot.ConeQpFitError = gpDiagnostics.ConeQpFitError;
    snapshot.ResidualNorm = gpDiagnostics.ResidualNorm;
    snapshot.ConeQpVarCount = gpDiagnostics.ConeQpVarCount;
    snapshot.FirstLsqrResidualForce = gpDiagnostics.FirstLsqrResidualForce;
    snapshot.HasFirstLsqrResidualForce = gpDiagnostics.HasFirstLsqrResidualForce;
}
else
{
    snapshot.ConeQpFitError = 0f;
    snapshot.ResidualNorm = 0f;
    snapshot.ConeQpVarCount = 0;
    snapshot.FirstLsqrResidualForce = Vector3.zero;
    snapshot.HasFirstLsqrResidualForce = false;
}
```

After removal, `FinalizeStepSnapshot` still sets:

```csharp
snapshot.MethodDiagnostics = diagnostics;
snapshot.ActiveContactMask = diagnostics != null && diagnostics.ActiveContactMask != null
    ? (bool[])diagnostics.ActiveContactMask.Clone()
    : null;
snapshot.ActiveContactCount = diagnostics != null
    ? diagnostics.ActiveContactCount
    : CountActiveContacts(snapshot.ActiveContactMask);
```

- [ ] **Step 4: Run source check for snapshot-only fields**

Run:

```powershell
rg -n "snapshot\.ConeQpFitError|snapshot\.ResidualNorm|snapshot\.ConeQpVarCount|snapshot\.FirstLsqrResidualForce|snapshot\.HasFirstLsqrResidualForce|public float ConeQpFitError;|public float ResidualNorm;|public int ConeQpVarCount;|public Vector3 FirstLsqrResidualForce;|public bool HasFirstLsqrResidualForce;" -S 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Assets\Projects\RealtimePose\Scripts\Physics'
```

Expected:

```text
Matches remain only in RealtimePosePhysicsMethodRuntime.cs and GlobalPoseFlow solver/runtime diagnostics, not in PhysicsStepSnapshot or driver snapshot assignments.
```

- [ ] **Step 5: Compile in Unity batchmode**

Run:

```powershell
$UnityExe = @(
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1c1\Editor\Unity.exe',
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1\Editor\Unity.exe',
  'C:\Program Files\Unity 2022.3.17f1c1\Editor\Unity.exe'
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $UnityExe) { throw 'Unity 2022.3.17f1c1 executable was not found in the known install locations.' }
& $UnityExe -batchmode -quit -projectPath 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' -logFile 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-task3-compile.log'
if ($LASTEXITCODE -ne 0) { Get-Content -Tail 160 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-task3-compile.log'; exit $LASTEXITCODE }
```

Expected:

```text
Exit code 0.
```

- [ ] **Step 6: Commit Task 3**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' add -- 'Assets/Projects/RealtimePose/Scripts/Physics/RealtimePosePhysicsDebugging.cs' 'Assets/Projects/RealtimePose/Scripts/Physics/RealtimePosePhysicsDriver.cs'
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' commit -m "refactor(realtime-pose): keep globalpose metrics in method diagnostics"
```

Expected:

```text
Commit succeeds and stages only the listed files.
```

---

### Task 4: Extract Frame Builder And Reference Velocity Sync

**Files:**
- Modify: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/RealtimePoseGlobalPoseMethodAsset.cs`
- Create: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/RealtimePoseGlobalPoseFrameBuilder.cs`
- Create: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/RealtimePoseGlobalPoseReferenceVelocitySync.cs`

- [ ] **Step 1: Create `RealtimePoseGlobalPoseReferenceVelocitySync.cs`**

Add this file with this class shell and move the existing method bodies from `RealtimePoseGlobalPoseMethodRuntime` unchanged:

```csharp
using AI4Animation;
using UnityEngine;

namespace RealtimePose.Physics
{
    internal sealed class RealtimePoseGlobalPoseReferenceVelocitySync
    {
        private readonly Quaternion[] previousJointWorldRot = new Quaternion[RealtimePosePhysicsSmplAdapter.SmplJointCount];
        private bool hasPrevious;

        public void Reset()
        {
            hasPrevious = false;
        }

        public void Sync(Actor referenceActor, Transform[] referenceSmplJoints, float dt)
        {
            if (referenceActor == null || referenceActor.Bones == null || referenceSmplJoints == null || dt < 1e-9f)
            {
                return;
            }

            for (int j = 0; j < RealtimePosePhysicsSmplAdapter.SmplJointCount; j++)
            {
                Transform tr = referenceSmplJoints[j];
                if (tr == null)
                {
                    continue;
                }

                Actor.Bone bone = FindReferenceBoneForTransform(referenceActor, tr);
                if (bone == null)
                {
                    continue;
                }

                Quaternion qCurr = tr.rotation;
                if (hasPrevious)
                {
                    Quaternion deltaQ = qCurr * Quaternion.Inverse(previousJointWorldRot[j]);
                    Vector3 omegaWorld = Frame.OmegaWorldFromBoneDeltaQuaternion(deltaQ, dt);
                    omegaWorld.x = -omegaWorld.x;
                    bone.SetAngularVelocity(omegaWorld);
                }
                else
                {
                    bone.SetAngularVelocity(Vector3.zero);
                }

                previousJointWorldRot[j] = qCurr;
            }

            hasPrevious = true;
        }

        private static Actor.Bone FindReferenceBoneForTransform(Actor referenceActor, Transform transform)
        {
            for (int i = 0; i < referenceActor.Bones.Length; i++)
            {
                if (referenceActor.Bones[i].GetTransform() == transform)
                {
                    return referenceActor.Bones[i];
                }
            }

            return null;
        }
    }
}
```

- [ ] **Step 2: Create `RealtimePoseGlobalPoseFrameBuilder.cs`**

Add this file and move these existing methods from `RealtimePoseGlobalPoseMethodRuntime` into it without changing their internal statements:

- `BuildRuntimeFrameInput`
- `InjectStationaryProb5`
- `BuildNetPyVrFrameInput`
- `BuildPredictedContactPointsFromReferenceJoints`

Use this class boundary:

```csharp
using System;
using System.Text;
using AI4Animation;
using UnityEngine;

namespace RealtimePose.Physics
{
    internal sealed class RealtimePoseGlobalPoseFrameBuilder
    {
        private readonly CarticulateGpNetPySolver solver;
        private readonly RealtimePosePhysicsMethodInitContext initContext;
        private readonly float[] poseTarget24x9;
        private readonly float[] refQuatBuf;
        private readonly float[] refRootBuf;
        private readonly float[] refQdotBuf;
        private readonly float[] refJointPositionBuf;
        private readonly float[] refJointVelocityBuf;
        private readonly float[] predJointWorld15;
        private readonly float[] stationaryProb5;
        private readonly ContactProbeResult[] probeResults;
        private readonly ContactProbeConfig[] probeConfigs;

        public RealtimePoseGlobalPoseFrameBuilder(
            CarticulateGpNetPySolver solver,
            RealtimePosePhysicsMethodInitContext initContext,
            float[] poseTarget24x9,
            float[] refQuatBuf,
            float[] refRootBuf,
            float[] refQdotBuf,
            float[] refJointPositionBuf,
            float[] refJointVelocityBuf,
            float[] predJointWorld15,
            float[] stationaryProb5,
            ContactProbeResult[] probeResults,
            ContactProbeConfig[] probeConfigs)
        {
            this.solver = solver;
            this.initContext = initContext;
            this.poseTarget24x9 = poseTarget24x9;
            this.refQuatBuf = refQuatBuf;
            this.refRootBuf = refRootBuf;
            this.refQdotBuf = refQdotBuf;
            this.refJointPositionBuf = refJointPositionBuf;
            this.refJointVelocityBuf = refJointVelocityBuf;
            this.predJointWorld15 = predJointWorld15;
            this.stationaryProb5 = stationaryProb5;
            this.probeResults = probeResults;
            this.probeConfigs = probeConfigs;
        }

        public ContactProbeResult[] ProbeResults => probeResults;
        public ContactProbeConfig[] ProbeConfigs => probeConfigs;
        public float[] PredJointWorld15 => predJointWorld15;

        public CarticulateGpNetPySolver.FrameInput Build(
            in RealtimePosePhysicsFrameInput input,
            in RealtimePosePhysicsStepContext context,
            PhysicsStepSnapshot snapshot,
            in CarticulateGpNetPySolver.Settings settings)
        {
            return BuildRuntimeFrameInput(input, context, snapshot, settings);
        }
    }
}
```

After adding this shell, move the current `BuildRuntimeFrameInput` method from `RealtimePoseGlobalPoseMethodAsset.cs` into this class below `Build`, preserving its full body. Then move `InjectStationaryProb5`, `BuildNetPyVrFrameInput`, and `BuildPredictedContactPointsFromReferenceJoints` into the same class. The moved methods keep their current signatures except that they now use the class fields declared above instead of runtime fields with the same names.

- [ ] **Step 3: Wire helpers into `RealtimePoseGlobalPoseMethodRuntime`**

In `RealtimePoseGlobalPoseMethodRuntime`, add fields:

```csharp
private RealtimePoseGlobalPoseFrameBuilder frameBuilder;
private readonly RealtimePoseGlobalPoseReferenceVelocitySync referenceVelocitySync = new RealtimePoseGlobalPoseReferenceVelocitySync();
```

In `Initialize`, after all buffers are allocated, add:

```csharp
frameBuilder = new RealtimePoseGlobalPoseFrameBuilder(
    solver,
    initContext,
    poseTarget24x9,
    refQuatBuf,
    refRootBuf,
    refQdotBuf,
    refJointPositionBuf,
    refJointVelocityBuf,
    predJointWorld15,
    stationaryProb5,
    probeResults,
    probeConfigs);
referenceVelocitySync.Reset();
```

In `StartSession`, replace:

```csharp
refPrevJointWorldRotHasPrev = false;
```

with:

```csharp
referenceVelocitySync.Reset();
```

In `StopSession`, replace:

```csharp
refPrevJointWorldRotHasPrev = false;
```

with:

```csharp
referenceVelocitySync.Reset();
```

In `TryStep`, replace:

```csharp
SyncReferenceActorAngularVelocityFromJointTransforms(input.ReferenceActor, input.ReferenceSmplJoints, context.Dt);
```

with:

```csharp
referenceVelocitySync.Sync(input.ReferenceActor, input.ReferenceSmplJoints, context.Dt);
```

Replace:

```csharp
CarticulateGpNetPySolver.FrameInput frame = BuildRuntimeFrameInput(input, context, input.Snapshot, settings);
```

with:

```csharp
CarticulateGpNetPySolver.FrameInput frame = frameBuilder.Build(input, context, input.Snapshot, settings);
```

In `Shutdown`, add:

```csharp
frameBuilder = null;
referenceVelocitySync.Reset();
```

Then remove these fields and methods from `RealtimePoseGlobalPoseMethodRuntime` because the new helpers own them:

```csharp
private readonly Quaternion[] refPrevJointWorldRot = new Quaternion[RealtimePosePhysicsSmplAdapter.SmplJointCount];
private bool refPrevJointWorldRotHasPrev;

private CarticulateGpNetPySolver.FrameInput BuildRuntimeFrameInput(...)
private void InjectStationaryProb5(...)
private CarticulateGpNetPySolver.FrameInput BuildNetPyVrFrameInput(...)
private float[] BuildPredictedContactPointsFromReferenceJoints(...)
private void SyncReferenceActorAngularVelocityFromJointTransforms(...)
private static Actor.Bone FindReferenceBoneForTransform(...)
```

- [ ] **Step 4: Compile in Unity batchmode**

Run:

```powershell
$UnityExe = @(
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1c1\Editor\Unity.exe',
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1\Editor\Unity.exe',
  'C:\Program Files\Unity 2022.3.17f1c1\Editor\Unity.exe'
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $UnityExe) { throw 'Unity 2022.3.17f1c1 executable was not found in the known install locations.' }
& $UnityExe -batchmode -quit -projectPath 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' -logFile 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-task4-compile.log'
if ($LASTEXITCODE -ne 0) { Get-Content -Tail 200 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-task4-compile.log'; exit $LASTEXITCODE }
```

Expected:

```text
Exit code 0.
```

- [ ] **Step 5: Commit Task 4**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' add -- 'Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/RealtimePoseGlobalPoseMethodAsset.cs' 'Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/RealtimePoseGlobalPoseFrameBuilder.cs' 'Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/RealtimePoseGlobalPoseReferenceVelocitySync.cs'
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' commit -m "refactor(realtime-pose): extract globalpose frame inputs"
```

Expected:

```text
Commit succeeds and includes the modified runtime plus two new helper files.
```

---

### Task 5: Extract Diagnostics Builder And Self-Test

**Files:**
- Modify: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/RealtimePoseGlobalPoseMethodAsset.cs`
- Create: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/RealtimePoseGlobalPoseDiagnosticsBuilder.cs`
- Create: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/RealtimePoseGlobalPoseSelfTest.cs`

- [ ] **Step 1: Create `RealtimePoseGlobalPoseDiagnosticsBuilder.cs`**

Add this file and move these existing methods from `RealtimePoseGlobalPoseMethodRuntime` into it without changing their internal calculations:

- `BuildContactObservations`
- `TryGetSolverJointWorldPosition`
- `BuildTauCsvFromCurrentSolverState`
- `BuildGrfJsonFromCurrentSolverState`
- `ComputeVirtualForceUnityFromCurrentSolverState`
- `ComputeExplainedVirtualForceUnityFromCurrentSolverState`
- `ComputeExplainedContactForceMagnitudeFromCurrentSolverState`
- `BuildSummary`
- `CountActiveContacts`
- `MapSolverContactJointToVisualizeJoint`

Use this class boundary:

```csharp
using System;
using System.Globalization;
using System.Text;
using UnityEngine;

namespace RealtimePose.Physics
{
    internal sealed class RealtimePoseGlobalPoseDiagnosticsBuilder
    {
        private readonly RealtimePosePhysicsMethodInitContext initContext;
        private readonly float[] localZero = new float[3];
        private readonly float[] gizmoWorldPoint3 = new float[3];

        public RealtimePoseGlobalPoseDiagnosticsBuilder(RealtimePosePhysicsMethodInitContext initContext)
        {
            this.initContext = initContext;
        }

        public RealtimePoseGlobalPoseMethodDiagnostics Build(
            string methodName,
            RealtimePoseGlobalPosePhysicsSettings settingsAsset,
            CarticulateGpNetPySolver solver,
            in CarticulateGpNetPySolver.Diagnostics solverDiagnostics,
            Transform[] referenceSmplJoints,
            long handle,
            float[] paperQddot,
            float[] predJointWorld15,
            ContactProbeResult[] probeResults,
            ContactProbeConfig[] probeConfigs)
        {
            RealtimePoseGlobalPoseMethodDiagnostics diagnostics = new RealtimePoseGlobalPoseMethodDiagnostics
            {
                MethodName = methodName,
                ActiveContactMask = solverDiagnostics.ActiveContactMask != null ? (bool[])solverDiagnostics.ActiveContactMask.Clone() : null,
                ConeQpFitError = solverDiagnostics.ConeQpFitError,
                ResidualNorm = solverDiagnostics.ResidualWrench6Norm,
                ConeQpVarCount = solverDiagnostics.ConeQpVarCount,
                FirstLsqrResidualForce = solverDiagnostics.FirstLsqrResidualForceRaw,
                HasFirstLsqrResidualForce = solverDiagnostics.HasFirstLsqrResidualForceRaw
            };

            diagnostics.ActiveContactCount = CountActiveContacts(diagnostics.ActiveContactMask);
            diagnostics.Contacts = BuildContactObservations(solver, referenceSmplJoints, handle, diagnostics.ActiveContactMask, predJointWorld15, probeResults, probeConfigs);
            diagnostics.TauCsv = BuildTauCsvFromCurrentSolverState(solver, paperQddot);
            diagnostics.GrfJson = BuildGrfJsonFromCurrentSolverState(solver, diagnostics.ActiveContactMask);

            if (!string.IsNullOrEmpty(diagnostics.TauCsv))
            {
                Vector3 virtualForceUnity = ComputeVirtualForceUnityFromCurrentSolverState(solverDiagnostics);
                RealtimePosePhysicsDebugProfile profile = initContext.DebugProfileProvider != null ? initContext.DebugProfileProvider.Invoke() : null;
                diagnostics.VirtualForceOverride =
                    profile != null && profile.ReactionForceMode == RealtimePosePhysicsReactionForceMode.VirtualForceExplained
                        ? ComputeExplainedVirtualForceUnityFromCurrentSolverState(solver, virtualForceUnity, diagnostics.ActiveContactMask)
                        : virtualForceUnity;
                diagnostics.HasVirtualForceOverride = true;
            }
            else
            {
                diagnostics.VirtualForceOverride = Vector3.zero;
                diagnostics.HasVirtualForceOverride = false;
            }

            diagnostics.Summary = BuildSummary(methodName, settingsAsset, diagnostics, paperQddot);
            return diagnostics;
        }
    }
}
```

When applying this step, add the moved private methods below `Build`. Replace their old dependencies on runtime fields with explicit parameters shown in `Build`. Do not leave duplicated method copies in `RealtimePoseGlobalPoseMethodAsset.cs`.

- [ ] **Step 2: Create `RealtimePoseGlobalPoseSelfTest.cs`**

Add this file:

```csharp
namespace RealtimePose.Physics
{
    internal static class RealtimePoseGlobalPoseSelfTest
    {
        public static bool TryRun(
            string methodName,
            CarticulateGpNetPySolver solver,
            in CarticulateGpNetPySolver.Settings settings,
            in CarticulateGpNetPySolver.FrameInput frame,
            in RealtimePosePhysicsStepContext context,
            float[] paperQddot,
            out CarticulateGpNetPySolver.Diagnostics solverDiagnostics,
            out RealtimePosePhysicsMethodDiagnostics diagnostics,
            out string message)
        {
            diagnostics = null;
            message = null;
            solverDiagnostics = default;

            if (solver == null || paperQddot == null || context.StateVel == null)
            {
                message = $"{methodName} self test requires Play Mode and an initialized runtime.";
                return false;
            }

            bool ok = CarticulateGpNetPySelfTestRunner.TryRunSingleStep(
                context.Handle,
                RealtimePosePhysicsDriver.PhysicsStepSeconds,
                solver,
                settings,
                frame,
                context.StateQuatWxyz,
                context.StateRoot,
                context.StateVel,
                paperQddot,
                out solverDiagnostics);

            if (!ok)
            {
                message = $"{methodName} self test failed or returned integrate=false.";
                return false;
            }

            RealtimePoseGlobalPoseMethodDiagnostics gpDiagnostics = new RealtimePoseGlobalPoseMethodDiagnostics
            {
                MethodName = methodName,
                ActiveContactMask = solverDiagnostics.ActiveContactMask != null ? (bool[])solverDiagnostics.ActiveContactMask.Clone() : null,
                ConeQpFitError = solverDiagnostics.ConeQpFitError,
                ResidualNorm = solverDiagnostics.ResidualWrench6Norm,
                ConeQpVarCount = solverDiagnostics.ConeQpVarCount,
                FirstLsqrResidualForce = solverDiagnostics.FirstLsqrResidualForceRaw,
                HasFirstLsqrResidualForce = solverDiagnostics.HasFirstLsqrResidualForceRaw
            };
            gpDiagnostics.ActiveContactCount = CountActiveContacts(gpDiagnostics.ActiveContactMask);
            gpDiagnostics.Summary =
                $"{methodName} self test ok | " +
                $"contactMode={settings.ContactMode} " +
                $"activeContacts={gpDiagnostics.ActiveContactCount} qpErr={gpDiagnostics.ConeQpFitError:F4} " +
                $"residual={gpDiagnostics.ResidualNorm:F4} qpVars={gpDiagnostics.ConeQpVarCount} " +
                $"qddotRoot=({paperQddot[0]:F4},{paperQddot[1]:F4},{paperQddot[2]:F4})";

            diagnostics = gpDiagnostics;
            message = gpDiagnostics.Summary;
            return true;
        }

        private static int CountActiveContacts(bool[] mask)
        {
            if (mask == null)
            {
                return 0;
            }

            int count = 0;
            for (int i = 0; i < mask.Length; i++)
            {
                if (mask[i])
                {
                    count++;
                }
            }

            return count;
        }
    }
}
```

- [ ] **Step 3: Wire diagnostics builder into runtime**

In `RealtimePoseGlobalPoseMethodRuntime`, add:

```csharp
private RealtimePoseGlobalPoseDiagnosticsBuilder diagnosticsBuilder;
```

In `Initialize`, after `frameBuilder` creation, add:

```csharp
diagnosticsBuilder = new RealtimePoseGlobalPoseDiagnosticsBuilder(initContext);
```

Replace `BuildStepDiagnostics` body with:

```csharp
public RealtimePosePhysicsMethodDiagnostics BuildStepDiagnostics(in RealtimePosePhysicsFrameInput input, in RealtimePosePhysicsStepContext context)
{
    return diagnosticsBuilder != null
        ? diagnosticsBuilder.Build(
            MethodName,
            settingsAsset,
            solver,
            solverDiagnostics,
            input.ReferenceSmplJoints,
            context.Handle,
            paperQddot,
            frameBuilder != null ? frameBuilder.PredJointWorld15 : null,
            frameBuilder != null ? frameBuilder.ProbeResults : null,
            frameBuilder != null ? frameBuilder.ProbeConfigs : null)
        : null;
}
```

In `Shutdown`, add:

```csharp
diagnosticsBuilder = null;
```

- [ ] **Step 4: Wire self-test helper into runtime**

Replace `TryRunSelfTest` body with:

```csharp
public bool TryRunSelfTest(in RealtimePosePhysicsFrameInput input, in RealtimePosePhysicsStepContext context, out RealtimePosePhysicsMethodDiagnostics diagnostics, out string message)
{
    CarticulateGpNetPySolver.Settings settings = BuildNetPyPhysicsSettings();
    CarticulateGpNetPySolver.FrameInput frame = frameBuilder.Build(input, context, null, settings);
    return RealtimePoseGlobalPoseSelfTest.TryRun(
        MethodName,
        solver,
        settings,
        frame,
        context,
        paperQddot,
        out solverDiagnostics,
        out diagnostics,
        out message);
}
```

- [ ] **Step 5: Remove moved methods from runtime file**

After wiring, remove these methods from `RealtimePoseGlobalPoseMethodAsset.cs`:

```csharp
public RealtimePosePhysicsMethodDiagnostics BuildStepDiagnostics(...)
public bool TryRunSelfTest(...)
private PhysicsContactObservation[] BuildContactObservations(...)
private bool TryGetSolverJointWorldPosition(...)
private string BuildTauCsvFromCurrentSolverState(...)
private string BuildGrfJsonFromCurrentSolverState(...)
private Vector3 ComputeVirtualForceUnityFromCurrentSolverState(...)
private Vector3 ComputeExplainedVirtualForceUnityFromCurrentSolverState(...)
private float ComputeExplainedContactForceMagnitudeFromCurrentSolverState(...)
private string BuildSummary(...)
private static int CountActiveContacts(...)
private static int MapSolverContactJointToVisualizeJoint(...)
```

Keep the public interface implementations `BuildStepDiagnostics` and `TryRunSelfTest` in runtime with the new bodies from Steps 3 and 4.

- [ ] **Step 6: Compile in Unity batchmode**

Run:

```powershell
$UnityExe = @(
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1c1\Editor\Unity.exe',
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1\Editor\Unity.exe',
  'C:\Program Files\Unity 2022.3.17f1c1\Editor\Unity.exe'
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $UnityExe) { throw 'Unity 2022.3.17f1c1 executable was not found in the known install locations.' }
& $UnityExe -batchmode -quit -projectPath 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' -logFile 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-task5-compile.log'
if ($LASTEXITCODE -ne 0) { Get-Content -Tail 240 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-task5-compile.log'; exit $LASTEXITCODE }
```

Expected:

```text
Exit code 0.
```

- [ ] **Step 7: Commit Task 5**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' add -- 'Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/RealtimePoseGlobalPoseMethodAsset.cs' 'Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/RealtimePoseGlobalPoseDiagnosticsBuilder.cs' 'Assets/Projects/RealtimePose/Scripts/Physics/GlobalPoseFlow/RealtimePoseGlobalPoseSelfTest.cs'
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' commit -m "refactor(realtime-pose): extract globalpose diagnostics"
```

Expected:

```text
Commit succeeds and includes the modified runtime plus two new helper files.
```

---

### Task 6: Final Cleanup, Unity Save, And Verification

**Files:**
- Modify only if Unity Editor rewrites orphan serialized data: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Scenes/RealtimePose_DiffusionPoser_Test.unity`
- Modify only if Unity Editor rewrites orphan serialized data: `D:/Projects/SchoolWorkProjects/firstPaperRalated/01_当前主线项目/SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Physics/GlobalPose/RealtimePoseGlobalPoseMethod.asset`

- [ ] **Step 1: Verify no generic layer directly constructs GlobalPose runtime**

Run:

```powershell
rg -n "new RealtimePoseGlobalPoseMethodRuntime|RealtimePoseGlobalPosePhysicsSettings|UsesLegacyGlobalPoseFallback|HasLegacyGlobalPoseSettings|StartGlobalPosePhysics|StopGlobalPosePhysics" -S 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Assets\Projects\RealtimePose\Scripts\Physics'
```

Expected:

```text
Matches for RealtimePoseGlobalPosePhysicsSettings may remain only under Physics/GlobalPoseFlow and asset/test-scene editor code. No matches in RealtimePosePhysicsDriver.cs or RealtimePosePhysicsRuntimeEditor.cs.
```

- [ ] **Step 2: Verify `RealtimePoseGlobalPoseMethodAsset.cs` is slim**

Run:

```powershell
(Get-Content -LiteralPath 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Assets\Projects\RealtimePose\Scripts\Physics\GlobalPoseFlow\RealtimePoseGlobalPoseMethodAsset.cs').Count
```

Expected:

```text
Line count is materially lower than the original 700+ line file, with asset class and runtime lifecycle remaining.
```

- [ ] **Step 3: Compile in Unity batchmode**

Run:

```powershell
$UnityExe = @(
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1c1\Editor\Unity.exe',
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1\Editor\Unity.exe',
  'C:\Program Files\Unity 2022.3.17f1c1\Editor\Unity.exe'
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $UnityExe) { throw 'Unity 2022.3.17f1c1 executable was not found in the known install locations.' }
& $UnityExe -batchmode -quit -projectPath 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' -logFile 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-final-compile.log'
if ($LASTEXITCODE -ne 0) { Get-Content -Tail 240 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Logs\codex-final-compile.log'; exit $LASTEXITCODE }
```

Expected:

```text
Exit code 0.
```

- [ ] **Step 4: Open Unity Editor and save affected scene/assets**

Run this command to open the project:

```powershell
$UnityExe = @(
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1c1\Editor\Unity.exe',
  'C:\Program Files\Unity\Hub\Editor\2022.3.17f1\Editor\Unity.exe',
  'C:\Program Files\Unity 2022.3.17f1c1\Editor\Unity.exe'
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $UnityExe) { throw 'Unity 2022.3.17f1c1 executable was not found in the known install locations.' }
Start-Process -FilePath $UnityExe -ArgumentList @('-projectPath', 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity') -WindowStyle Hidden
```

In Unity Editor:

```text
Open Assets/Projects/RealtimePose/Scenes/RealtimePose_DiffusionPoser_Test.unity.
Select the RealtimePosePhysicsDriver object.
Confirm Assigned Method Asset is not null and points to RealtimePoseGlobalPoseMethod.asset.
Save the scene and project assets.
Close Unity.
```

Expected:

```text
Unity may rewrite scene/asset binary data to drop deleted serialized fields. Review those diffs before committing.
```

- [ ] **Step 5: Manual Play Mode verification**

In Unity Editor after compile:

```text
Open RealtimePose_DiffusionPoser_Test.
Enter Play Mode.
Select the object with RealtimePosePhysicsDriver.
Click Initialize Driver.
Click Start Physics Session.
Click NetPy Physics Self Test.
Confirm console contains "GlobalPoseNetPy self test ok".
Click Stop Physics Session.
Exit Play Mode.
```

Expected:

```text
No compile errors, no missing method asset warning, and self-test reports a GlobalPose diagnostics summary.
```

- [ ] **Step 6: Commit Task 6**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' status --short
```

Stage only Unity files changed by the cleanup:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' add -- 'Assets/Projects/RealtimePose/Scripts/Physics' 'Assets/Projects/RealtimePose/Scenes/RealtimePose_DiffusionPoser_Test.unity' 'Assets/Projects/RealtimePose/Physics/GlobalPose/RealtimePoseGlobalPoseMethod.asset'
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' commit -m "refactor(realtime-pose): finish globalpose physics cleanup"
```

Expected:

```text
Commit succeeds. If scene or asset files were not rewritten by Unity, Git reports that no such file changes were staged for those paths; the scripts directory changes still commit.
```

---

## Final Checks

- [ ] **Check no unrelated DiffusionPoser changes were staged**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\DiffusionPoser' status --short
```

Expected:

```text
Only pre-existing stationary-signal changes remain. This GlobalPose cleanup implementation should not stage them.
```

- [ ] **Check Unity commits are focused**

Run:

```powershell
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' log --oneline -4
git -C 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity' status --short
```

Expected:

```text
Recent commits correspond to Tasks 2, 3, 4, 5, and 6. Remaining dirty files, if any, are reviewed and intentionally left unstaged.
```

- [ ] **Check source-level success criteria**

Run:

```powershell
rg -n "new RealtimePoseGlobalPoseMethodRuntime|UsesLegacyGlobalPoseFallback|HasLegacyGlobalPoseSettings|StartGlobalPosePhysics|StopGlobalPosePhysics|legacyRealtimePoseGlobalPosePhysicsSettings|legacySyncReferenceAngularVelocityFromJointTransforms" -S 'D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\SIGGRAPH2024Unity\Assets\Projects\RealtimePose\Scripts'
```

Expected:

```text
No matches.
```
