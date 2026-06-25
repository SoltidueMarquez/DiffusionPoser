const { app, BrowserWindow } = require("electron");
const { spawn } = require("child_process");
const fs = require("fs");
const http = require("http");
const path = require("path");

// 这台机器已经出现过 NVIDIA 显示驱动恢复记录。Electron/Chromium 的 GPU
// 进程在关闭 WebGL 页面时可能触发驱动层问题，所以桌面壳默认走软件渲染。
app.disableHardwareAcceleration();
app.commandLine.appendSwitch("disable-gpu");
app.commandLine.appendSwitch("disable-gpu-compositing");

const editorRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(editorRoot, "..");
const preferredDataDir = path.join(repoRoot, "dataset/generated/tasks/realtime_pose_stationary5_v1/amass_60hz_tasks");
const preferredSourceDir = path.join(repoRoot, "dataset/generated/sources/realtime_pose_stationary5_v1/amass_60hz");
const apiPort = process.env.REALTIME_POSE_EDITOR_PORT || "8765";
let serverProcess = null;
let shutdownTimer = null;

function envOrDefault(name, fallback) {
  return process.env[name] || fallback;
}

function pythonExecutable() {
  const localPython = path.join(editorRoot, ".venv", "Scripts", "python.exe");
  if (fs.existsSync(localPython)) {
    return localPython;
  }
  return process.env.PYTHON || "python";
}

function startServer() {
  const args = [
    "-m",
    "visual_editor.server",
    "--host",
    "127.0.0.1",
    "--port",
    apiPort,
    "--runtime_dir",
    envOrDefault("REALTIME_POSE_EDITOR_RUNTIME_DIR", path.join(editorRoot, ".runtime")),
    "--amass_dir",
    envOrDefault("REALTIME_POSE_EDITOR_AMASS_DIR", path.join(repoRoot, "dataset", "AMASS")),
    "--data_dir",
    envOrDefault("REALTIME_POSE_EDITOR_DATA_DIR", preferredDataDir),
    "--source_dir",
    envOrDefault("REALTIME_POSE_EDITOR_SOURCE_DIR", preferredSourceDir),
    "--result_dir",
    envOrDefault("REALTIME_POSE_EDITOR_RESULT_DIR", path.join(repoRoot, "output")),
    "--output_dir",
    envOrDefault("REALTIME_POSE_EDITOR_OUTPUT_DIR", path.join(editorRoot, ".runtime", "exports")),
  ];
  if (process.env.REALTIME_POSE_EDITOR_SMPL_MODEL_DIR) {
    args.push("--smpl_model_dir", process.env.REALTIME_POSE_EDITOR_SMPL_MODEL_DIR);
  }
  serverProcess = spawn(pythonExecutable(), args, {
    cwd: repoRoot,
    env: {
      ...process.env,
      PYTHONPATH: repoRoot,
      REALTIME_POSE_EDITOR_RUNTIME_DIR: envOrDefault("REALTIME_POSE_EDITOR_RUNTIME_DIR", path.join(editorRoot, ".runtime")),
      REALTIME_POSE_EDITOR_AMASS_DIR: envOrDefault("REALTIME_POSE_EDITOR_AMASS_DIR", path.join(repoRoot, "dataset", "AMASS")),
      REALTIME_POSE_EDITOR_DATA_DIR: envOrDefault("REALTIME_POSE_EDITOR_DATA_DIR", preferredDataDir),
      REALTIME_POSE_EDITOR_SOURCE_DIR: envOrDefault("REALTIME_POSE_EDITOR_SOURCE_DIR", preferredSourceDir),
      REALTIME_POSE_EDITOR_RESULT_DIR: envOrDefault("REALTIME_POSE_EDITOR_RESULT_DIR", path.join(repoRoot, "output")),
      REALTIME_POSE_EDITOR_OUTPUT_DIR: envOrDefault("REALTIME_POSE_EDITOR_OUTPUT_DIR", path.join(editorRoot, ".runtime", "exports")),
    },
    stdio: "inherit",
    windowsHide: true,
  });
}

function waitForApiReady(timeoutMs = 30000, intervalMs = 500) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve) => {
    const probe = () => {
      const request = http.get(
        {
          hostname: "127.0.0.1",
          port: Number(apiPort),
          path: "/api/health",
          timeout: 1000,
        },
        (response) => {
          response.resume();
          if (response.statusCode && response.statusCode < 500) {
            resolve(true);
            return;
          }
          retry();
        },
      );
      request.on("timeout", () => {
        request.destroy();
        retry();
      });
      request.on("error", retry);
    };
    const retry = () => {
      if (Date.now() >= deadline) {
        resolve(false);
        return;
      }
      setTimeout(probe, intervalMs);
    };
    probe();
  });
}

function stopServerProcess() {
  if (!serverProcess) {
    return;
  }
  const pid = serverProcess.pid;
  try {
    serverProcess.kill();
  } catch (_) {
    // 后续 taskkill 会兜底清理子进程。
  }
  shutdownTimer = setTimeout(() => {
    try {
      spawn("taskkill", ["/PID", String(pid), "/T", "/F"], {
        windowsHide: true,
        stdio: "ignore",
      });
    } catch (_) {
      // 关闭阶段不再向用户抛出错误。
    }
  }, 1200);
  serverProcess = null;
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1100,
    minHeight: 720,
    backgroundColor: "#10141c",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  const devUrl = process.env.ELECTRON_RENDERER_URL || "http://127.0.0.1:5177";
  const distIndex = path.join(editorRoot, "dist", "index.html");
  if (fs.existsSync(distIndex) && !process.env.ELECTRON_RENDERER_URL) {
    win.loadFile(distIndex);
  } else {
    win.loadURL(devUrl);
  }
}

app.whenReady().then(async () => {
  startServer();
  await waitForApiReady();
  createWindow();
});

app.on("window-all-closed", () => {
  stopServerProcess();
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  stopServerProcess();
});
