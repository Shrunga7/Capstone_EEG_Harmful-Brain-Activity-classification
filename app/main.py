"""
EEG Dashboard + Prediction API
------------------------------
What this version does:
- Serves dashboard at "/"
- Supports WebSocket live demo at "/ws"
- Supports health check at "/health"
- Supports real prediction by uploading a parquet file to "/predict"
- Uses Cloud Run compatible PORT
- Uses ws/wss automatically in browser

Expected files/folders:
.
├── main/dashboard.py
├── CNN_LSTM_KLD_v0.pth
└── sample_spectrograms/
    ├── 123.parquet
    ├── 456.parquet
    └── ...

Install:
pip install fastapi uvicorn websockets torch timm numpy pandas pyarrow opencv-python python-multipart

Run locally:
python dashboard.py

Open:
http://localhost:8000
"""

import os
import json
import asyncio
import tempfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse


# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Path to trained model weights
MODEL_PATH = BASE_DIR / "CNN_LSTM_KLD_v0.pth"

# Folder containing demo parquet files for auto-streaming on dashboard
SPECTROGRAM_DIR = BASE_DIR / "sample_spectrograms"


# Delay between demo predictions over WebSocket
STREAM_INTERVAL = 1.5

# Output classes in correct order used during training
CLASSES = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]

# Device selection: GPU if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# MODEL DEFINITION
# Must match the training notebook architecture exactly
# =============================================================================

class HybridLSTMSpectrogramModel(nn.Module):
    def __init__(self, num_classes=6, hidden_size=256, num_layers=2):
        super().__init__()

        # EfficientNet-B0 feature extractor
        self.cnn = timm.create_model(
            "efficientnet_b0",
            pretrained=False,      # do not load ImageNet weights here
            in_chans=1,            # single-channel spectrogram
            features_only=True,    # extract feature maps only
            out_indices=[4]        # deepest feature block
        )

        # Bidirectional LSTM over time dimension
        self.lstm = nn.LSTM(
            input_size=320,        # EfficientNet output channels at selected block
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Final classifier
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x shape: (B, 1, 128, 256)
        features = self.cnn(x)[0]             # (B, 320, H, W)
        features = features.mean(dim=2)       # average over frequency height -> (B, 320, W)
        features = features.permute(0, 2, 1)  # (B, W, 320) for LSTM
        lstm_out, _ = self.lstm(features)     # (B, W, hidden*2)
        pooled = lstm_out.mean(dim=1)         # global average over time -> (B, hidden*2)
        return self.classifier(pooled)        # (B, 6)


# =============================================================================
# MODEL LOADING
# =============================================================================

MODEL = None

def load_model() -> bool:
    """
    Load model weights once at startup.
    Returns True if successful, else False.
    """
    global MODEL

    print(f"Current working directory: {Path.cwd()}")
    print(f"Resolved model path: {MODEL_PATH}")
    print(f"Resolved spectrogram dir: {SPECTROGRAM_DIR}")

    if not MODEL_PATH.exists():
        print(f"WARNING: Model file not found: {MODEL_PATH}")
        return False

    try:
        # Create model object
        MODEL = HybridLSTMSpectrogramModel(num_classes=6, hidden_size=256).to(DEVICE)

        # Load checkpoint
        state = torch.load(MODEL_PATH, map_location=DEVICE)

        # Handle both raw state_dict and wrapped checkpoint dict
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        # Load weights
        MODEL.load_state_dict(state)
        MODEL.eval()

        print(f"SUCCESS: Model loaded from {MODEL_PATH} on {DEVICE}")
        return True

    except Exception as e:
        print(f"ERROR: Model load failed: {e}")
        return False


MODEL_LOADED = load_model()


# =============================================================================
# PREPROCESSING
# Must match training preprocessing as closely as possible
# =============================================================================

def preprocess_parquet(path: str):
    """
    Read a parquet spectrogram file and convert it to model input tensor.

    Steps:
    1. Read parquet
    2. Drop 'time' column if present
    3. Replace NaN with 0
    4. Apply log1p
    5. Center crop or zero pad time axis to 256
    6. Resize to final shape
    7. Standardize
    8. Convert to tensor with shape (1, 1, 128, 256)
    """
    try:
        # Read parquet file
        df = pd.read_parquet(path)

        # Remove non-spectrogram time column if present
        if "time" in df.columns:
            df = df.drop(columns=["time"])

        # Convert to numpy float32
        spec = df.values.astype(np.float32)

        # Replace NaNs with 0
        spec = np.nan_to_num(spec, nan=0.0)

        # Apply log transform for dynamic range compression
        spec = np.log1p(spec)

        # Make time axis exactly 256
        desired_time = 256
        t = spec.shape[0]

        if t > desired_time:
            # Center crop if too long
            start = (t - desired_time) // 2
            spec = spec[start:start + desired_time, :]
        else:
            # Zero pad if too short
            spec = np.pad(spec, ((0, desired_time - t), (0, 0)), mode="constant")

        # Resize to width=128, height=256
        # OpenCV format is (width, height)
        spec = cv2.resize(spec, (128, 256))

        # Standardize
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)

        # Transpose to (freq, time) => (128, 256)
        spec = spec.T

        # Convert to tensor and add batch + channel dimensions
        tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Move to CPU/GPU
        return tensor.to(DEVICE)

    except Exception as e:
        print(f"Preprocess error for {os.path.basename(path)}: {e}")
        return None

# This avoids repeating inference logic in multiple places.
def predict_tensor(tensor: torch.Tensor):
    """
    Run model inference on an already preprocessed tensor.
    Returns prediction dictionary or None if prediction cannot run.
    """
    if tensor is None:
        return None

    if not MODEL_LOADED or MODEL is None:
        return None

    with torch.no_grad():
        probs = torch.softmax(MODEL(tensor), dim=1)[0].cpu().numpy()

    idx  = int(np.argmax(probs))
    conf = float(probs[idx] * 100)
    risk = "HIGH" if idx == 0 else ("MEDIUM" if idx in (1, 2, 4) else "LOW")

    return {
        "predicted": CLASSES[idx],
        "confidence": round(conf, 1),
        "risk": risk,
        "probabilities": {c: round(float(p * 100), 1) for c, p in zip(CLASSES, probs)}
    }


# =============================================================================
# STREAMER FOR DEMO MODE
# Loops through local parquet files one by one for dashboard live updates
# =============================================================================

class SpectrogramStreamer:
    def __init__(self):
        if not SPECTROGRAM_DIR.exists():
          print(f"WARNING: Demo spectrogram folder not found: {SPECTROGRAM_DIR}")
          self.files = []
        else:
          self.files = sorted(str(p) for p in SPECTROGRAM_DIR.glob("*.parquet"))

        self.index = 0
        self.total = len(self.files)
        self.started_at = datetime.now()

        if self.total == 0:
            print(f"WARNING: No parquet files found in: {SPECTROGRAM_DIR}")
        else:
            print(f"INFO: Found {self.total} parquet files for demo streaming")

    def elapsed(self):
        """
        Return elapsed time since app started in HH:MM:SS format.
        """
        seconds = int((datetime.now() - self.started_at).total_seconds())
        return f"{seconds//3600:02d}:{(seconds%3600)//60:02d}:{seconds%60:02d}"

    def simulate(self):
        """
        Fallback prediction payload when model/data unavailable.
        """
        raw = np.random.rand(len(CLASSES))
        probs = raw / raw.sum()
        idx = int(np.argmax(probs))

        return {
            "source": "simulation",
            "spec_id": "N/A",
            "file_index": self.index,
            "total_files": self.total,
            "predicted": CLASSES[idx],
            "confidence": round(float(probs[idx] * 100), 1),
            "risk": "LOW",
            "probabilities": {
                cls_name: round(float(p * 100), 1)
                for cls_name, p in zip(CLASSES, probs)
            },
            "elapsed": self.elapsed(),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }

    def next(self):
        """
        Process next parquet file and return prediction payload for dashboard.
        """
        # Fallback if no files or model missing
        if self.total == 0 or not MODEL_LOADED:
            self.index += 1
            return self.simulate()

        # Pick next file in loop
        path = self.files[self.index % self.total]
        self.index += 1

        # Extract file id from filename
        spec_id = os.path.splitext(os.path.basename(path))[0]

        # Preprocess and predict
        tensor = preprocess_parquet(path)
        pred = predict_tensor(tensor)

        # Fallback if preprocessing or prediction failed
        if pred is None:
            return self.simulate()

        return {
            "source": "model",
            "spec_id": spec_id,
            "file_index": self.index,
            "total_files": self.total,
            "predicted": pred["predicted"],
            "confidence": pred["confidence"],
            "risk": pred["risk"],
            "probabilities": pred["probabilities"],
            "elapsed": self.elapsed(),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }


streamer = SpectrogramStreamer()


# =============================================================================
# DASHBOARD HTML
# Includes upload button + websocket live demo + proper ws/wss support
# =============================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EEG Live Classification Dashboard</title>
<style>
body{
  font-family:Arial,sans-serif;
  background:#0b1220;
  color:#e5eef8;
  margin:0;
  padding:0;
}
.container{
  max-width:1100px;
  margin:0 auto;
  padding:20px;
}
h1{margin-bottom:8px;}
.sub{color:#9eb3c9;margin-bottom:24px;}
.card{
  background:#121b2e;
  border:1px solid #22324f;
  border-radius:10px;
  padding:16px;
  margin-bottom:18px;
}
.row{
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:16px;
}
.big{
  font-size:28px;
  font-weight:700;
  margin-bottom:10px;
}
.small{color:#9eb3c9;font-size:14px;}
.bar-wrap{margin:10px 0;}
.bar-label{margin-bottom:4px;font-size:14px;}
.bar{
  width:100%;
  height:18px;
  background:#1b2740;
  border-radius:999px;
  overflow:hidden;
}
.fill{
  height:100%;
  width:0%;
  background:#4ea1ff;
  transition:width .4s ease;
}
.status{
  display:inline-block;
  padding:6px 10px;
  border-radius:999px;
  font-weight:700;
  font-size:13px;
}
.high{background:#5a1827;color:#ff8fa6;}
.medium{background:#5a4318;color:#ffc266;}
.low{background:#163926;color:#7ae7a1;}
input[type=file]{
  margin:10px 0;
}
button{
  background:#2563eb;
  color:white;
  border:none;
  border-radius:8px;
  padding:10px 14px;
  cursor:pointer;
  font-weight:700;
}
button:hover{background:#1d4ed8;}
pre{
  background:#0a1120;
  border:1px solid #22324f;
  padding:12px;
  border-radius:8px;
  overflow:auto;
}
#log{
  max-height:250px;
  overflow:auto;
}
.log-item{
  border-bottom:1px solid #22324f;
  padding:8px 0;
  font-size:14px;
}
@media(max-width:800px){
  .row{grid-template-columns:1fr;}
}
</style>
</head>
<body>
<div class="container">
  <h1>EEG Live Classification Dashboard</h1>
  <div class="sub">Live demo via WebSocket + real prediction via file upload</div>

  <div class="row">
    <div class="card">
      <div class="small">Current Prediction</div>
      <div class="big" id="predicted">—</div>
      <div class="small">Confidence: <span id="confidence">—</span>%</div>
      <div style="margin-top:10px">
        <span id="risk" class="status low">—</span>
      </div>
      <div style="margin-top:14px" class="small">Spec ID: <span id="specId">—</span></div>
      <div class="small">Source: <span id="source">—</span></div>
      <div class="small">Time: <span id="timestamp">—</span></div>
      <div class="small">Elapsed: <span id="elapsed">—</span></div>
      <div class="small">File: <span id="fileProgress">—</span></div>
    </div>

    <div class="card">
      <div class="small">Upload Parquet for Real Prediction</div>
      <input type="file" id="fileInput" accept=".parquet">
      <br>
      <button onclick="uploadParquet()">Predict Uploaded File</button>
      <div style="margin-top:12px" class="small">Upload Result</div>
      <pre id="uploadResult">No prediction yet.</pre>
    </div>
  </div>

  <div class="card">
    <div class="small">Class Probabilities</div>
    <div id="bars"></div>
  </div>

  <div class="card">
    <div class="small">Live Demo Log</div>
    <div id="log"></div>
  </div>
</div>

<script>
const CLASSES = ["Seizure","LPD","GPD","LRDA","GRDA","Other"];

// Build probability bars once
const barsContainer = document.getElementById("bars");
barsContainer.innerHTML = CLASSES.map(c => `
  <div class="bar-wrap">
    <div class="bar-label">${c}: <span id="txt-${c}">0%</span></div>
    <div class="bar">
      <div class="fill" id="bar-${c}"></div>
    </div>
  </div>
`).join("");

function setRiskBadge(risk){
  const el = document.getElementById("risk");
  el.textContent = risk;
  el.className = "status " + (risk === "HIGH" ? "high" : risk === "MEDIUM" ? "medium" : "low");
}

function updateDashboard(data){
  document.getElementById("predicted").textContent = data.predicted ?? "—";
  document.getElementById("confidence").textContent = data.confidence ?? "—";
  document.getElementById("specId").textContent = data.spec_id ?? "—";
  document.getElementById("source").textContent = data.source ?? "—";
  document.getElementById("timestamp").textContent = data.timestamp ?? "—";
  document.getElementById("elapsed").textContent = data.elapsed ?? "—";
  document.getElementById("fileProgress").textContent =
    `${data.file_index ?? 0} / ${data.total_files ?? 0}`;

  setRiskBadge(data.risk ?? "LOW");

  const probs = data.probabilities || {};
  CLASSES.forEach(c => {
    const v = probs[c] || 0;
    document.getElementById(`txt-${c}`).textContent = `${v.toFixed(1)}%`;
    document.getElementById(`bar-${c}`).style.width = `${v}%`;
  });

  const log = document.getElementById("log");
  const item = document.createElement("div");
  item.className = "log-item";
  item.textContent = `[${data.timestamp || "--:--:--"}] ${data.predicted || "—"} | ${data.confidence || 0}% | ${data.spec_id || "N/A"} | ${data.source || "—"}`;
  log.prepend(item);

  while (log.children.length > 30){
    log.removeChild(log.lastChild);
  }
}

// WebSocket connect with ws/wss auto selection
// This is required for Cloud Run because deployed apps usually run on HTTPS
function connectWebSocket(){
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${protocol}://${window.location.host}/ws`);

  ws.onopen = () => console.log("WebSocket connected");
  ws.onclose = () => {
    console.log("WebSocket disconnected, retrying...");
    setTimeout(connectWebSocket, 2500);
  };
  ws.onerror = (e) => console.error("WebSocket error:", e);
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateDashboard(data);
  };
}

connectWebSocket();

// Upload parquet file to /predict
async function uploadParquet(){
  const input = document.getElementById("fileInput");
  const output = document.getElementById("uploadResult");

  if (!input.files.length){
    output.textContent = "Please choose a .parquet file first.";
    return;
  }

  const formData = new FormData();
  formData.append("file", input.files[0]);

  output.textContent = "Uploading and predicting...";

  try{
    const response = await fetch("/predict", {
      method: "POST",
      body: formData
    });

    const result = await response.json();
    output.textContent = JSON.stringify(result, null, 2);

    // Also update main dashboard panel using upload result
    if (result.predicted){
      updateDashboard({
        source: "upload",
        spec_id: result.filename || "uploaded_file",
        file_index: 1,
        total_files: 1,
        predicted: result.predicted,
        confidence: result.confidence,
        risk: result.risk,
        probabilities: result.probabilities,
        elapsed: "manual",
        timestamp: new Date().toLocaleTimeString()
      });
    }
  }catch(err){
    output.textContent = "Upload failed: " + err;
  }
}
</script>
</body>
</html>
"""


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(title="EEG Dashboard API")


@app.get("/")
async def root():
    """
    Serve dashboard UI.
    """
    return HTMLResponse(content=DASHBOARD_HTML)


@app.get("/health")
async def health():
    """
    Simple health endpoint for Docker/Cloud Run testing.
    """
    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "device": str(DEVICE),
        "demo_files_found": streamer.total
    }


@app.post("/predict")
async def predict_uploaded_file(file: UploadFile = File(...)):
    """
    Predict from an uploaded parquet file.

    Usage:
    - Send multipart/form-data with file field named "file"
    - File should be a parquet spectrogram
    """
    if not MODEL_LOADED:
        return JSONResponse(
            status_code=500,
            content={"error": "Model not loaded"}
        )

    if not file.filename.lower().endswith(".parquet"):
        return JSONResponse(
            status_code=400,
            content={"error": "Only .parquet files are supported"}
        )

    temp_path = None

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp:
            temp_path = tmp.name
            tmp.write(await file.read())

        # Preprocess uploaded parquet
        tensor = preprocess_parquet(temp_path)
        pred = predict_tensor(tensor)

        if pred is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Prediction failed"}
            )

        # Return structured response
        return {
            "filename": file.filename,
            "predicted": pred["predicted"],
            "confidence": pred["confidence"],
            "risk": pred["risk"],
            "probabilities": pred["probabilities"]
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Server error: {str(e)}"}
        )

    finally:
        # Always remove temp file if created
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Live demo websocket:
    sends one prediction every STREAM_INTERVAL seconds
    using local parquet files from SPECTROGRAM_DIR.
    """
    await websocket.accept()

    try:
        while True:
            result = streamer.next()
            await websocket.send_text(json.dumps(result))
            await asyncio.sleep(STREAM_INTERVAL)

    except WebSocketDisconnect:
        print("WebSocket client disconnected")

    except Exception as e:
        print(f"WebSocket error: {e}")


# =============================================================================
# ENTRY POINT
# Cloud Run compatible: reads PORT environment variable
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print("HMS EEG LIVE DASHBOARD")
    print("=" * 65)
    print(f"  Model  : {'✅  ' + str(MODEL_PATH) if MODEL_LOADED else '⚠️   Not found — simulation mode'}")
    print(f"  Data   : {streamer.total} spectrogram files in '{SPECTROGRAM_DIR}/'")
    print(f"  Classes: {', '.join(CLASSES)}")
    print(f"  Device : {DEVICE}")
    print("=" * 65)
    port = int(os.environ.get("PORT", 8000))
    print(f"\n  🌐  Open  http://localhost:{port}\n")

    uvicorn.run(app, host="0.0.0.0", port=port) # For Cloud Run, bind to all interfaces and use PORT env variable