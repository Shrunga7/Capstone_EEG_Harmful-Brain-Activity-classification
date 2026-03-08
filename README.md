# Dashboard Branch — HMS EEG Real-Time Classification

Real-time web dashboard that streams spectrogram files through the trained **CNN+LSTM model** and displays live brain activity classifications.

## 📁 Branch Contents

```
Dashboard/
├── dashboard.py       ← single file: FastAPI server + model + preprocessing + UI
├── requirements.txt   ← pip dependencies
├── .gitignore
└── README.md
```

> **Note**: `best_transformer_model.pth` and the `train_spectrograms/` folder are not committed (see `.gitignore`). Place them locally before running.

---

## 🧠 Model

This dashboard loads the model saved by `CapstoneProjectLSTM.ipynb` from the `Hybrid_Models` branch.

**Architecture — `HybridLSTMSpectrogramModel`**

```
Input spectrogram  (1, 128, 256)
        ↓
EfficientNet-B0  (features_only, out_indices=[4])  →  320 channels
        ↓
Mean-pool frequency axis  →  (B, Time, 320)
        ↓
Bidirectional LSTM  hidden=256, layers=2  →  (B, Time, 512)
        ↓
Global average pool  →  (B, 512)
        ↓
Linear(512 → 6)  →  softmax probabilities
```

**Output classes**: Seizure · LPD · GPD · LRDA · GRDA · Other

---

## 🚀 Quick Start

### 1. Switch to this branch
```bash
git checkout Dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Place your files
```
your-project/
├── dashboard.py
├── best_transformer_model.pth      ← from CapstoneProjectLSTM.ipynb output
└── train_spectrograms/
    ├── 1000086677.parquet
    ├── 1000088078.parquet
    └── ...
```

### 4. Run
```bash
python dashboard.py
```

Open **http://localhost:8000** in your browser.

---

## ⚙️ Configuration

Edit the top of `dashboard.py` to change paths or speed:

```python
MODEL_PATH      = "best_transformer_model.pth"   # your saved model
SPECTROGRAM_DIR = "train_spectrograms"           # folder of .parquet files
STREAM_INTERVAL = 1.5                            # seconds between predictions
```

---

## 📊 Dashboard Panels

| Panel | Description |
|---|---|
| **Current Prediction** | Predicted class + confidence + risk level |
| **Class Probabilities** | Live bar chart for all 6 classes |
| **File Progress** | Which spectrogram is being read + progress bar |
| **EEG Visualisation** | Animated 8-channel waveform (modulated by predicted class) |
| **Classification Log** | Scrolling real-time log of every inference |

---

## 🔄 How It Works

```
train_spectrograms/*.parquet
        ↓  read one file per interval
preprocess  (NaN→0, log1p, crop→256, resize→128, standardise, transpose)
        ↓
HybridLSTMSpectrogramModel.forward()
        ↓
softmax probabilities  →  WebSocket  →  browser dashboard
```

Preprocessing mirrors `HMSDataset.__getitem__` from `CapstoneProjectLSTM.ipynb` exactly.

---

## ⚠️ Simulation Mode

If `best_transformer_model.pth` is not found, the dashboard starts automatically in **simulation mode** (random predictions). You will see an orange `Simulation` badge in the top-right corner instead of the green `LSTM Model` badge.

---

## 🔗 Related Branches

| Branch | Contents |
|---|---|
| `main` | Project overview |
| `CNN_model` | EfficientNet-B0 CNN training (Kaggle) |
| `Hybrid_Models` | CNN-LSTM and CNN-Transformer notebooks + hyperparameter tuning |
| `Spectrogram-Dataloader` | Spectrogram dataloader and CNN training example |
| `Dashboard` | ← You are here |
