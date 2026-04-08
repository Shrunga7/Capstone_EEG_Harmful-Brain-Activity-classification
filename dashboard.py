import os
import glob
import asyncio
import json
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel # <--- NEW LLM LAYER

# ─────────────────────────────────────────────────────────────────────────────
# NEW LLM LAYER: INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
# try:
#     from google import genai
#     from google.genai import types
#     gemini_client = genai.Client() # Automatically looks for GEMINI_API_KEY
#     LLM_READY = True
# except Exception as e:
#     print("⚠️  Gemini client failed to initialize. Did you set GEMINI_API_KEY?")
#     gemini_client = None
#     LLM_READY = False

# ─────────────────────────────────────────────────────────────────────────────
# NEW LLM LAYER: INITIALIZATION (DEBUG MODE)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from google import genai
    from google.genai import types
    
    # BRUTE FORCE TEST: Hardcode the key temporarily
    # Replace the string below with your actual key starting with "AIza..."
    gemini_client = genai.Client(api_key="[ENCRYPTION_KEY]") 
    
    LLM_READY = True
except Exception as e:
    # This will print the EXACT reason it is failing
    print(f"\n⚠️ GEMINI FATAL ERROR: {e}\n") 
    gemini_client = None
    LLM_READY = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH      = "best_transformer_model.pth"   
SPECTROGRAM_DIR = "train_spectrograms"           
STREAM_INTERVAL = 1.5                            
CLASSES         = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]

# ─────────────────────────────────────────────────────────────────────────────
# MODEL (Kept exactly the same)
# ─────────────────────────────────────────────────────────────────────────────
class HybridLSTMSpectrogramModel(nn.Module):
    def __init__(self, num_classes=6, hidden_size=256, num_layers=2):
        super().__init__()
        self.cnn = timm.create_model("efficientnet_b0", pretrained=False, in_chans=1, features_only=True, out_indices=[4])
        self.lstm = nn.LSTM(input_size=320, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        features = self.cnn(x)[0]             
        features = features.mean(dim=2)       
        features = features.permute(0, 2, 1)  
        lstm_out, _ = self.lstm(features)     
        x = lstm_out.mean(dim=1)              
        return self.classifier(x)             


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL  = None

def load_model():
    global MODEL
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  '{MODEL_PATH}' not found — running in simulation mode.")
        return False
    try:
        MODEL = HybridLSTMSpectrogramModel(num_classes=6, hidden_size=256).to(DEVICE)
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        MODEL.load_state_dict(state)
        MODEL.eval()
        print(f"✅ Model loaded  →  {MODEL_PATH}  [{DEVICE}]")
        return True
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        return False

MODEL_LOADED = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING & STREAMER (Kept exactly the same)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_parquet(path: str):
    try:
        df = pd.read_parquet(path)
        if "time" in df.columns:
            df = df.drop(columns=["time"])
        spec = df.values.astype(np.float32)
        spec = np.nan_to_num(spec, nan=0.0)
        spec = np.log1p(spec)
        desired_time = 256
        t = spec.shape[0]
        if t > desired_time:
            start = (t - desired_time) // 2
            spec = spec[start: start + desired_time, :]
        else:
            spec = np.pad(spec, ((0, desired_time - t), (0, 0)), mode="constant")
        spec = cv2.resize(spec, (128, 256))           
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        spec = spec.T                                  
        tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  
        return tensor.unsqueeze(0).to(DEVICE)                           
    except Exception as e:
        return None

class SpectrogramStreamer:
    def __init__(self):
        self.files = sorted(glob.glob(os.path.join(SPECTROGRAM_DIR, "*.parquet")))
        self.index = 0
        self.total = len(self.files)
        self.t0    = datetime.now()
        
    def next(self) -> dict:
        if self.total == 0 or not MODEL_LOADED:
            return self._simulate()
        path    = self.files[self.index % self.total]
        self.index += 1
        spec_id = os.path.splitext(os.path.basename(path))[0]
        tensor  = preprocess_parquet(path)
        if tensor is None:
            return self._simulate()
        with torch.no_grad():
            probs = torch.softmax(MODEL(tensor), dim=1)[0].cpu().numpy()
        idx  = int(np.argmax(probs))
        conf = float(probs[idx] * 100)
        risk = "HIGH" if idx == 0 else ("MEDIUM" if idx in (1, 2, 4) else "LOW")
        return {
            "source":        "model",
            "spec_id":       spec_id,
            "file_index":    self.index,
            "total_files":   self.total,
            "predicted":     CLASSES[idx],
            "confidence":    round(conf, 1),
            "risk":          risk,
            "probabilities": {c: round(float(p * 100), 1) for c, p in zip(CLASSES, probs)},
            "elapsed":       self._elapsed(),
            "timestamp":     datetime.now().strftime("%H:%M:%S"),
        }

    def _elapsed(self):
        s = int((datetime.now() - self.t0).total_seconds())
        return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

    def _simulate(self):
        import random
        raw   = [random.random() for _ in CLASSES]
        total = sum(raw)
        probs = [v / total for v in raw]
        idx   = int(np.argmax(probs))
        self.index += 1
        return {
            "source":        "simulation",
            "spec_id":       "N/A",
            "file_index":    self.index,
            "total_files":   self.total,
            "predicted":     CLASSES[idx],
            "confidence":    round(probs[idx] * 100, 1),
            "risk":          "LOW",
            "probabilities": {c: round(p * 100, 1) for c, p in zip(CLASSES, probs)},
            "elapsed":       self._elapsed(),
            "timestamp":     datetime.now().strftime("%H:%M:%S"),
        }

streamer = SpectrogramStreamer()

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD HTML (MAXIMUM READABILITY & OPTIMIZED GRID)
# ─────────────────────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HMS · EEG Live Classification</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&family=Syne:wght@700;800&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#06090f;--sur:#0b1118;--sur2:#0e1620;
  --bdr:#182535;--bdr2:#1e3348;
  --txt:#b8d4e8;--mut:#3d5a72;
  --sz:#ff2d55;--lpd:#ff8c00;--gpd:#f5c518;
  --lrda:#30d158;--grda:#a259ff;--oth:#32ade6;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body{height:100%;background:var(--bg);color:var(--txt);
  font-family:'IBM Plex Mono',monospace;font-size:18px;overflow:hidden;} /* BUMPED BASE FONT TO 18px */
body::before{content:'';position:fixed;inset:0;
  background-image:linear-gradient(var(--bdr) 1px,transparent 1px),
                   linear-gradient(90deg,var(--bdr) 1px,transparent 1px);
  background-size:48px 48px;opacity:.28;pointer-events:none;z-index:0;}
#app{position:relative;z-index:1;display:grid;
     grid-template-rows:75px 1fr 45px;height:100vh;} /* Taller header/footer */

header{display:flex;align-items:center;justify-content:space-between;
  padding:0 30px;border-bottom:1px solid var(--bdr2);
  background:rgba(11,17,24,.92);backdrop-filter:blur(8px);}
.brand{display:flex;align-items:center;gap:14px;}
.pulse{width:14px;height:14px;border-radius:50%;background:var(--sz);
  box-shadow:0 0 0 4px rgba(255,45,85,.15);animation:beat 1.8s ease-in-out infinite;}
@keyframes beat{0%,100%{transform:scale(1)}50%{transform:scale(1.3)}}
.brand-name{font-family:'Syne',sans-serif;font-weight:800;font-size:24px;
  letter-spacing:.1em;color:#fff;text-transform:uppercase;}
.brand-sub{font-size:13px;letter-spacing:.16em;color:var(--mut);text-transform:uppercase;}
.hpills{display:flex;align-items:center;gap:20px;}
.pill{display:flex;align-items:center;gap:8px;padding:6px 14px;
  border:1px solid var(--bdr2);border-radius:4px;
  font-size:13px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;}
.pill-dot{width:8px;height:8px;border-radius:50%;background:currentColor;
  animation:blink 1s ease-in-out infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.15}}
.pill.live{border-color:var(--sz);color:var(--sz);}
.pill.ok{border-color:var(--lrda);color:var(--lrda);}
.pill.sim{border-color:var(--lpd);color:var(--lpd);}
.hmeta{display:flex;gap:24px;}
.hm{display:flex;flex-direction:column;align-items:flex-end;}
.hml{font-size:12px;letter-spacing:.16em;color:var(--mut);text-transform:uppercase;}
.hmv{font-size:18px;font-weight:500;letter-spacing:.06em;}

main{display:grid;grid-template-columns:440px 1fr; /* MUCH wider left column to fit bigger text */
  gap:1px;background:var(--bdr);overflow:hidden;}
.col{background:var(--sur);display:flex;flex-direction:column;overflow:hidden;}
.sh{display:flex;align-items:center;gap:12px;
  padding:16px 24px 14px;border-bottom:1px solid var(--bdr);flex-shrink:0;}
.sh-line{flex:1;height:1px;background:var(--bdr);}
.sl{font-size:14px;font-weight:600;letter-spacing:.22em;text-transform:uppercase;
  color:var(--mut);white-space:nowrap;}

.state-card{margin:20px 24px 0;border:1px solid var(--bdr2);border-radius:4px;
  padding:24px;position:relative;overflow:hidden;
  transition:border-color .5s,box-shadow .5s;flex-shrink:0;}
.state-card::before{content:'';position:absolute;top:0;left:0;right:0;height:4px;
  background:var(--ca,var(--oth));transition:background .5s;}
.state-name{font-family:'Syne',sans-serif;font-weight:800;font-size:48px;
  color:var(--ca,var(--oth));transition:color .5s;line-height:1;letter-spacing:.03em;}
.state-row{display:flex;justify-content:space-between;align-items:flex-end;margin-top:12px;}
.state-conf{font-size:18px;color:var(--mut);letter-spacing:.1em;}
.state-conf b{color:var(--txt);font-weight:500;}
.risk{font-size:14px;letter-spacing:.16em;text-transform:uppercase;
  padding:6px 12px;border-radius:4px;font-weight:600;}
.risk-HIGH  {background:rgba(255,45,85,.14);color:var(--sz);}
.risk-MEDIUM{background:rgba(255,140,0,.14);color:var(--lpd);}
.risk-LOW   {background:rgba(48,209,88,.11);color:var(--lrda);}

.probs{padding:16px 24px;flex-shrink:0;}
.pb-row{display:flex;align-items:center;gap:12px;margin-bottom:12px;}
.pb-name{width:75px;font-size:16px;letter-spacing:.07em;color:var(--mut);
  text-align:right;flex-shrink:0;}
.pb-track{flex:1;height:14px;background:rgba(255,255,255,.04);
  border-radius:3px;overflow:hidden;}
.pb-fill{height:100%;border-radius:3px;transition:width .65s cubic-bezier(.4,0,.2,1);}
.pb-pct{width:55px;font-size:16px;text-align:right;color:var(--txt);flex-shrink:0;}

.llm-box { margin: 0 24px 20px; padding: 18px 20px; background: rgba(255,255,255,0.02); border: 1px solid var(--bdr2); border-radius: 4px; flex-shrink: 0;}
.llm-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.llm-title { font-size: 14px; font-weight: 600; letter-spacing: .12em; color: var(--mut); text-transform: uppercase; }
.btn-gen { background: var(--sz); color: #fff; border: none; padding: 8px 16px; border-radius: 4px; font-family: inherit; font-size: 13px; cursor: pointer; text-transform: uppercase; font-weight: 700; letter-spacing: .1em; transition: opacity 0.2s; }
.btn-gen:hover { opacity: 0.8; }
.btn-gen:disabled { background: var(--mut); cursor: not-allowed; }
.llm-text { font-size: 18px; color: var(--txt); line-height: 1.5; min-height: 54px;}

.file-info{margin:0 24px 20px;padding:16px 20px;
  border:1px solid var(--bdr2);border-radius:4px;flex-shrink:0;}
.fi-row{display:flex;justify-content:space-between;
  font-size:16px;color:var(--mut);letter-spacing:.08em;margin-bottom:10px;}
.fi-row b{color:var(--txt);}
.prog-track{height:6px;background:rgba(255,255,255,.05);border-radius:3px;overflow:hidden;}
.prog-fill{height:100%;border-radius:3px;transition:width .6s ease,background .5s;}

.eeg-wrap{flex-shrink:0;padding:16px 24px 16px;height:280px;} 
#eegCanvas{width:100%;height:100%;display:block;}
.stats{display:grid;grid-template-columns:repeat(4,1fr);
  gap:1px;background:var(--bdr);
  border-top:1px solid var(--bdr);border-bottom:1px solid var(--bdr);flex-shrink:0;}
.sb{background:var(--sur);padding:16px 24px;}
.sv{font-family:'Syne',sans-serif;font-weight:700;font-size:28px;color:#fff;}
.sk{font-size:13px;letter-spacing:.16em;text-transform:uppercase;color:var(--mut);margin-top:4px;}

/* THE FIX: SpecID gets 1fr space. Class, Conf, and Bar are grouped on the right */
.log-hdr{display:grid;grid-template-columns:100px 1fr 120px 80px 100px;
  padding:14px 24px;border-bottom:1px solid var(--bdr2);flex-shrink:0;
  font-size:14px;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:var(--mut);}
.log-wrap{flex:1;overflow-y:auto;min-height:0;
  scrollbar-width:thin;scrollbar-color:var(--bdr2) transparent;}
.log-row{display:grid;grid-template-columns:100px 1fr 120px 80px 100px;
  padding:14px 24px;border-bottom:1px solid var(--bdr);
  align-items:center;animation:si .3s ease;}
@keyframes si{from{opacity:0;transform:translateX(5px)}to{opacity:1;transform:translateX(0)}}
.lr-time{font-size:16px;color:var(--mut);}
.lr-id{font-size:16px;color:var(--mut);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.cls-tag{font-size:14px;letter-spacing:.12em;font-weight:600;text-transform:uppercase;
  padding:4px 10px;border-radius:4px;}
.lr-conf{font-size:16px;text-align:right;color:var(--txt);}
.lr-bar-wrap{padding:0 8px;}
.lr-bar{height:8px;border-radius:4px;}

footer{display:flex;align-items:center;justify-content:space-between;
  padding:0 30px;border-top:1px solid var(--bdr);background:var(--sur);
  font-size:13px;letter-spacing:.13em;text-transform:uppercase;color:var(--mut);}
#cdot{width:8px;height:8px;border-radius:50%;background:var(--mut);
  display:inline-block;margin-right:8px;transition:background .3s;}
#cdot.on{background:var(--lrda);}
#cdot.err{background:var(--sz);}
</style>
</head>
<body>
<div id="app">

<header>
  <div class="brand">
    <div class="pulse"></div>
    <div>
      <div class="brand-name">HMS · EEG Live Classification</div>
      <div class="brand-sub">HybridLSTM · EfficientNet-B0 + BiLSTM(256×2) · 6 Classes</div>
    </div>
  </div>
  <div class="hpills">
    <div class="hmeta">
      <div class="hm"><span class="hml">Elapsed</span><span class="hmv" id="hElapsed">00:00:00</span></div>
      <div class="hm"><span class="hml">File</span><span class="hmv" id="hFile">0 / 0</span></div>
    </div>
    <div class="pill live"><div class="pill-dot"></div>Streaming</div>
    <div class="pill sim" id="srcPill"><div class="pill-dot"></div><span id="srcTxt">—</span></div>
  </div>
</header>

<main>
  <div class="col">
    <div class="sh"><span class="sl">Current Prediction</span><div class="sh-line"></div></div>
    <div class="state-card" id="stateCard">
      <div class="state-name" id="stateName">—</div>
      <div class="state-row">
        <div class="state-conf">Confidence&nbsp;<b id="stateConf">—</b>%</div>
        <div class="risk risk-LOW" id="riskBadge">LOW</div>
      </div>
    </div>

    <div class="sh" style="margin-top:20px;">
      <span class="sl">Class Probabilities · All 6 Classes</span><div class="sh-line"></div>
    </div>
    <div class="probs" id="probs"></div>

    <div class="llm-box">
      <div class="llm-header">
        <span class="llm-title">Summary Report</span>
        <button class="btn-gen" id="btnGen" onclick="generateSummary()">Generate New</button>
      </div>
      <div class="llm-text" id="llmText">Click generate to translate the current streaming frame into plain English.</div>
    </div>

    <div class="sh"><span class="sl">File Progress</span><div class="sh-line"></div></div>
    <div class="file-info">
      <div class="fi-row">
        <span>Spec ID&nbsp;<b id="specId">—</b></span>
        <span><b id="fileIdx">0</b>&nbsp;/&nbsp;<b id="fileTotal">0</b></span>
      </div>
      <div class="prog-track">
        <div class="prog-fill" id="progFill" style="width:0%"></div>
      </div>
    </div>
  </div>

  <div class="col">
    <div class="sh"><span class="sl">EEG Signal · Visualisation</span><div class="sh-line"></div></div>
    <div class="eeg-wrap"><canvas id="eegCanvas"></canvas></div>

    <div class="stats">
      <div class="sb"><div class="sv" id="stInf">0000</div><div class="sk">Inferences</div></div>
      <div class="sb"><div class="sv" id="stCls" style="font-size:22px">—</div><div class="sk">Last Class</div></div>
      <div class="sb"><div class="sv" id="stConf">—</div><div class="sk">Confidence</div></div>
      <div class="sb"><div class="sv" id="stRisk">—</div><div class="sk">Risk Level</div></div>
    </div>

    <div class="sh"><span class="sl">Classification Log · Real-Time</span><div class="sh-line"></div></div>
    <div class="log-hdr">
      <span>Time</span><span>Spec ID</span><span>Class</span>
      <span style="text-align:right">Confidence</span><span></span>
    </div>
    <div class="log-wrap" id="logWrap"></div>
  </div>
</main>

<footer>
  <div><span id="cdot"></span><span id="ctext">Connecting…</span></div>
  <div>Architecture · EfficientNet-B0 → pool → BiLSTM(256) → Linear(6) · Loss: KLDivLoss</div>
  <div id="lastTs">—</div>
</footer>
</div>

<script>
const COLORS={Seizure:'#ff2d55',LPD:'#ff8c00',GPD:'#f5c518',
              LRDA:'#30d158',GRDA:'#a259ff',Other:'#32ade6'};
const CLASSES=['Seizure','LPD','GPD','LRDA','GRDA','Other'];
let currentProbabilities = {};

const eegCanvas=document.getElementById('eegCanvas');
const ctx=eegCanvas.getContext('2d');
const NCH=8;
const CNAMES=['Fp1','Fp2','C3','C4','P3','P4','O1','O2'];
const CF=[8.2,11.1,6.7,9.4,13.0,7.5,10.3,5.9];
const CPH=CF.map(()=>Math.random()*Math.PI*2);
let eegT=0,curCls='Other';

function resz(){
  eegCanvas.width=eegCanvas.offsetWidth;
  eegCanvas.height=eegCanvas.offsetHeight;
}
resz();window.addEventListener('resize',resz);

function drawEEG(){
  const w=eegCanvas.width,h=eegCanvas.height;
  ctx.clearRect(0,0,w,h);
  const rH=h/NCH;
  for(let c=0;c<NCH;c++){
    const yC=(c+.5)*rH,amp=rH*.36;
    const col=Object.values(COLORS)[c%6];
    ctx.beginPath();ctx.strokeStyle=col;ctx.lineWidth=1.5;ctx.globalAlpha=.8;
    for(let x=0;x<w;x++){
      const t=(x/w)+eegT;
      let y=Math.sin(t*CF[c]*Math.PI*2+CPH[c])*amp;
      y+=Math.sin(t*CF[c]*1.6*Math.PI*2+CPH[c]*.5)*amp*.2;
      y+=(Math.random()-.5)*amp*.08;
      if(curCls==='Seizure'&&c<3) y*=1.9+Math.abs(Math.sin(t*28+c))*1.1;
      else if(curCls==='LPD')     y*=1+Math.sin(t*8)*.5;
      else if(curCls==='GRDA')    y=c%2===0?y*1.4:y*.5;
      x===0?ctx.moveTo(x,yC+y):ctx.lineTo(x,yC+y);
    }
    ctx.stroke();ctx.globalAlpha=1;
    ctx.fillStyle='#3d5a72';ctx.font='14px "IBM Plex Mono"'; 
    ctx.fillText(CNAMES[c],8,c*rH+18);
  }
}
(function loop(){requestAnimationFrame(loop);eegT+=.0025;drawEEG();})();

(function(){
  const el=document.getElementById('probs');
  el.innerHTML=CLASSES.map(c=>`
    <div class="pb-row">
      <span class="pb-name">${c}</span>
      <div class="pb-track">
        <div class="pb-fill" id="pf-${c}" style="width:0%;background:${COLORS[c]}"></div>
      </div>
      <span class="pb-pct" id="pp-${c}">0%</span>
    </div>`).join('');
})();

let infCount=0;

function update(d){
  curCls=d.predicted;
  currentProbabilities=d.probabilities; 
  const col=COLORS[d.predicted]||'#32ade6';

  const card=document.getElementById('stateCard');
  card.style.setProperty('--ca',col);
  card.style.borderColor=col+'44';
  card.style.boxShadow=`0 0 20px ${col}1a`;
  document.getElementById('stateName').textContent=d.predicted;
  document.getElementById('stateConf').textContent=d.confidence;
  const rb=document.getElementById('riskBadge');
  rb.textContent=d.risk;rb.className=`risk risk-${d.risk}`;

  CLASSES.forEach(c=>{
    const v=d.probabilities[c]||0;
    document.getElementById(`pf-${c}`).style.width=v+'%';
    document.getElementById(`pp-${c}`).textContent=v.toFixed(1)+'%';
  });

  document.getElementById('specId').textContent=d.spec_id;
  document.getElementById('fileIdx').textContent=d.file_index;
  document.getElementById('fileTotal').textContent=d.total_files;
  const pct=d.total_files>0?(d.file_index/d.total_files*100):0;
  const pf=document.getElementById('progFill');
  pf.style.width=pct+'%';pf.style.background=col;

  document.getElementById('hElapsed').textContent=d.elapsed;
  document.getElementById('hFile').textContent=`${d.file_index} / ${d.total_files}`;

  const pill=document.getElementById('srcPill'),stxt=document.getElementById('srcTxt');
  if(d.source==='model'){pill.className='pill ok';stxt.textContent='LSTM Model';}
  else{pill.className='pill sim';stxt.textContent='Simulation';}

  infCount++;
  document.getElementById('stInf').textContent=String(infCount).padStart(4,'0');
  const sc=document.getElementById('stCls');
  sc.textContent=d.predicted;sc.style.color=col;
  const sf=document.getElementById('stConf');
  sf.textContent=d.confidence+'%';sf.style.color=col;
  document.getElementById('stRisk').textContent=d.risk;

  const wrap=document.getElementById('logWrap');
  const row=document.createElement('div');
  row.className='log-row';
  row.innerHTML=`
    <span class="lr-time">${d.timestamp}</span>
    <span class="lr-id" title="${d.spec_id}">${d.spec_id}</span>
    <span><span class="cls-tag"
      style="background:${col}22;color:${col};border:1px solid ${col}44">
      ${d.predicted}</span></span>
    <span class="lr-conf">${d.confidence}%</span>
    <span class="lr-bar-wrap">
      <div class="lr-bar" style="width:${Math.round(d.confidence)}%;background:${col}"></div>
    </span>`;
  wrap.prepend(row);
  while(wrap.children.length>60)wrap.removeChild(wrap.lastChild);

  document.getElementById('lastTs').textContent='Last · '+d.timestamp;
}

function connect(){
  const ws=new WebSocket(`ws://${window.location.host}/ws`);
  const dot=document.getElementById('cdot'),ct=document.getElementById('ctext');
  ws.onopen =()=>{dot.className='on'; ct.textContent='Connected · WebSocket';};
  ws.onclose=()=>{dot.className='err';ct.textContent='Disconnected · retrying…';setTimeout(connect,2500);};
  ws.onerror=()=>{dot.className='err';ct.textContent='Connection error';};
  ws.onmessage=(e)=>update(JSON.parse(e.data));
}
connect();

async function generateSummary() {
  const btn = document.getElementById('btnGen');
  const txt = document.getElementById('llmText');
  btn.disabled = true;
  btn.textContent = "Translating...";
  txt.textContent = "Analyzing current brain wave probabilities...";
  try {
    const response = await fetch('/api/summarize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ probabilities: currentProbabilities })
    });
    if (!response.ok) throw new Error("Network response was not ok");
    const data = await response.json();
    txt.textContent = data.summary;
  } catch (error) {
    txt.textContent = "Error generating report. Check console and API Key.";
    console.error("LLM Fetch Error:", error);
  } finally {
    btn.disabled = false;
    btn.textContent = "Generate New";
  }
}
</script>
</body>
</html>"""

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI ROUTES
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI()

@app.get("/")
async def root():
    return HTMLResponse(content=DASHBOARD_HTML)

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            result = streamer.next()
            await websocket.send_text(json.dumps(result))
            await asyncio.sleep(STREAM_INTERVAL)
    except (WebSocketDisconnect, Exception) as e:
        print(f"WebSocket closed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# NEW LLM LAYER API ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
class SummaryRequest(BaseModel):
    probabilities: dict

@app.post("/api/summarize")
async def summarize_probabilities(req: SummaryRequest):
    if not LLM_READY:
        return {"summary": "LLM API is not configured. Please export GEMINI_API_KEY in your server environment."}
    
    try:
        sys_instruct = (
            "You are a medical translator. Convert raw EEG classification probabilities "
            "into a simple, two-sentence maximum explanation for a layperson. "
            "Do NOT diagnose or give medical advice. Only state what the data indicates."
        )
        
        prompt = f"The CNN-LSTM model output probabilities are: {req.probabilities}. Please summarize."
        
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_instruct,
                temperature=0.2, 
            )
        )
        return {"summary": response.text}
        
    except Exception as e:
        return {"summary": f"Error generating summary: {str(e)}"}

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print("HMS EEG LIVE DASHBOARD + LLM")
    print("=" * 65)
    print(f"  Model  : {'✅  ' + MODEL_PATH if MODEL_LOADED else '⚠️   Not found — simulation mode'}")
    print(f"  LLM    : {'✅  Ready' if LLM_READY else '⚠️   API Key Missing'}")
    print(f"  Data   : {streamer.total} spectrogram files in '{SPECTROGRAM_DIR}/'")
    print(f"  Classes: {', '.join(CLASSES)}")
    print(f"  Device : {DEVICE}")
    print("=" * 65)
    print("\n  🌐  Open  http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)