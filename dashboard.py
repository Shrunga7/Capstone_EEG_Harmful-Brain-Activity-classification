"""
Real-time EEG Dashboard with Trained CNN+LSTM Model
"""

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import asyncio
import torch
import numpy as np
import json
from datetime import datetime, timedelta
import os

from model import create_model

app = FastAPI()

# Global model and configuration
MODEL = None
MODEL_INFO = None
DEVICE = None

def load_trained_model():
    """Load the trained model"""
    global MODEL, MODEL_INFO, DEVICE
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model info
    info_path = 'models/model_info.json'
    if not os.path.exists(info_path):
        print("⚠️  No trained model found. Using simulation mode.")
        return False
    
    with open(info_path, 'r') as f:
        MODEL_INFO = json.load(f)
    
    # Create and load model
    MODEL = create_model(
        n_channels=MODEL_INFO['n_channels'],
        seq_length=MODEL_INFO['seq_length'],
        n_classes=MODEL_INFO['n_classes']
    ).to(DEVICE)
    
    # Load weights
    checkpoint = torch.load('models/best_model.pth', map_location=DEVICE)
    MODEL.load_state_dict(checkpoint['model_state_dict'])
    MODEL.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   Device: {DEVICE}")
    print(f"   Classes: {MODEL_INFO['class_names']}")
    return True

# Try to load model at startup
MODEL_LOADED = load_trained_model()

class EEGDataGenerator:
    """Generate or process EEG data for dashboard"""
    
    def __init__(self):
        self.session_id = "ID-001"
        self.recording_id = "REC-154"
        self.start_time = datetime.now()
        self.alerts = []
        self.use_model = MODEL_LOADED
        
        if self.use_model:
            self.class_names = MODEL_INFO['class_names']
            self.n_channels = MODEL_INFO['n_channels']
            self.seq_length = MODEL_INFO['seq_length']
        else:
            self.class_names = ['Seizure', 'LPD', 'GPD', 'LRDA', 'Other']
            self.n_channels = 19
            self.seq_length = 2000
    
    def get_model_predictions(self, eeg_data=None):
        """
        Get predictions from trained model
        
        Args:
            eeg_data: EEG data of shape (n_channels, seq_length)
                     If None, generates random data for demo
        
        Returns:
            Dictionary with predictions and probabilities
        """
        if not self.use_model:
            return self.get_simulated_predictions()
        
        # Generate random EEG data if not provided (for demo)
        if eeg_data is None:
            eeg_data = np.random.randn(self.n_channels, self.seq_length).astype(np.float32)
        
        # Prepare data for model
        with torch.no_grad():
            data_tensor = torch.FloatTensor(eeg_data).unsqueeze(0).to(DEVICE)
            probs = MODEL.predict_proba(data_tensor)
            probs_np = probs.cpu().numpy()[0]
        
        # Get predicted class
        predicted_class = np.argmax(probs_np)
        confidence = probs_np[predicted_class] * 100
        
        # Create probabilities dictionary
        probabilities = {name: float(prob * 100) for name, prob in zip(self.class_names, probs_np)}
        
        # Determine risk level
        if predicted_class == 0:  # Seizure
            risk = "HIGH"
        elif predicted_class in [1, 2]:  # LPD, GPD
            risk = "MEDIUM"
        else:
            risk = "LOW"
        
        return {
            "state": self.class_names[predicted_class],
            "confidence": int(confidence),
            "risk": risk,
            "probabilities": probabilities,
            "predicted_class": int(predicted_class)
        }
    
    def get_simulated_predictions(self):
        """Fallback simulation when model is not available"""
        import random
        
        states = [
            {"state": "Seizure", "confidence": random.randint(75, 95), "risk": "HIGH"},
            {"state": "LPD", "confidence": random.randint(70, 90), "risk": "MEDIUM"},
            {"state": "Normal", "confidence": random.randint(85, 99), "risk": "LOW"},
            {"state": "GPD", "confidence": random.randint(65, 85), "risk": "MEDIUM"},
        ]
        prediction = random.choice(states)
        
        # Generate probabilities
        probs = {
            "Seizure": random.randint(1, 95) if prediction["state"] == "Seizure" else random.randint(1, 30),
            "LPD": random.randint(1, 90) if prediction["state"] == "LPD" else random.randint(1, 30),
            "GPD": random.randint(1, 85) if prediction["state"] == "GPD" else random.randint(1, 20),
            "LRDA": random.randint(1, 25),
            "Other": random.randint(1, 15)
        }
        
        # Normalize
        total = sum(probs.values())
        probabilities = {k: round(v/total * 100) for k, v in probs.items()}
        
        prediction["probabilities"] = probabilities
        return prediction
    
    def get_timeline_data(self):
        """Generate timeline data"""
        timeline = []
        current_time = datetime.now()
        
        for i in range(60):  # Show last 60 seconds
            time_point = current_time - timedelta(seconds=60-i)
            
            # Get prediction for this time point
            pred = self.get_model_predictions() if i % 5 == 0 else None
            if pred:
                event_type = pred["state"]
            else:
                event_type = timeline[-1]["event"] if timeline else "Other"
            
            timeline.append({
                "time": time_point.strftime("%H:%M:%S"),
                "event": event_type
            })
        
        return timeline
    
    def update_alerts(self, current_prediction):
        """Update alerts based on current prediction"""
        if current_prediction["state"] in ["Seizure", "LPD", "GPD"]:
            # Add new alert
            alert = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "event": current_prediction["state"],
                "duration": f"{np.random.randint(10, 90)}s",
                "probability": f"{current_prediction['confidence']}%"
            }
            self.alerts.insert(0, alert)
            
            # Keep only last 10 alerts
            self.alerts = self.alerts[:10]
    
    def get_elapsed_time(self):
        """Get elapsed recording time"""
        elapsed = datetime.now() - self.start_time
        hours = int(elapsed.total_seconds() // 3600)
        minutes = int((elapsed.total_seconds() % 3600) // 60)
        seconds = int(elapsed.total_seconds() % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

generator = EEGDataGenerator()

@app.get("/")
async def get_dashboard():
    """Serve the dashboard HTML"""
    
    # Read the HTML from previous main.py (simplified here for brevity)
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>EEG Seizure Detection Dashboard - Live Model</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #1a1d2e;
                color: #fff;
                overflow: hidden;
            }
            .header {
                background: #0f1117;
                padding: 15px 30px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 2px solid #2a2d3e;
            }
            .header-info {
                display: flex;
                gap: 25px;
                font-size: 14px;
            }
            .status-live {
                color: #ff4444;
                font-weight: bold;
            }
            .model-status {
                color: #4CAF50;
                font-weight: bold;
            }
            .dashboard-container {
                display: grid;
                grid-template-columns: 280px 1fr 350px;
                gap: 20px;
                padding: 20px;
                height: calc(100vh - 70px);
            }
            .panel {
                background: #242736;
                border-radius: 8px;
                padding: 20px;
                overflow-y: auto;
            }
            .panel-title {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 15px;
            }
            .current-state {
                padding: 15px;
                border-radius: 6px;
                margin-bottom: 15px;
                text-align: center;
            }
            .state-label {
                font-size: 20px;
                font-weight: bold;
                margin-bottom: 10px;
            }
            .state-info {
                font-size: 14px;
                opacity: 0.9;
            }
            .probabilities { margin-top: 20px; }
            .prob-item {
                display: flex;
                align-items: center;
                margin-bottom: 10px;
            }
            .prob-bar {
                height: 25px;
                border-radius: 4px;
                display: flex;
                align-items: center;
                padding: 0 10px;
                font-size: 12px;
                font-weight: bold;
                margin-right: 10px;
            }
            .prob-value {
                font-size: 14px;
                font-weight: bold;
            }
            .timeline {
                background: #1a1d2e;
                padding: 20px;
                border-radius: 8px;
                height: 100%;
            }
            .timeline-chart {
                height: 60px;
                background: #0f1117;
                border-radius: 4px;
                padding: 10px;
                display: flex;
                align-items: center;
                gap: 1px;
                margin-top: 10px;
            }
            .timeline-bar {
                flex: 1;
                height: 40px;
                border-radius: 2px;
            }
            .timeline-legend {
                display: flex;
                gap: 15px;
                margin-top: 15px;
                font-size: 12px;
                flex-wrap: wrap;
            }
            .legend-item {
                display: flex;
                align-items: center;
                gap: 5px;
            }
            .legend-color {
                width: 20px;
                height: 12px;
                border-radius: 2px;
            }
            .alerts-table {
                width: 100%;
                margin-top: 15px;
                font-size: 13px;
            }
            .alerts-table th {
                text-align: left;
                padding: 10px 5px;
                border-bottom: 1px solid #3a3d4e;
                font-weight: 600;
            }
            .alerts-table td {
                padding: 12px 5px;
                border-bottom: 1px solid #2a2d3e;
            }
            .event-badge {
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
                display: inline-block;
            }
            .spectrogram {
                background: #0f1117;
                height: 200px;
                border-radius: 6px;
                margin-top: 15px;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
                position: relative;
            }
            .seizure { background: #d32f2f; }
            .lpd { background: #f57c00; }
            .gpd { background: #fbc02d; }
            .lrda { background: #7cb342; }
            .other { background: #42a5f5; }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="header-info">
                <span><strong>Session:</strong> <span id="session-id">ID-001</span></span>
                <span><strong>Recording:</strong> <span id="recording-id">REC-154</span></span>
                <span><strong>Time:</strong> <span id="elapsed-time">00:00:00</span></span>
                <span><strong>Model:</strong> CNN + LSTM <span id="model-status" class="model-status">LOADED</span></span>
                <span><strong>Status:</strong> <span class="status-live">LIVE</span></span>
            </div>
        </div>
        
        <div class="dashboard-container">
            <div class="panel">
                <div class="panel-title">Current Brain State</div>
                <div class="current-state" id="current-state">
                    <div class="state-label" id="state-label">Loading...</div>
                    <div class="state-info">
                        Confidence: <span id="confidence">--</span>%<br>
                        Risk Level: <span id="risk-level">--</span>
                    </div>
                </div>
                
                <div class="panel-title">Class Probabilities</div>
                <div class="probabilities" id="probabilities"></div>
                
                <div class="panel-title">Live Spectrogram</div>
                <div class="spectrogram">
                    <canvas id="spectrogram-canvas" width="240" height="180"></canvas>
                </div>
            </div>
            
            <div class="timeline">
                <div class="panel-title">Live Activity Timeline</div>
                <div style="font-size: 12px; color: #888; margin-bottom: 10px;">Last 60 seconds</div>
                <div class="timeline-chart" id="timeline-chart"></div>
                <div class="timeline-legend">
                    <div class="legend-item">
                        <div class="legend-color seizure"></div>
                        <span>Seizure</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color lpd"></div>
                        <span>LPD</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color gpd"></div>
                        <span>GPD</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color lrda"></div>
                        <span>LRDA</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color other"></div>
                        <span>Other</span>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-title">Recent Alerts</div>
                <table class="alerts-table">
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Event</th>
                            <th>Duration</th>
                            <th>Prob.</th>
                        </tr>
                    </thead>
                    <tbody id="alerts-body"></tbody>
                </table>
            </div>
        </div>
        
        <script>
            const ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };
            
            function updateDashboard(data) {
                document.getElementById('session-id').textContent = data.session_id;
                document.getElementById('recording-id').textContent = data.recording_id;
                document.getElementById('elapsed-time').textContent = data.elapsed_time;
                
                const stateDiv = document.getElementById('current-state');
                const state = data.current_state;
                
                document.getElementById('state-label').textContent = state.state;
                document.getElementById('confidence').textContent = state.confidence;
                document.getElementById('risk-level').textContent = state.risk;
                
                const stateClass = state.state.toLowerCase().replace(' ', '-').split('/')[0];
                stateDiv.className = 'current-state ' + stateClass;
                
                const probsDiv = document.getElementById('probabilities');
                probsDiv.innerHTML = '';
                
                const classColors = {
                    'Seizure': '#d32f2f',
                    'LPD': '#f57c00',
                    'GPD': '#fbc02d',
                    'LRDA': '#7cb342',
                    'Other': '#42a5f5'
                };
                
                for (const [className, prob] of Object.entries(data.probabilities)) {
                    const item = document.createElement('div');
                    item.className = 'prob-item';
                    item.innerHTML = `
                        <div class="prob-bar" style="width: ${prob}%; background: ${classColors[className]}">
                            ${className}
                        </div>
                        <div class="prob-value">${Math.round(prob)}%</div>
                    `;
                    probsDiv.appendChild(item);
                }
                
                const timelineChart = document.getElementById('timeline-chart');
                timelineChart.innerHTML = '';
                
                const eventColors = {
                    'Seizure': '#d32f2f',
                    'LPD': '#f57c00',
                    'GPD': '#fbc02d',
                    'LRDA': '#7cb342',
                    'Other': '#42a5f5'
                };
                
                for (const point of data.timeline) {
                    const bar = document.createElement('div');
                    bar.className = 'timeline-bar';
                    bar.style.background = eventColors[point.event] || '#42a5f5';
                    timelineChart.appendChild(bar);
                }
                
                const alertsBody = document.getElementById('alerts-body');
                alertsBody.innerHTML = '';
                
                data.alerts.forEach(alert => {
                    const row = document.createElement('tr');
                    const eventClass = alert.event.toLowerCase().replace(' ', '-');
                    row.innerHTML = `
                        <td>${alert.time}</td>
                        <td><span class="event-badge ${eventClass}">${alert.event}</span></td>
                        <td>${alert.duration}</td>
                        <td>${alert.probability}</td>
                    `;
                    alertsBody.appendChild(row);
                });
                
                updateSpectrogram();
            }
            
            function updateSpectrogram() {
                const canvas = document.getElementById('spectrogram-canvas');
                const ctx = canvas.getContext('2d');
                
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                ctx.putImageData(imageData, -2, 0);
                
                for (let y = 0; y < canvas.height; y++) {
                    const intensity = Math.random() * Math.sin(y / 10) * 255;
                    const hue = 60 - (intensity / 255 * 60);
                    ctx.fillStyle = `hsl(${hue}, 100%, ${intensity / 5}%)`;
                    ctx.fillRect(canvas.width - 2, y, 2, 1);
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    
    try:
        while True:
            # Get model predictions (or simulated data)
            prediction = generator.get_model_predictions()
            
            # Update alerts
            generator.update_alerts(prediction)
            
            # Prepare data
            data = {
                "session_id": generator.session_id,
                "recording_id": generator.recording_id,
                "elapsed_time": generator.get_elapsed_time(),
                "current_state": {
                    "state": prediction["state"],
                    "confidence": prediction["confidence"],
                    "risk": prediction["risk"]
                },
                "probabilities": prediction["probabilities"],
                "timeline": generator.get_timeline_data(),
                "alerts": generator.alerts
            }
            
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 70)
    print("STARTING EEG DASHBOARD")
    print("=" * 70)
    if MODEL_LOADED:
        print("✅ Running with trained CNN+LSTM model")
        print(f"   Model classes: {MODEL_INFO['class_names']}")
        print(f"   Test accuracy: {MODEL_INFO.get('test_accuracy', 'N/A')}")
    else:
        print("⚠️  Running in SIMULATION mode (no trained model found)")
        print("   Train a model first using: python train_model.py")
    print("=" * 70)
    print("\n🌐 Open http://localhost:8000 in your browser\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
