Dataset: https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data  
**Objective**: To build a platform that streams (simulated) live biosignals to classify different types of harmful brain activities in real-time and visualize them along with the performance report on a dashboard. 

**Overview**:

This project analyzes EEG data to classify abnormal brain activity (e.g., seizures) using a CNN + LSTM model.
It provides a web-based dashboard where users can upload EEG files and receive:
  * Predicted brain activity class
  * Confidence score
  * Severity level
  * Human-readable explanation (via LLM)

**Architecture:**
Input: EEG Spectrogram (128 × 256)  
Spatial Feature Extraction: CNN (frozen layers until 3 epochs)  
Capture Temporal dependencies: LSTM  
Output: 6-class classification  

* User (Upload spectrogram .parquet file) -> Frontend Dashboard (HTML/CSS/JS)  -> Backend API (FastAPI) -> Preprocessing (Spectrograms) -> ML Model (PyTorch CNN + LSTM) -> Prediction Output (JSON: class, confidence, risk) -> LLM Layer (Readable Explanation) -> Dashboard Visualization -> Deployment (Docker → Google Cloud Run)  
 

**Feature:**  
* Upload EEG spectrograms .parquet files  
* Preprocessing: parquet -> image-like tensors  
* Deep Learning Model (CNN + LSTM)  
* Prediction outputs: Class label (6 categories)  
  Seizure  
  LPD (Lateralized Periodic Discharges)  
  GPD (Generalized Periodic Discharges)  
  LRDA  
  GRDA  
  Other  
* Model Confidence score  
* Severity level (LOW / MEDIUM / HIGH / UNCERTAIN) based on the predicted class and severity.
* LLM integration for simplified explanations  
* Interactive dashboard UI
* Dockerized application
* Deployed on Google Cloud Run

**Tech Stack:**  
Backend: FastAPI  
Frontend: HTML, CSS, JavaScript  
Deep Learning: PyTorch  
Data Processing: Pandas, NumPy  
File Format: Parquet  
Containerization: Docker   
Cloud Deployment: Google Cloud Run  
LLM Integration: Gemini API  

**Project Structure:**    
├── app/  
│ ├── main.py # FastAPI backend  
│ ├── model.py # Model training & evaluation  
│ ├── preprocessing.py # Spectrograms to tensors  
│ └── llm.py # Gemini API integration  
│  
├── frontend/  
│ ├── index.html  
│ ├── styles.css  
│ └── script.js  
│  
├── models/  
│ └── cnn_lstm_model.pt  
│  
├── data/  
│ └── sample_spectrograms/  
│
├── Dockerfile  
├── requirements.txt  
└── README.md  

### Installation & Setup  
1. Clone the repository    
2. Create a Virtual environment.  
       python -m venv venv  
       source venv/bin/activate   # Mac/Linux  
       venv\Scripts\activate      # Windows  
 
3. Install dependencies: pip install -r requirements.txt  
4. Environmental Variables:  
     export GEMINI_API_KEY=your_api_key   # Mac/Linux  
     set GEMINI_API_KEY=your_api_key      # Windows  
5. Run Locally: uvicorn app.main:app --reload  
6. Open in browser: http://localhost:8000  
7. Docker Deployment:  
   1. Build image: docker build -t eeg-dashboard .  
   2. Run Container: docker run -p 8000:8000 eeg-dashboard  
9. Google Cloud Deployment: gcloud builds submit --tag us-central1-docker.pkg.dev/PROJECT-ID/ml-deploy/eeg-dashboard  
10. gcloud run deploy eeg-dashboard \  
  --image us-central1-docker.pkg.dev/PROJECT-ID/ml-deploy/eeg-dashboard \  
  --platform managed \  
  --region us-central1 \  
  --allow-unauthenticated \  
  --set-env-vars GEMINI_API_KEY=YOUR_API_KEY  

11. Sample Output:  
    {  
  "filename": "sample.parquet",  
  "predicted": "GPD",  
  "confidence": 59.1,  
  "risk": "LOW",  
  "probabilities": {  
    "Seizure": 7.2,  
    "LPD": 1.6,  
    "GPD": 59.1,  
    "LRDA": 0.5,  
    "GRDA": 1.9,  
    "Other": 29.7  
  }  
}  
  





