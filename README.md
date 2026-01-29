Dataset: https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data  
**Objective**: To build a platform that streams (simulated) live biosignals to classify different types of harmful brain activities in real-time and visualize them along with the performance report on a dashboard. 

**Architecture:**

Computer Vision/Signal Processing: Process Spectrograms (2D images) of EEG readings with CNNs to classify the state of the brain activity. 
Sequence Modeling: Use LSTM for the time-series analysis of the temporal sequence of the cognitive states. 

**Cloud & MLOps:**

It requires a streaming pipeline to continuously feed the spectrograms to a CNN model and an API deployment (Docker/FastAPI). 

**The User Interface:**

Build a live Streamlit or React Dashboard that updates the classification results in real-time.

** Testing my pull request **

