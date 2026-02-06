# EEG Seizure Detection System with CNN+LSTM

Complete end-to-end system for EEG seizure detection using deep learning and real-time dashboard visualization.

## 📋 Overview

This system includes:
- **CNN+LSTM deep learning model** for EEG classification
- **Automated data preprocessing** for parquet files
- **Model training pipeline** with early stopping and learning rate scheduling
- **Real-time dashboard** using FastAPI and WebSockets
- **5-class classification**: Seizure, LPD, GPD, LRDA, Other

## 📁 Project Structure

```
eeg-dashboard/
├── inspect_data.py       # Analyze your parquet file structure
├── data_loader.py        # Data loading and preprocessing
├── model.py              # CNN+LSTM model architecture
├── train_model.py        # Training script
├── dashboard.py          # Real-time dashboard with trained model
├── requirements_full.txt # All dependencies
└── models/               # Saved models (created after training)
    ├── best_model.pth
    ├── model_info.json
    └── training_curves.png
```

## 🚀 Step-by-Step Guide

### Step 1: Setup Environment

Create project folder and navigate to it:
```bash
# Windows
mkdir C:\eeg-dashboard
cd C:\eeg-dashboard

# Mac/Linux
mkdir ~/eeg-dashboard
cd ~/eeg-dashboard
```

Download all the Python files I created and place them in this folder.

### Step 2: Install Dependencies

```bash
# Use pip3 if pip doesn't work
python3 -m pip install -r requirements_full.txt

# Or install individually if you have issues:
python3 -m pip install torch numpy pandas pyarrow scikit-learn matplotlib tqdm fastapi uvicorn
```

**Note**: If you don't have a GPU, PyTorch will automatically use CPU (which is fine for inference).

### Step 3: Prepare Your Data

Place your parquet file(s) in the project folder:
```
eeg-dashboard/
├── 1000086677.parquet    # Your data file
├── inspect_data.py
├── ...
```

### Step 4: Inspect Your Data (IMPORTANT!)

Run the data inspection script to understand your data structure:

```bash
python3 inspect_data.py
```

This will show you:
- Number of EEG channels
- Data shape and structure
- Column names
- Potential label columns
- Recommended configuration

**Important**: Review the output carefully. If the script can't automatically detect your data structure, you may need to modify the `label_column` parameter in the training script.

### Step 5: Train the Model

Edit `train_model.py` and update the file path if needed:
```python
PARQUET_FILE = "1000086677.parquet"  # Update this to your file name
```

Then run training:
```bash
python3 train_model.py
```

Training will:
- Load and preprocess your data
- Create train/validation/test splits
- Train the CNN+LSTM model
- Save the best model based on validation loss
- Generate training curves
- Show final test accuracy

**Expected time**: 10-60 minutes depending on data size and hardware

Training output will be saved in `models/` folder:
- `best_model.pth` - Trained model weights
- `model_info.json` - Model configuration and metrics
- `training_curves.png` - Training/validation curves

### Step 6: Run the Dashboard

Start the real-time dashboard:
```bash
python3 dashboard.py
```

You should see:
```
STARTING EEG DASHBOARD
======================================================================
✅ Running with trained CNN+LSTM model
   Model classes: ['Seizure', 'LPD', 'GPD', 'LRDA', 'Other']
   Test accuracy: 92.5
======================================================================

🌐 Open http://localhost:8000 in your browser
```

Open your browser and go to: **http://localhost:8000**

The dashboard will show:
- Real-time predictions from your trained model
- Class probabilities
- Activity timeline
- Recent alerts
- Live spectrogram

### Step 7: Use with Real-Time Data (Optional)

To integrate with real-time EEG data:

1. Modify `dashboard.py` in the `get_model_predictions()` function
2. Replace the random data generation with your real data source
3. Ensure data format matches: `(n_channels, seq_length)`

Example:
```python
def get_model_predictions(self, eeg_data=None):
    # YOUR CODE: Get real-time EEG data
    # eeg_data = your_data_source.get_latest_window()
    
    # Ensure correct shape: (n_channels, seq_length)
    eeg_data = preprocess_your_data(eeg_data)
    
    # Rest of the prediction code...
```

## 📊 Model Architecture

```
Input: (batch, channels, seq_length)
    ↓
Conv1D (64 filters) + BatchNorm + ReLU + MaxPool
    ↓
Conv1D (128 filters) + BatchNorm + ReLU + MaxPool
    ↓
Conv1D (256 filters) + BatchNorm + ReLU + MaxPool
    ↓
Bidirectional LSTM (128 units)
    ↓
Bidirectional LSTM (64 units)
    ↓
Fully Connected (64 units)
    ↓
Output: (batch, n_classes)
```

## 🎛️ Configuration

### Training Parameters (in `train_model.py`)

```python
TIME_WINDOW = 10          # Time window in seconds
N_EPOCHS = 50             # Maximum epochs
LEARNING_RATE = 0.001     # Initial learning rate
PATIENCE = 10             # Early stopping patience
```

### Model Parameters (in `model.py`)

```python
dropout = 0.5             # Dropout rate for regularization
```

## 🔧 Troubleshooting

### Problem: "pip: command not found"
**Solution**: Use `pip3` or `python3 -m pip` instead

### Problem: "No module named 'torch'"
**Solution**: Install PyTorch:
```bash
python3 -m pip install torch
```

### Problem: "Could not find label column"
**Solution**: 
1. Run `inspect_data.py` to see your columns
2. Specify label column manually in `train_model.py`:
```python
train_loader, val_loader, test_loader, info = load_parquet_data(
    PARQUET_FILE,
    label_column='your_label_column_name',  # Add this
    time_window_seconds=TIME_WINDOW
)
```

### Problem: "Out of memory" during training
**Solutions**:
1. Reduce batch size in `data_loader.py`:
```python
batch_size = 16  # Reduce from 32
```

2. Reduce sequence length by using smaller time windows

### Problem: Low accuracy
**Solutions**:
1. Check data quality with `inspect_data.py`
2. Ensure data is properly labeled
3. Increase training epochs
4. Try data augmentation
5. Adjust model architecture

### Problem: Dashboard shows "SIMULATION mode"
**Solution**: Make sure you've trained a model first:
```bash
python3 train_model.py
```

## 📈 Expected Results

- **Training time**: 10-60 minutes (depends on data size)
- **Expected accuracy**: 80-95% (depends on data quality)
- **Inference speed**: ~100-500 predictions/second on CPU

## 🔬 Advanced Usage

### Using Multiple Parquet Files

Modify `data_loader.py` to load multiple files:

```python
import glob

parquet_files = glob.glob("*.parquet")
dfs = [pd.read_parquet(f) for f in parquet_files]
df = pd.concat(dfs, ignore_index=True)
```

### Custom Preprocessing

Add your preprocessing steps in `data_loader.py` before training:

```python
# Example: Bandpass filter
from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut, highcut, fs):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

# Apply in load_parquet_data function
eeg_data = bandpass_filter(eeg_data, 0.5, 50, sampling_rate)
```

### Export Predictions

Save predictions to file:

```python
import csv

predictions = []
with torch.no_grad():
    for data, labels in test_loader:
        probs = model.predict_proba(data.to(device))
        predictions.extend(probs.cpu().numpy())

# Save to CSV
with open('predictions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Seizure', 'LPD', 'GPD', 'LRDA', 'Other'])
    writer.writerows(predictions)
```

## 📚 Files Description

- **inspect_data.py**: Analyzes parquet file structure
- **data_loader.py**: Loads and preprocesses EEG data
- **model.py**: Defines CNN+LSTM architecture
- **train_model.py**: Complete training pipeline
- **dashboard.py**: Real-time visualization dashboard

## 🤝 Support

If you encounter issues:
1. Check the error message carefully
2. Review the troubleshooting section
3. Ensure all dependencies are installed
4. Verify your data format matches expectations

## 📝 Notes

- The model uses 10-second time windows by default
- All data is automatically normalized during preprocessing
- The dashboard updates every second with new predictions
- Training uses early stopping to prevent overfitting
- Best model is automatically saved based on validation loss

## 🎯 Next Steps

1. ✅ Train your model
2. ✅ Evaluate on test set
3. ✅ Run dashboard
4. 🔄 Integrate with real-time data source
5. 🔄 Deploy to production server
6. 🔄 Add more features (alerts, logging, etc.)

Good luck with your EEG seizure detection system! 🧠⚡
