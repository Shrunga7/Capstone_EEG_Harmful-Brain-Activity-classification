"""
Data Loading and Preprocessing for EEG Data
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

class EEGDataset(Dataset):
    """PyTorch Dataset for EEG data"""
    
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data: numpy array of shape (n_samples, n_channels, seq_length)
            labels: numpy array of shape (n_samples,)
            transform: Optional transform to be applied on a sample
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label


def load_parquet_data(filepath, label_column=None, time_window_seconds=10, 
                      sampling_rate=None, test_size=0.2, random_state=42):
    """
    Load and preprocess EEG data from parquet file
    
    Args:
        filepath: Path to parquet file
        label_column: Name of the column containing labels (auto-detected if None)
        time_window_seconds: Length of time window in seconds
        sampling_rate: Sampling rate in Hz (auto-detected if None)
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader, test_loader, info_dict
    """
    
    print("=" * 70)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 70)
    
    # Load parquet file
    print(f"\n📂 Loading: {filepath}")
    df = pd.read_parquet(filepath)
    print(f"   Shape: {df.shape}")
    
    # Identify label column if not specified
    if label_column is None:
        print("\n🔍 Auto-detecting label column...")
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].nunique() <= 10:
                label_column = col
                print(f"   Found: {label_column}")
                break
    
    if label_column is None or label_column not in df.columns:
        raise ValueError("Could not find label column. Please specify label_column parameter.")
    
    # Get labels
    labels = df[label_column].values
    unique_labels = np.unique(labels)
    print(f"\n🏷️  Labels found: {unique_labels}")
    print(f"   Number of classes: {len(unique_labels)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    print("\n📊 Label distribution:")
    for i, label in enumerate(label_encoder.classes_):
        count = np.sum(encoded_labels == i)
        percentage = count / len(encoded_labels) * 100
        print(f"   {label}: {count} ({percentage:.1f}%)")
    
    # Identify EEG channels (all numeric columns except label)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    eeg_channels = [col for col in numeric_cols if col != label_column]
    
    print(f"\n📡 EEG channels detected: {len(eeg_channels)}")
    print(f"   Channels: {eeg_channels[:10]}{'...' if len(eeg_channels) > 10 else ''}")
    
    # Extract EEG data
    eeg_data = df[eeg_channels].values
    
    # Estimate sampling rate if not provided
    if sampling_rate is None:
        seq_length = len(df)
        # Assume the data is from one continuous recording
        # In real scenario, you might need to extract this from metadata
        sampling_rate = seq_length / time_window_seconds if seq_length < 10000 else 200
        print(f"\n⚠️  Sampling rate not provided. Estimated: {sampling_rate:.0f} Hz")
    
    print(f"   Sampling rate: {sampling_rate} Hz")
    print(f"   Time window: {time_window_seconds} seconds")
    
    # Calculate number of samples per window
    samples_per_window = int(sampling_rate * time_window_seconds)
    print(f"   Samples per window: {samples_per_window}")
    
    # Reshape data into windows
    n_windows = len(eeg_data) // samples_per_window
    print(f"\n✂️  Segmenting into windows...")
    print(f"   Total time points: {len(eeg_data)}")
    print(f"   Number of windows: {n_windows}")
    
    # Truncate to fit complete windows
    truncated_length = n_windows * samples_per_window
    eeg_data = eeg_data[:truncated_length]
    encoded_labels = encoded_labels[:truncated_length]
    
    # Reshape: (n_windows, samples_per_window, n_channels) -> (n_windows, n_channels, samples_per_window)
    eeg_data = eeg_data.reshape(n_windows, samples_per_window, len(eeg_channels))
    eeg_data = eeg_data.transpose(0, 2, 1)  # Now: (n_windows, n_channels, samples_per_window)
    
    # Take one label per window (use the most common label in each window)
    window_labels = encoded_labels.reshape(n_windows, samples_per_window)
    window_labels = np.array([np.bincount(w).argmax() for w in window_labels])
    
    print(f"   Final data shape: {eeg_data.shape}")
    print(f"   Final labels shape: {window_labels.shape}")
    
    # Normalize data (channel-wise)
    print(f"\n📏 Normalizing data...")
    scaler = StandardScaler()
    n_windows, n_channels, seq_len = eeg_data.shape
    
    # Reshape for scaling: (n_windows * seq_len, n_channels)
    eeg_data_flat = eeg_data.transpose(0, 2, 1).reshape(-1, n_channels)
    eeg_data_scaled = scaler.fit_transform(eeg_data_flat)
    
    # Reshape back: (n_windows, n_channels, seq_len)
    eeg_data = eeg_data_scaled.reshape(n_windows, seq_len, n_channels).transpose(0, 2, 1)
    
    # Split data
    print(f"\n✂️  Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        eeg_data, window_labels, test_size=test_size, random_state=random_state, stratify=window_labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=random_state, stratify=y_temp
    )
    
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Validation set: {X_val.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # Create datasets
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    test_dataset = EEGDataset(X_test, y_test)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n✅ Data loading complete!")
    
    info_dict = {
        'n_channels': n_channels,
        'seq_length': samples_per_window,
        'n_classes': len(label_encoder.classes_),
        'class_names': label_encoder.classes_.tolist(),
        'label_encoder': label_encoder,
        'scaler': scaler,
        'sampling_rate': sampling_rate
    }
    
    return train_loader, val_loader, test_loader, info_dict


if __name__ == "__main__":
    # Test data loading
    parquet_file = "1000086677.parquet"
    
    if os.path.exists(parquet_file):
        try:
            train_loader, val_loader, test_loader, info = load_parquet_data(
                parquet_file,
                time_window_seconds=10,
                test_size=0.2
            )
            
            print("\n" + "=" * 70)
            print("DATA INFO")
            print("=" * 70)
            print(f"Number of channels: {info['n_channels']}")
            print(f"Sequence length: {info['seq_length']}")
            print(f"Number of classes: {info['n_classes']}")
            print(f"Class names: {info['class_names']}")
            
            # Test a batch
            for batch_data, batch_labels in train_loader:
                print(f"\nBatch shapes:")
                print(f"  Data: {batch_data.shape}")
                print(f"  Labels: {batch_labels.shape}")
                break
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nPlease run inspect_data.py first to understand your data structure.")
    else:
        print(f"❌ File not found: {parquet_file}")
        print("Please update the file path.")
