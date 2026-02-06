"""
EEG Data Inspector
This script analyzes your parquet file to understand the data structure
"""

import pandas as pd
import numpy as np
import os

def inspect_parquet_file(filepath):
    """Analyze parquet file structure"""
    
    print("=" * 70)
    print("EEG DATA INSPECTION")
    print("=" * 70)
    
    # Read the parquet file
    df = pd.read_parquet(filepath)
    
    print(f"\n📊 Dataset Shape: {df.shape}")
    print(f"   - Number of rows: {len(df):,}")
    print(f"   - Number of columns: {len(df.columns)}")
    
    print("\n📋 Column Names:")
    print("-" * 70)
    for i, col in enumerate(df.columns, 1):
        print(f"   {i}. {col}")
    
    print("\n📈 Data Types:")
    print("-" * 70)
    print(df.dtypes)
    
    print("\n🔍 First Few Rows:")
    print("-" * 70)
    print(df.head(10))
    
    print("\n📊 Statistical Summary:")
    print("-" * 70)
    print(df.describe())
    
    # Identify potential label columns
    print("\n🏷️  Potential Label Columns:")
    print("-" * 70)
    label_candidates = []
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() <= 10:
            unique_vals = df[col].unique()
            print(f"\n   Column: {col}")
            print(f"   Unique values ({len(unique_vals)}): {unique_vals[:20]}")
            label_candidates.append(col)
    
    # Identify EEG channel columns
    print("\n📡 Potential EEG Channel Columns:")
    print("-" * 70)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"   Found {len(numeric_cols)} numeric columns")
    
    # Try to detect common EEG channel naming patterns
    eeg_patterns = ['EEG', 'Fp', 'F', 'C', 'P', 'O', 'T', 'A']
    eeg_channels = []
    for col in numeric_cols:
        col_upper = str(col).upper()
        if any(pattern in col_upper for pattern in eeg_patterns):
            eeg_channels.append(col)
    
    if eeg_channels:
        print(f"\n   Detected EEG channels ({len(eeg_channels)}):")
        for ch in eeg_channels[:20]:  # Show first 20
            print(f"   - {ch}")
        if len(eeg_channels) > 20:
            print(f"   ... and {len(eeg_channels) - 20} more")
    else:
        print(f"\n   Using all numeric columns as EEG channels")
        eeg_channels = numeric_cols
    
    # Check for missing values
    print("\n❓ Missing Values:")
    print("-" * 70)
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("   No missing values found! ✓")
    
    # Estimate sampling rate (if time column exists)
    print("\n⏱️  Time Information:")
    print("-" * 70)
    time_cols = [col for col in df.columns if 'time' in str(col).lower()]
    if time_cols:
        print(f"   Time column(s) found: {time_cols}")
    else:
        print("   No explicit time column detected")
        print(f"   Assuming sequential data with {len(df)} time points")
    
    # Summary for model configuration
    print("\n" + "=" * 70)
    print("🤖 RECOMMENDED MODEL CONFIGURATION")
    print("=" * 70)
    print(f"   Number of EEG channels: {len(eeg_channels)}")
    print(f"   Time window: 10 seconds (as specified)")
    print(f"   Number of classes: 5 (Seizure, LPD, GPD, LRDA, Other)")
    print(f"   Total samples: {len(df):,}")
    
    if label_candidates:
        print(f"   Suggested label column: {label_candidates[0]}")
    
    return {
        'n_channels': len(eeg_channels),
        'channel_names': eeg_channels,
        'label_columns': label_candidates,
        'n_samples': len(df),
        'shape': df.shape
    }

if __name__ == "__main__":
    # Example usage
    parquet_file = "1000086677.parquet"  # Change this to your file path
    
    if not os.path.exists(parquet_file):
        print(f"❌ File not found: {parquet_file}")
        print("\nPlease update the 'parquet_file' variable with the correct path.")
    else:
        info = inspect_parquet_file(parquet_file)
        
        print("\n" + "=" * 70)
        print("✅ Inspection complete!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Review the output above")
        print("2. Run the training script: python train_model.py")
        print("3. The model will automatically configure based on your data")
