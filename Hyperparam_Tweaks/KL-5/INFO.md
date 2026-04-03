# --- CONFIGURATION & HYPERPARAMETERS ---
CFG = {
    'model_name': 'CNN_LSTM_KLD_v5', #assigning a consistent name for the model

    # Data Settings
    'subset_frac': 1.00,      # Train on 10% for fast debugging
    'img_size': (128, 256),  # (Freq, Time)
    'seed': None,              # For reproducibility (or None for randomness)

    # Model Architecture
    'hidden_size': 32,      # LSTM hidden units
    'dropout': 0.5,          # Regularization

    # Training Loop
    'epochs': 10,            # Default 25
    'batch_size': 128,
    'lr': 1e-5,              # Learning Rate
    'weight_decay': 0.1,

    # Paths
    'train_csv': '/content/drive/MyDrive/Seneca/Capstone/spectrograms/train.csv',
    'val_csv': '/content/drive/MyDrive/Seneca/Capstone/spectrograms/val.csv',
    'train_dir': '/content/train_spectrograms',
    'val_dir': '/content/val_spectrograms'
}

print(f"Configuration Loaded. Mode: {'Debug (Subset)' if CFG['subset_frac'] < 1 else 'Full Training'}")
