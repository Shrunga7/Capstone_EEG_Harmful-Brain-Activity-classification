"""
CNN + LSTM Model for EEG Seizure Detection
Architecture: Conv1D layers for feature extraction + LSTM for temporal patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EEG_CNN_LSTM(nn.Module):
    """
    CNN+LSTM model for EEG seizure classification
    
    Args:
        n_channels: Number of EEG channels
        seq_length: Length of time sequence (number of time points)
        n_classes: Number of output classes (default: 5)
        dropout: Dropout rate (default: 0.5)
    """
    
    def __init__(self, n_channels, seq_length, n_classes=5, dropout=0.5):
        super(EEG_CNN_LSTM, self).__init__()
        
        self.n_channels = n_channels
        self.seq_length = seq_length
        self.n_classes = n_classes
        
        # CNN layers for spatial feature extraction
        # Input: (batch, channels, seq_length)
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate the output size after pooling
        self.cnn_output_size = seq_length // 8  # 3 pooling layers
        
        # LSTM layers for temporal pattern recognition
        self.lstm1 = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, n_classes)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, channels, seq_length)
        
        Returns:
            Output tensor of shape (batch, n_classes)
        """
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Reshape for LSTM: (batch, seq_length, features)
        x = x.permute(0, 2, 1)
        
        # LSTM temporal processing
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        
        x, _ = self.lstm2(x)
        
        # Take the last output of the sequence
        x = x[:, -1, :]
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def predict_proba(self, x):
        """
        Get probability predictions
        
        Args:
            x: Input tensor
        
        Returns:
            Probabilities for each class
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        return probs


def create_model(n_channels, seq_length, n_classes=5):
    """
    Factory function to create the model
    
    Args:
        n_channels: Number of EEG channels
        seq_length: Length of time sequence
        n_classes: Number of output classes
    
    Returns:
        Initialized model
    """
    model = EEG_CNN_LSTM(n_channels, seq_length, n_classes)
    return model


def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("=" * 70)
    print("MODEL ARCHITECTURE TEST")
    print("=" * 70)
    
    # Example configuration
    n_channels = 19  # Example: 19 EEG channels
    seq_length = 2000  # Example: 10 seconds at 200 Hz
    n_classes = 5
    
    model = create_model(n_channels, seq_length, n_classes)
    
    print(f"\nModel Configuration:")
    print(f"  - Input channels: {n_channels}")
    print(f"  - Sequence length: {seq_length}")
    print(f"  - Output classes: {n_classes}")
    print(f"  - Total parameters: {count_parameters(model):,}")
    
    print(f"\nModel Architecture:")
    print(model)
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, n_channels, seq_length)
    
    print(f"\nTesting forward pass...")
    print(f"  - Input shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"  - Output shape: {output.shape}")
    
    probs = model.predict_proba(dummy_input)
    print(f"  - Probabilities shape: {probs.shape}")
    print(f"  - Sample probabilities: {probs[0].detach().numpy()}")
    
    print("\n✅ Model test successful!")
