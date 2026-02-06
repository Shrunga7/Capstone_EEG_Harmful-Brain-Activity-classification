"""
Training Script for EEG Seizure Detection Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

from model import create_model
from data_loader import load_parquet_data

class EEGTrainer:
    """Trainer class for EEG model"""
    
    def __init__(self, model, device, save_dir='models'):
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for data, labels in pbar:
            data, labels = data.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc='Validation'):
                data, labels = data.to(self.device), labels.to(self.device)
                
                outputs = self.model(data)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc, all_preds, all_labels, all_probs
    
    def train(self, train_loader, val_loader, n_epochs=50, lr=0.001, patience=10):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of epochs
            lr: Learning rate
            patience: Early stopping patience
        """
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"Number of epochs: {n_epochs}")
        print(f"Learning rate: {lr}")
        print(f"Device: {self.device}")
        print(f"Early stopping patience: {patience}")
        print("=" * 70)
        
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            print("-" * 70)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc, _, _, _ = self.validate(val_loader, criterion)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model('best_model.pth')
                print(f"✅ Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"⏳ Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered after {epoch + 1} epochs")
                break
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        # Save training history
        self.save_training_history()
        self.plot_training_history()
    
    def save_model(self, filename):
        """Save model checkpoint"""
        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
        }, filepath)
    
    def save_training_history(self):
        """Save training history to JSON"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
        }
        
        filepath = os.path.join(self.save_dir, 'training_history.json')
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"\n💾 Saved training history to {filepath}")
    
    def plot_training_history(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        filepath = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(filepath, dpi=150)
        print(f"📊 Saved training curves to {filepath}")
        plt.close()


def main():
    """Main training function"""
    
    # Configuration
    PARQUET_FILE = "1000086677.parquet"  # Update this path
    TIME_WINDOW = 10  # seconds
    N_EPOCHS = 50
    LEARNING_RATE = 0.001
    PATIENCE = 10
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)
    
    train_loader, val_loader, test_loader, info = load_parquet_data(
        PARQUET_FILE,
        time_window_seconds=TIME_WINDOW,
        test_size=0.2,
        random_state=42
    )
    
    # Create model
    print("\n" + "=" * 70)
    print("STEP 2: CREATING MODEL")
    print("=" * 70)
    
    model = create_model(
        n_channels=info['n_channels'],
        seq_length=info['seq_length'],
        n_classes=info['n_classes']
    )
    
    print(f"\nModel created:")
    print(f"  - Input channels: {info['n_channels']}")
    print(f"  - Sequence length: {info['seq_length']}")
    print(f"  - Output classes: {info['n_classes']}")
    print(f"  - Class names: {info['class_names']}")
    
    # Train model
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING MODEL")
    print("=" * 70)
    
    trainer = EEGTrainer(model, device)
    trainer.train(
        train_loader,
        val_loader,
        n_epochs=N_EPOCHS,
        lr=LEARNING_RATE,
        patience=PATIENCE
    )
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("STEP 4: FINAL EVALUATION")
    print("=" * 70)
    
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_preds, test_labels, test_probs = trainer.validate(test_loader, criterion)
    
    print(f"\nTest Results:")
    print(f"  - Loss: {test_loss:.4f}")
    print(f"  - Accuracy: {test_acc:.2f}%")
    
    # Save model info
    model_info = {
        'n_channels': info['n_channels'],
        'seq_length': info['seq_length'],
        'n_classes': info['n_classes'],
        'class_names': info['class_names'],
        'sampling_rate': info['sampling_rate'],
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("\n✅ Training complete! Model saved to 'models/' directory")
    print("\nNext steps:")
    print("1. Check 'models/training_curves.png' for training visualization")
    print("2. Use 'inference.py' to make predictions on new data")
    print("3. Run the dashboard with the trained model")


if __name__ == "__main__":
    main()
