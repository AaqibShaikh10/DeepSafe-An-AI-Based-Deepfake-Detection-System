"""
Training script for audio deepfake detection model (LSTM RNN).
This script trains the model using PyTorch and saves checkpoints.
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


class DeepfakeAudioDataset(Dataset):
    """Dataset class for deepfake audio spectrograms"""

    def __init__(self, data_dir, split='train'):
        """
        Args:
            data_dir: Root directory containing processed spectrogram data
            split: 'train', 'val', or 'test'
        """
        self.data_dir = data_dir
        self.split = split

        # Load dataset split information
        split_file = os.path.join(data_dir, '..', 'splits', 'audio_split.json')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                self.split_data = json.load(f)[split]
        else:
            print(f"Warning: Split file not found. Using all data from {data_dir}")
            self.split_data = self._load_all_spectrograms()

    def _load_all_spectrograms(self):
        """Load all spectrograms from directory if split file doesn't exist"""
        spectrograms = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.npy'):
                # Assume filename format includes label information
                label = 1 if 'fake' in filename else 0
                spectrograms.append({'path': filename, 'label': label})
        return spectrograms

    def __len__(self):
        return len(self.split_data)

    def __getitem__(self, idx):
        item = self.split_data[idx]
        spec_path = os.path.join(self.data_dir, item['path'])

        # Load spectrogram (shape: time_steps, n_mels)
        spectrogram = np.load(spec_path)

        # Convert to tensor
        spectrogram = torch.FloatTensor(spectrogram)

        label = item['label']
        return spectrogram, label


class AudioDeepfakeDetector(nn.Module):
    """LSTM-based audio deepfake detector"""

    def __init__(self, input_size=128, hidden_size1=256, hidden_size2=128, num_classes=2):
        super(AudioDeepfakeDetector, self).__init__()

        # LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size1,
            batch_first=True,
            bidirectional=False
        )

        self.dropout1 = nn.Dropout(0.3)

        self.lstm2 = nn.LSTM(
            input_size=hidden_size1,
            hidden_size=hidden_size2,
            batch_first=True,
            bidirectional=False
        )

        self.dropout2 = nn.Dropout(0.3)

        # Dense layers
        self.fc1 = nn.Linear(hidden_size2, 64)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch, time_steps, features)

        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)

        # Second LSTM layer
        lstm2_out, (hidden, _) = self.lstm2(lstm1_out)
        # Use last hidden state
        last_hidden = hidden[-1]
        last_hidden = self.dropout2(last_hidden)

        # Dense layers
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc2(out)

        return out


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    for spectrograms, labels in tqdm(dataloader, desc='Training'):
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(spectrograms)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_predictions)

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for spectrograms, labels in tqdm(dataloader, desc='Validating'):
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            outputs = model(spectrograms)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    val_loss = running_loss / len(dataloader)
    val_acc = accuracy_score(all_labels, all_predictions)
    val_precision = precision_score(all_labels, all_predictions, average='binary')
    val_recall = recall_score(all_labels, all_predictions, average='binary')
    val_f1 = f1_score(all_labels, all_predictions, average='binary')

    return val_loss, val_acc, val_precision, val_recall, val_f1


def main(args):
    """Main training function"""

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = DeepfakeAudioDataset(
        data_dir=os.path.join(args.data_dir, 'audio_spectrograms'),
        split='train'
    )

    val_dataset = DeepfakeAudioDataset(
        data_dir=os.path.join(args.data_dir, 'audio_spectrograms'),
        split='val'
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create model
    model = AudioDeepfakeDetector(
        input_size=128,
        hidden_size1=256,
        hidden_size2=128,
        num_classes=2
    ).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, verbose=True
    )

    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.output_dir, 'audio_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, save_path)
            print(f"✓ Saved best model with val_acc: {val_acc:.4f}")

    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train audio deepfake detection model')
    parser.add_argument('--data_dir', type=str, default='../data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default='../models/audio_rnn',
                        help='Path to save trained model')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Train
    main(args)
