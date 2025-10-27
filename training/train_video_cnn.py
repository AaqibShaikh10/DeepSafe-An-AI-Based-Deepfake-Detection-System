"""
Training script for video deepfake detection model (EfficientNet-B4 CNN).
This script trains the model using PyTorch and saves checkpoints.
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt


class DeepfakeVideoDataset(Dataset):
    """Dataset class for deepfake video frames"""

    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir: Root directory containing processed data
            split: 'train', 'val', or 'test'
            transform: Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        # Load dataset split information
        split_file = os.path.join(data_dir, '..', 'splits', 'video_split.json')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                self.split_data = json.load(f)[split]
        else:
            print(f"Warning: Split file not found. Using all data from {data_dir}")
            self.split_data = self._load_all_frames()

    def _load_all_frames(self):
        """Load all frames from directory if split file doesn't exist"""
        frames = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # Assume filename format: {video_id}_frame_{idx}_{label}.jpg
                label = 1 if 'fake' in filename else 0
                frames.append({'path': filename, 'label': label})
        return frames

    def __len__(self):
        return len(self.split_data)

    def __getitem__(self, idx):
        item = self.split_data[idx]
        img_path = os.path.join(self.data_dir, item['path'])

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = item['label']
        return image, label


class VideoDeepfakeDetector(nn.Module):
    """EfficientNet-B4 based deepfake detector"""

    def __init__(self, num_classes=2, pretrained=True):
        super(VideoDeepfakeDetector, self).__init__()

        # Load EfficientNet-B4 backbone
        self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=0)

        # Get number of features from backbone
        num_features = self.backbone.num_features

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


def get_transforms(split='train'):
    """Get data transforms for training or validation"""

    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if split == 'train':
        # Training augmentation
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
    else:
        # Validation/test transforms
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc='Training'):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
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
        for images, labels in tqdm(dataloader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
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
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')

    train_dataset = DeepfakeVideoDataset(
        data_dir=os.path.join(args.data_dir, 'video_frames'),
        split='train',
        transform=train_transform
    )

    val_dataset = DeepfakeVideoDataset(
        data_dir=os.path.join(args.data_dir, 'video_frames'),
        split='val',
        transform=val_transform
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
    model = VideoDeepfakeDetector(num_classes=2, pretrained=True).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
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
            save_path = os.path.join(args.output_dir, 'video_model.pt')
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
    parser = argparse.ArgumentParser(description='Train video deepfake detection model')
    parser.add_argument('--data_dir', type=str, default='../data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default='../models/video_cnn',
                        help='Path to save trained model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Train
    main(args)
