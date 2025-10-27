# DeepSafe Model Training Guide

Complete step-by-step guide to train high-accuracy deepfake detection models (>85% accuracy).

## Table of Contents
1. [Hardware Requirements](#hardware-requirements)
2. [Software Requirements](#software-requirements)
3. [Dataset Acquisition](#dataset-acquisition)
4. [Environment Setup](#environment-setup)
5. [Video Model Training](#video-model-training)
6. [Audio Model Training](#audio-model-training)
7. [Model Evaluation](#model-evaluation)
8. [Model Conversion](#model-conversion)
9. [Troubleshooting](#troubleshooting)

---

## Hardware Requirements

### Minimum Requirements (Will be very slow)
- **CPU**: 8-core processor
- **RAM**: 32 GB
- **Storage**: 500 GB SSD
- **Training Time**: 7-14 days

### Recommended Requirements
- **GPU**: NVIDIA RTX 3090, RTX 4090, or A100 (24GB VRAM)
- **CPU**: 16-core processor
- **RAM**: 64 GB
- **Storage**: 1 TB NVMe SSD
- **Training Time**: 2-4 days

### Optimal Setup (Cloud)
- **AWS**: p3.2xlarge or p3.8xlarge (V100 GPUs)
- **Google Cloud**: n1-highmem-8 with 1x V100 GPU
- **Azure**: NC6s_v3 or NC12s_v3
- **Training Time**: 1-2 days

---

## Software Requirements

### Operating System
- Ubuntu 20.04 / 22.04 LTS (recommended)
- Windows 10/11 with WSL2
- macOS (M1/M2 with Metal acceleration)

### Python Environment
```bash
# Python 3.10 or 3.11 (PyTorch compatibility)
python --version  # Should be 3.10.x or 3.11.x
```

### CUDA Setup (For NVIDIA GPUs)
```bash
# Check CUDA version
nvidia-smi

# Install CUDA 11.8 or 12.1
# Visit: https://developer.nvidia.com/cuda-downloads
```

---

## Dataset Acquisition

### 1. FaceForensics++ Dataset (Video Model)

**Size**: ~500 GB
**Link**: https://github.com/ondyari/FaceForensics

#### Registration and Download

1. **Fill Registration Form**:
   ```
   Visit: https://github.com/ondyari/FaceForensics
   Submit research/educational use request
   Wait for approval email (1-2 days)
   ```

2. **Download Script**:
   ```bash
   # After approval, you'll receive download credentials
   cd data/raw
   mkdir -p faceforensics
   cd faceforensics

   # Download using provided script
   python download-FaceForensics.py \
       --server <server_url> \
       --username <your_username> \
       --password <your_password> \
       --compression c23 \
       --type videos
   ```

3. **Directory Structure** (after download):
   ```
   data/raw/faceforensics/
   ├── original_sequences/
   │   └── youtube/
   │       └── c23/
   │           └── videos/  # Real videos
   └── manipulated_sequences/
       ├── Deepfakes/c23/videos/
       ├── Face2Face/c23/videos/
       ├── FaceSwap/c23/videos/
       └── NeuralTextures/c23/videos/
   ```

#### Alternative: Smaller Subset for Testing
```bash
# Download only 100 videos for testing
python download-FaceForensics.py \
    --server <server_url> \
    --username <your_username> \
    --password <your_password> \
    --compression c23 \
    --type videos \
    --num_videos 100
```

### 2. ASVspoof 2019 Dataset (Audio Model)

**Size**: ~20 GB
**Link**: https://www.asvspoof.org/index2019.html

#### Download

1. **Register on ASVspoof Website**:
   - Visit: https://www.asvspoof.org/index2019.html
   - Register for LA (Logical Access) scenario
   - Download link will be sent via email

2. **Download Dataset**:
   ```bash
   cd data/raw
   mkdir -p asvspoof2019
   cd asvspoof2019

   # Download files (links from email)
   wget <LA_train_url>
   wget <LA_dev_url>
   wget <LA_eval_url>
   wget <LA_protocols_url>

   # Extract
   unzip LA.zip
   ```

3. **Directory Structure** (after extraction):
   ```
   data/raw/asvspoof2019/
   ├── LA/
   │   ├── ASVspoof2019_LA_train/
   │   │   └── flac/  # Training audio files
   │   ├── ASVspoof2019_LA_dev/
   │   │   └── flac/  # Validation audio files
   │   └── ASVspoof2019_LA_eval/
   │       └── flac/  # Test audio files
   └── ASVspoof2019_LA_cm_protocols/
       ├── ASVspoof2019.LA.cm.train.trn.txt
       ├── ASVspoof2019.LA.cm.dev.trl.txt
       └── ASVspoof2019.LA.cm.eval.trl.txt
   ```

---

## Environment Setup

### 1. Create Training Environment

```bash
# Clone the repository (if not already done)
git clone https://github.com/AaqibShaikh10/DeepSafe-An-AI-Based-Deepfake-Detection-System.git
cd DeepSafe-An-AI-Based-Deepfake-Detection-System

# Create virtual environment
python3.10 -m venv venv_training
source venv_training/bin/activate  # On Windows: venv_training\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 2. Install Training Dependencies

```bash
# Install PyTorch with CUDA support (for NVIDIA GPU)
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Install additional training tools
pip install tensorboard wandb
```

### 3. Verify GPU Setup

```python
# Test script
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('GPU count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
"
```

**Expected output** (if GPU is available):
```
CUDA available: True
CUDA version: 11.8
GPU count: 1
GPU name: NVIDIA GeForce RTX 3090
```

---

## Video Model Training

### Step 1: Preprocess Video Data

```bash
# This will extract faces from videos and save as images
python preprocessing/video_preprocessing.py \
    --input data/raw/faceforensics \
    --output data/processed/video_frames

# Expected output:
# - Processing time: 6-12 hours (depends on CPU/GPU)
# - Output: ~200,000 face images (224x224)
# - Storage: ~50 GB
```

**What this does**:
- Extracts 30 frames per video
- Detects faces using MTCNN
- Crops and resizes to 224x224
- Normalizes with ImageNet stats
- Saves metadata JSON

### Step 2: Create Train/Val/Test Splits

```bash
# Create data splits (70/15/15)
python preprocessing/create_splits.py \
    --video_frames data/processed/video_frames \
    --output data/splits
```

Create `preprocessing/create_splits.py`:
```python
import json
import os
import random
from pathlib import Path

def create_video_splits(frames_dir, output_dir):
    """Create train/val/test splits for video frames"""

    # Load metadata
    metadata_path = os.path.join(frames_dir, 'frames_metadata.json')
    with open(metadata_path, 'r') as f:
        all_frames = json.load(f)

    # Group by video_id
    video_groups = {}
    for frame in all_frames:
        video_id = frame['video_id']
        if video_id not in video_groups:
            video_groups[video_id] = []
        video_groups[video_id].append(frame)

    # Split videos (not frames)
    video_ids = list(video_groups.keys())
    random.shuffle(video_ids)

    n_total = len(video_ids)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    train_ids = video_ids[:n_train]
    val_ids = video_ids[n_train:n_train + n_val]
    test_ids = video_ids[n_train + n_val:]

    # Create split data
    splits = {
        'train': [f for vid in train_ids for f in video_groups[vid]],
        'val': [f for vid in val_ids for f in video_groups[vid]],
        'test': [f for vid in test_ids for f in video_groups[vid]]
    }

    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'video_split.json')
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)

    print(f"Splits created:")
    print(f"  Train: {len(splits['train'])} frames ({len(train_ids)} videos)")
    print(f"  Val: {len(splits['val'])} frames ({len(val_ids)} videos)")
    print(f"  Test: {len(splits['test'])} frames ({len(test_ids)} videos)")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_frames', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    create_video_splits(args.video_frames, args.output)
```

### Step 3: Train Video Model

```bash
# Start training
python training/train_video_cnn.py \
    --data_dir data/processed \
    --output_dir models/video_cnn \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 0.0001

# With GPU monitoring
watch -n 1 nvidia-smi  # In separate terminal
```

**Training Configuration**:
- **Epochs**: 30 (with early stopping)
- **Batch size**: 32 (adjust based on GPU memory)
- **Learning rate**: 0.0001
- **Optimizer**: Adam
- **Scheduler**: ReduceLROnPlateau
- **Augmentation**: Random flip, rotation, color jitter

**Expected Training Time**:
- RTX 3090: ~24-36 hours
- V100: ~18-24 hours
- A100: ~12-18 hours

**Monitoring Training**:
```bash
# Option 1: TensorBoard
tensorboard --logdir models/video_cnn/logs

# Option 2: Watch training log
tail -f models/video_cnn/training.log
```

**Expected Output**:
```
Epoch 1/30
Train Loss: 0.6234, Train Acc: 0.6543
Val Loss: 0.5123, Val Acc: 0.7234, Precision: 0.7124, Recall: 0.7345, F1: 0.7233
✓ Saved best model with val_acc: 0.7234

Epoch 2/30
Train Loss: 0.4521, Train Acc: 0.7856
Val Loss: 0.3987, Val Acc: 0.8123, Precision: 0.8234, Recall: 0.8012, F1: 0.8122
✓ Saved best model with val_acc: 0.8123

...

Epoch 25/30
Train Loss: 0.1234, Train Acc: 0.9534
Val Loss: 0.1456, Val Acc: 0.9223, Precision: 0.9334, Recall: 0.9112, F1: 0.9222
✓ Saved best model with val_acc: 0.9223

Training complete! Best validation accuracy: 0.9223
```

### Step 4: Fine-tuning (If accuracy < 85%)

If initial training doesn't reach 85%, try:

```bash
# Fine-tune with lower learning rate
python training/train_video_cnn.py \
    --data_dir data/processed \
    --output_dir models/video_cnn \
    --epochs 10 \
    --batch_size 16 \
    --learning_rate 0.00001 \
    --resume models/video_cnn/video_model.pt

# With more augmentation
python training/train_video_cnn.py \
    --data_dir data/processed \
    --output_dir models/video_cnn \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --augmentation_strength high
```

---

## Audio Model Training

### Step 1: Preprocess Audio Data

```bash
# Extract mel-spectrograms from audio
python preprocessing/audio_preprocessing.py \
    --input data/raw/asvspoof2019 \
    --output data/processed/audio_spectrograms

# Expected output:
# - Processing time: 2-4 hours
# - Output: ~100,000 spectrogram files
# - Storage: ~30 GB
```

### Step 2: Create Audio Splits

The ASVspoof dataset comes with official protocol files. Update `training/train_audio_rnn.py` to use them:

```python
# In train_audio_rnn.py, modify dataset loading:
def load_asvspoof_protocol(protocol_file):
    """Load ASVspoof protocol file"""
    data = []
    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            audio_id = parts[1]
            label = 0 if parts[4] == 'bonafide' else 1
            data.append({'audio_id': audio_id, 'label': label})
    return data
```

### Step 3: Train Audio Model

```bash
# Start training
python training/train_audio_rnn.py \
    --data_dir data/processed \
    --output_dir models/audio_rnn \
    --epochs 40 \
    --batch_size 64 \
    --learning_rate 0.001

# Monitor training
tensorboard --logdir models/audio_rnn/logs
```

**Training Configuration**:
- **Epochs**: 40 (with early stopping)
- **Batch size**: 64
- **Learning rate**: 0.001
- **Optimizer**: Adam
- **Scheduler**: ReduceLROnPlateau

**Expected Training Time**:
- RTX 3090: ~12-18 hours
- V100: ~8-12 hours
- A100: ~6-10 hours

**Expected Output**:
```
Epoch 1/40
Train Loss: 0.6891, Train Acc: 0.6234
Val Loss: 0.5678, Val Acc: 0.7123

...

Epoch 35/40
Train Loss: 0.2134, Train Acc: 0.9123
Val Loss: 0.2456, Val Acc: 0.8867, Precision: 0.8934, Recall: 0.8801, F1: 0.8867
✓ Saved best model with val_acc: 0.8867

Training complete! Best validation accuracy: 0.8867
```

---

## Model Evaluation

### Evaluate Video Model

```bash
python training/evaluate.py \
    --model models/video_cnn/video_model.pt \
    --model_type video \
    --data_dir data/processed \
    --output_dir docs

# This generates:
# - docs/video_model_performance.md
# - docs/video_confusion_matrix.png
# - docs/video_roc_curve.png
```

**Target Metrics**:
- ✅ Accuracy: ≥ 90%
- ✅ Precision: ≥ 88%
- ✅ Recall: ≥ 88%
- ✅ F1-Score: ≥ 88%
- ✅ AUC: ≥ 0.95

### Evaluate Audio Model

```bash
python training/evaluate.py \
    --model models/audio_rnn/audio_model.pt \
    --model_type audio \
    --data_dir data/processed \
    --output_dir docs

# This generates:
# - docs/audio_model_performance.md
# - docs/audio_confusion_matrix.png
# - docs/audio_roc_curve.png
```

**Target Metrics**:
- ✅ Accuracy: ≥ 88%
- ✅ Precision: ≥ 86%
- ✅ Recall: ≥ 86%
- ✅ F1-Score: ≥ 86%
- ✅ AUC: ≥ 0.93

---

## Model Conversion

### Convert PyTorch to TensorFlow

After training, convert models for deployment:

```bash
# Convert both models
python training/convert_models.py \
    --models_dir models \
    --convert_video \
    --convert_audio

# Verify conversion
python -c "
import tensorflow as tf
video_model = tf.keras.models.load_model('models/video_cnn/video_model.h5')
print('Video model loaded:', video_model is not None)
audio_model = tf.keras.models.load_model('models/audio_rnn/audio_model.h5')
print('Audio model loaded:', audio_model is not None)
"
```

**What this does**:
1. Loads PyTorch checkpoint
2. Exports to ONNX format
3. Converts ONNX to TensorFlow
4. Saves as .h5 file
5. Verifies accuracy matches PyTorch version

---

## Troubleshooting

### Issue 1: Out of Memory (OOM) Error

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
```bash
# Reduce batch size
python training/train_video_cnn.py --batch_size 16  # instead of 32

# Use gradient accumulation
python training/train_video_cnn.py --batch_size 16 --accumulation_steps 2

# Clear CUDA cache
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Issue 2: Low Accuracy (<85%)

**Possible causes and solutions**:

1. **Insufficient training data**:
   ```bash
   # Download full FaceForensics++ dataset (not subset)
   # Add Celeb-DF dataset for more diversity
   ```

2. **Hyperparameter tuning**:
   ```bash
   # Try different learning rates
   python training/train_video_cnn.py --learning_rate 0.00005

   # Increase epochs
   python training/train_video_cnn.py --epochs 50
   ```

3. **Model architecture**:
   ```python
   # In train_video_cnn.py, try different backbone:
   # EfficientNet-B5 (larger, more accurate)
   self.backbone = timm.create_model('efficientnet_b5', pretrained=True)
   ```

4. **Data augmentation**:
   ```python
   # Add more augmentation in get_transforms()
   transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
   transforms.RandomGrayscale(p=0.1),
   ```

### Issue 3: Training Too Slow

**Solutions**:
```bash
# Use mixed precision training
python training/train_video_cnn.py --use_amp

# Reduce workers if CPU is bottleneck
python training/train_video_cnn.py --num_workers 2

# Use cloud GPU
# AWS p3.2xlarge: ~$3/hour
# Google Cloud V100: ~$2.50/hour
```

### Issue 4: Model Conversion Fails

**Solutions**:
```bash
# Install ONNX tools
pip install onnx onnx-tf tf2onnx

# Convert manually with detailed logs
python training/convert_models.py --verbose

# Alternative: Use TorchScript
python training/convert_to_torchscript.py
```

---

## Expected Results Summary

After completing training:

### Video Model
- ✅ **Accuracy**: 90-93%
- ✅ **Model file**: `models/video_cnn/video_model.h5` (~80 MB)
- ✅ **Inference time**: 3-5 seconds per video
- ✅ **Training time**: 24-36 hours (RTX 3090)

### Audio Model
- ✅ **Accuracy**: 88-91%
- ✅ **Model file**: `models/audio_rnn/audio_model.h5` (~20 MB)
- ✅ **Inference time**: 1-2 seconds per audio
- ✅ **Training time**: 12-18 hours (RTX 3090)

### Combined System
- ✅ **Accuracy**: 92-95% (with fusion)
- ✅ **Ready for deployment**

---

## Next Steps

After successful training:
1. ✅ Verify model files exist in `models/` directory
2. ✅ Review performance reports in `docs/`
3. ✅ Proceed to [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
4. ✅ Test the complete system

---

**Training complete! Your models are ready for deployment.** 🎉
