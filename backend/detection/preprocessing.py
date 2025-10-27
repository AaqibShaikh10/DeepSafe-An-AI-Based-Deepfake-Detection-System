"""
Preprocessing module for video and audio deepfake detection.
Handles frame extraction, face detection, and audio feature extraction.
"""

import cv2
import numpy as np
import librosa
import torch
from PIL import Image
from facenet_pytorch import MTCNN
import filetype
from moviepy.editor import VideoFileClip
import tempfile
import os


# Initialize face detector (singleton pattern for efficiency)
_face_detector = None


def get_face_detector():
    """Get or initialize MTCNN face detector"""
    global _face_detector
    if _face_detector is None:
        _face_detector = MTCNN(
            image_size=224,
            margin=0,
            keep_all=False,
            select_largest=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    return _face_detector


def preprocess_video(video_path, max_frames=30):
    """
    Preprocess video for deepfake detection.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract (default: 30)

    Returns:
        List of preprocessed frame tensors (N, 3, 224, 224)

    Raises:
        ValueError: If file is invalid, corrupted, or no faces detected
    """

    # 1. File Validation
    if not os.path.exists(video_path):
        raise ValueError("Video file does not exist")

    # Check file format
    kind = filetype.guess(video_path)
    if kind is None or kind.extension not in ['mp4', 'avi', 'mov', 'webm', 'mkv']:
        raise ValueError("Unsupported video format")

    # Check file size (max 100MB)
    file_size = os.path.getsize(video_path)
    if file_size > 100 * 1024 * 1024:
        raise ValueError("File size exceeds 100MB")

    # 2. Video Loading
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot read video file")

    try:
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames == 0 or fps == 0:
            raise ValueError("Video file is corrupted or unreadable")

        # 3. Frame Extraction
        # Extract evenly-spaced frames
        frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)

        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

        if len(frames) == 0:
            raise ValueError("Failed to extract frames from video")

        # 4. Face Detection and Preprocessing
        face_detector = get_face_detector()
        preprocessed_frames = []

        for frame in frames:
            try:
                # Detect face
                boxes, _ = face_detector.detect(frame)

                if boxes is not None and len(boxes) > 0:
                    # Use largest face (first one from select_largest=True)
                    box = boxes[0]
                    x1, y1, x2, y2 = [int(b) for b in box]

                    # Add 30% padding
                    h, w = frame.shape[:2]
                    padding = int(0.3 * max(x2 - x1, y2 - y1))
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(w, x2 + padding)
                    y2 = min(h, y2 + padding)

                    # Crop face
                    face_crop = frame[y1:y2, x1:x2]

                    # 5. Image Preprocessing
                    # Resize to 224x224
                    face_img = Image.fromarray(face_crop)
                    face_img = face_img.resize((224, 224), Image.LANCZOS)

                    # Convert to numpy array
                    face_array = np.array(face_img, dtype=np.float32) / 255.0

                    # Normalize with ImageNet mean and std
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    face_array = (face_array - mean) / std

                    # Convert to CHW format (channels first)
                    face_tensor = torch.FloatTensor(face_array).permute(2, 0, 1)

                    preprocessed_frames.append(face_tensor)
            except Exception as e:
                # Skip frames where face detection fails
                continue

        # Check if we have enough frames with detected faces
        if len(preprocessed_frames) < 10:
            raise ValueError("Insufficient faces detected in video")

        # 6. Output
        # Stack tensors into batch
        return torch.stack(preprocessed_frames)

    finally:
        cap.release()


def preprocess_audio(audio_path, segment_duration=4, overlap=1):
    """
    Preprocess audio for deepfake detection.

    Args:
        audio_path: Path to audio file or video file (audio will be extracted)
        segment_duration: Duration of each segment in seconds (default: 4)
        overlap: Overlap between segments in seconds (default: 1)

    Returns:
        List of preprocessed mel-spectrogram tensors (N, 300, 128)

    Raises:
        ValueError: If file is invalid, corrupted, or silent
    """

    temp_audio_path = None

    try:
        # 1. Audio Extraction (if from video)
        kind = filetype.guess(audio_path)
        if kind and kind.extension in ['mp4', 'avi', 'mov', 'webm', 'mkv']:
            # Extract audio from video
            try:
                video = VideoFileClip(audio_path)
                if video.audio is None:
                    raise ValueError("Video contains no audio track")

                # Save temporary audio file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    temp_audio_path = tmp.name
                video.audio.write_audiofile(temp_audio_path, logger=None, verbose=False)
                video.close()
                audio_path = temp_audio_path
            except Exception as e:
                raise ValueError(f"Failed to extract audio from video: {str(e)}")

        # 2. File Validation
        if not os.path.exists(audio_path):
            raise ValueError("Audio file does not exist")

        # Check file size (max 50MB)
        file_size = os.path.getsize(audio_path)
        if file_size > 50 * 1024 * 1024:
            raise ValueError("Audio file too large")

        # 3. Audio Loading
        try:
            audio, sr = librosa.load(audio_path, sr=None, mono=False)
        except Exception as e:
            raise ValueError("Cannot read audio file")

        # 4. Audio Resampling
        target_sr = 16000
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)

        # Check if audio is silent
        if np.max(np.abs(audio)) < 1e-6:
            raise ValueError("Audio appears to be silent - cannot analyze")

        # 5. Audio Segmentation
        segment_samples = target_sr * segment_duration  # 64000 samples for 4 seconds
        hop_samples = target_sr * (segment_duration - overlap)  # 48000 samples for 3 second hop

        # Pad if audio is too short
        if len(audio) < segment_samples:
            audio = np.pad(audio, (0, segment_samples - len(audio)), mode='constant')

        # Create segments
        segments = []
        for start_idx in range(0, len(audio) - segment_samples + 1, hop_samples):
            segment = audio[start_idx:start_idx + segment_samples]
            segments.append(segment)

        if len(segments) == 0:
            segments = [audio[:segment_samples]]  # Use first segment if audio is very short

        # 6. Mel-Spectrogram Extraction
        preprocessed_spectrograms = []

        for segment in segments:
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=segment,
                sr=target_sr,
                n_fft=2048,
                hop_length=512,
                n_mels=128,
                fmax=8000
            )

            # Convert to log scale
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)

            # Normalize to [0, 1]
            log_mel_norm = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)

            # 7. Reshape for LSTM Input
            # Transpose to (time_steps, n_mels)
            log_mel_t = log_mel_norm.T

            # Pad or truncate to fixed length (300 time steps)
            target_time_steps = 300
            if log_mel_t.shape[0] < target_time_steps:
                # Pad
                padding = target_time_steps - log_mel_t.shape[0]
                log_mel_t = np.pad(log_mel_t, ((0, padding), (0, 0)), mode='constant')
            else:
                # Truncate
                log_mel_t = log_mel_t[:target_time_steps, :]

            # Convert to tensor
            spec_tensor = torch.FloatTensor(log_mel_t)
            preprocessed_spectrograms.append(spec_tensor)

        # 8. Output
        return torch.stack(preprocessed_spectrograms)

    finally:
        # Cleanup temporary audio file
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                pass
