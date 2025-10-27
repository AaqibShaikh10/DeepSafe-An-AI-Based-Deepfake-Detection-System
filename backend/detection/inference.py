"""
Model inference module for deepfake detection.
Handles model loading and prediction logic.
"""

import os
import numpy as np
import torch
from django.conf import settings


class ModelLoader:
    """
    Singleton class to load and cache AI models.
    Models are loaded once at startup and reused for all requests.
    """
    _instance = None
    video_model = None
    audio_model = None
    video_model_loaded = False
    audio_model_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.load_models()
        return cls._instance

    def load_models(self):
        """Load TensorFlow models from disk"""
        try:
            # Import TensorFlow here to avoid loading if not needed
            import tensorflow as tf

            # Load video model
            video_model_path = settings.VIDEO_MODEL_PATH
            if os.path.exists(video_model_path):
                self.video_model = tf.keras.models.load_model(video_model_path)
                self.video_model_loaded = True
                print(f"✓ Video model loaded from {video_model_path}")
            else:
                print(f"⚠ Video model not found at {video_model_path}")

            # Load audio model
            audio_model_path = settings.AUDIO_MODEL_PATH
            if os.path.exists(audio_model_path):
                self.audio_model = tf.keras.models.load_model(audio_model_path)
                self.audio_model_loaded = True
                print(f"✓ Audio model loaded from {audio_model_path}")
            else:
                print(f"⚠ Audio model not found at {audio_model_path}")

        except Exception as e:
            print(f"⚠ Error loading models: {e}")
            self.video_model_loaded = False
            self.audio_model_loaded = False


def predict_video(preprocessed_frames, model):
    """
    Run video model prediction on preprocessed frames.

    Args:
        preprocessed_frames: Tensor of shape (N, 3, 224, 224)
        model: Loaded TensorFlow model

    Returns:
        dict: {
            'real_probability': float,
            'fake_probability': float,
            'prediction': str ('real' or 'fake'),
            'confidence': float,
            'frames_analyzed': int
        }
    """
    if model is None:
        raise ValueError("Video model not loaded")

    try:
        # Convert PyTorch tensor to numpy and transpose to (N, 224, 224, 3)
        if isinstance(preprocessed_frames, torch.Tensor):
            frames_array = preprocessed_frames.cpu().numpy()
            frames_array = np.transpose(frames_array, (0, 2, 3, 1))
        else:
            frames_array = preprocessed_frames

        # Run prediction on all frames
        predictions = model.predict(frames_array, verbose=0)

        # Average predictions across all frames
        avg_prediction = np.mean(predictions, axis=0)

        # Extract probabilities (assuming [real_prob, fake_prob])
        real_prob = float(avg_prediction[0])
        fake_prob = float(avg_prediction[1])

        # Normalize probabilities to sum to 1.0
        total = real_prob + fake_prob
        if total > 0:
            real_prob /= total
            fake_prob /= total

        # Determine prediction
        prediction = 'real' if real_prob >= 0.5 else 'fake'
        confidence = max(real_prob, fake_prob)

        return {
            'real_probability': real_prob,
            'fake_probability': fake_prob,
            'prediction': prediction,
            'confidence': confidence,
            'frames_analyzed': len(frames_array)
        }

    except Exception as e:
        raise ValueError(f"Video prediction failed: {str(e)}")


def predict_audio(preprocessed_spectrograms, model):
    """
    Run audio model prediction on preprocessed spectrograms.

    Args:
        preprocessed_spectrograms: Tensor of shape (N, 300, 128)
        model: Loaded TensorFlow model

    Returns:
        dict: {
            'real_probability': float,
            'fake_probability': float,
            'prediction': str ('real' or 'fake'),
            'confidence': float,
            'segments_analyzed': int
        }
    """
    if model is None:
        raise ValueError("Audio model not loaded")

    try:
        # Convert PyTorch tensor to numpy
        if isinstance(preprocessed_spectrograms, torch.Tensor):
            specs_array = preprocessed_spectrograms.cpu().numpy()
        else:
            specs_array = preprocessed_spectrograms

        # Run prediction on all segments
        predictions = model.predict(specs_array, verbose=0)

        # Average predictions across all segments
        avg_prediction = np.mean(predictions, axis=0)

        # Extract probabilities (assuming [real_prob, fake_prob])
        real_prob = float(avg_prediction[0])
        fake_prob = float(avg_prediction[1])

        # Normalize probabilities to sum to 1.0
        total = real_prob + fake_prob
        if total > 0:
            real_prob /= total
            fake_prob /= total

        # Determine prediction
        prediction = 'real' if real_prob >= 0.5 else 'fake'
        confidence = max(real_prob, fake_prob)

        return {
            'real_probability': real_prob,
            'fake_probability': fake_prob,
            'prediction': prediction,
            'confidence': confidence,
            'segments_analyzed': len(specs_array)
        }

    except Exception as e:
        raise ValueError(f"Audio prediction failed: {str(e)}")


def combine_predictions(video_probs, audio_probs):
    """
    Combine video and audio predictions using weighted average.

    Args:
        video_probs: dict with 'real_probability' and 'fake_probability'
        audio_probs: dict with 'real_probability' and 'fake_probability'

    Returns:
        dict: {
            'real_probability': float,
            'fake_probability': float,
            'prediction': str ('real' or 'fake'),
            'confidence': float
        }
    """
    # Fusion weights (video: 0.6, audio: 0.4)
    video_weight = 0.6
    audio_weight = 0.4

    # Weighted average
    combined_fake_prob = (
        video_probs['fake_probability'] * video_weight +
        audio_probs['fake_probability'] * audio_weight
    )
    combined_real_prob = 1.0 - combined_fake_prob

    # Determine prediction
    prediction = 'real' if combined_real_prob >= 0.5 else 'fake'
    confidence = max(combined_real_prob, combined_fake_prob)

    return {
        'real_probability': combined_real_prob,
        'fake_probability': combined_fake_prob,
        'prediction': prediction,
        'confidence': confidence
    }
