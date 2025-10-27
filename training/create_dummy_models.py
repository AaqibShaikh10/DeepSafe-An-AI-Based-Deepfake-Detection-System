"""
Create dummy models for testing DeepSafe system.
This script creates simple TensorFlow models that match the expected input/output format.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_dummy_models():
    """Create dummy TensorFlow models for testing"""
    try:
        import tensorflow as tf
        from tensorflow import keras
        import numpy as np

        print("="*60)
        print("Creating Dummy Models for DeepSafe Testing")
        print("="*60)

        # Create models directory
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        video_dir = os.path.join(models_dir, 'video_cnn')
        audio_dir = os.path.join(models_dir, 'audio_rnn')

        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)

        # 1. Create Video Model (EfficientNet-B4 style)
        print("\n### Creating Video Model (Dummy CNN) ###")
        print("Input shape: (224, 224, 3)")
        print("Output shape: (2,) [real_prob, fake_prob]")

        video_model = keras.Sequential([
            keras.layers.Input(shape=(224, 224, 3)),
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(2, activation='softmax', name='predictions')
        ], name='video_deepfake_detector')

        video_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Initialize with random weights that slightly favor "real" predictions
        # This simulates a trained model behavior
        print("Initializing model weights...")
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        _ = video_model.predict(dummy_input, verbose=0)

        video_model_path = os.path.join(video_dir, 'video_model.h5')
        video_model.save(video_model_path)
        print(f"✓ Video model saved to: {video_model_path}")
        print(f"  Model size: {os.path.getsize(video_model_path) / 1024:.2f} KB")

        # Test the model
        print("\nTesting video model...")
        test_output = video_model.predict(dummy_input, verbose=0)
        print(f"  Test prediction: Real={test_output[0][0]:.4f}, Fake={test_output[0][1]:.4f}")

        # 2. Create Audio Model (LSTM style)
        print("\n### Creating Audio Model (Dummy LSTM) ###")
        print("Input shape: (300, 128)")
        print("Output shape: (2,) [real_prob, fake_prob]")

        audio_model = keras.Sequential([
            keras.layers.Input(shape=(300, 128)),
            keras.layers.LSTM(256, return_sequences=True),
            keras.layers.Dropout(0.3),
            keras.layers.LSTM(128),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(2, activation='softmax', name='predictions')
        ], name='audio_deepfake_detector')

        audio_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Initialize with random weights
        print("Initializing model weights...")
        dummy_audio = np.random.rand(1, 300, 128).astype(np.float32)
        _ = audio_model.predict(dummy_audio, verbose=0)

        audio_model_path = os.path.join(audio_dir, 'audio_model.h5')
        audio_model.save(audio_model_path)
        print(f"✓ Audio model saved to: {audio_model_path}")
        print(f"  Model size: {os.path.getsize(audio_model_path) / 1024:.2f} KB")

        # Test the model
        print("\nTesting audio model...")
        test_output = audio_model.predict(dummy_audio, verbose=0)
        print(f"  Test prediction: Real={test_output[0][0]:.4f}, Fake={test_output[0][1]:.4f}")

        # 3. Verify models can be loaded
        print("\n### Verifying Models ###")

        loaded_video = keras.models.load_model(video_model_path)
        print("✓ Video model can be loaded successfully")

        loaded_audio = keras.models.load_model(audio_model_path)
        print("✓ Audio model can be loaded successfully")

        print("\n" + "="*60)
        print("SUCCESS! Dummy models created and verified")
        print("="*60)
        print("\nModels created:")
        print(f"  1. Video: {video_model_path}")
        print(f"  2. Audio: {audio_model_path}")
        print("\nNOTE: These are dummy models for testing purposes.")
        print("For production use, train models on real datasets.")
        print("="*60)

        return True

    except ImportError as e:
        print(f"\n❌ Error: TensorFlow not installed")
        print(f"   {str(e)}")
        print("\nTo install TensorFlow:")
        print("   pip install tensorflow==2.15.0")
        return False

    except Exception as e:
        print(f"\n❌ Error creating models: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = create_dummy_models()
    sys.exit(0 if success else 1)
