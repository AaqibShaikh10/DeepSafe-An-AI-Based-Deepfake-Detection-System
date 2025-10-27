"""
Audio preprocessing script for training data preparation.
Extracts mel-spectrograms from audio files.
"""

import os
import argparse
import librosa
import numpy as np
from tqdm import tqdm
import json


def preprocess_audio_file(audio_path, output_dir, audio_id, label, segment_duration=4, overlap=1):
    """
    Preprocess audio file and extract mel-spectrograms.

    Args:
        audio_path: Path to audio file
        output_dir: Directory to save processed spectrograms
        audio_id: Unique identifier for the audio
        label: Label (0 for real, 1 for fake)
        segment_duration: Duration of each segment in seconds
        overlap: Overlap between segments in seconds

    Returns:
        List of successfully processed spectrogram paths
    """
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None, mono=True)

        # Resample to 16kHz
        target_sr = 16000
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        # Check if audio is valid
        if len(audio) == 0 or np.max(np.abs(audio)) < 1e-6:
            return []

        # Segment audio
        segment_samples = target_sr * segment_duration
        hop_samples = target_sr * (segment_duration - overlap)

        # Pad if too short
        if len(audio) < segment_samples:
            audio = np.pad(audio, (0, segment_samples - len(audio)), mode='constant')

        # Create segments
        segments = []
        for start_idx in range(0, len(audio) - segment_samples + 1, hop_samples):
            segment = audio[start_idx:start_idx + segment_samples]
            segments.append(segment)

        if len(segments) == 0:
            segments = [audio[:segment_samples]]

        # Process each segment
        processed_spectrograms = []

        for idx, segment in enumerate(segments):
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

            # Normalize
            log_mel_norm = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)

            # Transpose to (time_steps, n_mels)
            log_mel_t = log_mel_norm.T

            # Pad or truncate to fixed length
            target_time_steps = 300
            if log_mel_t.shape[0] < target_time_steps:
                padding = target_time_steps - log_mel_t.shape[0]
                log_mel_t = np.pad(log_mel_t, ((0, padding), (0, 0)), mode='constant')
            else:
                log_mel_t = log_mel_t[:target_time_steps, :]

            # Save spectrogram
            label_str = 'fake' if label == 1 else 'real'
            filename = f"{audio_id}_segment_{idx}_{label_str}.npy"
            save_path = os.path.join(output_dir, filename)
            np.save(save_path, log_mel_t)

            processed_spectrograms.append({
                'path': filename,
                'label': label,
                'audio_id': audio_id,
                'segment_idx': idx
            })

        return processed_spectrograms

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return []


def main(args):
    """Main preprocessing function"""

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load audio list
    # Assuming input directory structure:
    # input/
    #   real/
    #     audio1.wav
    #     audio2.wav
    #   fake/
    #     audio3.wav
    #     audio4.wav

    audio_list = []

    # Real audio files
    real_dir = os.path.join(args.input, 'real')
    if os.path.exists(real_dir):
        for filename in os.listdir(real_dir):
            if filename.endswith(('.wav', '.mp3', '.m4a', '.ogg', '.flac')):
                audio_list.append({
                    'path': os.path.join(real_dir, filename),
                    'label': 0,
                    'audio_id': os.path.splitext(filename)[0]
                })

    # Fake audio files
    fake_dir = os.path.join(args.input, 'fake')
    if os.path.exists(fake_dir):
        for filename in os.listdir(fake_dir):
            if filename.endswith(('.wav', '.mp3', '.m4a', '.ogg', '.flac')):
                audio_list.append({
                    'path': os.path.join(fake_dir, filename),
                    'label': 1,
                    'audio_id': os.path.splitext(filename)[0]
                })

    print(f"Found {len(audio_list)} audio files to process")

    # Process audio files
    all_processed_specs = []

    for audio_info in tqdm(audio_list, desc='Processing audio'):
        specs = preprocess_audio_file(
            audio_path=audio_info['path'],
            output_dir=args.output,
            audio_id=audio_info['audio_id'],
            label=audio_info['label']
        )
        all_processed_specs.extend(specs)

    # Save metadata
    metadata_path = os.path.join(args.output, 'spectrograms_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(all_processed_specs, f, indent=2)

    print(f"\nProcessing complete!")
    print(f"Total spectrograms processed: {len(all_processed_specs)}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess audio for training')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory containing real/ and fake/ subdirectories')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for processed spectrograms')

    args = parser.parse_args()
    main(args)
