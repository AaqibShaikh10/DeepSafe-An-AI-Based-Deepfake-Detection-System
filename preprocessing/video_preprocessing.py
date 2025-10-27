"""
Video preprocessing script for training data preparation.
Extracts frames from videos, detects faces, and saves preprocessed images.
"""

import os
import argparse
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import torch
from tqdm import tqdm
import json


def extract_and_preprocess_frames(video_path, output_dir, video_id, label, mtcnn, num_frames=30):
    """
    Extract frames from video, detect faces, and save preprocessed images.

    Args:
        video_path: Path to video file
        output_dir: Directory to save processed frames
        video_id: Unique identifier for the video
        label: Label (0 for real, 1 for fake)
        mtcnn: MTCNN face detector instance
        num_frames: Number of frames to extract

    Returns:
        List of successfully processed frame paths
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Video has no frames: {video_path}")
        return []

    # Select frame indices
    frame_indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)

    processed_frames = []

    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            # Detect face
            boxes, _ = mtcnn.detect(frame_rgb)

            if boxes is not None and len(boxes) > 0:
                # Use first detected face
                box = boxes[0]
                x1, y1, x2, y2 = [int(b) for b in box]

                # Add padding
                h, w = frame_rgb.shape[:2]
                padding = int(0.3 * max(x2 - x1, y2 - y1))
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)

                # Crop face
                face_crop = frame_rgb[y1:y2, x1:x2]

                # Resize to 224x224
                face_img = Image.fromarray(face_crop)
                face_img = face_img.resize((224, 224), Image.LANCZOS)

                # Save image
                label_str = 'fake' if label == 1 else 'real'
                filename = f"{video_id}_frame_{idx}_{label_str}.jpg"
                save_path = os.path.join(output_dir, filename)
                face_img.save(save_path, quality=95)

                processed_frames.append({
                    'path': filename,
                    'label': label,
                    'video_id': video_id,
                    'frame_idx': idx
                })
        except Exception as e:
            continue

    cap.release()
    return processed_frames


def main(args):
    """Main preprocessing function"""

    # Initialize MTCNN
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(
        image_size=224,
        margin=0,
        keep_all=False,
        select_largest=True,
        device=device
    )

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load video list
    # Assuming input directory structure:
    # input/
    #   real/
    #     video1.mp4
    #     video2.mp4
    #   fake/
    #     video3.mp4
    #     video4.mp4

    video_list = []

    # Real videos
    real_dir = os.path.join(args.input, 'real')
    if os.path.exists(real_dir):
        for filename in os.listdir(real_dir):
            if filename.endswith(('.mp4', '.avi', '.mov', '.webm')):
                video_list.append({
                    'path': os.path.join(real_dir, filename),
                    'label': 0,
                    'video_id': os.path.splitext(filename)[0]
                })

    # Fake videos
    fake_dir = os.path.join(args.input, 'fake')
    if os.path.exists(fake_dir):
        for filename in os.listdir(fake_dir):
            if filename.endswith(('.mp4', '.avi', '.mov', '.webm')):
                video_list.append({
                    'path': os.path.join(fake_dir, filename),
                    'label': 1,
                    'video_id': os.path.splitext(filename)[0]
                })

    print(f"Found {len(video_list)} videos to process")

    # Process videos
    all_processed_frames = []

    for video_info in tqdm(video_list, desc='Processing videos'):
        frames = extract_and_preprocess_frames(
            video_path=video_info['path'],
            output_dir=args.output,
            video_id=video_info['video_id'],
            label=video_info['label'],
            mtcnn=mtcnn,
            num_frames=30
        )
        all_processed_frames.extend(frames)

    # Save metadata
    metadata_path = os.path.join(args.output, 'frames_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(all_processed_frames, f, indent=2)

    print(f"\nProcessing complete!")
    print(f"Total frames processed: {len(all_processed_frames)}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess videos for training')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory containing real/ and fake/ subdirectories')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for processed frames')

    args = parser.parse_args()
    main(args)
