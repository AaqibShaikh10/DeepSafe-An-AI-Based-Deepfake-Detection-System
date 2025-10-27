"""
API views for deepfake detection endpoints.
"""

import os
import time
import uuid
from datetime import datetime
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .preprocessing import preprocess_video, preprocess_audio
from .inference import ModelLoader, predict_video, predict_audio, combine_predictions
from .models import DetectionHistory


class HealthCheckView(APIView):
    """Check if API and models are loaded"""

    def get(self, request):
        try:
            model_loader = ModelLoader()

            response_data = {
                'status': 'healthy' if (model_loader.video_model_loaded or model_loader.audio_model_loaded) else 'unhealthy',
                'video_model_loaded': model_loader.video_model_loaded,
                'audio_model_loaded': model_loader.audio_model_loaded,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }

            if response_data['status'] == 'unhealthy':
                response_data['error'] = 'Models not loaded'
                return Response(response_data, status=status.HTTP_503_SERVICE_UNAVAILABLE)

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({
                'status': 'unhealthy',
                'video_model_loaded': False,
                'audio_model_loaded': False,
                'error': str(e)
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)


class VideoDetectionView(APIView):
    """Analyze uploaded video for deepfake detection"""

    def post(self, request):
        start_time = time.time()
        uploaded_file_path = None

        try:
            # Validate file presence
            if 'video' not in request.FILES:
                return Response({
                    'success': False,
                    'error': 'No video file provided',
                    'error_type': 'validation_error'
                }, status=status.HTTP_400_BAD_REQUEST)

            video_file = request.FILES['video']
            analyze_audio_flag = request.POST.get('analyze_audio', 'false').lower() == 'true'

            # Validate file extension
            file_ext = os.path.splitext(video_file.name)[1].lower()
            if file_ext not in ['.mp4', '.avi', '.mov', '.webm']:
                return Response({
                    'success': False,
                    'error': 'Unsupported video format. Supported formats: MP4, AVI, MOV, WEBM',
                    'error_type': 'validation_error'
                }, status=status.HTTP_400_BAD_REQUEST)

            # Validate file size
            file_size = video_file.size
            max_size = settings.MAX_VIDEO_SIZE_MB * 1024 * 1024
            if file_size > max_size:
                return Response({
                    'success': False,
                    'error': f'File size exceeds {settings.MAX_VIDEO_SIZE_MB}MB',
                    'error_type': 'validation_error'
                }, status=status.HTTP_400_BAD_REQUEST)

            # Save uploaded file temporarily
            timestamp = int(time.time())
            random_str = uuid.uuid4().hex[:6]
            filename = f"video_{timestamp}_{random_str}{file_ext}"
            uploaded_file_path = os.path.join(settings.MEDIA_ROOT, filename)

            with open(uploaded_file_path, 'wb+') as destination:
                for chunk in video_file.chunks():
                    destination.write(chunk)

            # Preprocess video
            try:
                preprocessed_frames = preprocess_video(uploaded_file_path)
            except ValueError as e:
                return Response({
                    'success': False,
                    'error': str(e),
                    'error_type': 'processing_error'
                }, status=status.HTTP_400_BAD_REQUEST)

            # Load model and run inference
            model_loader = ModelLoader()
            if not model_loader.video_model_loaded:
                return Response({
                    'success': False,
                    'error': 'Video model not available',
                    'error_type': 'server_error'
                }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

            video_predictions = predict_video(preprocessed_frames, model_loader.video_model)

            # Initialize response
            response_data = {
                'success': True,
                'file_type': 'video',
                'predictions': {
                    'video': video_predictions
                },
                'processing_time': 0,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }

            # Optional: Analyze audio track
            audio_predictions = None
            if analyze_audio_flag:
                try:
                    if model_loader.audio_model_loaded:
                        preprocessed_audio = preprocess_audio(uploaded_file_path)
                        audio_predictions = predict_audio(preprocessed_audio, model_loader.audio_model)
                        response_data['predictions']['audio'] = audio_predictions

                        # Compute combined prediction
                        combined_predictions = combine_predictions(video_predictions, audio_predictions)
                        response_data['predictions']['combined'] = combined_predictions
                except Exception as e:
                    # Don't fail the entire request if audio analysis fails
                    print(f"Audio analysis failed: {e}")

            # Calculate processing time
            processing_time = time.time() - start_time
            response_data['processing_time'] = round(processing_time, 2)

            # Log to database
            DetectionHistory.objects.create(
                file_type='video',
                prediction=video_predictions['prediction'],
                confidence=video_predictions['confidence'],
                real_probability=video_predictions['real_probability'],
                fake_probability=video_predictions['fake_probability'],
                processing_time=processing_time,
                file_size=file_size,
                error_occurred=False
            )

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            # Log error to database
            processing_time = time.time() - start_time
            DetectionHistory.objects.create(
                file_type='video',
                prediction='unknown',
                confidence=0.0,
                real_probability=0.0,
                fake_probability=0.0,
                processing_time=processing_time,
                file_size=0,
                error_occurred=True,
                error_message=str(e)
            )

            return Response({
                'success': False,
                'error': 'Internal server error during video processing',
                'error_type': 'server_error'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        finally:
            # Cleanup: Delete uploaded file
            if uploaded_file_path and os.path.exists(uploaded_file_path):
                try:
                    os.remove(uploaded_file_path)
                except Exception as e:
                    print(f"Failed to delete file {uploaded_file_path}: {e}")


class AudioDetectionView(APIView):
    """Analyze uploaded audio for deepfake detection"""

    def post(self, request):
        start_time = time.time()
        uploaded_file_path = None

        try:
            # Validate file presence
            if 'audio' not in request.FILES:
                return Response({
                    'success': False,
                    'error': 'No audio file provided',
                    'error_type': 'validation_error'
                }, status=status.HTTP_400_BAD_REQUEST)

            audio_file = request.FILES['audio']

            # Validate file extension
            file_ext = os.path.splitext(audio_file.name)[1].lower()
            if file_ext not in ['.mp3', '.wav', '.m4a', '.ogg']:
                return Response({
                    'success': False,
                    'error': 'Unsupported audio format. Supported formats: MP3, WAV, M4A, OGG',
                    'error_type': 'validation_error'
                }, status=status.HTTP_400_BAD_REQUEST)

            # Validate file size
            file_size = audio_file.size
            max_size = settings.MAX_AUDIO_SIZE_MB * 1024 * 1024
            if file_size > max_size:
                return Response({
                    'success': False,
                    'error': f'Audio file too large. Maximum size: {settings.MAX_AUDIO_SIZE_MB}MB',
                    'error_type': 'validation_error'
                }, status=status.HTTP_400_BAD_REQUEST)

            # Save uploaded file temporarily
            timestamp = int(time.time())
            random_str = uuid.uuid4().hex[:6]
            filename = f"audio_{timestamp}_{random_str}{file_ext}"
            uploaded_file_path = os.path.join(settings.MEDIA_ROOT, filename)

            with open(uploaded_file_path, 'wb+') as destination:
                for chunk in audio_file.chunks():
                    destination.write(chunk)

            # Preprocess audio
            try:
                preprocessed_audio = preprocess_audio(uploaded_file_path)
            except ValueError as e:
                return Response({
                    'success': False,
                    'error': str(e),
                    'error_type': 'processing_error'
                }, status=status.HTTP_400_BAD_REQUEST)

            # Load model and run inference
            model_loader = ModelLoader()
            if not model_loader.audio_model_loaded:
                return Response({
                    'success': False,
                    'error': 'Audio model not available',
                    'error_type': 'server_error'
                }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

            audio_predictions = predict_audio(preprocessed_audio, model_loader.audio_model)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Prepare response
            response_data = {
                'success': True,
                'file_type': 'audio',
                'predictions': {
                    'audio': audio_predictions
                },
                'processing_time': round(processing_time, 2),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }

            # Log to database
            DetectionHistory.objects.create(
                file_type='audio',
                prediction=audio_predictions['prediction'],
                confidence=audio_predictions['confidence'],
                real_probability=audio_predictions['real_probability'],
                fake_probability=audio_predictions['fake_probability'],
                processing_time=processing_time,
                file_size=file_size,
                error_occurred=False
            )

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            # Log error to database
            processing_time = time.time() - start_time
            DetectionHistory.objects.create(
                file_type='audio',
                prediction='unknown',
                confidence=0.0,
                real_probability=0.0,
                fake_probability=0.0,
                processing_time=processing_time,
                file_size=0,
                error_occurred=True,
                error_message=str(e)
            )

            return Response({
                'success': False,
                'error': 'Internal server error during audio processing',
                'error_type': 'server_error'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        finally:
            # Cleanup: Delete uploaded file
            if uploaded_file_path and os.path.exists(uploaded_file_path):
                try:
                    os.remove(uploaded_file_path)
                except Exception as e:
                    print(f"Failed to delete file {uploaded_file_path}: {e}")
