"""
API endpoint tests for DeepSafe detection system.
Run with: python manage.py test tests.test_api
"""

from django.test import TestCase, Client
from django.urls import reverse
import json


class HealthCheckTestCase(TestCase):
    """Test cases for health check endpoint"""

    def setUp(self):
        self.client = Client()

    def test_health_endpoint_exists(self):
        """Test that health check endpoint is accessible"""
        response = self.client.get('/api/health/')
        self.assertIn(response.status_code, [200, 503])

    def test_health_endpoint_returns_json(self):
        """Test that health check returns JSON"""
        response = self.client.get('/api/health/')
        self.assertEqual(response['Content-Type'], 'application/json')

    def test_health_endpoint_structure(self):
        """Test that health check response has correct structure"""
        response = self.client.get('/api/health/')
        data = json.loads(response.content)
        self.assertIn('status', data)
        self.assertIn('video_model_loaded', data)
        self.assertIn('audio_model_loaded', data)
        self.assertIn('timestamp', data)


class VideoDetectionTestCase(TestCase):
    """Test cases for video detection endpoint"""

    def setUp(self):
        self.client = Client()

    def test_video_endpoint_requires_file(self):
        """Test that video endpoint requires a file"""
        response = self.client.post('/api/detect/video/')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.content)
        self.assertEqual(data['success'], False)
        self.assertIn('error', data)

    def test_video_endpoint_rejects_invalid_format(self):
        """Test that video endpoint rejects invalid file formats"""
        # This would require creating a test file
        # Implementation depends on test fixtures
        pass


class AudioDetectionTestCase(TestCase):
    """Test cases for audio detection endpoint"""

    def setUp(self):
        self.client = Client()

    def test_audio_endpoint_requires_file(self):
        """Test that audio endpoint requires a file"""
        response = self.client.post('/api/detect/audio/')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.content)
        self.assertEqual(data['success'], False)
        self.assertIn('error', data)

    def test_audio_endpoint_rejects_invalid_format(self):
        """Test that audio endpoint rejects invalid file formats"""
        # This would require creating a test file
        # Implementation depends on test fixtures
        pass


class DetectionHistoryTestCase(TestCase):
    """Test cases for detection history model"""

    def test_can_create_detection_history(self):
        """Test that DetectionHistory records can be created"""
        from detection.models import DetectionHistory

        history = DetectionHistory.objects.create(
            file_type='video',
            prediction='real',
            confidence=0.87,
            real_probability=0.87,
            fake_probability=0.13,
            processing_time=4.23,
            file_size=1024000,
            error_occurred=False
        )

        self.assertIsNotNone(history.id)
        self.assertEqual(history.prediction, 'real')
        self.assertEqual(history.confidence, 0.87)


# Add more test cases as needed:
# - Test preprocessing functions
# - Test model inference
# - Test error handling
# - Test file cleanup
# - Test validation logic
# - Integration tests with test fixtures
