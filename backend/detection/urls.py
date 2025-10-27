"""
URL routing for detection API endpoints.
"""

from django.urls import path
from .views import HealthCheckView, VideoDetectionView, AudioDetectionView

app_name = 'detection'

urlpatterns = [
    path('health/', HealthCheckView.as_view(), name='health'),
    path('detect/video/', VideoDetectionView.as_view(), name='detect_video'),
    path('detect/audio/', AudioDetectionView.as_view(), name='detect_audio'),
]
