from django.db import models


class DetectionHistory(models.Model):
    """Model to store detection history for analytics"""

    FILE_TYPE_CHOICES = [
        ('video', 'Video'),
        ('audio', 'Audio'),
    ]

    PREDICTION_CHOICES = [
        ('real', 'Real'),
        ('fake', 'Fake'),
    ]

    file_type = models.CharField(max_length=10, choices=FILE_TYPE_CHOICES)
    prediction = models.CharField(max_length=10, choices=PREDICTION_CHOICES)
    confidence = models.FloatField(help_text="Confidence score (0-1)")
    real_probability = models.FloatField(help_text="Probability of being real (0-1)")
    fake_probability = models.FloatField(help_text="Probability of being fake (0-1)")
    processing_time = models.FloatField(help_text="Processing time in seconds")
    file_size = models.IntegerField(help_text="File size in bytes")
    timestamp = models.DateTimeField(auto_now_add=True)
    error_occurred = models.BooleanField(default=False)
    error_message = models.TextField(null=True, blank=True)

    class Meta:
        verbose_name_plural = "Detection Histories"
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.file_type} - {self.prediction} ({self.confidence:.2%}) - {self.timestamp}"
