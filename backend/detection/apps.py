from django.apps import AppConfig


class DetectionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detection'

    def ready(self):
        """Load models once at Django startup"""
        from .inference import ModelLoader
        try:
            ModelLoader()  # Singleton, loads models once
            print("✓ Models loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Failed to load models: {e}")
            print("  Models will be loaded on first request if available")
