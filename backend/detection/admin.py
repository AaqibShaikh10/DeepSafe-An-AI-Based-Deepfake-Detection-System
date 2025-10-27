from django.contrib import admin
from .models import DetectionHistory


@admin.register(DetectionHistory)
class DetectionHistoryAdmin(admin.ModelAdmin):
    list_display = ('id', 'file_type', 'prediction', 'confidence', 'timestamp', 'error_occurred')
    list_filter = ('file_type', 'prediction', 'error_occurred', 'timestamp')
    search_fields = ('error_message',)
    readonly_fields = ('timestamp',)
    ordering = ('-timestamp',)
