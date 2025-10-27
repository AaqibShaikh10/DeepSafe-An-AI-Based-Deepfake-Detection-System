# DeepSafe API Documentation

Complete API reference for the DeepSafe deepfake detection system.

## Base URL

```
http://localhost:8000/api/
```

For production, replace with your deployed domain.

## Authentication

Currently, the API does not require authentication. This may be added in future versions for rate limiting and access control.

## API Endpoints

### 1. Health Check

**Endpoint:** `GET /api/health/`

**Description:** Check if the API is running and models are loaded.

**Request:**
```http
GET /api/health/ HTTP/1.1
Host: localhost:8000
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "video_model_loaded": true,
  "audio_model_loaded": true,
  "timestamp": "2025-10-27T12:34:56Z"
}
```

**Response (503 Service Unavailable):**
```json
{
  "status": "unhealthy",
  "video_model_loaded": false,
  "audio_model_loaded": false,
  "error": "Models not loaded",
  "timestamp": "2025-10-27T12:34:56Z"
}
```

---

### 2. Video Deepfake Detection

**Endpoint:** `POST /api/detect/video/`

**Description:** Analyze an uploaded video file for deepfake detection.

**Request:**
```http
POST /api/detect/video/ HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary

------WebKitFormBoundary
Content-Disposition: form-data; name="video"; filename="test_video.mp4"
Content-Type: video/mp4

[Binary video data]
------WebKitFormBoundary
Content-Disposition: form-data; name="analyze_audio"

true
------WebKitFormBoundary--
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| video | File | Yes | Video file to analyze (MP4, AVI, MOV, WEBM) |
| analyze_audio | Boolean | No | Whether to also analyze audio track (default: false) |

**Supported Formats:** MP4, AVI, MOV, WEBM

**File Size Limit:** 100MB

**Response (200 OK) - Video Only:**
```json
{
  "success": true,
  "file_type": "video",
  "predictions": {
    "video": {
      "real_probability": 0.87,
      "fake_probability": 0.13,
      "prediction": "real",
      "confidence": 0.87,
      "frames_analyzed": 28
    }
  },
  "processing_time": 4.23,
  "timestamp": "2025-10-27T12:34:56Z"
}
```

**Response (200 OK) - With Audio Analysis:**
```json
{
  "success": true,
  "file_type": "video",
  "predictions": {
    "video": {
      "real_probability": 0.87,
      "fake_probability": 0.13,
      "prediction": "real",
      "confidence": 0.87,
      "frames_analyzed": 28
    },
    "audio": {
      "real_probability": 0.92,
      "fake_probability": 0.08,
      "prediction": "real",
      "confidence": 0.92,
      "segments_analyzed": 5
    },
    "combined": {
      "real_probability": 0.892,
      "fake_probability": 0.108,
      "prediction": "real",
      "confidence": 0.892
    }
  },
  "processing_time": 5.67,
  "timestamp": "2025-10-27T12:34:56Z"
}
```

**Error Response (400 Bad Request) - Validation Error:**
```json
{
  "success": false,
  "error": "Unsupported video format. Supported formats: MP4, AVI, MOV, WEBM",
  "error_type": "validation_error"
}
```

**Error Response (400 Bad Request) - Processing Error:**
```json
{
  "success": false,
  "error": "No faces detected in video - cannot analyze",
  "error_type": "processing_error"
}
```

**Error Response (500 Internal Server Error):**
```json
{
  "success": false,
  "error": "Internal server error during video processing",
  "error_type": "server_error"
}
```

---

### 3. Audio Deepfake Detection

**Endpoint:** `POST /api/detect/audio/`

**Description:** Analyze an uploaded audio file for deepfake detection.

**Request:**
```http
POST /api/detect/audio/ HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary

------WebKitFormBoundary
Content-Disposition: form-data; name="audio"; filename="test_audio.wav"
Content-Type: audio/wav

[Binary audio data]
------WebKitFormBoundary--
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| audio | File | Yes | Audio file to analyze (MP3, WAV, M4A, OGG) |

**Supported Formats:** MP3, WAV, M4A, OGG

**File Size Limit:** 50MB

**Response (200 OK):**
```json
{
  "success": true,
  "file_type": "audio",
  "predictions": {
    "audio": {
      "real_probability": 0.92,
      "fake_probability": 0.08,
      "prediction": "real",
      "confidence": 0.92,
      "segments_analyzed": 5,
      "duration_seconds": 15.3
    }
  },
  "processing_time": 1.87,
  "timestamp": "2025-10-27T12:34:56Z"
}
```

**Error Response (400 Bad Request) - Validation Error:**
```json
{
  "success": false,
  "error": "Audio file too large. Maximum size: 50MB",
  "error_type": "validation_error"
}
```

**Error Response (400 Bad Request) - Processing Error:**
```json
{
  "success": false,
  "error": "Audio appears to be silent - cannot analyze",
  "error_type": "processing_error"
}
```

**Error Response (500 Internal Server Error):**
```json
{
  "success": false,
  "error": "Internal server error during audio processing",
  "error_type": "server_error"
}
```

---

## Response Fields

### Success Response

| Field | Type | Description |
|-------|------|-------------|
| success | Boolean | Whether the request was successful |
| file_type | String | Type of file analyzed ("video" or "audio") |
| predictions | Object | Prediction results |
| processing_time | Float | Time taken to process in seconds |
| timestamp | String | ISO 8601 timestamp of the response |

### Prediction Object (Video)

| Field | Type | Description |
|-------|------|-------------|
| real_probability | Float | Probability that content is real (0.0-1.0) |
| fake_probability | Float | Probability that content is fake (0.0-1.0) |
| prediction | String | Final prediction ("real" or "fake") |
| confidence | Float | Confidence score (0.0-1.0) |
| frames_analyzed | Integer | Number of frames analyzed |

### Prediction Object (Audio)

| Field | Type | Description |
|-------|------|-------------|
| real_probability | Float | Probability that audio is real (0.0-1.0) |
| fake_probability | Float | Probability that audio is fake (0.0-1.0) |
| prediction | String | Final prediction ("real" or "fake") |
| confidence | Float | Confidence score (0.0-1.0) |
| segments_analyzed | Integer | Number of audio segments analyzed |
| duration_seconds | Float | Duration of audio in seconds (optional) |

### Error Response

| Field | Type | Description |
|-------|------|-------------|
| success | Boolean | Always false for errors |
| error | String | Human-readable error message |
| error_type | String | Error category (see below) |

**Error Types:**
- `validation_error`: Invalid input (wrong format, file too large, etc.)
- `processing_error`: Error during processing (no faces, silent audio, etc.)
- `server_error`: Internal server error
- `network_error`: Network-related error (client-side)

---

## Code Examples

### Python (requests)

```python
import requests

# Health check
response = requests.get('http://localhost:8000/api/health/')
print(response.json())

# Video detection
with open('test_video.mp4', 'rb') as video_file:
    files = {'video': video_file}
    data = {'analyze_audio': 'true'}
    response = requests.post(
        'http://localhost:8000/api/detect/video/',
        files=files,
        data=data
    )
    print(response.json())

# Audio detection
with open('test_audio.wav', 'rb') as audio_file:
    files = {'audio': audio_file}
    response = requests.post(
        'http://localhost:8000/api/detect/audio/',
        files=files
    )
    print(response.json())
```

### JavaScript (Fetch API)

```javascript
// Health check
fetch('http://localhost:8000/api/health/')
  .then(response => response.json())
  .then(data => console.log(data));

// Video detection
const formData = new FormData();
formData.append('video', videoFile);
formData.append('analyze_audio', 'true');

fetch('http://localhost:8000/api/detect/video/', {
  method: 'POST',
  body: formData
})
  .then(response => response.json())
  .then(data => console.log(data));

// Audio detection
const audioFormData = new FormData();
audioFormData.append('audio', audioFile);

fetch('http://localhost:8000/api/detect/audio/', {
  method: 'POST',
  body: audioFormData
})
  .then(response => response.json())
  .then(data => console.log(data));
```

### cURL

```bash
# Health check
curl http://localhost:8000/api/health/

# Video detection
curl -X POST http://localhost:8000/api/detect/video/ \
  -F "video=@test_video.mp4" \
  -F "analyze_audio=true"

# Audio detection
curl -X POST http://localhost:8000/api/detect/audio/ \
  -F "audio=@test_audio.wav"
```

---

## Rate Limiting

Currently, no rate limiting is implemented. For production deployments, consider implementing rate limiting using:
- Django rate limit middleware
- NGINX rate limiting
- API Gateway rate limiting (AWS, Azure, GCP)

Recommended limits:
- 10 requests per minute per IP
- 100 requests per hour per IP

---

## CORS (Cross-Origin Resource Sharing)

CORS is configured in Django settings:
- Development: All origins allowed
- Production: Configure `CORS_ALLOWED_ORIGINS` in `.env`

Example `.env` configuration:
```
CORS_ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

---

## Error Handling Best Practices

1. **Always check the `success` field** before processing results
2. **Handle different error types** appropriately:
   - `validation_error`: Show user-friendly message, allow retry
   - `processing_error`: Explain issue (no faces, silent audio)
   - `server_error`: Generic error message, retry after delay
3. **Implement timeout handling** for long-running requests
4. **Add retry logic** for server errors (with exponential backoff)

---

## Performance Considerations

### Processing Times

Typical processing times (varies by hardware):
- Video (10 seconds, 720p): 3-7 seconds
- Audio (10 seconds): 1-3 seconds
- Video with audio analysis: 5-10 seconds

### Optimization Tips

1. **Compress files** before uploading (affects quality vs speed tradeoff)
2. **Use appropriate video resolution** (720p recommended, 1080p slower)
3. **Batch requests** if analyzing multiple files (use parallel requests)
4. **Implement caching** for repeated analyses (same file)

---

## Support

For API issues or questions:
- GitHub Issues: https://github.com/yourusername/DeepSafe/issues
- Email: support@deepsafe.example.com

---

*Last updated: October 27, 2025*
