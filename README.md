# DeepSafe: AI-Based Deepfake Detection System

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![Django](https://img.shields.io/badge/django-4.2-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

DeepSafe is a comprehensive AI-powered web application that detects deepfakes in both video and audio files using state-of-the-art deep learning models.

## Features

- **Video Deepfake Detection**: Analyzes facial manipulations using EfficientNet-B4 CNN
- **Audio Deepfake Detection**: Detects voice forgeries using LSTM RNN
- **Multi-Modal Analysis**: Combined video + audio analysis for enhanced accuracy
- **Web Interface**: User-friendly HTML/CSS/JavaScript interface for file uploads
- **REST API**: Django REST Framework backend for easy integration
- **Real-Time Results**: Displays probability scores and confidence levels
- **Production Ready**: Docker containerization and comprehensive testing

## Technology Stack

### Backend
- **Framework**: Django 4.2 + Django REST Framework
- **AI/ML**: TensorFlow 2.15 (deployment), PyTorch 2.2 (training)
- **Computer Vision**: OpenCV, MTCNN face detection
- **Audio Processing**: Librosa, MoviePy

### Frontend
- **Interface**: HTML5, CSS3, JavaScript (ES6+)
- **Styling**: Custom responsive CSS with animations
- **Communication**: Fetch API for async requests

### Models
- **Video Model**: EfficientNet-B4 (fine-tuned on FaceForensics++)
- **Audio Model**: LSTM RNN (trained on ASVspoof 2019)

## Installation

### Prerequisites

- Python 3.12 or higher
- Docker (optional, for containerized deployment)
- CUDA-capable GPU (optional, for faster inference)

### Option 1: Docker Deployment (Recommended)

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/DeepSafe-An-AI-Based-Deepfake-Detection-System.git
cd DeepSafe-An-AI-Based-Deepfake-Detection-System
```

2. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env file with your settings
```

3. **Build and run with Docker Compose:**
```bash
docker-compose up -d
```

4. **Access the application:**
- Web Interface: http://localhost:8000
- API Health Check: http://localhost:8000/api/health/

### Option 2: Manual Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/DeepSafe-An-AI-Based-Deepfake-Detection-System.git
cd DeepSafe-An-AI-Based-Deepfake-Detection-System
```

2. **Create and activate virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env file with your settings
```

5. **Run database migrations:**
```bash
cd backend
python manage.py migrate
```

6. **Collect static files:**
```bash
python manage.py collectstatic --noinput
```

7. **Create superuser (optional):**
```bash
python manage.py createsuperuser
```

8. **Run the development server:**
```bash
python manage.py runserver 0.0.0.0:8000
```

9. **Access the application:**
- Web Interface: http://localhost:8000
- Admin Panel: http://localhost:8000/admin/
- API Health Check: http://localhost:8000/api/health/

## Model Training

### Dataset Preparation

DeepSafe models are trained on industry-standard datasets:

**Video Model:**
- [FaceForensics++](https://github.com/ondyari/FaceForensics) (primary)
- [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics) (optional)

**Audio Model:**
- [ASVspoof 2019](https://www.asvspoof.org/index2019.html) (LA scenario)
- [WaveFake](https://github.com/RUB-SysSec/WaveFake) (optional)

### Training Workflow

1. **Download and organize datasets:**
```bash
# Place datasets in data/raw/ following the structure:
# data/raw/faceforensics/
# data/raw/asvspoof2019/
```

2. **Preprocess video data:**
```bash
python preprocessing/video_preprocessing.py \
    --input data/raw/faceforensics \
    --output data/processed/video_frames
```

3. **Preprocess audio data:**
```bash
python preprocessing/audio_preprocessing.py \
    --input data/raw/asvspoof2019 \
    --output data/processed/audio_spectrograms
```

4. **Train video model:**
```bash
python training/train_video_cnn.py \
    --data_dir data/processed \
    --output_dir models/video_cnn \
    --epochs 30 \
    --batch_size 32
```

5. **Train audio model:**
```bash
python training/train_audio_rnn.py \
    --data_dir data/processed \
    --output_dir models/audio_rnn \
    --epochs 40 \
    --batch_size 64
```

6. **Evaluate models:**
```bash
python training/evaluate.py \
    --model models/video_cnn/video_model.pt \
    --model_type video
```

7. **Convert models to TensorFlow:**
```bash
python training/convert_models.py \
    --models_dir models
```

### Model Performance

**Video Model (EfficientNet-B4):**
- Accuracy: ≥ 90%
- AUC: ≥ 0.95
- Inference Time: < 5 seconds (720p video, 10 seconds)

**Audio Model (LSTM):**
- Accuracy: ≥ 88%
- AUC: ≥ 0.93
- Inference Time: < 2 seconds (10-second audio clip)

## Usage

### Web Interface

1. **Navigate to the application:**
Open http://localhost:8000 in your browser

2. **Upload a file:**
- Click the upload area or drag and drop a video/audio file
- Supported formats:
  - **Video**: MP4, AVI, MOV, WEBM (max 100MB)
  - **Audio**: MP3, WAV, M4A, OGG (max 50MB)

3. **Optional - Analyze audio from video:**
Check "Also analyze audio track from video" for multi-modal analysis

4. **Click "Analyze for Deepfakes":**
Wait for processing (2-10 seconds typically)

5. **View results:**
- **Prediction**: REAL or FAKE
- **Probabilities**: Real vs Fake percentages
- **Confidence**: Overall confidence score
- **Metadata**: Processing time, frames/segments analyzed

### REST API

#### Health Check
```bash
GET /api/health/

Response:
{
  "status": "healthy",
  "video_model_loaded": true,
  "audio_model_loaded": true,
  "timestamp": "2025-10-27T12:34:56Z"
}
```

#### Video Detection
```bash
POST /api/detect/video/
Content-Type: multipart/form-data

Parameters:
- video (file): Video file to analyze
- analyze_audio (boolean, optional): Also analyze audio track

Response:
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

#### Audio Detection
```bash
POST /api/detect/audio/
Content-Type: multipart/form-data

Parameters:
- audio (file): Audio file to analyze

Response:
{
  "success": true,
  "file_type": "audio",
  "predictions": {
    "audio": {
      "real_probability": 0.92,
      "fake_probability": 0.08,
      "prediction": "real",
      "confidence": 0.92,
      "segments_analyzed": 5
    }
  },
  "processing_time": 1.87,
  "timestamp": "2025-10-27T12:34:56Z"
}
```

For complete API documentation, see [docs/api_documentation.md](docs/api_documentation.md).

## Project Structure

```
DeepSafe-An-AI-Based-Deepfake-Detection-System/
├── backend/                    # Django backend
│   ├── deepsafe_project/      # Django project settings
│   └── detection/             # Detection app (models, views, API)
├── frontend/                   # Web interface
│   ├── static/                # CSS, JavaScript, images
│   └── templates/             # HTML templates
├── models/                     # Trained AI models (.h5 files)
│   ├── video_cnn/
│   └── audio_rnn/
├── training/                   # Model training scripts
├── preprocessing/              # Data preprocessing utilities
├── tests/                      # Test suite
├── docs/                       # Documentation
├── data/                       # Datasets (not in git)
├── uploads/                    # Temporary file storage
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose configuration
└── README.md                   # This file
```

## Testing

Run the test suite:

```bash
# Run all tests
python manage.py test

# Run with coverage
pytest --cov=backend --cov-report=html

# Run specific test file
python manage.py test tests.test_api
```

## Deployment

### Production Checklist

Before deploying to production:

- [ ] Change `SECRET_KEY` in `.env` to a strong random string
- [ ] Set `DEBUG=False` in `.env`
- [ ] Configure `ALLOWED_HOSTS` with your domain
- [ ] Use PostgreSQL instead of SQLite
- [ ] Set up HTTPS (use nginx as reverse proxy)
- [ ] Configure proper CORS settings
- [ ] Set up monitoring and logging
- [ ] Implement rate limiting
- [ ] Configure backup strategy

### Deployment Options

**Docker (Recommended):**
```bash
docker-compose up -d
```

**Manual with Gunicorn:**
```bash
cd backend
gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 120 deepsafe_project.wsgi:application
```

**Cloud Platforms:**
- AWS: Use ECS or Elastic Beanstalk
- Google Cloud: Use Cloud Run or App Engine
- Azure: Use App Service
- Heroku: Use Heroku Buildpack

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 for Python code
- Use Black for code formatting: `black .`
- Run linters: `flake8 .`
- Write tests for new features

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

### Datasets
- [FaceForensics++](https://github.com/ondyari/FaceForensics) by Röss et al.
- [ASVspoof 2019](https://www.asvspoof.org/index2019.html) Challenge
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics) by Li et al.

### Research
- EfficientNet: Tan & Le, 2019
- Face Detection: MTCNN by Zhang et al., 2016
- Audio Processing: Librosa by McFee et al.

### Libraries
- Django and Django REST Framework
- TensorFlow and PyTorch
- OpenCV and Librosa
- All open-source contributors

## Support

For questions, issues, or suggestions:

- **Issues**: [GitHub Issues](https://github.com/yourusername/DeepSafe/issues)
- **Email**: your.email@example.com
- **Documentation**: [Project Wiki](https://github.com/yourusername/DeepSafe/wiki)

## Disclaimer

DeepSafe is a research and educational tool for deepfake detection. While it achieves high accuracy on test datasets, no deepfake detector is 100% accurate. Always use critical thinking and verify important information through multiple sources.

---

**Built with ❤️ by the DeepSafe Team**

*Protecting digital authenticity with AI*