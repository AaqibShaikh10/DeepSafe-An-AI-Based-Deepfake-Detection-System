# DeepSafe Deployment Guide

Complete step-by-step guide to deploy DeepSafe with trained models in production.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Local Deployment](#local-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Production Optimization](#production-optimization)
7. [Monitoring & Maintenance](#monitoring--maintenance)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Files
Before deployment, ensure you have:
- ✅ Trained models:
  - `models/video_cnn/video_model.h5` (~80 MB)
  - `models/audio_rnn/audio_model.h5` (~20 MB)
- ✅ All code from GitHub repository
- ✅ Environment variables configured

### Verify Models
```bash
# Check if models exist and are valid
ls -lh models/video_cnn/video_model.h5
ls -lh models/audio_rnn/audio_model.h5

# Test models
python -c "
import tensorflow as tf
vm = tf.keras.models.load_model('models/video_cnn/video_model.h5')
am = tf.keras.models.load_model('models/audio_rnn/audio_model.h5')
print('✓ Video model loaded successfully')
print('✓ Audio model loaded successfully')
"
```

**Expected output**:
```
✓ Video model loaded successfully
✓ Audio model loaded successfully
```

---

## Pre-Deployment Checklist

### Security Configuration

1. **Generate Secret Key**:
```bash
python -c "import secrets; print(secrets.token_urlsafe(50))"
# Copy output to .env file
```

2. **Configure Environment**:
```bash
cp .env.example .env
nano .env  # or use your preferred editor
```

Edit `.env`:
```bash
# Production settings
DEBUG=False
SECRET_KEY=<your-generated-secret-key>
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com,your-server-ip

# Database (use PostgreSQL for production)
DATABASE_URL=postgresql://user:password@localhost:5432/deepsafe

# File limits
MAX_VIDEO_SIZE_MB=100
MAX_AUDIO_SIZE_MB=50

# Model paths (verify these exist)
VIDEO_MODEL_PATH=models/video_cnn/video_model.h5
AUDIO_MODEL_PATH=models/audio_rnn/audio_model.h5

# Processing timeouts
VIDEO_TIMEOUT_SECONDS=60
AUDIO_TIMEOUT_SECONDS=30

# CORS (adjust for your domain)
CORS_ALLOWED_ORIGINS=https://yourdomain.com
```

3. **Verify Directory Permissions**:
```bash
# Ensure uploads directory is writable
mkdir -p uploads
chmod 755 uploads

# Ensure models are readable
chmod 644 models/video_cnn/video_model.h5
chmod 644 models/audio_rnn/audio_model.h5
```

---

## Local Deployment

### Option 1: Quick Start (Development)

```bash
# 1. Navigate to project
cd DeepSafe-An-AI-Based-Deepfake-Detection-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run migrations
cd backend
python manage.py migrate

# 4. Create superuser (optional)
python manage.py createsuperuser

# 5. Collect static files
python manage.py collectstatic --noinput

# 6. Start server
python manage.py runserver 0.0.0.0:8000
```

**Access**:
- Web Interface: http://localhost:8000
- Admin Panel: http://localhost:8000/admin/
- API Health: http://localhost:8000/api/health/

### Option 2: Production Server (Gunicorn)

```bash
# 1. Install gunicorn
pip install gunicorn

# 2. Navigate to backend
cd backend

# 3. Run with gunicorn
gunicorn --bind 0.0.0.0:8000 \
         --workers 4 \
         --timeout 120 \
         --access-logfile access.log \
         --error-logfile error.log \
         deepsafe_project.wsgi:application

# Or use systemd service (see below)
```

---

## Docker Deployment

### Step 1: Prepare Models

```bash
# Ensure models are in the correct location
ls -lh models/video_cnn/video_model.h5
ls -lh models/audio_rnn/audio_model.h5

# If models are large, consider using Docker volumes
```

### Step 2: Configure Environment

```bash
# Copy environment file
cp .env.example .env

# Edit with production values
nano .env
```

### Step 3: Build and Run

```bash
# Build Docker image
docker build -t deepsafe:latest .

# Run container
docker run -d \
  --name deepsafe \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/uploads:/app/uploads \
  --env-file .env \
  --restart unless-stopped \
  deepsafe:latest

# Check logs
docker logs -f deepsafe

# Check health
curl http://localhost:8000/api/health/
```

**Expected output**:
```json
{
  "status": "healthy",
  "video_model_loaded": true,
  "audio_model_loaded": true,
  "timestamp": "2025-01-27T12:00:00Z"
}
```

### Step 4: Docker Compose (Recommended)

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Restart
docker-compose restart
```

**Verify deployment**:
```bash
# Test health endpoint
curl http://localhost:8000/api/health/

# Test with sample file
curl -X POST http://localhost:8000/api/detect/video/ \
  -F "video=@test_video.mp4"
```

---

## Cloud Deployment

### AWS Deployment

#### Option 1: EC2 Instance

**1. Launch EC2 Instance**:
```
- AMI: Ubuntu 22.04 LTS
- Instance Type: t3.xlarge or c5.2xlarge
- Storage: 100 GB SSD
- Security Group:
  - Port 22 (SSH)
  - Port 80 (HTTP)
  - Port 443 (HTTPS)
```

**2. Connect and Setup**:
```bash
# SSH to instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Clone repository
git clone https://github.com/AaqibShaikh10/DeepSafe-An-AI-Based-Deepfake-Detection-System.git
cd DeepSafe-An-AI-Based-Deepfake-Detection-System

# Copy models (upload via SCP or download from S3)
scp -i your-key.pem models/*.h5 ubuntu@your-instance-ip:~/DeepSafe-An-AI-Based-Deepfake-Detection-System/models/

# Configure environment
cp .env.example .env
nano .env  # Set production values

# Deploy with Docker
docker-compose up -d
```

**3. Setup NGINX Reverse Proxy**:
```bash
# Install NGINX
sudo apt update
sudo apt install nginx

# Configure NGINX
sudo nano /etc/nginx/sites-available/deepsafe
```

Add configuration:
```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    client_max_body_size 100M;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }

    location /static/ {
        alias /home/ubuntu/DeepSafe-An-AI-Based-Deepfake-Detection-System/backend/staticfiles/;
    }
}
```

Enable and restart:
```bash
sudo ln -s /etc/nginx/sites-available/deepsafe /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

**4. Setup SSL with Let's Encrypt**:
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal is configured automatically
```

#### Option 2: AWS ECS (Elastic Container Service)

**1. Push Docker Image to ECR**:
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account-id.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag deepsafe:latest your-account-id.dkr.ecr.us-east-1.amazonaws.com/deepsafe:latest
docker push your-account-id.dkr.ecr.us-east-1.amazonaws.com/deepsafe:latest
```

**2. Create ECS Task Definition**:
```json
{
  "family": "deepsafe-task",
  "networkMode": "awsvpc",
  "containerDefinitions": [
    {
      "name": "deepsafe",
      "image": "your-account-id.dkr.ecr.us-east-1.amazonaws.com/deepsafe:latest",
      "memory": 4096,
      "cpu": 2048,
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "DEBUG", "value": "False"},
        {"name": "SECRET_KEY", "value": "your-secret-key"}
      ]
    }
  ],
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096"
}
```

**3. Create ECS Service with Load Balancer**:
- Use Application Load Balancer
- Configure health check: `/api/health/`
- Setup auto-scaling based on CPU/Memory

### Google Cloud Platform

#### Deploy to Cloud Run

```bash
# 1. Build and push to GCR
gcloud builds submit --tag gcr.io/your-project-id/deepsafe

# 2. Deploy to Cloud Run
gcloud run deploy deepsafe \
  --image gcr.io/your-project-id/deepsafe \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --timeout 120s \
  --max-instances 10 \
  --allow-unauthenticated

# 3. Get service URL
gcloud run services describe deepsafe --platform managed --region us-central1 --format 'value(status.url)'
```

### Azure

#### Deploy to Azure App Service

```bash
# 1. Create resource group
az group create --name DeepSafeRG --location eastus

# 2. Create App Service plan
az appservice plan create \
  --name DeepSafePlan \
  --resource-group DeepSafeRG \
  --sku P1V2 \
  --is-linux

# 3. Create web app
az webapp create \
  --resource-group DeepSafeRG \
  --plan DeepSafePlan \
  --name deepsafe-app \
  --deployment-container-image-name deepsafe:latest

# 4. Configure environment
az webapp config appsettings set \
  --resource-group DeepSafeRG \
  --name deepsafe-app \
  --settings DEBUG=False SECRET_KEY=your-secret-key
```

---

## Production Optimization

### 1. Database Configuration

**Switch to PostgreSQL**:
```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb deepsafe
sudo -u postgres createuser deepsafe_user
sudo -u postgres psql

# In PostgreSQL:
ALTER USER deepsafe_user WITH PASSWORD 'secure-password';
GRANT ALL PRIVILEGES ON DATABASE deepsafe TO deepsafe_user;
```

Update `.env`:
```
DATABASE_URL=postgresql://deepsafe_user:secure-password@localhost:5432/deepsafe
```

Update `settings.py`:
```python
import dj_database_url

DATABASES = {
    'default': dj_database_url.config(
        default=config('DATABASE_URL')
    )
}
```

### 2. Caching Setup

**Install Redis**:
```bash
# Install Redis
sudo apt install redis-server

# Configure Django caching
pip install django-redis
```

Update `settings.py`:
```python
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}

# Cache API responses
REST_FRAMEWORK = {
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle'
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '100/hour',
        'user': '1000/hour'
    }
}
```

### 3. Static Files with CDN

**Use WhiteNoise**:
```bash
pip install whitenoise
```

Update `settings.py`:
```python
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',  # Add this
    # ... other middleware
]

STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
```

**Or use S3/CloudFront**:
```bash
pip install django-storages boto3
```

Update `settings.py`:
```python
# AWS S3 Configuration
AWS_ACCESS_KEY_ID = config('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = config('AWS_SECRET_ACCESS_KEY')
AWS_STORAGE_BUCKET_NAME = config('AWS_STORAGE_BUCKET_NAME')
AWS_S3_REGION_NAME = 'us-east-1'
AWS_S3_CUSTOM_DOMAIN = f'{AWS_STORAGE_BUCKET_NAME}.s3.amazonaws.com'

# Static files
STATICFILES_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
STATIC_URL = f'https://{AWS_S3_CUSTOM_DOMAIN}/static/'
```

### 4. Performance Tuning

**Gunicorn Configuration**:
```bash
# Create gunicorn config
nano gunicorn_config.py
```

```python
# gunicorn_config.py
import multiprocessing

bind = "0.0.0.0:8000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 5

# Logging
accesslog = "access.log"
errorlog = "error.log"
loglevel = "info"

# Process naming
proc_name = "deepsafe"

# Server mechanics
daemon = False
pidfile = "gunicorn.pid"
```

Run with config:
```bash
gunicorn -c gunicorn_config.py deepsafe_project.wsgi:application
```

### 5. Systemd Service

Create service file:
```bash
sudo nano /etc/systemd/system/deepsafe.service
```

```ini
[Unit]
Description=DeepSafe Deepfake Detection Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/DeepSafe-An-AI-Based-Deepfake-Detection-System/backend
Environment="PATH=/home/ubuntu/DeepSafe-An-AI-Based-Deepfake-Detection-System/venv/bin"
ExecStart=/home/ubuntu/DeepSafe-An-AI-Based-Deepfake-Detection-System/venv/bin/gunicorn \
    --config /home/ubuntu/DeepSafe-An-AI-Based-Deepfake-Detection-System/gunicorn_config.py \
    deepsafe_project.wsgi:application
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable deepsafe
sudo systemctl start deepsafe
sudo systemctl status deepsafe
```

---

## Monitoring & Maintenance

### 1. Health Monitoring

**Setup monitoring script**:
```bash
# Create monitoring script
nano monitor_health.sh
```

```bash
#!/bin/bash
ENDPOINT="http://localhost:8000/api/health/"
RESPONSE=$(curl -s $ENDPOINT)

if echo "$RESPONSE" | grep -q '"status":"healthy"'; then
    echo "$(date): Service is healthy"
else
    echo "$(date): Service is DOWN!"
    # Send alert (email, Slack, etc.)
    # Restart service
    sudo systemctl restart deepsafe
fi
```

**Setup cron job**:
```bash
crontab -e

# Add line (check every 5 minutes):
*/5 * * * * /home/ubuntu/monitor_health.sh >> /home/ubuntu/monitor.log 2>&1
```

### 2. Logging

**Configure Django logging**:
```python
# settings.py
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/deepsafe/django.log',
            'maxBytes': 1024*1024*15,  # 15MB
            'backupCount': 10,
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['file'],
        'level': 'INFO',
    },
}
```

**Setup log rotation**:
```bash
sudo nano /etc/logrotate.d/deepsafe
```

```
/var/log/deepsafe/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 ubuntu ubuntu
    sharedscripts
    postrotate
        systemctl reload deepsafe > /dev/null
    endscript
}
```

### 3. Backup Strategy

**Backup models and database**:
```bash
# Create backup script
nano backup.sh
```

```bash
#!/bin/bash
BACKUP_DIR="/home/ubuntu/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup database
pg_dump deepsafe > "$BACKUP_DIR/db_$DATE.sql"

# Backup models
tar -czf "$BACKUP_DIR/models_$DATE.tar.gz" models/

# Backup config
cp .env "$BACKUP_DIR/env_$DATE"

# Delete backups older than 30 days
find $BACKUP_DIR -type f -mtime +30 -delete

echo "Backup completed: $DATE"
```

**Setup daily backups**:
```bash
crontab -e

# Add line (daily at 2 AM):
0 2 * * * /home/ubuntu/backup.sh >> /home/ubuntu/backup.log 2>&1
```

### 4. Update Strategy

```bash
# Create update script
nano update.sh
```

```bash
#!/bin/bash
set -e

echo "Starting update..."

# Pull latest code
git pull origin main

# Backup current state
./backup.sh

# Update dependencies
pip install -r requirements.txt

# Run migrations
cd backend
python manage.py migrate

# Collect static files
python manage.py collectstatic --noinput

# Restart service
sudo systemctl restart deepsafe

echo "Update completed successfully"
```

---

## Troubleshooting

### Issue 1: Models Not Loading

**Symptoms**:
```
⚠ Warning: Failed to load models
```

**Solutions**:
```bash
# Check file exists
ls -lh models/video_cnn/video_model.h5
ls -lh models/audio_rnn/audio_model.h5

# Check permissions
chmod 644 models/video_cnn/video_model.h5
chmod 644 models/audio_rnn/audio_model.h5

# Test loading manually
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('models/video_cnn/video_model.h5')
print('Model loaded successfully')
"

# Check environment variable
echo $VIDEO_MODEL_PATH
```

### Issue 2: CORS Errors

**Symptoms**:
```
Access to fetch at 'http://...' from origin 'http://...' has been blocked by CORS policy
```

**Solutions**:
```python
# In settings.py, update CORS settings:
CORS_ALLOWED_ORIGINS = [
    "https://yourdomain.com",
    "https://www.yourdomain.com",
]

# Or for development:
CORS_ALLOW_ALL_ORIGINS = True  # Only for development!
```

### Issue 3: File Upload Errors

**Symptoms**:
```
File size exceeds 100MB
```

**Solutions**:
```bash
# Increase NGINX upload limit
sudo nano /etc/nginx/nginx.conf

# Add in http block:
client_max_body_size 100M;

# Restart NGINX
sudo systemctl restart nginx

# Update Django settings
FILE_UPLOAD_MAX_MEMORY_SIZE = 100 * 1024 * 1024
DATA_UPLOAD_MAX_MEMORY_SIZE = 100 * 1024 * 1024
```

### Issue 4: Slow Inference

**Symptoms**:
- Video analysis takes >10 seconds
- High CPU usage

**Solutions**:
```bash
# 1. Use GPU for inference (if available)
# Ensure TensorFlow can access GPU:
python -c "
import tensorflow as tf
print('GPUs available:', tf.config.list_physical_devices('GPU'))
"

# 2. Optimize model inference
# Use TensorFlow Lite or ONNX Runtime

# 3. Add request queue with Celery
pip install celery redis

# 4. Increase worker count
gunicorn --workers 8  # More workers
```

### Issue 5: High Memory Usage

**Symptoms**:
- Server runs out of memory
- OOM killer terminates process

**Solutions**:
```bash
# 1. Reduce worker count
gunicorn --workers 2  # Fewer workers

# 2. Add memory limits in Docker
docker run --memory="4g" --memory-swap="4g" deepsafe:latest

# 3. Setup swap space
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 4. Monitor memory usage
htop  # or
watch -n 1 free -h
```

---

## Performance Benchmarks

### Expected Performance (Production)

**Server Specs**: AWS c5.2xlarge (8 vCPU, 16 GB RAM)

| Operation | Time | Throughput |
|-----------|------|------------|
| Video analysis (720p, 10s) | 3-5 seconds | 12-20 videos/min |
| Audio analysis (10s) | 1-2 seconds | 30-60 audios/min |
| Combined analysis | 5-7 seconds | 8-12 requests/min |

**Concurrent Requests**:
- 4 workers: 4-8 concurrent requests
- 8 workers: 8-16 concurrent requests

### Load Testing

```bash
# Install Apache Bench
sudo apt install apache2-utils

# Test API endpoint
ab -n 100 -c 10 http://localhost:8000/api/health/

# Expected results:
# - Requests per second: >100
# - Time per request: <100ms
```

---

## Security Checklist

Before going live:

- [ ] `DEBUG=False` in production
- [ ] Strong `SECRET_KEY` configured
- [ ] `ALLOWED_HOSTS` restricted to your domain
- [ ] HTTPS enabled with valid SSL certificate
- [ ] Database using strong password
- [ ] File upload size limits enforced
- [ ] Rate limiting configured
- [ ] CORS properly configured
- [ ] Security headers enabled (HSTS, CSP)
- [ ] Regular backups automated
- [ ] Monitoring and alerts setup
- [ ] Logs being collected and rotated
- [ ] Firewall configured (UFW or security groups)

---

## Post-Deployment Verification

```bash
# 1. Health check
curl https://yourdomain.com/api/health/

# 2. Test video upload
curl -X POST https://yourdomain.com/api/detect/video/ \
  -F "video=@test.mp4"

# 3. Test audio upload
curl -X POST https://yourdomain.com/api/detect/audio/ \
  -F "audio=@test.wav"

# 4. Check response times
time curl https://yourdomain.com/api/health/

# 5. Monitor logs
tail -f /var/log/deepsafe/django.log

# 6. Check resource usage
htop
```

---

## Success Criteria

Your deployment is successful when:

✅ Health endpoint returns `"status": "healthy"`
✅ Both models load successfully
✅ Video analysis completes in <5 seconds
✅ Audio analysis completes in <2 seconds
✅ No errors in logs
✅ HTTPS working correctly
✅ Monitoring is active
✅ Backups are running
✅ Performance meets benchmarks

---

**Deployment complete! Your DeepSafe system is live.** 🚀

For support:
- GitHub Issues: https://github.com/AaqibShaikh10/DeepSafe-An-AI-Based-Deepfake-Detection-System/issues
- Email: support@deepsafe.example.com
