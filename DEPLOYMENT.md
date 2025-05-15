# Deployment Guide

This guide explains how to deploy the Hate Speech Classification System in a production environment.

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Virtual environment
- Web server (e.g., Nginx, Apache)
- Process manager (e.g., Gunicorn, uWSGI)
- SSL certificate for HTTPS

## Environment Setup

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install production dependencies:
```bash
pip install -r requirements.txt
pip install gunicorn  # or uwsgi
```

3. Create `.env` file with production settings:
```bash
# Flask configuration
FLASK_ENV=production
FLASK_APP=app.py

# Server configuration
HOST=0.0.0.0
PORT=8080

# Model configuration
MODEL_DIR=models
CONFIDENCE_THRESHOLD=0.3
HIGH_CONFIDENCE_THRESHOLD=0.8
HATE_SPEECH_THRESHOLD=0.35

# Logging configuration
LOG_LEVEL=INFO
```

## Directory Structure

Ensure your production environment has the following structure:
```
/opt/hate-speech-classifier/
├── app.py
├── src/
│   └── main.py
├── models/
│   ├── naive_bayes.pkl
│   ├── logistic_regression.pkl
│   ├── tfidf_vectorizer.pkl
│   └── config.json
├── logs/
│   └── app.log
├── templates/
│   └── index.html
├── venv/
├── .env
└── requirements.txt
```

## Running with Gunicorn

1. Test Gunicorn configuration:
```bash
gunicorn --bind 0.0.0.0:8080 app:app
```

2. Create a systemd service file `/etc/systemd/system/hate-speech-classifier.service`:
```ini
[Unit]
Description=Hate Speech Classification Service
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/opt/hate-speech-classifier
Environment="PATH=/opt/hate-speech-classifier/venv/bin"
ExecStart=/opt/hate-speech-classifier/venv/bin/gunicorn --workers 4 --bind unix:hate-speech-classifier.sock -m 007 app:app

[Install]
WantedBy=multi-user.target
```

3. Start and enable the service:
```bash
sudo systemctl start hate-speech-classifier
sudo systemctl enable hate-speech-classifier
```

## Nginx Configuration

1. Create Nginx configuration `/etc/nginx/sites-available/hate-speech-classifier`:
```nginx
server {
    listen 80;
    server_name your_domain.com;

    location / {
        include proxy_params;
        proxy_pass http://unix:/opt/hate-speech-classifier/hate-speech-classifier.sock;
    }
}
```

2. Enable the site and restart Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/hate-speech-classifier /etc/nginx/sites-enabled
sudo systemctl restart nginx
```

## SSL Configuration

1. Install Certbot:
```bash
sudo apt install certbot python3-certbot-nginx
```

2. Obtain SSL certificate:
```bash
sudo certbot --nginx -d your_domain.com
```

## Monitoring and Maintenance

1. Check application logs:
```bash
tail -f /opt/hate-speech-classifier/logs/app.log
```

2. Monitor system resources:
```bash
htop
```

3. Check service status:
```bash
sudo systemctl status hate-speech-classifier
```

## Health Check

The application provides a health check endpoint at `/health`. Monitor this endpoint to ensure the service is running correctly:
```bash
curl http://your_domain.com/health
```

Expected response:
```json
{
    "status": "healthy",
    "models_loaded": true
}
```

## Security Considerations

1. Ensure proper file permissions:
```bash
sudo chown -R www-data:www-data /opt/hate-speech-classifier
sudo chmod -R 755 /opt/hate-speech-classifier
```

2. Configure firewall:
```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
```

3. Set secure headers in Nginx:
```nginx
add_header X-Frame-Options "SAMEORIGIN";
add_header X-XSS-Protection "1; mode=block";
add_header X-Content-Type-Options "nosniff";
```

## Backup and Recovery

1. Backup model files regularly:
```bash
rsync -av /opt/hate-speech-classifier/models/ /backup/models/
```

2. Backup configuration:
```bash
cp /opt/hate-speech-classifier/.env /backup/
```

## Troubleshooting

1. If the service fails to start:
```bash
sudo journalctl -u hate-speech-classifier
```

2. If Nginx returns 502 Bad Gateway:
```bash
sudo nginx -t
sudo tail -f /var/log/nginx/error.log
```

3. If models fail to load:
```bash
sudo -u www-data python3 -c "from app import load_models; load_models()"
```

## Performance Tuning

1. Adjust Gunicorn workers:
```bash
workers = (2 * cpu_cores) + 1
```

2. Configure Nginx worker connections:
```nginx
worker_processes auto;
worker_connections 1024;
```

3. Enable Nginx caching for static files:
```nginx
location /static/ {
    expires 1h;
    add_header Cache-Control "public, no-transform";
}
``` 