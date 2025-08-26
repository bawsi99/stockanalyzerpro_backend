# Stock Analyzer Pro - Server Deployment Guide

This guide explains how to deploy the Stock Analyzer Pro backend services on a server.

## Overview

The backend consists of two main services:
- **Data Service** (Port 8000): Handles data fetching, WebSocket connections, and real-time data streaming
- **Analysis Service** (Port 8001): Handles stock analysis, AI processing, and chart generation

## Prerequisites

### System Requirements
- Python 3.8 or higher
- pip3
- Git
- At least 2GB RAM
- At least 10GB disk space

### Required API Keys
- **Zerodha API** (for live data streaming)
- **Google Gemini API** (for AI analysis)
- **Supabase** (for analysis storage)

## Installation

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd StockAnalyzer\ Pro/version3.0/3.0/backend
```

### 2. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the backend directory:

```bash
# Zerodha Configuration
ZERODHA_API_KEY=your_zerodha_api_key
ZERODHA_ACCESS_TOKEN=your_zerodha_access_token

# Google Gemini Configuration
GEMINI_API_KEY=your_gemini_api_key
# or
GOOGLE_GEMINI_API_KEY=your_gemini_api_key

# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key

# Optional: Service Configuration
DATA_SERVICE_PORT=8000
ANALYSIS_SERVICE_PORT=8001
HOST=0.0.0.0
LOG_LEVEL=info

# Optional: CORS Configuration
CORS_ORIGINS=http://localhost:3000,http://localhost:8080,http://your-frontend-domain.com
```

## Deployment Options

### Option 1: Using the Shell Script (Recommended)

The easiest way to deploy both services:

```bash
# Make the script executable (if not already)
chmod +x start_services.sh

# Start both services
./start_services.sh start

# Check service status
./start_services.sh status

# View logs
./start_services.sh logs data
./start_services.sh logs analysis

# Stop services
./start_services.sh stop

# Restart services
./start_services.sh restart
```

### Option 2: Using Python Script

```bash
# Simple multiprocessing version
python3 run_services_simple.py

# With custom ports
python3 run_services_simple.py --data-port 8000 --analysis-port 8001 --host 0.0.0.0

# Advanced asyncio version
python3 run_services.py --data-port 8000 --analysis-port 8001 --host 0.0.0.0
```

### Option 3: Manual Deployment

Start each service separately:

```bash
# Terminal 1 - Data Service
python3 -c "
import sys
sys.path.insert(0, '.')
from data_service import app
import uvicorn
uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')
"

# Terminal 2 - Analysis Service
python3 -c "
import sys
sys.path.insert(0, '.')
from analysis_service import app
import uvicorn
uvicorn.run(app, host='0.0.0.0', port=8001, log_level='info')
"
```

## Production Deployment

### Using Systemd (Linux)

1. Create systemd service files:

**Data Service** (`/etc/systemd/system/stock-analyzer-data.service`):
```ini
[Unit]
Description=Stock Analyzer Data Service
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/your/backend
Environment=PATH=/usr/bin:/usr/local/bin
ExecStart=/usr/bin/python3 -c "import sys; sys.path.insert(0, '.'); from data_service import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')"
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Analysis Service** (`/etc/systemd/system/stock-analyzer-analysis.service`):
```ini
[Unit]
Description=Stock Analyzer Analysis Service
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/your/backend
Environment=PATH=/usr/bin:/usr/local/bin
ExecStart=/usr/bin/python3 -c "import sys; sys.path.insert(0, '.'); from analysis_service import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8001, log_level='info')"
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

2. Enable and start services:
```bash
sudo systemctl daemon-reload
sudo systemctl enable stock-analyzer-data
sudo systemctl enable stock-analyzer-analysis
sudo systemctl start stock-analyzer-data
sudo systemctl start stock-analyzer-analysis
```

3. Check status:
```bash
sudo systemctl status stock-analyzer-data
sudo systemctl status stock-analyzer-analysis
```

### Using Docker

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  data-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ZERODHA_API_KEY=${ZERODHA_API_KEY}
      - ZERODHA_ACCESS_TOKEN=${ZERODHA_ACCESS_TOKEN}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_ANON_KEY=${SUPABASE_ANON_KEY}
    volumes:
      - ./logs:/app/logs
    command: python3 -c "import sys; sys.path.insert(0, '.'); from data_service import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)"

  analysis-service:
    build: .
    ports:
      - "8001:8001"
    environment:
      - ZERODHA_API_KEY=${ZERODHA_API_KEY}
      - ZERODHA_ACCESS_TOKEN=${ZERODHA_ACCESS_TOKEN}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_ANON_KEY=${SUPABASE_ANON_KEY}
    volumes:
      - ./logs:/app/logs
      - ./output:/app/output
    command: python3 -c "import sys; sys.path.insert(0, '.'); from analysis_service import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8001)"
```

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000 8001

CMD ["python3", "run_services_simple.py"]
```

Run with Docker Compose:
```bash
docker-compose up -d
```

## Reverse Proxy Configuration

### Nginx Configuration

Create `/etc/nginx/sites-available/stock-analyzer`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Data Service
    location /api/data/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Analysis Service
    location /api/analysis/ {
        proxy_pass http://localhost:8001/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket support for data service
    location /ws/ {
        proxy_pass http://localhost:8000/ws/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/stock-analyzer /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Monitoring and Logging

### Health Checks

Both services provide health check endpoints:

```bash
# Data Service Health
curl http://localhost:8000/health

# Analysis Service Health
curl http://localhost:8001/health
```

### Log Management

Logs are stored in:
- `data_service.log` - Data service logs
- `analysis_service.log` - Analysis service logs

For production, consider using log rotation:

```bash
# Install logrotate
sudo apt-get install logrotate

# Create logrotate configuration
sudo nano /etc/logrotate.d/stock-analyzer
```

Add this configuration:
```
/path/to/your/backend/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 your-user your-group
    postrotate
        systemctl reload stock-analyzer-data
        systemctl reload stock-analyzer-analysis
    endscript
}
```

## Security Considerations

### Firewall Configuration

```bash
# Allow only necessary ports
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

### SSL/TLS Configuration

Use Let's Encrypt for free SSL certificates:

```bash
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### API Key Security

- Store API keys in environment variables, not in code
- Use a secrets management service in production
- Rotate API keys regularly
- Monitor API usage for unusual activity

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using the port
   sudo lsof -i :8000
   sudo lsof -i :8001
   
   # Kill the process if needed
   sudo kill -9 <PID>
   ```

2. **Permission Denied**
   ```bash
   # Make sure the user has proper permissions
   sudo chown -R your-user:your-group /path/to/your/backend
   chmod +x start_services.sh
   ```

3. **Import Errors**
   ```bash
   # Make sure you're in the correct directory
   cd /path/to/your/backend
   
   # Check Python path
   python3 -c "import sys; print(sys.path)"
   ```

4. **API Key Issues**
   ```bash
   # Test API keys
   python3 -c "
   import os
   print('ZERODHA_API_KEY:', 'SET' if os.getenv('ZERODHA_API_KEY') else 'NOT SET')
   print('GEMINI_API_KEY:', 'SET' if os.getenv('GEMINI_API_KEY') else 'NOT SET')
   "
   ```

### Debug Mode

Run services in debug mode for more verbose output:

```bash
# Set log level to debug
export LOG_LEVEL=debug
./start_services.sh start
```

## Performance Optimization

### Resource Limits

For high-traffic deployments:

1. **Increase file descriptors**:
   ```bash
   # Add to /etc/security/limits.conf
   your-user soft nofile 65536
   your-user hard nofile 65536
   ```

2. **Optimize Python settings**:
   ```bash
   export PYTHONOPTIMIZE=1
   export PYTHONUNBUFFERED=1
   ```

3. **Use a production ASGI server**:
   ```bash
   # Install Gunicorn with Uvicorn workers
   pip install gunicorn[uvicorn]
   
   # Run with Gunicorn
   gunicorn data_service:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   gunicorn analysis_service:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001
   ```

## Backup and Recovery

### Database Backup

If using Supabase, enable automatic backups in the Supabase dashboard.

### Configuration Backup

```bash
# Backup configuration files
tar -czf backup-$(date +%Y%m%d).tar.gz .env requirements.txt *.py *.sh
```

### Recovery Procedure

1. Restore configuration files
2. Reinstall dependencies: `pip install -r requirements.txt`
3. Restart services: `./start_services.sh restart`

## Support

For issues and questions:
1. Check the logs: `./start_services.sh logs data`
2. Verify configuration: `./start_services.sh status`
3. Test endpoints: `curl http://localhost:8000/health`

## API Documentation

Once services are running, you can access the interactive API documentation:

- Data Service: http://localhost:8000/docs
- Analysis Service: http://localhost:8001/docs

