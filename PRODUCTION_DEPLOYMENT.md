# Stock Analyzer Pro - Production Deployment Guide

This document explains how to deploy the Stock Analyzer Pro services in a production environment using a unified API on a single port.

## Overview

For production deployment, we've created a unified service that combines both the Data Service and Analysis Service into a single FastAPI application running on one port. This approach has several benefits:

- **Simplified Deployment**: Only one service to manage instead of two
- **Unified API**: Single endpoint for clients to connect to
- **Reduced Infrastructure**: Only one port needs to be exposed
- **Improved Resource Utilization**: Worker processes handle all requests

## Prerequisites

- Python 3.8+
- Required packages installed (`pip install -r requirements.txt`)
- Environment variables properly configured

## Running the Production Service

### Command Line

```bash
python run_production_services.py [--port 8000] [--host 0.0.0.0] [--workers 4]
```

### Options

- `--port`: Port number for the service (default: 8000)
- `--host`: Host address (default: 0.0.0.0)
- `--workers`: Number of worker processes (default: 4)
- `--log-level`: Logging level (default: info)

### Environment Variables

- `SERVICE_PORT`: Port for the service
- `HOST`: Host address
- `WORKERS`: Number of worker processes
- `LOG_LEVEL`: Logging level

## API Structure

The unified API organizes endpoints under specific prefixes:

- `/` - Main service information
- `/health` - Service health check
- `/data/...` - All Data Service endpoints
- `/analysis/...` - All Analysis Service endpoints

### Key Endpoints

#### Core Service

- `GET /` - Root endpoint with service information
- `GET /health` - Health check for the entire service

#### Data Service

- `GET /data/` - Data service information
- `GET /data/health` - Data service health check
- `GET /data/market/status` - Current market status

#### Analysis Service

- `GET /analysis/` - Analysis service information
- `GET /analysis/health` - Analysis service health check
- `POST /analysis/analyze` - Analyze a stock (main analysis endpoint)

## Testing the Service

A test script is included to verify that all endpoints are working correctly:

```bash
python test_production_endpoints.py [--host localhost] [--port 8000]
```

This script will test all key endpoints and report their status.

## Production Deployment Recommendations

### Using Docker

For containerized deployment, create a Dockerfile:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "run_production_services.py", "--workers", "4"]
```

### Using a Process Manager

For systemd-based deployment:

1. Create a systemd service file:

```ini
[Unit]
Description=Stock Analyzer Pro Production Service
After=network.target

[Service]
User=stockanalyzer
WorkingDirectory=/path/to/stockanalyzer
ExecStart=/path/to/python /path/to/stockanalyzer/backend/run_production_services.py
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

2. Enable and start the service:

```bash
sudo systemctl enable stockanalyzer
sudo systemctl start stockanalyzer
```

### Using NGINX as a Reverse Proxy

For a production setup with NGINX:

```nginx
server {
    listen 80;
    server_name stockanalyzer.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Scaling Considerations

- Increase the number of workers to handle more concurrent requests
- Consider using a load balancer for horizontal scaling
- Use a Redis cache for improved performance
- Monitor memory usage and adjust worker count accordingly

## Monitoring and Maintenance

- Health check endpoint at `/health` can be used for monitoring
- Use the `/data/health` and `/analysis/health` endpoints for service-specific health checks
- Log to a centralized logging system for easier troubleshooting
- Set up alerts based on endpoint response times and error rates
