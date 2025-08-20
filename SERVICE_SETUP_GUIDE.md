# Service Setup Guide

## Overview

The Stock Analyzer Pro system consists of multiple services that need to be running for the frontend to work correctly.

## Service Architecture

### Main Services (Required for Frontend)

1. **Data Service (Port 8000)**
   - Handles stock data fetching
   - WebSocket connections for real-time data
   - Market status and token mapping
   - Authentication

2. **Analysis Service (Port 8001)**
   - AI-powered stock analysis
   - Technical indicators calculation
   - Pattern recognition
   - Chart generation
   - Sector analysis

### Testing Service (Optional)

3. **Service Endpoints (Port 8002)**
   - Individual component testing endpoints
   - NOT used by the frontend
   - For development and debugging only

## Correct Way to Run Services

### Option 1: Run All Services Together (Recommended)

```bash
cd backend
python start_all_services.py
```

This will start both the data service (8000) and analysis service (8001) together.

### Option 2: Run Services Individually

```bash
# Terminal 1 - Data Service
cd backend
python start_data_service.py

# Terminal 2 - Analysis Service  
cd backend
python start_analysis_service.py
```

### Option 3: Using Uvicorn Directly

```bash
# Terminal 1 - Data Service
cd backend
uvicorn data_service:app --host 0.0.0.0 --port 8000

# Terminal 2 - Analysis Service
cd backend
uvicorn analysis_service:app --host 0.0.0.0 --port 8001
```

## Frontend Configuration

The frontend is configured to connect to:
- **Data Service**: `http://localhost:8000`
- **Analysis Service**: `http://localhost:8001`

## Health Checks

Verify services are running:

```bash
# Data Service Health
curl http://localhost:8000/health

# Analysis Service Health  
curl http://localhost:8001/health

# Service Endpoints Health (optional)
curl http://localhost:8002/health
```

## Common Issues

### Issue: "Service endpoints at 8002 doesn't communicate with frontend"

**Problem**: You're running the wrong service. The service endpoints at port 8002 is for testing individual components, not for the main application.

**Solution**: Run the correct services:
- Data service at port 8000
- Analysis service at port 8001

### Issue: Frontend can't connect to services

**Check**:
1. Are both services running?
2. Are they on the correct ports?
3. Check browser console for CORS errors
4. Verify the frontend config points to the right URLs

### Issue: Services fail to start

**Common causes**:
1. Missing environment variables
2. Port already in use
3. Missing dependencies
4. Zerodha API credentials not configured

## Environment Variables

Make sure you have the required environment variables set:

```bash
# Zerodha API credentials
ZERODHA_API_KEY=your_api_key
ZERODHA_ACCESS_TOKEN=your_access_token

# Service ports (optional, defaults shown)
DATA_SERVICE_PORT=8000
ANALYSIS_SERVICE_PORT=8001

# CORS origins
CORS_ORIGINS=http://localhost:3000,http://localhost:8080,http://localhost:5173
```

## Development Workflow

1. Start the backend services first
2. Start the frontend development server
3. Verify health checks pass
4. Test frontend functionality

## Troubleshooting

### Check if ports are in use:
```bash
lsof -i :8000
lsof -i :8001
lsof -i :8002
```

### Kill processes on specific ports:
```bash
kill -9 $(lsof -t -i:8000)
kill -9 $(lsof -t -i:8001)
```

### View service logs:
Check the terminal output where you started each service for error messages.
