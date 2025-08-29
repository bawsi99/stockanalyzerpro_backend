# 🚀 Multi-Instance Deployment Guide

This guide will help you deploy your Stock Analyzer Pro services across multiple Render instances for optimal cost and performance.

## 🏗️ **Architecture Overview**

```
┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Frontend      │
│   (Vercel)      │    │   (Vercel)      │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│ Data + WebSocket│    │  Analysis       │
│ Service         │    │  Service        │
│ (Free Tier)     │    │  (Standard)     │
│ 512 MB RAM      │    │  2 GB RAM       │
│ $0/month        │    │  $25/month      │
└─────────────────┘    └─────────────────┘
```

## 📋 **Prerequisites**

1. **Render Account**: [Sign up at render.com](https://render.com)
2. **GitHub Repository**: Your code should be in a GitHub repo
3. **Environment Variables**: API keys and configuration ready
4. **Requirements.txt**: Python dependencies file

## 🔧 **Step 1: Prepare Your Repository**

### **File Structure**
```
backend/
├── data_service.py          # Combined data + WebSocket service
├── analysis_service.py      # Analysis service
├── requirements.txt         # Python dependencies
├── env.data_websocket      # Environment config for data service
├── env.analysis            # Environment config for analysis service
├── render.data_websocket.yaml  # Render config for data service
├── render.analysis.yaml    # Render config for analysis service
└── services.config.js      # Frontend service mapping
```

### **Requirements.txt**
Ensure your `requirements.txt` includes all necessary packages:
```txt
fastapi
uvicorn
pandas
numpy
psutil
aiohttp
websockets
python-dotenv
# ... other dependencies
```

## 🚀 **Step 2: Deploy Data + WebSocket Service**

### **2.1 Create New Web Service on Render**

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure the service:

**Basic Settings:**
- **Name**: `stock-analyzer-data-websocket`
- **Environment**: `Python 3`
- **Region**: `Oregon` (or closest to your users)
- **Branch**: `main` (or your default branch)
- **Root Directory**: `backend` (if your code is in a backend folder)

**Build & Deploy:**
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python data_service.py`

### **2.2 Set Environment Variables**

In the Render dashboard, go to **Environment** tab and add:

```bash
# Service Configuration
SERVICE_NAME=data_websocket_service
SERVICE_PORT=8001
SERVICE_HOST=0.0.0.0

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,http://localhost:8080,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:8080,http://127.0.0.1:5173,https://stock-analyzer-pro.vercel.app,https://stock-analyzer-pro-git-prototype-aaryan-manawats-projects.vercel.app,https://stock-analyzer-cl9o3tivx-aaryan-manawats-projects.vercel.app,https://stockanalyzer-pro.vercel.app

# Zerodha API (Required)
ZERODHA_API_KEY=your_actual_api_key
ZERODHA_ACCESS_TOKEN=your_actual_access_token

# Authentication
JWT_SECRET=your_secure_jwt_secret
API_KEYS=your_api_key_1,your_api_key_2
REQUIRE_AUTH=false

# Memory Management (Important for 512 MB)
MAX_CONCURRENT_CONNECTIONS=50
MAX_WEBSOCKET_CONNECTIONS=30
DATA_CACHE_SIZE_MB=100
WEBSOCKET_HEARTBEAT_INTERVAL=30

# Logging
LOG_LEVEL=INFO
ENABLE_DEBUG=false
```

### **2.3 Deploy and Test**

1. Click **"Create Web Service"**
2. Wait for build and deployment (5-10 minutes)
3. Test the service:
   - Health check: `https://your-service.onrender.com/health`
   - Stock data: `https://your-service.onrender.com/stock/RELIANCE/history`
   - WebSocket: `wss://your-service.onrender.com/ws/stream`

**Note the URL**: `https://your-data-websocket-service.onrender.com`

## 🔍 **Step 3: Deploy Analysis Service**

### **3.1 Create Second Web Service**

1. Go back to Render Dashboard
2. Click **"New +"** → **"Web Service"**
3. Connect the same GitHub repository
4. Configure the service:

**Basic Settings:**
- **Name**: `stock-analyzer-analysis`
- **Environment**: `Python 3`
- **Region**: `Oregon` (same as data service)
- **Branch**: `main`
- **Root Directory**: `backend`

**Build & Deploy:**
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python analysis_service.py`

**Important**: Change the plan to **"Standard"** (2 GB RAM) for this service.

### **3.2 Set Environment Variables**

```bash
# Service Configuration
SERVICE_NAME=analysis_service
SERVICE_PORT=8002
SERVICE_HOST=0.0.0.0

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,http://localhost:8080,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:8080,http://127.0.0.1:5173,https://stock-analyzer-pro.vercel.app,https://stock-analyzer-pro-git-prototype-aaryan-manawats-projects.vercel.app,https://stock-analyzer-cl9o3tivx-aaryan-manawats-projects.vercel.app,https://stockanalyzer-pro.vercel.app

# Zerodha API
ZERODHA_API_KEY=your_actual_api_key
ZERODHA_ACCESS_TOKEN=your_actual_access_token

# Gemini AI (Required for analysis)
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_GEMINI_API_KEY=your_gemini_api_key

# Database (if using Supabase)
DATABASE_URL=your_supabase_url

# Redis (if using for caching)
REDIS_URL=your_redis_url

# Service Communication
DATA_SERVICE_URL=https://your-data-websocket-service.onrender.com
FRONTEND_URL=https://stock-analyzer-pro.vercel.app

# ML Configuration
ENABLE_ML_TRAINING=true
ML_MODEL_CACHE_SIZE_MB=500
PATTERN_RECOGNITION_ENABLED=true

# Chart Configuration
CHART_STORAGE_TYPE=redis
CHART_MAX_AGE_HOURS=24
CHART_MAX_TOTAL_SIZE_MB=200

# Memory Management (Important for 2 GB)
MAX_CONCURRENT_ANALYSES=20
MAX_CHART_GENERATION_JOBS=10
ANALYSIS_TIMEOUT_SECONDS=300
ENABLE_BACKGROUND_CALIBRATION=true

# Logging
LOG_LEVEL=INFO
ENABLE_DEBUG=false
ENABLE_PERFORMANCE_MONITORING=true
```

### **3.3 Deploy and Test**

1. Click **"Create Web Service"**
2. Wait for build and deployment (10-15 minutes for Standard plan)
3. Test the service:
   - Health check: `https://your-analysis-service.onrender.com/health`
   - Sector list: `https://your-analysis-service.onrender.com/sector/list`
   - Analysis: `POST https://your-analysis-service.onrender.com/analyze`

**Note the URL**: `https://your-analysis-service.onrender.com`

## 🔗 **Step 4: Configure Service Communication**

### **4.1 Update Analysis Service URL**

In your **Analysis Service** environment variables, update:
```bash
DATA_SERVICE_URL=https://your-actual-data-websocket-service.onrender.com
```

### **4.2 Test Cross-Service Communication**

The analysis service should now be able to call the data service for stock data.

## 🌐 **Step 5: Update Frontend Configuration**

### **5.1 Environment Variables**

In your Vercel frontend, add these environment variables:

```bash
REACT_APP_DATA_WEBSOCKET_URL=https://your-data-websocket-service.onrender.com
REACT_APP_ANALYSIS_URL=https://your-analysis-service.onrender.com
```

### **5.2 Update API Calls**

Use the `services.config.js` file to route requests to the correct service:

```javascript
import { buildServiceUrl } from './services.config.js';

// For stock data
const stockDataUrl = buildServiceUrl('dataWebsocket', 'stockHistory', { symbol: 'RELIANCE' });

// For analysis
const analysisUrl = buildServiceUrl('analysis', 'analyze');
```

## 📊 **Step 6: Monitor and Optimize**

### **6.1 Health Checks**

Both services have health endpoints:
- Data Service: `/health`
- Analysis Service: `/health`

### **6.2 Memory Monitoring**

- **Data Service**: Monitor WebSocket connections and data cache usage
- **Analysis Service**: Monitor ML model memory and chart generation

### **6.3 Performance Metrics**

- Response times for each service
- Memory usage patterns
- Error rates and logs

## 💰 **Cost Breakdown**

- **Data + WebSocket Service**: Free tier ($0/month)
- **Analysis Service**: Standard tier ($25/month)
- **Total**: $25/month (vs $50-100/month for single large instance)

## 🚨 **Troubleshooting**

### **Common Issues**

1. **Build Failures**: Check requirements.txt and Python version
2. **Environment Variables**: Ensure all required vars are set
3. **CORS Issues**: Verify CORS_ORIGINS includes your frontend URL
4. **Memory Issues**: Monitor logs for out-of-memory errors

### **Debug Commands**

```bash
# Check service logs in Render dashboard
# Test endpoints with curl
curl https://your-service.onrender.com/health

# Check WebSocket connection
wscat -c wss://your-service.onrender.com/ws/stream
```

## ✅ **Deployment Checklist**

- [ ] Data + WebSocket service deployed (Free tier)
- [ ] Analysis service deployed (Standard tier)
- [ ] Environment variables configured
- [ ] Cross-service communication working
- [ ] Frontend environment variables updated
- [ ] All endpoints tested
- [ ] Health checks passing
- [ ] Memory usage within limits

## 🎯 **Next Steps**

1. **Monitor Performance**: Watch memory usage and response times
2. **Scale if Needed**: Upgrade plans if you hit limits
3. **Add Caching**: Implement Redis for better performance
4. **Load Balancing**: Add more instances if traffic increases

## 📞 **Support**

If you encounter issues:
1. Check Render service logs
2. Verify environment variables
3. Test endpoints individually
4. Check CORS configuration
5. Monitor memory usage

---

**🎉 Congratulations!** You now have a cost-effective, scalable multi-instance architecture running on Render!
