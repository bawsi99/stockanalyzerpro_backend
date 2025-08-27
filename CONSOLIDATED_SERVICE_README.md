# Consolidated Service - Stock Analyzer Pro

## Overview

The consolidated service (`consolidated_service.py`) is a single FastAPI application that combines both the Data Service and Analysis Service into one deployment. This solves the port management issues that occur when running multiple services separately in cloud environments.

## Architecture

```
Consolidated Service (Port 8000)
├── /data/*          → Data Service endpoints
├── /analysis/*      → Analysis Service endpoints  
├── /ws/stream       → WebSocket endpoint (real-time data)
├── /auth/*          → Authentication endpoints
└── /health          → Health check endpoint
```

## Key Features

- **Single Port Deployment**: All services run on one port (8000 by default)
- **Unified CORS**: Consistent CORS configuration across all services
- **WebSocket Support**: Real-time data streaming via WebSocket
- **Authentication**: JWT-based authentication for WebSocket connections
- **Health Monitoring**: Comprehensive health checks for all services

## Quick Start

### 1. Environment Setup

Create a `.env` file in the backend directory:

```env
# Service Configuration
PORT=8000
HOST=0.0.0.0
LOG_LEVEL=info

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,http://localhost:5173,https://your-frontend-domain.vercel.app

# Authentication
REQUIRE_AUTH=false
JWT_SECRET=your-secret-key-change-in-production

# Zerodha Configuration (Optional)
ZERODHA_API_KEY=your_api_key
ZERODHA_ACCESS_TOKEN=your_access_token

# Gemini AI Configuration (Optional)
GEMINI_API_KEY=your_gemini_api_key

# Supabase Configuration (Optional)
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
```

### 2. Start the Service

```bash
# Option 1: Direct start
python consolidated_service.py

# Option 2: Using the startup script (recommended)
python start_consolidated_service.py

# Option 3: Using uvicorn directly
uvicorn consolidated_service:app --host 0.0.0.0 --port 8000
```

### 3. Test the Service

```bash
# Run the test script
python test_consolidated_service.py
```

## API Endpoints

### Health Checks
- `GET /health` - Main service health
- `GET /data/health` - Data service health
- `GET /analysis/health` - Analysis service health

### Authentication
- `GET /auth/verify?token=<token>` - Verify JWT token
- `POST /auth/token?user_id=<user_id>` - Create JWT token

### Data Service (mounted at `/data`)
- `GET /data/stock/{symbol}/history` - Get historical data
- `GET /data/stock/{symbol}/info` - Get stock information
- `GET /data/market/status` - Get market status
- `GET /data/ws/health` - WebSocket health check

### Analysis Service (mounted at `/analysis`)
- `POST /analysis/analyze` - Perform stock analysis
- `GET /analysis/sector/list` - Get sector list
- `GET /analysis/stock/{symbol}/sector` - Get stock sector info

### WebSocket
- `WS /ws/stream` - Real-time data streaming

## WebSocket Usage

### Connection
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/stream?token=your_jwt_token');
```

### Authentication
The WebSocket requires authentication via:
1. **JWT Token** (recommended): Pass as query parameter `?token=<jwt_token>`
2. **API Key**: Pass in headers as `X-API-Key: <api_key>`
3. **No Auth**: Set `REQUIRE_AUTH=false` in environment

### Message Format
```json
{
  "action": "subscribe",
  "symbols": ["RELIANCE", "TCS"],
  "timeframes": ["1d", "1h"],
  "throttle_ms": 1000,
  "batch": false,
  "format": "json"
}
```

## Troubleshooting

### WebSocket Connection Issues

1. **403 Forbidden Error**
   - Check CORS configuration
   - Verify authentication token
   - Ensure `REQUIRE_AUTH` setting matches your setup

2. **Connection Refused**
   - Verify service is running on correct port
   - Check firewall settings
   - Ensure WebSocket endpoint is accessible

3. **Authentication Failures**
   - Verify JWT token is valid and not expired
   - Check `JWT_SECRET` environment variable
   - Ensure token is passed correctly in WebSocket URL

### CORS Issues

1. **Frontend can't connect**
   - Add your frontend domain to `CORS_ORIGINS`
   - Include both HTTP and HTTPS versions if needed
   - Restart service after changing CORS configuration

2. **WebSocket CORS errors**
   - WebSocket connections also respect CORS settings
   - Ensure origin is included in `CORS_ORIGINS`

### Service Import Issues

1. **Module not found errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python path includes backend directory
   - Verify all service files are present

2. **Import errors in consolidated service**
   - Check that `data_service.py` and `analysis_service.py` exist
   - Verify no syntax errors in imported services
   - Ensure all required dependencies are available

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Service port |
| `HOST` | 0.0.0.0 | Service host |
| `LOG_LEVEL` | info | Logging level |
| `CORS_ORIGINS` | localhost domains | Allowed CORS origins |
| `REQUIRE_AUTH` | false | Require WebSocket authentication |
| `JWT_SECRET` | your-secret-key | JWT signing secret |
| `ZERODHA_API_KEY` | - | Zerodha API key |
| `ZERODHA_ACCESS_TOKEN` | - | Zerodha access token |
| `GEMINI_API_KEY` | - | Gemini AI API key |
| `SUPABASE_URL` | - | Supabase URL |
| `SUPABASE_ANON_KEY` | - | Supabase anonymous key |

## Deployment

### Local Development
```bash
python start_consolidated_service.py
```

### Production (Render)
1. Set `PORT` environment variable to match Render's port
2. Set `HOST` to `0.0.0.0`
3. Configure `CORS_ORIGINS` with your frontend domain
4. Set `REQUIRE_AUTH=true` for production
5. Configure all API keys and secrets

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "start_consolidated_service.py"]
```

## Monitoring

### Health Checks
- Monitor `/health` endpoint for overall service health
- Check individual service health at `/data/health` and `/analysis/health`
- WebSocket health available at `/data/ws/health`

### Logs
- Service logs include connection information
- WebSocket authentication attempts are logged
- CORS rejections are logged with origin details

## Security Considerations

1. **JWT Secret**: Use a strong, unique JWT secret in production
2. **CORS**: Restrict CORS origins to only necessary domains
3. **Authentication**: Enable authentication for production deployments
4. **API Keys**: Store sensitive API keys securely
5. **HTTPS**: Use HTTPS in production for secure WebSocket connections

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Run the test script: `python test_consolidated_service.py`
3. Check service logs for error details
4. Verify environment configuration
