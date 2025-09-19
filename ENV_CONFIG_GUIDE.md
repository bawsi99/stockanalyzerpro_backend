# Environment Configuration Guide for Distributed Services Architecture

## Overview
The distributed services architecture uses separate services for different functionalities. Each service runs on its own port and requires specific environment variables defined in `/backend/config/.env`.

### Service Architecture
The distributed backend consists of:
- **Data Service**: Runs on port 8001 (`http://localhost:8001`) - Handles data fetching, WebSocket streaming
- **Analysis Service**: Runs on port 8002 (`http://localhost:8002`) - Handles analysis, AI processing, charts
- **Database Service**: Runs on port 8003 (`http://localhost:8003`) - Handles database operations

Services communicate with each other through HTTP requests using environment-configured URLs.

## Required Environment Variables

### 🔐 **Authentication & Security**
```bash
# JWT Authentication
JWT_SECRET=your-secure-jwt-secret-key
REQUIRE_AUTH=false  # Set to 'true' to enable authentication

# API Keys for additional security
API_KEYS=test-api-key-1,test-api-key-2  # Comma-separated API keys
```

### 🌐 **Server Configuration**
```bash
# Server binding
SERVICE_HOST=0.0.0.0  # Host address (0.0.0.0 for all interfaces)

# Service ports (distributed architecture)
DATA_PORT=8001        # Data service port (includes WebSocket)
ANALYSIS_PORT=8002    # Analysis service port
DATABASE_PORT=8003    # Database service port

# CORS Configuration (comma-separated origins)
CORS_ORIGINS=http://localhost:3000,http://localhost:5173,https://your-frontend-domain.com
```

### 📊 **Zerodha API Configuration**
```bash
# Zerodha API credentials
ZERODHA_API_KEY=your-zerodha-api-key
ZERODHA_API_SECRET=your-zerodha-api-secret
ZERODHA_ACCESS_TOKEN=your-zerodha-access-token
ZERODHA_REQUEST_TOKEN=your-zerodha-request-token
```

### 🗄️ **Database Configuration (Supabase)**
```bash
# Supabase database connection
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-supabase-service-role-key
```

### 🤖 **AI Service Configuration**
```bash
# Google Gemini API for AI analysis
GEMINI_API_KEY=your-gemini-api-key
```

### 📈 **Market Data Configuration**
```bash
# Market hours and data processing
ENABLE_POST_MARKET=true        # Enable post-market data processing
ENABLE_EXTENDED_HOURS=true     # Enable extended hours trading data
MARKET_HOURS_TEST_MODE=true    # Test mode for development
FORCE_MARKET_OPEN=true         # Force market open status for testing
```

### 🗃️ **Redis Configuration**
```bash
# Redis connection
REDIS_URL=redis://username:password@host:port

# Redis cache settings
REDIS_CACHE_ENABLE_COMPRESSION=true
REDIS_CACHE_ENABLE_LOCAL_FALLBACK=true
REDIS_CACHE_LOCAL_SIZE=1000
REDIS_CACHE_CLEANUP_INTERVAL_MINUTES=60

# Cache TTL settings (in seconds)
CACHE_TTL_TECHNICAL_INDICATORS=120
CACHE_TTL_AI_ANALYSIS=900
CACHE_TTL_DEFAULT=300
```

### 📊 **Chart & Analytics Configuration**
```bash
# Chart storage and cleanup
CHART_MAX_AGE_HOURS=6
CHART_MAX_SIZE_MB=200
CHART_CLEANUP_INTERVAL_MINUTES=1
CHART_ENABLE_CLEANUP=true
```

### 🔧 **Development & Debug Configuration**
```bash
# Environment settings
ENVIRONMENT=development  # or 'production'
DEBUG=true              # Enable debug mode
LOG_LEVEL=INFO          # Logging level

# Service Communication (distributed architecture)
DATABASE_SERVICE_URL=http://localhost:8003  # URL for analysis service to reach database service
DATA_SERVICE_URL=http://localhost:8001      # URL for inter-service data requests

# Scheduled tasks
ENABLE_SCHEDULED_CALIBRATION=0
```

## Environment File Location

The environment file must be located at:
```
/backend/config/.env
```

## Usage Examples

### Starting the Distributed Services
```bash
# Start each service separately in different terminals

# Terminal 1 - Data Service (port 8001)
cd /path/to/backend/services
python data_service.py

# Terminal 2 - Analysis Service (port 8002) 
cd /path/to/backend/services
python analysis_service.py

# Terminal 3 - Database Service (port 8003)
cd /path/to/backend/services
python database_service.py
```

### Testing Configuration
```bash
# Test environment variable loading
python -c "
import dotenv
dotenv.load_dotenv('../config/.env')
import os
print('PORT:', os.getenv('PORT'))
print('HOST:', os.getenv('HOST'))
print('CORS_ORIGINS:', os.getenv('CORS_ORIGINS'))
"
```

## Security Considerations

1. **Never commit `.env` files to version control**
2. **Use strong, unique JWT secrets in production**
3. **Limit CORS origins to trusted domains only**
4. **Use environment-specific configurations for different deployment stages**
5. **Regularly rotate API keys and tokens**

## Production Deployment

For production deployments:

1. Set `REQUIRE_AUTH=true`
2. Use strong JWT secrets
3. Restrict CORS origins to production domains only
4. Set `ENVIRONMENT=production`
5. Set `DEBUG=false`
6. Use secure Redis connections (TLS)
7. Monitor and rotate access tokens regularly

## Troubleshooting

### Common Issues

1. **Service won't start**: Check that all required environment variables are set
2. **CORS errors**: Verify CORS_ORIGINS includes your frontend URL
3. **Database connection failures**: Verify Supabase credentials and URL
4. **Authentication failures**: Check JWT_SECRET and token validity
5. **Redis connection issues**: Verify Redis URL and credentials

### Debug Commands

```bash
# Check if environment variables are loaded
python -c "import os; import dotenv; dotenv.load_dotenv('../config/.env'); print('PORT:', os.getenv('PORT'))"

# Test service compilation
python -m py_compile data_service.py
python -m py_compile analysis_service.py
python -m py_compile database_service.py

# Check service health after startup
curl http://localhost:8001/health  # Data Service
curl http://localhost:8002/health  # Analysis Service
curl http://localhost:8003/health  # Database Service
```

## API Endpoints

Once configured and running, the distributed services provide:

**Data Service (Port 8001):**
- **Health Check**: `GET /health`
- **Stock Data**: `GET /stock/{symbol}/history`
- **WebSocket Stream**: `WS /ws/stream`
- **Market Status**: `GET /market/status`
- **Authentication**: `POST /auth/token`

**Analysis Service (Port 8002):**
- **Health Check**: `GET /health`
- **Stock Analysis**: `POST /analyze`
- **Enhanced Analysis**: `POST /analyze/enhanced`
- **Sector Analysis**: `GET /sector/list`

**Database Service (Port 8003):**
- **Health Check**: `GET /health`
- **Store Analysis**: `POST /analyses/store`
- **User Analyses**: `GET /analyses/user/{user_id}`

## Backup and Recovery

- Keep backup copies of your `.env` file securely
- Document any custom environment variable changes
- Use version control for `.env.example` template files (without actual secrets)