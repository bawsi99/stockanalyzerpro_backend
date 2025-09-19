# Environment Configuration Guide for Unified Backend Service

## Overview
The unified backend service (`data_database_service.py`) combines both data and database services into a single service. It requires a comprehensive set of environment variables defined in `/backend/config/.env`.

### Service Architecture
The unified backend consists of:
- **Main Service**: Runs on port 8000 (`http://localhost:8000`)
- **Database Endpoints**: Mounted at `/database` (`http://localhost:8000/database/*`)
- **Data Endpoints**: Mounted at `/data` (`http://localhost:8000/data/*`)
- **WebSocket Streaming**: Available at `/data/ws/stream`

The analysis service communicates with the database endpoints through the `DATABASE_SERVICE_URL` environment variable.

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
HOST=0.0.0.0          # Host address (0.0.0.0 for all interfaces)
PORT=8000             # Main service port
DATA_PORT=8001        # Internal data service port
SERVICE_HOST=0.0.0.0  # Service host binding

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

# Service Communication
DATABASE_SERVICE_URL=http://localhost:8000/database  # URL for analysis service to reach database endpoints
DATABASE_SERVICE_PING_CRON=*/5 * * * *               # Cron schedule for database health checks

# Scheduled tasks
ENABLE_SCHEDULED_CALIBRATION=0
```

## Environment File Location

The environment file must be located at:
```
/backend/config/.env
```

## Usage Examples

### Starting the Unified Service
```bash
cd /path/to/backend/services
python data_database_service.py
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
python -m py_compile data_database_service.py

# Check service health after startup
curl http://localhost:8000/health
```

## API Endpoints

Once configured and running, the service provides:

- **Health Check**: `GET /health`
- **Stock Data**: `GET /data/stock/{symbol}/history`
- **Database Operations**: `POST /database/analyses/store`
- **WebSocket Stream**: `WS /data/ws/stream`
- **Market Status**: `GET /data/market/status`

## Backup and Recovery

- Keep backup copies of your `.env` file securely
- Document any custom environment variable changes
- Use version control for `.env.example` template files (without actual secrets)