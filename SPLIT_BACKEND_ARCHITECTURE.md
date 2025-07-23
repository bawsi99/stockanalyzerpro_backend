# Split Backend Architecture

This document explains the new split backend architecture where the backend is divided into two separate services for better scalability, maintainability, and resource management.

## Architecture Overview

The backend is now split into two independent services:

### 1. Data Service (Port 8000)
**Purpose**: Handles all data fetching, WebSocket connections, and real-time data streaming.

**Responsibilities**:
- Historical data retrieval from Zerodha API
- Real-time data streaming via WebSocket
- Market data caching and optimization
- Token/symbol mapping
- Market status monitoring
- Alert management
- WebSocket connection management

**Key Endpoints**:
- `GET /health` - Service health check
- `GET /stock/{symbol}/history` - Historical OHLCV data
- `POST /data/optimized` - Optimized data fetching
- `GET /stock/{symbol}/info` - Stock information
- `GET /market/status` - Market status
- `GET /mapping/token-to-symbol` - Token to symbol mapping
- `GET /mapping/symbol-to-token` - Symbol to token mapping
- `WebSocket /ws/stream` - Real-time data streaming

### 2. Analysis Service (Port 8001)
**Purpose**: Handles all analysis, AI processing, and chart generation.

**Responsibilities**:
- Stock analysis and AI processing
- Technical indicator calculations
- Chart generation and visualization
- Sector analysis and benchmarking
- Pattern recognition
- Enhanced analysis with code execution
- Analysis result storage

**Key Endpoints**:
- `GET /health` - Service health check
- `POST /analyze` - Comprehensive stock analysis
- `POST /analyze/enhanced` - Enhanced analysis with code execution
- `GET /stock/{symbol}/indicators` - Technical indicators
- `GET /patterns/{symbol}` - Pattern recognition
- `GET /charts/{symbol}` - Chart generation
- `GET /sector/list` - Sector information
- `POST /sector/benchmark` - Sector benchmarking
- `GET /sector/{sector_name}/stocks` - Sector stocks
- `GET /sector/{sector_name}/performance` - Sector performance

## Running the Services

### Prerequisites
1. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```bash
# Zerodha API credentials
ZERODHA_API_KEY=your_api_key
ZERODHA_ACCESS_TOKEN=your_access_token

# JWT Secret (for authentication)
JWT_SECRET=your_jwt_secret

# API Keys (comma-separated)
API_KEYS=key1,key2,key3

# Supabase credentials (for analysis storage)
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Google Gemini API (for AI analysis)
GOOGLE_API_KEY=your_google_api_key
```

### Starting the Services

#### Option 1: Using Startup Scripts (Recommended)

**Terminal 1 - Data Service:**
```bash
cd backend
python start_data_service.py
```

**Terminal 2 - Analysis Service:**
```bash
cd backend
python start_analysis_service.py
```

#### Option 2: Using Uvicorn Directly

**Terminal 1 - Data Service:**
```bash
cd backend
uvicorn data_service:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Analysis Service:**
```bash
cd backend
uvicorn analysis_service:app --host 0.0.0.0 --port 8001 --reload
```

#### Option 3: Using Environment Variables

**Terminal 1 - Data Service:**
```bash
cd backend
DATA_SERVICE_PORT=8000 python start_data_service.py
```

**Terminal 2 - Analysis Service:**
```bash
cd backend
ANALYSIS_SERVICE_PORT=8001 python start_analysis_service.py
```

## Service Communication

### Frontend Integration

The frontend should be updated to communicate with both services:

**Data Service (Port 8000):**
```javascript
// Real-time data
const ws = new WebSocket('ws://localhost:8000/ws/stream');

// Historical data
const response = await fetch('http://localhost:8000/stock/RELIANCE/history?interval=1day');
```

**Analysis Service (Port 8001):**
```javascript
// Stock analysis
const response = await fetch('http://localhost:8001/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    stock: 'RELIANCE',
    exchange: 'NSE',
    period: 365,
    interval: 'day'
  })
});

// Technical indicators
const indicators = await fetch('http://localhost:8001/stock/RELIANCE/indicators?indicators=rsi,macd,sma');
```

### Service-to-Service Communication

If services need to communicate with each other, they can use HTTP requests:

```python
# From analysis service to data service
import httpx

async def get_stock_data(symbol: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"http://localhost:8000/stock/{symbol}/history")
        return response.json()
```

## Configuration

### Environment Variables

**Data Service:**
- `DATA_SERVICE_HOST` - Host address (default: 0.0.0.0)
- `DATA_SERVICE_PORT` - Port number (default: 8000)
- `DATA_SERVICE_RELOAD` - Enable auto-reload (default: false)

**Analysis Service:**
- `ANALYSIS_SERVICE_HOST` - Host address (default: 0.0.0.0)
- `ANALYSIS_SERVICE_PORT` - Port number (default: 8001)
- `ANALYSIS_SERVICE_RELOAD` - Enable auto-reload (default: false)

### Service Dependencies

**Data Service Dependencies:**
- `zerodha_client.py` - Zerodha API client
- `zerodha_ws_client.py` - WebSocket client
- `enhanced_data_service.py` - Enhanced data service
- `market_hours_manager.py` - Market hours management

**Analysis Service Dependencies:**
- `agent_capabilities.py` - Analysis orchestrator
- `sector_benchmarking.py` - Sector analysis
- `sector_classifier.py` - Sector classification
- `patterns/recognition.py` - Pattern recognition
- `technical_indicators.py` - Technical indicators
- `analysis_storage.py` - Analysis storage

## Monitoring and Health Checks

### Health Check Endpoints

**Data Service:**
```bash
curl http://localhost:8000/health
```

**Analysis Service:**
```bash
curl http://localhost:8001/health
```

### WebSocket Health

**Data Service:**
```bash
curl http://localhost:8000/ws/health
curl http://localhost:8000/ws/test
curl http://localhost:8000/ws/connections
```

### Service Status

Both services return health status with service identification:
```json
{
  "status": "healthy",
  "service": "data_service|analysis_service",
  "timestamp": "2024-01-01T00:00:00"
}
```

## Benefits of Split Architecture

### 1. **Scalability**
- Each service can be scaled independently
- Data service can handle high-frequency real-time data
- Analysis service can be scaled for heavy computational tasks

### 2. **Resource Management**
- Data service optimized for I/O operations
- Analysis service optimized for CPU-intensive tasks
- Better resource utilization

### 3. **Maintainability**
- Clear separation of concerns
- Easier to debug and maintain
- Independent deployment and updates

### 4. **Reliability**
- Service isolation prevents cascading failures
- Independent health monitoring
- Better error handling and recovery

### 5. **Development**
- Teams can work on services independently
- Different technology stacks if needed
- Easier testing and development

## Migration from Monolithic Backend

### 1. **Update Frontend Configuration**
Update your frontend to use the new service endpoints:

```javascript
// Old monolithic approach
const API_BASE = 'http://localhost:8000';

// New split approach
const DATA_SERVICE_BASE = 'http://localhost:8000';
const ANALYSIS_SERVICE_BASE = 'http://localhost:8001';
```

### 2. **Update API Calls**
Split API calls based on service responsibility:

```javascript
// Data-related calls go to data service
const stockData = await fetch(`${DATA_SERVICE_BASE}/stock/RELIANCE/history`);

// Analysis-related calls go to analysis service
const analysis = await fetch(`${ANALYSIS_SERVICE_BASE}/analyze`, {
  method: 'POST',
  body: JSON.stringify(request)
});
```

### 3. **WebSocket Connections**
WebSocket connections remain with the data service:

```javascript
const ws = new WebSocket(`ws://localhost:8000/ws/stream`);
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using the port
   lsof -i :8000
   lsof -i :8001
   
   # Kill the process
   kill -9 <PID>
   ```

2. **Import Errors**
   ```bash
   # Make sure you're in the backend directory
   cd backend
   
   # Check Python path
   python -c "import sys; print(sys.path)"
   ```

3. **Environment Variables**
   ```bash
   # Check if .env file is loaded
   python -c "import os; print(os.getenv('ZERODHA_API_KEY'))"
   ```

4. **Service Communication**
   ```bash
   # Test service connectivity
   curl http://localhost:8000/health
   curl http://localhost:8001/health
   ```

### Logs and Debugging

**Data Service Logs:**
- WebSocket connections and disconnections
- Data fetching operations
- Market status updates

**Analysis Service Logs:**
- Analysis processing steps
- AI model calls
- Chart generation progress

### Performance Monitoring

Monitor service performance using:
- Health check endpoints
- Response times
- Memory usage
- CPU utilization

## Future Enhancements

### 1. **Load Balancing**
- Add load balancers for high availability
- Implement service discovery

### 2. **Caching Layer**
- Redis for data caching
- CDN for static assets

### 3. **Message Queue**
- RabbitMQ/Kafka for async processing
- Event-driven architecture

### 4. **Containerization**
- Docker containers for each service
- Kubernetes orchestration

### 5. **Monitoring**
- Prometheus metrics
- Grafana dashboards
- Distributed tracing

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review service logs
3. Test individual service endpoints
4. Verify environment configuration

The split architecture provides a solid foundation for scaling and maintaining the trading platform while keeping services focused and efficient. 