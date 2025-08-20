# Service Endpoints - Comprehensive Testing Guide

This guide provides comprehensive endpoints for testing each service component in the Stock Analysis System individually. This allows you to understand the data flow, debug issues, and test specific functionality without running the full analysis pipeline.

## 🚀 Quick Start

### 1. Start All Services
```bash
cd backend
python run_services.py
```

### 2. Start Individual Services
```bash
# Data Service (Port 8000)
python start_data_service.py

# Analysis Service (Port 8001)
python start_analysis_service.py

# WebSocket Stream Service (Port 8081)
python start_websocket_service.py

# Service Endpoints (Port 8002) - Component Testing
python start_service_endpoints.py
```

### 3. Test All Endpoints
```bash
# Test all services
python test_service_endpoints.py all

# Test specific service
python test_service_endpoints.py data
python test_service_endpoints.py analysis
python test_service_endpoints.py websocket
python test_service_endpoints.py service_endpoints
```

## 📊 Service Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Service  │    │ Analysis Service│    │ WebSocket Stream│
│   (Port 8000)   │    │   (Port 8001)   │    │   (Port 8081)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │Service Endpoints│
                    │   (Port 8002)   │
                    └─────────────────┘
```

## 🔧 Service Endpoints (Port 8002)

The Service Endpoints provide individual testing for each component. This is the main focus for component-level testing.

### Health & Status
- `GET /health` - Comprehensive health check for all components
- `GET /status` - Detailed status and available endpoints

### Data Service Endpoints
- `POST /data/fetch` - Fetch historical stock data
- `GET /data/stock-info/{symbol}` - Get basic stock information
- `GET /data/market-status` - Get current market status
- `GET /data/token-mapping` - Get token to symbol mapping

### Technical Analysis Endpoints
- `POST /technical/indicators` - Calculate technical indicators
- `POST /technical/market-metrics` - Get basic market metrics
- `POST /technical/enhanced-metrics` - Get enhanced market metrics with sector context

### Pattern Recognition Endpoints
- `POST /patterns/detect` - Detect all patterns for a stock
- `POST /patterns/candlestick` - Detect candlestick patterns
- `POST /patterns/chart` - Detect chart patterns
- `POST /patterns/volume` - Detect volume patterns

### Sector Analysis Endpoints
- `POST /sectors/info` - Get sector information for a stock
- `POST /sectors/benchmarking` - Get sector benchmarking
- `GET /sectors/rotation` - Get sector rotation analysis
- `GET /sectors/correlation` - Get sector correlation matrix
- `GET /sectors/performance` - Get sector performance metrics

### Analysis Endpoints
- `POST /analysis/full` - Perform full analysis for a stock
- `POST /analysis/enhanced` - Perform enhanced analysis with code execution
- `POST /analysis/mtf` - Perform multi-timeframe analysis
- `POST /analysis/risk` - Perform risk assessment
- `POST /analysis/backtest` - Perform backtesting

### Machine Learning Endpoints
- `POST /ml/train` - Train machine learning model
- `POST /ml/predict` - Make prediction using trained model
- `GET /ml/model-info` - Get information about trained models

### Chart Generation Endpoints
- `POST /charts/generate` - Generate charts for a stock
- `POST /charts/data` - Get chart data for visualization

## 📋 Detailed Endpoint Examples

### 1. Data Service Testing

#### Fetch Stock Data
```bash
curl -X POST "http://localhost:8002/data/fetch" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "RELIANCE",
    "exchange": "NSE",
    "period": 365,
    "interval": "day"
  }'
```

#### Get Stock Info
```bash
curl "http://localhost:8002/data/stock-info/RELIANCE"
```

#### Get Market Status
```bash
curl "http://localhost:8002/data/market-status"
```

### 2. Technical Analysis Testing

#### Calculate Indicators
```bash
curl -X POST "http://localhost:8002/technical/indicators" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "RELIANCE",
    "exchange": "NSE",
    "period": 365,
    "interval": "day",
    "indicators": "rsi,macd,sma,ema"
  }'
```

#### Get Market Metrics
```bash
curl -X POST "http://localhost:8002/technical/market-metrics" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "RELIANCE",
    "exchange": "NSE",
    "period": 365,
    "interval": "day"
  }'
```

### 3. Pattern Recognition Testing

#### Detect All Patterns
```bash
curl -X POST "http://localhost:8002/patterns/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "RELIANCE",
    "exchange": "NSE",
    "period": 365,
    "interval": "day",
    "pattern_types": "all"
  }'
```

#### Detect Candlestick Patterns
```bash
curl -X POST "http://localhost:8002/patterns/candlestick" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "RELIANCE",
    "exchange": "NSE",
    "period": 365,
    "interval": "day"
  }'
```

### 4. Sector Analysis Testing

#### Get Sector Info
```bash
curl -X POST "http://localhost:8002/sectors/info" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "RELIANCE"
  }'
```

#### Get Sector Benchmarking
```bash
curl -X POST "http://localhost:8002/sectors/benchmarking" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "RELIANCE",
    "exchange": "NSE",
    "period": 365,
    "interval": "day"
  }'
```

#### Get Sector Rotation
```bash
curl "http://localhost:8002/sectors/rotation?period=1M"
```

### 5. Analysis Testing

#### Full Analysis
```bash
curl -X POST "http://localhost:8002/analysis/full" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "RELIANCE",
    "exchange": "NSE",
    "period": 365,
    "interval": "day"
  }'
```

#### Enhanced Analysis
```bash
curl -X POST "http://localhost:8002/analysis/enhanced" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "RELIANCE",
    "exchange": "NSE",
    "period": 365,
    "interval": "day"
  }'
```

#### Risk Assessment
```bash
curl -X POST "http://localhost:8002/analysis/risk" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "RELIANCE",
    "exchange": "NSE",
    "period": 365,
    "interval": "day"
  }'
```

### 6. Machine Learning Testing

#### Train Model
```bash
curl -X POST "http://localhost:8002/ml/train"
```

#### Make Prediction
```bash
curl -X POST "http://localhost:8002/ml/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "RELIANCE",
    "features": {
      "rsi": 65.5,
      "macd": 0.25,
      "sma_20": 2450.0,
      "volume": 1000000
    },
    "model_type": "global"
  }'
```

### 7. Chart Generation Testing

#### Generate Charts
```bash
curl -X POST "http://localhost:8002/charts/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "RELIANCE",
    "exchange": "NSE",
    "period": 365,
    "interval": "day"
  }'
```

#### Get Chart Data
```bash
curl -X POST "http://localhost:8002/charts/data" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "RELIANCE",
    "exchange": "NSE",
    "period": 365,
    "interval": "day"
  }'
```

## 🧪 Automated Testing

### Run All Tests
```bash
python test_service_endpoints.py all
```

### Run Specific Service Tests
```bash
# Test data service only
python test_service_endpoints.py data

# Test analysis service only
python test_service_endpoints.py analysis

# Test WebSocket service only
python test_service_endpoints.py websocket

# Test service endpoints only
python test_service_endpoints.py service_endpoints
```

### Test Results
The test script will:
1. Test each endpoint individually
2. Show success/failure status
3. Display response times
4. Generate a summary report
5. Save detailed results to a JSON file

## 🔍 Debugging Tips

### 1. Check Service Health
```bash
# Check all services
curl "http://localhost:8000/health"  # Data Service
curl "http://localhost:8001/health"  # Analysis Service
curl "http://localhost:8081/health"  # WebSocket Service
curl "http://localhost:8002/health"  # Service Endpoints
```

### 2. Check Service Status
```bash
curl "http://localhost:8002/status"
```

### 3. Monitor Logs
Each service generates logs that can help with debugging:
- Data Service: Check console output
- Analysis Service: Check console output
- WebSocket Service: Check console output
- Service Endpoints: Check console output

### 4. Common Issues

#### Authentication Issues
- Ensure Zerodha API credentials are set in `.env`
- Check if tokens are valid and not expired

#### Data Issues
- Verify stock symbol exists in Zerodha instruments
- Check if market is open for real-time data
- Ensure sufficient historical data is available

#### Analysis Issues
- Check if all required dependencies are installed
- Verify Gemini API key is configured
- Ensure sufficient memory for large datasets

## 📈 Performance Monitoring

### Response Times
- Data fetching: < 5 seconds
- Technical indicators: < 10 seconds
- Pattern detection: < 15 seconds
- Full analysis: < 60 seconds
- Enhanced analysis: < 120 seconds

### Memory Usage
- Monitor memory usage during large dataset processing
- Consider reducing period/interval for testing

### Error Handling
- All endpoints return proper HTTP status codes
- Error messages include details for debugging
- Failed requests are logged with timestamps

## 🔧 Configuration

### Environment Variables
```bash
# Service Ports
DATA_SERVICE_PORT=8000
ANALYSIS_SERVICE_PORT=8001
WEBSOCKET_SERVICE_PORT=8081
SERVICE_ENDPOINTS_PORT=8002

# API Keys
ZERODHA_API_KEY=your_api_key
ZERODHA_ACCESS_TOKEN=your_access_token
GEMINI_API_KEY=your_gemini_key

# CORS Origins
CORS_ORIGINS=http://localhost:3000,http://localhost:8080,http://localhost:5173
```

### Service Configuration
Each service can be configured independently:
- Host binding
- Port assignment
- Log levels
- CORS settings
- Authentication requirements

## 📚 Additional Resources

- [Data Service Documentation](data_service.py)
- [Analysis Service Documentation](analysis_service.py)
- [WebSocket Service Documentation](websocket_stream_service.py)
- [Technical Indicators Documentation](technical_indicators.py)
- [Pattern Recognition Documentation](patterns/recognition.py)
- [Sector Analysis Documentation](sector_benchmarking.py)

## 🤝 Contributing

When adding new endpoints:
1. Add the endpoint to the appropriate service
2. Update this documentation
3. Add test cases to `test_service_endpoints.py`
4. Update the status endpoint to include new endpoints

## 📞 Support

For issues or questions:
1. Check the health endpoints first
2. Review the logs for error messages
3. Test individual components using the service endpoints
4. Check the test results for specific failures
