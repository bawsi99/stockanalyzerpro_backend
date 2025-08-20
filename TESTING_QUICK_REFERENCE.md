# Quantitative System Testing - Quick Reference

## 🚀 Quick Start

### 1. Setup Testing Environment
```bash
cd backend
python setup_testing.py --all
```

### 2. Run Quick Validation
```bash
python run_quant_tests.py quick
```

### 3. Run Basic Tests
```bash
python run_basic_tests.py
```

## 📋 Essential Test Commands

### Quick Tests
```bash
# Quick validation (recommended first)
python run_quant_tests.py quick

# Basic functionality tests
python run_basic_tests.py

# Service endpoint tests
python test_service_endpoints.py all
```

### Comprehensive Testing
```bash
# All tests
python run_quant_tests.py all

# Specific test types
python run_quant_tests.py unit
python run_quant_tests.py integration
python run_quant_tests.py performance
python run_quant_tests.py service
```

### Backtesting
```bash
# Basic backtesting
python run_quant_tests.py backtest --symbols RELIANCE

# Multiple symbols
python run_quant_tests.py backtest --symbols RELIANCE TCS INFY --days 60

# Direct backtesting
python -m backend.run_backtests --symbols RELIANCE --days 30
```

### Coverage Testing
```bash
# Coverage with terminal report
python run_quant_tests.py coverage

# Coverage with HTML report
python run_quant_tests.py coverage --html
```

## 🔧 Manual Testing

### Service Testing
```bash
# Start services
python run_services.py

# Test individual services
python test_service_endpoints.py data
python test_service_endpoints.py analysis
python test_service_endpoints.py websocket
```

### Component Testing
```bash
# Test ML models
python -c "from ml.model import load_model; print('ML Model:', load_model() is not None)"

# Test technical indicators
python -c "from technical_indicators import TechnicalIndicators; print('Technical Indicators: OK')"

# Test pattern recognition
python -c "from patterns.recognition import PatternRecognition; print('Pattern Recognition: OK')"
```

## 📊 Test Categories

### 1. Unit Tests
- **Purpose**: Test individual functions and classes
- **Location**: `tests/unit/`
- **Command**: `python run_quant_tests.py unit`

### 2. Integration Tests
- **Purpose**: Test component interactions
- **Location**: `tests/integration/`
- **Command**: `python run_quant_tests.py integration`

### 3. Performance Tests
- **Purpose**: Test system performance and load handling
- **Location**: `tests/performance/`
- **Command**: `python run_quant_tests.py performance`

### 4. Backtesting
- **Purpose**: Validate trading strategies and patterns
- **Command**: `python run_quant_tests.py backtest`

### 5. Service Tests
- **Purpose**: Test API endpoints and services
- **Command**: `python run_quant_tests.py service`

## 🎯 Testing Focus Areas

### Core Components
- ✅ Machine Learning Models (`ml/model.py`)
- ✅ Technical Indicators (`technical_indicators.py`)
- ✅ Pattern Recognition (`patterns/recognition.py`)
- ✅ Data Pipeline (`zerodha_client.py`)
- ✅ Service Endpoints (`service_endpoints.py`)

### Key Validations
- ✅ Data fetching and processing
- ✅ Indicator calculations
- ✅ Pattern detection accuracy
- ✅ ML model predictions
- ✅ API response times
- ✅ Error handling

## 📈 Performance Benchmarks

### Response Time Targets
- **Data Service**: < 2 seconds
- **Analysis Service**: < 10 seconds
- **Service Endpoints**: < 5 seconds
- **WebSocket**: < 1 second

### Load Testing
- **Concurrent Requests**: 10+ simultaneous
- **Memory Usage**: < 500MB increase
- **Success Rate**: > 80%

## 🔍 Debugging

### Common Issues
```bash
# Check if services are running
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health

# Check logs
tail -f logs/analysis.log
tail -f logs/data.log

# Test specific endpoint
curl -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol": "RELIANCE", "exchange": "NSE", "period": 30}'
```

### Environment Variables
```bash
# Enable test mode
export TEST_MODE=true
export MARKET_HOURS_TEST_MODE=true
export FORCE_MARKET_OPEN=true

# Check configuration
python -c "import os; print('Test Mode:', os.getenv('TEST_MODE'))"
```

## 📝 Test Reports

### Generated Files
- `test_report.json` - Detailed test results
- `htmlcov/` - HTML coverage report
- `coverage.xml` - Coverage data for CI

### Report Analysis
```bash
# View test report
cat test_report.json | jq '.summary'

# Check coverage
python -m coverage report
```

## 🚨 Troubleshooting

### Test Failures
1. **Import Errors**: Check dependencies `pip install -r requirements.txt`
2. **Service Errors**: Ensure services are running `python run_services.py`
3. **Data Errors**: Verify Zerodha credentials and market hours
4. **ML Errors**: Check if models are trained and available

### Performance Issues
1. **Slow Tests**: Use `--verbose` for detailed output
2. **Memory Issues**: Monitor with `psutil`
3. **Timeout Errors**: Increase timeout values in test config

## 📚 Additional Resources

- **Full Guide**: `QUANT_SYSTEM_TESTING_GUIDE.md`
- **Service Documentation**: `SERVICE_ENDPOINTS_README.md`
- **Backtesting Guide**: `README.md` (backtesting section)

## 🎯 Best Practices

1. **Start with Quick Tests**: Always run `python run_quant_tests.py quick` first
2. **Test Incrementally**: Run specific test types before running all
3. **Monitor Performance**: Use coverage reports and performance metrics
4. **Validate Data**: Ensure test data is realistic and comprehensive
5. **Check Services**: Verify all services are running before integration tests

---

**Remember**: The quantitative system is complex. Start with quick validation tests and gradually move to more comprehensive testing as needed.
