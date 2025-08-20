# Quantitative System Testing Guide

This comprehensive guide covers all aspects of testing the StockAnalyzer Pro quantitative system, from unit tests to backtesting and performance validation.

## 🎯 Testing Overview

The quantitative system consists of multiple components that need thorough testing:

1. **Machine Learning Models** - Pattern recognition, signal scoring, risk assessment
2. **Technical Indicators** - All technical analysis calculations
3. **Pattern Recognition** - Chart pattern detection algorithms
4. **Backtesting Engine** - Strategy validation and performance testing
5. **Data Pipeline** - Data fetching, processing, and validation
6. **API Services** - All service endpoints and integrations
7. **Real-time Systems** - WebSocket streaming and live data

## 🧪 1. Unit Testing

### Setup Testing Environment

```bash
# Install testing dependencies
pip install pytest pytest-asyncio pytest-cov pytest-mock

# Create test configuration
mkdir -p backend/tests/{unit,integration,performance}
```

### Machine Learning Model Testing

```python
# backend/tests/unit/test_ml_models.py
import pytest
import numpy as np
import pandas as pd
from ml.model import train_global_model, load_model, load_registry
from ml.dataset import build_pooled_dataset

class TestMLModels:
    def test_model_training(self):
        """Test model training functionality."""
        # Test with sample data
        dataset = build_pooled_dataset()
        report = train_global_model(dataset, n_splits=3)
        
        assert report is not None
        assert report.model_path is not None
        assert report.metrics['brier'] >= 0
        assert report.metrics['logloss'] >= 0
        assert report.metrics['n_samples'] > 0
    
    def test_model_loading(self):
        """Test model loading functionality."""
        model = load_model()
        registry = load_registry()
        
        if model is not None:
            assert hasattr(model, 'predict_proba')
            assert registry is not None
            assert 'metrics' in registry
    
    def test_model_prediction(self):
        """Test model prediction functionality."""
        model = load_model()
        if model is None:
            pytest.skip("No trained model available")
        
        # Create test features
        test_features = pd.DataFrame({
            'rsi': [65.0],
            'macd': [0.25],
            'volume_ratio': [1.2],
            'pattern_type': ['bullish']
        })
        
        predictions = model.predict_proba(test_features)
        assert predictions.shape[1] == 2  # Binary classification
        assert np.all(predictions >= 0) and np.all(predictions <= 1)
```

### Technical Indicators Testing

```python
# backend/tests/unit/test_technical_indicators.py
import pytest
import pandas as pd
import numpy as np
from technical_indicators import TechnicalIndicators

class TestTechnicalIndicators:
    def setup_method(self):
        """Setup test data."""
        self.test_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
    
    def test_rsi_calculation(self):
        """Test RSI calculation."""
        rsi = TechnicalIndicators.calculate_rsi(self.test_data, period=14)
        assert len(rsi) == len(self.test_data)
        assert all(0 <= val <= 100 for val in rsi.dropna())
    
    def test_macd_calculation(self):
        """Test MACD calculation."""
        macd = TechnicalIndicators.calculate_macd(self.test_data)
        assert 'macd_line' in macd
        assert 'signal_line' in macd
        assert 'histogram' in macd
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        bb = TechnicalIndicators.calculate_bollinger_bands(self.test_data)
        assert 'upper_band' in bb
        assert 'middle_band' in bb
        assert 'lower_band' in bb
        assert 'percent_b' in bb
```

### Pattern Recognition Testing

```python
# backend/tests/unit/test_pattern_recognition.py
import pytest
import pandas as pd
from patterns.recognition import PatternRecognition

class TestPatternRecognition:
    def setup_method(self):
        """Setup test data with known patterns."""
        # Create data with a clear double top pattern
        self.test_data = pd.DataFrame({
            'open': [100] * 20,
            'high': [110, 105, 100, 95, 90, 85, 80, 85, 90, 95, 100, 105, 110, 105, 100, 95, 90, 85, 80, 75],
            'low': [90] * 20,
            'close': [105, 100, 95, 90, 85, 80, 85, 90, 95, 100, 105, 100, 95, 90, 85, 80, 75, 70, 65, 60],
            'volume': [1000] * 20
        })
    
    def test_double_top_detection(self):
        """Test double top pattern detection."""
        patterns = PatternRecognition.detect_double_tops(self.test_data)
        assert len(patterns) > 0
        assert any(p['pattern_type'] == 'double_top' for p in patterns)
    
    def test_support_resistance_detection(self):
        """Test support and resistance level detection."""
        levels = PatternRecognition.detect_support_resistance_levels(self.test_data)
        assert 'support_levels' in levels
        assert 'resistance_levels' in levels
    
    def test_volume_analysis(self):
        """Test volume pattern analysis."""
        volume_analysis = PatternRecognition.analyze_volume_patterns(self.test_data)
        assert 'volume_trend' in volume_analysis
        assert 'volume_anomalies' in volume_analysis
```

## 🔄 2. Integration Testing

### Service Integration Testing

```python
# backend/tests/integration/test_service_integration.py
import pytest
import asyncio
import aiohttp
from test_service_endpoints import ServiceEndpointTester

class TestServiceIntegration:
    @pytest.mark.asyncio
    async def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline integration."""
        async with ServiceEndpointTester() as tester:
            # Test data service
            data_results = await tester.test_data_service()
            assert all(r['success'] for r in data_results)
            
            # Test analysis service
            analysis_results = await tester.test_analysis_service()
            assert all(r['success'] for r in analysis_results)
            
            # Test service endpoints
            endpoint_results = await tester.test_service_endpoints()
            assert all(r['success'] for r in endpoint_results)
    
    @pytest.mark.asyncio
    async def test_websocket_integration(self):
        """Test WebSocket streaming integration."""
        async with ServiceEndpointTester() as tester:
            websocket_results = await tester.test_websocket_service()
            assert all(r['success'] for r in websocket_results)
```

### Data Pipeline Testing

```python
# backend/tests/integration/test_data_pipeline.py
import pytest
import asyncio
from zerodha_client import ZerodhaDataClient
from technical_indicators import TechnicalIndicators
from patterns.recognition import PatternRecognition

class TestDataPipeline:
    @pytest.mark.asyncio
    async def test_data_fetch_to_analysis(self):
        """Test complete data pipeline from fetch to analysis."""
        # 1. Fetch data
        client = ZerodhaDataClient()
        assert client.authenticate()
        
        data = await client.get_historical_data(
            symbol="RELIANCE",
            exchange="NSE",
            interval="day",
            period=30
        )
        assert data is not None and not data.empty
        
        # 2. Calculate indicators
        indicators = TechnicalIndicators.calculate_all_indicators(data)
        assert 'rsi' in indicators
        assert 'macd' in indicators
        assert 'bollinger_bands' in indicators
        
        # 3. Detect patterns
        patterns = PatternRecognition.detect_all_patterns(data)
        assert isinstance(patterns, dict)
        
        # 4. Validate data consistency
        assert len(data) == len(indicators['rsi'])
```

## 📊 3. Backtesting

### Strategy Backtesting

```bash
# Run backtesting for specific symbols
python -m backend.run_backtests --symbols RELIANCE TCS INFY --days 180 --lookahead 3

# Run backtesting with custom parameters
python -m backend.run_backtests \
    --symbols RELIANCE \
    --days 365 \
    --lookahead 5 \
    --threshold 2.0 \
    --workers 4
```

### Backtesting Validation

```python
# backend/tests/integration/test_backtesting.py
import pytest
from backtesting import Backtester
from datetime import datetime, timedelta

class TestBacktesting:
    def test_backtest_execution(self):
        """Test backtesting execution."""
        bt = Backtester(exchange="NSE")
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)
        
        results = bt.run(
            symbol="RELIANCE",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            interval="day",
            lookahead_days=3,
            success_threshold_pct=1.0
        )
        
        assert isinstance(results, dict)
        for pattern_type, stats in results.items():
            assert 'detected' in stats
            assert 'confirmed' in stats
            assert 'success' in stats
            assert 'success_rate' in stats
            assert 0 <= stats['success_rate'] <= 100
    
    def test_pattern_backtesting(self):
        """Test individual pattern backtesting."""
        from patterns.recognition import PatternRecognition
        
        # Create test data
        test_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 104, 103, 102, 101],
            'volume': [1000] * 10
        })
        
        # Test pattern backtesting
        results = PatternRecognition.backtest_pattern(
            test_data,
            PatternRecognition.detect_double_tops,
            window=5,
            hold_period=3
        )
        
        assert 'win_rate' in results
        assert 'avg_return' in results
        assert 'expectancy' in results
        assert 'n_trades' in results
```

## ⚡ 4. Performance Testing

### Load Testing

```python
# backend/tests/performance/test_load.py
import pytest
import asyncio
import time
import aiohttp
from concurrent.futures import ThreadPoolExecutor

class TestPerformance:
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test system performance under concurrent load."""
        async with aiohttp.ClientSession() as session:
            # Test concurrent analysis requests
            tasks = []
            for i in range(10):
                task = session.post(
                    "http://localhost:8001/analyze",
                    json={
                        "symbol": "RELIANCE",
                        "exchange": "NSE",
                        "period": 30,
                        "interval": "day"
                    }
                )
                tasks.append(task)
            
            start_time = time.time()
            responses = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # Validate responses
            success_count = sum(1 for r in responses if r.status == 200)
            assert success_count >= 8  # At least 80% success rate
            
            # Performance metrics
            total_time = end_time - start_time
            avg_time = total_time / len(responses)
            assert avg_time < 5.0  # Average response time under 5 seconds
    
    def test_memory_usage(self):
        """Test memory usage during analysis."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple analyses
        for i in range(5):
            # Simulate analysis
            pass
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 500  # Less than 500MB increase
```

### Response Time Testing

```python
# backend/tests/performance/test_response_times.py
import pytest
import time
import asyncio
from test_service_endpoints import ServiceEndpointTester

class TestResponseTimes:
    @pytest.mark.asyncio
    async def test_endpoint_response_times(self):
        """Test response times for all endpoints."""
        async with ServiceEndpointTester() as tester:
            # Test data service response times
            data_results = await tester.test_data_service()
            for result in data_results:
                assert result['duration'] < 2.0  # Under 2 seconds
            
            # Test analysis service response times
            analysis_results = await tester.test_analysis_service()
            for result in analysis_results:
                assert result['duration'] < 10.0  # Under 10 seconds
            
            # Test service endpoints response times
            endpoint_results = await tester.test_service_endpoints()
            for result in endpoint_results:
                assert result['duration'] < 5.0  # Under 5 seconds
```

## 🔍 5. System Validation

### End-to-End Testing

```python
# backend/tests/integration/test_e2e.py
import pytest
import asyncio
from agent_capabilities import StockAnalysisOrchestrator

class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_complete_analysis_workflow(self):
        """Test complete analysis workflow from start to finish."""
        orchestrator = StockAnalysisOrchestrator()
        
        # Run complete analysis
        result = await orchestrator.analyze_stock(
            symbol="RELIANCE",
            exchange="NSE",
            period=365,
            interval="day"
        )
        
        # Validate complete result structure
        assert result['success'] is True
        assert 'technical_analysis' in result
        assert 'pattern_analysis' in result
        assert 'ai_analysis' in result
        assert 'sector_analysis' in result
        assert 'risk_assessment' in result
        assert 'recommendations' in result
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in complete workflow."""
        orchestrator = StockAnalysisOrchestrator()
        
        # Test with invalid symbol
        result = await orchestrator.analyze_stock(
            symbol="INVALID_SYMBOL",
            exchange="NSE",
            period=30,
            interval="day"
        )
        
        # Should handle error gracefully
        assert result['success'] is False
        assert 'error' in result
        assert 'message' in result
```

## 🚀 6. Running Tests

### Test Execution Commands

```bash
# Run all tests
cd backend
python -m pytest tests/ -v --cov=. --cov-report=html

# Run specific test categories
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/performance/ -v

# Run with specific markers
python -m pytest tests/ -m "slow" -v
python -m pytest tests/ -m "not slow" -v

# Run tests in parallel
python -m pytest tests/ -n auto -v

# Generate coverage report
python -m pytest tests/ --cov=. --cov-report=html --cov-report=term-missing
```

### Continuous Integration Testing

```yaml
# .github/workflows/test.yml
name: Quantitative System Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        cd backend
        pip install -r requirements.txt
        pip install pytest pytest-asyncio pytest-cov pytest-mock
    
    - name: Run unit tests
      run: |
        cd backend
        python -m pytest tests/unit/ -v --cov=. --cov-report=xml
    
    - name: Run integration tests
      run: |
        cd backend
        python -m pytest tests/integration/ -v
    
    - name: Run performance tests
      run: |
        cd backend
        python -m pytest tests/performance/ -v -m "not slow"
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
      with:
        file: ./backend/coverage.xml
```

## 📈 7. Test Data Management

### Fixture Management

```python
# backend/tests/conftest.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_stock_data():
    """Provide sample stock data for testing."""
    return pd.DataFrame({
        'open': [100, 101, 102, 103, 104, 105, 104, 103, 102, 101],
        'high': [105, 106, 107, 108, 109, 110, 109, 108, 107, 106],
        'low': [95, 96, 97, 98, 99, 100, 99, 98, 97, 96],
        'close': [101, 102, 103, 104, 105, 106, 105, 104, 103, 102],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1400, 1300, 1200, 1100]
    })

@pytest.fixture
def mock_zerodha_client(mocker):
    """Mock Zerodha client for testing."""
    mock_client = mocker.patch('zerodha_client.ZerodhaDataClient')
    mock_client.return_value.authenticate.return_value = True
    return mock_client

@pytest.fixture
def test_symbols():
    """Provide test symbols."""
    return ["RELIANCE", "TCS", "INFY", "HDFC", "ICICIBANK"]
```

## 🔧 8. Test Configuration

### Environment Setup

```bash
# backend/tests/test_config.py
import os
import pytest

class TestConfig:
    """Test configuration management."""
    
    @pytest.fixture(autouse=True)
    def setup_test_env(self):
        """Setup test environment variables."""
        os.environ['TEST_MODE'] = 'true'
        os.environ['MARKET_HOURS_TEST_MODE'] = 'true'
        os.environ['FORCE_MARKET_OPEN'] = 'true'
        yield
        # Cleanup
        os.environ.pop('TEST_MODE', None)
        os.environ.pop('MARKET_HOURS_TEST_MODE', None)
        os.environ.pop('FORCE_MARKET_OPEN', None)
```

## 📊 9. Test Reporting

### Test Results Analysis

```python
# backend/tests/reporting/test_reporter.py
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

class TestReporter:
    """Generate comprehensive test reports."""
    
    def generate_test_report(self, results_file: str):
        """Generate detailed test report."""
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Create summary
        summary = {
            'total_tests': len(results),
            'passed': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'avg_response_time': sum(r['duration'] for r in results) / len(results),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save report
        report_file = Path('test_report.json')
        with open(report_file, 'w') as f:
            json.dump({'summary': summary, 'details': results}, f, indent=2)
        
        return summary
```

## 🎯 10. Testing Best Practices

### Code Quality

1. **Test Coverage**: Aim for >90% code coverage
2. **Test Isolation**: Each test should be independent
3. **Mocking**: Use mocks for external dependencies
4. **Data Validation**: Always validate input and output data
5. **Error Testing**: Test both success and failure scenarios

### Performance Testing

1. **Baseline Metrics**: Establish performance baselines
2. **Load Testing**: Test under realistic load conditions
3. **Memory Monitoring**: Track memory usage patterns
4. **Response Time**: Monitor response time degradation

### Continuous Testing

1. **Automated Runs**: Run tests on every code change
2. **Regression Testing**: Ensure new changes don't break existing functionality
3. **Integration Testing**: Test component interactions
4. **End-to-End Testing**: Test complete workflows

## 🚀 Quick Start Testing

```bash
# 1. Setup testing environment
cd backend
pip install pytest pytest-asyncio pytest-cov pytest-mock

# 2. Run basic tests
python -m pytest tests/unit/ -v

# 3. Run integration tests
python -m pytest tests/integration/ -v

# 4. Run backtesting
python -m backend.run_backtests --symbols RELIANCE --days 30

# 5. Run service tests
python test_service_endpoints.py all

# 6. Generate coverage report
python -m pytest tests/ --cov=. --cov-report=html
```

This comprehensive testing guide ensures your quantitative system is thoroughly validated across all components and scenarios.
