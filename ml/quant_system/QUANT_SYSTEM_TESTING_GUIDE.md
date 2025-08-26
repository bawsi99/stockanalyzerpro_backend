# Quantitative System Testing Guide

This guide provides comprehensive instructions for testing the quantitative trading system in StockAnalyzer Pro.

## 🚀 Quick Start

### 1. **Quick Validation Test** (Recommended for first-time testing)
```bash
cd backend/ml/quant_system
python run_quant_tests.py quick
```

### 2. **Comprehensive Test Suite**
```bash
cd backend/ml/quant_system
python run_quant_tests.py all
```

## 📋 Test Categories

### **A. Core System Tests**

#### **1. Simplified System Test** (`test_simplified_system.py`)
**Purpose**: Tests the core working components of the advanced trading system
**Components Tested**:
- Advanced feature engineering (100+ features)
- Multi-modal fusion models (price, news, social media)
- Dynamic ensemble management
- N-BEATS model
- System integration
- Performance metrics

**Run Command**:
```bash
python test_simplified_system.py
```

#### **2. Unified ML System Test** (`test_unified_ml_system.py`)
**Purpose**: Tests the unified machine learning system architecture
**Components Tested**:
- Core ML components and configuration
- Feature engineering pipeline
- Pattern ML engine
- Raw data ML engine
- Hybrid ML engine
- Unified manager
- System integration

**Run Command**:
```bash
python test_unified_ml_system.py
```

#### **3. Advanced Trading System Test** (`test_advanced_trading_system.py`)
**Purpose**: Tests the complete advanced trading system
**Components Tested**:
- Enhanced data pipeline
- Advanced feature engineering
- Multi-modal fusion models
- N-BEATS model
- Dynamic ensemble manager
- Complete trading system
- System integration
- Performance metrics

**Run Command**:
```bash
python test_advanced_trading_system.py
```

### **B. Advanced System Tests**

#### **4. Phase 2 Advanced System Test** (`test_phase2_advanced_system.py`)
**Purpose**: Tests Phase 2 advanced components
**Components Tested**:
- Neural Architecture Search (NAS)
- Meta-Learning Framework
- Advanced Training Strategies
- Temporal Fusion Transformer (TFT)
- Phase 2 Integration Manager
- End-to-end system integration

**Run Command**:
```bash
python test_phase2_advanced_system.py
```

#### **5. Real-Time System Test** (`test_realtime_system.py`)
**Purpose**: Tests real-time capabilities
**Components Tested**:
- Real-time data integration
- Live model predictions
- Actual ensemble management
- Real portfolio data

**Run Command**:
```bash
python test_realtime_system.py
```

### **C. Model-Specific Tests**

#### **6. N-BEATS Model Test** (`test_nbeats.py`)
**Purpose**: Tests the N-BEATS neural network model
**Components Tested**:
- Basic N-BEATS functionality
- Interpretability features
- Performance metrics
- Integration capabilities

**Run Command**:
```bash
python test_nbeats.py
```

### **D. Backtesting Tests**

#### **7. Simple Backtest** (`test_backtest_simple.py`)
**Purpose**: Tests basic backtesting functionality
**Components Tested**:
- Backtesting engine
- Strategy execution
- Performance calculation
- Risk metrics

**Run Command**:
```bash
python test_backtest_simple.py
```

#### **8. Production System Test** (`test_production_system.py`)
**Purpose**: Tests production-ready components
**Components Tested**:
- Market data integration
- Trading strategy execution
- Production deployment readiness

**Run Command**:
```bash
python test_production_system.py
```

### **E. Validation Tests**

#### **9. Step 1 Validation** (`step1_validation.py`)
**Purpose**: Validates system architecture and data flow
**Components Tested**:
- System architecture validation
- Component integration
- Data flow validation
- Feature engineering pipeline

**Run Command**:
```bash
python step1_validation.py
```

#### **10. Unified ML Validation** (`validation_unified_ml.py`)
**Purpose**: Validates unified ML system
**Components Tested**:
- ML system validation
- Model performance
- Integration testing

**Run Command**:
```bash
python validation_unified_ml.py
```

### **F. Feature Engineering Tests**

#### **11. Feature Engineering Test** (`test_feature_engineering.py`)
**Purpose**: Tests feature engineering pipeline
**Components Tested**:
- Technical indicators
- Price features
- Volume features
- Volatility features
- Pattern features

**Run Command**:
```bash
python test_feature_engineering.py
```

#### **12. Debug Feature Engineering** (`debug_feature_engineering.py`)
**Purpose**: Debug feature engineering issues
**Components Tested**:
- Feature calculation debugging
- Data validation
- Error handling

**Run Command**:
```bash
python debug_feature_engineering.py
```

### **G. Price ML Tests**

#### **13. Simple Price ML Test** (`test_price_ml_simple.py`)
**Purpose**: Tests price-based ML models
**Components Tested**:
- Price prediction models
- Model training
- Prediction accuracy

**Run Command**:
```bash
python test_price_ml_simple.py
```

#### **14. Debug Price ML** (`debug_price_ml.py`)
**Purpose**: Debug price ML issues
**Components Tested**:
- Model debugging
- Data preprocessing
- Training issues

**Run Command**:
```bash
python debug_price_ml.py
```

## 🛠️ Using the Test Runner

The `run_quant_tests.py` script provides a unified interface for running tests:

### **Basic Usage**
```bash
# Quick validation (recommended)
python run_quant_tests.py quick

# Run all tests
python run_quant_tests.py all

# Run specific test type
python run_quant_tests.py unified
python run_quant_tests.py advanced
python run_quant_tests.py phase2
python run_quant_tests.py realtime
python run_quant_tests.py nbeats
python run_quant_tests.py backtest
python run_quant_tests.py validation
```

### **Advanced Usage**
```bash
# Run backtesting with specific symbols
python run_quant_tests.py backtest --symbols RELIANCE,TCS,INFY

# Enable verbose output
python run_quant_tests.py quick --verbose

# Enable debug mode
python run_quant_tests.py quick --debug
```

## 📊 Test Results Interpretation

### **Success Indicators**
- ✅ **PASSED**: Component is working correctly
- 🎉 **All tests passed**: System is ready for production
- 📊 **Performance metrics**: Within expected ranges

### **Failure Indicators**
- ❌ **FAILED**: Component has issues that need fixing
- ⚠️ **Warnings**: Non-critical issues that should be addressed
- 🔧 **Configuration issues**: Missing dependencies or setup

### **Common Test Results**

#### **Simplified System Test Results**
```
🚀 Starting Simplified Advanced Trading System Tests...
============================================================

📋 Running Advanced Feature Engineering Test...
✅ Advanced feature engineering working correctly

📋 Running Multi-Modal Fusion Model Test...
✅ Multi-modal fusion model operational

📋 Running N-BEATS Model Test...
✅ N-BEATS model functioning properly

📊 Simplified Advanced Trading System Test Results Summary:
============================================================
Advanced Feature Engineering    ✅ PASSED
Multi-Modal Fusion Model       ✅ PASSED
N-BEATS Model                  ✅ PASSED
Dynamic Ensemble Manager       ✅ PASSED
System Integration             ✅ PASSED
Performance Metrics            ✅ PASSED

Overall: 6/6 tests passed

🎉 All tests passed! The core advanced trading system is working.
```

## 🔧 Troubleshooting

### **Common Issues**

#### **1. Import Errors**
```bash
# Solution: Install missing dependencies
pip install -r requirements.txt
```

#### **2. Module Not Found**
```bash
# Solution: Check Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/backend/ml/quant_system"
```

#### **3. Data Access Issues**
```bash
# Solution: Check Zerodha credentials
# Ensure .env file is properly configured
```

#### **4. Memory Issues**
```bash
# Solution: Reduce test data size
# Modify n_points parameter in test functions
```

### **Debug Mode**
Enable debug mode for detailed error information:
```bash
python run_quant_tests.py quick --debug
```

### **Verbose Output**
Enable verbose output for detailed test progress:
```bash
python run_quant_tests.py quick --verbose
```

## 📈 Performance Testing

### **Load Testing**
```bash
# Test with larger datasets
python test_advanced_trading_system.py  # Uses 500 data points by default
```

### **Memory Testing**
```bash
# Monitor memory usage during tests
python -m memory_profiler test_simplified_system.py
```

### **Speed Testing**
```bash
# Time the execution
time python test_simplified_system.py
```

## 🚀 Production Readiness

### **Pre-Production Checklist**
- [ ] All quick tests pass
- [ ] All advanced tests pass
- [ ] Real-time tests pass
- [ ] Backtesting tests pass
- [ ] Validation tests pass
- [ ] Performance metrics within acceptable ranges
- [ ] Memory usage optimized
- [ ] Error handling verified

### **Production Deployment**
```bash
# Run comprehensive test suite
python run_quant_tests.py all

# If all tests pass, system is ready for production
```

## 📚 Additional Resources

### **Documentation**
- `README.md`: Main system documentation
- `ANALYSIS_SERVICE_JSON_STRUCTURE.md`: API structure documentation
- `PROMPT_FORMATTING_ISSUE_FIX.md`: Prompt formatting fixes

### **Configuration**
- `config.py`: System configuration
- `.env`: Environment variables
- `requirements.txt`: Python dependencies

### **Logs**
- Check `logs/` directory for detailed execution logs
- Monitor `*.log` files for specific component logs

## 🆘 Getting Help

If you encounter issues:

1. **Check the logs**: Look for error messages in log files
2. **Run debug mode**: Use `--debug` flag for detailed output
3. **Check dependencies**: Ensure all required packages are installed
4. **Verify configuration**: Check environment variables and config files
5. **Review test output**: Look for specific failure messages

## 📝 Test Development

### **Adding New Tests**
1. Create test file following naming convention: `test_*.py`
2. Implement test functions with descriptive names
3. Add proper error handling and logging
4. Update `run_quant_tests.py` if needed
5. Document the test purpose and components

### **Test Best Practices**
- Use synthetic data for testing
- Implement proper cleanup
- Add comprehensive error handling
- Include performance metrics
- Document test assumptions
- Make tests reproducible

---

**Last Updated**: January 2025
**Version**: 3.0
**Maintainer**: StockAnalyzer Pro Team
