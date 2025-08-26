# Quant System Testing - Quick Reference

## 🚀 Quick Start Commands

### **1. Quick Validation** (Recommended)
```bash
cd backend/ml/quant_system
python run_quant_tests.py quick
```

### **2. Comprehensive Testing**
```bash
cd backend/ml/quant_system
python run_quant_tests.py all
```

### **3. Individual Test Suites**
```bash
# Core system tests
python test_simplified_system.py
python test_unified_ml_system.py
python test_advanced_trading_system.py

# Advanced system tests
python test_phase2_advanced_system.py
python test_realtime_system.py

# Model-specific tests
python test_nbeats.py

# Backtesting tests
python test_backtest_simple.py
python test_production_system.py

# Validation tests
python step1_validation.py
python validation_unified_ml.py
```

## 📋 Test Runner Options

```bash
# Available test types
python run_quant_tests.py quick      # Quick validation
python run_quant_tests.py unified    # Unified ML system
python run_quant_tests.py advanced   # Advanced trading system
python run_quant_tests.py phase2     # Phase 2 advanced system
python run_quant_tests.py realtime   # Real-time system
python run_quant_tests.py nbeats     # N-BEATS model
python run_quant_tests.py backtest   # Backtesting
python run_quant_tests.py validation # Validation tests
python run_quant_tests.py all        # All tests

# Options
python run_quant_tests.py backtest --symbols RELIANCE,TCS,INFY
python run_quant_tests.py quick --verbose
python run_quant_tests.py quick --debug
```

## ✅ Expected Test Results

### **Quick Test Success Output**
```
🚀 Running Quick Quant System Tests...
============================================================

🧪 Simplified Advanced Trading System
✅ Advanced Feature Engineering   ✅ PASSED
✅ Multi-Modal Fusion Model       ✅ PASSED
✅ N-BEATS Model                  ✅ PASSED
✅ Dynamic Ensemble Manager       ✅ PASSED
✅ System Integration             ✅ PASSED
✅ Performance Metrics            ✅ PASSED

🧪 Unified ML System
✅ Core Components: ✅ PASS
✅ Feature Engineering: ✅ PASS
✅ Pattern ML: ✅ PASS
✅ Raw Data ML: ✅ PASS
✅ Hybrid ML: ✅ PASS
✅ Unified Manager: ✅ PASS
✅ System Integration: ✅ PASS

🧪 N-BEATS Model
✅ Basic Functionality  ✅ PASSED
✅ Interpretability     ✅ PASSED
✅ Performance          ✅ PASSED
✅ Integration          ✅ PASSED

Overall: 3/3 tests passed
```

## 🔧 Troubleshooting

### **Common Issues & Solutions**

#### **1. Import Errors**
```bash
# Install dependencies
pip install -r requirements.txt
```

#### **2. Module Not Found**
```bash
# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### **3. Memory Issues**
```bash
# Reduce test data size in test files
# Look for n_points parameter and reduce it
```

#### **4. Configuration Issues**
```bash
# Check .env file exists and has proper credentials
# Ensure config.py is properly set up
```

## 📊 Test Categories Overview

| Test Type | Purpose | Components Tested |
|-----------|---------|-------------------|
| **Quick** | Core validation | Feature engineering, ML models, ensemble management |
| **Unified** | ML system architecture | Core ML, feature engineering, pattern/raw/hybrid ML |
| **Advanced** | Complete trading system | Data pipeline, feature engineering, fusion models |
| **Phase2** | Advanced components | NAS, meta-learning, TFT, advanced training |
| **Real-time** | Live capabilities | Real-time data, live predictions, portfolio tracking |
| **N-BEATS** | Neural model | N-BEATS functionality, interpretability, performance |
| **Backtest** | Strategy testing | Backtesting engine, strategy execution, risk metrics |
| **Validation** | System validation | Architecture, data flow, integration |

## 🚀 Production Readiness

### **Pre-Production Checklist**
- [ ] `python run_quant_tests.py quick` - ✅ PASSED
- [ ] `python run_quant_tests.py advanced` - ✅ PASSED
- [ ] `python run_quant_tests.py realtime` - ✅ PASSED
- [ ] `python run_quant_tests.py backtest` - ✅ PASSED
- [ ] `python run_quant_tests.py validation` - ✅ PASSED
- [ ] Performance metrics within acceptable ranges
- [ ] Memory usage optimized
- [ ] Error handling verified

### **Full Production Test**
```bash
python run_quant_tests.py all
```

## 📚 Documentation

- **Detailed Guide**: `QUANT_SYSTEM_TESTING_GUIDE.md`
- **Main Documentation**: `README.md`
- **API Structure**: `ANALYSIS_SERVICE_JSON_STRUCTURE.md`

## 🆘 Getting Help

1. **Check logs**: Look in `logs/` directory
2. **Debug mode**: Use `--debug` flag
3. **Verbose output**: Use `--verbose` flag
4. **Review test output**: Look for specific error messages
5. **Check dependencies**: Ensure all packages installed

---

**Quick Command Reference**: `python run_quant_tests.py quick`
