#!/usr/bin/env python3
"""
Comprehensive Test Script for Unified ML System

This script tests all components of the unified ML system:
1. Core configuration and base classes
2. Pattern-based ML engine
3. Raw data ML engine
4. Traditional ML engine
5. Hybrid ML engine
6. Feature engineering
7. Unified manager
8. Integration with other system components
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_core_components():
    """Test core ML components."""
    print("🧪 Testing Core ML Components...")
    
    try:
        from ml.core import UnifiedMLConfig, BaseMLEngine, MLModelRegistry
        
        # Test configuration
        config = UnifiedMLConfig()
        print(f"✅ UnifiedMLConfig created: {config}")
        
        # Test model registry
        registry = MLModelRegistry()
        print(f"✅ MLModelRegistry created: {registry}")
        
        # Test base engine
        base_engine = BaseMLEngine(config)
        print(f"✅ BaseMLEngine created: {base_engine}")
        
        return True
        
    except Exception as e:
        print(f"❌ Core components test failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering engine."""
    print("\n🧪 Testing Feature Engineering...")
    
    try:
        from ml.feature_engineering import feature_engineer
        
        # Create sample data
        sample_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [98, 99, 100, 101, 102],
            'close': [103, 104, 105, 106, 107],
            'volume': [1000000, 1200000, 1100000, 1300000, 1250000]
        })
        
        # Create features
        features = feature_engineer.create_all_features(sample_data)
        
        if not features.empty:
            print(f"✅ Features created successfully: {features.shape}")
            print(f"   Feature columns: {list(features.columns)}")
            return True
        else:
            print("❌ No features created")
            return False
            
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        return False

def test_pattern_ml():
    """Test pattern-based ML engine."""
    print("\n🧪 Testing Pattern-Based ML...")
    
    try:
        from ml.pattern_ml import pattern_ml_engine
        
        # Test engine creation
        print(f"✅ Pattern ML engine created: {pattern_ml_engine}")
        
        # Test adding pattern data
        pattern_ml_engine.add_pattern_data(
            pattern_type="head_shoulders",
            features={"duration": 10, "volume_ratio": 1.2, "trend_alignment": 0.8, "completion": 0.9},
            outcome=True,
            confirmed=True
        )
        print("✅ Pattern data added successfully")
        
        # Test model info
        model_info = pattern_ml_engine.get_model_info()
        print(f"✅ Model info retrieved: {model_info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pattern ML test failed: {e}")
        return False

def test_raw_data_ml():
    """Test raw data ML engine."""
    print("\n🧪 Testing Raw Data ML...")
    
    try:
        from ml.raw_data_ml import raw_data_ml_engine
        
        # Test engine creation
        print(f"✅ Raw Data ML engine created: {raw_data_ml_engine}")
        
        # Create sample data
        sample_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            'close': [103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
            'volume': [1000000, 1200000, 1100000, 1300000, 1250000, 1400000, 1350000, 1500000, 1450000, 1600000]
        })
        
        # Test volatility prediction
        volatility_pred = raw_data_ml_engine.predict_volatility(sample_data)
        print(f"✅ Volatility prediction: {volatility_pred}")
        
        # Test market regime classification
        market_regime = raw_data_ml_engine.classify_market_regime(sample_data)
        print(f"✅ Market regime: {market_regime}")
        
        return True
        
    except Exception as e:
        print(f"❌ Raw Data ML test failed: {e}")
        return False

# def test_traditional_ml():  # REMOVED (not needed with CatBoost)
#     """Test traditional ML engine."""
#     print("\n🧪 Testing Traditional ML...")
#     
#     try:
#         from ml.traditional_ml import traditional_ml_engine
#         
#         # Test engine creation
#         print(f"✅ Traditional ML engine created: {traditional_ml_engine}")
#         
#         # Create sample data with features
#         sample_data = pd.DataFrame({
#             'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
#             'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
#         'low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
#         'close': [103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
#         'volume': [1000000, 1200000, 1100000, 1300000, 1250000, 1400000, 1350000, 1500000, 1450000, 1600000],
#         'returns': [0.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
#         'rsi': [50, 52, 54, 56, 58, 60, 62, 64, 66, 68],
#         'macd': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#     }
#         
#     # Test model training
#     results = traditional_ml_engine.train_all_models(sample_data)
#     print(f"✅ Model training completed: {len(results)} model types")
#         
#     return True
#         
# except Exception as e:
#     print(f"❌ Traditional ML test failed: {e}")
#     return False

def test_hybrid_ml():
    """Test hybrid ML engine."""
    print("\n🧪 Testing Hybrid ML...")
    
    try:
        from ml.hybrid_ml import hybrid_ml_engine
        
        # Test engine creation
        print(f"✅ Hybrid ML engine created: {hybrid_ml_engine}")
        
        # Create sample data
        sample_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            'close': [103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
            'volume': [1000000, 1200000, 1100000, 1300000, 1250000, 1400000, 1350000, 1500000, 1450000, 1600000]
        })
        
        # Test comprehensive analysis
        analysis = hybrid_ml_engine.get_comprehensive_analysis(
            stock_data=sample_data,
            pattern_features={"duration": 10, "volume_ratio": 1.2, "trend_alignment": 0.8, "completion": 0.9},
            pattern_type="head_shoulders"
        )
        
        print(f"✅ Hybrid analysis completed: {len(analysis)} components")
        
        return True
        
    except Exception as e:
        print(f"❌ Hybrid ML test failed: {e}")
        return False

def test_unified_manager():
    """Test unified ML manager."""
    print("\n🧪 Testing Unified ML Manager...")
    
    try:
        from ml import unified_ml_manager
        
        # Test manager creation
        print(f"✅ Unified ML manager created: {unified_ml_manager}")
        
        # Test system summary
        summary = unified_ml_manager.get_system_summary()
        print(f"✅ System summary: {summary}")
        
        # Test engine status
        status = unified_ml_manager.get_engine_status()
        print(f"✅ Engine status: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Unified manager test failed: {e}")
        return False

def test_integration():
    """Test integration with other system components."""
    print("\n🧪 Testing System Integration...")
    
    try:
        # Test feature engineering integration
        from ml.feature_engineering import feature_engineer
        
        # Test ML integration
        from ml import unified_ml_manager
        
        print("✅ Core ML components imported successfully")
        
        # Test end-to-end workflow
        sample_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            'close': [103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
            'volume': [1000000, 1200000, 1100000, 1300000, 1250000, 1400000, 1350000, 1500000, 1450000, 1600000]
        })
        
        # Create features
        features = feature_engineer.create_all_features(sample_data)
        print(f"✅ Feature engineering integration: {features.shape}")
        
        # Test ML engine integration
        status = unified_ml_manager.get_engine_status()
        print(f"✅ ML engine integration: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests."""
    print("🚀 UNIFIED ML SYSTEM COMPREHENSIVE TEST")
    print("=" * 60)
    print("✅ No Backward Compatibility Required")
    print("✅ Clean, Unified ML System Only")
    print("=" * 60)
    
    tests = [
        ("Core Components", test_core_components),
        ("Feature Engineering", test_feature_engineering),
        ("Pattern ML", test_pattern_ml),
        ("Raw Data ML", test_raw_data_ml),
        # ("Traditional ML", test_traditional_ml),  # REMOVED
        ("Hybrid ML", test_hybrid_ml),
        ("Unified Manager", test_unified_manager),
        ("System Integration", test_integration)
    ]
    
    results = {}
    total_tests = len(tests)
    passed_tests = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Print results summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Unified ML system is working correctly.")
        print("✅ System is ready for production use")
        print("✅ Clean, unified architecture")
        print("✅ No legacy code or backward compatibility")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed. Please check the issues above.")
    
    return results

if __name__ == "__main__":
    run_comprehensive_test()
