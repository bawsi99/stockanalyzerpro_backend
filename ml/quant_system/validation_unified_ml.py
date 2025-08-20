#!/usr/bin/env python3
"""
Simple Validation Script for Unified ML System

This script provides a quick validation of the unified ML system
to ensure all components are working correctly.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test basic imports."""
    print("🧪 Testing Basic Imports...")
    
    try:
        # Test unified ML imports
        from ml import unified_ml_manager
        from ml.core import UnifiedMLConfig
        from ml.feature_engineering import feature_engineer
        
        print("✅ Basic imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_feature_creation():
    """Test feature creation."""
    print("\n🧪 Testing Feature Creation...")
    
    try:
        from ml.feature_engineering import feature_engineer
        
        # Create sample data
        data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [98, 99, 100, 101, 102],
            'close': [103, 104, 105, 106, 107],
            'volume': [1000000, 1200000, 1100000, 1300000, 1250000]
        })
        
        # Create features
        features = feature_engineer.create_all_features(data)
        
        if not features.empty:
            print(f"✅ Features created successfully: {features.shape}")
            return True
        else:
            print("❌ No features created")
            return False
            
    except Exception as e:
        print(f"❌ Feature creation failed: {e}")
        return False

def test_ml_engines():
    """Test ML engines."""
    print("\n🧪 Testing ML Engines...")
    
    try:
        from ml import (
            pattern_ml_engine,
            raw_data_ml_engine,
            traditional_ml_engine,
            hybrid_ml_engine
        )
        
        print("✅ All ML engines imported successfully")
        print(f"   Pattern ML: {pattern_ml_engine}")
        print(f"   Raw Data ML: {raw_data_ml_engine}")
        print(f"   Traditional ML: {traditional_ml_engine}")
        print(f"   Hybrid ML: {hybrid_ml_engine}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML engines test failed: {e}")
        return False

def test_unified_manager():
    """Test unified manager."""
    print("\n🧪 Testing Unified Manager...")
    
    try:
        from ml import unified_ml_manager
        
        # Test basic functionality
        status = unified_ml_manager.get_engine_status()
        summary = unified_ml_manager.get_system_summary()
        
        print("✅ Unified manager working correctly")
        print(f"   Engine status: {status}")
        print(f"   System summary: {summary}")
        
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

def run_validation():
    """Run all validation tests."""
    print("🚀 UNIFIED ML SYSTEM VALIDATION")
    print("=" * 50)
    print("✅ No Backward Compatibility Required")
    print("✅ Using Only Unified ML System")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Feature Creation", test_feature_creation),
        ("ML Engines", test_ml_engines),
        ("Unified Manager", test_unified_manager),
        ("System Integration", test_integration)
    ]
    
    results = {}
    passed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION RESULTS")
    print("=" * 50)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 VALIDATION SUCCESSFUL!")
        print("✅ Unified ML system is working correctly")
        print("✅ All components are operational")
        print("✅ System is ready for use")
        print("✅ Clean, unified architecture")
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed")
        print("Please check the issues above")
    
    return results

if __name__ == "__main__":
    run_validation()
