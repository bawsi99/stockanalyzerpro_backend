#!/usr/bin/env python3
"""
Validation Script for Step 1.1: Data Pipeline Setup

This script validates the data pipeline implementation by testing:
1. Data loading from Zerodha API
2. Data validation and quality checks
3. Multi-timeframe data management
4. Error handling and edge cases
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add the quant_system directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data_pipeline import OHLCVData, MultiTimeframeDataManager, DataConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_loading():
    """Test basic data loading functionality."""
    print("🔍 Testing Data Loading Functionality")
    print("=" * 50)
    
    # Test parameters
    symbol = "RELIANCE"
    timeframe = "day"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    try:
        # Create data container
        data_container = OHLCVData(symbol, timeframe)
        
        # Load data
        data_container.load(start_date, end_date)
        
        # Check if data was loaded
        if not data_container.data.empty:
            print(f"✅ Data loaded successfully")
            print(f"   Symbol: {symbol}")
            print(f"   Timeframe: {timeframe}")
            print(f"   Data points: {len(data_container.data)}")
            print(f"   Date range: {data_container.data.index[0]} to {data_container.data.index[-1]}")
            print(f"   Columns: {list(data_container.data.columns)}")
            return True
        else:
            print(f"❌ No data loaded for {symbol}")
            return False
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_data_validation():
    """Test data validation functionality."""
    print("\n🔍 Testing Data Validation")
    print("=" * 50)
    
    # Create test data
    test_data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [98, 99, 100, 101, 102],
        'close': [103, 104, 105, 106, 107],
        'volume': [1000000, 1200000, 1100000, 1300000, 1400000]
    }, index=pd.date_range('2024-01-01', periods=5, freq='D'))
    
    # Create data container with test data
    data_container = OHLCVData("TEST", "day")
    data_container.data = test_data
    
    # Test validation
    is_valid = data_container.validate_data()
    
    if is_valid:
        print("✅ Data validation passed")
        print(f"   Data shape: {data_container.data.shape}")
        print(f"   Latest price: {data_container.get_latest_price()}")
        print(f"   Date range: {data_container.get_data_range()}")
    else:
        print("❌ Data validation failed")
        
    return is_valid

def test_invalid_data_handling():
    """Test handling of invalid data."""
    print("\n🔍 Testing Invalid Data Handling")
    print("=" * 50)
    
    # Test with invalid price logic
    invalid_data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [95, 96, 97, 98, 99],  # High < Open (invalid)
        'low': [98, 99, 100, 101, 102],
        'close': [103, 104, 105, 106, 107],
        'volume': [1000000, 1200000, 1100000, 1300000, 1400000]
    }, index=pd.date_range('2024-01-01', periods=5, freq='D'))
    
    data_container = OHLCVData("TEST", "day")
    data_container.data = invalid_data
    
    # Should fail validation
    is_valid = data_container.validate_data()
    
    if not is_valid:
        print("✅ Invalid data correctly rejected")
    else:
        print("❌ Invalid data incorrectly accepted")
        
    return not is_valid

def test_multi_timeframe_manager():
    """Test multi-timeframe data management."""
    print("\n🔍 Testing Multi-Timeframe Data Manager")
    print("=" * 50)
    
    try:
        # Create manager
        timeframes = ["day", "hour"]
        manager = MultiTimeframeDataManager("RELIANCE", timeframes)
        
        # Load data for all timeframes
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Shorter period for testing
        
        data_containers = manager.load_all_timeframes(start_date, end_date)
        
        if data_containers:
            print("✅ Multi-timeframe data loaded")
            for tf, container in data_containers.items():
                print(f"   {tf}: {len(container.data)} data points")
                
            # Test getting latest prices
            latest_prices = manager.get_latest_prices()
            print(f"   Latest prices: {latest_prices}")
            
            return True
        else:
            print("❌ No data containers loaded")
            return False
            
    except Exception as e:
        print(f"❌ Error in multi-timeframe manager: {e}")
        return False

def test_error_handling():
    """Test error handling for edge cases."""
    print("\n🔍 Testing Error Handling")
    print("=" * 50)
    
    # Test with invalid symbol
    try:
        data_container = OHLCVData("INVALID_SYMBOL_123", "day")
        data_container.load(datetime.now() - timedelta(days=30), datetime.now())
        
        if data_container.data.empty:
            print("✅ Invalid symbol handled gracefully")
        else:
            print("❌ Invalid symbol should have returned empty data")
            
    except Exception as e:
        print(f"✅ Error handling working: {e}")
        
    # Test with invalid date range
    try:
        data_container = OHLCVData("RELIANCE", "day")
        data_container.load(datetime.now(), datetime.now() - timedelta(days=30))  # End before start
        
        if data_container.data.empty:
            print("✅ Invalid date range handled gracefully")
        else:
            print("❌ Invalid date range should have returned empty data")
            
    except Exception as e:
        print(f"✅ Error handling working: {e}")

def run_comprehensive_validation():
    """Run all validation tests."""
    print("🧪 STEP 1.1 VALIDATION: Data Pipeline Setup")
    print("=" * 60)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = []
    
    # Run all tests
    test_results.append(("Data Loading", test_data_loading()))
    test_results.append(("Data Validation", test_data_validation()))
    test_results.append(("Invalid Data Handling", test_invalid_data_handling()))
    test_results.append(("Multi-Timeframe Manager", test_multi_timeframe_manager()))
    test_error_handling()  # This test doesn't return boolean
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 STEP 1.1 VALIDATION COMPLETED SUCCESSFULLY!")
        print("✅ Data pipeline is ready for feature engineering")
    else:
        print("⚠️ Some tests failed. Please review and fix issues.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)
