#!/usr/bin/env python3
"""
Simplified Validation Script for Step 1.1: Data Pipeline Setup

This script validates the data pipeline implementation by testing:
1. Data validation and quality checks
2. Multi-timeframe data management
3. Error handling and edge cases
4. Core functionality without actual API calls
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

def test_data_validation():
    """Test data validation functionality."""
    print("🔍 Testing Data Validation")
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

def test_missing_columns():
    """Test handling of missing columns."""
    print("\n🔍 Testing Missing Columns")
    print("=" * 50)
    
    # Test with missing columns
    incomplete_data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [98, 99, 100, 101, 102],
        # Missing 'close' and 'volume'
    }, index=pd.date_range('2024-01-01', periods=5, freq='D'))
    
    data_container = OHLCVData("TEST", "day")
    data_container.data = incomplete_data
    
    # Should fail validation
    is_valid = data_container.validate_data()
    
    if not is_valid:
        print("✅ Missing columns correctly detected")
    else:
        print("❌ Missing columns not detected")
        
    return not is_valid

def test_negative_prices():
    """Test handling of negative prices."""
    print("\n🔍 Testing Negative Prices")
    print("=" * 50)
    
    # Test with negative prices
    negative_data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [98, 99, 100, 101, 102],
        'close': [103, 104, -105, 106, 107],  # Negative close price
        'volume': [1000000, 1200000, 1100000, 1300000, 1400000]
    }, index=pd.date_range('2024-01-01', periods=5, freq='D'))
    
    data_container = OHLCVData("TEST", "day")
    data_container.data = negative_data
    
    # Should fail validation
    is_valid = data_container.validate_data()
    
    if not is_valid:
        print("✅ Negative prices correctly detected")
    else:
        print("❌ Negative prices not detected")
        
    return not is_valid

def test_multi_timeframe_manager():
    """Test multi-timeframe data management."""
    print("\n🔍 Testing Multi-Timeframe Data Manager")
    print("=" * 50)
    
    try:
        # Create manager
        timeframes = ["day", "hour"]
        manager = MultiTimeframeDataManager("RELIANCE", timeframes)
        
        # Test manager creation
        if manager.symbol == "RELIANCE" and manager.timeframes == timeframes:
            print("✅ Multi-timeframe manager created successfully")
            print(f"   Symbol: {manager.symbol}")
            print(f"   Timeframes: {manager.timeframes}")
            
            # Test getting data (should be None since no data loaded)
            data = manager.get_timeframe_data("day")
            if data is None:
                print("✅ Empty data correctly handled")
            
            return True
        else:
            print("❌ Multi-timeframe manager creation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error in multi-timeframe manager: {e}")
        return False

def test_data_container_methods():
    """Test data container utility methods."""
    print("\n🔍 Testing Data Container Methods")
    print("=" * 50)
    
    # Create test data
    test_data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [98, 99, 100, 101, 102],
        'close': [103, 104, 105, 106, 107],
        'volume': [1000000, 1200000, 1100000, 1300000, 1400000]
    }, index=pd.date_range('2024-01-01', periods=5, freq='D'))
    
    data_container = OHLCVData("TEST", "day")
    data_container.data = test_data
    
    # Test get_latest_price
    latest_price = data_container.get_latest_price()
    if latest_price == 107:
        print("✅ get_latest_price works correctly")
    else:
        print(f"❌ get_latest_price failed: expected 107, got {latest_price}")
        return False
    
    # Test get_data_range
    start_date, end_date = data_container.get_data_range()
    if start_date == pd.Timestamp('2024-01-01') and end_date == pd.Timestamp('2024-01-05'):
        print("✅ get_data_range works correctly")
    else:
        print(f"❌ get_data_range failed: expected 2024-01-01 to 2024-01-05, got {start_date} to {end_date}")
        return False
    
    return True

def run_comprehensive_validation():
    """Run all validation tests."""
    print("🧪 STEP 1.1 VALIDATION: Data Pipeline Setup (Simplified)")
    print("=" * 60)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = []
    
    # Run all tests
    test_results.append(("Data Validation", test_data_validation()))
    test_results.append(("Invalid Data Handling", test_invalid_data_handling()))
    test_results.append(("Missing Columns", test_missing_columns()))
    test_results.append(("Negative Prices", test_negative_prices()))
    test_results.append(("Multi-Timeframe Manager", test_multi_timeframe_manager()))
    test_results.append(("Data Container Methods", test_data_container_methods()))
    
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
        print("✅ Data pipeline core functionality is working")
        print("✅ Ready to proceed to Step 1.2: Feature Engineering")
    else:
        print("⚠️ Some tests failed. Please review and fix issues.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)
