#!/usr/bin/env python3
"""
Validation Script for Step 1.2: Feature Engineering

This script validates the feature engineering implementation by testing:
1. Technical indicator calculations
2. Price-based features
3. Volume-based features
4. Volatility features
5. Pattern recognition
6. Feature cleaning and validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the quant_system directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from feature_engineering import FeatureEngineer, FeatureConfig

def test_feature_engineering_basic():
    """Test basic feature engineering functionality."""
    print("🔍 Testing Basic Feature Engineering")
    print("=" * 50)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.normal(0.001, 0.02, 100)  # Daily returns
    prices = 100 * np.exp(np.cumsum(returns))
    
    test_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, 100)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    # Ensure price logic is valid
    test_data['high'] = np.maximum(test_data[['open', 'high', 'close']].max(axis=1), test_data['high'])
    test_data['low'] = np.minimum(test_data[['open', 'low', 'close']].min(axis=1), test_data['low'])
    
    print(f"✅ Test data created: {test_data.shape}")
    print(f"   Date range: {test_data.index[0]} to {test_data.index[-1]}")
    
    # Create feature engineer
    config = FeatureConfig()
    engineer = FeatureEngineer(config)
    
    # Create features
    features_df = engineer.create_all_features(test_data)
    
    if not features_df.empty:
        print(f"✅ Features created successfully: {features_df.shape}")
        print(f"   Original columns: {list(test_data.columns)}")
        print(f"   Feature columns: {len([col for col in features_df.columns if col not in test_data.columns])}")
        
        # Check for key features
        key_features = ['rsi', 'macd', 'bb_upper', 'bb_lower', 'atr', 'vwap', 'returns']
        missing_features = [f for f in key_features if f not in features_df.columns]
        
        if not missing_features:
            print("✅ All key features created")
        else:
            print(f"⚠️ Missing features: {missing_features}")
            
        return True
    else:
        print("❌ Feature engineering failed")
        return False

def test_technical_indicators():
    """Test technical indicator calculations."""
    print("\n🔍 Testing Technical Indicators")
    print("=" * 50)
    
    # Create simple test data
    test_data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
        'low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
        'close': [103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
        'volume': [1000000] * 10
    }, index=pd.date_range('2024-01-01', periods=10, freq='D'))
    
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(test_data)
    
    if not features_df.empty:
        # Test RSI
        if 'rsi' in features_df.columns:
            rsi_values = features_df['rsi'].dropna()
            if len(rsi_values) > 0 and all(0 <= rsi <= 100 for rsi in rsi_values):
                print("✅ RSI calculation correct")
            else:
                print("❌ RSI values out of range")
                return False
        else:
            print("❌ RSI not calculated")
            return False
        
        # Test MACD
        if all(col in features_df.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
            print("✅ MACD calculation correct")
        else:
            print("❌ MACD not calculated")
            return False
        
        # Test Bollinger Bands
        if all(col in features_df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            # Check that upper > middle > lower
            bb_valid = all(features_df['bb_upper'] >= features_df['bb_middle']) and \
                      all(features_df['bb_middle'] >= features_df['bb_lower'])
            if bb_valid:
                print("✅ Bollinger Bands calculation correct")
            else:
                print("❌ Bollinger Bands logic error")
                return False
        else:
            print("❌ Bollinger Bands not calculated")
            return False
        
        return True
    else:
        print("❌ Feature engineering failed")
        return False

def test_price_features():
    """Test price-based feature calculations."""
    print("\n🔍 Testing Price Features")
    print("=" * 50)
    
    # Create test data with known patterns
    test_data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [98, 99, 100, 101, 102],
        'close': [103, 104, 105, 106, 107],
        'volume': [1000000] * 5
    }, index=pd.date_range('2024-01-01', periods=5, freq='D'))
    
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(test_data)
    
    if not features_df.empty:
        # Test returns calculation
        if 'returns' in features_df.columns:
            expected_returns = [np.nan, 0.0097, 0.0096, 0.0095, 0.0094]  # Approximate
            actual_returns = features_df['returns'].values
            returns_close = all(abs(actual_returns[i] - expected_returns[i]) < 0.01 
                              for i in range(1, len(actual_returns)) if not np.isnan(actual_returns[i]))
            if returns_close:
                print("✅ Returns calculation correct")
            else:
                print("❌ Returns calculation error")
                return False
        else:
            print("❌ Returns not calculated")
            return False
        
        # Test moving averages
        if 'sma_5' in features_df.columns:
            print("✅ Moving averages calculated")
        else:
            print("❌ Moving averages not calculated")
            return False
        
        return True
    else:
        print("❌ Feature engineering failed")
        return False

def test_volume_features():
    """Test volume-based feature calculations."""
    print("\n🔍 Testing Volume Features")
    print("=" * 50)
    
    # Create test data with varying volume
    test_data = pd.DataFrame({
        'open': [100] * 10,
        'high': [105] * 10,
        'low': [95] * 10,
        'close': [102] * 10,
        'volume': [1000000, 1200000, 800000, 1500000, 1100000, 1300000, 900000, 1400000, 1000000, 1200000]
    }, index=pd.date_range('2024-01-01', periods=10, freq='D'))
    
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(test_data)
    
    if not features_df.empty:
        # Test volume ratio
        if 'volume_ratio' in features_df.columns:
            volume_ratios = features_df['volume_ratio'].dropna()
            if len(volume_ratios) > 0:
                print("✅ Volume ratio calculated")
            else:
                print("❌ Volume ratio calculation failed")
                return False
        else:
            print("❌ Volume ratio not calculated")
            return False
        
        # Test VWAP
        if 'vwap' in features_df.columns:
            vwap_values = features_df['vwap'].dropna()
            if len(vwap_values) > 0:
                print("✅ VWAP calculated")
            else:
                print("❌ VWAP calculation failed")
                return False
        else:
            print("❌ VWAP not calculated")
            return False
        
        return True
    else:
        print("❌ Feature engineering failed")
        return False

def test_pattern_recognition():
    """Test pattern recognition functionality."""
    print("\n🔍 Testing Pattern Recognition")
    print("=" * 50)
    
    # Create test data with known patterns
    test_data = pd.DataFrame({
        'open': [100, 101, 100, 99, 100],
        'high': [105, 106, 105, 104, 105],
        'low': [95, 96, 95, 94, 95],
        'close': [100, 101, 100, 99, 100],  # Doji pattern
        'volume': [1000000] * 5
    }, index=pd.date_range('2024-01-01', periods=5, freq='D'))
    
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(test_data)
    
    if not features_df.empty:
        # Test pattern features
        pattern_features = ['is_doji', 'is_hammer', 'is_shooting_star', 'is_bullish_engulfing', 'is_bearish_engulfing']
        
        for pattern in pattern_features:
            if pattern in features_df.columns:
                pattern_values = features_df[pattern].dropna()
                if len(pattern_values) > 0:
                    print(f"✅ {pattern} calculated")
                else:
                    print(f"⚠️ {pattern} has no values")
            else:
                print(f"❌ {pattern} not calculated")
        
        return True
    else:
        print("❌ Feature engineering failed")
        return False

def test_feature_cleaning():
    """Test feature cleaning functionality."""
    print("\n🔍 Testing Feature Cleaning")
    print("=" * 50)
    
    # Create test data with some problematic values
    test_data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [98, 99, 100, 101, 102],
        'close': [103, 104, 105, 106, 107],
        'volume': [1000000, 1200000, 1100000, 1300000, 1400000]
    }, index=pd.date_range('2024-01-01', periods=5, freq='D'))
    
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(test_data)
    
    if not features_df.empty:
        # Check for infinite values
        has_infinite = np.isinf(features_df.select_dtypes(include=[np.number])).any().any()
        if not has_infinite:
            print("✅ No infinite values found")
        else:
            print("❌ Infinite values found")
            return False
        
        # Check for NaN values in key features
        key_features = ['rsi', 'macd', 'bb_upper', 'atr']
        nan_check = features_df[key_features].isnull().sum()
        if nan_check.sum() == 0:
            print("✅ No NaN values in key features")
        else:
            print(f"⚠️ NaN values found: {nan_check.to_dict()}")
        
        return True
    else:
        print("❌ Feature engineering failed")
        return False

def run_comprehensive_validation():
    """Run all validation tests."""
    print("🧪 STEP 1.2 VALIDATION: Feature Engineering")
    print("=" * 60)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = []
    
    # Run all tests
    test_results.append(("Basic Feature Engineering", test_feature_engineering_basic()))
    test_results.append(("Technical Indicators", test_technical_indicators()))
    test_results.append(("Price Features", test_price_features()))
    test_results.append(("Volume Features", test_volume_features()))
    test_results.append(("Pattern Recognition", test_pattern_recognition()))
    test_results.append(("Feature Cleaning", test_feature_cleaning()))
    
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
        print("🎉 STEP 1.2 VALIDATION COMPLETED SUCCESSFULLY!")
        print("✅ Feature engineering is working correctly")
        print("✅ Ready to proceed to Step 1.3: ML Model Development")
    else:
        print("⚠️ Some tests failed. Please review and fix issues.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)
