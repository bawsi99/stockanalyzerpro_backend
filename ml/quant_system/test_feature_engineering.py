#!/usr/bin/env python3
"""
Test script for simplified feature engineering module.
Tests that it can import from backend and create features.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add quant_system to path
sys.path.append(os.path.dirname(__file__))

def create_sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.normal(0.001, 0.02, 100)
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, 100)),
        'high': prices * (1 + np.abs(np.random.normal(0.01, 0.01, 100))),
        'low': prices * (1 - np.abs(np.random.normal(0.01, 0.01, 100))),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    return data

def test_feature_engineering():
    """Test the simplified feature engineering module."""
    print("🧪 Testing Simplified Feature Engineering Module")
    print("=" * 50)
    
    try:
        # Import the module
        from ml.feature_engineering import FeatureEngineer, feature_engineer
        
        print("✅ Successfully imported FeatureEngineer")
        
        # Create sample data
        data = create_sample_data()
        print(f"✅ Created sample data: {data.shape}")
        
        # Test feature creation
        print("\n🔧 Testing feature creation...")
        features_df = feature_engineer.create_all_features(data)
        
        if not features_df.empty:
            print(f"✅ Features created successfully: {features_df.shape}")
            print(f"📊 Feature columns: {list(features_df.columns)}")
            
            # Check if key features are present
            expected_features = ['sma_20', 'rsi_14', 'returns', 'volume_ratio']
            missing_features = [f for f in expected_features if f not in features_df.columns]
            
            if missing_features:
                print(f"⚠️  Missing features: {missing_features}")
            else:
                print("✅ All expected features present")
                
            # Show sample features
            print("\n📈 Sample features (last 5 rows):")
            feature_cols = [col for col in features_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            print(features_df[feature_cols].tail())
            
        else:
            print("❌ Feature creation failed - empty DataFrame")
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("💡 Make sure backend modules are accessible")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_backend_availability():
    """Test if backend modules are available."""
    print("\n🔍 Testing Backend Module Availability")
    print("=" * 40)
    
    try:
        # Try to import backend modules
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
        
        try:
            from technical_indicators import TechnicalIndicators
            print("✅ Backend technical_indicators available")
        except ImportError:
            print("❌ Backend technical_indicators not available")
            
        try:
            from patterns.recognition import PatternRecognition
            print("✅ Backend patterns.recognition available")
        except ImportError:
            print("❌ Backend patterns.recognition not available")
            
    except Exception as e:
        print(f"❌ Backend test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting Feature Engineering Tests")
    print("=" * 50)
    
    test_backend_availability()
    test_feature_engineering()
    
    print("\n🏁 Tests completed!")
