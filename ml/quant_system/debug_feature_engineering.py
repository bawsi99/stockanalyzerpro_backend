#!/usr/bin/env python3
"""
Debug script for feature engineering
"""

import sys
import os
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_feature_engineering():
    """Debug feature engineering step by step."""
    print("🔍 Debugging Feature Engineering...")
    
    try:
        # Test import
        print("1. Testing import...")
        from ml.feature_engineering import feature_engineer
        print("✅ Import successful")
        
        # Test data creation
        print("\n2. Testing data creation...")
        sample_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [98, 99, 100, 101, 102],
            'close': [103, 104, 105, 106, 107],
            'volume': [1000000, 1200000, 1100000, 1300000, 1250000]
        })
        print(f"✅ Sample data created: {sample_data.shape}")
        print(f"   Columns: {list(sample_data.columns)}")
        
        # Test feature creation
        print("\n3. Testing feature creation...")
        features = feature_engineer.create_all_features(sample_data)
        
        if not features.empty:
            print(f"✅ Features created successfully: {features.shape}")
            print(f"   Feature columns: {list(features.columns)}")
            print(f"   First few rows:")
            print(features.head())
        else:
            print("❌ No features created")
            print("   Sample data info:")
            print(f"   - Empty: {sample_data.empty}")
            print(f"   - Shape: {sample_data.shape}")
            print(f"   - Columns: {list(sample_data.columns)}")
            print(f"   - Data types: {sample_data.dtypes}")
            
            # Check for NaN values
            print(f"   - NaN count: {sample_data.isna().sum().sum()}")
            
            # Check for infinite values
            print(f"   - Infinite count: {np.isinf(sample_data.select_dtypes(include=[np.number])).sum().sum()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_feature_engineering()
