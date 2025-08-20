#!/usr/bin/env python3
"""
Simple test for price ML training
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

# Add project root to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from ml.quant_system.ml.raw_data_ml import raw_data_ml_engine, RawDataFeatureEngineer
from zerodha_client import ZerodhaDataClient

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_simple_training():
    """Test simple training with one symbol."""
    print("🧪 TESTING SIMPLE PRICE ML TRAINING")
    print("=" * 50)
    
    # Fetch data
    client = ZerodhaDataClient()
    data = client.get_historical_data('RELIANCE', 'NSE', 'day', period=100)
    
    if data is None or data.empty:
        print("❌ No data fetched")
        return
    
    print(f"📊 Data shape: {data.shape}")
    print(f"📊 Data columns: {list(data.columns)}")
    print(f"📊 Data sample:")
    print(data.head())
    
    # Test feature engineering
    print(f"\n🔧 Testing feature engineering...")
    feature_engineer = RawDataFeatureEngineer()
    
    try:
        features_df = feature_engineer.create_technical_features(data)
        print(f"✅ Feature engineering successful")
        print(f"📊 Features shape: {features_df.shape}")
        print(f"📊 Features columns: {list(features_df.columns)}")
        
        # Check for NaN
        nan_counts = features_df.isnull().sum()
        print(f"📊 NaN counts: {nan_counts[nan_counts > 0]}")
        
        # Drop NaN
        features_clean = features_df.dropna()
        print(f"📊 After dropna(): {features_clean.shape}")
        
        if features_clean.empty:
            print("❌ All data removed by dropna()")
            return
        
        # Test target creation
        print(f"\n🎯 Testing target creation...")
        target_horizon = 1
        features_clean['future_return'] = features_clean['close'].shift(-target_horizon) / features_clean['close'] - 1
        features_clean['future_direction'] = np.where(features_clean['future_return'] > 0, 1, 0)
        
        print(f"✅ Target creation successful")
        print(f"📊 Future return sample: {features_clean['future_return'].head()}")
        print(f"📊 Future direction sample: {features_clean['future_direction'].head()}")
        
        # Test feature selection
        print(f"\n🔍 Testing feature selection...")
        exclude_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'future_return', 'future_direction']
        feature_columns = [col for col in features_clean.columns if col not in exclude_columns]
        
        print(f"✅ Feature selection successful")
        print(f"📊 Feature columns: {feature_columns}")
        print(f"📊 Number of features: {len(feature_columns)}")
        
        if len(feature_columns) == 0:
            print("❌ No feature columns found")
            return
        
        # Test data extraction
        print(f"\n📊 Testing data extraction...")
        X = features_clean[feature_columns].values[:-target_horizon]
        y_direction = features_clean['future_direction'].values[:-target_horizon]
        y_magnitude = features_clean['future_return'].values[:-target_horizon]
        
        print(f"✅ Data extraction successful")
        print(f"📊 X shape: {X.shape}")
        print(f"📊 y_direction shape: {y_direction.shape}")
        print(f"📊 y_magnitude shape: {y_magnitude.shape}")
        
        # Test model training
        print(f"\n🤖 Testing model training...")
        try:
            success = raw_data_ml_engine.train(data, target_horizon=1)
            if success:
                print("✅ Model training successful!")
                print(f"📊 Model trained: {raw_data_ml_engine.is_trained}")
                print(f"📊 Direction model: {raw_data_ml_engine.direction_model is not None}")
                print(f"📊 Magnitude model: {raw_data_ml_engine.magnitude_model is not None}")
            else:
                print("❌ Model training failed")
        except Exception as e:
            print(f"❌ Model training error: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Feature engineering error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_training()
