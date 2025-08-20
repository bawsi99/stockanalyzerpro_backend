#!/usr/bin/env python3
"""
Debug Price ML Model Training

This script helps debug why the price ML model training is failing.
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

def debug_data_processing():
    """Debug the data processing pipeline."""
    print("🔍 DEBUGGING PRICE ML DATA PROCESSING")
    print("=" * 50)
    
    # Fetch sample data
    zerodha_client = ZerodhaDataClient()
    data = zerodha_client.get_historical_data(
        symbol='RELIANCE',
        exchange='NSE',
        interval='day',
        period=100
    )
    
    if data is None or data.empty:
        print("❌ No data fetched")
        return
    
    print(f"📊 Original Data Shape: {data.shape}")
    print(f"📊 Original Data Columns: {list(data.columns)}")
    print(f"📊 Original Data Sample:")
    print(data.head())
    print(f"📊 Original Data Info:")
    print(data.info())
    
    # Check for NaN values in original data
    print(f"\n🔍 NaN Analysis in Original Data:")
    nan_counts = data.isnull().sum()
    print(nan_counts[nan_counts > 0])
    
    # Create feature engineer
    feature_engineer = RawDataFeatureEngineer()
    
    # Create technical features
    print(f"\n🔧 Creating Technical Features...")
    features_df = feature_engineer.create_technical_features(data)
    
    print(f"📊 Features Data Shape: {features_df.shape}")
    print(f"📊 Features Data Columns: {list(features_df.columns)}")
    
    # Check for NaN values in features
    print(f"\n🔍 NaN Analysis in Features Data:")
    nan_counts = features_df.isnull().sum()
    nan_columns = nan_counts[nan_counts > 0]
    print(f"Columns with NaN values: {len(nan_columns)}")
    if len(nan_columns) > 0:
        print(nan_columns.head(10))  # Show first 10 columns with NaN
    
    # Check which columns have the most NaN values
    if len(nan_columns) > 0:
        print(f"\n🔍 Top 10 Columns with Most NaN Values:")
        top_nan = nan_counts.sort_values(ascending=False).head(10)
        for col, count in top_nan.items():
            percentage = (count / len(features_df)) * 100
            print(f"  {col}: {count} NaN values ({percentage:.1f}%)")
    
    # Try dropna() and see what happens
    print(f"\n🧹 Applying dropna()...")
    features_clean = features_df.dropna()
    print(f"📊 After dropna() Shape: {features_clean.shape}")
    
    if features_clean.empty:
        print("❌ All data removed by dropna()!")
        
        # Try to understand why
        print(f"\n🔍 Understanding dropna() behavior:")
        
        # Check if any row has all NaN values
        all_nan_rows = features_df.isnull().all(axis=1).sum()
        print(f"Rows with all NaN values: {all_nan_rows}")
        
        # Check if any column has all NaN values
        all_nan_cols = features_df.isnull().all().sum()
        print(f"Columns with all NaN values: {all_nan_cols}")
        
        # Try dropna() with different strategies
        print(f"\n🔍 Trying different dropna() strategies:")
        
        # Drop rows with all NaN
        features_partial = features_df.dropna(how='all')
        print(f"After dropna(how='all'): {features_partial.shape}")
        
        # Drop columns with all NaN
        features_partial = features_partial.dropna(axis=1, how='all')
        print(f"After dropna(axis=1, how='all'): {features_partial.shape}")
        
        # Drop rows with any NaN in remaining columns
        features_partial = features_partial.dropna()
        print(f"After final dropna(): {features_partial.shape}")
        
        if not features_partial.empty:
            print("✅ Success! Some data remains after selective dropna()")
            print(f"Remaining columns: {list(features_partial.columns)}")
        else:
            print("❌ Still no data after selective dropna()")
            
            # Show sample of problematic data
            print(f"\n🔍 Sample of problematic data (first 5 rows, first 10 columns):")
            sample_data = features_df.iloc[:5, :10]
            print(sample_data)
            
    else:
        print("✅ Data remains after dropna()")
        print(f"Remaining columns: {list(features_clean.columns)}")
        
        # Show sample of clean data
        print(f"\n📊 Sample of Clean Data:")
        print(features_clean.head())

def debug_feature_engineering():
    """Debug the feature engineering process."""
    print(f"\n🔧 DEBUGGING FEATURE ENGINEERING")
    print("=" * 50)
    
    # Create sample data with known structure
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(200, 300, 100),
        'low': np.random.uniform(50, 150, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000000, 5000000, 100)
    })
    sample_data.set_index('date', inplace=True)
    
    print(f"📊 Sample Data Shape: {sample_data.shape}")
    print(f"📊 Sample Data Sample:")
    print(sample_data.head())
    
    # Create feature engineer
    feature_engineer = RawDataFeatureEngineer()
    
    # Create technical features
    print(f"\n🔧 Creating Technical Features...")
    features_df = feature_engineer.create_technical_features(sample_data)
    
    print(f"📊 Features Data Shape: {features_df.shape}")
    print(f"📊 Features Data Columns: {list(features_df.columns)}")
    
    # Check for NaN values
    print(f"\n🔍 NaN Analysis:")
    nan_counts = features_df.isnull().sum()
    nan_columns = nan_counts[nan_counts > 0]
    print(f"Columns with NaN values: {len(nan_columns)}")
    if len(nan_columns) > 0:
        print(nan_columns.head(10))
    
    # Try dropna()
    features_clean = features_df.dropna()
    print(f"📊 After dropna() Shape: {features_clean.shape}")
    
    if not features_clean.empty:
        print("✅ Feature engineering works with sample data")
        print(f"Remaining columns: {list(features_clean.columns)}")
    else:
        print("❌ Feature engineering fails even with sample data")

def main():
    """Main debug function."""
    print("🚀 PRICE ML MODEL DEBUG")
    print("=" * 60)
    
    # Debug with real data
    debug_data_processing()
    
    # Debug with sample data
    debug_feature_engineering()
    
    print(f"\n📋 DEBUG SUMMARY")
    print("=" * 40)
    print("Debug completed. Check output above for issues.")

if __name__ == "__main__":
    main()
