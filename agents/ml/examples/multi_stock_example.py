#!/usr/bin/env python3
"""
Example: Multi-Stock ML Data Processing Pipeline

This example demonstrates how to use the MultiStockProcessor to:
1. Process data from multiple stocks uniformly
2. Create consistent features and labels across all symbols
3. Generate unified training/validation/test datasets
4. Train models on the consolidated multi-stock dataset

Usage:
    python multi_stock_example.py

Prerequisites:
    1. Raw data should be available in the ML data directory
    2. Run data extraction first if needed:
       cd backend/agents/ml/data_processing
       python data_extractor.py

This will consolidate data from all configured symbols and timeframes,
apply uniform feature engineering, create labels, and prepare datasets
ready for multi-stock model training.
"""

import os
import sys
from pathlib import Path

# Add project root to path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
sys.path.insert(0, project_root)

from backend.agents.ml.data_processing.multi_stock_processor import MultiStockProcessor
from backend.agents.ml.training.train_multi_models import main as train_models
from backend.agents.ml.config.config import ml_defaults, DEFAULT_TIMEFRAMES


def run_multi_stock_pipeline_example():
    """
    Complete example of multi-stock ML pipeline from raw data to trained models.
    """
    
    print("=" * 80)
    print("MULTI-STOCK ML PIPELINE EXAMPLE")
    print("=" * 80)
    
    # Step 1: Configure symbols and timeframes
    # You can customize these or use defaults from config
    symbols = [
        "RELIANCE", 
        "TCS", 
        "INFY", 
        "HDFCBANK",
        "ICICIBANK"
    ]
    
    timeframes = ["5m", "15m", "1h", "1d"]
    
    print(f"Processing {len(symbols)} symbols across {len(timeframes)} timeframes:")
    print(f"  Symbols: {symbols}")
    print(f"  Timeframes: {timeframes}")
    
    # Step 2: Initialize the processor
    processor = MultiStockProcessor(
        symbols=symbols,
        timeframes=timeframes,
        # base_dir will default to ml_defaults["base_dir"]
        # output_dir will be created automatically
    )
    
    # Step 3: Configure custom labeling strategy (optional)
    # Different timeframes may require different prediction horizons and thresholds
    custom_label_config = {
        "5m": {
            "horizon": 12,      # Look ahead 12 periods (1 hour for 5min bars)
            "threshold": 0.015, # 1.5% move threshold
            "method": "fixed_threshold"
        },
        "15m": {
            "horizon": 8,       # Look ahead 8 periods (2 hours for 15min bars)
            "threshold": 0.02,  # 2% move threshold
            "method": "fixed_threshold"
        },
        "1h": {
            "horizon": 12,      # Look ahead 12 periods (12 hours)
            "threshold": 0.025, # 2.5% move threshold
            "method": "fixed_threshold"
        },
        "1d": {
            "horizon": 5,       # Look ahead 5 periods (5 days)
            "threshold": 0.04,  # 4% move threshold
            "method": "fixed_threshold"
        }
    }
    
    # Step 4: Run the complete processing pipeline
    try:
        file_paths = processor.run_full_pipeline(
            label_config=custom_label_config,
            train_pct=0.7,      # 70% for training
            val_pct=0.15,       # 15% for validation
            test_pct=0.15,      # 15% for testing
            suffix="_example"   # Add suffix to output directory
        )
        
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        print(f"\nGenerated files:")
        for name, path in file_paths.items():
            print(f"  {name}: {path}")
        
        # Step 5: Optional - Train models on the processed data
        print("\n" + "=" * 40)
        print("OPTIONAL: Train models on processed data")
        print("=" * 40)
        
        # Get the directory containing the processed splits
        splits_dir = os.path.dirname(file_paths["train"])
        
        print(f"To train models on this processed dataset, run:")
        print(f"  cd backend/agents/ml")
        print(f"  python training/train_multi_models.py --splits_dir {splits_dir}")
        
        # Or train automatically (uncomment to enable)
        # print("Training models automatically...")
        # import subprocess
        # subprocess.run([
        #     sys.executable, "-m", "backend.agents.ml.training.train_multi_models",
        #     "--splits_dir", splits_dir,
        #     "--models", "logistic", "random_forest", "xgboost"
        # ], cwd=project_root)
        
        return file_paths
        
    except Exception as e:
        print(f"\nPIPELINE FAILED: {e}")
        print("\nPossible issues:")
        print("1. No raw data available - run data extraction first")
        print("2. Insufficient data for some symbols/timeframes")
        print("3. Missing dependencies for feature calculations")
        raise


def check_data_availability():
    """
    Check if raw data is available for processing.
    """
    
    print("Checking data availability...")
    
    base_dir = ml_defaults["base_dir"]
    print(f"Base data directory: {base_dir}")
    
    if not os.path.exists(base_dir):
        print("❌ Raw data directory does not exist!")
        print(f"Expected: {base_dir}")
        print("\nTo create raw data, run:")
        print("  cd backend/agents/ml/data_processing")
        print("  python data_extractor.py")
        return False
    
    # Check for some expected symbol/timeframe combinations
    found_data = False
    for symbol in ["RELIANCE", "TCS", "INFY"]:
        for tf in ["5m", "15m", "1h", "1d"]:
            symbol_dir = os.path.join(base_dir, f"symbol={symbol}")
            tf_dir = os.path.join(symbol_dir, f"timeframe={tf}")
            
            parquet_file = os.path.join(tf_dir, "bars.parquet")
            csv_file = os.path.join(tf_dir, "bars.csv")
            
            if os.path.exists(parquet_file) or os.path.exists(csv_file):
                print(f"✓ Found data: {symbol} {tf}")
                found_data = True
            else:
                print(f"✗ Missing data: {symbol} {tf}")
    
    if found_data:
        print("\n✅ Some raw data is available for processing")
        return True
    else:
        print("\n❌ No raw data found!")
        print("\nTo extract raw data, run:")
        print("  cd backend/agents/ml/data_processing")
        print("  python data_extractor.py")
        return False


if __name__ == "__main__":
    print("Multi-Stock ML Pipeline Example")
    print("=" * 40)
    
    # Check if we have data to work with
    if not check_data_availability():
        print("\nSkipping pipeline - no raw data available")
        print("Please extract data first using data_extractor.py")
        sys.exit(1)
    
    # Run the example pipeline
    try:
        file_paths = run_multi_stock_pipeline_example()
        
        print("\n" + "🎉" * 20)
        print("EXAMPLE COMPLETED SUCCESSFULLY!")
        print("🎉" * 20)
        
        print(f"\nNext steps:")
        print(f"1. Review the generated datasets in the output directory")
        print(f"2. Train ML models using train_multi_models.py")
        print(f"3. Analyze model performance and feature importance")
        print(f"4. Deploy best-performing models for live trading")
        
    except Exception as e:
        print(f"\nExample failed: {e}")
        print("Check the error messages above for troubleshooting guidance")
        sys.exit(1)