#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from pathlib import Path

# Import the build_dataset functions directly without circular dependencies
script_dir = os.path.dirname(__file__)
sys.path.insert(0, script_dir)

# Import build_dataset functions
from build_dataset import add_features, read_raw_csv

# Define constants locally to avoid import issues
DEFAULT_UNIVERSE = [
    "RELIANCE",
    "TCS", 
    "INFY",
    "HDFCBANK",
    "ICICIBANK",
    "ITC",
    "SBIN",
    "BAJFINANCE",
    "BHARTIARTL",
    "HINDUNILVR"
]

DEFAULT_TIMEFRAMES = {
    "5m": {"horizon_bars": 12, "est_cost_bps": 8.0},
    "15m": {"horizon_bars": 8, "est_cost_bps": 7.0},
    "1h": {"horizon_bars": 12, "est_cost_bps": 6.0},
    "1d": {"horizon_bars": 5, "est_cost_bps": 5.0},
}

# ML defaults
ml_defaults = {
    "universe": DEFAULT_UNIVERSE,
    "timeframes": DEFAULT_TIMEFRAMES,
    "base_dir": os.path.join(os.path.dirname(__file__), "data", "raw"),
    "exchange": "NSE",
}

# Inline implementations for missing functions
class LabelConfig:
    """Configuration for label creation"""
    def __init__(self, horizon: int, threshold: float, method: str = "fixed_threshold"):
        self.horizon = horizon
        self.threshold = threshold
        self.method = method

def create_labels(df: pd.DataFrame, config: LabelConfig) -> pd.DataFrame:
    """Create forward-looking labels based on returns"""
    out = df.copy()
    
    # Calculate forward returns
    close = out["close"].astype(float)
    fwd_price = close.shift(-config.horizon)
    log_ret = np.log(fwd_price / close)
    
    # Create binary and regression labels
    out["y_reg"] = log_ret
    out["y_cls"] = (log_ret > config.threshold).astype(int)
    
    # Remove rows where we can't calculate forward returns
    if config.horizon > 0:
        out = out.iloc[:-config.horizon]
    
    return out

def compute_scaler(df: pd.DataFrame, exclude_cols: set) -> Dict:
    """Compute standardization parameters"""
    feature_cols = []
    means = {}
    stds = {}
    binary_cols = []
    
    for col in df.columns:
        if col in exclude_cols:
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            continue
            
        # Check if binary
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            binary_cols.append(col)
            continue
        
        # Compute stats
        values = pd.to_numeric(df[col], errors='coerce')
        mean_val = float(np.nanmean(values))
        std_val = float(np.nanstd(values, ddof=0))
        
        if not np.isfinite(std_val) or std_val == 0.0:
            std_val = 1.0
            
        means[col] = mean_val
        stds[col] = std_val
        feature_cols.append(col)
    
    return {
        "feature_cols": feature_cols,
        "means": means,
        "stds": stds,
        "excluded_columns": list(exclude_cols),
        "binary_excluded": binary_cols
    }

def apply_scaler(df: pd.DataFrame, scaler: Dict) -> pd.DataFrame:
    """Apply standardization using computed scaler"""
    out = df.copy()
    
    for col in scaler["feature_cols"]:
        if col in out.columns:
            values = pd.to_numeric(out[col], errors='coerce')
            mean_val = scaler["means"].get(col, 0.0)
            std_val = scaler["stds"].get(col, 1.0) or 1.0
            out[col] = (values - mean_val) / std_val
    
    return out

def apply_qc_checks(
    df: pd.DataFrame, 
    max_feature_na_pct: float = 0.1, 
    min_periods_per_group: int = 100, 
    outlier_method: str = "iqr",
    outlier_factor: float = 3.0
) -> pd.DataFrame:
    """Apply basic quality control checks"""
    out = df.copy()
    initial_len = len(out)
    
    # Remove rows with too many NaN features
    feature_cols = [col for col in out.columns 
                   if col not in ['symbol', 'timeframe', 'seq_id', 'y_cls', 'y_reg',
                                'open', 'high', 'low', 'close', 'volume']]
    
    if feature_cols:
        na_pct = out[feature_cols].isna().sum(axis=1) / len(feature_cols)
        out = out[na_pct <= max_feature_na_pct]
        
    # Remove groups with insufficient data
    if 'symbol' in out.columns and 'timeframe' in out.columns:
        group_counts = out.groupby(['symbol', 'timeframe']).size()
        valid_groups = group_counts[group_counts >= min_periods_per_group].index
        
        if len(valid_groups) < len(group_counts):
            mask = out.set_index(['symbol', 'timeframe']).index.isin(valid_groups)
            out = out[mask]
    
    # Basic outlier removal using IQR
    if outlier_method == "iqr" and feature_cols:
        for col in feature_cols:
            if out[col].dtype in ['int64', 'float64']:
                Q1 = out[col].quantile(0.25)
                Q3 = out[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - outlier_factor * IQR
                upper = Q3 + outlier_factor * IQR
                out = out[(out[col] >= lower) & (out[col] <= upper)]
    
    removed_count = initial_len - len(out)
    if removed_count > 0:
        print(f"    QC removed {removed_count} rows ({removed_count/initial_len:.1%})")
    
    return out


class MultiStockProcessor:
    """
    Unified multi-stock data processing pipeline that:
    1. Consolidates raw data from multiple symbols/timeframes
    2. Applies uniform feature engineering across all stocks
    3. Creates consistent labels using configurable strategies
    4. Performs standardization across the entire dataset
    5. Applies quality control checks uniformly
    6. Splits data while maintaining temporal order within each symbol
    """
    
    def __init__(
        self, 
        base_dir: str = None, 
        output_dir: str = None,
        symbols: List[str] = None,
        timeframes: List[str] = None
    ):
        self.base_dir = base_dir or ml_defaults["base_dir"]
        self.output_dir = output_dir or os.path.join(os.path.dirname(self.base_dir), "processed", "multi_stock")
        self.symbols = symbols or ml_defaults["universe"]
        self.timeframes = timeframes or list(DEFAULT_TIMEFRAMES.keys())
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"MultiStockProcessor initialized:")
        print(f"  Base dir: {self.base_dir}")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Symbols: {self.symbols}")
        print(f"  Timeframes: {self.timeframes}")
    
    def discover_available_data(self) -> Dict[str, Dict[str, str]]:
        """
        Discover available data files for each symbol/timeframe combination.
        Returns: Dict[symbol][timeframe] -> file_path
        """
        available_data = {}
        
        for symbol in self.symbols:
            available_data[symbol] = {}
            for tf in self.timeframes:
                # Look for data in the expected raw data structure
                symbol_dir = os.path.join(self.base_dir, f"symbol={symbol}")
                tf_dir = os.path.join(symbol_dir, f"timeframe={tf}")
                
                # Check for parquet first, then CSV
                parquet_file = os.path.join(tf_dir, "bars.parquet")
                csv_file = os.path.join(tf_dir, "bars.csv")
                
                if os.path.exists(parquet_file):
                    available_data[symbol][tf] = parquet_file
                elif os.path.exists(csv_file):
                    available_data[symbol][tf] = csv_file
                else:
                    print(f"WARNING: No data found for {symbol} {tf} in {tf_dir}")
        
        return available_data
    
    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """Load raw data from parquet or CSV with consistent formatting."""
        try:
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
                    df = df.set_index(pd.to_datetime(df.iloc[:, 0]))
            else:
                df = read_raw_csv(file_path)
            
            # Ensure consistent datetime index
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert("UTC").tz_localize(None)
            
            return df.sort_index()
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return pd.DataFrame()
    
    def consolidate_raw_data(self) -> pd.DataFrame:
        """
        Consolidate raw OHLCV data from all symbols/timeframes into a single DataFrame.
        Adds 'symbol' and 'timeframe' columns for identification.
        """
        print("Consolidating raw data across all symbols and timeframes...")
        
        available_data = self.discover_available_data()
        consolidated_dfs = []
        
        total_combinations = sum(len(tfs) for tfs in available_data.values())
        processed = 0
        
        for symbol, timeframe_files in available_data.items():
            for timeframe, file_path in timeframe_files.items():
                processed += 1
                print(f"  Processing [{processed}/{total_combinations}] {symbol} {timeframe}...")
                
                df = self.load_raw_data(file_path)
                if df.empty:
                    continue
                
                # Add identifier columns
                df['symbol'] = symbol
                df['timeframe'] = timeframe
                
                # Add sequence info within each symbol-timeframe for later splitting
                df['seq_id'] = range(len(df))
                
                consolidated_dfs.append(df)
        
        if not consolidated_dfs:
            raise ValueError("No valid data found for any symbol/timeframe combination")
        
        # Combine all data
        consolidated = pd.concat(consolidated_dfs, ignore_index=False)
        consolidated = consolidated.sort_values(['symbol', 'timeframe', consolidated.index.name or 'timestamp'])
        
        print(f"Consolidated {len(consolidated)} rows from {len(consolidated_dfs)} symbol/timeframe combinations")
        print(f"Date range: {consolidated.index.min()} to {consolidated.index.max()}")
        print(f"Symbols: {sorted(consolidated['symbol'].unique())}")
        print(f"Timeframes: {sorted(consolidated['timeframe'].unique())}")
        
        return consolidated
    
    def add_features_unified(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering uniformly across all symbol/timeframe combinations.
        Processes each group separately to maintain indicator continuity.
        """
        print("Adding features across all symbol/timeframe combinations...")
        
        featured_dfs = []
        groups = df.groupby(['symbol', 'timeframe'])
        total_groups = len(groups)
        
        for i, ((symbol, timeframe), group) in enumerate(groups, 1):
            print(f"  Adding features [{i}/{total_groups}] {symbol} {timeframe} ({len(group)} bars)...")
            
            # Sort by timestamp within group to ensure proper indicator calculation
            group_sorted = group.sort_index()
            
            # Apply feature engineering to this symbol/timeframe group
            try:
                featured_group = add_features(group_sorted)
                featured_dfs.append(featured_group)
            except Exception as e:
                print(f"    ERROR adding features for {symbol} {timeframe}: {e}")
                continue
        
        if not featured_dfs:
            raise ValueError("No features could be computed for any symbol/timeframe")
        
        # Combine all featured data
        featured_df = pd.concat(featured_dfs)
        
        print(f"Features added successfully. Final shape: {featured_df.shape}")
        print(f"Feature columns: {[col for col in featured_df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'symbol', 'timeframe', 'seq_id']]}")
        
        return featured_df
    
    def create_labels_unified(
        self, 
        df: pd.DataFrame, 
        label_config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Create labels uniformly across all symbol/timeframe combinations.
        Uses forward-looking returns within each symbol/timeframe group.
        """
        print("Creating labels across all symbol/timeframe combinations...")
        
        if label_config is None:
            # Default label configuration - can be customized per timeframe
            label_config = {
                "5m": {"horizon": 12, "threshold": 0.02, "method": "fixed_threshold"},
                "15m": {"horizon": 8, "threshold": 0.025, "method": "fixed_threshold"}, 
                "1h": {"horizon": 12, "threshold": 0.03, "method": "fixed_threshold"},
                "1d": {"horizon": 5, "threshold": 0.05, "method": "fixed_threshold"},
            }
        
        labeled_dfs = []
        groups = df.groupby(['symbol', 'timeframe'])
        total_groups = len(groups)
        
        for i, ((symbol, timeframe), group) in enumerate(groups, 1):
            print(f"  Creating labels [{i}/{total_groups}] {symbol} {timeframe}...")
            
            # Get label configuration for this timeframe
            tf_config = label_config.get(timeframe, {
                "horizon": 5, 
                "threshold": 0.03, 
                "method": "fixed_threshold"
            })
            
            # Sort by timestamp within group
            group_sorted = group.sort_index()
            
            try:
                # Create LabelConfig object
                config = LabelConfig(
                    horizon=tf_config["horizon"],
                    threshold=tf_config["threshold"],
                    method=tf_config["method"]
                )
                
                # Create labels for this group
                labeled_group = create_labels(group_sorted, config)
                labeled_dfs.append(labeled_group)
                
                # Report label statistics for this group
                if 'y_cls' in labeled_group.columns:
                    pos_rate = labeled_group['y_cls'].mean()
                    print(f"    {symbol} {timeframe}: {pos_rate:.3f} positive rate ({pos_rate*len(labeled_group):.0f}/{len(labeled_group)} samples)")
                
            except Exception as e:
                print(f"    ERROR creating labels for {symbol} {timeframe}: {e}")
                continue
        
        if not labeled_dfs:
            raise ValueError("No labels could be created for any symbol/timeframe")
        
        labeled_df = pd.concat(labeled_dfs)
        
        # Overall label statistics
        if 'y_cls' in labeled_df.columns:
            overall_pos_rate = labeled_df['y_cls'].mean()
            print(f"Overall positive label rate: {overall_pos_rate:.3f} ({overall_pos_rate*len(labeled_df):.0f}/{len(labeled_df)} samples)")
        
        return labeled_df
    
    def apply_quality_control(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply uniform quality control checks across all data."""
        print("Applying quality control checks...")
        
        initial_shape = df.shape
        
        # Apply QC checks
        try:
            qc_df = apply_qc_checks(
                df, 
                max_feature_na_pct=0.1,  # Allow up to 10% missing features
                min_periods_per_group=100,  # Minimum samples per symbol/timeframe
                outlier_method="iqr",
                outlier_factor=3.0
            )
            
            print(f"QC applied: {initial_shape[0]} -> {qc_df.shape[0]} rows ({initial_shape[0] - qc_df.shape[0]} removed)")
            
            # Report QC statistics per group
            if 'symbol' in qc_df.columns and 'timeframe' in qc_df.columns:
                group_counts = qc_df.groupby(['symbol', 'timeframe']).size()
                print("Post-QC sample counts by group:")
                for (symbol, tf), count in group_counts.items():
                    print(f"  {symbol} {tf}: {count}")
            
            return qc_df
            
        except Exception as e:
            print(f"WARNING: QC checks failed, using original data: {e}")
            return df
    
    def create_train_val_test_splits(
        self, 
        df: pd.DataFrame, 
        train_pct: float = 0.6,
        val_pct: float = 0.2,
        test_pct: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create temporally-aware train/val/test splits.
        Maintains chronological order within each symbol/timeframe group.
        """
        print("Creating train/validation/test splits...")
        
        if not abs(train_pct + val_pct + test_pct - 1.0) < 1e-6:
            raise ValueError("Split percentages must sum to 1.0")
        
        train_dfs, val_dfs, test_dfs = [], [], []
        
        groups = df.groupby(['symbol', 'timeframe'])
        for (symbol, timeframe), group in groups:
            group_sorted = group.sort_index()
            n = len(group_sorted)
            
            if n < 10:
                print(f"  WARNING: {symbol} {timeframe} has only {n} samples, skipping...")
                continue
            
            # Calculate split indices
            train_end = int(n * train_pct)
            val_end = int(n * (train_pct + val_pct))
            
            train_split = group_sorted.iloc[:train_end]
            val_split = group_sorted.iloc[train_end:val_end]
            test_split = group_sorted.iloc[val_end:]
            
            print(f"  {symbol} {timeframe}: Train={len(train_split)}, Val={len(val_split)}, Test={len(test_split)}")
            
            if len(train_split) > 0:
                train_dfs.append(train_split)
            if len(val_split) > 0:
                val_dfs.append(val_split)
            if len(test_split) > 0:
                test_dfs.append(test_split)
        
        train_df = pd.concat(train_dfs) if train_dfs else pd.DataFrame()
        val_df = pd.concat(val_dfs) if val_dfs else pd.DataFrame()
        test_df = pd.concat(test_dfs) if test_dfs else pd.DataFrame()
        
        print(f"Final splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def standardize_features(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Compute standardization parameters from training data and apply to all splits.
        Excludes categorical and target columns from standardization.
        """
        print("Computing standardization parameters and applying to all splits...")
        
        # Exclude columns from standardization
        exclude_cols = {
            'symbol', 'timeframe', 'seq_id', 'y_cls', 'y_reg',
            'open', 'high', 'low', 'close', 'volume',  # OHLCV data
            'dow', 'hour',  # Categorical time features
        }
        
        # Find binary/categorical columns (0/1 values only)
        binary_cols = []
        for col in train_df.columns:
            if col not in exclude_cols and train_df[col].dtype in ['int64', 'float64']:
                unique_vals = train_df[col].dropna().unique()
                if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                    binary_cols.append(col)
        
        print(f"Excluding {len(exclude_cols | set(binary_cols))} columns from standardization")
        print(f"Binary columns: {binary_cols}")
        
        # Compute scaler from training data
        scaler = compute_scaler(
            train_df, 
            exclude_cols=exclude_cols | set(binary_cols)
        )
        
        # Apply standardization
        train_std = apply_scaler(train_df, scaler)
        val_std = apply_scaler(val_df, scaler)
        test_std = apply_scaler(test_df, scaler)
        
        print("Standardization completed")
        
        return train_std, val_std, test_std, scaler
    
    def save_processed_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame,
        scaler: Dict,
        suffix: str = ""
    ) -> Dict[str, str]:
        """Save all processed datasets and metadata."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.output_dir, f"run_{timestamp}{suffix}")
        os.makedirs(run_dir, exist_ok=True)
        
        file_paths = {}
        
        # Save split datasets
        datasets = {
            "train": train_df,
            "val": val_df, 
            "test": test_df
        }
        
        for name, df in datasets.items():
            if not df.empty:
                path = os.path.join(run_dir, f"{name}.csv")
                df.to_csv(path, index=True)
                file_paths[name] = path
                print(f"Saved {name}: {len(df)} rows -> {path}")
        
        # Save scaler
        scaler_path = os.path.join(run_dir, "scaler.json")
        with open(scaler_path, 'w') as f:
            json.dump(scaler, f, indent=2)
        file_paths["scaler"] = scaler_path
        
        # Save metadata
        metadata = {
            "processing_timestamp": timestamp,
            "symbols": sorted(train_df['symbol'].unique()) if 'symbol' in train_df.columns else [],
            "timeframes": sorted(train_df['timeframe'].unique()) if 'timeframe' in train_df.columns else [],
            "dataset_shapes": {name: list(df.shape) for name, df in datasets.items() if not df.empty},
            "feature_columns": [col for col in train_df.columns 
                              if col not in ['symbol', 'timeframe', 'seq_id', 'y_cls', 'y_reg', 
                                           'open', 'high', 'low', 'close', 'volume']],
            "standardization_info": {
                "excluded_columns": scaler.get("excluded_columns", []),
                "binary_columns": scaler.get("binary_excluded", []),
                "standardized_features": scaler.get("feature_cols", [])
            }
        }
        
        metadata_path = os.path.join(run_dir, "processing_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        file_paths["metadata"] = metadata_path
        
        print(f"All files saved to: {run_dir}")
        return file_paths
    
    def run_full_pipeline(
        self,
        label_config: Optional[Dict] = None,
        train_pct: float = 0.6,
        val_pct: float = 0.2, 
        test_pct: float = 0.2,
        suffix: str = ""
    ) -> Dict[str, str]:
        """
        Run the complete multi-stock processing pipeline:
        1. Consolidate raw data
        2. Add features uniformly  
        3. Create labels uniformly
        4. Apply quality control
        5. Create temporal splits
        6. Standardize features
        7. Save processed datasets
        """
        
        print("="*80)
        print("MULTI-STOCK PROCESSING PIPELINE")
        print("="*80)
        
        try:
            # Step 1: Consolidate raw data
            consolidated_df = self.consolidate_raw_data()
            
            # Step 2: Add features
            featured_df = self.add_features_unified(consolidated_df)
            
            # Step 3: Create labels
            labeled_df = self.create_labels_unified(featured_df, label_config)
            
            # Step 4: Apply QC
            qc_df = self.apply_quality_control(labeled_df)
            
            # Step 5: Create splits
            train_df, val_df, test_df = self.create_train_val_test_splits(
                qc_df, train_pct, val_pct, test_pct
            )
            
            # Step 6: Standardize
            train_std, val_std, test_std, scaler = self.standardize_features(
                train_df, val_df, test_df
            )
            
            # Step 7: Save everything
            file_paths = self.save_processed_data(
                train_std, val_std, test_std, scaler, suffix
            )
            
            print("="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"Final dataset sizes:")
            print(f"  Training: {len(train_std)}")
            print(f"  Validation: {len(val_std)}")
            print(f"  Test: {len(test_std)}")
            print(f"  Features: {len(scaler.get('feature_cols', []))}")
            
            return file_paths
            
        except Exception as e:
            print(f"PIPELINE FAILED: {e}")
            raise


def main():
    """CLI interface for multi-stock processing"""
    parser = argparse.ArgumentParser(description="Multi-stock ML data processing pipeline")
    
    parser.add_argument("--base_dir", default=None, 
                       help="Base directory containing raw data (default: from ml_defaults)")
    parser.add_argument("--output_dir", default=None,
                       help="Output directory for processed data") 
    parser.add_argument("--symbols", nargs="+", default=None,
                       help="List of symbols to process (default: from ml_defaults)")
    parser.add_argument("--timeframes", nargs="+", default=None,
                       help="List of timeframes to process (default: all available)")
    
    parser.add_argument("--train_pct", type=float, default=0.6,
                       help="Training split percentage")
    parser.add_argument("--val_pct", type=float, default=0.2, 
                       help="Validation split percentage")
    parser.add_argument("--test_pct", type=float, default=0.2,
                       help="Test split percentage")
    
    parser.add_argument("--label_horizon", type=int, default=None,
                       help="Default label horizon (overrides timeframe-specific configs)")
    parser.add_argument("--label_threshold", type=float, default=None,
                       help="Default label threshold (overrides timeframe-specific configs)")
    
    parser.add_argument("--suffix", default="",
                       help="Suffix to add to output directory name")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = MultiStockProcessor(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        symbols=args.symbols,
        timeframes=args.timeframes
    )
    
    # Prepare label configuration if custom values provided
    label_config = None
    if args.label_horizon is not None or args.label_threshold is not None:
        label_config = {}
        for tf in processor.timeframes:
            label_config[tf] = {
                "horizon": args.label_horizon or DEFAULT_TIMEFRAMES.get(tf, {}).get("horizon_bars", 5),
                "threshold": args.label_threshold or 0.03,
                "method": "fixed_threshold"
            }
    
    # Run pipeline
    file_paths = processor.run_full_pipeline(
        label_config=label_config,
        train_pct=args.train_pct,
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        suffix=args.suffix
    )
    
    print("\nGenerated files:")
    for name, path in file_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()