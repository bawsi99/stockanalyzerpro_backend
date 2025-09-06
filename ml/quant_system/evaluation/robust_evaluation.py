"""
Robust evaluation with time-based splits and walk-forward analysis.

This script implements:
1. Time-based train/validation/test splits (70/15/15)
2. Walk-forward analysis with expanding windows
3. Per-pattern-type analysis
4. Statistical significance testing

Usage:
  python -m backend.ml.quant_system.robust_evaluation \
    --dataset backend/ml/quant_system/datasets/robust_patterns.parquet
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy import stats

# Ensure package paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from ..engines.pattern_ml import PatternMLEngine
from ..core import UnifiedMLConfig
from ..scripts.train_models import load_dataset, build_training_data


def time_based_split(df: pd.DataFrame, train_pct: float = 0.7, val_pct: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset by time, ensuring no future leakage."""
    df = df.copy()
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    total_samples = len(df)
    train_end = int(total_samples * train_pct)
    val_end = int(total_samples * (train_pct + val_pct))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"Time-based split:")
    print(f"  Train: {len(train_df)} samples ({train_df['event_date'].min()} to {train_df['event_date'].max()})")
    print(f"  Val:   {len(val_df)} samples ({val_df['event_date'].min()} to {val_df['event_date'].max()})")
    print(f"  Test:  {len(test_df)} samples ({test_df['event_date'].min()} to {test_df['event_date'].max()})")
    
    return train_df, val_df, test_df


def walk_forward_split(df: pd.DataFrame, initial_train_pct: float = 0.3, step_pct: float = 0.1) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate walk-forward splits for time series validation."""
    df = df.copy()
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    splits = []
    total_samples = len(df)
    initial_train_size = int(total_samples * initial_train_pct)
    step_size = int(total_samples * step_pct)
    
    for i in range(initial_train_size, total_samples - step_size, step_size):
        train_df = df.iloc[:i]
        test_df = df.iloc[i:i+step_size]
        splits.append((train_df, test_df))
    
    print(f"Walk-forward splits: {len(splits)} periods")
    return splits


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        'positive_rate': np.mean(y_true),
        'predicted_positive_rate': np.mean(y_pred),
    }


def statistical_significance_test(metric1: float, metric2: float, n1: int, n2: int, alpha: float = 0.05) -> Dict[str, Any]:
    """Perform statistical significance test between two metrics."""
    # Simplified t-test for proportions
    pooled_p = (metric1 * n1 + metric2 * n2) / (n1 + n2)
    pooled_se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
    t_stat = (metric1 - metric2) / pooled_se
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'alpha': alpha
    }


def evaluate_time_based_split(dataset_path: str) -> Dict[str, Any]:
    """Evaluate with time-based train/val/test split."""
    print("=== TIME-BASED SPLIT EVALUATION ===")
    
    df = load_dataset(dataset_path)
    train_df, val_df, test_df = time_based_split(df)
    
    # Train on training set
    X_train, y_train, meta_train = build_training_data(train_df)
    engine = PatternMLEngine(config=UnifiedMLConfig(catboost_iterations=400))
    train_success = engine.train(data=(X_train, y_train, meta_train))
    
    if not train_success:
        print("Training failed")
        return {}
    
    results = {}
    
    # Evaluate on validation set
    X_val, y_val, meta_val = build_training_data(val_df)
    val_metrics = engine.evaluate(data=(X_val, y_val, meta_val))
    results['validation'] = val_metrics
    
    # Evaluate on test set
    X_test, y_test, meta_test = build_training_data(test_df)
    test_metrics = engine.evaluate(data=(X_test, y_test, meta_test))
    results['test'] = test_metrics
    
    # Statistical significance test
    if 'accuracy' in val_metrics and 'accuracy' in test_metrics:
        sig_test = statistical_significance_test(
            val_metrics['accuracy'], test_metrics['accuracy'],
            len(y_val), len(y_test)
        )
        results['significance_test'] = sig_test
    
    return results


def evaluate_walk_forward(dataset_path: str) -> Dict[str, Any]:
    """Evaluate with walk-forward analysis."""
    print("=== WALK-FORWARD EVALUATION ===")
    
    df = load_dataset(dataset_path)
    splits = walk_forward_split(df)
    
    all_metrics = []
    
    for i, (train_df, test_df) in enumerate(splits):
        print(f"Walk-forward period {i+1}/{len(splits)}: train={len(train_df)}, test={len(test_df)}")
        
        # Train on expanding window
        X_train, y_train, meta_train = build_training_data(train_df)
        engine = PatternMLEngine(config=UnifiedMLConfig(catboost_iterations=200))  # Faster for walk-forward
        train_success = engine.train(data=(X_train, y_train, meta_train))
        
        if not train_success:
            continue
        
        # Evaluate on test window
        X_test, y_test, meta_test = build_training_data(test_df)
        metrics = engine.evaluate(data=(X_test, y_test, meta_test))
        
        if isinstance(metrics, dict):
            metrics['period'] = i + 1
            metrics['train_size'] = len(train_df)
            metrics['test_size'] = len(test_df)
            all_metrics.append(metrics)
    
    if not all_metrics:
        return {}
    
    # Aggregate results
    metrics_df = pd.DataFrame(all_metrics)
    results = {
        'periods': len(all_metrics),
        'mean_accuracy': metrics_df['accuracy'].mean(),
        'std_accuracy': metrics_df['accuracy'].std(),
        'mean_auc': metrics_df['auc'].mean() if 'auc' in metrics_df.columns else 0.5,
        'std_auc': metrics_df['auc'].std() if 'auc' in metrics_df.columns else 0.0,
        'period_metrics': all_metrics
    }
    
    return results


def analyze_pattern_types(dataset_path: str) -> Dict[str, Any]:
    """Analyze performance by pattern type."""
    print("=== PATTERN-TYPE ANALYSIS ===")
    
    df = load_dataset(dataset_path)
    pattern_analysis = {}
    
    for pattern_type in df['pattern_type'].unique():
        pattern_df = df[df['pattern_type'] == pattern_type]
        if len(pattern_df) < 10:  # Skip patterns with too few samples
            continue
        
        print(f"\nPattern: {pattern_type} (n={len(pattern_df)})")
        
        # Time-based split for this pattern
        train_df, val_df, test_df = time_based_split(pattern_df)
        
        if len(train_df) < 5 or len(test_df) < 5:
            continue
        
        # Train and evaluate
        X_train, y_train, meta_train = build_training_data(train_df)
        engine = PatternMLEngine(config=UnifiedMLConfig(catboost_iterations=200))
        train_success = engine.train(data=(X_train, y_train, meta_train))
        
        if not train_success:
            continue
        
        X_test, y_test, meta_test = build_training_data(test_df)
        metrics = engine.evaluate(data=(X_test, y_test, meta_test))
        
        pattern_analysis[pattern_type] = {
            'total_samples': len(pattern_df),
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'positive_rate': np.mean(y_test),
            'metrics': metrics
        }
    
    return pattern_analysis


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Robust evaluation with time-based splits")
    p.add_argument("--dataset", required=True)
    args = p.parse_args(argv)
    
    if not os.path.exists(args.dataset):
        print(f"Dataset not found: {args.dataset}")
        return 1
    
    print(f"Loading dataset: {args.dataset}")
    df = load_dataset(args.dataset)
    print(f"Dataset size: {len(df)} events")
    print(f"Date range: {df['event_date'].min()} to {df['event_date'].max()}")
    print(f"Pattern types: {df['pattern_type'].value_counts().to_dict()}")
    print(f"Positive rate: {df['y_success'].mean():.3f}")
    
    results = {}
    
    # 1. Time-based split evaluation
    try:
        time_results = evaluate_time_based_split(args.dataset)
        results['time_based'] = time_results
    except Exception as e:
        print(f"Time-based evaluation failed: {e}")
    
    # 2. Walk-forward evaluation
    try:
        walk_results = evaluate_walk_forward(args.dataset)
        results['walk_forward'] = walk_results
    except Exception as e:
        print(f"Walk-forward evaluation failed: {e}")
    
    # 3. Pattern-type analysis
    try:
        pattern_results = analyze_pattern_types(args.dataset)
        results['pattern_analysis'] = pattern_results
    except Exception as e:
        print(f"Pattern analysis failed: {e}")
    
    # Print summary
    print("\n=== SUMMARY ===")
    if 'time_based' in results and 'test' in results['time_based']:
        test_metrics = results['time_based']['test']
        print(f"Test Accuracy: {test_metrics.get('accuracy', 'N/A')}")
        print(f"Test AUC: {test_metrics.get('auc', 'N/A')}")
    
    if 'walk_forward' in results:
        wf = results['walk_forward']
        print(f"Walk-forward Mean Accuracy: {wf.get('mean_accuracy', 'N/A')} ± {wf.get('std_accuracy', 'N/A')}")
        print(f"Walk-forward Mean AUC: {wf.get('mean_auc', 'N/A')} ± {wf.get('std_auc', 'N/A')}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
