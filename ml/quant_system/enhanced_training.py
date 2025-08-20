"""
Enhanced Training Script for Pattern ML

This script combines enhanced feature engineering with advanced ML techniques:
1. Enhanced feature engineering (market regime, technical indicators, etc.)
2. Ensemble methods with hyperparameter optimization
3. Feature selection and scaling
4. Advanced validation strategies

Usage:
    python -m backend.ml.quant_system.enhanced_training \
        --dataset backend/ml/quant_system/datasets/robust_patterns.parquet \
        --output backend/ml/quant_system/models/enhanced_pattern_model.joblib
"""

import os
import sys
import argparse
from typing import Dict, Any
import numpy as np
import pandas as pd
import logging
from datetime import datetime

# Ensure package paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from ml.quant_system.enhanced_feature_engineering import EnhancedFeatureEngine
from ml.quant_system.enhanced_ml_engine import EnhancedMLEngine
from ml.quant_system.train_evaluate_pattern import load_dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_enhanced_config() -> Dict[str, Any]:
    """Create enhanced configuration for training."""
    return {
        'ensemble_method': 'voting',  # Use voting ensemble for better stability
        'base_models': ['catboost', 'random_forest', 'gradient_boosting'],
        'meta_model': 'logistic_regression',
        'cv_folds': 5,
        'optimization_trials': 30,  # Reduced for faster training
        'feature_selection': True,
        'feature_selection_method': 'kbest',
        'n_features': 25,  # Increased from 20
        'scaling': True,
        'calibration': True,
        'random_state': 42
    }


def time_based_split(df: pd.DataFrame, train_pct: float = 0.7, val_pct: float = 0.15) -> tuple:
    """Split dataset by time."""
    df = df.copy()
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    total_samples = len(df)
    train_end = int(total_samples * train_pct)
    val_end = int(total_samples * (train_pct + val_pct))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    logger.info(f"Time-based split:")
    logger.info(f"  Train: {len(train_df)} samples ({train_df['event_date'].min()} to {train_df['event_date'].max()})")
    logger.info(f"  Val:   {len(val_df)} samples ({val_df['event_date'].min()} to {val_df['event_date'].max()})")
    logger.info(f"  Test:  {len(test_df)} samples ({test_df['event_date'].min()} to {test_df['event_date'].max()})")
    
    return train_df, val_df, test_df


def train_enhanced_model(dataset_path: str, output_path: str) -> Dict[str, Any]:
    """Train enhanced model with advanced features and ensemble methods."""
    
    # Load dataset
    logger.info(f"Loading dataset: {dataset_path}")
    df = load_dataset(dataset_path)
    logger.info(f"Dataset loaded: {len(df)} events, {df['pattern_type'].nunique()} pattern types")
    
    # Time-based split
    train_df, val_df, test_df = time_based_split(df)
    
    # Initialize enhanced feature engineering
    logger.info("Initializing enhanced feature engineering...")
    feature_engine = EnhancedFeatureEngine()
    
    # Fit feature engineering on training data
    logger.info("Fitting feature engineering...")
    feature_engine.fit(train_df)
    
    # Transform all datasets
    logger.info("Transforming datasets with enhanced features...")
    X_train_enhanced = feature_engine.transform(train_df)
    X_val_enhanced = feature_engine.transform(val_df)
    X_test_enhanced = feature_engine.transform(test_df)
    
    # Ensure all datasets have the same features
    common_features = list(set(X_train_enhanced.columns) & set(X_val_enhanced.columns) & set(X_test_enhanced.columns))
    logger.info(f"Common features across all datasets: {len(common_features)}")
    
    X_train_enhanced = X_train_enhanced[common_features]
    X_val_enhanced = X_val_enhanced[common_features]
    X_test_enhanced = X_test_enhanced[common_features]
    
    # Extract targets
    y_train = train_df['y_success'].astype(int).to_numpy()
    y_val = val_df['y_success'].astype(int).to_numpy()
    y_test = test_df['y_success'].astype(int).to_numpy()
    
    logger.info(f"Enhanced features created: {X_train_enhanced.shape[1]} features")
    
    # Get feature summary
    feature_summary = feature_engine.get_feature_summary()
    logger.info(f"Feature categories: {feature_summary['feature_categories']}")
    
    # Initialize enhanced ML engine
    logger.info("Initializing enhanced ML engine...")
    config = create_enhanced_config()
    ml_engine = EnhancedMLEngine(config)
    
    # Train model
    logger.info("Training enhanced model...")
    train_success = ml_engine.train(X_train_enhanced, y_train)
    
    if not train_success:
        logger.error("Enhanced model training failed")
        return {"error": "Training failed"}
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_metrics = ml_engine.evaluate(X_val_enhanced, y_val)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = ml_engine.evaluate(X_test_enhanced, y_test)
    
    # Get feature importance
    feature_importance = ml_engine.get_feature_importance()
    
    # Save model
    logger.info(f"Saving enhanced model to {output_path}")
    save_success = ml_engine.save_model(output_path)
    
    # Compile results
    results = {
        'training_success': train_success,
        'save_success': save_success,
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'feature_importance': feature_importance,
        'feature_summary': feature_summary,
        'model_info': ml_engine.get_model_info(),
        'dataset_info': {
            'total_events': len(df),
            'train_events': len(train_df),
            'val_events': len(val_df),
            'test_events': len(test_df),
            'pattern_types': df['pattern_type'].value_counts().to_dict(),
            'positive_rate': df['y_success'].mean()
        }
    }
    
    return results


def print_results(results: Dict[str, Any]):
    """Print training results."""
    print("\n" + "="*60)
    print("ENHANCED MODEL TRAINING RESULTS")
    print("="*60)
    
    if 'error' in results:
        print(f"❌ Training failed: {results['error']}")
        return
    
    # Dataset info
    print(f"\n📊 DATASET INFO:")
    dataset_info = results['dataset_info']
    print(f"  Total events: {dataset_info['total_events']}")
    print(f"  Train/Val/Test: {dataset_info['train_events']}/{dataset_info['val_events']}/{dataset_info['test_events']}")
    print(f"  Pattern types: {len(dataset_info['pattern_types'])}")
    print(f"  Positive rate: {dataset_info['positive_rate']:.3f}")
    
    # Feature info
    print(f"\n🔧 FEATURE ENGINEERING:")
    feature_summary = results['feature_summary']
    print(f"  Total features: {feature_summary['total_features']}")
    for category, features in feature_summary['feature_categories'].items():
        print(f"  {category}: {len(features)} features")
    
    # Model info
    print(f"\n🤖 MODEL INFO:")
    model_info = results['model_info']
    print(f"  Ensemble method: {model_info['ensemble_method']}")
    print(f"  Base models: {', '.join(model_info['base_models'])}")
    print(f"  Feature selection: {model_info['feature_selection']}")
    print(f"  Scaling: {model_info['scaling']}")
    print(f"  Calibration: {model_info['calibration']}")
    
    # Performance metrics
    print(f"\n📈 PERFORMANCE METRICS:")
    
    val_metrics = results['validation_metrics']
    test_metrics = results['test_metrics']
    
    print(f"  Validation Set:")
    print(f"    Accuracy:  {val_metrics.get('accuracy', 'N/A')}")
    print(f"    Precision: {val_metrics.get('precision', 'N/A')}")
    print(f"    Recall:    {val_metrics.get('recall', 'N/A')}")
    print(f"    F1-Score:  {val_metrics.get('f1', 'N/A')}")
    print(f"    AUC:       {val_metrics.get('auc', 'N/A')}")
    print(f"    Brier:     {val_metrics.get('brier', 'N/A')}")
    
    print(f"  Test Set:")
    print(f"    Accuracy:  {test_metrics.get('accuracy', 'N/A')}")
    print(f"    Precision: {test_metrics.get('precision', 'N/A')}")
    print(f"    Recall:    {test_metrics.get('recall', 'N/A')}")
    print(f"    F1-Score:  {test_metrics.get('f1', 'N/A')}")
    print(f"    AUC:       {test_metrics.get('auc', 'N/A')}")
    print(f"    Brier:     {test_metrics.get('brier', 'N/A')}")
    
    # Top features
    print(f"\n🏆 TOP 10 FEATURES:")
    feature_importance = results['feature_importance']
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
        print(f"  {i+1:2d}. {feature}: {importance}")
    
    # Save status
    print(f"\n💾 MODEL SAVED: {'✅' if results['save_success'] else '❌'}")
    
    print("\n" + "="*60)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train enhanced pattern ML model")
    parser.add_argument("--dataset", required=True, help="Path to dataset file")
    parser.add_argument("--output", required=True, help="Output path for trained model")
    parser.add_argument("--config", help="Path to custom config file (optional)")
    
    args = parser.parse_args(argv)
    
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset not found: {args.dataset}")
        return 1
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Train enhanced model
    logger.info("Starting enhanced model training...")
    results = train_enhanced_model(args.dataset, args.output)
    
    # Print results
    print_results(results)
    
    if 'error' in results:
        return 1
    
    logger.info("Enhanced model training completed successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
