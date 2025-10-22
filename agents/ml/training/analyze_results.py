#!/usr/bin/env python3
"""
Analysis script for multi-model training results.
Provides detailed insights and comparisons between different models.
"""

import json
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any


def load_results(model_dir: str) -> Dict[str, Any]:
    """Load the comprehensive results from the multi-model training"""
    metrics_path = os.path.join(model_dir, "all_models_metrics.json")
    with open(metrics_path, "r") as f:
        return json.load(f)


def print_detailed_analysis(results: Dict[str, Any]) -> None:
    """Print detailed analysis of model results"""
    
    dataset_info = results["dataset_info"]
    models = results["models"]
    
    print("=" * 100)
    print("STOCKANALYZER PRO - MULTI-MODEL TRAINING ANALYSIS")
    print("=" * 100)
    
    print(f"\nDATASET INFORMATION:")
    print(f"  • Source: {dataset_info['splits_dir']}")
    print(f"  • Training samples: {dataset_info['n']['train']:,}")
    print(f"  • Validation samples: {dataset_info['n']['val']:,}")
    print(f"  • Test samples: {dataset_info['n']['test']:,}")
    print(f"  • Features: {len(dataset_info['features'])} total")
    
    # Top features (show first 10)
    print(f"\n  Top features:")
    for i, feature in enumerate(dataset_info['features'][:10]):
        print(f"    {i+1:2d}. {feature}")
    if len(dataset_info['features']) > 10:
        print(f"    ... and {len(dataset_info['features']) - 10} more")
    
    print(f"\n{'MODEL PERFORMANCE DETAILED ANALYSIS':^100}")
    print("=" * 100)
    
    # Sort models by test AUC
    sorted_models = sorted(
        models.items(),
        key=lambda x: x[1]["auc"]["test"] if not np.isnan(x[1]["auc"]["test"]) else -1,
        reverse=True
    )
    
    for rank, (model_name, metrics) in enumerate(sorted_models, 1):
        print(f"\n{rank}. {model_name.replace('_', ' ').upper()} MODEL")
        print("-" * 80)
        
        # Classification Performance
        train_auc = metrics["auc"]["train"]
        val_auc = metrics["auc"]["val"]
        test_auc = metrics["auc"]["test"]
        val_ap = metrics["ap_val"]
        
        print(f"  Classification Performance:")
        print(f"    • Train AUC: {train_auc:.4f}")
        print(f"    • Val AUC:   {val_auc:.4f}")
        print(f"    • Test AUC:  {test_auc:.4f}")
        print(f"    • Val AP:    {val_ap:.4f}")
        
        # Overfitting Analysis
        if not np.isnan(train_auc) and not np.isnan(val_auc):
            overfitting = train_auc - val_auc
            if overfitting > 0.1:
                print(f"    ⚠️  High overfitting detected (Δ: {overfitting:.4f})")
            elif overfitting > 0.05:
                print(f"    ⚠️  Moderate overfitting (Δ: {overfitting:.4f})")
            else:
                print(f"    ✅ Good generalization (Δ: {overfitting:.4f})")
        
        # Trading Strategy Performance
        threshold = metrics["threshold"]
        coverage = metrics["test_performance"]["coverage"]
        avg_return = metrics["test_performance"]["avg_y_reg"]
        cum_return = metrics["test_performance"]["cum_y_reg"]
        
        print(f"  Trading Strategy Performance:")
        print(f"    • Threshold: {threshold:.4f}")
        print(f"    • Coverage:  {coverage:.2%}")
        
        if not np.isnan(avg_return):
            print(f"    • Avg Return per Trade: {avg_return:.4f} ({avg_return*100:.2f}%)")
            print(f"    • Cumulative Return: {cum_return:.4f} ({cum_return*100:.2f}%)")
            
            # Performance assessment
            if avg_return > 0.02:
                print(f"    🚀 Excellent trading performance")
            elif avg_return > 0.01:
                print(f"    ✅ Good trading performance") 
            elif avg_return > 0.005:
                print(f"    ⚠️  Modest trading performance")
            else:
                print(f"    ❌ Poor trading performance")
        else:
            print(f"    • Avg Return per Trade: No trades generated")
            print(f"    • Cumulative Return: {cum_return:.4f}")
            print(f"    ❌ No trading opportunities identified")
        
        # Model-specific insights
        if model_name == "logistic":
            print(f"  Model Insights:")
            print(f"    • Linear relationships captured effectively")
            print(f"    • Fast training and inference")
            print(f"    • Interpretable coefficients")
            
        elif "forest" in model_name:
            if train_auc > 0.95:
                print(f"  Model Insights:")
                print(f"    • High training accuracy indicates complex pattern capture")
                print(f"    • May be overfitting due to ensemble complexity")
                print(f"    • Consider reducing n_estimators or increasing min_samples_split")
                
        elif "boosting" in model_name or model_name in ["xgboost", "lightgbm"]:
            if train_auc >= 1.0:
                print(f"  Model Insights:")
                print(f"    • Perfect training performance - likely overfitting")
                print(f"    • Consider regularization (lower learning rate, fewer estimators)")
                print(f"    • Early stopping could help prevent overfitting")

    print(f"\n{'COMPARATIVE ANALYSIS':^100}")
    print("=" * 100)
    
    # Best performing model
    best_model = sorted_models[0]
    worst_model = sorted_models[-1]
    
    print(f"\n🏆 BEST PERFORMING MODEL: {best_model[0].replace('_', ' ').upper()}")
    print(f"   Test AUC: {best_model[1]['auc']['test']:.4f}")
    print(f"   Avg Return: {best_model[1]['test_performance']['avg_y_reg']:.4f}")
    
    print(f"\n📉 LEAST PERFORMING MODEL: {worst_model[0].replace('_', ' ').upper()}")
    print(f"   Test AUC: {worst_model[1]['auc']['test']:.4f}")
    
    # Calculate performance spreads
    test_aucs = [m[1]["auc"]["test"] for m in sorted_models if not np.isnan(m[1]["auc"]["test"])]
    if len(test_aucs) > 1:
        auc_spread = max(test_aucs) - min(test_aucs)
        print(f"\n📊 PERFORMANCE SPREAD:")
        print(f"   AUC Range: {min(test_aucs):.4f} - {max(test_aucs):.4f}")
        print(f"   AUC Spread: {auc_spread:.4f}")
        
        if auc_spread < 0.05:
            print(f"   All models perform similarly")
        elif auc_spread < 0.1:
            print(f"   Moderate performance differences between models")
        else:
            print(f"   Significant performance differences - model selection matters")
    
    # Overfitting analysis
    print(f"\n🔍 OVERFITTING ANALYSIS:")
    for model_name, metrics in models.items():
        train_auc = metrics["auc"]["train"]
        val_auc = metrics["auc"]["val"]
        if not np.isnan(train_auc) and not np.isnan(val_auc):
            overfitting = train_auc - val_auc
            status = "🔴 HIGH" if overfitting > 0.1 else "🟡 MOD" if overfitting > 0.05 else "🟢 LOW"
            print(f"   {model_name:15s}: {status} (Train: {train_auc:.3f}, Val: {val_auc:.3f}, Δ: {overfitting:.3f})")
    
    # Trading strategy insights
    print(f"\n💰 TRADING STRATEGY INSIGHTS:")
    profitable_models = 0
    for model_name, metrics in models.items():
        avg_ret = metrics["test_performance"]["avg_y_reg"]
        if not np.isnan(avg_ret) and avg_ret > 0:
            profitable_models += 1
    
    print(f"   Profitable models: {profitable_models}/{len(models)}")
    
    if profitable_models > 0:
        print(f"   ✅ Some models show profitable trading potential")
        print(f"   Consider ensemble methods combining top performers")
    else:
        print(f"   ⚠️  No models show consistent profitability")
        print(f"   May need feature engineering or different labeling strategy")
    
    print(f"\n{'RECOMMENDATIONS':^100}")
    print("=" * 100)
    
    best_auc = best_model[1]["auc"]["test"]
    best_return = best_model[1]["test_performance"]["avg_y_reg"]
    
    print(f"\n🎯 MODEL SELECTION:")
    if best_auc > 0.6 and not np.isnan(best_return) and best_return > 0.01:
        print(f"   • Deploy {best_model[0].replace('_', ' ').title()} model for production")
        print(f"   • Strong classification and trading performance")
    elif best_auc > 0.55:
        print(f"   • {best_model[0].replace('_', ' ').title()} shows promise but needs improvement")
        print(f"   • Consider ensemble methods or feature engineering")
    else:
        print(f"   • All models show weak performance")
        print(f"   • Recommend data quality review and feature engineering")
    
    print(f"\n🔧 MODEL IMPROVEMENT:")
    high_overfitting_models = []
    for model_name, metrics in models.items():
        train_auc = metrics["auc"]["train"]
        val_auc = metrics["auc"]["val"]
        if not np.isnan(train_auc) and not np.isnan(val_auc) and (train_auc - val_auc) > 0.1:
            high_overfitting_models.append(model_name)
    
    if high_overfitting_models:
        print(f"   • Reduce overfitting in: {', '.join(high_overfitting_models)}")
        print(f"   • Try regularization, cross-validation, or simpler models")
    
    if best_auc < 0.6:
        print(f"   • Overall weak performance suggests:")
        print(f"     - Review feature engineering")
        print(f"     - Check data quality and labeling")
        print(f"     - Consider different time horizons or targets")
    
    print(f"\n📈 DATA STRATEGY:")
    total_samples = sum(dataset_info["n"].values())
    if total_samples < 2000:
        print(f"   • Small dataset ({total_samples:,} samples) may limit model performance")
        print(f"   • Consider collecting more historical data")
    
    if dataset_info["n"]["test"] < 100:
        print(f"   • Small test set may lead to unstable performance estimates")
        print(f"   • Consider using cross-validation for more robust evaluation")
    
    print(f"\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Analyze multi-model training results")
    parser.add_argument("--model_dir", required=True, help="Directory containing model results")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    
    results = load_results(args.model_dir)
    print_detailed_analysis(results)


if __name__ == "__main__":
    main()