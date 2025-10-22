#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# Optional imports for XGBoost and LightGBM
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

EXCLUDE_COLS = {"y_cls", "y_reg", "open", "high", "low", "close", "volume"}


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    return df.sort_index()


def load_scaler(scaler_json: str) -> Dict:
    with open(scaler_json, "r") as f:
        return json.load(f)


def transform_with_scaler(df: pd.DataFrame, scaler: Dict) -> pd.DataFrame:
    out = df.copy()
    feature_cols: List[str] = scaler["feature_cols"]
    means: Dict[str, float] = scaler["means"]
    stds: Dict[str, float] = scaler["stds"]
    # Scale only known feature_cols
    for c in feature_cols:
        if c in out.columns:
            s = pd.to_numeric(out[c], errors="coerce")
            mu = means.get(c, 0.0)
            sd = stds.get(c, 1.0) or 1.0
            out[c] = (s - mu) / sd
    return out


def build_X(df: pd.DataFrame, scaler: Dict) -> Tuple[np.ndarray, List[str]]:
    # Use scaled feature_cols + unscaled binary flags registered in scaler
    feature_cols: List[str] = scaler["feature_cols"]
    binary_cols: List[str] = scaler.get("binary_excluded", [])
    cols: List[str] = []
    for c in feature_cols + binary_cols:
        if c in df.columns and c not in EXCLUDE_COLS:
            cols.append(c)
    X = df[cols].values
    return X, cols


def select_best_threshold(p: np.ndarray, y_reg: np.ndarray) -> float:
    # Grid over percentiles of p to be threshold-robust
    qs = np.linspace(0.5, 0.99, 50)
    thr_grid = np.quantile(p, qs)
    best_thr, best_er = 0.5, -1e9
    for thr in thr_grid:
        mask = p >= thr
        if mask.sum() == 0:
            continue
        er = float(y_reg[mask].mean())
        if er > best_er:
            best_er, best_thr = er, thr
    return float(best_thr)


def create_model_configs() -> Dict[str, object]:
    """Create configurations for different models"""
    configs = {
        "logistic": LogisticRegression(
            penalty="l2", 
            class_weight="balanced", 
            max_iter=2000, 
            solver="lbfgs",
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            min_samples_split=20,
            min_samples_leaf=5,
            random_state=42
        )
    }
    
    if HAS_XGB:
        configs["xgboost"] = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
    
    if HAS_LGBM:
        configs["lightgbm"] = LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
            class_weight="balanced"
        )
    
    return configs


def train_and_evaluate_model(
    model, 
    model_name: str, 
    X_tr: np.ndarray, 
    y_tr: np.ndarray, 
    X_va: np.ndarray, 
    y_va: np.ndarray, 
    X_te: np.ndarray, 
    y_te: np.ndarray, 
    yr_va: np.ndarray, 
    yr_te: np.ndarray
) -> Dict:
    """Train a model and return comprehensive metrics"""
    print(f"  Training {model_name}...")
    
    # Handle XGBoost eval_set for early stopping potential
    if model_name == "xgboost" and hasattr(model, 'fit'):
        try:
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        except Exception:
            # Fallback if eval_set causes issues
            model.fit(X_tr, y_tr)
    else:
        model.fit(X_tr, y_tr)
    
    # Get probabilities
    if hasattr(model, "predict_proba"):
        p_tr = model.predict_proba(X_tr)[:, 1]
        p_va = model.predict_proba(X_va)[:, 1]
        p_te = model.predict_proba(X_te)[:, 1]
    else:
        # Fallback for models without predict_proba
        p_tr = model.decision_function(X_tr)
        p_va = model.decision_function(X_va)
        p_te = model.decision_function(X_te)
    
    # Classification metrics
    auc_tr = roc_auc_score(y_tr, p_tr) if len(np.unique(y_tr)) > 1 else float("nan")
    auc_va = roc_auc_score(y_va, p_va) if len(np.unique(y_va)) > 1 else float("nan")
    auc_te = roc_auc_score(y_te, p_te) if len(np.unique(y_te)) > 1 else float("nan")
    ap_va = average_precision_score(y_va, p_va) if len(np.unique(y_va)) > 1 else float("nan")
    
    # Threshold by expected return on validation
    thr = select_best_threshold(p_va, yr_va)
    
    # Test evaluation with threshold
    long_mask_te = p_te >= thr
    coverage = float(long_mask_te.mean())
    avg_ret = float(yr_te[long_mask_te].mean()) if long_mask_te.any() else float("nan")
    cum_ret = float(yr_te[long_mask_te].sum()) if long_mask_te.any() else 0.0
    
    return {
        "model": model,
        "model_name": model_name,
        "predictions": {"train": p_tr, "val": p_va, "test": p_te},
        "threshold": thr,
        "metrics": {
            "auc": {"train": auc_tr, "val": auc_va, "test": auc_te},
            "ap_val": ap_va,
            "test": {"coverage": coverage, "avg_y_reg": avg_ret, "cum_y_reg": cum_ret}
        }
    }


def plot_roc_curves_multi(
    all_results: Dict, 
    y_sets: Dict[str, np.ndarray], 
    save_path: str
) -> None:
    """Plot ROC curves for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Individual plots per model
    model_names = list(all_results.keys())
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for idx, (model_name, result) in enumerate(all_results.items()):
        if idx >= 4:  # Only plot first 4 models individually
            break
            
        ax = axes[idx]
        
        for set_name, color_alpha in [('train', 0.6), ('val', 0.8), ('test', 1.0)]:
            if set_name not in y_sets:
                continue
                
            y_true = y_sets[set_name]
            y_scores = result["predictions"][set_name]
            auc = result["metrics"]["auc"][set_name]
            
            if len(np.unique(y_true)) < 2 or np.isnan(auc):
                continue
                
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            ax.plot(fpr, tpr, alpha=color_alpha, lw=2,
                   label=f'{set_name} (AUC = {auc:.3f})',
                   color=colors[0] if set_name == 'train' else 
                         colors[1] if set_name == 'val' else colors[2])
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves - {model_name.replace("_", " ").title()}')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    # Comparison plot (test set only)
    if len(all_results) > 1:
        ax = axes[-1] if len(all_results) <= 3 else axes[3]
        
        for idx, (model_name, result) in enumerate(all_results.items()):
            y_true = y_sets['test']
            y_scores = result["predictions"]["test"]
            auc = result["metrics"]["auc"]["test"]
            
            if len(np.unique(y_true)) < 2 or np.isnan(auc):
                continue
                
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            ax.plot(fpr, tpr, lw=2, color=colors[idx % len(colors)],
                   label=f'{model_name.replace("_", " ").title()} (AUC = {auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, lw=1, label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison - Test Set')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_results_summary(all_results: Dict) -> pd.DataFrame:
    """Create a summary DataFrame of all model results"""
    summary_data = []
    
    for model_name, result in all_results.items():
        metrics = result["metrics"]
        summary_data.append({
            "Model": model_name.replace("_", " ").title(),
            "Train_AUC": metrics["auc"]["train"],
            "Val_AUC": metrics["auc"]["val"],
            "Test_AUC": metrics["auc"]["test"],
            "Val_AP": metrics["ap_val"],
            "Threshold": result["threshold"],
            "Test_Coverage": metrics["test"]["coverage"],
            "Test_Avg_Return": metrics["test"]["avg_y_reg"],
            "Test_Cum_Return": metrics["test"]["cum_y_reg"]
        })
    
    return pd.DataFrame(summary_data)


def main():
    ap = argparse.ArgumentParser(description="Train multiple models on splits with standardization")
    ap.add_argument("--splits_dir", required=True, help="Directory containing train.csv/val.csv/test.csv and scaler.json")
    ap.add_argument("--model_dir", default=None, help="Output directory for model artifacts (default: backend/agents/ml/models/multi_model_TIMESTAMP)")
    ap.add_argument("--models", nargs="*", default=["logistic", "random_forest", "gradient_boosting", "xgboost", "lightgbm"], 
                   help="Models to train")
    args = ap.parse_args()

    splits = args.splits_dir
    train_std = os.path.join(splits, "train_standardized.csv")
    train_raw = os.path.join(splits, "train.csv")
    val_raw = os.path.join(splits, "val.csv")
    test_raw = os.path.join(splits, "test.csv")
    scaler_json = os.path.join(splits, "scaler.json")

    if not os.path.exists(scaler_json):
        raise FileNotFoundError(f"scaler.json not found in {splits}")
    scaler = load_scaler(scaler_json)

    # Load and transform data
    print("Loading and transforming data...")
    if os.path.exists(train_std):
        df_tr = read_csv(train_std)
    else:
        df_tr = transform_with_scaler(read_csv(train_raw), scaler)

    df_va = transform_with_scaler(read_csv(val_raw), scaler)
    df_te = transform_with_scaler(read_csv(test_raw), scaler)

    # Build matrices
    X_tr, cols = build_X(df_tr, scaler)
    X_va, _ = build_X(df_va, scaler)
    X_te, _ = build_X(df_te, scaler)

    y_tr = df_tr["y_cls"].values.astype(int)
    y_va = df_va["y_cls"].values.astype(int)
    y_te = df_te["y_cls"].values.astype(int)

    yr_tr = df_tr["y_reg"].values.astype(float)
    yr_va = df_va["y_reg"].values.astype(float)
    yr_te = df_te["y_reg"].values.astype(float)

    print(f"Data shapes - Train: {X_tr.shape}, Val: {X_va.shape}, Test: {X_te.shape}")
    print(f"Features: {len(cols)}")

    # Get available models
    all_model_configs = create_model_configs()
    models_to_train = {name: config for name, config in all_model_configs.items() 
                      if name in args.models}
    
    if not models_to_train:
        raise ValueError(f"No valid models found. Available: {list(all_model_configs.keys())}")
    
    print(f"Training {len(models_to_train)} models: {list(models_to_train.keys())}")

    # Setup output directory
    if args.model_dir:
        model_dir = args.model_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ml_dir = os.path.dirname(script_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(ml_dir, "models", f"multi_model_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    # Train all models
    print("Training models...")
    all_results = {}
    
    for model_name, model_config in models_to_train.items():
        try:
            result = train_and_evaluate_model(
                model_config, model_name, 
                X_tr, y_tr, X_va, y_va, X_te, y_te, 
                yr_va, yr_te
            )
            all_results[model_name] = result
            
            # Save individual model
            model_file = os.path.join(model_dir, f"{model_name}.joblib")
            joblib.dump({
                "model": result["model"], 
                "features": cols, 
                "threshold": result["threshold"], 
                "scaler_json": scaler_json
            }, model_file)
            
            # Save individual predictions
            preds = pd.DataFrame({
                "p": result["predictions"]["test"],
                "y_cls": y_te,
                "y_reg": yr_te,
                "decision": (result["predictions"]["test"] >= result["threshold"]).astype(int)
            }, index=df_te.index)
            preds.to_csv(os.path.join(model_dir, f"{model_name}_test_predictions.csv"))
            
            print(f"  ✓ {model_name} - Test AUC: {result['metrics']['auc']['test']:.3f}")
            
        except Exception as e:
            print(f"  ✗ {model_name} failed: {e}")
            continue

    if not all_results:
        raise RuntimeError("No models trained successfully")

    # Create summary
    print("\nCreating results summary...")
    summary_df = create_results_summary(all_results)
    summary_df.to_csv(os.path.join(model_dir, "model_comparison_summary.csv"), index=False)
    
    # Plot ROC curves
    print("Plotting ROC curves...")
    y_sets = {'train': y_tr, 'val': y_va, 'test': y_te}
    roc_plot_path = os.path.join(model_dir, "roc_curves_comparison.png")
    plot_roc_curves_multi(all_results, y_sets, roc_plot_path)
    
    # Compile comprehensive metrics
    all_metrics = {
        "dataset_info": {
            "n": {"train": int(X_tr.shape[0]), "val": int(X_va.shape[0]), "test": int(X_te.shape[0])},
            "features": cols,
            "splits_dir": splits
        },
        "models": {}
    }
    
    for model_name, result in all_results.items():
        all_metrics["models"][model_name] = {
            "auc": result["metrics"]["auc"],
            "ap_val": result["metrics"]["ap_val"],
            "threshold": result["threshold"],
            "test_performance": result["metrics"]["test"]
        }
    
    # Save combined metrics
    with open(os.path.join(model_dir, "all_models_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETED - {len(all_results)} models trained successfully")
    print(f"Model directory: {model_dir}")
    print(f"{'='*80}")
    
    print("\nMODEL PERFORMANCE COMPARISON:")
    print("-" * 80)
    header = f"{'Model':<15} {'Test AUC':<10} {'Val AP':<10} {'Coverage':<10} {'Avg Return':<12} {'Cum Return':<12}"
    print(header)
    print("-" * 80)
    
    # Sort by test AUC for display
    sorted_results = sorted(all_results.items(), 
                          key=lambda x: x[1]["metrics"]["auc"]["test"], 
                          reverse=True)
    
    for model_name, result in sorted_results:
        m = result["metrics"]
        model_display = model_name.replace("_", " ").title()
        test_auc = m["auc"]["test"]
        val_ap = m["ap_val"]
        coverage = m["test"]["coverage"]
        avg_ret = m["test"]["avg_y_reg"]
        cum_ret = m["test"]["cum_y_reg"]
        
        print(f"{model_display:<15} {test_auc:<10.3f} {val_ap:<10.3f} {coverage:<10.1%} {avg_ret:<12.4f} {cum_ret:<12.2f}")
    
    print("-" * 80)
    print(f"\nFiles created:")
    print(f"  - Combined metrics: all_models_metrics.json")
    print(f"  - Summary table: model_comparison_summary.csv") 
    print(f"  - ROC curves: roc_curves_comparison.png")
    print(f"  - Individual models: {{model_name}}.joblib")
    print(f"  - Individual predictions: {{model_name}}_test_predictions.csv")


if __name__ == "__main__":
    main()