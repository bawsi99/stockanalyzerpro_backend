#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


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


def build_X(df: pd.DataFrame, scaler: Dict) -> (np.ndarray, List[str]):
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


def plot_roc_curves(y_sets: Dict[str, np.ndarray], p_sets: Dict[str, np.ndarray], 
                   auc_scores: Dict[str, float], save_path: str) -> None:
    """Plot and save ROC curves for train/val/test sets."""
    plt.figure(figsize=(10, 8))
    
    colors = {'train': 'blue', 'val': 'orange', 'test': 'green'}
    
    for set_name in ['train', 'val', 'test']:
        if set_name not in y_sets or set_name not in p_sets:
            continue
            
        y_true = y_sets[set_name]
        y_scores = p_sets[set_name]
        auc = auc_scores.get(set_name, float('nan'))
        
        # Skip if we don't have both classes or AUC is NaN
        if len(np.unique(y_true)) < 2 or np.isnan(auc):
            continue
            
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color=colors[set_name], lw=2, 
                label=f'{set_name.capitalize()} (AUC = {auc:.3f})')
    
    # Plot diagonal line for random classifier
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.7, label='Random (AUC = 0.500)')
    
    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Logistic Regression', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add some styling
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory


def main():
    ap = argparse.ArgumentParser(description="Train LogisticRegression on splits with standardization")
    ap.add_argument("--splits_dir", required=True, help="Directory containing train.csv/val.csv/test.csv and scaler.json")
    ap.add_argument("--model_dir", default=None, help="Output directory for model artifacts (default: backend/agents/ml/models/model_TIMESTAMP)")
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

    # Train data (already standardized file preferred)
    if os.path.exists(train_std):
        df_tr = read_csv(train_std)
    else:
        df_tr = transform_with_scaler(read_csv(train_raw), scaler)

    # Val/Test transformed using scaler
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

    # Train LR
    clf = LogisticRegression(penalty="l2", class_weight="balanced", max_iter=2000, solver="lbfgs")
    clf.fit(X_tr, y_tr)

    # Probs
    p_tr = clf.predict_proba(X_tr)[:, 1]
    p_va = clf.predict_proba(X_va)[:, 1]
    p_te = clf.predict_proba(X_te)[:, 1]

    # Metrics
    auc_tr = roc_auc_score(y_tr, p_tr) if len(np.unique(y_tr)) > 1 else float("nan")
    auc_va = roc_auc_score(y_va, p_va) if len(np.unique(y_va)) > 1 else float("nan")
    auc_te = roc_auc_score(y_te, p_te) if len(np.unique(y_te)) > 1 else float("nan")
    ap_va = average_precision_score(y_va, p_va) if len(np.unique(y_va)) > 1 else float("nan")

    # Threshold by expected return on validation
    thr = select_best_threshold(p_va, yr_va)

    # Test evaluation
    long_mask_te = p_te >= thr
    coverage = float(long_mask_te.mean())
    avg_ret = float(yr_te[long_mask_te].mean()) if long_mask_te.any() else float("nan")
    cum_ret = float(yr_te[long_mask_te].sum()) if long_mask_te.any() else 0.0

    metrics = {
        "auc": {"train": auc_tr, "val": auc_va, "test": auc_te},
        "ap_val": ap_va,
        "threshold": thr,
        "test": {"coverage": coverage, "avg_y_reg": avg_ret, "cum_y_reg": cum_ret},
        "n": {"train": int(X_tr.shape[0]), "val": int(X_va.shape[0]), "test": int(X_te.shape[0])},
        "features": cols,
    }

    # Prepare data for ROC curve plotting
    y_sets = {'train': y_tr, 'val': y_va, 'test': y_te}
    p_sets = {'train': p_tr, 'val': p_va, 'test': p_te}
    auc_scores = {'train': auc_tr, 'val': auc_va, 'test': auc_te}

    # Default to backend/agents/ml/models/model_TIMESTAMP directory
    if args.model_dir:
        model_dir = args.model_dir
    else:
        # Get the path to backend/agents/ml/models/model_TIMESTAMP (script is now in training/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ml_dir = os.path.dirname(script_dir)  # Go up one level from training/ to ml/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(ml_dir, "models", f"model_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Plot and save ROC curves
    roc_plot_path = os.path.join(model_dir, "roc_curves.png")
    plot_roc_curves(y_sets, p_sets, auc_scores, roc_plot_path)
    
    # Save model and metrics
    joblib.dump({"model": clf, "features": cols, "threshold": thr, "scaler_json": scaler_json}, os.path.join(model_dir, "logreg.joblib"))
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save test predictions
    preds = pd.DataFrame({
        "p": p_te,
        "y_cls": y_te,
        "y_reg": yr_te,
        "decision": (p_te >= thr).astype(int)
    }, index=df_te.index)
    preds.to_csv(os.path.join(model_dir, "test_predictions.csv"))

    print({
        "model_dir": model_dir, 
        "metrics": os.path.join(model_dir, "metrics.json"),
        "roc_curves": roc_plot_path
    })


if __name__ == "__main__":
    main()
