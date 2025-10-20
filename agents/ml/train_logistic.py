#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score


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


def main():
    ap = argparse.ArgumentParser(description="Train LogisticRegression on splits with standardization")
    ap.add_argument("--splits_dir", required=True, help="Directory containing train.csv/val.csv/test.csv and scaler.json")
    ap.add_argument("--model_dir", default=None, help="Output directory for model artifacts (default: <splits_dir>/model)")
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

    model_dir = args.model_dir or os.path.join(splits, "model")
    os.makedirs(model_dir, exist_ok=True)
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

    print({"model_dir": model_dir, "metrics": os.path.join(model_dir, "metrics.json")})


if __name__ == "__main__":
    main()
