#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

EXCLUDE_COLS = {"y_cls", "y_reg", "open", "high", "low", "close", "volume"}


def read_labeled(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    df = df.sort_index()
    # force numeric where possible
    for c in list(df.columns):
        if not np.issubdtype(df[c].dtype, np.number):
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            except Exception:
                pass
    return df


def time_split_index(n: int, train_frac=0.7, val_frac=0.15) -> Tuple[slice, slice, slice]:
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)


def select_features(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    # Empirical KS (two-sample), no SciPy
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    data = np.sort(np.unique(np.concatenate([a, b])))
    # ECDFs at data points
    a_cdf = np.searchsorted(np.sort(a), data, side="right") / len(a)
    b_cdf = np.searchsorted(np.sort(b), data, side="right") / len(b)
    return float(np.max(np.abs(a_cdf - b_cdf)))


def psi(train: np.ndarray, test: np.ndarray, bins: int = 10, eps: float = 1e-9) -> float:
    # PSI using train quantile bins
    x = train[~np.isnan(train)]
    y = test[~np.isnan(test)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(x, qs))
    # if too few unique edges, fallback to min/max
    if len(edges) < 3:
        lo = np.nanmin(x)
        hi = np.nanmax(x)
        edges = np.linspace(lo, hi, bins + 1)
    # bin counts
    x_counts, _ = np.histogram(x, bins=edges)
    y_counts, _ = np.histogram(y, bins=edges)
    x_prop = x_counts / max(1, x_counts.sum()) + eps
    y_prop = y_counts / max(1, y_counts.sum()) + eps
    return float(np.sum((x_prop - y_prop) * np.log(x_prop / y_prop)))


def summarize_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for c in df.columns:
        s = df[c]
        if not np.issubdtype(s.dtype, np.number):
            continue
        q = s.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
        stats[c] = {
            "mean": float(np.nanmean(s)),
            "std": float(np.nanstd(s, ddof=0)),
            "min": float(np.nanmin(s)),
            "p01": float(q.get(0.01, np.nan)),
            "p05": float(q.get(0.05, np.nan)),
            "p50": float(q.get(0.5, np.nan)),
            "p95": float(q.get(0.95, np.nan)),
            "p99": float(q.get(0.99, np.nan)),
            "max": float(np.nanmax(s)),
        }
    return stats


def main():
    ap = argparse.ArgumentParser(description="Time split + drift QC (KS/PSI), train correlation, and schema stats")
    ap.add_argument("labels_csv", help="Path to labeled CSV")
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--out_dir", default=None, help="Output dir (default: alongside input)")
    args = ap.parse_args()

    df = read_labeled(args.labels_csv)
    n = len(df)
    idx_tr, idx_va, idx_te = time_split_index(n, args.train_frac, args.val_frac)

    df_tr = df.iloc[idx_tr]
    df_te = df.iloc[idx_te]

    feature_cols = select_features(df)
    X_tr = df_tr[feature_cols]
    X_te = df_te[feature_cols]

    # KS/PSI per feature
    ks: Dict[str, float] = {}
    psi_vals: Dict[str, float] = {}
    for c in feature_cols:
        a = X_tr[c].values.astype(float)
        b = X_te[c].values.astype(float)
        ks[c] = ks_statistic(a, b)
        psi_vals[c] = psi(a, b)

    # Correlation on train only
    corr_train = X_tr.corr(method="pearson").round(2)

    # Schema stats on train (for serving)
    schema = summarize_stats(X_tr)

    out_dir = args.out_dir or os.path.dirname(args.labels_csv)
    os.makedirs(out_dir, exist_ok=True)

    corr_path = os.path.join(out_dir, "train_correlation_matrix.csv")
    corr_train.to_csv(corr_path)

    drift_report = {
        "rows": {"train": int(len(df_tr)), "test": int(len(df_te))},
        "ks": ks,
        "psi": psi_vals,
        "top_ks": sorted(((k, v) for k, v in ks.items() if not np.isnan(v)), key=lambda x: x[1], reverse=True)[:15],
        "top_psi": sorted(((k, v) for k, v in psi_vals.items() if not np.isnan(v)), key=lambda x: x[1], reverse=True)[:15],
    }
    drift_path = os.path.join(out_dir, "drift_report.json")
    with open(drift_path, "w") as f:
        json.dump(drift_report, f, indent=2)

    schema_path = os.path.join(out_dir, "schema_stats.json")
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)

    print({
        "train_rows": len(df_tr),
        "test_rows": len(df_te),
        "train_corr": corr_path,
        "drift_report": drift_path,
        "schema_stats": schema_path,
        "top_ks": drift_report["top_ks"],
        "top_psi": drift_report["top_psi"],
    })


if __name__ == "__main__":
    main()
