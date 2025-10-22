#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

HEAVY_TAIL_CANDS = [
    "gap_pct",
    "range_pct",
    "atr_14_pct",
    "dist_sma50_pct",
    "vwap_dist",
]

META_COLS = {"symbol", "timeframe", "y_cls", "y_reg", "open", "high", "low", "close", "volume", "dow", "hour"}


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    return df.sort_index()


def qc_combined(
    df: pd.DataFrame,
    max_row_nan_pct: float = 0.15,
    min_rows_per_group: int = 200,
    outlier_method: str = "iqr",
    outlier_factor: float = 3.0,
) -> (pd.DataFrame, Dict):
    rep: Dict = {}

    # Initial stats
    rep["initial_rows"] = int(len(df))
    rep["initial_cols"] = int(df.shape[1])
    rep["symbols"] = sorted(df["symbol"].unique()) if "symbol" in df.columns else []
    rep["timeframes"] = sorted(df["timeframe"].unique()) if "timeframe" in df.columns else []

    # Drop duplicate index rows (keep last)
    dup_count = int(df.index.duplicated().sum())
    rep["index_duplicate_count"] = dup_count
    if dup_count:
        df = df[~df.index.duplicated(keep="last")]

    # Remove rows with too many NaNs across feature columns
    feature_cols = [c for c in df.columns if c not in META_COLS]
    if feature_cols:
        row_nan_pct = df[feature_cols].isna().mean(axis=1)
        mask = row_nan_pct <= max_row_nan_pct
        rep["rows_removed_nan_pct"] = int((~mask).sum())
        df = df[mask]

    # Remove groups with insufficient rows
    if "symbol" in df.columns and "timeframe" in df.columns:
        counts = df.groupby(["symbol", "timeframe"]).size()
        keep_groups = counts[counts >= min_rows_per_group].index
        before = len(df)
        df = df.set_index(["symbol", "timeframe"], append=True)
        # Filter via boolean mask on (symbol, timeframe)
        keep_set = set(keep_groups)
        mask = [(s, t) in keep_set for s, t in zip(df.index.get_level_values(1), df.index.get_level_values(2))]
        df = df[mask]
        df.index = df.index.droplevel([1, 2])
        rep["rows_removed_small_groups"] = int(before - len(df))
        rep["groups_kept"] = [
            {"symbol": s, "timeframe": t, "rows": int(counts[(s, t)])} for s, t in keep_groups
        ]

    # Outlier filtering (IQR) on numeric features
    removed_outliers = 0
    if outlier_method == "iqr":
        for c in feature_cols:
            if not np.issubdtype(df[c].dtype, np.number):
                continue
            s = df[c].dropna()
            if len(s) < 50:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr == 0:
                continue
            lo, hi = q1 - outlier_factor * iqr, q3 + outlier_factor * iqr
            mask = (df[c] >= lo) & (df[c] <= hi)
            removed_outliers += int((~mask).sum())
            df = df[mask]
    rep["rows_removed_outliers"] = removed_outliers

    # Inf check
    inf_counts = {c: int(np.isinf(pd.to_numeric(df[c], errors="coerce")).sum()) for c in feature_cols}
    rep["inf_counts"] = {k: v for k, v in inf_counts.items() if v > 0}

    # Label balance
    if "y_cls" in df.columns:
        yvc = df["y_cls"].value_counts(dropna=False)
        total = float(yvc.sum())
        rep["y_cls_distribution"] = {str(int(k)): {"count": int(v), "pct": float(v) / total} for k, v in yvc.items()}

    rep["final_rows"] = int(len(df))
    rep["final_cols"] = int(df.shape[1])
    rep["date_range"] = {
        "start": df.index.min().isoformat() if len(df) else None,
        "end": df.index.max().isoformat() if len(df) else None,
    }
    return df, rep


def main():
    ap = argparse.ArgumentParser(description="QC on combined_raw.csv before splitting")
    ap.add_argument("--input_csv", required=True, help="Path to combined_raw.csv")
    ap.add_argument("--out_csv", default=None, help="Output path for combined_qc.csv (default: alongside input)")
    ap.add_argument("--report_json", default=None, help="Path for QC report JSON (default: alongside input)")
    ap.add_argument("--max_row_nan_pct", type=float, default=0.15)
    ap.add_argument("--min_rows_per_group", type=int, default=200)
    ap.add_argument("--outlier_method", default="iqr", choices=["none", "iqr"])
    ap.add_argument("--outlier_factor", type=float, default=3.0)
    args = ap.parse_args()

    df = read_csv(args.input_csv)
    out_df, report = qc_combined(
        df,
        max_row_nan_pct=args.max_row_nan_pct,
        min_rows_per_group=args.min_rows_per_group,
        outlier_method=(None if args.outlier_method == "none" else args.outlier_method),
        outlier_factor=args.outlier_factor,
    )

    base_dir = os.path.dirname(args.input_csv)
    out_csv = args.out_csv or os.path.join(base_dir, "combined_qc.csv")
    report_json = args.report_json or os.path.join(base_dir, "qc_combined_report.json")

    out_df.to_csv(out_csv)
    with open(report_json, "w") as f:
        json.dump(report, f, indent=2)

    print({
        "input_rows": int(len(df)),
        "output_rows": int(len(out_df)),
        "combined_qc": out_csv,
        "report": report_json,
    })


if __name__ == "__main__":
    main()
