#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

META_EXCLUDE = {"symbol", "timeframe", "y_cls", "y_reg", "open", "high", "low", "close", "volume", "dow", "hour"}


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    return df.sort_index()


def temporal_split(df: pd.DataFrame, train_pct: float, val_pct: float, test_pct: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(train_pct + val_pct + test_pct - 1.0) < 1e-6
    trains, vals, tests = [], [], []
    groups = df.groupby(["symbol", "timeframe"]) if {"symbol", "timeframe"}.issubset(df.columns) else [(None, df)]
    for _, g in groups:
        g = g.sort_index()
        n = len(g)
        if n < 5:
            continue
        i1 = int(n * train_pct)
        i2 = int(n * (train_pct + val_pct))
        trains.append(g.iloc[:i1])
        vals.append(g.iloc[i1:i2])
        tests.append(g.iloc[i2:])
    tr = pd.concat(trains) if trains else pd.DataFrame()
    va = pd.concat(vals) if vals else pd.DataFrame()
    te = pd.concat(tests) if tests else pd.DataFrame()
    return tr, va, te


def is_binary(s: pd.Series) -> bool:
    vals = pd.unique(s.dropna())
    if len(vals) == 0:
        return False
    try:
        fv = set(pd.to_numeric(pd.Series(vals), errors="coerce").dropna().astype(float))
        return fv.issubset({0.0, 1.0}) and len(fv) <= 2
    except Exception:
        return False


def compute_scaler(train_df: pd.DataFrame) -> Dict:
    feature_cols: List[str] = []
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    binary_excluded: List[str] = []

    for c in train_df.columns:
        if c in META_EXCLUDE:
            continue
        if not np.issubdtype(train_df[c].dtype, np.number):
            # try conversion for numeric-like
            try:
                _ = pd.to_numeric(train_df[c], errors="coerce")
            except Exception:
                continue
        if is_binary(train_df[c]):
            binary_excluded.append(c)
            continue
        feature_cols.append(c)
        s = pd.to_numeric(train_df[c], errors="coerce")
        mu = float(np.nanmean(s))
        sd = float(np.nanstd(s, ddof=0))
        if not np.isfinite(sd) or sd == 0.0:
            sd = 1.0
        means[c] = mu
        stds[c] = sd

    return {
        "feature_cols": feature_cols,
        "means": means,
        "stds": stds,
        "exclude_cols": sorted(list(META_EXCLUDE)),
        "binary_excluded": binary_excluded,
    }


def apply_scaler(df: pd.DataFrame, scaler: Dict) -> pd.DataFrame:
    out = df.copy()
    for c in scaler["feature_cols"]:
        if c in out.columns:
            s = pd.to_numeric(out[c], errors="coerce")
            mu = scaler["means"].get(c, 0.0)
            sd = scaler["stds"].get(c, 1.0) or 1.0
            out[c] = (s - mu) / sd
    return out


def main():
    ap = argparse.ArgumentParser(description="Split combined QC dataset and standardize train")
    ap.add_argument("--input_csv", required=True, help="Path to combined_qc.csv")
    ap.add_argument("--train_pct", type=float, default=0.7)
    ap.add_argument("--val_pct", type=float, default=0.15)
    ap.add_argument("--test_pct", type=float, default=0.15)
    args = ap.parse_args()

    df = read_csv(args.input_csv)
    tr, va, te = temporal_split(df, args.train_pct, args.val_pct, args.test_pct)

    base_dir = os.path.dirname(args.input_csv)
    train_csv = os.path.join(base_dir, "train.csv")
    val_csv = os.path.join(base_dir, "val.csv")
    test_csv = os.path.join(base_dir, "test.csv")

    tr.to_csv(train_csv)
    va.to_csv(val_csv)
    te.to_csv(test_csv)

    # Standardize train and save scaler
    scaler = compute_scaler(tr)
    train_std = apply_scaler(tr, scaler)
    train_std_csv = os.path.join(base_dir, "train_standardized.csv")
    train_std.to_csv(train_std_csv)

    scaler_json = os.path.join(base_dir, "scaler.json")
    with open(scaler_json, "w") as f:
        json.dump(scaler, f, indent=2)

    # Simple metadata
    meta = {
        "rows": {
            "train": int(len(tr)), "val": int(len(va)), "test": int(len(te))
        },
        "features": scaler["feature_cols"],
    }
    with open(os.path.join(base_dir, "split_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print({
        "train": train_csv,
        "val": val_csv,
        "test": test_csv,
        "train_standardized": train_std_csv,
        "scaler": scaler_json,
    })


if __name__ == "__main__":
    main()
