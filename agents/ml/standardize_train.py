#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

EXCLUDE_COLS = {"y_cls", "y_reg", "open", "high", "low", "close", "volume"}


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    return df.sort_index()


def is_binary(series: pd.Series) -> bool:
    s = series.dropna().unique()
    if len(s) == 0:
        return False
    # Consider binary if subset of {0,1}
    return set(np.unique(series.dropna().astype(float))).issubset({0.0, 1.0})


def select_numeric_features(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c in EXCLUDE_COLS:
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            # try convert
            try:
                _ = pd.to_numeric(df[c], errors="coerce")
            except Exception:
                continue
        # skip binary flags
        if is_binary(df[c]):
            continue
        cols.append(c)
    return cols


def standardize_train(train_csv: str, out_csv: str, scaler_json: str) -> Dict:
    df = read_csv(train_csv)

    # Identify feature columns to scale
    feature_cols = select_numeric_features(df)

    # Compute mean/std on train; guard std=0
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    scaled = df.copy()

    for c in feature_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        mu = float(np.nanmean(s))
        sd = float(np.nanstd(s, ddof=0))
        if not np.isfinite(sd) or sd == 0.0:
            sd = 1.0
        means[c] = mu
        stds[c] = sd
        scaled[c] = (s - mu) / sd

    # Write standardized train
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    scaled.to_csv(out_csv)

    # Persist scaler parameters for later use on val/test
    scaler = {
        "feature_cols": feature_cols,
        "means": means,
        "stds": stds,
        "exclude_cols": sorted(list(EXCLUDE_COLS)),
        "binary_excluded": [c for c in df.columns if c not in EXCLUDE_COLS and is_binary(df[c])],
        "source": os.path.abspath(train_csv),
    }
    with open(scaler_json, "w") as f:
        json.dump(scaler, f, indent=2)

    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "scaled_features": len(feature_cols),
        "out_csv": out_csv,
        "scaler_json": scaler_json,
    }


def main():
    ap = argparse.ArgumentParser(description="Standardize training split (z-score), save standardized CSV and scaler JSON")
    ap.add_argument("train_csv", help="Path to splits/train.csv")
    ap.add_argument("--out_csv", default=None, help="Output path for standardized train (default: splits/train_standardized.csv)")
    ap.add_argument("--scaler_json", default=None, help="Output path for scaler JSON (default: splits/scaler.json)")
    args = ap.parse_args()

    train_csv = args.train_csv
    splits_dir = os.path.dirname(train_csv)
    out_csv = args.out_csv or os.path.join(splits_dir, "train_standardized.csv")
    scaler_json = args.scaler_json or os.path.join(splits_dir, "scaler.json")

    info = standardize_train(train_csv, out_csv, scaler_json)
    print(info)


if __name__ == "__main__":
    main()
