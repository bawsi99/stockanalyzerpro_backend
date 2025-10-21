#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Default features to cap due to drift/tails
DEFAULT_CAP_FEATURES = [
    "atr_14_pct",
    "vol_cv_20",
    "ret_skew_60",
    "ret_kurt_60",
    "ret_skew_20",
    "ret_kurt_20",
    "dist_sma50_pct",
    "range_pct",
    "up_down_vol_ratio_20",
    "ret_vol_corr_20",
    "cmf_20",
    "bb_bw_20",
    "atr_vol_20",
    "gap_pct",
]

EXCLUDE_COLS = {"y_cls", "y_reg", "open", "high", "low", "close", "volume"}


def read_labeled(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    df = df.sort_index()
    return df


def time_split_index(n: int, train_frac=0.7, val_frac=0.15) -> Tuple[slice, slice, slice]:
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)


def cap_with_train_quantiles(df: pd.DataFrame, features: List[str], p_low: float, p_high: float, train_idx: slice) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    capped = df.copy()
    bounds: Dict[str, Dict[str, float]] = {}

    train_df = df.iloc[train_idx]
    for c in features:
        if c not in df.columns:
            continue
        s_tr = train_df[c].astype(float)
        lo = float(s_tr.quantile(p_low))
        hi = float(s_tr.quantile(p_high))
        bounds[c] = {"lo": lo, "hi": hi}
        capped[c] = capped[c].clip(lower=lo, upper=hi)
    return capped, bounds


def derive_output(input_csv: str) -> str:
    base, ext = os.path.splitext(input_csv)
    return f"{base}_capped{ext}"


def main():
    ap = argparse.ArgumentParser(description="Cap heavy-tailed features based on train quantiles and write capped CSV")
    ap.add_argument("labels_csv", help="Path to labeled CSV")
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--p_low", type=float, default=0.01, help="Lower quantile for capping (train)")
    ap.add_argument("--p_high", type=float, default=0.99, help="Upper quantile for capping (train)")
    ap.add_argument("--features", nargs="*", default=None, help="Explicit list of features to cap")
    ap.add_argument("--output_csv", default=None, help="Output path; default: *_capped.csv next to input")
    args = ap.parse_args()

    df = read_labeled(args.labels_csv)
    n = len(df)
    idx_tr, idx_va, idx_te = time_split_index(n, args.train_frac, args.val_frac)

    features = args.features or DEFAULT_CAP_FEATURES
    # Only cap numeric, existing columns
    features = [c for c in features if c in df.columns and np.issubdtype(df[c].dtype, np.number)]

    capped, bounds = cap_with_train_quantiles(df, features, args.p_low, args.p_high, idx_tr)

    out_csv = args.output_csv or derive_output(args.labels_csv)
    capped.to_csv(out_csv)

    # Save bounds for reproducibility
    bounds_path = os.path.join(os.path.dirname(out_csv), "capping_bounds.json")
    with open(bounds_path, "w") as f:
        json.dump(bounds, f, indent=2)

    print({
        "rows": len(capped),
        "cols": int(capped.shape[1]),
        "output_csv": out_csv,
        "bounds": bounds_path,
        "capped_features": features,
        "quantiles": {"low": args.p_low, "high": args.p_high},
    })


if __name__ == "__main__":
    main()
