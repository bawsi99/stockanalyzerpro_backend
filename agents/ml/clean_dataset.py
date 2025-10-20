#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

EXCLUDE_COLS = {"y_cls", "y_reg", "open", "high", "low", "close", "volume"}


def read_labeled(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    df = df.sort_index()
    return df


def time_split_index(n: int, train_frac=0.7, val_frac=0.15) -> Tuple[slice, slice, slice]:
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)


def cap_feature(train_s: pd.Series, full_s: pd.Series, p_low: float, p_high: float) -> Tuple[pd.Series, Dict[str, float]]:
    lo = float(train_s.quantile(p_low))
    hi = float(train_s.quantile(p_high))
    return full_s.clip(lower=lo, upper=hi), {"lo": lo, "hi": hi}


def main():
    ap = argparse.ArgumentParser(description="Drop unstable features, optional column drops, log-transform, and tighter cap selected features")
    ap.add_argument("input_csv", help="Path to labeled CSV (capped or not)")
    ap.add_argument("--output_csv", default=None, help="Output path; default *_cleaned.csv next to input")
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--drop", nargs="*", default=None, help="Additional columns to drop before transforms (e.g., vol_cv_20_log)")
    args = ap.parse_args()

    df = read_labeled(args.input_csv)
    n = len(df)
    idx_tr, idx_va, idx_te = time_split_index(n, args.train_frac, args.val_frac)
    df_tr = df.iloc[idx_tr]

    # 1) Drop unstable features if present
    to_drop = ["ret_skew_60", "ret_kurt_60"]
    if args.drop:
        to_drop.extend(args.drop)
    actually_dropped = []
    for c in to_drop:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
            actually_dropped.append(c)

    # 2) Log-transform vol_cv_20 -> vol_cv_20_log and drop original
    if "vol_cv_20" in df.columns:
        s = df["vol_cv_20"].astype(float).clip(lower=0)
        df["vol_cv_20_log"] = np.log1p(s)
        df.drop(columns=["vol_cv_20"], inplace=True)

    # 3) Tighter capping for atr_14_pct and vol_cv_20_log (2–98% train quantiles)
    bounds: Dict[str, Dict[str, float]] = {}
    df_tr = df.iloc[idx_tr]  # refresh train slice after transformations
    for c in ["atr_14_pct", "vol_cv_20_log"]:
        if c in df.columns:
            capped_s, b = cap_feature(df_tr[c].astype(float), df[c].astype(float), 0.02, 0.98)
            df[c] = capped_s
            bounds[c] = b

    out_csv = args.output_csv or os.path.splitext(args.input_csv)[0] + "_cleaned.csv"
    df.to_csv(out_csv)

    bounds_path = os.path.join(os.path.dirname(out_csv), "cleaning_bounds.json")
    with open(bounds_path, "w") as f:
        json.dump(bounds, f, indent=2)

    print({
        "rows": len(df),
        "cols": int(df.shape[1]),
        "output_csv": out_csv,
        "dropped": actually_dropped,
        "log_feature": "vol_cv_20_log" if "vol_cv_20_log" in df.columns else None,
        "tight_bounds": bounds_path,
    })


if __name__ == "__main__":
    main()
