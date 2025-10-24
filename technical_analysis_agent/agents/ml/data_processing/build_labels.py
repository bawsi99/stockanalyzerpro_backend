#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import numpy as np
import pandas as pd

# Ensure project root is on sys.path for 'backend' package imports when run as a script
import os, sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import timeframe spec/costs from project config
from backend.agents.ml.config.config import ml_defaults, TimeframeSpec

EPS = 1e-12


def infer_timeframe_from_path(path: str) -> str | None:
    m = re.search(r"timeframe=([^/]+)", path)
    return m.group(1) if m else None


def get_spec(tf_key: str) -> TimeframeSpec:
    specs = ml_defaults["timeframes"]
    if tf_key not in specs:
        raise ValueError(f"Unknown timeframe key: {tf_key}. Valid: {list(specs.keys())}")
    return specs[tf_key]


def read_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    # Ensure monotonic datetime index
    df = df.sort_index()
    return df


def make_labels(df: pd.DataFrame, horizon_bars: int, est_cost_bps: float) -> pd.DataFrame:
    out = df.copy()

    # Use log-return for stability
    close = out["close"].astype(float).clip(lower=EPS)
    fwd_price = close.shift(-horizon_bars)
    log_ret = np.log(fwd_price / close)

    # Convert costs (bps) to approximate log return subtraction
    # For small x, log(1 - x) ≈ -x. bps -> decimal
    cost = (est_cost_bps or 0.0) / 10000.0
    y_reg = log_ret - cost
    y_cls = (y_reg > 0.0).astype(int)

    out["y_reg"] = y_reg
    out["y_cls"] = y_cls

    # Drop rows with NaN labels at the tail
    out = out.iloc[:-horizon_bars] if horizon_bars > 0 else out
    return out


def derive_default_output(input_path: str) -> str:
    # features.csv -> labels.csv in same folder
    if input_path.endswith("features.csv"):
        return input_path.replace("features.csv", "labels.csv")
    root, _ = os.path.splitext(input_path)
    return f"{root}_labels.csv"


def main():
    p = argparse.ArgumentParser(description="Append labels y_cls/y_reg to features CSV using timeframe spec")
    p.add_argument("features_csv", help="Path to features CSV (e.g., .../data/processed/symbol=RELIANCE/timeframe=15m/features.csv)")
    p.add_argument("--timeframe", default=None, help="Timeframe key (5m,15m,1h,1d). If omitted, inferred from path")
    p.add_argument("--output_csv", default=None, help="Optional output CSV path; default: labels.csv next to input")
    args = p.parse_args()

    tf = args.timeframe or infer_timeframe_from_path(args.features_csv) or "1d"
    spec = get_spec(tf)

    df = read_features(args.features_csv)
    labeled = make_labels(df, horizon_bars=spec.horizon_bars, est_cost_bps=spec.est_cost_bps)

    out_path = args.output_csv or derive_default_output(args.features_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    labeled.to_csv(out_path)
    print({"wrote": len(labeled), "timeframe": tf, "horizon_bars": spec.horizon_bars, "output_csv": out_path})


if __name__ == "__main__":
    main()
