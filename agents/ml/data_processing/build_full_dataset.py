#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import importlib.util
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

# Ensure project root is on sys.path when run as a script
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Reuse existing modules with robust fallback if package import fails
try:
    from backend.agents.ml.data_processing import build_dataset as bd
    from backend.agents.ml.data_processing import build_labels as bl
except ModuleNotFoundError:
    # Fallback: import by file paths to avoid package import issues
    this_dir = os.path.dirname(os.path.abspath(__file__))  # .../data_processing

    # build_dataset.py
    bd_path = os.path.join(this_dir, "build_dataset.py")
    spec_bd = importlib.util.spec_from_file_location("bd", bd_path)
    bd = importlib.util.module_from_spec(spec_bd)
    assert spec_bd and spec_bd.loader
    sys.modules[spec_bd.name] = bd
    spec_bd.loader.exec_module(bd)

    # Load config for labeling logic
    ml_dir = os.path.dirname(this_dir)  # .../ml
    cfg_path = os.path.join(ml_dir, "config", "config.py")
    spec_cfg = importlib.util.spec_from_file_location("cfg", cfg_path)
    cfg = importlib.util.module_from_spec(spec_cfg)
    assert spec_cfg and spec_cfg.loader
    sys.modules[spec_cfg.name] = cfg
    spec_cfg.loader.exec_module(cfg)

    # Provide minimal replacements for build_labels API
    import re as _re
    EPS = 1e-12

    class _BL:
        @staticmethod
        def infer_timeframe_from_path(path: str) -> str | None:
            m = _re.search(r"timeframe=([^/]+)", path)
            return m.group(1) if m else None

        @staticmethod
        def get_spec(tf_key: str):
            specs = cfg.ml_defaults["timeframes"]
            if tf_key not in specs:
                raise ValueError(f"Unknown timeframe key: {tf_key}. Valid: {list(specs.keys())}")
            return specs[tf_key]

        @staticmethod
        def derive_default_output(input_path: str) -> str:
            if input_path.endswith("features.csv"):
                return input_path.replace("features.csv", "labels.csv")
            root, _ = os.path.splitext(input_path)
            return f"{root}_labels.csv"

        @staticmethod
        def make_labels(df: pd.DataFrame, horizon_bars: int, est_cost_bps: float) -> pd.DataFrame:
            out = df.copy()
            close = out["close"].astype(float).clip(lower=EPS)
            fwd_price = close.shift(-horizon_bars)
            log_ret = np.log(fwd_price / close)
            cost = (est_cost_bps or 0.0) / 10000.0
            y_reg = log_ret - cost
            y_cls = (y_reg > 0.0).astype(int)
            out["y_reg"] = y_reg
            out["y_cls"] = y_cls
            out = out.iloc[:-horizon_bars] if horizon_bars > 0 else out
            return out

    bl = _BL  # assign replacement namespace

EPS = 1e-12


def time_split_index(n: int, train_frac=0.7, val_frac=0.15) -> Tuple[slice, slice, slice]:
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)


def cap_feature(train_s: pd.Series, full_s: pd.Series, p_low: float, p_high: float) -> Tuple[pd.Series, Dict[str, float]]:
    lo = float(train_s.quantile(p_low))
    hi = float(train_s.quantile(p_high))
    return full_s.clip(lower=lo, upper=hi), {"lo": lo, "hi": hi}


def derive_pipeline_outputs(input_csv: str, output_csv: Optional[str]) -> Tuple[str, str, str]:
    """Return (features_csv, labels_csv, cleaned_csv) following project conventions unless overridden.

    Cleaned CSV defaults to data.archive/processed/.../labels_capped_cleaned.csv mirroring the raw path structure.
    """
    features_csv = bd.derive_default_output(input_csv)
    labels_csv = bl.derive_default_output(features_csv)

    if output_csv:
        cleaned_csv = output_csv
    else:
        in_dir = os.path.dirname(input_csv)
        if "/data.archive/raw/" in in_dir:
            proc_dir = in_dir.replace("/data.archive/raw/", "/data.archive/processed/")
            cleaned_csv = os.path.join(proc_dir, "labels_capped_cleaned.csv")
        else:
            # Fallback: place next to input
            root, _ = os.path.splitext(input_csv)
            cleaned_csv = f"{root}_labels_capped_cleaned.csv"

    return features_csv, labels_csv, cleaned_csv


def run_pipeline(
    input_csv: str,
    timeframe: Optional[str] = None,
    output_csv: Optional[str] = None,
    drop_warmup: bool = False,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    extra_drop: Optional[List[str]] = None,
) -> Dict[str, object]:
    # 1) Build features from raw OHLCV
    raw_df = bd.read_raw_csv(input_csv)
    features = bd.add_features(raw_df)
    if drop_warmup:
        features = features.dropna()

    # 2) Append labels using timeframe spec
    tf = timeframe or bl.infer_timeframe_from_path(input_csv) or "1d"
    spec = bl.get_spec(tf)
    labeled = bl.make_labels(features, horizon_bars=spec.horizon_bars, est_cost_bps=spec.est_cost_bps)

    # 3) Clean dataset (transform + tighter caps)
    n = len(labeled)
    idx_tr, idx_va, idx_te = time_split_index(n, train_frac, val_frac)

    df = labeled.copy()

    # Drop unstable or user-specified features if present
    to_drop = ["ret_skew_60", "ret_kurt_60"]
    if extra_drop:
        to_drop.extend(extra_drop)
    actually_dropped = []
    for c in to_drop:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
            actually_dropped.append(c)

    # Log-transform vol_cv_20 -> vol_cv_20_log and drop original
    if "vol_cv_20" in df.columns:
        s = df["vol_cv_20"].astype(float).clip(lower=0)
        df["vol_cv_20_log"] = np.log1p(s)
        df.drop(columns=["vol_cv_20"], inplace=True)

    # Tighter capping for atr_14_pct and vol_cv_20_log (2–98% train quantiles)
    bounds: Dict[str, Dict[str, float]] = {}
    df_tr = df.iloc[idx_tr]
    for c in ["atr_14_pct", "vol_cv_20_log"]:
        if c in df.columns:
            capped_s, b = cap_feature(df_tr[c].astype(float), df[c].astype(float), 0.02, 0.98)
            df[c] = capped_s
            bounds[c] = b

    # 4) Write outputs following conventions
    features_csv, labels_csv, cleaned_csv = derive_pipeline_outputs(input_csv, output_csv)
    os.makedirs(os.path.dirname(cleaned_csv), exist_ok=True)

    # Save only the final cleaned dataset by default; intermediates can be saved on demand if needed
    df.to_csv(cleaned_csv)
    bounds_path = os.path.join(os.path.dirname(cleaned_csv), "cleaning_bounds.json")
    with open(bounds_path, "w") as f:
        json.dump(bounds, f, indent=2)

    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "timeframe": tf,
        "horizon_bars": int(spec.horizon_bars),
        "features_csv_default": features_csv,
        "labels_csv_default": labels_csv,
        "output_csv": cleaned_csv,
        "bounds_path": bounds_path,
        "dropped": actually_dropped,
    }


def main():
    p = argparse.ArgumentParser(description="End-to-end pipeline: raw CSV -> features -> labels -> cleaned dataset")
    p.add_argument("input_csv", help="Path to raw bars CSV")
    p.add_argument("--timeframe", default=None, help="Timeframe key (5m,15m,1h,1d). Default: infer from path or 1d")
    p.add_argument("--output_csv", default=None, help="Final output CSV path. Default: <...>/labels_cleaned.csv next to derived labels path")
    p.add_argument("--drop_warmup", action="store_true", help="Drop indicator warmup NaNs after feature build")
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--drop", nargs="*", default=None, help="Additional columns to drop before transforms")
    args = p.parse_args()

    res = run_pipeline(
        input_csv=args.input_csv,
        timeframe=args.timeframe,
        output_csv=args.output_csv,
        drop_warmup=args.drop_warmup,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        extra_drop=args.drop,
    )

    print(res)


if __name__ == "__main__":
    main()
