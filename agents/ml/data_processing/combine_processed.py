#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Paths
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))
DEFAULT_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "agents/ml/data/processed")
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "agents/ml/data")

LABEL_FILENAMES = [
    "labels_capped_cleaned.csv",
    "labels_capped.csv",
    "labels.csv",
]


def discover_label_files(processed_dir: str, symbols: Optional[List[str]], timeframes: Optional[List[str]]) -> List[Tuple[str, str, str]]:
    """Return list of (symbol, timeframe, file_path) for available labeled datasets."""
    out: List[Tuple[str, str, str]] = []

    # Pattern: processed/symbol=XXX/timeframe=YYY/<labels*.csv>
    sym_dirs = [d for d in os.listdir(processed_dir) if d.startswith("symbol=")]
    for sym_dir in sym_dirs:
        symbol = sym_dir.split("=", 1)[1]
        if symbols and symbol not in symbols:
            continue
        tf_root = os.path.join(processed_dir, sym_dir)
        tf_dirs = [d for d in os.listdir(tf_root) if d.startswith("timeframe=")]
        for tf_dir in tf_dirs:
            tf = tf_dir.split("=", 1)[1]
            if timeframes and tf not in timeframes:
                continue
            leaf = os.path.join(tf_root, tf_dir)
            # choose best label file
            chosen = None
            for name in LABEL_FILENAMES:
                cand = os.path.join(leaf, name)
                if os.path.exists(cand):
                    chosen = cand
                    break
            if chosen:
                out.append((symbol, tf, chosen))
    return out


def read_labeled_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    df = df.sort_index()
    return df


def consolidate(processed_dir: str, symbols: Optional[List[str]], timeframes: Optional[List[str]]) -> Tuple[pd.DataFrame, Dict]:
    files = discover_label_files(processed_dir, symbols, timeframes)
    if not files:
        raise FileNotFoundError("No labeled datasets found under processed directory")

    frames: List[pd.DataFrame] = []
    meta_entries: List[Dict] = []
    for symbol, tf, path in files:
        try:
            df = read_labeled_csv(path)
            if df.empty:
                continue
            df["symbol"] = symbol
            df["timeframe"] = tf
            frames.append(df)
            meta_entries.append({
                "symbol": symbol,
                "timeframe": tf,
                "rows": int(len(df)),
                "file": os.path.relpath(path, PROJECT_ROOT),
            })
        except Exception as e:
            meta_entries.append({
                "symbol": symbol,
                "timeframe": tf,
                "error": str(e),
                "file": os.path.relpath(path, PROJECT_ROOT),
            })
            continue

    if not frames:
        raise RuntimeError("All labeled datasets failed to load or were empty")

    combined = pd.concat(frames).sort_index()
    coverage = (
        combined.groupby(["symbol", "timeframe"]).size().reset_index(name="rows")
    )

    meta = {
        "processed_dir": os.path.relpath(processed_dir, PROJECT_ROOT),
        "included": meta_entries,
        "coverage": coverage.to_dict(orient="records"),
        "columns": list(combined.columns),
        "rows": int(len(combined)),
        "date_range": {
            "start": combined.index.min().isoformat() if len(combined) else None,
            "end": combined.index.max().isoformat() if len(combined) else None,
        },
    }
    return combined, meta


def main():
    ap = argparse.ArgumentParser(description="Combine processed per-symbol/timeframe labeled datasets into a single combined_raw.csv")
    ap.add_argument("--processed_dir", default=DEFAULT_PROCESSED_DIR, help="Root of processed data (symbol=*/timeframe=*/labels*.csv)")
    ap.add_argument("--symbols", nargs="*", default=None, help="Optional subset of symbols to include")
    ap.add_argument("--timeframes", nargs="*", default=None, help="Optional subset of timeframes to include (e.g., 5m 15m 1h 1d)")
    ap.add_argument("--out_root", default=DEFAULT_DATA_DIR, help="Root output directory (default: agents/ml/data)")
    ap.add_argument("--run_dir", default=None, help="Optional explicit run directory name (default: data_YYYYMMDD_HHMMSS)")
    args = ap.parse_args()

    # Create run directory under out_root
    if args.run_dir:
        run_dir_name = args.run_dir
    else:
        run_dir_name = f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_dir = os.path.join(args.out_root, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)

    # Consolidate
    combined, meta = consolidate(args.processed_dir, args.symbols, args.timeframes)

    combined_raw_csv = os.path.join(run_dir, "combined_raw.csv")
    combined.to_csv(combined_raw_csv)

    meta_path = os.path.join(run_dir, "combine_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Quick summary to stdout
    syms = sorted(combined["symbol"].unique())
    tfs = sorted(combined["timeframe"].unique())
    print({
        "run_dir": os.path.relpath(run_dir, PROJECT_ROOT),
        "combined_raw": os.path.relpath(combined_raw_csv, PROJECT_ROOT),
        "rows": int(len(combined)),
        "symbols": syms,
        "timeframes": tfs,
    })


if __name__ == "__main__":
    main()
