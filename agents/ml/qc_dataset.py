#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

HEAVY_TAIL_CANDIDATES = [
    "gap_pct",
    "range_pct",
    "atr_14_pct",
    "dist_sma50_pct",
    "vwap_dist",
]


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    df = df.sort_index()
    return df


def qc_report(df: pd.DataFrame) -> Dict:
    rep: Dict = {}

    # Index checks
    idx = df.index
    rep["index_monotonic_increasing"] = bool(idx.is_monotonic_increasing)
    dup_count = int(idx.duplicated().sum())
    rep["index_duplicate_count"] = dup_count

    # NaN/Inf checks
    nan_counts = df.isna().sum().to_dict()
    rep["nan_counts"] = {k: int(v) for k, v in nan_counts.items() if int(v) > 0}

    # Infinite values
    inf_counts = {c: int(np.isinf(df[c].values).sum()) for c in df.columns}
    rep["inf_counts"] = {k: v for k, v in inf_counts.items() if v > 0}

    # Label distribution
    if "y_cls" in df:
        vc = df["y_cls"].value_counts(dropna=False)
        total = float(vc.sum()) if len(vc) else 0.0
        rep["y_cls_distribution"] = {
            str(k): {"count": int(v), "pct": (float(v) / total) if total else 0.0}
            for k, v in vc.to_dict().items()
        }
    if "y_reg" in df:
        desc = df["y_reg"].describe(percentiles=[0.01, 0.05, 0.95, 0.99]).to_dict()
        rep["y_reg_summary"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in desc.items()}

    # Heavy-tail summaries (no winsorization here)
    tails: Dict[str, Dict[str, float]] = {}
    for c in HEAVY_TAIL_CANDIDATES:
        if c in df.columns:
            s = df[c].dropna()
            if len(s) > 0:
                tails[c] = {
                    "min": float(s.min()),
                    "p01": float(s.quantile(0.01)),
                    "p05": float(s.quantile(0.05)),
                    "p95": float(s.quantile(0.95)),
                    "p99": float(s.quantile(0.99)),
                    "max": float(s.max()),
                }
    rep["heavy_tail_candidates"] = tails

    # Basic shape
    rep["rows"] = int(len(df))
    rep["cols"] = int(df.shape[1])
    rep["start"] = df.index.min().isoformat() if len(df) else None
    rep["end"] = df.index.max().isoformat() if len(df) else None

    return rep


def main():
    ap = argparse.ArgumentParser(description="QC report for labeled/features dataset (no mutation)")
    ap.add_argument("input_csv", help="Path to CSV (e.g., .../labels.csv)")
    ap.add_argument("--output_json", default=None, help="Optional path to write QC JSON report next to CSV by default")
    args = ap.parse_args()

    df = read_csv(args.input_csv)
    rep = qc_report(df)

    out_json = args.output_json or os.path.join(os.path.dirname(args.input_csv), "qc_report.json")
    with open(out_json, "w") as f:
        json.dump(rep, f, indent=2)

    # Print concise summary
    print({
        "rows": rep["rows"],
        "cols": rep["cols"],
        "index_monotonic_increasing": rep["index_monotonic_increasing"],
        "index_duplicate_count": rep["index_duplicate_count"],
        "nan_cols": list(rep["nan_counts"].keys()),
        "y_cls_distribution": rep.get("y_cls_distribution"),
        "qc_report": out_json,
    })


if __name__ == "__main__":
    main()
