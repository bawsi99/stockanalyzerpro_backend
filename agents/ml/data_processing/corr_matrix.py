#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd


def read_csv_numeric(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    df = df.sort_index()
    # Keep only numeric columns
    num_cols: List[str] = []
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.number):
            num_cols.append(c)
        else:
            # try coercion
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                if np.issubdtype(df[c].dtype, np.number):
                    num_cols.append(c)
            except Exception:
                pass
    if not num_cols:
        raise ValueError("No numeric columns found for correlation analysis")
    return df[num_cols]


def derive_output_path(input_csv: str) -> str:
    # Save next to input as correlation_matrix.csv by default
    base_dir = os.path.dirname(input_csv)
    return os.path.join(base_dir, "correlation_matrix.csv")


def main():
    ap = argparse.ArgumentParser(description="Generate Pearson correlation matrix from a labeled/features CSV")
    ap.add_argument("input_csv", help="Path to CSV (e.g., .../data/processed/.../labels.csv)")
    ap.add_argument("--output_csv", default=None, help="Output CSV path for correlation matrix")
    ap.add_argument("--top", type=int, default=15, help="Number of top absolute-correlation pairs to print")
    args = ap.parse_args()

    df = read_csv_numeric(args.input_csv)
    corr = df.corr(method="pearson").round(2)

    out_path = args.output_csv or derive_output_path(args.input_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    corr.to_csv(out_path)

    # Print a readable table of top |corr| pairs (excluding self and duplicates)
    # Use the upper triangle mask to avoid duplicate pairs
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    corr_pairs = pd.DataFrame({
        'feature_i': np.repeat(corr.columns.values, corr.shape[1])[mask.ravel()],
        'feature_j': np.tile(corr.columns.values, corr.shape[0])[mask.ravel()],
        'corr': corr.values[mask]
    })
    corr_pairs['abs_corr'] = corr_pairs['corr'].abs()
    corr_pairs = corr_pairs.sort_values('abs_corr', ascending=False).head(args.top)

    # Round for printing
    corr_pairs['corr'] = corr_pairs['corr'].round(2)
    corr_pairs['abs_corr'] = corr_pairs['abs_corr'].round(2)

    print(f"Saved correlation matrix to: {out_path}")
    print("Top correlated pairs (abs):")
    print(corr_pairs.to_string(index=False))


if __name__ == "__main__":
    main()
