#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd


def read_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    return df.sort_index()


def time_split_index(n: int, train_frac=0.7, val_frac=0.15) -> Tuple[slice, slice, slice]:
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)


def derive_out_dir(input_csv: str) -> str:
    base_dir = os.path.dirname(input_csv)
    return os.path.join(base_dir, "splits")


def main():
    ap = argparse.ArgumentParser(description="Time-based split of labeled dataset into train/val/test CSVs")
    ap.add_argument("input_csv", help="Path to labeled CSV (e.g., .../labels_final.csv)")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--out_dir", default=None, help="Directory to write splits (default: <input_dir>/splits)")
    args = ap.parse_args()

    df = read_df(args.input_csv)
    idx_tr, idx_va, idx_te = time_split_index(len(df), args.train_frac, args.val_frac)

    df_tr = df.iloc[idx_tr]
    df_va = df.iloc[idx_va]
    df_te = df.iloc[idx_te]

    out_dir = args.out_dir or derive_out_dir(args.input_csv)
    os.makedirs(out_dir, exist_ok=True)

    train_csv = os.path.join(out_dir, "train.csv")
    val_csv = os.path.join(out_dir, "val.csv")
    test_csv = os.path.join(out_dir, "test.csv")

    df_tr.to_csv(train_csv)
    df_va.to_csv(val_csv)
    df_te.to_csv(test_csv)

    manifest = {
        "input_csv": args.input_csv,
        "out_dir": out_dir,
        "rows": {"total": int(len(df)), "train": int(len(df_tr)), "val": int(len(df_va)), "test": int(len(df_te))},
        "dates": {
            "train": {"start": df_tr.index.min().isoformat() if len(df_tr) else None, "end": df_tr.index.max().isoformat() if len(df_tr) else None},
            "val":   {"start": df_va.index.min().isoformat() if len(df_va) else None, "end": df_va.index.max().isoformat() if len(df_va) else None},
            "test":  {"start": df_te.index.min().isoformat() if len(df_te) else None, "end": df_te.index.max().isoformat() if len(df_te) else None},
        },
        "files": {"train": train_csv, "val": val_csv, "test": test_csv},
        "fractions": {"train": args.train_frac, "val": args.val_frac, "test": round(1.0 - args.train_frac - args.val_frac, 6)},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print({
        "out_dir": out_dir,
        "train": train_csv,
        "val": val_csv,
        "test": test_csv,
        "rows": manifest["rows"],
    })


if __name__ == "__main__":
    main()
