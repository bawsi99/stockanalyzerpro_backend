"""
Train and evaluate the pattern-based ML model on a generated event dataset.

This script expects a Parquet/CSV dataset built by dataset_builder.py with columns:
  - symbol, pattern_type, event_date, duration, y_success, volume_ratio20, ret_20, completion_status

It maps available columns to the pattern engine's core feature schema:
  duration -> duration
  volume_ratio -> volume_ratio20 (clipped)
  trend_alignment -> sign(ret_20) mapped to {-1,0,1} then scaled to numeric
  completion -> 1 if completion_status == 'completed' else 0

Usage:
  python -m backend.ml.quant_system.train_evaluate_pattern \
    --dataset backend/ml/quant_system/datasets/test_general_patterns.parquet
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Tuple
import numpy as np
import pandas as pd

# Ensure package paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from ml.quant_system.ml.pattern_ml import PatternMLEngine, PatternDataset
from ml.quant_system.ml.core import UnifiedMLConfig


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        # Try CSV fallback
        csv_path = os.path.splitext(path)[0] + ".csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found: {path}")
        return pd.read_csv(csv_path)
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    req_cols = ["pattern_type", "duration", "y_success"]
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column in dataset: {c}")

    # Map columns to core schema
    X = pd.DataFrame()
    X["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)

    # volume_ratio: prefer volume_ratio20; else fallback to 1.0
    if "volume_ratio20" in df.columns:
        vr = pd.to_numeric(df["volume_ratio20"], errors="coerce").fillna(1.0).clip(0, 10)
    else:
        vr = pd.Series(1.0, index=df.index)
    X["volume_ratio"] = vr

    # trend_alignment: sign of 20-bar return mapped to {0,1} for simplicity
    if "ret_20" in df.columns:
        r20 = pd.to_numeric(df["ret_20"], errors="coerce").fillna(0.0)
        trend = np.where(r20 > 0, 1.0, np.where(r20 < 0, 0.0, 0.5))
    else:
        trend = np.full(len(df), 0.5)
    X["trend_alignment"] = trend

    # completion: from completion_status
    if "completion_status" in df.columns:
        comp = df["completion_status"].astype(str).str.lower()
        X["completion"] = np.where(comp == "completed", 1.0, 0.0)
    else:
        X["completion"] = 0.0

    # Add categorical pattern_type
    X["pattern_type"] = df["pattern_type"].astype(str)

    # Targets
    y = pd.to_numeric(df["y_success"], errors="coerce").fillna(0).astype(int).to_numpy()

    # Meta
    meta = df[["pattern_type"]].copy()
    if "event_date" in df.columns:
        meta["timestamp"] = pd.to_datetime(df["event_date"], errors="coerce")
    else:
        meta["timestamp"] = pd.NaT

    return X, y, meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train/evaluate pattern ML on an event dataset")
    p.add_argument("--dataset", required=True)
    args = p.parse_args(argv)

    df = load_dataset(args.dataset)
    if len(df) == 0:
        print("Empty dataset; nothing to train")
        return 0

    X, y, meta = build_training_data(df)

    engine = PatternMLEngine(config=UnifiedMLConfig(catboost_iterations=400))
    ok = engine.train(data=(X, y, meta), n_splits=3)
    if not ok:
        print("Training failed or CatBoost unavailable")
        return 1

    metrics = engine.evaluate(data=(X, y, meta))
    info = engine.get_model_info()

    # Report
    print("=== DATASET ===")
    print(f"samples: {len(df)}")
    pos = int((y == 1).sum())
    print(f"positives: {pos} ({pos/len(df):.2%})")
    print("=== TRAIN METRICS ===")
    if isinstance(metrics, dict):
        for k in ["accuracy", "brier", "logloss", "n_samples"]:
            if k in metrics:
                print(f"{k}: {metrics[k]}")
    print("=== MODEL INFO ===")
    if isinstance(info, dict):
        print({k: info.get(k) for k in ["status", "trained_at", "model_path"]})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


