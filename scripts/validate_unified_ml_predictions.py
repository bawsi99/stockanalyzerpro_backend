#!/usr/bin/env python3
"""
Validate unified ML predictions end-to-end on synthetic OHLCV data.
Outputs a concise JSON summary and basic pass/fail checks.
Run:
  PYTHONPATH=backend python backend/scripts/validate_unified_ml_predictions.py
"""

import os
import sys
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def make_synthetic_ohlcv(num_days: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [datetime.now() - timedelta(days=i) for i in range(num_days)][::-1]
    prices = np.cumsum(rng.normal(0, 1, size=num_days)) + 100
    prices = np.maximum(prices, 1.0)
    high = prices + rng.normal(0.5, 0.3, size=num_days)
    low = prices - rng.normal(0.5, 0.3, size=num_days)
    open_ = prices + rng.normal(0.0, 0.2, size=num_days)
    close = prices
    volume = np.abs(rng.normal(1e6, 2e5, size=num_days))
    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }, index=pd.to_datetime(dates))
    df.index.name = 'date'
    return df


def add_quant_system_to_path():
    # Add backend/ml/quant_system to sys.path so that `from ml.unified_manager ...` works
    here = os.path.abspath(os.path.dirname(__file__))
    quant_root = os.path.abspath(os.path.join(here, '..', 'ml', 'quant_system'))
    if quant_root not in sys.path:
        sys.path.insert(0, quant_root)


def main():
    add_quant_system_to_path()
    try:
        from ml.unified_manager import unified_ml_manager
    except Exception as e:
        print(f"❌ Failed to import unified ML manager: {e}")
        sys.exit(1)

    df = make_synthetic_ohlcv()

    # Train engines (best-effort) and get predictions
    try:
        _ = unified_ml_manager.train_all_engines(df, None)
    except Exception as e:
        print(f"⚠️ Training error (continuing): {e}")

    try:
        preds = unified_ml_manager.get_comprehensive_prediction(df)
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        sys.exit(1)

    # Basic assertions
    passed = True
    if not isinstance(preds, dict) or not preds:
        print("❌ Predictions are empty or not a dict")
        passed = False
    consensus = preds.get('consensus') if isinstance(preds, dict) else None
    if not consensus or 'overall_signal' not in consensus:
        print("❌ Missing ML consensus in predictions")
        passed = False

    # Print compact summary
    print("\n✅ Unified ML predictions summary:")
    try:
        print(json.dumps({
            'has_raw_data_ml': 'raw_data_ml' in preds,
            'has_pattern_ml': 'pattern_ml' in preds,
            'has_hybrid_ml': 'hybrid_ml' in preds,
            'consensus': preds.get('consensus', {}),
        }, indent=2, default=lambda o: float(o) if hasattr(o, '__float__') else str(o)))
    except Exception:
        print(str(preds)[:1000])

    if passed:
        print("\n✅ validate_unified_ml_predictions: PASS")
        sys.exit(0)
    else:
        print("\n❌ validate_unified_ml_predictions: FAIL")
        sys.exit(2)


if __name__ == "__main__":
    main()


