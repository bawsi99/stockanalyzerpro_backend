#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import csv
import json
import os
from typing import Dict, List

import pandas as pd

# Ensure parent directory (project backend root) is on sys.path for absolute imports
CURRENT_DIR = os.path.dirname(__file__)
BACKEND_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from signals.scoring import compute_signals_summary
from signals.config import save_timeframe_weights, load_timeframe_weights


def load_fixture(path: str) -> Dict[str, Dict]:
    with open(path, "r") as f:
        return json.load(f)


def evaluate_fixture(fixture: Dict[str, Dict], target_bias: str) -> float:
    summary = compute_signals_summary(fixture)
    correct = 1.0 if summary.consensus_bias == target_bias else 0.0
    # Weight by confidence to reward decisive correct predictions
    return correct * (0.5 + 0.5 * summary.confidence)


def _filter_timeframe_data(raw: Dict[str, Dict]) -> Dict[str, Dict]:
    allowed_timeframes = {
        "minute",
        "3minute",
        "5minute",
        "10minute",
        "15minute",
        "30minute",
        "60minute",
        "hour",
        "day",
        "week",
        "month",
    }
    return {tf: (data or {}) for tf, data in raw.items() if tf in allowed_timeframes and isinstance(data, dict)}


def _evaluate_with_candidate_weights(fixtures: List[str], candidate_weights: Dict[str, float]) -> float:
    import tempfile

    # Prepare a temporary weights file and point loader to it
    tmp_dir = tempfile.mkdtemp(prefix="calib_weights_")
    tmp_path = os.path.join(tmp_dir, "weights.json")
    with open(tmp_path, "w") as f:
        json.dump({"timeframe_weights": candidate_weights}, f)

    prev_env = os.environ.get("SIGNALS_WEIGHTS_CONFIG")
    os.environ["SIGNALS_WEIGHTS_CONFIG"] = tmp_path
    try:
        total = 0.0
        for fp in fixtures:
            data = load_fixture(fp)
            filtered = _filter_timeframe_data(data)
            if not filtered:
                continue
            target = data.get("target_bias") or data.get("target") or "neutral"
            total += evaluate_fixture(filtered, str(target))
        return total / max(1, len(fixtures))
    finally:
        if prev_env is not None:
            os.environ["SIGNALS_WEIGHTS_CONFIG"] = prev_env
        else:
            os.environ.pop("SIGNALS_WEIGHTS_CONFIG", None)


def run_calibration(fixtures_dir: str, output: str | None = None) -> Dict[str, float]:
    # Collect fixtures
    fixtures: List[str] = []
    for root, _, files in os.walk(fixtures_dir):
        for fn in files:
            if fn.endswith(".json"):
                fixtures.append(os.path.join(root, fn))

    if not fixtures:
        print("No fixtures found.")
        return load_timeframe_weights()

    # Candidate profiles to test
    candidates = [
        {"day": 1.0, "week": 1.2, "month": 1.4, "hour": 0.9, "15minute": 0.7},
        {"day": 1.2, "week": 1.0, "month": 1.5, "hour": 0.8, "15minute": 0.6},
        {"day": 1.0, "week": 1.0, "month": 1.0, "hour": 1.0, "15minute": 1.0},
    ]

    # Fill with defaults for missing keys
    defaults = load_timeframe_weights()
    filled_candidates: List[Dict[str, float]] = []
    for c in candidates:
        tmp = defaults.copy()
        tmp.update(c)
        filled_candidates.append(tmp)

    best_score = -1.0
    best_weights = defaults

    # Evaluate each candidate by injecting it via env-backed config
    for weights in filled_candidates:
        avg = _evaluate_with_candidate_weights(fixtures, weights)
        if avg > best_score:
            best_score = avg
            best_weights = weights

    save_timeframe_weights(best_weights, output)
    print("Saved calibrated weights to", output or "signals/weights_config.json")
    return best_weights


def main():
    parser = argparse.ArgumentParser(description="Calibrate timeframe weights using fixtures")
    parser.add_argument("fixtures_dir", help="Directory with JSON fixtures of per-timeframe indicators")
    parser.add_argument("--output", default=None, help="Output config path (weights_config.json)")
    args = parser.parse_args()
    run_calibration(args.fixtures_dir, args.output)


if __name__ == "__main__":
    main()


