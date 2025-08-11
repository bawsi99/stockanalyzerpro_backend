#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import json
import os
import shutil
from datetime import datetime
from typing import List

# Ensure parent directory (project backend root) is on sys.path for absolute imports
CURRENT_DIR = os.path.dirname(__file__)
BACKEND_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from signals.config import save_timeframe_weights, load_timeframe_weights
from scripts.calibrate_weights import load_fixture, evaluate_fixture, run_calibration


def collect_fixtures(fixtures_dir: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(fixtures_dir):
        for fn in files:
            if fn.endswith('.json'):
                paths.append(os.path.join(root, fn))
    return paths


def evaluate_set(paths: List[str]) -> float:
    total = 0.0
    for p in paths:
        data = load_fixture(p)
        target = data.get('target_bias', 'neutral')
        total += evaluate_fixture({k: v for k, v in data.items() if k in ('minute','3minute','5minute','10minute','15minute','30minute','60minute','hour','day','week','month')}, target)
    return total / max(1, len(paths))


def main():
    parser = argparse.ArgumentParser(description='End-to-end calibration with validation and backup')
    parser.add_argument('fixtures_dir', help='Root directory of fixtures')
    parser.add_argument('--weights', default=None, help='Weights file path (default signals/weights_config.json)')
    parser.add_argument('--min_improvement', type=float, default=0.02, help='Minimal absolute improvement to accept')
    parser.add_argument('--backup_dir', default=None, help='Directory to store weights history')
    args = parser.parse_args()

    fixtures = collect_fixtures(args.fixtures_dir)
    if not fixtures:
        print('No fixtures found.')
        return

    # Train/validate split: simple chronological split by filename
    fixtures.sort()
    split = int(0.7 * len(fixtures))
    train, val = fixtures[:split], fixtures[split:]

    # If a weights path is provided, ensure the scorer uses it for baseline
    if args.weights:
        os.environ['SIGNALS_WEIGHTS_CONFIG'] = args.weights
    # Baseline score (using current weights and env override)
    baseline = evaluate_set(val)
    print('Baseline validation score:', round(baseline, 4))

    # Run calibration programmatically to avoid argparse conflicts
    best_weights = run_calibration(args.fixtures_dir, args.weights)

    # Evaluate new weights
    improved = evaluate_set(val)
    print('New validation score:', round(improved, 4))

    if improved >= baseline + args.min_improvement:
        # Accept: ensure saved into weights file and backup
        if not args.backup_dir:
            args.backup_dir = os.path.join(os.path.dirname(__file__), '..', 'signals', 'weights_history')
        os.makedirs(args.backup_dir, exist_ok=True)
        weights_path = args.weights or os.path.join(os.path.dirname(__file__), '..', 'signals', 'weights_config.json')
        # Copy current weights to history with timestamp
        if os.path.exists(weights_path):
            ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            shutil.copyfile(weights_path, os.path.join(args.backup_dir, f'weights_{ts}.json'))
        print('Calibration accepted and weights backed up.')
    else:
        print('Calibration rejected (insufficient improvement). Retaining existing weights.')


if __name__ == '__main__':
    main()


