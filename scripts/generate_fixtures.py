#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import json
import os
from datetime import timedelta
from typing import Dict, List, Any

import pandas as pd

# Ensure parent directory (project backend root) is on sys.path for absolute imports
CURRENT_DIR = os.path.dirname(__file__)
BACKEND_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from agent_capabilities import StockAnalysisOrchestrator
def forward_return_label(df: pd.DataFrame, horizon_days: int, bull_thr: float, bear_thr: float) -> str:
    if len(df) < horizon_days + 1:
        return "neutral"
    start = float(df['close'].iloc[0])
    end = float(df['close'].iloc[min(horizon_days, len(df)-1)])
    ret = (end / start) - 1.0
    if ret >= bull_thr:
        return "bullish"
    if ret <= bear_thr:
        return "bearish"
    return "neutral"


def to_json_safe(obj: Any):
    import numpy as np
    import pandas as pd
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if np.isnan(val) or np.isinf(val) else val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Series,)):
        return [to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (pd.DataFrame,)):
        return obj.to_dict(orient='list')
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(v) for v in obj]
    try:
        return float(obj)
    except Exception:
        return str(obj)


def main():
    parser = argparse.ArgumentParser(description="Generate fixtures with auto-labels")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g., RELIANCE,TCS")
    parser.add_argument("--period", type=int, default=365*3, help="Days of history to fetch")
    parser.add_argument("--interval", default="day", help="Base interval")
    parser.add_argument("--horizon", type=int, default=10, help="Forward horizon in days for labeling")
    parser.add_argument("--bull", type=float, default=0.02, help="Bullish threshold (e.g., 0.02=+2%%)")
    parser.add_argument("--bear", type=float, default=-0.02, help="Bearish threshold (e.g., -0.02=-2%%)")
    parser.add_argument("--stride", type=int, default=10, help="Step days between samples")
    parser.add_argument("--out", required=True, help="Output directory for fixtures")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    orchestrator = StockAnalysisOrchestrator()

    # Known index symbol mapping to Zerodha format
    index_symbol_map = {
        "NIFTY_50": "NIFTY 50",
        "NIFTY_BANK": "NIFTY BANK",
        "NIFTY_IT": "NIFTY IT",
        "NIFTY_PHARMA": "NIFTY PHARMA",
        "NIFTY_AUTO": "NIFTY AUTO",
        "NIFTY_FMCG": "NIFTY FMCG",
        "NIFTY_ENERGY": "NIFTY ENERGY",
        "NIFTY_METAL": "NIFTY METAL",
        "NIFTY_REALTY": "NIFTY REALTY",
        "NIFTY_MEDIA": "NIFTY MEDIA",
        "NIFTY_CONSUMER_DURABLES": "NIFTY CONSR DURBL",
        "NIFTY_HEALTHCARE": "NIFTY HEALTHCARE",
        "NIFTY_INFRA": "NIFTY INFRA",
        "NIFTY_OIL_GAS": "NIFTY OIL AND GAS",
        "NIFTY_SERV_SECTOR": "NIFTY SERV SECTOR",
    }

    # Normalize symbol list and drop empties (handles trailing commas/spaces)
    symbol_list = [s.strip() for s in args.symbols.split(',') if s.strip()]

    # Prefetch shared benchmarks once
    prefetch = {}
    try:
        from technical_indicators import TechnicalIndicators as _TI
        import asyncio as _asyncio
        _ti = _TI()
        prefetch["NIFTY_50"] = _asyncio.run(_ti.get_nifty_50_data_async(365))
        prefetch["INDIA_VIX"] = _asyncio.run(_ti.get_india_vix_data_async(30))
    except Exception:
        prefetch = {}

    for sym in symbol_list:
        req_symbol = index_symbol_map.get(sym, sym.replace('_', ' '))
        # Retrieve data synchronously via orchestrator
        import asyncio
        try:
            data = asyncio.run(orchestrator.retrieve_stock_data(req_symbol, "NSE", args.interval, args.period))
        except Exception:
            continue

        if data is None or data.empty:
            continue
        data = data.sort_index()

        # Iterate with walk-forward windows
        idx = 0
        dates = list(data.index)
        while idx + args.horizon < len(dates):
            t = dates[idx]
            window = data.loc[:t]
            # compute indicators optimized for the snapshot
            from technical_indicators import TechnicalIndicators
            inds = TechnicalIndicators.calculate_all_indicators_optimized(window, prefetch=prefetch)
            # Reduce to keys used by scoring and sanitize
            allowed_keys = {
                'rsi', 'macd', 'moving_averages', 'adx', 'bollinger_bands', 'volume',
                'enhanced_momentum', 'cmf', 'ichimoku', 'keltner', 'supertrend'
            }
            reduced: Dict[str, Any] = {k: inds.get(k) for k in allowed_keys if k in inds}
            safe_inds = to_json_safe(reduced)
            # Build per-timeframe dict using base snapshot only
            per_tf = {args.interval: safe_inds}
            # Label from future horizon
            future_slice = data.loc[t:]
            target = forward_return_label(future_slice.reset_index(drop=True), args.horizon, args.bull, args.bear)

            fixture = per_tf
            fixture["target_bias"] = target
            out_path = os.path.join(args.out, f"{sym}_{t.strftime('%Y%m%d')}.json")
            # Atomic write to avoid partial/invalid JSON
            import tempfile
            with tempfile.NamedTemporaryFile('w', delete=False, dir=args.out, suffix='.json') as tmpf:
                json.dump(fixture, tmpf, indent=2)
                tmp_name = tmpf.name
            os.replace(tmp_name, out_path)
            idx += args.stride

    print("Fixtures generated in", args.out)


if __name__ == "__main__":
    main()


