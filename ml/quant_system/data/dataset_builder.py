"""
General cross-stock pattern dataset builder.

This script:
- selects a universe of symbols (via Zerodha or local CSV)
- fetches historical OHLCV (daily by default)
- detects a set of chart patterns at event time t
- labels each event for TP/SL success within a fixed horizon
- saves an event-level Parquet dataset with metadata

Usage (non-interactive example):
  python -m backend.ml.quant_system.dataset_builder \
    --symbols_file backend/ml/quant_system/zerodha_instruments.csv \
    --exchange NSE \
    --num_symbols 200 \
    --from_date 2019-01-01 \
    --to_date 2025-08-01 \
    --horizon_days 20 \
    --tp_pct 0.04 \
    --sl_pct 0.02 \
    --output backend/ml/quant_system/datasets/general_patterns.parquet

Notes:
- Requires valid Zerodha credentials for live API fetching; will fall back to symbols_file for universe.
- Safe to run repeatedly; output overwrites unless --append is provided in future revisions.
"""

from __future__ import annotations

import os
import sys
import json
import math
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
import logging

# Ensure backend is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Project root is three levels up from this file: backend/ml/quant_system/
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

try:
    from zerodha_client import ZerodhaDataClient
except Exception:  # pragma: no cover
    ZerodhaDataClient = None  # type: ignore

try:
    from patterns.recognition import PatternRecognition
    PATTERN_RECOGNITION_AVAILABLE = True
except Exception as e:  # pragma: no cover
    PatternRecognition = None
    PATTERN_RECOGNITION_AVAILABLE = False
    logging.warning(f"Pattern recognition module not available: {e}")


# ----------------------------- Config & Schema -----------------------------

@dataclass
class BuildConfig:
    exchange: str = "NSE"
    num_symbols: int = 200
    from_date: Optional[str] = None  # ISO date string
    to_date: Optional[str] = None    # ISO date string
    interval: str = "day"
    horizon_days: int = 20
    tp_pct: float = 0.04
    sl_pct: float = 0.02
    symbols_file: Optional[str] = None
    output: str = os.path.join(CURRENT_DIR, "datasets", "general_patterns.parquet")


EVENT_COLUMNS: List[str] = [
    # identifiers
    "event_id", "symbol", "exchange", "event_date", "pattern_type",
    # pattern descriptors (subset)
    "duration", "quality_score", "completion_status",
    # entry and risk params
    "entry_price", "tp_price", "sl_price", "horizon_days",
    # label
    "y_success",
    # simple context features
    "close", "volume", "volume_ratio20", "ret_5", "ret_20"
]


# ----------------------------- Utilities -----------------------------

def _parse_date(d: Optional[str]) -> Optional[datetime]:
    if d is None:
        return None
    return datetime.strptime(d, "%Y-%m-%d")


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def safe_pct_change(series: pd.Series, periods: int) -> pd.Series:
    try:
        return series.pct_change(periods)
    except Exception:
        return pd.Series([np.nan] * len(series), index=series.index)


def compute_volume_ratio20(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    rolling = df["volume"].rolling(window=20, min_periods=1).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = df["volume"] / rolling
    return ratio.replace([np.inf, -np.inf], np.nan)


def first_hit_outcome(next_df: pd.DataFrame, entry_price: float, tp_pct: float, sl_pct: float) -> int:
    """Return 1 if TP is hit before SL, 0 if SL first or none.
    Uses intrabar high/low to determine barrier hits.
    """
    if next_df.empty:
        return 0
    tp_level = entry_price * (1.0 + tp_pct)
    sl_level = entry_price * (1.0 - sl_pct)

    for _, row in next_df.iterrows():
        high = float(row.get("high", np.nan))
        low = float(row.get("low", np.nan))
        if math.isnan(high) or math.isnan(low):
            # fall back to close if needed
            close_val = float(row.get("close", np.nan))
            if not math.isnan(close_val):
                high = max(high if not math.isnan(high) else close_val, close_val)
                low = min(low if not math.isnan(low) else close_val, close_val)
        # Check order: if both are hit in same bar, assume TP-first if close >= entry (conservative bias towards success configurable)
        hit_tp = high >= tp_level
        hit_sl = low <= sl_level
        if hit_tp and not hit_sl:
            return 1
        if hit_sl and not hit_tp:
            return 0
        if hit_tp and hit_sl:
            # tie-breaker: use closest to open
            return 1 if (tp_level - row.get("open", entry_price)) <= (row.get("open", entry_price) - sl_level) else 0
    return 0


# ----------------------------- Symbol Universe -----------------------------

def load_symbols_from_csv(csv_path: str, exchange: str, limit: int) -> List[str]:
    if not os.path.exists(csv_path):
        return []
    try:
        df = pd.read_csv(csv_path)
        col = None
        for c in ["tradingsymbol", "symbol", "SYMBOL", "Tradingsymbol"]:
            if c in df.columns:
                col = c
                break
        if col is None:
            return []
        if "exchange" in df.columns:
            df = df[df["exchange"] == exchange]
        syms = df[col].dropna().astype(str).unique().tolist()
        return syms[:limit]
    except Exception:
        return []


def select_universe(config: BuildConfig) -> List[str]:
    # Priority 1: symbols file if provided or local default
    candidates: List[str] = []
    search_csvs = [
        config.symbols_file,
        os.path.join(CURRENT_DIR, "zerodha_instruments.csv"),
        os.path.join(PROJECT_ROOT, "zerodha_instruments.csv"),
    ]
    for path in search_csvs:
        if path:
            symbols = load_symbols_from_csv(path, config.exchange, config.num_symbols)
            if symbols:
                candidates = symbols
                break

    # Fallback 2: query Zerodha instruments if available
    if not candidates and ZerodhaDataClient is not None:
        try:
            client = ZerodhaDataClient()
            if client.authenticate():
                inst = client.get_instruments(exchange=config.exchange)
                if inst is not None and len(inst) > 0:
                    if "tradingsymbol" in inst.columns:
                        candidates = inst["tradingsymbol"].dropna().astype(str).unique().tolist()[: config.num_symbols]
        except Exception:
            pass

    # Final fallback: minimal demo set
    if not candidates:
        candidates = [
            "RELIANCE", "HDFCBANK", "TCS", "INFY", "ICICIBANK",
            "HINDUNILVR", "KOTAKBANK", "LT", "SBIN", "ITC",
        ][: config.num_symbols]

    return candidates


# ----------------------------- Pattern Extraction -----------------------------

def detect_events_for_symbol(df: pd.DataFrame, symbol: str, config: BuildConfig) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if df is None or df.empty:
        return events

    # Normalize
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df.set_index("date", inplace=True)
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    else:
        # Attempt to coerce
        if "datetime" in df.columns:
            df["date"] = pd.to_datetime(df["datetime"])  # noqa: E501
            df = df.sort_values("date").reset_index(drop=True)
            df.set_index("date", inplace=True)

    close = df.get("close")
    if close is None or close.isna().all():
        return events

    # Precompute context features
    vol_ratio = compute_volume_ratio20(df)
    ret_5 = safe_pct_change(df["close"], 5)
    ret_20 = safe_pct_change(df["close"], 20)

    # 1) Double Top / Bottom
    try:
        dtops = PatternRecognition.detect_double_top(close)
        for a, b in dtops:
            event_idx = min(int(b), len(df) - 1)
            events.append({
                "symbol": symbol,
                "pattern_type": "double_top",
                "event_iloc": event_idx,
                "duration": int(max(0, b - a)),
                "quality_score": np.nan,
                "completion_status": "completed",
            })
    except Exception:
        pass

    try:
        dbots = PatternRecognition.detect_double_bottom(close)
        for a, b in dbots:
            event_idx = min(int(b), len(df) - 1)
            events.append({
                "symbol": symbol,
                "pattern_type": "double_bottom",
                "event_iloc": event_idx,
                "duration": int(max(0, b - a)),
                "quality_score": np.nan,
                "completion_status": "completed",
            })
    except Exception:
        pass

    # 2) Head & Shoulders (use neckline index as event)
    try:
        hs = PatternRecognition.detect_head_and_shoulders(close)
        for pat in hs:
            event_idx = int(pat.get("neckline", {}).get("index", pat.get("right_shoulder", {}).get("index", 0)))
            event_idx = min(event_idx, len(df) - 1)
            events.append({
                "symbol": symbol,
                "pattern_type": "head_and_shoulders",
                "event_iloc": event_idx,
                "duration": int(pat.get("right_shoulder", {}).get("index", 0) - pat.get("left_shoulder", {}).get("index", 0)),
                "quality_score": float(pat.get("quality_score", np.nan)),
                "completion_status": str(pat.get("completion_status", "forming")),
            })
    except Exception:
        pass

    try:
        ihs = PatternRecognition.detect_inverse_head_and_shoulders(close)
        for pat in ihs:
            event_idx = int(pat.get("neckline", {}).get("index", pat.get("right_shoulder", {}).get("index", 0)))
            event_idx = min(event_idx, len(df) - 1)
            events.append({
                "symbol": symbol,
                "pattern_type": "inverse_head_and_shoulders",
                "event_iloc": event_idx,
                "duration": int(pat.get("right_shoulder", {}).get("index", 0) - pat.get("left_shoulder", {}).get("index", 0)),
                "quality_score": float(pat.get("quality_score", np.nan)),
                "completion_status": str(pat.get("completion_status", "forming")),
            })
    except Exception:
        pass

    # 3) Triangle & Flag (use segment end as event)
    try:
        tris = PatternRecognition.detect_triangle(close)
        for seg in tris:
            if not seg:
                continue
            a, b = int(seg[0]), int(seg[-1])
            event_idx = min(b, len(df) - 1)
            events.append({
                "symbol": symbol,
                "pattern_type": "triangle",
                "event_iloc": event_idx,
                "duration": int(max(0, b - a)),
                "quality_score": np.nan,
                "completion_status": "forming",
            })
    except Exception:
        pass

    try:
        flags = PatternRecognition.detect_flag(close)
        for seg in flags:
            if not seg:
                continue
            a, b = int(seg[0]), int(seg[-1])
            event_idx = min(b, len(df) - 1)
            events.append({
                "symbol": symbol,
                "pattern_type": "flag",
                "event_iloc": event_idx,
                "duration": int(max(0, b - a)),
                "quality_score": np.nan,
                "completion_status": "forming",
            })
    except Exception:
        pass

    # Label each event
    labeled: List[Dict[str, Any]] = []
    for i, ev in enumerate(events):
        idx = int(ev["event_iloc"])  # iloc index
        if idx < 0 or idx >= len(df):
            continue
        event_ts = df.index[idx]
        entry_price = float(df["close"].iloc[idx])
        next_window = df.iloc[idx + 1 : idx + 1 + config.horizon_days]
        y = first_hit_outcome(next_window, entry_price, config.tp_pct, config.sl_pct)
        record = {
            "event_id": f"{symbol}_{event_ts.strftime('%Y%m%d')}_{i}",
            "symbol": symbol,
            "exchange": config.exchange,
            "event_date": event_ts.isoformat(),
            "pattern_type": ev["pattern_type"],
            "duration": int(ev.get("duration", 0)),
            "quality_score": float(ev.get("quality_score", np.nan)) if not pd.isna(ev.get("quality_score", np.nan)) else np.nan,
            "completion_status": str(ev.get("completion_status", "forming")),
            "entry_price": entry_price,
            "tp_price": entry_price * (1.0 + config.tp_pct),
            "sl_price": entry_price * (1.0 - config.sl_pct),
            "horizon_days": int(config.horizon_days),
            "y_success": int(y),
            # context features
            "close": float(df["close"].iloc[idx]) if not pd.isna(df["close"].iloc[idx]) else np.nan,
            "volume": float(df["volume"].iloc[idx]) if "volume" in df.columns and not pd.isna(df["volume"].iloc[idx]) else np.nan,
            "volume_ratio20": float(vol_ratio.iloc[idx]) if not pd.isna(vol_ratio.iloc[idx]) else np.nan,
            "ret_5": float(ret_5.iloc[idx]) if not pd.isna(ret_5.iloc[idx]) else np.nan,
            "ret_20": float(ret_20.iloc[idx]) if not pd.isna(ret_20.iloc[idx]) else np.nan,
        }
        labeled.append(record)

    return labeled


# ----------------------------- Main Build Flow -----------------------------

def fetch_ohlcv_for_symbol(client: Optional[ZerodhaDataClient], symbol: str, config: BuildConfig) -> Optional[pd.DataFrame]:
    from_d = _parse_date(config.from_date)
    to_d = _parse_date(config.to_date)
    if client is None:
        return None
    try:
        df = client.get_historical_data(
            symbol=symbol,
            exchange=config.exchange,
            interval=config.interval,
            from_date=from_d,
            to_date=to_d,
        )
        return df
    except Exception:
        return None


def build_dataset(config: BuildConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    symbols = select_universe(config)
    if not symbols:
        raise RuntimeError("No symbols available to build the dataset")

    client = None
    if ZerodhaDataClient is not None:
        try:
            client = ZerodhaDataClient()
            client.authenticate()  # If this fails, client methods will return None
        except Exception:
            client = None

    all_events: List[Dict[str, Any]] = []
    for s in symbols:
        df = fetch_ohlcv_for_symbol(client, s, config)
        if df is None or len(df) < 60:  # minimal history requirement
            continue
        events = detect_events_for_symbol(df, s, config)
        if events:
            all_events.extend(events)

    if not all_events:
        # Create empty frame with schema
        dataset = pd.DataFrame(columns=EVENT_COLUMNS)
    else:
        dataset = pd.DataFrame(all_events)
        # enforce column order and presence
        for col in EVENT_COLUMNS:
            if col not in dataset.columns:
                dataset[col] = np.nan
        dataset = dataset[EVENT_COLUMNS]

    meta = {
        "generated_at": datetime.utcnow().isoformat(),
        "config": asdict(config),
        "num_symbols": len(symbols),
        "num_events": int(len(dataset)),
        "columns": EVENT_COLUMNS,
    }
    return dataset, meta


def save_outputs(dataset: pd.DataFrame, meta: Dict[str, Any], output_path: str) -> None:
    ensure_dir(output_path)
    # Save Parquet if available, else CSV
    try:
        dataset.to_parquet(output_path, index=False)
        meta_path = os.path.splitext(output_path)[0] + "_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        # Fallback to CSV
        csv_path = output_path if output_path.lower().endswith(".csv") else os.path.splitext(output_path)[0] + ".csv"
        dataset.to_csv(csv_path, index=False)
        meta_path = os.path.splitext(csv_path)[0] + "_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


def parse_args(argv: Optional[List[str]] = None) -> BuildConfig:
    p = argparse.ArgumentParser(description="Build a general cross-stock pattern dataset")
    p.add_argument("--exchange", default="NSE")
    p.add_argument("--num_symbols", type=int, default=200)
    p.add_argument("--from_date", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument("--to_date", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument("--interval", type=str, default="day")
    p.add_argument("--horizon_days", type=int, default=20)
    p.add_argument("--tp_pct", type=float, default=0.04)
    p.add_argument("--sl_pct", type=float, default=0.02)
    p.add_argument("--symbols_file", type=str, default=None)
    p.add_argument("--output", type=str, default=os.path.join(CURRENT_DIR, "datasets", "general_patterns.parquet"))
    args = p.parse_args(argv)
    cfg = BuildConfig(
        exchange=args.exchange,
        num_symbols=args.num_symbols,
        from_date=args.from_date,
        to_date=args.to_date,
        interval=args.interval,
        horizon_days=args.horizon_days,
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
        symbols_file=args.symbols_file,
        output=args.output,
    )
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    dataset, meta = build_dataset(cfg)
    save_outputs(dataset, meta, cfg.output)
    print(f"Dataset built: {len(dataset)} events → {cfg.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


