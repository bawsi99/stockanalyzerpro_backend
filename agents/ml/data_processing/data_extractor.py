from __future__ import annotations

import os
import sys

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
sys.path.insert(0, project_root)

import os
import sys
import asyncio
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Iterable, List

import pandas as pd

from backend.agents.ml.config.config import TimeframeSpec, ml_defaults
from backend.core.orchestrator import StockAnalysisOrchestrator


class MLDataExtractor:
    """
    Fetch OHLCV via the existing orchestrator and persist cleaned bars for ML.
    - Partitions by symbol/timeframe/date under base_dir.
    - Writes parquet if available (pyarrow/fastparquet), else CSV.
    """

    def __init__(self, base_dir: str | None = None, exchange: str | None = None) -> None:
        self.base_dir = base_dir or ml_defaults["base_dir"]
        self.exchange = exchange or ml_defaults["exchange"]
        os.makedirs(self.base_dir, exist_ok=True)
        self._orch = StockAnalysisOrchestrator()

    @staticmethod
    def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        # Normalize columns
        cols = {c.lower(): c for c in df.columns}
        rename_map = {}
        for want in ["open", "high", "low", "close", "volume"]:
            if want in df.columns:
                continue
            # Try common variants
            for cand in [want, want.capitalize(), want[0].upper() + want[1:], want[:1]]:
                if cand in df.columns:
                    rename_map[cand] = want
                    break
        if rename_map:
            df = df.rename(columns=rename_map)
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index(pd.to_datetime(df["timestamp"], utc=True))
                df = df.drop(columns=["timestamp"])  # type: ignore
            else:
                raise ValueError("Expected datetime index or timestamp column")
        # Make tz-naive UTC for consistency
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("UTC").tz_localize(None)
        # Sort, drop dups, drop NA close
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        if "close" in df.columns:
            df = df.dropna(subset=["close"])  # type: ignore
        return df[[c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]]

    def fetch_ohlcv(self, symbol: str, interval: str, backfill_days: int) -> pd.DataFrame:
        """Fetch data via async orchestrator from a sync context."""
        try:
            coro = self._orch.retrieve_stock_data(symbol, self.exchange, interval, backfill_days)
            df = asyncio.run(coro)  # run the async call
        except RuntimeError as e:
            # Fallback if already inside an event loop (unlikely in CLI)
            try:
                loop = asyncio.get_event_loop()
                df = loop.run_until_complete(coro)
            except Exception as inner:
                raise inner
        return self._clean_df(df)

    @staticmethod
    def _supports_parquet() -> bool:
        try:
            import pyarrow  # noqa: F401
            return True
        except Exception:
            try:
                import fastparquet  # noqa: F401
                return True
            except Exception:
                return False

    def _write_partitioned(self, df: pd.DataFrame, symbol: str, timeframe_key: str) -> List[str]:
        """Write a single consolidated file per (symbol, timeframe).
        If a file exists, merge and de-duplicate by index, then overwrite.
        """
        if df.empty:
            return []
        base = os.path.join(self.base_dir, f"symbol={symbol}", f"timeframe={timeframe_key}")
        os.makedirs(base, exist_ok=True)
        use_parquet = self._supports_parquet()
        file_path = os.path.join(base, f"bars.{ 'parquet' if use_parquet else 'csv'}")

        # Merge with existing if present
        try:
            if os.path.exists(file_path):
                if use_parquet:
                    existing = pd.read_parquet(file_path)
                else:
                    existing = pd.read_csv(file_path, index_col=0, parse_dates=True)
                if not isinstance(existing.index, pd.DatetimeIndex):
                    # Best-effort parse
                    existing.index = pd.to_datetime(existing.index, utc=True).tz_convert("UTC").tz_localize(None)
                combined = pd.concat([existing, df])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            else:
                combined = df.sort_index()
        except Exception:
            # If reading/merging fails, fall back to current df
            combined = df.sort_index()

        # Persist consolidated file
        if use_parquet:
            combined.to_parquet(file_path, index=True)
        else:
            combined.to_csv(file_path, index=True)
        return [file_path]

    def backfill_universe(self, universe: Iterable[str], tf_specs: Dict[str, TimeframeSpec]) -> Dict[str, Dict[str, List[str]]]:
        """
        For each symbol and timeframe, fetch and persist.
        Returns mapping: symbol -> timeframe_key -> [written_files]
        """
        report: Dict[str, Dict[str, List[str]]] = {}
        for symbol in universe:
            report[symbol] = {}
            for tf_key, spec in tf_specs.items():
                try:
                    df = self.fetch_ohlcv(symbol, spec.interval, spec.backfill_days)
                    files = self._write_partitioned(df, symbol, tf_key)
                    report[symbol][tf_key] = files
                except Exception as e:
                    # Record failure with empty list; real logging could be added
                    report[symbol][tf_key] = []
                    print(f"[WARN] {symbol} {tf_key} backfill failed: {e}", file=sys.stderr)
        return report


if __name__ == "__main__":
    # Simple CLI to backfill defaults
    extractor = MLDataExtractor()
    universe = ml_defaults["universe"]
    tf_specs = ml_defaults["timeframes"]
    summary = extractor.backfill_universe(universe, tf_specs)
    # Emit a tiny summary
    total_files = sum(len(v2) and sum(len(lst) for lst in v2.values()) for v2 in summary.values())
    print({"wrote": total_files, "base_dir": extractor.base_dir})
