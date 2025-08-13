import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .pattern_database import PatternDatabase, PatternRecord

logger = logging.getLogger(__name__)


def _parse_date(d: str | date) -> date:
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d), "%Y-%m-%d").date()


def _infer_direction_from_pattern(pattern_type: str, pattern: dict | None) -> Optional[str]:
    # 1) Explicit direction from payload
    if isinstance(pattern, dict):
        d = pattern.get("direction") or pattern.get("breakout_direction")
        if isinstance(d, str) and d.lower() in {"up", "down", "bullish", "bearish"}:
            return "bullish" if d.lower() in {"up", "bullish"} else "bearish"
        subtype = pattern.get("subtype")
        if isinstance(subtype, str):
            if subtype.lower() in {"rising", "ascending"}:
                return "bearish" if "wedge" in (pattern_type or "").lower() else None
            if subtype.lower() in {"falling", "descending"}:
                return "bullish" if "wedge" in (pattern_type or "").lower() else None
    # 2) Type-based defaults
    t = (pattern_type or "").lower()
    if t in {"inverse_head_and_shoulders", "cup_and_handle", "triple_bottoms", "double_bottoms", "rounding_bottom"}:
        return "bullish"
    if t in {"head_and_shoulders", "triple_tops", "double_tops", "rounding_top"}:
        return "bearish"
    # 3) Unknown
    return None


def _extract_basic_features(pattern: dict) -> Dict[str, float]:
    # Align with features used by Bayesian scorer
    try:
        return {
            "duration": float(pattern.get("duration") or pattern.get("length") or 0.0),
            "volume_ratio": float(pattern.get("volume_ratio") or 1.0),
            "trend_alignment": float(pattern.get("trend_alignment") or 0.0),
            "completion": float(pattern.get("completion") or pattern.get("quality_score") or pattern.get("confidence") or 0.0),
        }
    except Exception:
        return {"duration": 0.0, "volume_ratio": 1.0, "trend_alignment": 0.0, "completion": 0.0}


class Backtester:
    """Walk-forward backtesting harness that replays the overlay detector on a rolling window
    and validates outcomes after a lookahead period. Records results in PatternDatabase.
    """

    def __init__(self, exchange: str = "NSE") -> None:
        self.exchange = exchange
        self.db = PatternDatabase()

    def _fetch_history(self, symbol: str, start: date, end: date, interval: str = "day") -> Optional[pd.DataFrame]:
        try:
            from .zerodha_client import ZerodhaDataClient
            z = ZerodhaDataClient()
            if not z.authenticate():
                logger.error("Zerodha authentication failed for backtest")
                return None
            df = z.get_historical_data(symbol=symbol, exchange=self.exchange, interval=interval, from_date=start, to_date=end)
            if df is None or df.empty:
                return None
            # Normalize index
            if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            elif 'datetime' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            df = df.sort_index()
            # Ensure required columns
            for col in ("open", "high", "low", "close", "volume"):
                if col not in df.columns:
                    df[col] = 0.0
            return df
        except Exception as exc:
            logger.error(f"Failed to fetch history for {symbol}: {exc}")
            return None

    def _detect_patterns(self, data_slice: pd.DataFrame) -> Dict[str, List[dict]]:
        try:
            from .agent_capabilities import StockAnalysisOrchestrator
            orch = StockAnalysisOrchestrator()
            overlays = orch._create_overlays(data_slice, {}) or {}
            adv = overlays.get("advanced_patterns", {}) or {}
            return adv
        except Exception as exc:
            logger.error(f"Pattern detection failed: {exc}")
            return {}

    def _determine_outcome(self, dir_hint: Optional[str], entry_price: float, future_price: float, threshold_pct: float = 1.0) -> Optional[bool]:
        try:
            if entry_price is None or future_price is None:
                return None
            change_pct = (future_price - entry_price) / entry_price * 100.0
            if dir_hint == "bullish":
                return change_pct >= threshold_pct
            if dir_hint == "bearish":
                return change_pct <= -threshold_pct
            # Unknown direction: consider absolute move
            return abs(change_pct) >= threshold_pct
        except Exception:
            return None

    def run(self, symbol: str, start_date: str, end_date: str, interval: str = "day", lookahead_days: int = 3, min_candles: int = 60, success_threshold_pct: float = 1.0) -> Dict[str, dict]:
        """Run walk-forward backtest. Returns summary metrics per pattern type.

        Summary schema: { pattern_type: {"detected": int, "confirmed": int, "success": int, "success_rate": float } }
        """
        start = _parse_date(start_date)
        end = _parse_date(end_date)

        # Fetch a broader window to include lookahead for the last detections
        hist = self._fetch_history(symbol, start, end + timedelta(days=lookahead_days + 7), interval=interval)
        if hist is None or hist.empty:
            logger.error(f"No historical data for backtest {symbol}")
            return {}

        # Prepare counters and index mapping for updates: pattern_type -> list[(record_index_in_db, detection_date)]
        index_map: Dict[str, List[Tuple[int, date]]] = {}
        summary: Dict[str, dict] = {}

        # Pre-existing counts to compute new indices
        pre_counts: Dict[str, int] = {ptype: len(records) for ptype, records in self.db.get_all().items()}

        current = start
        while current <= end:
            # Slice data up to current day (inclusive)
            ds = hist.loc[hist.index.date <= current]
            if ds is None or ds.empty or len(ds) < min_candles:
                current += timedelta(days=1)
                continue

            # Detect advanced patterns at this point in time
            adv = self._detect_patterns(ds)
            if not adv:
                current += timedelta(days=1)
                continue

            # Entry price = close on current day
            try:
                entry_row = hist.loc[hist.index.date == current]
                if entry_row.empty:
                    current += timedelta(days=1)
                    continue
                entry_price = float(entry_row['close'].iloc[0])
            except Exception:
                current += timedelta(days=1)
                continue

            for ptype, plist in adv.items():
                if not isinstance(plist, list) or not plist:
                    continue
                # Ensure summary bucket
                if ptype not in summary:
                    summary[ptype] = {"detected": 0, "confirmed": 0, "success": 0, "success_rate": 0.0}

                # For each pattern, record features
                for p in plist:
                    features = _extract_basic_features(p)
                    rec = PatternRecord(
                        symbol=symbol,
                        timestamp=current.isoformat(),
                        features=features,
                        outcome=None,
                        confirmed=False,
                    )
                    # Append and compute the new index in DB for this pattern type
                    try:
                        self.db.record_pattern(ptype, rec)
                        new_idx = pre_counts.get(ptype, 0)
                        pre_counts[ptype] = new_idx + 1
                        index_map.setdefault(ptype, []).append((new_idx, current))
                        summary[ptype]["detected"] += 1
                    except Exception as exc:
                        logger.warning(f"Failed to record pattern {ptype}: {exc}")

            current += timedelta(days=1)

        # Second pass: evaluate outcomes after lookahead_days
        for ptype, idx_list in index_map.items():
            for rec_idx, det_date in idx_list:
                # Entry price at detection date
                try:
                    entry_row = hist.loc[hist.index.date == det_date]
                    if entry_row.empty:
                        continue
                    entry_price = float(entry_row['close'].iloc[0])
                except Exception:
                    continue
                # Future price at earliest available day >= detection + lookahead
                fut_date = det_date + timedelta(days=lookahead_days)
                fut_row = hist.loc[hist.index.date >= fut_date].head(1)
                if fut_row is None or fut_row.empty:
                    continue
                future_price = float(fut_row['close'].iloc[0])

                # Determine direction hint from payload when possible
                direction = None
                try:
                    # Use the first recorded instance on that day to infer direction more precisely
                    # (best-effort only; direction primarily matters for channels/wedges)
                    direction = _infer_direction_from_pattern(ptype, None)
                except Exception:
                    direction = _infer_direction_from_pattern(ptype, None)

                # Compute outcome
                outcome = self._determine_outcome(direction, entry_price, future_price, threshold_pct=success_threshold_pct)
                if outcome is None:
                    continue
                try:
                    self.db.update_outcome(ptype, rec_idx, bool(outcome))
                    summary[ptype]["confirmed"] += 1
                    if outcome:
                        summary[ptype]["success"] += 1
                except Exception as exc:
                    logger.warning(f"Failed to update outcome for {ptype}[{rec_idx}]: {exc}")

        # Compute success rates
        for ptype, s in summary.items():
            conf = s["confirmed"]
            s["success_rate"] = round(100.0 * s["success"] / conf, 2) if conf > 0 else 0.0

        return summary


