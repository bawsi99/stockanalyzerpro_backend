from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List


# Default universe: placeholder; replace/extend as needed
# Add more symbols for better cross-stock generalization
DEFAULT_UNIVERSE: List[str] = [
    "RELIANCE"
]


@dataclass(frozen=True)
class TimeframeSpec:
    backfill_days: int
    horizon_bars: int
    est_cost_bps: float
    interval: str  # orchestrator interval string


# Timeframe configuration aligned with orchestrator intervals
# DEFAULT_TIMEFRAMES: Dict[str, TimeframeSpec] = {
#     "5m": TimeframeSpec(backfill_days=180, horizon_bars=12, est_cost_bps=8.0, interval="5min"),
#     "15m": TimeframeSpec(backfill_days=250, horizon_bars=8, est_cost_bps=7.0, interval="15min"),
#     "1h": TimeframeSpec(backfill_days=365, horizon_bars=12, est_cost_bps=6.0, interval="1hour"),
#     "1d": TimeframeSpec(backfill_days=2000, horizon_bars=5, est_cost_bps=5.0, interval="day"),
# }

DEFAULT_TIMEFRAMES: Dict[str, TimeframeSpec] = {
    "1d": TimeframeSpec(backfill_days=10000, horizon_bars=5, est_cost_bps=5.0, interval="day"),
}


def get_default_base_dir() -> str:
    """Default storage directory for raw ML data (under ml/data/raw, not ml/config/data/raw)."""
    ml_dir = os.path.dirname(os.path.dirname(__file__))  # .../ml
    return os.path.join(ml_dir, "data", "raw")


ml_defaults = {
    "universe": DEFAULT_UNIVERSE,
    "timeframes": DEFAULT_TIMEFRAMES,
    "base_dir": get_default_base_dir(),
    "exchange": "NSE",
}
