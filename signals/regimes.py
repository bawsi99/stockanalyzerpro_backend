from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def detect_market_regime(data: pd.DataFrame, indicators: Dict) -> Dict[str, str]:
    """
    Lightweight regime detection using ADX and ATR%.
    Returns tags to guide scoring weights.
    """
    result = {
        "trend": "unknown",
        "volatility": "normal",
    }

    try:
        # ADX (if available as series or current value)
        adx_val = None
        if isinstance(indicators.get("adx"), (list, tuple, pd.Series)) and len(indicators.get("adx")):
            adx_val = float(indicators["adx"][-1])
        elif isinstance(indicators.get("ADX"), dict) and indicators["ADX"].get("adx"):
            adx_series = indicators["ADX"]["adx"]
            adx_val = float(adx_series[-1]) if len(adx_series) else None

        if adx_val is not None:
            if adx_val >= 25:
                result["trend"] = "trending"
            else:
                result["trend"] = "ranging"

        # ATR% as volatility proxy
        atrp = None
        if isinstance(indicators.get("atr_percent"), (list, tuple, pd.Series)) and len(indicators.get("atr_percent")):
            atrp = float(indicators["atr_percent"][-1])
        elif isinstance(indicators.get("ATR"), dict) and indicators["ATR"].get("atr_percent"):
            atrp_series = indicators["ATR"]["atr_percent"]
            atrp = float(atrp_series[-1]) if len(atrp_series) else None

        if atrp is not None:
            if atrp >= 3.0:
                result["volatility"] = "high"
            elif atrp <= 1.0:
                result["volatility"] = "low"
            else:
                result["volatility"] = "normal"
    except Exception:
        # Fallback silently to defaults
        pass

    return result


