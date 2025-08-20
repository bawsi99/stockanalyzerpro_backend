from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .schema import SignalReason, TimeframeScore, SignalsSummary
from .config import load_timeframe_weights


def _safe_last(series_like) -> float:
    try:
        # Handle simple numeric values
        if isinstance(series_like, (int, float)):
            return float(series_like)
        
        # Handle None
        if series_like is None:
            return np.nan
            
        # Handle lists and tuples
        if isinstance(series_like, (list, tuple)) and series_like:
            return float(series_like[-1])
        
        # Handle pandas Series
        if hasattr(series_like, "iloc") and len(series_like) > 0:
            return float(series_like.iloc[-1])
        
        # Handle numpy arrays
        if hasattr(series_like, "__array__"):
            arr = np.asarray(series_like)
            return float(arr[-1]) if arr.size else np.nan
            
    except Exception:
        return np.nan
    return np.nan


def _score_rsi(indicators: Dict) -> Tuple[float, List[SignalReason]]:
    r = []
    score = 0.0
    rsi_val = _safe_last(indicators.get("rsi_14"))
    if np.isnan(rsi_val):
        rsi_block = indicators.get("rsi") or {}
        rsi_val = _safe_last(rsi_block.get("rsi_14"))
    if np.isnan(rsi_val):
        return 0.0, r

    if rsi_val >= 60:
        score += 0.25
        r.append(SignalReason("RSI", f"RSI 14 at {rsi_val:.1f} (bullish momentum)", 0.25, "bullish"))
    elif rsi_val <= 40:
        score -= 0.25
        r.append(SignalReason("RSI", f"RSI 14 at {rsi_val:.1f} (bearish momentum)", 0.25, "bearish"))
    else:
        r.append(SignalReason("RSI", f"RSI 14 neutral at {rsi_val:.1f}", 0.05, "neutral"))
    return score, r


def _score_macd(indicators: Dict) -> Tuple[float, List[SignalReason]]:
    r = []
    score = 0.0
    macd = _safe_last(indicators.get("macd_line"))
    signal = _safe_last(indicators.get("signal_line"))
    if np.isnan(macd) or np.isnan(signal):
        macd_block = indicators.get("macd") or {}
        macd = _safe_last(macd_block.get("macd_line"))
        signal = _safe_last(macd_block.get("signal_line"))
    if np.isnan(macd) or np.isnan(signal):
        return 0.0, r
    if macd > signal:
        score += 0.2
        r.append(SignalReason("MACD", f"MACD above signal ({macd:.2f}>{signal:.2f})", 0.2, "bullish"))
    elif macd < signal:
        score -= 0.2
        r.append(SignalReason("MACD", f"MACD below signal ({macd:.2f}<{signal:.2f})", 0.2, "bearish"))
    return score, r


def _score_ma(indicators: Dict) -> Tuple[float, List[SignalReason]]:
    r = []
    score = 0.0
    sma50 = _safe_last(indicators.get("sma_50"))
    sma200 = _safe_last(indicators.get("sma_200"))
    if np.isnan(sma50) or np.isnan(sma200):
        ma_block = indicators.get("moving_averages") or {}
        sma50 = _safe_last(ma_block.get("sma_50"))
        sma200 = _safe_last(ma_block.get("sma_200"))
    if np.isnan(sma50) or np.isnan(sma200):
        return 0.0, r
    if sma50 > sma200:
        score += 0.2
        r.append(SignalReason("MA", "SMA50 above SMA200 (uptrend bias)", 0.2, "bullish"))
    elif sma50 < sma200:
        score -= 0.2
        r.append(SignalReason("MA", "SMA50 below SMA200 (downtrend bias)", 0.2, "bearish"))
    return score, r


def _score_adx(indicators: Dict) -> Tuple[float, List[SignalReason]]:
    r = []
    score = 0.0
    adx_val = _safe_last(indicators.get("adx"))
    if np.isnan(adx_val):
        adx_block = indicators.get("adx") or {}
        if isinstance(adx_block, dict):
            adx_val = _safe_last(adx_block.get("adx"))
    if np.isnan(adx_val):
        return 0.0, r
    if adx_val >= 25:
        # amplify existing bias elsewhere via confidence, not score
        r.append(SignalReason("ADX", f"ADX strong at {adx_val:.1f}", 0.1, "neutral"))
    else:
        r.append(SignalReason("ADX", f"ADX weak at {adx_val:.1f}", 0.05, "neutral"))
    return score, r


def _score_bbands(indicators: Dict) -> Tuple[float, List[SignalReason]]:
    r = []
    score = 0.0
    # Structured first
    bb_block = indicators.get("bollinger_bands") or {}
    percent_b = _safe_last(bb_block.get("percent_b"))
    if np.isnan(percent_b):
        percent_b = _safe_last(indicators.get("percent_b"))
    if np.isnan(percent_b):
        return 0.0, r
    if percent_b >= 0.8:
        score += 0.1
        r.append(SignalReason("BollingerBands", f"%B high at {percent_b:.2f}", 0.1, "bullish"))
    elif percent_b <= 0.2:
        score -= 0.1
        r.append(SignalReason("BollingerBands", f"%B low at {percent_b:.2f}", 0.1, "bearish"))
    else:
        r.append(SignalReason("BollingerBands", f"%B neutral at {percent_b:.2f}", 0.05, "neutral"))
    return score, r


def _score_volume(indicators: Dict) -> Tuple[float, List[SignalReason]]:
    r = []
    score = 0.0
    vol_block = indicators.get("volume") or {}
    vr = _safe_last(vol_block.get("volume_ratio"))
    if np.isnan(vr):
        vr = _safe_last(indicators.get("volume_ratio"))
    if np.isnan(vr):
        return 0.0, r
    if vr >= 1.5:
        score += 0.1
        r.append(SignalReason("Volume", f"Volume ratio elevated at {vr:.2f}x", 0.1, "bullish"))
    elif vr <= 0.5:
        score -= 0.1
        r.append(SignalReason("Volume", f"Volume ratio depressed at {vr:.2f}x", 0.1, "bearish"))
    else:
        r.append(SignalReason("Volume", f"Volume ratio neutral at {vr:.2f}x", 0.05, "neutral"))
    return score, r


def _score_mfi_cmf(indicators: Dict) -> Tuple[float, List[SignalReason]]:
    r = []
    score = 0.0
    # MFI
    mfi = np.nan
    ev = indicators.get("enhanced_volume") or {}
    mfi = _safe_last(ev.get("mfi")) if isinstance(ev.get("mfi"), (list, tuple, pd.Series)) else ev.get("mfi")
    if mfi is not None and not np.isnan(float(mfi)):
        mfi = float(mfi)
        if mfi >= 80:
            score -= 0.1
            r.append(SignalReason("MFI", f"MFI overbought at {mfi:.1f}", 0.1, "bearish"))
        elif mfi <= 20:
            score += 0.1
            r.append(SignalReason("MFI", f"MFI oversold at {mfi:.1f}", 0.1, "bullish"))
        else:
            r.append(SignalReason("MFI", f"MFI neutral at {mfi:.1f}", 0.05, "neutral"))
    # CMF
    cmf_block = indicators.get("cmf") or {}
    cmf_val = cmf_block.get("value")
    if cmf_val is not None and not np.isnan(float(cmf_val)):
        cmf_val = float(cmf_val)
        if cmf_val > 0:
            score += 0.1
            r.append(SignalReason("CMF", f"CMF positive at {cmf_val:.2f}", 0.1, "bullish"))
        elif cmf_val < 0:
            score -= 0.1
            r.append(SignalReason("CMF", f"CMF negative at {cmf_val:.2f}", 0.1, "bearish"))
        else:
            r.append(SignalReason("CMF", f"CMF neutral at {cmf_val:.2f}", 0.05, "neutral"))
    return score, r


def _score_stochrsi(indicators: Dict) -> Tuple[float, List[SignalReason]]:
    r = []
    score = 0.0
    mom = indicators.get("enhanced_momentum") or {}
    k = _safe_last(mom.get("stochrsi_k"))
    d = _safe_last(mom.get("stochrsi_d"))
    if np.isnan(k) or np.isnan(d):
        return 0.0, r
    if k > 80 and d > 80:
        score -= 0.1
        r.append(SignalReason("StochRSI", f"StochRSI overbought (K={k:.1f}, D={d:.1f})", 0.1, "bearish"))
    elif k < 20 and d < 20:
        score += 0.1
        r.append(SignalReason("StochRSI", f"StochRSI oversold (K={k:.1f}, D={d:.1f})", 0.1, "bullish"))
    else:
        r.append(SignalReason("StochRSI", f"StochRSI neutral (K={k:.1f}, D={d:.1f})", 0.05, "neutral"))
    return score, r


def _score_ichimoku(indicators: Dict) -> Tuple[float, List[SignalReason]]:
    r = []
    score = 0.0
    ich = indicators.get("ichimoku") or {}
    sig = ich.get("signal")
    # Only append a reason if a signal is explicitly available
    if sig == "bullish":
        score += 0.15
        r.append(SignalReason("Ichimoku", "Price above cloud", 0.15, "bullish"))
    elif sig == "bearish":
        score -= 0.15
        r.append(SignalReason("Ichimoku", "Price below cloud", 0.15, "bearish"))
    elif sig in ("inside", "neutral"):
        r.append(SignalReason("Ichimoku", "Price inside cloud", 0.05, "neutral"))
    # If no signal present, omit neutral filler
    return score, r


def _score_keltner(indicators: Dict) -> Tuple[float, List[SignalReason]]:
    r = []
    score = 0.0
    kel = indicators.get("keltner") or {}
    sig = kel.get("signal")
    # Only append a reason if a signal is explicitly available
    if sig == "overbought":
        score -= 0.1
        r.append(SignalReason("Keltner", "Price above upper channel", 0.1, "bearish"))
    elif sig == "oversold":
        score += 0.1
        r.append(SignalReason("Keltner", "Price below lower channel", 0.1, "bullish"))
    elif sig in ("inside", "neutral"):
        r.append(SignalReason("Keltner", "Price within channel", 0.05, "neutral"))
    # If no signal present, omit neutral filler
    return score, r


def _score_supertrend(indicators: Dict) -> Tuple[float, List[SignalReason]]:
    r = []
    score = 0.0
    st = indicators.get("supertrend") or {}
    direction = st.get("direction")
    # Only append a reason if a direction is explicitly available
    if direction == "up":
        score += 0.15
        r.append(SignalReason("Supertrend", "Supertrend up", 0.15, "bullish"))
    elif direction == "down":
        score -= 0.15
        r.append(SignalReason("Supertrend", "Supertrend down", 0.15, "bearish"))
    elif direction in ("neutral", "flat"):
        r.append(SignalReason("Supertrend", "Supertrend neutral", 0.05, "neutral"))
    # If no direction present, omit neutral filler
    return score, r


def compute_timeframe_score(indicators: Dict, timeframe: str, regime: Dict[str, str]) -> TimeframeScore:
    parts: List[SignalReason] = []
    total = 0.0

    for scorer in (
        _score_rsi,
        _score_macd,
        _score_ma,
        _score_adx,
        _score_bbands,
        _score_volume,
        _score_mfi_cmf,
        _score_stochrsi,
        _score_ichimoku,
        _score_keltner,
        _score_supertrend,
    ):
        s, reasons = scorer(indicators)
        total += s
        parts.extend(reasons)

    score = float(np.clip(total, -1.0, 1.0))

    # Confidence shaped by regime and ADX
    confidence = 0.5
    if regime.get("trend") == "trending":
        confidence += 0.1
    if regime.get("volatility") == "high":
        confidence -= 0.05
    confidence = float(np.clip(confidence, 0.1, 0.95))

    bias = "bullish" if score > 0.1 else ("bearish" if score < -0.1 else "neutral")

    return TimeframeScore(
        timeframe=timeframe,
        score=score,
        confidence=confidence,
        bias=bias,  # type: ignore
        reasons=parts,
    )


def compute_signals_summary(
    per_timeframe_indicators: Dict[str, Dict]
) -> SignalsSummary:
    # Derive a simple regime from the base timeframe if available
    base_tf = next(iter(per_timeframe_indicators.keys())) if per_timeframe_indicators else "day"
    regime = {"trend": "unknown", "volatility": "normal"}

    # Allow passing of 'data' or meta if at hand; else compute regime from indicators if possible
    # Here we infer from indicators only
    regime = {**regime}

    scores: List[TimeframeScore] = []
    for tf, inds in per_timeframe_indicators.items():
        tf_regime = regime
        scores.append(compute_timeframe_score(inds or {}, tf, tf_regime))

    if not scores:
        return SignalsSummary(consensus_score=0.0, consensus_bias="neutral", confidence=0.3, per_timeframe=[], regime=regime)

    # Determine a coarse regime label for weighting profile selection
    strong_adx_ct = 0
    for inds in per_timeframe_indicators.values():
        adx_block = (inds or {}).get("adx") or {}
        try:
            adx_val = float(adx_block.get("adx")) if isinstance(adx_block, dict) else float(adx_block)
        except Exception:
            adx_val = None
        if adx_val is not None and adx_val >= 25:
            strong_adx_ct += 1
    regime_label = "trending" if strong_adx_ct >= max(1, len(per_timeframe_indicators)//2) else "ranging"

    # Base weights per timeframe, possibly regime-specific
    tf_base_weights = load_timeframe_weights(regime=regime_label)

    weights = []
    for s in scores:
        base_w = tf_base_weights.get(s.timeframe, 1.0)
        weights.append(float(base_w * s.confidence))
    weights = np.array(weights, dtype=float)
    vals = np.array([s.score for s in scores], dtype=float)
    consensus = float(np.clip((vals * weights).sum() / (weights.sum() + 1e-9), -1.0, 1.0))
    consensus_bias = "bullish" if consensus > 0.1 else ("bearish" if consensus < -0.1 else "neutral")
    consensus_confidence = float(np.clip(weights.mean(), 0.1, 0.95))

    # Conflict penalty: reduce confidence when biases disagree significantly
    bullish_ct = sum(1 for s in scores if s.bias == "bullish")
    bearish_ct = sum(1 for s in scores if s.bias == "bearish")
    if bullish_ct > 0 and bearish_ct > 0:
        conflict_ratio = min(bullish_ct, bearish_ct) / max(bullish_ct, bearish_ct)
        penalty = 0.15 * conflict_ratio  # up to 15% reduction
        consensus_confidence = float(np.clip(consensus_confidence * (1.0 - penalty), 0.1, 0.95))

    return SignalsSummary(
        consensus_score=consensus,
        consensus_bias=consensus_bias,  # type: ignore
        confidence=consensus_confidence,
        per_timeframe=scores,
        regime=regime,
    )


