import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

# --- Helper: Resample DataFrame to a higher timeframe ---
def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a higher timeframe using standard rules.
    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume'] and a DateTimeIndex
        rule: Pandas resample rule (e.g., 'W' for weekly, 'M' for monthly, 'H' for hourly)
    Returns:
        Resampled DataFrame
    """
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    resampled = df.resample(rule).apply(ohlc_dict).dropna()
    return resampled

# --- Helper: Compute indicators for a given timeframe ---
def compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute key technical indicators for the given DataFrame.
    Args:
        df: DataFrame with ['close'] and DateTimeIndex
    Returns:
        Dict with indicators and sufficiency messages
    """
    results = {}
    n = len(df)
    close = df['close']
    # SMA/EMA
    if n >= 200:
        results['sma_50'] = float(close.rolling(50).mean().iloc[-1])
        results['sma_200'] = float(close.rolling(200).mean().iloc[-1])
        results['ma_trend'] = (
            'bullish' if results['sma_50'] > results['sma_200'] else 'bearish'
        )
    elif n >= 50:
        results['sma_50'] = float(close.rolling(50).mean().iloc[-1])
        results['sma_200'] = None
        results['ma_trend'] = 'insufficient_data_for_sma_200'
    else:
        results['sma_50'] = None
        results['sma_200'] = None
        results['ma_trend'] = 'insufficient_data_for_ma_trend'
    # RSI
    if n >= 15:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        results['rsi_14'] = float(rsi.iloc[-1])
    else:
        results['rsi_14'] = None
    # MACD
    if n >= 35:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        results['macd'] = float(macd_line.iloc[-1])
        results['macd_signal'] = float(signal_line.iloc[-1])
        results['macd_hist'] = float((macd_line - signal_line).iloc[-1])
    else:
        results['macd'] = results['macd_signal'] = results['macd_hist'] = None
    # Trend
    if results['sma_50'] is not None and results['sma_200'] is not None:
        if results['sma_50'] > results['sma_200']:
            results['trend'] = 'bullish'
        elif results['sma_50'] < results['sma_200']:
            results['trend'] = 'bearish'
        else:
            results['trend'] = 'neutral'
    else:
        results['trend'] = 'unknown'
    return results

# --- Main: Multi-Timeframe Analysis ---
def multi_timeframe_analysis(
    df: pd.DataFrame, base_interval: str = 'day'
) -> Dict[str, Any]:
    """
    Perform multi-timeframe and long-term technical analysis.
    Args:
        df: DataFrame with OHLCV columns and DateTimeIndex
        base_interval: 'minute', 'hour', 'day', 'week', 'month'
    Returns:
        Dict with per-timeframe indicators, trend alignment, and sufficiency messages
    """
    timeframes = {}
    messages = []
    # Always analyze base timeframe
    timeframes[base_interval] = compute_indicators(df)
    # Determine which higher timeframes to compute
    resample_map = {
        'minute': [('hour', 'H'), ('day', 'D'), ('week', 'W'), ('month', 'ME')],
        'hour': [('day', 'D'), ('week', 'W'), ('month', 'ME')],
        'day': [('week', 'W'), ('month', 'ME')],
        'week': [('month', 'ME')],
    }
    if base_interval in resample_map:
        for tf_name, rule in resample_map[base_interval]:
            try:
                tf_df = resample_ohlcv(df, rule)
                if len(tf_df) < 10:
                    messages.append(f"Not enough data for {tf_name} timeframe analysis.")
                    continue
                timeframes[tf_name] = compute_indicators(tf_df)
            except Exception as e:
                messages.append(f"Error computing {tf_name} timeframe: {e}")
    # Trend alignment
    trends = [v.get('trend') for v in timeframes.values() if v.get('trend') in ['bullish', 'bearish', 'neutral']]
    if len(set(trends)) == 1 and len(trends) > 0:
        alignment = f"all_{trends[0]}"
    else:
        alignment = 'not_aligned'
    # Compose result
    result = {
        'timeframes': timeframes,
        'alignment': alignment,
    }
    if messages:
        result['messages'] = messages
    return result 