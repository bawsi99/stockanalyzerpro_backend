#!/usr/bin/env python3
import asyncio
import pandas as pd
import numpy as np

from backend.agents.indicators.trend.processor import TrendIndicatorsProcessor
from backend.agents.indicators.momentum.processor import MomentumIndicatorsProcessor


def make_df(n=50):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = np.linspace(100, 110, n) + np.random.normal(0, 0.5, n)
    high = close + 1
    low = close - 1
    open_ = close
    volume = np.random.randint(1000, 2000, n)
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume
    }, index=idx)


def optimized_indicators_sample(df: pd.DataFrame):
    # Mimic optimized indicators shape (scalars in moving_averages, nested dicts for rsi/macd/adx)
    last_close = float(df['close'].iloc[-1])
    return {
        'moving_averages': {
            'sma_20': last_close * 0.99,
            'sma_50': last_close * 1.00,
            'sma_200': last_close * 0.95,
            'ema_20': last_close * 0.995,
            'ema_50': last_close * 1.002,
            'price_to_sma_200': 0.03,
            'sma_20_to_sma_50': -0.01,
            'golden_cross': False,
            'death_cross': False,
        },
        'rsi': {
            'rsi_14': 52.3,
            'recent_values': [49.0, 51.2, 52.3],
            'trend': 'up',
            'status': 'neutral',
        },
        'macd': {
            'macd_line': 0.5,
            'signal_line': 0.3,
            'histogram': 0.2,
        },
        'adx': {
            'adx': 22.0,
            'plus_di': 18.0,
            'minus_di': 15.0,
            'trend_direction': 'bullish',
            'trend_strength': 'weak'
        }
    }


async def run_once():
    df = make_df(60)
    ind = optimized_indicators_sample(df)

    trend = TrendIndicatorsProcessor()
    momentum = MomentumIndicatorsProcessor()

    t_res = await trend.analyze_async(df, ind)
    m_res = await momentum.analyze_async(df, ind)

    assert 'error' not in t_res, f"Trend error: {t_res.get('error')}"
    assert 'error' not in m_res, f"Momentum error: {m_res.get('error')}"
    assert isinstance(t_res.get('overall_trend', {}), dict)
    assert isinstance(m_res.get('overall_momentum', {}), dict)
    print("OK: processors handle optimized schema without errors")


if __name__ == "__main__":
    asyncio.run(run_once())
