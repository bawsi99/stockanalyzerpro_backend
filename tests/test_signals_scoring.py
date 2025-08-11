import math

from signals.scoring import compute_signals_summary


def test_basic_bullish_scores():
    indicators = {
        "rsi": {"rsi_14": 65},
        "macd": {"macd_line": 1.0, "signal_line": 0.5, "histogram": 0.5},
        "moving_averages": {"sma_50": 100, "sma_200": 90},
        "adx": {"adx": 28},
        "bollinger_bands": {"percent_b": 0.85},
        "volume": {"volume_ratio": 1.6},
        "cmf": {"value": 0.1},
        "enhanced_momentum": {"stochrsi_k": 30, "stochrsi_d": 35},
        "ichimoku": {"signal": "bullish"},
        "keltner": {"signal": "inside"},
        "supertrend": {"direction": "up"},
    }
    summary = compute_signals_summary({"day": indicators})
    assert summary.consensus_score > 0
    assert summary.consensus_bias == "bullish"
    assert 0.1 <= summary.confidence <= 0.95
    assert summary.per_timeframe and summary.per_timeframe[0].bias == "bullish"


def test_basic_bearish_scores():
    indicators = {
        "rsi": {"rsi_14": 35},
        "macd": {"macd_line": -1.0, "signal_line": 0.5, "histogram": -1.5},
        "moving_averages": {"sma_50": 100, "sma_200": 110},
        "adx": {"adx": 20},
        "bollinger_bands": {"percent_b": 0.15},
        "volume": {"volume_ratio": 0.4},
        "cmf": {"value": -0.1},
        "enhanced_momentum": {"stochrsi_k": 10, "stochrsi_d": 15},
        "ichimoku": {"signal": "bearish"},
        "keltner": {"signal": "oversold"},
        "supertrend": {"direction": "down"},
    }
    summary = compute_signals_summary({"day": indicators})
    assert summary.consensus_score < 0
    assert summary.consensus_bias == "bearish"
    assert 0.1 <= summary.confidence <= 0.95


