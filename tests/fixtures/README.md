# Fixtures for quick backtests

Put JSON files here representing per-timeframe indicators for a symbol. Example:

```
{
  "day": { "rsi": {"rsi_14": 62}, "macd": {"macd_line": 0.5, "signal_line": 0.2}, "moving_averages": {"sma_50": 100, "sma_200": 95} },
  "week": { "rsi": {"rsi_14": 58}, "macd": {"macd_line": 0.3, "signal_line": 0.1} },
  "target_bias": "bullish"
}
```

The `target_bias` field is used to score calibration.


