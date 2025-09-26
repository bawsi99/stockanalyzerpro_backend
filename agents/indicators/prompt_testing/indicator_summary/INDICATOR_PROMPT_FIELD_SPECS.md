# Indicator Summary Prompt Field Specifications

This document explains exactly how each field in the indicator_summary prompt is computed, including file locations and formulas used. It references the testing prompt:

backend/agents/indicators/prompt testing/indicator_summary/prompt_HDFCBANK_20250926_165218.txt

Sections below map to the "Technical Data" blocks in the prompt, the "Levels" block, and the confidence formatting.

---

## Data sources and orchestration

- Deterministic indicator calculations (numeric values) are produced by:
  - File: backend/ml/indicators/technical_indicators.py
  - Function: TechnicalIndicators.calculate_all_indicators_optimized(data: pd.DataFrame, stock_symbol: str)

- Agent qualitative summaries (trend and momentum direction/strength/confidence) are produced by:
  - Trend: backend/agents/indicators/trend/processor.py (class TrendIndicatorsProcessor)
  - Momentum: backend/agents/indicators/momentum/processor.py (class MomentumIndicatorsProcessor)

- Curation/merging into the prompt context happens in:
  - File: backend/analysis/orchestrator.py
  - Method: StockAnalysisOrchestrator._curate_indicators_from_agents(unified, raw_indicators, stock_data)
    - Merges numeric values (SMA/EMA/RSI/MACD/volume) into key_indicators
    - Derives critical support/resistance via TechnicalIndicators.detect_support_resistance
    - Computes detected_conflicts via ContextEngineer

- Prompt formatting and confidence percentage rendering are handled by:
  - File: backend/gemini/context_engineer.py
  - Method: ContextEngineer._structure_indicator_summary_context
    - Formats trend/momentum "confidence" as a percentage string with two decimals for prompt output only.

---

## Technical Data: trend_indicators

Fields and formulas:

- sma_20, sma_50, sma_200 (floats)
  - Source: TechnicalIndicators.calculate_sma(data, 'close', N)
  - Mathematical definition (simple moving average over N periods):
    - SMA_N(t) = (1/N) * Σ_{i=0..N-1} close[t - i]
  - In optimized path, we use only the current (last) value for each period N:
    - sma_20 = float(SMA_20 at the latest bar)
    - sma_50 = float(SMA_50 at the latest bar)
    - sma_200 = float(SMA_200 at the latest bar)
  - Edge-case fallback (insufficient data): SMA_200 may proxy to SMA_50 or last close per calculate_all_indicators_optimized safeguards.

- ema_20, ema_50 (floats)
  - Source: TechnicalIndicators.calculate_ema(data, 'close', N)
  - Mathematical definition (exponential moving average, smoothing factor α = 2/(N+1)):
    - EMA_N(t) = α * close[t] + (1 - α) * EMA_N(t-1)
    - Initialization uses pandas ewm(span=N) convention for the series; we surface the latest (current) EMA.
  - In optimized path, we use only the current (last) value for each period N:
    - ema_20 = float(EMA_20 at the latest bar)
    - ema_50 = float(EMA_50 at the latest bar)

- price_to_sma_200 (float, decimal ratio)
  - Source: TechnicalIndicators.calculate_all_indicators_optimized
  - Formula: (current_price / sma_200_last) - 1
  - Example: 0.03 means current price is 3% above the SMA 200.

- sma_20_to_sma_50 (float, decimal ratio)
  - Source: TechnicalIndicators.calculate_all_indicators_optimized
  - Formula: (sma_20_last / sma_50_last) - 1
  - Example: -0.01 means SMA 20 is 1% below SMA 50.

- golden_cross, death_cross (bool)
  - Source: TechnicalIndicators.calculate_all_indicators_optimized
  - Formulas (using legacy arrays internally when available):
    - golden_cross = (SMA20_last > SMA50_last) and (SMA20_prev <= SMA50_prev)
    - death_cross = (SMA20_last < SMA50_last) and (SMA20_prev >= SMA50_prev)
  - In optimized output, these are provided as booleans.

- direction, strength, confidence (qualitative + numeric)
  - Source: TrendIndicatorsProcessor (backend/agents/indicators/trend/processor.py)

  Trend Direction determination
  - Short/Medium/Long-term directions from _calculate_direction(prices_window):
    - Fit linear regression slope via numpy.polyfit over windows (10/30/60 bars).
    - Normalize slope by mean price in the window.
    - Thresholds:
      - normalized_slope > +0.002 => 'bullish'
      - normalized_slope < -0.002 => 'bearish'
      - otherwise => 'neutral'
  - direction_consensus = majority of [short, medium, long]

  Trend Strength determination
  - _assess_overall_strength(strength_analysis):
    - Combines ADX-based reading and trend consistency:
      - ADX strength level: 'weak' (<20), 'moderate' (<40), 'strong' (>=40)
      - Consistency: blend of direction consistency and slope consistency across recent windows
    - Rules (simplified):
      - strong if ADX='strong' and consistency > 0.7
      - moderate if ADX in {'moderate','strong'} and consistency > 0.5
      - else weak

  Confidence score (numeric 0..1)
  - _calculate_confidence_score(ma_analysis, strength_analysis, direction_analysis): average of:
    - direction_confidence = max(count of one direction in [short,medium,long]) / 3
    - strength_score mapped from strength:
      - strong => 1.0, moderate => 0.7, weak => 0.3
    - alignment_score if bullish/bearish alignment detected in _assess_ma_alignment
      - alignment_strength ~ min(1.0, num_MAs_considered/3.0)

  Prompt display of confidence (percentage)
  - ContextEngineer formats confidence to a percentage string in the prompt:
    - If value in [0,1], multiply by 100; then format with 2 decimals, e.g., "74.33%".
    - Only for prompt rendering — underlying numeric confidence remains unchanged in code.

---

## Technical Data: momentum_indicators

Fields and formulas:

- rsi_current (float)
  - Source: TechnicalIndicators.calculate_rsi(data)
  - In optimized path, we expose RSI via indicators['rsi'] dict:
    - rsi_14 = last 14-period RSI value
  - In prompt curation: momentum_indicators.rsi_current = round(rsi_14, 2)

- rsi_status (string)
  - Source: TechnicalIndicators.calculate_all_indicators_optimized
  - Mapping from RSI value:
    - > 70 => 'overbought'
    - 60..70 => 'near_overbought'
    - < 30 => 'oversold'
    - <= 40 => 'near_oversold'
    - else => 'neutral'

- macd.histogram (float) and macd trend (string)
  - Source: TechnicalIndicators.calculate_macd(data)
    - Returns MACD line, signal line, and histogram (MACD - signal).
  - In optimized path, we provide a dict with last macd_line/signal_line/histogram values.
  - In prompt curation: histogram is rounded; trend set to the momentum direction or 'neutral' if not assessed.

- direction, strength, confidence (qualitative + numeric)
  - Source: MomentumIndicatorsProcessor (backend/agents/indicators/momentum/processor.py)

  Momentum Direction determination
  - Signals gathered from:
    - RSI signal: overbought/oversold/bullish/bearish (>=70, <=30, >=50, <50)
    - MACD signal: 'bullish' if macd_line > signal_line else 'bearish'
    - Stochastic: 'bullish' if %K > %D else 'bearish' (when available)
  - direction = majority of available {bullish, bearish}; 'neutral' if tie or no signals

  Momentum Strength determination
  - If confidence > 0.7 => 'strong'
  - Else if confidence > 0.5 => 'moderate'
  - Else 'weak'

  Confidence score (numeric 0..1)
  - _calculate_confidence_score(rsi_analysis, macd_analysis, stoch_analysis)
  - Reflects alignment/consistency of RSI/MACD/Stoch signals (more reinforcing signals => higher confidence)

  Prompt display of confidence (percentage)
  - Same formatting rule as for trend: shown as a percentage string with two decimals only in the prompt.

---

## Technical Data: volume_indicators

- volume_ratio (float)
  - Source: TechnicalIndicators.calculate_all_indicators_optimized
  - Formula: current_volume / SMA(volume, 20)
  - Safe fallback to 1.0 if the denominator is zero/NaN.

- volume_trend (string)
  - Currently set to a basic default ('neutral') in the curated prompt unless enhanced volume logic is present.
  - Note: Additional volume metrics (OBV, trend) are computed in indicators['volume'] but not all are surfaced in this specific prompt block.

---

## Levels (critical_levels)

- support, resistance (arrays of floats)
  - Source (in prompt curation): TechnicalIndicators.detect_support_resistance(stock_data)
  - The orchestration deduplicates and rounds, keeping up to 3 most relevant:
    - support: top 3 highest support levels (descending)
    - resistance: top 3 lowest resistance levels (ascending)
  - Implementation details reside in TechnicalIndicators; typically derived from recent swing points or pivot logic. If unavailable, basic fallbacks may be used.

---

## Signal Conflicts

- Source: ContextEngineer (backend/gemini/context_engineer.py)
  - Method: _comprehensive_conflict_analysis(key_indicators)
  - Uses multiple rule layers to detect conflicts:
    - Momentum vs Trend (e.g., RSI overbought while MACD bullish)
    - Timeframe conflicts (e.g., SMA alignment inconsistent with long-term bias)
    - Volume confirmation, signal strength consistency, market regime checks, statistical divergences, meta-analysis
  - Output: conflict_count, severity (none/low/medium/high/critical), and categorized lists.
  - In prompt, shown as a simplified “Signal Conflicts” section summarizing top conflicts.

---

## Confidence Formatting in Prompt

- File: backend/gemini/context_engineer.py
- Method: _structure_indicator_summary_context
- Behaviour:
  - Deep-copies the key_indicators dict and transforms:
    - trend_indicators.confidence
    - momentum_indicators.confidence
  - If numeric in [0,1], multiply by 100; render as "XX.XX%".
  - If already in 0..100, just render as "XX.XX%".
  - This is purely for prompt text; computation values remain unchanged elsewhere.

---

## Notes

- Rounding in the prompt is applied at curation/formatting time: generally to two decimals for numeric prints.
- All formulas are robust to insufficient data via safe fallbacks implemented in TechnicalIndicators and the agent processors.
- If you need exact code references (line numbers), search for the functions listed above in the specified files — the optimized indicator calculations are clustered around the calculate_all_indicators_optimized implementation, and the agent logic is in the two processor classes described.
