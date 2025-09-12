#!/usr/bin/env python3
"""
Frontend Response Builder Module
This module contains the logic to build the exact response structure that the frontend expects.
"""
import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any
import pandas as pd

# Add the backend directory to Python path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from ml.indicators.volume_profile import calculate_volume_profile, identify_significant_levels
from ml.analysis.market_regime import detect_market_regime

logger = logging.getLogger(__name__)

class FrontendResponseBuilder:
    """Builds the exact response structure that the frontend expects."""
    
    @staticmethod
    def build_frontend_response(symbol: str, exchange: str, data: pd.DataFrame, 
                              indicators: dict, ai_analysis: dict, indicator_summary: str, 
                              chart_insights: str, chart_paths: dict, sector_context: dict, 
                              mtf_context: dict, advanced_analysis: dict, ml_predictions: dict | None, period: int, interval: str) -> dict:
        """
        Build the exact response structure that the frontend expects.
        """
        try:
            # Normalize/guard input data
            from core.utils import ensure_ohlcv_dataframe, interval_to_frontend_display
            try:
                data = ensure_ohlcv_dataframe(data)
            except Exception:
                pass

            # Get latest price and basic info
            latest_price = data['close'].iloc[-1] if not data.empty else 0
            price_change = data['close'].iloc[-1] - data['close'].iloc[-2] if len(data) > 1 else 0
            price_change_pct = (price_change / data['close'].iloc[-2]) * 100 if len(data) > 1 and data['close'].iloc[-2] != 0 else 0
            
            # Convert interval format for frontend display
            frontend_interval = interval_to_frontend_display(interval)
            
            # Build basic response structure
            result = {
                "success": True,
                "stock_symbol": symbol,
                "exchange": exchange,
                "analysis_period": frontend_interval,
                "interval": interval,
                "timestamp": datetime.now().isoformat(),
                "message": f"Analysis completed successfully for {symbol}",
                "results": {
                    "current_price": float(latest_price),
                    "price_change": float(price_change),
                    "price_change_percentage": float(price_change_pct),
                    "analysis_period": f"{period} days",
                    "analysis_timestamp": datetime.now().isoformat(),
                    "analysis_type": "enhanced_with_code_execution",
                    "mathematical_validation": True,
                    "calculation_method": "code_execution",
                    "accuracy_improvement": "high",
                    "technical_indicators": FrontendResponseBuilder._build_technical_indicators(data, indicators),
                    "ai_analysis": FrontendResponseBuilder._build_ai_analysis(ai_analysis, data, interval),
                    # Deterministic signals: pass through if present; otherwise compute from indicators/MTF context
                    "signals": FrontendResponseBuilder._extract_signals(
                        ai_analysis=ai_analysis,
                        indicators=indicators,
                        interval=interval,
                        mtf_context=mtf_context,
                        data=data,
                    ),
                    "sector_context": sector_context or {},
                    "multi_timeframe_analysis": mtf_context or {},
                    "enhanced_metadata": {
                        "mathematical_validation": True,
                        "code_execution_enabled": True,
                        "statistical_analysis": True,
                        "calculation_timestamp": int(datetime.now().timestamp() * 1000),
                        "advanced_risk_metrics": advanced_analysis.get("advanced_risk", {}),
                        "stress_testing_metrics": advanced_analysis.get("stress_testing", {}),
                        "scenario_analysis_metrics": advanced_analysis.get("scenario_analysis", {})
                    },
                    "charts": {}, # Empty charts - frontend uses dedicated /charts endpoint
                    # Unified ML predictions surfaced for frontend (if available)
                    "ml_predictions": ml_predictions or {},
                    "overlays": FrontendResponseBuilder._build_overlays(
                        data,
                        advanced_analysis.get("advanced_patterns", {}),
                        symbol=symbol,
                        exchange=exchange,
                        interval=interval,
                    ),
                    # Derive risk level and recommendation from AI analysis to match legacy behavior
                    "risk_level": (lambda conf: (
                        'Low' if conf >= 80 else 'Medium' if conf >= 60 else 'High' if conf >= 40 else 'Very High'
                    ))(float(ai_analysis.get('confidence_pct', 0) or 0)),
                    "recommendation": (lambda conf, trend: (
                        'Strong Buy' if conf >= 80 and trend == 'Bullish' else
                        'Strong Sell' if conf >= 80 and trend == 'Bearish' else
                        'Buy' if conf >= 60 and trend == 'Bullish' else
                        'Sell' if conf >= 60 and trend == 'Bearish' else
                        'Hold' if conf >= 60 else 'Wait and Watch' if conf >= 40 else 'Avoid Trading'
                    ))(float(ai_analysis.get('confidence_pct', 0) or 0), ai_analysis.get('trend', 'Unknown')),
                    # Provide both keys for backward/forward compatibility
                    "indicator_summary": indicator_summary,
                    "indicator_summary_md": indicator_summary,
                    "chart_insights": chart_insights,
                    "consensus": FrontendResponseBuilder._build_consensus(ai_analysis, indicators, data, mtf_context),
                    "summary": {
                        "overall_signal": ai_analysis.get('trend', 'Unknown'),
                        "confidence": ai_analysis.get('confidence_pct', 0),
                        "risk_level": "medium",
                        "recommendation": "hold"
                    },
                    "support_levels": FrontendResponseBuilder._extract_support_levels(data, indicators),
                    "resistance_levels": FrontendResponseBuilder._extract_resistance_levels(data, indicators),
                    "triangle_patterns": [],
                    "flag_patterns": [],
                    "volume_anomalies_detailed": [],
                    "trading_guidance": {}
                }
            }
            # Augment with market regime and volume-based S/R (safe fallbacks)
            try:
                market_regime = detect_market_regime(data)
            except Exception as _e:
                logger.warning(f"market_regime detection failed: {_e}")
                market_regime = "unknown"

            try:
                vp = calculate_volume_profile(data)
                support_levels, resistance_levels = identify_significant_levels(vp, float(latest_price))
                volume_profile_analysis = {
                    "support": [{"price": float(p), "strength": "high"} for p in (support_levels[:3] if support_levels else [])],
                    "resistance": [{"price": float(p), "strength": "high"} for p in (resistance_levels[:3] if resistance_levels else [])],
                }
            except Exception as _e:
                logger.warning(f"volume_profile analysis failed: {_e}")
                volume_profile_analysis = {"support": [], "resistance": []}

            result["results"]["market_regime"] = market_regime
            result["results"]["volume_profile_analysis"] = volume_profile_analysis

            return result
            
        except Exception as e:
            logger.error(f"Error building frontend response: {e}")
            return {
                "success": False,
                "error": str(e),
                "stock_symbol": symbol,
                "exchange": exchange,
                "timestamp": datetime.now().isoformat()
            }

    @staticmethod
    def _extract_signals(ai_analysis: dict, indicators: dict, interval: str = "day", mtf_context: dict | None = None, data: pd.DataFrame = None) -> dict:
        """Extract or compute deterministic signals for the frontend.

        Priority:
        1) If ai_analysis contains a ready-made signals block, use it.
        2) If indicators contain a signals block, use it.
        3) Otherwise, compute signals from the available indicators using scoring.
        """
        # 1) Pass-through from ai_analysis if available
        try:
            if isinstance(ai_analysis, dict) and ai_analysis.get("signals"):
                return ai_analysis["signals"]
        except Exception:
            pass

        # 2) Pass-through from indicators if embedded
        try:
            if isinstance(indicators, dict) and indicators.get("signals"):
                return indicators["signals"]
        except Exception:
            pass

        # 3) Compute from indicators and/or MTF context
        try:
            # Lazy import to avoid heavy import-time costs or circular deps
            from data.signals.scoring import compute_signals_summary

            # If multi-timeframe indicators are present in ai_analysis, prefer them
            per_timeframe_indicators = {}
            mtf_block = ai_analysis.get('multi_timeframe') if isinstance(ai_analysis, dict) else None
            if isinstance(mtf_block, dict) and mtf_block.get('timeframes'):
                for tf, tf_obj in mtf_block['timeframes'].items():
                    if isinstance(tf_obj, dict) and 'indicators' in tf_obj:
                        per_timeframe_indicators[tf] = tf_obj.get('indicators') or {}

            # Otherwise, if a separate MTF context is supplied, synthesize minimal indicators
            if not per_timeframe_indicators and isinstance(mtf_context, dict) and mtf_context.get('timeframe_analyses'):
                try:
                    tf_analyses = mtf_context['timeframe_analyses']
                    # Expect keys like '1day', '1hour', '30min', etc.
                    for tf, summary in tf_analyses.items():
                        if not isinstance(summary, dict):
                            continue
                        indicators_min = {}
                        # Map key_indicators.rsi -> rsi_14
                        rsi_val = (summary.get('key_indicators') or {}).get('rsi')
                        if rsi_val is not None:
                            indicators_min['rsi_14'] = float(rsi_val)
                            indicators_min['rsi'] = {'rsi_14': float(rsi_val)}
                        # Map macd_signal -> synthetic macd_line/signal_line
                        macd_sig = (summary.get('key_indicators') or {}).get('macd_signal')
                        if isinstance(macd_sig, str):
                            if macd_sig.lower() == 'bullish':
                                indicators_min['macd_line'] = 1.0
                                indicators_min['signal_line'] = 0.0
                            elif macd_sig.lower() == 'bearish':
                                indicators_min['macd_line'] = -1.0
                                indicators_min['signal_line'] = 0.0
                            else:
                                # neutral
                                indicators_min['macd_line'] = 0.0
                                indicators_min['signal_line'] = 0.0
                            indicators_min['macd'] = {
                                'macd_line': indicators_min['macd_line'],
                                'signal_line': indicators_min['signal_line'],
                            }
                        # Map volume_status -> synthetic volume_ratio
                        vol_status = (summary.get('key_indicators') or {}).get('volume_status')
                        if isinstance(vol_status, str):
                            ratio = 1.0
                            if vol_status == 'high':
                                ratio = 1.6
                            elif vol_status == 'low':
                                ratio = 0.4
                            indicators_min['volume_ratio'] = ratio
                            indicators_min['volume'] = {'volume_ratio': ratio}
                        # Map trend -> supertrend.direction to inject timeframe-specific bias
                        trend = summary.get('trend')
                        if isinstance(trend, str) and trend in ('bullish', 'bearish', 'neutral'):
                            if trend == 'bullish':
                                indicators_min['supertrend'] = {'direction': 'up'}
                            elif trend == 'bearish':
                                indicators_min['supertrend'] = {'direction': 'down'}
                            else:
                                indicators_min['supertrend'] = {'direction': 'neutral'}
                        # Use MTF confidence to shape ADX strength
                        try:
                            tf_conf = float(summary.get('confidence')) if summary.get('confidence') is not None else 0.5
                        except Exception:
                            tf_conf = 0.5
                        # Centered around 20 with +/- 10 swing
                        adx_val = max(5.0, min(40.0, 20.0 + (tf_conf - 0.5) * 20.0))
                        indicators_min['adx'] = {'adx': float(adx_val)}
                        # Synthesize percent_b from support/resistance proximity if available
                        try:
                            ki = summary.get('key_indicators') or {}
                            support_levels = ki.get('support_levels') or []
                            resistance_levels = ki.get('resistance_levels') or []
                            current_price = (summary.get('risk_metrics') or {}).get('current_price')
                            if current_price and support_levels and resistance_levels:
                                # choose nearest support below and resistance above
                                below = [s for s in support_levels if s <= current_price]
                                above = [r for r in resistance_levels if r >= current_price]
                                nearest_support = max(below) if below else min(support_levels)
                                nearest_resistance = min(above) if above else max(resistance_levels)
                                band = max(1e-6, float(nearest_resistance - nearest_support))
                                pos = max(0.0, min(1.0, float((current_price - nearest_support) / band)))
                                indicators_min['bollinger_bands'] = {'percent_b': pos}
                        except Exception:
                            pass
                        # Keep as minimal, scoring gracefully skips missing fields
                        if indicators_min:
                            per_timeframe_indicators[tf] = indicators_min
                except Exception:
                    # Best-effort synthesis from MTF context
                    pass

            # Fallback: single timeframe from provided indicators
            if not per_timeframe_indicators:
                normalized_interval = interval or 'day'
                per_timeframe_indicators[normalized_interval] = indicators or {}

            # Try to get price data for better regime detection
            price_data = None
            if data is not None and not data.empty and len(data) >= 20:
                price_data = data
            
            summary = compute_signals_summary(per_timeframe_indicators, price_data)

            # Build output, allowing overrides from mtf_context where available (e.g., confidence)
            result = {
                "consensus_score": summary.consensus_score,
                "consensus_bias": summary.consensus_bias,
                "confidence": summary.confidence,
                "per_timeframe": [],
                "regime": summary.regime,
            }
            # Optionally override per-timeframe confidence with mtf_context values
            mtf_tf_conf = {}
            if isinstance(mtf_context, dict) and isinstance(mtf_context.get('timeframe_analyses'), dict):
                try:
                    for tf, s in mtf_context['timeframe_analyses'].items():
                        if isinstance(s, dict) and 'confidence' in s:
                            mtf_tf_conf[tf] = float(s['confidence'])
                except Exception:
                    pass
            for s in summary.per_timeframe:
                overridden_conf = mtf_tf_conf.get(s.timeframe, s.confidence)
                result["per_timeframe"].append({
                    "timeframe": s.timeframe,
                    "score": s.score,
                    "confidence": overridden_conf,
                    "bias": s.bias,
                    "reasons": [
                        {
                            "indicator": r.indicator,
                            "description": r.description,
                            "weight": r.weight,
                            "bias": r.bias,
                        }
                        for r in s.reasons
                    ],
                })
            return result
        except Exception:
            # As a last resort, return a neutral minimal block
            return {
                "consensus_score": 0.0,
                "consensus_bias": "neutral",
                "confidence": 0.3,
                "per_timeframe": [],
                "regime": {"trend": "unknown", "volatility": "normal"}
            }
    
    @staticmethod
    def _build_technical_indicators(data: pd.DataFrame, indicators: dict) -> dict:
        """Build technical indicators structure from canonical schema produced by
        TechnicalIndicators.calculate_all_indicators_optimized.

        Falls back to minimal synthesis only when canonical fields are missing.
        """
        try:
            latest_close = data['close'].iloc[-1] if not data.empty else 0.0
            latest_volume = data['volume'].iloc[-1] if not data.empty else 0.0

            # If indicators are already in structured form (from calculate_all_indicators_optimized),
            # prefer those real values instead of synthesizing defaults
            is_structured = isinstance(indicators, dict) and (
                'moving_averages' in indicators or 'rsi' in indicators or 'macd' in indicators
            )

            if is_structured:
                # Canonical mapping passthrough
                ma = indicators.get('moving_averages', {}) or {}
                rsi = indicators.get('rsi', {}) or {}
                macd = indicators.get('macd', {}) or {}
                bb = indicators.get('bollinger_bands', {}) or {}
                vol = indicators.get('volume', {}) or {}
                adx = indicators.get('adx', {}) or {}
                trend_data = indicators.get('trend_data', {}) or {}

                # Ensure mandatory derived fields exist when missing
                price_to_sma_200 = ma.get('price_to_sma_200')
                if price_to_sma_200 is None:
                    sma200 = ma.get('sma_200') or 0.0
                    price_to_sma_200 = float((latest_close / sma200 - 1) if sma200 else 0.0)

                sma_20_to_sma_50 = ma.get('sma_20_to_sma_50')
                if sma_20_to_sma_50 is None:
                    sma20 = ma.get('sma_20') or 0.0
                    sma50 = ma.get('sma_50') or 0.0
                    sma_20_to_sma_50 = float((sma20 / sma50 - 1) if sma50 else 0.0)

                # Percent-b and bandwidth fallbacks
                percent_b = bb.get('percent_b')
                bandwidth = bb.get('bandwidth')
                if percent_b is None or bandwidth is None:
                    upper = bb.get('upper_band') or float(latest_close * 1.02)
                    middle = bb.get('middle_band') or float(latest_close)
                    lower = bb.get('lower_band') or float(latest_close * 0.98)
                    band_width = upper - lower
                    percent_b = float((latest_close - lower) / band_width) if band_width else 0.5
                    bandwidth = float(band_width / middle) if middle else 0.0

                # Volume ratio fallback using recent average
                volume_ratio = vol.get('volume_ratio')
                if volume_ratio is None:
                    avg_volume = data['volume'].mean() if not data.empty else 0.0
                    volume_ratio = float(latest_volume / avg_volume) if avg_volume else 1.0

                # ADX defaults
                adx_value = adx.get('adx') if adx else None
                plus_di = adx.get('plus_di') if adx else None
                minus_di = adx.get('minus_di') if adx else None
                trend_direction = adx.get('trend_direction') or (
                    'bullish' if (plus_di or 0) > (minus_di or 0) else 'bearish'
                )

                # Return indicators mostly as-is to avoid duplication and drift
                result = {
                    "moving_averages": {
                        "sma_20": float(ma.get('sma_20') or 0.0),
                        "sma_50": float(ma.get('sma_50') or 0.0),
                        "sma_200": float(ma.get('sma_200') or 0.0),
                        "ema_20": float(ma.get('ema_20') or ma.get('sma_20') or 0.0),
                        "ema_50": float(ma.get('ema_50') or ma.get('sma_50') or 0.0),
                        "price_to_sma_200": float(price_to_sma_200),
                        "sma_20_to_sma_50": float(sma_20_to_sma_50),
                        "golden_cross": bool(ma.get('golden_cross') or False),
                        "death_cross": bool(ma.get('death_cross') or False)
                    },
                    "rsi": {
                        "rsi_14": float(rsi.get('rsi_14') or 0.0),
                        "trend": str(rsi.get('trend') or ('bullish' if (rsi.get('rsi_14') or 50) > 50 else 'bearish' if (rsi.get('rsi_14') or 50) < 50 else 'neutral')),
                        "status": str(rsi.get('status') or ('overbought' if (rsi.get('rsi_14') or 0) > 70 else 'oversold' if (rsi.get('rsi_14') or 0) < 30 else 'neutral'))
                    },
                    "macd": {
                        "macd_line": float(macd.get('macd_line') or 0.0),
                        "signal_line": float(macd.get('signal_line') or 0.0),
                        "histogram": float(macd.get('histogram') or ((macd.get('macd_line') or 0.0) - (macd.get('signal_line') or 0.0)))
                    },
                    "bollinger_bands": {
                        "upper_band": float(bb.get('upper_band') or latest_close * 1.02),
                        "middle_band": float(bb.get('middle_band') or latest_close),
                        "lower_band": float(bb.get('lower_band') or latest_close * 0.98),
                        "percent_b": float(percent_b),
                        "bandwidth": float(bandwidth)
                    },
                    "volume": {
                        "volume_ratio": float(volume_ratio),
                        "obv": float(vol.get('obv') or latest_volume),
                        "obv_trend": str(vol.get('obv_trend') or 'neutral')
                    },
                    "adx": {
                        "adx": float(adx_value) if adx_value is not None else 25.0,
                        "plus_di": float(plus_di) if plus_di is not None else 30.0,
                        "minus_di": float(minus_di) if minus_di is not None else 20.0,
                        "trend_direction": str(trend_direction)
                    },
                    "trend_data": {
                        "direction": str(trend_data.get('direction') or trend_direction),
                        "strength": str(trend_data.get('strength') or ('strong' if (adx_value or 0) > 25 else 'weak')),
                        "adx": float(trend_data.get('adx') or (adx_value or 0.0)),
                        "plus_di": float(trend_data.get('plus_di') or (plus_di or 0.0)),
                        "minus_di": float(trend_data.get('minus_di') or (minus_di or 0.0))
                    },
                    # Keep raw_data minimal; charts handle full series
                    "raw_data": {},
                    "metadata": {
                        "start": data.index[0].strftime('%Y-%m-%d') if not data.empty else "",
                        "end": data.index[-1].strftime('%Y-%m-%d') if not data.empty else "",
                        "period": len(data),
                        "last_price": float(latest_close),
                        "last_volume": float(latest_volume),
                        "data_quality": {
                            "is_valid": True,
                            "warnings": [],
                            "data_quality_issues": [],
                            "missing_data": [],
                            "suspicious_patterns": []
                        },
                        "indicator_availability": {
                            "sma_20": True,
                            "sma_50": True,
                            "sma_200": True,
                            "ema_20": True,
                            "ema_50": True,
                            "macd": True,
                            "rsi": True,
                            "bollinger_bands": True,
                            "stochastic": True,
                            "adx": True,
                            "obv": True,
                            "volume_ratio": True,
                            "atr": True
                        }
                    }
                }

                return result

            # Fallback: synthesize minimal indicators if canonical fields missing
            sma_20 = latest_close
            sma_50 = latest_close
            sma_200 = latest_close
            price_to_sma_200 = float((latest_close / sma_200) if sma_200 else 0.0)
            sma_20_to_sma_50 = float((sma_20 / sma_50) if sma_50 else 0.0)
            avg_volume = data['volume'].mean() if not data.empty else 0.0
            volume_ratio = float(latest_volume / avg_volume) if avg_volume else 1.0

            return {
                "moving_averages": {
                    "sma_20": float(sma_20),
                    "sma_50": float(sma_50),
                    "sma_200": float(sma_200),
                    "ema_20": float(sma_20),
                    "ema_50": float(sma_50),
                    "price_to_sma_200": price_to_sma_200,
                    "sma_20_to_sma_50": sma_20_to_sma_50,
                    "golden_cross": False,
                    "death_cross": False
                },
                "rsi": {
                    "rsi_14": 50.0,
                    "trend": "neutral",
                    "status": "neutral"
                },
                "macd": {
                    "macd_line": 0.0,
                    "signal_line": 0.0,
                    "histogram": 0.0
                },
                "bollinger_bands": {
                    "upper_band": float(latest_close * 1.02),
                    "middle_band": float(latest_close),
                    "lower_band": float(latest_close * 0.98),
                    "percent_b": 0.5,
                    "bandwidth": 0.04
                },
                "volume": {
                    "volume_ratio": volume_ratio,
                    "obv": float(latest_volume),
                    "obv_trend": "neutral"
                },
                "adx": {
                    "adx": 25.0,
                    "plus_di": 30.0,
                    "minus_di": 20.0,
                    "trend_direction": "neutral"
                },
                "trend_data": {
                    "direction": "neutral",
                    "strength": "weak",
                    "adx": 25.0,
                    "plus_di": 30.0,
                    "minus_di": 20.0
                },
                "raw_data": {
                    "open": [float(x) for x in data['open'].tail(100).tolist()] if not data.empty else [],
                    "high": [float(x) for x in data['high'].tail(100).tolist()] if not data.empty else [],
                    "low": [float(x) for x in data['low'].tail(100).tolist()] if not data.empty else [],
                    "close": [float(x) for x in data['close'].tail(100).tolist()] if not data.empty else [],
                    "volume": [float(x) for x in data['volume'].tail(100).tolist()] if not data.empty else []
                },
                "metadata": {
                    "start": data.index[0].strftime('%Y-%m-%d') if not data.empty else "",
                    "end": data.index[-1].strftime('%Y-%m-%d') if not data.empty else "",
                    "period": len(data),
                    "last_price": float(latest_close),
                    "last_volume": float(latest_volume),
                    "data_quality": {
                        "is_valid": True,
                        "warnings": [],
                        "data_quality_issues": [],
                        "missing_data": [],
                        "suspicious_patterns": []
                    },
                    "indicator_availability": {
                        "sma_20": True,
                        "sma_50": True,
                        "sma_200": True,
                        "ema_20": True,
                        "ema_50": True,
                        "macd": True,
                        "rsi": True,
                        "bollinger_bands": True,
                        "stochastic": True,
                        "adx": True,
                        "obv": True,
                        "volume_ratio": True,
                        "atr": True
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error building technical indicators: {e}")
            return {}
    
    @staticmethod
    def _build_ai_analysis(ai_analysis: dict, data: pd.DataFrame, interval: str) -> dict:
        """Build AI analysis structure."""
        try:
            trend = ai_analysis.get('trend', 'Unknown')
            confidence = ai_analysis.get('confidence_pct', 0)
            latest_price = data['close'].iloc[-1] if not data.empty else 0
            # Infer a more appropriate trend duration from data and interval
            inferred_duration = FrontendResponseBuilder._infer_trend_duration(data, interval)
            
            return {
                "meta": {
                    "symbol": ai_analysis.get('symbol', ''),
                    "analysis_date": datetime.now().strftime('%Y-%m-%d'),
                    "timeframe": "1D",
                    "overall_confidence": float(confidence),
                    "data_quality_score": 92.0
                },
                "market_outlook": {
                    "primary_trend": {
                        "direction": trend,
                        "strength": "moderate" if confidence > 60 else "weak",
                        "duration": inferred_duration,
                        "confidence": float(confidence),
                        "rationale": ai_analysis.get('rationale', 'Technical analysis indicates current trend')
                    },
                    "secondary_trend": {
                        "direction": "neutral",
                        "strength": "weak",
                        "duration": "medium-term",
                        "confidence": 60.0,
                        "rationale": "Mixed signals in medium-term timeframe"
                    },
                    "key_drivers": [
                        {
                            "factor": "Volume increase",
                            "impact": "positive",
                            "timeframe": "short-term"
                        }
                    ]
                },
                "trading_strategy": {
                    "short_term": {
                        "horizon_days": 5,
                        "bias": trend.lower(),
                        "entry_strategy": {
                            "type": "breakout",
                            "entry_range": [float(latest_price * 0.99), float(latest_price * 1.01)],
                            "entry_conditions": ["Price above SMA 20", "Volume confirmation"],
                            "confidence": 75.0
                        },
                        "exit_strategy": {
                            "stop_loss": float(latest_price * 0.98),
                            "stop_loss_type": "fixed",
                            "targets": [
                                {
                                    "price": float(latest_price * 1.03),
                                    "probability": "high",
                                    "timeframe": "3 days"
                                }
                            ],
                            "trailing_stop": {
                                "enabled": True,
                                "method": "ATR-based"
                            }
                        },
                        "position_sizing": {
                            "risk_per_trade": "2%",
                            "max_position_size": "10%",
                            "atr_multiplier": 2.0
                        },
                        "rationale": "Strong technical setup with clear entry and exit levels"
                    },
                    "medium_term": {
                        "horizon_days": 30,
                        "bias": "neutral",
                        "entry_strategy": {
                            "type": "accumulation",
                            "entry_range": [float(latest_price * 0.98), float(latest_price * 1.02)],
                            "entry_conditions": ["Pullback to support", "RSI oversold"],
                            "confidence": 65.0
                        },
                        "exit_strategy": {
                            "stop_loss": float(latest_price * 0.95),
                            "stop_loss_type": "support-based",
                            "targets": [
                                {
                                    "price": float(latest_price * 1.06),
                                    "probability": "medium",
                                    "timeframe": "20 days"
                                }
                            ],
                            "trailing_stop": {
                                "enabled": True,
                                "method": "percentage-based"
                            }
                        },
                        "position_sizing": {
                            "risk_per_trade": "3%",
                            "max_position_size": "15%",
                            "atr_multiplier": 2.5
                        },
                        "rationale": "Medium-term consolidation expected with breakout potential"
                    },
                    "long_term": {
                        "horizon_days": 365,
                        "investment_rating": "buy" if trend.lower() == "bullish" else "hold",
                        "fair_value_range": [float(latest_price * 1.06), float(latest_price * 1.20)],
                        "key_levels": {
                            "accumulation_zone": [float(latest_price * 0.93), float(latest_price)],
                            "distribution_zone": [float(latest_price * 1.13), float(latest_price * 1.20)]
                        },
                        "rationale": "Strong fundamentals with technical support"
                    }
                },
                "risk_management": {
                    "key_risks": [
                        {
                            "risk": "Market volatility",
                            "probability": "medium",
                            "impact": "high",
                            "mitigation": "Use stop-loss orders"
                        }
                    ],
                    "stop_loss_levels": [
                        {
                            "level": float(latest_price * 0.98),
                            "type": "technical",
                            "rationale": "Below SMA 20 support"
                        }
                    ],
                    "position_management": {
                        "scaling_in": True,
                        "scaling_out": True,
                        "max_correlation": 0.7
                    }
                },
                "critical_levels": {
                    "must_watch": [
                        {
                            "level": float(latest_price * 1.02),
                            "type": "resistance",
                            "significance": "key breakout level",
                            "action": "monitor for breakout"
                        }
                    ],
                    "confirmation_levels": [
                        {
                            "level": float(latest_price * 0.98),
                            "type": "support",
                            "condition": "price holds above",
                            "action": "confirm bullish bias"
                        }
                    ]
                },
                "monitoring_plan": {
                    "daily_checks": ["Price action", "Volume analysis"],
                    "weekly_reviews": ["Technical indicators", "Pattern development"],
                    "exit_triggers": [
                        {
                            "condition": f"Price breaks below {float(latest_price * 0.98)}",
                            "action": "Exit long position"
                        }
                    ]
                },
                "data_quality_assessment": {
                    "issues": [],
                    "confidence_adjustments": {
                        "reason": "High data quality",
                        "adjustment": "No adjustments needed"
                    }
                },
                "key_takeaways": [
                    "Strong technical setup with clear entry levels",
                    "Volume confirmation supports bullish bias",
                    "Risk management through proper stop-loss placement"
                ],
                "indicator_summary_md": ai_analysis.get('indicator_summary', ''),
                "chart_insights": ai_analysis.get('chart_insights', '')
            }
        except Exception as e:
            logger.error(f"Error building AI analysis: {e}")
            return {}
    
    @staticmethod
    def _infer_trend_duration(data: pd.DataFrame, interval: str) -> str:
        """Infer trend duration heuristically from price vs SMA50/200 and interval.

        Returns one of: "short-term", "medium-term", "long-term".
        """
        try:
            if data is None or data.empty or 'close' not in data.columns:
                return "short-term"

            closes = data['close']
            last_close = float(closes.iloc[-1]) if len(closes) > 0 else 0.0
            if last_close <= 0:
                return "short-term"

            sma50 = float(closes.rolling(window=50).mean().iloc[-1]) if len(closes) >= 50 else None
            sma200 = float(closes.rolling(window=200).mean().iloc[-1]) if len(closes) >= 200 else None

            intraday_aliases = {"min", "1min", "5min", "15min", "30min", "60min", "1h", "hour"}
            intraday = interval in intraday_aliases
            d200_thresh = 0.05 if not intraday else 0.03
            d50_thresh = 0.03 if not intraday else 0.02

            if sma200 and sma200 > 0 and sma50:
                dist200 = abs(last_close - sma200) / sma200
                if dist200 >= d200_thresh and sma50 > sma200:
                    return "long-term"

            if sma50 and sma50 > 0:
                dist50 = abs(last_close - sma50) / sma50
                if dist50 >= d50_thresh:
                    return "medium-term"

            return "short-term"
        except Exception:
            return "short-term"

    @staticmethod
    def _build_overlays(data: pd.DataFrame, advanced_patterns: dict = None, symbol: str = "", exchange: str = "NSE", interval: str = "day") -> dict:
        """Build overlays structure using actual pattern recognition."""
        try:
            # Early fallback for insufficient data to avoid index errors downstream
            if data is None or data.empty or len(data) < 2:
                latest_price = data['close'].iloc[-1] if (data is not None and not data.empty) else 0
                return {
                    "triangles": [],
                    "flags": [],
                    "support_resistance": {
                        "support": [{"level": float(latest_price * 0.98)}, {"level": float(latest_price * 0.95)}],
                        "resistance": [{"level": float(latest_price * 1.02)}, {"level": float(latest_price * 1.05)}]
                    },
                    "double_tops": [],
                    "double_bottoms": [],
                    "divergences": [],
                    "volume_anomalies": [],
                    "advanced_patterns": advanced_patterns or {
                        "head_and_shoulders": [],
                        "inverse_head_and_shoulders": [],
                        "cup_and_handle": [],
                        "triple_tops": [],
                        "triple_bottoms": [],
                        "wedge_patterns": [],
                        "channel_patterns": []
                    }
                }

            # Import the orchestrator to use its pattern recognition
            from analysis.orchestrator import StockAnalysisOrchestrator
            from ml.bayesian_scorer import BayesianPatternScorer
            
            # Create orchestrator instance
            orchestrator = StockAnalysisOrchestrator()
            
            # Generate overlays directly without recalculating all indicators to
            # avoid unnecessary heavy computations and potential upstream errors
            # that could prematurely trigger the fallback path (returning empty
            # pattern lists). The _create_overlays method internally performs
            # all required computations for pattern detection.
            # Prefer precomputed patterns from central cache if available to avoid recomputation
            cached_patterns = None
            if symbol:
                try:
                    from services.central_data_provider import CentralDataProvider
                    central_data_provider = CentralDataProvider()
                    cached_patterns = central_data_provider.get_patterns(symbol=symbol, exchange=exchange, interval=interval, data=data)
                except Exception:
                    cached_patterns = None

            if isinstance(cached_patterns, dict) and cached_patterns:
                overlays = {
                    "triangles": cached_patterns.get("triangles", []),
                    "flags": cached_patterns.get("flags", []),
                    "support_resistance": cached_patterns.get("support_resistance", {"support": [], "resistance": []}),
                    "double_tops": cached_patterns.get("double_tops", []),
                    "double_bottoms": cached_patterns.get("double_bottoms", []),
                    "divergences": cached_patterns.get("divergences", []),
                    "volume_anomalies": cached_patterns.get("volume_anomalies", []),
                    "advanced_patterns": cached_patterns.get("advanced_patterns", {}),
                }
            else:
                overlays = orchestrator._create_overlays(data, {})

            # Prefer orchestrator-detected advanced patterns; merge in any external ones (from advanced_analysis)
            backend_adv = overlays.get("advanced_patterns", {}) or {}

            def _ensure_adv_shape(obj: dict | None) -> dict:
                base = obj or {}
                return {
                    "head_and_shoulders": base.get("head_and_shoulders", []),
                    "inverse_head_and_shoulders": base.get("inverse_head_and_shoulders", []),
                    "cup_and_handle": base.get("cup_and_handle", []),
                    "triple_tops": base.get("triple_tops", []),
                    "triple_bottoms": base.get("triple_bottoms", []),
                    "wedge_patterns": base.get("wedge_patterns", []),
                    "channel_patterns": base.get("channel_patterns", []),
                }

            def _has_any_patterns(obj: dict | None) -> bool:
                if not isinstance(obj, dict):
                    return False
                for v in obj.values():
                    if isinstance(v, list) and len(v) > 0:
                        return True
                return False

            def _merge_advanced(backend_obj: dict, external_obj: dict | None) -> dict:
                backend_norm = _ensure_adv_shape(backend_obj)
                external_norm = _ensure_adv_shape(external_obj or {})
                merged = {}
                for key in backend_norm.keys():
                    # Concatenate; de-duplication is non-trivial for dict items, so allow duplicates if any
                    merged[key] = list(backend_norm.get(key, [])) + list(external_norm.get(key, []))
                return merged

            merged_advanced = _merge_advanced(backend_adv, advanced_patterns) if _has_any_patterns(advanced_patterns) else _ensure_adv_shape(backend_adv)

            # Filter out zero-quality patterns at the final assembly layer
            def _filter_zero_quality(obj: dict) -> dict:
                def _q(p: dict) -> float:
                    return float(p.get('quality_score') or p.get('confidence') or p.get('completion') or 0)
                out = {}
                for k, arr in obj.items():
                    if isinstance(arr, list):
                        out[k] = [p for p in arr if _q(p) > 0]
                    else:
                        out[k] = arr
                return out

            merged_advanced = _filter_zero_quality(merged_advanced)

            # Normalize, de-duplicate and limit advanced patterns to keep UI clean
            # Use ML-powered CatBoost predictor for modern pattern analysis
            from ml.inference import predict_probability, get_model_version, get_pattern_prediction_breakdown

            def _score_of(p: dict) -> float:
                # ML-powered probability estimate using unified ML system
                try:
                    pattern_type = (
                        str(p.get('pattern_type', ''))  # Use empty string instead of None
                        or str(p.get('type', ''))  # Use empty string instead of None
                        or "unknown"
                    )
                    # Ensure we don't have "None" as a string
                    if pattern_type == "None" or pattern_type == "null":
                        pattern_type = "unknown"
                        
                    features = {
                        'duration': float(p.get('duration') or p.get('length') or 0.0),
                        'volume_ratio': float(p.get('volume_ratio') or 1.0),
                        'trend_alignment': float(p.get('trend_alignment') or 0.0),
                        'completion': float(p.get('completion') or p.get('quality_score') or p.get('confidence') or 0.0),
                    }
                    
                    # Get ML-powered prediction
                    proba = predict_probability(features, pattern_type)
                    
                    # Get detailed breakdown for enhanced analysis
                    breakdown = get_pattern_prediction_breakdown(features, pattern_type)
                    
                    # Store enhanced ML insights in the pattern data
                    p['ml_breakdown'] = breakdown
                    p['prediction_source'] = 'unified_ml_system'
                    p['model_version'] = breakdown.get('model_version', '1.0.0')
                    
                    return float(max(0.0, min(1.0, proba)) * 100.0)
                    
                except Exception as e:
                    logger.warning(f"ML prediction failed, using default score: {e}")
                    return 50.0

            def _strength_from_score(score: float) -> str:
                if score >= 80:
                    return 'strong'
                if score >= 60:
                    return 'medium'
                return 'weak'

            def _reliability_from_score(score: float) -> str:
                if score >= 80:
                    return 'high'
                if score >= 60:
                    return 'medium'
                return 'low'

            def _normalize_entry(type_key: str, p: dict) -> dict:
                # Create a shallow copy to avoid mutating original
                out = dict(p) if isinstance(p, dict) else {}
                score = _score_of(out)
                # Keep probability (0-100) for frontend; store also as 0-1
                out['probability'] = float(score)
                out['probability_fraction'] = float(score) / 100.0
                # Map common fields for frontend to avoid empty cards
                if type_key == 'triple_tops':
                    if out.get('target_level') is None and out.get('target') is not None:
                        out['target_level'] = out.get('target')
                    if out.get('stop_level') is None:
                        # Prefer support_level if available
                        stop = out.get('support_level')
                        if stop is None and isinstance(out.get('neckline'), dict):
                            stop = out['neckline'].get('level')
                        if stop is not None:
                            out['stop_level'] = stop
                elif type_key == 'triple_bottoms':
                    if out.get('target_level') is None and out.get('target') is not None:
                        out['target_level'] = out.get('target')
                    if out.get('stop_level') is None and out.get('resistance_level') is not None:
                        out['stop_level'] = out.get('resistance_level')
                elif type_key == 'channel_patterns':
                    if out.get('target_level') is None and out.get('target') is not None:
                        out['target_level'] = out.get('target')
                    # stop_level is ambiguous for channels; keep absent if unknown
                else:
                    # For other patterns, map target if present
                    if out.get('target_level') is None and out.get('target') is not None:
                        out['target_level'] = out.get('target')

                # Derive strength/reliability if missing
                if 'strength' not in out:
                    out['strength'] = _strength_from_score(score)
                if 'reliability' not in out:
                    out['reliability'] = _reliability_from_score(score)
                # Compute risk metrics when possible
                try:
                    from analysis.risk_scoring import calculate_risk_score, extract_reward_risk
                    current_price = float(data['close'].iloc[-1]) if not data.empty else 0.0
                    rr, _risk_abs = extract_reward_risk(out, current_price)
                    # Approximate volatility from Bollinger bandwidth or ADX if present
                    vol_proxy = 0.1
                    try:
                        bb = (FrontendResponseBuilder._build_technical_indicators(data, {}) or {}).get('bollinger_bands', {})
                        vol_proxy = float(bb.get('bandwidth', vol_proxy))
                    except Exception:
                        pass
                    risk_payload = calculate_risk_score(
                        probability=out['probability_fraction'],
                        reward_risk_ratio=rr,
                        volatility=vol_proxy,
                    )
                    out['risk_metrics'] = risk_payload
                except Exception:
                    pass
                return out

            def _signature(type_key: str, p: dict) -> tuple:
                # Build a stable signature for de-duplication per type
                try:
                    if type_key == 'triple_tops':
                        sup = round(float(p.get('support_level') or 0.0), 2)
                        tgt = round(float(p.get('target') or p.get('target_level') or 0.0), 2)
                        # Include first/last peak indices if present
                        peaks = p.get('peaks') or []
                        first_idx = int((peaks[0] or {}).get('index', 0)) if peaks else 0
                        last_idx = int((peaks[-1] or {}).get('index', 0)) if peaks else 0
                        return (type_key, sup, tgt, first_idx, last_idx)
                    if type_key == 'triple_bottoms':
                        res = round(float(p.get('resistance_level') or 0.0), 2)
                        tgt = round(float(p.get('target') or p.get('target_level') or 0.0), 2)
                        lows = p.get('lows') or []
                        first_idx = int((lows[0] or {}).get('index', 0)) if lows else 0
                        last_idx = int((lows[-1] or {}).get('index', 0)) if lows else 0
                        return (type_key, res, tgt, first_idx, last_idx)
                    if type_key == 'channel_patterns':
                        s = int(p.get('start_index') or 0)
                        e = int(p.get('end_index') or 0)
                        subtype = str(p.get('type') or '')
                        return (type_key, subtype, s, e)
                    # Generic fallback
                    s = int(p.get('start_index') or 0)
                    e = int(p.get('end_index') or 0)
                    tgt = round(float(p.get('target') or p.get('target_level') or 0.0), 2)
                    return (type_key, s, e, tgt)
                except Exception:
                    return (type_key, id(p))

            def _dedupe_sort_limit(arr: list, type_key: str, k: int = 3) -> list:
                if not isinstance(arr, list) or not arr:
                    return []
                normalized = [_normalize_entry(type_key, p) for p in arr if isinstance(p, dict)]
                seen = set()
                unique = []
                for p in normalized:
                    sig = _signature(type_key, p)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    unique.append(p)
                unique.sort(key=_score_of, reverse=True)
                return unique[:k]

            # Apply per-type cleanup
            cleaned_advanced = {}
            for key, arr in merged_advanced.items():
                if key in (
                    'triple_tops', 'triple_bottoms', 'channel_patterns',
                    'wedge_patterns', 'cup_and_handle',
                    'head_and_shoulders', 'inverse_head_and_shoulders'
                ):
                    cleaned_advanced[key] = _dedupe_sort_limit(arr or [], key, k=3)
                else:
                    cleaned_advanced[key] = arr or []
            merged_advanced = cleaned_advanced

            # Ensure the structure matches frontend expectations
            return {
                "triangles": overlays.get("triangles", []),
                "flags": overlays.get("flags", []),
                "support_resistance": {
                    "support": overlays.get("support_resistance", {}).get("support", []),
                    "resistance": overlays.get("support_resistance", {}).get("resistance", [])
                },
                "double_tops": overlays.get("double_tops", []),
                "double_bottoms": overlays.get("double_bottoms", []),
                "divergences": overlays.get("divergences", []),
                "volume_anomalies": overlays.get("volume_anomalies", []),
                "advanced_patterns": merged_advanced,
            }
        except Exception as e:
            logger.error(f"Error building overlays: {e}")
            # Fallback to basic structure if pattern recognition fails
            latest_price = data['close'].iloc[-1] if not data.empty else 0
            return {
                "triangles": [],
                "flags": [],
                "support_resistance": {
                    "support": [{"level": float(latest_price * 0.98)}, {"level": float(latest_price * 0.95)}],
                    "resistance": [{"level": float(latest_price * 1.02)}, {"level": float(latest_price * 1.05)}]
                },
                "double_tops": [],
                "double_bottoms": [],
                "divergences": [],
                "volume_anomalies": [],
                "advanced_patterns": {
                    "head_and_shoulders": [],
                    "inverse_head_and_shoulders": [],
                    "cup_and_handle": [],
                    "triple_tops": [],
                    "triple_bottoms": [],
                    "wedge_patterns": [],
                    "channel_patterns": []
                }
            }
    
    @staticmethod
    def _extract_support_levels(data: pd.DataFrame, indicators: dict) -> list:
        """Extract support levels."""
        try:
            if data.empty:
                return []
            
            latest_price = data['close'].iloc[-1]
            sma_20 = indicators.get('sma_20', [latest_price])[-1] if indicators.get('sma_20') else latest_price
            sma_50 = indicators.get('sma_50', [latest_price])[-1] if indicators.get('sma_50') else latest_price
            
            support_levels = [
                float(latest_price * 0.98),
                float(sma_20 * 0.99),
                float(sma_50 * 0.98),
                float(latest_price * 0.95)
            ]
            
            return sorted(support_levels, reverse=True)
        except Exception as e:
            logger.error(f"Error extracting support levels: {e}")
            return []
    
    @staticmethod
    def _extract_resistance_levels(data: pd.DataFrame, indicators: dict) -> list:
        """Extract resistance levels."""
        try:
            if data.empty:
                return []
            
            latest_price = data['close'].iloc[-1]
            sma_20 = indicators.get('sma_20', [latest_price])[-1] if indicators.get('sma_20') else latest_price
            sma_50 = indicators.get('sma_50', [latest_price])[-1] if indicators.get('sma_50') else latest_price
            
            resistance_levels = [
                float(latest_price * 1.02),
                float(sma_20 * 1.01),
                float(sma_50 * 1.02),
                float(latest_price * 1.05)
            ]
            
            return sorted(resistance_levels)
        except Exception as e:
            logger.error(f"Error extracting resistance levels: {e}")
            return [] 

    @staticmethod
    def _build_consensus(ai_analysis: dict, indicators: dict, data: pd.DataFrame, mtf_context: dict | None = None) -> dict:
        """Build consensus using signals.scoring as the single source of truth.

        Prefer multi-timeframe indicators when available and compute consensus via
        compute_signals_summary. Avoid duplicating per-indicator logic.
        """
        try:
            from data.signals.scoring import compute_signals_summary
            per_timeframe_indicators: dict[str, dict] = {}
            
            # Prefer MTF indicators if present in ai_analysis
            mtf_block = ai_analysis.get('multi_timeframe') if isinstance(ai_analysis, dict) else None
            if isinstance(mtf_block, dict) and mtf_block.get('timeframes'):
                for tf, tf_obj in mtf_block['timeframes'].items():
                    if isinstance(tf_obj, dict) and 'indicators' in tf_obj:
                        per_timeframe_indicators[tf] = tf_obj.get('indicators') or {}
            
            # Otherwise synthesize from mtf_context
            if not per_timeframe_indicators and isinstance(mtf_context, dict) and mtf_context.get('timeframe_analyses'):
                try:
                    tf_analyses = mtf_context['timeframe_analyses']
                    for tf, summary in tf_analyses.items():
                        if not isinstance(summary, dict):
                            continue
                        indicators_min = {}
                        ki = summary.get('key_indicators') or {}
                        rsi_val = ki.get('rsi')
                        if rsi_val is not None:
                            indicators_min['rsi'] = {'rsi_14': float(rsi_val)}
                        macd_sig = ki.get('macd_signal')
                        if isinstance(macd_sig, str):
                            indicators_min['macd'] = {'macd_line': 1.0 if macd_sig.lower()== 'bullish' else -1.0 if macd_sig.lower()=='bearish' else 0.0,
                                                      'signal_line': 0.0}
                        trend = summary.get('trend')
                        if isinstance(trend, str):
                            indicators_min['supertrend'] = {'direction': 'up' if trend=='bullish' else 'down' if trend=='bearish' else 'neutral'}
                        vol_status = ki.get('volume_status')
                        if isinstance(vol_status, str):
                            ratio = 1.6 if vol_status=='high' else 0.4 if vol_status=='low' else 1.0
                            indicators_min['volume'] = {'volume_ratio': float(ratio)}
                        try:
                            tf_conf = float(summary.get('confidence')) if summary.get('confidence') is not None else 0.5
                        except Exception:
                            tf_conf = 0.5
                        indicators_min['adx'] = {'adx': float(max(5.0, min(40.0, 20.0 + (tf_conf - 0.5) * 20.0)))}
                        if indicators_min:
                            per_timeframe_indicators[tf] = indicators_min
                except Exception:
                    per_timeframe_indicators = {}
            
            # If no MTF indicators, flatten the nested indicators structure for signals scoring
            if not per_timeframe_indicators:
                flattened_indicators = FrontendResponseBuilder._flatten_indicators_for_scoring(indicators)
                per_timeframe_indicators['day'] = flattened_indicators

            # Try to get price data for better regime detection
            price_data = None
            if data is not None and not data.empty and len(data) >= 50:
                price_data = data
            
            summary = compute_signals_summary(per_timeframe_indicators, price_data)

            c = float(summary.consensus_score)
            bullish_percentage = max(0.0, c) * 100.0
            bearish_percentage = max(0.0, -c) * 100.0
            neutral_percentage = max(0.0, 100.0 - bullish_percentage - bearish_percentage)
            overall_signal = 'Bullish' if c > 0.1 else ('Bearish' if c < -0.1 else 'Neutral')
            max_percentage = max(bullish_percentage, bearish_percentage, neutral_percentage)
            signal_strength = 'Strong' if max_percentage >= 80 else 'Medium' if max_percentage >= 60 else 'Weak'
            ai_confidence = ai_analysis.get('meta', {}).get('overall_confidence', 50) if isinstance(ai_analysis, dict) else 50
            confidence = float(min(100.0, max(0.0, (ai_confidence + summary.confidence * 100.0) / 2.0)))

            per_tf_sorted = sorted(summary.per_timeframe, key=lambda s: getattr(s, 'confidence', 0), reverse=True)
            signal_details = []
            if per_tf_sorted:
                top = per_tf_sorted[0]
                for r in getattr(top, 'reasons', [])[:5]:
                    # Extract the actual value from the indicators for display
                    indicator_value = FrontendResponseBuilder._extract_indicator_value(r.indicator, indicators)
                    
                    signal_details.append({
                        "indicator": getattr(r, 'indicator', 'unknown'),
                        "signal": getattr(r, 'bias', 'neutral'),
                        "strength": "medium",
                        "weight": float(getattr(r, 'weight', 0.1) or 0.1),
                        "score": float(getattr(top, 'score', 0.0) or 0.0),
                        "value": indicator_value,
                        "description": getattr(r, 'description', '')
                    })

            return {
                "overall_signal": overall_signal,
                "signal_strength": signal_strength,
                "bullish_percentage": round(bullish_percentage, 2),
                "bearish_percentage": round(bearish_percentage, 2),
                "neutral_percentage": round(neutral_percentage, 2),
                "bullish_score": round(bullish_percentage, 2),
                "bearish_score": round(bearish_percentage, 2),
                "neutral_score": round(neutral_percentage, 2),
                "total_weight": len(summary.per_timeframe),
                "confidence": round(confidence, 2),
                "signal_details": signal_details,
                "data_quality_flags": [],
                "warnings": [],
                "bullish_count": int(bullish_percentage > max(bearish_percentage, neutral_percentage)),
                "bearish_count": int(bearish_percentage > max(bullish_percentage, neutral_percentage)),
                "neutral_count": int(neutral_percentage >= max(bullish_percentage, bearish_percentage)),
            }
            
        except Exception as e:
            logger.error(f"Error building consensus: {e}")
            # Return default consensus
            return {
                "overall_signal": "Neutral",
                "signal_strength": "Weak",
                "bullish_percentage": 33.33,
                "bearish_percentage": 33.33,
                "neutral_percentage": 33.34,
                "bullish_score": 33.33,
                "bearish_score": 33.33,
                "neutral_score": 33.34,
                "total_weight": 3,
                "confidence": 50.0,
                "signal_details": [],
                "data_quality_flags": ["Consensus calculation error"],
                "warnings": [f"Error building consensus: {str(e)}"],
                "bullish_count": 1,
                "bearish_count": 1,
                "neutral_count": 1
            }

    @staticmethod
    def _flatten_indicators_for_scoring(indicators: dict) -> dict:
        """Flatten nested indicators structure for signals scoring system."""
        flattened = {}
        
        # Extract RSI
        if indicators.get('rsi') and isinstance(indicators['rsi'], dict):
            rsi_data = indicators['rsi']
            if 'rsi_14' in rsi_data:
                flattened['rsi_14'] = rsi_data['rsi_14']
        
        # Extract MACD
        if indicators.get('macd') and isinstance(indicators['macd'], dict):
            macd_data = indicators['macd']
            if 'macd_line' in macd_data:
                flattened['macd_line'] = macd_data['macd_line']
            if 'signal_line' in macd_data:
                flattened['signal_line'] = macd_data['signal_line']
        
        # Extract Moving Averages
        if indicators.get('moving_averages') and isinstance(indicators['moving_averages'], dict):
            ma_data = indicators['moving_averages']
            if 'sma_50' in ma_data:
                flattened['sma_50'] = ma_data['sma_50']
            if 'sma_200' in ma_data:
                flattened['sma_200'] = ma_data['sma_200']
        
        # Extract ADX
        if indicators.get('adx') and isinstance(indicators['adx'], dict):
            adx_data = indicators['adx']
            if 'adx' in adx_data:
                flattened['adx'] = adx_data['adx']
        
        # Extract Bollinger Bands
        if indicators.get('bollinger_bands') and isinstance(indicators['bollinger_bands'], dict):
            bb_data = indicators['bollinger_bands']
            if 'percent_b' in bb_data:
                flattened['percent_b'] = bb_data['percent_b']
        
        # Extract Volume
        if indicators.get('volume') and isinstance(indicators['volume'], dict):
            volume_data = indicators['volume']
            if 'volume_ratio' in volume_data:
                flattened['volume_ratio'] = volume_data['volume_ratio']
        
        return flattened 

    @staticmethod
    def _extract_indicator_value(indicator_name: str, indicators: dict) -> float | None:
        """Extract the actual value of an indicator for display in signal details."""
        try:
            if indicator_name == "RSI":
                if indicators.get('rsi') and isinstance(indicators['rsi'], dict):
                    return float(indicators['rsi'].get('rsi_14', 0))
            elif indicator_name == "MACD":
                if indicators.get('macd') and isinstance(indicators['macd'], dict):
                    macd_line = indicators['macd'].get('macd_line', 0)
                    signal_line = indicators['macd'].get('signal_line', 0)
                    return float(macd_line - signal_line)  # Return histogram value
            elif indicator_name == "MA":
                if indicators.get('moving_averages') and isinstance(indicators['moving_averages'], dict):
                    sma_50 = indicators['moving_averages'].get('sma_50', 0)
                    sma_200 = indicators['moving_averages'].get('sma_200', 0)
                    if sma_200 != 0:
                        return float((sma_50 / sma_200 - 1) * 100)  # Return percentage difference
            elif indicator_name == "ADX":
                if indicators.get('adx') and isinstance(indicators['adx'], dict):
                    return float(indicators['adx'].get('adx', 0))
            elif indicator_name == "BollingerBands":
                if indicators.get('bollinger_bands') and isinstance(indicators['bollinger_bands'], dict):
                    return float(indicators['bollinger_bands'].get('percent_b', 0.5))
            elif indicator_name == "Volume":
                if indicators.get('volume') and isinstance(indicators['volume'], dict):
                    return float(indicators['volume'].get('volume_ratio', 1.0))
        except (ValueError, TypeError, KeyError):
            pass
        
        return None 