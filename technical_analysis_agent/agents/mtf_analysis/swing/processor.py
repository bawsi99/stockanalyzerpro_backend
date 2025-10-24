#!/usr/bin/env python3
"""
Swing Multi-Timeframe Analysis Agent

Targets 15min/30min/1hour timeframes to identify swing opportunities
with focus on trend continuation, pullbacks, and breakouts.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SwingMTFProcessor:
    """Processor for swing multi-timeframe analysis"""

    def __init__(self) -> None:
        self.agent_name = "swing_mtf"
        self.timeframes = ["15min", "30min", "1hour"]
        self.primary_timeframe = "30min"

    async def analyze_async(self, mtf_data: Dict[str, pd.DataFrame], indicators: Dict, context: str = "") -> Dict[str, Any]:
        try:
            logger.info("[SWING_MTF] Starting swing multi-timeframe analysis...")
            start_time = datetime.now()

            swing_data = {tf: df for tf, df in mtf_data.items() if tf in self.timeframes}
            if not swing_data:
                return {
                    "agent_name": self.agent_name,
                    "error": "No swing timeframe data available",
                    "confidence_score": 0.0,
                    "analysis_timestamp": datetime.now().isoformat(),
                }

            # Analyze each timeframe
            tf_analysis: Dict[str, Dict[str, Any]] = {}
            for tf in self.timeframes:
                if tf in swing_data:
                    tf_analysis[tf] = await self._analyze_timeframe(swing_data[tf], tf)

            validation = self._validate_swing_signals(tf_analysis)
            setups = self._identify_swing_setups(tf_analysis)
            confidence = self._calculate_confidence(tf_analysis, validation)

            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"[SWING_MTF] Analysis completed in {processing_time:.2f}s")

            return {
                "agent_name": self.agent_name,
                "analysis_timestamp": datetime.now().isoformat(),
                "processing_time": processing_time,
                "context": context,
                "timeframe_analysis": tf_analysis,
                "cross_timeframe_validation": validation,
                "swing_setups": setups,
                "dominant_timeframe": self._get_dominant_timeframe(tf_analysis),
                "confidence_score": confidence,
            }
        except Exception as e:  # noqa: E722
            logger.error(f"[SWING_MTF] Analysis failed: {e}")
            return {
                "agent_name": self.agent_name,
                "error": str(e),
                "confidence_score": 0.0,
                "analysis_timestamp": datetime.now().isoformat(),
            }

    async def _analyze_timeframe(self, data: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        if len(data) < 50:
            return {
                "timeframe": timeframe,
                "status": "insufficient_data",
                "trend": "unknown",
                "strength": "unknown",
                "signals": [],
            }

        close = data["close"]
        sma_20 = close.rolling(20, min_periods=15).mean()
        sma_50 = close.rolling(50, min_periods=30).mean()
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal

        current_price = float(close.iloc[-1])
        trend = "bullish" if sma_20.iloc[-1] > sma_50.iloc[-1] else "bearish"
        momentum = float(close.pct_change().rolling(20, min_periods=10).mean().iloc[-1])
        vol_ma = data["volume"].rolling(20, min_periods=15).mean()
        volume_ratio = float(data["volume"].iloc[-1] / vol_ma.iloc[-1]) if vol_ma.iloc[-1] > 0 else 1.0

        # Key levels
        highs = data["high"].rolling(50, min_periods=30).max()
        lows = data["low"].rolling(50, min_periods=30).min()

        signals = self._generate_timeframe_signals(
            timeframe=timeframe,
            current_price=current_price,
            sma_20=float(sma_20.iloc[-1]),
            sma_50=float(sma_50.iloc[-1]) if not np.isnan(sma_50.iloc[-1]) else None,
            macd=float(macd.iloc[-1]),
            macd_hist=float(macd_hist.iloc[-1]),
            volume_ratio=volume_ratio,
            resistance=float(highs.iloc[-1]),
            support=float(lows.iloc[-1]),
        )

        return {
            "timeframe": timeframe,
            "status": "analyzed",
            "trend": trend,
            "strength": "strong" if abs(momentum) > 0.003 else "medium" if abs(momentum) > 0.0015 else "weak",
            "momentum": momentum,
            "current_price": current_price,
            "sma_20": float(sma_20.iloc[-1]),
            "sma_50": float(sma_50.iloc[-1]) if not np.isnan(sma_50.iloc[-1]) else None,
            "macd": float(macd.iloc[-1]),
            "macd_hist": float(macd_hist.iloc[-1]),
            "volume_ratio": volume_ratio,
            "resistance_level": float(highs.iloc[-1]),
            "support_level": float(lows.iloc[-1]),
            "signals": signals,
            "confidence": min(0.95, len(data) / 200.0),
        }

    def _generate_timeframe_signals(
        self,
        timeframe: str,
        current_price: float,
        sma_20: float,
        sma_50: Optional[float],
        macd: float,
        macd_hist: float,
        volume_ratio: float,
        resistance: float,
        support: float,
    ) -> List[Dict[str, Any]]:
        signals: List[Dict[str, Any]] = []

        # Trend continuation
        if sma_50 is not None and sma_20 > sma_50:
            signals.append({
                "type": "trend_continuation",
                "direction": "bullish",
                "strength": 0.6,
                "description": f"SMA20>SMA50 on {timeframe}",
            })
        elif sma_50 is not None and sma_20 < sma_50:
            signals.append({
                "type": "trend_continuation",
                "direction": "bearish",
                "strength": 0.6,
                "description": f"SMA20<SMA50 on {timeframe}",
            })

        # Pullback to SMA20
        if abs(current_price - sma_20) / current_price < 0.005:
            signals.append({
                "type": "pullback",
                "direction": "bullish" if current_price >= sma_20 else "bearish",
                "strength": 0.5,
                "description": f"Price near SMA20 on {timeframe}",
            })

        # MACD momentum
        if macd_hist > 0:
            signals.append({
                "type": "momentum",
                "direction": "bullish",
                "strength": min(0.8, 0.5 + min(0.3, abs(macd_hist))),
                "description": f"MACD histogram positive on {timeframe}",
            })
        else:
            signals.append({
                "type": "momentum",
                "direction": "bearish",
                "strength": min(0.8, 0.5 + min(0.3, abs(macd_hist))),
                "description": f"MACD histogram negative on {timeframe}",
            })

        # Breakout proximity
        if current_price >= resistance:
            signals.append({
                "type": "breakout",
                "direction": "bullish",
                "strength": 0.7,
                "description": f"Price at/above resistance on {timeframe}",
            })
        elif current_price <= support:
            signals.append({
                "type": "breakdown",
                "direction": "bearish",
                "strength": 0.7,
                "description": f"Price at/below support on {timeframe}",
            })

        # Volume confirmation
        if volume_ratio > 1.5:
            signals.append({
                "type": "volume_confirmation",
                "direction": "bullish" if current_price >= sma_20 else "bearish",
                "strength": 0.6,
                "description": f"Elevated volume on {timeframe}",
            })

        return signals

    def _validate_swing_signals(self, tf_analysis: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        analyzed = [v for v in tf_analysis.values() if v.get("status") == "analyzed"]
        if not analyzed:
            return {
                "trend_alignment": "unknown",
                "signal_strength": 0.0,
                "consensus_trend": "neutral",
                "supporting_timeframes": [],
                "conflicting_timeframes": [],
            }

        bullish = sum(1 for v in analyzed if v.get("trend") == "bullish")
        bearish = sum(1 for v in analyzed if v.get("trend") == "bearish")
        total = len(analyzed)

        if bullish >= total * 0.6:
            consensus = "bullish"
            align = "aligned"
        elif bearish >= total * 0.6:
            consensus = "bearish"
            align = "aligned"
        else:
            consensus = "neutral"
            align = "conflicting"

        supporting = []
        conflicting = []
        for tf, v in tf_analysis.items():
            if v.get("status") != "analyzed":
                continue
            if v.get("trend") == consensus:
                supporting.append(tf)
            elif consensus != "neutral" and v.get("trend") != "neutral":
                conflicting.append(tf)

        strength = max(bullish, bearish) / total if total > 0 else 0.0

        return {
            "trend_alignment": align,
            "signal_strength": strength,
            "consensus_trend": consensus,
            "supporting_timeframes": supporting,
            "conflicting_timeframes": conflicting,
            "total_analyzed_timeframes": total,
        }

    def _identify_swing_setups(self, tf_analysis: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        setups: Dict[str, List[Dict[str, Any]]] = {
            "pullback_continuation": [],
            "breakout": [],
            "range_reversal": [],
        }

        for tf, v in tf_analysis.items():
            if v.get("status") != "analyzed":
                continue
            price = v.get("current_price")
            sma_20 = v.get("sma_20")
            res = v.get("resistance_level")
            sup = v.get("support_level")

            # Pullback continuation setup
            if abs(price - sma_20) / price < 0.005 and v.get("trend") in ("bullish", "bearish"):
                setups["pullback_continuation"].append({
                    "timeframe": tf,
                    "direction": v.get("trend"),
                    "confidence": 0.6,
                })

            # Breakout setup
            if price >= res:
                setups["breakout"].append({
                    "timeframe": tf,
                    "direction": "bullish",
                    "confidence": 0.7,
                })
            elif price <= sup:
                setups["breakout"].append({
                    "timeframe": tf,
                    "direction": "bearish",
                    "confidence": 0.7,
                })

            # Range reversal heuristic
            if (res - sup) / price < 0.02 and abs(price - (res + sup) / 2) / price < 0.005:
                setups["range_reversal"].append({
                    "timeframe": tf,
                    "direction": "bullish" if v.get("trend") == "bearish" else "bearish",
                    "confidence": 0.5,
                })

        return setups

    def _calculate_confidence(self, tf_analysis: Dict[str, Dict[str, Any]], validation: Dict[str, Any]) -> float:
        analyzed = sum(1 for v in tf_analysis.values() if v.get("status") == "analyzed")
        total = len(self.timeframes)
        data_conf = analyzed / total
        sig_strength = float(validation.get("signal_strength", 0.0))
        return (data_conf * 0.4) + (sig_strength * 0.6)

    def _get_dominant_timeframe(self, tf_analysis: Dict[str, Dict[str, Any]]) -> str:
        best_tf = self.primary_timeframe
        best_score = 0.0
        for tf, v in tf_analysis.items():
            if v.get("status") != "analyzed":
                continue
            score = v.get("confidence", 0.0) * len(v.get("signals", []))
            if score > best_score:
                best_score = score
                best_tf = tf
        return best_tf
