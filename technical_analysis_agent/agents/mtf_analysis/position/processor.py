#!/usr/bin/env python3
"""
Position (Higher Timeframe) Multi-Timeframe Analysis Agent

Targets 1hour and 1day timeframes for position/longer-horizon alignment,
emphasizing trend confirmation and risk context.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PositionMTFProcessor:
    """Processor for higher-timeframe (position) analysis"""

    def __init__(self) -> None:
        self.agent_name = "position_mtf"
        self.timeframes = ["1hour", "1day"]
        self.primary_timeframe = "1day"

    async def analyze_async(self, mtf_data: Dict[str, pd.DataFrame], indicators: Dict, context: str = "") -> Dict[str, Any]:
        try:
            logger.info("[POSITION_MTF] Starting position timeframe analysis...")
            start_time = datetime.now()

            pos_data = {tf: df for tf, df in mtf_data.items() if tf in self.timeframes}
            if not pos_data:
                return {
                    "agent_name": self.agent_name,
                    "error": "No position timeframe data available",
                    "confidence_score": 0.0,
                    "analysis_timestamp": datetime.now().isoformat(),
                }

            tf_analysis: Dict[str, Dict[str, Any]] = {}
            for tf in self.timeframes:
                if tf in pos_data:
                    tf_analysis[tf] = await self._analyze_timeframe(pos_data[tf], tf)

            validation = self._validate_position(tf_analysis)
            confidence = self._calculate_confidence(tf_analysis, validation)

            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"[POSITION_MTF] Analysis completed in {processing_time:.2f}s")

            return {
                "agent_name": self.agent_name,
                "analysis_timestamp": datetime.now().isoformat(),
                "processing_time": processing_time,
                "context": context,
                "timeframe_analysis": tf_analysis,
                "cross_timeframe_validation": validation,
                "dominant_timeframe": self._get_dominant_timeframe(tf_analysis),
                "confidence_score": confidence,
                "position_context": self._position_context(tf_analysis),
            }
        except Exception as e:  # noqa: E722
            logger.error(f"[POSITION_MTF] Analysis failed: {e}")
            return {
                "agent_name": self.agent_name,
                "error": str(e),
                "confidence_score": 0.0,
                "analysis_timestamp": datetime.now().isoformat(),
            }

    async def _analyze_timeframe(self, data: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        if len(data) < 60:
            return {
                "timeframe": timeframe,
                "status": "insufficient_data",
                "trend": "unknown",
                "strength": "unknown",
                "signals": [],
            }

        close = data["close"]
        sma_50 = close.rolling(50, min_periods=30).mean()
        sma_200 = close.rolling(200, min_periods=60).mean()
        current_price = float(close.iloc[-1])

        # Trend based on SMA50/SMA200 (use what we have)
        if not np.isnan(sma_200.iloc[-1]):
            trend = "bullish" if sma_50.iloc[-1] > sma_200.iloc[-1] else "bearish"
        elif not np.isnan(sma_50.iloc[-1]):
            trend = "bullish" if current_price > sma_50.iloc[-1] else "bearish"
        else:
            trend = "unknown"

        # Volatility proxy
        volatility = float(close.pct_change().std() * np.sqrt(252)) if len(close) > 1 else 0.0

        # Risk levels
        highs = data["high"].rolling(100, min_periods=50).max()
        lows = data["low"].rolling(100, min_periods=50).min()

        signals: List[Dict[str, Any]] = []
        if trend == "bullish" and current_price >= highs.iloc[-1]:
            signals.append({
                "type": "trend_breakout",
                "direction": "bullish",
                "strength": 0.7,
                "description": f"Price breaking 100-period high on {timeframe}",
            })
        if trend == "bearish" and current_price <= lows.iloc[-1]:
            signals.append({
                "type": "trend_breakdown",
                "direction": "bearish",
                "strength": 0.7,
                "description": f"Price breaking 100-period low on {timeframe}",
            })

        return {
            "timeframe": timeframe,
            "status": "analyzed",
            "trend": trend,
            "strength": "strong" if abs(volatility) > 0.4 else "medium" if abs(volatility) > 0.25 else "weak",
            "current_price": current_price,
            "sma_50": float(sma_50.iloc[-1]) if not np.isnan(sma_50.iloc[-1]) else None,
            "sma_200": float(sma_200.iloc[-1]) if not np.isnan(sma_200.iloc[-1]) else None,
            "volatility": volatility,
            "resistance_level": float(highs.iloc[-1]),
            "support_level": float(lows.iloc[-1]),
            "signals": signals,
            "confidence": min(0.98, len(data) / 250.0),
        }

    def _validate_position(self, tf_analysis: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
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

        if bullish >= total * 0.5:
            consensus = "bullish"
            align = "aligned"
        elif bearish >= total * 0.5:
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

    def _position_context(self, tf_analysis: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {}
        day = tf_analysis.get("1day", {})
        hr = tf_analysis.get("1hour", {})
        ctx["bias"] = day.get("trend", "unknown")
        ctx["daily_volatility"] = day.get("volatility")
        ctx["alignment"] = "aligned" if day.get("trend") == hr.get("trend") else "mixed"
        return ctx
