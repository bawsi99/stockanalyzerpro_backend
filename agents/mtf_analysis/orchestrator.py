#!/usr/bin/env python3
"""
MTF Orchestrator

Coordinates Multi-Timeframe Analysis Agents (intraday, swing, position)
and integrates with the CoreMTFProcessor for data and validation.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd

from agents.mtf_analysis.core.processor import CoreMTFProcessor
from agents.mtf_analysis.core.data_models import TimeframeAnalysisResult
from agents.mtf_analysis.intraday import IntradayMTFProcessor

try:
    from agents.mtf_analysis.swing import SwingMTFProcessor
except Exception:  # noqa: E722
    SwingMTFProcessor = None  # Will be available after creation

try:
    from agents.mtf_analysis.position import PositionMTFProcessor
except Exception:  # noqa: E722
    PositionMTFProcessor = None  # Will be available after creation

logger = logging.getLogger(__name__)


class MTFOrchestrator:
    """
    Orchestrates multi-timeframe data fetching, agent processing, and cross-timeframe validation.
    """

    def __init__(self, include_agents: Optional[List[str]] = None) -> None:
        self.include_agents = include_agents or ["intraday", "swing", "position"]
        self.core_processor = CoreMTFProcessor()
        self.default_timeframes: List[str] = [
            "1min",
            "5min",
            "15min",
            "30min",
            "1hour",
            "1day",
        ]

    async def _fetch_mtf_data(self, symbol: str, exchange: str) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV DataFrames for configured timeframes using the core processor's client."""
        # Ensure authentication
        if not await self.core_processor.authenticate():
            raise RuntimeError("Failed to authenticate with Zerodha API")

        async def fetch_tf(tf: str):
            try:
                df = await self.core_processor.fetch_timeframe_data(symbol, exchange, tf)
                return tf, df
            except Exception as e:  # noqa: E722
                logger.warning(f"Fetch failed for {tf}: {e}")
                return tf, None

        results = await asyncio.gather(*[fetch_tf(tf) for tf in self.default_timeframes])
        return {tf: df for tf, df in results if df is not None}

    async def _analyze_all_timeframes(self, symbol: str, exchange: str) -> Dict[str, TimeframeAnalysisResult]:
        """Use the core processor to compute full analyses per timeframe (indicators/signals/etc.)."""
        result = await self.core_processor.analyze_comprehensive_mtf(symbol, exchange)
        if result.success:
            return result.timeframe_analyses
        else:
            logger.error(f"Core processor analysis failed: {result.error_message}")
            return {}

    async def run(self, symbol: str, exchange: str = "NSE", include: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute orchestrated multi-timeframe analysis.

        Args:
            symbol: Trading symbol
            exchange: Exchange (default: NSE)
            include: Optional list of agents to include ["intraday", "swing", "position"]
        Returns:
            Dictionary with agent outputs and cross-timeframe overview
        """
        selected_agents = include or self.include_agents
        start_ts = datetime.now()

        # Fetch raw OHLCV for processors and compute analyzer overview in parallel
        fetch_task = asyncio.create_task(self._fetch_mtf_data(symbol, exchange))
        overview_task = asyncio.create_task(self._analyze_all_timeframes(symbol, exchange))

        mtf_data, analyzer_overview = await asyncio.gather(fetch_task, overview_task)

        # Prepare processors
        processors = []
        if "intraday" in selected_agents:
            processors.append(("intraday", IntradayMTFProcessor()))
        if "swing" in selected_agents and SwingMTFProcessor is not None:
            processors.append(("swing", SwingMTFProcessor()))
        if "position" in selected_agents and PositionMTFProcessor is not None:
            processors.append(("position", PositionMTFProcessor()))

        # Run processors concurrently
        async def run_proc(name: str, proc):
            try:
                return name, await proc.analyze_async(mtf_data, indicators={}, context=f"symbol={symbol} exchange={exchange}")
            except Exception as e:  # noqa: E722
                logger.error(f"Agent {name} failed: {e}")
                return name, {
                    "agent_name": name,
                    "error": str(e),
                    "confidence_score": 0.0,
                    "analysis_timestamp": datetime.now().isoformat(),
                }

        agent_results_pairs = await asyncio.gather(*[run_proc(n, p) for n, p in processors])
        agent_results = {name: res for name, res in agent_results_pairs}

        # Cross-timeframe validation using core processor results
        validation = self.core_processor.validate_cross_timeframe(analyzer_overview)

        # Summarize core processor overview for quick reference
        overview_summary: Dict[str, Any] = {
            tf: {
                "trend": ta.get('trend', 'neutral'),
                "confidence": ta.get('confidence', 0.0),
                "data_points": ta.get('data_points', 0),
            }
            for tf, ta in analyzer_overview.items()
            if isinstance(ta, dict)
        }

        total_time = (datetime.now() - start_ts).total_seconds()

        return {
            "symbol": symbol,
            "exchange": exchange,
            "analysis_timestamp": datetime.now().isoformat(),
            "processing_time": total_time,
            "agents": agent_results,
            "mtf_overview": overview_summary,
            "cross_timeframe_validation": {
                "consensus_trend": validation.consensus_trend,
                "signal_strength": validation.signal_strength,
                "confidence_score": validation.confidence_score,
                "supporting_timeframes": validation.supporting_timeframes,
                "conflicting_timeframes": validation.conflicting_timeframes,
                "neutral_timeframes": validation.neutral_timeframes,
                "divergence_detected": validation.divergence_detected,
                "divergence_type": validation.divergence_type,
                "key_conflicts": validation.key_conflicts,
            },
        }
