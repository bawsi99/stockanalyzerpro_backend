"""
Indicators Analysis Agents Package

This package contains specialized agents for analyzing different categories of technical indicators.
It provides comprehensive technical analysis through distributed indicator-specific agents.

Agents:
- Trend: Moving averages, trend direction, and strength analysis
- Momentum: RSI, MACD, Stochastic oscillator analysis
- Volatility: Bollinger Bands, ATR, volatility measures
- Support/Resistance: Key levels and breakout analysis
"""

# Import all indicator agents
from .trend import TrendIndicatorsProcessor
from .momentum import MomentumIndicatorsProcessor
from .indicators_agents import IndicatorAgentsOrchestrator, indicators_orchestrator

__all__ = [
    'TrendIndicatorsProcessor',
    'MomentumIndicatorsProcessor',
    'IndicatorAgentsOrchestrator',
    'indicators_orchestrator'
]

__version__ = "1.0.0"
__author__ = "Indicators Analysis System"
__description__ = "Comprehensive technical indicators analysis through specialized agents"