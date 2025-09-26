"""
Reversal Patterns Analysis Module

This module provides comprehensive analysis of trend reversal patterns including:
- Divergence analysis (RSI, MACD, other oscillators)
- Double top and double bottom patterns
- Head and shoulders patterns
- Other reversal indicators and signals

The module consists of:
- ReversalPatternsProcessor: Core pattern detection and analysis logic
- ReversalPatternsCharts: Specialized chart generation for reversal patterns
"""

from .processor import ReversalPatternsProcessor, ReversalPattern
from .charts import ReversalPatternsCharts

__all__ = [
    'ReversalPatternsProcessor',
    'ReversalPattern', 
    'ReversalPatternsCharts'
]