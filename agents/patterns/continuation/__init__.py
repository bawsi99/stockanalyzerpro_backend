"""
Continuation Patterns Analysis Module

This module provides comprehensive analysis of trend continuation patterns including:
- Triangle patterns (ascending, descending, symmetrical)
- Flag and pennant patterns
- Channel patterns (ascending, descending, horizontal)
- Support and resistance level identification
- Breakout analysis and target calculation

The module consists of:
- ContinuationPatternsProcessor: Core pattern detection and analysis logic
- ContinuationPatternsCharts: Specialized chart generation for continuation patterns
"""

from .processor import ContinuationPatternsProcessor, ContinuationPattern
from .charts import ContinuationPatternsCharts

__all__ = [
    'ContinuationPatternsProcessor',
    'ContinuationPattern',
    'ContinuationPatternsCharts'
]