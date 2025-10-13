#!/usr/bin/env python3
"""
Pattern Detection Agent - Pattern Analysis Module

This module provides comprehensive chart pattern detection including:
- Classic chart patterns (triangles, flags, channels, etc.)
- Price pattern analysis and classification
- Pattern completion tracking and validation
- Entry/exit point identification from patterns

The agent follows the distributed architecture pattern with separate components
for technical processing and chart generation.
"""

from .agent import PatternDetectionAgent
from .processor import PatternDetectionProcessor
from .charts import PatternDetectionChartGenerator

__all__ = [
    'PatternDetectionAgent',
    'PatternDetectionProcessor', 
    'PatternDetectionChartGenerator'
]

__version__ = "1.0.0"
