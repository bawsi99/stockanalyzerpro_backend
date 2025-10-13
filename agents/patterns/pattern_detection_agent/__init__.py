#!/usr/bin/env python3
"""
Pattern Detection Agent - Pattern Analysis Module

This module provides comprehensive chart pattern detection including:
- Classic chart patterns (triangles, flags, channels, etc.)
- Price pattern analysis and classification
- Pattern completion tracking and validation
- Entry/exit point identification from patterns

The agent follows the distributed architecture pattern with separate components
for technical processing, chart generation, and LLM-powered insights.
"""

from .agent import PatternDetectionAgent
from .processor import PatternDetectionProcessor
from .charts import PatternDetectionCharts
from .llm_agent import PatternDetectionLLMAgent

__all__ = [
    'PatternDetectionAgent',
    'PatternDetectionProcessor', 
    'PatternDetectionCharts',
    'PatternDetectionLLMAgent'
]

__version__ = "1.0.0"