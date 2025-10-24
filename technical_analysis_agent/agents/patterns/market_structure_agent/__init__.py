#!/usr/bin/env python3
"""
Market Structure Agent - Pattern Analysis Module

This module provides comprehensive market structure analysis including:
- Swing point identification and analysis
- BOS (Break of Structure) and CHOCH (Change of Character) detection  
- Trend analysis and structural integrity assessment
- Support/resistance level identification from market structure

The agent follows the distributed architecture pattern with separate components
for technical processing, chart generation, and LLM-powered insights.
"""

from .agent import MarketStructureAgent
from .processor import MarketStructureProcessor
from .charts import MarketStructureCharts
from .llm_agent import MarketStructureLLMAgent

__all__ = [
    'MarketStructureAgent',
    'MarketStructureProcessor', 
    'MarketStructureCharts',
    'MarketStructureLLMAgent'
]

__version__ = "1.0.0"