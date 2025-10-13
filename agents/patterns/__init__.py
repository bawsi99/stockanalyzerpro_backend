"""
Pattern Analysis Agents Package

This package contains all pattern-based analysis agents for technical analysis.
It includes agents for reversal patterns, continuation patterns, technical overview,
and general pattern recognition.

Agents:
- Reversal: Analyzes reversal patterns like divergences, double tops/bottoms
- Continuation: Analyzes continuation patterns like triangles, flags, channels
- Technical Overview: Provides comprehensive technical analysis overview
- Pattern Recognition: General pattern identification and analysis
"""

# Import all pattern agents (technical_overview removed - use dedicated indicators system)
from .reversal import ReversalPatternsProcessor, ReversalPatternsCharts
from .continuation import ContinuationPatternsProcessor, ContinuationPatternsCharts
from .pattern_recognition import PatternRecognitionProcessor, PatternRecognitionCharts
from .patterns_agents import PatternAgentsOrchestrator

# Also export core components
from .market_structure_analyzer import MarketStructureAnalyzer
from .pattern_context_builder import PatternContextBuilder
from .pattern_llm_agent import PatternLLMAgent

__all__ = [
    'ReversalPatternsProcessor',
    'ReversalPatternsCharts',
    'ContinuationPatternsProcessor',
    'ContinuationPatternsCharts', 
    'PatternRecognitionProcessor',
    'PatternRecognitionCharts',
    'PatternAgentsOrchestrator',
    'MarketStructureAnalyzer',
    'PatternContextBuilder',
    'PatternLLMAgent'
]
