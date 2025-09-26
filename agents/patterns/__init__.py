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

# Import all pattern agents
from .reversal import ReversalPatternsProcessor, ReversalPatternsCharts
from .continuation import ContinuationPatternsProcessor, ContinuationPatternsCharts
from .technical_overview import TechnicalOverviewProcessor, TechnicalOverviewCharts
from .pattern_recognition import PatternRecognitionProcessor, PatternRecognitionCharts
from .patterns_agents import PatternAgentsOrchestrator, patterns_orchestrator

__all__ = [
    'ReversalPatternsProcessor',
    'ReversalPatternsCharts',
    'ContinuationPatternsProcessor',
    'ContinuationPatternsCharts', 
    'TechnicalOverviewProcessor',
    'TechnicalOverviewCharts',
    'PatternRecognitionProcessor',
    'PatternRecognitionCharts',
    'PatternAgentsOrchestrator',
    'patterns_orchestrator'
]
