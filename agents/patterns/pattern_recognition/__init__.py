"""
Pattern Recognition Agent Package

Comprehensive pattern recognition module that provides market structure analysis,
cross-pattern relationships, and general pattern identification capabilities.
"""

from .processor import PatternRecognitionProcessor
from .charts import PatternRecognitionCharts

__all__ = [
    'PatternRecognitionProcessor',
    'PatternRecognitionCharts'
]

__version__ = "1.0.0"
__author__ = "Pattern Analysis System"
__description__ = "Advanced pattern recognition for comprehensive market analysis"