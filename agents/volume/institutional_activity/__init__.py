"""
Institutional Activity Analysis Agent

This agent specializes in analyzing institutional trading patterns and behavior.
Focuses on detecting large order flows, block trades, and institutional accumulation/distribution patterns.
"""

from .processor import InstitutionalActivityProcessor
from .charts import InstitutionalActivityChartGenerator as InstitutionalActivityCharts
from .integration import InstitutionalActivityIntegration

__all__ = [
    'InstitutionalActivityProcessor',
    'InstitutionalActivityCharts',
    'InstitutionalActivityIntegration'
]
