"""
Volume Anomaly Detection Agent

This agent specializes in detecting unusual volume spikes and patterns using statistical analysis.
Focuses on identifying statistical outliers rather than institutional activity.
"""

from .processor import VolumeAnomalyProcessor
from .charts import VolumeAnomalyChartGenerator as VolumeAnomalyCharts

__all__ = [
    'VolumeAnomalyProcessor',
    'VolumeAnomalyCharts'
]
