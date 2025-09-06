"""
Data Pipeline for Quantitative Trading System

This module provides data processing and management capabilities:
- Data pipeline for processing raw market data
- Dataset builder for creating training datasets
- Market data integration from multiple sources
- Data storage and caching
"""

from .pipeline import *
from .dataset_builder import *
from .market_data_integration import *

__version__ = "2.0.0"
__author__ = "Quantitative Trading System Team"
