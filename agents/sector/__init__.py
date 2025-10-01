#!/usr/bin/env python3
"""
Sector Analysis Agent Package

Consolidates all sector-related functionality:
- SectorSynthesisProcessor: LLM synthesis of sector metrics
- SectorBenchmarkingProvider: Comprehensive sector benchmarking
- SectorClassifier: Sector classification and mapping
- EnhancedSectorClassifier: Enhanced classification with filtering
- SectorCacheManager: Cache management for sector analysis
- SectorPriceMonitor: Price monitoring for cache refresh triggers
- SectorAnalysisScheduler: Weekly scheduled cache refresh
"""

from .processor import SectorSynthesisProcessor
from .benchmarking import SectorBenchmarkingProvider
from .classifier import SectorClassifier
from .enhanced_classifier import EnhancedSectorClassifier, enhanced_sector_classifier
from .cache_manager import SectorCacheManager
from .price_monitor import SectorPriceMonitor
from .scheduler import SectorAnalysisScheduler

__all__ = [
    "SectorSynthesisProcessor",
    "SectorBenchmarkingProvider",
    "SectorClassifier",
    "EnhancedSectorClassifier",
    "enhanced_sector_classifier",
    "SectorCacheManager",
    "SectorPriceMonitor",
    "SectorAnalysisScheduler"
]
