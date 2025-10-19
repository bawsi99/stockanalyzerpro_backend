"""
ML Engines for Quantitative Trading System

This module provides the core ML engines for the system:
- Pattern-based ML (CatBoost) - for pattern success modeling
- Raw Data ML (LSTM, Random Forest) - for direct OHLCV analysis
- Hybrid ML - combining both approaches
- Unified Manager - orchestrating all engines
"""

from .pattern_ml import PatternMLEngine, PatternRecord, PatternDataset
from .raw_data_ml import (
    RawDataMLEngine, 
    PricePrediction, 
    VolatilityPrediction, 
    MarketRegime,
    RawDataFeatureEngineer
)
from .hybrid_ml import HybridMLEngine, HybridPrediction
from .unified_manager import UnifiedMLManager

__version__ = "2.0.0"
__author__ = "Quantitative Trading System Team"

# Main exports
__all__ = [
    # Pattern ML
    'PatternMLEngine',
    'PatternRecord',
    'PatternDataset',
    
    # Raw Data ML
    'RawDataMLEngine',
    'PricePrediction',
    'VolatilityPrediction',
    'MarketRegime',
    'RawDataFeatureEngineer',
    
    # Hybrid ML
    'HybridMLEngine',
    'HybridPrediction',
    
    # Unified Manager
    'UnifiedMLManager'
]
