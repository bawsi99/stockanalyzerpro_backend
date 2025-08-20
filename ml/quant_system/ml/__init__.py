"""
Unified ML Module for Quantitative Trading System

This module consolidates all machine learning functionality:
1. Pattern-based ML (CatBoost) - for pattern success modeling
2. Raw Data ML - for direct OHLCV data analysis
3. Hybrid ML - combining both approaches
4. Traditional ML Models - Random Forest, XGBoost, etc.
5. Feature Engineering - technical indicators and features
6. Model Training & Evaluation - comprehensive ML pipeline
"""

from .core import *
from .pattern_ml import *
from .raw_data_ml import *
from .hybrid_ml import *
# from .traditional_ml import *  # REMOVED (not needed with CatBoost)
from .feature_engineering import *
from .unified_manager import *

__version__ = "2.0.0"
__author__ = "Quantitative Trading System Team"

# Main entry point
__all__ = [
    'UnifiedMLConfig',
    'BaseMLEngine', 
    'MLModelRegistry',
    'pattern_ml_engine',
    'raw_data_ml_engine',
    'hybrid_ml_engine',
    # 'traditional_ml_engine',  # REMOVED
    'feature_engineer',
    'unified_ml_manager'
]
