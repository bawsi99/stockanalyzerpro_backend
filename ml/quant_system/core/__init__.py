"""
Core Infrastructure for Quantitative Trading System

This module provides the foundational components for the entire system:
- Unified configuration management
- Base model classes and interfaces
- Centralized model registry
- Common utility functions
"""

from .config import (
    MLConfig,
    TradingConfig,
    DataConfig,
    AdvancedConfig,
    SystemConfig,
    UnifiedConfig,
    config,
    UnifiedMLConfig  # Backward compatibility
)

from .base_models import (
    PredictionResult,
    TradingSignal,
    PerformanceMetrics,
    BaseMLEngine,
    BaseTradingStrategy,
    BaseDataProcessor,
    BaseRiskManager,
    BaseEvaluator,
    validate_data_format,
    calculate_returns,
    calculate_volatility,
    normalize_features,
    create_lagged_features
)

from .registry import (
    ModelMetadata,
    ModelRegistry,
    global_registry,
    MLModelRegistry  # Backward compatibility
)

from .utils import (
    setup_logging,
    timing_decorator,
    validate_dataframe,
    clean_dataframe,
    calculate_technical_indicators,
    calculate_returns,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    normalize_features,
    create_lagged_features,
    create_rolling_features,
    save_object,
    load_object,
    format_currency,
    format_percentage,
    get_trading_days,
    suppress_warnings,
    memory_usage_mb,
    log_memory_usage,
    TRADING_DAYS_PER_YEAR,
    TIME_PERIODS
)

__version__ = "2.0.0"
__author__ = "Quantitative Trading System Team"

# Main exports
__all__ = [
    # Configuration
    'MLConfig',
    'TradingConfig', 
    'DataConfig',
    'AdvancedConfig',
    'SystemConfig',
    'UnifiedConfig',
    'config',
    'UnifiedMLConfig',  # Backward compatibility
    
    # Base Models
    'PredictionResult',
    'TradingSignal',
    'PerformanceMetrics',
    'BaseMLEngine',
    'BaseTradingStrategy',
    'BaseDataProcessor',
    'BaseRiskManager',
    'BaseEvaluator',
    
    # Registry
    'ModelMetadata',
    'ModelRegistry',
    'global_registry',
    'MLModelRegistry',  # Backward compatibility
    
    # Utilities
    'setup_logging',
    'timing_decorator',
    'validate_dataframe',
    'clean_dataframe',
    'calculate_technical_indicators',
    'calculate_returns',
    'calculate_volatility',
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'normalize_features',
    'create_lagged_features',
    'create_rolling_features',
    'save_object',
    'load_object',
    'format_currency',
    'format_percentage',
    'get_trading_days',
    'suppress_warnings',
    'memory_usage_mb',
    'log_memory_usage',
    'TRADING_DAYS_PER_YEAR',
    'TIME_PERIODS'
]
