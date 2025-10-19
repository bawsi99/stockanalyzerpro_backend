"""
Unified Configuration System for Quantitative Trading

This module provides centralized configuration management for all components
of the quantitative trading system.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class MLConfig:
    """Core ML configuration settings."""
    
    # General ML settings
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.1
    cv_folds: int = 5
    
    # Pattern-based ML settings
    pattern_ml_enabled: bool = True
    catboost_iterations: int = 1000
    catboost_depth: int = 6
    catboost_learning_rate: float = 0.03
    
    # Raw data ML settings
    raw_data_ml_enabled: bool = True
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    
    # Feature engineering settings
    feature_engineering_enabled: bool = True
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    bb_period: int = 20
    bb_std: float = 2.0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.test_size + self.validation_size >= 1.0:
            raise ValueError("test_size + validation_size must be < 1.0")
        
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be >= 2")

@dataclass
class TradingConfig:
    """Trading system configuration settings."""
    
    # Risk management settings
    risk_management_enabled: bool = True
    max_position_size: float = 0.1
    default_stop_loss: float = 0.02
    default_take_profit: float = 0.04
    max_drawdown: float = 0.20
    kelly_fraction: float = 0.25
    
    # Backtesting settings
    backtesting_enabled: bool = True
    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    
    # Trading session settings
    session_duration_minutes: int = 60
    cycle_interval_seconds: int = 60
    max_concurrent_positions: int = 10

@dataclass
class DataConfig:
    """Data pipeline configuration settings."""
    
    # Data sources
    enable_yahoo_finance: bool = True
    enable_alpha_vantage: bool = False
    enable_news_api: bool = False
    enable_social_api: bool = False
    
    # Data processing
    cache_duration: int = 300  # seconds
    max_data_points: int = 10000
    missing_threshold: float = 0.1
    
    # Pattern detection
    horizon_days: int = 20
    tp_pct: float = 0.04
    sl_pct: float = 0.02

@dataclass
class AdvancedConfig:
    """Advanced ML configuration settings."""
    
    # Neural Architecture Search
    nas_enabled: bool = False
    nas_max_architectures: int = 10
    nas_accuracy_weight: float = 0.6
    nas_speed_weight: float = 0.2
    nas_complexity_weight: float = 0.1
    nas_interpretability_weight: float = 0.1
    
    # Meta Learning
    meta_learning_enabled: bool = False
    meta_learning_episodes: int = 100
    meta_learning_inner_lr: float = 0.01
    meta_learning_outer_lr: float = 0.001
    
    # Temporal Fusion Transformer
    tft_enabled: bool = False
    tft_hidden_size: int = 64
    tft_num_layers: int = 3
    tft_dropout: float = 0.1

@dataclass
class SystemConfig:
    """System-wide configuration settings."""
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "quant_system.log"
    
    # Performance
    enable_profiling: bool = False
    memory_limit_mb: int = 4096
    
    # Security
    enable_encryption: bool = False
    api_key_encryption: bool = False

class UnifiedConfig:
    """Unified configuration manager for the entire system."""
    
    def __init__(self):
        self.ml = MLConfig()
        self.trading = TradingConfig()
        self.data = DataConfig()
        self.advanced = AdvancedConfig()
        self.system = SystemConfig()
        
        logger.info("Unified configuration initialized")
    
    def update_config(self, section: str, **kwargs):
        """Update configuration for a specific section."""
        if hasattr(self, section):
            config_section = getattr(self, section)
            for key, value in kwargs.items():
                if hasattr(config_section, key):
                    setattr(config_section, key, value)
                    logger.info(f"Updated {section}.{key} = {value}")
                else:
                    logger.warning(f"Unknown configuration key: {section}.{key}")
        else:
            logger.error(f"Unknown configuration section: {section}")
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary."""
        return {
            'ml': self.ml.__dict__,
            'trading': self.trading.__dict__,
            'data': self.data.__dict__,
            'advanced': self.advanced.__dict__,
            'system': self.system.__dict__
        }
    
    def validate_config(self) -> bool:
        """Validate all configuration settings."""
        try:
            # Validate ML config
            self.ml.__post_init__()
            
            # Validate trading config
            if self.trading.max_position_size <= 0 or self.trading.max_position_size > 1:
                raise ValueError("max_position_size must be between 0 and 1")
            
            if self.trading.default_stop_loss <= 0 or self.trading.default_take_profit <= 0:
                raise ValueError("stop_loss and take_profit must be positive")
            
            # Validate data config
            if self.data.cache_duration <= 0:
                raise ValueError("cache_duration must be positive")
            
            logger.info("Configuration validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

# Global configuration instance
config = UnifiedConfig()

# Backward compatibility
UnifiedMLConfig = MLConfig  # For existing code compatibility
