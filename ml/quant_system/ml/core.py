"""
Core ML Configuration and Base Classes

This module provides unified configuration and base classes for all ML components.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class UnifiedMLConfig:
    """Unified configuration for all ML components."""
    
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
    
    # Traditional ML settings - REMOVED (not needed with CatBoost)
    # traditional_ml_enabled: bool = False
    # rf_n_estimators: int = 100
    # xgb_n_estimators: int = 100
    
    # Feature engineering settings
    feature_engineering_enabled: bool = True
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    bb_period: int = 20
    bb_std: float = 2.0
    
    # Risk management settings
    risk_management_enabled: bool = True
    max_position_size: float = 0.1
    default_stop_loss: float = 0.02
    default_take_profit: float = 0.04
    
    # Backtesting settings
    backtesting_enabled: bool = True
    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.test_size + self.validation_size >= 1.0:
            raise ValueError("test_size + validation_size must be < 1.0")
        
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be >= 2")

@dataclass
class MLModelRegistry:
    """Registry for all trained ML models."""
    
    models: Dict[str, Any] = field(default_factory=dict)
    scalers: Dict[str, Any] = field(default_factory=dict)
    feature_columns: Dict[str, List[str]] = field(default_factory=dict)
    training_history: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def register_model(self, name: str, model: Any, scaler: Any = None, 
                      feature_columns: List[str] = None, training_info: Dict[str, Any] = None):
        """Register a trained model."""
        self.models[name] = model
        if scaler is not None:
            self.scalers[name] = scaler
        if feature_columns is not None:
            self.feature_columns[name] = feature_columns
        if training_info is not None:
            self.training_history[name] = training_info
        
        logger.info(f"Model '{name}' registered successfully")
    
    def get_model(self, name: str) -> Optional[Any]:
        """Get a registered model."""
        return self.models.get(name)
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self.models.keys())
    
    def remove_model(self, name: str) -> bool:
        """Remove a registered model."""
        if name in self.models:
            del self.models[name]
            if name in self.scalers:
                del self.scalers[name]
            if name in self.feature_columns:
                del self.feature_columns[name]
            if name in self.training_history:
                del self.training_history[name]
            logger.info(f"Model '{name}' removed successfully")
            return True
        return False

class BaseMLEngine:
    """Base class for all ML engines."""
    
    def __init__(self, config: UnifiedMLConfig = None):
        self.config = config or UnifiedMLConfig()
        self.registry = MLModelRegistry()
        self.is_trained = False
    
    def train(self, data: Any, **kwargs) -> bool:
        """Train the ML engine. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement train method")
    
    def predict(self, data: Any, **kwargs) -> Any:
        """Make predictions. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement predict method")
    
    def evaluate(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Evaluate model performance. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement evaluate method")
    
    def save_model(self, path: str) -> bool:
        """Save the trained model. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement save_model method")
    
    def load_model(self, path: str) -> bool:
        """Load a trained model. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement load_model method")

# Global configuration instance
default_config = UnifiedMLConfig()

# Global model registry
global_registry = MLModelRegistry()
