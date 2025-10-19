"""
Base Model Classes for Quantitative Trading System

This module provides base classes and interfaces for all ML models
and trading components in the system.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Standardized prediction result structure."""
    
    prediction: Union[float, int, str]
    confidence: float
    direction: Optional[str] = None  # 'up', 'down', 'sideways'
    magnitude: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class TradingSignal:
    """Standardized trading signal structure."""
    
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    position_size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: Optional[str] = None
    risk_level: str = 'medium'  # 'low', 'medium', 'high'
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class PerformanceMetrics:
    """Standardized performance metrics structure."""
    
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_return: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseMLEngine(ABC):
    """Base class for all ML engines."""
    
    def __init__(self, config: Any = None):
        self.config = config
        self.is_trained = False
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.training_history = {}
        
        logger.info(f"{self.__class__.__name__} initialized")
    
    @abstractmethod
    def train(self, data: Any, **kwargs) -> bool:
        """Train the ML engine. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def predict(self, data: Any, **kwargs) -> PredictionResult:
        """Make predictions. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def evaluate(self, data: Any, **kwargs) -> PerformanceMetrics:
        """Evaluate model performance. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> bool:
        """Save the trained model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> bool:
        """Load a trained model. Must be implemented by subclasses."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'is_trained': self.is_trained,
            'model_type': self.__class__.__name__,
            'feature_columns': self.feature_columns,
            'training_history': self.training_history
        }

class BaseTradingStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, config: Any = None):
        self.config = config
        self.positions = {}
        self.trade_history = []
        self.performance_metrics = {}
        
        logger.info(f"{self.__class__.__name__} initialized")
    
    @abstractmethod
    def generate_signal(self, market_data: Dict[str, Any], 
                       predictions: Dict[str, Any]) -> TradingSignal:
        """Generate trading signal based on market data and predictions."""
        pass
    
    @abstractmethod
    def execute_trade(self, symbol: str, signal: TradingSignal) -> Dict[str, Any]:
        """Execute a trade based on the signal."""
        pass
    
    @abstractmethod
    def calculate_position_size(self, symbol: str, signal: TradingSignal, 
                              portfolio_value: float) -> float:
        """Calculate appropriate position size for the trade."""
        pass
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status."""
        return {
            'positions': self.positions.copy(),
            'total_trades': len(self.trade_history),
            'active_positions': len([p for p in self.positions.values() if p != 0]),
            'performance_metrics': self.performance_metrics
        }

class BaseDataProcessor(ABC):
    """Base class for all data processors."""
    
    def __init__(self, config: Any = None):
        self.config = config
        self.data_cache = {}
        self.feature_cache = {}
        
        logger.info(f"{self.__class__.__name__} initialized")
    
    @abstractmethod
    def process_data(self, raw_data: Any, **kwargs) -> Any:
        """Process raw data into usable format."""
        pass
    
    @abstractmethod
    def create_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Create features from processed data."""
        pass
    
    def clear_cache(self):
        """Clear all cached data."""
        self.data_cache.clear()
        self.feature_cache.clear()
        logger.info("Data cache cleared")

class BaseRiskManager(ABC):
    """Base class for all risk management components."""
    
    def __init__(self, config: Any = None):
        self.config = config
        self.risk_metrics = {}
        self.risk_history = []
        
        logger.info(f"{self.__class__.__name__} initialized")
    
    @abstractmethod
    def assess_risk(self, position: Dict[str, Any], 
                   market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk for a given position."""
        pass
    
    @abstractmethod
    def calculate_var(self, portfolio: Dict[str, Any], 
                     confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk for the portfolio."""
        pass
    
    @abstractmethod
    def should_stop_trading(self, portfolio: Dict[str, Any]) -> bool:
        """Determine if trading should be stopped due to risk."""
        pass
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk summary."""
        return {
            'risk_metrics': self.risk_metrics.copy(),
            'risk_history_count': len(self.risk_history),
            'current_risk_level': self.risk_metrics.get('risk_level', 'unknown')
        }

class BaseEvaluator(ABC):
    """Base class for all evaluation components."""
    
    def __init__(self, config: Any = None):
        self.config = config
        self.evaluation_results = {}
        
        logger.info(f"{self.__class__.__name__} initialized")
    
    @abstractmethod
    def evaluate_model(self, model: Any, test_data: Any, **kwargs) -> PerformanceMetrics:
        """Evaluate a model's performance."""
        pass
    
    @abstractmethod
    def evaluate_strategy(self, strategy: Any, historical_data: Any, **kwargs) -> Dict[str, Any]:
        """Evaluate a trading strategy's performance."""
        pass
    
    @abstractmethod
    def compare_models(self, models: List[Any], test_data: Any, **kwargs) -> Dict[str, Any]:
        """Compare multiple models."""
        pass
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations."""
        return {
            'evaluation_results': self.evaluation_results.copy(),
            'total_evaluations': len(self.evaluation_results)
        }

# Utility functions for common operations
def validate_data_format(data: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that data has required columns."""
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    return True

def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate returns from price series."""
    return prices.pct_change().dropna()

def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Calculate rolling volatility."""
    return returns.rolling(window=window).std()

def normalize_features(features: pd.DataFrame) -> pd.DataFrame:
    """Normalize features to [0, 1] range."""
    return (features - features.min()) / (features.max() - features.min())

def create_lagged_features(data: pd.DataFrame, columns: List[str], 
                          lags: List[int]) -> pd.DataFrame:
    """Create lagged features for time series data."""
    result = data.copy()
    for col in columns:
        for lag in lags:
            result[f"{col}_lag_{lag}"] = data[col].shift(lag)
    return result
