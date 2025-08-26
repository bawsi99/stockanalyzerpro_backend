"""
Dynamic Ensemble Manager for Advanced Trading System

This module provides dynamic ensemble management capabilities including:
1. Market regime detection
2. Dynamic model selection
3. Adaptive weight optimization
4. Performance tracking
5. Uncertainty quantification
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import warnings
import torch
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class EnsembleConfig:
    """Configuration for dynamic ensemble manager."""
    
    # Model pool
    model_names: List[str] = None
    base_weights: Dict[str, float] = None
    
    # Regime detection
    regime_window: int = 60
    regime_threshold: float = 0.1
    num_regimes: int = 4  # Trending Bull, Trending Bear, Sideways, Volatile
    
    # Performance tracking
    performance_window: int = 30
    weight_update_frequency: int = 5  # Update weights every N days
    
    # Uncertainty quantification
    confidence_threshold: float = 0.6
    uncertainty_weight: float = 0.3
    
    def __post_init__(self):
        if self.model_names is None:
            self.model_names = ['multimodal_fusion', 'nbeats', 'temporal_fusion', 'cross_domain']
        if self.base_weights is None:
            self.base_weights = {name: 1.0/len(self.model_names) for name in self.model_names}

class MarketRegimeDetector:
    """Detects market regimes for dynamic ensemble selection."""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.regime_history = []
        self.regime_transitions = []
        
    def detect_regime(self, market_data: pd.DataFrame) -> str:
        """
        Detect current market regime.
        
        Args:
            market_data: Market data with price and volume information
        
        Returns:
            Market regime string
        """
        if len(market_data) < self.config.regime_window:
            return 'unknown'
        
        # Calculate regime indicators
        returns = market_data['close'].pct_change().dropna()
        volatility = returns.rolling(window=20).std()
        trend = market_data['close'].rolling(window=50).mean()
        momentum = market_data['close'] / trend - 1
        
        # Current values
        current_volatility = volatility.iloc[-1]
        current_momentum = momentum.iloc[-1]
        current_trend = (market_data['close'].iloc[-1] - market_data['close'].iloc[-self.config.regime_window]) / market_data['close'].iloc[-self.config.regime_window]
        
        # Regime classification
        if current_trend > self.config.regime_threshold and current_momentum > 0:
            regime = 'trending_bull'
        elif current_trend < -self.config.regime_threshold and current_momentum < 0:
            regime = 'trending_bear'
        elif current_volatility > volatility.quantile(0.8):
            regime = 'volatile'
        else:
            regime = 'sideways'
        
        # Store regime history
        self.regime_history.append({
            'timestamp': market_data.index[-1],
            'regime': regime,
            'volatility': current_volatility,
            'momentum': current_momentum,
            'trend': current_trend
        })
        
        return regime
    
    def get_regime_probabilities(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Get probabilities for each market regime."""
        regimes = ['trending_bull', 'trending_bear', 'sideways', 'volatile']
        probabilities = {}
        
        # Calculate regime indicators
        returns = market_data['close'].pct_change().dropna()
        volatility = returns.rolling(window=20).std()
        trend = market_data['close'].rolling(window=50).mean()
        momentum = market_data['close'] / trend - 1
        
        current_volatility = volatility.iloc[-1]
        current_momentum = momentum.iloc[-1]
        current_trend = (market_data['close'].iloc[-1] - market_data['close'].iloc[-self.config.regime_window]) / market_data['close'].iloc[-self.config.regime_window]
        
        # Calculate probabilities based on distance to regime centers
        trend_prob = np.exp(-abs(current_trend) / self.config.regime_threshold)
        momentum_prob = np.exp(-abs(current_momentum) / 0.1)
        volatility_prob = np.exp(-abs(current_volatility - volatility.mean()) / volatility.std())
        
        # Assign probabilities
        if current_trend > 0 and current_momentum > 0:
            probabilities['trending_bull'] = trend_prob * momentum_prob
        else:
            probabilities['trending_bull'] = 0.1
        
        if current_trend < 0 and current_momentum < 0:
            probabilities['trending_bear'] = trend_prob * momentum_prob
        else:
            probabilities['trending_bear'] = 0.1
        
        if current_volatility < volatility.quantile(0.6):
            probabilities['sideways'] = volatility_prob
        else:
            probabilities['sideways'] = 0.1
        
        if current_volatility > volatility.quantile(0.8):
            probabilities['volatile'] = volatility_prob
        else:
            probabilities['volatile'] = 0.1
        
        # Normalize probabilities
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {k: v/total_prob for k, v in probabilities.items()}
        
        return probabilities

class PerformanceTracker:
    """Tracks model performance for dynamic weight adjustment."""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.performance_history = {name: [] for name in config.model_names}
        self.recent_performance = {name: 0.0 for name in config.model_names}
        
    def update_performance(self, model_name: str, prediction: float, actual: float, timestamp: datetime):
        """Update performance for a specific model."""
        # Calculate prediction error
        error = abs(prediction - actual)
        accuracy = 1.0 / (1.0 + error)  # Convert error to accuracy-like metric
        
        # Store performance
        self.performance_history[model_name].append({
            'timestamp': timestamp,
            'prediction': prediction,
            'actual': actual,
            'error': error,
            'accuracy': accuracy
        })
        
        # Update recent performance (rolling average)
        recent_performances = [p['accuracy'] for p in self.performance_history[model_name][-self.config.performance_window:]]
        if recent_performances:
            self.recent_performance[model_name] = np.mean(recent_performances)
    
    def get_model_performance(self, model_name: str) -> Dict[str, float]:
        """Get performance metrics for a specific model."""
        if model_name not in self.performance_history:
            return {'accuracy': 0.0, 'recent_accuracy': 0.0, 'stability': 0.0}
        
        performances = self.performance_history[model_name]
        if not performances:
            return {'accuracy': 0.0, 'recent_accuracy': 0.0, 'stability': 0.0}
        
        accuracies = [p['accuracy'] for p in performances]
        recent_accuracies = [p['accuracy'] for p in performances[-self.config.performance_window:]]
        
        return {
            'accuracy': np.mean(accuracies),
            'recent_accuracy': np.mean(recent_accuracies) if recent_accuracies else 0.0,
            'stability': 1.0 - np.std(accuracies) if len(accuracies) > 1 else 0.0
        }
    
    def get_all_performances(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all models."""
        return {name: self.get_model_performance(name) for name in self.config.model_names}

class AdaptiveWeightOptimizer:
    """Optimizes ensemble weights based on recent performance and market conditions."""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.weight_history = []
        
    def optimize_weights(self, performances: Dict[str, Dict[str, float]], 
                        regime_probabilities: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize ensemble weights based on performance and market regime.
        
        Args:
            performances: Performance metrics for each model
            regime_probabilities: Probabilities for each market regime
        
        Returns:
            Optimized weights for each model
        """
        # Base weights from performance
        performance_weights = {}
        total_performance = 0.0
        
        for model_name, perf in performances.items():
            # Combine accuracy and stability
            combined_score = perf['recent_accuracy'] * 0.7 + perf['stability'] * 0.3
            performance_weights[model_name] = max(combined_score, 0.1)  # Minimum weight
            total_performance += performance_weights[model_name]
        
        # Normalize performance weights
        if total_performance > 0:
            performance_weights = {k: v/total_performance for k, v in performance_weights.items()}
        
        # Adjust weights based on market regime
        regime_adjusted_weights = self._adjust_weights_for_regime(performance_weights, regime_probabilities)
        
        # Store weight history
        self.weight_history.append({
            'timestamp': datetime.now(),
            'weights': regime_adjusted_weights.copy(),
            'performances': performances,
            'regime_probabilities': regime_probabilities
        })
        
        return regime_adjusted_weights
    
    def _adjust_weights_for_regime(self, base_weights: Dict[str, float], 
                                  regime_probabilities: Dict[str, float]) -> Dict[str, float]:
        """Adjust weights based on market regime probabilities."""
        # Define regime-specific model preferences
        regime_preferences = {
            'trending_bull': {
                'multimodal_fusion': 1.2,
                'nbeats': 1.1,
                'temporal_fusion': 1.0,
                'cross_domain': 0.9
            },
            'trending_bear': {
                'multimodal_fusion': 1.1,
                'nbeats': 1.0,
                'temporal_fusion': 1.2,
                'cross_domain': 1.1
            },
            'sideways': {
                'multimodal_fusion': 1.0,
                'nbeats': 1.2,
                'temporal_fusion': 0.9,
                'cross_domain': 1.1
            },
            'volatile': {
                'multimodal_fusion': 1.3,
                'nbeats': 0.8,
                'temporal_fusion': 1.1,
                'cross_domain': 1.2
            }
        }
        
        # Calculate regime-adjusted weights
        adjusted_weights = base_weights.copy()
        
        for regime, prob in regime_probabilities.items():
            if regime in regime_preferences:
                regime_multipliers = regime_preferences[regime]
                for model_name in adjusted_weights:
                    if model_name in regime_multipliers:
                        adjusted_weights[model_name] *= (1 + (regime_multipliers[model_name] - 1) * prob * 0.5)
        
        # Normalize adjusted weights
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
        
        return adjusted_weights

class UncertaintyQuantifier:
    """Quantifies prediction uncertainty for ensemble decisions."""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        
    def quantify_uncertainty(self, predictions: Dict[str, float], 
                           weights: Dict[str, float]) -> Dict[str, float]:
        """
        Quantify uncertainty in ensemble predictions.
        
        Args:
            predictions: Predictions from each model
            weights: Current ensemble weights
        
        Returns:
            Uncertainty metrics
        """
        # Calculate weighted prediction
        weighted_prediction = sum(predictions[model] * weights[model] for model in predictions)
        
        # Calculate prediction variance (ensemble uncertainty)
        prediction_variance = sum(weights[model] * (predictions[model] - weighted_prediction) ** 2 
                                for model in predictions)
        
        # Calculate weight entropy (model disagreement)
        weight_entropy = -sum(weights[model] * np.log(weights[model] + 1e-10) for model in weights)
        max_entropy = np.log(len(weights))
        normalized_entropy = weight_entropy / max_entropy
        
        # Calculate confidence score
        confidence = 1.0 / (1.0 + prediction_variance + normalized_entropy)
        
        return {
            'weighted_prediction': weighted_prediction,
            'prediction_variance': prediction_variance,
            'weight_entropy': weight_entropy,
            'normalized_entropy': normalized_entropy,
            'confidence': confidence,
            'uncertainty': 1.0 - confidence
        }

class DynamicEnsembleManager:
    """Manages dynamic ensemble selection and weight optimization."""
    
    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()
        self.regime_detector = MarketRegimeDetector(self.config)
        self.performance_tracker = PerformanceTracker(self.config)
        self.weight_optimizer = AdaptiveWeightOptimizer(self.config)
        self.uncertainty_quantifier = UncertaintyQuantifier(self.config)
        
        # Model pool (placeholders - will be replaced with actual models)
        self.model_pool = {name: None for name in self.config.model_names}
        self.current_weights = self.config.base_weights.copy()
        
        # Performance tracking
        self.last_weight_update = None
        self.ensemble_history = []
        
    def register_model(self, model_name: str, model_instance: Any):
        """Register a model in the ensemble."""
        if model_name in self.config.model_names:
            self.model_pool[model_name] = model_instance
            logger.info(f"Registered model: {model_name}")
        else:
            logger.warning(f"Unknown model name: {model_name}")
    
    def select_optimal_ensemble(self, market_data: pd.DataFrame) -> Tuple[List[str], Dict[str, float]]:
        """
        Select optimal ensemble for current market conditions.
        
        Args:
            market_data: Current market data
        
        Returns:
            Tuple of (selected_models, weights)
        """
        # Detect market regime
        current_regime = self.regime_detector.detect_regime(market_data)
        regime_probabilities = self.regime_detector.get_regime_probabilities(market_data)
        
        # Get model performances
        performances = self.performance_tracker.get_all_performances()
        
        # Optimize weights
        optimized_weights = self.weight_optimizer.optimize_weights(performances, regime_probabilities)
        
        # Select models based on weights (exclude models with very low weights)
        min_weight_threshold = 0.05
        selected_models = [model for model, weight in optimized_weights.items() 
                          if weight >= min_weight_threshold]
        
        # Normalize weights for selected models
        selected_weights = {model: optimized_weights[model] for model in selected_models}
        total_weight = sum(selected_weights.values())
        if total_weight > 0:
            selected_weights = {k: v/total_weight for k, v in selected_weights.items()}
        
        self.current_weights = selected_weights
        
        logger.info(f"Selected ensemble: {selected_models}")
        logger.info(f"Current weights: {selected_weights}")
        logger.info(f"Market regime: {current_regime}")
        
        return selected_models, selected_weights
    
    def get_ensemble_prediction(self, market_data: pd.DataFrame, 
                               features: pd.DataFrame) -> Dict[str, Any]:
        """
        Get ensemble prediction with uncertainty quantification.
        
        Args:
            market_data: Market data
            features: Engineered features
        
        Returns:
            Ensemble prediction with uncertainty
        """
        # Select optimal ensemble
        selected_models, weights = self.select_optimal_ensemble(market_data)
        
        # Get predictions from each model
        predictions = {}
        for model_name in selected_models:
            if self.model_pool[model_name] is not None:
                try:
                    # This is a placeholder - actual prediction would depend on model interface
                    prediction = self._get_model_prediction(model_name, features)
                    predictions[model_name] = prediction
                except Exception as e:
                    logger.error(f"Error getting prediction from {model_name}: {e}")
                    predictions[model_name] = 0.0
            else:
                logger.warning(f"Model {model_name} not available")
                predictions[model_name] = 0.0
        
        # Quantify uncertainty
        uncertainty_metrics = self.uncertainty_quantifier.quantify_uncertainty(predictions, weights)
        
        # Calculate weighted prediction
        weighted_prediction = uncertainty_metrics['weighted_prediction']
        
        # Store ensemble history
        self.ensemble_history.append({
            'timestamp': market_data.index[-1] if len(market_data) > 0 else datetime.now(),
            'selected_models': selected_models,
            'weights': weights,
            'predictions': predictions,
            'weighted_prediction': weighted_prediction,
            'uncertainty_metrics': uncertainty_metrics
        })
        
        return {
            'prediction': weighted_prediction,
            'confidence': uncertainty_metrics['confidence'],
            'uncertainty': uncertainty_metrics['uncertainty'],
            'model_predictions': predictions,
            'model_weights': weights,
            'selected_models': selected_models
        }
    
    def update_performance(self, model_name: str, prediction: float, actual: float, timestamp: datetime):
        """Update performance for a specific model."""
        self.performance_tracker.update_performance(model_name, prediction, actual, timestamp)
    
    def _get_model_prediction(self, model_name: str, features: pd.DataFrame) -> float:
        """Get prediction from a specific model."""
        if model_name not in self.model_pool or self.model_pool[model_name] is None:
            logger.warning(f"Model {model_name} not available in pool")
            return 0.0
        
        try:
            model = self.model_pool[model_name]
            
            # Get actual prediction based on model type
            if hasattr(model, 'predict'):
                # For models with standard predict interface
                prediction = model.predict(features)
                if isinstance(prediction, (list, np.ndarray)):
                    prediction = prediction[0] if len(prediction) > 0 else 0.0
                return float(prediction)
            
            elif hasattr(model, 'model') and hasattr(model.model, 'forward'):
                # For PyTorch models
                with torch.no_grad():
                    # Convert features to tensor format expected by model
                    if isinstance(features, pd.DataFrame):
                        features_tensor = torch.tensor(features.values, dtype=torch.float32)
                    else:
                        features_tensor = features
                    
                    if len(features_tensor.shape) == 2:
                        features_tensor = features_tensor.unsqueeze(0)  # Add batch dimension
                    
                    prediction = model.model(features_tensor)
                    if isinstance(prediction, torch.Tensor):
                        prediction = prediction.cpu().numpy()
                        if len(prediction.shape) > 1:
                            prediction = prediction[0, 0]  # Get first prediction
                        else:
                            prediction = prediction[0]
                    return float(prediction)
            
            else:
                logger.error(f"Model {model_name} has no recognized prediction interface")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error getting prediction from {model_name}: {e}")
            return 0.0
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get summary of ensemble performance and configuration."""
        performances = self.performance_tracker.get_all_performances()
        
        return {
            'current_weights': self.current_weights,
            'model_performances': performances,
            'ensemble_history_length': len(self.ensemble_history),
            'registered_models': list(self.model_pool.keys()),
            'available_models': [name for name, model in self.model_pool.items() if model is not None]
        }

# Global instance for easy access
dynamic_ensemble_manager = DynamicEnsembleManager()
