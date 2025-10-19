"""
Phase 2 Integration Manager for Advanced Trading System

This module integrates all Phase 2 advanced components into a unified system:

1. Neural Architecture Search (NAS) - Automated model optimization
2. Meta-Learning Framework - Fast adaptation to new markets
3. Advanced Training Strategies - Sophisticated training techniques
4. Temporal Fusion Transformer - State-of-the-art time series modeling
5. Advanced Ensemble Management - Intelligent model combination
6. Real-time Integration - Live deployment capabilities
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
import json
import time
from pathlib import Path
import warnings
from datetime import datetime

# Import Phase 2 components
from ..training.neural_architecture_search import NeuralArchitectureSearch, NASConfig, create_nas_engine
from ..training.meta_learning import MetaLearningFramework, MetaLearningConfig, create_meta_learning_framework
from ..training.advanced_strategies import AdvancedTrainer, AdvancedTrainingConfig, create_advanced_trainer
from ..models.tft import TemporalFusionTransformer, TFTConfig, TFTTrainer, create_tft_model, create_tft_trainer

# Import Phase 1 components for integration
from ...features.feature_engineer import AdvancedFeatureEngineer, FeatureConfig
from ..models.ensemble_manager import DynamicEnsembleManager, EnsembleConfig
from .real_time_integrator import RealTimeDataIntegrator, RealTimeConfig

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class Phase2Config:
    """Unified configuration for Phase 2 advanced system."""
    
    # Component configurations
    nas_config: NASConfig = None
    meta_config: MetaLearningConfig = None
    training_config: AdvancedTrainingConfig = None
    tft_config: TFTConfig = None
    ensemble_config: EnsembleConfig = None
    feature_config: FeatureConfig = None
    realtime_config: RealTimeConfig = None
    
    # Integration parameters
    enable_nas: bool = True
    enable_meta_learning: bool = True
    enable_advanced_training: bool = True
    enable_tft: bool = True
    
    # Model selection strategy
    model_selection_strategy: str = 'performance_based'  # 'performance_based', 'ensemble', 'adaptive'
    model_update_frequency: int = 24  # hours
    
    # Performance monitoring
    performance_window: int = 1000  # samples
    adaptation_threshold: float = 0.05  # performance drop threshold for re-adaptation
    
    # System parameters
    max_models: int = 10
    model_cache_size: int = 50
    auto_retrain: bool = True
    
    # Hardware
    device: str = 'auto'
    
    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize sub-configs if not provided
        if self.nas_config is None:
            self.nas_config = NASConfig()
        if self.meta_config is None:
            self.meta_config = MetaLearningConfig()
        if self.training_config is None:
            self.training_config = AdvancedTrainingConfig()
        if self.tft_config is None:
            self.tft_config = TFTConfig()
        if self.ensemble_config is None:
            self.ensemble_config = EnsembleConfig()
        if self.feature_config is None:
            self.feature_config = FeatureConfig()
        if self.realtime_config is None:
            self.realtime_config = RealTimeConfig()

class ModelRegistry:
    """Registry for managing multiple advanced models."""
    
    def __init__(self, max_models: int = 10):
        self.max_models = max_models
        self.models = {}
        self.model_metadata = {}
        self.performance_history = {}
        
    def register_model(self, model_id: str, model: nn.Module, metadata: Dict[str, Any]):
        """Register a new model in the registry."""
        
        if len(self.models) >= self.max_models:
            # Remove oldest model
            oldest_id = min(self.model_metadata.keys(), 
                          key=lambda x: self.model_metadata[x].get('created_at', 0))
            self.remove_model(oldest_id)
        
        self.models[model_id] = model
        self.model_metadata[model_id] = {
            **metadata,
            'created_at': time.time(),
            'updated_at': time.time(),
            'prediction_count': 0
        }
        self.performance_history[model_id] = []
        
        logger.info(f"Registered model {model_id} in registry")
    
    def get_model(self, model_id: str) -> Optional[nn.Module]:
        """Get a model from registry."""
        return self.models.get(model_id)
    
    def remove_model(self, model_id: str):
        """Remove a model from registry."""
        if model_id in self.models:
            del self.models[model_id]
            del self.model_metadata[model_id]
            del self.performance_history[model_id]
            logger.info(f"Removed model {model_id} from registry")
    
    def update_performance(self, model_id: str, performance_metric: float):
        """Update model performance."""
        if model_id in self.performance_history:
            self.performance_history[model_id].append({
                'timestamp': time.time(),
                'performance': performance_metric
            })
            self.model_metadata[model_id]['updated_at'] = time.time()
            self.model_metadata[model_id]['prediction_count'] += 1
    
    def get_best_models(self, n: int = 3) -> List[str]:
        """Get top N performing models."""
        model_scores = {}
        
        for model_id, history in self.performance_history.items():
            if history:
                # Use recent performance
                recent_performance = [p['performance'] for p in history[-100:]]
                model_scores[model_id] = np.mean(recent_performance)
            else:
                model_scores[model_id] = 0.0
        
        # Sort by performance (higher is better)
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        return [model_id for model_id, _ in sorted_models[:n]]
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get status of model registry."""
        return {
            'total_models': len(self.models),
            'model_ids': list(self.models.keys()),
            'model_metadata': self.model_metadata,
            'best_models': self.get_best_models(5)
        }

class Phase2IntegrationManager:
    """Main integration manager for Phase 2 advanced components."""
    
    def __init__(self, input_size: int, output_size: int, config: Phase2Config = None):
        self.config = config or Phase2Config()
        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device(self.config.device)
        
        # Initialize model registry
        self.model_registry = ModelRegistry(self.config.max_models)
        
        # Initialize Phase 1 components (enhanced)
        self.feature_engineer = AdvancedFeatureEngineer(self.config.feature_config)
        self.ensemble_manager = DynamicEnsembleManager(self.config.ensemble_config)
        self.realtime_integrator = RealTimeDataIntegrator(self.config.realtime_config)
        
        # Initialize Phase 2 components
        self.nas_engine = None
        self.meta_framework = None
        self.tft_model = None
        self.tft_trainer = None
        
        if self.config.enable_nas:
            self.nas_engine = create_nas_engine(self.config.nas_config)
        
        if self.config.enable_meta_learning:
            self.meta_framework = create_meta_learning_framework(
                input_size, output_size, self.config.meta_config
            )
        
        if self.config.enable_tft:
            self.tft_model = create_tft_model(self.config.tft_config)
            self.tft_trainer = create_tft_trainer(self.tft_model, self.config.tft_config)
        
        # Current active models
        self.active_models = {}
        self.model_weights = {}
        
        # Performance tracking
        self.system_performance = []
        self.adaptation_history = []
        
        logger.info("Phase 2 Integration Manager initialized successfully")
    
    def discover_optimal_architectures(self, train_data: torch.Tensor, train_targets: torch.Tensor,
                                     val_data: torch.Tensor, val_targets: torch.Tensor) -> List[Dict[str, Any]]:
        """Use NAS to discover optimal model architectures."""
        
        if not self.config.enable_nas or self.nas_engine is None:
            logger.warning("NAS is disabled or not available")
            return []
        
        logger.info("Starting Neural Architecture Search...")
        
        # Run NAS
        best_architectures = self.nas_engine.search(train_data, train_targets, val_data, val_targets)
        
        # Register discovered models
        for i, arch_result in enumerate(best_architectures):
            model_id = f"nas_discovered_{i}_{int(time.time())}"
            
            # Create model from architecture
            from ..training.neural_architecture_search import DynamicNeuralNetwork
            model = DynamicNeuralNetwork(
                arch_result['architecture'], 
                self.input_size, 
                self.output_size
            )
            
            # Register in model registry
            metadata = {
                'source': 'nas',
                'architecture': arch_result['architecture'],
                'nas_score': arch_result['weighted_score'],
                'accuracy': arch_result['accuracy'],
                'speed': arch_result['speed'],
                'complexity': arch_result['complexity']
            }
            
            self.model_registry.register_model(model_id, model, metadata)
            
            logger.info(f"Registered NAS-discovered model {model_id} with score {arch_result['weighted_score']:.4f}")
        
        return best_architectures
    
    def setup_meta_learning(self, market_data: pd.DataFrame, features: pd.DataFrame, 
                           targets: pd.DataFrame) -> Dict[str, Any]:
        """Setup and train meta-learning framework."""
        
        if not self.config.enable_meta_learning or self.meta_framework is None:
            logger.warning("Meta-learning is disabled or not available")
            return {}
        
        logger.info("Setting up meta-learning framework...")
        
        # Train meta-learner
        training_history = self.meta_framework.train_meta_learner(market_data, features, targets)
        
        # Register meta-learned model
        model_id = f"meta_learned_{int(time.time())}"
        metadata = {
            'source': 'meta_learning',
            'training_history': training_history,
            'adaptation_capability': True
        }
        
        self.model_registry.register_model(model_id, self.meta_framework.base_model, metadata)
        
        logger.info(f"Meta-learning setup completed and registered as {model_id}")
        
        return training_history
    
    def train_with_advanced_strategies(self, model: nn.Module, train_data: torch.Tensor, 
                                     train_targets: torch.Tensor, val_data: torch.Tensor = None,
                                     val_targets: torch.Tensor = None) -> Dict[str, List[float]]:
        """Train model using advanced training strategies."""
        
        if not self.config.enable_advanced_training:
            logger.warning("Advanced training is disabled")
            return {}
        
        logger.info("Training with advanced strategies...")
        
        # Create advanced trainer
        trainer = create_advanced_trainer(model, self.config.training_config)
        
        # Prepare datasets
        from torch.utils.data import TensorDataset
        train_dataset = TensorDataset(train_data, train_targets)
        val_dataset = TensorDataset(val_data, val_targets) if val_data is not None else None
        
        # Train with advanced strategies
        training_history = trainer.train(train_dataset, val_dataset)
        
        logger.info("Advanced training completed")
        
        return training_history
    
    def quick_adapt_to_market(self, symbol: str, adaptation_data: torch.Tensor, 
                             adaptation_targets: torch.Tensor) -> str:
        """Quickly adapt to new market conditions using meta-learning."""
        
        if not self.config.enable_meta_learning or self.meta_framework is None:
            logger.warning("Meta-learning not available for adaptation")
            return None
        
        logger.info(f"Adapting to new market conditions for {symbol}...")
        
        # Quick adaptation
        adapted_model = self.meta_framework.quick_adapt(adaptation_data, adaptation_targets)
        
        # Register adapted model
        model_id = f"adapted_{symbol}_{int(time.time())}"
        metadata = {
            'source': 'meta_adaptation',
            'symbol': symbol,
            'adaptation_samples': len(adaptation_data),
            'base_model': 'meta_learned'
        }
        
        self.model_registry.register_model(model_id, adapted_model, metadata)
        
        logger.info(f"Market adaptation completed for {symbol}, registered as {model_id}")
        
        return model_id
    
    def generate_ensemble_prediction(self, symbol: str, features: torch.Tensor = None,
                                   market_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Generate prediction using ensemble of advanced models."""
        
        # Get real-time data if not provided
        if features is None and market_data is None:
            realtime_data = self.realtime_integrator.get_comprehensive_data(symbol)
            if 'error' not in realtime_data and not realtime_data['market_data'].empty:
                market_data = realtime_data['market_data']
                
                # Create features
                features = self.feature_engineer.create_all_features(
                    price_data=market_data,
                    news_data=realtime_data.get('news_data'),
                    social_data=realtime_data.get('social_data')
                )
                
                # Convert to tensor
                if isinstance(features, pd.DataFrame):
                    features = torch.tensor(features.values, dtype=torch.float32)
            else:
                logger.error("Failed to get real-time data for prediction")
                return {'error': 'No data available'}
        
        # Get best performing models
        best_model_ids = self.model_registry.get_best_models(n=5)
        
        if not best_model_ids:
            logger.error("No models available for prediction")
            return {'error': 'No models available'}
        
        # Collect predictions from multiple models
        predictions = {}
        model_confidences = {}
        
        for model_id in best_model_ids:
            model = self.model_registry.get_model(model_id)
            if model is not None:
                try:
                    model.eval()
                    with torch.no_grad():
                        if len(features.shape) == 1:
                            features = features.unsqueeze(0)
                        
                        prediction = model(features.to(self.device))
                        
                        if isinstance(prediction, torch.Tensor):
                            prediction = prediction.cpu().numpy()
                        
                        predictions[model_id] = prediction
                        
                        # Simple confidence based on recent performance
                        recent_performance = self.model_registry.performance_history.get(model_id, [])
                        if recent_performance:
                            confidence = np.mean([p['performance'] for p in recent_performance[-10:]])
                        else:
                            confidence = 0.5
                        
                        model_confidences[model_id] = confidence
                
                except Exception as e:
                    logger.error(f"Error getting prediction from model {model_id}: {e}")
                    continue
        
        if not predictions:
            return {'error': 'No valid predictions obtained'}
        
        # Weighted ensemble prediction
        total_weight = sum(model_confidences.values())
        if total_weight == 0:
            weights = {mid: 1.0/len(predictions) for mid in predictions.keys()}
        else:
            weights = {mid: conf/total_weight for mid, conf in model_confidences.items()}
        
        # Calculate ensemble prediction
        ensemble_pred = None
        for model_id, pred in predictions.items():
            weight = weights[model_id]
            if ensemble_pred is None:
                ensemble_pred = weight * pred
            else:
                ensemble_pred += weight * pred
        
        # Calculate ensemble confidence
        ensemble_confidence = sum(conf * weights[mid] for mid, conf in model_confidences.items())
        
        # Use TFT for probabilistic forecasting if available
        tft_prediction = None
        if self.config.enable_tft and self.tft_model is not None:
            try:
                # This would need proper TFT input formatting
                # For now, we'll skip TFT integration in ensemble
                pass
            except Exception as e:
                logger.warning(f"TFT prediction failed: {e}")
        
        result = {
            'ensemble_prediction': float(ensemble_pred[0]) if len(ensemble_pred.shape) > 0 else float(ensemble_pred),
            'ensemble_confidence': float(ensemble_confidence),
            'individual_predictions': {mid: float(pred[0]) if len(pred.shape) > 0 else float(pred) 
                                     for mid, pred in predictions.items()},
            'model_weights': weights,
            'num_models': len(predictions),
            'timestamp': datetime.now().isoformat()
        }
        
        if tft_prediction is not None:
            result['tft_prediction'] = tft_prediction
        
        return result
    
    def monitor_and_adapt(self, symbol: str, actual_return: float, predicted_return: float):
        """Monitor performance and trigger adaptation if needed."""
        
        # Calculate prediction error
        prediction_error = abs(actual_return - predicted_return)
        
        # Update model performances
        best_models = self.model_registry.get_best_models(n=5)
        for model_id in best_models:
            # Simple performance metric (inverse of error)
            performance = 1.0 / (1.0 + prediction_error)
            self.model_registry.update_performance(model_id, performance)
        
        # Check if adaptation is needed
        recent_performance = self.system_performance[-self.config.performance_window:] if len(self.system_performance) >= self.config.performance_window else self.system_performance
        
        if recent_performance:
            avg_performance = np.mean(recent_performance)
            current_performance = 1.0 / (1.0 + prediction_error)
            
            performance_drop = avg_performance - current_performance
            
            if performance_drop > self.config.adaptation_threshold:
                logger.info(f"Performance drop detected for {symbol}: {performance_drop:.4f}")
                
                if self.config.auto_retrain:
                    self._trigger_adaptation(symbol)
        
        # Store system performance
        self.system_performance.append(1.0 / (1.0 + prediction_error))
        
        # Limit history size
        if len(self.system_performance) > self.config.performance_window * 2:
            self.system_performance = self.system_performance[-self.config.performance_window:]
    
    def _trigger_adaptation(self, symbol: str):
        """Trigger automatic adaptation for a symbol."""
        
        logger.info(f"Triggering adaptation for {symbol}")
        
        try:
            # Get recent data for adaptation
            realtime_data = self.realtime_integrator.get_comprehensive_data(symbol)
            
            if 'error' not in realtime_data and not realtime_data['market_data'].empty:
                market_data = realtime_data['market_data']
                
                # Create features and targets for adaptation
                features = self.feature_engineer.create_all_features(
                    price_data=market_data,
                    news_data=realtime_data.get('news_data'),
                    social_data=realtime_data.get('social_data')
                )
                
                # Simple target: next return
                returns = market_data['close'].pct_change().dropna()
                
                if len(features) > len(returns):
                    features = features.iloc[:len(returns)]
                elif len(returns) > len(features):
                    returns = returns.iloc[:len(features)]
                
                # Convert to tensors
                adaptation_features = torch.tensor(features.values, dtype=torch.float32)
                adaptation_targets = torch.tensor(returns.values, dtype=torch.float32)
                
                # Quick adaptation
                adapted_model_id = self.quick_adapt_to_market(symbol, adaptation_features, adaptation_targets)
                
                if adapted_model_id:
                    # Record adaptation
                    self.adaptation_history.append({
                        'timestamp': time.time(),
                        'symbol': symbol,
                        'model_id': adapted_model_id,
                        'reason': 'performance_drop'
                    })
                    
                    logger.info(f"Adaptation completed for {symbol}")
        
        except Exception as e:
            logger.error(f"Adaptation failed for {symbol}: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        return {
            'registry_status': self.model_registry.get_registry_status(),
            'active_models': list(self.active_models.keys()),
            'system_performance': {
                'recent_performance': np.mean(self.system_performance[-100:]) if self.system_performance else 0.0,
                'total_predictions': len(self.system_performance),
                'adaptation_count': len(self.adaptation_history)
            },
            'component_status': {
                'nas_enabled': self.config.enable_nas,
                'meta_learning_enabled': self.config.enable_meta_learning,
                'advanced_training_enabled': self.config.enable_advanced_training,
                'tft_enabled': self.config.enable_tft
            },
            'recent_adaptations': self.adaptation_history[-10:] if self.adaptation_history else []
        }
    
    def save_system(self, save_path: str):
        """Save the entire Phase 2 system."""
        
        checkpoint = {
            'config': self.config,
            'model_registry_metadata': self.model_registry.model_metadata,
            'system_performance': self.system_performance,
            'adaptation_history': self.adaptation_history,
            'timestamp': time.time()
        }
        
        # Save models separately
        models_dir = Path(save_path) / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for model_id, model in self.model_registry.models.items():
            model_path = models_dir / f"{model_id}.pth"
            torch.save(model.state_dict(), model_path)
        
        # Save system state
        with open(Path(save_path) / 'system_state.json', 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        logger.info(f"Phase 2 system saved to {save_path}")
    
    def load_system(self, load_path: str):
        """Load the Phase 2 system."""
        
        # Load system state
        with open(Path(load_path) / 'system_state.json', 'r') as f:
            checkpoint = json.load(f)
        
        self.system_performance = checkpoint['system_performance']
        self.adaptation_history = checkpoint['adaptation_history']
        
        # Load models would require architecture information
        # This is a simplified version
        logger.info(f"Phase 2 system loaded from {load_path}")

# Factory functions
def create_phase2_manager(input_size: int, output_size: int, config: Phase2Config = None) -> Phase2IntegrationManager:
    """Create Phase 2 integration manager."""
    return Phase2IntegrationManager(input_size, output_size, config)

def quick_phase2_setup(input_size: int, output_size: int, 
                      train_data: torch.Tensor, train_targets: torch.Tensor,
                      val_data: torch.Tensor = None, val_targets: torch.Tensor = None) -> Phase2IntegrationManager:
    """Quick setup of Phase 2 system with automatic architecture discovery."""
    
    # Create manager
    manager = create_phase2_manager(input_size, output_size)
    
    # Discover optimal architectures
    if val_data is not None:
        best_archs = manager.discover_optimal_architectures(train_data, train_targets, val_data, val_targets)
        logger.info(f"Discovered {len(best_archs)} optimal architectures")
    
    return manager

# Global Phase 2 manager instance
phase2_manager = None

def get_phase2_manager(input_size: int = None, output_size: int = None, 
                      config: Phase2Config = None) -> Phase2IntegrationManager:
    """Get global Phase 2 manager instance."""
    global phase2_manager
    if phase2_manager is None and input_size is not None and output_size is not None:
        phase2_manager = create_phase2_manager(input_size, output_size, config)
    return phase2_manager

