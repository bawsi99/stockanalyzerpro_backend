"""
Unified ML Manager

This module provides a unified interface for all ML engines:
1. Pattern-based ML (CatBoost)
2. Raw Data ML (LSTM, Random Forest)
3. Hybrid ML (Combined approach)
4. Traditional ML (Random Forest, XGBoost)
5. Feature Engineering

Provides a single entry point for all ML operations.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import pandas as pd
import numpy as np

from .core import UnifiedMLConfig, global_registry
from .pattern_ml import pattern_ml_engine
from .raw_data_ml import raw_data_ml_engine
from .hybrid_ml import hybrid_ml_engine
# from .traditional_ml import traditional_ml_engine  # REMOVED
from .feature_engineering import feature_engineer

logger = logging.getLogger(__name__)

class UnifiedMLManager:
    """Unified manager for all ML engines."""
    
    def __init__(self, config: UnifiedMLConfig = None):
        self.config = config or UnifiedMLConfig()
        
        # Initialize all ML engines
        self.pattern_engine = pattern_ml_engine
        self.raw_data_engine = raw_data_ml_engine
        self.hybrid_engine = hybrid_ml_engine
        # self.traditional_engine = traditional_ml_engine  # REMOVED
        self.feature_engine = feature_engineer
        
        # Engine status
        self.engine_status = {
            'pattern_ml': False,
            'raw_data_ml': False,
            'hybrid_ml': False,
            # 'traditional_ml': False,  # REMOVED
            'feature_engineering': True
        }
        
        logger.info("Unified ML Manager initialized")
    
    def train_all_engines(self, stock_data: pd.DataFrame, pattern_data: Optional[Dict] = None) -> Dict[str, bool]:
        """Train all ML engines."""
        logger.info("Training all ML engines...")
        
        results = {}
        
        # Validate stock data
        if stock_data is None or stock_data.empty:
            logger.warning("Empty stock data provided for training")
            return {
                'pattern_ml': False,
                'raw_data_ml': False,
                'hybrid_ml': False,
                'error': 'No stock data available for training'
            }
            
        # Check minimum required columns for stock data
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in stock_data.columns]
        if missing_cols:
            logger.warning(f"Missing required columns for training: {missing_cols}")
            return {
                'pattern_ml': False,
                'raw_data_ml': False,
                'hybrid_ml': False,
                'error': f"Missing required columns: {missing_cols}"
            }
            
        # Check minimum data points
        if len(stock_data) < 50:
            logger.warning(f"Insufficient data points for training: {len(stock_data)} (minimum 50 required)")
            return {
                'pattern_ml': False,
                'raw_data_ml': False,
                'hybrid_ml': False,
                'error': f"Insufficient data points: {len(stock_data)} (minimum 50 required)"
            }
        
        # Train pattern-based ML
        if self.config.pattern_ml_enabled:
            try:
                if pattern_data:
                    # Add pattern data to engine
                    for pattern_type, records in pattern_data.items():
                        for record in records:
                            self.pattern_engine.add_pattern_data(
                                pattern_type=pattern_type,
                                features=record.get('features', {}),
                                outcome=record.get('outcome', False),
                                confirmed=record.get('confirmed', True),
                                timestamp=record.get('timestamp')
                            )
                
                results['pattern_ml'] = self.pattern_engine.train()
                self.engine_status['pattern_ml'] = results['pattern_ml']
                logger.info(f"Pattern ML training: {'SUCCESS' if results['pattern_ml'] else 'FAILED'}")
            except Exception as e:
                logger.error(f"Pattern ML training failed: {e}")
                results['pattern_ml'] = False
                self.engine_status['pattern_ml'] = False
        
        # Train raw data ML
        if self.config.raw_data_ml_enabled:
            try:
                results['raw_data_ml'] = self.raw_data_engine.train(stock_data)
                self.engine_status['raw_data_ml'] = results['raw_data_ml']
                logger.info(f"Raw Data ML training: {'SUCCESS' if results['raw_data_ml'] else 'FAILED'}")
            except Exception as e:
                logger.error(f"Raw Data ML training failed: {e}")
                results['raw_data_ml'] = False
                self.engine_status['raw_data_ml'] = False
        
        # Traditional ML - REMOVED (not needed with CatBoost)
        # if self.config.traditional_ml_enabled:
        #     try:
        #         # Create features first
        #         features_df = self.feature_engine.create_all_features(stock_data)
        #         if not features_df.empty:
        #             results['traditional_ml'] = self.traditional_engine.train_all_models(features_df)
        #             self.engine_status['traditional_ml'] = bool(results['traditional_ml'])
        #             logger.info(f"Traditional ML training: {'SUCCESS' if self.engine_status['traditional_ml'] else 'FAILED'}")
        #         else:
        #             logger.warning("No features available for traditional ML training")
        #             results['traditional_ml'] = False
        #             self.engine_status['traditional_ml'] = False
        #     except Exception as e:
        #         logger.error(f"Traditional ML training failed: {e}")
        #             results['traditional_ml'] = False
        #             self.engine_status['traditional_ml'] = False
        
        # Train hybrid ML
        if self.engine_status['pattern_ml'] or self.engine_status['raw_data_ml']:
            try:
                results['hybrid_ml'] = self.hybrid_engine.train(stock_data, pattern_data)
                self.engine_status['hybrid_ml'] = results['hybrid_ml']
                logger.info(f"Hybrid ML training: {'SUCCESS' if results['hybrid_ml'] else 'FAILED'}")
            except Exception as e:
                logger.error(f"Hybrid ML training failed: {e}")
                results['hybrid_ml'] = False
                self.engine_status['hybrid_ml'] = False
        
        logger.info(f"Training completed. Results: {results}")
        return results
    
    def get_comprehensive_prediction(self, stock_data: pd.DataFrame, 
                                   pattern_features: Optional[Dict] = None,
                                   pattern_type: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive predictions from all available engines."""
        logger.info("Generating comprehensive predictions...")
        
        predictions = {}
        
        # Validate input data
        if stock_data is None or stock_data.empty:
            logger.warning("Empty stock data provided for prediction")
            return {
                'error': 'No stock data available for prediction',
                'consensus': {
                    'overall_signal': 'hold',
                    'confidence': 0.5,
                    'risk_level': 'high',
                    'recommendation': 'Insufficient data for analysis'
                }
            }
        
        # Pattern-based prediction
        if self.engine_status['pattern_ml'] and pattern_features:
            try:
                pattern_prob = self.pattern_engine.predict(pattern_features, pattern_type)
                predictions['pattern_ml'] = {
                    'success_probability': pattern_prob,
                    'confidence': pattern_prob,
                    'signal': 'buy' if pattern_prob > 0.6 else 'sell'
                }
            except Exception as e:
                logger.error(f"Pattern ML prediction failed: {e}")
                predictions['pattern_ml'] = {'error': str(e)}
        
        # Raw data prediction
        if self.engine_status['raw_data_ml']:
            try:
                # Check if we have the minimum required columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in stock_data.columns]
                
                if missing_cols:
                    logger.warning(f"Missing required columns for prediction: {missing_cols}")
                    predictions['raw_data_ml'] = {'error': f"Missing required columns: {missing_cols}"}
                else:
                    # Check if we have enough data points
                    if len(stock_data) < 50:
                        logger.warning(f"Insufficient data points for prediction: {len(stock_data)} (minimum 50 required)")
                        predictions['raw_data_ml'] = {'error': f"Insufficient data points: {len(stock_data)} (minimum 50 required)"}
                    else:
                        price_pred = self.raw_data_engine.predict(stock_data)
                        volatility_pred = self.raw_data_engine.predict_volatility(stock_data)
                        market_regime = self.raw_data_engine.classify_market_regime(stock_data)
                        
                        predictions['raw_data_ml'] = {
                            'price_prediction': {
                                'direction': price_pred.direction,
                                'magnitude': price_pred.magnitude,
                                'confidence': price_pred.confidence
                            },
                            'volatility_prediction': {
                                'current': volatility_pred.current_volatility,
                                'predicted': volatility_pred.predicted_volatility,
                                'regime': volatility_pred.volatility_regime
                            },
                            'market_regime': {
                                'regime': market_regime.regime,
                                'strength': market_regime.strength,
                                'confidence': market_regime.confidence
                            }
                        }
            except Exception as e:
                logger.error(f"Raw Data ML prediction failed: {e}")
                predictions['raw_data_ml'] = {'error': str(e)}
        
        # Traditional ML prediction - REMOVED (not needed with CatBoost)
        # if self.engine_status['traditional_ml']:
        #     try:
        #         features_df = self.feature_engine.create_all_features(stock_data)
        #         if not features_df.empty:
        #             trad_predictions = {}
        #                     
        #                     # Get predictions for each model type
        #         for model_type in ['price_prediction', 'direction_prediction', 'volatility_prediction']:
        #             if model_type in self.traditional_engine.models:
        #                 pred_result = self.traditional_engine.predict(features_df, model_type)
        #                 if pred_result:
        #                     trad_predictions[model_type] = pred_result
        #                     
        #                     predictions['traditional_ml'] = trad_predictions
        #         else:
        #             predictions['traditional_ml'] = {'error': 'No features available'}
        #     except Exception as e:
        #         logger.error(f"Traditional ML prediction failed: {e}")
        #             predictions['traditional_ml'] = {'error': str(e)}
        
        # Hybrid prediction
        if self.engine_status['hybrid_ml']:
            try:
                hybrid_analysis = self.hybrid_engine.get_comprehensive_analysis(
                    stock_data, pattern_features, pattern_type
                )
                predictions['hybrid_ml'] = hybrid_analysis
            except Exception as e:
                logger.error(f"Hybrid ML prediction failed: {e}")
                predictions['hybrid_ml'] = {'error': str(e)}
        
        # Generate consensus
        consensus = self._generate_consensus(predictions)
        predictions['consensus'] = consensus
        
        logger.info("Comprehensive predictions generated successfully")
        return predictions
    
    def _generate_consensus(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate consensus from all available predictions."""
        consensus = {
            'overall_signal': 'hold',
            'confidence': 0.5,
            'risk_level': 'medium',
            'recommendation': 'Wait for clearer signals'
        }
        
        try:
            signals = []
            confidences = []
            
            # Collect signals and confidences
            for engine, pred in predictions.items():
                if engine == 'consensus' or 'error' in pred:
                    continue
                
                if engine == 'pattern_ml' and 'signal' in pred:
                    signals.append(pred['signal'])
                    confidences.append(pred['confidence'])
                
                elif engine == 'raw_data_ml' and 'price_prediction' in pred:
                    price_pred = pred['price_prediction']
                    if price_pred['direction'] == 'up':
                        signals.append('buy')
                    elif price_pred['direction'] == 'down':
                        signals.append('sell')
                    else:
                        signals.append('hold')
                    confidences.append(price_pred['confidence'])
                
                elif engine == 'hybrid_ml' and 'hybrid_prediction' in pred:
                    hybrid_pred = pred['hybrid_prediction']
                    signals.append(hybrid_pred['consensus_signal'])
                    confidences.append(hybrid_pred['combined_confidence'])
            
            if signals and confidences:
                # Determine consensus signal
                buy_count = signals.count('buy') + signals.count('strong_buy')
                sell_count = signals.count('sell') + signals.count('strong_sell')
                
                if buy_count > sell_count:
                    consensus['overall_signal'] = 'buy'
                elif sell_count > buy_count:
                    consensus['overall_signal'] = 'sell'
                else:
                    consensus['overall_signal'] = 'hold'
                
                # Calculate average confidence
                consensus['confidence'] = np.mean(confidences)
                
                # Determine risk level
                if consensus['confidence'] > 0.7:
                    consensus['risk_level'] = 'low'
                elif consensus['confidence'] < 0.4:
                    consensus['risk_level'] = 'high'
                else:
                    consensus['risk_level'] = 'medium'
                
                # Generate recommendation
                if consensus['overall_signal'] == 'buy' and consensus['confidence'] > 0.6:
                    consensus['recommendation'] = 'Strong buy signal with good confidence'
                elif consensus['overall_signal'] == 'sell' and consensus['confidence'] > 0.6:
                    consensus['recommendation'] = 'Strong sell signal with good confidence'
                elif consensus['overall_signal'] == 'hold':
                    consensus['recommendation'] = 'Mixed signals, wait for clearer setup'
                else:
                    consensus['recommendation'] = 'Weak signal, exercise caution'
        
        except Exception as e:
            logger.error(f"Consensus generation failed: {e}")
            consensus['error'] = str(e)
        
        return consensus
    
    def get_engine_status(self) -> Dict[str, bool]:
        """Get status of all ML engines."""
        return self.engine_status.copy()
    
    def get_model_registry(self) -> Dict[str, Any]:
        """Get information about all registered models."""
        return {
            'registered_models': global_registry.list_models(),
            'engine_status': self.engine_status,
            'total_models': len(global_registry.list_models())
        }
    
    def save_all_models(self, base_path: str) -> Dict[str, bool]:
        """Save all trained models."""
        logger.info("Saving all trained models...")
        
        results = {}
        
        # Save pattern ML model
        if self.engine_status['pattern_ml']:
            try:
                results['pattern_ml'] = self.pattern_engine.save_model(f"{base_path}/pattern_ml.joblib")
            except Exception as e:
                logger.error(f"Failed to save pattern ML model: {e}")
                results['pattern_ml'] = False
        
        # Save raw data ML model
        if self.engine_status['raw_data_ml']:
            try:
                results['raw_data_ml'] = self.raw_data_engine.save_model(f"{base_path}/raw_data_ml.joblib")
            except Exception as e:
                logger.error(f"Failed to save raw data ML model: {e}")
                results['raw_data_ml'] = False
        
        # Save hybrid ML model
        if self.engine_status['hybrid_ml']:
            try:
                results['hybrid_ml'] = self.hybrid_engine.save_model(f"{base_path}/hybrid_ml.joblib")
            except Exception as e:
                logger.error(f"Failed to save hybrid ML model: {e}")
                results['hybrid_ml'] = False
        
        # Save traditional ML model
        # Traditional ML save - REMOVED (not needed with CatBoost)
        # if self.engine_status['traditional_ml']:
        #     try:
        #         results['traditional_ml'] = self.traditional_engine.save_model(f"{base_path}/traditional_ml.joblib")
        #     except Exception as e:
        #         logger.error(f"Failed to save traditional ML model: {e}")
        #         results['traditional_ml'] = False
        
        logger.info(f"Model saving completed. Results: {results}")
        return results
    
    def load_all_models(self, base_path: str) -> Dict[str, bool]:
        """Load all saved models."""
        logger.info("Loading all saved models...")
        
        results = {}
        
        # Load pattern ML model
        try:
            results['pattern_ml'] = self.pattern_engine.load_model(f"{base_path}/pattern_ml.joblib")
            self.engine_status['pattern_ml'] = results['pattern_ml']
        except Exception as e:
            logger.error(f"Failed to load pattern ML model: {e}")
            results['pattern_ml'] = False
        
        # Load raw data ML model
        try:
            results['raw_data_ml'] = self.raw_data_engine.load_model(f"{base_path}/raw_data_ml.joblib")
            self.engine_status['raw_data_ml'] = results['raw_data_ml']
        except Exception as e:
            logger.error(f"Failed to load raw data ML model: {e}")
            results['raw_data_ml'] = False
        
        # Load hybrid ML model
        try:
            results['hybrid_ml'] = self.hybrid_engine.load_model(f"{base_path}/hybrid_ml.joblib")
            self.engine_status['hybrid_ml'] = results['hybrid_ml']
        except Exception as e:
            logger.error(f"Failed to load hybrid ML model: {e}")
            results['hybrid_ml'] = False
        
        # Load traditional ML model
        # Load traditional ML - REMOVED (not needed with CatBoost)
        # try:
        #     results['traditional_ml'] = self.traditional_engine.load_model(f"{base_path}/traditional_ml.joblib")
        #     self.engine_status['traditional_ml'] = False
        # except Exception as e:
        #     logger.error(f"Failed to load traditional ML model: {e}")
        #     results['traditional_ml'] = False
        
        logger.info(f"Model loading completed. Results: {results}")
        return results
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features using the feature engineering engine."""
        return self.feature_engine.create_all_features(data)
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary."""
        return {
            'engine_status': self.engine_status,
            'registered_models': global_registry.list_models(),
            'total_models': len(global_registry.list_models()),
            'configuration': {
                'pattern_ml_enabled': self.config.pattern_ml_enabled,
                'raw_data_ml_enabled': self.config.raw_data_ml_enabled,
                'feature_engineering_enabled': self.config.feature_engineering_enabled
                # 'traditional_ml_enabled': self.config.traditional_ml_enabled,  # REMOVED
            },
            'timestamp': datetime.now().isoformat()
        }

# Global instance
unified_ml_manager = UnifiedMLManager()
