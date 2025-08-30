"""
ML Inference Module for Backend Integration

This module provides ML-powered pattern prediction capabilities by integrating
with the unified ML system (quant_system/ml/).
"""

import sys
import os
import logging
from typing import Dict, Optional, Any
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Global ML manager instance
_ml_manager = None
_model_version = "1.0.0"

def get_ml_manager():
    """Get or create the unified ML manager instance."""
    global _ml_manager
    
    if _ml_manager is None:
        try:
            # Import from quant_system ML module using the new path
            quant_system_path = os.path.join(os.path.dirname(__file__), 'quant_system')
            logger.info(f"🔍 Quant system path: {quant_system_path}")
            logger.info(f"🔍 Absolute quant system path: {os.path.abspath(quant_system_path)}")
            
            # Add quant_system to Python path
            if quant_system_path not in sys.path:
                sys.path.insert(0, quant_system_path)
                logger.info(f"✅ Added {quant_system_path} to sys.path")
            
            # Import from the quant_system ML module
            logger.info("🔍 Attempting to import pattern_ml_engine...")
            from .quant_system.ml.pattern_ml import pattern_ml_engine
            logger.info("✅ Successfully imported pattern_ml_engine")
            
            # Load the trained model if not already loaded
            if not pattern_ml_engine.is_trained:
                model_path = os.path.join(quant_system_path, 'models', 'pattern_catboost.joblib')
                if os.path.exists(model_path):
                    pattern_ml_engine.load_model(model_path)
                    logger.info("✅ Trained CatBoost model loaded successfully")
                else:
                    logger.warning("No trained model found, will use fallback")
            
            # Create a simple ML manager interface
            class SimpleMLManager:
                def __init__(self):
                    self.pattern_engine = pattern_ml_engine
                    self.engine_status = {'pattern_ml': pattern_ml_engine.is_trained}
                
                def get_system_summary(self):
                    return {
                        'engine_status': self.engine_status,
                        'pattern_ml_trained': pattern_ml_engine.is_trained
                    }
            
            _ml_manager = SimpleMLManager()
            logger.info("✅ Simple ML manager loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load unified ML manager: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            _ml_manager = None
    
    return _ml_manager

def predict_probability(features: Dict[str, float], pattern_type: str) -> float:
    """
    Predict pattern success probability using the unified ML system.
    
    Args:
        features: Pattern features (duration, volume_ratio, trend_alignment, completion)
        pattern_type: Type of pattern (e.g., 'head_shoulders', 'triple_tops')
    
    Returns:
        float: Success probability (0.0 to 1.0)
    """
    try:
        # Get ML manager
        ml_manager = get_ml_manager()
        if ml_manager is None:
            logger.warning("ML manager not available, using fallback")
            return 0.5
        
        # Check if pattern ML is trained
        if not ml_manager.engine_status.get('pattern_ml', False):
            logger.info("Pattern ML not trained, attempting to load model...")
            # The SimpleMLManager doesn't have train_all_engines method
            # Instead, we should try to load the model if available
            pattern_engine = ml_manager.pattern_engine
            if pattern_engine:
                model_path = os.path.join(os.path.dirname(__file__), 'quant_system', 'models', 'pattern_catboost.joblib')
                if os.path.exists(model_path):
                    pattern_engine.load_model(model_path)
                    ml_manager.engine_status['pattern_ml'] = pattern_engine.is_trained
                    logger.info(f"✅ Pattern ML model loaded: {pattern_engine.is_trained}")
                else:
                    logger.warning("No trained model found, will use fallback")
        
        # Get pattern ML engine
        pattern_engine = ml_manager.pattern_engine
        
        # Handle fallback case where pattern_engine is None
        if pattern_engine is None:
            logger.warning("Pattern engine not available, using fallback")
            return 0.5
        
        if pattern_engine.is_trained:
            # Make prediction using trained model
            probability = pattern_engine.predict(features, pattern_type)
            logger.info(f"✅ ML prediction for {pattern_type}: {probability:.3f}")
            return probability
        else:
            logger.warning("Pattern ML model not trained, using fallback")
            return 0.5
            
    except Exception as e:
        logger.error(f"❌ ML prediction failed: {e}")
        return 0.5

def get_model_version() -> str:
    """Get the current ML model version."""
    return _model_version

def get_ml_system_status() -> Dict[str, Any]:
    """Get the status of the ML system."""
    try:
        ml_manager = get_ml_manager()
        if ml_manager:
            return ml_manager.get_system_summary()
        else:
            return {"status": "ml_manager_not_available"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def get_pattern_prediction_breakdown(features: Dict[str, float], pattern_type: str) -> Dict[str, Any]:
    """
    Get detailed pattern prediction breakdown including confidence and risk metrics.
    
    Args:
        features: Pattern features
        pattern_type: Type of pattern
    
    Returns:
        Dict containing prediction details
    """
    try:
        probability = predict_probability(features, pattern_type)
        
        # Calculate confidence levels
        if probability >= 0.8:
            confidence = "very_high"
            strength = "strong"
        elif probability >= 0.6:
            confidence = "high"
            strength = "medium"
        elif probability >= 0.4:
            confidence = "medium"
            strength = "weak"
        else:
            confidence = "low"
            strength = "very_weak"
        
        # Risk assessment
        risk_level = "low" if probability >= 0.7 else "medium" if probability >= 0.5 else "high"
        
        return {
            "probability": probability,
            "confidence": confidence,
            "strength": strength,
            "risk_level": risk_level,
            "pattern_type": pattern_type,
            "features": features,
            "model_version": _model_version,
            "prediction_source": "unified_ml_system"
        }
        
    except Exception as e:
        logger.error(f"❌ Pattern breakdown failed: {e}")
        return {
            "probability": 0.5,
            "confidence": "unknown",
            "strength": "unknown",
            "risk_level": "unknown",
            "pattern_type": pattern_type,
            "features": features,
            "model_version": _model_version,
            "prediction_source": "fallback",
            "error": str(e)
        }
