#!/usr/bin/env python3
"""
ML Inference Module

This module provides ML inference capabilities for pattern recognition
and prediction using the existing CatBoost infrastructure.
"""

import logging
import os
from typing import Dict, Any, Optional
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

logger = logging.getLogger(__name__)

def get_model_version() -> str:
    """
    Get the current ML model version.
    
    Returns:
        str: Model version string
    """
    try:
        # Try to load from the existing pattern registry
        from ml.quant_system.engines.pattern_ml import PatternMLEngine
        from ml.quant_system.core import UnifiedMLConfig
        
        # Initialize the engine to get model info
        engine = PatternMLEngine(UnifiedMLConfig())
        
        # Check if registry file exists
        if os.path.exists(engine.registry_path):
            import json
            try:
                with open(engine.registry_path, 'r') as f:
                    registry = json.load(f)
                return registry.get('model_version', '1.0.0')
            except Exception:
                pass
        
        return "1.0.0"  # Default version
    except Exception as e:
        logger.warning(f"Could not determine model version: {e}")
        return "1.0.0"

def predict_probability(features: Dict[str, float], pattern_type: Optional[str] = None) -> float:
    """
    Predict the probability of pattern success using ML model.
    
    Args:
        features: Dictionary of feature values
        pattern_type: Optional pattern type for specialized prediction
    
    Returns:
        float: Probability score between 0.0 and 1.0
    """
    try:
        # Try to use the existing CatBoost pattern ML engine
        from ml.quant_system.engines.pattern_ml import PatternMLEngine
        from ml.quant_system.core import UnifiedMLConfig
        
        # Initialize the engine
        config = UnifiedMLConfig()
        engine = PatternMLEngine(config)
        
        # Check if model file exists
        if not os.path.exists(engine.model_path):
            logger.debug("ML model not available, using heuristic scoring")
            return _heuristic_prediction(features, pattern_type)
        
        # Load the model
        import joblib
        try:
            model = joblib.load(engine.model_path)
            
            # Prepare features for prediction
            import pandas as pd
            import numpy as np
            
            # Ensure all required features are present
            required_features = ['duration', 'volume_ratio', 'trend_alignment', 'completion']
            feature_values = []
            
            for feature_name in required_features:
                value = features.get(feature_name, 0.0)
                # Ensure numeric and handle edge cases
                if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                    feature_values.append(float(value))
                else:
                    feature_values.append(0.0)
            
            # Add pattern type as categorical feature (if model supports it)
            if pattern_type and hasattr(model, 'get_feature_names_in'):
                try:
                    # Try to include pattern type if model expects it
                    feature_values.append(pattern_type)
                except:
                    # If pattern type isn't supported, just use the numeric features
                    pass
            
            # Create prediction array
            X_pred = np.array(feature_values[:4]).reshape(1, -1)  # Use only numeric features
            
            # Get prediction probability
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_pred)[0]
                # Return probability of positive class (success)
                return float(proba[1]) if len(proba) > 1 else float(proba[0])
            elif hasattr(model, 'predict'):
                # For models without predict_proba, use predict and convert to probability
                prediction = model.predict(X_pred)[0]
                # Convert prediction to probability (assuming 0-1 range or classification)
                return float(max(0.0, min(1.0, prediction)))
            else:
                logger.warning("Model doesn't support prediction, using heuristic")
                return _heuristic_prediction(features, pattern_type)
                
        except Exception as e:
            logger.warning(f"Error loading or using ML model: {e}")
            return _heuristic_prediction(features, pattern_type)
            
    except ImportError as e:
        logger.debug(f"ML dependencies not available: {e}")
        return _heuristic_prediction(features, pattern_type)
    except Exception as e:
        logger.warning(f"ML prediction failed: {e}")
        return _heuristic_prediction(features, pattern_type)

def get_pattern_prediction_breakdown(features: Dict[str, float], pattern_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Get detailed breakdown of pattern prediction.
    
    Args:
        features: Dictionary of feature values
        pattern_type: Optional pattern type
    
    Returns:
        Dict containing prediction breakdown and analysis
    """
    try:
        # Get base probability
        probability = predict_probability(features, pattern_type)
        
        # Create detailed breakdown
        breakdown = {
            'model_version': get_model_version(),
            'probability': probability,
            'features_analyzed': features.copy(),
            'pattern_type': pattern_type or 'unknown',
            'confidence': _get_confidence_level(probability, features),
            'strength': _get_pattern_strength(probability, features),
            'risk_level': _get_risk_level(probability, features),
            'contributing_factors': _analyze_contributing_factors(features),
            'reliability_score': _calculate_reliability_score(probability, features)
        }
        
        return breakdown
        
    except Exception as e:
        logger.error(f"Error generating prediction breakdown: {e}")
        return {
            'model_version': get_model_version(),
            'probability': 0.5,
            'features_analyzed': features.copy(),
            'pattern_type': pattern_type or 'unknown',
            'confidence': 'low',
            'strength': 'weak',
            'risk_level': 'high',
            'contributing_factors': {},
            'reliability_score': 0.5,
            'error': str(e)
        }

def _heuristic_prediction(features: Dict[str, float], pattern_type: Optional[str] = None) -> float:
    """
    Fallback heuristic prediction when ML model is not available.
    
    Args:
        features: Dictionary of feature values
        pattern_type: Optional pattern type
    
    Returns:
        float: Probability score between 0.0 and 1.0
    """
    try:
        # Base score from completion
        completion = float(features.get('completion', 0.0))
        base_score = max(0.0, min(1.0, completion))
        
        # Adjust based on volume ratio
        volume_ratio = float(features.get('volume_ratio', 1.0))
        if volume_ratio > 1.5:  # High volume confirmation
            base_score += 0.1
        elif volume_ratio < 0.8:  # Low volume warning
            base_score -= 0.1
        
        # Adjust based on trend alignment
        trend_alignment = float(features.get('trend_alignment', 0.0))
        base_score += trend_alignment * 0.2  # Up to 20% bonus for good alignment
        
        # Adjust based on duration
        duration = float(features.get('duration', 5.0))
        if duration > 10:  # Longer patterns may be more reliable
            base_score += 0.05
        elif duration < 3:  # Very short patterns less reliable
            base_score -= 0.1
        
        # Pattern-specific adjustments
        if pattern_type:
            pattern_multipliers = {
                'head_and_shoulders': 0.75,
                'inverse_head_and_shoulders': 0.75,
                'double_top': 0.70,
                'double_bottom': 0.70,
                'triple_top': 0.75,
                'triple_bottom': 0.75,
                'cup_and_handle': 0.80,
                'flag': 0.65,
                'pennant': 0.65,
                'triangle': 0.60,
                'wedge': 0.55,
                'channel': 0.50
            }
            multiplier = pattern_multipliers.get(pattern_type.lower(), 0.60)
            base_score *= multiplier
        else:
            base_score *= 0.60  # Default multiplier for unknown patterns
        
        # Ensure score is within bounds
        final_score = max(0.0, min(1.0, base_score))
        
        logger.debug(f"Heuristic prediction for {pattern_type}: {final_score:.3f}")
        return final_score
        
    except Exception as e:
        logger.error(f"Error in heuristic prediction: {e}")
        return 0.5  # Neutral score

def _get_confidence_level(probability: float, features: Dict[str, float]) -> str:
    """Get confidence level based on probability and features."""
    try:
        completion = float(features.get('completion', 0.0))
        volume_ratio = float(features.get('volume_ratio', 1.0))
        
        if probability >= 0.8 and completion >= 0.8 and volume_ratio >= 1.2:
            return 'very_high'
        elif probability >= 0.7 and completion >= 0.7:
            return 'high'
        elif probability >= 0.5:
            return 'medium'
        else:
            return 'low'
    except:
        return 'medium'

def _get_pattern_strength(probability: float, features: Dict[str, float]) -> str:
    """Get pattern strength based on probability and features."""
    try:
        if probability >= 0.8:
            return 'strong'
        elif probability >= 0.6:
            return 'medium'
        else:
            return 'weak'
    except:
        return 'weak'

def _get_risk_level(probability: float, features: Dict[str, float]) -> str:
    """Get risk level based on probability and features."""
    try:
        if probability >= 0.8:
            return 'low'
        elif probability >= 0.6:
            return 'medium'
        else:
            return 'high'
    except:
        return 'medium'

def _analyze_contributing_factors(features: Dict[str, float]) -> Dict[str, str]:
    """Analyze which features contribute most to the prediction."""
    try:
        factors = {}
        
        completion = float(features.get('completion', 0.0))
        if completion >= 0.8:
            factors['completion'] = 'positive'
        elif completion <= 0.3:
            factors['completion'] = 'negative'
        else:
            factors['completion'] = 'neutral'
        
        volume_ratio = float(features.get('volume_ratio', 1.0))
        if volume_ratio >= 1.5:
            factors['volume'] = 'positive'
        elif volume_ratio <= 0.7:
            factors['volume'] = 'negative'
        else:
            factors['volume'] = 'neutral'
        
        trend_alignment = float(features.get('trend_alignment', 0.0))
        if trend_alignment >= 0.7:
            factors['trend'] = 'positive'
        elif trend_alignment <= -0.3:
            factors['trend'] = 'negative'
        else:
            factors['trend'] = 'neutral'
        
        duration = float(features.get('duration', 5.0))
        if duration >= 10:
            factors['duration'] = 'positive'
        elif duration <= 2:
            factors['duration'] = 'negative'
        else:
            factors['duration'] = 'neutral'
        
        return factors
    except:
        return {}

def _calculate_reliability_score(probability: float, features: Dict[str, float]) -> float:
    """Calculate overall reliability score."""
    try:
        base_reliability = probability
        
        # Adjust based on feature quality
        completion = float(features.get('completion', 0.0))
        volume_ratio = float(features.get('volume_ratio', 1.0))
        trend_alignment = float(features.get('trend_alignment', 0.0))
        
        # Higher completion increases reliability
        if completion >= 0.8:
            base_reliability += 0.05
        
        # Good volume confirmation increases reliability
        if volume_ratio >= 1.3:
            base_reliability += 0.05
        
        # Good trend alignment increases reliability
        if abs(trend_alignment) >= 0.6:
            base_reliability += 0.05
        
        return max(0.0, min(1.0, base_reliability))
    except:
        return probability

# Initialize logging
logger.info(f"ML Inference module initialized. Model version: {get_model_version()}")
