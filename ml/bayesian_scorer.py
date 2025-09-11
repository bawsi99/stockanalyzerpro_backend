#!/usr/bin/env python3
"""
ml/bayesian_scorer.py

Bayesian Pattern Scorer for ML-based pattern analysis.
This module provides comprehensive Bayesian probability scoring for stock pattern analysis,
including both ML-based models and heuristic fallback methods.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add the backend directory to Python path for imports
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import numpy as np

# Optional ML dependencies
try:
    from sklearn.naive_bayes import GaussianNB
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Optional pattern database
try:
    from pattern_database import PatternDatabase, PatternRecord
    HAS_PATTERN_DB = True
except ImportError:
    HAS_PATTERN_DB = False
    PatternDatabase = None
    PatternRecord = None

logger = logging.getLogger(__name__)


class BayesianPatternScorer:
    """
    Comprehensive Bayesian Pattern Scorer for ML-based pattern analysis.
    
    This class provides Bayesian probability scoring using both ML models (when available)
    and intelligent heuristic fallback methods. It combines:
    - Gaussian Naive Bayes models trained on historical pattern data
    - Heuristic probability calculation based on pattern features
    - Graceful degradation when dependencies are missing
    """
    
    def __init__(self):
        """Initialize the Bayesian Pattern Scorer."""
        self.db = None
        self.models: Dict[str, Tuple] = {}
        
        # Initialize pattern database if available
        if HAS_PATTERN_DB and PatternDatabase:
            try:
                self.db = PatternDatabase()
                logger.info("Successfully initialized with pattern database")
            except Exception as e:
                logger.warning(f"Could not initialize pattern database: {e}")
                self.db = None
        
        if not HAS_SKLEARN:
            logger.warning("scikit-learn not available; using heuristic methods only")
        
        if not HAS_PATTERN_DB:
            logger.info("Pattern database not available; using heuristic methods only")
    
    def predict_probability(self, pattern_type: str, features: Dict[str, float]) -> float:
        """
        Predict the success probability of a pattern using Bayesian methods.
        
        Args:
            pattern_type: Type of pattern (e.g., 'triangle', 'flag', 'double_top')
            features: Dictionary of pattern features for prediction
            
        Returns:
            Probability score between 0.0 and 1.0
        """
        try:
            # Try ML-based prediction first (if dependencies available)
            if HAS_SKLEARN and self.db:
                ml_probability = self._predict_with_ml(pattern_type, features)
                if ml_probability is not None:
                    return ml_probability
            
            # Fallback to heuristic probability calculation
            return self._calculate_fallback_probability(pattern_type, features)
            
        except Exception as e:
            logger.warning(f"Probability prediction failed for {pattern_type}: {e}")
            return 0.5
    
    def _predict_with_ml(self, pattern_type: str, features: Dict[str, float]) -> Optional[float]:
        """
        Predict probability using ML models (Gaussian Naive Bayes).
        
        Returns None if ML prediction is not possible.
        """
        try:
            if not HAS_SKLEARN or not self.db:
                return None
                
            # Check if model exists for this pattern type
            if pattern_type not in self.models:
                # Try to train the model
                if not self._train_pattern_model(pattern_type):
                    return None
            
            model, feature_keys = self.models[pattern_type]
            
            # Prepare features for prediction
            x = np.array([[float(features.get(k, 0.0)) for k in feature_keys]], dtype=float)
            proba = model.predict_proba(x)[0]
            
            # Return probability of success (class 1)
            return float(proba[1]) if len(proba) > 1 else 0.5
            
        except Exception as e:
            logger.debug(f"ML prediction failed for {pattern_type}: {e}")
            return None
    
    def _train_pattern_model(self, pattern_type: str) -> bool:
        """
        Train a Gaussian Naive Bayes model for a specific pattern type.
        
        Returns True if training was successful, False otherwise.
        """
        try:
            if not HAS_SKLEARN or not self.db:
                return False
                
            # Get historical data for this pattern type
            historical = self.db.get_historical(pattern_type)
            if len(historical) < 20:  # Need minimum data for training
                logger.debug(f"Insufficient historical data for {pattern_type}: {len(historical)} < 20")
                return False
            
            # Prepare dataset
            dataset_result = self._prepare_dataset(historical)
            if dataset_result is None:
                return False
                
            X, y, feature_keys = dataset_result
            
            # Train Gaussian Naive Bayes model
            model = GaussianNB()
            model.fit(X, y)
            
            # Store the trained model
            self.models[pattern_type] = (model, feature_keys)
            logger.info(f"Successfully trained ML model for {pattern_type} with {len(X)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Model training failed for {pattern_type}: {e}")
            return False
    
    def _prepare_dataset(self, historical_records) -> Optional[Tuple[np.ndarray, np.ndarray, list]]:
        """
        Prepare training dataset from historical records.
        
        Returns (X, y, feature_keys) or None if dataset preparation fails.
        """
        try:
            if not historical_records:
                return None
                
            # Establish feature schema from first record
            feature_keys = sorted(historical_records[0].features.keys())
            X = []
            y = []
            
            # Process each historical record
            for record in historical_records:
                try:
                    # Extract features in consistent order
                    feature_vector = [float(record.features.get(k, 0.0)) for k in feature_keys]
                    X.append(feature_vector)
                    
                    # Extract outcome (success/failure)
                    y.append(1 if record.outcome else 0)
                    
                except Exception:
                    # Skip records with invalid data
                    continue
            
            # Validate dataset
            if not X or len(set(y)) < 2:  # Need both success and failure examples
                return None
                
            return np.array(X, dtype=float), np.array(y, dtype=int), feature_keys
            
        except Exception as e:
            logger.warning(f"Dataset preparation failed: {e}")
            return None
    
    def train(self, pattern_type: str) -> bool:
        """
        Train the Bayesian model for a specific pattern type.
        
        Args:
            pattern_type: Type of pattern to train
            
        Returns:
            True if training was successful, False otherwise
        """
        return self._train_pattern_model(pattern_type)
    
    def _calculate_fallback_probability(self, pattern_type: str, features: Dict[str, float]) -> float:
        """
        Calculate pattern probability using fallback heuristic methods.
        
        This method is used when the risk-based scorer is not available.
        """
        try:
            # Base probabilities for different pattern types
            base_probabilities = {
                'triangle': 0.62,
                'flag': 0.68,
                'double_top': 0.55,
                'double_bottom': 0.58,
                'head_and_shoulders': 0.72,
                'inverse_head_and_shoulders': 0.71,
                'cup_and_handle': 0.75,
                'pennant': 0.64,
                'wedge': 0.59,
                'channel': 0.61,
                'support': 0.65,
                'resistance': 0.63
            }
            
            base_prob = base_probabilities.get(pattern_type, 0.50)
            
            # Apply feature-based adjustments
            adjustments = 0.0
            
            # Volume ratio adjustment
            volume_ratio = features.get('volume_ratio', 1.0)
            if volume_ratio > 1.2:
                adjustments += min(0.15, (volume_ratio - 1.0) * 0.1)
            elif volume_ratio < 0.8:
                adjustments -= min(0.15, (1.0 - volume_ratio) * 0.1)
            
            # Duration adjustment
            duration = features.get('duration', 0)
            if duration > 20:
                adjustments += min(0.1, duration / 200)
            elif duration < 5:
                adjustments -= 0.1
            
            # Confidence adjustment
            confidence = features.get('confidence', 0.5)
            adjustments += (confidence - 0.5) * 0.2
            
            # Calculate final probability
            final_prob = base_prob + adjustments
            
            # Ensure probability is within bounds
            return max(0.1, min(0.9, final_prob))
            
        except Exception as e:
            logger.warning(f"Fallback probability calculation failed: {e}")
            return 0.5
    
    def is_available(self) -> bool:
        """Check if the Bayesian scorer is properly initialized and available."""
        return True  # Always available with fallback methods
    
    def has_ml_capabilities(self) -> bool:
        """Check if ML-based prediction capabilities are available."""
        return HAS_SKLEARN and HAS_PATTERN_DB and self.db is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model state."""
        info = {
            'sklearn_available': HAS_SKLEARN,
            'pattern_db_available': HAS_PATTERN_DB,
            'db_initialized': self.db is not None,
            'ml_mode': self.has_ml_capabilities(),
            'fallback_mode': not self.has_ml_capabilities(),
            'trained_patterns': list(self.models.keys()),
            'model_count': len(self.models)
        }
        
        return info


# For compatibility, also make the class available at module level
__all__ = ['BayesianPatternScorer']
