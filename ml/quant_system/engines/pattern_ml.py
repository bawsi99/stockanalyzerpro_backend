"""
Pattern-Based ML Module

This module provides CatBoost-based pattern success modeling capabilities.
Adapted from backend/ml/ for unified integration.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Any
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False
    logging.warning("CatBoost not available. Pattern ML will not work.")

from ..core import BaseMLEngine, UnifiedMLConfig

logger = logging.getLogger(__name__)

@dataclass
class PatternRecord:
    """Pattern record for training."""
    features: Dict[str, float]
    outcome: bool
    confirmed: bool = False
    timestamp: Optional[datetime] = None
    pattern_type: Optional[str] = None

@dataclass
class TrainReport:
    """Training report for pattern models."""
    model_path: str
    trained_at: str
    metrics: Dict[str, float]
    feature_schema: Dict[str, str]

class PatternDataset:
    """Dataset management for pattern-based ML."""
    
    CORE_SCHEMA: List[str] = [
        "duration",
        "volume_ratio", 
        "trend_alignment",
        "completion",
    ]
    
    def __init__(self):
        self.patterns: Dict[str, List[PatternRecord]] = {}
    
    def add_pattern(self, pattern_type: str, record: PatternRecord):
        """Add a pattern record."""
        if pattern_type not in self.patterns:
            self.patterns[pattern_type] = []
        self.patterns[pattern_type].append(record)
    
    def build_training_dataset(self) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """Build training dataset from pattern records."""
        rows: List[Dict] = []
        
        for pattern_type, records in self.patterns.items():
            for r in records:
                if not getattr(r, "confirmed", False):
                    continue
                try:
                    feat = {k: float(r.features.get(k, 0.0)) for k in self.CORE_SCHEMA}
                    rows.append({
                        **feat,
                        "y": 1 if bool(getattr(r, "outcome", False)) else 0,
                        "pattern_type": pattern_type,
                        "timestamp": getattr(r, "timestamp", None),
                    })
                except Exception as e:
                    logger.warning(f"Failed to process pattern record: {e}")
                    continue
        
        if not rows:
            return pd.DataFrame(columns=self.CORE_SCHEMA), np.array([], dtype=int), pd.DataFrame()
        
        df = pd.DataFrame(rows)
        
        # Sort by timestamp when available
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        
        X = df[self.CORE_SCHEMA + ["pattern_type"]].copy()
        y = df["y"].astype(int).to_numpy()
        meta = df[["pattern_type", "timestamp"]].copy()
        
        return X, y, meta
    
    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute class weights for imbalanced datasets."""
        if y.size == 0:
            return {0: 1.0, 1: 1.0}
        
        pos = float((y == 1).sum())
        neg = float((y == 0).sum())
        
        if pos == 0 or neg == 0:
            return {0: 1.0, 1: 1.0}
        
        total = pos + neg
        return {
            0: total / (2.0 * neg),
            1: total / (2.0 * pos),
        }

class PatternMLEngine(BaseMLEngine):
    """Pattern-based ML engine using CatBoost."""
    
    def __init__(self, config: UnifiedMLConfig = None):
        super().__init__(config)
        self.model = None
        self.dataset = PatternDataset()
        self.models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        self.registry_path = os.path.join(self.models_dir, "pattern_registry.json")
        self.model_path = os.path.join(self.models_dir, "pattern_catboost.joblib")
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
    
    def add_pattern_data(self, pattern_type: str, features: Dict[str, float], 
                        outcome: bool, confirmed: bool = True, timestamp: datetime = None):
        """Add pattern data for training."""
        record = PatternRecord(
            features=features,
            outcome=outcome,
            confirmed=confirmed,
            timestamp=timestamp,
            pattern_type=pattern_type
        )
        self.dataset.add_pattern(pattern_type, record)
        logger.info(f"Added pattern data: {pattern_type} -> {outcome}")
    
    def train(self, data: Optional[Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]] = None, 
              n_splits: int = 5) -> bool:
        """Train the pattern-based ML model."""
        if not HAS_CATBOOST:
            logger.error("CatBoost not available")
            return False
        
        # Build dataset
        if data is None:
            X, y, meta = self.dataset.build_training_dataset()
        else:
            X, y, meta = data
        
        # Check for empty dataset
        if X is None or X.empty or y is None or y.size == 0:
            logger.warning("No training data available")
            return False
            
        # Check if we have enough samples for training
        if len(y) < 10:
            logger.warning(f"Insufficient data for pattern ML training: {len(y)} samples (minimum 10 required)")
            return False
        
        try:
            # Prepare features
            columns = list(X.columns)
            cat_idx = self._get_categorical_indices(columns)
            
            # Time series cross-validation - ensure we have enough data for splits
            min_samples_per_split = 5
            max_possible_splits = max(2, len(y) // min_samples_per_split)
            actual_splits = min(n_splits, max_possible_splits)
            
            tscv = TimeSeriesSplit(n_splits=actual_splits)
            
            # Base CatBoost model
            clf = CatBoostClassifier(
                loss_function="Logloss",
                depth=self.config.catboost_depth,
                iterations=self.config.catboost_iterations,
                learning_rate=self.config.catboost_learning_rate,
                l2_leaf_reg=3.0,
                early_stopping_rounds=100,
                eval_metric="Logloss",
                verbose=False,
            )
            
            # Class weights
            class_weights = self.dataset.compute_class_weights(y)
            sample_weights = np.where(y == 1, class_weights[1], class_weights[0]).astype(float)
            
            # Check for class imbalance
            pos_count = np.sum(y == 1)
            neg_count = np.sum(y == 0)
            
            if pos_count == 0 or neg_count == 0:
                logger.warning(f"Imbalanced dataset: {pos_count} positive samples, {neg_count} negative samples. Need both classes for training.")
                return False
                
            # Fit model
            clf.fit(X, y, sample_weight=sample_weights, cat_features=cat_idx, verbose=False)
            
            # Generate out-of-fold probabilities for calibration
            oof_pred = np.zeros_like(y, dtype=float)
            
            # Only do cross-validation if we have enough data
            if len(y) >= 20:
                for train_idx, test_idx in tscv.split(X):
                    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                    y_tr = y[train_idx]
                    sw_tr = sample_weights[train_idx]
                    
                    # Check if we have both classes in the training set
                    if len(np.unique(y_tr)) < 2:
                        logger.warning(f"Skipping CV fold - only one class present in fold")
                        oof_pred[test_idx] = 0.5  # Default prediction
                        continue
                    
                    m = CatBoostClassifier(
                        loss_function="Logloss",
                        depth=self.config.catboost_depth,
                        iterations=int(self.config.catboost_iterations * 0.8),
                        learning_rate=self.config.catboost_learning_rate,
                        l2_leaf_reg=3.0,
                        early_stopping_rounds=100,
                        eval_metric="Logloss",
                        verbose=False,
                    )
                    m.fit(X_tr, y_tr, sample_weight=sw_tr, cat_features=cat_idx, verbose=False)
                    p = m.predict_proba(X_te)[:, 1]
                    oof_pred[test_idx] = p
            else:
                # For small datasets, use the same predictions
                oof_pred = clf.predict_proba(X)[:, 1]
            
            # Calibration - only if we have enough data
            if len(y) >= 10:
                cal = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
                cal.fit(X, y)
            else:
                # For very small datasets, skip calibration
                cal = clf
            
            # Calculate metrics
            eps = 1e-12
            metrics = {
                "brier": float(brier_score_loss(y, np.clip(oof_pred, eps, 1 - eps))),
                "logloss": float(log_loss(y, np.clip(oof_pred, eps, 1 - eps))),
                "n_samples": int(len(y)),
            }
            
            # Save model
            joblib.dump(cal, self.model_path)
            
            # Create feature schema
            feature_schema = {k: ("categorical" if k == "pattern_type" else "numeric") for k in columns}
            
            # Save registry
            registry = {
                "model_path": self.model_path,
                "trained_at": datetime.utcnow().isoformat(),
                "metrics": metrics,
                "feature_schema": feature_schema,
            }
            
            with open(self.registry_path, "w") as f:
                json.dump(registry, f, indent=2)
            
            # Store model
            self.model = cal
            self.is_trained = True
            
            # Register in global registry
            self.registry.register_model(
                "pattern_ml", 
                cal, 
                training_info={
                    "metrics": metrics,
                    "feature_schema": feature_schema,
                    "trained_at": registry["trained_at"]
                }
            )
            
            logger.info(f"Pattern ML model trained successfully: {metrics}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train pattern ML model: {e}")
            return False
    
    def predict(self, features: Dict[str, float], pattern_type: Optional[str] = None) -> float:
        """Predict pattern success probability."""
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained")
            return 0.5
        
        try:
            # Check if features is empty
            if not features:
                logger.warning("Empty features provided for prediction")
                return 0.5
                
            # Prepare features
            row = {k: float(features.get(k, 0.0)) for k in self.dataset.CORE_SCHEMA}
            if pattern_type is not None:
                row["pattern_type"] = str(pattern_type)
            else:
                row["pattern_type"] = "unknown"
            
            X = pd.DataFrame([row])
            
            # Check if model has predict_proba method (CatBoost or calibrated classifier)
            if hasattr(self.model, 'predict_proba'):
                # Check if model has classes_ attribute (sklearn models)
                if hasattr(self.model, 'classes_') and len(self.model.classes_) > 1:
                    proba = self.model.predict_proba(X)[:, 1][0]
                else:
                    # Default to raw prediction if classes not available
                    raw_pred = self.model.predict(X)[0]
                    proba = float(raw_pred)
            else:
                # Fallback for models without predict_proba
                raw_pred = self.model.predict(X)[0]
                proba = 1.0 if raw_pred > 0.5 else 0.0
                
            return float(max(0.0, min(1.0, float(proba))))
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.5
    
    def evaluate(self, data: Optional[Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]] = None) -> Dict[str, Any]:
        """Evaluate model performance."""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        # Use provided data or build from dataset
        if data is None:
            X, y, meta = self.dataset.build_training_dataset()
        else:
            X, y, meta = data
        
        if X.empty or y.size == 0:
            return {"error": "No evaluation data"}
        
        try:
            # Make predictions
            y_pred = self.model.predict(X)
            y_proba = self.model.predict_proba(X)[:, 1]
            
            # Calculate metrics
            metrics = {
                "accuracy": float(np.mean(y_pred == y)),
                "brier": float(brier_score_loss(y, y_proba)),
                "logloss": float(log_loss(y, y_proba)),
                "n_samples": int(len(y)),
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"error": str(e)}
    
    def save_model(self, path: str) -> bool:
        """Save the trained model."""
        if not self.is_trained or self.model is None:
            return False
        
        try:
            joblib.dump(self.model, path)
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Load a trained model."""
        try:
            self.model = joblib.load(path)
            self.is_trained = True
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _get_categorical_indices(self, columns: List[str]) -> List[int]:
        """Get indices of categorical features."""
        idx = []
        for i, c in enumerate(columns):
            if c == "pattern_type":
                idx.append(i)
        return idx
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        try:
            with open(self.registry_path, "r") as f:
                registry = json.load(f)
            
            return {
                "status": "trained",
                "trained_at": registry.get("trained_at"),
                "metrics": registry.get("metrics"),
                "feature_schema": registry.get("feature_schema"),
                "model_path": registry.get("model_path"),
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"status": "error", "error": str(e)}

# Global instance
# Global instance removed - instantiate locally as needed
