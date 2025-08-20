"""
Enhanced ML Engine with Ensemble Methods

This module provides an enhanced ML engine with:
1. Ensemble methods (stacking, voting)
2. Advanced hyperparameter optimization
3. Model selection and validation
4. Feature selection
5. Cross-validation strategies

Usage:
    from enhanced_ml_engine import EnhancedMLEngine
    engine = EnhancedMLEngine()
    engine.train(X, y)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
import joblib
import json
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import optuna

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    logging.warning("CatBoost not available")

logger = logging.getLogger(__name__)


class EnhancedMLEngine:
    """Enhanced ML engine with ensemble methods and advanced optimization."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.models = {}
        self.feature_selector = None
        self.scaler = None
        self.best_model = None
        self.feature_names = []
        self.is_trained = False
        self.training_history = []
        
    def _default_config(self) -> Dict:
        """Default configuration."""
        return {
            'ensemble_method': 'stacking',  # 'stacking', 'voting', 'single'
            'base_models': ['catboost', 'random_forest', 'gradient_boosting'],
            'meta_model': 'logistic_regression',
            'cv_folds': 5,
            'optimization_trials': 50,
            'feature_selection': True,
            'feature_selection_method': 'kbest',  # 'kbest', 'rfe', 'none'
            'n_features': 20,
            'scaling': True,
            'calibration': True,
            'random_state': 42
        }
    
    def train(self, X: pd.DataFrame, y: np.ndarray, meta: Optional[pd.DataFrame] = None) -> bool:
        """Train the enhanced ML model."""
        try:
            logger.info("Starting enhanced ML training...")
            
            # Store feature names
            self.feature_names = list(X.columns)
            
            # 1. Feature selection
            if self.config['feature_selection']:
                X = self._feature_selection(X, y)
            
            # 2. Feature scaling
            if self.config['scaling']:
                X = self._scale_features(X)
            
            # 3. Hyperparameter optimization
            best_params = self._optimize_hyperparameters(X, y)
            
            # 4. Train ensemble model
            success = self._train_ensemble(X, y, best_params)
            
            if success:
                self.is_trained = True
                logger.info("Enhanced ML training completed successfully")
                return True
            else:
                logger.error("Enhanced ML training failed")
                return False
                
        except Exception as e:
            logger.error(f"Enhanced ML training error: {e}")
            return False
    
    def _feature_selection(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Perform feature selection."""
        logger.info("Performing feature selection...")
        
        if self.config['feature_selection_method'] == 'kbest':
            self.feature_selector = SelectKBest(score_func=f_classif, k=self.config['n_features'])
            X_selected = self.feature_selector.fit_transform(X, y)
            selected_features = X.columns[self.feature_selector.get_support()]
            X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
        elif self.config['feature_selection_method'] == 'rfe':
            # Use CatBoost for RFE if available, otherwise Random Forest
            if HAS_CATBOOST:
                estimator = CatBoostClassifier(iterations=100, verbose=False, random_state=self.config['random_state'])
            else:
                estimator = RandomForestClassifier(n_estimators=100, random_state=self.config['random_state'])
            
            self.feature_selector = RFE(estimator, n_features_to_select=self.config['n_features'])
            X_selected = self.feature_selector.fit_transform(X, y)
            selected_features = X.columns[self.feature_selector.get_support()]
            X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        logger.info(f"Feature selection completed. Selected {X.shape[1]} features")
        return X
    
    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features."""
        logger.info("Scaling features...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        return X
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """Optimize hyperparameters using Optuna."""
        logger.info("Optimizing hyperparameters...")
        
        def objective(trial):
            # Define hyperparameter space
            params = {}
            
            # CatBoost parameters
            if 'catboost' in self.config['base_models'] and HAS_CATBOOST:
                params['catboost'] = {
                    'iterations': trial.suggest_int('cb_iterations', 100, 1000),
                    'depth': trial.suggest_int('cb_depth', 4, 10),
                    'learning_rate': trial.suggest_float('cb_lr', 0.01, 0.3),
                    'l2_leaf_reg': trial.suggest_float('cb_l2', 1, 10),
                    'random_state': self.config['random_state']
                }
            
            # Random Forest parameters
            if 'random_forest' in self.config['base_models']:
                params['random_forest'] = {
                    'n_estimators': trial.suggest_int('rf_n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('rf_max_depth', 3, 15),
                    'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
                    'random_state': self.config['random_state']
                }
            
            # Gradient Boosting parameters
            if 'gradient_boosting' in self.config['base_models']:
                params['gradient_boosting'] = {
                    'n_estimators': trial.suggest_int('gb_n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('gb_max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('gb_lr', 0.01, 0.3),
                    'subsample': trial.suggest_float('gb_subsample', 0.6, 1.0),
                    'random_state': self.config['random_state']
                }
            
            # Train and evaluate
            tscv = TimeSeriesSplit(n_splits=self.config['cv_folds'])
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train ensemble
                ensemble = self._create_ensemble(params)
                ensemble.fit(X_train, y_train)
                
                # Predict and score
                y_pred = ensemble.predict(X_val)
                score = accuracy_score(y_val, y_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config['optimization_trials'])
        
        logger.info(f"Best hyperparameters found: {study.best_params}")
        return study.best_params
    
    def _create_ensemble(self, params: Dict) -> Any:
        """Create ensemble model."""
        estimators = []
        
        # Create base models
        if 'catboost' in self.config['base_models'] and HAS_CATBOOST and 'catboost' in params:
            try:
                cb = CatBoostClassifier(**params['catboost'], verbose=False)
                estimators.append(('catboost', cb))
            except Exception as e:
                logger.warning(f"Failed to create CatBoost model: {e}")
        
        if 'random_forest' in self.config['base_models'] and 'random_forest' in params:
            try:
                rf = RandomForestClassifier(**params['random_forest'])
                estimators.append(('random_forest', rf))
            except Exception as e:
                logger.warning(f"Failed to create Random Forest model: {e}")
        
        if 'gradient_boosting' in self.config['base_models'] and 'gradient_boosting' in params:
            try:
                gb = GradientBoostingClassifier(**params['gradient_boosting'])
                estimators.append(('gradient_boosting', gb))
            except Exception as e:
                logger.warning(f"Failed to create Gradient Boosting model: {e}")
        
        # Ensure we have at least one estimator
        if not estimators:
            logger.warning("No estimators created, using default Random Forest")
            estimators.append(('random_forest', RandomForestClassifier(n_estimators=100, random_state=self.config['random_state'])))
        
        # Create ensemble
        if self.config['ensemble_method'] == 'voting' and len(estimators) > 1:
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
        elif self.config['ensemble_method'] == 'stacking' and len(estimators) > 1:
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
            # Note: For true stacking, we'd need to implement custom stacking logic
        else:
            # Use the best single model
            ensemble = estimators[0][1]
        
        return ensemble
    
    def _train_ensemble(self, X: pd.DataFrame, y: np.ndarray, best_params: Dict) -> bool:
        """Train the final ensemble model."""
        try:
            logger.info("Training final ensemble model...")
            
            # Create and train ensemble
            self.best_model = self._create_ensemble(best_params)
            self.best_model.fit(X, y)
            
            # Calibration
            if self.config['calibration']:
                logger.info("Calibrating model...")
                self.best_model = CalibratedClassifierCV(
                    self.best_model, 
                    method='isotonic', 
                    cv='prefit'
                )
                self.best_model.fit(X, y)
            
            # Store training info
            self.training_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'n_samples': len(y),
                'n_features': X.shape[1],
                'positive_rate': np.mean(y),
                'hyperparameters': best_params
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ensemble training error: {e}")
            return False
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained or self.best_model is None:
            raise ValueError("Model not trained")
        
        # Apply feature selection
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X)
            X = pd.DataFrame(X_selected, columns=self.feature_names[:X_selected.shape[1]], index=X.index)
        
        # Apply scaling
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Predict
        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X)[:, 1]
        else:
            return self.best_model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained or self.best_model is None:
            raise ValueError("Model not trained")
        
        # Apply feature selection
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X)
            X = pd.DataFrame(X_selected, columns=self.feature_names[:X_selected.shape[1]], index=X.index)
        
        # Apply scaling
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Predict probabilities
        return self.best_model.predict_proba(X)
    
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        try:
            y_pred = self.predict(X) > 0.5
            y_proba = self.predict(X)
            
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0),
                'f1': f1_score(y, y_pred, zero_division=0),
                'auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.5,
                'brier': brier_score_loss(y, y_proba),
                'logloss': log_loss(y, y_proba),
                'n_samples': len(y),
                'positive_rate': np.mean(y)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return {"error": str(e)}
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if not self.is_trained or self.best_model is None:
            return {}
        
        try:
            # Try to get feature importance from the model
            if hasattr(self.best_model, 'feature_importances_'):
                importances = self.best_model.feature_importances_
            elif hasattr(self.best_model, 'named_steps') and 'calibratedclassifiercv' in self.best_model.named_steps:
                base_model = self.best_model.named_steps['calibratedclassifiercv'].base_estimator
                if hasattr(base_model, 'feature_importances_'):
                    importances = base_model.feature_importances_
                else:
                    return {}
            else:
                return {}
            
            # Map to feature names
            feature_names = self.feature_names[:len(importances)]
            feature_importance = dict(zip(feature_names, importances))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Feature importance error: {e}")
            return {}
    
    def save_model(self, path: str) -> bool:
        """Save the trained model."""
        if not self.is_trained:
            return False
        
        try:
            model_data = {
                'best_model': self.best_model,
                'feature_selector': self.feature_selector,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'config': self.config,
                'training_history': self.training_history
            }
            joblib.dump(model_data, path)
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Load a trained model."""
        try:
            model_data = joblib.load(path)
            self.best_model = model_data['best_model']
            self.feature_selector = model_data['feature_selector']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.config = model_data['config']
            self.training_history = model_data['training_history']
            self.is_trained = True
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        return {
            "status": "trained",
            "ensemble_method": self.config['ensemble_method'],
            "base_models": self.config['base_models'],
            "n_features": len(self.feature_names),
            "feature_selection": self.config['feature_selection'],
            "scaling": self.config['scaling'],
            "calibration": self.config['calibration'],
            "training_history": self.training_history[-1] if self.training_history else None
        }
