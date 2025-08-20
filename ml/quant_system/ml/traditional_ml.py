"""
Traditional ML Module

This module provides traditional machine learning capabilities for:
1. Price prediction models (regression)
2. Direction prediction models (classification)
3. Volatility prediction models
4. Model evaluation and validation
5. Hyperparameter optimization

Adapted from quant_system/ml_models.py for unified integration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.svm import SVR, SVC
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logging.warning("scikit-learn not available. Traditional ML models will not work.")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logging.warning("XGBoost not available. XGBoost models will not work.")

from .core import BaseMLEngine, UnifiedMLConfig

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for traditional ML models."""
    # Target variables
    price_target: str = 'close'  # Target for price prediction
    direction_target: str = 'direction'  # Target for direction prediction
    volatility_target: str = 'volatility'  # Target for volatility prediction
    
    # Feature selection
    feature_columns: List[str] = None  # Columns to use as features
    exclude_columns: List[str] = None  # Columns to exclude
    
    # Model parameters
    test_size: float = 0.2  # Test set size
    validation_size: float = 0.1  # Validation set size
    random_state: int = 42
    
    # Cross-validation
    cv_folds: int = 5
    
    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = []
        if self.exclude_columns is None:
            self.exclude_columns = ['open', 'high', 'low', 'close', 'volume']

class TargetGenerator:
    """Generate target variables for ML models."""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
    
    def generate_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate all target variables."""
        df = data.copy()
        
        # Generate direction target (1 for up, 0 for down)
        df[self.config.direction_target] = (df[self.config.price_target].shift(-1) > df[self.config.price_target]).astype(int)
        
        # Generate volatility target (next period's volatility)
        df[self.config.volatility_target] = df[self.config.price_target].pct_change().rolling(window=5).std().shift(-1)
        
        # Generate return target (next period's return)
        df['return_target'] = df[self.config.price_target].pct_change().shift(-1)
        
        # Generate price target (next period's price)
        df['price_target'] = df[self.config.price_target].shift(-1)
        
        return df
    
    def generate_multi_horizon_targets(self, data: pd.DataFrame, horizons: List[int] = None) -> pd.DataFrame:
        """Generate targets for multiple prediction horizons."""
        if horizons is None:
            horizons = [1, 3, 5, 10, 20]
        
        df = data.copy()
        
        for horizon in horizons:
            # Direction targets
            df[f'direction_{horizon}'] = (df[self.config.price_target].shift(-horizon) > df[self.config.price_target]).astype(int)
            
            # Return targets
            df[f'return_{horizon}'] = df[self.config.price_target].pct_change(periods=horizon).shift(-horizon)
            
            # Volatility targets
            df[f'volatility_{horizon}'] = df[self.config.price_target].pct_change().rolling(window=horizon).std().shift(-horizon)
        
        return df

class FeatureSelector:
    """Feature selection for ML models."""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
    
    def select_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Select features for ML models."""
        df = data.copy()
        
        # Get all numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove excluded columns
        feature_columns = [col for col in numeric_columns if col not in self.config.exclude_columns]
        
        # Remove target columns
        target_columns = [self.config.direction_target, self.config.volatility_target, 'return_target', 'price_target']
        feature_columns = [col for col in feature_columns if col not in target_columns]
        
        # Remove multi-horizon targets
        feature_columns = [col for col in feature_columns if not any(col.startswith(f'{target}_') for target in ['direction', 'return', 'volatility'])]
        
        # If specific features are specified, use only those
        if self.config.feature_columns:
            feature_columns = [col for col in feature_columns if col in self.config.feature_columns]
        
        logger.info(f"Selected {len(feature_columns)} features: {feature_columns}")
        
        return df[feature_columns], feature_columns

class DataPreprocessor:
    """Data preprocessing for ML models."""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_columns = []
    
    def preprocess_data(self, data: pd.DataFrame, target_column: str, fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for ML models."""
        df = data.copy()
        
        # Remove rows with NaN values
        df = df.dropna()
        
        if df.empty:
            logger.warning("No data after removing NaN values")
            return np.array([]), np.array([])
        
        # Separate features and target
        feature_selector = FeatureSelector(self.config)
        features_df, feature_columns = feature_selector.select_features(df)
        
        # Store feature columns
        self.feature_columns = feature_columns
        
        if features_df.empty:
            logger.warning("No features available")
            return np.array([]), np.array([])
        
        # Get target
        if target_column not in df.columns:
            logger.error(f"Target column {target_column} not found")
            return np.array([]), np.array([])
        
        target = df[target_column].values
        
        # Scale features
        if fit_scaler:
            features_scaled = self.scaler.fit_transform(features_df)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                logger.warning("Scaler not fitted. Fitting now.")
                features_scaled = self.scaler.fit_transform(features_df)
                self.is_fitted = True
            else:
                features_scaled = self.scaler.transform(features_df)
        
        logger.info(f"Preprocessed data shape: {features_scaled.shape}")
        
        return features_scaled, target

class TraditionalMLEngine(BaseMLEngine):
    """Traditional ML engine for quantitative analysis."""
    
    def __init__(self, config: UnifiedMLConfig = None):
        super().__init__(config)
        self.model_config = ModelConfig()
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        
    def train_price_prediction_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train price prediction model (regression)."""
        logger.info("Training price prediction model...")
        
        # Generate targets
        target_generator = TargetGenerator(self.model_config)
        df = target_generator.generate_targets(data)
        
        # Preprocess data
        preprocessor = DataPreprocessor(self.model_config)
        X, y = preprocessor.preprocess_data(df, 'price_target')
        
        if len(X) == 0 or len(y) == 0:
            logger.error("No data available for training")
            return {}
        
        # Split data
        split_idx = int(len(X) * (1 - self.model_config.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train models
        models = {}
        results = {}
        
        if HAS_SKLEARN:
            # Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=self.config.rf_n_estimators, 
                random_state=self.model_config.random_state
            )
            rf_model.fit(X_train, y_train)
            models['random_forest'] = rf_model
            
            # Linear Regression
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            models['linear_regression'] = lr_model
            
            # SVR
            svr_model = SVR(kernel='rbf')
            svr_model.fit(X_train, y_train)
            models['svr'] = svr_model
        
        if HAS_XGBOOST:
            # XGBoost
            xgb_model = xgb.XGBRegressor(
                n_estimators=self.config.xgb_n_estimators, 
                random_state=self.model_config.random_state
            )
            xgb_model.fit(X_train, y_train)
            models['xgboost'] = xgb_model
        
        # Evaluate models
        for name, model in models.items():
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred
            }
            
            logger.info(f"{name}: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        
        # Store best model
        if results:
            best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
            self.models['price_prediction'] = results[best_model_name]['model']
            self.scalers['price_prediction'] = preprocessor.scaler
            self.feature_columns['price_prediction'] = preprocessor.feature_columns
            
            # Register in global registry
            self.registry.register_model(
                "traditional_price_prediction",
                results[best_model_name]['model'],
                preprocessor.scaler,
                preprocessor.feature_columns
            )
            
            logger.info(f"Best price prediction model: {best_model_name}")
        
        return results
    
    def train_direction_prediction_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train direction prediction model (classification)."""
        logger.info("Training direction prediction model...")
        
        # Generate targets
        target_generator = TargetGenerator(self.model_config)
        df = target_generator.generate_targets(data)
        
        # Preprocess data
        preprocessor = DataPreprocessor(self.model_config)
        X, y = preprocessor.preprocess_data(df, self.model_config.direction_target)
        
        if len(X) == 0 or len(y) == 0:
            logger.error("No data available for training")
            return {}
        
        # Split data
        split_idx = int(len(X) * (1 - self.model_config.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train models
        models = {}
        results = {}
        
        if HAS_SKLEARN:
            # Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=self.config.rf_n_estimators, 
                random_state=self.model_config.random_state
            )
            rf_model.fit(X_train, y_train)
            models['random_forest'] = rf_model
            
            # Logistic Regression
            lr_model = LogisticRegression(random_state=self.model_config.random_state)
            lr_model.fit(X_train, y_train)
            models['logistic_regression'] = lr_model
            
            # SVC
            svc_model = SVC(probability=True, random_state=self.model_config.random_state)
            svc_model.fit(X_train, y_train)
            models['svc'] = svc_model
        
        if HAS_XGBOOST:
            # XGBoost
            xgb_model = xgb.XGBClassifier(
                n_estimators=self.config.xgb_n_estimators, 
                random_state=self.model_config.random_state
            )
            xgb_model.fit(X_train, y_train)
            models['xgboost'] = xgb_model
        
        # Evaluate models
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_prob
            }
            
            logger.info(f"{name}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            if auc is not None:
                logger.info(f"{name}: AUC={auc:.4f}")
        
        # Store best model
        if results:
            best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
            self.models['direction_prediction'] = results[best_model_name]['model']
            self.scalers['direction_prediction'] = preprocessor.scaler
            self.feature_columns['direction_prediction'] = preprocessor.feature_columns
            
            # Register in global registry
            self.registry.register_model(
                "traditional_direction_prediction",
                results[best_model_name]['model'],
                preprocessor.scaler,
                preprocessor.feature_columns
            )
            
            logger.info(f"Best direction prediction model: {best_model_name}")
        
        return results
    
    def train_volatility_prediction_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train volatility prediction model (regression)."""
        logger.info("Training volatility prediction model...")
        
        # Generate targets
        target_generator = TargetGenerator(self.model_config)
        df = target_generator.generate_targets(data)
        
        # Preprocess data
        preprocessor = DataPreprocessor(self.model_config)
        X, y = preprocessor.preprocess_data(df, self.model_config.volatility_target)
        
        if len(X) == 0 or len(y) == 0:
            logger.error("No data available for training")
            return {}
        
        # Split data
        split_idx = int(len(X) * (1 - self.model_config.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train models
        models = {}
        results = {}
        
        if HAS_SKLEARN:
            # Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=self.config.rf_n_estimators, 
                random_state=self.model_config.random_state
            )
            rf_model.fit(X_train, y_train)
            models['random_forest'] = rf_model
            
            # Linear Regression
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            models['linear_regression'] = lr_model
        
        if HAS_XGBOOST:
            # XGBoost
            xgb_model = xgb.XGBRegressor(
                n_estimators=self.config.xgb_n_estimators, 
                random_state=self.model_config.random_state
            )
            xgb_model.fit(X_train, y_train)
            models['xgboost'] = xgb_model
        
        # Evaluate models
        for name, model in models.items():
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred
            }
            
            logger.info(f"{name}: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.4f}")
        
        # Store best model
        if results:
            best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
            self.models['volatility_prediction'] = results[best_model_name]['model']
            self.scalers['volatility_prediction'] = preprocessor.scaler
            self.feature_columns['volatility_prediction'] = preprocessor.feature_columns
            
            # Register in global registry
            self.registry.register_model(
                "traditional_volatility_prediction",
                results[best_model_name]['model'],
                preprocessor.scaler,
                preprocessor.feature_columns
            )
            
            logger.info(f"Best volatility prediction model: {best_model_name}")
        
        return results
    
    def train_all_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train all traditional ML models."""
        logger.info("Training all traditional ML models...")
        
        results = {}
        
        # Train price prediction model
        results['price_prediction'] = self.train_price_prediction_model(data)
        
        # Train direction prediction model
        results['direction_prediction'] = self.train_direction_prediction_model(data)
        
        # Train volatility prediction model
        results['volatility_prediction'] = self.train_volatility_prediction_model(data)
        
        self.is_trained = len(self.models) > 0
        logger.info("All traditional ML models trained successfully")
        
        return results
    
    def predict(self, data: pd.DataFrame, model_type: str) -> Dict[str, Any]:
        """Make predictions using trained models."""
        if model_type not in self.models:
            logger.error(f"Model {model_type} not found")
            return {}
        
        # Preprocess data
        preprocessor = DataPreprocessor(self.model_config)
        X, _ = preprocessor.preprocess_data(data, 'close', fit_scaler=False)
        
        if len(X) == 0:
            logger.error("No data available for prediction")
            return {}
        
        # Make predictions
        model = self.models[model_type]
        predictions = model.predict(X)
        
        # Get probabilities for classification models
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[:, 1]
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'model_type': model_type
        }
    
    def evaluate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Evaluate traditional ML models."""
        if not self.is_trained:
            return {"error": "Models not trained"}
        
        try:
            evaluation = {}
            
            # Evaluate each model type
            for model_type in self.models.keys():
                if model_type in ['price_prediction', 'volatility_prediction']:
                    # Regression models
                    pred_result = self.predict(data, model_type)
                    if pred_result:
                        evaluation[model_type] = {
                            "predictions": pred_result['predictions'],
                            "model_type": "regression"
                        }
                elif model_type == 'direction_prediction':
                    # Classification models
                    pred_result = self.predict(data, model_type)
                    if pred_result:
                        evaluation[model_type] = {
                            "predictions": pred_result['predictions'],
                            "probabilities": pred_result['probabilities'],
                            "model_type": "classification"
                        }
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"error": str(e)}
    
    def save_model(self, path: str) -> bool:
        """Save the traditional ML models."""
        if not self.is_trained:
            return False
        
        try:
            import joblib
            
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_columns': self.feature_columns,
                'config': self.config
            }
            
            joblib.dump(model_data, path)
            logger.info(f"Traditional ML models saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save traditional ML models: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Load the traditional ML models."""
        try:
            import joblib
            
            model_data = joblib.load(path)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_columns = model_data['feature_columns']
            self.config = model_data['config']
            self.is_trained = True
            
            logger.info(f"Traditional ML models loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load traditional ML models: {e}")
            return False

# Global instance
traditional_ml_engine = TraditionalMLEngine()
