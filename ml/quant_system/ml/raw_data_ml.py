"""
Raw Data ML Module

This module provides traditional quantitative analysis capabilities by processing
raw stock data (OHLCV) directly to predict price movements, volatility, and market regimes.
Adapted from backend/ml/raw_data_engine.py for unified integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

# ML/DL imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logging.warning("PyTorch not available. LSTM models will not work.")

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import f1_score, roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logging.warning("scikit-learn not available. Traditional ML models will not work.")

from .core import BaseMLEngine, UnifiedMLConfig

logger = logging.getLogger(__name__)

@dataclass
class PricePrediction:
    """Price movement prediction result."""
    direction: str  # "up", "down", "sideways"
    magnitude: float  # Expected price change as percentage
    confidence: float  # Prediction confidence (0-1)
    timeframe: str  # Prediction timeframe
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None

@dataclass
class VolatilityPrediction:
    """Volatility prediction result."""
    current_volatility: float
    predicted_volatility: float
    volatility_regime: str  # "low", "medium", "high", "increasing", "decreasing"
    confidence: float

@dataclass
class MarketRegime:
    """Market regime classification."""
    regime: str  # "trending_bull", "trending_bear", "sideways", "volatile"
    strength: float  # Regime strength (0-1)
    duration: int  # Expected duration in periods
    confidence: float

class RawDataFeatureEngineer:
    """Feature engineering for raw stock data."""
    
    def __init__(self, config: UnifiedMLConfig = None):
        self.config = config or UnifiedMLConfig()
        self.scaler = StandardScaler()
        self.feature_columns = []
    
    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators from raw OHLCV data."""
        df = data.copy()
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Missing required columns. Available: {df.columns.tolist()}")
            return df
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
        
        # Moving averages (use min_periods to reduce NaN loss on early rows)
        window_to_min = {5: 3, 10: 5, 20: 10, 50: 20}
        for window in [5, 10, 20, 50]:
            min_p = window_to_min[window]
            df[f'sma_{window}'] = df['close'].rolling(window=window, min_periods=min_p).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            df[f'price_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
            df[f'price_ema_{window}_ratio'] = df['close'] / df[f'ema_{window}']
            df[f'sma_{window}_slope'] = df[f'sma_{window}'].diff(3) / df[f'sma_{window}'].shift(3)
            df[f'ema_{window}_slope'] = df[f'ema_{window}'].diff(3) / df[f'ema_{window}'].shift(3)
        
        # Volatility features
        df['volatility_5'] = df['returns'].rolling(window=5, min_periods=3).std()
        df['volatility_20'] = df['returns'].rolling(window=20, min_periods=10).std()
        df['volatility_50'] = df['returns'].rolling(window=50, min_periods=20).std()
        df['volatility_change_5'] = df['volatility_5'].pct_change()
        df['volatility_change_20'] = df['volatility_20'].pct_change()
        df['volatility_ratio_5_20'] = df['volatility_5'] / df['volatility_20']
        df['volatility_ratio_5_50'] = df['volatility_5'] / df['volatility_50']
        
        # Volume features
        df['volume_sma_20'] = df['volume'].rolling(window=20, min_periods=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['volume_price_trend'] = (df['volume'] * df['returns']).rolling(10).sum()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(50, min_periods=25).mean()
        df['volume_volatility'] = df['volume'].rolling(20, min_periods=10).std() / df['volume_sma_20']
        
        # Momentum features
        df['rsi_14'] = self._calculate_rsi(df['close'], self.config.rsi_period)
        df['macd'] = self._calculate_macd(df['close'])
        df['stochastic'] = self._calculate_stochastic(df['high'], df['low'], df['close'])
        df['rsi_regime'] = np.where(df['rsi_14'] > 70, 2, np.where(df['rsi_14'] < 30, 0, 1))
        df['stoch_regime'] = np.where(df['stochastic'] > 80, 2, np.where(df['stochastic'] < 20, 0, 1))
        
        # Trend features
        df['adx'] = self._calculate_adx(df['high'], df['low'], df['close'])
        df['trend_strength'] = abs(df['ema_20'] - df['ema_50']) / df['ema_50']
        df['trend_direction'] = np.where(df['ema_20'] > df['ema_50'], 1, -1)
        df['trend_consistency'] = ((df['ema_20'] > df['ema_20'].shift(1)) & (df['ema_50'] > df['ema_50'].shift(1))).astype(int)
        
        # Bollinger Bands
        df['bb_upper'] = df['sma_20'] + (df['volatility_20'] * 2 * df['close'])
        df['bb_lower'] = df['sma_20'] - (df['volatility_20'] * 2 * df['close'])
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']
        
        # ATR and related
        df['atr'] = self._calculate_atr(df['high'], df['low'], df['close'])
        df['atr_ratio'] = df['atr'] / df['close']
        df['atr_change'] = df['atr'].pct_change()
        
        # Time-based features
        if 'date' in df.columns:
            # If date is a column, use it
            df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
            df['month'] = pd.to_datetime(df['date']).dt.month
            df['quarter'] = pd.to_datetime(df['date']).dt.quarter
        else:
            # If date is the index, use it
            df['day_of_week'] = pd.to_datetime(df.index).dayofweek
            df['month'] = pd.to_datetime(df.index).month
            df['quarter'] = pd.to_datetime(df.index).quarter
        
        # Remove any unnamed columns
        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
        
        # Add market regime features
        df['market_regime'] = self._classify_market_regime(df)
        df['regime_volatility'] = np.where(df['market_regime'] == 'volatile', 1, 0)
        df['regime_trending'] = np.where(df['market_regime'] == 'trending_bull', 1, np.where(df['market_regime'] == 'trending_bear', -1, 0))
        
        # Ensure all features are numeric
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Try to convert to numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Fill NaN with 0
                    df[col] = df[col].fillna(0)
                except:
                    # If conversion fails, drop the column
                    df = df.drop(columns=[col])
                    logger.warning(f"Dropped non-numeric column: {col}")
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        min_p = max(3, window // 3)
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=min_p).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=min_p).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Stochastic oscillator."""
        min_p = max(3, window // 3)
        lowest_low = low.rolling(window=window, min_periods=min_p).min()
        highest_high = high.rolling(window=window, min_periods=min_p).max()
        stochastic = 100 * (close - lowest_low) / (highest_high - lowest_low)
        return stochastic
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window, min_periods=max(3, window // 3)).mean()
        return atr
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate ADX indicator."""
        # Simplified ADX calculation using ATR
        atr = self._calculate_atr(high, low, close, window)
        return atr / close * 100
    
    def _classify_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Classify market regime based on volatility and trend."""
        returns = df['returns'].rolling(20, min_periods=10).mean()
        volatility = df['volatility_20']
        
        regime = pd.Series('sideways', index=df.index)
        regime = np.where((abs(returns) > 0.001) & (volatility < 0.02), 
                         np.where(returns > 0, 'trending_bull', 'trending_bear'), regime)
        regime = np.where(volatility > 0.03, 'volatile', regime)
        
        return pd.Series(regime, index=df.index)

class LSTMPredictor(nn.Module):
    """LSTM model for price prediction."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 3)  # 3 outputs: direction, magnitude, confidence
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take last output
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output

class RawDataMLEngine(BaseMLEngine):
    """Main ML engine for raw stock data processing."""
    
    def __init__(self, config: UnifiedMLConfig = None):
        super().__init__(config)
        self.feature_engineer = RawDataFeatureEngineer(config)
        self.direction_model = None
        self.magnitude_model = None
        self.volatility_model = None
        self.regime_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def train(self, data: pd.DataFrame, target_horizon: int = 1) -> bool:
        """Train price prediction model."""
        if not HAS_SKLEARN:
            logger.warning("scikit-learn not available")
            return False
        
        try:
            # Prepare features
            features_df = self.feature_engineer.create_technical_features(data)
            # Clean NaNs and infinities conservatively to retain early samples
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.ffill()
            features_df = features_df.fillna(0)
            
            # Create target (future price movement)
            features_df['future_return'] = features_df['close'].shift(-target_horizon) / features_df['close'] - 1
            features_df['future_direction'] = np.where(features_df['future_return'] > 0, 1, 0)
            
            # Select features - exclude date and other non-feature columns
            exclude_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'future_return', 'future_direction', 'symbol']
            feature_columns = [col for col in features_df.columns if col not in exclude_columns]
            
            # Ensure we have features
            if len(feature_columns) == 0:
                logger.error("No feature columns found")
                return False
            
            logger.info(f"Feature columns: {feature_columns}")
            logger.info(f"Features shape before target horizon: {features_df.shape}")
            
            # Get feature data
            X = features_df[feature_columns].values[:-target_horizon]
            y_direction = features_df['future_direction'].values[:-target_horizon]
            y_magnitude = features_df['future_return'].values[:-target_horizon]
            
            logger.info(f"X shape: {X.shape}, y_direction shape: {y_direction.shape}")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train direction model (classification)
            self.direction_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=2,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            
            # Calibrate the classifier for better probability estimates
            self.direction_model = CalibratedClassifierCV(
                self.direction_model, 
                cv=3, 
                method='isotonic'
            )
            self.direction_model.fit(X_scaled, y_direction.astype(int))
            
            # Find optimal threshold on validation set
            self.optimal_threshold = self._find_optimal_threshold(X_scaled, y_direction.astype(int))
            
            # Train magnitude model (regression)
            self.magnitude_model = GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                random_state=self.config.random_state
            )
            self.magnitude_model.fit(X_scaled, y_magnitude)
            
            self.feature_columns = feature_columns
            self.is_trained = True
            
            # Register models
            self.registry.register_model(
                "raw_data_direction", 
                self.direction_model, 
                self.scaler,
                self.feature_columns
            )
            self.registry.register_model(
                "raw_data_magnitude", 
                self.magnitude_model, 
                self.scaler,
                self.feature_columns
            )
            
            logger.info(f"Optimal threshold: {self.optimal_threshold:.3f}")
            logger.info(f"Raw data ML model trained with {len(X)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train raw data ML model: {e}")
            return False
    
    def predict(self, data: pd.DataFrame, horizon: int = 1) -> PricePrediction:
        """Predict price movement."""
        if not self.is_trained or self.direction_model is None or self.magnitude_model is None:
            return PricePrediction("sideways", 0.0, 0.5, f"{horizon}period")
        
        try:
            # Prepare features
            features_df = self.feature_engineer.create_technical_features(data)
            features_df = features_df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            latest_features = features_df[self.feature_columns].iloc[-1:].values
            latest_features_scaled = self.scaler.transform(latest_features)
            
            # Predict direction and magnitude
            if hasattr(self.direction_model, 'predict_proba'):
                proba = self.direction_model.predict_proba(latest_features_scaled)[0]
                # Probability of class 1 (up)
                direction_prob = float(proba[1]) if proba.shape[-1] > 1 else float(proba[0])
            else:
                direction_pred = self.direction_model.predict(latest_features_scaled)[0]
                direction_prob = float(direction_pred)
            magnitude = self.magnitude_model.predict(latest_features_scaled)[0]
            
            # Determine direction using probability thresholds
            # Use optimal threshold for direction classification
            threshold = getattr(self, 'optimal_threshold', 0.5)
            if direction_prob >= threshold:
                direction = "up"
                confidence = direction_prob
            elif direction_prob <= (1 - threshold):
                direction = "down"
                confidence = 1 - direction_prob
            else:
                direction = "sideways"
                confidence = 0.5
            
            return PricePrediction(
                direction=direction,
                magnitude=abs(magnitude),
                confidence=confidence,
                timeframe=f"{horizon}period"
            )
            
        except Exception as e:
            logger.error(f"Price prediction failed: {e}")
            return PricePrediction("sideways", 0.0, 0.5, f"{horizon}period")
    
    def predict_volatility(self, data: pd.DataFrame) -> VolatilityPrediction:
        """Predict future volatility."""
        try:
            # Calculate current volatility
            returns = data['close'].pct_change().dropna()
            current_vol = returns.rolling(20).std().iloc[-1]
            
            # Simple volatility prediction (can be enhanced with ML)
            # Predict based on recent volatility trend
            vol_trend = returns.rolling(10).std().diff().iloc[-1]
            
            if vol_trend > 0:
                predicted_vol = current_vol * 1.1
                regime = "increasing"
            elif vol_trend < 0:
                predicted_vol = current_vol * 0.9
                regime = "decreasing"
            else:
                predicted_vol = current_vol
                regime = "stable"
            
            return VolatilityPrediction(
                current_volatility=current_vol,
                predicted_volatility=predicted_vol,
                volatility_regime=regime,
                confidence=0.7
            )
            
        except Exception as e:
            logger.error(f"Volatility prediction failed: {e}")
            return VolatilityPrediction(0.02, 0.02, "stable", 0.5)
    
    def classify_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """Classify current market regime."""
        try:
            # Calculate regime indicators
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1]
            trend = returns.rolling(50).mean().iloc[-1]
            
            # Determine regime
            if abs(trend) > 0.001 and volatility < 0.02:
                if trend > 0:
                    regime = "trending_bull"
                    strength = min(abs(trend) * 1000, 1.0)
                else:
                    regime = "trending_bear"
                    strength = min(abs(trend) * 1000, 1.0)
            elif volatility > 0.03:
                regime = "volatile"
                strength = min(volatility * 20, 1.0)
            else:
                regime = "sideways"
                strength = 0.5
            
            return MarketRegime(
                regime=regime,
                strength=strength,
                duration=20,  # Default duration
                confidence=0.8
            )
            
        except Exception as e:
            logger.error(f"Market regime classification failed: {e}")
            return MarketRegime("sideways", 0.5, 20, 0.5)
    
    def evaluate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Evaluate model performance."""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        try:
            # Make predictions
            price_pred = self.predict(data)
            volatility_pred = self.predict_volatility(data)
            regime_pred = self.classify_market_regime(data)
            
            return {
                "price_prediction": {
                    "direction": price_pred.direction,
                    "magnitude": price_pred.magnitude,
                    "confidence": price_pred.confidence
                },
                "volatility_prediction": {
                    "current": volatility_pred.current_volatility,
                    "predicted": volatility_pred.predicted_volatility,
                    "regime": volatility_pred.volatility_regime
                },
                "market_regime": {
                    "regime": regime_pred.regime,
                    "strength": regime_pred.strength,
                    "confidence": regime_pred.confidence
                }
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"error": str(e)}
    
    def save_model(self, path: str) -> bool:
        """Save the trained model."""
        if not self.is_trained:
            return False
        
        try:
            import joblib
            model_data = {
                'direction_model': self.direction_model,
                'magnitude_model': self.magnitude_model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }
            joblib.dump(model_data, path)
            logger.info(f"Raw data ML model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Load a trained model."""
        try:
            import joblib
            model_data = joblib.load(path)
            
            self.direction_model = model_data['direction_model']
            self.magnitude_model = model_data['magnitude_model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = True
            
            logger.info(f"Raw data ML model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _find_optimal_threshold(self, X: np.ndarray, y: np.ndarray) -> float:
        """Find optimal classification threshold using validation set."""
        try:
            # Use time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            thresholds = np.arange(0.3, 0.8, 0.05)
            best_f1 = 0
            best_threshold = 0.5
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train on train split
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.config.random_state
                )
                model.fit(X_train, y_train)
                
                # Predict probabilities on validation
                proba = model.predict_proba(X_val)[:, 1]
                
                # Find best threshold
                for threshold in thresholds:
                    y_pred = (proba >= threshold).astype(int)
                    f1 = f1_score(y_val, y_pred, average='weighted')
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
            
            return best_threshold
            
        except Exception as e:
            logger.warning(f"Threshold optimization failed: {e}, using default 0.5")
            return 0.5

# Global instance
raw_data_ml_engine = RawDataMLEngine()
