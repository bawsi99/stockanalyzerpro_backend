"""
Enhanced Feature Engineering for Pattern ML

This module provides advanced feature engineering capabilities including:
1. Market regime features (volatility, trend strength, sector rotation)
2. Pattern-specific features (breakout strength, volume confirmation)
3. Technical indicators (RSI, MACD, Bollinger Bands)
4. Time-based features (day of week, month, seasonality)
5. Cross-pattern features (pattern frequency, success rate by type)

Usage:
    from enhanced_feature_engineering import EnhancedFeatureEngine
    engine = EnhancedFeatureEngine()
    X_enhanced = engine.transform(df)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EnhancedFeatureEngine:
    """Enhanced feature engineering for pattern ML."""
    
    def __init__(self):
        self.feature_names = []
        self.is_fitted = False
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the feature engineering and transform the data."""
        self.fit(df)
        return self.transform(df)
    
    def fit(self, df: pd.DataFrame):
        """Fit the feature engineering (compute statistics, etc.)."""
        # Compute pattern success rates by type
        self.pattern_success_rates = {}
        if 'pattern_type' in df.columns and 'y_success' in df.columns:
            for pattern_type in df['pattern_type'].unique():
                pattern_df = df[df['pattern_type'] == pattern_type]
                if len(pattern_df) > 10:  # Only if we have enough samples
                    success_rate = pattern_df['y_success'].mean()
                    self.pattern_success_rates[pattern_type] = success_rate
        
        # Compute market regime statistics
        if 'ret_20' in df.columns:
            self.market_volatility = df['ret_20'].std()
            self.market_trend = df['ret_20'].mean()
        
        self.is_fitted = True
        logger.info(f"Enhanced feature engineering fitted with {len(self.pattern_success_rates)} pattern types")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data with enhanced features."""
        if not self.is_fitted:
            self.fit(df)
        
        X = df.copy()
        
        # 1. Basic features (existing)
        X = self._add_basic_features(X)
        
        # 2. Market regime features
        X = self._add_market_regime_features(X)
        
        # 3. Pattern-specific features
        X = self._add_pattern_specific_features(X)
        
        # 4. Technical indicators
        X = self._add_technical_indicators(X)
        
        # 5. Time-based features
        X = self._add_time_features(X)
        
        # 6. Cross-pattern features
        X = self._add_cross_pattern_features(X)
        
        # 7. Volatility features
        X = self._add_volatility_features(X)
        
        # 8. Volume analysis features
        X = self._add_volume_features(X)
        
        # Store feature names (only numeric columns)
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        self.feature_names = [col for col in numeric_columns if col not in ['y_success']]
        
        # Keep only numeric features
        X = X[self.feature_names]
        
        # Clean NaN values
        X = X.fillna(0.0)  # Fill NaN with 0 for numeric features
        
        # Remove any remaining infinite values
        X = X.replace([np.inf, -np.inf], 0.0)
        
        return X
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic features (existing logic)."""
        X = df.copy()
        
        # Duration (normalized)
        if 'duration' in X.columns:
            X['duration_norm'] = (X['duration'] - X['duration'].mean()) / (X['duration'].std() + 1e-8)
        
        # Volume ratio (clipped and normalized)
        if 'volume_ratio20' in X.columns:
            vr = pd.to_numeric(X['volume_ratio20'], errors='coerce').fillna(1.0).clip(0, 10)
            X['volume_ratio_norm'] = (vr - vr.mean()) / (vr.std() + 1e-8)
        
        # Trend alignment
        if 'ret_20' in X.columns:
            r20 = pd.to_numeric(X['ret_20'], errors='coerce').fillna(0.0)
            X['trend_alignment'] = np.where(r20 > 0, 1.0, np.where(r20 < 0, 0.0, 0.5))
            X['trend_strength'] = np.abs(r20)
        
        # Completion status
        if 'completion_status' in X.columns:
            comp = X['completion_status'].astype(str).str.lower()
            X['completion'] = np.where(comp == 'completed', 1.0, 0.0)
        
        return X
    
    def _add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime features."""
        X = df.copy()
        
        # Market volatility regime
        if 'ret_20' in X.columns:
            r20 = pd.to_numeric(X['ret_20'], errors='coerce').fillna(0.0)
            
            # Rolling volatility (if we have time series)
            if 'event_date' in X.columns:
                X['event_date'] = pd.to_datetime(X['event_date'])
                X = X.sort_values('event_date')
                
                # 30-day rolling volatility
                rolling_vol = r20.rolling(window=30, min_periods=10).std().fillna(r20.std())
                X['market_volatility_30d'] = rolling_vol
                
                # Volatility regime (high/medium/low)
                vol_quantiles = rolling_vol.quantile([0.33, 0.67])
                X['volatility_regime'] = np.where(
                    rolling_vol <= vol_quantiles.iloc[0], 0,  # Low
                    np.where(rolling_vol <= vol_quantiles.iloc[1], 1, 2)  # Medium/High
                )
            
            # Trend regime
            rolling_trend = r20.rolling(window=20, min_periods=10).mean().fillna(r20.mean())
            X['trend_regime'] = np.where(
                rolling_trend > 0.01, 1,  # Bullish
                np.where(rolling_trend < -0.01, -1, 0)  # Bearish/Neutral
            )
        
        return X
    
    def _add_pattern_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern-specific features."""
        X = df.copy()
        
        # Pattern type encoding
        if 'pattern_type' in X.columns:
            # One-hot encoding for pattern types
            pattern_dummies = pd.get_dummies(X['pattern_type'], prefix='pattern')
            X = pd.concat([X, pattern_dummies], axis=1)
            
            # Pattern success rate (from historical data)
            X['pattern_success_rate'] = X['pattern_type'].map(self.pattern_success_rates).fillna(0.5)
        
        # Breakout strength (for reversal patterns)
        if 'ret_20' in X.columns and 'pattern_type' in X.columns:
            r20 = pd.to_numeric(X['ret_20'], errors='coerce').fillna(0.0)
            
            # For reversal patterns, stronger reversal = better
            reversal_patterns = ['double_top', 'head_and_shoulders', 'double_bottom', 'inverse_head_and_shoulders']
            X['breakout_strength'] = np.where(
                X['pattern_type'].isin(reversal_patterns),
                np.abs(r20),  # Stronger reversal
                1.0 - np.abs(r20)  # For continuation patterns, weaker move
            )
        
        return X
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features."""
        X = df.copy()
        
        # RSI approximation (if we have price data)
        if 'ret_20' in X.columns:
            r20 = pd.to_numeric(X['ret_20'], errors='coerce').fillna(0.0)
            
            # Simple RSI approximation based on return distribution
            positive_returns = pd.Series(np.where(r20 > 0, r20, 0), index=X.index)
            negative_returns = pd.Series(np.where(r20 < 0, -r20, 0), index=X.index)
            
            avg_gain = positive_returns.rolling(window=14, min_periods=5).mean().fillna(0)
            avg_loss = negative_returns.rolling(window=14, min_periods=5).mean().fillna(0)
            
            rs = avg_gain / (avg_loss + 1e-8)
            X['rsi_approx'] = 100 - (100 / (1 + rs))
            
            # RSI regime
            X['rsi_regime'] = np.where(
                X['rsi_approx'] > 70, 1,  # Overbought
                np.where(X['rsi_approx'] < 30, -1, 0)  # Oversold/Neutral
            )
        
        return X
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        X = df.copy()
        
        if 'event_date' in X.columns:
            X['event_date'] = pd.to_datetime(X['event_date'])
            
            # Day of week (0=Monday, 6=Sunday)
            X['day_of_week'] = X['event_date'].dt.dayofweek
            
            # Month
            X['month'] = X['event_date'].dt.month
            
            # Quarter
            X['quarter'] = X['event_date'].dt.quarter
            
            # Day of month
            X['day_of_month'] = X['event_date'].dt.day
            
            # Week of year
            X['week_of_year'] = X['event_date'].dt.isocalendar().week
            
            # Seasonality (sin/cos encoding)
            X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
            X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)
            
            # Day of week encoding
            X['dow_sin'] = np.sin(2 * np.pi * X['day_of_week'] / 7)
            X['dow_cos'] = np.cos(2 * np.pi * X['day_of_week'] / 7)
        
        return X
    
    def _add_cross_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-pattern features."""
        X = df.copy()
        
        if 'event_date' in X.columns and 'pattern_type' in X.columns:
            X['event_date'] = pd.to_datetime(X['event_date'])
            X = X.sort_values('event_date')
            
            # Pattern frequency (how common is this pattern type recently)
            pattern_counts = X.groupby('pattern_type').rolling(
                window=30, min_periods=1
            ).count().iloc[:, 0].reset_index(0, drop=True)
            
            X['pattern_frequency'] = pattern_counts
            
            # Recent pattern success rate
            if 'y_success' in X.columns:
                recent_success = X.groupby('pattern_type').rolling(
                    window=20, min_periods=5
                )['y_success'].mean().reset_index(0, drop=True)
                
                X['recent_pattern_success'] = recent_success.fillna(0.5)
        
        return X
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-related features."""
        X = df.copy()
        
        if 'ret_20' in X.columns:
            r20 = pd.to_numeric(X['ret_20'], errors='coerce').fillna(0.0)
            
            # Realized volatility
            X['realized_volatility'] = np.abs(r20)
            
            # Volatility of volatility
            if 'event_date' in X.columns:
                X['event_date'] = pd.to_datetime(X['event_date'])
                X = X.sort_values('event_date')
                
                vol_of_vol = np.abs(r20).rolling(window=10, min_periods=5).std().fillna(0)
                X['volatility_of_volatility'] = vol_of_vol
            
            # Volatility regime change
            if 'market_volatility_30d' in X.columns:
                vol_change = X['market_volatility_30d'].diff()
                X['volatility_change'] = vol_change.fillna(0)
        
        return X
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume analysis features."""
        X = df.copy()
        
        # Volume confirmation
        if 'volume_ratio20' in X.columns:
            vr = pd.to_numeric(X['volume_ratio20'], errors='coerce').fillna(1.0)
            
            # Volume spike
            X['volume_spike'] = np.where(vr > 2.0, 1.0, 0.0)
            
            # Volume trend
            if 'event_date' in X.columns:
                X['event_date'] = pd.to_datetime(X['event_date'])
                X = X.sort_values('event_date')
                
                volume_trend = vr.rolling(window=10, min_periods=5).mean().fillna(1.0)
                X['volume_trend'] = np.where(volume_trend > 1.2, 1.0, np.where(volume_trend < 0.8, -1.0, 0.0))
        
        return X
    
    def get_feature_importance(self, model) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if not hasattr(model, 'feature_importances_'):
            return {}
        
        if hasattr(model, 'named_steps') and 'calibratedclassifiercv' in model.named_steps:
            # For calibrated models, get the base estimator
            base_model = model.named_steps['calibratedclassifiercv'].base_estimator
            if hasattr(base_model, 'feature_importances_'):
                importances = base_model.feature_importances_
            else:
                return {}
        else:
            importances = model.feature_importances_
        
        feature_importance = dict(zip(self.feature_names, importances))
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of engineered features."""
        return {
            'total_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'feature_categories': {
                'basic': [f for f in self.feature_names if any(x in f for x in ['duration', 'volume_ratio', 'trend', 'completion'])],
                'market_regime': [f for f in self.feature_names if any(x in f for x in ['volatility', 'trend_regime', 'market'])],
                'pattern_specific': [f for f in self.feature_names if any(x in f for x in ['pattern_', 'breakout'])],
                'technical': [f for f in self.feature_names if any(x in f for x in ['rsi', 'technical'])],
                'time': [f for f in self.feature_names if any(x in f for x in ['day', 'month', 'week', 'quarter', 'sin', 'cos'])],
                'cross_pattern': [f for f in self.feature_names if any(x in f for x in ['frequency', 'recent'])],
                'volatility': [f for f in self.feature_names if any(x in f for x in ['volatility'])],
                'volume': [f for f in self.feature_names if any(x in f for x in ['volume'])]
            }
        }
