"""
Simplified Feature Engineering Module

This module provides ML-focused feature engineering by importing from backend modules.
No code duplication - uses the robust backend technical indicators and patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass

# Import from backend instead of duplicating
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

try:
    from technical_indicators import TechnicalIndicators
    from patterns.recognition import PatternRecognition
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    logging.warning("Backend modules not available, using simplified features")

from .core import UnifiedMLConfig

logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # Use backend defaults
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    
    # Volatility features
    atr_period: int = 14
    volatility_period: int = 20
    
    # Volume features
    vwap_period: int = 20
    volume_sma_period: int = 20
    
    # Momentum features
    momentum_periods: List[int] = None
    
    def __post_init__(self):
        if self.momentum_periods is None:
            self.momentum_periods = [5, 10, 20, 50]

class FeatureEngineer:
    """Simplified feature engineering using backend modules."""
    
    def __init__(self, config: UnifiedMLConfig = None):
        self.config = config or UnifiedMLConfig()
        self.feature_config = FeatureConfig()
        
        # Initialize backend modules if available
        if BACKEND_AVAILABLE:
            self.technical_indicators = TechnicalIndicators()
            self.pattern_recognition = PatternRecognition()
        else:
            self.technical_indicators = None
            self.pattern_recognition = None
            logger.warning("Using simplified feature engineering (backend not available)")
        
    def create_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create all features using backend modules when available."""
        if data.empty:
            logger.warning("Empty data provided for feature engineering")
            return pd.DataFrame()
            
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns: {required_columns}")
            return pd.DataFrame()
        
        logger.info(f"Creating features for {len(df)} data points")
        
        # Use backend modules if available, otherwise fallback to simplified
        if BACKEND_AVAILABLE and self.technical_indicators:
            df = self._create_features_with_backend(df)
        else:
            df = self._create_features_simplified(df)
        
        # Remove any infinite or NaN values
        df = self._clean_features(df)
        
        logger.info(f"Feature engineering completed. Final shape: {df.shape}")
        return df
    
    def _create_features_with_backend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features using robust backend modules."""
        logger.info("Using backend technical indicators for feature engineering")
        
        try:
            # Get all technical indicators from backend
            indicators = self.technical_indicators.calculate_all_indicators_optimized(df)
            
            # Add key indicators as features
            if 'moving_averages' in indicators:
                ma = indicators['moving_averages']
                df['sma_20'] = ma.get('sma_20', np.nan)
                df['sma_50'] = ma.get('sma_50', np.nan)
                df['sma_200'] = ma.get('sma_200', np.nan)
                df['ema_20'] = ma.get('ema_20', np.nan)
                df['ema_50'] = ma.get('ema_50', np.nan)
                df['price_to_sma_200'] = ma.get('price_to_sma_200', np.nan)
                df['sma_20_to_sma_50'] = ma.get('sma_20_to_sma_50', np.nan)
            
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                df['rsi_14'] = rsi.get('rsi_14', np.nan)
                df['rsi_trend'] = 1 if rsi.get('trend') == 'up' else -1 if rsi.get('trend') == 'down' else 0
            
            if 'macd' in indicators:
                macd = indicators['macd']
                df['macd_line'] = macd.get('macd_line', np.nan)
                df['macd_signal'] = macd.get('signal_line', np.nan)
                df['macd_histogram'] = macd.get('histogram', np.nan)
            
            if 'bollinger_bands' in indicators:
                bb = indicators['bollinger_bands']
                df['bb_upper'] = bb.get('upper_band', np.nan)
                df['bb_middle'] = bb.get('middle_band', np.nan)
                df['bb_lower'] = bb.get('lower_band', np.nan)
                df['bb_percent_b'] = bb.get('percent_b', np.nan)
                df['bb_bandwidth'] = bb.get('bandwidth', np.nan)
            
            if 'volume' in indicators:
                vol = indicators['volume']
                df['volume_ratio'] = vol.get('volume_ratio', np.nan)
                df['obv'] = vol.get('obv', np.nan)
                df['obv_trend'] = 1 if vol.get('obv_trend') == 'up' else -1 if vol.get('obv_trend') == 'down' else 0
            
            if 'adx' in indicators:
                adx = indicators['adx']
                df['adx'] = adx.get('adx', np.nan)
                df['plus_di'] = adx.get('plus_di', np.nan)
                df['minus_di'] = adx.get('minus_di', np.nan)
                df['trend_direction'] = 1 if adx.get('trend_direction') == 'bullish' else -1
            
            if 'volatility' in indicators:
                vol = indicators['volatility']
                df['atr'] = vol.get('atr', np.nan)
                df['volatility_ratio'] = vol.get('volatility_ratio', np.nan)
                df['bb_squeeze'] = 1 if vol.get('bb_squeeze') else 0
            
            # Add basic price features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['price_range'] = (df['high'] - df['low']) / df['close']
            
            logger.info("Backend feature engineering completed successfully")
            
        except Exception as e:
            logger.error(f"Backend feature engineering failed: {e}, falling back to simplified")
            df = self._create_features_simplified(df)
        
        return df
    
    def _create_features_simplified(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic features when backend is not available."""
        logger.info("Using simplified feature engineering (backend not available)")
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_range'] = (df['high'] - df['low']) / df['close']
        
        # Simple moving averages
        for period in [5, 10, 20]:
            if len(df) >= period:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                df[f'price_vs_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
        
        # Simple RSI
        if len(df) >= 14:
            df['rsi_14'] = self._calculate_simple_rsi(df['close'], 14)
        
        # Simple volume features
        if len(df) >= 20:
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        return df
    
    def _calculate_simple_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Simple RSI calculation for fallback."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean features by removing infinite and NaN values."""
        logger.info("Cleaning features...")
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Get feature columns (excluding original OHLCV data)
        feature_columns = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        if not feature_columns:
            logger.warning("No feature columns found, returning original data")
            return df
        
        # Handle NaN values more gracefully
        for col in feature_columns:
            if col in df.columns:
                # For technical indicators, forward fill then fill remaining with 0
                if any(indicator in col.lower() for indicator in ['rsi', 'macd', 'bb', 'adx', 'atr', 'obv']):
                    df[col] = df[col].fillna(method='ffill').fillna(0)
                # For price-based features, fill with 0
                elif any(price_feature in col.lower() for price_feature in ['returns', 'log_returns', 'price_range', 'sma', 'ema']):
                    df[col] = df[col].fillna(0)
                # For volume features, fill with 1 (neutral ratio)
                elif 'volume' in col.lower():
                    df[col] = df[col].fillna(1)
                # For trend features, fill with 0 (neutral)
                elif 'trend' in col.lower():
                    df[col] = df[col].fillna(0)
                # Default: fill with 0
                else:
                    df[col] = df[col].fillna(0)
        
        # Check if we still have any NaN values
        remaining_nans = df[feature_columns].isna().sum().sum()
        if remaining_nans > 0:
            logger.warning(f"Still have {remaining_nans} NaN values after cleaning, filling with 0")
            df[feature_columns] = df[feature_columns].fillna(0)
        
        # Ensure we have data
        if len(df) == 0:
            logger.error("DataFrame is empty after cleaning, this should not happen")
            return df
        
        logger.info(f"Feature cleaning completed. Final shape: {df.shape}")
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names for ML training."""
        return [
            'sma_20', 'sma_50', 'sma_200', 'ema_20', 'ema_50',
            'price_to_sma_200', 'sma_20_to_sma_50',
            'rsi_14', 'rsi_trend',
            'macd_line', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_percent_b', 'bb_bandwidth',
            'volume_ratio', 'obv', 'obv_trend',
            'adx', 'plus_di', 'minus_di', 'trend_direction',
            'atr', 'volatility_ratio', 'bb_squeeze',
            'returns', 'log_returns', 'price_range'
        ]

# Global instance
feature_engineer = FeatureEngineer()
