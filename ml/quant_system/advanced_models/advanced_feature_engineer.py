"""
Advanced Feature Engineering for Enhanced Trading System

This module provides advanced feature engineering capabilities including:
1. Market microstructure features (order book dynamics)
2. Cross-asset correlation modeling
3. Volatility surface analysis
4. Cross-domain feature extraction
5. Alternative data integration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration for advanced feature engineering."""
    
    # Technical indicators
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    
    # Volatility features
    volatility_windows: List[int] = None
    garch_p: int = 1
    garch_q: int = 1
    
    # Cross-asset features
    correlation_window: int = 60
    beta_window: int = 252
    sector_correlation: bool = True
    
    # Market microstructure
    order_flow_window: int = 20
    imbalance_threshold: float = 0.1
    
    # Alternative data
    sentiment_window: int = 5
    news_impact_window: int = 3
    
    # Data quality thresholds
    missing_threshold: float = 0.1
    
    def __post_init__(self):
        if self.volatility_windows is None:
            self.volatility_windows = [5, 10, 20, 60]

class AdvancedFeatureEngineer:
    """Advanced feature engineering for enhanced trading system."""
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.feature_cache = {}
        self.correlation_matrix = None
        self.sector_mapping = {}
        
    def create_all_features(self, price_data: pd.DataFrame, 
                           volume_data: pd.DataFrame = None,
                           news_data: pd.DataFrame = None,
                           social_data: pd.DataFrame = None,
                           market_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create comprehensive feature set for advanced trading.
        
        Args:
            price_data: OHLCV price data
            volume_data: Detailed volume data (optional)
            news_data: News sentiment data (optional)
            social_data: Social media sentiment data (optional)
            market_data: Market index data for cross-asset features (optional)
        
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Creating comprehensive feature set...")
        
        # Start with basic technical indicators
        features = self._create_technical_indicators(price_data)
        
        # Add market microstructure features
        if volume_data is not None:
            microstructure_features = self._create_microstructure_features(price_data, volume_data)
            features = pd.concat([features, microstructure_features], axis=1)
        
        # Add volatility features
        volatility_features = self._create_volatility_features(price_data)
        features = pd.concat([features, volatility_features], axis=1)
        
        # Add cross-asset features
        if market_data is not None:
            cross_asset_features = self._create_cross_asset_features(price_data, market_data)
            features = pd.concat([features, cross_asset_features], axis=1)
        
        # Add alternative data features
        if news_data is not None:
            news_features = self._create_news_features(news_data)
            features = pd.concat([features, news_features], axis=1)
        
        if social_data is not None:
            social_features = self._create_social_features(social_data)
            features = pd.concat([features, social_features], axis=1)
        
        # Add cross-domain features
        cross_domain_features = self._create_cross_domain_features(features)
        features = pd.concat([features, cross_domain_features], axis=1)
        
        # Clean and validate features
        features = self._clean_features(features)
        
        logger.info(f"Created {features.shape[1]} features for {features.shape[0]} samples")
        return features
    
    def _create_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical indicators."""
        features = pd.DataFrame(index=data.index)
        
        # Basic price features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['high_low_ratio'] = data['high'] / data['low']
        features['close_open_ratio'] = data['close'] / data['open']
        features['body_size'] = abs(data['close'] - data['open']) / data['open']
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            features[f'sma_{period}'] = data['close'].rolling(window=period, min_periods=period//2).mean()
            features[f'ema_{period}'] = data['close'].ewm(span=period).mean()
            features[f'price_sma_{period}_ratio'] = data['close'] / features[f'sma_{period}']
            features[f'price_ema_{period}_ratio'] = data['close'] / features[f'ema_{period}']
        
        # RSI
        features['rsi'] = self._calculate_rsi(data['close'], self.config.rsi_period)
        
        # MACD
        macd_line, signal_line, macd_histogram = self._calculate_macd(data['close'])
        features['macd_line'] = macd_line
        features['macd_signal'] = signal_line
        features['macd_histogram'] = macd_histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(data['close'])
        features['bb_upper'] = bb_upper
        features['bb_middle'] = bb_middle
        features['bb_lower'] = bb_lower
        features['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
        features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # ATR and volatility
        features['atr'] = self._calculate_atr(data, self.config.atr_period)
        features['volatility'] = features['returns'].rolling(window=20).std()
        
        # Volume features
        if 'volume' in data.columns:
            features['volume_sma'] = data['volume'].rolling(window=20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_sma']
            features['volume_price_trend'] = (data['volume'] * features['returns']).rolling(window=20).sum()
        
        # Momentum indicators
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
            features[f'roc_{period}'] = (data['close'] - data['close'].shift(period)) / data['close'].shift(period)
        
        # Support and resistance
        features['support_level'] = data['low'].rolling(window=20).min()
        features['resistance_level'] = data['high'].rolling(window=20).max()
        features['support_distance'] = (data['close'] - features['support_level']) / data['close']
        features['resistance_distance'] = (features['resistance_level'] - data['close']) / data['close']
        
        return features
    
    def _create_microstructure_features(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features."""
        features = pd.DataFrame(index=price_data.index)
        
        # Order flow imbalance
        if 'bid_volume' in volume_data.columns and 'ask_volume' in volume_data.columns:
            features['order_imbalance'] = (volume_data['bid_volume'] - volume_data['ask_volume']) / (volume_data['bid_volume'] + volume_data['ask_volume'])
            features['order_imbalance_ma'] = features['order_imbalance'].rolling(window=self.config.order_flow_window).mean()
            features['order_imbalance_std'] = features['order_imbalance'].rolling(window=self.config.order_flow_window).std()
        
        # Volume profile
        if 'volume' in volume_data.columns:
            features['volume_vwap'] = (volume_data['volume'] * price_data['close']).rolling(window=20).sum() / volume_data['volume'].rolling(window=20).sum()
            features['volume_price_divergence'] = (price_data['close'] - features['volume_vwap']) / features['volume_vwap']
        
        # Bid-ask spread (if available)
        if 'bid' in price_data.columns and 'ask' in price_data.columns:
            features['bid_ask_spread'] = (price_data['ask'] - price_data['bid']) / price_data['bid']
            features['spread_ma'] = features['bid_ask_spread'].rolling(window=20).mean()
        
        # Market impact
        features['price_impact'] = features['returns'].abs() / (volume_data['volume'] if 'volume' in volume_data.columns else price_data['volume'])
        
        return features
    
    def _create_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create advanced volatility features."""
        features = pd.DataFrame(index=data.index)
        
        # Multiple timeframe volatility
        for window in self.config.volatility_windows:
            features[f'volatility_{window}d'] = data['close'].pct_change().rolling(window=window).std()
            features[f'log_volatility_{window}d'] = np.log(features[f'volatility_{window}d'])
        
        # Volatility ratios
        features['volatility_ratio_5_20'] = features['volatility_5d'] / features['volatility_20d']
        features['volatility_ratio_10_60'] = features['volatility_10d'] / features['volatility_60d']
        
        # Realized volatility
        features['realized_volatility'] = np.sqrt((data['close'].pct_change() ** 2).rolling(window=20).sum())
        
        # Parkinson volatility (high-low based)
        features['parkinson_volatility'] = np.sqrt((np.log(data['high'] / data['low']) ** 2).rolling(window=20).sum() / (4 * np.log(2)))
        
        # Garman-Klass volatility
        features['garman_klass_volatility'] = np.sqrt(((0.5 * np.log(data['high'] / data['low']) ** 2) - 
                                                      ((2 * np.log(2) - 1) * np.log(data['close'] / data['open']) ** 2)).rolling(window=20).sum())
        
        # Volatility clustering (simplified version)
        features['volatility_clustering'] = features['volatility_20d'].rolling(window=60).corr(features['volatility_20d'].shift(1))
        
        return features
    
    def _create_cross_asset_features(self, asset_data: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create cross-asset correlation and beta features."""
        features = pd.DataFrame(index=asset_data.index)
        
        # Calculate returns
        asset_returns = asset_data['close'].pct_change()
        market_returns = market_data['close'].pct_change()
        
        # Rolling correlation
        features['market_correlation'] = asset_returns.rolling(window=self.config.correlation_window).corr(market_returns)
        
        # Rolling beta
        features['market_beta'] = asset_returns.rolling(window=self.config.beta_window).cov(market_returns) / market_returns.rolling(window=self.config.beta_window).var()
        
        # Alpha (excess return)
        features['alpha'] = asset_returns - features['market_beta'] * market_returns
        
        # Information ratio
        features['information_ratio'] = features['alpha'].rolling(window=60).mean() / features['alpha'].rolling(window=60).std()
        
        # Relative strength
        features['relative_strength'] = asset_data['close'] / market_data['close']
        features['relative_strength_ma'] = features['relative_strength'].rolling(window=20).mean()
        
        # Sector correlation (if sector data available)
        if self.config.sector_correlation and 'sector' in market_data.columns:
            sector_features = self._create_sector_features(asset_data, market_data)
            features = pd.concat([features, sector_features], axis=1)
        
        return features
    
    def _create_sector_features(self, asset_data: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create sector-specific features."""
        features = pd.DataFrame(index=asset_data.index)
        
        # This would require sector classification data
        # For now, create placeholder features
        features['sector_momentum'] = 0.0  # Placeholder
        features['sector_rotation'] = 0.0  # Placeholder
        
        return features
    
    def _create_news_features(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """Create news sentiment features."""
        features = pd.DataFrame(index=news_data.index)
        
        if 'sentiment' in news_data.columns:
            # Sentiment features
            features['news_sentiment'] = news_data['sentiment']
            features['news_sentiment_ma'] = news_data['sentiment'].rolling(window=self.config.sentiment_window).mean()
            features['news_sentiment_std'] = news_data['sentiment'].rolling(window=self.config.sentiment_window).std()
            
            # Sentiment momentum
            features['sentiment_momentum'] = news_data['sentiment'] - features['news_sentiment_ma']
            
            # Sentiment surprise
            features['sentiment_surprise'] = news_data['sentiment'] - news_data['sentiment'].shift(1)
        
        if 'volume' in news_data.columns:
            # News volume features
            features['news_volume'] = news_data['volume']
            features['news_volume_ma'] = news_data['volume'].rolling(window=self.config.news_impact_window).mean()
            features['news_volume_ratio'] = news_data['volume'] / features['news_volume_ma']
        
        return features
    
    def _create_social_features(self, social_data: pd.DataFrame) -> pd.DataFrame:
        """Create social media sentiment features."""
        features = pd.DataFrame(index=social_data.index)
        
        if 'sentiment' in social_data.columns:
            # Social sentiment features
            features['social_sentiment'] = social_data['sentiment']
            features['social_sentiment_ma'] = social_data['sentiment'].rolling(window=self.config.sentiment_window).mean()
            features['social_sentiment_std'] = social_data['sentiment'].rolling(window=self.config.sentiment_window).std()
            
            # Sentiment divergence
            features['sentiment_divergence'] = social_data['sentiment'] - features['social_sentiment_ma']
        
        if 'volume' in social_data.columns:
            # Social volume features
            features['social_volume'] = social_data['volume']
            features['social_volume_ma'] = social_data['volume'].rolling(window=self.config.sentiment_window).mean()
            features['social_volume_ratio'] = social_data['volume'] / features['social_volume_ma']
        
        return features
    
    def _create_cross_domain_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create cross-domain inspired features."""
        cross_domain_features = pd.DataFrame(index=features.index)
        
        # Weather forecasting inspired features (ensemble-like)
        cross_domain_features['ensemble_volatility'] = features[['volatility_5d', 'volatility_10d', 'volatility_20d']].mean(axis=1)
        cross_domain_features['ensemble_volatility_std'] = features[['volatility_5d', 'volatility_10d', 'volatility_20d']].std(axis=1)
        
        # Epidemiology inspired features (contagion-like)
        cross_domain_features['price_contagion'] = features['returns'].rolling(window=5).apply(lambda x: (x > 0).sum() / len(x))
        cross_domain_features['volume_contagion'] = features['volume_ratio'].rolling(window=5).apply(lambda x: (x > 1).sum() / len(x))
        
        # Neuroscience inspired features (spike-like)
        cross_domain_features['price_spikes'] = (features['returns'].abs() > features['returns'].rolling(window=20).std() * 2).astype(int)
        cross_domain_features['spike_frequency'] = cross_domain_features['price_spikes'].rolling(window=20).sum()
        
        # Quantum inspired features (superposition-like)
        cross_domain_features['price_superposition'] = features['close_open_ratio'] * features['high_low_ratio']
        # Only create volume_superposition if price_impact exists
        if 'price_impact' in features.columns and 'volume_ratio' in features.columns:
            cross_domain_features['volume_superposition'] = features['volume_ratio'] * features['price_impact']
        else:
            cross_domain_features['volume_superposition'] = features.get('volume_ratio', 1.0) * features.get('returns', 0.0)
        
        return cross_domain_features
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features."""
        # Remove infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill missing values
        features = features.fillna(method='ffill')
        
        # Backward fill remaining missing values
        features = features.fillna(method='bfill')
        
        # Remove features with too many missing values
        missing_threshold = self.config.missing_threshold if hasattr(self.config, 'missing_threshold') else 0.1
        features = features.loc[:, features.isnull().sum() / len(features) < missing_threshold]
        
        # Remove constant features
        features = features.loc[:, features.std() > 0]
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=self.config.macd_fast).mean()
        ema_slow = prices.ewm(span=self.config.macd_slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.config.macd_signal).mean()
        macd_histogram = macd_line - signal_line
        return macd_line, signal_line, macd_histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=self.config.bb_period).mean()
        std = prices.rolling(window=self.config.bb_period).std()
        upper = middle + (std * self.config.bb_std)
        lower = middle - (std * self.config.bb_std)
        return upper, middle, lower
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def get_feature_importance(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """Calculate feature importance using correlation."""
        importance = {}
        for col in features.columns:
            if col in features.columns and not features[col].isnull().all():
                corr = abs(features[col].corr(target))
                importance[col] = corr if not np.isnan(corr) else 0.0
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return importance
    
    def select_top_features(self, features: pd.DataFrame, target: pd.Series, top_k: int = 50) -> pd.DataFrame:
        """Select top-k most important features."""
        importance = self.get_feature_importance(features, target)
        top_features = list(importance.keys())[:top_k]
        return features[top_features]

# Global instance for easy access
advanced_feature_engineer = AdvancedFeatureEngineer()
