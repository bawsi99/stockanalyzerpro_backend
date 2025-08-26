"""
Enhanced Data Pipeline for Multi-Timeframe Analysis

This module provides enhanced data loading and synchronization capabilities
for multi-timeframe quantitative analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zerodha_client import ZerodhaDataClient

logger = logging.getLogger(__name__)

@dataclass
class TimeframeConfig:
    """Configuration for timeframe-specific settings."""
    name: str
    interval: str
    min_data_points: int
    weight: float
    description: str

class EnhancedDataPipeline:
    """Enhanced data pipeline with multi-timeframe support."""
    
    # Predefined timeframe configurations
    TIMEFRAME_CONFIGS = {
        "1min": TimeframeConfig("1min", "1minute", 1440, 0.05, "Ultra-short term intraday"),
        "5min": TimeframeConfig("5min", "5minute", 288, 0.15, "Short term intraday"),
        "15min": TimeframeConfig("15min", "15minute", 96, 0.20, "Medium term intraday"),
        "30min": TimeframeConfig("30min", "30minute", 48, 0.20, "Long term intraday"),
        "1hour": TimeframeConfig("1hour", "60minute", 24, 0.25, "Swing trading"),
        "1day": TimeframeConfig("1day", "day", 252, 0.15, "Position trading")
    }
    
    def __init__(self, timeframes: Optional[List[str]] = None):
        """
        Initialize enhanced data pipeline.
        
        Args:
            timeframes: List of timeframes to support. Defaults to all available.
        """
        self.timeframes = timeframes or list(self.TIMEFRAME_CONFIGS.keys())
        self.data_cache = {}
        self.zerodha_client = ZerodhaDataClient()
        self.synchronization_manager = TimeframeSynchronizer()
        
        logger.info(f"Enhanced data pipeline initialized with timeframes: {self.timeframes}")
    
    def load_multi_timeframe_data(self, symbol: str, start_date: datetime, 
                                 end_date: datetime, timeframes: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple timeframes.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data loading
            end_date: End date for data loading
            timeframes: Specific timeframes to load. Defaults to all configured timeframes.
            
        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        timeframes = timeframes or self.timeframes
        multi_tf_data = {}
        
        logger.info(f"Loading multi-timeframe data for {symbol} from {start_date} to {end_date}")
        
        for timeframe in timeframes:
            if timeframe not in self.TIMEFRAME_CONFIGS:
                logger.warning(f"Unknown timeframe: {timeframe}")
                continue
                
            try:
                config = self.TIMEFRAME_CONFIGS[timeframe]
                data = self._load_single_timeframe_data(symbol, start_date, end_date, config)
                
                if not data.empty:
                    multi_tf_data[timeframe] = data
                    logger.info(f"Loaded {len(data)} data points for {timeframe}")
                else:
                    logger.warning(f"No data loaded for {timeframe}")
                    
            except Exception as e:
                logger.error(f"Error loading data for {timeframe}: {e}")
                continue
        
        # Synchronize data across timeframes
        if len(multi_tf_data) > 1:
            multi_tf_data = self.synchronization_manager.align_timeframes(multi_tf_data)
        
        return multi_tf_data
    
    def _load_single_timeframe_data(self, symbol: str, start_date: datetime, 
                                   end_date: datetime, config: TimeframeConfig) -> pd.DataFrame:
        """Load data for a single timeframe."""
        try:
            data = self.zerodha_client.get_historical_data(
                symbol=symbol,
                exchange="NSE",
                interval=config.interval,
                from_date=start_date,
                to_date=end_date
            )
            
            if not data.empty:
                # Standardize column names
                data.columns = [col.lower() for col in data.columns]
                data.index.name = 'datetime'
                
                # Sort by datetime
                data = data.sort_index()
                
                # Validate data quality
                if self._validate_data_quality(data, config):
                    return data
                else:
                    logger.warning(f"Data quality validation failed for {config.name}")
                    return pd.DataFrame()
            else:
                logger.warning(f"No data returned for {config.name}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading {config.name} data: {e}")
            return pd.DataFrame()
    
    def _validate_data_quality(self, data: pd.DataFrame, config: TimeframeConfig) -> bool:
        """Validate data quality for a specific timeframe."""
        if data.empty:
            return False
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Missing required columns for {config.name}")
            return False
        
        # Check for minimum data points
        if len(data) < config.min_data_points:
            logger.warning(f"Insufficient data points for {config.name}: {len(data)} < {config.min_data_points}")
            return False
        
        # Check for missing values
        missing_values = data[required_columns].isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Missing values in {config.name}: {missing_values.to_dict()}")
        
        # Check for negative prices
        if (data[['open', 'high', 'low', 'close']] <= 0).any().any():
            logger.error(f"Negative or zero prices found in {config.name}")
            return False
        
        return True
    
    def create_timeframe_features(self, multi_tf_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Create timeframe-specific features from multi-timeframe data.
        
        Args:
            multi_tf_data: Dictionary of timeframe data
            
        Returns:
            Dictionary of timeframe-specific features
        """
        timeframe_features = {}
        
        for timeframe, data in multi_tf_data.items():
            if data.empty:
                continue
                
            try:
                features = self._create_single_timeframe_features(data, timeframe)
                if not features.empty:
                    timeframe_features[timeframe] = features
                    
            except Exception as e:
                logger.error(f"Error creating features for {timeframe}: {e}")
                continue
        
        return timeframe_features
    
    def _create_single_timeframe_features(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Create features for a single timeframe."""
        features = data.copy()
        
        # Basic technical indicators
        features['returns'] = features['close'].pct_change()
        features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = features['close'].rolling(window=period, min_periods=period//2).mean()
            features[f'ema_{period}'] = features['close'].ewm(span=period).mean()
        
        # Volatility indicators
        features['atr'] = self._calculate_atr(features, period=14)
        features['volatility'] = features['returns'].rolling(window=20).std()
        
        # Volume indicators
        features['volume_sma'] = features['volume'].rolling(window=20).mean()
        features['volume_ratio'] = features['volume'] / features['volume_sma']
        
        # Timeframe-specific features
        if timeframe in ['5min', '15min', '30min']:
            features = self._add_intraday_features(features)
        elif timeframe in ['1hour', '1day']:
            features = self._add_swing_features(features)
        
        # Remove NaN values
        features = features.dropna()
        
        return features
    
    def _add_intraday_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add intraday-specific features."""
        features = data.copy()
        
        # Intraday momentum
        features['intraday_momentum'] = (features['close'] - features['open']) / features['open']
        
        # Time-based features (if datetime index is available)
        if hasattr(features.index, 'time'):
            features['hour'] = features.index.hour
            features['minute'] = features.index.minute
            features['is_market_open'] = ((features['hour'] >= 9) & (features['hour'] < 15)).astype(int)
        
        return features
    
    def _add_swing_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add swing trading features."""
        features = data.copy()
        
        # Trend indicators
        features['trend_strength'] = abs(features['close'] - features['sma_20']) / features['sma_20']
        
        # Support/resistance levels
        features['support_level'] = features['low'].rolling(window=20).min()
        features['resistance_level'] = features['high'].rolling(window=20).max()
        
        return features
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def get_timeframe_weights(self) -> Dict[str, float]:
        """Get configured weights for each timeframe."""
        return {tf: self.TIMEFRAME_CONFIGS[tf].weight for tf in self.timeframes 
                if tf in self.TIMEFRAME_CONFIGS}
    
    def cache_data(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Cache data for future use."""
        cache_key = f"{symbol}_{timeframe}"
        self.data_cache[cache_key] = data
        logger.info(f"Cached data for {cache_key}")
    
    def get_cached_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get cached data if available."""
        cache_key = f"{symbol}_{timeframe}"
        return self.data_cache.get(cache_key)

class TimeframeSynchronizer:
    """Synchronizes data across different timeframes."""
    
    def align_timeframes(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align data from different timeframes to common timestamps.
        
        Args:
            data_dict: Dictionary mapping timeframe to DataFrame
            
        Returns:
            Aligned data dictionary
        """
        if len(data_dict) <= 1:
            return data_dict
        
        # Find the highest frequency timeframe (most data points)
        highest_freq_tf = max(data_dict.keys(), key=lambda x: len(data_dict[x]))
        reference_data = data_dict[highest_freq_tf]
        
        aligned_data = {highest_freq_tf: reference_data}
        
        for timeframe, data in data_dict.items():
            if timeframe == highest_freq_tf:
                continue
            
            try:
                aligned_data[timeframe] = self._align_to_reference(data, reference_data)
                logger.info(f"Aligned {timeframe} data to {highest_freq_tf}")
                
            except Exception as e:
                logger.error(f"Error aligning {timeframe}: {e}")
                aligned_data[timeframe] = data  # Keep original if alignment fails
        
        return aligned_data
    
    def _align_to_reference(self, data: pd.DataFrame, reference_data: pd.DataFrame) -> pd.DataFrame:
        """Align data to reference timestamps using forward fill."""
        # Reindex to reference timestamps
        aligned_data = data.reindex(reference_data.index, method='ffill')
        
        # Remove any remaining NaN values
        aligned_data = aligned_data.dropna()
        
        return aligned_data

# Global instance for easy access
enhanced_data_pipeline = EnhancedDataPipeline()

