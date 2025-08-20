"""
Data Pipeline for Quantitative Trading System

This module provides the foundation for loading and managing OHLCV data
across multiple timeframes for quantitative analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zerodha_client import ZerodhaDataClient

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for data loading."""
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    min_data_points: int = 252  # Minimum data points required

class OHLCVData:
    """Core data structure for OHLCV data management."""
    
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = pd.DataFrame()
        self.zerodha_client = ZerodhaDataClient()
        
    def load(self, start_date: datetime, end_date: datetime) -> 'OHLCVData':
        """Load historical OHLCV data."""
        try:
            logger.info(f"Loading data for {self.symbol} ({self.timeframe}) from {start_date} to {end_date}")
            
            # Use existing Zerodha client with correct method signature
            self.data = self.zerodha_client.get_historical_data(
                symbol=self.symbol,
                exchange="NSE",
                interval=self.timeframe,
                from_date=start_date,
                to_date=end_date
            )
            
            # Ensure proper column names
            if not self.data.empty:
                self.data.columns = [col.lower() for col in self.data.columns]
                self.data.index.name = 'datetime'
                
                # Sort by datetime
                self.data = self.data.sort_index()
                
                logger.info(f"Loaded {len(self.data)} data points for {self.symbol}")
            else:
                logger.warning(f"No data loaded for {self.symbol}")
                
        except Exception as e:
            logger.error(f"Error loading data for {self.symbol}: {e}")
            self.data = pd.DataFrame()
            
        return self
    
    def validate_data(self) -> bool:
        """Validate data quality."""
        if self.data.empty:
            return False
            
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in self.data.columns for col in required_columns):
            logger.error(f"Missing required columns: {required_columns}")
            return False
            
        # Check for missing values
        missing_values = self.data[required_columns].isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Missing values found: {missing_values.to_dict()}")
            
        # Check for negative prices
        if (self.data[['open', 'high', 'low', 'close']] <= 0).any().any():
            logger.error("Negative or zero prices found")
            return False
            
        # Check for logical price relationships
        if not self._validate_price_logic():
            logger.error("Price logic validation failed")
            return False
            
        return True
    
    def _validate_price_logic(self) -> bool:
        """Validate that high >= low, high >= open, high >= close, etc."""
        data = self.data
        
        # Check each row individually to avoid alignment issues
        for idx in data.index:
            row = data.loc[idx]
            high = row['high']
            low = row['low']
            open_price = row['open']
            close = row['close']
            
            # High should be >= all other prices
            if not (high >= open_price and high >= low and high >= close):
                return False
                
            # Low should be <= all other prices
            if not (low <= open_price and low <= close):
                return False
                
        return True
    
    def get_latest_price(self) -> Optional[float]:
        """Get the latest closing price."""
        if not self.data.empty:
            return self.data['close'].iloc[-1]
        return None
    
    def get_data_range(self) -> Tuple[datetime, datetime]:
        """Get the date range of the data."""
        if self.data.empty:
            return None, None
        return self.data.index[0], self.data.index[-1]

class MultiTimeframeDataManager:
    """Manages data across multiple timeframes."""
    
    def __init__(self, symbol: str, timeframes: List[str]):
        self.symbol = symbol
        self.timeframes = timeframes
        self.data_containers = {}
        
    def load_all_timeframes(self, start_date: datetime, end_date: datetime) -> Dict[str, OHLCVData]:
        """Load data for all timeframes."""
        for timeframe in self.timeframes:
            data_container = OHLCVData(self.symbol, timeframe)
            data_container.load(start_date, end_date)
            
            if data_container.validate_data():
                self.data_containers[timeframe] = data_container
                logger.info(f"Successfully loaded {timeframe} data for {self.symbol}")
            else:
                logger.error(f"Failed to validate {timeframe} data for {self.symbol}")
                
        return self.data_containers
    
    def get_timeframe_data(self, timeframe: str) -> Optional[pd.DataFrame]:
        """Get data for a specific timeframe."""
        if timeframe in self.data_containers:
            return self.data_containers[timeframe].data
        return None
    
    def get_latest_prices(self) -> Dict[str, float]:
        """Get latest prices for all timeframes."""
        prices = {}
        for timeframe, container in self.data_containers.items():
            price = container.get_latest_price()
            if price is not None:
                prices[timeframe] = price
        return prices

# Global data manager instance
data_manager = None
