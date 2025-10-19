"""
Core Utilities for Quantitative Trading System

This module provides common utility functions used across all components
of the quantitative trading system.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import warnings
from functools import wraps
import time
import json
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration for the system."""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create logger
    logger = logging.getLogger('quant_system')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

def validate_dataframe(df: pd.DataFrame, required_columns: List[str], 
                      min_rows: int = 1) -> Tuple[bool, str]:
    """Validate DataFrame structure and content."""
    
    # Check if DataFrame is empty
    if df.empty:
        return False, "DataFrame is empty"
    
    # Check minimum number of rows
    if len(df) < min_rows:
        return False, f"DataFrame has {len(df)} rows, minimum {min_rows} required"
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    # Check for all NaN values
    if df.isnull().all().any():
        return False, "Some columns contain only NaN values"
    
    return True, "DataFrame validation passed"

def clean_dataframe(df: pd.DataFrame, 
                   fill_method: str = 'forward',
                   drop_threshold: float = 0.5) -> pd.DataFrame:
    """Clean DataFrame by handling missing values and outliers."""
    
    df_clean = df.copy()
    
    # Remove columns with too many missing values
    missing_ratio = df_clean.isnull().sum() / len(df_clean)
    columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    if len(columns_to_drop) > 0:
        logger.warning(f"Dropping columns with >{drop_threshold*100}% missing values: {list(columns_to_drop)}")
        df_clean = df_clean.drop(columns=columns_to_drop)
    
    # Fill missing values
    if fill_method == 'forward':
        df_clean = df_clean.fillna(method='ffill')
    elif fill_method == 'backward':
        df_clean = df_clean.fillna(method='bfill')
    elif fill_method == 'mean':
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_columns] = df_clean[numeric_columns].fillna(df_clean[numeric_columns].mean())
    elif fill_method == 'median':
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_columns] = df_clean[numeric_columns].fillna(df_clean[numeric_columns].median())
    
    # Remove remaining NaN values
    df_clean = df_clean.dropna()
    
    return df_clean

def calculate_technical_indicators(df: pd.DataFrame, 
                                 price_column: str = 'close',
                                 volume_column: str = 'volume') -> pd.DataFrame:
    """Calculate common technical indicators."""
    
    df_indicators = df.copy()
    
    # Price-based indicators
    df_indicators['sma_20'] = df_indicators[price_column].rolling(window=20).mean()
    df_indicators['sma_50'] = df_indicators[price_column].rolling(window=50).mean()
    df_indicators['ema_12'] = df_indicators[price_column].ewm(span=12).mean()
    df_indicators['ema_26'] = df_indicators[price_column].ewm(span=26).mean()
    
    # RSI
    delta = df_indicators[price_column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_indicators['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df_indicators['macd'] = df_indicators['ema_12'] - df_indicators['ema_26']
    df_indicators['macd_signal'] = df_indicators['macd'].ewm(span=9).mean()
    df_indicators['macd_histogram'] = df_indicators['macd'] - df_indicators['macd_signal']
    
    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    df_indicators['bb_middle'] = df_indicators[price_column].rolling(window=bb_period).mean()
    bb_std_dev = df_indicators[price_column].rolling(window=bb_period).std()
    df_indicators['bb_upper'] = df_indicators['bb_middle'] + (bb_std_dev * bb_std)
    df_indicators['bb_lower'] = df_indicators['bb_middle'] - (bb_std_dev * bb_std)
    df_indicators['bb_width'] = (df_indicators['bb_upper'] - df_indicators['bb_lower']) / df_indicators['bb_middle']
    df_indicators['bb_position'] = (df_indicators[price_column] - df_indicators['bb_lower']) / (df_indicators['bb_upper'] - df_indicators['bb_lower'])
    
    # Volume indicators
    if volume_column in df_indicators.columns:
        df_indicators['volume_sma'] = df_indicators[volume_column].rolling(window=20).mean()
        df_indicators['volume_ratio'] = df_indicators[volume_column] / df_indicators['volume_sma']
    
    # Price change indicators
    df_indicators['price_change'] = df_indicators[price_column].pct_change()
    df_indicators['price_change_abs'] = df_indicators['price_change'].abs()
    df_indicators['volatility'] = df_indicators['price_change'].rolling(window=20).std()
    
    return df_indicators

def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """Calculate returns from price series."""
    
    if method == 'simple':
        return prices.pct_change()
    elif method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unknown return method: {method}")

def calculate_volatility(returns: pd.Series, window: int = 20, 
                        annualized: bool = True) -> pd.Series:
    """Calculate rolling volatility."""
    
    volatility = returns.rolling(window=window).std()
    
    if annualized:
        # Assuming 252 trading days per year
        volatility = volatility * np.sqrt(252)
    
    return volatility

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio for a return series."""
    
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return excess_returns.mean() / returns.std() * np.sqrt(252)

def calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown from price series."""
    
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def normalize_features(features: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
    """Normalize features using specified method."""
    
    if method == 'minmax':
        return (features - features.min()) / (features.max() - features.min())
    elif method == 'zscore':
        return (features - features.mean()) / features.std()
    elif method == 'robust':
        median = features.median()
        mad = (features - median).abs().median()
        return (features - median) / mad
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def create_lagged_features(data: pd.DataFrame, columns: List[str], 
                          lags: List[int]) -> pd.DataFrame:
    """Create lagged features for time series data."""
    
    result = data.copy()
    for col in columns:
        for lag in lags:
            result[f"{col}_lag_{lag}"] = data[col].shift(lag)
    return result

def create_rolling_features(data: pd.DataFrame, columns: List[str], 
                           windows: List[int], functions: List[str] = ['mean', 'std']) -> pd.DataFrame:
    """Create rolling window features."""
    
    result = data.copy()
    for col in columns:
        for window in windows:
            for func in functions:
                if func == 'mean':
                    result[f"{col}_rolling_{func}_{window}"] = data[col].rolling(window=window).mean()
                elif func == 'std':
                    result[f"{col}_rolling_{func}_{window}"] = data[col].rolling(window=window).std()
                elif func == 'min':
                    result[f"{col}_rolling_{func}_{window}"] = data[col].rolling(window=window).min()
                elif func == 'max':
                    result[f"{col}_rolling_{func}_{window}"] = data[col].rolling(window=window).max()
                elif func == 'sum':
                    result[f"{col}_rolling_{func}_{window}"] = data[col].rolling(window=window).sum()
    return result

def save_object(obj: Any, filepath: str) -> bool:
    """Save object to file using appropriate method."""
    
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(obj, f, indent=2, default=str)
        elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
        else:
            # Default to pickle
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
        
        logger.info(f"Object saved to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save object to {filepath}: {e}")
        return False

def load_object(filepath: str) -> Optional[Any]:
    """Load object from file using appropriate method."""
    
    try:
        if not Path(filepath).exists():
            logger.warning(f"File not found: {filepath}")
            return None
        
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                return json.load(f)
        elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            # Try pickle first, then JSON
            try:
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            except:
                with open(filepath, 'r') as f:
                    return json.load(f)
        
    except Exception as e:
        logger.error(f"Failed to load object from {filepath}: {e}")
        return None

def format_currency(amount: float, currency: str = 'USD') -> str:
    """Format currency amount for display."""
    
    if currency == 'USD':
        return f"${amount:,.2f}"
    elif currency == 'INR':
        return f"₹{amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage value for display."""
    
    return f"{value * 100:.{decimals}f}%"

def get_trading_days(start_date: datetime, end_date: datetime) -> List[datetime]:
    """Get list of trading days between start and end dates."""
    
    # Simple implementation - excludes weekends
    # In practice, you might want to exclude holidays as well
    trading_days = []
    current_date = start_date
    
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Monday = 0, Sunday = 6
            trading_days.append(current_date)
        current_date += timedelta(days=1)
    
    return trading_days

def suppress_warnings(func):
    """Decorator to suppress warnings in a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapper

def memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    import psutil
    import os
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def log_memory_usage(logger: logging.Logger, context: str = ""):
    """Log current memory usage."""
    usage = memory_usage_mb()
    logger.info(f"Memory usage {context}: {usage:.2f} MB")

# Constants
TRADING_DAYS_PER_YEAR = 252
HOURS_PER_DAY = 24
MINUTES_PER_HOUR = 60
SECONDS_PER_MINUTE = 60

# Common time periods
TIME_PERIODS = {
    '1m': 1,
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1h': 60,
    '4h': 240,
    '1d': 1440,
    '1w': 10080
}
