import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Callable
import matplotlib.dates as mdates
import os
import logging
import asyncio

# Add new imports for pattern modules
from patterns.recognition import PatternRecognition
from patterns.visualization import PatternVisualizer

# Add new imports for configuration and optimization
from config import Config
from cache_manager import cached, monitor_performance
from zerodha_client import ZerodhaDataClient

# Add to existing IndianMarketMetricsProvider class
from sector_classifier import sector_classifier


class TechnicalIndicators:
    """
    Class for calculating various technical indicators used in stock analysis.
    """
    
    @staticmethod
    def get_available_indicators() -> Dict[str, Callable]:
        """
        Returns a dictionary of available technical indicators and their calculation functions.
        
        Returns:
            Dict[str, Callable]: Dictionary mapping indicator names to their calculation functions
        """
        return {
            'SMA': lambda data: TechnicalIndicators.calculate_sma(data),
            'EMA': lambda data: TechnicalIndicators.calculate_ema(data),
            'WMA': lambda data: TechnicalIndicators.calculate_wma(data),
            'MACD': lambda data: TechnicalIndicators.calculate_macd(data),
            'RSI': lambda data: TechnicalIndicators.calculate_rsi(data),
            'Bollinger Bands': lambda data: TechnicalIndicators.calculate_bollinger_bands(data),
            'ATR': lambda data: TechnicalIndicators.calculate_atr(data),
            'Stochastic': lambda data: TechnicalIndicators.calculate_stochastic_oscillator(data),
            'OBV': lambda data: TechnicalIndicators.calculate_obv(data),
            'ADX': lambda data: TechnicalIndicators.calculate_adx(data),
            'Ichimoku': lambda data: TechnicalIndicators.calculate_ichimoku(data),
            'Fibonacci': lambda data: TechnicalIndicators.calculate_fibonacci_retracement(data),
            'Pivot Points': lambda data: TechnicalIndicators.calculate_pivot_points(data),
            'Support/Resistance': lambda data: TechnicalIndicators.detect_support_resistance(data)
        }
    
    @staticmethod
    def calculate_sma(data: pd.DataFrame, column: str = 'close', window: int = 20) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            data: DataFrame containing price data
            column: Column name to use for calculation
            window: Window period for moving average
            
        Returns:
            pd.Series: Simple Moving Average values
        """
        return data[column].rolling(window=window).mean()
    
    @staticmethod
    def calculate_ema(data: pd.DataFrame, column: str = 'close', window: int = 20) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            data: DataFrame containing price data
            column: Column name to use for calculation
            window: Window period for moving average
            
        Returns:
            pd.Series: Exponential Moving Average values
        """
        return data[column].ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def calculate_wma(data: pd.DataFrame, column: str = 'close', window: int = 20) -> pd.Series:
        """
        Calculate Weighted Moving Average.
        
        Args:
            data: DataFrame containing price data
            column: Column name to use for calculation
            window: Window period for moving average
            
        Returns:
            pd.Series: Weighted Moving Average values
        """
        weights = np.arange(1, window + 1)
        return data[column].rolling(window=window).apply(lambda x: np.sum(weights * x) / weights.sum(), raw=True)
    
    @staticmethod
    @cached(ttl=600, key_prefix="macd")  # Cache for 10 minutes
    @monitor_performance("calculate_macd")
    def calculate_macd(data: pd.DataFrame, column: str = 'close', fast_period: int = None, 
                      slow_period: int = None, signal_period: int = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data: DataFrame containing price data
            column: Column name to use for calculation
            fast_period: Fast EMA period (uses config default if None)
            slow_period: Slow EMA period (uses config default if None)
            signal_period: Signal line period (uses config default if None)
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: MACD line, signal line, and histogram
        """
        # Use configuration defaults if not specified
        macd_config = Config.get("technical_indicators", "macd", {})
        if fast_period is None:
            fast_period = macd_config.get("fast_period", 12)
        if slow_period is None:
            slow_period = macd_config.get("slow_period", 26)
        if signal_period is None:
            signal_period = macd_config.get("signal_period", 9)
        
        fast_ema = TechnicalIndicators.calculate_ema(data, column, fast_period)
        slow_ema = TechnicalIndicators.calculate_ema(data, column, slow_period)
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    @cached(ttl=600, key_prefix="rsi")  # Cache for 10 minutes
    @monitor_performance("calculate_rsi")
    def calculate_rsi(data: pd.DataFrame, column: str = 'close', window: int = None) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            data: DataFrame containing price data
            column: Column name to use for calculation
            window: RSI period (uses config default if None)
            
        Returns:
            pd.Series: RSI values
        """
        # Use configuration default if window not specified
        if window is None:
            window = Config.get("technical_indicators", "rsi", {}).get("period", 14)
        
        delta = data[column].diff()
        
        gain = delta.copy()
        gain[gain < 0] = 0
        
        loss = delta.copy()
        loss[loss > 0] = 0
        loss = abs(loss)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, column: str = 'close', 
                                 window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: DataFrame containing price data
            column: Column name to use for calculation
            window: Window period for moving average
            num_std: Number of standard deviations for bands
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: Upper band, middle band, and lower band
        """
        middle_band = TechnicalIndicators.calculate_sma(data, column, window)
        std_dev = data[column].rolling(window=window).std()
        
        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def calculate_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            data: DataFrame containing price data
            window: ATR period
            
        Returns:
            pd.Series: ATR values
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    @staticmethod
    def calculate_fibonacci_retracement(data: pd.DataFrame, trend: str = 'up') -> Dict[str, float]:
        """
        Calculate Fibonacci Retracement levels.
        
        Args:
            data: DataFrame containing price data
            trend: Current trend direction ('up' or 'down')
            
        Returns:
            Dict[str, float]: Dictionary containing Fibonacci levels
        """
        if trend == 'up':
            # For uptrend, calculate retracement from low to high
            low = data['low'].min()
            high = data['high'].max()
            diff = high - low
            #0.0186
            levels = {
                '0.0': high,
                '0.236': high - (0.236 * diff),
                '0.382': high - (0.382 * diff),
                '0.5': high - (0.5 * diff),
                '0.618': high - (0.618 * diff),
                '0.786': high - (0.786 * diff),
                '1.0': low
            }
        else:
            # For downtrend, calculate retracement from high to low
            low = data['low'].min()
            high = data['high'].max()
            diff = high - low
            
            levels = {
                '0.0': low,
                '0.236': low + (0.236 * diff),
                '0.382': low + (0.382 * diff),
                '0.5': low + (0.5 * diff),
                '0.618': low + (0.618 * diff),
                '0.786': low + (0.786 * diff),
                '1.0': high
            }
        
        return levels
    
    @staticmethod
    def detect_support_resistance(data: pd.DataFrame, window: int = 20, 
                                 threshold: float = 0.02) -> Tuple[List[float], List[float]]:
        """
        Detect support and resistance levels using price action.
        
        Args:
            data: DataFrame containing price data
            window: Window size for detecting local extrema
            threshold: Percentage threshold for level proximity
            
        Returns:
            Tuple[List[float], List[float]]: Support and resistance levels
        """
        # Find local minima and maxima
        local_min = []
        local_max = []
        
        for i in range(window, len(data) - window):
            # Check if this point is a local minimum
            if all(data['low'].iloc[i] <= data['low'].iloc[i-j] for j in range(1, window+1)) and \
               all(data['low'].iloc[i] <= data['low'].iloc[i+j] for j in range(1, window+1)):
                local_min.append(data['low'].iloc[i])
            
            # Check if this point is a local maximum
            if all(data['high'].iloc[i] >= data['high'].iloc[i-j] for j in range(1, window+1)) and \
               all(data['high'].iloc[i] >= data['high'].iloc[i+j] for j in range(1, window+1)):
                local_max.append(data['high'].iloc[i])
        
        # Cluster similar levels
        support_levels = []
        for level in local_min:
            # Check if this level is close to any existing level
            if not any(abs(level - s) / s < threshold for s in support_levels):
                support_levels.append(level)
        
        resistance_levels = []
        for level in local_max:
            # Check if this level is close to any existing level
            if not any(abs(level - r) / r < threshold for r in resistance_levels):
                resistance_levels.append(level)
        
        return support_levels, resistance_levels
    
    @staticmethod
    def calculate_obv(data: pd.DataFrame) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            pd.Series: OBV values
        """
        if len(data) < 2:
            return pd.Series([0], index=data.index)
        
        obv = pd.Series(0.0, index=data.index)
        close_diff = data['close'].diff()
        
        # Initialize first value
        obv.iloc[0] = data['volume'].iloc[0]
        
        for i in range(1, len(data)):
            if close_diff.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + data['volume'].iloc[i]
            elif close_diff.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - data['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def calculate_stochastic_oscillator(data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            data: DataFrame containing OHLCV data
            k_period: %K period
            d_period: %D period
            
        Returns:
            Tuple[pd.Series, pd.Series]: %K and %D values
        """
        if len(data) < k_period:
            return pd.Series([0] * len(data), index=data.index), pd.Series([0] * len(data), index=data.index)
        
        # Calculate %K
        lowest_low = data['low'].rolling(window=k_period).min()
        highest_high = data['high'].rolling(window=k_period).max()
        
        k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
        
        # Calculate %D (SMA of %K)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    @staticmethod
    def calculate_adx(data: pd.DataFrame, window: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Average Directional Index.
        
        Args:
            data: DataFrame containing price data
            window: ADX period
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: ADX, +DI, and -DI values
        """
        # Calculate True Range
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        # Calculate +DM and -DM
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = pd.Series(plus_dm, index=data.index)
        minus_dm = pd.Series(minus_dm, index=data.index)
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def calculate_ichimoku(data: pd.DataFrame, tenkan_period: int = 9, 
                          kijun_period: int = 26, senkou_b_period: int = 52, 
                          displacement: int = 26) -> Dict[str, pd.Series]:
        """
        Calculate Ichimoku Cloud components.
        
        Args:
            data: DataFrame containing price data
            tenkan_period: Tenkan-sen (Conversion Line) period
            kijun_period: Kijun-sen (Base Line) period
            senkou_b_period: Senkou Span B period
            displacement: Displacement period for Senkou Span
            
        Returns:
            Dict[str, pd.Series]: Dictionary containing Ichimoku components
        """
        # Calculate Tenkan-sen (Conversion Line)
        tenkan_high = data['high'].rolling(window=tenkan_period).max()
        tenkan_low = data['low'].rolling(window=tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Calculate Kijun-sen (Base Line)
        kijun_high = data['high'].rolling(window=kijun_period).max()
        kijun_low = data['low'].rolling(window=kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Calculate Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        # Calculate Senkou Span B (Leading Span B)
        senkou_high = data['high'].rolling(window=senkou_b_period).max()
        senkou_low = data['low'].rolling(window=senkou_b_period).min()
        senkou_span_b = ((senkou_high + senkou_low) / 2).shift(displacement)
        
        # Calculate Chikou Span (Lagging Span)
        chikou_span = data['close'].shift(-displacement)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    @staticmethod
    def calculate_fibonacci_retracement(data: pd.DataFrame, trend: str = 'up') -> Dict[str, float]:
        """
        Calculate Fibonacci Retracement levels.
        
        Args:
            data: DataFrame containing price data
            trend: Current trend direction ('up' or 'down')
            
        Returns:
            Dict[str, float]: Dictionary containing Fibonacci levels
        """
        if trend == 'up':
            # For uptrend, calculate retracement from low to high
            low = data['low'].min()
            high = data['high'].max()
            diff = high - low
            #0.0186
            levels = {
                '0.0': high,
                '0.236': high - (0.236 * diff),
                '0.382': high - (0.382 * diff),
                '0.5': high - (0.5 * diff),
                '0.618': high - (0.618 * diff),
                '0.786': high - (0.786 * diff),
                '1.0': low
            }
        else:
            # For downtrend, calculate retracement from high to low
            low = data['low'].min()
            high = data['high'].max()
            diff = high - low
            
            levels = {
                '0.0': low,
                '0.236': low + (0.236 * diff),
                '0.382': low + (0.382 * diff),
                '0.5': low + (0.5 * diff),
                '0.618': low + (0.618 * diff),
                '0.786': low + (0.786 * diff),
                '1.0': high
            }
        
        return levels
    
    @staticmethod
    def calculate_pivot_points(data: pd.DataFrame, method: str = 'standard') -> Dict[str, float]:
        """
        Calculate Pivot Points.
        
        Args:
            data: DataFrame containing price data for the period
            method: Pivot point calculation method ('standard', 'fibonacci', 'woodie', 'camarilla', 'demark')
            
        Returns:
            Dict[str, float]: Dictionary containing pivot points
        """
        # Get high, low, close for the period
        high = data['high'].iloc[-1]
        low = data['low'].iloc[-1]
        close = data['close'].iloc[-1]
        open_price = data['open'].iloc[-1]
        
        if method == 'standard':
            # Standard pivot points
            pivot = (high + low + close) / 3
            s1 = (2 * pivot) - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)
            r1 = (2 * pivot) - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)
            
            return {
                'pivot': pivot,
                'support_1': s1,
                'support_2': s2,
                'support_3': s3,
                'resistance_1': r1,
                'resistance_2': r2,
                'resistance_3': r3
            }
        
        elif method == 'fibonacci':
            # Fibonacci pivot points
            pivot = (high + low + close) / 3
            s1 = pivot - 0.382 * (high - low)
            s2 = pivot - 0.618 * (high - low)
            s3 = pivot - 1.0 * (high - low)
            r1 = pivot + 0.382 * (high - low)
            r2 = pivot + 0.618 * (high - low)
            r3 = pivot + 1.0 * (high - low)
            
            return {
                'pivot': pivot,
                'support_1': s1,
                'support_2': s2,
                'support_3': s3,
                'resistance_1': r1,
                'resistance_2': r2,
                'resistance_3': r3
            }
        
        elif method == 'woodie':
            # Woodie pivot points
            pivot = (high + low + 2 * open_price) / 4
            s1 = (2 * pivot) - high
            s2 = pivot - (high - low)
            s3 = s1 - (high - low)
            r1 = (2 * pivot) - low
            r2 = pivot + (high - low)
            r3 = r1 + (high - low)
            
            return {
                'pivot': pivot,
                'support_1': s1,
                'support_2': s2,
                'support_3': s3,
                'resistance_1': r1,
                'resistance_2': r2,
                'resistance_3': r3
            }
        
        elif method == 'camarilla':
            # Camarilla pivot points
            pivot = (high + low + close) / 3
            s1 = close - (high - low) * 1.1 / 12
            s2 = close - (high - low) * 1.1 / 6
            s3 = close - (high - low) * 1.1 / 4
            s4 = close - (high - low) * 1.1 / 2
            r1 = close + (high - low) * 1.1 / 12
            r2 = close + (high - low) * 1.1 / 6
            r3 = close + (high - low) * 1.1 / 4
            r4 = close + (high - low) * 1.1 / 2
            
            return {
                'pivot': pivot,
                'support_1': s1,
                'support_2': s2,
                'support_3': s3,
                'support_4': s4,
                'resistance_1': r1,
                'resistance_2': r2,
                'resistance_3': r3,
                'resistance_4': r4
            }
        
        elif method == 'demark':
            # DeMark pivot points
            if close < open_price:
                pivot = high + (2 * low) + close
            elif close > open_price:
                pivot = (2 * high) + low + close
            else:
                pivot = high + low + (2 * close)
            
            pivot = pivot / 4
            s1 = pivot * 2 - high
            r1 = pivot * 2 - low
            
            return {
                'pivot': pivot,
                'support_1': s1,
                'resistance_1': r1
            }
        
        else:
            raise ValueError(f"Unknown pivot point method: {method}")
    
    @staticmethod
    def detect_support_resistance(data: pd.DataFrame, window: int = 20, 
                                 threshold: float = 0.02) -> Tuple[List[float], List[float]]:
        """
        Detect support and resistance levels using price action.
        
        Args:
            data: DataFrame containing price data
            window: Window size for detecting local extrema
            threshold: Percentage threshold for level proximity
            
        Returns:
            Tuple[List[float], List[float]]: Support and resistance levels
        """
        # Find local minima and maxima
        local_min = []
        local_max = []
        
        for i in range(window, len(data) - window):
            # Check if this point is a local minimum
            if all(data['low'].iloc[i] <= data['low'].iloc[i-j] for j in range(1, window+1)) and \
               all(data['low'].iloc[i] <= data['low'].iloc[i+j] for j in range(1, window+1)):
                local_min.append(data['low'].iloc[i])
            
            # Check if this point is a local maximum
            if all(data['high'].iloc[i] >= data['high'].iloc[i-j] for j in range(1, window+1)) and \
               all(data['high'].iloc[i] >= data['high'].iloc[i+j] for j in range(1, window+1)):
                local_max.append(data['high'].iloc[i])
        
        # Cluster similar levels
        support_levels = []
        for level in local_min:
            # Check if this level is close to any existing level
            if not any(abs(level - s) / s < threshold for s in support_levels):
                support_levels.append(level)
        
        resistance_levels = []
        for level in local_max:
            # Check if this level is close to any existing level
            if not any(abs(level - r) / r < threshold for r in resistance_levels):
                resistance_levels.append(level)
        
        return support_levels, resistance_levels
    
    @staticmethod
    def calculate_vwap(data: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            pd.Series: VWAP values
        """
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        return vwap
    
    @staticmethod
    def calculate_money_flow_index(data: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate Money Flow Index (MFI).
        
        Args:
            data: DataFrame containing OHLCV data
            window: Period for calculation
            
        Returns:
            pd.Series: MFI values
        """
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        
        positive_flow = pd.Series(0.0, index=data.index)
        negative_flow = pd.Series(0.0, index=data.index)
        
        # Calculate positive and negative money flow
        for i in range(1, len(data)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = money_flow.iloc[i]
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                negative_flow.iloc[i] = money_flow.iloc[i]
        
        positive_mf = positive_flow.rolling(window=window).sum()
        negative_mf = negative_flow.rolling(window=window).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    @staticmethod
    def calculate_volume_profile(data: pd.DataFrame, bins: int = 20) -> Dict[str, Any]:
        """
        Calculate volume profile analysis.
        
        Args:
            data: DataFrame containing OHLCV data
            bins: Number of price bins for analysis
            
        Returns:
            Dict containing volume profile data
        """
        price_range = data['high'].max() - data['low'].min()
        bin_size = price_range / bins
        
        volume_profile = {}
        for i in range(bins):
            price_level = data['low'].min() + (i * bin_size)
            volume_at_level = data[
                (data['low'] <= price_level + bin_size) & 
                (data['high'] >= price_level)
            ]['volume'].sum()
            volume_profile[f"level_{i}"] = {
                "price": price_level,
                "volume": volume_at_level
            }
        
        # Find high volume nodes
        volumes = [v['volume'] for v in volume_profile.values()]
        avg_volume = np.mean(volumes)
        high_volume_nodes = [
            level for level, data in volume_profile.items() 
            if data['volume'] > avg_volume * 1.5
        ]
        
        return {
            "profile": volume_profile,
            "high_volume_nodes": high_volume_nodes,
            "avg_volume": avg_volume
        }
    
    @staticmethod
    def calculate_williams_r(data: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate Williams %R oscillator.
        
        Args:
            data: DataFrame containing OHLCV data
            window: Period for calculation
            
        Returns:
            pd.Series: Williams %R values
        """
        highest_high = data['high'].rolling(window=window).max()
        lowest_low = data['low'].rolling(window=window).min()
        
        williams_r = ((highest_high - data['close']) / (highest_high - lowest_low)) * -100
        return williams_r
    
    @staticmethod
    def detect_rsi_divergence(prices: pd.Series, rsi: pd.Series, order: int = 5) -> Dict[str, Any]:
        """
        Detect RSI divergence patterns.
        
        Args:
            prices: Price series
            rsi: RSI series
            order: Order for peak detection
            
        Returns:
            Dict containing divergence information
        """
        from scipy.signal import argrelextrema
        
        price_peaks = argrelextrema(prices.values, np.greater, order=order)[0]
        price_lows = argrelextrema(prices.values, np.less, order=order)[0]
        rsi_peaks = argrelextrema(rsi.values, np.greater, order=order)[0]
        rsi_lows = argrelextrema(rsi.values, np.less, order=order)[0]
        
        divergences = {
            "bearish_divergence": [],
            "bullish_divergence": [],
            "hidden_bearish": [],
            "hidden_bullish": []
        }
        
        # Regular bearish divergence (price higher, RSI lower)
        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            if (prices.iloc[price_peaks[-1]] > prices.iloc[price_peaks[-2]] and 
                rsi.iloc[rsi_peaks[-1]] < rsi.iloc[rsi_peaks[-2]]):
                divergences["bearish_divergence"].append({
                    "strength": "strong",
                    "price_peaks": [price_peaks[-2], price_peaks[-1]],
                    "rsi_peaks": [rsi_peaks[-2], rsi_peaks[-1]]
                })
        
        # Regular bullish divergence (price lower, RSI higher)
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            if (prices.iloc[price_lows[-1]] < prices.iloc[price_lows[-2]] and 
                rsi.iloc[rsi_lows[-1]] > rsi.iloc[rsi_lows[-2]]):
                divergences["bullish_divergence"].append({
                    "strength": "strong",
                    "price_lows": [price_lows[-2], price_lows[-1]],
                    "rsi_lows": [rsi_lows[-2], rsi_lows[-1]]
                })
        
        return divergences
    
    @staticmethod
    def calculate_trend_strength(data: pd.DataFrame, sma_20: pd.Series, sma_50: pd.Series, sma_200: pd.Series) -> Dict[str, Any]:
        """
        Calculate comprehensive trend strength analysis.
        
        Args:
            data: DataFrame containing price data
            sma_20: 20-period SMA
            sma_50: 50-period SMA
            sma_200: 200-period SMA
            
        Returns:
            Dict containing trend strength analysis
        """
        current_price = data['close'].iloc[-1]
        
        # Price position relative to moving averages
        price_position = {
            "above_sma_20": current_price > sma_20.iloc[-1],
            "above_sma_50": current_price > sma_50.iloc[-1],
            "above_sma_200": current_price > sma_200.iloc[-1]
        }
        
        # Moving average alignment
        ma_alignment = {
            "sma_20_above_50": sma_20.iloc[-1] > sma_50.iloc[-1],
            "sma_50_above_200": sma_50.iloc[-1] > sma_200.iloc[-1],
            "all_bullish": sma_20.iloc[-1] > sma_50.iloc[-1] > sma_200.iloc[-1],
            "all_bearish": sma_20.iloc[-1] < sma_50.iloc[-1] < sma_200.iloc[-1]
        }
        
        # Trend consistency (last 20 periods)
        price_trend = data['close'].iloc[-20:].pct_change().mean()
        trend_consistency = "bullish" if price_trend > 0.001 else "bearish" if price_trend < -0.001 else "sideways"
        
        # Overall strength score (0-100)
        strength_score = 0
        
        # Add points for price position
        if price_position["above_sma_20"]: strength_score += 10
        if price_position["above_sma_50"]: strength_score += 15
        if price_position["above_sma_200"]: strength_score += 20
        
        # Add points for MA alignment
        if ma_alignment["sma_20_above_50"]: strength_score += 15
        if ma_alignment["sma_50_above_200"]: strength_score += 20
        if ma_alignment["all_bullish"]: strength_score += 20
        
        # Add points for trend consistency
        if trend_consistency == "bullish": strength_score += 20
        elif trend_consistency == "sideways": strength_score += 10
        
        return {
            "overall_strength": "strong" if strength_score >= 70 else "moderate" if strength_score >= 40 else "weak",
            "price_position": price_position,
            "ma_alignment": ma_alignment,
            "trend_consistency": trend_consistency,
            "strength_score": strength_score
        }
    
    @staticmethod
    def calculate_enhanced_support_resistance(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate enhanced support and resistance levels.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            Dict containing enhanced levels
        """
        # Dynamic support/resistance (recent swing points)
        support_levels, resistance_levels = TechnicalIndicators.detect_support_resistance(data)
        
        # Fibonacci retracement levels
        high = data['high'].max()
        low = data['low'].min()
        diff = high - low
        fibonacci_levels = {
            "0.236": low + 0.236 * diff,
            "0.382": low + 0.382 * diff,
            "0.500": low + 0.500 * diff,
            "0.618": low + 0.618 * diff,
            "0.786": low + 0.786 * diff
        }
        
        # Pivot points
        pivot_points = TechnicalIndicators.calculate_pivot_points(data)
        
        # Psychological levels (round numbers)
        current_price = data['close'].iloc[-1]
        psychological_levels = []
        for i in range(-5, 6):
            level = round(current_price / 100) * 100 + (i * 100)
            if 0 < level < current_price * 2:  # Reasonable range
                psychological_levels.append(level)
        
        return {
            "dynamic_support": support_levels[:3] if len(support_levels) >= 3 else support_levels,
            "dynamic_resistance": resistance_levels[:3] if len(resistance_levels) >= 3 else resistance_levels,
            "fibonacci_levels": fibonacci_levels,
            "pivot_points": pivot_points,
            "psychological_levels": psychological_levels
        }
    
    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame, stock_symbol: str = None) -> Dict[str, Any]:
        """
        Calculate all technical indicators.
        
        Args:
            data: DataFrame containing price and volume data
            stock_symbol: Stock symbol for sector classification
            
        Returns:
            Dict[str, Any]: Dictionary containing all calculated indicators
        """
        indicators = {}
        
        # Calculate Moving Averages
        sma_20 = TechnicalIndicators.calculate_sma(data, 'close', 20)
        sma_50 = TechnicalIndicators.calculate_sma(data, 'close', 50)
        sma_200 = TechnicalIndicators.calculate_sma(data, 'close', 200)
        ema_20 = TechnicalIndicators.calculate_ema(data, 'close', 20)
        ema_50 = TechnicalIndicators.calculate_ema(data, 'close', 50)
        
        # Calculate price to MA ratios
        current_price = data['close'].iloc[-1]
        # Handle division by zero or NaN for price to SMA 200
        if pd.isna(sma_200.iloc[-1]) or sma_200.iloc[-1] == 0:
            price_to_sma_200 = 0.0  # Default to neutral
        else:
            price_to_sma_200 = (current_price / sma_200.iloc[-1] - 1)
        
        # Handle division by zero or NaN for SMA 20 to SMA 50
        if pd.isna(sma_50.iloc[-1]) or sma_50.iloc[-1] == 0:
            sma_20_to_sma_50 = 0.0  # Default to neutral
        else:
            sma_20_to_sma_50 = (sma_20.iloc[-1] / sma_50.iloc[-1] - 1)
        
        # Check for Golden/Death Cross
        golden_cross = sma_20.iloc[-1] > sma_50.iloc[-1] and sma_20.iloc[-2] <= sma_50.iloc[-2]
        death_cross = sma_20.iloc[-1] < sma_50.iloc[-1] and sma_20.iloc[-2] >= sma_50.iloc[-2]
        
        # Add Moving Averages to indicators
        indicators['moving_averages'] = {
            'sma_20': float(sma_20.iloc[-1]),
            'sma_50': float(sma_50.iloc[-1]),
            'sma_200': float(sma_200.iloc[-1]),
            'ema_20': float(ema_20.iloc[-1]),
            'ema_50': float(ema_50.iloc[-1]),
            'price_to_sma_200': float(price_to_sma_200),
            'sma_20_to_sma_50': float(sma_20_to_sma_50),
            'golden_cross': bool(golden_cross),
            'death_cross': bool(death_cross)
        }
        
        # Calculate RSI
        rsi = TechnicalIndicators.calculate_rsi(data)
        rsi_value = float(rsi.iloc[-1])
        
        # Determine RSI status with more granular levels
        if rsi_value > 70:
            rsi_status = 'overbought'
        elif rsi_value >= 60:
            rsi_status = 'near_overbought'
        elif rsi_value <= 40:
            rsi_status = 'near_oversold'
        elif rsi_value < 30:
            rsi_status = 'oversold'
        else:
            rsi_status = 'neutral'
        
        indicators['rsi'] = {
            'rsi_14': rsi_value,
            'trend': 'up' if rsi.iloc[-1] > rsi.iloc[-2] else 'down',
            'status': rsi_status
        }
        
        # Calculate MACD
        macd_line, signal_line, histogram = TechnicalIndicators.calculate_macd(data)
        indicators['macd'] = {
            'macd_line': float(macd_line.iloc[-1]),
            'signal_line': float(signal_line.iloc[-1]),
            'histogram': float(histogram.iloc[-1])
        }
        
        # Calculate Bollinger Bands
        upper_band, middle_band, lower_band = TechnicalIndicators.calculate_bollinger_bands(data)
        current_price = data['close'].iloc[-1]
        
        # Handle division by zero or NaN for percent_b
        band_width = upper_band.iloc[-1] - lower_band.iloc[-1]
        if pd.isna(band_width) or band_width == 0:
            percent_b = 0.5  # Default to middle of band
        else:
            percent_b = (current_price - lower_band.iloc[-1]) / band_width
        
        # Handle division by zero or NaN for bandwidth
        if pd.isna(middle_band.iloc[-1]) or middle_band.iloc[-1] == 0:
            bandwidth = 0.0  # Default to zero bandwidth
        else:
            bandwidth = band_width / middle_band.iloc[-1]
        
        indicators['bollinger_bands'] = {
            'upper_band': float(upper_band.iloc[-1]),
            'middle_band': float(middle_band.iloc[-1]),
            'lower_band': float(lower_band.iloc[-1]),
            'percent_b': float(percent_b),
            'bandwidth': float(bandwidth)
        }
        
        # Calculate Volume indicators
        volume_ma = data['volume'].rolling(window=20).mean()
        # Handle division by zero or NaN
        if pd.isna(volume_ma.iloc[-1]) or volume_ma.iloc[-1] == 0:
            volume_ratio = 1.0  # Default to normal volume ratio
        else:
            volume_ratio = data['volume'].iloc[-1] / volume_ma.iloc[-1]
        obv = TechnicalIndicators.calculate_obv(data)
        
        indicators['volume'] = {
            'volume_ratio': float(volume_ratio),
            'obv': float(obv.iloc[-1]),
            'obv_trend': 'up' if obv.iloc[-1] > obv.iloc[-2] else 'down'
        }
        
        # Calculate ADX
        adx, plus_di, minus_di = TechnicalIndicators.calculate_adx(data)
        trend_direction = 'bullish' if plus_di.iloc[-1] > minus_di.iloc[-1] else 'bearish'
        
        indicators['adx'] = {
            'adx': float(adx.iloc[-1]),
            'plus_di': float(plus_di.iloc[-1]),
            'minus_di': float(minus_di.iloc[-1]),
            'trend_direction': trend_direction
        }
        
        # Add trend data
        indicators['trend_data'] = {
            'direction': trend_direction,
            'strength': 'strong' if adx.iloc[-1] > 25 else 'weak',
            'adx': float(adx.iloc[-1]),
            'plus_di': float(plus_di.iloc[-1]),
            'minus_di': float(minus_di.iloc[-1])
        }
        
        # Calculate Enhanced Volatility Indicators
        atr = TechnicalIndicators.calculate_atr(data)
        atr_20 = atr.rolling(window=20).mean()
        volatility_ratio = atr.iloc[-1] / atr_20.iloc[-1] if not pd.isna(atr_20.iloc[-1]) and atr_20.iloc[-1] != 0 else 1.0
        
        # Bollinger Band squeeze detection
        bb_squeeze = bandwidth < 0.1  # Low bandwidth indicates squeeze
        
        # Historical volatility percentile (20-period)
        volatility_percentile = (atr.iloc[-20:].rank().iloc[-1] / 20) * 100 if len(atr) >= 20 else 50
        
        indicators['volatility'] = {
            'atr': float(atr.iloc[-1]),
            'atr_20_avg': float(atr_20.iloc[-1]) if not pd.isna(atr_20.iloc[-1]) else None,
            'volatility_ratio': float(volatility_ratio),
            'bb_squeeze': bool(bb_squeeze),
            'volatility_percentile': float(volatility_percentile),
            'volatility_regime': 'high' if volatility_ratio > 1.5 else 'low' if volatility_ratio < 0.7 else 'normal'
        }
        
        # Calculate Enhanced Volume Indicators
        vwap = TechnicalIndicators.calculate_vwap(data)
        mfi = TechnicalIndicators.calculate_money_flow_index(data)
        
        # Volume profile analysis
        volume_profile = TechnicalIndicators.calculate_volume_profile(data)
        
        # Comprehensive volume analysis
        enhanced_volume_analysis = TechnicalIndicators.calculate_enhanced_volume_analysis(data)
        
        indicators['enhanced_volume'] = {
            'vwap': float(vwap.iloc[-1]) if not pd.isna(vwap.iloc[-1]) else None,
            'mfi': float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else None,
            'mfi_status': 'overbought' if mfi.iloc[-1] > 80 else 'oversold' if mfi.iloc[-1] < 20 else 'neutral',
            'volume_profile': volume_profile,
            'price_vs_vwap': float((current_price / vwap.iloc[-1] - 1) * 100) if not pd.isna(vwap.iloc[-1]) and vwap.iloc[-1] != 0 else 0.0,
            'comprehensive_analysis': enhanced_volume_analysis
        }
        
        # Calculate Enhanced Momentum Indicators
        stochastic_k, stochastic_d = TechnicalIndicators.calculate_stochastic_oscillator(data)
        williams_r = TechnicalIndicators.calculate_williams_r(data)
        
        # RSI divergence detection
        rsi_divergence = TechnicalIndicators.detect_rsi_divergence(data['close'], rsi)
        
        indicators['enhanced_momentum'] = {
            'stochastic_k': float(stochastic_k.iloc[-1]) if not pd.isna(stochastic_k.iloc[-1]) else None,
            'stochastic_d': float(stochastic_d.iloc[-1]) if not pd.isna(stochastic_d.iloc[-1]) else None,
            'stochastic_status': 'overbought' if stochastic_k.iloc[-1] > 80 else 'oversold' if stochastic_k.iloc[-1] < 20 else 'neutral',
            'williams_r': float(williams_r.iloc[-1]) if not pd.isna(williams_r.iloc[-1]) else None,
            'williams_r_status': 'overbought' if williams_r.iloc[-1] < -80 else 'oversold' if williams_r.iloc[-1] > -20 else 'neutral',
            'rsi_divergence': rsi_divergence
        }
        
        # Calculate Enhanced Trend Strength
        trend_strength = TechnicalIndicators.calculate_trend_strength(data, sma_20, sma_50, sma_200)
        indicators['trend_strength'] = trend_strength
        
        # Calculate Enhanced Support/Resistance
        enhanced_levels = TechnicalIndicators.calculate_enhanced_support_resistance(data)
        indicators['enhanced_levels'] = enhanced_levels
        
        # === PHASE 2 FEATURES ===
        
        # Calculate Multi-Timeframe Analysis
        multi_timeframe = TechnicalIndicators.calculate_multi_timeframe_analysis(data)
        indicators['multi_timeframe'] = multi_timeframe
        
        # Calculate Advanced Risk Metrics
        advanced_risk = TechnicalIndicators.calculate_advanced_risk_metrics(data)
        indicators['advanced_risk'] = advanced_risk
        
        # Calculate Phase 3 Advanced Risk Metrics
        stress_testing = TechnicalIndicators.calculate_stress_testing_metrics(data)
        scenario_analysis = TechnicalIndicators.calculate_scenario_analysis_metrics(data)
        indicators['stress_testing'] = stress_testing
        indicators['scenario_analysis'] = scenario_analysis
        
        return indicators

    @staticmethod
    def calculate_enhanced_volume_analysis(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive volume analysis including daily metrics and ratios.
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            Dict containing comprehensive volume analysis
        """
        if len(data) < 20:
            return {"error": "insufficient_data"}
        
        # Basic volume metrics
        current_volume = data['volume'].iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # Volume moving averages
        volume_ma_5 = data['volume'].rolling(window=5).mean()
        volume_ma_10 = data['volume'].rolling(window=10).mean()
        volume_ma_20 = data['volume'].rolling(window=20).mean()
        volume_ma_50 = data['volume'].rolling(window=50).mean()
        
        # Volume ratios
        volume_ratio_5 = current_volume / volume_ma_5.iloc[-1] if volume_ma_5.iloc[-1] > 0 else 1.0
        volume_ratio_10 = current_volume / volume_ma_10.iloc[-1] if volume_ma_10.iloc[-1] > 0 else 1.0
        volume_ratio_20 = current_volume / volume_ma_20.iloc[-1] if volume_ma_20.iloc[-1] > 0 else 1.0
        volume_ratio_50 = current_volume / volume_ma_50.iloc[-1] if volume_ma_50.iloc[-1] > 0 else 1.0
        
        # Volume trend analysis
        volume_trend_5 = "up" if current_volume > volume_ma_5.iloc[-1] else "down"
        volume_trend_10 = "up" if current_volume > volume_ma_10.iloc[-1] else "down"
        volume_trend_20 = "up" if current_volume > volume_ma_20.iloc[-1] else "down"
        volume_trend_50 = "up" if current_volume > volume_ma_50.iloc[-1] else "down"
        
        # Volume volatility
        volume_std_20 = data['volume'].rolling(window=20).std()
        volume_volatility = volume_std_20.iloc[-1] / volume_ma_20.iloc[-1] if volume_ma_20.iloc[-1] > 0 else 0.0
        
        # Volume anomaly detection
        volume_anomalies = []
        for i in range(20, len(data)):
            vol_ma = volume_ma_20.iloc[i]
            vol_std = volume_std_20.iloc[i]
            current_vol = data['volume'].iloc[i]
            
            if current_vol > (vol_ma + 2 * vol_std):
                anomaly = {
                    "date": data.index[i].strftime('%Y-%m-%d'),
                    "volume": float(current_vol),
                    "avg_volume": float(vol_ma),
                    "volume_ratio": float(current_vol / vol_ma),
                    "price": float(data['close'].iloc[i]),
                    "price_change": float(data['close'].iloc[i] - data['close'].iloc[i-1]),
                    "anomaly_strength": "high" if current_vol > (vol_ma + 3 * vol_std) else "medium"
                }
                volume_anomalies.append(anomaly)
        
        # Recent volume anomalies (last 10 days)
        recent_anomalies = [a for a in volume_anomalies if len(volume_anomalies) <= 10]
        
        # Volume price correlation
        price_changes = data['close'].pct_change().dropna()
        volume_changes = data['volume'].pct_change().dropna()
        
        # Align series
        min_length = min(len(price_changes), len(volume_changes))
        if min_length > 10:
            correlation_20 = price_changes.iloc[-20:].corr(volume_changes.iloc[-20:])
            correlation_50 = price_changes.iloc[-50:].corr(volume_changes.iloc[-50:]) if len(price_changes) >= 50 else correlation_20
        else:
            correlation_20 = correlation_50 = 0.0
        
        # Volume confirmation analysis
        price_trend = "up" if data['close'].iloc[-1] > data['close'].iloc[-2] else "down"
        volume_confirmation = "confirmed" if (price_trend == "up" and volume_trend_20 == "up") or (price_trend == "down" and volume_trend_20 == "up") else "diverging"
        
        # Volume accumulation/distribution
        obv = TechnicalIndicators.calculate_obv(data)
        obv_trend = "up" if obv.iloc[-1] > obv.iloc[-5] else "down"
        
        # Money Flow Index
        mfi = TechnicalIndicators.calculate_money_flow_index(data)
        mfi_status = "overbought" if mfi.iloc[-1] > 80 else "oversold" if mfi.iloc[-1] < 20 else "neutral"
        
        # VWAP analysis
        vwap = TechnicalIndicators.calculate_vwap(data)
        price_vs_vwap = ((current_price / vwap.iloc[-1] - 1) * 100) if vwap.iloc[-1] > 0 else 0.0
        
        # Volume profile analysis
        volume_profile = TechnicalIndicators.calculate_volume_profile(data)
        
        # Volume strength scoring (0-100)
        volume_strength = 0
        if volume_ratio_20 > 1.5:
            volume_strength += 30
        elif volume_ratio_20 > 1.2:
            volume_strength += 20
        elif volume_ratio_20 > 0.8:
            volume_strength += 10
            
        if volume_confirmation == "confirmed":
            volume_strength += 25
            
        if obv_trend == "up":
            volume_strength += 20
            
        if abs(correlation_20) > 0.3:
            volume_strength += 15
            
        if mfi_status == "neutral":
            volume_strength += 10
            
        volume_strength = min(volume_strength, 100)
        
        return {
            "daily_metrics": {
                "current_volume": float(current_volume),
                "current_price": float(current_price),
                "volume_price_ratio": float(current_volume / current_price) if current_price > 0 else 0.0
            },
            "volume_ratios": {
                "ratio_5d": float(volume_ratio_5),
                "ratio_10d": float(volume_ratio_10),
                "ratio_20d": float(volume_ratio_20),
                "ratio_50d": float(volume_ratio_50),
                "primary_ratio": float(volume_ratio_20)  # Most commonly used
            },
            "volume_trends": {
                "trend_5d": volume_trend_5,
                "trend_10d": volume_trend_10,
                "trend_20d": volume_trend_20,
                "trend_50d": volume_trend_50,
                "primary_trend": volume_trend_20
            },
            "volume_volatility": {
                "volatility_ratio": float(volume_volatility),
                "volatility_regime": "high" if volume_volatility > 0.5 else "low" if volume_volatility < 0.2 else "normal"
            },
            "volume_anomalies": {
                "total_anomalies": len(volume_anomalies),
                "recent_anomalies": len(recent_anomalies),
                "anomaly_list": recent_anomalies,
                "last_anomaly_date": volume_anomalies[-1]["date"] if volume_anomalies else None
            },
            "price_volume_correlation": {
                "correlation_20d": float(correlation_20),
                "correlation_50d": float(correlation_50),
                "correlation_strength": "strong" if abs(correlation_20) > 0.5 else "moderate" if abs(correlation_20) > 0.3 else "weak"
            },
            "volume_confirmation": {
                "price_trend": price_trend,
                "volume_trend": volume_trend_20,
                "confirmation_status": volume_confirmation,
                "strength": "strong" if volume_ratio_20 > 1.5 and volume_confirmation == "confirmed" else "moderate" if volume_confirmation == "confirmed" else "weak"
            },
            "advanced_indicators": {
                "obv": float(obv.iloc[-1]),
                "obv_trend": obv_trend,
                "mfi": float(mfi.iloc[-1]),
                "mfi_status": mfi_status,
                "vwap": float(vwap.iloc[-1]),
                "price_vs_vwap_pct": float(price_vs_vwap)
            },
            "volume_profile": volume_profile,
            "volume_strength_score": volume_strength,
            "volume_quality": {
                "data_quality": "good" if current_volume > 0 else "poor",
                "reliability": "high" if volume_strength > 70 else "medium" if volume_strength > 40 else "low"
            }
        }

    @staticmethod
    def calculate_multi_timeframe_analysis(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive multi-timeframe analysis.
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            Dict containing multi-timeframe analysis
        """
        if len(data) < 200:
            return {"error": "insufficient_data_for_multi_timeframe"}
        
        # Define timeframes
        timeframes = {
            "short_term": {"periods": [5, 10, 20], "name": "Short-term (5-20 days)"},
            "medium_term": {"periods": [50, 100, 200], "name": "Medium-term (50-200 days)"},
            "long_term": {"periods": [200, 365], "name": "Long-term (200+ days)"}
        }
        
        analysis = {}
        
        for tf_name, tf_config in timeframes.items():
            tf_analysis = {}
            
            for period in tf_config["periods"]:
                if len(data) >= period:
                    # Calculate indicators for this period
                    recent_data = data.tail(period)
                    
                    # Moving averages
                    sma = TechnicalIndicators.calculate_sma(recent_data, 'close', 20)
                    ema = TechnicalIndicators.calculate_ema(recent_data, 'close', 20)
                    
                    # RSI
                    rsi = TechnicalIndicators.calculate_rsi(recent_data)
                    
                    # MACD
                    macd_line, signal_line, histogram = TechnicalIndicators.calculate_macd(recent_data)
                    
                    # Bollinger Bands
                    upper_bb, middle_bb, lower_bb = TechnicalIndicators.calculate_bollinger_bands(recent_data)
                    
                    # Current price analysis
                    current_price = recent_data['close'].iloc[-1]
                    price_position = {
                        "above_sma": current_price > sma.iloc[-1],
                        "above_ema": current_price > ema.iloc[-1],
                        "bb_position": "upper" if current_price > upper_bb.iloc[-1] else "lower" if current_price < lower_bb.iloc[-1] else "middle",
                        "rsi_zone": "overbought" if rsi.iloc[-1] > 70 else "oversold" if rsi.iloc[-1] < 30 else "neutral",
                        "macd_signal": "bullish" if macd_line.iloc[-1] > signal_line.iloc[-1] else "bearish"
                    }
                    
                    # Trend analysis
                    trend_direction = "bullish" if sma.iloc[-1] > sma.iloc[-5] else "bearish" if sma.iloc[-1] < sma.iloc[-5] else "sideways"
                    
                    # Momentum analysis
                    momentum = {
                        "rsi_trend": "up" if rsi.iloc[-1] > rsi.iloc[-5] else "down",
                        "macd_trend": "up" if macd_line.iloc[-1] > macd_line.iloc[-5] else "down",
                        "volume_trend": "up" if recent_data['volume'].iloc[-5:].mean() > recent_data['volume'].iloc[-10:-5].mean() else "down"
                    }
                    
                    tf_analysis[f"{period}d"] = {
                        "price_position": price_position,
                        "trend_direction": trend_direction,
                        "momentum": momentum,
                        "indicators": {
                            "sma_20": float(sma.iloc[-1]),
                            "ema_20": float(ema.iloc[-1]),
                            "rsi": float(rsi.iloc[-1]),
                            "macd": float(macd_line.iloc[-1]),
                            "signal": float(signal_line.iloc[-1]),
                            "bb_upper": float(upper_bb.iloc[-1]),
                            "bb_lower": float(lower_bb.iloc[-1])
                        }
                    }
            
            # Timeframe analysis (AI-ready format)
            if tf_analysis:
                analysis[tf_name] = {
                    "name": tf_config["name"],
                    "periods": tf_analysis,
                    "ai_confidence": 0,  # Will be populated by AI analysis
                    "ai_trend": "neutral"  # Will be populated by AI analysis
                }
        
        # Overall multi-timeframe AI analysis
        if analysis:
            short_term = analysis.get("short_term", {})
            medium_term = analysis.get("medium_term", {})
            long_term = analysis.get("long_term", {})
            
            # Calculate overall AI confidence (weighted average)
            short_confidence = short_term.get("ai_confidence", 0)
            medium_confidence = medium_term.get("ai_confidence", 0)
            long_confidence = long_term.get("ai_confidence", 0)
            
            # Weighted average (short-term: 30%, medium-term: 40%, long-term: 30%)
            overall_confidence = (short_confidence * 0.3 + medium_confidence * 0.4 + long_confidence * 0.3)
            
            # Determine primary trend based on AI analysis
            trends = [
                short_term.get("ai_trend", "neutral"),
                medium_term.get("ai_trend", "neutral"),
                long_term.get("ai_trend", "neutral")
            ]
            
            bullish_count = trends.count("bullish")
            bearish_count = trends.count("bearish")
            neutral_count = trends.count("neutral")
            
            if bullish_count > bearish_count:
                primary_trend = "bullish"
            elif bearish_count > bullish_count:
                primary_trend = "bearish"
            else:
                primary_trend = "neutral"
            
            analysis["overall_ai_analysis"] = {
                "confidence": overall_confidence,
                "primary_trend": primary_trend,
                "short_term": short_term.get("ai_trend", "neutral"),
                "medium_term": medium_term.get("ai_trend", "neutral"),
                "long_term": long_term.get("ai_trend", "neutral")
            }
        
        return analysis

    @staticmethod
    def get_market_metrics(data: pd.DataFrame, stock_symbol: str = None) -> Dict[str, float]:
        """
        Calculate real Indian market metrics using VERIFIED data sources.
        Now supports sector-specific indices for more accurate calculations.
        
        Args:
            data: DataFrame containing OHLCV data
            stock_symbol: Stock symbol for sector classification
            
        Returns:
            Dict containing enhanced market metrics
        """
        if stock_symbol:
            return market_metrics_provider.get_enhanced_market_metrics(data, stock_symbol)
        else:
            # Fallback to original method
            return market_metrics_provider.get_basic_market_metrics(data)

    @staticmethod
    def calculate_advanced_risk_metrics(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate advanced risk metrics and assessment.
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            Dict containing advanced risk analysis
        """
        if len(data) < 50:
            return {"error": "insufficient_data_for_risk_analysis"}
        
        # Calculate returns
        returns = data['close'].pct_change().dropna()
        
        # Basic risk metrics
        volatility = returns.std()
        annualized_volatility = volatility * np.sqrt(252)  # Assuming daily data
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)  # 95% VaR
        var_99 = np.percentile(returns, 1)  # 99% VaR
        
        # Expected Shortfall (Conditional VaR)
        es_95 = returns[returns <= var_95].mean()
        es_99 = returns[returns <= var_99].mean()
        
        # Maximum Drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
        mean_return = returns.mean()
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std()
        sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar Ratio
        calmar_ratio = (mean_return * 252) / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Get real market data for beta and correlation
        try:
            market_metrics = TechnicalIndicators.get_market_metrics(data)
            beta = market_metrics["beta"]
            market_correlation = market_metrics["correlation"]
            risk_free_rate = market_metrics["risk_free_rate"]
        except Exception as e:
            # Fallback to default values if market metrics calculation fails
            beta = 1.0
            market_correlation = 0.6
            risk_free_rate = 6.5
        
        # Risk-adjusted returns using real risk-free rate
        risk_adjusted_return = (mean_return - risk_free_rate / 252) / annualized_volatility if annualized_volatility > 0 else 0
        
        # Volatility regime analysis
        rolling_vol = returns.rolling(window=20).std()
        current_vol = rolling_vol.iloc[-1]
        vol_percentile = (rolling_vol.rank().iloc[-1] / len(rolling_vol)) * 100
        
        # Volatility regime classification using configurable thresholds
        vol_thresholds = Config.get("risk", "stress_testing", {}).get("volatility_percentiles", [0.95, 0.99])
        vol_high_threshold = vol_thresholds[0] * 100 if len(vol_thresholds) > 0 else 80
        vol_low_threshold = (1 - vol_thresholds[0]) * 100 if len(vol_thresholds) > 0 else 20
        
        if vol_percentile > vol_high_threshold:
            vol_regime = "high"
        elif vol_percentile < vol_low_threshold:
            vol_regime = "low"
        else:
            vol_regime = "normal"
        
        # Tail risk analysis
        tail_events = len(returns[returns < var_95])
        tail_frequency = tail_events / len(returns)
        
        # Liquidity risk (if volume data available)
        volume_volatility = 0  # Initialize to default value
        if 'volume' in data.columns:
            avg_volume = data['volume'].mean()
            volume_volatility = data['volume'].std() / avg_volume if avg_volume > 0 else 0
            liquidity_score = min(100, max(0, 100 - volume_volatility * 100))
            print(f"DEBUG: Volume data available - avg_volume: {avg_volume}, volume_volatility: {volume_volatility}, liquidity_score: {liquidity_score}")
        else:
            liquidity_score = 50  # Neutral if no volume data
            print(f"DEBUG: No volume data available in columns: {data.columns.tolist()}")
        
        # Overall risk score (0-100, higher = more risky)
        risk_score = 0
        
        # Volatility component (30%)
        vol_component = min(30, (annualized_volatility * 100))
        risk_score += vol_component
        
        # Drawdown component (25%)
        drawdown_component = min(25, abs(max_drawdown) * 100)
        risk_score += drawdown_component
        
        # Tail risk component (20%)
        tail_component = min(20, tail_frequency * 100)
        risk_score += tail_component
        
        # Liquidity component (15%)
        liquidity_component = (100 - liquidity_score) * 0.15
        risk_score += liquidity_component
        
        # Correlation component (10%)
        correlation_component = market_correlation * 10
        risk_score += correlation_component
        
        # Risk level classification
        if risk_score >= 70:
            risk_level = "high"
        elif risk_score >= 40:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Risk mitigation recommendations
        mitigation_strategies = []
        
        if vol_component > 20:
            mitigation_strategies.append("Consider volatility-based position sizing")
            mitigation_strategies.append("Use wider stop-loss levels")
        
        if drawdown_component > 15:
            mitigation_strategies.append("Implement strict risk management")
            mitigation_strategies.append("Consider hedging strategies")
        
        if tail_component > 10:
            mitigation_strategies.append("Prepare for extreme market moves")
            mitigation_strategies.append("Diversify across uncorrelated assets")
        
        if liquidity_component > 10:
            mitigation_strategies.append("Avoid large position sizes")
            mitigation_strategies.append("Use limit orders for exits")
        
        return {
            "basic_metrics": {
                "volatility": float(volatility),
                "annualized_volatility": float(annualized_volatility),
                "mean_return": float(mean_return),
                "annualized_return": float(mean_return * 252)
            },
            "var_metrics": {
                "var_95": float(var_95),
                "var_99": float(var_99),
                "es_95": float(es_95),
                "es_99": float(es_99)
            },
            "drawdown_metrics": {
                "max_drawdown": float(max_drawdown),
                "current_drawdown": float(drawdown.iloc[-1]),
                "drawdown_duration": int(float((drawdown < 0).astype(int).rolling(window=len(drawdown)).sum().iloc[-1])) if not pd.isna((drawdown < 0).astype(int).rolling(window=len(drawdown)).sum().iloc[-1]) else 0
            },
            "risk_adjusted_metrics": {
                "sharpe_ratio": float(sharpe_ratio),
                "sortino_ratio": float(sortino_ratio),
                "calmar_ratio": float(calmar_ratio),
                "risk_adjusted_return": float(risk_adjusted_return)
            },
            "distribution_metrics": {
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
                "tail_frequency": float(tail_frequency)
            },
            "volatility_analysis": {
                "current_volatility": float(current_vol),
                "volatility_percentile": float(vol_percentile),
                "volatility_regime": vol_regime
            },
            "liquidity_analysis": {
                "liquidity_score": float(liquidity_score),
                "volume_volatility": float(volume_volatility) if 'volume' in data.columns else 0
            },
            "correlation_analysis": {
                "market_correlation": float(market_correlation),
                "beta": float(beta)
            },
            "risk_assessment": {
                "overall_risk_score": float(risk_score),
                "risk_level": risk_level,
                "risk_components": {
                    "volatility_risk": float(vol_component),
                    "drawdown_risk": float(drawdown_component),
                    "tail_risk": float(tail_component),
                    "liquidity_risk": float(liquidity_component),
                    "correlation_risk": float(correlation_component)
                },
                "mitigation_strategies": mitigation_strategies
            }
        }

    @staticmethod
    def calculate_stress_testing_metrics(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate stress testing metrics for advanced risk assessment.
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            Dict containing stress testing analysis
        """
        if len(data) < 100:
            return {"error": "insufficient_data_for_stress_testing"}
        
        # Calculate returns
        returns = data['close'].pct_change().dropna()
        
        # Historical stress scenarios
        stress_scenarios = {}
        
        # 1. Historical worst periods
        rolling_20d = returns.rolling(window=20).sum()
        rolling_60d = returns.rolling(window=60).sum()
        rolling_252d = returns.rolling(window=252).sum()
        
        stress_scenarios["worst_20d"] = {
            "return": float(rolling_20d.min()),
            "date": str(rolling_20d.idxmin()) if rolling_20d.idxmin() else None,
            "percentile": float((rolling_20d.rank().iloc[-1] / len(rolling_20d)) * 100)
        }
        
        # Helper function to safely get idxmin with proper NA handling
        def safe_idxmin(series):
            if series.empty or series.isna().all():
                return None
            try:
                # Use skipna=True to avoid the deprecation warning
                min_idx = series.idxmin(skipna=True)
                return str(min_idx) if min_idx is not None else None
            except (ValueError, TypeError):
                return None
        
        stress_scenarios["worst_60d"] = {
            "return": float(rolling_60d.min()),
            "date": safe_idxmin(rolling_60d),
            "percentile": float((rolling_60d.rank().iloc[-1] / len(rolling_60d)) * 100)
        }
        
        stress_scenarios["worst_252d"] = {
            "return": float(rolling_252d.min()),
            "date": safe_idxmin(rolling_252d),
            "percentile": float((rolling_252d.rank().iloc[-1] / len(rolling_252d)) * 100)
        }
        
        # 2. Volatility stress scenarios
        rolling_vol = returns.rolling(window=20).std()
        vol_stress = rolling_vol.quantile([0.95, 0.99])
        
        stress_scenarios["volatility_stress"] = {
            "95th_percentile": float(vol_stress[0.95]),
            "99th_percentile": float(vol_stress[0.99]),
            "current_vol": float(rolling_vol.iloc[-1]),
            "vol_percentile": float((rolling_vol.rank().iloc[-1] / len(rolling_vol)) * 100)
        }
        
        # 3. Drawdown stress scenarios
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        
        stress_scenarios["drawdown_stress"] = {
            "max_drawdown": float(drawdown.min()),
            "current_drawdown": float(drawdown.iloc[-1]),
            "avg_drawdown": float(drawdown[drawdown < 0].mean()),
            "drawdown_frequency": float(len(drawdown[drawdown < -0.05]) / len(drawdown))
        }
        
        # 4. Tail risk stress scenarios
        tail_events_2std = len(returns[returns < -2 * returns.std()])
        tail_events_3std = len(returns[returns < -3 * returns.std()])
        
        stress_scenarios["tail_risk_stress"] = {
            "2std_events": tail_events_2std,
            "3std_events": tail_events_3std,
            "2std_frequency": float(tail_events_2std / len(returns)),
            "3std_frequency": float(tail_events_3std / len(returns)),
            "largest_daily_loss": float(returns.min()),
            "largest_daily_gain": float(returns.max())
        }
        
        # 5. Liquidity stress scenarios (if volume data available)
        if 'volume' in data.columns:
            volume = data['volume']
            avg_volume = volume.mean()
            volume_stress = volume.quantile([0.05, 0.01])
            
            stress_scenarios["liquidity_stress"] = {
                "low_volume_periods": len(volume[volume < avg_volume * 0.5]),
                "5th_percentile_volume": float(volume_stress[0.05]),
                "1st_percentile_volume": float(volume_stress[0.01]),
                "current_volume": float(volume.iloc[-1]),
                "volume_percentile": float((volume.rank().iloc[-1] / len(volume)) * 100)
            }
        
        # 6. Correlation stress scenarios (placeholder for market correlation)
        stress_scenarios["correlation_stress"] = {
            "market_correlation": 0.75,  # Placeholder
            "correlation_regime": "high",  # high/medium/low
            "correlation_impact": "moderate"  # high/moderate/low
        }
        
        # 7. Scenario analysis
        scenario_analysis = {}
        
        # Bear market scenario
        bear_scenario = {
            "market_decline": -0.20,  # 20% market decline
            "correlation_increase": 0.9,  # Increased correlation
            "volatility_increase": 2.0,  # Doubled volatility
            "expected_return": float(returns.mean() - 0.20 * 0.75),  # Market impact
            "expected_volatility": float(returns.std() * 2.0),
            "var_95": float(returns.quantile(0.05) * 2.0),
            "max_drawdown": float(drawdown.min() * 1.5)
        }
        scenario_analysis["bear_market"] = bear_scenario
        
        # Financial crisis scenario
        crisis_scenario = {
            "market_decline": -0.40,  # 40% market decline
            "correlation_increase": 0.95,  # Very high correlation
            "volatility_increase": 3.0,  # Tripled volatility
            "liquidity_decrease": 0.3,  # 70% liquidity reduction
            "expected_return": float(returns.mean() - 0.40 * 0.75),
            "expected_volatility": float(returns.std() * 3.0),
            "var_95": float(returns.quantile(0.05) * 3.0),
            "max_drawdown": float(drawdown.min() * 2.0)
        }
        scenario_analysis["financial_crisis"] = crisis_scenario
        
        # Black swan scenario
        black_swan_scenario = {
            "market_decline": -0.60,  # 60% market decline
            "correlation_increase": 0.98,  # Near perfect correlation
            "volatility_increase": 5.0,  # 5x volatility
            "liquidity_decrease": 0.1,  # 90% liquidity reduction
            "expected_return": float(returns.mean() - 0.60 * 0.75),
            "expected_volatility": float(returns.std() * 5.0),
            "var_95": float(returns.quantile(0.05) * 5.0),
            "max_drawdown": float(drawdown.min() * 3.0)
        }
        scenario_analysis["black_swan"] = black_swan_scenario
        
        # 8. Stress test summary
        stress_summary = {
            "overall_stress_score": 0,
            "stress_level": "low",
            "key_risk_factors": [],
            "mitigation_recommendations": []
        }
        
        # Calculate overall stress score
        stress_score = 0
        
        # Volatility stress component
        vol_percentile = stress_scenarios["volatility_stress"]["vol_percentile"]
        if vol_percentile > 80:
            stress_score += 25
            stress_summary["key_risk_factors"].append("High current volatility")
        elif vol_percentile > 60:
            stress_score += 15
        
        # Drawdown stress component
        current_dd = stress_scenarios["drawdown_stress"]["current_drawdown"]
        if current_dd < -0.10:
            stress_score += 25
            stress_summary["key_risk_factors"].append("Significant current drawdown")
        elif current_dd < -0.05:
            stress_score += 15
        
        # Tail risk stress component
        tail_freq = stress_scenarios["tail_risk_stress"]["2std_frequency"]
        if tail_freq > 0.05:
            stress_score += 20
            stress_summary["key_risk_factors"].append("Frequent extreme events")
        elif tail_freq > 0.025:
            stress_score += 10
        
        # Liquidity stress component
        if 'volume' in data.columns:
            vol_percentile = stress_scenarios["liquidity_stress"]["volume_percentile"]
            if vol_percentile < 20:
                stress_score += 15
                stress_summary["key_risk_factors"].append("Low current liquidity")
            elif vol_percentile < 40:
                stress_score += 10
        
        # Correlation stress component
        stress_score += 15  # Base correlation risk
        
        # Determine stress level
        if stress_score >= 70:
            stress_summary["stress_level"] = "high"
        elif stress_score >= 40:
            stress_summary["stress_level"] = "medium"
        else:
            stress_summary["stress_level"] = "low"
        
        stress_summary["overall_stress_score"] = stress_score
        
        # Generate mitigation recommendations
        if vol_percentile > 60:
            stress_summary["mitigation_recommendations"].append("Implement volatility-based position sizing")
            stress_summary["mitigation_recommendations"].append("Use wider stop-loss levels")
        
        if current_dd < -0.05:
            stress_summary["mitigation_recommendations"].append("Reduce position sizes")
            stress_summary["mitigation_recommendations"].append("Consider hedging strategies")
        
        if tail_freq > 0.025:
            stress_summary["mitigation_recommendations"].append("Prepare for extreme market moves")
            stress_summary["mitigation_recommendations"].append("Diversify across uncorrelated assets")
        
        if 'volume' in data.columns and vol_percentile < 40:
            stress_summary["mitigation_recommendations"].append("Avoid large position sizes")
            stress_summary["mitigation_recommendations"].append("Use limit orders for exits")
        
        return {
            "stress_scenarios": stress_scenarios,
            "scenario_analysis": scenario_analysis,
            "stress_summary": stress_summary
        }

    @staticmethod
    def calculate_scenario_analysis_metrics(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive scenario analysis metrics.
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            Dict containing scenario analysis
        """
        if len(data) < 100:
            return {"error": "insufficient_data_for_scenario_analysis"}
        
        # Calculate returns
        returns = data['close'].pct_change().dropna()
        
        # Monte Carlo simulation parameters from configuration
        n_simulations = Config.get("performance", "monte_carlo_simulations", 1000)
        n_days = Config.get("performance", "monte_carlo_days", 252)  # One year
        
        # Historical parameters
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Generate Monte Carlo simulations
        np.random.seed(42)  # For reproducibility
        simulations = np.random.normal(mean_return, std_return, (n_simulations, n_days))
        
        # Calculate cumulative returns for each simulation
        cumulative_simulations = np.cumprod(1 + simulations, axis=1)
        
        # Scenario analysis results
        scenario_results = {}
        
        # 1. Probability of different return scenarios
        final_returns = cumulative_simulations[:, -1] - 1
        
        scenario_results["return_probabilities"] = {
            "probability_positive_return": float(len(final_returns[final_returns > 0]) / n_simulations),
            "probability_10_percent_gain": float(len(final_returns[final_returns > 0.10]) / n_simulations),
            "probability_20_percent_gain": float(len(final_returns[final_returns > 0.20]) / n_simulations),
            "probability_10_percent_loss": float(len(final_returns[final_returns < -0.10]) / n_simulations),
            "probability_20_percent_loss": float(len(final_returns[final_returns < -0.20]) / n_simulations),
            "probability_30_percent_loss": float(len(final_returns[final_returns < -0.30]) / n_simulations)
        }
        
        # 2. Expected value analysis
        scenario_results["expected_values"] = {
            "expected_annual_return": float(np.mean(final_returns)),
            "expected_annual_volatility": float(np.std(final_returns)),
            "best_case_scenario": float(np.percentile(final_returns, 95)),
            "worst_case_scenario": float(np.percentile(final_returns, 5)),
            "median_scenario": float(np.percentile(final_returns, 50))
        }
        
        # 3. Risk-adjusted scenario analysis
        sharpe_ratios = final_returns / np.std(final_returns)
        scenario_results["risk_adjusted_scenarios"] = {
            "probability_positive_sharpe": float(len(sharpe_ratios[sharpe_ratios > 0]) / n_simulations),
            "probability_sharpe_above_1": float(len(sharpe_ratios[sharpe_ratios > 1]) / n_simulations),
            "probability_sharpe_above_2": float(len(sharpe_ratios[sharpe_ratios > 2]) / n_simulations),
            "average_sharpe_ratio": float(np.mean(sharpe_ratios)),
            "sharpe_ratio_95th_percentile": float(np.percentile(sharpe_ratios, 95))
        }
        
        # 4. Drawdown analysis for scenarios
        max_drawdowns = []
        for sim in cumulative_simulations:
            rolling_max = np.maximum.accumulate(sim)
            drawdown = (sim - rolling_max) / rolling_max
            max_drawdowns.append(np.min(drawdown))
        
        scenario_results["drawdown_scenarios"] = {
            "probability_max_dd_less_10": float(len([dd for dd in max_drawdowns if dd > -0.10]) / n_simulations),
            "probability_max_dd_less_20": float(len([dd for dd in max_drawdowns if dd > -0.20]) / n_simulations),
            "probability_max_dd_less_30": float(len([dd for dd in max_drawdowns if dd > -0.30]) / n_simulations),
            "expected_max_drawdown": float(np.mean(max_drawdowns)),
            "worst_case_drawdown": float(np.percentile(max_drawdowns, 5))
        }
        
        # 5. Time to recovery analysis
        recovery_times = []
        for sim in cumulative_simulations:
            rolling_max = np.maximum.accumulate(sim)
            drawdown = (sim - rolling_max) / rolling_max
            # Find time to recover from maximum drawdown
            max_dd_idx = np.argmin(drawdown)
            if max_dd_idx < len(sim) - 1:
                recovery_idx = max_dd_idx + np.argmax(sim[max_dd_idx:] >= rolling_max[max_dd_idx])
                recovery_time = recovery_idx - max_dd_idx
                recovery_times.append(recovery_time)
        
        if recovery_times:
            scenario_results["recovery_analysis"] = {
                "expected_recovery_time": float(np.mean(recovery_times)),
                "median_recovery_time": float(np.median(recovery_times)),
                "probability_recovery_within_30_days": float(len([rt for rt in recovery_times if rt <= 30]) / len(recovery_times)),
                "probability_recovery_within_60_days": float(len([rt for rt in recovery_times if rt <= 60]) / len(recovery_times)),
                "probability_recovery_within_90_days": float(len([rt for rt in recovery_times if rt <= 90]) / len(recovery_times))
            }
        else:
            scenario_results["recovery_analysis"] = {
                "expected_recovery_time": None,
                "median_recovery_time": None,
                "probability_recovery_within_30_days": None,
                "probability_recovery_within_60_days": None,
                "probability_recovery_within_90_days": None
            }
        
        # 6. Volatility regime scenarios
        volatility_scenarios = {
            "low_volatility": std_return * 0.5,
            "normal_volatility": std_return,
            "high_volatility": std_return * 2.0,
            "extreme_volatility": std_return * 3.0
        }
        
        vol_scenario_results = {}
        for vol_name, vol_level in volatility_scenarios.items():
            vol_simulations = np.random.normal(mean_return, vol_level, (n_simulations, n_days))
            vol_final_returns = np.cumprod(1 + vol_simulations, axis=1)[:, -1] - 1
            
            vol_scenario_results[vol_name] = {
                "expected_return": float(np.mean(vol_final_returns)),
                "expected_volatility": float(np.std(vol_final_returns)),
                "probability_positive_return": float(len(vol_final_returns[vol_final_returns > 0]) / n_simulations),
                "probability_20_percent_loss": float(len(vol_final_returns[vol_final_returns < -0.20]) / n_simulations)
            }
        
        scenario_results["volatility_regime_scenarios"] = vol_scenario_results
        
        # 7. Correlation scenarios (placeholder for market correlation)
        correlation_scenarios = {
            "low_correlation": 0.3,
            "normal_correlation": 0.6,
            "high_correlation": 0.8,
            "extreme_correlation": 0.95
        }
        
        corr_scenario_results = {}
        for corr_name, corr_level in correlation_scenarios.items():
            # Simulate market impact based on correlation
            market_return = -0.10  # 10% market decline
            correlated_return = mean_return + corr_level * market_return
            
            corr_scenario_results[corr_name] = {
                "expected_return": float(correlated_return),
                "market_impact": float(corr_level * market_return),
                "correlation_level": corr_level
            }
        
        scenario_results["correlation_scenarios"] = corr_scenario_results
        
        # 8. Scenario summary and recommendations
        scenario_summary = {
            "overall_risk_level": "medium",
            "key_scenario_insights": [],
            "recommended_actions": []
        }
        
        # Determine overall risk level
        prob_20pct_loss = scenario_results["return_probabilities"]["probability_20_percent_loss"]
        expected_max_dd = scenario_results["drawdown_scenarios"]["expected_max_drawdown"]
        
        if prob_20pct_loss > 0.15 or expected_max_dd < -0.25:
            scenario_summary["overall_risk_level"] = "high"
        elif prob_20pct_loss > 0.08 or expected_max_dd < -0.15:
            scenario_summary["overall_risk_level"] = "medium"
        else:
            scenario_summary["overall_risk_level"] = "low"
        
        # Generate insights
        if prob_20pct_loss > 0.10:
            scenario_summary["key_scenario_insights"].append("Significant probability of large losses")
        
        if expected_max_dd < -0.20:
            scenario_summary["key_scenario_insights"].append("Expected large maximum drawdowns")
        
        prob_positive = scenario_results["return_probabilities"]["probability_positive_return"]
        if prob_positive < 0.6:
            scenario_summary["key_scenario_insights"].append("Low probability of positive returns")
        
        # Generate recommendations
        if scenario_summary["overall_risk_level"] == "high":
            scenario_summary["recommended_actions"].append("Reduce position sizes significantly")
            scenario_summary["recommended_actions"].append("Implement strict stop-loss levels")
            scenario_summary["recommended_actions"].append("Consider hedging strategies")
        elif scenario_summary["overall_risk_level"] == "medium":
            scenario_summary["recommended_actions"].append("Moderate position sizing")
            scenario_summary["recommended_actions"].append("Use standard risk management")
        else:
            scenario_summary["recommended_actions"].append("Normal position sizing acceptable")
            scenario_summary["recommended_actions"].append("Standard risk management sufficient")
        
        return {
            "scenario_results": scenario_results,
            "scenario_summary": scenario_summary
        }

    @staticmethod
    def calculate_all_indicators_optimized(data: pd.DataFrame, stock_symbol: str = None) -> Dict[str, Any]:
        """
        Calculate all technical indicators with optimized data reduction (95-98% reduction in historical data).
        
        Args:
            data: DataFrame containing price and volume data
            stock_symbol: Stock symbol for sector classification
            
        Returns:
            Dict[str, Any]: Dictionary containing all calculated indicators with reduced historical data
        """
        indicators = {}
        
        # Calculate Moving Averages (only current values)
        sma_20 = TechnicalIndicators.calculate_sma(data, 'close', 20)
        sma_50 = TechnicalIndicators.calculate_sma(data, 'close', 50)
        sma_200 = TechnicalIndicators.calculate_sma(data, 'close', 200)
        ema_20 = TechnicalIndicators.calculate_ema(data, 'close', 20)
        ema_50 = TechnicalIndicators.calculate_ema(data, 'close', 50)
        
        # Calculate price to MA ratios
        current_price = data['close'].iloc[-1]
        # Handle division by zero or NaN for price to SMA 200
        if pd.isna(sma_200.iloc[-1]) or sma_200.iloc[-1] == 0:
            price_to_sma_200 = 0.0  # Default to neutral
        else:
            price_to_sma_200 = (current_price / sma_200.iloc[-1] - 1)
        
        # Handle division by zero or NaN for SMA 20 to SMA 50
        if pd.isna(sma_50.iloc[-1]) or sma_50.iloc[-1] == 0:
            sma_20_to_sma_50 = 0.0  # Default to neutral
        else:
            sma_20_to_sma_50 = (sma_20.iloc[-1] / sma_50.iloc[-1] - 1)
        
        # Check for Golden/Death Cross
        golden_cross = sma_20.iloc[-1] > sma_50.iloc[-1] and sma_20.iloc[-2] <= sma_50.iloc[-2]
        death_cross = sma_20.iloc[-1] < sma_50.iloc[-1] and sma_20.iloc[-2] >= sma_50.iloc[-2]
        
        # Add Moving Averages to indicators (only current values, no historical arrays)
        indicators['moving_averages'] = {
            'sma_20': float(sma_20.iloc[-1]),
            'sma_50': float(sma_50.iloc[-1]),
            'sma_200': float(sma_200.iloc[-1]),
            'ema_20': float(ema_20.iloc[-1]),
            'ema_50': float(ema_50.iloc[-1]),
            'price_to_sma_200': float(price_to_sma_200),
            'sma_20_to_sma_50': float(sma_20_to_sma_50),
            'golden_cross': bool(golden_cross),
            'death_cross': bool(death_cross),
            'signal': 'bullish' if sma_20.iloc[-1] > sma_50.iloc[-1] else 'bearish'
        }
        
        # Calculate RSI (only current and recent values)
        rsi = TechnicalIndicators.calculate_rsi(data)
        rsi_value = float(rsi.iloc[-1])
        
        # Determine RSI status with more granular levels
        if rsi_value > 70:
            rsi_status = 'overbought'
        elif rsi_value >= 60:
            rsi_status = 'near_overbought'
        elif rsi_value <= 40:
            rsi_status = 'near_oversold'
        elif rsi_value < 30:
            rsi_status = 'oversold'
        else:
            rsi_status = 'neutral'
        
        # Keep only last 5 RSI values for trend analysis
        rsi_recent = rsi.tail(5).tolist() if len(rsi) >= 5 else rsi.tolist()
        
        indicators['rsi'] = {
            'rsi_14': rsi_value,
            'recent_values': rsi_recent,
            'trend': 'up' if rsi.iloc[-1] > rsi.iloc[-2] else 'down',
            'status': rsi_status,
            'signal': 'oversold' if rsi_value < 30 else 'overbought' if rsi_value > 70 else 'neutral'
        }
        
        # Calculate MACD (only current values)
        macd_line, signal_line, histogram = TechnicalIndicators.calculate_macd(data)
        indicators['macd'] = {
            'macd_line': float(macd_line.iloc[-1]),
            'signal_line': float(signal_line.iloc[-1]),
            'histogram': float(histogram.iloc[-1]),
            'signal': 'bullish' if macd_line.iloc[-1] > signal_line.iloc[-1] else 'bearish'
        }
        
        # Calculate Bollinger Bands (only current values)
        upper_band, middle_band, lower_band = TechnicalIndicators.calculate_bollinger_bands(data)
        current_price = data['close'].iloc[-1]
        
        # Handle division by zero or NaN for percent_b
        band_width = upper_band.iloc[-1] - lower_band.iloc[-1]
        if pd.isna(band_width) or band_width == 0:
            percent_b = 0.5  # Default to middle of band
        else:
            percent_b = (current_price - lower_band.iloc[-1]) / band_width
        
        # Handle division by zero or NaN for bandwidth
        if pd.isna(middle_band.iloc[-1]) or middle_band.iloc[-1] == 0:
            bandwidth = 0.0  # Default to zero bandwidth
        else:
            bandwidth = band_width / middle_band.iloc[-1]
        
        indicators['bollinger_bands'] = {
            'upper_band': float(upper_band.iloc[-1]),
            'middle_band': float(middle_band.iloc[-1]),
            'lower_band': float(lower_band.iloc[-1]),
            'percent_b': float(percent_b),
            'bandwidth': float(bandwidth),
            'signal': 'squeeze' if bandwidth < 0.1 else 'expansion'
        }
        
        # Calculate Volume indicators (only current values)
        volume_ma = data['volume'].rolling(window=20).mean()
        # Handle division by zero or NaN
        if pd.isna(volume_ma.iloc[-1]) or volume_ma.iloc[-1] == 0:
            volume_ratio = 1.0  # Default to normal volume ratio
        else:
            volume_ratio = data['volume'].iloc[-1] / volume_ma.iloc[-1]
        obv = TechnicalIndicators.calculate_obv(data)
        
        indicators['volume'] = {
            'volume_ratio': float(volume_ratio),
            'obv': float(obv.iloc[-1]),
            'obv_trend': 'up' if obv.iloc[-1] > obv.iloc[-2] else 'down',
            'signal': 'high_volume' if volume_ratio > 1.5 else 'low_volume' if volume_ratio < 0.5 else 'normal'
        }
        
        # Calculate ADX (only current values)
        adx, plus_di, minus_di = TechnicalIndicators.calculate_adx(data)
        trend_direction = 'bullish' if plus_di.iloc[-1] > minus_di.iloc[-1] else 'bearish'
        
        indicators['adx'] = {
            'adx': float(adx.iloc[-1]),
            'plus_di': float(plus_di.iloc[-1]),
            'minus_di': float(minus_di.iloc[-1]),
            'trend_direction': trend_direction,
            'trend_strength': 'strong' if adx.iloc[-1] > 25 else 'weak'
        }
        
        # Add trend data (consolidated)
        indicators['trend_data'] = {
            'direction': trend_direction,
            'strength': 'strong' if adx.iloc[-1] > 25 else 'weak',
            'adx': float(adx.iloc[-1]),
            'plus_di': float(plus_di.iloc[-1]),
            'minus_di': float(minus_di.iloc[-1])
        }
        
        # Calculate Enhanced Volatility Indicators (only current values)
        atr = TechnicalIndicators.calculate_atr(data)
        atr_20 = atr.rolling(window=20).mean()
        volatility_ratio = atr.iloc[-1] / atr_20.iloc[-1] if not pd.isna(atr_20.iloc[-1]) and atr_20.iloc[-1] != 0 else 1.0
        
        # Bollinger Band squeeze detection
        bb_squeeze = bandwidth < 0.1  # Low bandwidth indicates squeeze
        
        # Historical volatility percentile (20-period)
        volatility_percentile = (atr.iloc[-20:].rank().iloc[-1] / 20) * 100 if len(atr) >= 20 else 50
        
        indicators['volatility'] = {
            'atr': float(atr.iloc[-1]),
            'atr_20_avg': float(atr_20.iloc[-1]) if not pd.isna(atr_20.iloc[-1]) else None,
            'volatility_ratio': float(volatility_ratio),
            'bb_squeeze': bool(bb_squeeze),
            'volatility_percentile': float(volatility_percentile),
            'volatility_regime': 'high' if volatility_ratio > 1.5 else 'low' if volatility_ratio < 0.7 else 'normal'
        }
        
        # Calculate Enhanced Volume Indicators (only current values)
        vwap = TechnicalIndicators.calculate_vwap(data)
        mfi = TechnicalIndicators.calculate_money_flow_index(data)
        
        # Volume profile analysis (simplified)
        volume_profile = TechnicalIndicators.calculate_volume_profile(data)
        
        # Comprehensive volume analysis (simplified)
        enhanced_volume_analysis = TechnicalIndicators.calculate_enhanced_volume_analysis(data)
        
        indicators['enhanced_volume'] = {
            'vwap': float(vwap.iloc[-1]) if not pd.isna(vwap.iloc[-1]) else None,
            'mfi': float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else None,
            'mfi_status': 'overbought' if mfi.iloc[-1] > 80 else 'oversold' if mfi.iloc[-1] < 20 else 'neutral',
            'price_vs_vwap': float((current_price / vwap.iloc[-1] - 1) * 100) if not pd.isna(vwap.iloc[-1]) and vwap.iloc[-1] != 0 else 0.0,
            'comprehensive_analysis': enhanced_volume_analysis
        }
        
        # Calculate Enhanced Momentum Indicators (only current values)
        stochastic_k, stochastic_d = TechnicalIndicators.calculate_stochastic_oscillator(data)
        williams_r = TechnicalIndicators.calculate_williams_r(data)
        
        # RSI divergence detection (simplified)
        rsi_divergence = TechnicalIndicators.detect_rsi_divergence(data['close'], rsi)
        
        indicators['enhanced_momentum'] = {
            'stochastic_k': float(stochastic_k.iloc[-1]) if not pd.isna(stochastic_k.iloc[-1]) else None,
            'stochastic_d': float(stochastic_d.iloc[-1]) if not pd.isna(stochastic_d.iloc[-1]) else None,
            'stochastic_status': 'overbought' if stochastic_k.iloc[-1] > 80 else 'oversold' if stochastic_k.iloc[-1] < 20 else 'neutral',
            'williams_r': float(williams_r.iloc[-1]) if not pd.isna(williams_r.iloc[-1]) else None,
            'williams_r_status': 'overbought' if williams_r.iloc[-1] < -80 else 'oversold' if williams_r.iloc[-1] > -20 else 'neutral',
            'rsi_divergence': rsi_divergence
        }
        
        # Calculate Enhanced Trend Strength (simplified)
        trend_strength = TechnicalIndicators.calculate_trend_strength(data, sma_20, sma_50, sma_200)
        indicators['trend_strength'] = trend_strength
        
        # Calculate Enhanced Support/Resistance (simplified - only top levels)
        enhanced_levels = TechnicalIndicators.calculate_enhanced_support_resistance(data)
        indicators['enhanced_levels'] = enhanced_levels
        
        # === PHASE 2 FEATURES (Simplified) ===
        
        # Calculate Multi-Timeframe Analysis (simplified)
        multi_timeframe = TechnicalIndicators.calculate_multi_timeframe_analysis(data)
        indicators['multi_timeframe'] = multi_timeframe
        
        # Calculate Advanced Risk Metrics (simplified)
        advanced_risk = TechnicalIndicators.calculate_advanced_risk_metrics(data)
        indicators['advanced_risk'] = advanced_risk
        
        # Calculate Phase 3 Advanced Risk Metrics (simplified)
        stress_testing = TechnicalIndicators.calculate_stress_testing_metrics(data)
        scenario_analysis = TechnicalIndicators.calculate_scenario_analysis_metrics(data)
        indicators['stress_testing'] = stress_testing
        indicators['scenario_analysis'] = scenario_analysis
        
        return indicators


class DataCollector:
    @staticmethod
    def collect_all_data(data: pd.DataFrame, stock_symbol: str = None) -> Dict[str, Any]:
        """
        Collect all technical analysis data with sector awareness.
        
        Args:
            data: DataFrame containing price and volume data
            stock_symbol: Stock symbol for sector classification
            
        Returns:
            Dict[str, Any]: Dictionary containing all technical analysis data
        """
        # Ensure index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except:
                data.index = pd.date_range(start='2020-01-01', periods=len(data))
        
        # Calculate all indicators with sector-aware market metrics
        indicators = TechnicalIndicators.calculate_all_indicators(data, stock_symbol)
        
        # Add metadata
        metadata = {
            'start': data.index[0].isoformat(),
            'end': data.index[-1].isoformat(),
            'period': len(data),
            'last_price': float(data['close'].iloc[-1]),
            'last_volume': float(data['volume'].iloc[-1])
        }
        
        if stock_symbol:
            sector = sector_classifier.get_stock_sector(stock_symbol)
            metadata['sector'] = sector
            metadata['sector_indices'] = sector_classifier.get_sector_indices(sector) if sector else []
        
        return indicators


class IndianMarketMetricsProvider:
    """
    Provides real Indian market metrics using Zerodha API.
    All symbols and tokens verified from zerodha_instruments.csv
    """
    def __init__(self):
        self.zerodha_client = ZerodhaDataClient()
        self.cache = {}
        self.cache_duration = 900  # 15 minutes cache
        self.sector_classifier = sector_classifier  # Import from sector_classifier
        
        # Enhanced market indices with sector-specific indices
        self.market_indices = {
            'NIFTY_50': {
                'symbol': 'NIFTY 50',
                'token': 256265,
                'exchange': 'NSE'
            },
            'INDIA_VIX': {
                'symbol': 'INDIA VIX',
                'token': 264969,
                'exchange': 'NSE'
            },
            # Sector-specific indices
            'NIFTY_BANK': {
                'symbol': 'NIFTY BANK',
                'token': 260105,
                'exchange': 'NSE'
            },
            'NIFTY_IT': {
                'symbol': 'NIFTY IT',
                'token': 259849,
                'exchange': 'NSE'
            },
            'NIFTY_PHARMA': {
                'symbol': 'NIFTY PHARMA',
                'token': 262409,
                'exchange': 'NSE'
            },
            'NIFTY_AUTO': {
                'symbol': 'NIFTY AUTO',
                'token': 263433,
                'exchange': 'NSE'
            },
            'NIFTY_FMCG': {
                'symbol': 'NIFTY FMCG',
                'token': 261897,
                'exchange': 'NSE'
            },
            'NIFTY_ENERGY': {
                'symbol': 'NIFTY ENERGY',
                'token': 261641,
                'exchange': 'NSE'
            },
            'NIFTY_METAL': {
                'symbol': 'NIFTY METAL',
                'token': 263689,
                'exchange': 'NSE'
            },
            'NIFTY_REALTY': {
                'symbol': 'NIFTY REALTY',
                'token': 261129,
                'exchange': 'NSE'
            },
            'NIFTY_MEDIA': {
                'symbol': 'NIFTY MEDIA',
                'token': 263945,
                'exchange': 'NSE'
            },
            'NIFTY_CONSUMER_DURABLES': {
                'symbol': 'NIFTY CONSR DURBL',
                'token': 288777,
                'exchange': 'NSE'
            },
            'NIFTY_HEALTHCARE': {
                'symbol': 'NIFTY HEALTHCARE',
                'token': 288521,
                'exchange': 'NSE'
            },
            'NIFTY_INFRA': {
                'symbol': 'NIFTY INFRA',
                'token': 261385,
                'exchange': 'NSE'
            },
            'NIFTY_OIL_GAS': {
                'symbol': 'NIFTY OIL AND GAS',
                'token': 289033,
                'exchange': 'NSE'
            },
            'NIFTY_SERV_SECTOR': {
                'symbol': 'NIFTY SERV SECTOR',
                'token': 263177,
                'exchange': 'NSE'
            }
        }
    
    def get_sector_index_data(self, sector: str, period: int = 365) -> pd.DataFrame:
        """
        Get sector-specific index data.
        
        Args:
            sector: Sector name (e.g., 'BANKING', 'IT')
            period: Number of days of historical data
            
        Returns:
            pd.DataFrame: Historical data for sector index
        """
        try:
            primary_index = self.sector_classifier.get_primary_sector_index(sector)
            if not primary_index:
                logging.warning(f"No primary index found for sector: {sector}")
                return None
            
            # Find the index mapping
            index_key = None
            for key, data in self.market_indices.items():
                if data['symbol'] == primary_index:
                    index_key = key
                    break
            
            if not index_key:
                logging.error(f"Index mapping not found for: {primary_index}")
                return None
            
            return self.zerodha_client.get_historical_data(
                symbol=self.market_indices[index_key]['symbol'],
                exchange=self.market_indices[index_key]['exchange'],
                period=period
            )
        except Exception as e:
            logging.error(f"Error fetching sector index data for {sector}: {e}")
            return None
    
    def get_basic_market_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Get basic market metrics without sector-specific data.
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            Dict containing basic market metrics
        """
        try:
            # Get Nifty 50 data for market comparison
            nifty_data = self.get_nifty_50_data(period=365)
            
            if nifty_data is not None and len(nifty_data) > 0:
                # Calculate stock returns
                stock_returns = data['close'].pct_change().dropna()
                market_returns = nifty_data['close'].pct_change().dropna()
                
                # Align the series
                common_dates = stock_returns.index.intersection(market_returns.index)
                if len(common_dates) > 30:
                    stock_returns = stock_returns.loc[common_dates]
                    market_returns = market_returns.loc[common_dates]
                    
                    # Calculate beta and correlation
                    beta = self.calculate_beta(stock_returns, market_returns)
                    correlation = self.calculate_correlation(stock_returns, market_returns)
                else:
                    beta = 1.0
                    correlation = 0.0
            else:
                beta = 1.0
                correlation = 0.0
            
            # Get risk-free rate
            risk_free_rate = self.get_risk_free_rate()
            
            return {
                "beta": float(beta),
                "correlation": float(correlation),
                "risk_free_rate": float(risk_free_rate)
            }
            
        except Exception as e:
            # Return default values if calculation fails
            return {
                "beta": 1.0,
                "correlation": 0.0,
                "risk_free_rate": 6.5  # Default Indian risk-free rate
            }
    
    def get_enhanced_market_metrics(self, data: pd.DataFrame, stock_symbol: str) -> Dict[str, float]:
        """
        Calculate enhanced market metrics using sector-specific indices with real-time data.
        
        Args:
            data: DataFrame containing OHLCV data
            stock_symbol: Stock symbol for sector classification
            
        Returns:
            Dict containing enhanced market metrics
        """
        try:
            logging.info(f"Calculating enhanced market metrics for {stock_symbol}")
            stock_returns = data['close'].pct_change().dropna()
            
            # Get sector for the stock
            sector = self.sector_classifier.get_stock_sector(stock_symbol)
            
            # Initialize metrics
            beta = 1.0
            correlation = 0.6
            sector_beta = 1.0
            sector_correlation = 0.6
            market_beta = 1.0
            market_correlation = 0.6
            
            # Get NIFTY 50 data (market benchmark) - real-time
            nifty_data = self.get_nifty_50_data(365)
            if nifty_data is not None and len(nifty_data) > 30:
                market_returns = nifty_data['close'].pct_change().dropna()
                market_beta = self.calculate_beta(stock_returns, market_returns)
                market_correlation = self.calculate_correlation(stock_returns, market_returns)
                
                # Calculate market performance metrics
                market_cumulative_return = (1 + market_returns).prod() - 1
                market_volatility = market_returns.std() * np.sqrt(252)
                market_sharpe = (market_returns.mean() * 252 - self.get_risk_free_rate()) / market_volatility if market_volatility > 0 else 0
            else:
                market_cumulative_return = 0
                market_volatility = 0.2
                market_sharpe = 0
            
            # Get sector-specific data if available - real-time
            sector_cumulative_return = 0
            sector_volatility = 0.2
            sector_sharpe = 0
            
            if sector:
                sector_data = self.get_sector_index_data(sector, 365)
                if sector_data is not None and len(sector_data) > 30:
                    sector_returns = sector_data['close'].pct_change().dropna()
                    sector_beta = self.calculate_beta(stock_returns, sector_returns)
                    sector_correlation = self.calculate_correlation(stock_returns, sector_returns)
                    
                    # Calculate sector performance metrics
                    sector_cumulative_return = (1 + sector_returns).prod() - 1
                    sector_volatility = sector_returns.std() * np.sqrt(252)
                    sector_sharpe = (sector_returns.mean() * 252 - self.get_risk_free_rate()) / sector_volatility if sector_volatility > 0 else 0
                    
                    # Use sector metrics as primary, market metrics as secondary
                    beta = sector_beta
                    correlation = sector_correlation
                    logging.info(f"Using sector-specific metrics for {stock_symbol} ({sector})")
                else:
                    logging.warning(f"Sector data unavailable for {sector}, using market metrics")
            else:
                logging.info(f"No sector classification for {stock_symbol}, using market metrics")
            
            # Get real-time market data
            risk_free_rate = self.get_risk_free_rate()
            vix_data = self.get_india_vix_data(30)
            current_vix = vix_data['close'].iloc[-1] if vix_data is not None else 15.0
            
            # Calculate stock performance metrics
            stock_cumulative_return = (1 + stock_returns).prod() - 1
            stock_volatility = stock_returns.std() * np.sqrt(252)
            stock_sharpe = (stock_returns.mean() * 252 - risk_free_rate) / stock_volatility if stock_volatility > 0 else 0
            
            # Calculate relative performance
            market_excess_return = stock_cumulative_return - market_cumulative_return
            sector_excess_return = stock_cumulative_return - sector_cumulative_return if sector else 0
            
            # Calculate momentum metrics
            momentum_20d = (data['close'].iloc[-1] / data['close'].iloc[-20] - 1) if len(data) >= 20 else 0
            momentum_50d = (data['close'].iloc[-1] / data['close'].iloc[-50] - 1) if len(data) >= 50 else 0
            
            # Market sentiment indicators
            market_sentiment = "Bullish" if market_cumulative_return > 0.1 else "Bearish" if market_cumulative_return < -0.1 else "Neutral"
            sector_sentiment = "Bullish" if sector_cumulative_return > 0.1 else "Bearish" if sector_cumulative_return < -0.1 else "Neutral" if sector else "Unknown"
            
            return {
                "beta": float(beta),
                "correlation": float(correlation),
                "sector_beta": float(sector_beta),
                "sector_correlation": float(sector_correlation),
                "market_beta": float(market_beta),
                "market_correlation": float(market_correlation),
                "sector": sector,
                "risk_free_rate": float(risk_free_rate),
                "current_vix": float(current_vix),
                "data_source": f"Zerodha API - {'Sector-specific' if sector else 'Market'} Data",
                "nifty_data_points": len(nifty_data) if nifty_data is not None else 0,
                "sector_data_points": len(sector_data) if sector and sector_data is not None else 0,
                
                # Performance metrics
                "stock_return": float(stock_cumulative_return),
                "market_return": float(market_cumulative_return),
                "sector_return": float(sector_cumulative_return),
                "market_excess_return": float(market_excess_return),
                "sector_excess_return": float(sector_excess_return),
                
                # Risk metrics
                "stock_volatility": float(stock_volatility),
                "market_volatility": float(market_volatility),
                "sector_volatility": float(sector_volatility),
                "stock_sharpe": float(stock_sharpe),
                "market_sharpe": float(market_sharpe),
                "sector_sharpe": float(sector_sharpe),
                
                # Momentum metrics
                "momentum_20d": float(momentum_20d),
                "momentum_50d": float(momentum_50d),
                
                # Sentiment indicators
                "market_sentiment": market_sentiment,
                "sector_sentiment": sector_sentiment,
                
                # Real-time indicators
                "last_update": pd.Timestamp.now().isoformat(),
                "data_freshness": "real_time" if self._is_market_open() else "last_close",
                "market_status": "open" if self._is_market_open() else "closed"
            }
            
        except Exception as e:
            logging.error(f"Error calculating enhanced market metrics for {stock_symbol}: {e}")
            return self._get_default_market_metrics(stock_symbol)
    
    def _is_market_open(self) -> bool:
        """Check if Indian market is currently open."""
        try:
            now = pd.Timestamp.now(tz='Asia/Kolkata')
            is_weekday = now.weekday() < 5  # Monday to Friday
            market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            return is_weekday and market_open <= now <= market_close
        except Exception:
            return False
    
    def _get_default_market_metrics(self, stock_symbol: str) -> Dict[str, float]:
        """Get default market metrics when real data is unavailable."""
        return {
            "beta": 1.0,
            "correlation": 0.6,
            "sector_beta": 1.0,
            "sector_correlation": 0.6,
            "market_beta": 1.0,
            "market_correlation": 0.6,
            "sector": None,
            "risk_free_rate": 0.07,
            "current_vix": 15.0,
            "data_source": "Default values - real data unavailable",
            "stock_return": 0.0,
            "market_return": 0.0,
            "sector_return": 0.0,
            "market_excess_return": 0.0,
            "sector_excess_return": 0.0,
            "stock_volatility": 0.2,
            "market_volatility": 0.2,
            "sector_volatility": 0.2,
            "stock_sharpe": 0.0,
            "market_sharpe": 0.0,
            "sector_sharpe": 0.0,
            "momentum_20d": 0.0,
            "momentum_50d": 0.0,
            "market_sentiment": "Neutral",
            "sector_sentiment": "Unknown",
            "last_update": pd.Timestamp.now().isoformat(),
            "data_freshness": "default",
            "market_status": "unknown"
        }
    
    def get_nifty_50_data(self, period: int = 365) -> pd.DataFrame:
        try:
            return self.zerodha_client.get_historical_data(
                symbol=self.market_indices['NIFTY_50']['symbol'],
                exchange=self.market_indices['NIFTY_50']['exchange'],
                period=period
            )
        except Exception as e:
            logging.error(f"Error fetching NIFTY 50 data: {e}")
            return None
    def get_india_vix_data(self, period: int = 30) -> pd.DataFrame:
        try:
            return self.zerodha_client.get_historical_data(
                symbol=self.market_indices['INDIA_VIX']['symbol'],
                exchange=self.market_indices['INDIA_VIX']['exchange'],
                period=period
            )
        except Exception as e:
            logging.error(f"Error fetching INDIA VIX data: {e}")
            return None
    def calculate_beta(self, stock_returns: pd.Series, market_returns: pd.Series) -> float:
        try:
            aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
            if len(aligned) < 30:
                return 1.0
            cov = np.cov(aligned.iloc[:,0], aligned.iloc[:,1])[0,1]
            var = np.var(aligned.iloc[:,1])
            if var == 0:
                return 1.0
            beta = cov / var
            return float(max(0.1, min(3.0, beta)))
        except Exception as e:
            logging.error(f"Error calculating beta: {e}")
            return 1.0
    def calculate_correlation(self, stock_returns: pd.Series, market_returns: pd.Series) -> float:
        try:
            aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
            if len(aligned) < 30:
                return 0.5
            corr = np.corrcoef(aligned.iloc[:,0], aligned.iloc[:,1])[0,1]
            if np.isnan(corr):
                return 0.5
            return float(corr)
        except Exception as e:
            logging.error(f"Error calculating correlation: {e}")
            return 0.5
    def get_risk_free_rate(self) -> float:
        try:
            return 0.072  # 7.2% as decimal
        except Exception as e:
            logging.error(f"Error fetching risk-free rate: {e}")
            return 0.07

    async def get_nifty_50_data_async(self, period: int = 365) -> pd.DataFrame:
        """Async version of get_nifty_50_data."""
        try:
            return await self.zerodha_client.get_historical_data_async(
                symbol=self.market_indices['NIFTY_50']['symbol'],
                exchange=self.market_indices['NIFTY_50']['exchange'],
                period=period
            )
        except Exception as e:
            logging.error(f"Error fetching NIFTY 50 data: {e}")
            return None

    async def get_india_vix_data_async(self, period: int = 30) -> pd.DataFrame:
        """Async version of get_india_vix_data."""
        try:
            return await self.zerodha_client.get_historical_data_async(
                symbol=self.market_indices['INDIA_VIX']['symbol'],
                exchange=self.market_indices['INDIA_VIX']['exchange'],
                period=period
            )
        except Exception as e:
            logging.error(f"Error fetching INDIA VIX data: {e}")
            return None

    async def get_sector_index_data_async(self, sector: str, period: int = 365) -> pd.DataFrame:
        """Async version of get_sector_index_data."""
        try:
            primary_index = self.sector_classifier.get_primary_sector_index(sector)
            if not primary_index:
                logging.warning(f"No primary index found for sector: {sector}")
                return None
            
            # Find the index mapping
            index_key = None
            for key, data in self.market_indices.items():
                if data['symbol'] == primary_index:
                    index_key = key
                    break
            
            if not index_key:
                logging.error(f"Index mapping not found for: {primary_index}")
                return None
            
            return await self.zerodha_client.get_historical_data_async(
                symbol=self.market_indices[index_key]['symbol'],
                exchange=self.market_indices[index_key]['exchange'],
                period=period
            )
        except Exception as e:
            logging.error(f"Error fetching sector index data for {sector}: {e}")
            return None

    async def get_enhanced_market_metrics_async(self, data: pd.DataFrame, stock_symbol: str) -> Dict[str, float]:
        """Async version of get_enhanced_market_metrics."""
        try:
            # Check if market is open
            if not self._is_market_open():
                return self._get_default_market_metrics(stock_symbol)
            
            # Get stock returns
            stock_returns = data['close'].pct_change().dropna()
            if len(stock_returns) < 30:
                return self._get_default_market_metrics(stock_symbol)
            
            # Fetch all index data concurrently
            tasks = [
                self.get_nifty_50_data_async(365),
                self.get_india_vix_data_async(30)
            ]
            
            # Get sector info if available
            sector = None
            if stock_symbol:
                try:
                    sector = self.sector_classifier.get_stock_sector(stock_symbol)
                    if sector and sector != 'UNKNOWN':
                        tasks.append(self.get_sector_index_data_async(sector, 365))
                except Exception as e:
                    logging.warning(f"Could not classify stock {stock_symbol}: {e}")
            
            # Execute all async tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            nifty_data = results[0] if not isinstance(results[0], Exception) else None
            vix_data = results[1] if not isinstance(results[1], Exception) else None
            sector_data = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else None
            
            # Calculate metrics
            metrics = {}
            
            # Market metrics (NIFTY 50)
            if nifty_data is not None and len(nifty_data) >= 30:
                market_returns = nifty_data['close'].pct_change().dropna()
                aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
                if len(aligned_data) >= 30:
                    stock_aligned = aligned_data.iloc[:, 0]
                    market_aligned = aligned_data.iloc[:, 1]
                    
                    metrics.update({
                        "beta": self.calculate_beta(stock_aligned, market_aligned),
                        "correlation": self.calculate_correlation(stock_aligned, market_aligned),
                        "market_beta": self.calculate_beta(stock_aligned, market_aligned),
                        "market_correlation": self.calculate_correlation(stock_aligned, market_aligned),
                        "market_return": float((1 + market_aligned).prod() - 1),
                        "market_volatility": float(market_aligned.std() * np.sqrt(252)),
                        "market_sharpe": float((market_aligned.mean() * 252 - 0.07) / (market_aligned.std() * np.sqrt(252))) if market_aligned.std() > 0 else 0.0
                    })
            
            # VIX metrics
            if vix_data is not None and len(vix_data) > 0:
                current_vix = vix_data['close'].iloc[-1]
                metrics["current_vix"] = float(current_vix)
                metrics["market_sentiment"] = "Bearish" if current_vix > 25 else "Bullish" if current_vix < 15 else "Neutral"
            
            # Sector metrics
            if sector_data is not None and len(sector_data) >= 30:
                sector_returns = sector_data['close'].pct_change().dropna()
                aligned_data = pd.concat([stock_returns, sector_returns], axis=1).dropna()
                if len(aligned_data) >= 30:
                    stock_aligned = aligned_data.iloc[:, 0]
                    sector_aligned = aligned_data.iloc[:, 1]
                    
                    metrics.update({
                        "sector_beta": self.calculate_beta(stock_aligned, sector_aligned),
                        "sector_correlation": self.calculate_correlation(stock_aligned, sector_aligned),
                        "sector_return": float((1 + sector_aligned).prod() - 1),
                        "sector_volatility": float(sector_aligned.std() * np.sqrt(252)),
                        "sector_sharpe": float((sector_aligned.mean() * 252 - 0.07) / (sector_aligned.std() * np.sqrt(252))) if sector_aligned.std() > 0 else 0.0,
                        "sector_excess_return": float((1 + stock_aligned).prod() - 1) - float((1 + sector_aligned).prod() - 1)
                    })
            
            # Add common metrics
            metrics.update({
                "stock_return": float((1 + stock_returns).prod() - 1),
                "stock_volatility": float(stock_returns.std() * np.sqrt(252)),
                "stock_sharpe": float((stock_returns.mean() * 252 - 0.07) / (stock_returns.std() * np.sqrt(252))) if stock_returns.std() > 0 else 0.0,
                "risk_free_rate": 0.07,
                "sector": sector,
                "data_source": "Real-time async data",
                "last_update": pd.Timestamp.now().isoformat(),
                "data_freshness": "real-time",
                "market_status": "open"
            })
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error in async market metrics calculation: {e}")
            return self._get_default_market_metrics(stock_symbol)

market_metrics_provider = IndianMarketMetricsProvider()


