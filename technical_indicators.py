import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Callable
import matplotlib.dates as mdates

import os

# Add new imports for pattern modules
from patterns.recognition import PatternRecognition
from patterns.visualization import PatternVisualizer


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
    def calculate_macd(data: pd.DataFrame, column: str = 'close', fast_period: int = 12, 
                      slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data: DataFrame containing price data
            column: Column name to use for calculation
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: MACD line, signal line, and histogram
        """
        fast_ema = TechnicalIndicators.calculate_ema(data, column, fast_period)
        slow_ema = TechnicalIndicators.calculate_ema(data, column, slow_period)
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, column: str = 'close', window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            data: DataFrame containing price data
            column: Column name to use for calculation
            window: RSI period
            
        Returns:
            pd.Series: RSI values
        """
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
    def calculate_stochastic_oscillator(data: pd.DataFrame, k_window: int = 14, 
                                       d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            data: DataFrame containing price data
            k_window: %K period
            d_window: %D period
            
        Returns:
            Tuple[pd.Series, pd.Series]: %K and %D values
        """
        low_min = data['low'].rolling(window=k_window).min()
        high_max = data['high'].rolling(window=k_window).max()
        
        # Calculate %K
        k = 100 * ((data['close'] - low_min) / (high_max - low_min))
        
        # Calculate %D
        d = k.rolling(window=d_window).mean()
        
        return k, d
    
    @staticmethod
    def calculate_obv(data: pd.DataFrame) -> pd.Series:
        """
        Calculate On-Balance Volume.
        
        Args:
            data: DataFrame containing price and volume data
            
        Returns:
            pd.Series: OBV values
        """
        close_diff = data['close'].diff()
        
        obv = pd.Series(index=data.index)
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
    def calculate_all_indicators(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all technical indicators.
        
        Args:
            data: DataFrame containing price and volume data
            
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
        
        return indicators


class IndicatorComparisonAnalyzer:
    """
    Class for comparing and analyzing multiple technical indicators to generate consensus signals.
    """
    
    def _convert_numpy_types(self, obj):
        """
        Convert numpy types to Python native types.
        
        Args:
            obj: Object that may contain numpy types
            
        Returns:
            Object with numpy types converted to Python native types
        """
        import numpy as np
        import math
        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isinf(obj) or np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, float):
            if math.isinf(obj) or math.isnan(obj):
                return None
            return obj
        return obj
    
    def analyze(self, data: pd.DataFrame, indicators: Dict[str, Any], patterns: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze stock data using technical indicators and patterns.
        
        Args:
            data: DataFrame containing price and volume data
            indicators: Dictionary containing calculated indicators
            patterns: Optional dictionary containing detected patterns
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        # Get indicator consensus
        consensus = self.analyze_indicator_consensus(indicators)
        
        # Prepare analysis results
        analysis_results = {
            'consensus': consensus,
            'indicators': indicators,
            'patterns': patterns,
            'summary': {
                'overall_signal': consensus['overall_signal'],
                'signal_strength': consensus['signal_strength'],
                'bullish_percentage': consensus['bullish_percentage'],
                'bearish_percentage': consensus['bearish_percentage'],
                'neutral_percentage': consensus['neutral_percentage']
            }
        }
        
        # Convert numpy types to Python native types
        return self._convert_numpy_types(analysis_results)
    
    @staticmethod
    def analyze_indicator_consensus(indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze multiple indicators to determine consensus signals.
        
        Args:
            indicators: Dictionary containing calculated indicators
            
        Returns:
            Dict[str, Any]: Consensus analysis results
        """
        # Initialize signal counters
        bullish_signals = 0
        bearish_signals = 0
        neutral_signals = 0
        
        # Initialize signal details
        signal_details = []
        
        # Analyze Moving Averages
        ma_data = indicators['moving_averages']
        
        # Price vs 200-day MA
        if ma_data['price_to_sma_200'] > 0:
            bullish_signals += 1
            signal_details.append({
                'indicator': 'Price vs 200-day MA',
                'signal': 'bullish',
                'strength': 'strong',
                'description': f"Price is {ma_data['price_to_sma_200']*100:.2f}% above 200-day MA, indicating long-term uptrend"
            })
        elif ma_data['price_to_sma_200'] < 0:
            bearish_signals += 1
            signal_details.append({
                'indicator': 'Price vs 200-day MA',
                'signal': 'bearish',
                'strength': 'strong',
                'description': f"Price is {abs(ma_data['price_to_sma_200'])*100:.2f}% below 200-day MA, indicating long-term downtrend"
            })
        else:
            neutral_signals += 1
            signal_details.append({
                'indicator': 'Price vs 200-day MA',
                'signal': 'neutral',
                'strength': 'weak',
                'description': "Price is at the 200-day MA, indicating potential trend change"
            })
        
        # 20-day MA vs 50-day MA (Golden/Death Cross)
        if ma_data['sma_20_to_sma_50'] > 0:
            bullish_signals += 1
            signal_details.append({
                'indicator': '20-day MA vs 50-day MA',
                'signal': 'bullish',
                'strength': 'moderate',
                'description': f"20-day MA is above 50-day MA by {ma_data['sma_20_to_sma_50']*100:.2f}%, indicating medium-term uptrend"
            })
        elif ma_data['sma_20_to_sma_50'] < 0:
            bearish_signals += 1
            signal_details.append({
                'indicator': '20-day MA vs 50-day MA',
                'signal': 'bearish',
                'strength': 'moderate',
                'description': f"20-day MA is below 50-day MA by {abs(ma_data['sma_20_to_sma_50'])*100:.2f}%, indicating medium-term downtrend"
            })
        else:
            neutral_signals += 1
            signal_details.append({
                'indicator': '20-day MA vs 50-day MA',
                'signal': 'neutral',
                'strength': 'weak',
                'description': "20-day MA is at the 50-day MA, indicating potential medium-term trend change"
            })
        
        # Analyze MACD
        macd_data = indicators['macd']
        
        # MACD Line vs Signal Line
        if macd_data['macd_line'] > macd_data['signal_line']:
            bullish_signals += 1
            signal_details.append({
                'indicator': 'MACD',
                'signal': 'bullish',
                'strength': 'moderate',
                'description': f"MACD line ({macd_data['macd_line']:.2f}) is above signal line ({macd_data['signal_line']:.2f}), indicating bullish momentum"
            })
        elif macd_data['macd_line'] < macd_data['signal_line']:
            bearish_signals += 1
            signal_details.append({
                'indicator': 'MACD',
                'signal': 'bearish',
                'strength': 'moderate',
                'description': f"MACD line ({macd_data['macd_line']:.2f}) is below signal line ({macd_data['signal_line']:.2f}), indicating bearish momentum"
            })
        else:
            neutral_signals += 1
            signal_details.append({
                'indicator': 'MACD',
                'signal': 'neutral',
                'strength': 'weak',
                'description': "MACD line is at the signal line, indicating potential momentum change"
            })
        
        # MACD Histogram
        if macd_data['histogram'] > 0 and macd_data['histogram'] > macd_data['histogram'] * 0.1:  # Increasing histogram
            bullish_signals += 1
            signal_details.append({
                'indicator': 'MACD Histogram',
                'signal': 'bullish',
                'strength': 'moderate',
                'description': f"MACD histogram is positive and increasing, indicating strengthening bullish momentum"
            })
        elif macd_data['histogram'] < 0 and macd_data['histogram'] < macd_data['histogram'] * 0.1:  # Decreasing histogram
            bearish_signals += 1
            signal_details.append({
                'indicator': 'MACD Histogram',
                'signal': 'bearish',
                'strength': 'moderate',
                'description': f"MACD histogram is negative and decreasing, indicating strengthening bearish momentum"
            })
        else:
            neutral_signals += 1
            signal_details.append({
                'indicator': 'MACD Histogram',
                'signal': 'neutral',
                'strength': 'weak',
                'description': "MACD histogram is flat or changing direction, indicating potential momentum shift"
            })
        
        # Analyze RSI
        rsi_data = indicators['rsi']
        
        # RSI Level
        if rsi_data['rsi_14'] > 70:
            bearish_signals += 1  # Overbought condition
            signal_details.append({
                'indicator': 'RSI',
                'signal': 'bearish',
                'strength': 'strong',
                'description': f"RSI at {rsi_data['rsi_14']:.2f} indicates overbought conditions, potential reversal or consolidation"
            })
        elif rsi_data['rsi_14'] < 30:
            bullish_signals += 1  # Oversold condition
            signal_details.append({
                'indicator': 'RSI',
                'signal': 'bullish',
                'strength': 'strong',
                'description': f"RSI at {rsi_data['rsi_14']:.2f} indicates oversold conditions, potential reversal or bounce"
            })
        elif rsi_data['rsi_14'] >= 60:
            bearish_signals += 0.5  # Near overbought condition
            signal_details.append({
                'indicator': 'RSI',
                'signal': 'bearish',
                'strength': 'moderate',
                'description': f"RSI at {rsi_data['rsi_14']:.2f} is nearing overbought territory, potential reversal or consolidation"
            })
        elif rsi_data['rsi_14'] <= 40:
            bullish_signals += 0.5  # Near oversold condition
            signal_details.append({
                'indicator': 'RSI',
                'signal': 'bullish',
                'strength': 'moderate',
                'description': f"RSI at {rsi_data['rsi_14']:.2f} is nearing oversold territory, potential reversal or bounce"
            })
        elif rsi_data['rsi_14'] > 50:
            bullish_signals += 0.5  # Moderately bullish
            signal_details.append({
                'indicator': 'RSI',
                'signal': 'bullish',
                'strength': 'weak',
                'description': f"RSI at {rsi_data['rsi_14']:.2f} is above centerline, indicating moderate bullish momentum"
            })
        elif rsi_data['rsi_14'] < 50:
            bearish_signals += 0.5  # Moderately bearish
            signal_details.append({
                'indicator': 'RSI',
                'signal': 'bearish',
                'strength': 'weak',
                'description': f"RSI at {rsi_data['rsi_14']:.2f} is below centerline, indicating moderate bearish momentum"
            })
        else:
            neutral_signals += 1
            signal_details.append({
                'indicator': 'RSI',
                'signal': 'neutral',
                'strength': 'weak',
                'description': f"RSI at {rsi_data['rsi_14']:.2f} is at centerline, indicating neutral momentum"
            })
        
        # Analyze Bollinger Bands
        bb_data = indicators['bollinger_bands']
        
        # Price position within Bollinger Bands
        if bb_data['percent_b'] > 1:
            bearish_signals += 1  # Price above upper band
            signal_details.append({
                'indicator': 'Bollinger Bands',
                'signal': 'bearish',
                'strength': 'moderate',
                'description': f"Price is above upper Bollinger Band, indicating overbought conditions or strong uptrend"
            })
        elif bb_data['percent_b'] < 0:
            bullish_signals += 1  # Price below lower band
            signal_details.append({
                'indicator': 'Bollinger Bands',
                'signal': 'bullish',
                'strength': 'moderate',
                'description': f"Price is below lower Bollinger Band, indicating oversold conditions or strong downtrend"
            })
        elif bb_data['percent_b'] > 0.8:
            bearish_signals += 0.5  # Price near upper band
            signal_details.append({
                'indicator': 'Bollinger Bands',
                'signal': 'bearish',
                'strength': 'weak',
                'description': f"Price is near upper Bollinger Band, approaching overbought conditions"
            })
        elif bb_data['percent_b'] < 0.2:
            bullish_signals += 0.5  # Price near lower band
            signal_details.append({
                'indicator': 'Bollinger Bands',
                'signal': 'bullish',
                'strength': 'weak',
                'description': f"Price is near lower Bollinger Band, approaching oversold conditions"
            })
        else:
            neutral_signals += 1
            signal_details.append({
                'indicator': 'Bollinger Bands',
                'signal': 'neutral',
                'strength': 'moderate',
                'description': f"Price is within Bollinger Bands, indicating normal volatility"
            })
        
        # Analyze Bollinger Band Width
        if bb_data['bandwidth'] < 0.1:  # Narrow bands
            neutral_signals += 1
            signal_details.append({
                'indicator': 'Bollinger Band Width',
                'signal': 'neutral',
                'strength': 'strong',
                'description': f"Narrow Bollinger Bands (width: {bb_data['bandwidth']:.2f}) indicate low volatility, potential breakout ahead"
            })
        elif bb_data['bandwidth'] > 0.3:  # Wide bands
            neutral_signals += 0.5
            signal_details.append({
                'indicator': 'Bollinger Band Width',
                'signal': 'neutral',
                'strength': 'moderate',
                'description': f"Wide Bollinger Bands (width: {bb_data['bandwidth']:.2f}) indicate high volatility"
            })
        
        # Analyze Volume
        volume_data = indicators['volume']
        
        # Volume trend
        if volume_data['volume_ratio'] > 1.5:
            if bullish_signals > bearish_signals:
                bullish_signals += 1
                signal_details.append({
                    'indicator': 'Volume',
                    'signal': 'bullish',
                    'strength': 'strong',
                    'description': f"Volume is {volume_data['volume_ratio']:.2f}x average, confirming price movement"
                })
            elif bearish_signals > bullish_signals:
                bearish_signals += 1
                signal_details.append({
                    'indicator': 'Volume',
                    'signal': 'bearish',
                    'strength': 'strong',
                    'description': f"Volume is {volume_data['volume_ratio']:.2f}x average, confirming price movement"
                })
            else:
                neutral_signals += 1
                signal_details.append({
                    'indicator': 'Volume',
                    'signal': 'neutral',
                    'strength': 'moderate',
                    'description': f"Volume is {volume_data['volume_ratio']:.2f}x average, but price direction is unclear"
                })
        elif volume_data['volume_ratio'] < 0.5:
            neutral_signals += 1
            signal_details.append({
                'indicator': 'Volume',
                'signal': 'neutral',
                'strength': 'weak',
                'description': f"Volume is low ({volume_data['volume_ratio']:.2f}x average), indicating lack of conviction"
            })
        
        # OBV trend
        if 'obv_trend' in volume_data:
            if volume_data['obv_trend'] == 'up':
                bullish_signals += 1
                signal_details.append({
                    'indicator': 'On-Balance Volume',
                    'signal': 'bullish',
                    'strength': 'moderate',
                    'description': "OBV is in uptrend, indicating accumulation and buying pressure"
                })
            elif volume_data['obv_trend'] == 'down':
                bearish_signals += 1
                signal_details.append({
                    'indicator': 'On-Balance Volume',
                    'signal': 'bearish',
                    'strength': 'moderate',
                    'description': "OBV is in downtrend, indicating distribution and selling pressure"
                })
        
        # Analyze ADX
        adx_data = indicators['adx']
        
        # Trend strength
        if adx_data['adx'] > 25:
            if adx_data['trend_direction'] == 'bullish':
                bullish_signals += 1
                signal_details.append({
                    'indicator': 'ADX',
                    'signal': 'bullish',
                    'strength': 'strong',
                    'description': f"ADX at {adx_data['adx']:.2f} indicates strong trend, +DI above -DI confirms bullish direction"
                })
            elif adx_data['trend_direction'] == 'bearish':
                bearish_signals += 1
                signal_details.append({
                    'indicator': 'ADX',
                    'signal': 'bearish',
                    'strength': 'strong',
                    'description': f"ADX at {adx_data['adx']:.2f} indicates strong trend, -DI above +DI confirms bearish direction"
                })
        else:
            neutral_signals += 1
            signal_details.append({
                'indicator': 'ADX',
                'signal': 'neutral',
                'strength': 'moderate',
                'description': f"ADX at {adx_data['adx']:.2f} indicates weak or no trend"
            })
        
        # Calculate overall consensus
        total_signals = bullish_signals + bearish_signals + neutral_signals
        bullish_percentage = (bullish_signals / total_signals) * 100 if total_signals > 0 else 0
        bearish_percentage = (bearish_signals / total_signals) * 100 if total_signals > 0 else 0
        neutral_percentage = (neutral_signals / total_signals) * 100 if total_signals > 0 else 0
        
        # Determine overall signal
        if bullish_percentage > 60:
            overall_signal = 'bullish'
            signal_strength = 'strong' if bullish_percentage > 75 else 'moderate'
        elif bearish_percentage > 60:
            overall_signal = 'bearish'
            signal_strength = 'strong' if bearish_percentage > 75 else 'moderate'
        else:
            overall_signal = 'neutral'
            signal_strength = 'moderate'
        
        # Compile consensus results
        consensus = {
            'overall_signal': overall_signal,
            'signal_strength': signal_strength,
            'bullish_percentage': bullish_percentage,
            'bearish_percentage': bearish_percentage,
            'neutral_percentage': neutral_percentage,
            'bullish_count': bullish_signals,
            'bearish_count': bearish_signals,
            'neutral_count': neutral_signals,
            'signal_details': signal_details
        }
        
        return consensus


class DataCollector:
    @staticmethod
    def collect_all_data(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Collect all technical analysis data.
        
        Args:
            data: DataFrame containing price and volume data
            
        Returns:
            Dict[str, Any]: Dictionary containing all technical analysis data
        """
        # Ensure index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except:
                # If conversion fails, create a dummy datetime index
                data.index = pd.date_range(start='2020-01-01', periods=len(data))
        
        # Calculate all indicators
        indicators = TechnicalIndicators.calculate_all_indicators(data)
        
        # Add metadata
        metadata = {
            'start': data.index[0].isoformat(),
            'end': data.index[-1].isoformat(),
            'period': len(data),
            'last_price': float(data['close'].iloc[-1]),
            'last_volume': float(data['volume'].iloc[-1])
        }
        
        # Return indicators directly without wrapping in technical_indicators
        return indicators


