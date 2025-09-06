from __future__ import annotations

from typing import Dict, Tuple
import logging
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def detect_market_regime(data: pd.DataFrame, indicators: Dict) -> Dict[str, str]:
    """
    Comprehensive market regime detection using multiple validated approaches.
    
    Returns:
        Dict with keys: "trend", "volatility", "confidence"
        
    Regime Classification:
        Trend: "bullish", "bearish", "sideways", "unknown"
        Volatility: "high", "medium", "low", "unknown"
        Confidence: "high", "medium", "low"
    """
    result = {
        "trend": "unknown",
        "volatility": "unknown",
        "confidence": "low"
    }
    
    try:
        if data is None or data.empty or len(data) < 50:
            logger.warning("Insufficient data for regime detection")
            return result
            
        # Validate required columns
        required_cols = ['close', 'high', 'low']
        if not all(col in data.columns for col in required_cols):
            logger.warning("Missing required price columns for regime detection")
            return result
        
        # Calculate comprehensive regime metrics
        trend_metrics = _calculate_trend_metrics(data)
        volatility_metrics = _calculate_volatility_metrics(data)
        momentum_metrics = _calculate_momentum_metrics(data)
        
        # Determine trend regime with confidence
        trend_result = _classify_trend_regime(trend_metrics, momentum_metrics)
        result["trend"] = trend_result["regime"]
        
        # Determine volatility regime with confidence
        volatility_result = _classify_volatility_regime(volatility_metrics)
        result["volatility"] = volatility_result["regime"]
        
        # Calculate overall confidence
        result["confidence"] = _calculate_overall_confidence(trend_result, volatility_result)
        
        logger.info(f"Regime detection completed: {result}")
        
    except Exception as e:
        logger.error(f"Regime detection failed: {e}")
        result = {"trend": "unknown", "volatility": "unknown", "confidence": "low"}
    
    return result


def _calculate_trend_metrics(data: pd.DataFrame) -> Dict[str, float]:
    """Calculate comprehensive trend strength and direction metrics."""
    try:
        close = data['close']
        high = data['high']
        low = data['low']
        
        # 1. ADX (Average Directional Index) - Trend Strength
        adx = _calculate_adx(high, low, close, period=14)
        
        # 2. Linear Regression R-squared - Trend Consistency
        x = np.arange(len(close))
        y = close.values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2
        
        # 3. Price Position Relative to Moving Averages
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()
        
        current_price = close.iloc[-1]
        price_vs_sma20 = (current_price / sma_20.iloc[-1] - 1) * 100 if not pd.isna(sma_20.iloc[-1]) else 0
        price_vs_sma50 = (current_price / sma_50.iloc[-1] - 1) * 100 if not pd.isna(sma_50.iloc[-1]) else 0
        price_vs_sma200 = (current_price / sma_200.iloc[-1] - 1) * 100 if not pd.isna(sma_200.iloc[-1]) else 0
        
        # 4. Moving Average Alignment
        ma_alignment = 0
        if not pd.isna(sma_20.iloc[-1]) and not pd.isna(sma_50.iloc[-1]) and not pd.isna(sma_200.iloc[-1]):
            if sma_20.iloc[-1] > sma_50.iloc[-1] > sma_200.iloc[-1]:
                ma_alignment = 1  # Bullish alignment
            elif sma_20.iloc[-1] < sma_50.iloc[-1] < sma_200.iloc[-1]:
                ma_alignment = -1  # Bearish alignment
            else:
                ma_alignment = 0  # Mixed alignment
        
        # 5. Price Channel Position
        highest_high = high.rolling(20).max()
        lowest_low = low.rolling(20).min()
        channel_position = (current_price - lowest_low.iloc[-1]) / (highest_high.iloc[-1] - lowest_low.iloc[-1]) if not pd.isna(highest_high.iloc[-1]) and not pd.isna(lowest_low.iloc[-1]) else 0.5
        
        return {
            "adx": adx,
            "r_squared": r_squared,
            "slope": slope,
            "price_vs_sma20": price_vs_sma20,
            "price_vs_sma50": price_vs_sma50,
            "price_vs_sma200": price_vs_sma200,
            "ma_alignment": ma_alignment,
            "channel_position": channel_position
        }
        
    except Exception as e:
        logger.error(f"Trend metrics calculation failed: {e}")
        return {}


def _calculate_volatility_metrics(data: pd.DataFrame) -> Dict[str, float]:
    """Calculate comprehensive volatility metrics."""
    try:
        close = data['close']
        high = data['high']
        low = data['low']
        
        # 1. ATR (Average True Range) - Absolute Volatility
        atr = _calculate_atr(high, low, close, period=14)
        atr_percent = (atr / close) * 100
        
        # 2. Rolling Standard Deviation - Return Volatility
        returns = close.pct_change().dropna()
        rolling_std_20 = returns.rolling(20).std() * np.sqrt(252)  # Annualized
        rolling_std_50 = returns.rolling(50).std() * np.sqrt(252)
        
        # 3. Historical Volatility Percentiles
        vol_20_percentile = rolling_std_20.quantile(0.2)
        vol_80_percentile = rolling_std_20.quantile(0.8)
        current_vol = rolling_std_20.iloc[-1] if not pd.isna(rolling_std_20.iloc[-1]) else 0
        
        # 4. Bollinger Bands Volatility
        bb_volatility = _calculate_bollinger_volatility(close, period=20)
        
        # 5. Parkinson Volatility (High-Low based)
        hl_volatility = np.sqrt(np.log(high / low) ** 2 / (4 * np.log(2)))
        hl_volatility_annualized = hl_volatility.rolling(20).mean() * np.sqrt(252)
        
        return {
            "atr": atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0,
            "atr_percent": atr_percent.iloc[-1] if not pd.isna(atr_percent.iloc[-1]) else 0,
            "rolling_std_20": current_vol,
            "vol_20_percentile": vol_20_percentile if not pd.isna(vol_20_percentile) else 0,
            "vol_80_percentile": vol_80_percentile if not pd.isna(vol_80_percentile) else 0,
            "bb_volatility": bb_volatility,
            "hl_volatility": hl_volatility_annualized.iloc[-1] if not pd.isna(hl_volatility_annualized.iloc[-1]) else 0
        }
        
    except Exception as e:
        logger.error(f"Volatility metrics calculation failed: {e}")
        return {}


def _calculate_momentum_metrics(data: pd.DataFrame) -> Dict[str, float]:
    """Calculate momentum and strength metrics."""
    try:
        close = data['close']
        
        # 1. RSI
        rsi = _calculate_rsi(close, period=14)
        
        # 2. MACD
        macd, signal, histogram = _calculate_macd(close)
        
        # 3. Price Momentum
        momentum_5 = (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0
        momentum_20 = (close.iloc[-1] / close.iloc[-20] - 1) * 100 if len(close) >= 20 else 0
        
        # 4. Rate of Change
        roc_10 = ((close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]) * 100 if len(close) >= 10 else 0
        
        return {
            "rsi": rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50,
            "macd": macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0,
            "macd_signal": signal.iloc[-1] if not pd.isna(signal.iloc[-1]) else 0,
            "macd_histogram": histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0,
            "momentum_5": momentum_5,
            "momentum_20": momentum_20,
            "roc_10": roc_10
        }
        
    except Exception as e:
        logger.error(f"Momentum metrics calculation failed: {e}")
        return {}


def _classify_trend_regime(trend_metrics: Dict[str, float], momentum_metrics: Dict[str, float]) -> Dict[str, str]:
    """Classify trend regime using multiple validated indicators."""
    try:
        # Extract key metrics
        adx_value = trend_metrics.get("adx", 0)
        if isinstance(adx_value, pd.Series):
            adx = float(adx_value.iloc[-1]) if not pd.isna(adx_value.iloc[-1]) else 0
        else:
            adx = float(adx_value)
            
        r_squared = trend_metrics.get("r_squared", 0)
        slope = trend_metrics.get("slope", 0)
        ma_alignment = trend_metrics.get("ma_alignment", 0)
        price_vs_sma200 = trend_metrics.get("price_vs_sma200", 0)
        
        rsi = momentum_metrics.get("rsi", 50)
        macd_histogram = momentum_metrics.get("macd_histogram", 0)
        
        # Calculate trend strength score (0-100)
        trend_strength = 0
        
        # ADX contribution (0-30 points)
        if adx >= 25:
            trend_strength += 30
        elif adx >= 20:
            trend_strength += 20
        elif adx >= 15:
            trend_strength += 10
        
        # R-squared contribution (0-25 points)
        if r_squared >= 0.7:
            trend_strength += 25
        elif r_squared >= 0.5:
            trend_strength += 15
        elif r_squared >= 0.3:
            trend_strength += 10
        
        # Moving average alignment contribution (0-20 points)
        if ma_alignment == 1:
            trend_strength += 20
        elif ma_alignment == -1:
            trend_strength += 20
        else:
            trend_strength += 5
        
        # Price vs SMA200 contribution (0-15 points)
        if abs(price_vs_sma200) > 10:
            trend_strength += 15
        elif abs(price_vs_sma200) > 5:
            trend_strength += 10
        
        # RSI momentum contribution (0-10 points)
        if isinstance(rsi, (int, float)) and isinstance(slope, (int, float)):
            if (rsi < 30 and slope > 0) or (rsi > 70 and slope < 0):
                trend_strength += 10
        
        # Determine trend regime
        if trend_strength >= 70:
            if isinstance(slope, (int, float)) and isinstance(ma_alignment, (int, float)):
                if slope > 0 and ma_alignment >= 0:
                    regime = "bullish"
                elif slope < 0 and ma_alignment <= 0:
                    regime = "bearish"
                else:
                    regime = "sideways"
            else:
                regime = "sideways"
        elif trend_strength >= 40:
            if isinstance(slope, (int, float)):
                if slope > 0:
                    regime = "bullish"
                elif slope < 0:
                    regime = "bearish"
                else:
                    regime = "sideways"
            else:
                regime = "sideways"
        else:
            regime = "sideways"
        
        return {
            "regime": regime,
            "strength": trend_strength,
            "confidence": "high" if trend_strength >= 60 else "medium" if trend_strength >= 30 else "low"
        }
        
    except Exception as e:
        logger.error(f"Trend classification failed: {e}")
        return {"regime": "unknown", "strength": 0, "confidence": "low"}


def _classify_volatility_regime(volatility_metrics: Dict[str, float]) -> Dict[str, str]:
    """Classify volatility regime using statistical percentiles."""
    try:
        # Extract key metrics and ensure they're scalar values
        atr_percent = volatility_metrics.get("atr_percent", 0)
        if isinstance(atr_percent, pd.Series):
            atr_percent = float(atr_percent.iloc[-1]) if not pd.isna(atr_percent.iloc[-1]) else 0
        else:
            atr_percent = float(atr_percent)
            
        rolling_std_20 = volatility_metrics.get("rolling_std_20", 0)
        if isinstance(rolling_std_20, pd.Series):
            rolling_std_20 = float(rolling_std_20.iloc[-1]) if not pd.isna(rolling_std_20.iloc[-1]) else 0
        else:
            rolling_std_20 = float(rolling_std_20)
            
        vol_20_percentile = volatility_metrics.get("vol_20_percentile", 0)
        if isinstance(vol_20_percentile, pd.Series):
            vol_20_percentile = float(vol_20_percentile.iloc[-1]) if not pd.isna(vol_20_percentile.iloc[-1]) else 0
        else:
            vol_20_percentile = float(vol_20_percentile)
            
        vol_80_percentile = volatility_metrics.get("vol_80_percentile", 0)
        if isinstance(vol_80_percentile, pd.Series):
            vol_80_percentile = float(vol_80_percentile.iloc[-1]) if not pd.isna(vol_80_percentile.iloc[-1]) else 0
        else:
            vol_80_percentile = float(vol_80_percentile)
            
        bb_volatility = volatility_metrics.get("bb_volatility", 0)
        if isinstance(bb_volatility, pd.Series):
            bb_volatility = float(bb_volatility.iloc[-1]) if not pd.isna(bb_volatility.iloc[-1]) else 0
        else:
            bb_volatility = float(bb_volatility)
        
        # Calculate volatility score (0-100)
        volatility_score = 0
        
        # ATR% contribution (0-30 points)
        if atr_percent >= 4.0:
            volatility_score += 30
        elif atr_percent >= 2.5:
            volatility_score += 20
        elif atr_percent >= 1.5:
            volatility_score += 10
        
        # Rolling std percentile contribution (0-40 points)
        if vol_80_percentile > 0 and rolling_std_20 > 0:
            if rolling_std_20 >= vol_80_percentile:
                volatility_score += 40
            elif rolling_std_20 >= vol_20_percentile:
                volatility_score += 20
            else:
                volatility_score += 5
        
        # Bollinger Bands contribution (0-30 points)
        if bb_volatility > 0.2:
            volatility_score += 30
        elif bb_volatility > 0.1:
            volatility_score += 20
        elif bb_volatility > 0.05:
            volatility_score += 10
        
        # Determine volatility regime
        if volatility_score >= 70:
            regime = "high"
        elif volatility_score >= 30:
            regime = "medium"
        else:
            regime = "low"
        
        return {
            "regime": regime,
            "score": volatility_score,
            "confidence": "high" if volatility_score >= 60 else "medium" if volatility_score >= 30 else "low"
        }
        
    except Exception as e:
        logger.error(f"Volatility classification failed: {e}")
        return {"regime": "unknown", "score": 0, "confidence": "low"}


def _calculate_overall_confidence(trend_result: Dict[str, str], volatility_result: Dict[str, str]) -> str:
    """Calculate overall confidence in regime classification."""
    try:
        trend_conf = trend_result.get("confidence", "low")
        vol_conf = volatility_result.get("confidence", "low")
        
        # Convert confidence levels to scores
        conf_scores = {"high": 3, "medium": 2, "low": 1}
        trend_score = conf_scores.get(trend_conf, 1)
        vol_score = conf_scores.get(vol_conf, 1)
        
        avg_score = (trend_score + vol_score) / 2
        
        if avg_score >= 2.5:
            return "high"
        elif avg_score >= 1.5:
            return "medium"
        else:
            return "low"
            
    except Exception:
        return "low"


# Technical Indicator Calculation Functions
def _calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate ADX (Average Directional Index)."""
    try:
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR
        atr = tr.rolling(period).mean()
        
        # Directional Movement
        plus_dm = (high.diff()).clip(lower=0.0)
        minus_dm = (-low.diff()).clip(lower=0.0)
        
        # +DI and -DI
        plus_di = 100.0 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100.0 * (minus_dm.rolling(period).mean() / atr)
        
        # DX and ADX
        dx = 100.0 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(period).mean()
        
        return adx
        
    except Exception as e:
        logger.error(f"ADX calculation failed: {e}")
        return pd.Series(dtype=float)


def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate ATR (Average True Range)."""
    try:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
        
    except Exception as e:
        logger.error(f"ATR calculation failed: {e}")
        return pd.Series(dtype=float)


def _calculate_bollinger_volatility(close: pd.Series, period: int = 20) -> float:
    """Calculate Bollinger Bands volatility."""
    try:
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        bandwidth = (upper - lower) / sma
        return bandwidth.iloc[-1] if not pd.isna(bandwidth.iloc[-1]) else 0
        
    except Exception as e:
        logger.error(f"Bollinger volatility calculation failed: {e}")
        return 0


def _calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI (Relative Strength Index)."""
    try:
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    except Exception as e:
        logger.error(f"RSI calculation failed: {e}")
        return pd.Series(dtype=float)


def _calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    try:
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
        
    except Exception as e:
        logger.error(f"MACD calculation failed: {e}")
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)


