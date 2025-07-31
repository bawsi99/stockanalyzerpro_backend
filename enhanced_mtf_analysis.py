"""
Enhanced Multi-Timeframe Analysis System

This module provides comprehensive multi-timeframe analysis by:
1. Fetching data for all available Zerodha intervals
2. Calculating appropriate indicators for each timeframe
3. Performing cross-timeframe signal validation
4. Identifying supporting and conflicting signals
5. Providing confidence-weighted analysis
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Local imports
from zerodha_client import ZerodhaDataClient
from technical_indicators import TechnicalIndicators
from patterns.recognition import PatternRecognition

logger = logging.getLogger(__name__)

@dataclass
class TimeframeConfig:
    """Configuration for each timeframe analysis."""
    interval: str
    period_days: int
    min_data_points: int
    indicators: List[str]
    weight: float  # Weight for signal confidence
    description: str

@dataclass
class TimeframeAnalysis:
    """Results of analysis for a single timeframe."""
    timeframe: str
    data_points: int
    indicators: Dict[str, Any]
    signals: Dict[str, Any]
    confidence: float
    trend: str
    support_levels: List[float]
    resistance_levels: List[float]
    key_levels: List[float]
    volume_analysis: Dict[str, Any]
    pattern_analysis: Dict[str, Any]
    risk_metrics: Dict[str, Any]

@dataclass
class CrossTimeframeValidation:
    """Cross-timeframe validation results."""
    supporting_timeframes: List[str]
    conflicting_timeframes: List[str]
    neutral_timeframes: List[str]
    signal_strength: float
    consensus_trend: str
    divergence_detected: bool
    divergence_type: Optional[str]
    key_conflicts: List[str]
    confidence_score: float 

class EnhancedMultiTimeframeAnalyzer:
    """
    Enhanced multi-timeframe analysis system with comprehensive signal validation.
    """
    
    def __init__(self):
        self.zerodha_client = ZerodhaDataClient()
        self.technical_indicators = TechnicalIndicators()
        self.pattern_recognition = PatternRecognition()
        
        # Define timeframe configurations with appropriate indicators and weights
        self.timeframe_configs = {
            '1min': TimeframeConfig(
                interval='minute',
                period_days=30,  # 30 days for 1min data
                min_data_points=100,
                indicators=['sma_20', 'ema_12', 'rsi_14', 'macd', 'bollinger_bands', 'volume_ratio', 'atr'],
                weight=0.05,  # Lowest weight for noise
                description='Intraday scalping timeframe'
            ),
            '5min': TimeframeConfig(
                interval='5minute',
                period_days=60,  # 60 days for 5min data
                min_data_points=80,
                indicators=['sma_20', 'ema_12', 'rsi_14', 'macd', 'bollinger_bands', 'volume_ratio', 'atr', 'stochastic'],
                weight=0.10,
                description='Short-term intraday trading'
            ),
            '15min': TimeframeConfig(
                interval='15minute',
                period_days=90,  # 90 days for 15min data
                min_data_points=60,
                indicators=['sma_20', 'sma_50', 'ema_12', 'rsi_14', 'macd', 'bollinger_bands', 'volume_ratio', 'atr', 'stochastic', 'adx'],
                weight=0.15,
                description='Medium-term intraday trading'
            ),
            '30min': TimeframeConfig(
                interval='30minute',
                period_days=120,  # 120 days for 30min data
                min_data_points=50,
                indicators=['sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi_14', 'macd', 'bollinger_bands', 'volume_ratio', 'atr', 'stochastic', 'adx', 'obv'],
                weight=0.20,
                description='Swing trading timeframe'
            ),
            '1hour': TimeframeConfig(
                interval='60minute',
                period_days=180,  # 180 days for 1hour data
                min_data_points=40,
                indicators=['sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi_14', 'macd', 'bollinger_bands', 'volume_ratio', 'atr', 'stochastic', 'adx', 'obv', 'ichimoku'],
                weight=0.25,
                description='Position trading timeframe'
            ),
            '1day': TimeframeConfig(
                interval='day',
                period_days=365,  # 1 year for daily data
                min_data_points=30,
                indicators=['sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26', 'rsi_14', 'macd', 'bollinger_bands', 'volume_ratio', 'atr', 'stochastic', 'adx', 'obv', 'ichimoku', 'fibonacci'],
                weight=0.25,  # Highest weight for trend confirmation
                description='Long-term trend analysis'
            )
        }
        
        # Signal mapping for consistency
        self.signal_mapping = {
            'bullish': 1,
            'bearish': -1,
            'neutral': 0,
            'strong_bullish': 2,
            'strong_bearish': -2
        }
    
    async def authenticate(self) -> bool:
        """Authenticate with Zerodha API."""
        return self.zerodha_client.authenticate() 
    
    async def fetch_timeframe_data(self, symbol: str, exchange: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch data for a specific timeframe."""
        try:
            config = self.timeframe_configs[timeframe]
            
            # Fetch historical data
            data = await self.zerodha_client.get_historical_data_async(
                symbol=symbol,
                exchange=exchange,
                interval=config.interval,
                period=config.period_days
            )
            
            if data is None or len(data) < config.min_data_points:
                logger.warning(f"Insufficient data for {timeframe}: {len(data) if data is not None else 0} points")
                return None
            
            # Ensure proper datetime index
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {timeframe}: {e}")
            return None
    
    def calculate_timeframe_indicators(self, data: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Calculate indicators for a specific timeframe."""
        config = self.timeframe_configs[timeframe]
        indicators = {}
        
        try:
            # Calculate basic indicators
            if 'sma_20' in config.indicators:
                sma_20 = self.technical_indicators.calculate_sma(data, 'close', 20)
                indicators['sma_20'] = float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else None
            
            if 'sma_50' in config.indicators:
                sma_50 = self.technical_indicators.calculate_sma(data, 'close', 50)
                indicators['sma_50'] = float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else None
            
            if 'sma_200' in config.indicators:
                sma_200 = self.technical_indicators.calculate_sma(data, 'close', 200)
                indicators['sma_200'] = float(sma_200.iloc[-1]) if not pd.isna(sma_200.iloc[-1]) else None
            
            if 'ema_12' in config.indicators:
                ema_12 = self.technical_indicators.calculate_ema(data, 'close', 12)
                indicators['ema_12'] = float(ema_12.iloc[-1]) if not pd.isna(ema_12.iloc[-1]) else None
            
            if 'ema_26' in config.indicators:
                ema_26 = self.technical_indicators.calculate_ema(data, 'close', 26)
                indicators['ema_26'] = float(ema_26.iloc[-1]) if not pd.isna(ema_26.iloc[-1]) else None
            
            # RSI
            if 'rsi_14' in config.indicators:
                rsi = self.technical_indicators.calculate_rsi(data, 'close', 14)
                rsi_value = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
                indicators['rsi_14'] = rsi_value
                indicators['rsi_trend'] = 'up' if rsi.iloc[-1] > rsi.iloc[-2] else 'down' if len(rsi) > 1 else 'neutral'
                indicators['rsi_status'] = self._get_rsi_status(rsi_value)
            
            # MACD
            if 'macd' in config.indicators:
                macd_line, signal_line, histogram = self.technical_indicators.calculate_macd(data, 'close')
                indicators['macd_line'] = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None
                indicators['macd_signal_line'] = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else None
                indicators['macd_histogram'] = float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else None
                indicators['macd_signal'] = self._get_macd_signal(macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1])
            
            # Bollinger Bands
            if 'bollinger_bands' in config.indicators:
                upper, middle, lower = self.technical_indicators.calculate_bollinger_bands(data, 'close')
                current_price = data['close'].iloc[-1]
                indicators['bb_upper'] = float(upper.iloc[-1]) if not pd.isna(upper.iloc[-1]) else None
                indicators['bb_middle'] = float(middle.iloc[-1]) if not pd.isna(middle.iloc[-1]) else None
                indicators['bb_lower'] = float(lower.iloc[-1]) if not pd.isna(lower.iloc[-1]) else None
                indicators['bb_position'] = self._get_bb_position(current_price, upper.iloc[-1], lower.iloc[-1])
            
            # Volume analysis
            if 'volume_ratio' in config.indicators:
                volume_ma = data['volume'].rolling(window=20).mean()
                current_volume = data['volume'].iloc[-1]
                volume_ratio = current_volume / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1.0
                indicators['volume_ratio'] = float(volume_ratio)
                indicators['volume_status'] = 'high' if volume_ratio > 1.5 else 'low' if volume_ratio < 0.5 else 'normal'
            
            # ATR
            if 'atr' in config.indicators:
                atr = self.technical_indicators.calculate_atr(data)
                indicators['atr'] = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None
            
            # Stochastic
            if 'stochastic' in config.indicators:
                stoch_k, stoch_d = self.technical_indicators.calculate_stochastic_oscillator(data)
                indicators['stoch_k'] = float(stoch_k.iloc[-1]) if not pd.isna(stoch_k.iloc[-1]) else None
                indicators['stoch_d'] = float(stoch_d.iloc[-1]) if not pd.isna(stoch_d.iloc[-1]) else None
                indicators['stoch_signal'] = self._get_stochastic_signal(stoch_k.iloc[-1], stoch_d.iloc[-1])
            
            # ADX
            if 'adx' in config.indicators:
                adx, plus_di, minus_di = self.technical_indicators.calculate_adx(data)
                indicators['adx'] = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else None
                indicators['plus_di'] = float(plus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else None
                indicators['minus_di'] = float(minus_di.iloc[-1]) if not pd.isna(minus_di.iloc[-1]) else None
                indicators['adx_trend'] = 'bullish' if plus_di.iloc[-1] > minus_di.iloc[-1] else 'bearish'
                indicators['trend_strength'] = 'strong' if adx.iloc[-1] > 25 else 'weak'
            
            # OBV
            if 'obv' in config.indicators:
                obv = self.technical_indicators.calculate_obv(data)
                indicators['obv'] = float(obv.iloc[-1]) if not pd.isna(obv.iloc[-1]) else None
                indicators['obv_trend'] = 'up' if obv.iloc[-1] > obv.iloc[-2] else 'down' if len(obv) > 1 else 'neutral'
            
            # Ichimoku (for higher timeframes)
            if 'ichimoku' in config.indicators and timeframe in ['1hour', '1day']:
                try:
                    ichimoku = self.technical_indicators.calculate_ichimoku(data)
                    current_price = data['close'].iloc[-1]
                    
                    # Check if ichimoku calculation was successful
                    if isinstance(ichimoku, dict) and 'tenkan' in ichimoku:
                        indicators['ichimoku'] = {
                            'tenkan': float(ichimoku['tenkan'].iloc[-1]) if not pd.isna(ichimoku['tenkan'].iloc[-1]) else None,
                            'kijun': float(ichimoku['kijun'].iloc[-1]) if not pd.isna(ichimoku['kijun'].iloc[-1]) else None,
                            'senkou_a': float(ichimoku['senkou_a'].iloc[-1]) if not pd.isna(ichimoku['senkou_a'].iloc[-1]) else None,
                            'senkou_b': float(ichimoku['senkou_b'].iloc[-1]) if not pd.isna(ichimoku['senkou_b'].iloc[-1]) else None,
                            'chikou': float(ichimoku['chikou'].iloc[-1]) if not pd.isna(ichimoku['chikou'].iloc[-1]) else None
                        }
                        indicators['ichimoku_signal'] = self._get_ichimoku_signal(current_price, ichimoku)
                    else:
                        indicators['ichimoku'] = None
                        indicators['ichimoku_signal'] = 'neutral'
                except Exception as e:
                    logger.warning(f"Ichimoku calculation failed for {timeframe}: {e}")
                    indicators['ichimoku'] = None
                    indicators['ichimoku_signal'] = 'neutral'
            
            # Fibonacci (for daily timeframe)
            if 'fibonacci' in config.indicators and timeframe == '1day':
                try:
                    fib_levels = self.technical_indicators.calculate_fibonacci_retracement(data)
                    indicators['fibonacci'] = fib_levels
                except Exception as e:
                    logger.warning(f"Fibonacci calculation failed for {timeframe}: {e}")
                    indicators['fibonacci'] = None
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {timeframe}: {e}")
        
        return indicators 
    
    def generate_timeframe_signals(self, indicators: Dict[str, Any], timeframe: str) -> Dict[str, Any]:
        """Generate trading signals for a timeframe based on indicators."""
        signals = {}
        
        try:
            # Trend signals
            trend_signals = []
            
            # Moving average signals
            if indicators.get('sma_20') is not None and indicators.get('sma_50') is not None:
                if float(indicators['sma_20']) > float(indicators['sma_50']):
                    trend_signals.append(('sma_cross', 'bullish', 0.3))
                else:
                    trend_signals.append(('sma_cross', 'bearish', 0.3))
            
            if indicators.get('sma_50') is not None and indicators.get('sma_200') is not None:
                if float(indicators['sma_50']) > float(indicators['sma_200']):
                    trend_signals.append(('sma_200_cross', 'bullish', 0.4))
                else:
                    trend_signals.append(('sma_200_cross', 'bearish', 0.4))
            
            # RSI signals
            if indicators.get('rsi_14') is not None:
                rsi = float(indicators['rsi_14'])
                if rsi < 30:
                    trend_signals.append(('rsi_oversold', 'bullish', 0.2))
                elif rsi > 70:
                    trend_signals.append(('rsi_overbought', 'bearish', 0.2))
                elif rsi > 50 and indicators.get('rsi_trend') == 'up':
                    trend_signals.append(('rsi_momentum', 'bullish', 0.15))
                elif rsi < 50 and indicators.get('rsi_trend') == 'down':
                    trend_signals.append(('rsi_momentum', 'bearish', 0.15))
            
            # MACD signals
            if indicators.get('macd_line') is not None and indicators.get('macd_signal_line') is not None:
                macd_line = float(indicators['macd_line'])
                macd_signal_line = float(indicators['macd_signal_line'])
                if macd_line > macd_signal_line:
                    trend_signals.append(('macd_cross', 'bullish', 0.25))
                else:
                    trend_signals.append(('macd_cross', 'bearish', 0.25))
            
            # Bollinger Bands signals
            if indicators.get('bb_position'):
                bb_pos = indicators['bb_position']
                if bb_pos == 'oversold':
                    trend_signals.append(('bb_oversold', 'bullish', 0.2))
                elif bb_pos == 'overbought':
                    trend_signals.append(('bb_overbought', 'bearish', 0.2))
            
            # ADX trend strength
            if indicators.get('trend_strength') == 'strong':
                if indicators.get('adx_trend') == 'bullish':
                    trend_signals.append(('adx_strong_bullish', 'bullish', 0.3))
                else:
                    trend_signals.append(('adx_strong_bearish', 'bearish', 0.3))
            
            # Volume confirmation
            if indicators.get('volume_status') == 'high':
                bullish_count = len([s for s in trend_signals if s[1] == 'bullish'])
                bearish_count = len([s for s in trend_signals if s[1] == 'bearish'])
                if bullish_count > bearish_count:
                    trend_signals.append(('volume_confirmation', 'bullish', 0.15))
                else:
                    trend_signals.append(('volume_confirmation', 'bearish', 0.15))
            
            # Calculate overall trend
            bullish_weight = sum([s[2] for s in trend_signals if s[1] == 'bullish'])
            bearish_weight = sum([s[2] for s in trend_signals if s[1] == 'bearish'])
            
            if bullish_weight > bearish_weight + 0.2:
                overall_trend = 'bullish'
                confidence = min(bullish_weight, 1.0)
            elif bearish_weight > bullish_weight + 0.2:
                overall_trend = 'bearish'
                confidence = min(bearish_weight, 1.0)
            else:
                overall_trend = 'neutral'
                confidence = 0.5
            
            signals = {
                'trend_signals': trend_signals,
                'overall_trend': overall_trend,
                'confidence': confidence,
                'bullish_weight': bullish_weight,
                'bearish_weight': bearish_weight,
                'signal_count': len(trend_signals)
            }
            
        except Exception as e:
            logger.error(f"Error generating signals for {timeframe}: {e}")
            signals = {
                'overall_trend': 'neutral',
                'confidence': 0.0,
                'error': str(e)
            }
        
        return signals
    
    def calculate_support_resistance(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Calculate support and resistance levels."""
        try:
            support, resistance = self.technical_indicators.detect_support_resistance(data)
            return support, resistance
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return [], []
    
    def detect_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect chart patterns."""
        try:
            patterns = {}
            
            # Candlestick patterns
            candlestick_patterns = self.pattern_recognition.detect_candlestick_patterns(data)
            patterns['candlestick'] = candlestick_patterns[-3:] if candlestick_patterns else []
            
            # Double tops/bottoms
            double_tops = self.pattern_recognition.detect_double_top(data['close'])
            double_bottoms = self.pattern_recognition.detect_double_bottom(data['close'])
            patterns['double_tops'] = double_tops
            patterns['double_bottoms'] = double_bottoms
            
            # Triangles
            triangles = self.pattern_recognition.detect_triangle(data['close'])
            patterns['triangles'] = triangles
            
            # Head and shoulders
            head_shoulders = self.pattern_recognition.detect_head_and_shoulders(data['close'])
            patterns['head_shoulders'] = head_shoulders
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return {}
    
    async def analyze_timeframe(self, symbol: str, exchange: str, timeframe: str) -> Optional[TimeframeAnalysis]:
        """Analyze a single timeframe comprehensively."""
        try:
            # Fetch data
            data = await self.fetch_timeframe_data(symbol, exchange, timeframe)
            if data is None:
                return None
            
            # Calculate indicators
            indicators = self.calculate_timeframe_indicators(data, timeframe)
            
            # Generate signals
            signals = self.generate_timeframe_signals(indicators, timeframe)
            
            # Calculate support/resistance
            support_levels, resistance_levels = self.calculate_support_resistance(data)
            
            # Detect patterns
            patterns = self.detect_patterns(data)
            
            # Volume analysis
            volume_analysis = {
                'current_volume': float(data['volume'].iloc[-1]),
                'volume_ma_20': float(data['volume'].rolling(20).mean().iloc[-1]),
                'volume_ratio': indicators.get('volume_ratio', 1.0),
                'volume_trend': indicators.get('obv_trend', 'neutral')
            }
            
            # Risk metrics
            current_price = data['close'].iloc[-1]
            risk_metrics = {
                'current_price': current_price,
                'atr': indicators.get('atr'),
                'volatility': float(data['close'].pct_change().std() * np.sqrt(252)) if len(data) > 1 else None,
                'max_drawdown': self._calculate_max_drawdown(data['close']),
                'risk_reward_ratio': self._calculate_risk_reward_ratio(current_price, support_levels, resistance_levels)
            }
            
            return TimeframeAnalysis(
                timeframe=timeframe,
                data_points=len(data),
                indicators=indicators,
                signals=signals,
                confidence=signals.get('confidence', 0.0),
                trend=signals.get('overall_trend', 'neutral'),
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                key_levels=support_levels + resistance_levels,
                volume_analysis=volume_analysis,
                pattern_analysis=patterns,
                risk_metrics=risk_metrics
            )
            
        except Exception as e:
            logger.error(f"Error analyzing timeframe {timeframe}: {e}")
            return None 
    
    async def analyze_all_timeframes(self, symbol: str, exchange: str = "NSE") -> Dict[str, TimeframeAnalysis]:
        """Analyze all timeframes concurrently."""
        # Authenticate first
        if not await self.authenticate():
            logger.error("Failed to authenticate with Zerodha")
            return {}
        
        # Analyze all timeframes concurrently
        tasks = []
        for timeframe in self.timeframe_configs.keys():
            task = self.analyze_timeframe(symbol, exchange, timeframe)
            tasks.append((timeframe, task))
        
        # Execute all tasks concurrently
        results = {}
        for timeframe, task in tasks:
            try:
                result = await task
                if result:
                    results[timeframe] = result
            except Exception as e:
                logger.error(f"Error in timeframe {timeframe}: {e}")
        
        return results
    
    def validate_cross_timeframe(self, timeframe_analyses: Dict[str, TimeframeAnalysis]) -> CrossTimeframeValidation:
        """Validate signals across timeframes and identify conflicts."""
        if not timeframe_analyses:
            return CrossTimeframeValidation(
                supporting_timeframes=[],
                conflicting_timeframes=[],
                neutral_timeframes=[],
                signal_strength=0.0,
                consensus_trend='neutral',
                divergence_detected=False,
                divergence_type=None,
                key_conflicts=[],
                confidence_score=0.0
            )
        
        # Collect trends and weights
        trends = {}
        total_weight = 0
        
        for timeframe, analysis in timeframe_analyses.items():
            config = self.timeframe_configs[timeframe]
            # Use dynamic weighting based on signal quality instead of fixed weights
            signal_quality = self.calculate_signal_quality(analysis)
            dynamic_weight = config.weight * signal_quality * analysis.confidence
            
            trends[timeframe] = {
                'trend': analysis.trend,
                'weight': dynamic_weight,
                'confidence': analysis.confidence,
                'signal_quality': signal_quality,
                'base_weight': config.weight
            }
            total_weight += dynamic_weight
        
        # Calculate weighted consensus
        bullish_weight = sum([t['weight'] for t in trends.values() if t['trend'] == 'bullish'])
        bearish_weight = sum([t['weight'] for t in trends.values() if t['trend'] == 'bearish'])
        neutral_weight = sum([t['weight'] for t in trends.values() if t['trend'] == 'neutral'])
        
        # Determine consensus trend
        if bullish_weight > bearish_weight and bullish_weight > neutral_weight:
            consensus_trend = 'bullish'
            signal_strength = bullish_weight / total_weight if total_weight > 0 else 0
        elif bearish_weight > bullish_weight and bearish_weight > neutral_weight:
            consensus_trend = 'bearish'
            signal_strength = bearish_weight / total_weight if total_weight > 0 else 0
        else:
            consensus_trend = 'neutral'
            signal_strength = 0.5
        
        # Identify supporting and conflicting timeframes
        supporting_timeframes = [tf for tf, t in trends.items() if t['trend'] == consensus_trend]
        conflicting_timeframes = [tf for tf, t in trends.items() if t['trend'] != consensus_trend and t['trend'] != 'neutral']
        neutral_timeframes = [tf for tf, t in trends.items() if t['trend'] == 'neutral']
        
        # Detect divergences
        divergence_detected = False
        divergence_type = None
        
        # Check for trend divergence between timeframes
        if len(supporting_timeframes) > 0 and len(conflicting_timeframes) > 0:
            # Check if higher timeframes conflict with lower timeframes
            timeframe_order = ['1day', '1hour', '30min', '15min', '5min', '1min']
            higher_timeframes = [tf for tf in timeframe_order if tf in trends]
            lower_timeframes = [tf for tf in reversed(timeframe_order) if tf in trends]
            
            higher_trends = [trends[tf]['trend'] for tf in higher_timeframes[:2] if tf in trends]
            lower_trends = [trends[tf]['trend'] for tf in lower_timeframes[:2] if tf in trends]
            
            if higher_trends and lower_trends:
                if all(t == 'bullish' for t in higher_trends) and all(t == 'bearish' for t in lower_trends):
                    divergence_detected = True
                    divergence_type = 'bearish_divergence'
                elif all(t == 'bearish' for t in higher_trends) and all(t == 'bullish' for t in lower_trends):
                    divergence_detected = True
                    divergence_type = 'bullish_divergence'
        
        # Identify key conflicts
        key_conflicts = []
        if conflicting_timeframes:
            for tf in conflicting_timeframes:
                config = self.timeframe_configs[tf]
                if trends[tf]['weight'] > 0.1:  # Only significant conflicts
                    key_conflicts.append(f"{tf} ({config.description}): {trends[tf]['trend']}")
        
        # Calculate overall confidence score
        confidence_score = signal_strength * (len(supporting_timeframes) / len(timeframe_analyses))
        
        return CrossTimeframeValidation(
            supporting_timeframes=supporting_timeframes,
            conflicting_timeframes=conflicting_timeframes,
            neutral_timeframes=neutral_timeframes,
            signal_strength=signal_strength,
            consensus_trend=consensus_trend,
            divergence_detected=divergence_detected,
            divergence_type=divergence_type,
            key_conflicts=key_conflicts,
            confidence_score=confidence_score
        )
    
    async def comprehensive_mtf_analysis(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """Perform comprehensive multi-timeframe analysis."""
        try:
            # Analyze all timeframes
            timeframe_analyses = await self.analyze_all_timeframes(symbol, exchange)
            
            if not timeframe_analyses:
                return {
                    'error': 'No timeframe data available',
                    'success': False
                }
            
            # Validate cross-timeframe
            validation = self.validate_cross_timeframe(timeframe_analyses)
            
            # Compile results
            results = {
                'success': True,
                'symbol': symbol,
                'exchange': exchange,
                'analysis_timestamp': datetime.now().isoformat(),
                'timeframe_analyses': {
                    tf: {
                        'trend': analysis.trend,
                        'confidence': analysis.confidence,
                        'data_points': analysis.data_points,
                        'key_indicators': {
                            'rsi': analysis.indicators.get('rsi_14'),
                            'macd_signal': analysis.signals.get('overall_trend'),
                            'volume_status': analysis.volume_analysis.get('volume_status'),
                            'support_levels': analysis.support_levels[:3],  # Top 3
                            'resistance_levels': analysis.resistance_levels[:3]  # Top 3
                        },
                        'patterns': list(analysis.pattern_analysis.keys()),
                        'risk_metrics': {
                            'current_price': analysis.risk_metrics.get('current_price'),
                            'volatility': analysis.risk_metrics.get('volatility'),
                            'max_drawdown': analysis.risk_metrics.get('max_drawdown')
                        }
                    }
                    for tf, analysis in timeframe_analyses.items()
                },
                'cross_timeframe_validation': {
                    'consensus_trend': validation.consensus_trend,
                    'signal_strength': validation.signal_strength,
                    'confidence_score': validation.confidence_score,
                    'supporting_timeframes': validation.supporting_timeframes,
                    'conflicting_timeframes': validation.conflicting_timeframes,
                    'neutral_timeframes': validation.neutral_timeframes,
                    'divergence_detected': validation.divergence_detected,
                    'divergence_type': validation.divergence_type,
                    'key_conflicts': validation.key_conflicts
                },
                'summary': {
                    'overall_signal': validation.consensus_trend,
                    'confidence': validation.confidence_score,
                    'timeframes_analyzed': len(timeframe_analyses),
                    'signal_alignment': 'aligned' if len(validation.conflicting_timeframes) == 0 else 'conflicting',
                    'risk_level': self._determine_risk_level(validation.confidence_score, validation.divergence_detected),
                    'recommendation': self._generate_recommendation(validation.consensus_trend, validation.confidence_score, validation.divergence_detected)
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive MTF analysis: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def calculate_signal_quality(self, analysis) -> float:
        """Calculate signal quality score for dynamic weighting."""
        try:
            quality_score = 0.0
            
            # Handle both TimeframeAnalysis objects and dictionaries
            if hasattr(analysis, 'indicators'):
                # TimeframeAnalysis object
                indicators = analysis.indicators
                signals = analysis.signals
                timeframe = analysis.timeframe
                trend = analysis.trend
                confidence = analysis.confidence
                support_levels = analysis.support_levels
                resistance_levels = analysis.resistance_levels
            else:
                # Dictionary format
                indicators = analysis.get('indicators', {})
                signals = analysis.get('signals', {})
                timeframe = analysis.get('timeframe', 'unknown')
                trend = analysis.get('trend', 'neutral')
                confidence = analysis.get('confidence', 0)
                support_levels = analysis.get('support_levels', [])
                resistance_levels = analysis.get('resistance_levels', [])
            
            # 1. Signal strength (how many indicators agree)
            if 'signal_count' in signals:
                signal_count = signals['signal_count']
                max_possible_signals = len(self.timeframe_configs[timeframe].indicators)
                signal_strength = min(signal_count / max_possible_signals, 1.0)
                quality_score += signal_strength * 0.3
            
            # 2. Volume confirmation
            if 'volume_ratio' in indicators:
                volume_ratio = indicators.get('volume_ratio', 1.0)
                if volume_ratio > 1.5:  # High volume
                    quality_score += 0.2
                elif volume_ratio > 1.0:  # Normal volume
                    quality_score += 0.1
                elif volume_ratio < 0.5:  # Low volume - reduce quality
                    quality_score -= 0.1
            
            # 3. Trend consistency (RSI trend vs overall trend)
            if 'rsi_trend' in indicators and trend != 'neutral':
                rsi_trend = indicators['rsi_trend']
                if (rsi_trend == 'up' and trend == 'bullish') or (rsi_trend == 'down' and trend == 'bearish'):
                    quality_score += 0.15  # Trend alignment
                else:
                    quality_score -= 0.1   # Trend divergence
            
            # 4. MACD signal strength
            if 'macd_histogram' in indicators:
                macd_hist = indicators.get('macd_histogram', 0)
                if abs(macd_hist) > 0.5:  # Strong MACD signal
                    quality_score += 0.1
                elif abs(macd_hist) < 0.1:  # Weak MACD signal
                    quality_score -= 0.05
            
            # 5. Support/Resistance proximity
            current_price = indicators.get('current_price', 0)
            if current_price and support_levels and resistance_levels:
                nearest_support = min(support_levels, key=lambda x: abs(x - current_price))
                nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
                
                support_distance = abs(current_price - nearest_support) / current_price
                resistance_distance = abs(nearest_resistance - current_price) / current_price
                
                # Closer to levels = higher quality signal
                if min(support_distance, resistance_distance) < 0.02:  # Within 2%
                    quality_score += 0.15
                elif min(support_distance, resistance_distance) < 0.05:  # Within 5%
                    quality_score += 0.1
            
            # 6. Confidence boost
            quality_score += confidence * 0.1
            
            # Normalize to 0.1 to 2.0 range (0.1 = poor quality, 2.0 = excellent quality)
            return max(0.1, min(2.0, quality_score))
            
        except Exception as e:
            logger.error(f"Error calculating signal quality for {timeframe if 'timeframe' in locals() else 'unknown'}: {e}")
            return 1.0  # Default neutral quality
    
    # Helper methods for signal interpretation
    def _get_rsi_status(self, rsi_value: float) -> str:
        if rsi_value is None:
            return 'neutral'
        if rsi_value < 30:
            return 'oversold'
        elif rsi_value > 70:
            return 'overbought'
        elif rsi_value > 50:
            return 'bullish'
        else:
            return 'bearish'
    
    def _get_macd_signal(self, macd_line: float, signal_line: float, histogram: float) -> str:
        if macd_line is None or signal_line is None:
            return 'neutral'
        if macd_line > signal_line and histogram > 0:
            return 'bullish'
        elif macd_line < signal_line and histogram < 0:
            return 'bearish'
        else:
            return 'neutral'
    
    def _get_bb_position(self, price: float, upper: float, lower: float) -> str:
        if upper is None or lower is None:
            return 'neutral'
        if price <= lower:
            return 'oversold'
        elif price >= upper:
            return 'overbought'
        else:
            return 'neutral'
    
    def _get_stochastic_signal(self, k_value: float, d_value: float) -> str:
        if k_value is None or d_value is None:
            return 'neutral'
        if k_value < 20 and d_value < 20:
            return 'oversold'
        elif k_value > 80 and d_value > 80:
            return 'overbought'
        elif k_value > d_value:
            return 'bullish'
        else:
            return 'bearish'
    
    def _get_ichimoku_signal(self, price: float, ichimoku: Dict[str, pd.Series]) -> str:
        try:
            tenkan = ichimoku['tenkan'].iloc[-1]
            kijun = ichimoku['kijun'].iloc[-1]
            senkou_a = ichimoku['senkou_a'].iloc[-1]
            senkou_b = ichimoku['senkou_b'].iloc[-1]
            
            if pd.isna(tenkan) or pd.isna(kijun) or pd.isna(senkou_a) or pd.isna(senkou_b):
                return 'neutral'
            
            # Price above cloud and tenkan > kijun
            if price > senkou_a and price > senkou_b and tenkan > kijun:
                return 'bullish'
            # Price below cloud and tenkan < kijun
            elif price < senkou_a and price < senkou_b and tenkan < kijun:
                return 'bearish'
            else:
                return 'neutral'
        except:
            return 'neutral'
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            peak = prices.expanding().max()
            drawdown = (prices - peak) / peak
            return float(drawdown.min())
        except:
            return 0.0
    
    def _calculate_risk_reward_ratio(self, current_price: float, support_levels: List[float], resistance_levels: List[float]) -> float:
        """Calculate risk/reward ratio based on nearest support and resistance."""
        try:
            if not support_levels or not resistance_levels:
                return 1.0
            
            nearest_support = max([s for s in support_levels if s < current_price], default=current_price * 0.95)
            nearest_resistance = min([r for r in resistance_levels if r > current_price], default=current_price * 1.05)
            
            risk = current_price - nearest_support
            reward = nearest_resistance - current_price
            
            if risk <= 0:
                return 1.0
            
            return reward / risk
        except:
            return 1.0
    
    def _determine_risk_level(self, confidence_score: float, divergence_detected: bool) -> str:
        """Determine risk level based on confidence and divergence."""
        if divergence_detected:
            return 'High'
        elif confidence_score >= 0.8:
            return 'Low'
        elif confidence_score >= 0.6:
            return 'Medium'
        else:
            return 'High'
    
    def _generate_recommendation(self, trend: str, confidence: float, divergence_detected: bool) -> str:
        """Generate trading recommendation."""
        if divergence_detected:
            return 'Wait for confirmation - divergence detected'
        elif confidence >= 0.8:
            if trend == 'bullish':
                return 'Strong Buy'
            elif trend == 'bearish':
                return 'Strong Sell'
            else:
                return 'Hold'
        elif confidence >= 0.6:
            if trend == 'bullish':
                return 'Buy'
            elif trend == 'bearish':
                return 'Sell'
            else:
                return 'Hold'
        else:
            return 'Wait for better signals'

# Global instance
enhanced_mtf_analyzer = EnhancedMultiTimeframeAnalyzer() 