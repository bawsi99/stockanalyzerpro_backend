"""
Trend Indicators Agent Processor

Specialized agent for analyzing trend indicators including moving averages,
trend lines, and directional indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import asyncio
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TrendSignal:
    """Data structure for a trend signal"""
    signal_type: str
    direction: str  # bullish/bearish/neutral
    strength: str   # strong/moderate/weak
    confidence: float
    price_level: float
    description: str

class TrendIndicatorsProcessor:
    """
    Processor for analyzing trend indicators in stock data.
    
    This processor specializes in:
    - Moving average analysis (SMA, EMA)
    - Trend direction and strength assessment
    - Trend line identification
    - ADX-based trend strength
    """
    
    def __init__(self):
        self.name = "trend_indicators"
        self.description = "Analyzes trend direction and strength using moving averages and trend indicators"
    
    async def analyze_async(self, stock_data: pd.DataFrame, indicators: Dict[str, Any] = None, 
                          context: str = "", chart_image: bytes = None) -> Dict[str, Any]:
        """
        Asynchronous analysis of trend indicators
        
        Args:
            stock_data: OHLCV price data
            indicators: Technical indicators dictionary
            context: Additional context for analysis
            chart_image: Chart image for visual analysis
            
        Returns:
            Dictionary containing trend analysis results
        """
        try:
            start_time = datetime.now()
            
            # Extract price data
            prices = stock_data['close'].values
            
            # Analyze different trend aspects
            moving_average_analysis = await self._analyze_moving_averages(prices, indicators)
            trend_strength_analysis = await self._analyze_trend_strength(prices, indicators)
            trend_direction_analysis = await self._analyze_trend_direction(prices, indicators)
            trend_continuity_analysis = await self._analyze_trend_continuity(prices, indicators)
            
            # Generate trend signals
            trend_signals = await self._generate_trend_signals(
                moving_average_analysis, trend_strength_analysis, 
                trend_direction_analysis, trend_continuity_analysis
            )
            
            # Calculate overall trend assessment
            overall_trend = await self._assess_overall_trend(
                moving_average_analysis, trend_strength_analysis,
                trend_direction_analysis, prices
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Build comprehensive analysis result
            analysis_result = {
                'agent_name': self.name,
                'processing_time': processing_time,
                'timestamp': start_time.isoformat(),
                
                # Trend Analysis
                'moving_average_analysis': moving_average_analysis,
                'trend_strength_analysis': trend_strength_analysis,
                'trend_direction_analysis': trend_direction_analysis,
                'trend_continuity_analysis': trend_continuity_analysis,
                
                # Signals and Assessment
                'trend_signals': [self._signal_to_dict(signal) for signal in trend_signals],
                'overall_trend': overall_trend,
                
                # Trading Insights
                'entry_signals': self._identify_entry_signals(trend_signals, prices),
                'support_resistance_from_trend': self._identify_trend_based_levels(prices, indicators),
                
                # Risk Assessment
                'trend_risk_assessment': self._assess_trend_risks(overall_trend, trend_signals),
                
                # Confidence Metrics
                'confidence_score': self._calculate_confidence_score(
                    moving_average_analysis, trend_strength_analysis, trend_direction_analysis
                ),
                
                # Context
                'market_context': context,
                'current_price': prices[-1] if len(prices) > 0 else 0.0
            }
            
            logger.info(f"[TREND_INDICATORS] Analysis completed in {processing_time:.2f}s")
            return analysis_result
            
        except Exception as e:
            logger.error(f"[TREND_INDICATORS] Analysis failed: {str(e)}")
            return {
                'agent_name': self.name,
                'error': str(e),
                'success': False,
                'confidence_score': 0.0
            }
    
    async def _analyze_moving_averages(self, prices: np.ndarray, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze moving average indicators (supports optimized and legacy schemas)"""
        
        ma_analysis = {
            'sma_analysis': {},
            'ema_analysis': {},
            'ma_crossover_signals': [],
            'ma_trend_alignment': {}
        }
        
        def get_ma_value(indics: Dict[str, Any], key: str) -> Optional[float]:
            try:
                if not indics:
                    return None
                if 'moving_averages' in indics and isinstance(indics['moving_averages'], dict) and key in indics['moving_averages']:
                    val = indics['moving_averages'][key]
                    return float(val) if val is not None else None
                if key in indics:
                    seq = indics[key]
                    if hasattr(seq, '__len__') and len(seq) > 0:
                        return float(seq[-1])
                return None
            except Exception:
                return None
        
        def get_ma_series(indics: Dict[str, Any], key: str) -> Optional[np.ndarray]:
            try:
                if not indics:
                    return None
                if key in indics and hasattr(indics[key], '__len__') and len(indics[key]) > 0:
                    return np.array(indics[key], dtype=float)
                return None
            except Exception:
                return None
        
        current_price = float(prices[-1]) if len(prices) > 0 else 0.0
        
        # SMA Analysis (use last values when available)
        sma20_val = get_ma_value(indicators, 'sma_20')
        sma50_val = get_ma_value(indicators, 'sma_50')
        if sma20_val is not None and sma50_val is not None:
            sma20_series = get_ma_series(indicators, 'sma_20')
            sma50_series = get_ma_series(indicators, 'sma_50')
            sma20_trend = self._calculate_ma_trend(sma20_series) if isinstance(sma20_series, np.ndarray) else 'neutral'
            sma50_trend = self._calculate_ma_trend(sma50_series) if isinstance(sma50_series, np.ndarray) else 'neutral'
            
            # Crossover only if we have series; else neutral
            sma_cross = self._detect_ma_crossover(sma20_series, sma50_series) if isinstance(sma20_series, np.ndarray) and isinstance(sma50_series, np.ndarray) else {'type': 'none', 'direction': 'neutral'}
            
            ma_analysis['sma_analysis'] = {
                'sma_20_trend': sma20_trend,
                'sma_50_trend': sma50_trend,
                'price_vs_sma20': 'above' if current_price > sma20_val else 'below',
                'price_vs_sma50': 'above' if current_price > sma50_val else 'below',
                'sma_20_50_cross': sma_cross
            }
        
        # EMA Analysis (support ema_12/26 legacy and ema_20/50 optimized)
        # Legacy ema_12/26 if series present
        ema12_series = get_ma_series(indicators, 'ema_12')
        ema26_series = get_ma_series(indicators, 'ema_26')
        if isinstance(ema12_series, np.ndarray) and isinstance(ema26_series, np.ndarray):
            ma_analysis['ema_analysis'] = {
                'ema_12_trend': self._calculate_ma_trend(ema12_series),
                'ema_26_trend': self._calculate_ma_trend(ema26_series),
                'price_vs_ema12': 'above' if current_price > float(ema12_series[-1]) else 'below',
                'price_vs_ema26': 'above' if current_price > float(ema26_series[-1]) else 'below',
                'ema_12_26_cross': self._detect_ma_crossover(ema12_series, ema26_series)
            }
        else:
            # Optimized ema_20/50 as a proxy
            ema20_val = get_ma_value(indicators, 'ema_20')
            ema50_val = get_ma_value(indicators, 'ema_50')
            if ema20_val is not None and ema50_val is not None:
                ma_analysis['ema_analysis'] = {
                    'ema_20_trend': 'neutral',
                    'ema_50_trend': 'neutral',
                    'price_vs_ema20': 'above' if current_price > ema20_val else 'below',
                    'price_vs_ema50': 'above' if current_price > ema50_val else 'below',
                    'ema_20_50_cross': {'type': 'none', 'direction': 'neutral'}
                }
        
        # MA Crossover Signals
        ma_analysis['ma_crossover_signals'] = await self._identify_ma_crossovers(indicators)
        
        # Trend Alignment
        ma_analysis['ma_trend_alignment'] = await self._assess_ma_alignment(indicators, prices)
        
        return ma_analysis
    
    async def _analyze_trend_strength(self, prices: np.ndarray, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trend strength using various methods"""
        
        strength_analysis = {
            'adx_strength': {},
            'price_momentum_strength': {},
            'trend_consistency': {},
            'overall_strength': 'weak'
        }
        
        # ADX-based strength
        if indicators and 'adx' in indicators:
            adx_obj = indicators['adx']
            current_adx = 25
            adx_trend_value = 'neutral'
            try:
                if isinstance(adx_obj, dict):
                    # Optimized indicators shape: {'adx': float, 'plus_di': float, 'minus_di': float, ...}
                    if adx_obj.get('adx') is not None:
                        current_adx = float(adx_obj.get('adx'))
                    # No reliable history here; keep trend neutral
                    adx_trend_value = 'neutral'
                else:
                    # Legacy/array-like shape
                    if len(adx_obj) > 0:
                        current_adx = float(adx_obj[-1])
                    if hasattr(adx_obj, '__len__') and len(adx_obj) >= 10:
                        adx_trend_value = self._calculate_ma_trend(adx_obj[-10:])
            except Exception:
                # Fall back to defaults
                current_adx = 25
                adx_trend_value = 'neutral'
            
            strength_analysis['adx_strength'] = {
                'current_adx': current_adx,
                'strength_level': self._interpret_adx_strength(current_adx),
                'adx_trend': adx_trend_value
            }
        
        # Price momentum strength
        if len(prices) >= 20:
            recent_returns = np.diff(prices[-20:]) / prices[-20:-1]
            strength_analysis['price_momentum_strength'] = {
                'average_return': np.mean(recent_returns),
                'return_consistency': self._calculate_return_consistency(recent_returns),
                'momentum_acceleration': self._calculate_momentum_acceleration(prices)
            }
        
        # Trend consistency
        strength_analysis['trend_consistency'] = await self._calculate_trend_consistency(prices)
        
        # Overall strength assessment
        strength_analysis['overall_strength'] = self._assess_overall_strength(strength_analysis)
        
        return strength_analysis
    
    async def _analyze_trend_direction(self, prices: np.ndarray, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trend direction using multiple methods"""
        
        direction_analysis = {
            'short_term_direction': 'neutral',
            'medium_term_direction': 'neutral',
            'long_term_direction': 'neutral',
            'direction_consensus': 'neutral',
            'direction_confidence': 0.0
        }
        
        # Short-term direction (last 10 periods)
        if len(prices) >= 10:
            direction_analysis['short_term_direction'] = self._calculate_direction(prices[-10:])
        
        # Medium-term direction (last 30 periods)
        if len(prices) >= 30:
            direction_analysis['medium_term_direction'] = self._calculate_direction(prices[-30:])
        
        # Long-term direction (last 60 periods)
        if len(prices) >= 60:
            direction_analysis['long_term_direction'] = self._calculate_direction(prices[-60:])
        
        # Direction consensus
        directions = [
            direction_analysis['short_term_direction'],
            direction_analysis['medium_term_direction'], 
            direction_analysis['long_term_direction']
        ]
        direction_analysis['direction_consensus'] = self._get_direction_consensus(directions)
        
        # Direction confidence
        direction_analysis['direction_confidence'] = self._calculate_direction_confidence(directions)
        
        return direction_analysis
    
    async def _analyze_trend_continuity(self, prices: np.ndarray, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trend continuity and potential for continuation or reversal"""
        
        continuity_analysis = {
            'trend_maturity': 'young',
            'continuation_probability': 0.5,
            'reversal_signals': [],
            'trend_exhaustion_indicators': []
        }
        
        if len(prices) < 20:
            return continuity_analysis
        
        # Calculate trend maturity
        continuity_analysis['trend_maturity'] = self._assess_trend_maturity(prices)
        
        # Continuation probability
        continuity_analysis['continuation_probability'] = self._calculate_continuation_probability(prices, indicators)
        
        # Reversal signals
        continuity_analysis['reversal_signals'] = self._identify_reversal_signals(prices, indicators)
        
        # Trend exhaustion indicators
        continuity_analysis['trend_exhaustion_indicators'] = self._identify_exhaustion_signals(prices, indicators)
        
        return continuity_analysis
    
    async def _generate_trend_signals(self, ma_analysis: Dict, strength_analysis: Dict, 
                                    direction_analysis: Dict, continuity_analysis: Dict) -> List[TrendSignal]:
        """Generate actionable trend signals"""
        
        signals = []
        
        # Generate MA crossover signals
        if 'ma_crossover_signals' in ma_analysis:
            for crossover in ma_analysis['ma_crossover_signals']:
                signal = TrendSignal(
                    signal_type="ma_crossover",
                    direction=crossover['direction'],
                    strength=crossover['strength'],
                    confidence=crossover['confidence'],
                    price_level=crossover['price'],
                    description=crossover['description']
                )
                signals.append(signal)
        
        # Generate trend strength signals
        overall_strength = strength_analysis.get('overall_strength', 'weak')
        direction_consensus = direction_analysis.get('direction_consensus', 'neutral')
        
        if overall_strength in ['strong', 'moderate'] and direction_consensus != 'neutral':
            signal = TrendSignal(
                signal_type="trend_strength",
                direction=direction_consensus,
                strength=overall_strength,
                confidence=direction_analysis.get('direction_confidence', 0.5),
                price_level=0.0,  # Will be set by calling function
                description=f"{overall_strength.title()} {direction_consensus} trend identified"
            )
            signals.append(signal)
        
        return signals
    
    async def _assess_overall_trend(self, ma_analysis: Dict, strength_analysis: Dict, 
                                  direction_analysis: Dict, prices: np.ndarray) -> Dict[str, Any]:
        """Assess overall trend condition"""
        
        overall_trend = {
            'direction': direction_analysis.get('direction_consensus', 'neutral'),
            'strength': strength_analysis.get('overall_strength', 'weak'),
            'confidence': direction_analysis.get('direction_confidence', 0.5),
            'current_price': prices[-1] if len(prices) > 0 else 0.0,
            'trend_quality': 'low',
            'actionable': False
        }
        
        # Assess trend quality
        overall_trend['trend_quality'] = self._assess_trend_quality(ma_analysis, strength_analysis, direction_analysis)
        
        # Determine if trend is actionable
        overall_trend['actionable'] = (
            overall_trend['strength'] in ['strong', 'moderate'] and
            overall_trend['direction'] != 'neutral' and
            overall_trend['confidence'] > 0.6
        )
        
        return overall_trend
    
    def _calculate_ma_trend(self, ma_values: np.ndarray) -> str:
        """Calculate moving average trend direction"""
        if len(ma_values) < 3:
            return 'neutral'
        
        recent_slope = (ma_values[-1] - ma_values[-3]) / ma_values[-3]
        
        if recent_slope > 0.001:
            return 'uptrend'
        elif recent_slope < -0.001:
            return 'downtrend'
        else:
            return 'sideways'
    
    def _detect_ma_crossover(self, fast_ma: Any, slow_ma: Any) -> Dict[str, Any]:
        """Detect moving average crossover. Supports only sequence inputs; scalars return neutral."""
        try:
            if not hasattr(fast_ma, '__len__') or not hasattr(slow_ma, '__len__'):
                return {'type': 'none', 'direction': 'neutral'}
            if len(fast_ma) < 2 or len(slow_ma) < 2:
                return {'type': 'none', 'direction': 'neutral'}
        except Exception:
            return {'type': 'none', 'direction': 'neutral'}
        
        # Current and previous relationships
        current_above = fast_ma[-1] > slow_ma[-1]
        previous_above = fast_ma[-2] > slow_ma[-2]
        
        if current_above and not previous_above:
            return {'type': 'golden_cross', 'direction': 'bullish'}
        elif not current_above and previous_above:
            return {'type': 'death_cross', 'direction': 'bearish'}
        else:
            return {'type': 'none', 'direction': 'neutral'}
    
    async def _identify_ma_crossovers(self, indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify all moving average crossovers"""
        crossovers = []
        
        # Check various MA combinations
        ma_pairs = [
            ('ema_12', 'ema_26', 'EMA 12/26'),
            ('sma_20', 'sma_50', 'SMA 20/50'),
            ('sma_50', 'sma_200', 'SMA 50/200'),
            # Optimized proxies (likely scalars -> neutral cross)
            ('ema_20', 'ema_50', 'EMA 20/50'),
        ]
        
        def get_series_or_none(indics: Dict[str, Any], key: str) -> Optional[np.ndarray]:
            try:
                if key in indics and hasattr(indics[key], '__len__'):
                    arr = indics[key]
                    if len(arr) > 0:
                        return np.array(arr, dtype=float)
                return None
            except Exception:
                return None
        
        def get_from_moving(indics: Dict[str, Any], key: str) -> Optional[float]:
            try:
                mv = indics.get('moving_averages') if indics else None
                if isinstance(mv, dict) and key in mv:
                    v = mv[key]
                    return float(v) if v is not None else None
                return None
            except Exception:
                return None
        
        for fast_key, slow_key, description in ma_pairs:
            if not indicators:
                continue
            fast_seq = get_series_or_none(indicators, fast_key)
            slow_seq = get_series_or_none(indicators, slow_key)
            if isinstance(fast_seq, np.ndarray) and isinstance(slow_seq, np.ndarray):
                crossover = self._detect_ma_crossover(fast_seq, slow_seq)
            else:
                # Try optimized scalar-only case: cannot detect; neutral
                fval = get_from_moving(indicators, fast_key)
                sval = get_from_moving(indicators, slow_key)
                if fval is None or sval is None:
                    continue
                crossover = {'type': 'none', 'direction': 'neutral'}
            if crossover['type'] != 'none':
                crossovers.append({
                    'pair': description,
                    'type': crossover['type'],
                    'direction': crossover['direction'],
                    'strength': 'moderate',
                    'confidence': 0.7,
                    'price': 0.0,  # Would need current price context
                    'description': f"{description} {crossover['type']}"
                })
        
        return crossovers
    
    async def _assess_ma_alignment(self, indicators: Dict[str, Any], prices: np.ndarray) -> Dict[str, Any]:
        """Assess moving average alignment for trend confirmation"""
        
        alignment = {
            'bullish_alignment': False,
            'bearish_alignment': False,
            'alignment_strength': 0.0,
            'alignment_quality': 'poor'
        }
        
        if not indicators:
            return alignment
        
        current_price = float(prices[-1]) if len(prices) > 0 else 0.0
        ma_keys = ['ema_12', 'ema_26', 'ema_20', 'ema_50', 'sma_20', 'sma_50', 'sma_200']
        available_mas: Dict[str, float] = {}
        # Prefer optimized moving_averages values
        if 'moving_averages' in indicators and isinstance(indicators['moving_averages'], dict):
            for k in ma_keys:
                if k in indicators['moving_averages'] and indicators['moving_averages'][k] is not None:
                    try:
                        available_mas[k] = float(indicators['moving_averages'][k])
                    except Exception:
                        pass
        # Fallback to legacy arrays
        for k in ma_keys:
            if k not in available_mas and k in indicators and hasattr(indicators[k], '__len__') and len(indicators[k]) > 0:
                try:
                    available_mas[k] = float(indicators[k][-1])
                except Exception:
                    pass
        
        if len(available_mas) < 2:
            return alignment
        
        # Check for bullish alignment (price > all MAs, MAs in ascending order)
        ma_values = list(available_mas.values())
        ma_values_with_price = [current_price] + ma_values
        
        bullish_aligned = all(ma_values_with_price[i] >= ma_values_with_price[i+1] for i in range(len(ma_values_with_price)-1))
        bearish_aligned = all(ma_values_with_price[i] <= ma_values_with_price[i+1] for i in range(len(ma_values_with_price)-1))
        
        alignment['bullish_alignment'] = bullish_aligned
        alignment['bearish_alignment'] = bearish_aligned
        
        # Calculate alignment strength
        if bullish_aligned or bearish_aligned:
            alignment['alignment_strength'] = min(1.0, len(available_mas) / 3.0)
            alignment['alignment_quality'] = 'excellent' if len(available_mas) >= 4 else 'good'
        
        return alignment
    
    def _calculate_direction(self, prices: np.ndarray) -> str:
        """Calculate trend direction using linear regression"""
        if len(prices) < 3:
            return 'neutral'
        
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        
        # Normalize slope by average price
        normalized_slope = slope / np.mean(prices)
        
        if normalized_slope > 0.002:
            return 'bullish'
        elif normalized_slope < -0.002:
            return 'bearish'
        else:
            return 'neutral'
    
    def _get_direction_consensus(self, directions: List[str]) -> str:
        """Get consensus direction from multiple timeframes"""
        bullish_count = directions.count('bullish')
        bearish_count = directions.count('bearish')
        
        if bullish_count > bearish_count:
            return 'bullish'
        elif bearish_count > bullish_count:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_direction_confidence(self, directions: List[str]) -> float:
        """Calculate confidence in direction consensus"""
        if not directions:
            return 0.0
        
        # Count occurrences of each direction
        direction_counts = {}
        for direction in directions:
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        
        # Calculate confidence based on consensus
        max_count = max(direction_counts.values())
        return max_count / len(directions)
    
    def _interpret_adx_strength(self, adx_value: float) -> str:
        """Interpret ADX value for trend strength"""
        if adx_value < 20:
            return 'weak'
        elif adx_value < 40:
            return 'moderate'
        else:
            return 'strong'
    
    def _calculate_return_consistency(self, returns: np.ndarray) -> float:
        """Calculate consistency of returns direction"""
        if len(returns) == 0:
            return 0.0
        
        positive_returns = np.sum(returns > 0)
        negative_returns = np.sum(returns < 0)
        total_returns = len(returns)
        
        # Return ratio of dominant direction
        return max(positive_returns, negative_returns) / total_returns
    
    def _calculate_momentum_acceleration(self, prices: np.ndarray) -> float:
        """Calculate momentum acceleration"""
        if len(prices) < 10:
            return 0.0
        
        # Calculate recent vs older momentum
        recent_momentum = (prices[-1] - prices[-5]) / prices[-5]
        older_momentum = (prices[-5] - prices[-10]) / prices[-10]
        
        return recent_momentum - older_momentum
    
    async def _calculate_trend_consistency(self, prices: np.ndarray) -> Dict[str, Any]:
        """Calculate trend consistency metrics"""
        
        consistency = {
            'direction_consistency': 0.0,
            'slope_consistency': 0.0,
            'overall_consistency': 0.0
        }
        
        if len(prices) < 10:
            return consistency
        
        # Calculate direction consistency
        daily_changes = np.diff(prices)
        up_days = np.sum(daily_changes > 0)
        down_days = np.sum(daily_changes < 0)
        total_days = len(daily_changes)
        
        consistency['direction_consistency'] = max(up_days, down_days) / total_days
        
        # Calculate slope consistency (using rolling windows)
        window_size = 5
        slopes = []
        
        for i in range(window_size, len(prices)):
            window_prices = prices[i-window_size:i]
            x = np.arange(len(window_prices))
            slope = np.polyfit(x, window_prices, 1)[0]
            slopes.append(slope)
        
        if slopes:
            # Consistency based on slope direction
            positive_slopes = np.sum(np.array(slopes) > 0)
            negative_slopes = np.sum(np.array(slopes) < 0)
            consistency['slope_consistency'] = max(positive_slopes, negative_slopes) / len(slopes)
        
        # Overall consistency
        consistency['overall_consistency'] = (
            consistency['direction_consistency'] * 0.6 + 
            consistency['slope_consistency'] * 0.4
        )
        
        return consistency
    
    def _assess_overall_strength(self, strength_analysis: Dict[str, Any]) -> str:
        """Assess overall trend strength"""
        
        # Get ADX strength if available
        adx_strength = strength_analysis.get('adx_strength', {}).get('strength_level', 'weak')
        
        # Get consistency metrics
        consistency = strength_analysis.get('trend_consistency', {}).get('overall_consistency', 0.0)
        
        # Combine factors
        if adx_strength == 'strong' and consistency > 0.7:
            return 'strong'
        elif adx_strength in ['moderate', 'strong'] and consistency > 0.5:
            return 'moderate'
        else:
            return 'weak'
    
    def _assess_trend_maturity(self, prices: np.ndarray) -> str:
        """Assess how mature/old the current trend is"""
        if len(prices) < 20:
            return 'young'
        
        # Simple assessment based on consecutive moves in same direction
        daily_changes = np.diff(prices[-20:])
        
        # Count consecutive moves in dominant direction
        current_direction = 1 if daily_changes[-1] > 0 else -1
        consecutive_count = 0
        
        for change in reversed(daily_changes):
            if (change > 0 and current_direction > 0) or (change < 0 and current_direction < 0):
                consecutive_count += 1
            else:
                break
        
        if consecutive_count < 3:
            return 'young'
        elif consecutive_count < 7:
            return 'mature'
        else:
            return 'old'
    
    def _calculate_continuation_probability(self, prices: np.ndarray, indicators: Dict[str, Any]) -> float:
        """Calculate probability of trend continuation"""
        if len(prices) < 10:
            return 0.5
        
        factors = []
        
        # Factor 1: Trend strength
        if indicators and 'adx' in indicators:
            adx_val = 25
            try:
                adx_obj = indicators['adx']
                if isinstance(adx_obj, dict):
                    if adx_obj.get('adx') is not None:
                        adx_val = float(adx_obj.get('adx'))
                else:
                    if len(adx_obj) > 0:
                        adx_val = float(adx_obj[-1])
            except Exception:
                adx_val = 25
            factors.append(min(1.0, adx_val / 50.0))  # Normalize ADX
        
        # Factor 2: Price momentum
        recent_momentum = (prices[-1] - prices[-10]) / prices[-10]
        factors.append(min(1.0, abs(recent_momentum) * 10))  # Scale momentum
        
        # Factor 3: Trend consistency
        daily_changes = np.diff(prices[-10:])
        consistent_direction = np.sum(daily_changes > 0) if np.sum(daily_changes) > 0 else np.sum(daily_changes < 0)
        factors.append(consistent_direction / len(daily_changes))
        
        return np.mean(factors) if factors else 0.5
    
    def _identify_reversal_signals(self, prices: np.ndarray, indicators: Dict[str, Any]) -> List[str]:
        """Identify potential trend reversal signals"""
        reversal_signals = []
        
        if len(prices) < 10:
            return reversal_signals
        
        # Signal 1: Momentum divergence (simplified)
        if len(prices) >= 20:
            price_high = np.max(prices[-10:])
            price_high_old = np.max(prices[-20:-10])
            
            if price_high > price_high_old:  # New price high
                # Check if momentum is weakening (simplified check)
                recent_momentum = abs(prices[-1] - prices[-5]) / prices[-5]
                older_momentum = abs(prices[-6] - prices[-10]) / prices[-10]
                
                if recent_momentum < older_momentum * 0.7:
                    reversal_signals.append("momentum_divergence")
        
        # Signal 2: Extreme movements
        if len(prices) >= 5:
            recent_change = abs(prices[-1] - prices[-5]) / prices[-5]
            if recent_change > 0.1:  # 10% move in 5 periods
                reversal_signals.append("extreme_movement")
        
        return reversal_signals
    
    def _identify_exhaustion_signals(self, prices: np.ndarray, indicators: Dict[str, Any]) -> List[str]:
        """Identify trend exhaustion signals"""
        exhaustion_signals = []
        
        # Signal 1: Extended moves without pullback
        if len(prices) >= 10:
            daily_changes = np.diff(prices[-10:])
            same_direction_count = 0
            direction = 1 if daily_changes[-1] > 0 else -1
            
            for change in reversed(daily_changes):
                if (change > 0 and direction > 0) or (change < 0 and direction < 0):
                    same_direction_count += 1
                else:
                    break
            
            if same_direction_count >= 7:
                exhaustion_signals.append("extended_move")
        
        # Signal 2: Decreasing volume with price extension (if volume available)
        # This would require volume data integration
        
        return exhaustion_signals
    
    def _identify_entry_signals(self, trend_signals: List[TrendSignal], prices: np.ndarray) -> List[Dict[str, Any]]:
        """Identify entry signals based on trend analysis"""
        entry_signals = []
        
        for signal in trend_signals:
            if signal.signal_type == "ma_crossover" and signal.strength in ['strong', 'moderate']:
                entry_signals.append({
                    'type': 'trend_entry',
                    'direction': signal.direction,
                    'entry_price': prices[-1] if len(prices) > 0 else 0.0,
                    'confidence': signal.confidence,
                    'rationale': signal.description
                })
        
        return entry_signals
    
    def _identify_trend_based_levels(self, prices: np.ndarray, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Identify support/resistance levels based on trend analysis"""
        
        levels = {
            'dynamic_support': 0.0,
            'dynamic_resistance': 0.0,
            'trend_channel_upper': 0.0,
            'trend_channel_lower': 0.0
        }
        
        if not indicators:
            return levels
        
        current_price = prices[-1] if len(prices) > 0 else 0.0
        
        # Dynamic support/resistance from moving averages
        if 'sma_20' in indicators and len(indicators['sma_20']) > 0:
            sma_20 = indicators['sma_20'][-1]
            if current_price > sma_20:
                levels['dynamic_support'] = sma_20
            else:
                levels['dynamic_resistance'] = sma_20
        
        return levels
    
    def _assess_trend_risks(self, overall_trend: Dict[str, Any], trend_signals: List[TrendSignal]) -> Dict[str, Any]:
        """Assess risks related to trend analysis"""
        
        risk_assessment = {
            'trend_reversal_risk': 'low',
            'false_signal_risk': 'low',
            'trend_exhaustion_risk': 'low',
            'overall_risk': 'low'
        }
        
        # Assess based on trend strength and signals
        if overall_trend.get('strength') == 'weak':
            risk_assessment['false_signal_risk'] = 'high'
        
        # Check for conflicting signals
        bullish_signals = [s for s in trend_signals if s.direction == 'bullish']
        bearish_signals = [s for s in trend_signals if s.direction == 'bearish']
        
        if len(bullish_signals) > 0 and len(bearish_signals) > 0:
            risk_assessment['false_signal_risk'] = 'medium'
        
        # Overall risk
        risk_levels = [risk_assessment[key] for key in risk_assessment if key != 'overall_risk']
        if 'high' in risk_levels:
            risk_assessment['overall_risk'] = 'high'
        elif 'medium' in risk_levels:
            risk_assessment['overall_risk'] = 'medium'
        
        return risk_assessment
    
    def _calculate_confidence_score(self, ma_analysis: Dict, strength_analysis: Dict, direction_analysis: Dict) -> float:
        """Calculate overall confidence score for trend analysis"""
        
        confidence_factors = []
        
        # Factor 1: Direction consensus confidence
        direction_confidence = direction_analysis.get('direction_confidence', 0.0)
        confidence_factors.append(direction_confidence)
        
        # Factor 2: Trend strength
        strength = strength_analysis.get('overall_strength', 'weak')
        strength_score = {'strong': 1.0, 'moderate': 0.7, 'weak': 0.3}.get(strength, 0.3)
        confidence_factors.append(strength_score)
        
        # Factor 3: MA alignment quality
        alignment = ma_analysis.get('ma_trend_alignment', {})
        if alignment.get('bullish_alignment') or alignment.get('bearish_alignment'):
            alignment_score = alignment.get('alignment_strength', 0.0)
            confidence_factors.append(alignment_score)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _assess_trend_quality(self, ma_analysis: Dict, strength_analysis: Dict, direction_analysis: Dict) -> str:
        """Assess overall trend quality"""
        
        # Get key metrics
        direction_confidence = direction_analysis.get('direction_confidence', 0.0)
        strength = strength_analysis.get('overall_strength', 'weak')
        alignment = ma_analysis.get('ma_trend_alignment', {})
        
        # Quality assessment
        if (direction_confidence > 0.8 and 
            strength == 'strong' and 
            (alignment.get('bullish_alignment') or alignment.get('bearish_alignment'))):
            return 'excellent'
        elif (direction_confidence > 0.6 and 
              strength in ['strong', 'moderate']):
            return 'good'
        elif direction_confidence > 0.5:
            return 'fair'
        else:
            return 'poor'
    
    def _signal_to_dict(self, signal: TrendSignal) -> Dict[str, Any]:
        """Convert TrendSignal to dictionary"""
        return {
            'signal_type': signal.signal_type,
            'direction': signal.direction,
            'strength': signal.strength,
            'confidence': signal.confidence,
            'price_level': signal.price_level,
            'description': signal.description
        }