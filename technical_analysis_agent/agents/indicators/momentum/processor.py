"""
Momentum Indicators Agent Processor

Specialized agent for analyzing momentum indicators including RSI, MACD,
Stochastic, and other oscillators.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class MomentumIndicatorsProcessor:
    """
    Processor for analyzing momentum indicators in stock data.
    
    This processor specializes in:
    - RSI analysis and signals
    - MACD analysis and crossovers
    - Stochastic oscillator signals
    - Momentum divergences
    """
    
    def __init__(self):
        self.name = "momentum_indicators"
        self.description = "Analyzes momentum indicators and oscillator signals"
    
    async def analyze_async(self, stock_data: pd.DataFrame, indicators: Dict[str, Any] = None, 
                          context: str = "", chart_image: bytes = None) -> Dict[str, Any]:
        """
        Asynchronous analysis of momentum indicators
        
        Args:
            stock_data: OHLCV price data
            indicators: Technical indicators dictionary
            context: Additional context for analysis
            chart_image: Chart image for visual analysis
            
        Returns:
            Dictionary containing momentum analysis results
        """
        try:
            start_time = datetime.now()
            
            # Extract price data
            prices = stock_data['close'].values
            
            # Analyze different momentum aspects
            rsi_analysis = await self._analyze_rsi(indicators, prices)
            macd_analysis = await self._analyze_macd(indicators, prices)
            stochastic_analysis = await self._analyze_stochastic(indicators, prices)
            divergence_analysis = await self._analyze_divergences(indicators, prices)
            
            # Generate momentum signals
            momentum_signals = await self._generate_momentum_signals(
                rsi_analysis, macd_analysis, stochastic_analysis, divergence_analysis
            )
            
            # Calculate overall momentum assessment
            overall_momentum = await self._assess_overall_momentum(
                rsi_analysis, macd_analysis, stochastic_analysis, prices
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Build comprehensive analysis result
            analysis_result = {
                'agent_name': self.name,
                'processing_time': processing_time,
                'timestamp': start_time.isoformat(),
                
                # Momentum Analysis
                'rsi_analysis': rsi_analysis,
                'macd_analysis': macd_analysis,
                'stochastic_analysis': stochastic_analysis,
                'divergence_analysis': divergence_analysis,
                
                # Signals and Assessment
                'momentum_signals': momentum_signals,
                'overall_momentum': overall_momentum,
                
                # Trading Insights
                'overbought_oversold_signals': self._identify_extreme_signals(rsi_analysis, stochastic_analysis),
                'crossover_signals': self._identify_crossover_signals(macd_analysis, stochastic_analysis),
                
                # Confidence Metrics
                'confidence_score': self._calculate_confidence_score(
                    rsi_analysis, macd_analysis, stochastic_analysis
                ),
                
                # Context
                'market_context': context,
                'current_price': prices[-1] if len(prices) > 0 else 0.0
            }
            
            logger.info(f"[MOMENTUM_INDICATORS] Analysis completed in {processing_time:.2f}s")
            return analysis_result
            
        except Exception as e:
            logger.error(f"[MOMENTUM_INDICATORS] Analysis failed: {str(e)}")
            return {
                'agent_name': self.name,
                'error': str(e),
                'success': False,
                'confidence_score': 0.0
            }
    
    async def _analyze_rsi(self, indicators: Dict[str, Any], prices: np.ndarray) -> Dict[str, Any]:
        """Analyze RSI indicator"""
        
        rsi_analysis = {
            'current_rsi': 50.0,
            'rsi_trend': 'neutral',
            'signal': 'neutral',
            'extremes': [],
            'divergences': []
        }
        
        if not indicators or 'rsi' not in indicators:
            return rsi_analysis
        
        rsi_obj = indicators['rsi']
        # Support both optimized dict shape and legacy array-like
        recent_values = []
        if isinstance(rsi_obj, dict):
            current_rsi = float(rsi_obj.get('rsi_14', 50.0))
            recent_values = rsi_obj.get('recent_values') or []
            try:
                recent_values = [float(x) for x in recent_values if x is not None]
            except Exception:
                recent_values = []
            rsi_series = np.array(recent_values, dtype=float) if recent_values else np.array([current_rsi], dtype=float)
        else:
            # Assume sequence/array-like
            try:
                current_rsi = float(rsi_obj[-1]) if len(rsi_obj) > 0 else 50.0
            except Exception:
                current_rsi = 50.0
            rsi_series = np.array(rsi_obj, dtype=float) if hasattr(rsi_obj, '__len__') else np.array([current_rsi], dtype=float)
        
        rsi_analysis['current_rsi'] = current_rsi
        
        # RSI trend
        if len(rsi_series) >= 5:
            recent_rsi_trend = (np.mean(rsi_series[-3:]) - np.mean(rsi_series[-6:-3])) if len(rsi_series) >= 6 else 0
            rsi_analysis['rsi_trend'] = 'rising' if recent_rsi_trend > 1 else 'falling' if recent_rsi_trend < -1 else 'sideways'
        
        # RSI signal
        if current_rsi >= 70:
            rsi_analysis['signal'] = 'overbought'
        elif current_rsi <= 30:
            rsi_analysis['signal'] = 'oversold'
        elif current_rsi >= 50:
            rsi_analysis['signal'] = 'bullish'
        else:
            rsi_analysis['signal'] = 'bearish'
        
        # Extreme levels
        rsi_analysis['extremes'] = self._identify_rsi_extremes(rsi_series)
        
        return rsi_analysis
    
    async def _analyze_macd(self, indicators: Dict[str, Any], prices: np.ndarray) -> Dict[str, Any]:
        """Analyze MACD indicator"""
        
        macd_analysis = {
            'macd_signal': 'neutral',
            'histogram_signal': 'neutral',
            'crossover_signals': [],
            'divergences': []
        }
        
        if not indicators or 'macd' not in indicators:
            return macd_analysis
        
        macd_obj = indicators['macd']
        # Optimized dict shape
        if isinstance(macd_obj, dict):
            try:
                macd_line = macd_obj.get('macd_line')
                signal_line = macd_obj.get('signal_line')
                histogram = macd_obj.get('histogram')
                if macd_line is not None and signal_line is not None:
                    macd_analysis['macd_signal'] = 'bullish' if float(macd_line) > float(signal_line) else 'bearish'
                # Histogram trend is not available without history; classify by sign if present
                if histogram is not None:
                    h = float(histogram)
                    macd_analysis['histogram_signal'] = 'improving' if h > 0 else 'deteriorating' if h < 0 else 'neutral'
            except Exception:
                pass
            # No reliable crossover signals without history
            macd_analysis['crossover_signals'] = []
            return macd_analysis
        
        # Legacy array-like shape
        if 'macd_signal' not in indicators:
            return macd_analysis
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        if len(macd) < 2 or len(macd_signal) < 2:
            return macd_analysis
        
        # Current MACD vs Signal
        current_macd = macd[-1]
        current_signal = macd_signal[-1]
        macd_analysis['macd_signal'] = 'bullish' if current_macd > current_signal else 'bearish'
        
        # Histogram signal (legacy path)
        if 'macd_histogram' in indicators:
            histogram = indicators['macd_histogram']
            if len(histogram) >= 2:
                macd_analysis['histogram_signal'] = 'improving' if histogram[-1] > histogram[-2] else 'deteriorating'
        
        # Crossover signals
        macd_analysis['crossover_signals'] = self._identify_macd_crossovers(macd, macd_signal)
        
        return macd_analysis
    
    async def _analyze_stochastic(self, indicators: Dict[str, Any], prices: np.ndarray) -> Dict[str, Any]:
        """Analyze Stochastic oscillator"""
        
        stoch_analysis = {
            'stoch_signal': 'neutral',
            'k_d_relationship': 'neutral',
            'crossover_signals': [],
            'extremes': []
        }
        
        if not indicators or 'stoch_k' not in indicators or 'stoch_d' not in indicators:
            return stoch_analysis
        
        stoch_k = indicators['stoch_k']
        stoch_d = indicators['stoch_d']
        
        if len(stoch_k) == 0 or len(stoch_d) == 0:
            return stoch_analysis
        
        current_k = stoch_k[-1]
        current_d = stoch_d[-1]
        
        # Stochastic signal based on levels
        if current_k >= 80:
            stoch_analysis['stoch_signal'] = 'overbought'
        elif current_k <= 20:
            stoch_analysis['stoch_signal'] = 'oversold'
        else:
            stoch_analysis['stoch_signal'] = 'neutral'
        
        # K vs D relationship
        if current_k > current_d:
            stoch_analysis['k_d_relationship'] = 'bullish'
        else:
            stoch_analysis['k_d_relationship'] = 'bearish'
        
        # Crossover signals
        stoch_analysis['crossover_signals'] = self._identify_stoch_crossovers(stoch_k, stoch_d)
        
        return stoch_analysis
    
    async def _analyze_divergences(self, indicators: Dict[str, Any], prices: np.ndarray) -> Dict[str, Any]:
        """Analyze momentum divergences"""
        
        divergence_analysis = {
            'rsi_divergences': [],
            'macd_divergences': [],
            'overall_divergence_signal': 'none'
        }
        
        if len(prices) < 20:
            return divergence_analysis
        
        # RSI divergences (simplified)
        if indicators and 'rsi' in indicators:
            rsi = indicators['rsi']
            divergence_analysis['rsi_divergences'] = self._detect_rsi_divergences(prices, rsi)
        
        # MACD divergences (simplified)
        if indicators and 'macd' in indicators:
            macd = indicators['macd']
            divergence_analysis['macd_divergences'] = self._detect_macd_divergences(prices, macd)
        
        # Overall divergence signal
        total_divergences = len(divergence_analysis['rsi_divergences']) + len(divergence_analysis['macd_divergences'])
        if total_divergences > 0:
            # Determine dominant divergence type
            bullish_divs = sum(1 for div in divergence_analysis['rsi_divergences'] + divergence_analysis['macd_divergences'] if div.get('type') == 'bullish')
            bearish_divs = sum(1 for div in divergence_analysis['rsi_divergences'] + divergence_analysis['macd_divergences'] if div.get('type') == 'bearish')
            
            if bullish_divs > bearish_divs:
                divergence_analysis['overall_divergence_signal'] = 'bullish'
            elif bearish_divs > bullish_divs:
                divergence_analysis['overall_divergence_signal'] = 'bearish'
        
        return divergence_analysis
    
    async def _generate_momentum_signals(self, rsi_analysis: Dict, macd_analysis: Dict,
                                       stoch_analysis: Dict, divergence_analysis: Dict) -> List[Dict[str, Any]]:
        """Generate actionable momentum signals"""
        
        signals = []
        
        # RSI signals
        rsi_signal = rsi_analysis.get('signal', 'neutral')
        if rsi_signal in ['overbought', 'oversold']:
            signals.append({
                'type': 'rsi_extreme',
                'signal': rsi_signal,
                'confidence': 0.7,
                'description': f"RSI {rsi_signal} at {rsi_analysis.get('current_rsi', 50):.1f}"
            })
        
        # MACD signals
        macd_crossovers = macd_analysis.get('crossover_signals', [])
        for crossover in macd_crossovers:
            signals.append({
                'type': 'macd_crossover',
                'signal': crossover['type'],
                'confidence': 0.6,
                'description': f"MACD {crossover['type']} crossover"
            })
        
        # Divergence signals
        if divergence_analysis.get('overall_divergence_signal') != 'none':
            signals.append({
                'type': 'divergence',
                'signal': divergence_analysis['overall_divergence_signal'],
                'confidence': 0.8,
                'description': f"{divergence_analysis['overall_divergence_signal'].title()} divergence detected"
            })
        
        return signals
    
    async def _assess_overall_momentum(self, rsi_analysis: Dict, macd_analysis: Dict,
                                     stoch_analysis: Dict, prices: np.ndarray) -> Dict[str, Any]:
        """Assess overall momentum condition"""
        
        overall_momentum = {
            'direction': 'neutral',
            'strength': 'weak',
            'confidence': 0.5,
            'current_price': prices[-1] if len(prices) > 0 else 0.0,
            'momentum_quality': 'poor',
            'actionable': False
        }
        
        # Collect momentum signals
        momentum_signals = []
        
        # RSI contribution
        rsi_signal = rsi_analysis.get('signal', 'neutral')
        if rsi_signal in ['bullish', 'bearish']:
            momentum_signals.append(rsi_signal)
        
        # MACD contribution
        macd_signal = macd_analysis.get('macd_signal', 'neutral')
        if macd_signal in ['bullish', 'bearish']:
            momentum_signals.append(macd_signal)
        
        # Stochastic contribution
        stoch_signal = stoch_analysis.get('k_d_relationship', 'neutral')
        if stoch_signal in ['bullish', 'bearish']:
            momentum_signals.append(stoch_signal)
        
        # Determine consensus
        if momentum_signals:
            bullish_count = momentum_signals.count('bullish')
            bearish_count = momentum_signals.count('bearish')
            
            if bullish_count > bearish_count:
                overall_momentum['direction'] = 'bullish'
                overall_momentum['confidence'] = bullish_count / len(momentum_signals)
            elif bearish_count > bullish_count:
                overall_momentum['direction'] = 'bearish'
                overall_momentum['confidence'] = bearish_count / len(momentum_signals)
            else:
                overall_momentum['direction'] = 'neutral'
        
        # Assess strength
        if overall_momentum['confidence'] > 0.7:
            overall_momentum['strength'] = 'strong'
        elif overall_momentum['confidence'] > 0.5:
            overall_momentum['strength'] = 'moderate'
        else:
            overall_momentum['strength'] = 'weak'
        
        # Quality assessment
        overall_momentum['momentum_quality'] = self._assess_momentum_quality(rsi_analysis, macd_analysis, stoch_analysis)
        
        # Actionable assessment
        overall_momentum['actionable'] = (
            overall_momentum['strength'] in ['strong', 'moderate'] and
            overall_momentum['direction'] != 'neutral' and
            overall_momentum['confidence'] > 0.6
        )
        
        return overall_momentum
    
    def _identify_rsi_extremes(self, rsi: np.ndarray) -> List[Dict[str, Any]]:
        """Identify RSI extreme levels"""
        extremes = []
        
        if len(rsi) < 5:
            return extremes
        
        # Look for recent extremes
        recent_rsi = rsi[-10:] if len(rsi) >= 10 else rsi
        
        for i, value in enumerate(recent_rsi):
            if value >= 70:
                extremes.append({'type': 'overbought', 'value': value, 'recent_index': i})
            elif value <= 30:
                extremes.append({'type': 'oversold', 'value': value, 'recent_index': i})
        
        return extremes
    
    def _identify_macd_crossovers(self, macd: np.ndarray, macd_signal: np.ndarray) -> List[Dict[str, Any]]:
        """Identify MACD crossovers"""
        crossovers = []
        
        if len(macd) < 2 or len(macd_signal) < 2:
            return crossovers
        
        # Check for recent crossover
        current_above = macd[-1] > macd_signal[-1]
        previous_above = macd[-2] > macd_signal[-2]
        
        if current_above and not previous_above:
            crossovers.append({'type': 'bullish', 'description': 'MACD crossed above signal line'})
        elif not current_above and previous_above:
            crossovers.append({'type': 'bearish', 'description': 'MACD crossed below signal line'})
        
        return crossovers
    
    def _identify_stoch_crossovers(self, stoch_k: np.ndarray, stoch_d: np.ndarray) -> List[Dict[str, Any]]:
        """Identify Stochastic crossovers"""
        crossovers = []
        
        if len(stoch_k) < 2 or len(stoch_d) < 2:
            return crossovers
        
        # Check for recent crossover
        current_above = stoch_k[-1] > stoch_d[-1]
        previous_above = stoch_k[-2] > stoch_d[-2]
        
        if current_above and not previous_above:
            crossovers.append({'type': 'bullish', 'description': '%K crossed above %D'})
        elif not current_above and previous_above:
            crossovers.append({'type': 'bearish', 'description': '%K crossed below %D'})
        
        return crossovers
    
    def _detect_rsi_divergences(self, prices: np.ndarray, rsi: np.ndarray) -> List[Dict[str, Any]]:
        """Detect RSI divergences (simplified)"""
        divergences = []
        
        if len(prices) < 20 or len(rsi) < 20:
            return divergences
        
        # Simple divergence detection
        recent_price_high = np.max(prices[-10:])
        older_price_high = np.max(prices[-20:-10])
        recent_rsi_high = np.max(rsi[-10:])
        older_rsi_high = np.max(rsi[-20:-10])
        
        # Bearish divergence: higher price high, lower RSI high
        if recent_price_high > older_price_high and recent_rsi_high < older_rsi_high:
            divergences.append({
                'type': 'bearish',
                'description': 'Bearish RSI divergence: higher price highs, lower RSI highs'
            })
        
        # Similar check for bullish divergence with lows
        recent_price_low = np.min(prices[-10:])
        older_price_low = np.min(prices[-20:-10])
        recent_rsi_low = np.min(rsi[-10:])
        older_rsi_low = np.min(rsi[-20:-10])
        
        if recent_price_low < older_price_low and recent_rsi_low > older_rsi_low:
            divergences.append({
                'type': 'bullish',
                'description': 'Bullish RSI divergence: lower price lows, higher RSI lows'
            })
        
        return divergences
    
    def _detect_macd_divergences(self, prices: np.ndarray, macd: np.ndarray) -> List[Dict[str, Any]]:
        """Detect MACD divergences (simplified)"""
        divergences = []
        
        if len(prices) < 20 or len(macd) < 20:
            return divergences
        
        # Simple MACD divergence detection (similar to RSI)
        recent_price_high = np.max(prices[-10:])
        older_price_high = np.max(prices[-20:-10])
        recent_macd_high = np.max(macd[-10:])
        older_macd_high = np.max(macd[-20:-10])
        
        if recent_price_high > older_price_high and recent_macd_high < older_macd_high:
            divergences.append({
                'type': 'bearish',
                'description': 'Bearish MACD divergence detected'
            })
        
        return divergences
    
    def _identify_extreme_signals(self, rsi_analysis: Dict, stoch_analysis: Dict) -> List[Dict[str, Any]]:
        """Identify overbought/oversold signals"""
        extreme_signals = []
        
        # RSI extremes
        rsi_signal = rsi_analysis.get('signal', 'neutral')
        if rsi_signal in ['overbought', 'oversold']:
            extreme_signals.append({
                'indicator': 'RSI',
                'signal': rsi_signal,
                'value': rsi_analysis.get('current_rsi', 50),
                'confidence': 0.7
            })
        
        # Stochastic extremes
        stoch_signal = stoch_analysis.get('stoch_signal', 'neutral')
        if stoch_signal in ['overbought', 'oversold']:
            extreme_signals.append({
                'indicator': 'Stochastic',
                'signal': stoch_signal,
                'confidence': 0.6
            })
        
        return extreme_signals
    
    def _identify_crossover_signals(self, macd_analysis: Dict, stoch_analysis: Dict) -> List[Dict[str, Any]]:
        """Identify crossover signals"""
        crossover_signals = []
        
        # MACD crossovers
        macd_crossovers = macd_analysis.get('crossover_signals', [])
        crossover_signals.extend(macd_crossovers)
        
        # Stochastic crossovers
        stoch_crossovers = stoch_analysis.get('crossover_signals', [])
        crossover_signals.extend(stoch_crossovers)
        
        return crossover_signals
    
    def _calculate_confidence_score(self, rsi_analysis: Dict, macd_analysis: Dict, stoch_analysis: Dict) -> float:
        """Calculate overall confidence score for momentum analysis"""
        
        confidence_factors = []
        
        # RSI confidence
        rsi_signal = rsi_analysis.get('signal', 'neutral')
        if rsi_signal != 'neutral':
            # Higher confidence for extreme readings
            if rsi_signal in ['overbought', 'oversold']:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.6)
        
        # MACD confidence
        macd_signal = macd_analysis.get('macd_signal', 'neutral')
        if macd_signal != 'neutral':
            confidence_factors.append(0.7)
        
        # Stochastic confidence
        stoch_signal = stoch_analysis.get('stoch_signal', 'neutral')
        if stoch_signal != 'neutral':
            confidence_factors.append(0.6)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _assess_momentum_quality(self, rsi_analysis: Dict, macd_analysis: Dict, stoch_analysis: Dict) -> str:
        """Assess overall momentum quality"""
        
        quality_factors = 0
        total_factors = 0
        
        # RSI quality
        rsi_signal = rsi_analysis.get('signal', 'neutral')
        if rsi_signal in ['overbought', 'oversold']:
            quality_factors += 1
        total_factors += 1
        
        # MACD quality
        macd_crossovers = macd_analysis.get('crossover_signals', [])
        if macd_crossovers:
            quality_factors += 1
        total_factors += 1
        
        # Overall assessment
        quality_ratio = quality_factors / total_factors if total_factors > 0 else 0
        
        if quality_ratio >= 0.7:
            return 'excellent'
        elif quality_ratio >= 0.5:
            return 'good'
        elif quality_ratio >= 0.3:
            return 'fair'
        else:
            return 'poor'