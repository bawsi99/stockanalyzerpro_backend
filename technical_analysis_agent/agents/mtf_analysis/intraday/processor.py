#!/usr/bin/env python3
"""
Intraday Multi-Timeframe Analysis Agent

Analyzes short-term intraday timeframes (1min, 5min, 15min) for scalping
and short-term trading opportunities with high-frequency signal validation.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class IntradayMTFProcessor:
    """Processor for intraday multi-timeframe analysis"""
    
    def __init__(self):
        self.agent_name = "intraday_mtf"
        self.timeframes = ['1min', '5min', '15min']
        self.primary_timeframe = '5min'  # Primary focus for signal generation
        self.timeframe_weights = {
            '1min': 0.2,   # Lower weight due to noise
            '5min': 0.5,   # Primary weight for signals
            '15min': 0.3   # Confirmation weight
        }
    
    async def analyze_async(self, mtf_data: Dict[str, pd.DataFrame], indicators: Dict, context: str = "") -> Dict:
        """
        Perform comprehensive intraday multi-timeframe analysis
        
        Args:
            mtf_data: Dictionary of timeframe -> DataFrame mappings
            indicators: Technical indicators dictionary  
            context: Additional context for analysis
            
        Returns:
            Dictionary containing intraday MTF analysis
        """
        try:
            logger.info(f"[INTRADAY_MTF] Starting intraday multi-timeframe analysis...")
            start_time = datetime.now()
            
            # Filter for intraday timeframes
            intraday_data = {tf: data for tf, data in mtf_data.items() if tf in self.timeframes}
            
            if not intraday_data:
                return {
                    'agent_name': self.agent_name,
                    'error': 'No intraday timeframe data available',
                    'confidence_score': 0.0,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            
            # Analyze each intraday timeframe
            timeframe_analysis = {}
            for timeframe in self.timeframes:
                if timeframe in intraday_data:
                    timeframe_analysis[timeframe] = await self._analyze_timeframe(
                        intraday_data[timeframe], timeframe, indicators
                    )
            
            # Cross-timeframe validation for intraday
            validation = await self._validate_intraday_signals(timeframe_analysis)
            
            # Generate scalping opportunities
            scalping_signals = await self._generate_scalping_signals(timeframe_analysis, intraday_data)
            
            # Determine entry/exit timing
            timing_analysis = await self._analyze_entry_exit_timing(timeframe_analysis)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_confidence(timeframe_analysis, validation)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"[INTRADAY_MTF] Analysis completed in {processing_time:.2f}s")
            
            return {
                'agent_name': self.agent_name,
                'analysis_timestamp': datetime.now().isoformat(),
                'processing_time': processing_time,
                'context': context,
                
                # Timeframe Analysis
                'timeframe_analysis': timeframe_analysis,
                'cross_timeframe_validation': validation,
                
                # Trading Opportunities
                'scalping_signals': scalping_signals,
                'entry_exit_timing': timing_analysis,
                
                # Overall Assessment
                'dominant_timeframe': self._get_dominant_timeframe(timeframe_analysis),
                'signal_strength': validation.get('signal_strength', 0.0),
                'trend_alignment': validation.get('trend_alignment', 'neutral'),
                'confidence_score': overall_confidence,
                
                # Intraday Specific
                'noise_level': self._calculate_noise_level(timeframe_analysis),
                'momentum_shifts': self._detect_momentum_shifts(timeframe_analysis),
                'breakout_potential': self._assess_breakout_potential(timeframe_analysis),
                
                # Trading Recommendations
                'trading_style': self._recommend_trading_style(timeframe_analysis),
                'optimal_timeframe': self._get_optimal_trading_timeframe(timeframe_analysis),
                'risk_considerations': self._get_intraday_risks(timeframe_analysis)
            }
            
        except Exception as e:
            logger.error(f"[INTRADAY_MTF] Analysis failed: {str(e)}")
            return {
                'agent_name': self.agent_name,
                'error': str(e),
                'confidence_score': 0.0,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    async def _analyze_timeframe(self, data: pd.DataFrame, timeframe: str, indicators: Dict) -> Dict:
        """Analyze a single intraday timeframe"""
        
        if len(data) < 20:  # Minimum data requirement
            return {
                'timeframe': timeframe,
                'status': 'insufficient_data',
                'trend': 'unknown',
                'strength': 'unknown',
                'signals': []
            }
        
        # Basic trend analysis
        sma_20 = data['close'].rolling(window=20, min_periods=10).mean()
        current_price = data['close'].iloc[-1]
        trend_direction = 'bullish' if current_price > sma_20.iloc[-1] else 'bearish'
        
        # Momentum analysis
        returns = data['close'].pct_change()
        momentum = returns.rolling(window=10, min_periods=5).mean().iloc[-1]
        momentum_strength = 'strong' if abs(momentum) > 0.002 else 'moderate' if abs(momentum) > 0.001 else 'weak'
        
        # Volume analysis
        avg_volume = data['volume'].rolling(window=20, min_periods=10).mean().iloc[-1]
        current_volume = data['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Support/Resistance levels
        highs = data['high'].rolling(window=20, min_periods=10).max()
        lows = data['low'].rolling(window=20, min_periods=10).min()
        
        # Pattern detection (simplified)
        patterns = []
        if len(data) >= 30:
            # Simple consolidation detection
            price_range = (data['high'].iloc[-10:].max() - data['low'].iloc[-10:].min()) / current_price
            if price_range < 0.02:  # 2% range
                patterns.append('consolidation')
            
            # Simple breakout detection
            if current_price > highs.iloc[-2]:
                patterns.append('upward_breakout')
            elif current_price < lows.iloc[-2]:
                patterns.append('downward_breakout')
        
        return {
            'timeframe': timeframe,
            'status': 'analyzed',
            'data_points': len(data),
            'trend': trend_direction,
            'strength': momentum_strength,
            'momentum': float(momentum),
            'volume_ratio': float(volume_ratio),
            'current_price': float(current_price),
            'sma_20': float(sma_20.iloc[-1]),
            'resistance_level': float(highs.iloc[-1]),
            'support_level': float(lows.iloc[-1]),
            'patterns': patterns,
            'signals': self._generate_timeframe_signals(data, timeframe),
            'confidence': min(0.9, len(data) / 100)  # Higher data = higher confidence
        }
    
    def _generate_timeframe_signals(self, data: pd.DataFrame, timeframe: str) -> List[Dict]:
        """Generate trading signals for a specific timeframe"""
        
        signals = []
        
        if len(data) < 20:
            return signals
        
        current_price = data['close'].iloc[-1]
        sma_20 = data['close'].rolling(window=20, min_periods=10).mean().iloc[-1]
        
        # Moving average crossover signals
        if current_price > sma_20:
            signals.append({
                'type': 'trend_following',
                'direction': 'bullish',
                'strength': 0.6,
                'description': f'Price above SMA20 on {timeframe}'
            })
        else:
            signals.append({
                'type': 'trend_following', 
                'direction': 'bearish',
                'strength': 0.6,
                'description': f'Price below SMA20 on {timeframe}'
            })
        
        # Volume confirmation
        avg_volume = data['volume'].rolling(window=20, min_periods=10).mean().iloc[-1]
        if data['volume'].iloc[-1] > avg_volume * 1.5:
            signals.append({
                'type': 'volume_confirmation',
                'direction': 'bullish' if current_price > sma_20 else 'bearish',
                'strength': 0.7,
                'description': f'High volume confirmation on {timeframe}'
            })
        
        return signals
    
    async def _validate_intraday_signals(self, timeframe_analysis: Dict) -> Dict:
        """Validate signals across intraday timeframes"""
        
        analyzed_timeframes = [tf for tf in timeframe_analysis.values() if tf.get('status') == 'analyzed']
        
        if not analyzed_timeframes:
            return {
                'trend_alignment': 'unknown',
                'signal_strength': 0.0,
                'supporting_timeframes': [],
                'conflicting_timeframes': [],
                'consensus_trend': 'neutral'
            }
        
        # Count trend directions
        bullish_count = sum(1 for tf in analyzed_timeframes if tf.get('trend') == 'bullish')
        bearish_count = sum(1 for tf in analyzed_timeframes if tf.get('trend') == 'bearish')
        total_count = len(analyzed_timeframes)
        
        # Determine consensus
        if bullish_count >= total_count * 0.6:
            consensus_trend = 'bullish'
            trend_alignment = 'aligned'
        elif bearish_count >= total_count * 0.6:
            consensus_trend = 'bearish'
            trend_alignment = 'aligned'
        else:
            consensus_trend = 'neutral'
            trend_alignment = 'conflicting'
        
        # Calculate signal strength
        signal_strength = max(bullish_count, bearish_count) / total_count if total_count > 0 else 0.0
        
        # Supporting and conflicting timeframes
        supporting_timeframes = []
        conflicting_timeframes = []
        
        for tf_name, tf_data in timeframe_analysis.items():
            if tf_data.get('status') == 'analyzed':
                tf_trend = tf_data.get('trend', 'neutral')
                if tf_trend == consensus_trend:
                    supporting_timeframes.append(tf_name)
                elif tf_trend != 'neutral' and consensus_trend != 'neutral':
                    conflicting_timeframes.append(tf_name)
        
        return {
            'trend_alignment': trend_alignment,
            'signal_strength': signal_strength,
            'supporting_timeframes': supporting_timeframes,
            'conflicting_timeframes': conflicting_timeframes,
            'consensus_trend': consensus_trend,
            'total_analyzed_timeframes': total_count
        }
    
    async def _generate_scalping_signals(self, timeframe_analysis: Dict, data: Dict[str, pd.DataFrame]) -> Dict:
        """Generate scalping-specific signals"""
        
        scalping_signals = {
            'quick_moves': [],
            'momentum_plays': [],
            'breakout_trades': [],
            'mean_reversion': []
        }
        
        # Focus on 1min and 5min for scalping
        for tf in ['1min', '5min']:
            if tf in timeframe_analysis and timeframe_analysis[tf].get('status') == 'analyzed':
                tf_data = timeframe_analysis[tf]
                
                # Quick momentum moves
                if abs(tf_data.get('momentum', 0)) > 0.003:  # Strong momentum
                    scalping_signals['momentum_plays'].append({
                        'timeframe': tf,
                        'direction': 'bullish' if tf_data.get('momentum', 0) > 0 else 'bearish',
                        'strength': min(abs(tf_data.get('momentum', 0)) * 100, 1.0),
                        'entry_type': 'momentum_follow'
                    })
                
                # Breakout detection
                patterns = tf_data.get('patterns', [])
                for pattern in patterns:
                    if 'breakout' in pattern:
                        scalping_signals['breakout_trades'].append({
                            'timeframe': tf,
                            'pattern': pattern,
                            'direction': 'bullish' if 'upward' in pattern else 'bearish',
                            'strength': 0.8
                        })
        
        return scalping_signals
    
    async def _analyze_entry_exit_timing(self, timeframe_analysis: Dict) -> Dict:
        """Analyze optimal entry and exit timing for intraday trades"""
        
        # Default timing analysis
        timing = {
            'optimal_entry_timeframe': '5min',
            'optimal_exit_timeframe': '1min',
            'entry_conditions': [],
            'exit_conditions': [],
            'timing_confidence': 0.5
        }
        
        # Determine best timeframe for entries based on signal strength
        best_entry_tf = '5min'
        best_signal_strength = 0.0
        
        for tf_name, tf_data in timeframe_analysis.items():
            if tf_data.get('status') == 'analyzed':
                tf_confidence = tf_data.get('confidence', 0.0)
                if tf_confidence > best_signal_strength:
                    best_signal_strength = tf_confidence
                    best_entry_tf = tf_name
        
        timing['optimal_entry_timeframe'] = best_entry_tf
        timing['timing_confidence'] = best_signal_strength
        
        # Entry conditions
        if best_entry_tf in timeframe_analysis:
            tf_data = timeframe_analysis[best_entry_tf]
            if tf_data.get('trend') == 'bullish':
                timing['entry_conditions'] = [
                    f"Price breaks above resistance on {best_entry_tf}",
                    "Volume confirms the move",
                    "Lower timeframe shows momentum"
                ]
            elif tf_data.get('trend') == 'bearish':
                timing['entry_conditions'] = [
                    f"Price breaks below support on {best_entry_tf}",
                    "Volume confirms the move", 
                    "Lower timeframe shows momentum"
                ]
        
        return timing
    
    def _calculate_confidence(self, timeframe_analysis: Dict, validation: Dict) -> float:
        """Calculate overall confidence score"""
        
        analyzed_count = sum(1 for tf in timeframe_analysis.values() if tf.get('status') == 'analyzed')
        total_timeframes = len(self.timeframes)
        
        # Base confidence from data availability
        data_confidence = analyzed_count / total_timeframes
        
        # Signal alignment confidence
        signal_strength = validation.get('signal_strength', 0.0)
        
        # Weighted confidence
        overall_confidence = (data_confidence * 0.4) + (signal_strength * 0.6)
        
        return float(overall_confidence)
    
    def _get_dominant_timeframe(self, timeframe_analysis: Dict) -> str:
        """Get the timeframe with the strongest signals"""
        
        best_tf = '5min'  # Default
        best_score = 0.0
        
        for tf_name, tf_data in timeframe_analysis.items():
            if tf_data.get('status') == 'analyzed':
                # Score based on confidence and number of signals
                score = tf_data.get('confidence', 0.0) * len(tf_data.get('signals', []))
                if score > best_score:
                    best_score = score
                    best_tf = tf_name
        
        return best_tf
    
    def _calculate_noise_level(self, timeframe_analysis: Dict) -> str:
        """Calculate market noise level for intraday trading"""
        
        noise_scores = []
        
        for tf_name, tf_data in timeframe_analysis.items():
            if tf_data.get('status') == 'analyzed' and 'momentum' in tf_data:
                # Higher momentum volatility = more noise
                momentum_vol = abs(tf_data.get('momentum', 0))
                if tf_name == '1min':
                    noise_scores.append(momentum_vol * 2)  # 1min is noisier
                else:
                    noise_scores.append(momentum_vol)
        
        if noise_scores:
            avg_noise = sum(noise_scores) / len(noise_scores)
            if avg_noise > 0.004:
                return 'high'
            elif avg_noise > 0.002:
                return 'moderate'
            else:
                return 'low'
        
        return 'unknown'
    
    def _detect_momentum_shifts(self, timeframe_analysis: Dict) -> List[Dict]:
        """Detect momentum shifts across intraday timeframes"""
        
        shifts = []
        
        # Compare momentum between timeframes
        timeframes = ['1min', '5min', '15min']
        momentums = {}
        
        for tf in timeframes:
            if tf in timeframe_analysis and timeframe_analysis[tf].get('status') == 'analyzed':
                momentums[tf] = timeframe_analysis[tf].get('momentum', 0)
        
        # Detect shifts
        if '1min' in momentums and '5min' in momentums:
            if momentums['1min'] * momentums['5min'] < 0:  # Opposite directions
                shifts.append({
                    'type': 'direction_shift',
                    'timeframes': ['1min', '5min'],
                    'description': 'Short-term momentum diverging from medium-term'
                })
        
        return shifts
    
    def _assess_breakout_potential(self, timeframe_analysis: Dict) -> Dict:
        """Assess potential for breakouts across timeframes"""
        
        breakout_assessment = {
            'potential': 'low',
            'direction': 'neutral',
            'timeframes_supporting': [],
            'confidence': 0.0
        }
        
        # Count consolidation patterns
        consolidation_count = 0
        breakout_patterns = []
        
        for tf_name, tf_data in timeframe_analysis.items():
            if tf_data.get('status') == 'analyzed':
                patterns = tf_data.get('patterns', [])
                if 'consolidation' in patterns:
                    consolidation_count += 1
                
                for pattern in patterns:
                    if 'breakout' in pattern:
                        breakout_patterns.append((tf_name, pattern))
        
        # High consolidation + some breakouts = high potential
        if consolidation_count >= 2 and breakout_patterns:
            breakout_assessment['potential'] = 'high'
            breakout_assessment['timeframes_supporting'] = [tf for tf, _ in breakout_patterns]
            breakout_assessment['confidence'] = 0.8
            
            # Determine direction from breakout patterns
            upward_breaks = sum(1 for _, pattern in breakout_patterns if 'upward' in pattern)
            downward_breaks = sum(1 for _, pattern in breakout_patterns if 'downward' in pattern)
            
            if upward_breaks > downward_breaks:
                breakout_assessment['direction'] = 'bullish'
            elif downward_breaks > upward_breaks:
                breakout_assessment['direction'] = 'bearish'
        
        return breakout_assessment
    
    def _recommend_trading_style(self, timeframe_analysis: Dict) -> str:
        """Recommend optimal trading style based on intraday analysis"""
        
        # Count signal strengths and patterns
        strong_signals = 0
        momentum_plays = 0
        
        for tf_data in timeframe_analysis.values():
            if tf_data.get('status') == 'analyzed':
                signals = tf_data.get('signals', [])
                for signal in signals:
                    if signal.get('strength', 0) > 0.7:
                        strong_signals += 1
                    if signal.get('type') == 'momentum':
                        momentum_plays += 1
        
        if strong_signals >= 2:
            return 'aggressive_scalping'
        elif momentum_plays >= 1:
            return 'momentum_trading'
        else:
            return 'conservative_scalping'
    
    def _get_optimal_trading_timeframe(self, timeframe_analysis: Dict) -> str:
        """Get the optimal timeframe for trading execution"""
        
        # Score each timeframe
        scores = {}
        
        for tf_name, tf_data in timeframe_analysis.items():
            if tf_data.get('status') == 'analyzed':
                score = 0
                
                # Base score from confidence
                score += tf_data.get('confidence', 0) * 50
                
                # Bonus for signal strength
                signals = tf_data.get('signals', [])
                avg_signal_strength = sum(s.get('strength', 0) for s in signals) / len(signals) if signals else 0
                score += avg_signal_strength * 30
                
                # Timeframe preference (5min is often optimal for intraday)
                if tf_name == '5min':
                    score += 20
                
                scores[tf_name] = score
        
        if scores:
            return max(scores, key=scores.get)
        
        return '5min'  # Default
    
    def _get_intraday_risks(self, timeframe_analysis: Dict) -> List[str]:
        """Identify intraday-specific risks"""
        
        risks = []
        
        # Check for conflicting signals
        trends = [tf.get('trend') for tf in timeframe_analysis.values() if tf.get('status') == 'analyzed']
        if len(set(trends)) > 1:
            risks.append("Conflicting trends across intraday timeframes")
        
        # Check for high noise
        if self._calculate_noise_level(timeframe_analysis) == 'high':
            risks.append("High market noise may cause false signals")
        
        # Check data quality
        insufficient_data = sum(1 for tf in timeframe_analysis.values() if tf.get('status') == 'insufficient_data')
        if insufficient_data > 0:
            risks.append(f"Insufficient data for {insufficient_data} timeframes")
        
        # Volume considerations
        low_volume_count = 0
        for tf_data in timeframe_analysis.values():
            if tf_data.get('status') == 'analyzed' and tf_data.get('volume_ratio', 1.0) < 0.5:
                low_volume_count += 1
        
        if low_volume_count >= 2:
            risks.append("Low volume may affect execution quality")
        
        return risks[:5]  # Return top 5 risks