"""
Reversal Patterns Processor

This module handles the analysis of reversal patterns including divergences, 
double tops/bottoms, and other reversal indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import json
import asyncio
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ReversalPattern:
    """Data structure for a reversal pattern"""
    pattern_type: str
    strength: str  # strong/medium/weak
    completion: float  # 0-100 percentage
    reliability: str  # high/medium/low
    entry_level: float
    stop_loss: float
    target_levels: List[float]
    confidence: float
    description: str

class ReversalPatternsProcessor:
    """
    Processor for analyzing reversal patterns in stock data.
    
    This processor specializes in:
    - Divergence analysis (bullish/bearish)
    - Double top/bottom patterns
    - Head and shoulders patterns
    - Other reversal indicators
    """
    
    def __init__(self):
        self.name = "reversal_patterns"
        self.description = "Analyzes trend reversal patterns and signals"
    
    async def analyze_async(self, stock_data: pd.DataFrame, indicators: Dict[str, Any] = None, 
                          context: str = "", chart_image: bytes = None) -> Dict[str, Any]:
        """
        Asynchronous analysis of reversal patterns
        
        Args:
            stock_data: OHLCV price data
            indicators: Technical indicators dictionary
            context: Additional context for analysis
            chart_image: Chart image for visual analysis
            
        Returns:
            Dictionary containing reversal pattern analysis results
        """
        try:
            # Start pattern detection
            start_time = pd.Timestamp.now()
            
            # Detect various reversal patterns
            divergences = self._detect_divergences(stock_data, indicators)
            double_patterns = self._detect_double_patterns(stock_data)
            other_reversals = self._detect_other_reversals(stock_data, indicators)
            
            # Calculate pattern confluence
            pattern_confluence = self._analyze_pattern_confluence(divergences, double_patterns, other_reversals)
            
            # Assess signal strength
            signal_strength = self._assess_signal_strength(divergences, double_patterns, other_reversals)
            
            # Calculate risk-reward ratios
            risk_reward = self._calculate_risk_reward(stock_data, divergences + double_patterns + other_reversals)
            
            # Determine primary reversal signal
            primary_signal = self._determine_primary_signal(divergences, double_patterns, other_reversals)
            
            processing_time = (pd.Timestamp.now() - start_time).total_seconds()
            
            # Build comprehensive analysis result
            analysis_result = {
                'agent_name': self.name,
                'processing_time': processing_time,
                'timestamp': pd.Timestamp.now().isoformat(),
                
                # Pattern Analysis
                'reversal_patterns': {
                    'divergences': [self._pattern_to_dict(d) for d in divergences],
                    'double_patterns': [self._pattern_to_dict(d) for d in double_patterns],
                    'other_reversals': [self._pattern_to_dict(o) for o in other_reversals]
                },
                
                # Signal Analysis
                'primary_signal': primary_signal,
                'signal_strength': signal_strength,
                'pattern_confluence': pattern_confluence,
                'risk_reward_ratio': risk_reward,
                
                # Entry/Exit Analysis
                'entry_points': self._identify_entry_points(divergences + double_patterns + other_reversals),
                'stop_loss_levels': self._calculate_stop_loss_levels(stock_data, divergences + double_patterns + other_reversals),
                'target_levels': self._calculate_target_levels(stock_data, divergences + double_patterns + other_reversals),
                
                # Risk Assessment
                'false_signal_risk': self._assess_false_signal_risk(divergences, double_patterns, other_reversals),
                'pattern_failure_risk': self._assess_pattern_failure_risk(stock_data, divergences + double_patterns + other_reversals),
                
                # Confidence Metrics
                'confidence_score': self._calculate_confidence_score(divergences, double_patterns, other_reversals, indicators),
                'reliability_assessment': self._assess_reliability(divergences + double_patterns + other_reversals),
                
                # Trading Recommendations
                'trading_strategy': self._generate_trading_strategy(primary_signal, divergences + double_patterns + other_reversals),
                'position_sizing': self._recommend_position_sizing(signal_strength, risk_reward),
                
                # Additional Context
                'market_context': self._analyze_market_context(stock_data, indicators),
                'volume_confirmation': self._check_volume_confirmation(stock_data, divergences + double_patterns + other_reversals)
            }
            
            logger.info(f"[REVERSAL_PATTERNS] Analysis completed in {processing_time:.2f}s")
            return analysis_result
            
        except Exception as e:
            logger.error(f"[REVERSAL_PATTERNS] Analysis failed: {str(e)}")
            return {
                'agent_name': self.name,
                'error': str(e),
                'success': False,
                'confidence_score': 0.0
            }
    
    def _detect_divergences(self, stock_data: pd.DataFrame, indicators: Dict[str, Any]) -> List[ReversalPattern]:
        """Detect bullish and bearish divergences"""
        divergences = []
        
        if not indicators or 'rsi' not in indicators:
            return divergences
        
        prices = stock_data['close'].values
        rsi = indicators['rsi']
        
        # Look for divergences in recent periods
        lookback = min(20, len(prices) - 1)
        
        for i in range(lookback, len(prices)):
            # Bullish divergence: price makes lower low, RSI makes higher low
            if i >= 2:
                price_window = prices[i-10:i+1] if i >= 10 else prices[:i+1]
                rsi_window = rsi[i-10:i+1] if i >= 10 else rsi[:i+1]
                
                if len(price_window) >= 3 and len(rsi_window) >= 3:
                    price_min_idx = np.argmin(price_window)
                    rsi_min_idx = np.argmin(rsi_window)
                    
                    # Check for bullish divergence
                    if (price_min_idx == len(price_window) - 1 and rsi_min_idx < len(rsi_window) - 1):
                        if rsi_window[-1] > rsi_window[rsi_min_idx]:
                            divergence = ReversalPattern(
                                pattern_type="bullish_divergence",
                                strength="medium",
                                completion=75.0,
                                reliability="medium",
                                entry_level=prices[i] * 1.02,
                                stop_loss=prices[i] * 0.97,
                                target_levels=[prices[i] * 1.05, prices[i] * 1.08],
                                confidence=0.65,
                                description=f"Bullish RSI divergence detected at {prices[i]:.2f}"
                            )
                            divergences.append(divergence)
        
        return divergences
    
    def _detect_double_patterns(self, stock_data: pd.DataFrame) -> List[ReversalPattern]:
        """Detect double top and double bottom patterns"""
        patterns = []
        prices = stock_data['close'].values
        highs = stock_data['high'].values
        lows = stock_data['low'].values
        
        if len(prices) < 20:
            return patterns
        
        # Look for double tops (resistance at similar levels)
        for i in range(10, len(highs) - 10):
            current_high = highs[i]
            
            # Look for another high at similar level
            for j in range(i + 5, min(i + 15, len(highs))):
                if abs(highs[j] - current_high) / current_high < 0.02:  # Within 2%
                    # Check if there's a valley between the peaks
                    valley_low = np.min(lows[i:j+1])
                    if (current_high - valley_low) / current_high > 0.03:  # At least 3% decline
                        pattern = ReversalPattern(
                            pattern_type="double_top",
                            strength="medium",
                            completion=80.0,
                            reliability="high",
                            entry_level=valley_low * 0.98,
                            stop_loss=current_high * 1.01,
                            target_levels=[valley_low * 0.95, valley_low * 0.92],
                            confidence=0.70,
                            description=f"Double top pattern at {current_high:.2f}"
                        )
                        patterns.append(pattern)
                        break
        
        # Look for double bottoms (support at similar levels)
        for i in range(10, len(lows) - 10):
            current_low = lows[i]
            
            # Look for another low at similar level
            for j in range(i + 5, min(i + 15, len(lows))):
                if abs(lows[j] - current_low) / current_low < 0.02:  # Within 2%
                    # Check if there's a peak between the valleys
                    peak_high = np.max(highs[i:j+1])
                    if (peak_high - current_low) / current_low > 0.03:  # At least 3% rally
                        pattern = ReversalPattern(
                            pattern_type="double_bottom",
                            strength="medium",
                            completion=80.0,
                            reliability="high",
                            entry_level=peak_high * 1.02,
                            stop_loss=current_low * 0.99,
                            target_levels=[peak_high * 1.05, peak_high * 1.08],
                            confidence=0.70,
                            description=f"Double bottom pattern at {current_low:.2f}"
                        )
                        patterns.append(pattern)
                        break
        
        return patterns
    
    def _detect_other_reversals(self, stock_data: pd.DataFrame, indicators: Dict[str, Any]) -> List[ReversalPattern]:
        """Detect other reversal patterns like head and shoulders"""
        patterns = []
        
        # Simple head and shoulders detection (simplified version)
        highs = stock_data['high'].values
        lows = stock_data['low'].values
        
        if len(highs) < 15:
            return patterns
        
        # Look for potential head and shoulders patterns
        for i in range(5, len(highs) - 10):
            left_shoulder = highs[i-5:i]
            head = highs[i:i+3]
            right_shoulder = highs[i+3:i+8]
            
            if len(left_shoulder) > 0 and len(head) > 0 and len(right_shoulder) > 0:
                left_peak = np.max(left_shoulder)
                head_peak = np.max(head)
                right_peak = np.max(right_shoulder)
                
                # Check if head is higher than shoulders
                if (head_peak > left_peak * 1.02 and head_peak > right_peak * 1.02 and
                    abs(left_peak - right_peak) / left_peak < 0.05):
                    
                    # Find neckline (support level)
                    neckline = np.min(lows[i-5:i+8])
                    
                    pattern = ReversalPattern(
                        pattern_type="head_and_shoulders",
                        strength="strong",
                        completion=85.0,
                        reliability="high",
                        entry_level=neckline * 0.98,
                        stop_loss=head_peak * 1.01,
                        target_levels=[neckline * 0.95, neckline * 0.90],
                        confidence=0.75,
                        description=f"Head and shoulders pattern, neckline at {neckline:.2f}"
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _analyze_pattern_confluence(self, divergences: List[ReversalPattern], 
                                  double_patterns: List[ReversalPattern], 
                                  other_reversals: List[ReversalPattern]) -> Dict[str, Any]:
        """Analyze confluence between different reversal patterns"""
        all_patterns = divergences + double_patterns + other_reversals
        
        if not all_patterns:
            return {'has_confluence': False, 'confluence_strength': 'none'}
        
        # Check for patterns at similar price levels
        bullish_patterns = [p for p in all_patterns if 'bullish' in p.pattern_type or 'bottom' in p.pattern_type]
        bearish_patterns = [p for p in all_patterns if 'bearish' in p.pattern_type or 'top' in p.pattern_type or 'head' in p.pattern_type]
        
        confluence = {
            'has_confluence': len(bullish_patterns) > 1 or len(bearish_patterns) > 1,
            'bullish_confluence': len(bullish_patterns),
            'bearish_confluence': len(bearish_patterns),
            'confluence_strength': 'strong' if max(len(bullish_patterns), len(bearish_patterns)) >= 2 else 'weak',
            'dominant_direction': 'bullish' if len(bullish_patterns) > len(bearish_patterns) else 'bearish'
        }
        
        return confluence
    
    def _assess_signal_strength(self, divergences: List[ReversalPattern], 
                              double_patterns: List[ReversalPattern], 
                              other_reversals: List[ReversalPattern]) -> str:
        """Assess overall signal strength"""
        all_patterns = divergences + double_patterns + other_reversals
        
        if not all_patterns:
            return "weak"
        
        strong_patterns = [p for p in all_patterns if p.strength == "strong"]
        medium_patterns = [p for p in all_patterns if p.strength == "medium"]
        
        if len(strong_patterns) >= 1:
            return "strong"
        elif len(medium_patterns) >= 2:
            return "medium"
        else:
            return "weak"
    
    def _calculate_risk_reward(self, stock_data: pd.DataFrame, patterns: List[ReversalPattern]) -> float:
        """Calculate risk-reward ratio for reversal patterns"""
        if not patterns:
            return 0.0
        
        current_price = stock_data['close'].iloc[-1]
        
        # Average risk-reward from all patterns
        risk_rewards = []
        for pattern in patterns:
            if pattern.target_levels and pattern.stop_loss:
                avg_target = np.mean(pattern.target_levels)
                if 'bullish' in pattern.pattern_type or 'bottom' in pattern.pattern_type:
                    reward = abs(avg_target - current_price)
                    risk = abs(current_price - pattern.stop_loss)
                else:
                    reward = abs(current_price - avg_target)
                    risk = abs(pattern.stop_loss - current_price)
                
                if risk > 0:
                    risk_rewards.append(reward / risk)
        
        return np.mean(risk_rewards) if risk_rewards else 1.0
    
    def _determine_primary_signal(self, divergences: List[ReversalPattern], 
                                double_patterns: List[ReversalPattern], 
                                other_reversals: List[ReversalPattern]) -> str:
        """Determine the primary reversal signal"""
        all_patterns = divergences + double_patterns + other_reversals
        
        if not all_patterns:
            return "neutral"
        
        # Count bullish vs bearish signals
        bullish_count = len([p for p in all_patterns if 'bullish' in p.pattern_type or 'bottom' in p.pattern_type])
        bearish_count = len([p for p in all_patterns if 'bearish' in p.pattern_type or 'top' in p.pattern_type or 'head' in p.pattern_type])
        
        if bullish_count > bearish_count:
            return "bullish_reversal"
        elif bearish_count > bullish_count:
            return "bearish_reversal"
        else:
            return "mixed_signals"
    
    def _pattern_to_dict(self, pattern: ReversalPattern) -> Dict[str, Any]:
        """Convert ReversalPattern to dictionary"""
        return {
            'type': pattern.pattern_type,
            'strength': pattern.strength,
            'completion': pattern.completion,
            'reliability': pattern.reliability,
            'entry_level': pattern.entry_level,
            'stop_loss': pattern.stop_loss,
            'target_levels': pattern.target_levels,
            'confidence': pattern.confidence,
            'description': pattern.description
        }
    
    def _identify_entry_points(self, patterns: List[ReversalPattern]) -> List[float]:
        """Identify optimal entry points from patterns"""
        return [p.entry_level for p in patterns if p.entry_level]
    
    def _calculate_stop_loss_levels(self, stock_data: pd.DataFrame, patterns: List[ReversalPattern]) -> List[float]:
        """Calculate stop loss levels"""
        return [p.stop_loss for p in patterns if p.stop_loss]
    
    def _calculate_target_levels(self, stock_data: pd.DataFrame, patterns: List[ReversalPattern]) -> List[float]:
        """Calculate target levels"""
        all_targets = []
        for pattern in patterns:
            if pattern.target_levels:
                all_targets.extend(pattern.target_levels)
        return all_targets
    
    def _assess_false_signal_risk(self, divergences: List[ReversalPattern], 
                                double_patterns: List[ReversalPattern], 
                                other_reversals: List[ReversalPattern]) -> str:
        """Assess risk of false signals"""
        all_patterns = divergences + double_patterns + other_reversals
        
        if not all_patterns:
            return "high"
        
        high_reliability = len([p for p in all_patterns if p.reliability == "high"])
        total_patterns = len(all_patterns)
        
        reliability_ratio = high_reliability / total_patterns
        
        if reliability_ratio > 0.7:
            return "low"
        elif reliability_ratio > 0.4:
            return "medium"
        else:
            return "high"
    
    def _assess_pattern_failure_risk(self, stock_data: pd.DataFrame, patterns: List[ReversalPattern]) -> str:
        """Assess risk of pattern failure"""
        if not patterns:
            return "high"
        
        # Assess based on completion percentage and strength
        avg_completion = np.mean([p.completion for p in patterns])
        strong_patterns = len([p for p in patterns if p.strength == "strong"])
        
        if avg_completion > 80 and strong_patterns > 0:
            return "low"
        elif avg_completion > 60:
            return "medium"
        else:
            return "high"
    
    def _calculate_confidence_score(self, divergences: List[ReversalPattern], 
                                  double_patterns: List[ReversalPattern], 
                                  other_reversals: List[ReversalPattern], 
                                  indicators: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        all_patterns = divergences + double_patterns + other_reversals
        
        if not all_patterns:
            return 0.0
        
        # Base confidence on pattern quality
        pattern_confidence = np.mean([p.confidence for p in all_patterns])
        
        # Adjust for number of patterns (confluence)
        pattern_count_bonus = min(0.2, len(all_patterns) * 0.05)
        
        # Adjust for pattern strength
        strong_patterns = len([p for p in all_patterns if p.strength == "strong"])
        strength_bonus = strong_patterns * 0.1
        
        total_confidence = pattern_confidence + pattern_count_bonus + strength_bonus
        return min(1.0, total_confidence)
    
    def _assess_reliability(self, patterns: List[ReversalPattern]) -> str:
        """Assess overall reliability of patterns"""
        if not patterns:
            return "low"
        
        high_reliability = len([p for p in patterns if p.reliability == "high"])
        total_patterns = len(patterns)
        
        reliability_ratio = high_reliability / total_patterns
        
        if reliability_ratio > 0.6:
            return "high"
        elif reliability_ratio > 0.3:
            return "medium"
        else:
            return "low"
    
    def _generate_trading_strategy(self, primary_signal: str, patterns: List[ReversalPattern]) -> Dict[str, Any]:
        """Generate trading strategy based on reversal patterns"""
        if not patterns:
            return {'action': 'hold', 'reasoning': 'No clear reversal patterns identified'}
        
        strategy = {
            'primary_action': 'buy' if 'bullish' in primary_signal else 'sell' if 'bearish' in primary_signal else 'hold',
            'entry_strategy': 'breakout' if primary_signal != 'neutral' else 'wait',
            'position_type': 'reversal',
            'time_horizon': 'short_to_medium',
            'patterns_supporting': len(patterns),
            'reasoning': f"Based on {len(patterns)} reversal pattern(s) with {primary_signal} bias"
        }
        
        return strategy
    
    def _recommend_position_sizing(self, signal_strength: str, risk_reward: float) -> Dict[str, Any]:
        """Recommend position sizing based on signal strength and risk-reward"""
        if signal_strength == "strong" and risk_reward > 2.0:
            size = "large"
        elif signal_strength == "medium" and risk_reward > 1.5:
            size = "medium"
        else:
            size = "small"
        
        return {
            'recommended_size': size,
            'risk_reward_ratio': risk_reward,
            'signal_strength': signal_strength,
            'reasoning': f"Position sizing based on {signal_strength} signal strength and {risk_reward:.2f} risk-reward ratio"
        }
    
    def _analyze_market_context(self, stock_data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze broader market context for reversal patterns"""
        current_price = stock_data['close'].iloc[-1]
        recent_prices = stock_data['close'].tail(20)
        
        context = {
            'trend_direction': 'up' if recent_prices.iloc[-1] > recent_prices.iloc[0] else 'down',
            'volatility': recent_prices.std() / recent_prices.mean(),
            'price_position': 'high' if current_price > recent_prices.quantile(0.8) else 'low' if current_price < recent_prices.quantile(0.2) else 'middle'
        }
        
        return context
    
    def _check_volume_confirmation(self, stock_data: pd.DataFrame, patterns: List[ReversalPattern]) -> Dict[str, Any]:
        """Check volume confirmation for reversal patterns"""
        if 'volume' not in stock_data.columns:
            return {'volume_confirmation': 'unavailable'}
        
        recent_volume = stock_data['volume'].tail(10).mean()
        avg_volume = stock_data['volume'].mean()
        
        volume_confirmation = {
            'has_volume_support': recent_volume > avg_volume * 1.2,
            'volume_trend': 'increasing' if recent_volume > avg_volume else 'decreasing',
            'confirmation_strength': 'strong' if recent_volume > avg_volume * 1.5 else 'weak'
        }
        
        return volume_confirmation