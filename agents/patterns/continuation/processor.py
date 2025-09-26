"""
Continuation Patterns Processor

This module handles the analysis of continuation patterns including triangles,
flags, pennants, channels, and support/resistance levels.
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
class ContinuationPattern:
    """Data structure for a continuation pattern"""
    pattern_type: str
    strength: str  # strong/medium/weak
    completion: float  # 0-100 percentage
    reliability: str  # high/medium/low
    breakout_level: float
    stop_loss: float
    target_levels: List[float]
    confidence: float
    description: str
    direction: str  # bullish/bearish/neutral

class ContinuationPatternsProcessor:
    """
    Processor for analyzing continuation patterns in stock data.
    
    This processor specializes in:
    - Triangle patterns (ascending, descending, symmetrical)
    - Flag and pennant patterns
    - Channel patterns
    - Support and resistance level analysis
    """
    
    def __init__(self):
        self.name = "continuation_patterns"
        self.description = "Analyzes trend continuation patterns and key levels"
    
    async def analyze_async(self, stock_data: pd.DataFrame, indicators: Dict[str, Any] = None, 
                          context: str = "", chart_image: bytes = None) -> Dict[str, Any]:
        """
        Asynchronous analysis of continuation patterns
        
        Args:
            stock_data: OHLCV price data
            indicators: Technical indicators dictionary
            context: Additional context for analysis
            chart_image: Chart image for visual analysis
            
        Returns:
            Dictionary containing continuation pattern analysis results
        """
        try:
            start_time = pd.Timestamp.now()
            
            # Detect various continuation patterns
            triangles = self._detect_triangle_patterns(stock_data)
            flags_pennants = self._detect_flags_pennants(stock_data)
            channels = self._detect_channel_patterns(stock_data)
            key_levels = self._identify_key_levels(stock_data, indicators)
            
            # Analyze pattern confluence
            pattern_confluence = self._analyze_pattern_confluence(triangles, flags_pennants, channels)
            
            # Assess breakout potential
            breakout_potential = self._assess_breakout_potential(stock_data, triangles + flags_pennants + channels)
            
            # Calculate risk-reward ratios
            risk_reward = self._calculate_risk_reward(stock_data, triangles + flags_pennants + channels)
            
            # Determine primary continuation signal
            primary_signal = self._determine_primary_signal(triangles, flags_pennants, channels)
            
            processing_time = (pd.Timestamp.now() - start_time).total_seconds()
            
            # Build comprehensive analysis result
            analysis_result = {
                'agent_name': self.name,
                'processing_time': processing_time,
                'timestamp': pd.Timestamp.now().isoformat(),
                
                # Pattern Analysis
                'continuation_patterns': {
                    'triangles': [self._pattern_to_dict(t) for t in triangles],
                    'flags_pennants': [self._pattern_to_dict(f) for f in flags_pennants],
                    'channels': [self._pattern_to_dict(c) for c in channels]
                },
                
                # Key Levels Analysis
                'key_levels': key_levels,
                
                # Signal Analysis
                'primary_signal': primary_signal,
                'pattern_confluence': pattern_confluence,
                'breakout_potential': breakout_potential,
                'risk_reward_ratio': risk_reward,
                
                # Entry/Exit Analysis
                'breakout_levels': self._identify_breakout_levels(triangles + flags_pennants + channels),
                'entry_points': self._calculate_entry_points(stock_data, triangles + flags_pennants + channels),
                'stop_loss_levels': self._calculate_stop_loss_levels(stock_data, triangles + flags_pennants + channels),
                'target_levels': self._calculate_target_levels(stock_data, triangles + flags_pennants + channels),
                
                # Risk Assessment
                'pattern_failure_risk': self._assess_pattern_failure_risk(stock_data, triangles + flags_pennants + channels),
                'false_breakout_risk': self._assess_false_breakout_risk(triangles + flags_pennants + channels),
                
                # Confidence Metrics
                'confidence_score': self._calculate_confidence_score(triangles, flags_pennants, channels, indicators),
                'reliability_assessment': self._assess_reliability(triangles + flags_pennants + channels),
                
                # Trading Recommendations
                'trading_strategy': self._generate_trading_strategy(primary_signal, triangles + flags_pennants + channels),
                'position_sizing': self._recommend_position_sizing(breakout_potential, risk_reward),
                
                # Additional Context
                'market_context': self._analyze_market_context(stock_data, indicators),
                'volume_confirmation': self._check_volume_confirmation(stock_data, triangles + flags_pennants + channels)
            }
            
            logger.info(f"[CONTINUATION_PATTERNS] Analysis completed in {processing_time:.2f}s")
            return analysis_result
            
        except Exception as e:
            logger.error(f"[CONTINUATION_PATTERNS] Analysis failed: {str(e)}")
            return {
                'agent_name': self.name,
                'error': str(e),
                'success': False,
                'confidence_score': 0.0
            }
    
    def _detect_triangle_patterns(self, stock_data: pd.DataFrame) -> List[ContinuationPattern]:
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        patterns = []
        highs = stock_data['high'].values
        lows = stock_data['low'].values
        
        if len(highs) < 20:
            return patterns
        
        # Look for triangle patterns in recent data
        lookback = min(30, len(highs) - 5)
        
        for i in range(lookback, len(highs) - 5):
            # Get recent price action for triangle analysis
            recent_highs = highs[i-lookback:i]
            recent_lows = lows[i-lookback:i]
            
            # Analyze trend lines
            upper_trend = self._calculate_trend_line(recent_highs, ascending=False)
            lower_trend = self._calculate_trend_line(recent_lows, ascending=True)
            
            if upper_trend and lower_trend:
                # Determine triangle type
                upper_slope = upper_trend['slope']
                lower_slope = lower_trend['slope']
                
                triangle_type = None
                direction = "neutral"
                
                if upper_slope < -0.01 and abs(lower_slope) < 0.01:  # Descending triangle
                    triangle_type = "descending_triangle"
                    direction = "bearish"
                elif lower_slope > 0.01 and abs(upper_slope) < 0.01:  # Ascending triangle
                    triangle_type = "ascending_triangle" 
                    direction = "bullish"
                elif upper_slope < -0.005 and lower_slope > 0.005:  # Symmetrical triangle
                    triangle_type = "symmetrical_triangle"
                    direction = "neutral"
                
                if triangle_type:
                    # Calculate breakout levels
                    current_price = stock_data['close'].iloc[i-1]
                    upper_level = upper_trend['level']
                    lower_level = lower_trend['level']
                    
                    pattern = ContinuationPattern(
                        pattern_type=triangle_type,
                        strength="medium",
                        completion=self._calculate_triangle_completion(recent_highs, recent_lows),
                        reliability="high",
                        breakout_level=upper_level if direction == "bullish" else lower_level,
                        stop_loss=lower_level if direction == "bullish" else upper_level,
                        target_levels=self._calculate_triangle_targets(current_price, upper_level, lower_level, direction),
                        confidence=0.7,
                        description=f"{triangle_type.replace('_', ' ').title()} pattern",
                        direction=direction
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_flags_pennants(self, stock_data: pd.DataFrame) -> List[ContinuationPattern]:
        """Detect flag and pennant patterns"""
        patterns = []
        prices = stock_data['close'].values
        highs = stock_data['high'].values
        lows = stock_data['low'].values
        volume = stock_data.get('volume', None)
        
        if len(prices) < 15:
            return patterns
        
        # Look for flag/pennant patterns after strong moves
        for i in range(10, len(prices) - 5):
            # Check for strong prior move (flagpole)
            flagpole_start = max(0, i - 10)
            flagpole_move = (prices[i] - prices[flagpole_start]) / prices[flagpole_start]
            
            if abs(flagpole_move) > 0.05:  # At least 5% move
                # Analyze consolidation period
                consolidation_highs = highs[i:i+5]
                consolidation_lows = lows[i:i+5]
                consolidation_range = np.max(consolidation_highs) - np.min(consolidation_lows)
                avg_price = np.mean(prices[i:i+5])
                
                if consolidation_range / avg_price < 0.05:  # Tight consolidation
                    pattern_type = "bull_flag" if flagpole_move > 0 else "bear_flag"
                    direction = "bullish" if flagpole_move > 0 else "bearish"
                    
                    breakout_level = np.max(consolidation_highs) if direction == "bullish" else np.min(consolidation_lows)
                    stop_loss = np.min(consolidation_lows) if direction == "bullish" else np.max(consolidation_highs)
                    
                    # Calculate targets based on flagpole height
                    flagpole_height = abs(prices[i] - prices[flagpole_start])
                    target1 = breakout_level + flagpole_height if direction == "bullish" else breakout_level - flagpole_height
                    target2 = breakout_level + flagpole_height * 1.5 if direction == "bullish" else breakout_level - flagpole_height * 1.5
                    
                    pattern = ContinuationPattern(
                        pattern_type=pattern_type,
                        strength="strong" if abs(flagpole_move) > 0.1 else "medium",
                        completion=80.0,
                        reliability="high",
                        breakout_level=breakout_level,
                        stop_loss=stop_loss,
                        target_levels=[target1, target2],
                        confidence=0.75,
                        description=f"{pattern_type.replace('_', ' ').title()} after {flagpole_move:.1%} move",
                        direction=direction
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_channel_patterns(self, stock_data: pd.DataFrame) -> List[ContinuationPattern]:
        """Detect channel patterns"""
        patterns = []
        highs = stock_data['high'].values
        lows = stock_data['low'].values
        
        if len(highs) < 25:
            return patterns
        
        # Look for parallel channel patterns
        lookback = min(30, len(highs) - 5)
        
        for i in range(lookback, len(highs) - 5):
            # Get data for channel analysis
            recent_highs = highs[i-lookback:i]
            recent_lows = lows[i-lookback:i]
            
            # Calculate trend lines for channel
            upper_trend = self._calculate_trend_line(recent_highs, ascending=None)
            lower_trend = self._calculate_trend_line(recent_lows, ascending=None)
            
            if upper_trend and lower_trend:
                # Check if lines are roughly parallel
                slope_diff = abs(upper_trend['slope'] - lower_trend['slope'])
                
                if slope_diff < 0.02:  # Roughly parallel
                    # Determine channel type
                    avg_slope = (upper_trend['slope'] + lower_trend['slope']) / 2
                    
                    if avg_slope > 0.01:
                        channel_type = "ascending_channel"
                        direction = "bullish"
                    elif avg_slope < -0.01:
                        channel_type = "descending_channel"
                        direction = "bearish"
                    else:
                        channel_type = "horizontal_channel"
                        direction = "neutral"
                    
                    current_price = stock_data['close'].iloc[i-1]
                    upper_level = upper_trend['level']
                    lower_level = lower_trend['level']
                    
                    # Determine likely breakout direction and levels
                    if current_price > (upper_level + lower_level) / 2:
                        breakout_level = upper_level
                        stop_loss = lower_level
                    else:
                        breakout_level = lower_level
                        stop_loss = upper_level
                    
                    channel_height = upper_level - lower_level
                    target1 = breakout_level + channel_height if breakout_level == upper_level else breakout_level - channel_height
                    
                    pattern = ContinuationPattern(
                        pattern_type=channel_type,
                        strength="medium",
                        completion=85.0,
                        reliability="medium",
                        breakout_level=breakout_level,
                        stop_loss=stop_loss,
                        target_levels=[target1],
                        confidence=0.65,
                        description=f"{channel_type.replace('_', ' ').title()} pattern",
                        direction=direction
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _identify_key_levels(self, stock_data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Identify key support and resistance levels"""
        highs = stock_data['high'].values
        lows = stock_data['low'].values
        
        # Find significant levels using pivot points
        support_levels = []
        resistance_levels = []
        
        window = min(10, len(highs) // 5)
        
        # Find resistance levels (local maxima)
        for i in range(window, len(highs) - window):
            if highs[i] == np.max(highs[i-window:i+window+1]):
                resistance_levels.append({
                    'level': highs[i],
                    'strength': self._calculate_level_strength(highs, highs[i]),
                    'touches': self._count_level_touches(highs, highs[i], tolerance=0.02)
                })
        
        # Find support levels (local minima)
        for i in range(window, len(lows) - window):
            if lows[i] == np.min(lows[i-window:i+window+1]):
                support_levels.append({
                    'level': lows[i],
                    'strength': self._calculate_level_strength(lows, lows[i], is_support=True),
                    'touches': self._count_level_touches(lows, lows[i], tolerance=0.02)
                })
        
        # Sort by strength and keep top levels
        support_levels = sorted(support_levels, key=lambda x: x['strength'], reverse=True)[:5]
        resistance_levels = sorted(resistance_levels, key=lambda x: x['strength'], reverse=True)[:5]
        
        return {
            'support_levels': [s['level'] for s in support_levels],
            'resistance_levels': [r['level'] for r in resistance_levels],
            'support_strength': [s['strength'] for s in support_levels],
            'resistance_strength': [r['strength'] for r in resistance_levels],
            'level_analysis': {
                'strongest_support': support_levels[0]['level'] if support_levels else None,
                'strongest_resistance': resistance_levels[0]['level'] if resistance_levels else None,
                'support_quality': 'strong' if support_levels and support_levels[0]['strength'] > 0.7 else 'weak',
                'resistance_quality': 'strong' if resistance_levels and resistance_levels[0]['strength'] > 0.7 else 'weak'
            }
        }
    
    def _calculate_trend_line(self, prices: np.ndarray, ascending: bool = None) -> Optional[Dict[str, float]]:
        """Calculate trend line for prices"""
        if len(prices) < 5:
            return None
        
        x = np.arange(len(prices))
        
        # Use linear regression to find trend line
        slope, intercept = np.polyfit(x, prices, 1)
        current_level = slope * (len(prices) - 1) + intercept
        
        # Calculate R-squared for trend line quality
        y_pred = slope * x + intercept
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        if r_squared > 0.3:  # Reasonable trend line fit
            return {
                'slope': slope,
                'intercept': intercept,
                'level': current_level,
                'r_squared': r_squared
            }
        
        return None
    
    def _calculate_triangle_completion(self, highs: np.ndarray, lows: np.ndarray) -> float:
        """Calculate completion percentage of triangle pattern"""
        if len(highs) < 5 or len(lows) < 5:
            return 0.0
        
        # Calculate convergence of highs and lows
        high_range = np.max(highs) - np.min(highs)
        low_range = np.max(lows) - np.min(lows)
        
        recent_range = np.max(highs[-5:]) - np.min(lows[-5:])
        initial_range = np.max(highs[:5]) - np.min(lows[:5])
        
        if initial_range == 0:
            return 0.0
        
        convergence_ratio = 1 - (recent_range / initial_range)
        return min(100.0, max(0.0, convergence_ratio * 100))
    
    def _calculate_triangle_targets(self, current_price: float, upper_level: float, 
                                  lower_level: float, direction: str) -> List[float]:
        """Calculate target levels for triangle breakout"""
        triangle_height = upper_level - lower_level
        
        if direction == "bullish":
            target1 = upper_level + triangle_height * 0.5
            target2 = upper_level + triangle_height
        elif direction == "bearish":
            target1 = lower_level - triangle_height * 0.5
            target2 = lower_level - triangle_height
        else:  # neutral - use both directions
            target1 = upper_level + triangle_height * 0.5
            target2 = lower_level - triangle_height * 0.5
        
        return [target1, target2] if direction != "neutral" else [target1, target2]
    
    def _calculate_level_strength(self, prices: np.ndarray, level: float, is_support: bool = False) -> float:
        """Calculate strength of support/resistance level"""
        touches = self._count_level_touches(prices, level, tolerance=0.02)
        
        # More touches = stronger level
        touch_score = min(1.0, touches / 5.0)
        
        # Consider how well level held
        violations = self._count_level_violations(prices, level, is_support, tolerance=0.01)
        hold_score = max(0.0, 1.0 - violations / max(1, touches))
        
        return (touch_score * 0.6) + (hold_score * 0.4)
    
    def _count_level_touches(self, prices: np.ndarray, level: float, tolerance: float = 0.02) -> int:
        """Count how many times price touched a level"""
        touches = 0
        for price in prices:
            if abs(price - level) / level <= tolerance:
                touches += 1
        return touches
    
    def _count_level_violations(self, prices: np.ndarray, level: float, 
                              is_support: bool, tolerance: float = 0.01) -> int:
        """Count violations of support/resistance level"""
        violations = 0
        for price in prices:
            if is_support and price < level * (1 - tolerance):
                violations += 1
            elif not is_support and price > level * (1 + tolerance):
                violations += 1
        return violations
    
    def _analyze_pattern_confluence(self, triangles: List[ContinuationPattern], 
                                  flags_pennants: List[ContinuationPattern], 
                                  channels: List[ContinuationPattern]) -> Dict[str, Any]:
        """Analyze confluence between different continuation patterns"""
        all_patterns = triangles + flags_pennants + channels
        
        if not all_patterns:
            return {'has_confluence': False, 'confluence_strength': 'none'}
        
        # Group patterns by direction
        bullish_patterns = [p for p in all_patterns if p.direction == "bullish"]
        bearish_patterns = [p for p in all_patterns if p.direction == "bearish"]
        neutral_patterns = [p for p in all_patterns if p.direction == "neutral"]
        
        confluence = {
            'has_confluence': len(bullish_patterns) > 1 or len(bearish_patterns) > 1,
            'bullish_confluence': len(bullish_patterns),
            'bearish_confluence': len(bearish_patterns),
            'neutral_patterns': len(neutral_patterns),
            'confluence_strength': 'strong' if max(len(bullish_patterns), len(bearish_patterns)) >= 2 else 'weak',
            'dominant_direction': ('bullish' if len(bullish_patterns) > len(bearish_patterns) 
                                 else 'bearish' if len(bearish_patterns) > len(bullish_patterns) 
                                 else 'neutral')
        }
        
        return confluence
    
    def _assess_breakout_potential(self, stock_data: pd.DataFrame, patterns: List[ContinuationPattern]) -> Dict[str, Any]:
        """Assess breakout potential of continuation patterns"""
        if not patterns:
            return {'potential': 'low', 'direction': 'neutral'}
        
        current_price = stock_data['close'].iloc[-1]
        
        # Calculate proximity to breakout levels
        breakout_proximities = []
        for pattern in patterns:
            if pattern.breakout_level:
                proximity = abs(current_price - pattern.breakout_level) / current_price
                breakout_proximities.append({
                    'pattern': pattern.pattern_type,
                    'proximity': proximity,
                    'direction': pattern.direction,
                    'completion': pattern.completion
                })
        
        if not breakout_proximities:
            return {'potential': 'low', 'direction': 'neutral'}
        
        # Find closest breakout
        closest_breakout = min(breakout_proximities, key=lambda x: x['proximity'])
        
        potential = 'high' if closest_breakout['proximity'] < 0.02 else 'medium' if closest_breakout['proximity'] < 0.05 else 'low'
        
        return {
            'potential': potential,
            'direction': closest_breakout['direction'],
            'closest_pattern': closest_breakout['pattern'],
            'proximity': closest_breakout['proximity'],
            'completion': closest_breakout['completion']
        }
    
    # Additional helper methods would continue here...
    # For brevity, I'll include the key remaining methods
    
    def _calculate_risk_reward(self, stock_data: pd.DataFrame, patterns: List[ContinuationPattern]) -> float:
        """Calculate risk-reward ratio for continuation patterns"""
        if not patterns:
            return 0.0
        
        current_price = stock_data['close'].iloc[-1]
        risk_rewards = []
        
        for pattern in patterns:
            if pattern.target_levels and pattern.stop_loss:
                avg_target = np.mean(pattern.target_levels)
                reward = abs(avg_target - current_price)
                risk = abs(current_price - pattern.stop_loss)
                
                if risk > 0:
                    risk_rewards.append(reward / risk)
        
        return np.mean(risk_rewards) if risk_rewards else 1.0
    
    def _determine_primary_signal(self, triangles: List[ContinuationPattern], 
                                flags_pennants: List[ContinuationPattern], 
                                channels: List[ContinuationPattern]) -> str:
        """Determine primary continuation signal"""
        all_patterns = triangles + flags_pennants + channels
        
        if not all_patterns:
            return "neutral"
        
        # Weight patterns by strength and reliability
        bullish_score = 0
        bearish_score = 0
        
        for pattern in all_patterns:
            weight = 1.0
            if pattern.strength == "strong":
                weight = 1.5
            elif pattern.strength == "weak":
                weight = 0.5
            
            if pattern.direction == "bullish":
                bullish_score += weight
            elif pattern.direction == "bearish":
                bearish_score += weight
        
        if bullish_score > bearish_score * 1.2:
            return "bullish_continuation"
        elif bearish_score > bullish_score * 1.2:
            return "bearish_continuation"
        else:
            return "mixed_signals"
    
    def _pattern_to_dict(self, pattern: ContinuationPattern) -> Dict[str, Any]:
        """Convert ContinuationPattern to dictionary"""
        return {
            'type': pattern.pattern_type,
            'strength': pattern.strength,
            'completion': pattern.completion,
            'reliability': pattern.reliability,
            'breakout_level': pattern.breakout_level,
            'stop_loss': pattern.stop_loss,
            'target_levels': pattern.target_levels,
            'confidence': pattern.confidence,
            'description': pattern.description,
            'direction': pattern.direction
        }
    
    def _identify_breakout_levels(self, patterns: List[ContinuationPattern]) -> List[float]:
        """Identify breakout levels from patterns"""
        return [p.breakout_level for p in patterns if p.breakout_level]
    
    def _calculate_entry_points(self, stock_data: pd.DataFrame, patterns: List[ContinuationPattern]) -> List[float]:
        """Calculate optimal entry points"""
        return [p.breakout_level for p in patterns if p.breakout_level]
    
    def _calculate_stop_loss_levels(self, stock_data: pd.DataFrame, patterns: List[ContinuationPattern]) -> List[float]:
        """Calculate stop loss levels"""
        return [p.stop_loss for p in patterns if p.stop_loss]
    
    def _calculate_target_levels(self, stock_data: pd.DataFrame, patterns: List[ContinuationPattern]) -> List[float]:
        """Calculate target levels"""
        all_targets = []
        for pattern in patterns:
            if pattern.target_levels:
                all_targets.extend(pattern.target_levels)
        return all_targets
    
    def _assess_pattern_failure_risk(self, stock_data: pd.DataFrame, patterns: List[ContinuationPattern]) -> str:
        """Assess risk of pattern failure"""
        if not patterns:
            return "high"
        
        avg_completion = np.mean([p.completion for p in patterns])
        high_reliability = len([p for p in patterns if p.reliability == "high"])
        
        if avg_completion > 80 and high_reliability > 0:
            return "low"
        elif avg_completion > 60:
            return "medium"
        else:
            return "high"
    
    def _assess_false_breakout_risk(self, patterns: List[ContinuationPattern]) -> str:
        """Assess risk of false breakouts"""
        if not patterns:
            return "high"
        
        strong_patterns = len([p for p in patterns if p.strength == "strong"])
        total_patterns = len(patterns)
        
        if strong_patterns / total_patterns > 0.6:
            return "low"
        else:
            return "medium"
    
    def _calculate_confidence_score(self, triangles: List[ContinuationPattern], 
                                  flags_pennants: List[ContinuationPattern], 
                                  channels: List[ContinuationPattern], 
                                  indicators: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        all_patterns = triangles + flags_pennants + channels
        
        if not all_patterns:
            return 0.0
        
        pattern_confidence = np.mean([p.confidence for p in all_patterns])
        pattern_count_bonus = min(0.15, len(all_patterns) * 0.03)
        
        return min(1.0, pattern_confidence + pattern_count_bonus)
    
    def _assess_reliability(self, patterns: List[ContinuationPattern]) -> str:
        """Assess overall reliability of patterns"""
        if not patterns:
            return "low"
        
        high_reliability = len([p for p in patterns if p.reliability == "high"])
        reliability_ratio = high_reliability / len(patterns)
        
        if reliability_ratio > 0.6:
            return "high"
        elif reliability_ratio > 0.3:
            return "medium"
        else:
            return "low"
    
    def _generate_trading_strategy(self, primary_signal: str, patterns: List[ContinuationPattern]) -> Dict[str, Any]:
        """Generate trading strategy based on continuation patterns"""
        if not patterns:
            return {'action': 'hold', 'reasoning': 'No clear continuation patterns identified'}
        
        strategy = {
            'primary_action': ('buy' if 'bullish' in primary_signal 
                             else 'sell' if 'bearish' in primary_signal 
                             else 'hold'),
            'entry_strategy': 'breakout_confirmation',
            'position_type': 'continuation',
            'time_horizon': 'medium_term',
            'patterns_supporting': len(patterns),
            'reasoning': f"Based on {len(patterns)} continuation pattern(s) with {primary_signal} bias"
        }
        
        return strategy
    
    def _recommend_position_sizing(self, breakout_potential: Dict[str, Any], risk_reward: float) -> Dict[str, Any]:
        """Recommend position sizing"""
        potential = breakout_potential.get('potential', 'low')
        
        if potential == 'high' and risk_reward > 2.0:
            size = "large"
        elif potential == 'medium' and risk_reward > 1.5:
            size = "medium"
        else:
            size = "small"
        
        return {
            'recommended_size': size,
            'breakout_potential': potential,
            'risk_reward_ratio': risk_reward,
            'reasoning': f"Position sizing based on {potential} breakout potential and {risk_reward:.2f} R:R"
        }
    
    def _analyze_market_context(self, stock_data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market context for continuation patterns"""
        current_price = stock_data['close'].iloc[-1]
        recent_prices = stock_data['close'].tail(20)
        
        return {
            'trend_direction': 'up' if recent_prices.iloc[-1] > recent_prices.iloc[0] else 'down',
            'price_momentum': 'strong' if abs(recent_prices.iloc[-1] / recent_prices.iloc[0] - 1) > 0.05 else 'weak',
            'consolidation_phase': recent_prices.std() / recent_prices.mean() < 0.02
        }
    
    def _check_volume_confirmation(self, stock_data: pd.DataFrame, patterns: List[ContinuationPattern]) -> Dict[str, Any]:
        """Check volume confirmation for continuation patterns"""
        if 'volume' not in stock_data.columns:
            return {'volume_confirmation': 'unavailable'}
        
        recent_volume = stock_data['volume'].tail(10).mean()
        avg_volume = stock_data['volume'].mean()
        
        return {
            'has_volume_support': recent_volume > avg_volume * 1.1,
            'volume_trend': 'increasing' if recent_volume > avg_volume else 'stable',
            'confirmation_strength': 'strong' if recent_volume > avg_volume * 1.3 else 'weak'
        }