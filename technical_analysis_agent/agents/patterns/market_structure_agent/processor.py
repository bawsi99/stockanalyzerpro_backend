#!/usr/bin/env python3
"""
Market Structure Processor - Technical Analysis Module

This module handles the technical analysis of market structure including:
- Swing point identification and classification
- BOS (Break of Structure) and CHOCH (Change of Character) detection
- Trend analysis and structural integrity assessment
- Support/resistance level identification from market structure
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MarketStructureProcessor:
    """
    Processor for analyzing market structure in stock data.
    
    This processor specializes in:
    - Swing high/low identification using multiple methods
    - BOS/CHOCH detection and classification
    - Trend structure analysis and quality assessment
    - Key level identification from structural points
    """
    
    def __init__(self):
        self.name = "market_structure"
        self.description = "Analyzes market structure, swing points, and trend changes"
        self.version = "1.0.0"
    
    def process_market_structure_data(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Process market structure analysis from stock data.
        
        Args:
            stock_data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing comprehensive market structure analysis
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"[MARKET_STRUCTURE] Starting market structure analysis")
            
            if stock_data is None or stock_data.empty or len(stock_data) < 20:
                return self._build_error_result("Insufficient data for market structure analysis")
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in stock_data.columns]
            if missing_columns:
                return self._build_error_result(f"Missing required columns: {missing_columns}")
            
            # 1. Swing Point Analysis
            swing_points = self._identify_swing_points(stock_data)
            
            # 2. BOS/CHOCH Analysis
            bos_choch_analysis = self._analyze_bos_choch(stock_data, swing_points)
            
            # 3. Trend Analysis
            trend_analysis = self._analyze_trend_structure(stock_data, swing_points)
            
            # 4. Support/Resistance Levels
            key_levels = self._identify_key_levels(stock_data, swing_points)
            
            # 5. Structure Quality Assessment
            structure_quality = self._assess_structure_quality(swing_points, bos_choch_analysis, trend_analysis)
            
            # 6. Current Market State
            current_state = self._analyze_current_state(stock_data, swing_points, trend_analysis)
            
            # 7. Fractal Analysis (multi-timeframe structure simulation)
            fractal_analysis = self._analyze_fractal_structure(stock_data, swing_points)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Build comprehensive result
            result = {
                'success': True,
                'agent_name': self.name,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                
                # Core Analysis
                'swing_points': swing_points,
                'bos_choch_analysis': bos_choch_analysis,
                'trend_analysis': trend_analysis,
                'key_levels': key_levels,
                
                # Assessment
                'structure_quality': structure_quality,
                'current_state': current_state,
                'fractal_analysis': fractal_analysis,
                
                # Statistics
                'analysis_period_days': len(stock_data),
                'total_swing_highs': len(swing_points.get('swing_highs', [])),
                'total_swing_lows': len(swing_points.get('swing_lows', [])),
                
                # Confidence Metrics
                'confidence_score': self._calculate_confidence_score(swing_points, structure_quality, trend_analysis),
                'data_quality': self._assess_data_quality(stock_data)
            }
            
            logger.info(f"[MARKET_STRUCTURE] Analysis completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"[MARKET_STRUCTURE] Analysis failed: {str(e)}")
            return self._build_error_result(str(e), processing_time)
    
    def _identify_swing_points(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """Identify swing highs and swing lows"""
        try:
            highs = stock_data['high'].values
            lows = stock_data['low'].values
            dates = stock_data.index
            
            swing_highs = []
            swing_lows = []
            
            # Parameters for swing detection
            lookback = 5  # Look back period for swing confirmation
            
            # Identify swing highs
            for i in range(lookback, len(highs) - lookback):
                current_high = highs[i]
                
                # Check if current high is higher than surrounding points
                left_side = highs[i-lookback:i]
                right_side = highs[i+1:i+lookback+1]
                
                if (current_high > np.max(left_side) and 
                    current_high > np.max(right_side)):
                    swing_highs.append({
                        'date': str(dates[i]),
                        'price': round(float(current_high), 2),
                        'index': int(i),
                        'strength': self._calculate_swing_strength(highs, i, True),
                        'type': 'swing_high'
                    })
            
            # Identify swing lows
            for i in range(lookback, len(lows) - lookback):
                current_low = lows[i]
                
                # Check if current low is lower than surrounding points
                left_side = lows[i-lookback:i]
                right_side = lows[i+1:i+lookback+1]
                
                if (current_low < np.min(left_side) and 
                    current_low < np.min(right_side)):
                    swing_lows.append({
                        'date': str(dates[i]),
                        'price': round(float(current_low), 2),
                        'index': int(i),
                        'strength': self._calculate_swing_strength(lows, i, False),
                        'type': 'swing_low'
                    })
            
            # Sort by date
            swing_highs.sort(key=lambda x: x['date'])
            swing_lows.sort(key=lambda x: x['date'])
            
            return {
                'swing_highs': swing_highs,
                'swing_lows': swing_lows,
                'total_swings': len(swing_highs) + len(swing_lows),
                'swing_density': round((len(swing_highs) + len(swing_lows)) / len(stock_data) if len(stock_data) > 0 else 0, 2),
                'analysis_method': 'local_extrema',
                'lookback_period': lookback
            }
            
        except Exception as e:
            logger.error(f"[MARKET_STRUCTURE] Swing point identification failed: {e}")
            return {'error': str(e)}
    
    def _calculate_swing_strength(self, price_array: np.ndarray, index: int, is_high: bool) -> str:
        """Calculate the strength of a swing point"""
        try:
            lookback = min(10, index, len(price_array) - index - 1)
            if lookback < 3:
                return 'weak'
            
            current_price = price_array[index]
            surrounding_prices = np.concatenate([
                price_array[index-lookback:index],
                price_array[index+1:index+lookback+1]
            ])
            
            if is_high:
                # For swing highs, check how much higher than surrounding prices
                price_diff = current_price - np.max(surrounding_prices)
            else:
                # For swing lows, check how much lower than surrounding prices
                price_diff = np.min(surrounding_prices) - current_price
            
            price_percentage = abs(price_diff) / current_price * 100
            
            if price_percentage > 3.0:
                return 'strong'
            elif price_percentage > 1.5:
                return 'medium'
            else:
                return 'weak'
                
        except Exception:
            return 'unknown'
    
    def _analyze_bos_choch(self, stock_data: pd.DataFrame, swing_points: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Break of Structure (BOS) and Change of Character (CHOCH)"""
        try:
            if 'swing_highs' not in swing_points or 'swing_lows' not in swing_points:
                return {'error': 'No swing points available for BOS/CHOCH analysis'}
            
            bos_events = []
            choch_events = []
            
            swing_highs = swing_points['swing_highs']
            swing_lows = swing_points['swing_lows']
            
            # Analyze for bullish BOS (breaking above previous swing high)
            if len(swing_highs) >= 2:
                for i in range(1, len(swing_highs)):
                    current_high = swing_highs[i]
                    previous_high = swing_highs[i-1]
                    
                    if current_high['price'] > previous_high['price'] * 1.01:  # At least 1% break
                        bos_events.append({
                            'type': 'bullish_bos',
                            'date': current_high['date'],
                            'break_price': round(current_high['price'], 2),
                            'previous_level': round(previous_high['price'], 2),
                            'percentage_break': round(((current_high['price'] - previous_high['price']) / previous_high['price']) * 100, 2),
                            'strength': 'strong' if current_high['price'] > previous_high['price'] * 1.03 else 'medium'
                        })
            
            # Analyze for bearish BOS (breaking below previous swing low)
            if len(swing_lows) >= 2:
                for i in range(1, len(swing_lows)):
                    current_low = swing_lows[i]
                    previous_low = swing_lows[i-1]
                    
                    if current_low['price'] < previous_low['price'] * 0.99:  # At least 1% break
                        bos_events.append({
                            'type': 'bearish_bos',
                            'date': current_low['date'],
                            'break_price': round(current_low['price'], 2),
                            'previous_level': round(previous_low['price'], 2),
                            'percentage_break': round(((previous_low['price'] - current_low['price']) / previous_low['price']) * 100, 2),
                            'strength': 'strong' if current_low['price'] < previous_low['price'] * 0.97 else 'medium'
                        })
            
            # CHOCH analysis (simplified - would need more complex logic for true CHOCH)
            # For now, identify potential character changes
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                recent_highs = swing_highs[-2:]
                recent_lows = swing_lows[-2:]
                
                # Check for potential trend change signals
                if (recent_highs[-1]['price'] < recent_highs[-2]['price'] and 
                    recent_lows[-1]['price'] < recent_lows[-2]['price']):
                    choch_events.append({
                        'type': 'bearish_choch',
                        'date': recent_lows[-1]['date'],
                        'description': 'Lower highs and lower lows pattern suggesting bearish character change',
                        'confidence': 'medium'
                    })
                elif (recent_highs[-1]['price'] > recent_highs[-2]['price'] and 
                      recent_lows[-1]['price'] > recent_lows[-2]['price']):
                    choch_events.append({
                        'type': 'bullish_choch',
                        'date': recent_highs[-1]['date'],
                        'description': 'Higher highs and higher lows pattern suggesting bullish character change',
                        'confidence': 'medium'
                    })
            
            return {
                'bos_events': bos_events,
                'choch_events': choch_events,
                'total_bos_events': len(bos_events),
                'total_choch_events': len(choch_events),
                'recent_structural_break': bos_events[-1] if bos_events else None,
                'structural_bias': self._determine_structural_bias(bos_events, choch_events)
            }
            
        except Exception as e:
            logger.error(f"[MARKET_STRUCTURE] BOS/CHOCH analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_trend_structure(self, stock_data: pd.DataFrame, swing_points: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall trend structure with market regime awareness"""
        try:
            if 'swing_highs' not in swing_points or 'swing_lows' not in swing_points:
                return {'error': 'No swing points available for trend analysis'}
            
            swing_highs = swing_points['swing_highs']
            swing_lows = swing_points['swing_lows']
            
            # Detect market regime first
            market_regime = self._detect_market_regime(stock_data, swing_highs, swing_lows)
            
            # Analyze trend from swing points with regime awareness
            trend_direction = self._determine_trend_direction(swing_highs, swing_lows)
            trend_strength = self._calculate_trend_strength(swing_highs, swing_lows, market_regime)
            trend_consistency = self._assess_trend_consistency(swing_highs, swing_lows)
            
            # Price action analysis
            current_price = stock_data['close'].iloc[-1]
            price_range = stock_data['high'].max() - stock_data['low'].min()
            price_position = (current_price - stock_data['low'].min()) / price_range if price_range > 0 else 0.5
            
            return {
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'trend_consistency': trend_consistency,
                'price_position_in_range': round(price_position, 2),
                'current_price': round(float(current_price), 2),
                'range_high': round(float(stock_data['high'].max()), 2),
                'range_low': round(float(stock_data['low'].min()), 2),
                'trend_quality': self._assess_trend_quality(trend_direction, trend_strength, trend_consistency),
                'market_regime': market_regime  # Add regime information
            }
            
        except Exception as e:
            logger.error(f"[MARKET_STRUCTURE] Trend analysis failed: {e}")
            return {'error': str(e)}
    
    def _determine_trend_direction(self, swing_highs: List[Dict], swing_lows: List[Dict]) -> str:
        """Determine the overall trend direction from swing points with improved logic"""
        try:
            if len(swing_highs) < 2 or len(swing_lows) < 2:
                return 'insufficient_data'
            
            # Use more swing points for better trend determination
            min_swings = min(len(swing_highs), len(swing_lows))
            analysis_count = min(4, min_swings)  # Use up to 4 most recent swings
            
            if analysis_count < 2:
                return 'insufficient_data'
            
            recent_highs = swing_highs[-analysis_count:]
            recent_lows = swing_lows[-analysis_count:]
            
            # Calculate trend scores based on multiple criteria
            uptrend_score = 0
            downtrend_score = 0
            
            # Criterion 1: Higher Highs and Higher Lows pattern
            hh_count = sum(1 for i in range(1, len(recent_highs)) 
                          if recent_highs[i]['price'] > recent_highs[i-1]['price'])
            hl_count = sum(1 for i in range(1, len(recent_lows)) 
                          if recent_lows[i]['price'] > recent_lows[i-1]['price'])
            
            lh_count = sum(1 for i in range(1, len(recent_highs)) 
                          if recent_highs[i]['price'] < recent_highs[i-1]['price'])
            ll_count = sum(1 for i in range(1, len(recent_lows)) 
                          if recent_lows[i]['price'] < recent_lows[i-1]['price'])
            
            # Score based on HH/HL vs LH/LL patterns
            uptrend_score += (hh_count + hl_count) * 2
            downtrend_score += (lh_count + ll_count) * 2
            
            # Criterion 2: Overall price progression
            if len(swing_highs) >= 3:
                first_high = swing_highs[0]['price']
                last_high = swing_highs[-1]['price']
                high_change_pct = (last_high - first_high) / first_high * 100
                
                if high_change_pct > 2.0:  # Strong upward progression
                    uptrend_score += 3
                elif high_change_pct < -2.0:  # Strong downward progression
                    downtrend_score += 3
            
            if len(swing_lows) >= 3:
                first_low = swing_lows[0]['price']
                last_low = swing_lows[-1]['price']
                low_change_pct = (last_low - first_low) / first_low * 100
                
                if low_change_pct > 2.0:  # Lows moving higher
                    uptrend_score += 3
                elif low_change_pct < -2.0:  # Lows moving lower
                    downtrend_score += 3
            
            # Criterion 3: Recent momentum (last 2 swings weighted more)
            if len(recent_highs) >= 2:
                recent_high_change = (recent_highs[-1]['price'] - recent_highs[-2]['price']) / recent_highs[-2]['price'] * 100
                if recent_high_change > 1.0:
                    uptrend_score += 2
                elif recent_high_change < -1.0:
                    downtrend_score += 2
            
            if len(recent_lows) >= 2:
                recent_low_change = (recent_lows[-1]['price'] - recent_lows[-2]['price']) / recent_lows[-2]['price'] * 100
                if recent_low_change > 1.0:
                    uptrend_score += 2
                elif recent_low_change < -1.0:
                    downtrend_score += 2
            
            # Make decision based on scores with threshold
            score_diff = abs(uptrend_score - downtrend_score)
            total_score = uptrend_score + downtrend_score
            
            # If scores are too close, consider it sideways
            if total_score == 0 or score_diff / total_score < 0.3:  # Less than 30% difference
                return 'sideways'
            elif uptrend_score > downtrend_score:
                return 'uptrend'
            else:
                return 'downtrend'
                
        except Exception:
            return 'unknown'
    
    def _detect_market_regime(self, stock_data: pd.DataFrame, swing_highs: List[Dict], swing_lows: List[Dict]) -> Dict[str, Any]:
        """Detect the current market regime to adjust analysis parameters"""
        try:
            # Calculate volatility metrics
            returns = stock_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Calculate price momentum
            price_change = (stock_data['close'].iloc[-1] - stock_data['close'].iloc[0]) / stock_data['close'].iloc[0] * 100
            
            # Calculate swing frequency
            swing_frequency = len(swing_highs) + len(swing_lows)
            periods = len(stock_data)
            swing_rate = swing_frequency / periods if periods > 0 else 0
            
            # Determine regime based on multiple factors
            regime_scores = {
                'trending': 0,
                'volatile': 0, 
                'consolidating': 0,
                'mixed': 0
            }
            
            # Price momentum scoring
            if abs(price_change) > 10:  # Strong momentum
                regime_scores['trending'] += 3
            elif abs(price_change) > 5:  # Moderate momentum
                regime_scores['trending'] += 2
                regime_scores['mixed'] += 1
            elif abs(price_change) < 2:  # Low momentum
                regime_scores['consolidating'] += 2
            
            # Volatility scoring
            if volatility > 0.4:  # High volatility
                regime_scores['volatile'] += 3
            elif volatility > 0.25:  # Medium volatility
                regime_scores['volatile'] += 1
                regime_scores['mixed'] += 1
            else:  # Low volatility
                regime_scores['consolidating'] += 2
            
            # Swing frequency scoring
            if swing_rate > 0.15:  # High swing frequency
                regime_scores['volatile'] += 2
            elif swing_rate > 0.1:  # Medium swing frequency
                regime_scores['mixed'] += 2
            else:  # Low swing frequency
                regime_scores['trending'] += 1
                regime_scores['consolidating'] += 1
            
            # Determine primary regime
            primary_regime = max(regime_scores.items(), key=lambda x: x[1])[0]
            confidence = regime_scores[primary_regime] / max(sum(regime_scores.values()), 1)
            
            return {
                'regime': primary_regime,
                'confidence': round(confidence, 2),
                'volatility': round(volatility, 2),
                'price_change_pct': round(price_change, 2),
                'swing_rate': round(swing_rate, 2),
                'regime_scores': regime_scores
            }
            
        except Exception as e:
            logger.error(f"[MARKET_STRUCTURE] Market regime detection failed: {e}")
            return {'regime': 'unknown', 'confidence': 0}
    
    def _calculate_trend_strength(self, swing_highs: List[Dict], swing_lows: List[Dict], market_regime: Dict[str, Any] = None) -> str:
        """Calculate trend strength based on swing point progression with regime awareness"""
        try:
            if len(swing_highs) < 2 or len(swing_lows) < 2:  # Reduced minimum requirement
                return 'weak'
            
            # Get market regime info
            regime_type = market_regime.get('regime', 'unknown') if market_regime else 'unknown'
            volatility = market_regime.get('volatility', 0.3) if market_regime else 0.3
            
            # Analyze progression consistency
            high_progressions = []
            for i in range(1, len(swing_highs)):
                progression = (swing_highs[i]['price'] - swing_highs[i-1]['price']) / swing_highs[i-1]['price']
                high_progressions.append(progression)
            
            low_progressions = []
            for i in range(1, len(swing_lows)):
                progression = (swing_lows[i]['price'] - swing_lows[i-1]['price']) / swing_lows[i-1]['price']
                low_progressions.append(progression)
            
            # Calculate consistency with regime-adjusted thresholds
            high_consistency = np.std(high_progressions) if high_progressions else 1
            low_consistency = np.std(low_progressions) if low_progressions else 1
            avg_consistency = (high_consistency + low_consistency) / 2
            
            # Regime-specific thresholds
            if regime_type == 'volatile':
                # More lenient thresholds for volatile markets
                strong_threshold = 0.04
                medium_threshold = 0.08
            elif regime_type == 'trending':
                # Standard thresholds for trending markets
                strong_threshold = 0.02
                medium_threshold = 0.05
            elif regime_type == 'consolidating':
                # Stricter thresholds for consolidating markets
                strong_threshold = 0.015
                medium_threshold = 0.03
            else:
                # Default balanced thresholds
                strong_threshold = 0.025
                medium_threshold = 0.06
            
            # Additional strength factors
            strength_bonus = 0
            
            # Bonus for consistent directional movement
            if high_progressions and low_progressions:
                avg_high_progression = np.mean([abs(p) for p in high_progressions])
                avg_low_progression = np.mean([abs(p) for p in low_progressions])
                
                if avg_high_progression > 0.02 or avg_low_progression > 0.02:
                    strength_bonus += 0.01  # Reduce consistency threshold
            
            # Adjust thresholds with bonus
            adjusted_strong = strong_threshold + strength_bonus
            adjusted_medium = medium_threshold + strength_bonus
            
            if avg_consistency < adjusted_strong:
                return 'strong'
            elif avg_consistency < adjusted_medium:
                return 'medium'
            else:
                return 'weak'
                
        except Exception:
            return 'unknown'
    
    def _assess_trend_consistency(self, swing_highs: List[Dict], swing_lows: List[Dict]) -> Dict[str, Any]:
        """Assess the consistency of the trend"""
        try:
            total_swings = len(swing_highs) + len(swing_lows)
            if total_swings < 4:
                return {'consistency': 'insufficient_data', 'score': 0}
            
            # Count swings that follow the trend vs those that don't
            consistent_swings = 0
            
            # Simple consistency check (would need more sophisticated logic)
            trend_direction = self._determine_trend_direction(swing_highs, swing_lows)
            
            if trend_direction == 'uptrend':
                # Count how many swings support uptrend
                for i in range(1, len(swing_highs)):
                    if swing_highs[i]['price'] > swing_highs[i-1]['price']:
                        consistent_swings += 1
                for i in range(1, len(swing_lows)):
                    if swing_lows[i]['price'] > swing_lows[i-1]['price']:
                        consistent_swings += 1
            elif trend_direction == 'downtrend':
                # Count how many swings support downtrend
                for i in range(1, len(swing_highs)):
                    if swing_highs[i]['price'] < swing_highs[i-1]['price']:
                        consistent_swings += 1
                for i in range(1, len(swing_lows)):
                    if swing_lows[i]['price'] < swing_lows[i-1]['price']:
                        consistent_swings += 1
            
            consistency_ratio = consistent_swings / (total_swings - 2) if total_swings > 2 else 0
            
            if consistency_ratio > 0.7:
                consistency_level = 'high'
            elif consistency_ratio > 0.5:
                consistency_level = 'medium'
            else:
                consistency_level = 'low'
            
            return {
                'consistency': consistency_level,
                'score': round(consistency_ratio, 2),
                'consistent_swings': consistent_swings,
                'total_evaluated_swings': total_swings - 2
            }
            
        except Exception:
            return {'consistency': 'unknown', 'score': 0}
    
    def _identify_key_levels(self, stock_data: pd.DataFrame, swing_points: Dict[str, Any]) -> Dict[str, Any]:
        """Identify key support and resistance levels from market structure"""
        try:
            if 'swing_highs' not in swing_points or 'swing_lows' not in swing_points:
                return {'error': 'No swing points available for key level identification'}
            
            swing_highs = swing_points['swing_highs']
            swing_lows = swing_points['swing_lows']
            
            # Extract resistance levels from swing highs
            resistance_levels = []
            for swing_high in swing_highs:
                resistance_levels.append({
                    'level': round(swing_high['price'], 2),
                    'date': swing_high['date'],
                    'strength': swing_high['strength'],
                    'type': 'resistance'
                })
            
            # Extract support levels from swing lows
            support_levels = []
            for swing_low in swing_lows:
                support_levels.append({
                    'level': round(swing_low['price'], 2),
                    'date': swing_low['date'],
                    'strength': swing_low['strength'],
                    'type': 'support'
                })
            
            # Find current nearest levels
            current_price = stock_data['close'].iloc[-1]
            
            # Find nearest resistance (above current price)
            nearest_resistance = None
            for level in sorted(resistance_levels, key=lambda x: abs(x['level'] - current_price)):
                if level['level'] > current_price:
                    nearest_resistance = level
                    break
            
            # Find nearest support (below current price)
            nearest_support = None
            for level in sorted(support_levels, key=lambda x: abs(x['level'] - current_price)):
                if level['level'] < current_price:
                    nearest_support = level
                    break
            
            return {
                'resistance_levels': resistance_levels,
                'support_levels': support_levels,
                'nearest_resistance': nearest_resistance,
                'nearest_support': nearest_support,
                'current_price': round(float(current_price), 2),
                'total_key_levels': len(resistance_levels) + len(support_levels)
            }
            
        except Exception as e:
            logger.error(f"[MARKET_STRUCTURE] Key level identification failed: {e}")
            return {'error': str(e)}
    
    def _assess_structure_quality(self, swing_points: Dict[str, Any], 
                                 bos_choch: Dict[str, Any], 
                                 trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the overall quality of the market structure with improved calibration"""
        try:
            quality_score = 60  # Higher base score to be less conservative
            quality_factors = []
            
            # Factor 1: Swing point density (more lenient thresholds)
            swing_density = swing_points.get('swing_density', 0)
            if 0.05 <= swing_density <= 0.15:  # Expanded optimal density range
                quality_score += 15
                quality_factors.append('optimal_swing_density')
            elif 0.02 <= swing_density < 0.05:
                quality_score += 10  # Less penalty for lower density
                quality_factors.append('adequate_swing_density')
            elif 0.15 < swing_density <= 0.2:  # More lenient on higher density
                quality_score += 5  # Small bonus instead of penalty
                quality_factors.append('active_swing_density')
            elif swing_density > 0.2:
                quality_score -= 5  # Reduced penalty for very high density
                quality_factors.append('excessive_swing_density')
            else:
                quality_score -= 3  # Reduced penalty for low density
                quality_factors.append('insufficient_swing_density')
            
            # Factor 2: Trend consistency (adjusted thresholds)
            trend_consistency = trend_analysis.get('trend_consistency', {})
            consistency_score = trend_consistency.get('score', 0)
            if consistency_score > 0.6:  # Lowered threshold from 0.7
                quality_score += 18
                quality_factors.append('high_trend_consistency')
            elif consistency_score > 0.4:  # Lowered threshold from 0.5
                quality_score += 12  # Increased bonus
                quality_factors.append('medium_trend_consistency')
            elif consistency_score > 0.2:  # Added intermediate level
                quality_score += 5
                quality_factors.append('moderate_trend_consistency')
            else:
                quality_score -= 3  # Reduced penalty
                quality_factors.append('low_trend_consistency')
            
            # Factor 3: BOS events quality (more generous)
            bos_events = bos_choch.get('bos_events', [])
            total_bos = len(bos_events)
            strong_bos = len([event for event in bos_events if event.get('strength') == 'strong'])
            medium_bos = len([event for event in bos_events if event.get('strength') == 'medium'])
            
            if strong_bos > 0:
                quality_score += 12  # Increased bonus
                quality_factors.append('strong_bos_present')
            elif medium_bos > 0:
                quality_score += 8  # New bonus for medium BOS
                quality_factors.append('medium_bos_present')
            elif total_bos > 0:
                quality_score += 4  # Small bonus for any BOS
                quality_factors.append('bos_events_present')
            
            # Factor 4: Market structure clarity (improved assessment)
            trend_direction = trend_analysis.get('trend_direction', 'unknown')
            if trend_direction in ['uptrend', 'downtrend']:
                quality_score += 12  # Increased bonus for clear trends
                quality_factors.append('clear_trend_direction')
            elif trend_direction == 'sideways':
                quality_score += 8  # Higher bonus for sideways (it's still valid structure)
                quality_factors.append('sideways_structure')
            
            # Factor 5: Data completeness bonus
            total_swings = swing_points.get('total_swings', 0)
            if total_swings >= 8:  # Good amount of swing data
                quality_score += 5
                quality_factors.append('sufficient_swing_data')
            
            # Factor 6: Structural bias coherence
            structural_bias = bos_choch.get('structural_bias', 'neutral')
            if structural_bias in ['bullish', 'bearish']:  # Any bias is better than neutral
                quality_score += 5
                quality_factors.append('coherent_structural_bias')
            
            # Normalize score
            quality_score = max(0, min(100, quality_score))
            
            # More balanced rating thresholds
            if quality_score >= 85:
                quality_rating = 'excellent'
            elif quality_score >= 70:
                quality_rating = 'good'
            elif quality_score >= 55:
                quality_rating = 'fair'
            elif quality_score >= 40:
                quality_rating = 'adequate'
            else:
                quality_rating = 'poor'
            
            return {
                'quality_score': quality_score,
                'quality_rating': quality_rating,
                'quality_factors': quality_factors,
                'assessment_criteria': {
                    'swing_density': round(swing_density, 2),
                    'trend_consistency_score': round(consistency_score, 2),
                    'strong_bos_count': strong_bos,
                    'total_bos_count': total_bos,
                    'trend_clarity': trend_direction != 'unknown',
                    'total_swings': total_swings
                }
            }
            
        except Exception as e:
            logger.error(f"[MARKET_STRUCTURE] Structure quality assessment failed: {e}")
            return {'error': str(e)}
    
    def _analyze_current_state(self, stock_data: pd.DataFrame, 
                              swing_points: Dict[str, Any], 
                              trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the current market state"""
        try:
            current_price = stock_data['close'].iloc[-1]
            
            # Recent price action
            recent_prices = stock_data['close'].tail(10)
            price_momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0] * 100
            
            # Position relative to recent swings
            swing_highs = swing_points.get('swing_highs', [])
            swing_lows = swing_points.get('swing_lows', [])
            
            recent_high = max([sh['price'] for sh in swing_highs[-3:]], default=current_price) if swing_highs else current_price
            recent_low = min([sl['price'] for sl in swing_lows[-3:]], default=current_price) if swing_lows else current_price
            
            position_in_range = ((current_price - recent_low) / (recent_high - recent_low)) if recent_high != recent_low else 0.5
            
            # Current state assessment
            if position_in_range > 0.8:
                price_position = 'near_resistance'
            elif position_in_range < 0.2:
                price_position = 'near_support'
            else:
                price_position = 'mid_range'
            
            return {
                'current_price': round(float(current_price), 2),
                'price_momentum_10d': round(float(price_momentum), 2),
                'position_in_recent_range': round(float(position_in_range), 2),
                'price_position_description': price_position,
                'recent_range_high': round(float(recent_high), 2),
                'recent_range_low': round(float(recent_low), 2),
                'trend_alignment': trend_analysis.get('trend_direction', 'unknown'),
                'structure_state': self._determine_structure_state(position_in_range, price_momentum, trend_analysis)
            }
            
        except Exception as e:
            logger.error(f"[MARKET_STRUCTURE] Current state analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_fractal_structure(self, stock_data: pd.DataFrame, swing_points: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze fractal market structure (multi-timeframe simulation)"""
        try:
            # Simulate higher timeframe structure by looking at larger patterns
            data_length = len(stock_data)
            
            # Create different "timeframes" by sampling data
            weekly_simulation = stock_data.iloc[::5] if data_length > 25 else stock_data  # Every 5th bar
            monthly_simulation = stock_data.iloc[::20] if data_length > 100 else stock_data  # Every 20th bar
            
            fractal_analysis = {
                'daily_structure': {
                    'swing_count': swing_points.get('total_swings', 0),
                    'trend': self._determine_trend_direction(
                        swing_points.get('swing_highs', []), 
                        swing_points.get('swing_lows', [])
                    )
                }
            }
            
            # Analyze weekly structure if we have enough data
            if len(weekly_simulation) >= 10:
                weekly_swings = self._identify_swing_points(weekly_simulation)
                fractal_analysis['weekly_structure'] = {
                    'swing_count': weekly_swings.get('total_swings', 0),
                    'trend': self._determine_trend_direction(
                        weekly_swings.get('swing_highs', []), 
                        weekly_swings.get('swing_lows', [])
                    )
                }
            
            # Analyze monthly structure if we have enough data
            if len(monthly_simulation) >= 6:
                monthly_swings = self._identify_swing_points(monthly_simulation)
                fractal_analysis['monthly_structure'] = {
                    'swing_count': monthly_swings.get('total_swings', 0),
                    'trend': self._determine_trend_direction(
                        monthly_swings.get('swing_highs', []), 
                        monthly_swings.get('swing_lows', [])
                    )
                }
            
            # Analyze timeframe alignment
            trends = [structure.get('trend', 'unknown') for structure in fractal_analysis.values()]
            unique_trends = set(trends)
            
            if len(unique_trends) == 1 and 'unknown' not in unique_trends:
                alignment = 'aligned'
            elif len(unique_trends) == 2 and 'unknown' in unique_trends:
                alignment = 'partially_aligned'
            else:
                alignment = 'conflicting'
            
            fractal_analysis['timeframe_alignment'] = alignment
            fractal_analysis['trend_consensus'] = max(set(trends), key=trends.count) if trends else 'unknown'
            
            return fractal_analysis
            
        except Exception as e:
            logger.error(f"[MARKET_STRUCTURE] Fractal analysis failed: {e}")
            return {'error': str(e)}
    
    def _determine_structural_bias(self, bos_events: List[Dict], choch_events: List[Dict]) -> str:
        """Determine current structural bias from BOS/CHOCH events"""
        try:
            if not bos_events and not choch_events:
                return 'neutral'
            
            # Count recent bullish vs bearish events
            recent_events = (bos_events + choch_events)[-5:]  # Last 5 events
            
            bullish_count = len([event for event in recent_events 
                               if 'bullish' in event.get('type', '')])
            bearish_count = len([event for event in recent_events 
                               if 'bearish' in event.get('type', '')])
            
            if bullish_count > bearish_count:
                return 'bullish'
            elif bearish_count > bullish_count:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception:
            return 'unknown'
    
    def _assess_trend_quality(self, direction: str, strength: str, consistency: Dict[str, Any]) -> str:
        """Assess overall trend quality"""
        try:
            if direction in ['uptrend', 'downtrend'] and strength == 'strong' and consistency.get('consistency') == 'high':
                return 'excellent'
            elif direction in ['uptrend', 'downtrend'] and (strength in ['strong', 'medium']) and consistency.get('consistency') in ['high', 'medium']:
                return 'good'
            elif direction != 'unknown':
                return 'fair'
            else:
                return 'poor'
                
        except Exception:
            return 'unknown'
    
    def _determine_structure_state(self, position_in_range: float, momentum: float, trend_analysis: Dict[str, Any]) -> str:
        """Determine current structure state"""
        try:
            trend = trend_analysis.get('trend_direction', 'unknown')
            
            if trend == 'uptrend' and momentum > 0 and position_in_range > 0.5:
                return 'trending_up'
            elif trend == 'downtrend' and momentum < 0 and position_in_range < 0.5:
                return 'trending_down'
            elif abs(momentum) < 1 and 0.3 <= position_in_range <= 0.7:
                return 'consolidating'
            elif position_in_range > 0.8:
                return 'testing_resistance'
            elif position_in_range < 0.2:
                return 'testing_support'
            else:
                return 'transitional'
                
        except Exception:
            return 'unknown'
    
    def _calculate_confidence_score(self, swing_points: Dict[str, Any], 
                                   structure_quality: Dict[str, Any], 
                                   trend_analysis: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the analysis"""
        try:
            base_confidence = 0.5
            
            # Factor in swing point quality
            swing_count = swing_points.get('total_swings', 0)
            if swing_count >= 6:
                base_confidence += 0.1
            elif swing_count >= 4:
                base_confidence += 0.05
            
            # Factor in structure quality
            quality_score = structure_quality.get('quality_score', 50) / 100
            base_confidence += quality_score * 0.3
            
            # Factor in trend clarity
            trend_direction = trend_analysis.get('trend_direction', 'unknown')
            if trend_direction in ['uptrend', 'downtrend']:
                base_confidence += 0.1
            elif trend_direction == 'sideways':
                base_confidence += 0.05
            
            return round(min(1.0, max(0.0, base_confidence)), 2)
            
        except Exception:
            return 0.5
    
    def _assess_data_quality(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess the quality of input data"""
        try:
            data_length = len(stock_data)
            
            # Check for missing values
            missing_data = stock_data.isnull().sum().sum()
            missing_percentage = (missing_data / (data_length * len(stock_data.columns))) * 100
            
            # Check data length adequacy
            if data_length >= 60:
                length_quality = 'excellent'
            elif data_length >= 30:
                length_quality = 'good'
            elif data_length >= 20:
                length_quality = 'fair'
            else:
                length_quality = 'poor'
            
            # Overall data quality score
            quality_score = 100
            if missing_percentage > 5:
                quality_score -= 20
            elif missing_percentage > 1:
                quality_score -= 10
            
            if data_length < 20:
                quality_score -= 30
            elif data_length < 30:
                quality_score -= 15
            
            return {
                'data_length': data_length,
                'missing_data_percentage': round(missing_percentage, 2),
                'length_quality': length_quality,
                'overall_quality_score': max(0, quality_score),
                'sufficient_for_analysis': data_length >= 20 and missing_percentage < 10
            }
            
        except Exception as e:
            return {'error': str(e), 'sufficient_for_analysis': False}
    
    def _build_error_result(self, error_message: str, processing_time: float = 0.0) -> Dict[str, Any]:
        """Build error result dictionary"""
        return {
            'success': False,
            'agent_name': self.name,
            'error': error_message,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'confidence_score': 0.0
        }