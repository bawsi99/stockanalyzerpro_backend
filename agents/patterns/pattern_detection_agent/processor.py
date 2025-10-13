#!/usr/bin/env python3
"""
Pattern Detection Processor - Technical Analysis Module

This module handles the detection and analysis of classic chart patterns including:
- Triangle patterns (ascending, descending, symmetrical)
- Flag and pennant patterns
- Channel and rectangle patterns
- Head and shoulders patterns
- Double top/bottom patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PatternDetectionProcessor:
    """
    Processor for detecting classic chart patterns in stock data.
    
    This processor specializes in:
    - Triangle pattern detection (ascending, descending, symmetrical)
    - Flag and pennant identification
    - Channel and rectangle patterns
    - Head and shoulders patterns
    - Double top/bottom detection
    """
    
    def __init__(self):
        self.name = "pattern_detection"
        self.description = "Detects and analyzes classic chart patterns"
        self.version = "1.0.0"
    
    def process_pattern_detection_data(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Process pattern detection analysis from stock data.
        
        Args:
            stock_data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing comprehensive pattern detection analysis
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"[PATTERN_DETECTION] Starting pattern detection analysis")
            
            if stock_data is None or stock_data.empty or len(stock_data) < 20:
                return self._build_error_result("Insufficient data for pattern detection analysis")
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in stock_data.columns]
            if missing_columns:
                return self._build_error_result(f"Missing required columns: {missing_columns}")
            
            # 1. Triangle Pattern Detection
            triangle_patterns = self._detect_triangle_patterns(stock_data)
            
            # 2. Flag and Pennant Detection
            flag_pennant_patterns = self._detect_flag_pennant_patterns(stock_data)
            
            # 3. Channel and Rectangle Detection
            channel_patterns = self._detect_channel_patterns(stock_data)
            
            # 4. Head and Shoulders Detection
            head_shoulders_patterns = self._detect_head_shoulders_patterns(stock_data)
            
            # 5. Double Top/Bottom Detection
            double_patterns = self._detect_double_patterns(stock_data)
            
            # 6. Pattern Summary and Analysis
            pattern_summary = self._analyze_pattern_summary(
                triangle_patterns, flag_pennant_patterns, channel_patterns, 
                head_shoulders_patterns, double_patterns
            )
            
            # 7. Formation Stage Analysis
            formation_stage = self._analyze_formation_stage(stock_data, pattern_summary)
            
            # 8. Key Levels from Patterns
            key_levels = self._identify_pattern_levels(
                stock_data, triangle_patterns, flag_pennant_patterns, 
                channel_patterns, head_shoulders_patterns, double_patterns
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Build comprehensive result
            result = {
                'success': True,
                'agent_name': self.name,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                
                # Core Pattern Detection
                'detected_patterns': self._compile_all_patterns(
                    triangle_patterns, flag_pennant_patterns, channel_patterns,
                    head_shoulders_patterns, double_patterns
                ),
                'pattern_summary': pattern_summary,
                'formation_stage': formation_stage,
                'key_levels': key_levels,
                
                # Statistics
                'analysis_period_days': len(stock_data),
                'total_patterns_detected': pattern_summary.get('total_patterns', 0),
                
                # Confidence Metrics
                'confidence_score': self._calculate_confidence_score(pattern_summary, formation_stage),
                'data_quality': self._assess_data_quality(stock_data)
            }
            
            logger.info(f"[PATTERN_DETECTION] Analysis completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"[PATTERN_DETECTION] Analysis failed: {str(e)}")
            return self._build_error_result(str(e), processing_time)
    
    def _detect_triangle_patterns(self, stock_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        try:
            patterns = []
            highs = stock_data['high'].values
            lows = stock_data['low'].values
            dates = stock_data.index
            
            if len(stock_data) < 20:
                return patterns
            
            # Look for triangle patterns in recent data
            for start_idx in range(10, len(stock_data) - 10):
                window_size = min(20, len(stock_data) - start_idx)
                if window_size < 10:
                    continue
                
                end_idx = start_idx + window_size
                window_highs = highs[start_idx:end_idx]
                window_lows = lows[start_idx:end_idx]
                
                # Calculate trend lines
                high_trend = self._calculate_trend_line(window_highs)
                low_trend = self._calculate_trend_line(window_lows)
                
                if high_trend and low_trend:
                    # Classify triangle type
                    triangle_type = self._classify_triangle(high_trend, low_trend)
                    
                    if triangle_type:
                        # Calculate pattern metrics
                        completion = self._calculate_pattern_completion(window_highs, window_lows, triangle_type)
                        reliability = self._assess_pattern_reliability(window_highs, window_lows, triangle_type)
                        
                        pattern = {
                            'pattern_name': triangle_type,
                            'pattern_type': 'continuation' if 'symmetrical' in triangle_type else 'continuation',
                            'completion_status': 'forming' if completion < 80 else 'completed',
                            'completion_percentage': completion,
                            'reliability': reliability,
                            'pattern_quality': 'strong' if reliability == 'high' and completion > 70 else 'medium' if reliability == 'medium' else 'weak',
                            'start_date': str(dates[start_idx]),
                            'pattern_data': {
                                'high_trend': high_trend,
                                'low_trend': low_trend,
                                'apex_price': (high_trend[-1] + low_trend[-1]) / 2
                            }
                        }
                        patterns.append(pattern)
                        
                        # Only keep the most significant patterns
                        if len(patterns) > 3:
                            patterns = sorted(patterns, key=lambda x: x['completion_percentage'], reverse=True)[:3]
            
            return patterns
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION] Triangle detection failed: {e}")
            return []
    
    def _detect_flag_pennant_patterns(self, stock_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect flag and pennant patterns"""
        try:
            patterns = []
            highs = stock_data['high'].values
            lows = stock_data['low'].values
            volumes = stock_data['volume'].values if 'volume' in stock_data.columns else None
            dates = stock_data.index
            
            if len(stock_data) < 15:
                return patterns
            
            # Look for flag/pennant patterns (short-term consolidation after strong moves)
            for i in range(10, len(stock_data) - 5):
                # Check for strong prior move (flagpole)
                flagpole_length = min(10, i)
                if flagpole_length < 5:
                    continue
                
                flagpole_start = i - flagpole_length
                flagpole_move = (highs[i] - lows[flagpole_start]) / lows[flagpole_start]
                
                # Need at least 3% move to consider
                if abs(flagpole_move) > 0.03:
                    # Check consolidation period after the move
                    consolidation_end = min(i + 8, len(stock_data))
                    consolidation_highs = highs[i:consolidation_end]
                    consolidation_lows = lows[i:consolidation_end]
                    
                    if len(consolidation_highs) >= 5:
                        # Check if it's a flag (parallel lines) or pennant (converging lines)
                        consolidation_range = np.max(consolidation_highs) - np.min(consolidation_lows)
                        flagpole_range = highs[i] - lows[flagpole_start]
                        
                        # Consolidation should be smaller than the flagpole
                        if consolidation_range < flagpole_range * 0.5:
                            pattern_type = 'flag' if consolidation_range > flagpole_range * 0.2 else 'pennant'
                            
                            # Determine direction
                            direction = 'bullish' if flagpole_move > 0 else 'bearish'
                            pattern_name = f'{direction}_{pattern_type}'
                            
                            # Volume confirmation (if available)
                            volume_confirmation = self._check_volume_confirmation(volumes, flagpole_start, i, consolidation_end) if volumes is not None else 'unknown'
                            
                            pattern = {
                                'pattern_name': pattern_name,
                                'pattern_type': 'continuation',
                                'completion_status': 'forming',
                                'completion_percentage': 75,  # Flags/pennants are usually near completion when detected
                                'reliability': 'high' if abs(flagpole_move) > 0.05 and volume_confirmation == 'present' else 'medium',
                                'pattern_quality': 'strong' if abs(flagpole_move) > 0.05 else 'medium',
                                'start_date': str(dates[flagpole_start]),
                                'pattern_data': {
                                    'flagpole_move': flagpole_move,
                                    'consolidation_range': consolidation_range,
                                    'volume_confirmation': volume_confirmation
                                }
                            }
                            patterns.append(pattern)
                            
                            # Limit to most significant patterns
                            if len(patterns) > 2:
                                patterns = sorted(patterns, key=lambda x: abs(x['pattern_data']['flagpole_move']), reverse=True)[:2]
            
            return patterns
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION] Flag/pennant detection failed: {e}")
            return []
    
    def _detect_channel_patterns(self, stock_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect channel and rectangle patterns"""
        try:
            patterns = []
            highs = stock_data['high'].values
            lows = stock_data['low'].values
            dates = stock_data.index
            
            if len(stock_data) < 20:
                return patterns
            
            # Look for channel patterns (parallel support and resistance)
            for start_idx in range(0, len(stock_data) - 20, 5):  # Step by 5 to avoid too many overlapping patterns
                window_size = min(25, len(stock_data) - start_idx)
                if window_size < 15:
                    continue
                
                end_idx = start_idx + window_size
                window_highs = highs[start_idx:end_idx]
                window_lows = lows[start_idx:end_idx]
                
                # Check for consistent support and resistance levels
                high_level = np.mean(np.sort(window_highs)[-3:])  # Average of top 3 highs
                low_level = np.mean(np.sort(window_lows)[:3])     # Average of bottom 3 lows
                
                # Check if price stayed within the channel
                channel_width = high_level - low_level
                if channel_width > 0:
                    # Count how many times price respected the levels
                    resistance_tests = np.sum((window_highs >= high_level * 0.98) & (window_highs <= high_level * 1.02))
                    support_tests = np.sum((window_lows <= low_level * 1.02) & (window_lows >= low_level * 0.98))
                    
                    # Need at least 2 tests of each level
                    if resistance_tests >= 2 and support_tests >= 2:
                        # Determine channel type
                        price_trend = (window_highs[-1] - window_highs[0]) / window_highs[0]
                        
                        if abs(price_trend) < 0.05:
                            pattern_name = 'rectangle'
                            pattern_type = 'consolidation'
                        elif price_trend > 0.05:
                            pattern_name = 'ascending_channel'
                            pattern_type = 'continuation'
                        else:
                            pattern_name = 'descending_channel'
                            pattern_type = 'continuation'
                        
                        # Calculate completion and reliability
                        completion = min(90, resistance_tests * 10 + support_tests * 10)
                        reliability = 'high' if (resistance_tests + support_tests) >= 6 else 'medium'
                        
                        pattern = {
                            'pattern_name': pattern_name,
                            'pattern_type': pattern_type,
                            'completion_status': 'forming',
                            'completion_percentage': completion,
                            'reliability': reliability,
                            'pattern_quality': 'strong' if reliability == 'high' and completion > 70 else 'medium',
                            'start_date': str(dates[start_idx]),
                            'pattern_data': {
                                'resistance_level': high_level,
                                'support_level': low_level,
                                'channel_width': channel_width,
                                'resistance_tests': resistance_tests,
                                'support_tests': support_tests
                            }
                        }
                        patterns.append(pattern)
                        
                        # Limit patterns
                        if len(patterns) > 2:
                            break
            
            return patterns
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION] Channel detection failed: {e}")
            return []
    
    def _detect_head_shoulders_patterns(self, stock_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect head and shoulders patterns"""
        try:
            patterns = []
            highs = stock_data['high'].values
            lows = stock_data['low'].values
            dates = stock_data.index
            
            if len(stock_data) < 25:
                return patterns
            
            # Look for head and shoulders patterns
            for i in range(10, len(stock_data) - 15):
                # Look for three peaks pattern
                left_shoulder_start = max(0, i - 10)
                head_center = i
                right_shoulder_end = min(i + 15, len(stock_data))
                
                if right_shoulder_end - left_shoulder_start < 20:
                    continue
                
                # Identify peaks in the range
                peaks = []
                for j in range(left_shoulder_start + 2, right_shoulder_end - 2):
                    if (highs[j] > highs[j-1] and highs[j] > highs[j+1] and 
                        highs[j] > highs[j-2] and highs[j] > highs[j+2]):
                        peaks.append((j, highs[j]))
                
                # Need at least 3 peaks for head and shoulders
                if len(peaks) >= 3:
                    # Sort peaks by height
                    peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)
                    
                    # Check if highest peak is in the middle (head)
                    head_idx, head_price = peaks_sorted[0]
                    
                    # Find left and right shoulders
                    left_shoulders = [p for p in peaks if p[0] < head_idx]
                    right_shoulders = [p for p in peaks if p[0] > head_idx]
                    
                    if left_shoulders and right_shoulders:
                        left_shoulder = max(left_shoulders, key=lambda x: x[1])
                        right_shoulder = max(right_shoulders, key=lambda x: x[1])
                        
                        # Check if shoulders are roughly equal and lower than head
                        shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1]
                        head_higher = (head_price > left_shoulder[1] * 1.02 and 
                                     head_price > right_shoulder[1] * 1.02)
                        
                        if shoulder_diff < 0.05 and head_higher:
                            # Find neckline (support between shoulders)
                            neckline_start = left_shoulder[0]
                            neckline_end = right_shoulder[0]
                            neckline_lows = lows[neckline_start:neckline_end+1]
                            neckline_level = np.min(neckline_lows)
                            
                            # Determine if it's regular or inverse H&S
                            current_price = highs[-1]
                            if head_price > neckline_level:
                                pattern_name = 'head_and_shoulders'
                                pattern_type = 'reversal'
                            else:
                                pattern_name = 'inverse_head_and_shoulders'
                                pattern_type = 'reversal'
                            
                            # Calculate reliability
                            reliability = 'high' if shoulder_diff < 0.03 else 'medium'
                            
                            pattern = {
                                'pattern_name': pattern_name,
                                'pattern_type': pattern_type,
                                'completion_status': 'forming',
                                'completion_percentage': 80,
                                'reliability': reliability,
                                'pattern_quality': 'strong' if reliability == 'high' else 'medium',
                                'start_date': str(dates[left_shoulder_start]),
                                'pattern_data': {
                                    'head_price': head_price,
                                    'left_shoulder_price': left_shoulder[1],
                                    'right_shoulder_price': right_shoulder[1],
                                    'neckline_level': neckline_level,
                                    'shoulder_symmetry': 1 - shoulder_diff
                                }
                            }
                            patterns.append(pattern)
                            
                            # Only keep one H&S pattern to avoid duplicates
                            if len(patterns) > 0:
                                break
            
            return patterns
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION] Head and shoulders detection failed: {e}")
            return []
    
    def _detect_double_patterns(self, stock_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect double top and double bottom patterns"""
        try:
            patterns = []
            highs = stock_data['high'].values
            lows = stock_data['low'].values
            dates = stock_data.index
            
            if len(stock_data) < 20:
                return patterns
            
            # Look for double tops
            for i in range(10, len(stock_data) - 10):
                current_high = highs[i]
                
                # Look for another high at similar level (within recent data)
                for j in range(i + 5, min(i + 20, len(stock_data))):
                    if abs(highs[j] - current_high) / current_high < 0.03:  # Within 3%
                        # Check if there's a valley between peaks
                        valley_start = i
                        valley_end = j
                        valley_low = np.min(lows[valley_start:valley_end+1])
                        
                        # Valley should be at least 3% below the peaks
                        if (current_high - valley_low) / current_high > 0.03:
                            pattern = {
                                'pattern_name': 'double_top',
                                'pattern_type': 'reversal',
                                'completion_status': 'completed',
                                'completion_percentage': 90,
                                'reliability': 'high',
                                'pattern_quality': 'strong',
                                'start_date': str(dates[i]),
                                'pattern_data': {
                                    'first_peak': current_high,
                                    'second_peak': highs[j],
                                    'valley_low': valley_low,
                                    'peak_similarity': 1 - abs(highs[j] - current_high) / current_high
                                }
                            }
                            patterns.append(pattern)
                            break
            
            # Look for double bottoms
            for i in range(10, len(stock_data) - 10):
                current_low = lows[i]
                
                # Look for another low at similar level
                for j in range(i + 5, min(i + 20, len(stock_data))):
                    if abs(lows[j] - current_low) / current_low < 0.03:  # Within 3%
                        # Check if there's a peak between valleys
                        peak_start = i
                        peak_end = j
                        peak_high = np.max(highs[peak_start:peak_end+1])
                        
                        # Peak should be at least 3% above the lows
                        if (peak_high - current_low) / current_low > 0.03:
                            pattern = {
                                'pattern_name': 'double_bottom',
                                'pattern_type': 'reversal',
                                'completion_status': 'completed',
                                'completion_percentage': 90,
                                'reliability': 'high',
                                'pattern_quality': 'strong',
                                'start_date': str(dates[i]),
                                'pattern_data': {
                                    'first_bottom': current_low,
                                    'second_bottom': lows[j],
                                    'peak_high': peak_high,
                                    'bottom_similarity': 1 - abs(lows[j] - current_low) / current_low
                                }
                            }
                            patterns.append(pattern)
                            break
            
            # Limit to most significant patterns
            if len(patterns) > 2:
                patterns = sorted(patterns, key=lambda x: x['completion_percentage'], reverse=True)[:2]
            
            return patterns
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION] Double pattern detection failed: {e}")
            return []
    
    def _calculate_trend_line(self, values: np.ndarray) -> Optional[List[float]]:
        """Calculate trend line for a series of values"""
        try:
            if len(values) < 3:
                return None
            
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)
            trend_line = [slope * i + intercept for i in x]
            return trend_line
            
        except Exception:
            return None
    
    def _classify_triangle(self, high_trend: List[float], low_trend: List[float]) -> Optional[str]:
        """Classify triangle type based on trend lines"""
        try:
            high_slope = high_trend[-1] - high_trend[0]
            low_slope = low_trend[-1] - low_trend[0]
            
            # Normalize slopes
            high_slope_norm = high_slope / (high_trend[0] if high_trend[0] != 0 else 1)
            low_slope_norm = low_slope / (low_trend[0] if low_trend[0] != 0 else 1)
            
            if abs(high_slope_norm) < 0.01 and low_slope_norm > 0.01:
                return 'ascending_triangle'
            elif high_slope_norm < -0.01 and abs(low_slope_norm) < 0.01:
                return 'descending_triangle'
            elif high_slope_norm < -0.01 and low_slope_norm > 0.01:
                return 'symmetrical_triangle'
            
            return None
            
        except Exception:
            return None
    
    def _calculate_pattern_completion(self, highs: np.ndarray, lows: np.ndarray, pattern_type: str) -> float:
        """Calculate pattern completion percentage"""
        try:
            # Simple completion calculation based on pattern development
            data_length = len(highs)
            if data_length < 10:
                return 30
            elif data_length < 15:
                return 50
            elif data_length < 20:
                return 70
            else:
                return 85
                
        except Exception:
            return 50
    
    def _assess_pattern_reliability(self, highs: np.ndarray, lows: np.ndarray, pattern_type: str) -> str:
        """Assess pattern reliability"""
        try:
            # Calculate price volatility within pattern
            price_range = np.max(highs) - np.min(lows)
            avg_price = (np.max(highs) + np.min(lows)) / 2
            volatility = price_range / avg_price if avg_price > 0 else 1
            
            if volatility < 0.05:
                return 'high'
            elif volatility < 0.10:
                return 'medium'
            else:
                return 'low'
                
        except Exception:
            return 'low'
    
    def _check_volume_confirmation(self, volumes: np.ndarray, flagpole_start: int, flagpole_end: int, consolidation_end: int) -> str:
        """Check volume confirmation for flag/pennant patterns"""
        try:
            if volumes is None or len(volumes) <= consolidation_end:
                return 'unknown'
            
            flagpole_volume = np.mean(volumes[flagpole_start:flagpole_end])
            consolidation_volume = np.mean(volumes[flagpole_end:consolidation_end])
            
            # Volume should decrease during consolidation
            if consolidation_volume < flagpole_volume * 0.8:
                return 'present'
            else:
                return 'absent'
                
        except Exception:
            return 'unknown'
    
    def _compile_all_patterns(self, *pattern_lists) -> List[Dict[str, Any]]:
        """Compile all detected patterns into a single list"""
        all_patterns = []
        for pattern_list in pattern_lists:
            all_patterns.extend(pattern_list)
        
        # Sort by reliability and completion
        all_patterns.sort(key=lambda x: (x['reliability'] == 'high', x['completion_percentage']), reverse=True)
        
        # Limit to top patterns to avoid overwhelming output
        return all_patterns[:5]
    
    def _analyze_pattern_summary(self, *pattern_lists) -> Dict[str, Any]:
        """Analyze overall pattern summary"""
        try:
            all_patterns = []
            for pattern_list in pattern_lists:
                all_patterns.extend(pattern_list)
            
            if not all_patterns:
                return {
                    'total_patterns': 0,
                    'dominant_pattern': 'none',
                    'pattern_confluence': 'none',
                    'overall_bias': 'neutral'
                }
            
            # Count pattern types
            reversal_patterns = [p for p in all_patterns if p['pattern_type'] == 'reversal']
            continuation_patterns = [p for p in all_patterns if p['pattern_type'] == 'continuation']
            
            # Determine dominant pattern
            if all_patterns:
                dominant_pattern = max(all_patterns, key=lambda x: x['completion_percentage'])['pattern_name']
            else:
                dominant_pattern = 'none'
            
            # Assess confluence
            if len(all_patterns) >= 3:
                confluence = 'high'
            elif len(all_patterns) == 2:
                confluence = 'medium'
            elif len(all_patterns) == 1:
                confluence = 'low'
            else:
                confluence = 'none'
            
            # Determine bias
            bullish_patterns = [p for p in all_patterns if 'bullish' in p['pattern_name'] or 'ascending' in p['pattern_name'] or 'inverse' in p['pattern_name']]
            bearish_patterns = [p for p in all_patterns if 'bearish' in p['pattern_name'] or 'descending' in p['pattern_name'] or ('head_and_shoulders' in p['pattern_name'] and 'inverse' not in p['pattern_name'])]
            
            if len(bullish_patterns) > len(bearish_patterns):
                bias = 'bullish'
            elif len(bearish_patterns) > len(bullish_patterns):
                bias = 'bearish'
            else:
                bias = 'neutral'
            
            return {
                'total_patterns': len(all_patterns),
                'dominant_pattern': dominant_pattern,
                'pattern_confluence': confluence,
                'overall_bias': bias,
                'reversal_patterns': len(reversal_patterns),
                'continuation_patterns': len(continuation_patterns)
            }
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION] Pattern summary analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_formation_stage(self, stock_data: pd.DataFrame, pattern_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pattern formation stage"""
        try:
            total_patterns = pattern_summary.get('total_patterns', 0)
            
            if total_patterns == 0:
                return {
                    'primary_stage': 'early_formation',
                    'pattern_maturity': 'early',
                    'breakout_potential': 'low'
                }
            elif total_patterns == 1:
                return {
                    'primary_stage': 'development',
                    'pattern_maturity': 'developing',
                    'breakout_potential': 'medium'
                }
            else:
                return {
                    'primary_stage': 'completion',
                    'pattern_maturity': 'mature',
                    'breakout_potential': 'high'
                }
                
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION] Formation stage analysis failed: {e}")
            return {'error': str(e)}
    
    def _identify_pattern_levels(self, stock_data: pd.DataFrame, *pattern_lists) -> Dict[str, Any]:
        """Identify key levels from detected patterns"""
        try:
            current_price = stock_data['close'].iloc[-1]
            
            # Collect levels from all patterns
            resistance_levels = []
            support_levels = []
            breakout_levels = []
            
            for pattern_list in pattern_lists:
                for pattern in pattern_list:
                    pattern_data = pattern.get('pattern_data', {})
                    
                    # Extract levels based on pattern type
                    if 'resistance_level' in pattern_data:
                        resistance_levels.append(pattern_data['resistance_level'])
                    if 'support_level' in pattern_data:
                        support_levels.append(pattern_data['support_level'])
                    if 'neckline_level' in pattern_data:
                        breakout_levels.append(pattern_data['neckline_level'])
                    if 'apex_price' in pattern_data:
                        breakout_levels.append(pattern_data['apex_price'])
            
            # Find nearest levels
            nearest_resistance = None
            nearest_support = None
            breakout_level = None
            
            if resistance_levels:
                resistance_above = [r for r in resistance_levels if r > current_price]
                nearest_resistance = min(resistance_above) if resistance_above else max(resistance_levels)
            
            if support_levels:
                support_below = [s for s in support_levels if s < current_price]
                nearest_support = max(support_below) if support_below else min(support_levels)
            
            if breakout_levels:
                breakout_level = min(breakout_levels, key=lambda x: abs(x - current_price))
            
            return {
                'nearest_resistance': nearest_resistance or current_price * 1.05,
                'nearest_support': nearest_support or current_price * 0.95,
                'breakout_level': breakout_level or current_price,
                'current_price': float(current_price),
                'total_levels': len(resistance_levels) + len(support_levels)
            }
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION] Level identification failed: {e}")
            return {'error': str(e)}
    
    def _calculate_confidence_score(self, pattern_summary: Dict[str, Any], formation_stage: Dict[str, Any]) -> float:
        """Calculate overall confidence score for pattern detection"""
        try:
            base_confidence = 0.3
            
            # Factor in number of patterns
            total_patterns = pattern_summary.get('total_patterns', 0)
            if total_patterns >= 3:
                base_confidence += 0.3
            elif total_patterns >= 2:
                base_confidence += 0.2
            elif total_patterns >= 1:
                base_confidence += 0.1
            
            # Factor in pattern confluence
            confluence = pattern_summary.get('pattern_confluence', 'none')
            if confluence == 'high':
                base_confidence += 0.2
            elif confluence == 'medium':
                base_confidence += 0.1
            
            # Factor in formation maturity
            maturity = formation_stage.get('pattern_maturity', 'early')
            if maturity == 'mature':
                base_confidence += 0.2
            elif maturity == 'developing':
                base_confidence += 0.1
            
            return min(1.0, max(0.0, base_confidence))
            
        except Exception:
            return 0.5
    
    def _assess_data_quality(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess the quality of input data for pattern detection"""
        try:
            data_length = len(stock_data)
            
            # Check for missing values
            missing_data = stock_data.isnull().sum().sum()
            missing_percentage = (missing_data / (data_length * len(stock_data.columns))) * 100
            
            # Check data length adequacy for pattern detection
            if data_length >= 60:
                length_quality = 'excellent'
            elif data_length >= 40:
                length_quality = 'good'
            elif data_length >= 25:
                length_quality = 'fair'
            else:
                length_quality = 'poor'
            
            # Overall quality score
            quality_score = 100
            if missing_percentage > 5:
                quality_score -= 30
            elif missing_percentage > 1:
                quality_score -= 15
            
            if data_length < 25:
                quality_score -= 40
            elif data_length < 40:
                quality_score -= 20
            
            return {
                'data_length': data_length,
                'missing_data_percentage': missing_percentage,
                'length_quality': length_quality,
                'overall_quality_score': max(0, quality_score),
                'sufficient_for_analysis': data_length >= 25 and missing_percentage < 10
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