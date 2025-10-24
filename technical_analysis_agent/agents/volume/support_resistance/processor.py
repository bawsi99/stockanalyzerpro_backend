#!/usr/bin/env python3
"""
Volume-Based Support/Resistance Agent - Data Processing Module

Identifies volume-validated price levels for support and resistance analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class SupportResistanceProcessor:
    def __init__(self):
        self.price_level_tolerance = 0.02  # 2% tolerance for level grouping
        self.min_volume_threshold = 0.1  # Minimum 10% of average volume
        self.min_test_count = 2  # Minimum number of tests to validate level
        self.volume_bins = 100  # Granularity for volume-at-price analysis
    
    def process_support_resistance_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Main processing function for support/resistance analysis"""
        try:
            # 1. Calculate Volume-at-Price (VAP) data
            vap_analysis = self._calculate_volume_at_price(data)
            
            # 2. Identify potential support/resistance levels
            potential_levels = self._identify_potential_levels(data, vap_analysis)
            
            # 3. Validate levels through testing history
            validated_levels = self._validate_levels_with_history(data, potential_levels)
            
            # 4. Analyze current position relative to levels
            current_analysis = self._analyze_current_position(data, validated_levels)
            
            # 5. Calculate level strength and reliability
            level_ratings = self._calculate_level_ratings(validated_levels, vap_analysis)
            
            # 6. Generate trading implications
            trading_implications = self._generate_trading_implications(
                current_analysis, level_ratings, data['close'].iloc[-1]
            )
            
            # 7. Quality assessment
            quality = self._assess_analysis_quality(validated_levels, vap_analysis)
            
            return {
                'volume_at_price_analysis': vap_analysis,
                'potential_levels': potential_levels,
                'validated_levels': validated_levels,
                'current_position_analysis': current_analysis,
                'level_ratings': level_ratings,
                'trading_implications': trading_implications,
                'quality_assessment': quality,
                'volume_based_support_levels': self._format_support_levels(validated_levels, 'support'),
                'volume_based_resistance_levels': self._format_resistance_levels(validated_levels, 'resistance')
            }
            
        except Exception as e:
            return {'error': f"Support/resistance analysis failed: {str(e)}"}
    
    def _calculate_volume_at_price(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume distribution at different price levels"""
        try:
            # Get overall price range
            price_min = data['low'].min()
            price_max = data['high'].max()
            price_range = price_max - price_min
            
            if price_range == 0:
                return {'error': 'No price range available'}
            
            # Create price bins
            price_bins = np.linspace(price_min, price_max, self.volume_bins)
            bin_width = price_range / self.volume_bins
            
            # Calculate volume at each price level
            volume_profile = []
            
            for i in range(len(price_bins) - 1):
                bin_low = price_bins[i]
                bin_high = price_bins[i + 1]
                bin_center = (bin_low + bin_high) / 2
                
                # Find overlapping price ranges
                overlapping_data = data[
                    (data['low'] <= bin_high) & (data['high'] >= bin_low)
                ]
                
                total_volume = 0
                touch_count = 0
                
                for _, row in overlapping_data.iterrows():
                    # Calculate overlap proportion
                    if row['high'] != row['low']:
                        overlap_low = max(bin_low, row['low'])
                        overlap_high = min(bin_high, row['high'])
                        overlap_range = max(0, overlap_high - overlap_low)
                        proportion = overlap_range / (row['high'] - row['low'])
                        volume_contribution = row['volume'] * proportion
                    else:
                        # Single price point
                        if bin_low <= row['close'] <= bin_high:
                            volume_contribution = row['volume']
                        else:
                            volume_contribution = 0
                    
                    total_volume += volume_contribution
                    if volume_contribution > 0:
                        touch_count += 1
                
                volume_profile.append({
                    'price_level': bin_center,
                    'volume': total_volume,
                    'touch_count': touch_count,
                    'price_range': [bin_low, bin_high],
                    'volume_density': total_volume / bin_width if bin_width > 0 else 0
                })
            
            # Sort by volume and identify high-volume nodes
            volume_profile.sort(key=lambda x: x['volume'], reverse=True)
            
            # Calculate statistics
            total_volume = sum(level['volume'] for level in volume_profile)
            avg_volume_per_level = total_volume / len(volume_profile) if volume_profile else 0
            
            # Identify significant volume levels (above average)
            significant_levels = [
                level for level in volume_profile 
                if level['volume'] > avg_volume_per_level * 1.5
            ]
            
            return {
                'volume_profile': volume_profile,
                'significant_volume_levels': significant_levels,
                'total_volume': total_volume,
                'average_volume_per_level': avg_volume_per_level,
                'price_range': [price_min, price_max],
                'highest_volume_level': volume_profile[0] if volume_profile else None
            }
            
        except Exception as e:
            return {'error': f"Volume-at-price calculation failed: {str(e)}"}
    
    def _identify_potential_levels(self, data: pd.DataFrame, vap_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Identify potential support/resistance levels"""
        try:
            if 'error' in vap_analysis:
                return {'error': 'VAP analysis unavailable'}
            
            significant_levels = vap_analysis.get('significant_volume_levels', [])
            
            # Also consider swing highs and lows
            swing_levels = self._identify_swing_levels(data)
            
            # Combine volume levels with swing levels
            combined_levels = []
            
            # Process volume-based levels
            for level in significant_levels:
                combined_levels.append({
                    'price': level['price_level'],
                    'volume': level['volume'],
                    'type': 'volume_node',
                    'strength': self._calculate_initial_strength(level),
                    'touch_count': level['touch_count']
                })
            
            # Process swing levels
            for level in swing_levels:
                # Check if this swing level has volume support
                volume_support = self._get_volume_at_level(level['price'], vap_analysis)
                
                combined_levels.append({
                    'price': level['price'],
                    'volume': volume_support,
                    'type': level['type'],
                    'strength': level['strength'],
                    'touch_count': level['touch_count'],
                    'swing_data': level
                })
            
            # Remove duplicates (levels too close together)
            deduplicated_levels = self._deduplicate_levels(combined_levels)
            
            # Sort by strength
            deduplicated_levels.sort(key=lambda x: x['strength'], reverse=True)
            
            return {
                'combined_levels': deduplicated_levels,
                'volume_levels_count': len(significant_levels),
                'swing_levels_count': len(swing_levels),
                'total_potential_levels': len(deduplicated_levels)
            }
            
        except Exception as e:
            return {'error': f"Level identification failed: {str(e)}"}
    
    def _identify_swing_levels(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify swing highs and lows"""
        try:
            swing_levels = []
            window = 5  # Look at 5 days before and after
            
            # Identify swing highs
            for i in range(window, len(data) - window):
                current_high = data['high'].iloc[i]
                is_swing_high = True
                
                # Check if it's higher than surrounding days
                for j in range(i - window, i + window + 1):
                    if j != i and data['high'].iloc[j] >= current_high:
                        is_swing_high = False
                        break
                
                if is_swing_high:
                    swing_levels.append({
                        'price': current_high,
                        'type': 'swing_high',
                        'date': data.index[i],
                        'strength': self._calculate_swing_strength(data, i, 'high'),
                        'touch_count': 1  # Initial count
                    })
            
            # Identify swing lows
            for i in range(window, len(data) - window):
                current_low = data['low'].iloc[i]
                is_swing_low = True
                
                # Check if it's lower than surrounding days
                for j in range(i - window, i + window + 1):
                    if j != i and data['low'].iloc[j] <= current_low:
                        is_swing_low = False
                        break
                
                if is_swing_low:
                    swing_levels.append({
                        'price': current_low,
                        'type': 'swing_low',
                        'date': data.index[i],
                        'strength': self._calculate_swing_strength(data, i, 'low'),
                        'touch_count': 1  # Initial count
                    })
            
            return swing_levels
            
        except Exception as e:
            print(f"Swing level identification failed: {e}")
            return []
    
    def _validate_levels_with_history(self, data: pd.DataFrame, potential_levels: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate levels by checking historical testing"""
        try:
            if 'error' in potential_levels:
                return []
            
            combined_levels = potential_levels.get('combined_levels', [])
            validated_levels = []
            
            for level in combined_levels:
                level_price = level['price']
                tolerance = level_price * self.price_level_tolerance
                
                # Count tests of this level
                tests = self._count_level_tests(data, level_price, tolerance)
                
                # Calculate success rate
                success_rate = self._calculate_level_success_rate(data, level_price, tolerance, tests)
                
                # Determine level type (support or resistance)
                level_type = self._determine_level_type(data, level_price, tests)
                
                if tests['total_tests'] >= self.min_test_count:
                    validated_level = {
                        'price': level_price,
                        'type': level_type,
                        'strength': level['strength'],
                        'volume': level['volume'],
                        'total_tests': tests['total_tests'],
                        'successful_tests': tests['successful_tests'],
                        'failed_tests': tests['failed_tests'],
                        'success_rate': success_rate,
                        'last_test_date': tests['last_test_date'],
                        'reliability': self._assess_level_reliability(success_rate, tests['total_tests']),
                        'original_type': level['type']
                    }
                    validated_levels.append(validated_level)
            
            # Sort by reliability and strength
            validated_levels.sort(key=lambda x: (x['reliability'], x['strength']), reverse=True)
            
            return validated_levels
            
        except Exception as e:
            print(f"Level validation failed: {e}")
            return []
    
    def _count_level_tests(self, data: pd.DataFrame, level_price: float, tolerance: float) -> Dict[str, Any]:
        """Count how many times a level has been tested"""
        try:
            tests = []
            successful_tests = 0
            failed_tests = 0
            
            for i, (date, row) in enumerate(data.iterrows()):
                # Check if price tested the level (came within tolerance)
                if (row['low'] <= level_price + tolerance and 
                    row['high'] >= level_price - tolerance):
                    
                    # Determine if test was successful (level held)
                    if i < len(data) - 1:  # Not the last day
                        next_day = data.iloc[i + 1]
                        
                        # For support: successful if next day doesn't close below level
                        # For resistance: successful if next day doesn't close above level
                        if row['low'] <= level_price <= row['high']:
                            # Direct test
                            if data['close'].iloc[i + 1] > level_price:
                                success = True
                                successful_tests += 1
                            else:
                                success = False
                                failed_tests += 1
                        else:
                            # Indirect test (just touched)
                            success = True
                            successful_tests += 1
                        
                        tests.append({
                            'date': date,
                            'price_range': [row['low'], row['high']],
                            'success': success,
                            'volume': row['volume']
                        })
            
            return {
                'total_tests': len(tests),
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'test_details': tests,
                'last_test_date': tests[-1]['date'] if tests else None
            }
            
        except Exception as e:
            return {
                'total_tests': 0,
                'successful_tests': 0,
                'failed_tests': 0,
                'test_details': [],
                'last_test_date': None
            }
    
    def _calculate_level_success_rate(self, data: pd.DataFrame, level_price: float, 
                                    tolerance: float, tests: Dict[str, Any]) -> float:
        """Calculate success rate of level holds"""
        total_tests = tests['total_tests']
        successful_tests = tests['successful_tests']
        
        if total_tests == 0:
            return 0.0
        
        return successful_tests / total_tests
    
    def _determine_level_type(self, data: pd.DataFrame, level_price: float, 
                            tests: Dict[str, Any]) -> str:
        """Determine if level acts as support or resistance"""
        if not tests['test_details']:
            return 'unknown'
        
        support_tests = 0
        resistance_tests = 0
        
        for test in tests['test_details']:
            test_date = test['date']
            try:
                test_idx = data.index.get_loc(test_date)
                
                # Look at price before the test
                if test_idx > 0:
                    prev_close = data['close'].iloc[test_idx - 1]
                    
                    if prev_close > level_price:
                        # Price came from above, level acted as support
                        support_tests += 1
                    else:
                        # Price came from below, level acted as resistance
                        resistance_tests += 1
            except:
                continue
        
        if support_tests > resistance_tests:
            return 'support'
        elif resistance_tests > support_tests:
            return 'resistance'
        else:
            return 'both'
    
    def _analyze_current_position(self, data: pd.DataFrame, validated_levels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze current price position relative to key levels"""
        try:
            if not validated_levels:
                return {'error': 'No validated levels available'}
            
            current_price = data['close'].iloc[-1]
            
            # Find nearest support and resistance levels
            support_levels = [level for level in validated_levels if level['type'] in ['support', 'both']]
            resistance_levels = [level for level in validated_levels if level['type'] in ['resistance', 'both']]
            
            # Nearest support (below current price)
            supports_below = [level for level in support_levels if level['price'] < current_price]
            nearest_support = max(supports_below, key=lambda x: x['price']) if supports_below else None
            
            # Nearest resistance (above current price)
            resistance_above = [level for level in resistance_levels if level['price'] > current_price]
            nearest_resistance = min(resistance_above, key=lambda x: x['price']) if resistance_above else None
            
            # Calculate distances and percentages
            support_distance = current_price - nearest_support['price'] if nearest_support else float('inf')
            resistance_distance = nearest_resistance['price'] - current_price if nearest_resistance else float('inf')
            
            support_distance_pct = (support_distance / current_price) * 100 if nearest_support else float('inf')
            resistance_distance_pct = (resistance_distance / current_price) * 100 if nearest_resistance else float('inf')
            
            # Determine current range
            if nearest_support and nearest_resistance:
                range_width = nearest_resistance['price'] - nearest_support['price']
                position_in_range = (current_price - nearest_support['price']) / range_width
                range_position = self._classify_range_position(position_in_range)
            else:
                range_width = None
                position_in_range = None
                range_position = 'unknown'
            
            return {
                'current_price': current_price,
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'support_distance': support_distance,
                'resistance_distance': resistance_distance,
                'support_distance_percentage': support_distance_pct,
                'resistance_distance_percentage': resistance_distance_pct,
                'trading_range_width': range_width,
                'position_in_range': position_in_range,
                'range_position_classification': range_position
            }
            
        except Exception as e:
            return {'error': f"Current position analysis failed: {str(e)}"}
    
    def _calculate_level_ratings(self, validated_levels: List[Dict[str, Any]], 
                               vap_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate strength ratings for each level"""
        try:
            level_ratings = []
            
            for level in validated_levels:
                # Base score from success rate
                success_score = level['success_rate'] * 40
                
                # Volume score (normalized)
                if 'error' not in vap_analysis:
                    max_volume = vap_analysis.get('total_volume', 1) / len(vap_analysis.get('volume_profile', [1]))
                    volume_score = min((level['volume'] / max_volume) * 30, 30) if max_volume > 0 else 0
                else:
                    volume_score = 0
                
                # Test frequency score
                test_score = min(level['total_tests'] * 5, 20)
                
                # Recency bonus (recent tests are more relevant)
                recency_score = self._calculate_recency_score(level.get('last_test_date'), level['total_tests'])
                
                total_score = success_score + volume_score + test_score + recency_score
                
                level_rating = {
                    'price': level['price'],
                    'type': level['type'],
                    'overall_score': min(total_score, 100),
                    'success_score': success_score,
                    'volume_score': volume_score,
                    'test_frequency_score': test_score,
                    'recency_score': recency_score,
                    'strength_classification': self._classify_level_strength(total_score)
                }
                
                level_ratings.append(level_rating)
            
            # Sort by overall score
            level_ratings.sort(key=lambda x: x['overall_score'], reverse=True)
            
            return {
                'individual_ratings': level_ratings,
                'strongest_support': self._find_strongest_level(level_ratings, 'support'),
                'strongest_resistance': self._find_strongest_level(level_ratings, 'resistance'),
                'average_strength': sum(r['overall_score'] for r in level_ratings) / len(level_ratings) if level_ratings else 0
            }
            
        except Exception as e:
            return {'error': f"Level rating calculation failed: {str(e)}"}
    
    # Helper methods
    def _calculate_initial_strength(self, level: Dict[str, Any]) -> float:
        """Calculate initial strength based on volume and touch count"""
        volume_factor = level['volume'] / 1000000 if level['volume'] > 0 else 0  # Normalize
        touch_factor = min(level['touch_count'] * 0.1, 1.0)
        
        return min(volume_factor + touch_factor, 1.0)
    
    def _get_volume_at_level(self, price: float, vap_analysis: Dict[str, Any]) -> float:
        """Get volume support at a specific price level"""
        if 'error' in vap_analysis:
            return 0.0
        
        volume_profile = vap_analysis.get('volume_profile', [])
        tolerance = price * 0.01  # 1% tolerance
        
        total_volume = 0
        for level in volume_profile:
            if abs(level['price_level'] - price) <= tolerance:
                total_volume += level['volume']
        
        return total_volume
    
    def _deduplicate_levels(self, levels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate levels that are too close together"""
        if not levels:
            return []
        
        levels.sort(key=lambda x: x['price'])
        deduplicated = [levels[0]]
        
        for level in levels[1:]:
            last_level = deduplicated[-1]
            price_diff_pct = abs(level['price'] - last_level['price']) / last_level['price']
            
            if price_diff_pct > self.price_level_tolerance:
                deduplicated.append(level)
            else:
                # Keep the stronger level
                if level['strength'] > last_level['strength']:
                    deduplicated[-1] = level
        
        return deduplicated
    
    def _calculate_swing_strength(self, data: pd.DataFrame, index: int, high_or_low: str) -> float:
        """Calculate strength of swing high/low"""
        window = 10
        
        if high_or_low == 'high':
            center_price = data['high'].iloc[index]
            surrounding_prices = []
            
            for i in range(max(0, index - window), min(len(data), index + window + 1)):
                if i != index:
                    surrounding_prices.append(data['high'].iloc[i])
            
            if not surrounding_prices:
                return 0.0
            
            max_surrounding = max(surrounding_prices)
            if max_surrounding == 0:
                return 0.0
            
            return (center_price - max_surrounding) / center_price
            
        else:  # low
            center_price = data['low'].iloc[index]
            surrounding_prices = []
            
            for i in range(max(0, index - window), min(len(data), index + window + 1)):
                if i != index:
                    surrounding_prices.append(data['low'].iloc[i])
            
            if not surrounding_prices:
                return 0.0
            
            min_surrounding = min(surrounding_prices)
            if center_price == 0:
                return 0.0
            
            return (min_surrounding - center_price) / center_price
    
    def _assess_level_reliability(self, success_rate: float, test_count: int) -> str:
        """Assess reliability of a level"""
        reliability_score = success_rate * 0.7 + min(test_count / 10, 0.3)
        
        if reliability_score >= 0.8:
            return 'very_high'
        elif reliability_score >= 0.6:
            return 'high'
        elif reliability_score >= 0.4:
            return 'medium'
        elif reliability_score >= 0.2:
            return 'low'
        else:
            return 'very_low'
    
    def _classify_range_position(self, position: float) -> str:
        """Classify position within trading range"""
        if position <= 0.2:
            return 'near_support'
        elif position <= 0.4:
            return 'lower_third'
        elif position <= 0.6:
            return 'middle_range'
        elif position <= 0.8:
            return 'upper_third'
        else:
            return 'near_resistance'
    
    def _calculate_recency_score(self, last_test_date, test_count: int) -> float:
        """Calculate recency bonus for recent tests"""
        if not last_test_date or test_count == 0:
            return 0
        
        # This is a simplified version - in practice, you'd calculate days since last test
        return min(test_count * 0.5, 10)  # Max 10 points for recency
    
    def _classify_level_strength(self, score: float) -> str:
        """Classify level strength"""
        if score >= 80:
            return 'very_strong'
        elif score >= 60:
            return 'strong'
        elif score >= 40:
            return 'moderate'
        elif score >= 20:
            return 'weak'
        else:
            return 'very_weak'
    
    def _find_strongest_level(self, ratings: List[Dict[str, Any]], level_type: str) -> Dict[str, Any]:
        """Find strongest level of given type"""
        type_levels = [r for r in ratings if r['type'] in [level_type, 'both']]
        return max(type_levels, key=lambda x: x['overall_score']) if type_levels else None
    
    def _generate_trading_implications(self, current_analysis: Dict[str, Any], 
                                     level_ratings: Dict[str, Any], 
                                     current_price: float) -> Dict[str, Any]:
        """Generate trading implications based on analysis"""
        try:
            implications = {
                'risk_levels': {},
                'target_levels': {},
                'breakout_levels': {},
                'trading_strategy': 'unknown',
                'risk_reward_ratio': None
            }
            
            if 'error' in current_analysis or 'error' in level_ratings:
                return implications
            
            nearest_support = current_analysis.get('nearest_support')
            nearest_resistance = current_analysis.get('nearest_resistance')
            
            # Risk levels (stop loss suggestions)
            if nearest_support:
                implications['risk_levels']['support_break'] = {
                    'price': nearest_support['price'] * 0.99,  # 1% below support
                    'distance_pct': ((current_price - nearest_support['price'] * 0.99) / current_price) * 100
                }
            
            # Target levels
            if nearest_resistance:
                implications['target_levels']['resistance_target'] = {
                    'price': nearest_resistance['price'] * 0.99,  # Just below resistance
                    'distance_pct': ((nearest_resistance['price'] * 0.99 - current_price) / current_price) * 100
                }
            
            # Breakout levels
            if nearest_resistance:
                implications['breakout_levels']['resistance_breakout'] = {
                    'price': nearest_resistance['price'] * 1.01,  # 1% above resistance
                    'distance_pct': ((nearest_resistance['price'] * 1.01 - current_price) / current_price) * 100
                }
            
            # Risk-reward ratio
            if (nearest_support and nearest_resistance and 
                implications['risk_levels'].get('support_break') and
                implications['target_levels'].get('resistance_target')):
                
                risk = implications['risk_levels']['support_break']['distance_pct']
                reward = implications['target_levels']['resistance_target']['distance_pct']
                
                if risk > 0:
                    implications['risk_reward_ratio'] = reward / risk
            
            # Trading strategy suggestion
            range_position = current_analysis.get('range_position_classification', 'unknown')
            
            if range_position == 'near_support':
                implications['trading_strategy'] = 'buy_near_support'
            elif range_position == 'near_resistance':
                implications['trading_strategy'] = 'sell_near_resistance'
            elif range_position == 'middle_range':
                implications['trading_strategy'] = 'wait_for_direction'
            else:
                implications['trading_strategy'] = 'assess_breakout_potential'
            
            return implications
            
        except Exception as e:
            return {'error': f"Trading implications generation failed: {str(e)}"}
    
    def extract_volume_bands(self, data: pd.DataFrame, top_n: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build volume-based support/resistance bands from validated levels and VAP bins.
        Returns { 'support': [...], 'resistance': [...] } with entries including
        low, high, center, reliability, success_rate, total_tests, volume_strength.
        """
        try:
            results = self.process_support_resistance_data(data)
            if 'error' in results:
                return {'support': [], 'resistance': []}
            validated_levels = results.get('validated_levels', []) or []
            vap = results.get('volume_at_price_analysis', {}) or {}
            volume_profile = vap.get('volume_profile', []) or []

            # Helper to find the bin range containing a price
            def find_bin_range(price: float) -> Tuple[float, float]:
                for lvl in volume_profile:
                    pr = lvl.get('price_range') or [None, None]
                    if pr and pr[0] is not None and pr[1] is not None:
                        if pr[0] <= price <= pr[1]:
                            return float(pr[0]), float(pr[1])
                # Fallback: build a small ATR-like band if no exact bin found
                try:
                    current_price = float(data['close'].iloc[-1])
                    width = max(0.005 * current_price, 0.005 * price)  # ~0.5%
                except Exception:
                    width = 0.005 * price if price else 0.0
                return float(price - width), float(price + width)

            # Ranking key for reliability
            def reliability_rank(rel: str) -> int:
                order = {'very_high': 5, 'high': 4, 'medium': 3, 'low': 2, 'very_low': 1}
                return order.get(str(rel), 0)

            supports: List[Dict[str, Any]] = []
            resistances: List[Dict[str, Any]] = []
            for lvl in validated_levels:
                price = float(lvl.get('price', 0.0) or 0.0)
                ltype = str(lvl.get('type', 'unknown'))
                low, high = find_bin_range(price)
                band = {
                    'low': float(low),
                    'high': float(high),
                    'center': price,
                    'reliability': lvl.get('reliability'),
                    'success_rate': float(lvl.get('success_rate') or 0.0),
                    'total_tests': int(lvl.get('total_tests') or 0),
                    'volume_strength': self._classify_volume_strength(float(lvl.get('volume') or 0.0))
                }
                if ltype in ('support', 'both'):
                    supports.append(band)
                if ltype in ('resistance', 'both'):
                    resistances.append(band)

            # Ensure reasonable band width bounds (min 0.5% / max 2.0% around center)
            def normalize_width(b: Dict[str, Any]) -> Dict[str, Any]:
                c = float(b['center'])
                if c <= 0:
                    return b
                min_w = 0.005 * c
                max_w = 0.02 * c
                low, high = float(b['low']), float(b['high'])
                width = max(0.0, high - low)
                if width < min_w:
                    pad = (min_w - width) / 2.0
                    low, high = c - (width/2 + pad), c + (width/2 + pad)
                elif width > max_w:
                    # shrink symmetrically around center
                    low, high = c - max_w/2.0, c + max_w/2.0
                b['low'], b['high'] = float(low), float(high)
                return b

            supports = [normalize_width(b) for b in supports]
            resistances = [normalize_width(b) for b in resistances]

            # Sort by reliability, then success_rate, then total_tests
            supports.sort(key=lambda b: (reliability_rank(b['reliability']), b['success_rate'], b['total_tests']), reverse=True)
            resistances.sort(key=lambda b: (reliability_rank(b['reliability']), b['success_rate'], b['total_tests']), reverse=True)

            # Fallback: if very few validated bands, derive from VAP significant levels
            need_sup = max(0, top_n - len(supports))
            need_res = max(0, top_n - len(resistances))
            if (need_sup > 0 or need_res > 0) and volume_profile:
                try:
                    # Use significant levels if available, otherwise top volume nodes
                    # significant_volume_levels were computed in _calculate_volume_at_price
                    sig_levels = vap.get('significant_volume_levels', []) or []
                    if not sig_levels:
                        # Build minimal proxy from highest volume bins
                        sig_levels = sorted(volume_profile, key=lambda x: x.get('volume', 0), reverse=True)[:top_n*2]
                    current_price = float(data['close'].iloc[-1]) if len(data) > 0 else 0.0

                    def band_from_level(lvl: Dict[str, Any]) -> Dict[str, Any]:
                        pr = lvl.get('price_range') or [None, None]
                        low = float(pr[0]) if pr and pr[0] is not None else float(lvl.get('price_level') or 0.0)
                        high = float(pr[1]) if pr and pr[1] is not None else float(lvl.get('price_level') or 0.0)
                        center = float(lvl.get('price_level') or (low + high) / 2.0)
                        return {
                            'low': low,
                            'high': high if high >= low else low,
                            'center': center,
                            'reliability': 'low',
                            'success_rate': 0.0,
                            'total_tests': 0,
                            'volume_strength': self._classify_volume_strength(float(lvl.get('volume') or 0.0))
                        }

                    # Fill supports
                    if need_sup > 0:
                        candidates = [band_from_level(l) for l in sig_levels if float(l.get('price_level') or 0.0) < current_price]
                        # Normalize widths
                        candidates = [normalize_width(b) for b in candidates]
                        # De-dup close centers
                        def unique_by_center(arr):
                            seen = set()
                            out = []
                            for b in arr:
                                key = round(b['center'], 4)
                                if key in seen:
                                    continue
                                seen.add(key)
                                out.append(b)
                            return out
                        candidates = unique_by_center(candidates)
                        supports.extend(candidates[:need_sup])

                    # Fill resistances
                    if need_res > 0:
                        candidates = [band_from_level(l) for l in sig_levels if float(l.get('price_level') or 0.0) > current_price]
                        candidates = [normalize_width(b) for b in candidates]
                        def unique_by_center(arr):
                            seen = set()
                            out = []
                            for b in arr:
                                key = round(b['center'], 4)
                                if key in seen:
                                    continue
                                seen.add(key)
                                out.append(b)
                            return out
                        candidates = unique_by_center(candidates)
                        resistances.extend(candidates[:need_res])

                except Exception:
                    pass

            return {
                'support': supports[:max(1, top_n)],
                'resistance': resistances[:max(1, top_n)]
            }
        except Exception as e:
            return {'support': [], 'resistance': [], 'error': f'volume_bands_failed: {e}'}

    def _assess_analysis_quality(self, validated_levels: List[Dict[str, Any]], 
                               vap_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall analysis quality"""
        try:
            quality_score = 0
            factors = []
            
            # Level count quality
            level_count = len(validated_levels)
            if level_count >= 5:
                quality_score += 25
                factors.append('sufficient_levels')
            elif level_count >= 3:
                quality_score += 15
                factors.append('adequate_levels')
            
            # Level reliability quality
            if validated_levels:
                avg_reliability = sum(1 for level in validated_levels 
                                    if level['reliability'] in ['high', 'very_high']) / len(validated_levels)
                quality_score += avg_reliability * 25
                if avg_reliability > 0.5:
                    factors.append('high_reliability_levels')
            
            # Volume analysis quality
            if 'error' not in vap_analysis:
                quality_score += 25
                factors.append('volume_analysis_valid')
                
                if vap_analysis.get('total_volume', 0) > 0:
                    quality_score += 15
                    factors.append('sufficient_volume_data')
            
            # Test frequency quality
            if validated_levels:
                avg_tests = sum(level['total_tests'] for level in validated_levels) / len(validated_levels)
                if avg_tests >= 3:
                    quality_score += 10
                    factors.append('well_tested_levels')
            
            return {
                'overall_score': min(quality_score, 100),
                'quality_factors': factors,
                'level_count': level_count,
                'reliability_rating': self._determine_overall_reliability(quality_score)
            }
            
        except Exception as e:
            return {
                'overall_score': 0,
                'error': f"Quality assessment failed: {str(e)}"
            }
    
    def _determine_overall_reliability(self, score: int) -> str:
        """Determine overall analysis reliability"""
        if score >= 80:
            return 'very_high'
        elif score >= 60:
            return 'high'
        elif score >= 40:
            return 'medium'
        elif score >= 20:
            return 'low'
        else:
            return 'very_low'
    
    def _format_support_levels(self, validated_levels: List[Dict[str, Any]], level_type: str) -> List[Dict[str, Any]]:
        """Format support levels for output"""
        support_levels = [level for level in validated_levels if level['type'] in ['support', 'both']]
        
        formatted_levels = []
        for level in support_levels[:5]:  # Top 5 levels
            formatted_levels.append({
                'price_level': level['price'],
                'volume_strength': self._classify_volume_strength(level['volume']),
                'reliability': level['reliability'],
                'success_rate': level['success_rate'],
                'total_tests': level['total_tests']
            })
        
        return formatted_levels
    
    def _format_resistance_levels(self, validated_levels: List[Dict[str, Any]], level_type: str) -> List[Dict[str, Any]]:
        """Format resistance levels for output"""
        resistance_levels = [level for level in validated_levels if level['type'] in ['resistance', 'both']]
        
        formatted_levels = []
        for level in resistance_levels[:5]:  # Top 5 levels
            formatted_levels.append({
                'price_level': level['price'],
                'volume_strength': self._classify_volume_strength(level['volume']),
                'reliability': level['reliability'],
                'success_rate': level['success_rate'],
                'total_tests': level['total_tests']
            })
        
        return formatted_levels
    
    def _classify_volume_strength(self, volume: float) -> str:
        """Classify volume strength"""
        if volume > 5000000:
            return 'very_high'
        elif volume > 2000000:
            return 'high'
        elif volume > 1000000:
            return 'medium'
        elif volume > 500000:
            return 'low'
        else:
            return 'very_low'


def test_support_resistance_processor():
    """Test function for Support/Resistance Processor"""
    print("📊 Testing Support/Resistance Processor")
    print("=" * 60)
    
    # Create test data with clear support/resistance levels
    dates = pd.date_range(start='2024-07-01', end='2024-10-20', freq='D')
    np.random.seed(42)
    
    base_price = 2400
    
    # Create price data with defined support/resistance levels
    support_level = 2350
    resistance_level = 2450
    
    prices = []
    current_price = base_price
    
    for i in range(len(dates)):
        # Random walk with boundaries
        change = np.random.normal(0, 15)
        new_price = current_price + change
        
        # Bounce off support/resistance with some probability
        if new_price <= support_level and np.random.random() > 0.3:
            new_price = support_level + np.random.uniform(5, 20)
        elif new_price >= resistance_level and np.random.random() > 0.3:
            new_price = resistance_level - np.random.uniform(5, 20)
        
        prices.append(new_price)
        current_price = new_price
    
    # Generate volume (higher at support/resistance tests)
    volumes = []
    for i, price in enumerate(prices):
        base_vol = np.random.lognormal(np.log(1500000), 0.3)
        
        # Increase volume near support/resistance
        if abs(price - support_level) < 20 or abs(price - resistance_level) < 20:
            base_vol *= np.random.uniform(1.5, 3.0)
        
        volumes.append(int(base_vol))
    
    # Create OHLC data
    test_data = pd.DataFrame({
        'open': [p + np.random.normal(0, 5) for p in prices],
        'high': [p + abs(np.random.normal(8, 4)) for p in prices],
        'low': [p - abs(np.random.normal(8, 4)) for p in prices],
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    # Ensure realistic OHLC relationships
    test_data['high'] = np.maximum(test_data[['open', 'close']].max(axis=1), test_data['high'])
    test_data['low'] = np.minimum(test_data[['open', 'close']].min(axis=1), test_data['low'])
    
    print(f"✅ Created test data: {len(test_data)} days")
    print(f"   Price range: ${test_data['close'].min():.2f} - ${test_data['close'].max():.2f}")
    print(f"   Volume range: {test_data['volume'].min():,} - {test_data['volume'].max():,}")
    print(f"   Expected support around: ${support_level}")
    print(f"   Expected resistance around: ${resistance_level}")
    
    # Process data
    processor = SupportResistanceProcessor()
    results = processor.process_support_resistance_data(test_data)
    
    if 'error' in results:
        print(f"❌ Analysis failed: {results['error']}")
        return False

    # Extract volume bands for smoke test
    bands = processor.extract_volume_bands(test_data, top_n=3)
    print("\n🟦 Volume S/R Bands (top 3 per side):")
    for side in ("support","resistance"):
        arr = bands.get(side, []) if isinstance(bands, dict) else []
        print(f"  {side.title()}:")
        for b in arr:
            print(f"    {b['reliability'] or 'n/a'} band: [{b['low']:.2f}, {b['high']:.2f}] (center {b['center']:.2f})")
    
    print("✅ Analysis completed successfully")
    
    # Display results
    validated_levels = results.get('validated_levels', [])
    current_analysis = results.get('current_position_analysis', {})
    level_ratings = results.get('level_ratings', {})
    quality = results.get('quality_assessment', {})
    
    print(f"\n📊 Analysis Results:")
    print(f"   Validated levels: {len(validated_levels)}")
    print(f"   Quality score: {quality.get('overall_score', 0)}/100")
    
    # Show top support/resistance levels
    support_levels = results.get('volume_based_support_levels', [])
    resistance_levels = results.get('volume_based_resistance_levels', [])
    
    if support_levels:
        print(f"\n📉 Top Support Levels:")
        for i, level in enumerate(support_levels[:3], 1):
            print(f"   {i}. ${level['price_level']:.2f} - {level['reliability']} reliability ({level['total_tests']} tests)")
    
    if resistance_levels:
        print(f"\n📈 Top Resistance Levels:")
        for i, level in enumerate(resistance_levels[:3], 1):
            print(f"   {i}. ${level['price_level']:.2f} - {level['reliability']} reliability ({level['total_tests']} tests)")
    
    # Current position
    if 'error' not in current_analysis:
        print(f"\n📍 Current Position:")
        print(f"   Price: ${current_analysis['current_price']:.2f}")
        print(f"   Range position: {current_analysis.get('range_position_classification', 'unknown')}")
        
        nearest_support = current_analysis.get('nearest_support')
        nearest_resistance = current_analysis.get('nearest_resistance')
        
        if nearest_support:
            print(f"   Nearest support: ${nearest_support['price']:.2f} ({current_analysis.get('support_distance_percentage', 0):.1f}% away)")
        
        if nearest_resistance:
            print(f"   Nearest resistance: ${nearest_resistance['price']:.2f} ({current_analysis.get('resistance_distance_percentage', 0):.1f}% away)")
    
    return True

if __name__ == "__main__":
    test_support_resistance_processor()