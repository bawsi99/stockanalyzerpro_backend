import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import List, Tuple, Dict, Any


class PatternRecognition:
    """
    Central class for all pattern detection logic (peaks/lows, divergences, double tops/bottoms, triangles, flags, volume anomalies).
    """
    @staticmethod
    def identify_peaks_lows(prices: pd.Series, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Identify local peaks and lows."""
        prices_np = prices.values
        peaks = argrelextrema(prices_np, np.greater, order=order)[0]
        lows = argrelextrema(prices_np, np.less, order=order)[0]
        return peaks, lows

    @staticmethod
    def get_swing_points(prices: pd.Series, order: int = 5) -> Dict[str, np.ndarray]:
        """Return swing highs and lows."""
        highs, lows = PatternRecognition.identify_peaks_lows(prices, order=order)
        return {'swing_highs': highs, 'swing_lows': lows}

    @staticmethod
    def detect_divergence(prices: pd.Series, indicator: pd.Series, order: int = 5) -> List[Tuple[int, int, str]]:
        """Detect bullish/bearish divergence between price and indicator (e.g., RSI)."""
        price_np = prices.values
        indicator_np = indicator.values
        peaks = argrelextrema(price_np, np.greater_equal, order=order)[0]
        lows = argrelextrema(price_np, np.less_equal, order=order)[0]
        divergences = []
        for i in range(1, len(peaks)):
            p1, p2 = peaks[i-1], peaks[i]
            if price_np[p2] > price_np[p1] and indicator_np[p2] < indicator_np[p1]:
                divergences.append((p1, p2, 'bearish'))
        for i in range(1, len(lows)):
            l1, l2 = lows[i-1], lows[i]
            if price_np[l2] < price_np[l1] and indicator_np[l2] > indicator_np[l1]:
                divergences.append((l1, l2, 'bullish'))
        return divergences

    @staticmethod
    def detect_volume_anomalies(volume: pd.Series, threshold: float = 2.0):
        """Detect volume spikes (anomalies) where volume is threshold times above rolling mean."""
        rolling_mean = volume.rolling(window=20, min_periods=1).mean()
        anomalies = volume[volume > threshold * rolling_mean].index.tolist()
        return anomalies

    @staticmethod
    def detect_double_top(prices: pd.Series, threshold: float = 0.02, order: int = 5) -> List[Tuple[int, int]]:
        """Detect double top patterns in price data."""
        peaks, _ = PatternRecognition.identify_peaks_lows(prices, order=order)
        patterns = []
        for i in range(1, len(peaks)):
            price_diff = abs(prices.iloc[peaks[i]] - prices.iloc[peaks[i - 1]])
            if price_diff / prices.iloc[peaks[i]] < threshold:
                patterns.append((peaks[i - 1], peaks[i]))
        return patterns

    @staticmethod
    def detect_double_bottom(prices: pd.Series, threshold: float = 0.02, order: int = 5) -> List[Tuple[int, int]]:
        """Detect double bottom patterns in price data."""
        lows = argrelextrema(prices.values, np.less, order=order)[0]
        patterns = []
        for i in range(1, len(lows)):
            first = lows[i-1]
            second = lows[i]
            if abs(prices.iloc[first] - prices.iloc[second]) / prices.iloc[first] < threshold:
                peak_idx = np.argmax(prices.iloc[first:second+1]) + first
                if peak_idx > first and peak_idx < second:
                    patterns.append((first, second))
        return patterns

    @staticmethod
    def detect_triangle(prices: pd.Series, min_points: int = 5) -> List[List[int]]:
        """
        Detect symmetrical triangles (descending highs + ascending lows).

        Returns: list of lists with indices that belong to each triangle window.
        """
        patterns: List[List[int]] = []

        for start in range(len(prices) - min_points):
            end = start + min_points
            segment = prices.iloc[start:end]

            # Significant local extremes -------------------------------------------------
            order = max(min_points // 12, 4)
            local_highs = argrelextrema(segment.values, np.greater, order=order)[0]
            local_lows  = argrelextrema(segment.values, np.less,    order=order)[0]

            if len(local_highs) < 2 or len(local_lows) < 2:
                continue

            # Use the SAME window for highs & lows ---------------------------------------
            x_hi = start + local_highs
            x_lo = start + local_lows
            y_hi = prices.iloc[x_hi]
            y_lo = prices.iloc[x_lo]

            # Regression lines -----------------------------------------------------------
            slope_hi, _ = np.polyfit(x_hi, y_hi, 1)   # should be negative
            slope_lo, _ = np.polyfit(x_lo, y_lo, 1)   # should be positive

            if slope_hi >= 0 or slope_lo <= 0:
                continue

            # Do the slopes have *similar magnitude*?  -----------------------------------
            mag_hi = abs(slope_hi)
            mag_lo = slope_lo
            rel_diff = abs(mag_hi - mag_lo) / max(mag_hi, mag_lo)

            if rel_diff > 0.35:
                continue

            # Passed all checks → record the whole sub-window (or just [start,end])
            patterns.append(list(range(start, end)))

        return patterns

    @staticmethod
    def detect_flag(
            prices: pd.Series,
            impulse: int = 15,           # bars that define the 'flag pole'
            channel: int = 20,           # length of consolidation window
            pullback_ratio: float = .35  # fraction of the impulse it may retrace
        ) -> List[List[int]]:
        """
        Very simple bullish/bearish flag detector:
        1) looks for an 'impulse' move (straight up or down)
        2) then a sideways / slightly drifting consolidation
        """
        patterns: List[List[int]] = []
        N = len(prices)

        i = impulse
        while i < N - channel:
            # 1) impulse magnitude
            pole_return = (prices.iloc[i] - prices.iloc[i-impulse]) / prices.iloc[i-impulse]

            # bullish pole
            if pole_return > 0.08:       # > +8 % in 'impulse' bars
                seg = prices.iloc[i:i+channel]
                max_pullback = pole_return * pullback_ratio
                retr = (seg.min() - prices.iloc[i]) / prices.iloc[i]

                # sideways channel? (small retracement and std-dev)
                if abs(retr) <= max_pullback and seg.pct_change().std() < 0.02:
                    patterns.append(list(range(i-impulse, i+channel)))
                    i += channel       # skip overlapping windows
                    continue

            # bearish pole
            if pole_return < -0.08:
                seg = prices.iloc[i:i+channel]
                max_pullback = abs(pole_return) * pullback_ratio
                retr = (seg.max() - prices.iloc[i]) / prices.iloc[i]
                if abs(retr) <= max_pullback and seg.pct_change().std() < 0.02:
                    patterns.append(list(range(i-impulse, i+channel)))
                    i += channel
                    continue

            i += 1

        return patterns
    
    @staticmethod
    def calculate_pattern_reliability(pattern_type: str, pattern_data: Dict, volume_data: pd.Series = None, 
                                    market_conditions: Dict = None) -> Dict[str, Any]:
        """
        Calculate pattern reliability score and analysis.
        
        Args:
            pattern_type: Type of pattern (triangle, flag, double_top, double_bottom)
            pattern_data: Pattern-specific data
            volume_data: Volume data for confirmation
            market_conditions: Market conditions data
            
        Returns:
            Dict containing reliability analysis
        """
        reliability_score = 0
        factors = {}
        
        # Base reliability scores for different patterns
        base_scores = {
            'triangle': 60,
            'flag': 65,
            'double_top': 70,
            'double_bottom': 70,
            'head_shoulders': 75,
            'cup_handle': 80
        }
        
        reliability_score = base_scores.get(pattern_type, 50)
        
        # Volume confirmation (if available)
        if volume_data is not None and len(volume_data) > 0:
            recent_volume_avg = volume_data.iloc[-20:].mean()
            current_volume = volume_data.iloc[-1]
            volume_ratio = current_volume / recent_volume_avg if recent_volume_avg > 0 else 1.0
            
            if volume_ratio > 1.5:
                reliability_score += 15
                factors['volume_confirmation'] = 'strong'
            elif volume_ratio > 1.2:
                reliability_score += 10
                factors['volume_confirmation'] = 'moderate'
            else:
                factors['volume_confirmation'] = 'weak'
            
            factors['volume_ratio'] = volume_ratio
        
        # Pattern completion percentage
        if 'completion_percentage' in pattern_data:
            completion = pattern_data['completion_percentage']
            if completion > 80:
                reliability_score += 20
                factors['completion'] = 'high'
            elif completion > 60:
                reliability_score += 10
                factors['completion'] = 'moderate'
            else:
                factors['completion'] = 'low'
        
        # Market condition correlation
        if market_conditions:
            volatility = market_conditions.get('volatility_regime', 'normal')
            if volatility == 'normal':
                reliability_score += 10
                factors['market_conditions'] = 'favorable'
            elif volatility == 'high':
                reliability_score -= 5
                factors['market_conditions'] = 'challenging'
            else:
                factors['market_conditions'] = 'neutral'
        
        # Pattern quality factors
        if 'quality_score' in pattern_data:
            quality = pattern_data['quality_score']
            if quality > 0.8:
                reliability_score += 15
                factors['pattern_quality'] = 'excellent'
            elif quality > 0.6:
                reliability_score += 10
                factors['pattern_quality'] = 'good'
            else:
                factors['pattern_quality'] = 'poor'
        
        # Cap the score at 100
        reliability_score = min(reliability_score, 100)
        
        # Determine reliability level
        if reliability_score >= 80:
            reliability_level = 'high'
        elif reliability_score >= 60:
            reliability_level = 'moderate'
        else:
            reliability_level = 'low'
        
        return {
            'reliability_score': reliability_score,
            'reliability_level': reliability_level,
            'factors': factors,
            'pattern_type': pattern_type,
            'recommendation': 'strong' if reliability_score >= 80 else 'moderate' if reliability_score >= 60 else 'weak'
        }
    
    @staticmethod
    def analyze_pattern_failure_risk(pattern_type: str, pattern_data: Dict, 
                                   current_price: float, support_resistance: Dict) -> Dict[str, Any]:
        """
        Analyze potential pattern failure risks.
        
        Args:
            pattern_type: Type of pattern
            pattern_data: Pattern-specific data
            current_price: Current price
            support_resistance: Support and resistance levels
            
        Returns:
            Dict containing failure risk analysis
        """
        risk_factors = []
        risk_score = 0
        
        # Distance to key levels
        if 'target_level' in pattern_data:
            target = pattern_data['target_level']
            distance_to_target = abs(current_price - target) / current_price
            
            if distance_to_target > 0.1:  # More than 10% away
                risk_factors.append('far_from_target')
                risk_score += 20
        
        # Support/resistance proximity
        if support_resistance:
            nearest_support = min(support_resistance.get('support', [current_price * 0.9]))
            nearest_resistance = max(support_resistance.get('resistance', [current_price * 1.1]))
            
            support_distance = (current_price - nearest_support) / current_price
            resistance_distance = (nearest_resistance - current_price) / current_price
            
            if support_distance < 0.02:  # Very close to support
                risk_factors.append('near_support_breakdown')
                risk_score += 15
            
            if resistance_distance < 0.02:  # Very close to resistance
                risk_factors.append('near_resistance_rejection')
                risk_score += 15
        
        # Pattern-specific risks
        if pattern_type == 'triangle':
            if 'breakout_direction' not in pattern_data:
                risk_factors.append('unclear_breakout_direction')
                risk_score += 25
        
        elif pattern_type in ['double_top', 'double_bottom']:
            if 'confirmation_level' not in pattern_data:
                risk_factors.append('missing_confirmation')
                risk_score += 20
        
        # Market volatility risk
        if 'volatility' in pattern_data:
            volatility = pattern_data['volatility']
            if volatility > 0.03:  # High volatility
                risk_factors.append('high_volatility')
                risk_score += 15
        
        # Determine risk level
        if risk_score >= 50:
            risk_level = 'high'
        elif risk_score >= 30:
            risk_level = 'moderate'
        else:
            risk_level = 'low'
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'mitigation_strategies': PatternRecognition._get_risk_mitigation_strategies(pattern_type, risk_factors)
        }
    
    @staticmethod
    def _get_risk_mitigation_strategies(pattern_type: str, risk_factors: List[str]) -> List[str]:
        """
        Get risk mitigation strategies based on pattern type and risk factors.
        
        Args:
            pattern_type: Type of pattern
            risk_factors: List of identified risk factors
            
        Returns:
            List of mitigation strategies
        """
        strategies = []
        
        if 'far_from_target' in risk_factors:
            strategies.append('Use tighter stop-loss levels')
            strategies.append('Consider partial profit taking')
        
        if 'near_support_breakdown' in risk_factors:
            strategies.append('Set stop-loss below support level')
            strategies.append('Wait for support confirmation')
        
        if 'near_resistance_rejection' in risk_factors:
            strategies.append('Set stop-loss above resistance level')
            strategies.append('Wait for breakout confirmation')
        
        if 'unclear_breakout_direction' in risk_factors:
            strategies.append('Wait for clear breakout signal')
            strategies.append('Use volume confirmation')
        
        if 'missing_confirmation' in risk_factors:
            strategies.append('Wait for pattern confirmation')
            strategies.append('Use additional technical indicators')
        
        if 'high_volatility' in risk_factors:
            strategies.append('Use wider stop-loss levels')
            strategies.append('Reduce position size')
        
        return strategies 

    @staticmethod
    def detect_head_and_shoulders(prices: pd.Series, order: int = 5, tolerance: float = 0.02) -> List[Dict[str, Any]]:
        """
        Detect Head and Shoulders patterns.
        
        Args:
            prices: Price series
            order: Order for peak detection
            tolerance: Tolerance for shoulder height similarity
            
        Returns:
            List of detected H&S patterns
        """
        peaks, _ = PatternRecognition.identify_peaks_lows(prices, order=order)
        
        if len(peaks) < 3:
            return []
        
        patterns = []
        
        for i in range(len(peaks) - 2):
            left_shoulder = peaks[i]
            head = peaks[i + 1]
            right_shoulder = peaks[i + 2]
            
            left_price = prices.iloc[left_shoulder]
            head_price = prices.iloc[head]
            right_price = prices.iloc[right_shoulder]
            
            # Check H&S conditions
            # 1. Head should be higher than shoulders
            if head_price <= left_price or head_price <= right_price:
                continue
            
            # 2. Shoulders should be at similar levels
            shoulder_diff = abs(left_price - right_price) / left_price
            if shoulder_diff > tolerance:
                continue
            
            # 3. Check for neckline (support level between shoulders)
            neckline_start = left_shoulder
            neckline_end = right_shoulder
            
            # Find the lowest point between shoulders for neckline
            neckline_low = prices.iloc[neckline_start:neckline_end + 1].min()
            neckline_idx = prices.iloc[neckline_start:neckline_end + 1].idxmin()
            # Convert Timestamp to integer index
            neckline_idx_int = prices.index.get_loc(neckline_idx)
            
            # 4. Calculate pattern metrics
            head_height = head_price - neckline_low
            shoulder_height = (left_price + right_price) / 2 - neckline_low
            
            # 5. Pattern quality assessment
            quality_score = 0
            
            # Head prominence (should be significantly higher than shoulders)
            head_prominence = (head_price - max(left_price, right_price)) / head_price
            if head_prominence > 0.02:  # 2% higher
                quality_score += 30
            
            # Shoulder symmetry
            shoulder_symmetry = 1 - shoulder_diff
            quality_score += shoulder_symmetry * 20
            
            # Volume confirmation (if available)
            if 'volume' in prices.index.name or hasattr(prices, 'volume'):
                # This would need volume data integration
                quality_score += 10
            
            # Pattern completion
            current_price = prices.iloc[-1]
            if current_price < neckline_low:
                completion_status = "completed"
                quality_score += 20
            else:
                completion_status = "forming"
            
            pattern = {
                "type": "head_and_shoulders",
                "left_shoulder": {
                    "index": int(left_shoulder),
                    "price": float(left_price),
                    "date": str(prices.index[left_shoulder]) if hasattr(prices.index, 'strftime') else str(left_shoulder)
                },
                "head": {
                    "index": int(head),
                    "price": float(head_price),
                    "date": str(prices.index[head]) if hasattr(prices.index, 'strftime') else str(head)
                },
                "right_shoulder": {
                    "index": int(right_shoulder),
                    "price": float(right_price),
                    "date": str(prices.index[right_shoulder]) if hasattr(prices.index, 'strftime') else str(right_shoulder)
                },
                "neckline": {
                    "level": float(neckline_low),
                    "index": int(neckline_idx_int),
                    "date": str(prices.index[neckline_idx_int]) if hasattr(prices.index, 'strftime') else str(neckline_idx_int)
                },
                "target": float(neckline_low - head_height),  # Measured move target
                "quality_score": quality_score,
                "completion_status": completion_status,
                "head_prominence": float(head_prominence),
                "shoulder_symmetry": float(shoulder_symmetry)
            }
            
            patterns.append(pattern)
        
        return patterns

    @staticmethod
    def detect_inverse_head_and_shoulders(prices: pd.Series, order: int = 5, tolerance: float = 0.02) -> List[Dict[str, Any]]:
        """
        Detect Inverse Head and Shoulders patterns.
        
        Args:
            prices: Price series
            order: Order for low detection
            tolerance: Tolerance for shoulder height similarity
            
        Returns:
            List of detected inverse H&S patterns
        """
        _, lows = PatternRecognition.identify_peaks_lows(prices, order=order)
        
        if len(lows) < 3:
            return []
        
        patterns = []
        
        for i in range(len(lows) - 2):
            left_shoulder = lows[i]
            head = lows[i + 1]
            right_shoulder = lows[i + 2]
            
            left_price = prices.iloc[left_shoulder]
            head_price = prices.iloc[head]
            right_price = prices.iloc[right_shoulder]
            
            # Check inverse H&S conditions
            # 1. Head should be lower than shoulders
            if head_price >= left_price or head_price >= right_price:
                continue
            
            # 2. Shoulders should be at similar levels
            shoulder_diff = abs(left_price - right_price) / left_price
            if shoulder_diff > tolerance:
                continue
            
            # 3. Check for neckline (resistance level between shoulders)
            neckline_start = left_shoulder
            neckline_end = right_shoulder
            
            # Find the highest point between shoulders for neckline
            neckline_high = prices.iloc[neckline_start:neckline_end + 1].max()
            neckline_idx = prices.iloc[neckline_start:neckline_end + 1].idxmax()
            # Convert Timestamp to integer index
            neckline_idx_int = prices.index.get_loc(neckline_idx)
            
            # 4. Calculate pattern metrics
            head_depth = neckline_high - head_price
            shoulder_depth = neckline_high - (left_price + right_price) / 2
            
            # 5. Pattern quality assessment
            quality_score = 0
            
            # Head prominence (should be significantly lower than shoulders)
            head_prominence = (min(left_price, right_price) - head_price) / head_price
            if head_prominence > 0.02:  # 2% lower
                quality_score += 30
            
            # Shoulder symmetry
            shoulder_symmetry = 1 - shoulder_diff
            quality_score += shoulder_symmetry * 20
            
            # Pattern completion
            current_price = prices.iloc[-1]
            if current_price > neckline_high:
                completion_status = "completed"
                quality_score += 20
            else:
                completion_status = "forming"
            
            pattern = {
                "type": "inverse_head_and_shoulders",
                "left_shoulder": {
                    "index": int(left_shoulder),
                    "price": float(left_price),
                    "date": str(prices.index[left_shoulder]) if hasattr(prices.index, 'strftime') else str(left_shoulder)
                },
                "head": {
                    "index": int(head),
                    "price": float(head_price),
                    "date": str(prices.index[head]) if hasattr(prices.index, 'strftime') else str(head)
                },
                "right_shoulder": {
                    "index": int(right_shoulder),
                    "price": float(right_price),
                    "date": str(prices.index[right_shoulder]) if hasattr(prices.index, 'strftime') else str(right_shoulder)
                },
                "neckline": {
                    "level": float(neckline_high),
                    "index": int(neckline_idx_int),
                    "date": str(prices.index[neckline_idx_int]) if hasattr(prices.index, 'strftime') else str(neckline_idx_int)
                },
                "target": float(neckline_high + head_depth),  # Measured move target
                "quality_score": quality_score,
                "completion_status": completion_status,
                "head_prominence": float(head_prominence),
                "shoulder_symmetry": float(shoulder_symmetry)
            }
            
            patterns.append(pattern)
        
        return patterns

    @staticmethod
    def detect_cup_and_handle(prices: pd.Series, min_cup_duration: int = 20, max_cup_duration: int = 100,
                            handle_duration_ratio: float = 0.3, depth_tolerance: float = 0.15) -> List[Dict[str, Any]]:
        """
        Detect Cup and Handle patterns.
        
        Args:
            prices: Price series
            min_cup_duration: Minimum duration for cup formation
            max_cup_duration: Maximum duration for cup formation
            handle_duration_ratio: Handle duration as ratio of cup duration
            depth_tolerance: Tolerance for cup depth consistency
            
        Returns:
            List of detected Cup and Handle patterns
        """
        if len(prices) < min_cup_duration * 2:
            return []
        
        patterns = []
        
        for start_idx in range(len(prices) - min_cup_duration):
            # Try different cup durations
            for cup_duration in range(min_cup_duration, min(max_cup_duration, len(prices) - start_idx)):
                cup_end = start_idx + cup_duration
                cup_data = prices.iloc[start_idx:cup_end + 1]
                
                # Check if cup formation is valid
                cup_start_price = cup_data.iloc[0]
                cup_end_price = cup_data.iloc[-1]
                cup_low = cup_data.min()
                cup_low_idx = cup_data.idxmin()
                # Convert Timestamp to integer index if needed
                cup_low_idx_int = cup_data.index.get_loc(cup_low_idx)
                
                # Cup should have similar start and end prices
                price_diff = abs(cup_start_price - cup_end_price) / cup_start_price
                if price_diff > depth_tolerance:
                    continue
                
                # Cup should have a clear bottom
                cup_depth = (cup_start_price - cup_low) / cup_start_price
                if cup_depth < 0.05:  # At least 5% depth
                    continue
                
                # Check for handle formation
                handle_start = cup_end
                handle_duration = int(cup_duration * handle_duration_ratio)
                handle_end = min(handle_start + handle_duration, len(prices) - 1)
                
                if handle_end <= handle_start:
                    continue
                
                handle_data = prices.iloc[handle_start:handle_end + 1]
                
                # Handle should be a downward drift or consolidation
                handle_start_price = handle_data.iloc[0]
                handle_end_price = handle_data.iloc[-1]
                handle_low = handle_data.min()
                
                # Handle should not break below cup low significantly
                handle_breakdown = (cup_low - handle_low) / cup_low
                if handle_breakdown > 0.02:  # More than 2% breakdown
                    continue
                
                # Handle should show downward drift
                handle_drift = (handle_start_price - handle_end_price) / handle_start_price
                if handle_drift < 0.01:  # At least 1% downward drift
                    continue
                
                # Calculate pattern metrics
                cup_height = cup_start_price - cup_low
                handle_height = handle_start_price - handle_low
                
                # Pattern quality assessment
                quality_score = 0
                
                # Cup symmetry
                cup_symmetry = 1 - price_diff
                quality_score += cup_symmetry * 25
                
                # Cup depth
                if 0.05 <= cup_depth <= 0.25:  # Ideal depth range
                    quality_score += 25
                elif 0.25 < cup_depth <= 0.4:
                    quality_score += 15
                
                # Handle quality
                handle_quality = 1 - handle_breakdown
                quality_score += handle_quality * 20
                
                # Volume confirmation (placeholder)
                quality_score += 10
                
                # Pattern completion
                current_price = prices.iloc[-1]
                breakout_level = handle_start_price
                if current_price > breakout_level:
                    completion_status = "completed"
                    quality_score += 20
                else:
                    completion_status = "forming"
                
                pattern = {
                    "type": "cup_and_handle",
                    "cup": {
                        "start_index": int(start_idx),
                        "end_index": int(cup_end),
                        "start_price": float(cup_start_price),
                        "end_price": float(cup_end_price),
                        "low_price": float(cup_low),
                        "low_index": int(cup_low_idx_int),
                        "depth": float(cup_depth),
                        "duration": cup_duration
                    },
                    "handle": {
                        "start_index": int(handle_start),
                        "end_index": int(handle_end),
                        "start_price": float(handle_start_price),
                        "end_price": float(handle_end_price),
                        "low_price": float(handle_low),
                        "drift": float(handle_drift),
                        "duration": handle_duration
                    },
                    "breakout_level": float(breakout_level),
                    "target": float(breakout_level + cup_height),  # Measured move target
                    "quality_score": quality_score,
                    "completion_status": completion_status,
                    "cup_symmetry": float(cup_symmetry),
                    "handle_quality": float(handle_quality)
                }
                
                patterns.append(pattern)
                break  # Found a valid pattern for this start point
        
        return patterns 

    @staticmethod
    def detect_triple_top(prices: pd.Series, order: int = 5, tolerance: float = 0.02) -> List[Dict[str, Any]]:
        """
        Detect Triple Top patterns.
        
        Args:
            prices: Price series
            order: Order for peak detection
            tolerance: Price tolerance for peak similarity
            
        Returns:
            List of detected Triple Top patterns
        """
        if len(prices) < 30:
            return []
        
        peaks, _ = PatternRecognition.identify_peaks_lows(prices, order=order)
        patterns = []
        
        for i in range(len(peaks) - 2):
            peak1, peak2, peak3 = peaks[i], peaks[i+1], peaks[i+2]
            
            # Get peak prices
            price1, price2, price3 = prices.iloc[peak1], prices.iloc[peak2], prices.iloc[peak3]
            
            # Check if peaks are similar in price
            max_price = max(price1, price2, price3)
            min_price = min(price1, price2, price3)
            price_range = (max_price - min_price) / max_price
            
            if price_range > tolerance:
                continue
            
            # Check if peaks are well-spaced
            spacing1 = peak2 - peak1
            spacing2 = peak3 - peak2
            
            if spacing1 < 5 or spacing2 < 5:  # Minimum spacing
                continue
            
            # Check for valleys between peaks
            valley1 = prices.iloc[peak1:peak2].min()
            valley2 = prices.iloc[peak2:peak3].min()
            
            # Valleys should be significantly lower than peaks
            valley1_ratio = (max_price - valley1) / max_price
            valley2_ratio = (max_price - valley2) / max_price
            
            if valley1_ratio < 0.03 or valley2_ratio < 0.03:  # At least 3% drop
                continue
            
            # Calculate pattern metrics
            avg_peak_price = (price1 + price2 + price3) / 3
            support_level = max(valley1, valley2)
            
            # Pattern quality assessment
            quality_score = 0
            
            # Peak similarity
            peak_similarity = 1 - price_range
            quality_score += peak_similarity * 30
            
            # Valley depth
            avg_valley_ratio = (valley1_ratio + valley2_ratio) / 2
            quality_score += min(30, avg_valley_ratio * 100)
            
            # Spacing consistency
            spacing_ratio = min(spacing1, spacing2) / max(spacing1, spacing2)
            quality_score += spacing_ratio * 20
            
            # Pattern completion
            current_price = prices.iloc[-1]
            if current_price < support_level:
                completion_status = "completed"
                quality_score += 20
            else:
                completion_status = "forming"
            
            pattern = {
                "type": "triple_top",
                "peaks": [
                    {
                        "index": int(peak1),
                        "price": float(price1),
                        "date": str(prices.index[peak1]) if hasattr(prices.index, 'strftime') else str(peak1)
                    },
                    {
                        "index": int(peak2),
                        "price": float(price2),
                        "date": str(prices.index[peak2]) if hasattr(prices.index, 'strftime') else str(peak2)
                    },
                    {
                        "index": int(peak3),
                        "price": float(price3),
                        "date": str(prices.index[peak3]) if hasattr(prices.index, 'strftime') else str(peak3)
                    }
                ],
                "valleys": [
                    {
                        "price": float(valley1),
                        "ratio": float(valley1_ratio)
                    },
                    {
                        "price": float(valley2),
                        "ratio": float(valley2_ratio)
                    }
                ],
                "support_level": float(support_level),
                "target": float(support_level - (avg_peak_price - support_level)),  # Measured move target
                "quality_score": quality_score,
                "completion_status": completion_status,
                "peak_similarity": float(peak_similarity),
                "avg_valley_ratio": float(avg_valley_ratio),
                "spacing_consistency": float(spacing_ratio)
            }
            
            patterns.append(pattern)
        
        return patterns

    @staticmethod
    def detect_triple_bottom(prices: pd.Series, order: int = 5, tolerance: float = 0.02) -> List[Dict[str, Any]]:
        """
        Detect Triple Bottom patterns.
        
        Args:
            prices: Price series
            order: Order for low detection
            tolerance: Price tolerance for low similarity
            
        Returns:
            List of detected Triple Bottom patterns
        """
        if len(prices) < 30:
            return []
        
        _, lows = PatternRecognition.identify_peaks_lows(prices, order=order)
        patterns = []
        
        for i in range(len(lows) - 2):
            low1, low2, low3 = lows[i], lows[i+1], lows[i+2]
            
            # Get low prices
            price1, price2, price3 = prices.iloc[low1], prices.iloc[low2], prices.iloc[low3]
            
            # Check if lows are similar in price
            max_price = max(price1, price2, price3)
            min_price = min(price1, price2, price3)
            price_range = (max_price - min_price) / max_price
            
            if price_range > tolerance:
                continue
            
            # Check if lows are well-spaced
            spacing1 = low2 - low1
            spacing2 = low3 - low2
            
            if spacing1 < 5 or spacing2 < 5:  # Minimum spacing
                continue
            
            # Check for peaks between lows
            peak1 = prices.iloc[low1:low2].max()
            peak2 = prices.iloc[low2:low3].max()
            
            # Peaks should be significantly higher than lows
            peak1_ratio = (peak1 - min_price) / min_price
            peak2_ratio = (peak2 - min_price) / min_price
            
            if peak1_ratio < 0.03 or peak2_ratio < 0.03:  # At least 3% rise
                continue
            
            # Calculate pattern metrics
            avg_low_price = (price1 + price2 + price3) / 3
            resistance_level = min(peak1, peak2)
            
            # Pattern quality assessment
            quality_score = 0
            
            # Low similarity
            low_similarity = 1 - price_range
            quality_score += low_similarity * 30
            
            # Peak height
            avg_peak_ratio = (peak1_ratio + peak2_ratio) / 2
            quality_score += min(30, avg_peak_ratio * 100)
            
            # Spacing consistency
            spacing_ratio = min(spacing1, spacing2) / max(spacing1, spacing2)
            quality_score += spacing_ratio * 20
            
            # Pattern completion
            current_price = prices.iloc[-1]
            if current_price > resistance_level:
                completion_status = "completed"
                quality_score += 20
            else:
                completion_status = "forming"
            
            pattern = {
                "type": "triple_bottom",
                "lows": [
                    {
                        "index": int(low1),
                        "price": float(price1),
                        "date": str(prices.index[low1]) if hasattr(prices.index, 'strftime') else str(low1)
                    },
                    {
                        "index": int(low2),
                        "price": float(price2),
                        "date": str(prices.index[low2]) if hasattr(prices.index, 'strftime') else str(low2)
                    },
                    {
                        "index": int(low3),
                        "price": float(price3),
                        "date": str(prices.index[low3]) if hasattr(prices.index, 'strftime') else str(low3)
                    }
                ],
                "peaks": [
                    {
                        "price": float(peak1),
                        "ratio": float(peak1_ratio)
                    },
                    {
                        "price": float(peak2),
                        "ratio": float(peak2_ratio)
                    }
                ],
                "resistance_level": float(resistance_level),
                "target": float(resistance_level + (resistance_level - avg_low_price)),  # Measured move target
                "quality_score": quality_score,
                "completion_status": completion_status,
                "low_similarity": float(low_similarity),
                "avg_peak_ratio": float(avg_peak_ratio),
                "spacing_consistency": float(spacing_ratio)
            }
            
            patterns.append(pattern)
        
        return patterns

    @staticmethod
    def detect_wedge_patterns(prices: pd.Series, min_points: int = 6, min_duration: int = 20) -> List[Dict[str, Any]]:
        """
        Detect Wedge patterns (Rising Wedge, Falling Wedge).
        
        Args:
            prices: Price series
            min_points: Minimum number of swing points
            min_duration: Minimum duration for pattern
            
        Returns:
            List of detected Wedge patterns
        """
        if len(prices) < min_duration:
            return []
        
        patterns = []
        
        for start_idx in range(len(prices) - min_duration):
            for end_idx in range(start_idx + min_duration, len(prices)):
                segment = prices.iloc[start_idx:end_idx + 1]
                
                # Find swing points in segment
                order = max(min_points // 6, 3)
                highs = argrelextrema(segment.values, np.greater, order=order)[0]
                lows = argrelextrema(segment.values, np.less, order=order)[0]
                
                if len(highs) < 2 or len(lows) < 2:
                    continue
                
                # Convert to absolute indices
                x_highs = start_idx + highs
                x_lows = start_idx + lows
                y_highs = prices.iloc[x_highs]
                y_lows = prices.iloc[x_lows]
                
                # Fit regression lines
                if len(x_highs) >= 2 and len(x_lows) >= 2:
                    slope_highs, _ = np.polyfit(x_highs, y_highs, 1)
                    slope_lows, _ = np.polyfit(x_lows, y_lows, 1)
                    
                    # Detect wedge type
                    wedge_type = None
                    quality_score = 0
                    
                    # Rising Wedge: Both lines have positive slopes, highs slope > lows slope
                    if slope_highs > 0 and slope_lows > 0 and slope_highs > slope_lows:
                        wedge_type = "rising_wedge"
                        # Check convergence
                        convergence = slope_highs - slope_lows
                        quality_score = min(100, convergence * 1000)
                    
                    # Falling Wedge: Both lines have negative slopes, highs slope > lows slope
                    elif slope_highs < 0 and slope_lows < 0 and slope_highs > slope_lows:
                        wedge_type = "falling_wedge"
                        # Check convergence
                        convergence = slope_highs - slope_lows
                        quality_score = min(100, convergence * 1000)
                    
                    if wedge_type and quality_score > 20:
                        # Calculate pattern metrics
                        duration = end_idx - start_idx
                        price_range = (segment.max() - segment.min()) / segment.min()
                        
                        # Pattern completion
                        current_price = prices.iloc[-1]
                        if wedge_type == "rising_wedge":
                            # Bearish pattern - price should break below lower line
                            lower_line_end = slope_lows * end_idx + np.polyfit(x_lows, y_lows, 1)[1]
                            completion_status = "completed" if current_price < lower_line_end else "forming"
                        else:  # falling_wedge
                            # Bullish pattern - price should break above upper line
                            upper_line_end = slope_highs * end_idx + np.polyfit(x_highs, y_highs, 1)[1]
                            completion_status = "completed" if current_price > upper_line_end else "forming"
                        
                        # Calculate target based on pattern height
                        pattern_height = segment.max() - segment.min()
                        if wedge_type == "rising_wedge":
                            target = current_price - pattern_height
                        else:
                            target = current_price + pattern_height
                        
                        pattern = {
                            "type": wedge_type,
                            "start_index": int(start_idx),
                            "end_index": int(end_idx),
                            "duration": duration,
                            "slope_highs": float(slope_highs),
                            "slope_lows": float(slope_lows),
                            "convergence": float(slope_highs - slope_lows),
                            "price_range": float(price_range),
                            "quality_score": quality_score,
                            "completion_status": completion_status,
                            "target": float(target),
                            "swing_points": {
                                "highs": [{"index": int(x), "price": float(prices.iloc[x])} for x in x_highs],
                                "lows": [{"index": int(x), "price": float(prices.iloc[x])} for x in x_lows]
                            }
                        }
                        
                        patterns.append(pattern)
        
        return patterns

    @staticmethod
    def detect_channel_patterns(prices: pd.Series, min_points: int = 4, min_duration: int = 15) -> List[Dict[str, Any]]:
        """
        Detect Channel patterns (Horizontal, Ascending, Descending).
        
        Args:
            prices: Price series
            min_points: Minimum number of swing points
            min_duration: Minimum duration for pattern
            
        Returns:
            List of detected Channel patterns
        """
        if len(prices) < min_duration:
            return []
        
        patterns = []
        
        for start_idx in range(len(prices) - min_duration):
            for end_idx in range(start_idx + min_duration, len(prices)):
                segment = prices.iloc[start_idx:end_idx + 1]
                
                # Find swing points in segment
                order = max(min_points // 4, 2)
                highs = argrelextrema(segment.values, np.greater, order=order)[0]
                lows = argrelextrema(segment.values, np.less, order=order)[0]
                
                if len(highs) < 2 or len(lows) < 2:
                    continue
                
                # Convert to absolute indices
                x_highs = start_idx + highs
                x_lows = start_idx + lows
                y_highs = prices.iloc[x_highs]
                y_lows = prices.iloc[x_lows]
                
                # Fit regression lines
                if len(x_highs) >= 2 and len(x_lows) >= 2:
                    slope_highs, intercept_highs = np.polyfit(x_highs, y_highs, 1)
                    slope_lows, intercept_lows = np.polyfit(x_lows, y_lows, 1)
                    
                    # Calculate channel characteristics
                    slope_diff = abs(slope_highs - slope_lows)
                    avg_slope = (slope_highs + slope_lows) / 2
                    
                    # Channel quality assessment
                    quality_score = 0
                    channel_type = None
                    
                    # Check if lines are roughly parallel (similar slopes)
                    if slope_diff < 0.001:  # Very similar slopes
                        quality_score += 40
                        
                        # Determine channel type based on slope
                        if abs(avg_slope) < 0.0001:
                            channel_type = "horizontal_channel"
                            quality_score += 20
                        elif avg_slope > 0.0001:
                            channel_type = "ascending_channel"
                            quality_score += 20
                        else:
                            channel_type = "descending_channel"
                            quality_score += 20
                        
                        # Check channel width consistency
                        channel_widths = []
                        for i in range(len(x_highs)):
                            for j in range(len(x_lows)):
                                if abs(x_highs[i] - x_lows[j]) < 5:  # Close in time
                                    width = y_highs.iloc[i] - y_lows.iloc[j]
                                    channel_widths.append(width)
                        
                        if channel_widths:
                            width_std = np.std(channel_widths)
                            width_mean = np.mean(channel_widths)
                            width_cv = width_std / width_mean if width_mean > 0 else 1
                            
                            # Lower coefficient of variation = better quality
                            width_quality = max(0, 40 - width_cv * 100)
                            quality_score += width_quality
                        
                        # Check for touches
                        touches = len(highs) + len(lows)
                        if touches >= 4:
                            quality_score += 20
                        
                        if channel_type and quality_score > 40:
                            # Calculate pattern metrics
                            duration = end_idx - start_idx
                            channel_height = segment.max() - segment.min()
                            
                            # Pattern completion
                            current_price = prices.iloc[-1]
                            upper_line_current = slope_highs * len(prices) + intercept_highs
                            lower_line_current = slope_lows * len(prices) + intercept_lows
                            
                            if current_price > upper_line_current or current_price < lower_line_current:
                                completion_status = "completed"
                                quality_score += 20
                            else:
                                completion_status = "forming"
                            
                            # Calculate target based on channel height
                            if current_price > upper_line_current:
                                target = current_price + channel_height
                            elif current_price < lower_line_current:
                                target = current_price - channel_height
                            else:
                                target = current_price + channel_height  # Default
                            
                            pattern = {
                                "type": channel_type,
                                "start_index": int(start_idx),
                                "end_index": int(end_idx),
                                "duration": duration,
                                "slope_highs": float(slope_highs),
                                "slope_lows": float(slope_lows),
                                "slope_difference": float(slope_diff),
                                "avg_slope": float(avg_slope),
                                "channel_height": float(channel_height),
                                "quality_score": quality_score,
                                "completion_status": completion_status,
                                "target": float(target),
                                "touches": touches,
                                "swing_points": {
                                    "highs": [{"index": int(x), "price": float(prices.iloc[x])} for x in x_highs],
                                    "lows": [{"index": int(x), "price": float(prices.iloc[x])} for x in x_lows]
                                }
                            }
                            
                            patterns.append(pattern)
        
        return patterns 

    @staticmethod
    def detect_candlestick_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect classic candlestick patterns in OHLCV data.
        Patterns detected: doji, hammer, inverted hammer, shooting star, bullish engulfing, bearish engulfing, hanging man, etc.
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close'] (and optionally 'date' as index)
        Returns:
            List of dicts, each with keys: 'type', 'index', 'open', 'high', 'low', 'close', and pattern-specific metrics
        """
        patterns = []
        for i in range(1, len(df)):
            o = df['open'].iloc[i]
            h = df['high'].iloc[i]
            l = df['low'].iloc[i]
            c = df['close'].iloc[i]
            prev_o = df['open'].iloc[i-1]
            prev_c = df['close'].iloc[i-1]
            body = abs(c - o)
            upper_shadow = h - max(c, o)
            lower_shadow = min(c, o) - l
            range_ = h - l
            # Avoid division by zero
            if range_ == 0:
                continue
            body_pct = body / range_
            upper_shadow_pct = upper_shadow / range_
            lower_shadow_pct = lower_shadow / range_
            # Doji
            if body_pct < 0.1 and upper_shadow_pct > 0.2 and lower_shadow_pct > 0.2:
                patterns.append({
                    'type': 'doji',
                    'index': i,
                    'open': o, 'high': h, 'low': l, 'close': c,
                    'body_size': body, 'upper_shadow': upper_shadow, 'lower_shadow': lower_shadow,
                    'quality_score': 1 - body_pct
                })
            # Hammer
            elif body_pct < 0.3 and lower_shadow_pct > 0.5 and upper_shadow_pct < 0.2 and c > o:
                patterns.append({
                    'type': 'hammer',
                    'index': i,
                    'open': o, 'high': h, 'low': l, 'close': c,
                    'body_size': body, 'upper_shadow': upper_shadow, 'lower_shadow': lower_shadow,
                    'quality_score': lower_shadow_pct - body_pct
                })
            # Inverted Hammer
            elif body_pct < 0.3 and upper_shadow_pct > 0.5 and lower_shadow_pct < 0.2 and c > o:
                patterns.append({
                    'type': 'inverted_hammer',
                    'index': i,
                    'open': o, 'high': h, 'low': l, 'close': c,
                    'body_size': body, 'upper_shadow': upper_shadow, 'lower_shadow': lower_shadow,
                    'quality_score': upper_shadow_pct - body_pct
                })
            # Shooting Star
            elif body_pct < 0.3 and upper_shadow_pct > 0.5 and lower_shadow_pct < 0.2 and c < o:
                patterns.append({
                    'type': 'shooting_star',
                    'index': i,
                    'open': o, 'high': h, 'low': l, 'close': c,
                    'body_size': body, 'upper_shadow': upper_shadow, 'lower_shadow': lower_shadow,
                    'quality_score': upper_shadow_pct - body_pct
                })
            # Hanging Man
            elif body_pct < 0.3 and lower_shadow_pct > 0.5 and upper_shadow_pct < 0.2 and c < o:
                patterns.append({
                    'type': 'hanging_man',
                    'index': i,
                    'open': o, 'high': h, 'low': l, 'close': c,
                    'body_size': body, 'upper_shadow': upper_shadow, 'lower_shadow': lower_shadow,
                    'quality_score': lower_shadow_pct - body_pct
                })
            # Bullish Engulfing
            elif c > o and prev_c < prev_o and c > prev_o and o < prev_c:
                patterns.append({
                    'type': 'bullish_engulfing',
                    'index': i,
                    'engulfed_index': i-1,
                    'open': o, 'high': h, 'low': l, 'close': c,
                    'prev_open': prev_o, 'prev_close': prev_c,
                    'body_size': body,
                    'quality_score': (c - o) / (prev_o - prev_c + 1e-9)
                })
            # Bearish Engulfing
            elif c < o and prev_c > prev_o and c < prev_o and o > prev_c:
                patterns.append({
                    'type': 'bearish_engulfing',
                    'index': i,
                    'engulfed_index': i-1,
                    'open': o, 'high': h, 'low': l, 'close': c,
                    'prev_open': prev_o, 'prev_close': prev_c,
                    'body_size': body,
                    'quality_score': (o - c) / (prev_c - prev_o + 1e-9)
                })
        return patterns 

    @staticmethod
    def backtest_pattern(
        df: pd.DataFrame,
        pattern_func: callable,
        window: int = 100,
        hold_period: int = 10
    ) -> dict:
        """
        Backtest a pattern detection function over a DataFrame.
        Args:
            df: DataFrame with OHLCV data
            pattern_func: function that takes a price series and returns pattern indices or spans
            window: int, sliding window size for pattern detection
            hold_period: int, number of bars to hold after pattern completion
        Returns:
            dict with win_rate, avg_return, expectancy, n_trades, returns, etc.
        """
        results = []
        prices = df['close']
        for start in range(0, len(df) - window - hold_period):
            sub_df = df.iloc[start:start+window]
            sub_prices = sub_df['close']
            # Detect patterns in this window
            patterns = pattern_func(sub_prices)
            for pattern in patterns:
                # For tuple patterns (e.g., (start_idx, end_idx)), use end_idx as completion
                if isinstance(pattern, (tuple, list)) and len(pattern) > 1:
                    entry_idx = pattern[-1]
                elif isinstance(pattern, int):
                    entry_idx = pattern
                else:
                    continue
                global_entry = start + entry_idx
                if global_entry + hold_period >= len(df):
                    continue
                entry_price = df['close'].iloc[global_entry]
                exit_price = df['close'].iloc[global_entry + hold_period]
                ret = (exit_price - entry_price) / entry_price
                results.append(ret)
        n_trades = len(results)
        win_trades = [r for r in results if r > 0]
        loss_trades = [r for r in results if r <= 0]
        win_rate = len(win_trades) / n_trades if n_trades > 0 else 0
        avg_return = np.mean(results) if results else 0
        expectancy = (np.mean(win_trades) if win_trades else 0) * win_rate + (np.mean(loss_trades) if loss_trades else 0) * (1 - win_rate)
        return {
            'win_rate': win_rate,
            'avg_return': avg_return,
            'expectancy': expectancy,
            'n_trades': n_trades,
            'returns': results
        } 