#!/usr/bin/env python3
"""
Volume Trend Momentum Agent - Data Processing Module

Analyzes volume trend momentum for trend continuation assessment
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class VolumeTrendMomentumProcessor:
    def __init__(self):
        self.short_period = 10  # Short-term trend period
        self.medium_period = 20  # Medium-term trend period
        self.long_period = 50   # Long-term trend period
        self.momentum_threshold = 0.15  # 15% change threshold for momentum signals
        self.cycle_min_length = 5  # Minimum cycle length in days
    
    def process_volume_trend_momentum_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Main processing function for volume trend momentum analysis"""
        try:
            # 1. Calculate volume trends (multiple timeframes)
            volume_trends = self._calculate_volume_trends(data)
            
            # 2. Analyze volume momentum
            momentum_analysis = self._analyze_volume_momentum(data)
            
            # 3. Identify momentum cycles and phases
            cycle_analysis = self._analyze_momentum_cycles(data, momentum_analysis)
            
            # 4. Compare volume vs price momentum
            momentum_comparison = self._compare_volume_price_momentum(data, momentum_analysis)
            
            # 5. Calculate future trend implications
            future_implications = self._calculate_future_implications(
                volume_trends, momentum_analysis, cycle_analysis
            )
            
            # 6. Assess momentum sustainability
            sustainability = self._assess_momentum_sustainability(
                volume_trends, cycle_analysis, data
            )
            
            # 7. Quality assessment
            quality = self._assess_analysis_quality(volume_trends, momentum_analysis)
            
            return {
                'volume_trend_analysis': volume_trends,
                'momentum_analysis': momentum_analysis,
                'cycle_analysis': cycle_analysis,
                'momentum_comparison': momentum_comparison,
                'future_implications': future_implications,
                'sustainability_assessment': sustainability,
                'quality_assessment': quality,
                'volume_trend_direction': volume_trends.get('primary_trend_direction', 'unknown'),
                'trend_strength': volume_trends.get('trend_strength_classification', 'unknown'),
                'momentum_phase': cycle_analysis.get('current_phase', 'unknown')
            }
            
        except Exception as e:
            return {'error': f"Volume trend momentum analysis failed: {str(e)}"}
    
    def _calculate_volume_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume trends across multiple timeframes"""
        try:
            volume_trends = {}
            
            # Calculate trends for different periods
            for period_name, period in [('short', self.short_period), 
                                      ('medium', self.medium_period), 
                                      ('long', self.long_period)]:
                
                if len(data) < period:
                    continue
                
                # Get recent data for this period
                recent_data = data.tail(period)
                volumes = recent_data['volume'].values
                
                # Linear regression for trend
                x = np.arange(len(volumes))
                coeffs = np.polyfit(x, volumes, 1)
                trend_slope = coeffs[0]
                
                # Normalize slope by average volume
                avg_volume = volumes.mean()
                normalized_slope = trend_slope / avg_volume if avg_volume > 0 else 0
                
                # Calculate R-squared for trend strength
                y_pred = np.poly1d(coeffs)(x)
                ss_res = np.sum((volumes - y_pred) ** 2)
                ss_tot = np.sum((volumes - np.mean(volumes)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Trend direction and strength
                if normalized_slope > 0.02:
                    direction = 'increasing'
                elif normalized_slope < -0.02:
                    direction = 'decreasing'
                else:
                    direction = 'stable'
                
                strength = self._classify_trend_strength(abs(normalized_slope), r_squared)
                
                volume_trends[f'{period_name}_term'] = {
                    'period_days': period,
                    'trend_slope': trend_slope,
                    'normalized_slope': normalized_slope,
                    'direction': direction,
                    'strength': strength,
                    'r_squared': r_squared,
                    'average_volume': avg_volume,
                    'start_volume': volumes[0],
                    'end_volume': volumes[-1],
                    'volume_change_pct': ((volumes[-1] - volumes[0]) / volumes[0]) * 100 if volumes[0] > 0 else 0
                }
            
            # Determine primary trend (medium-term focus)
            primary_trend = volume_trends.get('medium_term', {})
            
            # Trend agreement analysis
            short_dir = volume_trends.get('short_term', {}).get('direction', 'unknown')
            medium_dir = volume_trends.get('medium_term', {}).get('direction', 'unknown')
            long_dir = volume_trends.get('long_term', {}).get('direction', 'unknown')
            
            trend_agreement = self._assess_trend_agreement(short_dir, medium_dir, long_dir)
            
            return {
                **volume_trends,
                'primary_trend_direction': primary_trend.get('direction', 'unknown'),
                'primary_trend_strength': primary_trend.get('strength', 'unknown'),
                'trend_strength_classification': primary_trend.get('strength', 'unknown'),
                'trend_agreement': trend_agreement,
                'trend_agreement_score': self._calculate_agreement_score(short_dir, medium_dir, long_dir)
            }
            
        except Exception as e:
            return {'error': f"Volume trend calculation failed: {str(e)}"}
    
    def _analyze_volume_momentum(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume momentum using rate of change and momentum oscillators"""
        try:
            volumes = data['volume'].values
            
            if len(volumes) < self.short_period:
                return {'error': 'Insufficient data for momentum analysis'}
            
            # Volume Rate of Change (ROC) - multiple periods
            roc_periods = [5, 10, 20]
            momentum_indicators = {}
            
            for period in roc_periods:
                if len(volumes) >= period:
                    roc_values = []
                    for i in range(period, len(volumes)):
                        if volumes[i - period] > 0:
                            roc = ((volumes[i] - volumes[i - period]) / volumes[i - period]) * 100
                            roc_values.append(roc)
                        else:
                            roc_values.append(0)
                    
                    current_roc = roc_values[-1] if roc_values else 0
                    avg_roc = np.mean(roc_values) if roc_values else 0
                    
                    momentum_indicators[f'roc_{period}d'] = {
                        'current_value': current_roc,
                        'average_value': avg_roc,
                        'values': roc_values,
                        'trend': 'positive' if current_roc > avg_roc else 'negative'
                    }
            
            # Volume momentum oscillator (similar to price momentum)
            if len(volumes) >= 14:
                momentum_values = []
                for i in range(14, len(volumes)):
                    momentum = volumes[i] - volumes[i - 14]
                    momentum_values.append(momentum)
                
                current_momentum = momentum_values[-1] if momentum_values else 0
                momentum_ma = np.mean(momentum_values[-10:]) if len(momentum_values) >= 10 else current_momentum
            else:
                momentum_values = []
                current_momentum = 0
                momentum_ma = 0
            
            # Volume acceleration (second derivative)
            volume_acceleration = self._calculate_volume_acceleration(volumes)
            
            # Overall momentum classification
            overall_momentum = self._classify_overall_momentum(momentum_indicators, current_momentum)
            
            # Momentum strength
            momentum_strength = self._assess_momentum_strength(momentum_indicators, volume_acceleration)
            
            return {
                'rate_of_change_indicators': momentum_indicators,
                'momentum_oscillator': {
                    'current_value': current_momentum,
                    'moving_average': momentum_ma,
                    'values': momentum_values,
                    'signal': 'bullish' if current_momentum > momentum_ma else 'bearish'
                },
                'volume_acceleration': volume_acceleration,
                'overall_momentum_direction': overall_momentum,
                'momentum_strength': momentum_strength,
                'momentum_signals': self._generate_momentum_signals(momentum_indicators, current_momentum)
            }
            
        except Exception as e:
            return {'error': f"Volume momentum analysis failed: {str(e)}"}
    
    def _analyze_momentum_cycles(self, data: pd.DataFrame, momentum_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volume momentum cycles and phases"""
        try:
            if 'error' in momentum_analysis:
                return {'error': 'Momentum analysis unavailable'}
            
            # Get momentum values for cycle analysis
            roc_10d = momentum_analysis.get('rate_of_change_indicators', {}).get('roc_10d', {})
            momentum_values = roc_10d.get('values', [])
            
            if len(momentum_values) < self.cycle_min_length * 2:
                return {'error': 'Insufficient data for cycle analysis'}
            
            # Identify momentum peaks and troughs
            peaks_troughs = self._identify_momentum_peaks_troughs(momentum_values)
            
            # Calculate cycle metrics
            cycles = self._calculate_momentum_cycles(peaks_troughs, momentum_values)
            
            # Determine current phase
            current_phase = self._determine_current_momentum_phase(
                momentum_values, peaks_troughs
            )
            
            # Phase duration analysis
            phase_duration = self._analyze_phase_duration(cycles, current_phase)
            
            # Cycle strength assessment
            cycle_strength = self._assess_cycle_strength(cycles, momentum_values)
            
            return {
                'momentum_peaks_troughs': peaks_troughs,
                'identified_cycles': cycles,
                'current_phase': current_phase,
                'phase_duration_analysis': phase_duration,
                'cycle_strength_assessment': cycle_strength,
                'cycle_count': len(cycles),
                'average_cycle_length': np.mean([c['duration'] for c in cycles]) if cycles else 0,
                'cycle_regularity': self._assess_cycle_regularity(cycles)
            }
            
        except Exception as e:
            return {'error': f"Momentum cycle analysis failed: {str(e)}"}
    
    def _compare_volume_price_momentum(self, data: pd.DataFrame, momentum_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compare volume momentum with price momentum"""
        try:
            if 'error' in momentum_analysis:
                return {'error': 'Momentum analysis unavailable'}
            
            # Calculate price momentum (similar to volume momentum)
            prices = data['close'].values
            price_momentum_values = []
            
            if len(prices) >= 14:
                for i in range(14, len(prices)):
                    price_momentum = prices[i] - prices[i - 14]
                    price_momentum_values.append(price_momentum)
            
            # Get volume momentum
            volume_momentum = momentum_analysis.get('momentum_oscillator', {})
            volume_momentum_values = volume_momentum.get('values', [])
            
            if not price_momentum_values or not volume_momentum_values:
                return {'error': 'Insufficient momentum data for comparison'}
            
            # Align the arrays (same length)
            min_length = min(len(price_momentum_values), len(volume_momentum_values))
            price_mom = price_momentum_values[-min_length:] if price_momentum_values else []
            volume_mom = volume_momentum_values[-min_length:] if volume_momentum_values else []
            
            if not price_mom or not volume_mom or min_length < 5:
                return {'error': 'Insufficient aligned momentum data'}
            
            # Calculate correlation
            correlation = np.corrcoef(price_mom, volume_mom)[0, 1] if len(price_mom) > 1 else 0
            
            # Momentum divergence analysis
            divergence_analysis = self._analyze_momentum_divergence(
                price_mom, volume_mom, data.index[-min_length:]
            )
            
            # Current momentum comparison
            current_price_momentum = price_mom[-1] if price_mom else 0
            current_volume_momentum = volume_mom[-1] if volume_mom else 0
            
            # Momentum agreement
            price_direction = 'positive' if current_price_momentum > 0 else 'negative'
            volume_direction = 'positive' if current_volume_momentum > 0 else 'negative'
            momentum_agreement = price_direction == volume_direction
            
            return {
                'momentum_correlation': correlation,
                'correlation_strength': self._classify_correlation_strength(correlation),
                'divergence_analysis': divergence_analysis,
                'current_comparison': {
                    'price_momentum': current_price_momentum,
                    'volume_momentum': current_volume_momentum,
                    'price_direction': price_direction,
                    'volume_direction': volume_direction,
                    'agreement': momentum_agreement
                },
                'momentum_signals': self._generate_comparison_signals(
                    correlation, divergence_analysis, momentum_agreement
                )
            }
            
        except Exception as e:
            return {'error': f"Momentum comparison failed: {str(e)}"}
    
    def _calculate_future_implications(self, volume_trends: Dict[str, Any], 
                                     momentum_analysis: Dict[str, Any], 
                                     cycle_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate future trend implications based on volume momentum"""
        try:
            implications = {
                'trend_continuation_probability': 0.5,
                'momentum_exhaustion_warning': False,
                'volume_acceleration_signal': 'neutral',
                'predicted_momentum_phase': 'unknown',
                'confidence_level': 'low'
            }
            
            if any('error' in analysis for analysis in [volume_trends, momentum_analysis, cycle_analysis]):
                return implications
            
            # Trend continuation probability
            trend_strength = volume_trends.get('primary_trend_strength', 'unknown')
            momentum_strength = momentum_analysis.get('momentum_strength', 'unknown')
            current_phase = cycle_analysis.get('current_phase', 'unknown')
            
            continuation_score = 0.5  # Base probability
            
            # Adjust based on trend strength
            if trend_strength in ['very_strong', 'strong']:
                continuation_score += 0.2
            elif trend_strength in ['weak', 'very_weak']:
                continuation_score -= 0.2
            
            # Adjust based on momentum strength
            if momentum_strength in ['very_strong', 'strong']:
                continuation_score += 0.15
            elif momentum_strength in ['weak', 'very_weak']:
                continuation_score -= 0.15
            
            # Adjust based on momentum phase
            if current_phase in ['building', 'peak']:
                continuation_score += 0.1
            elif current_phase in ['declining', 'trough']:
                continuation_score -= 0.1
            
            implications['trend_continuation_probability'] = max(0, min(1, continuation_score))
            
            # Momentum exhaustion warning
            volume_acceleration = momentum_analysis.get('volume_acceleration', {})
            if (current_phase == 'peak' and 
                volume_acceleration.get('trend', 'unknown') == 'decelerating'):
                implications['momentum_exhaustion_warning'] = True
            
            # Volume acceleration signal
            acc_trend = volume_acceleration.get('trend', 'unknown')
            if acc_trend == 'accelerating' and volume_acceleration.get('strength', 0) > 0.1:
                implications['volume_acceleration_signal'] = 'strong_acceleration'
            elif acc_trend == 'decelerating' and volume_acceleration.get('strength', 0) > 0.1:
                implications['volume_acceleration_signal'] = 'strong_deceleration'
            else:
                implications['volume_acceleration_signal'] = 'neutral'
            
            # Predicted next momentum phase
            if current_phase == 'building':
                implications['predicted_momentum_phase'] = 'peak'
            elif current_phase == 'peak':
                implications['predicted_momentum_phase'] = 'declining'
            elif current_phase == 'declining':
                implications['predicted_momentum_phase'] = 'trough'
            elif current_phase == 'trough':
                implications['predicted_momentum_phase'] = 'building'
            
            # Overall confidence level
            trend_agreement = volume_trends.get('trend_agreement_score', 0)
            momentum_correlation = 0.5  # Default if not available from comparison
            
            confidence_score = (trend_agreement + momentum_correlation) / 2
            
            if confidence_score > 0.8:
                implications['confidence_level'] = 'very_high'
            elif confidence_score > 0.6:
                implications['confidence_level'] = 'high'
            elif confidence_score > 0.4:
                implications['confidence_level'] = 'medium'
            elif confidence_score > 0.2:
                implications['confidence_level'] = 'low'
            else:
                implications['confidence_level'] = 'very_low'
            
            return implications
            
        except Exception as e:
            return {'error': f"Future implications calculation failed: {str(e)}"}
    
    def _assess_momentum_sustainability(self, volume_trends: Dict[str, Any], 
                                      cycle_analysis: Dict[str, Any], 
                                      data: pd.DataFrame) -> Dict[str, Any]:
        """Assess sustainability of current volume momentum"""
        try:
            sustainability = {
                'overall_sustainability': 'unknown',
                'sustainability_score': 0,
                'risk_factors': [],
                'supporting_factors': [],
                'sustainability_timeframe': 'unknown'
            }
            
            if 'error' in volume_trends or 'error' in cycle_analysis:
                return sustainability
            
            score = 50  # Base score
            
            # Volume trend consistency
            agreement_score = volume_trends.get('trend_agreement_score', 0)
            score += agreement_score * 20
            
            if agreement_score > 0.7:
                sustainability['supporting_factors'].append('strong_trend_agreement')
            elif agreement_score < 0.3:
                sustainability['risk_factors'].append('conflicting_trends')
            
            # Momentum phase assessment
            current_phase = cycle_analysis.get('current_phase', 'unknown')
            phase_duration = cycle_analysis.get('phase_duration_analysis', {})
            
            if current_phase in ['building', 'peak']:
                score += 10
                sustainability['supporting_factors'].append(f'favorable_momentum_phase_{current_phase}')
            elif current_phase in ['declining', 'trough']:
                score -= 10
                sustainability['risk_factors'].append(f'unfavorable_momentum_phase_{current_phase}')
            
            # Phase duration analysis
            if phase_duration.get('phase_maturity', 'unknown') == 'early':
                score += 5
            elif phase_duration.get('phase_maturity', 'unknown') == 'mature':
                score -= 5
                sustainability['risk_factors'].append('mature_momentum_phase')
            
            # Volume level analysis
            recent_volume = data['volume'].tail(10).mean()
            long_term_volume = data['volume'].mean()
            
            volume_ratio = recent_volume / long_term_volume if long_term_volume > 0 else 1
            
            if volume_ratio > 1.2:
                score += 10
                sustainability['supporting_factors'].append('elevated_volume_levels')
            elif volume_ratio < 0.8:
                score -= 10
                sustainability['risk_factors'].append('below_average_volume')
            
            # Trend strength
            trend_strength = volume_trends.get('primary_trend_strength', 'unknown')
            if trend_strength in ['very_strong', 'strong']:
                score += 15
                sustainability['supporting_factors'].append('strong_trend_strength')
            elif trend_strength in ['weak', 'very_weak']:
                score -= 15
                sustainability['risk_factors'].append('weak_trend_strength')
            
            # Final score and classification
            sustainability['sustainability_score'] = max(0, min(100, score))
            
            if score >= 70:
                sustainability['overall_sustainability'] = 'very_high'
                sustainability['sustainability_timeframe'] = 'long_term'
            elif score >= 60:
                sustainability['overall_sustainability'] = 'high'
                sustainability['sustainability_timeframe'] = 'medium_term'
            elif score >= 50:
                sustainability['overall_sustainability'] = 'moderate'
                sustainability['sustainability_timeframe'] = 'short_term'
            elif score >= 40:
                sustainability['overall_sustainability'] = 'low'
                sustainability['sustainability_timeframe'] = 'very_short_term'
            else:
                sustainability['overall_sustainability'] = 'very_low'
                sustainability['sustainability_timeframe'] = 'unsustainable'
            
            return sustainability
            
        except Exception as e:
            return {'error': f"Sustainability assessment failed: {str(e)}"}
    
    # Helper methods
    def _classify_trend_strength(self, normalized_slope: float, r_squared: float) -> str:
        """Classify trend strength based on slope and R-squared"""
        if abs(normalized_slope) > 0.1 and r_squared > 0.7:
            return 'very_strong'
        elif abs(normalized_slope) > 0.05 and r_squared > 0.5:
            return 'strong'
        elif abs(normalized_slope) > 0.02 and r_squared > 0.3:
            return 'moderate'
        elif abs(normalized_slope) > 0.01:
            return 'weak'
        else:
            return 'very_weak'
    
    def _assess_trend_agreement(self, short: str, medium: str, long: str) -> str:
        """Assess agreement between different timeframe trends"""
        trends = [short, medium, long]
        trends = [t for t in trends if t != 'unknown']
        
        if not trends:
            return 'unknown'
        
        if len(set(trends)) == 1:
            return 'strong_agreement'
        elif trends.count(trends[0]) >= 2:
            return 'moderate_agreement'
        else:
            return 'conflicting'
    
    def _calculate_agreement_score(self, short: str, medium: str, long: str) -> float:
        """Calculate numerical agreement score"""
        trends = [short, medium, long]
        trends = [t for t in trends if t != 'unknown']
        
        if not trends:
            return 0.0
        
        if len(set(trends)) == 1:
            return 1.0
        elif len(trends) == 2:
            return 0.5
        else:
            # Count most common trend
            most_common_count = max([trends.count(t) for t in set(trends)])
            return most_common_count / len(trends)
    
    def _calculate_volume_acceleration(self, volumes: np.ndarray) -> Dict[str, Any]:
        """Calculate volume acceleration (second derivative)"""
        if len(volumes) < 3:
            return {'error': 'Insufficient data for acceleration calculation'}
        
        # First derivative (velocity)
        velocity = np.diff(volumes)
        
        # Second derivative (acceleration)
        acceleration = np.diff(velocity)
        
        # Recent acceleration trend
        if len(acceleration) >= 5:
            recent_acceleration = acceleration[-5:]
            avg_acceleration = np.mean(recent_acceleration)
            
            if avg_acceleration > np.std(acceleration) * 0.5:
                trend = 'accelerating'
            elif avg_acceleration < -np.std(acceleration) * 0.5:
                trend = 'decelerating'
            else:
                trend = 'stable'
                
            strength = abs(avg_acceleration) / np.std(acceleration) if np.std(acceleration) > 0 else 0
        else:
            trend = 'unknown'
            avg_acceleration = 0
            strength = 0
        
        return {
            'values': acceleration.tolist(),
            'current_acceleration': acceleration[-1] if len(acceleration) > 0 else 0,
            'average_acceleration': avg_acceleration,
            'trend': trend,
            'strength': strength
        }
    
    def _classify_overall_momentum(self, momentum_indicators: Dict, current_momentum: float) -> str:
        """Classify overall momentum direction"""
        positive_indicators = 0
        negative_indicators = 0
        
        for indicator in momentum_indicators.values():
            if indicator.get('current_value', 0) > 0:
                positive_indicators += 1
            elif indicator.get('current_value', 0) < 0:
                negative_indicators += 1
        
        if current_momentum > 0:
            positive_indicators += 1
        elif current_momentum < 0:
            negative_indicators += 1
        
        if positive_indicators > negative_indicators:
            return 'positive'
        elif negative_indicators > positive_indicators:
            return 'negative'
        else:
            return 'neutral'
    
    def _assess_momentum_strength(self, momentum_indicators: Dict, volume_acceleration: Dict) -> str:
        """Assess overall momentum strength"""
        if not momentum_indicators or 'error' in volume_acceleration:
            return 'unknown'
        
        # Average ROC strength
        roc_strengths = []
        for indicator in momentum_indicators.values():
            current_val = abs(indicator.get('current_value', 0))
            avg_val = abs(indicator.get('average_value', 0))
            if avg_val > 0:
                strength = current_val / avg_val
                roc_strengths.append(strength)
        
        avg_roc_strength = np.mean(roc_strengths) if roc_strengths else 0
        
        # Acceleration strength
        acc_strength = volume_acceleration.get('strength', 0)
        
        # Combined strength
        combined_strength = (avg_roc_strength + acc_strength) / 2
        
        if combined_strength > 2:
            return 'very_strong'
        elif combined_strength > 1.5:
            return 'strong'
        elif combined_strength > 1:
            return 'moderate'
        elif combined_strength > 0.5:
            return 'weak'
        else:
            return 'very_weak'
    
    def _generate_momentum_signals(self, momentum_indicators: Dict, current_momentum: float) -> List[str]:
        """Generate momentum-based signals"""
        signals = []
        
        # ROC-based signals
        for period, indicator in momentum_indicators.items():
            current_val = indicator.get('current_value', 0)
            avg_val = indicator.get('average_value', 0)
            
            if current_val > avg_val * 1.5:
                signals.append(f'strong_positive_{period}')
            elif current_val < avg_val * -1.5:
                signals.append(f'strong_negative_{period}')
        
        # Overall momentum signals
        if current_momentum > 0:
            signals.append('momentum_bullish')
        elif current_momentum < 0:
            signals.append('momentum_bearish')
        
        return signals
    
    def _identify_momentum_peaks_troughs(self, momentum_values: List[float]) -> Dict[str, List]:
        """Identify peaks and troughs in momentum data"""
        if len(momentum_values) < self.cycle_min_length:
            return {'peaks': [], 'troughs': []}
        
        peaks = []
        troughs = []
        window = 3
        
        for i in range(window, len(momentum_values) - window):
            current_val = momentum_values[i]
            
            # Check for peak
            is_peak = True
            for j in range(i - window, i + window + 1):
                if j != i and momentum_values[j] >= current_val:
                    is_peak = False
                    break
            
            if is_peak:
                peaks.append({'index': i, 'value': current_val})
            
            # Check for trough
            is_trough = True
            for j in range(i - window, i + window + 1):
                if j != i and momentum_values[j] <= current_val:
                    is_trough = False
                    break
            
            if is_trough:
                troughs.append({'index': i, 'value': current_val})
        
        return {'peaks': peaks, 'troughs': troughs}
    
    def _calculate_momentum_cycles(self, peaks_troughs: Dict, momentum_values: List[float]) -> List[Dict]:
        """Calculate momentum cycles from peaks and troughs"""
        peaks = peaks_troughs.get('peaks', [])
        troughs = peaks_troughs.get('troughs', [])
        
        if not peaks or not troughs:
            return []
        
        # Combine and sort by index
        all_points = []
        for peak in peaks:
            all_points.append(('peak', peak['index'], peak['value']))
        for trough in troughs:
            all_points.append(('trough', trough['index'], trough['value']))
        
        all_points.sort(key=lambda x: x[1])
        
        # Identify complete cycles (trough to trough or peak to peak)
        cycles = []
        
        for i in range(len(all_points) - 1):
            current_type, current_idx, current_val = all_points[i]
            next_type, next_idx, next_val = all_points[i + 1]
            
            # Look for complete cycles
            if i < len(all_points) - 2:
                third_type, third_idx, third_val = all_points[i + 2]
                
                if current_type == third_type:  # Same type = complete cycle
                    cycle_duration = third_idx - current_idx
                    cycle_amplitude = abs(next_val - min(current_val, third_val))
                    
                    cycles.append({
                        'start_index': current_idx,
                        'end_index': third_idx,
                        'duration': cycle_duration,
                        'amplitude': cycle_amplitude,
                        'start_value': current_val,
                        'mid_value': next_val,
                        'end_value': third_val,
                        'cycle_type': f"{current_type}_to_{current_type}"
                    })
        
        return cycles
    
    def _determine_current_momentum_phase(self, momentum_values: List[float], 
                                        peaks_troughs: Dict) -> str:
        """Determine current momentum phase"""
        if len(momentum_values) < 5:
            return 'unknown'
        
        recent_values = momentum_values[-5:]
        current_value = momentum_values[-1]
        
        # Recent trend
        if len(recent_values) > 2:
            recent_slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
        else:
            recent_slope = 0
        
        # Position relative to recent peaks/troughs
        peaks = peaks_troughs.get('peaks', [])
        troughs = peaks_troughs.get('troughs', [])
        
        # Find most recent peak and trough
        recent_peak = max(peaks, key=lambda x: x['index']) if peaks else None
        recent_trough = min(troughs, key=lambda x: x['index']) if troughs else None
        
        # Determine phase based on recent trend and position
        if recent_slope > 0.1:
            return 'building'
        elif recent_slope < -0.1:
            return 'declining'
        elif current_value > 0 and recent_slope >= -0.1:
            return 'peak'
        elif current_value < 0 and recent_slope <= 0.1:
            return 'trough'
        else:
            return 'transitioning'
    
    def _analyze_phase_duration(self, cycles: List[Dict], current_phase: str) -> Dict[str, Any]:
        """Analyze duration characteristics of momentum phases"""
        if not cycles:
            return {'error': 'No cycles available for analysis'}
        
        # Calculate average cycle duration
        durations = [cycle['duration'] for cycle in cycles]
        avg_duration = np.mean(durations)
        
        # Estimate current phase maturity (simplified)
        phase_maturity = 'early'  # This would need more sophisticated logic
        
        return {
            'average_cycle_duration': avg_duration,
            'phase_maturity': phase_maturity,
            'expected_phase_remaining': max(1, avg_duration // 2)  # Simplified
        }
    
    def _assess_cycle_strength(self, cycles: List[Dict], momentum_values: List[float]) -> str:
        """Assess overall strength of momentum cycles"""
        if not cycles:
            return 'unknown'
        
        amplitudes = [cycle['amplitude'] for cycle in cycles]
        avg_amplitude = np.mean(amplitudes)
        std_momentum = np.std(momentum_values)
        
        if std_momentum == 0:
            return 'unknown'
        
        normalized_amplitude = avg_amplitude / std_momentum
        
        if normalized_amplitude > 2:
            return 'very_strong'
        elif normalized_amplitude > 1.5:
            return 'strong'
        elif normalized_amplitude > 1:
            return 'moderate'
        elif normalized_amplitude > 0.5:
            return 'weak'
        else:
            return 'very_weak'
    
    def _assess_cycle_regularity(self, cycles: List[Dict]) -> str:
        """Assess regularity of momentum cycles"""
        if len(cycles) < 3:
            return 'insufficient_data'
        
        durations = [cycle['duration'] for cycle in cycles]
        duration_cv = np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else float('inf')
        
        if duration_cv < 0.3:
            return 'very_regular'
        elif duration_cv < 0.5:
            return 'regular'
        elif duration_cv < 0.8:
            return 'moderately_irregular'
        else:
            return 'irregular'
    
    def _analyze_momentum_divergence(self, price_momentum: List[float], 
                                   volume_momentum: List[float], 
                                   dates: pd.DatetimeIndex) -> Dict[str, Any]:
        """Analyze divergence between price and volume momentum"""
        if len(price_momentum) != len(volume_momentum) or len(price_momentum) < 5:
            return {'error': 'Insufficient data for divergence analysis'}
        
        # Calculate trends for both series
        x = np.arange(len(price_momentum))
        price_trend = np.polyfit(x, price_momentum, 1)[0]
        volume_trend = np.polyfit(x, volume_momentum, 1)[0]
        
        # Determine divergence
        price_direction = 'positive' if price_trend > 0 else 'negative'
        volume_direction = 'positive' if volume_trend > 0 else 'negative'
        
        has_divergence = price_direction != volume_direction
        divergence_strength = abs(price_trend - volume_trend) / (abs(price_trend) + abs(volume_trend)) if (abs(price_trend) + abs(volume_trend)) > 0 else 0
        
        divergence_type = 'none'
        if has_divergence:
            if price_direction == 'positive' and volume_direction == 'negative':
                divergence_type = 'bearish_divergence'
            elif price_direction == 'negative' and volume_direction == 'positive':
                divergence_type = 'bullish_divergence'
        
        return {
            'has_divergence': has_divergence,
            'divergence_type': divergence_type,
            'divergence_strength': divergence_strength,
            'price_momentum_trend': price_direction,
            'volume_momentum_trend': volume_direction
        }
    
    def _classify_correlation_strength(self, correlation: float) -> str:
        """Classify correlation strength"""
        abs_corr = abs(correlation)
        
        if abs_corr > 0.8:
            return 'very_strong'
        elif abs_corr > 0.6:
            return 'strong'
        elif abs_corr > 0.4:
            return 'moderate'
        elif abs_corr > 0.2:
            return 'weak'
        else:
            return 'very_weak'
    
    def _generate_comparison_signals(self, correlation: float, 
                                   divergence_analysis: Dict, 
                                   momentum_agreement: bool) -> List[str]:
        """Generate signals from momentum comparison"""
        signals = []
        
        # Correlation signals
        if abs(correlation) > 0.7:
            if correlation > 0:
                signals.append('strong_positive_correlation')
            else:
                signals.append('strong_negative_correlation')
        
        # Divergence signals
        divergence_type = divergence_analysis.get('divergence_type', 'none')
        if divergence_type != 'none':
            signals.append(divergence_type)
        
        # Agreement signals
        if momentum_agreement:
            signals.append('momentum_agreement')
        else:
            signals.append('momentum_disagreement')
        
        return signals
    
    def _assess_analysis_quality(self, volume_trends: Dict, momentum_analysis: Dict) -> Dict[str, Any]:
        """Assess overall analysis quality"""
        try:
            quality_score = 0
            factors = []
            
            # Volume trend quality
            if 'error' not in volume_trends:
                quality_score += 25
                factors.append('volume_trends_valid')
                
                if volume_trends.get('trend_agreement_score', 0) > 0.5:
                    quality_score += 15
                    factors.append('trend_agreement')
            
            # Momentum analysis quality
            if 'error' not in momentum_analysis:
                quality_score += 25
                factors.append('momentum_analysis_valid')
                
                momentum_strength = momentum_analysis.get('momentum_strength', 'unknown')
                if momentum_strength in ['strong', 'very_strong']:
                    quality_score += 15
                    factors.append('strong_momentum_signals')
            
            # Data sufficiency
            roc_indicators = momentum_analysis.get('rate_of_change_indicators', {})
            if len(roc_indicators) >= 2:
                quality_score += 20
                factors.append('multiple_momentum_indicators')
            
            # Trend consistency
            primary_strength = volume_trends.get('primary_trend_strength', 'unknown')
            if primary_strength in ['strong', 'very_strong']:
                quality_score += 15
                factors.append('strong_primary_trend')
            
            return {
                'overall_score': min(quality_score, 100),
                'quality_factors': factors,
                'analysis_reliability': self._determine_analysis_reliability(quality_score)
            }
            
        except Exception as e:
            return {
                'overall_score': 0,
                'error': f"Quality assessment failed: {str(e)}"
            }
    
    def _determine_analysis_reliability(self, score: int) -> str:
        """Determine analysis reliability based on score"""
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


def test_volume_trend_momentum_processor():
    """Test function for Volume Trend Momentum Processor"""
    print("🚀 Testing Volume Trend Momentum Processor")
    print("=" * 60)
    
    # Create test data with momentum patterns
    dates = pd.date_range(start='2024-07-01', end='2024-10-20', freq='D')
    np.random.seed(42)
    
    base_price = 2400
    base_volume = 1500000
    
    # Generate trending price data
    price_changes = np.random.normal(0.003, 0.015, len(dates))  # Slight uptrend
    prices = base_price * np.cumprod(1 + price_changes)
    
    # Generate volume with momentum cycles
    volumes = []
    for i, date in enumerate(dates):
        # Base volume with trend
        trend_factor = 1 + (i / len(dates)) * 0.2  # Growing volume trend
        
        # Add cyclical momentum
        cycle_factor = 1 + 0.3 * np.sin(2 * np.pi * i / 20)  # 20-day cycles
        
        # Add random variation
        random_factor = np.random.lognormal(0, 0.3)
        
        volume = base_volume * trend_factor * cycle_factor * random_factor
        volumes.append(int(volume))
    
    test_data = pd.DataFrame({
        'open': prices + np.random.normal(0, 2, len(dates)),
        'high': prices + np.abs(np.random.normal(8, 4, len(dates))),
        'low': prices - np.abs(np.random.normal(8, 4, len(dates))),
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    # Ensure realistic OHLC relationships
    test_data['high'] = np.maximum(test_data[['open', 'close']].max(axis=1), test_data['high'])
    test_data['low'] = np.minimum(test_data[['open', 'close']].min(axis=1), test_data['low'])
    
    print(f"✅ Created test data: {len(test_data)} days")
    print(f"   Price range: ${test_data['close'].min():.2f} - ${test_data['close'].max():.2f}")
    print(f"   Volume range: {test_data['volume'].min():,} - {test_data['volume'].max():,}")
    
    # Process data
    processor = VolumeTrendMomentumProcessor()
    results = processor.process_volume_trend_momentum_data(test_data)
    
    if 'error' in results:
        print(f"❌ Analysis failed: {results['error']}")
        return False
    
    print("✅ Analysis completed successfully")
    
    # Display results
    volume_trends = results.get('volume_trend_analysis', {})
    momentum_analysis = results.get('momentum_analysis', {})
    cycle_analysis = results.get('cycle_analysis', {})
    future_implications = results.get('future_implications', {})
    sustainability = results.get('sustainability_assessment', {})
    quality = results.get('quality_assessment', {})
    
    print(f"\n📊 Analysis Results:")
    print(f"   Volume trend direction: {results.get('volume_trend_direction', 'unknown')}")
    print(f"   Trend strength: {results.get('trend_strength', 'unknown')}")
    print(f"   Momentum phase: {results.get('momentum_phase', 'unknown')}")
    print(f"   Quality score: {quality.get('overall_score', 0)}/100")
    
    # Volume trends
    if 'error' not in volume_trends:
        print(f"\n📈 Volume Trends:")
        for timeframe in ['short_term', 'medium_term', 'long_term']:
            trend_data = volume_trends.get(timeframe, {})
            if trend_data:
                direction = trend_data.get('direction', 'unknown')
                strength = trend_data.get('strength', 'unknown')
                change_pct = trend_data.get('volume_change_pct', 0)
                print(f"   {timeframe.title()}: {direction} ({strength}) - {change_pct:+.1f}%")
    
    # Momentum analysis
    if 'error' not in momentum_analysis:
        print(f"\n⚡ Momentum Analysis:")
        print(f"   Overall direction: {momentum_analysis.get('overall_momentum_direction', 'unknown')}")
        print(f"   Strength: {momentum_analysis.get('momentum_strength', 'unknown')}")
        
        roc_indicators = momentum_analysis.get('rate_of_change_indicators', {})
        for period, indicator in roc_indicators.items():
            current_roc = indicator.get('current_value', 0)
            print(f"   {period}: {current_roc:+.1f}%")
    
    # Cycle analysis
    if 'error' not in cycle_analysis:
        print(f"\n🔄 Cycle Analysis:")
        print(f"   Current phase: {cycle_analysis.get('current_phase', 'unknown')}")
        print(f"   Cycle count: {cycle_analysis.get('cycle_count', 0)}")
        print(f"   Average cycle length: {cycle_analysis.get('average_cycle_length', 0):.1f} days")
        print(f"   Cycle regularity: {cycle_analysis.get('cycle_regularity', 'unknown')}")
    
    # Future implications
    if 'error' not in future_implications:
        print(f"\n🔮 Future Implications:")
        print(f"   Trend continuation probability: {future_implications.get('trend_continuation_probability', 0):.1%}")
        print(f"   Momentum exhaustion warning: {future_implications.get('momentum_exhaustion_warning', False)}")
        print(f"   Acceleration signal: {future_implications.get('volume_acceleration_signal', 'unknown')}")
        print(f"   Confidence level: {future_implications.get('confidence_level', 'unknown')}")
    
    # Sustainability
    if 'error' not in sustainability:
        print(f"\n⏳ Sustainability Assessment:")
        print(f"   Overall sustainability: {sustainability.get('overall_sustainability', 'unknown')}")
        print(f"   Sustainability score: {sustainability.get('sustainability_score', 0)}/100")
        print(f"   Timeframe: {sustainability.get('sustainability_timeframe', 'unknown')}")
        
        supporting_factors = sustainability.get('supporting_factors', [])
        risk_factors = sustainability.get('risk_factors', [])
        
        if supporting_factors:
            print(f"   Supporting factors: {', '.join(supporting_factors[:3])}")
        if risk_factors:
            print(f"   Risk factors: {', '.join(risk_factors[:3])}")
    
    return True

if __name__ == "__main__":
    test_volume_trend_momentum_processor()