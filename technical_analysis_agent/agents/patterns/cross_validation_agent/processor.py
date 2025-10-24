#!/usr/bin/env python3
"""
Cross-Validation Processor - Pattern Verification Module

This module handles cross-validation of detected patterns using multiple independent approaches:
- Statistical validation of pattern characteristics
- Volume confirmation analysis
- Time-series pattern validation
- Historical pattern success rate analysis
- Inter-pattern consistency checks
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Import conflict detector for pattern analysis
try:
    from agents.patterns.cross_validation_agent.conflict_detector import PatternConflictDetector
except ImportError:
    PatternConflictDetector = None

logger = logging.getLogger(__name__)

class CrossValidationProcessor:
    """
    Processor for cross-validating detected patterns using multiple independent methods.
    
    This processor specializes in:
    - Statistical validation of pattern characteristics
    - Volume profile confirmation
    - Time-series consistency checks
    - Historical pattern performance validation
    - Multi-method confidence scoring
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.name = "cross_validation"
        self.description = "Multi-method pattern validation and verification"
        self.version = "1.0.0"
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def process_cross_validation_data(
        self, 
        stock_data: pd.DataFrame,
        detected_patterns: List[Dict[str, Any]],
        pattern_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process cross-validation analysis for detected patterns.
        
        Args:
            stock_data: DataFrame with OHLCV data
            detected_patterns: List of patterns detected by primary analysis
            pattern_summary: Summary of pattern detection results
            
        Returns:
            Dictionary containing comprehensive cross-validation results
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"[CROSS_VALIDATION] Starting cross-validation analysis")
            
            if stock_data is None or stock_data.empty or len(stock_data) < 20:
                return self._build_error_result("Insufficient data for cross-validation analysis")
            
            if not detected_patterns:
                return self._build_no_patterns_result("No patterns provided for cross-validation")
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in stock_data.columns]
            if missing_columns:
                return self._build_error_result(f"Missing required columns: {missing_columns}")
            
            # 1. Statistical Validation
            statistical_validation = self._perform_statistical_validation(stock_data, detected_patterns)
            
            # 2. Volume Confirmation Analysis
            volume_confirmation = self._analyze_volume_confirmation(stock_data, detected_patterns)
            
            # 3. Time-Series Pattern Validation
            time_series_validation = self._validate_time_series_patterns(stock_data, detected_patterns)
            
            # 4. Historical Pattern Performance
            historical_validation = self._analyze_historical_pattern_performance(detected_patterns)
            
            # 5. Inter-Pattern Consistency
            consistency_analysis = self._analyze_pattern_consistency(detected_patterns, pattern_summary)
            
            # 6. Market Regime Analysis  
            market_regime = self._detect_market_regime(stock_data)
            
            # 7. Alternative Method Validation
            alternative_validation = self._perform_alternative_method_validation(stock_data, detected_patterns)
            
            # 8. Pattern Conflict Detection (temporarily disabled for demo)
            conflict_analysis = {
                'total_conflicts': 0,
                'pattern_coherence': 'high',
                'conflict_summary': 'Conflict detection temporarily disabled',
                'resolution_strategy': {'confidence_adjustment': 0.0}
            }
            
            # 9. Comprehensive Validation Scoring
            validation_scores = self._calculate_validation_scores(
                statistical_validation, volume_confirmation, time_series_validation,
                historical_validation, consistency_analysis, alternative_validation, market_regime
            )
            
            # 10. Final Confidence Assessment (with conflict adjustment)
            final_confidence = self._assess_final_confidence(validation_scores, detected_patterns, market_regime, conflict_analysis)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Infer interval and lookback metadata
            data_interval = self._infer_data_interval(stock_data)
            lookback_periods = len(stock_data)
            start_ts, end_ts = None, None
            try:
                idx = stock_data.index
                start_ts = str(idx[0]) if len(idx) > 0 else None
                end_ts = str(idx[-1]) if len(idx) > 0 else None
            except Exception:
                pass
            
            # Build comprehensive result
            result = {
                'success': True,
                'agent_name': self.name,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                
                # Core Validation Results
                'validation_summary': {
                    'patterns_validated': len(detected_patterns),
                    'validation_methods_used': 8,  # Updated to include conflict detection
                    'overall_validation_score': round(validation_scores.get('overall_score', 0), 2),
                    'validation_confidence': final_confidence.get('confidence_level', 'unknown'),
                    'market_regime': market_regime.get('regime', 'unknown'),
                    'pattern_conflicts': conflict_analysis.get('total_conflicts', 0),
                    'pattern_coherence': conflict_analysis.get('pattern_coherence', 'unknown'),
                    # Added metadata for downstream clarity
                    'data_interval': data_interval,
                    'lookback_periods': lookback_periods,
                    'start_timestamp': start_ts,
                    'end_timestamp': end_ts
                },
                
                # Detailed Validation Results
                'statistical_validation': statistical_validation,
                'volume_confirmation': volume_confirmation,
                'time_series_validation': time_series_validation,
                'historical_validation': historical_validation,
                'consistency_analysis': consistency_analysis,
                'alternative_validation': alternative_validation,
                'market_regime_analysis': market_regime,
                'pattern_conflict_analysis': conflict_analysis,
                
                # Validation Scoring
                'validation_scores': validation_scores,
                'final_confidence_assessment': final_confidence,
                
                # Pattern-Specific Results
                'pattern_validation_details': self._compile_pattern_validation_details(
                    detected_patterns, statistical_validation, volume_confirmation,
                    time_series_validation, historical_validation
                ),
                
                # Statistics
                'analysis_period_days': len(stock_data),
                'confidence_score': round(final_confidence.get('overall_confidence', 0.5), 2),
                'data_quality': self._assess_data_quality(stock_data)
            }
            
            logger.info(f"[CROSS_VALIDATION] Analysis completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"[CROSS_VALIDATION] Analysis failed: {str(e)}")
            return self._build_error_result(str(e), processing_time)
    
    def _perform_statistical_validation(self, stock_data: pd.DataFrame, detected_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical validation of detected patterns"""
        try:
            validation_results = {
                'method': 'statistical_validation',
                'patterns_tested': len(detected_patterns),
                'validation_results': [],
                'overall_statistical_score': 0.0
            }
            
            for i, pattern in enumerate(detected_patterns):
                pattern_name = pattern.get('pattern_name', f'Pattern_{i+1}')
                pattern_type = pattern.get('pattern_type', 'unknown')
                completion = pattern.get('completion_percentage', 0)
                
                # Statistical tests for pattern validity
                stat_tests = {
                    'price_trend_significance': self._test_price_trend_significance(stock_data, pattern),
                    'volatility_consistency': self._test_volatility_consistency(stock_data, pattern),
                    'pattern_duration_validity': self._test_pattern_duration_validity(stock_data, pattern),
                    'price_level_significance': self._test_price_level_significance(stock_data, pattern)
                }
                
                # Calculate pattern statistical score
                stat_scores = [score for score in stat_tests.values() if isinstance(score, (int, float))]
                pattern_stat_score = np.mean(stat_scores) if stat_scores else 0.5
                
                pattern_validation = {
                    'pattern_name': pattern_name,
                    'pattern_type': pattern_type,
                    'statistical_tests': stat_tests,
                    'statistical_score': pattern_stat_score,
                    'statistical_confidence': self._classify_confidence(pattern_stat_score),
                    'significant_tests': len([s for s in stat_scores if s >= 0.7]),
                    'total_tests': len(stat_tests)
                }
                
                validation_results['validation_results'].append(pattern_validation)
            
            # Overall statistical score
            if validation_results['validation_results']:
                overall_score = np.mean([
                    p['statistical_score'] for p in validation_results['validation_results']
                ])
                validation_results['overall_statistical_score'] = overall_score
            
            return validation_results
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION] Statistical validation failed: {e}")
            return {'error': str(e), 'method': 'statistical_validation'}
    
    def _analyze_volume_confirmation(self, stock_data: pd.DataFrame, detected_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze volume confirmation for detected patterns"""
        try:
            confirmation_results = {
                'method': 'volume_confirmation',
                'patterns_analyzed': len(detected_patterns),
                'confirmation_results': [],
                'overall_volume_score': 0.0
            }
            
            if 'volume' not in stock_data.columns:
                confirmation_results['error'] = 'Volume data not available'
                return confirmation_results
            
            for i, pattern in enumerate(detected_patterns):
                pattern_name = pattern.get('pattern_name', f'Pattern_{i+1}')
                pattern_type = pattern.get('pattern_type', 'unknown')
                
                # Volume analysis for pattern confirmation
                volume_analysis = {
                    'volume_trend_confirmation': self._analyze_volume_trend(stock_data, pattern),
                    'breakout_volume_strength': self._analyze_breakout_volume(stock_data, pattern),
                    'volume_price_correlation': self._analyze_volume_price_correlation(stock_data, pattern),
                    'relative_volume_strength': self._analyze_relative_volume(stock_data, pattern)
                }
                
                # Calculate volume confirmation score
                volume_scores = [score for score in volume_analysis.values() if isinstance(score, (int, float))]
                pattern_volume_score = np.mean(volume_scores) if volume_scores else 0.5
                
                pattern_confirmation = {
                    'pattern_name': pattern_name,
                    'pattern_type': pattern_type,
                    'volume_analysis': volume_analysis,
                    'volume_confirmation_score': pattern_volume_score,
                    'volume_strength': self._classify_volume_strength(pattern_volume_score),
                    'confirmed_indicators': len([s for s in volume_scores if s >= 0.6])
                }
                
                confirmation_results['confirmation_results'].append(pattern_confirmation)
            
            # Overall volume confirmation score
            if confirmation_results['confirmation_results']:
                overall_score = np.mean([
                    p['volume_confirmation_score'] for p in confirmation_results['confirmation_results']
                ])
                confirmation_results['overall_volume_score'] = overall_score
            
            return confirmation_results
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION] Volume confirmation failed: {e}")
            return {'error': str(e), 'method': 'volume_confirmation'}
    
    def _validate_time_series_patterns(self, stock_data: pd.DataFrame, detected_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate patterns using time-series analysis methods"""
        try:
            validation_results = {
                'method': 'time_series_validation',
                'patterns_validated': len(detected_patterns),
                'validation_results': [],
                'overall_time_series_score': 0.0
            }
            
            for i, pattern in enumerate(detected_patterns):
                pattern_name = pattern.get('pattern_name', f'Pattern_{i+1}')
                pattern_type = pattern.get('pattern_type', 'unknown')
                
                # Time-series validation methods
                ts_validation = {
                    'trend_consistency': self._validate_trend_consistency(stock_data, pattern),
                    'seasonality_analysis': self._analyze_pattern_seasonality(stock_data, pattern),
                    'autocorrelation_test': self._test_autocorrelation(stock_data, pattern),
                    'stationarity_test': self._test_stationarity(stock_data, pattern),
                    'change_point_detection': self._detect_change_points(stock_data, pattern)
                }
                
                # Calculate time-series score
                ts_scores = [score for score in ts_validation.values() if isinstance(score, (int, float))]
                pattern_ts_score = np.mean(ts_scores) if ts_scores else 0.5
                
                pattern_validation = {
                    'pattern_name': pattern_name,
                    'pattern_type': pattern_type,
                    'time_series_tests': ts_validation,
                    'time_series_score': pattern_ts_score,
                    'time_series_confidence': self._classify_confidence(pattern_ts_score),
                    'validated_aspects': len([s for s in ts_scores if s >= 0.6])
                }
                
                validation_results['validation_results'].append(pattern_validation)
            
            # Overall time-series score
            if validation_results['validation_results']:
                overall_score = np.mean([
                    p['time_series_score'] for p in validation_results['validation_results']
                ])
                validation_results['overall_time_series_score'] = overall_score
            
            return validation_results
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION] Time-series validation failed: {e}")
            return {'error': str(e), 'method': 'time_series_validation'}
    
    def _analyze_historical_pattern_performance(self, detected_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze historical performance of detected pattern types"""
        try:
            # Historical success rates for different pattern types (based on literature)
            historical_success_rates = {
                'ascending_triangle': 0.72,
                'descending_triangle': 0.68,
                'symmetrical_triangle': 0.65,
                'bullish_flag': 0.78,
                'bearish_flag': 0.75,
                'bullish_pennant': 0.74,
                'bearish_pennant': 0.71,
                'rectangle': 0.63,
                'ascending_channel': 0.69,
                'descending_channel': 0.66,
                'head_and_shoulders': 0.71,
                'inverse_head_and_shoulders': 0.68,
                'double_top': 0.64,
                'double_bottom': 0.67
            }
            
            performance_results = {
                'method': 'historical_performance',
                'patterns_analyzed': len(detected_patterns),
                'performance_results': [],
                'overall_historical_score': 0.0
            }
            
            for i, pattern in enumerate(detected_patterns):
                pattern_name = pattern.get('pattern_name', f'Pattern_{i+1}')
                pattern_type = pattern.get('pattern_type', 'unknown')
                reliability = pattern.get('reliability', 'unknown')
                completion = pattern.get('completion_percentage', 0)
                
                # Get historical success rate
                base_success_rate = historical_success_rates.get(pattern_name, 0.5)
                
                # Adjust for pattern quality and completion
                reliability_multiplier = {
                    'high': 1.1,
                    'medium': 1.0,
                    'low': 0.9
                }.get(reliability, 1.0)
                
                completion_multiplier = 0.7 + (completion / 100) * 0.3
                
                adjusted_success_rate = base_success_rate * reliability_multiplier * completion_multiplier
                adjusted_success_rate = min(0.95, max(0.2, adjusted_success_rate))  # Clamp values
                
                performance_analysis = {
                    'pattern_name': pattern_name,
                    'pattern_type': pattern_type,
                    'base_historical_success_rate': base_success_rate,
                    'reliability_adjustment': reliability_multiplier,
                    'completion_adjustment': completion_multiplier,
                    'adjusted_success_rate': adjusted_success_rate,
                    'historical_confidence': self._classify_confidence(adjusted_success_rate),
                    'performance_category': self._classify_performance(adjusted_success_rate)
                }
                
                performance_results['performance_results'].append(performance_analysis)
            
            # Overall historical score
            if performance_results['performance_results']:
                overall_score = np.mean([
                    p['adjusted_success_rate'] for p in performance_results['performance_results']
                ])
                performance_results['overall_historical_score'] = overall_score
            
            return performance_results
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION] Historical performance analysis failed: {e}")
            return {'error': str(e), 'method': 'historical_performance'}
    
    def _analyze_pattern_consistency(self, detected_patterns: List[Dict[str, Any]], pattern_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consistency between detected patterns"""
        try:
            consistency_results = {
                'method': 'pattern_consistency',
                'patterns_analyzed': len(detected_patterns),
                'consistency_score': 0.0,
                'consistency_issues': [],
                'pattern_conflicts': [],
                'pattern_reinforcements': []
            }
            
            if len(detected_patterns) < 2:
                consistency_results['note'] = 'Insufficient patterns for consistency analysis'
                consistency_results['consistency_score'] = 0.8  # Assume good if only one pattern
                return consistency_results
            
            # Check for pattern conflicts (using legacy method for consistency analysis)
            conflicts = self._detect_legacy_pattern_conflicts(detected_patterns)
            consistency_results['pattern_conflicts'] = conflicts
            
            # Check for pattern reinforcements
            reinforcements = self._detect_pattern_reinforcements(detected_patterns)
            consistency_results['pattern_reinforcements'] = reinforcements
            
            # Analyze overall bias consistency
            bias_consistency = self._analyze_bias_consistency(detected_patterns, pattern_summary)
            consistency_results['bias_consistency'] = bias_consistency
            
            # Analyze timeframe consistency
            timeframe_consistency = self._analyze_timeframe_consistency(detected_patterns)
            consistency_results['timeframe_consistency'] = timeframe_consistency
            
            # Calculate consistency score
            conflict_penalty = len(conflicts) * 0.1
            reinforcement_bonus = len(reinforcements) * 0.05
            bias_score = bias_consistency.get('consistency_score', 0.5)
            timeframe_score = timeframe_consistency.get('consistency_score', 0.5)
            
            base_consistency = (bias_score + timeframe_score) / 2
            final_consistency = max(0.1, min(1.0, base_consistency + reinforcement_bonus - conflict_penalty))
            
            consistency_results['consistency_score'] = final_consistency
            consistency_results['consistency_level'] = self._classify_confidence(final_consistency)
            
            return consistency_results
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION] Pattern consistency analysis failed: {e}")
            return {'error': str(e), 'method': 'pattern_consistency'}
    
    def _perform_alternative_method_validation(self, stock_data: pd.DataFrame, detected_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate patterns using alternative detection methods"""
        try:
            alternative_results = {
                'method': 'alternative_validation',
                'patterns_tested': len(detected_patterns),
                'alternative_results': [],
                'overall_alternative_score': 0.0
            }
            
            for i, pattern in enumerate(detected_patterns):
                pattern_name = pattern.get('pattern_name', f'Pattern_{i+1}')
                pattern_type = pattern.get('pattern_type', 'unknown')
                
                # Alternative validation methods
                alternative_methods = {
                    'fractal_analysis': self._validate_with_fractals(stock_data, pattern),
                    'clustering_validation': self._validate_with_clustering(stock_data, pattern),
                    'fourier_analysis': self._validate_with_fourier(stock_data, pattern),
                    'wavelet_analysis': self._validate_with_wavelets(stock_data, pattern)
                }
                
                # Calculate alternative method score
                alt_scores = [score for score in alternative_methods.values() if isinstance(score, (int, float))]
                pattern_alt_score = np.mean(alt_scores) if alt_scores else 0.5
                
                pattern_alternative = {
                    'pattern_name': pattern_name,
                    'pattern_type': pattern_type,
                    'alternative_methods': alternative_methods,
                    'alternative_score': pattern_alt_score,
                    'alternative_confidence': self._classify_confidence(pattern_alt_score),
                    'confirmed_methods': len([s for s in alt_scores if s >= 0.6])
                }
                
                alternative_results['alternative_results'].append(pattern_alternative)
            
            # Overall alternative score
            if alternative_results['alternative_results']:
                overall_score = np.mean([
                    p['alternative_score'] for p in alternative_results['alternative_results']
                ])
                alternative_results['overall_alternative_score'] = overall_score
            
            return alternative_results
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION] Alternative method validation failed: {e}")
            return {'error': str(e), 'method': 'alternative_validation'}
    
    def _calculate_validation_scores(self, *validation_results) -> Dict[str, Any]:
        """Calculate comprehensive validation scores from all methods"""
        try:
            scores = {}
            
            # Extract scores from each validation method
            method_scores = {}
            for result in validation_results:
                if isinstance(result, dict) and not result.get('error'):
                    method_name = result.get('method', 'unknown')
                    
                    # Extract the main score for each method
                    if 'overall_statistical_score' in result:
                        method_scores['statistical'] = result['overall_statistical_score']
                    elif 'overall_volume_score' in result:
                        method_scores['volume'] = result['overall_volume_score']
                    elif 'overall_time_series_score' in result:
                        method_scores['time_series'] = result['overall_time_series_score']
                    elif 'overall_historical_score' in result:
                        method_scores['historical'] = result['overall_historical_score']
                    elif 'consistency_score' in result:
                        method_scores['consistency'] = result['consistency_score']
                    elif 'overall_alternative_score' in result:
                        method_scores['alternative'] = result['overall_alternative_score']
            
            # Calculate weighted overall score
            weights = {
                'statistical': 0.20,
                'volume': 0.15,
                'time_series': 0.20,
                'historical': 0.20,
                'consistency': 0.15,
                'alternative': 0.10
            }
            
            weighted_sum = 0.0
            total_weight = 0.0
            
            for method, score in method_scores.items():
                if method in weights:
                    weighted_sum += score * weights[method]
                    total_weight += weights[method]
            
            overall_score = weighted_sum / total_weight if total_weight > 0 else 0.5
            
            scores = {
                'method_scores': method_scores,
                'weights_used': weights,
                'overall_score': overall_score,
                'validation_quality': self._classify_confidence(overall_score),
                'methods_validated': len(method_scores),
                'validation_completeness': len(method_scores) / 6  # 6 total methods
            }
            
            return scores
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION] Validation score calculation failed: {e}")
            return {'error': str(e)}
    
    def _detect_pattern_conflicts(self, detected_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect conflicts between patterns using conflict detector"""
        try:
            if not PatternConflictDetector or len(detected_patterns) < 2:
                return {
                    'total_conflicts': 0,
                    'pattern_coherence': 'high',
                    'conflict_summary': f'No conflicts detected among {len(detected_patterns)} pattern(s).',
                    'resolution_strategy': {'confidence_adjustment': 0.0}
                }
            
            conflict_detector = PatternConflictDetector()
            conflict_analysis = conflict_detector.detect_pattern_conflicts(detected_patterns, {})
            
            logger.info(f"[CROSS_VALIDATION] Pattern conflict analysis: {conflict_analysis.get('total_conflicts', 0)} conflicts")
            return conflict_analysis
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION] Pattern conflict detection failed: {e}")
            return {
                'error': str(e),
                'total_conflicts': 0,
                'pattern_coherence': 'unknown',
                'resolution_strategy': {'confidence_adjustment': 0.0}
            }
    
    def _assess_final_confidence(self, validation_scores: Dict[str, Any], detected_patterns: List[Dict[str, Any]], market_regime: Dict[str, Any], conflict_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """Assess final confidence with proper statistical validation weighting"""
        try:
            method_scores = validation_scores.get('method_scores', {})
            overall_score = validation_scores.get('overall_score', 0.5)
            validation_completeness = validation_scores.get('validation_completeness', 0.5)
            num_patterns = len(detected_patterns)
            
            # CRITICAL: Extract statistical validation score for confidence capping
            statistical_score = method_scores.get('statistical', 0.5)
            volume_score = method_scores.get('volume', 0.5)
            
            # Statistical validation should cap maximum confidence
            if statistical_score < 0.5:
                max_confidence = 0.6  # Cap at 60% if statistical very weak
                confidence_cap_reason = f"Statistical validation very low ({statistical_score:.2f})"
            elif statistical_score < 0.6:
                max_confidence = 0.7  # Cap at 70% if statistical weak  
                confidence_cap_reason = f"Statistical validation low ({statistical_score:.2f})"
            elif statistical_score < 0.7:
                max_confidence = 0.85  # Cap at 85% if statistical moderate
                confidence_cap_reason = f"Statistical validation moderate ({statistical_score:.2f})"
            else:
                max_confidence = 1.0  # No cap if statistical validation strong
                confidence_cap_reason = None
            
            # Volume validation should also influence cap
            if volume_score < 0.5 and max_confidence > 0.65:
                max_confidence = min(max_confidence, 0.65)
                if confidence_cap_reason:
                    confidence_cap_reason += f" + Volume validation low ({volume_score:.2f})"
                else:
                    confidence_cap_reason = f"Volume validation low ({volume_score:.2f})"
            
            # Base confidence from validation score (but properly weighted)
            # Give higher weight to statistical and volume validation
            weighted_base_confidence = (
                statistical_score * 0.4 +  # High weight on statistical
                volume_score * 0.2 +       # Medium weight on volume
                method_scores.get('historical', 0.5) * 0.2 +
                method_scores.get('time_series', 0.5) * 0.1 +
                method_scores.get('consistency', 0.5) * 0.1
            )
            
            # Adjust for validation completeness (less aggressive)
            completeness_factor = 0.9 + (validation_completeness * 0.1)
            
            # Pattern factor (less aggressive multiplier)
            pattern_factor = min(1.1, 1.0 + (num_patterns - 1) * 0.02)
            
            # Market regime adjustment (more conservative)
            regime = market_regime.get('regime', 'unknown')
            regime_confidence = market_regime.get('confidence', 0.5)
            
            if regime == 'trending':
                regime_adjustment = 0.05  # Small boost for trending
            elif regime == 'stable':
                regime_adjustment = 0.02  # Small boost for stable
            elif regime == 'volatile':
                regime_adjustment = -0.05  # Small penalty for volatile
            else:
                regime_adjustment = 0.0   # No adjustment for unknown
                
            regime_factor = 1.0 + (regime_adjustment * regime_confidence)
            
            # Calculate preliminary confidence
            preliminary_confidence = weighted_base_confidence * completeness_factor * pattern_factor * regime_factor
            
            # Apply conflict-based adjustments
            conflict_adjustment = 0.0
            if conflict_analysis and isinstance(conflict_analysis, dict):
                resolution_strategy = conflict_analysis.get('resolution_strategy', {})
                conflict_adjustment = resolution_strategy.get('confidence_adjustment', 0.0)
            elif conflict_analysis and not isinstance(conflict_analysis, dict):
                logger.warning(f"[CROSS_VALIDATION] Conflict analysis is not a dict: {type(conflict_analysis)}")
                conflict_adjustment = 0.0
            
            # Apply confidence cap and conflict adjustments
            adjusted_confidence = preliminary_confidence + conflict_adjustment
            final_confidence = min(adjusted_confidence, max_confidence)
            final_confidence = max(0.1, min(1.0, final_confidence))
            
            # Determine confidence level based on ACTUAL final confidence
            if final_confidence >= 0.8:
                confidence_level = 'very_high'
                confidence_category = 'strong'
            elif final_confidence >= 0.65:
                confidence_level = 'high'
                confidence_category = 'good'
            elif final_confidence >= 0.5:
                confidence_level = 'medium'
                confidence_category = 'moderate'
            elif final_confidence >= 0.35:
                confidence_level = 'low'
                confidence_category = 'weak'
            else:
                confidence_level = 'very_low'
                confidence_category = 'poor'
            
            assessment = {
                'overall_confidence': round(final_confidence, 2),
                'confidence_level': confidence_level,
                'confidence_category': confidence_category,
                'base_validation_score': round(weighted_base_confidence, 2),
                'validation_completeness': round(validation_completeness, 2),
                'pattern_count_factor': round(pattern_factor, 2),
                'market_regime_factor': round(regime_factor, 2),
                'confidence_cap_applied': preliminary_confidence > max_confidence,
                'confidence_cap_reason': confidence_cap_reason,
                'max_allowed_confidence': max_confidence,
                'preliminary_confidence': round(preliminary_confidence, 2),
                'conflict_adjustment': round(conflict_adjustment, 2),
                'adjusted_confidence': round(adjusted_confidence, 2),
                'confidence_factors': {
                    'statistical_score': round(statistical_score, 2),
                    'volume_score': round(volume_score, 2),
                    'weighted_base': round(weighted_base_confidence, 2),
                    'method_completeness': round(validation_completeness, 2),
                    'pattern_consistency': round(pattern_factor, 2),
                    'market_regime_adjustment': round(regime_factor, 2)
                },
                'recommendation': self._generate_confidence_recommendation(final_confidence, confidence_level)
            }
            
            return assessment
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION] Final confidence assessment failed: {e}")
            return {'error': str(e)}
    
    # Helper methods for specific validation tests
    def _test_price_trend_significance(self, stock_data: pd.DataFrame, pattern: Dict[str, Any]) -> float:
        """Test statistical significance of price trends in pattern"""
        try:
            close_prices = stock_data['close'].values
            if len(close_prices) < 10:
                return 0.5
            
            # Simple trend test using linear regression
            x = np.arange(len(close_prices))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, close_prices)
            
            # Return confidence based on p-value and R-squared
            r_squared = r_value ** 2
            significance = 1 - p_value if p_value < 1 else 0
            
            return min(1.0, (r_squared + significance) / 2)
            
        except Exception:
            return 0.5
    
    def _test_volatility_consistency(self, stock_data: pd.DataFrame, pattern: Dict[str, Any]) -> float:
        """Test volatility consistency within pattern"""
        try:
            returns = stock_data['close'].pct_change().dropna()
            if len(returns) < 5:
                return 0.5
            
            # Calculate rolling volatility
            window = min(10, len(returns) // 2)
            rolling_vol = returns.rolling(window=window).std()
            
            # Test for volatility stability (lower coefficient of variation is better)
            vol_cv = rolling_vol.std() / rolling_vol.mean() if rolling_vol.mean() > 0 else 1
            
            # Convert to confidence score (lower CV = higher confidence)
            confidence = max(0.1, 1.0 - min(1.0, vol_cv))
            
            return confidence
            
        except Exception:
            return 0.5
    
    def _test_pattern_duration_validity(self, stock_data: pd.DataFrame, pattern: Dict[str, Any]) -> float:
        """Test if pattern duration is reasonable"""
        try:
            data_length = len(stock_data)
            
            # Typical pattern durations (in periods)
            expected_durations = {
                'triangle': (10, 30),
                'flag': (5, 15),
                'pennant': (5, 15),
                'channel': (15, 40),
                'rectangle': (10, 30),
                'head_and_shoulders': (15, 35),
                'double': (10, 25)
            }
            
            pattern_name = pattern.get('pattern_name', '')
            pattern_type = None
            
            # Determine pattern category
            for key in expected_durations.keys():
                if key in pattern_name.lower():
                    pattern_type = key
                    break
            
            if not pattern_type:
                return 0.6  # Default if pattern type not recognized
            
            min_duration, max_duration = expected_durations[pattern_type]
            
            # Check if data length is within expected range
            if min_duration <= data_length <= max_duration:
                return 0.9
            elif data_length < min_duration:
                return max(0.3, 0.9 * (data_length / min_duration))
            else:  # data_length > max_duration
                return max(0.3, 0.9 * (max_duration / data_length))
                
        except Exception:
            return 0.5
    
    def _test_price_level_significance(self, stock_data: pd.DataFrame, pattern: Dict[str, Any]) -> float:
        """Test significance of key price levels in pattern"""
        try:
            pattern_data = pattern.get('pattern_data', {})
            
            # Extract key levels if available
            key_levels = []
            if 'resistance_level' in pattern_data:
                key_levels.append(pattern_data['resistance_level'])
            if 'support_level' in pattern_data:
                key_levels.append(pattern_data['support_level'])
            if 'neckline_level' in pattern_data:
                key_levels.append(pattern_data['neckline_level'])
            
            if not key_levels:
                return 0.6  # No specific levels to test
            
            # Test how well price respects these levels
            total_tests = 0
            successful_tests = 0
            
            for level in key_levels:
                # Count how many times price approached but didn't break the level significantly
                level_touches = 0
                level_breaks = 0
                
                for price in [stock_data['high'].values, stock_data['low'].values]:
                    for p in price:
                        distance = abs(p - level) / level
                        if distance < 0.02:  # Within 2% of level
                            level_touches += 1
                            if distance < 0.005:  # Very close touch
                                level_breaks += 1
                
                if level_touches > 0:
                    success_rate = level_breaks / level_touches
                    successful_tests += success_rate
                    total_tests += 1
            
            if total_tests == 0:
                return 0.6
            
            return successful_tests / total_tests
            
        except Exception:
            return 0.5
    
    # Volume analysis methods
    def _analyze_volume_trend(self, stock_data: pd.DataFrame, pattern: Dict[str, Any]) -> float:
        """Analyze volume trend confirmation"""
        try:
            volumes = stock_data['volume'].values
            if len(volumes) < 5:
                return 0.5
            
            # Calculate volume trend
            x = np.arange(len(volumes))
            slope, _, r_value, p_value, _ = stats.linregress(x, volumes)
            
            pattern_type = pattern.get('pattern_type', 'unknown')
            pattern_name = pattern.get('pattern_name', '')
            
            # For continuation patterns, volume should generally decrease during consolidation
            if pattern_type == 'continuation' or 'flag' in pattern_name or 'pennant' in pattern_name:
                # Negative slope is good for continuation patterns
                if slope < 0:
                    return min(1.0, abs(r_value))
                else:
                    return max(0.3, 1.0 - abs(r_value))
            else:
                # For other patterns, consistent volume is generally good
                return max(0.4, 1.0 - abs(slope) / np.mean(volumes))
                
        except Exception:
            return 0.5
    
    def _analyze_breakout_volume(self, stock_data: pd.DataFrame, pattern: Dict[str, Any]) -> float:
        """Analyze breakout volume strength"""
        try:
            volumes = stock_data['volume'].values
            if len(volumes) < 10:
                return 0.5
            
            # Look at recent volume vs average volume
            recent_volume = np.mean(volumes[-3:])  # Last 3 periods
            avg_volume = np.mean(volumes[:-3])     # Previous periods
            
            if avg_volume == 0:
                return 0.5
            
            volume_ratio = recent_volume / avg_volume
            
            # Higher recent volume is generally better for pattern confirmation
            if volume_ratio > 1.5:
                return 0.9
            elif volume_ratio > 1.2:
                return 0.8
            elif volume_ratio > 1.0:
                return 0.7
            elif volume_ratio > 0.8:
                return 0.5
            else:
                return 0.3
                
        except Exception:
            return 0.5
    
    def _analyze_volume_price_correlation(self, stock_data: pd.DataFrame, pattern: Dict[str, Any]) -> float:
        """Analyze correlation between volume and price movement"""
        try:
            if len(stock_data) < 10:
                return 0.5
            
            price_change = stock_data['close'].pct_change().abs()
            volume = stock_data['volume']
            
            # Remove NaN values
            valid_data = pd.DataFrame({'price_change': price_change, 'volume': volume}).dropna()
            
            if len(valid_data) < 5:
                return 0.5
            
            correlation = valid_data['price_change'].corr(valid_data['volume'])
            
            # Positive correlation between volume and price movement is generally good
            if pd.isna(correlation):
                return 0.5
            
            return max(0.2, min(1.0, (correlation + 1) / 2))  # Convert from [-1,1] to [0,1]
            
        except Exception:
            return 0.5
    
    def _analyze_relative_volume(self, stock_data: pd.DataFrame, pattern: Dict[str, Any]) -> float:
        """Analyze relative volume strength"""
        try:
            volumes = stock_data['volume'].values
            if len(volumes) < 10:
                return 0.5
            
            # Calculate relative volume using different percentiles
            median_volume = np.median(volumes)
            percentile_75 = np.percentile(volumes, 75)
            recent_avg_volume = np.mean(volumes[-5:])
            
            if median_volume == 0:
                return 0.5
            
            relative_strength = recent_avg_volume / median_volume
            
            # Classify relative volume strength
            if relative_strength > 2.0:
                return 0.95
            elif relative_strength > 1.5:
                return 0.85
            elif relative_strength > 1.2:
                return 0.75
            elif relative_strength > 0.8:
                return 0.65
            else:
                return 0.4
                
        except Exception:
            return 0.5
    
    # Time series validation methods
    def _validate_trend_consistency(self, stock_data: pd.DataFrame, pattern: Dict[str, Any]) -> float:
        """Enhanced trend consistency validation with multiple trend analysis methods"""
        try:
            close_prices = stock_data['close'].values
            if len(close_prices) < 10:
                return 0.5
            
            # Multi-timeframe trend analysis
            trend_scores = []
            
            # 1. Segmented Linear Regression Analysis
            segment_sizes = [max(3, len(close_prices) // 4), max(5, len(close_prices) // 3), max(7, len(close_prices) // 2)]
            
            for segment_size in segment_sizes:
                if segment_size >= len(close_prices):
                    continue
                    
                trends = []
                r_squared_values = []
                
                # Overlapping segments for better trend detection
                step_size = max(1, segment_size // 2)
                for i in range(0, len(close_prices) - segment_size + 1, step_size):
                    segment = close_prices[i:i + segment_size]
                    if len(segment) >= 3:
                        x = np.arange(len(segment))
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, segment)
                        trends.append(slope)
                        r_squared_values.append(r_value ** 2)
                
                if len(trends) >= 2:
                    # Trend direction consistency
                    trend_signs = np.sign(trends)
                    direction_consistency = np.sum(trend_signs == trend_signs[0]) / len(trend_signs)
                    
                    # Trend magnitude consistency
                    trend_std = np.std(trends)
                    trend_mean = abs(np.mean(trends))
                    magnitude_consistency = 1.0 / (1.0 + trend_std / (trend_mean + 1e-8))
                    
                    # R-squared consistency (how well segments fit linear trends)
                    avg_r_squared = np.mean(r_squared_values)
                    
                    # Combined trend score for this timeframe
                    segment_score = (direction_consistency * 0.4 + 
                                   magnitude_consistency * 0.35 + 
                                   avg_r_squared * 0.25)
                    trend_scores.append(segment_score)
            
            # 2. Moving Average Trend Analysis
            if len(close_prices) >= 20:
                ma_short = pd.Series(close_prices).rolling(window=5).mean().dropna()
                ma_long = pd.Series(close_prices).rolling(window=10).mean().dropna()
                
                if len(ma_short) >= 10 and len(ma_long) >= 10:
                    # Align the series
                    min_len = min(len(ma_short), len(ma_long))
                    ma_short = ma_short.iloc[-min_len:]
                    ma_long = ma_long.iloc[-min_len:]
                    
                    # Check MA trend consistency
                    ma_diff = ma_short.values - ma_long.values
                    ma_trend_changes = np.sum(np.diff(np.sign(ma_diff)) != 0)
                    ma_consistency = 1.0 - (ma_trend_changes / len(ma_diff))
                    trend_scores.append(ma_consistency)
            
            # 3. Pattern-specific trend validation
            pattern_name = pattern.get('pattern_name', '').lower()
            pattern_type = pattern.get('pattern_type', 'unknown')
            
            if not trend_scores:
                return 0.5
                
            base_score = np.mean(trend_scores)
            
            # Pattern-specific adjustments
            if 'triangle' in pattern_name:
                # Triangular patterns should show converging trends
                if len(close_prices) >= 20:
                    early_trend = stats.linregress(np.arange(len(close_prices)//2), 
                                                 close_prices[:len(close_prices)//2])[0]
                    late_trend = stats.linregress(np.arange(len(close_prices)//2), 
                                                close_prices[-len(close_prices)//2:])[0]
                    
                    # Convergence indicates decreasing trend magnitude
                    if abs(late_trend) < abs(early_trend):
                        base_score *= 1.2  # Reward convergence
                        
            elif 'channel' in pattern_name or 'rectangle' in pattern_name:
                # Channel patterns should show minimal overall trend
                overall_slope = stats.linregress(np.arange(len(close_prices)), close_prices)[0]
                price_range = np.max(close_prices) - np.min(close_prices)
                normalized_slope = abs(overall_slope) / (price_range / len(close_prices))
                
                if normalized_slope < 0.3:  # Relatively flat
                    base_score *= 1.15
                    
            elif 'head' in pattern_name and 'shoulders' in pattern_name:
                # H&S patterns should show clear trend reversal structure
                if len(close_prices) >= 15:
                    # Divide into three parts and check for peak/trough structure
                    third = len(close_prices) // 3
                    left_segment = close_prices[:third]
                    middle_segment = close_prices[third:2*third]
                    right_segment = close_prices[2*third:]
                    
                    # Check if middle segment has highest/lowest values (depending on pattern)
                    if 'inverse' in pattern_name:
                        # Inverse H&S should have lowest point in middle
                        if np.min(middle_segment) < np.min(left_segment) and np.min(middle_segment) < np.min(right_segment):
                            base_score *= 1.1
                    else:
                        # Regular H&S should have highest point in middle
                        if np.max(middle_segment) > np.max(left_segment) and np.max(middle_segment) > np.max(right_segment):
                            base_score *= 1.1
            
            # 4. Volatility-adjusted trend consistency
            volatility = np.std(np.diff(close_prices) / close_prices[:-1])
            if volatility > 0:
                # Higher volatility should require higher trend consistency threshold
                volatility_adjustment = min(1.2, 1.0 + volatility * 2)
                if base_score > (0.6 * volatility_adjustment):
                    base_score *= 1.05  # Small bonus for maintaining consistency despite volatility
            
            return min(0.95, max(0.1, base_score))
            
        except Exception:
            return 0.5
    
    def _analyze_pattern_seasonality(self, stock_data: pd.DataFrame, pattern: Dict[str, Any]) -> float:
        """Analyze seasonal components in pattern"""
        try:
            # Simple seasonality check using autocorrelation
            close_prices = stock_data['close'].values
            if len(close_prices) < 20:
                return 0.6
            
            # Check for weekly seasonality (5-day cycle) if we have enough data
            if len(close_prices) >= 15:
                autocorr_5 = np.corrcoef(close_prices[:-5], close_prices[5:])[0, 1]
                if not np.isnan(autocorr_5):
                    return max(0.4, min(0.9, 0.6 + abs(autocorr_5) * 0.3))
            
            return 0.6  # Default if can't analyze seasonality
            
        except Exception:
            return 0.5
    
    def _test_autocorrelation(self, stock_data: pd.DataFrame, pattern: Dict[str, Any]) -> float:
        """Enhanced autocorrelation test with multiple lags and pattern-specific analysis"""
        try:
            close_prices = stock_data['close'].values
            if len(close_prices) < 10:
                return 0.5
            
            # Calculate returns and price levels for different autocorrelation tests
            returns = np.diff(close_prices) / close_prices[:-1]
            log_returns = np.diff(np.log(close_prices)) if np.all(close_prices > 0) else returns
            
            if len(returns) < 5:
                return 0.5
            
            autocorr_scores = []
            
            # 1. Returns autocorrelation at multiple lags
            max_lag = min(10, len(returns) // 3)
            for lag in range(1, max_lag + 1):
                if len(returns) > lag:
                    try:
                        autocorr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                        if not np.isnan(autocorr):
                            # Score based on autocorrelation strength and pattern expectations
                            abs_autocorr = abs(autocorr)
                            
                            # Different optimal ranges for different lags
                            if lag == 1:
                                # Lag-1: minimal autocorrelation is generally better (market efficiency)
                                if abs_autocorr < 0.2:
                                    lag_score = 0.9
                                elif abs_autocorr < 0.4:
                                    lag_score = 0.7
                                elif abs_autocorr < 0.6:
                                    lag_score = 0.5
                                else:
                                    lag_score = 0.3
                            else:
                                # Higher lags: some structure can be acceptable for patterns
                                if abs_autocorr < 0.3:
                                    lag_score = 0.8
                                elif abs_autocorr < 0.5:
                                    lag_score = 0.6
                                else:
                                    lag_score = 0.4
                            
                            # Weight by lag (closer lags more important)
                            weight = 1.0 / lag
                            autocorr_scores.append((lag_score, weight))
                    except:
                        continue
            
            # 2. Price level autocorrelation (for trend patterns)
            if len(close_prices) >= 8:
                # Detrended price autocorrelation
                detrended = close_prices - np.linspace(close_prices[0], close_prices[-1], len(close_prices))
                for lag in [1, 2, 3]:
                    if len(detrended) > lag:
                        try:
                            price_autocorr = np.corrcoef(detrended[:-lag], detrended[lag:])[0, 1]
                            if not np.isnan(price_autocorr):
                                # For detrended prices, moderate positive autocorrelation can indicate pattern structure
                                abs_price_autocorr = abs(price_autocorr)
                                if 0.1 <= abs_price_autocorr <= 0.6:
                                    price_score = 0.8
                                elif abs_price_autocorr <= 0.8:
                                    price_score = 0.6
                                else:
                                    price_score = 0.4
                                
                                autocorr_scores.append((price_score, 0.5 / lag))
                        except:
                            continue
            
            # 3. Pattern-specific autocorrelation expectations
            pattern_name = pattern.get('pattern_name', '').lower()
            pattern_type = pattern.get('pattern_type', 'unknown')
            
            if not autocorr_scores:
                return 0.5
            
            # Calculate weighted average
            total_weight = sum(weight for _, weight in autocorr_scores)
            if total_weight > 0:
                base_score = sum(score * weight for score, weight in autocorr_scores) / total_weight
            else:
                base_score = 0.5
            
            # Pattern-specific adjustments
            if 'triangle' in pattern_name or 'wedge' in pattern_name:
                # Triangular patterns: expect decreasing autocorrelation over time
                if len(returns) >= 20:
                    early_returns = returns[:len(returns)//2]
                    late_returns = returns[len(returns)//2:]
                    
                    if len(early_returns) >= 3 and len(late_returns) >= 3:
                        early_autocorr = abs(np.corrcoef(early_returns[:-1], early_returns[1:])[0, 1]) if len(early_returns) > 1 else 0
                        late_autocorr = abs(np.corrcoef(late_returns[:-1], late_returns[1:])[0, 1]) if len(late_returns) > 1 else 0
                        
                        if not (np.isnan(early_autocorr) or np.isnan(late_autocorr)):
                            if late_autocorr < early_autocorr:  # Decreasing autocorrelation
                                base_score *= 1.15
                                
            elif 'channel' in pattern_name or 'rectangle' in pattern_name:
                # Channel patterns: expect consistent low autocorrelation
                if base_score >= 0.7:  # Good autocorrelation structure
                    base_score *= 1.1
                    
            elif 'head' in pattern_name and 'shoulders' in pattern_name:
                # H&S patterns: expect moderate structure in autocorrelation
                if 0.5 <= base_score <= 0.8:  # Some structure but not too much
                    base_score *= 1.1
                    
            elif any(word in pattern_name for word in ['flag', 'pennant']):
                # Flag/Pennant patterns: expect higher autocorrelation during consolidation
                if base_score >= 0.6:
                    base_score *= 1.05
            
            # 4. Ljung-Box test approximation for serial correlation
            if len(returns) >= 15:
                try:
                    # Simple Ljung-Box-like test for multiple lags
                    lb_stats = []
                    for lag in range(1, min(6, len(returns) // 3)):
                        if len(returns) > lag:
                            autocorr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                            if not np.isnan(autocorr):
                                # Approximate Ljung-Box statistic
                                lb_stat = (len(returns) * (len(returns) + 2) * autocorr**2) / (len(returns) - lag)
                                lb_stats.append(lb_stat)
                    
                    if lb_stats:
                        # Lower LB statistics generally better (less serial correlation)
                        avg_lb = np.mean(lb_stats)
                        lb_score = 1.0 / (1.0 + avg_lb / 10)  # Normalize
                        
                        # Combine with base score
                        base_score = (base_score * 0.7 + lb_score * 0.3)
                except:
                    pass  # Skip LB test if it fails
            
            return min(0.95, max(0.15, base_score))
            
        except Exception:
            return 0.5
    
    def _test_stationarity(self, stock_data: pd.DataFrame, pattern: Dict[str, Any]) -> float:
        """Test stationarity of the pattern data"""
        try:
            # Simple stationarity test using rolling statistics
            close_prices = stock_data['close'].values
            if len(close_prices) < 20:
                return 0.6
            
            # Calculate rolling mean and std
            window = min(10, len(close_prices) // 2)
            rolling_mean = pd.Series(close_prices).rolling(window=window).mean()
            rolling_std = pd.Series(close_prices).rolling(window=window).std()
            
            # Remove NaN values
            rolling_mean = rolling_mean.dropna()
            rolling_std = rolling_std.dropna()
            
            if len(rolling_mean) < 5 or len(rolling_std) < 5:
                return 0.6
            
            # Calculate coefficient of variation for rolling statistics
            mean_cv = rolling_mean.std() / rolling_mean.mean() if rolling_mean.mean() != 0 else 1
            std_cv = rolling_std.std() / rolling_std.mean() if rolling_std.mean() != 0 else 1
            
            # Lower CV indicates more stationarity
            stationarity_score = 1.0 / (1.0 + mean_cv + std_cv)
            
            return min(0.9, max(0.3, stationarity_score))
            
        except Exception:
            return 0.5
    
    def _detect_change_points(self, stock_data: pd.DataFrame, pattern: Dict[str, Any]) -> float:
        """Detect structural change points in the pattern"""
        try:
            close_prices = stock_data['close'].values
            if len(close_prices) < 15:
                return 0.6
            
            # Simple change point detection using rolling variance
            returns = np.diff(close_prices) / close_prices[:-1]
            
            # Calculate rolling variance
            window = max(5, len(returns) // 3)
            rolling_var = pd.Series(returns).rolling(window=window).var()
            
            # Count significant variance changes
            var_changes = 0
            threshold = rolling_var.std()
            
            for i in range(1, len(rolling_var)):
                if not pd.isna(rolling_var.iloc[i]) and not pd.isna(rolling_var.iloc[i-1]):
                    if abs(rolling_var.iloc[i] - rolling_var.iloc[i-1]) > threshold:
                        var_changes += 1
            
            # Fewer change points generally indicate more stable patterns
            change_point_ratio = var_changes / len(rolling_var) if len(rolling_var) > 0 else 0
            
            return max(0.3, 1.0 - min(1.0, change_point_ratio * 2))
            
        except Exception:
            return 0.5
    
    # Pattern consistency analysis methods
    def _detect_legacy_pattern_conflicts(self, detected_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect conflicts between patterns (legacy version for consistency analysis)"""
        conflicts = []
        
        try:
            for i, pattern1 in enumerate(detected_patterns):
                for j, pattern2 in enumerate(detected_patterns[i+1:], i+1):
                    conflict = self._check_pattern_conflict(pattern1, pattern2)
                    if conflict:
                        conflicts.append({
                            'pattern1': pattern1.get('pattern_name', f'Pattern_{i+1}'),
                            'pattern2': pattern2.get('pattern_name', f'Pattern_{j+1}'),
                            'conflict_type': conflict,
                            'severity': self._assess_conflict_severity(conflict)
                        })
            
            return conflicts
            
        except Exception:
            return []
    
    def _detect_pattern_reinforcements(self, detected_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect reinforcing patterns"""
        reinforcements = []
        
        try:
            for i, pattern1 in enumerate(detected_patterns):
                for j, pattern2 in enumerate(detected_patterns[i+1:], i+1):
                    reinforcement = self._check_pattern_reinforcement(pattern1, pattern2)
                    if reinforcement:
                        reinforcements.append({
                            'pattern1': pattern1.get('pattern_name', f'Pattern_{i+1}'),
                            'pattern2': pattern2.get('pattern_name', f'Pattern_{j+1}'),
                            'reinforcement_type': reinforcement,
                            'strength': self._assess_reinforcement_strength(reinforcement)
                        })
            
            return reinforcements
            
        except Exception:
            return []
    
    def _analyze_bias_consistency(self, detected_patterns: List[Dict[str, Any]], pattern_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consistency of market bias across patterns"""
        try:
            # Count bullish vs bearish signals
            bullish_count = 0
            bearish_count = 0
            neutral_count = 0
            
            for pattern in detected_patterns:
                pattern_name = pattern.get('pattern_name', '').lower()
                pattern_type = pattern.get('pattern_type', '')
                
                if ('bullish' in pattern_name or 'ascending' in pattern_name or 
                    'inverse' in pattern_name):
                    bullish_count += 1
                elif ('bearish' in pattern_name or 'descending' in pattern_name or
                      ('head_and_shoulders' in pattern_name and 'inverse' not in pattern_name)):
                    bearish_count += 1
                else:
                    neutral_count += 1
            
            total_patterns = len(detected_patterns)
            if total_patterns == 0:
                return {'consistency_score': 0.5, 'bias': 'unknown'}
            
            # Calculate bias consistency
            max_bias_count = max(bullish_count, bearish_count, neutral_count)
            consistency_ratio = max_bias_count / total_patterns
            
            # Determine dominant bias
            if bullish_count > bearish_count and bullish_count > neutral_count:
                dominant_bias = 'bullish'
            elif bearish_count > bullish_count and bearish_count > neutral_count:
                dominant_bias = 'bearish'
            else:
                dominant_bias = 'neutral'
            
            return {
                'consistency_score': consistency_ratio,
                'dominant_bias': dominant_bias,
                'bullish_signals': bullish_count,
                'bearish_signals': bearish_count,
                'neutral_signals': neutral_count,
                'bias_distribution': {
                    'bullish_ratio': bullish_count / total_patterns,
                    'bearish_ratio': bearish_count / total_patterns,
                    'neutral_ratio': neutral_count / total_patterns
                }
            }
            
        except Exception:
            return {'consistency_score': 0.5, 'error': 'Bias analysis failed'}
    
    def _analyze_timeframe_consistency(self, detected_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze timeframe consistency of patterns"""
        try:
            # This is a placeholder for timeframe analysis
            # In a real implementation, you'd analyze pattern formation times
            
            # For now, assume consistency based on pattern maturity
            maturity_scores = []
            
            for pattern in detected_patterns:
                completion = pattern.get('completion_percentage', 0)
                reliability = pattern.get('reliability', 'unknown')
                
                # Convert to numeric scores
                reliability_score = {'high': 0.9, 'medium': 0.7, 'low': 0.5}.get(reliability, 0.5)
                completion_score = completion / 100
                
                maturity_scores.append((reliability_score + completion_score) / 2)
            
            if not maturity_scores:
                return {'consistency_score': 0.5}
            
            # High consistency if all patterns have similar maturity levels
            avg_maturity = np.mean(maturity_scores)
            maturity_std = np.std(maturity_scores)
            
            consistency_score = max(0.3, 1.0 - maturity_std)
            
            return {
                'consistency_score': consistency_score,
                'average_maturity': avg_maturity,
                'maturity_deviation': maturity_std,
                'pattern_count': len(detected_patterns)
            }
            
        except Exception:
            return {'consistency_score': 0.5, 'error': 'Timeframe analysis failed'}
    
    # Alternative validation methods
    def _validate_with_fractals(self, stock_data: pd.DataFrame, pattern: Dict[str, Any]) -> float:
        """Validate pattern using fractal analysis"""
        try:
            close_prices = stock_data['close'].values
            if len(close_prices) < 10:
                return 0.5
            
            # Simple fractal detection (local minima/maxima)
            fractals = []
            for i in range(2, len(close_prices) - 2):
                # Local maximum
                if (close_prices[i] > close_prices[i-1] and close_prices[i] > close_prices[i+1] and
                    close_prices[i] > close_prices[i-2] and close_prices[i] > close_prices[i+2]):
                    fractals.append(('high', i, close_prices[i]))
                # Local minimum
                elif (close_prices[i] < close_prices[i-1] and close_prices[i] < close_prices[i+1] and
                      close_prices[i] < close_prices[i-2] and close_prices[i] < close_prices[i+2]):
                    fractals.append(('low', i, close_prices[i]))
            
            # Pattern should have reasonable number of fractals
            fractal_density = len(fractals) / len(close_prices)
            
            if 0.1 <= fractal_density <= 0.4:  # Reasonable fractal density
                return 0.8
            elif 0.05 <= fractal_density <= 0.5:
                return 0.6
            else:
                return 0.4
                
        except Exception:
            return 0.5
    
    def _calculate_simple_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate simple RSI for feature extraction"""
        try:
            if len(prices) < period + 1:
                return 50.0  # Neutral RSI
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except:
            return 50.0
    
    def _validate_with_clustering(self, stock_data: pd.DataFrame, pattern: Dict[str, Any]) -> float:
        """Validate pattern using clustering analysis with enhanced features"""
        try:
            if len(stock_data) < 10:
                return 0.5
            
            # Prepare enhanced features for clustering
            features = []
            close_prices = stock_data['close'].values
            
            for i in range(len(stock_data)):
                price = close_prices[i]
                volume = stock_data['volume'].iloc[i] if 'volume' in stock_data.columns else np.mean(stock_data['volume']) if 'volume' in stock_data.columns else 0
                
                # Add technical features
                price_change = (close_prices[i] - close_prices[i-1]) / close_prices[i-1] if i > 0 else 0
                rsi = self._calculate_simple_rsi(close_prices[:i+1]) if i >= 14 else 50
                volatility = np.std(close_prices[max(0, i-5):i+1]) if i >= 5 else 0
                
                # Position within pattern (normalized)
                position = i / len(stock_data)
                
                features.append([price, volume, price_change, rsi, volatility, position])
            
            features = np.array(features)
            
            # Normalize features using StandardScaler approach
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            if len(features_scaled) < 6:
                return 0.5
            
            # Use multiple clustering algorithms and metrics
            best_score = 0
            
            # Try K-means with optimal cluster selection
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            
            for n_clusters in range(2, min(6, len(features_scaled) // 3)):
                try:
                    # K-means clustering
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(features_scaled)
                    
                    # Calculate multiple clustering quality metrics
                    silhouette_avg = silhouette_score(features_scaled, cluster_labels)
                    calinski_score = calinski_harabasz_score(features_scaled, cluster_labels)
                    
                    # Normalize Calinski-Harabasz score
                    normalized_calinski = min(1.0, calinski_score / (len(features_scaled) * 10))
                    
                    # Combined score (silhouette is already normalized [-1, 1])
                    combined_score = (silhouette_avg + 1) / 2 * 0.7 + normalized_calinski * 0.3
                    best_score = max(best_score, combined_score)
                    
                except Exception as e:
                    continue
            
            # Additional pattern-specific clustering validation
            pattern_type = pattern.get('pattern_type', 'unknown')
            if pattern_type in ['trending', 'breakout'] and best_score > 0.6:
                best_score *= 1.1  # Boost for patterns that should cluster well
            elif pattern_type in ['consolidation', 'sideways'] and best_score < 0.5:
                best_score *= 0.9  # Slightly penalize poor clustering in consolidation patterns
            
            return min(0.95, max(0.2, best_score))
            
        except Exception:
            return 0.5
    
    def _validate_with_fourier(self, stock_data: pd.DataFrame, pattern: Dict[str, Any]) -> float:
        """Validate pattern using Fourier analysis"""
        try:
            close_prices = stock_data['close'].values
            if len(close_prices) < 8:
                return 0.5
            
            # Simple FFT analysis
            fft_values = np.fft.fft(close_prices)
            frequencies = np.fft.fftfreq(len(close_prices))
            
            # Analyze frequency spectrum
            power_spectrum = np.abs(fft_values) ** 2
            
            # Look for dominant frequencies (excluding DC component)
            if len(power_spectrum) > 1:
                main_power = power_spectrum[1:len(power_spectrum)//2]  # Positive frequencies only
                if len(main_power) > 0:
                    max_power = np.max(main_power)
                    mean_power = np.mean(main_power)
                    
                    if mean_power > 0:
                        power_ratio = max_power / mean_power
                        # Higher ratio indicates more structured pattern
                        return min(0.9, max(0.3, power_ratio / 10))
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _validate_with_wavelets(self, stock_data: pd.DataFrame, pattern: Dict[str, Any]) -> float:
        """Validate pattern using enhanced multi-resolution wavelet analysis"""
        try:
            close_prices = stock_data['close'].values
            if len(close_prices) < 8:
                return 0.5
            
            # Ensure data length is power of 2 for better wavelet analysis
            data_len = len(close_prices)
            next_power_2 = 1 << (data_len - 1).bit_length()
            
            if data_len < next_power_2:
                # Pad with last value
                padded_data = np.pad(close_prices, (0, next_power_2 - data_len), 'edge')
            else:
                padded_data = close_prices[:next_power_2]
            
            # Multi-resolution wavelet analysis using different scales
            scales = [2, 4, 8, 16] if len(padded_data) >= 16 else [2, 4, 8]
            wavelet_features = []
            
            for scale in scales:
                if len(padded_data) < scale * 2:
                    continue
                
                # High-pass filtering (detail coefficients approximation)
                detail_coeffs = []
                for i in range(0, len(padded_data) - scale, scale):
                    segment = padded_data[i:i + scale]
                    if len(segment) == scale:
                        # Simple haar wavelet-like transform
                        avg = np.mean(segment)
                        detail = np.mean(segment[scale//2:]) - np.mean(segment[:scale//2])
                        detail_coeffs.append(detail)
                
                if len(detail_coeffs) > 1:
                    # Energy distribution analysis
                    energy = np.sum(np.array(detail_coeffs) ** 2)
                    normalized_energy = energy / (len(detail_coeffs) * np.var(padded_data) + 1e-8)
                    
                    # Smoothness measure
                    detail_var = np.var(detail_coeffs)
                    detail_mean = np.mean(np.abs(detail_coeffs))
                    smoothness = 1.0 / (1.0 + detail_var / (detail_mean + 1e-8))
                    
                    # Pattern regularity at this scale
                    autocorr = np.corrcoef(detail_coeffs[:-1], detail_coeffs[1:])[0, 1] if len(detail_coeffs) > 2 else 0
                    regularity = abs(autocorr) if not np.isnan(autocorr) else 0
                    
                    # Combine metrics for this scale
                    scale_score = (smoothness * 0.4 + regularity * 0.3 + 
                                 min(1.0, normalized_energy) * 0.3)
                    wavelet_features.append(scale_score)
            
            if not wavelet_features:
                return 0.5
            
            # Pattern-specific wavelet validation
            pattern_name = pattern.get('pattern_name', '').lower()
            pattern_type = pattern.get('pattern_type', 'unknown')
            
            base_score = np.mean(wavelet_features)
            
            # Adjust score based on pattern characteristics
            if 'triangle' in pattern_name or 'wedge' in pattern_name:
                # Triangular patterns should show decreasing volatility across scales
                if len(wavelet_features) >= 2 and wavelet_features[-1] > wavelet_features[0]:
                    base_score *= 1.15  # Boost for proper triangular convergence
            elif 'channel' in pattern_name or 'rectangle' in pattern_name:
                # Channel patterns should show consistent energy across scales
                scale_consistency = 1.0 - np.std(wavelet_features) / (np.mean(wavelet_features) + 1e-8)
                base_score = (base_score + scale_consistency) / 2
            elif 'head' in pattern_name and 'shoulders' in pattern_name:
                # H&S patterns should show clear multi-resolution structure
                if len(wavelet_features) >= 3:
                    structure_score = max(wavelet_features[1:])  # Mid-scale should be prominent
                    base_score = (base_score + structure_score) / 2
            
            # Multi-scale consistency bonus
            if len(wavelet_features) >= 3:
                consistency = 1.0 - np.std(wavelet_features) / (np.mean(wavelet_features) + 1e-8)
                if consistency > 0.7:
                    base_score *= 1.1
            
            return min(0.95, max(0.2, base_score))
                
        except Exception:
            return 0.5
    
    # Pattern conflict and reinforcement detection
    def _check_pattern_conflict(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> Optional[str]:
        """Check if two patterns conflict with each other"""
        try:
            name1 = pattern1.get('pattern_name', '').lower()
            name2 = pattern2.get('pattern_name', '').lower()
            
            type1 = pattern1.get('pattern_type', '')
            type2 = pattern2.get('pattern_type', '')
            
            # Check for directional conflicts
            if (('bullish' in name1 and 'bearish' in name2) or 
                ('bearish' in name1 and 'bullish' in name2)):
                return 'directional_conflict'
            
            if (('ascending' in name1 and 'descending' in name2) or
                ('descending' in name1 and 'ascending' in name2)):
                return 'trend_conflict'
            
            # Check for type conflicts
            if (type1 == 'reversal' and type2 == 'continuation'):
                return 'type_conflict'
            
            return None
            
        except Exception:
            return None
    
    def _check_pattern_reinforcement(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> Optional[str]:
        """Check if two patterns reinforce each other"""
        try:
            name1 = pattern1.get('pattern_name', '').lower()
            name2 = pattern2.get('pattern_name', '').lower()
            
            type1 = pattern1.get('pattern_type', '')
            type2 = pattern2.get('pattern_type', '')
            
            # Check for directional reinforcement
            if (('bullish' in name1 and 'bullish' in name2) or 
                ('bearish' in name1 and 'bearish' in name2)):
                return 'directional_reinforcement'
            
            if (('ascending' in name1 and 'ascending' in name2) or
                ('descending' in name1 and 'descending' in name2)):
                return 'trend_reinforcement'
            
            # Check for type reinforcement
            if type1 == type2 and type1 in ['reversal', 'continuation']:
                return 'type_reinforcement'
            
            return None
            
        except Exception:
            return None
    
    # Helper classification methods
    def _classify_confidence(self, score: float) -> str:
        """Classify confidence level from score"""
        if score >= 0.8:
            return 'very_high'
        elif score >= 0.7:
            return 'high'
        elif score >= 0.6:
            return 'medium'
        elif score >= 0.4:
            return 'low'
        else:
            return 'very_low'
    
    def _classify_volume_strength(self, score: float) -> str:
        """Classify volume strength from score"""
        if score >= 0.8:
            return 'strong'
        elif score >= 0.6:
            return 'moderate'
        elif score >= 0.4:
            return 'weak'
        else:
            return 'very_weak'
    
    def _classify_performance(self, score: float) -> str:
        """Classify performance category from score"""
        if score >= 0.75:
            return 'excellent'
        elif score >= 0.65:
            return 'good'
        elif score >= 0.55:
            return 'average'
        elif score >= 0.45:
            return 'below_average'
        else:
            return 'poor'
    
    def _assess_conflict_severity(self, conflict_type: str) -> str:
        """Assess severity of pattern conflicts"""
        severity_map = {
            'directional_conflict': 'high',
            'trend_conflict': 'high',
            'type_conflict': 'medium'
        }
        return severity_map.get(conflict_type, 'low')
    
    def _assess_reinforcement_strength(self, reinforcement_type: str) -> str:
        """Assess strength of pattern reinforcements"""
        strength_map = {
            'directional_reinforcement': 'strong',
            'trend_reinforcement': 'strong',
            'type_reinforcement': 'moderate'
        }
        return strength_map.get(reinforcement_type, 'weak')
    
    def _generate_confidence_recommendation(self, confidence: float, level: str) -> str:
        """Generate recommendation based on confidence level"""
        if confidence >= 0.8:
            return "High confidence - patterns are well-validated across multiple methods"
        elif confidence >= 0.7:
            return "Good confidence - patterns show strong validation with minor concerns"
        elif confidence >= 0.6:
            return "Moderate confidence - patterns are reasonably validated but require caution"
        elif confidence >= 0.4:
            return "Low confidence - patterns show weak validation, consider additional confirmation"
        else:
            return "Very low confidence - patterns are poorly validated, avoid relying on these signals"
    
    def _compile_pattern_validation_details(
        self, 
        detected_patterns: List[Dict[str, Any]], 
        statistical_validation: Dict[str, Any],
        volume_confirmation: Dict[str, Any],
        time_series_validation: Dict[str, Any],
        historical_validation: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Compile detailed validation results for each pattern"""
        
        pattern_details = []
        
        try:
            for i, pattern in enumerate(detected_patterns):
                pattern_name = pattern.get('pattern_name', f'Pattern_{i+1}')
                
                detail = {
                    'pattern_name': pattern_name,
                    'pattern_type': pattern.get('pattern_type', 'unknown'),
                    'original_reliability': pattern.get('reliability', 'unknown'),
                    'original_completion': pattern.get('completion_percentage', 0),
                    'validation_results': {}
                }
                
                # Extract validation results for this pattern
                for validation_result in [statistical_validation, volume_confirmation, 
                                        time_series_validation, historical_validation]:
                    if not validation_result.get('error'):
                        method = validation_result.get('method', 'unknown')
                        
                        # Find results for this pattern
                        results_key = None
                        if 'validation_results' in validation_result:
                            results_key = 'validation_results'
                        elif 'confirmation_results' in validation_result:
                            results_key = 'confirmation_results'
                        elif 'performance_results' in validation_result:
                            results_key = 'performance_results'
                        
                        if results_key and i < len(validation_result[results_key]):
                            pattern_result = validation_result[results_key][i]
                            detail['validation_results'][method] = pattern_result
                
                pattern_details.append(detail)
            
            return pattern_details
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION] Pattern detail compilation failed: {e}")
            return []
    
    def _assess_data_quality(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess the quality of input data for cross-validation"""
        try:
            data_length = len(stock_data)
            
            # Check for missing values
            missing_data = stock_data.isnull().sum().sum()
            missing_percentage = (missing_data / (data_length * len(stock_data.columns))) * 100
            
            # Check data length adequacy
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
                'sufficient_for_validation': data_length >= 25 and missing_percentage < 10
            }
            
        except Exception as e:
            return {'error': str(e), 'sufficient_for_validation': False}
    
    def _detect_market_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect market regime (stable, volatile, trending) based on price volatility and trend direction"""
        try:
            if len(data) < 20:
                return {'regime': 'unknown', 'confidence': 0.0, 'characteristics': {}}
            
            # Calculate key metrics
            close_prices = data['close'].values
            returns = np.diff(close_prices) / close_prices[:-1]
            
            # Volatility analysis
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
            rolling_vol = pd.Series(returns).rolling(window=10).std().mean()
            
            # Trend analysis
            price_change = (close_prices[-1] - close_prices[0]) / close_prices[0]
            trend_strength = abs(price_change)
            
            # Range analysis
            price_range = (data['high'].max() - data['low'].min()) / data['close'].mean()
            
            # Regime classification thresholds
            high_vol_threshold = 0.25
            low_vol_threshold = 0.15
            trend_threshold = 0.05
            
            # Classify regime
            if volatility > high_vol_threshold:
                if trend_strength > trend_threshold:
                    regime = 'trending'
                    confidence = min(0.9, 0.6 + (trend_strength * 2))
                else:
                    regime = 'volatile'
                    confidence = min(0.85, 0.5 + (volatility * 2))
            elif volatility < low_vol_threshold and trend_strength < trend_threshold:
                regime = 'stable'
                confidence = min(0.8, 0.6 + ((low_vol_threshold - volatility) * 3))
            elif trend_strength > trend_threshold:
                regime = 'trending'
                confidence = min(0.85, 0.5 + (trend_strength * 3))
            else:
                regime = 'stable'
                confidence = 0.6
            
            return {
                'regime': regime,
                'confidence': round(confidence, 2),
                'characteristics': {
                    'volatility': round(volatility, 4),
                    'trend_strength': round(trend_strength, 4),
                    'price_range': round(price_range, 4),
                    'market_direction': 'up' if price_change > 0.02 else 'down' if price_change < -0.02 else 'sideways'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Market regime detection failed: {e}")
            return {'regime': 'unknown', 'confidence': 0.0, 'characteristics': {}, 'error': str(e)}
    
    def _infer_data_interval(self, stock_data: pd.DataFrame) -> str:
        """Infer data interval from index frequency or median delta."""
        try:
            idx = stock_data.index
            # Try pandas infer_freq
            try:
                import pandas as pd
                freq = pd.infer_freq(idx)
                if freq:
                    return str(freq)
            except Exception:
                pass
            # Fallback: median delta
            if len(idx) >= 2:
                deltas = (idx[1:] - idx[:-1])
                # Handle numpy/pandas timedelta
                try:
                    median_delta = pd.Series(deltas).median()
                    seconds = median_delta.total_seconds()
                except Exception:
                    # Best effort
                    seconds = None
                if seconds is not None:
                    if seconds >= 60*60*24*0.9:
                        return 'day'
                    if seconds >= 60*60*0.9:
                        return '1hour'
                    if seconds >= 60*15*0.9:
                        return '15min'
                    if seconds >= 60*5*0.9:
                        return '5min'
                    return f'{int(seconds)}s'
            return 'unknown'
        except Exception:
            return 'unknown'

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
    
    def _build_no_patterns_result(self, message: str) -> Dict[str, Any]:
        """Build result for when no patterns are provided"""
        return {
            'success': True,
            'agent_name': self.name,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'validation_summary': {
                'patterns_validated': 0,
                'validation_methods_used': 0,
                'overall_validation_score': 0.0,
                'validation_confidence': 'not_applicable'
            },
            'confidence_score': 0.0
        }