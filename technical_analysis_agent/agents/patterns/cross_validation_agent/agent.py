#!/usr/bin/env python3
"""
Cross-Validation Agent - Main Coordinator

This module coordinates all cross-validation analysis components including:
- Multi-method pattern validation processing
- Comprehensive validation chart generation
- LLM-powered validation insights and recommendations
- Integration with the broader pattern analysis system
"""

import pandas as pd
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import sys
import os
import json
from copy import deepcopy

# Add the backend directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import agent components
from agents.patterns.cross_validation_agent.processor import CrossValidationProcessor
from agents.patterns.cross_validation_agent.pattern_chart_generator import PatternChartGenerator
from agents.patterns.cross_validation_agent.pattern_detection import PatternDetector
from agents.patterns.cross_validation_agent.llm_agent import CrossValidationLLMAgent
from agents.patterns.market_structure_agent.processor import MarketStructureProcessor
# Try importing LLM client - use try/except for robustness
try:
    from llm import get_llm_client
except ImportError:
    try:
        from backend.llm import get_llm_client
    except ImportError:
        # Fallback for testing
        def get_llm_client(name):
            return None

logger = logging.getLogger(__name__)

class CrossValidationAgent:
    """
    Main Cross-Validation Agent that coordinates all validation components.
    
    This agent orchestrates:
    - Multi-method pattern validation analysis
    - Validation visualization and charting
    - AI-powered validation interpretation
    - Comprehensive confidence assessment
    """
    
    def __init__(self):
        self.name = "cross_validation_agent"
        self.version = "1.0.0"
        self.description = "Comprehensive pattern cross-validation and confidence assessment system"
        
        # Set up logging first
        self.logger = logging.getLogger(__name__)
        
        # Initialize sub-components
        self.pattern_detector = PatternDetector()
        self.processor = CrossValidationProcessor()
        self.chart_generator = PatternChartGenerator()
        self.llm_client = self._initialize_llm()
        
        # Initialize enhanced LLM agent with temporal analysis capabilities
        self.llm_agent = CrossValidationLLMAgent()
        
        # Initialize Market Structure processor for regime/context summary
        self.ms_processor = MarketStructureProcessor()
        
        self.logger.info(f"{self.name.title().replace('_', ' ')} Agent v{self.version} initialized")
    
    async def analyze_and_validate_patterns(
        self, 
        stock_data: pd.DataFrame,
        symbol: str = "STOCK",
        include_charts: bool = True,
        include_llm_analysis: bool = True,
        market_context: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete analysis pipeline: detect patterns and then validate them.
        
        This is the primary method for end-to-end pattern analysis including:
        1. Pattern detection from stock data
        2. Cross-validation of detected patterns
        3. Chart generation and LLM analysis (optional)
        
        Args:
            stock_data: DataFrame with OHLCV data
            symbol: Stock symbol for analysis
            include_charts: Whether to generate charts
            include_llm_analysis: Whether to include LLM analysis
            market_context: Additional market context for analysis
            save_path: Optional path to save charts and results
            
        Returns:
            Dictionary containing comprehensive pattern analysis and validation results
        """
        analysis_start_time = datetime.now()
        
        try:
            logger.info(f"[CROSS_VALIDATION_AGENT] Starting complete pattern analysis for {symbol}")
            
            # Validate inputs
            if stock_data is None or stock_data.empty:
                return self._build_error_result("No stock data provided for analysis", symbol)
            
            if len(stock_data) < 20:
                return self._build_error_result("Insufficient data for pattern analysis (minimum 20 periods required)", symbol)
            
            # Initialize result structure
            results = {
                'success': False,
                'agent_name': self.name,
                'symbol': symbol,
                'analysis_timestamp': analysis_start_time.isoformat(),
                'components_executed': [],
                'total_processing_time': 0.0
            }
            
            # Step 1: Pattern Detection
            logger.info(f"[CROSS_VALIDATION_AGENT] Detecting patterns for {symbol}")
            pattern_detection_results = self.pattern_detector.detect_patterns(stock_data)
            
            if not pattern_detection_results.get('success', False):
                logger.warning(f"[CROSS_VALIDATION_AGENT] Pattern detection failed for {symbol}")
                # Still continue with validation using empty patterns if detection fails
                detected_patterns = []
                pattern_summary = {'total_patterns': 0, 'dominant_pattern': 'none'}
            else:
                detected_patterns = pattern_detection_results.get('detected_patterns', [])
                pattern_summary = pattern_detection_results.get('pattern_summary', {})
            
            results['pattern_detection'] = pattern_detection_results
            results['components_executed'].append('pattern_detection')
            
            # Step 2: Cross-Validation (proceed even if no patterns detected)
            validation_results = await self.validate_patterns(
                stock_data=stock_data,
                detected_patterns=detected_patterns,
                pattern_summary=pattern_summary,
                symbol=symbol,
                include_charts=include_charts,
                include_llm_analysis=include_llm_analysis,
                market_context=market_context,
                save_path=save_path
            )
            
            # Merge validation results
            results.update(validation_results)
            
            # Calculate total processing time
            total_time = (datetime.now() - analysis_start_time).total_seconds()
            results['total_processing_time'] = total_time
            
            # Final validation and summary
            results['success'] = self._validate_results(results)
            results['analysis_summary'] = self._generate_analysis_summary(results)
            
            logger.info(f"[CROSS_VALIDATION_AGENT] Complete analysis completed for {symbol} in {total_time:.2f}s")
            
            return results
            
        except Exception as e:
            total_time = (datetime.now() - analysis_start_time).total_seconds()
            logger.error(f"[CROSS_VALIDATION_AGENT] Complete analysis failed for {symbol}: {e}")
            return self._build_error_result(str(e), symbol, total_time)
    
    async def validate_patterns(
        self, 
        stock_data: pd.DataFrame,
        detected_patterns: List[Dict[str, Any]],
        pattern_summary: Dict[str, Any],
        symbol: str = "STOCK",
        include_charts: bool = True,
        include_llm_analysis: bool = True,
        market_context: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive cross-validation analysis for detected patterns.
        
        Args:
            stock_data: DataFrame with OHLCV data
            detected_patterns: List of patterns to validate
            pattern_summary: Summary of pattern detection results
            symbol: Stock symbol for analysis
            include_charts: Whether to generate charts
            include_llm_analysis: Whether to include LLM analysis
            market_context: Additional market context for analysis
            save_path: Optional path to save charts and results
            
        Returns:
            Dictionary containing comprehensive cross-validation results
        """
        analysis_start_time = datetime.now()
        
        try:
            logger.info(f"[CROSS_VALIDATION_AGENT] Starting comprehensive validation for {symbol}")
            
            # Validate inputs
            if stock_data is None or stock_data.empty:
                return self._build_error_result("No stock data provided for validation", symbol)
            
            if len(stock_data) < 20:
                return self._build_error_result("Insufficient data for cross-validation (minimum 20 periods required)", symbol)
            
            # Allow validation even with empty patterns for baseline analysis
            if not detected_patterns:
                logger.info(f"[CROSS_VALIDATION_AGENT] No patterns detected for {symbol}, proceeding with baseline validation")
            
            # Initialize result structure
            results = {
                'success': False,
                'agent_name': self.name,
                'symbol': symbol,
                'analysis_timestamp': analysis_start_time.isoformat(),
                'components_executed': [],
                'total_processing_time': 0.0
            }
            
            # Execute validation pipeline
            validation_results = await self._execute_validation_pipeline(
                stock_data, detected_patterns, pattern_summary, symbol,
                include_charts, include_llm_analysis, market_context, save_path
            )
            
            # Merge results
            results.update(validation_results)
            
            # Calculate total processing time
            total_time = (datetime.now() - analysis_start_time).total_seconds()
            results['total_processing_time'] = total_time
            
            # Final validation and summary
            results['success'] = self._validate_results(results)
            results['analysis_summary'] = self._generate_analysis_summary(results)
            
            logger.info(f"[CROSS_VALIDATION_AGENT] Validation completed for {symbol} in {total_time:.2f}s")
            
            return results
            
        except Exception as e:
            total_time = (datetime.now() - analysis_start_time).total_seconds()
            logger.error(f"[CROSS_VALIDATION_AGENT] Validation failed for {symbol}: {e}")
            return self._build_error_result(str(e), symbol, total_time)
    
    async def _execute_validation_pipeline(
        self, 
        stock_data: pd.DataFrame,
        detected_patterns: List[Dict[str, Any]],
        pattern_summary: Dict[str, Any],
        symbol: str,
        include_charts: bool,
        include_llm_analysis: bool,
        market_context: Optional[Dict[str, Any]],
        save_path: Optional[str]
    ) -> Dict[str, Any]:
        """Execute the complete validation pipeline"""
        
        results = {'components_executed': []}
        
        try:
            # Step 1: Cross-Validation Processing
            logger.info(f"[CROSS_VALIDATION_AGENT] Executing cross-validation processing for {symbol}")
            validation_results = self.processor.process_cross_validation_data(
                stock_data, detected_patterns, pattern_summary
            )
            
            results['cross_validation'] = validation_results
            results['components_executed'].append('cross_validation')
            
            if not validation_results.get('success', False):
                logger.warning(f"[CROSS_VALIDATION_AGENT] Cross-validation processing failed for {symbol}")
                return results
            
            # Early exit efficiency: if no patterns validated, skip charts and LLM
            try:
                patterns_validated = validation_results.get('validation_summary', {}).get('patterns_validated', 0)
            except Exception:
                patterns_validated = 0
            if patterns_validated == 0:
                logger.info(f"[CROSS_VALIDATION_AGENT] No patterns to analyze for {symbol} — skipping charts and LLM")
                return results
            
            # Step 2: Chart Generation (if requested)
            charts_results = None
            chart_image_bytes = None
            if include_charts:
                try:
                    logger.info(f"[CROSS_VALIDATION_AGENT] Generating pattern visualization chart for {symbol}")
                    
                    # Generate pattern chart in memory like market structure agent
                    chart_image_bytes = await asyncio.to_thread(
                        self.chart_generator.generate_pattern_chart_bytes,
                        stock_data, detected_patterns, symbol
                    )
                    
                    if chart_image_bytes:
                        logger.info(f"[CROSS_VALIDATION_AGENT] Chart image generated: {len(chart_image_bytes)} bytes")
                        charts_results = {
                            'success': True,
                            'chart_type': 'pattern_visualization',
                            'chart_size_bytes': len(chart_image_bytes),
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        logger.warning(f"[CROSS_VALIDATION_AGENT] Chart generation returned no data")
                        charts_results = {'success': False, 'error': 'No chart data generated'}
                    
                    results['charts'] = charts_results
                    results['chart_image_bytes'] = chart_image_bytes
                    results['components_executed'].append('charts')
                    
                except Exception as e:
                    logger.error(f"[CROSS_VALIDATION_AGENT] Chart generation failed for {symbol}: {e}")
                    results['charts'] = {'success': False, 'error': str(e)}
                    chart_image_bytes = None
            
            # Step 2.5: Build Market Structure context (lightweight)
            market_context = None
            try:
                logger.info(f"[CROSS_VALIDATION_AGENT] Building market structure context for {symbol}")
                ms_result = self.ms_processor.process_market_structure_data(stock_data)
                if ms_result and ms_result.get('success'):
                    market_context = self._build_market_context_from_structure(ms_result)
                else:
                    logger.warning(f"[CROSS_VALIDATION_AGENT] Market structure analysis unavailable for {symbol}")
            except Exception as e:
                logger.warning(f"[CROSS_VALIDATION_AGENT] Market structure context failed for {symbol}: {e}")
                market_context = None
            results['market_context'] = market_context
            # Persist into cross_validation block as well for any downstream consumers/savers
            try:
                if 'cross_validation' in results and isinstance(results['cross_validation'], dict):
                    results['cross_validation']['market_context'] = market_context
            except Exception:
                pass

            # Step 2.6: Score, filter, rank detected patterns for LLM focus
            try:
                filtering_output = self._score_filter_rank_patterns(
                    detected_patterns=detected_patterns,
                    validation_results=validation_results,
                    top_k=5,
                    recency_days=60
                )
                filtered_patterns = filtering_output['filtered_patterns']
                filtered_validation_data = filtering_output['filtered_validation_data']
                discarded_patterns = filtering_output['discarded_patterns']
                # Attach market context into filtered_validation_data for prompt fallback
                try:
                    if market_context:
                        filtered_validation_data['market_context'] = market_context
                except Exception:
                    pass
                results['filtered_patterns'] = filtered_patterns
                results['discarded_patterns'] = discarded_patterns
            except Exception as e:
                logger.warning(f"[CROSS_VALIDATION_AGENT] Pattern filtering/ranking failed for {symbol}: {e}")
                filtered_patterns = detected_patterns
                filtered_validation_data = validation_results
                results['filtering_error'] = str(e)
            
            # Step 3: LLM Analysis (if requested)
            llm_results = None
            if include_llm_analysis:
                try:
                    logger.info(f"[CROSS_VALIDATION_AGENT] Executing LLM validation analysis for {symbol}")
                    # Debug: confirm market_context presence and source
                    try:
                        mc_source = market_context.get('source') if isinstance(market_context, dict) else None
                        logger.info(f"[CROSS_VALIDATION_AGENT] Market context present: {bool(market_context)}; source={mc_source}")
                        if isinstance(market_context, dict):
                            logger.info("[CROSS_VALIDATION_AGENT] Market structure context payload:\n%s", json.dumps(market_context, indent=2, default=str))
                    except Exception:
                        logger.info(f"[CROSS_VALIDATION_AGENT] Market context present: False; source=None")

                    # Get chart image bytes if available
                    llm_chart_image_bytes = results.get('chart_image_bytes') if include_charts else None
                    
                    llm_results = await self._generate_llm_analysis(
                        filtered_validation_data, filtered_patterns, symbol, market_context, llm_chart_image_bytes
                    )
                    
                    results['llm_analysis'] = llm_results
                    results['components_executed'].append('llm_analysis')
                    
                except Exception as e:
                    logger.error(f"[CROSS_VALIDATION_AGENT] LLM analysis failed for {symbol}: {e}")
                    results['llm_analysis'] = {'success': False, 'error': str(e)}
            
            # Step 4: Integrate Results
            integrated_results = self._integrate_validation_components(
                validation_results, charts_results, llm_results, detected_patterns, symbol
            )
            results.update(integrated_results)
            
            return results
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_AGENT] Pipeline execution failed for {symbol}: {e}")
            results['pipeline_error'] = str(e)
            return results
    
    def _integrate_validation_components(
        self,
        validation_results: Dict[str, Any],
        charts_results: Optional[Dict[str, Any]],
        llm_results: Optional[Dict[str, Any]],
        detected_patterns: List[Dict[str, Any]],
        symbol: str
    ) -> Dict[str, Any]:
        """Integrate results from all validation components"""
        
        try:
            integrated = {
                'integration_success': True,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
            
            # Extract key metrics from cross-validation
            if validation_results.get('success'):
                integrated.update({
                    # Core Validation Results
                    'validation_summary': validation_results.get('validation_summary', {}),
                    'validation_scores': validation_results.get('validation_scores', {}),
                    'final_confidence_assessment': validation_results.get('final_confidence_assessment', {}),
                    
                    # Detailed Validation Results
                    'statistical_validation': validation_results.get('statistical_validation', {}),
                    'volume_confirmation': validation_results.get('volume_confirmation', {}),
                    'time_series_validation': validation_results.get('time_series_validation', {}),
                    'historical_validation': validation_results.get('historical_validation', {}),
                    'consistency_analysis': validation_results.get('consistency_analysis', {}),
                    'alternative_validation': validation_results.get('alternative_validation', {}),
                    
                    # Pattern-Specific Results
                    'pattern_validation_details': validation_results.get('pattern_validation_details', []),
                    
                    # Core Metrics
                    'patterns_validated': validation_results.get('validation_summary', {}).get('patterns_validated', 0),
                    'validation_methods_used': validation_results.get('validation_summary', {}).get('validation_methods_used', 0),
                    'overall_validation_score': validation_results.get('validation_scores', {}).get('overall_score', 0),
                    'validation_confidence': validation_results.get('confidence_score', 0),
                    'data_quality': validation_results.get('data_quality', {})
                })
            
            # Add chart information
            if charts_results and charts_results.get('success'):
                integrated['charts_generated'] = charts_results.get('charts_generated', 0)
                integrated['chart_types'] = list(charts_results.get('charts_data', {}).keys())
            
            # Add LLM insights
            if llm_results and llm_results.get('success'):
                integrated.update({
                    'validation_insights': {
                        'validation_reliability_assessment': llm_results.get('validation_reliability_assessment', ''),
                        'pattern_confidence_evaluation': llm_results.get('pattern_confidence_evaluation', ''),
                        'risk_assessment': llm_results.get('risk_assessment', ''),
                        'trading_decision_framework': llm_results.get('trading_decision_framework', ''),
                        'validation_insights': llm_results.get('validation_insights', ''),
                    },
                    'ai_analysis_quality': llm_results.get('analysis_quality', 'unknown'),
                    'ai_confidence_assessment': llm_results.get('confidence_score', 0)
                })
            
            # Calculate overall validation confidence
            integrated['overall_validation_confidence'] = self._calculate_overall_validation_confidence(
                validation_results, llm_results
            )
            
            # Generate validation executive summary
            integrated['validation_executive_summary'] = self._generate_validation_executive_summary(
                integrated, detected_patterns
            )
            
            # Generate validation recommendations
            integrated['validation_recommendations'] = self._generate_validation_recommendations(
                integrated, validation_results
            )
            
            return integrated
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_AGENT] Component integration failed: {e}")
            return {
                'integration_success': False,
                'integration_error': str(e),
                'symbol': symbol
            }
    
    def _calculate_overall_validation_confidence(
        self, 
        validation_results: Dict[str, Any], 
        llm_results: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate overall validation confidence from all components"""
        
        try:
            # Primary confidence from validation processing
            primary_confidence = validation_results.get('confidence_score', 0.0)
            
            # Validation completeness factor
            validation_scores = validation_results.get('validation_scores', {})
            completeness = validation_scores.get('validation_completeness', 0.5)
            
            # LLM analysis quality factor
            llm_quality_factor = 1.0
            if llm_results and llm_results.get('success'):
                ai_quality = llm_results.get('analysis_quality', 'unknown')
                llm_quality_factor = {
                    'excellent': 1.1,
                    'good': 1.05,
                    'fair': 1.0,
                    'poor': 0.95,
                    'fallback': 0.9,
                    'unknown': 1.0
                }.get(ai_quality, 1.0)
            
            # Method diversity factor (more methods = higher confidence)
            method_count = len(validation_scores.get('method_scores', {}))
            method_factor = min(1.2, 1.0 + (method_count - 1) * 0.03)
            
            # Overall calculation: Weight primary confidence, adjust for completeness and quality
            overall_confidence = primary_confidence * completeness * llm_quality_factor * method_factor
            
            return min(1.0, max(0.0, overall_confidence))
            
        except Exception:
            return 0.5
    
    def _generate_validation_executive_summary(
        self, 
        integrated_results: Dict[str, Any],
        detected_patterns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate executive summary of validation analysis"""
        
        try:
            symbol = integrated_results.get('symbol', 'STOCK')
            patterns_validated = integrated_results.get('patterns_validated', 0)
            methods_used = integrated_results.get('validation_methods_used', 0)
            overall_score = integrated_results.get('overall_validation_score', 0)
            validation_confidence = integrated_results.get('validation_confidence', 0)
            overall_confidence = integrated_results.get('overall_validation_confidence', 0)
            
            # Get final confidence assessment
            final_assessment = integrated_results.get('final_confidence_assessment', {})
            confidence_level = final_assessment.get('confidence_level', 'unknown')
            confidence_category = final_assessment.get('confidence_category', 'unknown')
            
            # Generate key findings
            key_findings = []
            
            key_findings.append(f"{patterns_validated} patterns validated using {methods_used} methods")
            key_findings.append(f"Overall validation score: {overall_score:.2f}")
            key_findings.append(f"Validation confidence: {confidence_level}")
            
            # Add method-specific findings
            validation_scores = integrated_results.get('validation_scores', {})
            method_scores = validation_scores.get('method_scores', {})
            
            if method_scores:
                best_method = max(method_scores.items(), key=lambda x: x[1])
                key_findings.append(f"Strongest validation method: {best_method[0]} ({best_method[1]:.2f})")
            
            # Pattern consistency findings
            consistency_analysis = integrated_results.get('consistency_analysis', {})
            if not consistency_analysis.get('error'):
                conflicts = len(consistency_analysis.get('pattern_conflicts', []))
                reinforcements = len(consistency_analysis.get('pattern_reinforcements', []))
                if conflicts > 0:
                    key_findings.append(f"{conflicts} pattern conflicts detected")
                if reinforcements > 0:
                    key_findings.append(f"{reinforcements} pattern reinforcements found")
            
            # Risk assessment
            if overall_confidence >= 0.8:
                risk_level = 'low'
                recommendation = 'High confidence validation - patterns are well-supported'
            elif overall_confidence >= 0.6:
                risk_level = 'medium'
                recommendation = 'Moderate confidence validation - patterns show reasonable support'
            elif overall_confidence >= 0.4:
                risk_level = 'high'
                recommendation = 'Low confidence validation - patterns require additional confirmation'
            else:
                risk_level = 'very high'
                recommendation = 'Very low confidence validation - patterns are poorly supported'
            
            return {
                'symbol': symbol,
                'patterns_validated': patterns_validated,
                'validation_methods_used': methods_used,
                'overall_validation_score': overall_score,
                'validation_confidence_level': confidence_level,
                'validation_confidence_category': confidence_category,
                'overall_confidence': f"{overall_confidence:.1%}",
                'risk_level': risk_level,
                'recommendation': recommendation,
                'key_findings': key_findings,
                'validation_quality': 'comprehensive' if methods_used >= 4 else 'partial'
            }
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_AGENT] Executive summary generation failed: {e}")
            return {'error': str(e)}
    
    def _generate_validation_recommendations(
        self,
        integrated_results: Dict[str, Any],
        validation_results: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable validation recommendations"""
        
        recommendations = []
        
        try:
            overall_score = integrated_results.get('overall_validation_score', 0)
            confidence_level = integrated_results.get('final_confidence_assessment', {}).get('confidence_level', 'unknown')
            methods_used = integrated_results.get('validation_methods_used', 0)
            patterns_validated = integrated_results.get('patterns_validated', 0)
            
            # Core validation recommendations
            if overall_score >= 0.8:
                recommendations.append("High validation confidence - patterns are well-supported across multiple methods")
                recommendations.append("Consider standard position sizing for these validated patterns")
            elif overall_score >= 0.6:
                recommendations.append("Moderate validation confidence - patterns show reasonable support")
                recommendations.append("Consider reduced position sizing or additional confirmation")
            elif overall_score >= 0.4:
                recommendations.append("Low validation confidence - patterns require additional confirmation")
                recommendations.append("Use smaller position sizes and tighter risk management")
            else:
                recommendations.append("Very low validation confidence - avoid relying solely on these patterns")
                recommendations.append("Seek additional technical or fundamental confirmation")
            
            # Method-specific recommendations
            if methods_used < 4:
                recommendations.append(f"Only {methods_used} validation methods applied - consider expanding validation coverage")
            
            # Pattern-specific recommendations
            pattern_details = integrated_results.get('pattern_validation_details', [])
            if pattern_details:
                high_confidence_patterns = []
                low_confidence_patterns = []
                
                for detail in pattern_details:
                    pattern_name = detail.get('pattern_name', 'unknown')
                    validation_results = detail.get('validation_results', {})
                    
                    # Calculate average validation score for this pattern
                    scores = []
                    for method_data in validation_results.values():
                        if isinstance(method_data, dict):
                            for key, value in method_data.items():
                                if 'score' in key and isinstance(value, (int, float)):
                                    scores.append(value)
                    
                    avg_score = sum(scores) / len(scores) if scores else 0
                    
                    if avg_score >= 0.7:
                        high_confidence_patterns.append(pattern_name)
                    elif avg_score < 0.4:
                        low_confidence_patterns.append(pattern_name)
                
                if high_confidence_patterns:
                    recommendations.append(f"Highest confidence patterns: {', '.join(high_confidence_patterns)}")
                if low_confidence_patterns:
                    recommendations.append(f"Lowest confidence patterns requiring caution: {', '.join(low_confidence_patterns)}")
            
            # Consistency recommendations
            consistency_analysis = integrated_results.get('consistency_analysis', {})
            if not consistency_analysis.get('error'):
                conflicts = consistency_analysis.get('pattern_conflicts', [])
                if conflicts:
                    recommendations.append(f"Pattern conflicts detected - review {len(conflicts)} conflicting signals carefully")
                
                reinforcements = consistency_analysis.get('pattern_reinforcements', [])
                if reinforcements:
                    recommendations.append(f"Pattern reinforcements found - {len(reinforcements)} patterns support each other")
            
            # Data quality recommendations
            data_quality = integrated_results.get('data_quality', {})
            if data_quality.get('overall_quality_score', 100) < 80:
                recommendations.append("Data quality issues detected - validation confidence may be impacted")
            
            # Ensure we have at least some recommendations
            if not recommendations:
                recommendations.append("Review validation results carefully before making trading decisions")
                recommendations.append("Consider market context and additional confirmation signals")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_AGENT] Recommendation generation failed: {e}")
            return ["Error generating recommendations - review validation results manually"]
    
    def _generate_analysis_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of the entire validation analysis process"""
        
        try:
            return {
                'components_executed': len(results.get('components_executed', [])),
                'components_list': results.get('components_executed', []),
                'cross_validation_success': results.get('cross_validation', {}).get('success', False),
                'charts_generated': results.get('charts', {}).get('success', False),
                'llm_analysis_success': results.get('llm_analysis', {}).get('success', False),
                'integration_success': results.get('integration_success', False),
                'total_processing_time': results.get('total_processing_time', 0),
                'overall_success': results.get('success', False),
                'patterns_validated': results.get('patterns_validated', 0),
                'validation_methods_used': results.get('validation_methods_used', 0),
                'final_validation_confidence': results.get('overall_validation_confidence', 0)
            }
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_AGENT] Analysis summary generation failed: {e}")
            return {'error': str(e)}
    
    def _validate_results(self, results: Dict[str, Any]) -> bool:
        """Validate that analysis results are complete and coherent"""
        
        try:
            # Check if cross-validation succeeded
            if not results.get('cross_validation', {}).get('success', False):
                return False
            
            # Check if we have validation results
            validation_summary = results.get('validation_summary', {})
            if not validation_summary or validation_summary.get('patterns_validated', 0) == 0:
                return False
            
            # Check integration success
            if not results.get('integration_success', False):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_AGENT] Results validation failed: {e}")
            return False
    
    def _initialize_llm(self):
        """Initialize LLM client for cross-validation analysis"""
        try:
            # Use cross_validation_agent configuration from llm_assignments.yaml
            llm_client = get_llm_client("cross_validation_agent")
            self.logger.info("✅ Cross Validation LLM Agent initialized with backend/llm")
            return llm_client
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Cross Validation LLM Agent: {e}")
            return None
    
    async def _generate_llm_analysis(self, validation_results: Dict[str, Any], detected_patterns: List[Dict[str, Any]], symbol: str, market_context: Optional[Dict[str, Any]] = None, chart_image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        """Generate LLM analysis for validation results using enhanced LLM agent"""
        try:
            # Use the enhanced LLM agent with temporal analysis capabilities
            return await self.llm_agent.generate_validation_analysis(
                validation_data=validation_results,
                detected_patterns=detected_patterns,
                symbol=symbol,
                market_context=market_context,  # pass market structure context when available
                chart_image_bytes=chart_image_bytes  # pass chart image bytes for multimodal analysis
            )
                
        except Exception as e:
            error_msg = f"Enhanced LLM analysis failed: {str(e)}"
            self.logger.error(f"[CROSS_VALIDATION_LLM] {error_msg} for {symbol}")
            return {'success': False, 'error': error_msg}
    
    # Note: Old simple prompt creation method removed - now using enhanced LLM agent
    # with comprehensive temporal analysis capabilities in CrossValidationLLMAgent
    
    def _build_error_result(self, error_message: str, symbol: str = "UNKNOWN", processing_time: float = 0.0) -> Dict[str, Any]:
        """Build standardized error result"""
        return {
            'success': False,
            'agent_name': self.name,
            'symbol': symbol,
            'error': error_message,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_processing_time': processing_time,
            'components_executed': [],
            'patterns_validated': 0,
            'validation_methods_used': 0,
            'overall_validation_confidence': 0.0
        }
    
    def _build_no_patterns_result(self, message: str, symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """Build result for when no patterns are provided"""
        return {
            'success': True,
            'agent_name': self.name,
            'symbol': symbol,
            'message': message,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_processing_time': 0.0,
            'components_executed': [],
            'patterns_validated': 0,
            'validation_methods_used': 0,
            'overall_validation_confidence': 0.0,
            'validation_executive_summary': {
                'symbol': symbol,
                'patterns_validated': 0,
                'recommendation': 'No patterns provided for validation',
                'key_findings': ['No patterns to validate'],
                'validation_quality': 'not_applicable'
            }
        }

    def _build_market_context_from_structure(self, ms_result: Dict[str, Any]) -> Dict[str, Any]:
        """Build compact market context payload from Market Structure analysis"""
        try:
            from datetime import datetime
            ta = ms_result.get('trend_analysis', {}) or {}
            bos = ms_result.get('bos_choch_analysis', {}) or {}
            kl = ms_result.get('key_levels', {}) or {}
            cs = ms_result.get('current_state', {}) or {}
            fr = ms_result.get('fractal_analysis', {}) or {}

            current_price = kl.get('current_price')
            def dist_pct(level):
                try:
                    if current_price is None or level is None or float(current_price) == 0:
                        return None
                    return round(100.0 * (float(level) - float(current_price)) / float(current_price), 2)
                except Exception:
                    return None

            ns = kl.get('nearest_support') or {}
            nr = kl.get('nearest_resistance') or {}
            regime = ta.get('market_regime', {}) or {}

            return {
                'source': 'market_structure_agent',
                'timestamp': datetime.now().isoformat(),
                'regime': {
                    'regime': regime.get('regime', 'unknown'),
                    'confidence': regime.get('confidence', 0.0)
                },
                'structure_bias': bos.get('structural_bias', 'unknown'),
                'trend': {
                    'direction': ta.get('trend_direction', 'unknown'),
                    'strength': ta.get('trend_strength', 'unknown'),
                    'quality': ta.get('trend_quality', 'unknown')
                },
                'bos_choch': {
                    'total_bos_events': bos.get('total_bos_events', 0),
                    'total_choch_events': bos.get('total_choch_events', 0),
                    'recent_structural_break': bos.get('recent_structural_break')
                },
                'key_levels': {
                    'current_price': current_price,
                    'nearest_support': (
                        {'level': ns.get('level'), 'distance_pct': dist_pct(ns.get('level'))} if ns else None
                    ),
                    'nearest_resistance': (
                        {'level': nr.get('level'), 'distance_pct': dist_pct(nr.get('level'))} if nr else None
                    ),
                    'price_position_description': cs.get('price_position_description', 'unknown')
                },
                'fractal': {
                    'timeframe_alignment': fr.get('timeframe_alignment', 'unknown'),
                    'trend_consensus': fr.get('trend_consensus', 'unknown')
                }
            }
        except Exception as e:
            self.logger.error(f"[CROSS_VALIDATION_AGENT] Failed to build market context from structure: {e}")
            return {'source': 'market_structure_agent', 'error': str(e)}

    def _score_filter_rank_patterns(
        self,
        detected_patterns: List[Dict[str, Any]],
        validation_results: Dict[str, Any],
        top_k: int = 5,
        recency_days: int = 60
    ) -> Dict[str, Any]:
        """Compute composite scores, filter stale/low-quality patterns, deduplicate, and rank.
        Returns filtered patterns list, filtered validation data copy, and discarded reasons.
        """
        # Defensive copies
        filtered_validation_data = deepcopy(validation_results)
        pattern_details = validation_results.get('pattern_validation_details', []) or []

        # Helper: extract per-pattern scores from detail
        def extract_scores(detail: Dict[str, Any]) -> Dict[str, float]:
            vr = detail.get('validation_results', {}) or {}
            # Initialize defaults
            s = {
                'statistical': None,
                'volume': None,
                'time_series': None,
                'historical': None
            }
            # Statistical
            stat = vr.get('statistical_validation', {}) or {}
            if isinstance(stat, dict):
                s['statistical'] = stat.get('statistical_score')
            # Volume
            vol = vr.get('volume_confirmation', {}) or {}
            if isinstance(vol, dict):
                s['volume'] = vol.get('volume_confirmation_score')
            # Time series
            ts = vr.get('time_series_validation', {}) or {}
            if isinstance(ts, dict):
                s['time_series'] = ts.get('time_series_score')
            # Historical
            hist = vr.get('historical_performance', {}) or {}
            if isinstance(hist, dict):
                s['historical'] = hist.get('adjusted_success_rate')
            return s

        def compute_composite(s: Dict[str, Optional[float]]) -> float:
            # Weights
            w = {
                'statistical': 0.40,
                'volume': 0.20,
                'time_series': 0.20,
                'historical': 0.20
            }
            total = 0.0
            total_w = 0.0
            for k, wt in w.items():
                val = s.get(k)
                if isinstance(val, (int, float)):
                    total += float(val) * wt
                    total_w += wt
            if total_w == 0:
                return 0.5
            return max(0.0, min(1.0, total / total_w))

        def reliability_rank(rel: str) -> int:
            r = (rel or '').lower()
            return {'high': 0, 'medium': 1, 'low': 2}.get(r, 3)

        # Build pattern rows with computed composite and reasons
        rows = []
        for idx, pattern in enumerate(detected_patterns):
            detail = pattern_details[idx] if idx < len(pattern_details) else {}
            scores = extract_scores(detail)
            composite = compute_composite(scores)
            age_days = pattern.get('pattern_age_days')
            status = (pattern.get('completion_status') or '').lower()
            completion = pattern.get('completion_percentage', 0) or 0
            reliability = (pattern.get('reliability') or 'unknown').lower()
            name = pattern.get('pattern_name', f'Pattern_{idx+1}')

            # Quality flags
            reasons = []
            # Recency filter
            if isinstance(age_days, (int, float)):
                if age_days > recency_days:
                    reasons.append('stale')
            # Completion for completed
            if status == 'completed' and completion < 80:
                reasons.append('low_completion')
            # Reliability gate
            allow_low = (reliability == 'low' and composite >= 0.65 and (not isinstance(age_days, (int, float)) or age_days <= recency_days))
            if reliability == 'low' and not allow_low:
                reasons.append('low_reliability')
            # Volume+Stat joint weakness
            vol_score = scores.get('volume')
            stat_score = scores.get('statistical')
            if isinstance(vol_score, (int, float)) and isinstance(stat_score, (int, float)):
                if vol_score < 0.5 and stat_score < 0.55:
                    # Exception: forming & very recent (<=7 days)
                    very_recent = isinstance(age_days, (int, float)) and age_days <= 7
                    if not (status == 'forming' and very_recent):
                        reasons.append('weak_vol_and_stats')

            rows.append({
                'index': idx,
                'pattern': pattern,
                'detail': detail,
                'scores': scores,
                'composite_score': round(composite, 3),
                'reliability': reliability,
                'age_days': age_days,
                'status': status,
                'completion': completion,
                'name': name,
                'reasons': reasons
            })

        # Apply initial filters (remove if any reasons exist)
        kept = [r for r in rows if len(r['reasons']) == 0]
        discarded = [
            {
                'name': r['name'],
                'age_days': r['age_days'],
                'status': r['status'],
                'reliability': r['reliability'],
                'composite_score': r['composite_score'],
                'reasons': r['reasons']
            }
            for r in rows if len(r['reasons']) > 0
        ]

        # Deduplicate by pattern_name (keep best: higher composite, then newer, then higher completion)
        dedup_map = {}
        for r in kept:
            key = (r['name'] or '').lower()
            prev = dedup_map.get(key)
            if prev is None:
                dedup_map[key] = r
            else:
                better = False
                if r['composite_score'] > prev['composite_score']:
                    better = True
                elif r['composite_score'] == prev['composite_score']:
                    # Prefer more recent (smaller age_days)
                    prev_age = prev['age_days'] if isinstance(prev['age_days'], (int, float)) else float('inf')
                    this_age = r['age_days'] if isinstance(r['age_days'], (int, float)) else float('inf')
                    if this_age < prev_age:
                        better = True
                    elif this_age == prev_age and r['completion'] > prev['completion']:
                        better = True
                if better:
                    # Move previous to discarded as duplicate lower rank
                    discarded.append({
                        'name': prev['name'], 'age_days': prev['age_days'], 'status': prev['status'],
                        'reliability': prev['reliability'], 'composite_score': prev['composite_score'],
                        'reasons': prev['reasons'] + ['duplicate_lower_rank']
                    })
                    dedup_map[key] = r
                else:
                    discarded.append({
                        'name': r['name'], 'age_days': r['age_days'], 'status': r['status'],
                        'reliability': r['reliability'], 'composite_score': r['composite_score'],
                        'reasons': r['reasons'] + ['duplicate_lower_rank']
                    })

        kept_dedup = list(dedup_map.values())

        # Rank: composite desc, age asc, reliability rank
        def sort_key(r):
            age = r['age_days'] if isinstance(r['age_days'], (int, float)) else float('inf')
            return (-r['composite_score'], age, reliability_rank(r['reliability']))

        kept_sorted = sorted(kept_dedup, key=sort_key)

        # Cap top_k
        final_rows = kept_sorted[:top_k]
        # Add any excess to discarded with reason 'exceeds_top_k'
        for r in kept_sorted[top_k:]:
            discarded.append({
                'name': r['name'], 'age_days': r['age_days'], 'status': r['status'],
                'reliability': r['reliability'], 'composite_score': r['composite_score'],
                'reasons': r['reasons'] + ['exceeds_top_k']
            })

        # Build filtered lists
        filtered_indices = [r['index'] for r in final_rows]
        filtered_patterns = [detected_patterns[i] for i in filtered_indices]
        filtered_details = [pattern_details[i] for i in filtered_indices]

        # Rewrite validation data copy for LLM (limit to filtered details and counts)
        filtered_validation_data['pattern_validation_details'] = filtered_details
        # Adjust patterns_validated count to filtered set for clarity in LLM output
        if 'validation_summary' in filtered_validation_data:
            filtered_validation_data['validation_summary'] = dict(filtered_validation_data['validation_summary'])
            filtered_validation_data['validation_summary']['patterns_validated'] = len(filtered_details)

        return {
            'filtered_patterns': filtered_patterns,
            'filtered_validation_data': filtered_validation_data,
            'discarded_patterns': discarded
        }

# Convenience function for external usage
async def validate_stock_patterns(
    stock_data: pd.DataFrame,
    detected_patterns: List[Dict[str, Any]],
    pattern_summary: Dict[str, Any],
    symbol: str = "STOCK",
    include_charts: bool = True,
    include_llm_analysis: bool = True,
    market_context: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to validate stock patterns using the Cross-Validation Agent.
    
    Args:
        stock_data: DataFrame with OHLCV data
        detected_patterns: List of patterns to validate
        pattern_summary: Summary of pattern detection results
        symbol: Stock symbol
        include_charts: Whether to generate visualization charts
        include_llm_analysis: Whether to include AI-powered insights
        market_context: Additional market context
        save_path: Path to save results and charts
        
    Returns:
        Comprehensive cross-validation analysis results
    """
    agent = CrossValidationAgent()
    return await agent.validate_patterns(
        stock_data, detected_patterns, pattern_summary, symbol,
        include_charts, include_llm_analysis, market_context, save_path
    )