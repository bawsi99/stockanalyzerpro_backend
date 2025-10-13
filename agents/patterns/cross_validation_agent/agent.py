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

# Add the backend directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import agent components
from agents.patterns.cross_validation_agent.processor import CrossValidationProcessor
from agents.patterns.cross_validation_agent.charts import CrossValidationChartGenerator
from llm import get_llm_client

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
        self.processor = CrossValidationProcessor()
        self.chart_generator = CrossValidationChartGenerator()
        self.llm_client = self._initialize_llm()
        
        self.logger.info(f"{self.name.title().replace('_', ' ')} Agent v{self.version} initialized")
    
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
            
            if not detected_patterns:
                return self._build_no_patterns_result("No patterns provided for cross-validation", symbol)
            
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
            
            # Step 2: Chart Generation (if requested)
            charts_results = None
            if include_charts:
                try:
                    logger.info(f"[CROSS_VALIDATION_AGENT] Generating validation charts for {symbol}")
                    charts_results = self.chart_generator.generate_cross_validation_charts(
                        stock_data, validation_results, symbol, save_path
                    )
                    
                    results['charts'] = charts_results
                    results['components_executed'].append('charts')
                    
                except Exception as e:
                    logger.error(f"[CROSS_VALIDATION_AGENT] Chart generation failed for {symbol}: {e}")
                    results['charts'] = {'success': False, 'error': str(e)}
            
            # Step 3: LLM Analysis (if requested)
            llm_results = None
            if include_llm_analysis:
                try:
                    logger.info(f"[CROSS_VALIDATION_AGENT] Executing LLM validation analysis for {symbol}")
                    
                    llm_results = await self._generate_llm_analysis(
                        validation_results, detected_patterns, symbol
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
    
    async def _generate_llm_analysis(self, validation_results: Dict[str, Any], detected_patterns: List[Dict[str, Any]], symbol: str) -> Dict[str, Any]:
        """Generate LLM analysis for validation results"""
        try:
            if not self.llm_client:
                return {'success': False, 'error': 'LLM client not initialized'}
            
            # Create analysis prompt
            prompt = self._create_validation_analysis_prompt(validation_results, detected_patterns, symbol)
            
            self.logger.info(f"[CROSS_VALIDATION_LLM] Sending analysis request for {symbol}")
            self.logger.info(f"[CROSS_VALIDATION_LLM] Prompt length: {len(prompt)} characters")
            
            # Get LLM response with timeout
            response, token_usage = await asyncio.wait_for(
                self.llm_client.generate_text(prompt, return_token_usage=True),
                timeout=90.0  # 90 second timeout
            )
            
            if response and len(response.strip()) > 0:
                self.logger.info(f"[CROSS_VALIDATION_LLM] Analysis completed for {symbol}")
                self.logger.info(f"[CROSS_VALIDATION_LLM] Response length: {len(response)} characters")
                
                return {
                    'success': True,
                    'analysis': response,
                    'token_usage': token_usage if token_usage else {},
                    'model_used': 'gemini-2.5-flash',
                    'response_time': 0  # Will be populated by the LLM client
                }
            else:
                error_msg = 'No response from LLM or empty response'
                self.logger.error(f"[CROSS_VALIDATION_LLM] Analysis failed for {symbol}: {error_msg}")
                return {'success': False, 'error': error_msg}
                
        except asyncio.TimeoutError:
            error_msg = "LLM analysis timed out after 90 seconds"
            self.logger.error(f"[CROSS_VALIDATION_LLM] {error_msg} for {symbol}")
            return {'success': False, 'error': error_msg}
        except Exception as e:
            error_msg = f"LLM analysis failed: {str(e)}"
            self.logger.error(f"[CROSS_VALIDATION_LLM] {error_msg} for {symbol}")
            return {'success': False, 'error': error_msg}
    
    def _create_validation_analysis_prompt(self, validation_results: Dict[str, Any], detected_patterns: List[Dict[str, Any]], symbol: str) -> str:
        """Create structured prompt for LLM validation analysis"""
        
        validation_summary = validation_results.get('validation_summary', {})
        market_regime = validation_results.get('market_regime_analysis', {})
        
        prompt = f"""Please analyze this comprehensive cross-validation analysis for {symbol} and provide actionable insights.

## VALIDATION SUMMARY
- Patterns Validated: {validation_summary.get('patterns_validated', 0)}
- Validation Methods Used: {validation_summary.get('validation_methods_used', 0)}
- Overall Validation Score: {validation_summary.get('overall_validation_score', 0):.2f}
- Validation Confidence: {validation_summary.get('validation_confidence', 'unknown')}
- Market Regime: {market_regime.get('regime', 'unknown')}

## DETECTED PATTERNS"""
        
        for i, pattern in enumerate(detected_patterns[:5]):  # Limit to top 5 patterns
            prompt += f"""
### Pattern {i+1}: {pattern.get('pattern_name', 'Unknown')}
- Type: {pattern.get('pattern_type', 'unknown')}
- Completion: {pattern.get('completion_percentage', 0):.1f}%
- Reliability: {pattern.get('reliability', 'unknown')}
- Pattern Quality: {pattern.get('pattern_quality', 'unknown')}"""
        
        # Add validation method results
        statistical_val = validation_results.get('statistical_validation', {})
        volume_conf = validation_results.get('volume_confirmation', {})
        historical_val = validation_results.get('historical_validation', {})
        
        prompt += f"""

## VALIDATION RESULTS
### Statistical Validation
- Overall Score: {statistical_val.get('overall_statistical_score', 0):.2f}
- Patterns Tested: {statistical_val.get('patterns_tested', 0)}

### Volume Confirmation  
- Overall Score: {volume_conf.get('overall_volume_score', 0):.2f}
- Patterns Analyzed: {volume_conf.get('patterns_analyzed', 0)}

### Historical Performance
- Overall Score: {historical_val.get('overall_historical_score', 0):.2f}
- Patterns Analyzed: {historical_val.get('patterns_analyzed', 0)}

## ANALYSIS REQUEST
Provide a comprehensive validation assessment including:
1. **Validation Confidence**: Overall confidence in the detected patterns based on cross-validation results
2. **Pattern Reliability**: Which patterns have the strongest validation support
3. **Risk Assessment**: Potential risks and limitations identified through validation
4. **Market Context**: How the current market regime affects pattern reliability
5. **Trading Implications**: Actionable insights for trading decisions
6. **Recommendations**: Specific recommendations based on validation findings

Format your response as clear, actionable insights for {symbol} pattern validation."""
        
        return prompt
    
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