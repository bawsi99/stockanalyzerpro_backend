#!/usr/bin/env python3
"""
Pattern Detection Agent - Main Coordinator

This module coordinates all pattern detection analysis components including:
- Technical pattern detection processing
- Chart generation for pattern visualization
- LLM-powered pattern analysis and insights
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
from agents.patterns.pattern_detection_agent.processor import PatternDetectionProcessor
from agents.patterns.pattern_detection_agent.charts import PatternDetectionChartGenerator
from agents.patterns.pattern_detection_agent.llm_agent import PatternDetectionLLMAgent

logger = logging.getLogger(__name__)

class PatternDetectionAgent:
    """
    Main Pattern Detection Agent that coordinates all analysis components.
    
    This agent orchestrates:
    - Technical pattern detection and analysis
    - Chart generation for visualization
    - AI-powered pattern interpretation
    - Comprehensive result aggregation
    """
    
    def __init__(self):
        self.name = "pattern_detection_agent"
        self.version = "1.0.0"
        self.description = "Comprehensive pattern detection and analysis system"
        
        # Initialize sub-components
        self.processor = PatternDetectionProcessor()
        self.chart_generator = PatternDetectionChartGenerator()
        self.llm_agent = PatternDetectionLLMAgent()
    
    async def analyze_patterns(
        self, 
        stock_data: pd.DataFrame,
        symbol: str = "STOCK",
        include_charts: bool = True,
        include_llm_analysis: bool = True,
        market_context: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive pattern detection analysis.
        
        Args:
            stock_data: DataFrame with OHLCV data
            symbol: Stock symbol for analysis
            include_charts: Whether to generate charts
            include_llm_analysis: Whether to include LLM analysis
            market_context: Additional market context for analysis
            save_path: Optional path to save charts and results
            
        Returns:
            Dictionary containing comprehensive pattern detection results
        """
        analysis_start_time = datetime.now()
        
        try:
            logger.info(f"[PATTERN_DETECTION_AGENT] Starting comprehensive analysis for {symbol}")
            
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
            
            # Execute analysis components
            analysis_results = await self._execute_analysis_pipeline(
                stock_data, symbol, include_charts, include_llm_analysis,
                market_context, save_path
            )
            
            # Merge results
            results.update(analysis_results)
            
            # Calculate total processing time
            total_time = (datetime.now() - analysis_start_time).total_seconds()
            results['total_processing_time'] = total_time
            
            # Final validation and summary
            results['success'] = self._validate_analysis_results(results)
            results['analysis_summary'] = self._generate_analysis_summary(results)
            
            logger.info(f"[PATTERN_DETECTION_AGENT] Analysis completed for {symbol} in {total_time:.2f}s")
            
            return results
            
        except Exception as e:
            total_time = (datetime.now() - analysis_start_time).total_seconds()
            logger.error(f"[PATTERN_DETECTION_AGENT] Analysis failed for {symbol}: {e}")
            return self._build_error_result(str(e), symbol, total_time)
    
    async def _execute_analysis_pipeline(
        self, 
        stock_data: pd.DataFrame,
        symbol: str,
        include_charts: bool,
        include_llm_analysis: bool,
        market_context: Optional[Dict[str, Any]],
        save_path: Optional[str]
    ) -> Dict[str, Any]:
        """Execute the complete analysis pipeline"""
        
        results = {'components_executed': []}
        
        try:
            # Step 1: Technical Pattern Detection
            logger.info(f"[PATTERN_DETECTION_AGENT] Executing technical pattern detection for {symbol}")
            pattern_results = self.processor.process_pattern_detection_data(stock_data)
            
            results['pattern_detection'] = pattern_results
            results['components_executed'].append('pattern_detection')
            
            if not pattern_results.get('success', False):
                logger.warning(f"[PATTERN_DETECTION_AGENT] Pattern detection failed for {symbol}")
                return results
            
            # Step 2: Chart Generation (if requested)
            charts_results = None
            if include_charts:
                try:
                    logger.info(f"[PATTERN_DETECTION_AGENT] Generating charts for {symbol}")
                    charts_results = self.chart_generator.generate_pattern_detection_charts(
                        stock_data, pattern_results, symbol, save_path
                    )
                    
                    results['charts'] = charts_results
                    results['components_executed'].append('charts')
                    
                except Exception as e:
                    logger.error(f"[PATTERN_DETECTION_AGENT] Chart generation failed for {symbol}: {e}")
                    results['charts'] = {'success': False, 'error': str(e)}
            
            # Step 3: LLM Analysis (if requested)
            llm_results = None
            if include_llm_analysis:
                try:
                    logger.info(f"[PATTERN_DETECTION_AGENT] Executing LLM analysis for {symbol}")
                    
                    # Prepare stock data summary for LLM
                    stock_data_summary = self._prepare_stock_data_summary(stock_data, symbol)
                    
                    llm_results = await self.llm_agent.generate_pattern_analysis(
                        stock_data_summary, pattern_results, symbol, market_context
                    )
                    
                    results['llm_analysis'] = llm_results
                    results['components_executed'].append('llm_analysis')
                    
                except Exception as e:
                    logger.error(f"[PATTERN_DETECTION_AGENT] LLM analysis failed for {symbol}: {e}")
                    results['llm_analysis'] = {'success': False, 'error': str(e)}
            
            # Step 4: Integrate Results
            integrated_results = self._integrate_analysis_components(
                pattern_results, charts_results, llm_results, symbol
            )
            results.update(integrated_results)
            
            return results
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_AGENT] Pipeline execution failed for {symbol}: {e}")
            results['pipeline_error'] = str(e)
            return results
    
    def _prepare_stock_data_summary(self, stock_data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Prepare stock data summary for LLM analysis"""
        try:
            current_price = float(stock_data['close'].iloc[-1])
            open_price = float(stock_data['open'].iloc[0])
            high_price = float(stock_data['high'].max())
            low_price = float(stock_data['low'].min())
            
            price_change = current_price - open_price
            price_change_pct = (price_change / open_price) * 100 if open_price > 0 else 0
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'period_open': open_price,
                'period_high': high_price,
                'period_low': low_price,
                'price_change': price_change,
                'price_change_percent': price_change_pct,
                'data_points': len(stock_data),
                'timeframe': 'daily',  # Could be made dynamic
                'volatility': float(stock_data['close'].pct_change().std() * 100) if len(stock_data) > 1 else 0
            }
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_AGENT] Stock data summary preparation failed: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def _integrate_analysis_components(
        self,
        pattern_results: Dict[str, Any],
        charts_results: Optional[Dict[str, Any]],
        llm_results: Optional[Dict[str, Any]],
        symbol: str
    ) -> Dict[str, Any]:
        """Integrate results from all analysis components"""
        
        try:
            integrated = {
                'integration_success': True,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
            
            # Extract key metrics from pattern detection
            if pattern_results.get('success'):
                integrated.update({
                    'detected_patterns': pattern_results.get('detected_patterns', []),
                    'total_patterns_detected': pattern_results.get('total_patterns_detected', 0),
                    'pattern_summary': pattern_results.get('pattern_summary', {}),
                    'formation_stage': pattern_results.get('formation_stage', {}),
                    'key_levels': pattern_results.get('key_levels', {}),
                    'technical_confidence': pattern_results.get('confidence_score', 0),
                    'data_quality': pattern_results.get('data_quality', {})
                })
            
            # Add chart information
            if charts_results and charts_results.get('success'):
                integrated['charts_generated'] = charts_results.get('charts_generated', 0)
                integrated['chart_types'] = list(charts_results.get('charts_data', {}).keys())
            
            # Add LLM insights
            if llm_results and llm_results.get('success'):
                integrated.update({
                    'ai_insights': {
                        'pattern_interpretation': llm_results.get('pattern_interpretation', ''),
                        'market_outlook': llm_results.get('market_outlook', ''),
                        'trading_strategy': llm_results.get('trading_strategy', ''),
                        'risk_assessment': llm_results.get('risk_assessment', ''),
                        'key_insights': llm_results.get('key_insights', ''),
                    },
                    'ai_analysis_quality': llm_results.get('analysis_quality', 'unknown'),
                    'ai_confidence': llm_results.get('confidence_score', 0)
                })
            
            # Calculate overall confidence score
            integrated['overall_confidence'] = self._calculate_overall_confidence(
                pattern_results, llm_results
            )
            
            # Generate executive summary
            integrated['executive_summary'] = self._generate_executive_summary(integrated)
            
            return integrated
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_AGENT] Component integration failed: {e}")
            return {
                'integration_success': False,
                'integration_error': str(e),
                'symbol': symbol
            }
    
    def _calculate_overall_confidence(
        self, 
        pattern_results: Dict[str, Any], 
        llm_results: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate overall confidence score from all components"""
        
        try:
            technical_confidence = pattern_results.get('confidence_score', 0.0)
            
            # Weight: 70% technical, 30% AI analysis quality
            if llm_results and llm_results.get('success'):
                ai_quality = llm_results.get('analysis_quality', 'unknown')
                ai_confidence_factor = {
                    'excellent': 1.0,
                    'good': 0.8,
                    'fair': 0.6,
                    'poor': 0.4,
                    'fallback': 0.2,
                    'unknown': 0.3
                }.get(ai_quality, 0.3)
                
                overall_confidence = (0.7 * technical_confidence) + (0.3 * ai_confidence_factor)
            else:
                # Only technical confidence available
                overall_confidence = technical_confidence * 0.8  # Slightly reduced due to missing AI component
            
            return min(1.0, max(0.0, overall_confidence))
            
        except Exception:
            return 0.5
    
    def _generate_executive_summary(self, integrated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of the analysis"""
        
        try:
            symbol = integrated_results.get('symbol', 'STOCK')
            total_patterns = integrated_results.get('total_patterns_detected', 0)
            overall_confidence = integrated_results.get('overall_confidence', 0)
            
            pattern_summary = integrated_results.get('pattern_summary', {})
            dominant_pattern = pattern_summary.get('dominant_pattern', 'none')
            overall_bias = pattern_summary.get('overall_bias', 'neutral')
            
            formation_stage = integrated_results.get('formation_stage', {})
            pattern_maturity = formation_stage.get('pattern_maturity', 'unknown')
            
            # Generate key findings
            key_findings = []
            
            if total_patterns > 0:
                key_findings.append(f"{total_patterns} pattern(s) detected")
                if dominant_pattern != 'none':
                    key_findings.append(f"Dominant pattern: {dominant_pattern.replace('_', ' ').title()}")
                if overall_bias != 'neutral':
                    key_findings.append(f"Market bias: {overall_bias}")
            else:
                key_findings.append("No significant patterns detected")
            
            # Risk level assessment
            if overall_confidence >= 0.8:
                risk_level = 'low'
                recommendation = 'Patterns provide strong signals'
            elif overall_confidence >= 0.6:
                risk_level = 'medium'
                recommendation = 'Patterns provide moderate signals'
            elif overall_confidence >= 0.4:
                risk_level = 'high'
                recommendation = 'Patterns provide weak signals'
            else:
                risk_level = 'very high'
                recommendation = 'Insufficient reliable pattern signals'
            
            return {
                'symbol': symbol,
                'patterns_detected': total_patterns,
                'dominant_pattern': dominant_pattern,
                'market_bias': overall_bias,
                'pattern_maturity': pattern_maturity,
                'confidence_level': f"{overall_confidence:.1%}",
                'risk_level': risk_level,
                'recommendation': recommendation,
                'key_findings': key_findings,
                'analysis_quality': 'comprehensive' if total_patterns > 0 else 'limited'
            }
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_AGENT] Executive summary generation failed: {e}")
            return {'error': str(e)}
    
    def _generate_analysis_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of the entire analysis process"""
        
        try:
            return {
                'components_executed': len(results.get('components_executed', [])),
                'components_list': results.get('components_executed', []),
                'pattern_detection_success': results.get('pattern_detection', {}).get('success', False),
                'charts_generated': results.get('charts', {}).get('success', False),
                'llm_analysis_success': results.get('llm_analysis', {}).get('success', False),
                'integration_success': results.get('integration_success', False),
                'total_processing_time': results.get('total_processing_time', 0),
                'overall_success': results.get('success', False)
            }
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_AGENT] Analysis summary generation failed: {e}")
            return {'error': str(e)}
    
    def _validate_analysis_results(self, results: Dict[str, Any]) -> bool:
        """Validate that analysis results are complete and coherent"""
        
        try:
            # Check if pattern detection succeeded
            if not results.get('pattern_detection', {}).get('success', False):
                return False
            
            # Check if we have the minimum required data
            if not results.get('detected_patterns') and not results.get('total_patterns_detected'):
                return False
            
            # Check integration success
            if not results.get('integration_success', False):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_AGENT] Results validation failed: {e}")
            return False
    
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
            'detected_patterns': [],
            'total_patterns_detected': 0,
            'overall_confidence': 0.0
        }

# Convenience function for external usage
async def analyze_stock_patterns(
    stock_data: pd.DataFrame,
    symbol: str = "STOCK",
    include_charts: bool = True,
    include_llm_analysis: bool = True,
    market_context: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to analyze stock patterns using the Pattern Detection Agent.
    
    Args:
        stock_data: DataFrame with OHLCV data
        symbol: Stock symbol
        include_charts: Whether to generate visualization charts
        include_llm_analysis: Whether to include AI-powered insights
        market_context: Additional market context
        save_path: Path to save results and charts
        
    Returns:
        Comprehensive pattern detection analysis results
    """
    agent = PatternDetectionAgent()
    return await agent.analyze_patterns(
        stock_data, symbol, include_charts, include_llm_analysis, market_context, save_path
    )