#!/usr/bin/env python3
"""
Comprehensive Test Suite for Integrated Market Structure Agent

This module provides extensive testing for the integrated market structure agent,
covering various market scenarios, error conditions, and performance validation.

Features:
- Multiple market scenario testing (uptrend, downtrend, sideways, volatile)
- Error handling and resilience testing
- Performance measurement and validation
- LLM response quality assessment
- Chart generation validation
- End-to-end integration testing
"""

import os
import sys
import asyncio
import logging
import json
import time
from typing import Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
import traceback

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the integrated agent
try:
    from integrated_market_structure_agent import IntegratedMarketStructureAgent
except ImportError:
    print("❌ Failed to import IntegratedMarketStructureAgent")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntegratedAgentTestSuite:
    """
    Comprehensive test suite for the integrated market structure agent.
    """
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = {}
        
        # Initialize test agent
        self.agent = IntegratedMarketStructureAgent(
            charts_output_dir="test_integrated_charts",
            results_output_dir="test_integrated_results",
            agent_name="pattern_agent"
        )
        
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """
        Run the complete test suite.
        
        Returns:
            Dictionary containing all test results and metrics
        """
        logger.info("🚀 Starting comprehensive test suite for Integrated Market Structure Agent")
        
        start_time = time.time()
        
        # Test scenarios
        test_scenarios = [
            await self._test_strong_uptrend_scenario(),
            await self._test_clear_downtrend_scenario(),
            await self._test_sideways_consolidation_scenario(),
            await self._test_volatile_market_scenario(),
            await self._test_breakout_scenario(),
            await self._test_error_handling(),
            await self._test_performance_validation()
        ]
        
        total_time = time.time() - start_time
        
        # Compile results
        results = self._compile_test_results(test_scenarios, total_time)
        
        # Save comprehensive report
        report_path = await self._save_test_report(results)
        results['report_path'] = report_path
        
        logger.info(f"✅ Test suite completed in {total_time:.2f}s - Report: {report_path}")
        return results
    
    async def _test_strong_uptrend_scenario(self) -> Dict[str, Any]:
        """Test strong uptrend market scenario."""
        logger.info("📈 Testing strong uptrend scenario...")
        
        try:
            # Create strong uptrend data
            stock_data, analysis_data = self._create_strong_uptrend_data()
            
            # Run analysis
            start_time = time.time()
            result = await self.agent.analyze_market_structure(
                stock_data=stock_data,
                analysis_data=analysis_data,
                symbol="UPTEST",
                scenario_description="Strong Uptrend Test"
            )
            execution_time = time.time() - start_time
            
            # Validate results
            validation = self._validate_analysis_result(result, "uptrend")
            
            return {
                'test_name': 'strong_uptrend_scenario',
                'success': result.get('success', False),
                'execution_time': execution_time,
                'result': result,
                'validation': validation,
                'chart_generated': result.get('chart_info', {}).get('generated', False),
                'llm_response_quality': result.get('llm_analysis', {}).get('response_quality', 'unknown'),
                'overall_quality_score': result.get('performance_metrics', {}).get('overall_quality_score', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Strong uptrend test failed: {e}")
            return {
                'test_name': 'strong_uptrend_scenario',
                'success': False,
                'error': str(e),
                'execution_time': 0
            }
    
    async def _test_clear_downtrend_scenario(self) -> Dict[str, Any]:
        """Test clear downtrend market scenario."""
        logger.info("📉 Testing clear downtrend scenario...")
        
        try:
            stock_data, analysis_data = self._create_clear_downtrend_data()
            
            start_time = time.time()
            result = await self.agent.analyze_market_structure(
                stock_data=stock_data,
                analysis_data=analysis_data,
                symbol="DOWNTEST",
                scenario_description="Clear Downtrend Test"
            )
            execution_time = time.time() - start_time
            
            validation = self._validate_analysis_result(result, "downtrend")
            
            return {
                'test_name': 'clear_downtrend_scenario',
                'success': result.get('success', False),
                'execution_time': execution_time,
                'result': result,
                'validation': validation,
                'chart_generated': result.get('chart_info', {}).get('generated', False),
                'llm_response_quality': result.get('llm_analysis', {}).get('response_quality', 'unknown'),
                'overall_quality_score': result.get('performance_metrics', {}).get('overall_quality_score', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Clear downtrend test failed: {e}")
            return {
                'test_name': 'clear_downtrend_scenario',
                'success': False,
                'error': str(e),
                'execution_time': 0
            }
    
    async def _test_sideways_consolidation_scenario(self) -> Dict[str, Any]:
        """Test sideways consolidation scenario."""
        logger.info("↔️ Testing sideways consolidation scenario...")
        
        try:
            stock_data, analysis_data = self._create_sideways_consolidation_data()
            
            start_time = time.time()
            result = await self.agent.analyze_market_structure(
                stock_data=stock_data,
                analysis_data=analysis_data,
                symbol="SIDETEST",
                scenario_description="Sideways Consolidation Test"
            )
            execution_time = time.time() - start_time
            
            validation = self._validate_analysis_result(result, "sideways")
            
            return {
                'test_name': 'sideways_consolidation_scenario',
                'success': result.get('success', False),
                'execution_time': execution_time,
                'result': result,
                'validation': validation,
                'chart_generated': result.get('chart_info', {}).get('generated', False),
                'llm_response_quality': result.get('llm_analysis', {}).get('response_quality', 'unknown'),
                'overall_quality_score': result.get('performance_metrics', {}).get('overall_quality_score', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Sideways consolidation test failed: {e}")
            return {
                'test_name': 'sideways_consolidation_scenario',
                'success': False,
                'error': str(e),
                'execution_time': 0
            }
    
    async def _test_volatile_market_scenario(self) -> Dict[str, Any]:
        """Test volatile market scenario."""
        logger.info("🌪️ Testing volatile market scenario...")
        
        try:
            stock_data, analysis_data = self._create_volatile_market_data()
            
            start_time = time.time()
            result = await self.agent.analyze_market_structure(
                stock_data=stock_data,
                analysis_data=analysis_data,
                symbol="VOLTEST",
                scenario_description="Volatile Market Test"
            )
            execution_time = time.time() - start_time
            
            validation = self._validate_analysis_result(result, "volatile")
            
            return {
                'test_name': 'volatile_market_scenario',
                'success': result.get('success', False),
                'execution_time': execution_time,
                'result': result,
                'validation': validation,
                'chart_generated': result.get('chart_info', {}).get('generated', False),
                'llm_response_quality': result.get('llm_analysis', {}).get('response_quality', 'unknown'),
                'overall_quality_score': result.get('performance_metrics', {}).get('overall_quality_score', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Volatile market test failed: {e}")
            return {
                'test_name': 'volatile_market_scenario',
                'success': False,
                'error': str(e),
                'execution_time': 0
            }
    
    async def _test_breakout_scenario(self) -> Dict[str, Any]:
        """Test breakout scenario."""
        logger.info("🚀 Testing breakout scenario...")
        
        try:
            stock_data, analysis_data = self._create_breakout_scenario_data()
            
            start_time = time.time()
            result = await self.agent.analyze_market_structure(
                stock_data=stock_data,
                analysis_data=analysis_data,
                symbol="BREAKTEST",
                scenario_description="Breakout Scenario Test"
            )
            execution_time = time.time() - start_time
            
            validation = self._validate_analysis_result(result, "breakout")
            
            return {
                'test_name': 'breakout_scenario',
                'success': result.get('success', False),
                'execution_time': execution_time,
                'result': result,
                'validation': validation,
                'chart_generated': result.get('chart_info', {}).get('generated', False),
                'llm_response_quality': result.get('llm_analysis', {}).get('response_quality', 'unknown'),
                'overall_quality_score': result.get('performance_metrics', {}).get('overall_quality_score', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Breakout scenario test failed: {e}")
            return {
                'test_name': 'breakout_scenario',
                'success': False,
                'error': str(e),
                'execution_time': 0
            }
    
    async def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling with invalid data."""
        logger.info("🚨 Testing error handling...")
        
        error_tests = []
        
        # Test 1: Empty data
        try:
            result = await self.agent.analyze_market_structure(
                stock_data={},
                analysis_data={},
                symbol="ERRORTEST1",
                scenario_description="Empty Data Test"
            )
            error_tests.append({
                'test': 'empty_data',
                'handled_gracefully': not result.get('success', True),
                'error_message': result.get('error', 'No error message')
            })
        except Exception as e:
            error_tests.append({
                'test': 'empty_data',
                'handled_gracefully': True,
                'error_message': str(e)
            })
        
        # Test 2: Malformed data
        try:
            malformed_data = {'invalid': 'structure'}
            result = await self.agent.analyze_market_structure(
                stock_data=malformed_data,
                analysis_data=malformed_data,
                symbol="ERRORTEST2",
                scenario_description="Malformed Data Test"
            )
            error_tests.append({
                'test': 'malformed_data',
                'handled_gracefully': not result.get('success', True),
                'error_message': result.get('error', 'No error message')
            })
        except Exception as e:
            error_tests.append({
                'test': 'malformed_data',
                'handled_gracefully': True,
                'error_message': str(e)
            })
        
        # Test 3: Missing required fields
        try:
            incomplete_stock_data = {'prices': [100, 101, 102]}  # Missing volumes, timestamps
            incomplete_analysis_data = {'swing_points': {}}  # Missing other required fields
            
            result = await self.agent.analyze_market_structure(
                stock_data=incomplete_stock_data,
                analysis_data=incomplete_analysis_data,
                symbol="ERRORTEST3",
                scenario_description="Incomplete Data Test"
            )
            error_tests.append({
                'test': 'incomplete_data',
                'handled_gracefully': True,  # Should still work with fallbacks
                'success': result.get('success', False),
                'error_message': result.get('error', 'No error')
            })
        except Exception as e:
            error_tests.append({
                'test': 'incomplete_data',
                'handled_gracefully': True,
                'error_message': str(e)
            })
        
        return {
            'test_name': 'error_handling',
            'error_tests': error_tests,
            'total_error_tests': len(error_tests),
            'gracefully_handled': sum(1 for test in error_tests if test.get('handled_gracefully', False))
        }
    
    async def _test_performance_validation(self) -> Dict[str, Any]:
        """Test performance characteristics."""
        logger.info("⚡ Testing performance validation...")
        
        # Create standard test data
        stock_data, analysis_data = self._create_strong_uptrend_data()
        
        # Multiple runs for performance measurement
        execution_times = []
        success_count = 0
        total_runs = 3
        
        for i in range(total_runs):
            try:
                start_time = time.time()
                result = await self.agent.analyze_market_structure(
                    stock_data=stock_data,
                    analysis_data=analysis_data,
                    symbol=f"PERFTEST{i+1}",
                    scenario_description=f"Performance Test {i+1}"
                )
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
                
                if result.get('success'):
                    success_count += 1
                    
            except Exception as e:
                logger.error(f"Performance test run {i+1} failed: {e}")
                execution_times.append(float('inf'))
        
        # Calculate performance metrics
        valid_times = [t for t in execution_times if t != float('inf')]
        avg_time = sum(valid_times) / len(valid_times) if valid_times else 0
        min_time = min(valid_times) if valid_times else 0
        max_time = max(valid_times) if valid_times else 0
        
        return {
            'test_name': 'performance_validation',
            'total_runs': total_runs,
            'successful_runs': success_count,
            'success_rate': success_count / total_runs,
            'execution_times': execution_times,
            'avg_execution_time': avg_time,
            'min_execution_time': min_time,
            'max_execution_time': max_time,
            'performance_acceptable': avg_time < 60  # Should complete within 60 seconds
        }
    
    def _create_strong_uptrend_data(self) -> tuple:
        """Create data for strong uptrend scenario."""
        # Strong uptrend with clear higher highs and higher lows
        prices = [100, 102, 104, 106, 109, 111, 114, 117, 119, 122, 125, 128, 130, 133, 136]
        volumes = [1000, 1200, 1100, 1500, 1800, 1600, 2000, 2200, 1900, 2500, 2800, 2400, 3000, 3200, 2900]
        timestamps = [(datetime.now() - timedelta(days=len(prices)-i)).isoformat() for i in range(len(prices))]
        
        stock_data = {
            'prices': prices,
            'volumes': volumes,
            'timestamps': timestamps,
            'symbol': 'UPTEST'
        }
        
        analysis_data = {
            'swing_points': {
                'total_swings': 8,
                'swing_highs': [
                    {'price': 106, 'index': 3, 'strength': 3},
                    {'price': 114, 'index': 6, 'strength': 4},
                    {'price': 125, 'index': 10, 'strength': 5},
                    {'price': 136, 'index': 14, 'strength': 4}
                ],
                'swing_lows': [
                    {'price': 100, 'index': 0, 'strength': 3},
                    {'price': 104, 'index': 2, 'strength': 2},
                    {'price': 111, 'index': 5, 'strength': 3},
                    {'price': 119, 'index': 8, 'strength': 2}
                ],
                'swing_density': 0.533,
                'quality_score': 88
            },
            'bos_choch_analysis': {
                'bos_events': [
                    {'type': 'bullish_bos', 'price': 107, 'index': 4, 'strength': 'strong'},
                    {'type': 'bullish_bos', 'price': 120, 'index': 9, 'strength': 'strong'},
                    {'type': 'bullish_bos', 'price': 131, 'index': 12, 'strength': 'strong'}
                ],
                'choch_events': [],
                'structural_bias': 'strongly_bullish',
                'recent_break_type': 'bullish_bos',
                'total_breaks': 3
            },
            'trend_analysis': {
                'trend_direction': 'strong_uptrend',
                'trend_strength': 'very_strong',
                'trend_quality': 'excellent',
                'structure_score': 92,
                'higher_highs_present': True,
                'higher_lows_present': True
            },
            'key_levels': {
                'nearest_resistance': {'level': 140, 'strength': 'medium', 'touches': 0},
                'nearest_support': {'level': 130, 'strength': 'strong', 'touches': 2},
                'current_price': 136,
                'levels': [
                    {'level': 100, 'type': 'support', 'strength': 'strong'},
                    {'level': 125, 'type': 'support', 'strength': 'medium'},
                    {'level': 130, 'type': 'support', 'strength': 'strong'},
                    {'level': 140, 'type': 'resistance', 'strength': 'medium'}
                ]
            },
            'volume_analysis': {
                'volume_trend': 'increasing',
                'correlation_strength': 'strong',
                'average_volume': 2000,
                'recent_pattern': 'confirming_trend'
            },
            'market_regime': {
                'regime': 'trending',
                'confidence': 0.91,
                'volatility': 'moderate',
                'trend_strength': 'very_strong'
            }
        }
        
        return stock_data, analysis_data
    
    def _create_clear_downtrend_data(self) -> tuple:
        """Create data for clear downtrend scenario."""
        # Clear downtrend with lower highs and lower lows
        prices = [150, 148, 145, 143, 140, 138, 134, 131, 129, 125, 122, 119, 116, 113, 110]
        volumes = [1500, 1800, 2000, 2200, 1900, 2500, 2800, 2600, 3000, 3200, 2900, 3500, 3800, 3600, 4000]
        timestamps = [(datetime.now() - timedelta(days=len(prices)-i)).isoformat() for i in range(len(prices))]
        
        stock_data = {
            'prices': prices,
            'volumes': volumes,
            'timestamps': timestamps,
            'symbol': 'DOWNTEST'
        }
        
        analysis_data = {
            'swing_points': {
                'total_swings': 7,
                'swing_highs': [
                    {'price': 150, 'index': 0, 'strength': 4},
                    {'price': 143, 'index': 3, 'strength': 3},
                    {'price': 134, 'index': 6, 'strength': 4},
                    {'price': 125, 'index': 9, 'strength': 3}
                ],
                'swing_lows': [
                    {'price': 145, 'index': 2, 'strength': 2},
                    {'price': 138, 'index': 5, 'strength': 3},
                    {'price': 129, 'index': 8, 'strength': 2},
                    {'price': 110, 'index': 14, 'strength': 4}
                ],
                'swing_density': 0.467,
                'quality_score': 85
            },
            'bos_choch_analysis': {
                'bos_events': [
                    {'type': 'bearish_bos', 'price': 142, 'index': 4, 'strength': 'strong'},
                    {'type': 'bearish_bos', 'price': 130, 'index': 8, 'strength': 'strong'},
                    {'type': 'bearish_bos', 'price': 115, 'index': 12, 'strength': 'very_strong'}
                ],
                'choch_events': [
                    {'type': 'bearish_choch', 'price': 147, 'index': 1, 'strength': 'medium'}
                ],
                'structural_bias': 'strongly_bearish',
                'recent_break_type': 'bearish_bos',
                'total_breaks': 4
            },
            'trend_analysis': {
                'trend_direction': 'strong_downtrend',
                'trend_strength': 'very_strong',
                'trend_quality': 'excellent',
                'structure_score': 89,
                'lower_highs_present': True,
                'lower_lows_present': True
            },
            'key_levels': {
                'nearest_resistance': {'level': 120, 'strength': 'strong', 'touches': 1},
                'nearest_support': {'level': 105, 'strength': 'medium', 'touches': 0},
                'current_price': 110,
                'levels': [
                    {'level': 150, 'type': 'resistance', 'strength': 'very_strong'},
                    {'level': 134, 'type': 'resistance', 'strength': 'strong'},
                    {'level': 120, 'type': 'resistance', 'strength': 'strong'},
                    {'level': 105, 'type': 'support', 'strength': 'medium'}
                ]
            },
            'volume_analysis': {
                'volume_trend': 'increasing',
                'correlation_strength': 'strong',
                'average_volume': 2800,
                'recent_pattern': 'selling_pressure'
            },
            'market_regime': {
                'regime': 'trending',
                'confidence': 0.87,
                'volatility': 'high',
                'trend_strength': 'very_strong'
            }
        }
        
        return stock_data, analysis_data
    
    def _create_sideways_consolidation_data(self) -> tuple:
        """Create data for sideways consolidation scenario."""
        # Sideways movement between support and resistance
        prices = [120, 122, 118, 124, 119, 123, 121, 117, 125, 120, 122, 118, 124, 121, 119]
        volumes = [800, 900, 1200, 700, 1100, 600, 850, 1300, 500, 900, 750, 1100, 650, 800, 950]
        timestamps = [(datetime.now() - timedelta(days=len(prices)-i)).isoformat() for i in range(len(prices))]
        
        stock_data = {
            'prices': prices,
            'volumes': volumes,
            'timestamps': timestamps,
            'symbol': 'SIDETEST'
        }
        
        analysis_data = {
            'swing_points': {
                'total_swings': 10,
                'swing_highs': [
                    {'price': 122, 'index': 1, 'strength': 2},
                    {'price': 124, 'index': 3, 'strength': 3},
                    {'price': 123, 'index': 5, 'strength': 2},
                    {'price': 125, 'index': 8, 'strength': 4},
                    {'price': 124, 'index': 12, 'strength': 3}
                ],
                'swing_lows': [
                    {'price': 118, 'index': 2, 'strength': 3},
                    {'price': 119, 'index': 4, 'strength': 2},
                    {'price': 117, 'index': 7, 'strength': 4},
                    {'price': 118, 'index': 11, 'strength': 3},
                    {'price': 119, 'index': 14, 'strength': 2}
                ],
                'swing_density': 0.667,
                'quality_score': 72
            },
            'bos_choch_analysis': {
                'bos_events': [],
                'choch_events': [
                    {'type': 'bullish_choch', 'price': 121, 'index': 6, 'strength': 'weak'},
                    {'type': 'bearish_choch', 'price': 120, 'index': 9, 'strength': 'weak'}
                ],
                'structural_bias': 'neutral',
                'recent_break_type': 'none',
                'total_breaks': 2
            },
            'trend_analysis': {
                'trend_direction': 'sideways',
                'trend_strength': 'weak',
                'trend_quality': 'fair',
                'structure_score': 45,
                'range_bound': True,
                'consolidation_pattern': True
            },
            'key_levels': {
                'nearest_resistance': {'level': 125, 'strength': 'strong', 'touches': 4},
                'nearest_support': {'level': 117, 'strength': 'strong', 'touches': 3},
                'current_price': 119,
                'levels': [
                    {'level': 125, 'type': 'resistance', 'strength': 'strong'},
                    {'level': 122, 'type': 'resistance', 'strength': 'medium'},
                    {'level': 120, 'type': 'neutral', 'strength': 'weak'},
                    {'level': 118, 'type': 'support', 'strength': 'medium'},
                    {'level': 117, 'type': 'support', 'strength': 'strong'}
                ]
            },
            'volume_analysis': {
                'volume_trend': 'decreasing',
                'correlation_strength': 'weak',
                'average_volume': 850,
                'recent_pattern': 'consolidating'
            },
            'market_regime': {
                'regime': 'consolidation',
                'confidence': 0.78,
                'volatility': 'low',
                'trend_strength': 'very_weak'
            }
        }
        
        return stock_data, analysis_data
    
    def _create_volatile_market_data(self) -> tuple:
        """Create data for volatile market scenario."""
        # High volatility with erratic price movements
        prices = [100, 105, 98, 110, 95, 108, 92, 112, 88, 115, 90, 118, 85, 120, 82]
        volumes = [2000, 3500, 4000, 2800, 4500, 3200, 5000, 3800, 5500, 4200, 5200, 4500, 6000, 4800, 6200]
        timestamps = [(datetime.now() - timedelta(days=len(prices)-i)).isoformat() for i in range(len(prices))]
        
        stock_data = {
            'prices': prices,
            'volumes': volumes,
            'timestamps': timestamps,
            'symbol': 'VOLTEST'
        }
        
        analysis_data = {
            'swing_points': {
                'total_swings': 12,
                'swing_highs': [
                    {'price': 105, 'index': 1, 'strength': 2},
                    {'price': 110, 'index': 3, 'strength': 3},
                    {'price': 108, 'index': 5, 'strength': 2},
                    {'price': 112, 'index': 7, 'strength': 3},
                    {'price': 115, 'index': 9, 'strength': 4},
                    {'price': 118, 'index': 11, 'strength': 4},
                    {'price': 120, 'index': 13, 'strength': 5}
                ],
                'swing_lows': [
                    {'price': 98, 'index': 2, 'strength': 2},
                    {'price': 95, 'index': 4, 'strength': 3},
                    {'price': 92, 'index': 6, 'strength': 3},
                    {'price': 88, 'index': 8, 'strength': 4},
                    {'price': 90, 'index': 10, 'strength': 2},
                    {'price': 85, 'index': 12, 'strength': 4},
                    {'price': 82, 'index': 14, 'strength': 5}
                ],
                'swing_density': 0.8,
                'quality_score': 65
            },
            'bos_choch_analysis': {
                'bos_events': [
                    {'type': 'bullish_bos', 'price': 101, 'index': 1, 'strength': 'weak'},
                    {'type': 'bearish_bos', 'price': 99, 'index': 2, 'strength': 'weak'},
                    {'type': 'bullish_bos', 'price': 109, 'index': 5, 'strength': 'medium'},
                    {'type': 'bearish_bos', 'price': 94, 'index': 6, 'strength': 'medium'}
                ],
                'choch_events': [
                    {'type': 'bullish_choch', 'price': 103, 'index': 7, 'strength': 'medium'},
                    {'type': 'bearish_choch', 'price': 91, 'index': 10, 'strength': 'medium'}
                ],
                'structural_bias': 'neutral',
                'recent_break_type': 'mixed',
                'total_breaks': 6
            },
            'trend_analysis': {
                'trend_direction': 'volatile',
                'trend_strength': 'weak',
                'trend_quality': 'poor',
                'structure_score': 35,
                'high_volatility': True,
                'trend_clarity': 'very_unclear'
            },
            'key_levels': {
                'nearest_resistance': {'level': 125, 'strength': 'weak', 'touches': 0},
                'nearest_support': {'level': 80, 'strength': 'medium', 'touches': 1},
                'current_price': 82,
                'levels': [
                    {'level': 120, 'type': 'resistance', 'strength': 'medium'},
                    {'level': 110, 'type': 'resistance', 'strength': 'weak'},
                    {'level': 100, 'type': 'neutral', 'strength': 'weak'},
                    {'level': 90, 'type': 'support', 'strength': 'weak'},
                    {'level': 80, 'type': 'support', 'strength': 'medium'}
                ]
            },
            'volume_analysis': {
                'volume_trend': 'irregular',
                'correlation_strength': 'medium',
                'average_volume': 4200,
                'recent_pattern': 'volatile_spikes'
            },
            'market_regime': {
                'regime': 'volatile',
                'confidence': 0.85,
                'volatility': 'very_high',
                'trend_strength': 'very_weak'
            }
        }
        
        return stock_data, analysis_data
    
    def _create_breakout_scenario_data(self) -> tuple:
        """Create data for breakout scenario."""
        # Consolidation followed by breakout
        prices = [50, 52, 49, 53, 48, 54, 47, 55, 46, 56, 58, 62, 65, 68, 72]
        volumes = [1000, 1200, 1100, 1300, 1400, 1500, 1600, 1800, 2000, 2200, 3500, 4000, 4500, 5000, 5500]
        timestamps = [(datetime.now() - timedelta(days=len(prices)-i)).isoformat() for i in range(len(prices))]
        
        stock_data = {
            'prices': prices,
            'volumes': volumes,
            'timestamps': timestamps,
            'symbol': 'BREAKTEST'
        }
        
        analysis_data = {
            'swing_points': {
                'total_swings': 9,
                'swing_highs': [
                    {'price': 52, 'index': 1, 'strength': 2},
                    {'price': 53, 'index': 3, 'strength': 2},
                    {'price': 54, 'index': 5, 'strength': 3},
                    {'price': 55, 'index': 7, 'strength': 3},
                    {'price': 56, 'index': 9, 'strength': 4},  # Pre-breakout high
                    {'price': 72, 'index': 14, 'strength': 5}  # Post-breakout high
                ],
                'swing_lows': [
                    {'price': 49, 'index': 2, 'strength': 2},
                    {'price': 48, 'index': 4, 'strength': 3},
                    {'price': 47, 'index': 6, 'strength': 3},
                    {'price': 46, 'index': 8, 'strength': 4}   # Final low before breakout
                ],
                'swing_density': 0.6,
                'quality_score': 82
            },
            'bos_choch_analysis': {
                'bos_events': [
                    {'type': 'bullish_bos', 'price': 57, 'index': 10, 'strength': 'very_strong'},  # Breakout point
                    {'type': 'bullish_bos', 'price': 63, 'index': 11, 'strength': 'strong'},
                    {'type': 'bullish_bos', 'price': 69, 'index': 13, 'strength': 'strong'}
                ],
                'choch_events': [],
                'structural_bias': 'strongly_bullish',
                'recent_break_type': 'bullish_bos',
                'total_breaks': 3,
                'breakout_confirmed': True
            },
            'trend_analysis': {
                'trend_direction': 'strong_uptrend',
                'trend_strength': 'very_strong',
                'trend_quality': 'excellent',
                'structure_score': 95,
                'breakout_pattern': True,
                'pre_breakout_consolidation': True
            },
            'key_levels': {
                'nearest_resistance': {'level': 75, 'strength': 'medium', 'touches': 0},
                'nearest_support': {'level': 65, 'strength': 'strong', 'touches': 1},
                'current_price': 72,
                'breakout_level': 56,  # Key breakout level
                'levels': [
                    {'level': 56, 'type': 'support', 'strength': 'very_strong'},  # Former resistance now support
                    {'level': 50, 'type': 'support', 'strength': 'strong'},
                    {'level': 65, 'type': 'support', 'strength': 'strong'},
                    {'level': 75, 'type': 'resistance', 'strength': 'medium'},
                    {'level': 80, 'type': 'resistance', 'strength': 'weak'}
                ]
            },
            'volume_analysis': {
                'volume_trend': 'dramatically_increasing',
                'correlation_strength': 'very_strong',
                'average_volume': 2800,
                'recent_pattern': 'breakout_confirmation',
                'volume_surge': True,
                'breakout_volume_ratio': 4.5  # Volume surge on breakout
            },
            'market_regime': {
                'regime': 'breakout',
                'confidence': 0.94,
                'volatility': 'moderate',
                'trend_strength': 'very_strong'
            }
        }
        
        return stock_data, analysis_data
    
    def _validate_analysis_result(self, result: Dict[str, Any], expected_pattern: str) -> Dict[str, Any]:
        """Validate analysis result against expected pattern."""
        validation = {
            'overall_success': result.get('success', False),
            'chart_generated': result.get('chart_info', {}).get('generated', False),
            'llm_analysis_present': bool(result.get('llm_analysis', {}).get('narrative_analysis')),
            'structured_data_present': bool(result.get('llm_analysis', {}).get('structured_data')),
            'expected_pattern_match': False,
            'quality_score': result.get('performance_metrics', {}).get('overall_quality_score', 0),
            'issues': []
        }
        
        # Check structured data for pattern match
        structured_data = result.get('llm_analysis', {}).get('structured_data', {})
        if structured_data:
            trend_structure = structured_data.get('trend_structure', {})
            market_regime = structured_data.get('market_regime', {})
            
            # Pattern-specific validation
            if expected_pattern == 'uptrend':
                trend_dir = trend_structure.get('trend_direction', '').lower()
                validation['expected_pattern_match'] = 'uptrend' in trend_dir
            elif expected_pattern == 'downtrend':
                trend_dir = trend_structure.get('trend_direction', '').lower()
                validation['expected_pattern_match'] = 'downtrend' in trend_dir
            elif expected_pattern == 'sideways':
                trend_dir = trend_structure.get('trend_direction', '').lower()
                regime = market_regime.get('current_regime', '').lower()
                validation['expected_pattern_match'] = 'sideways' in trend_dir or 'consolidation' in regime
            elif expected_pattern == 'volatile':
                regime = market_regime.get('current_regime', '').lower()
                validation['expected_pattern_match'] = 'volatile' in regime or trend_structure.get('trend_strength_score', 100) < 40
            elif expected_pattern == 'breakout':
                regime = market_regime.get('current_regime', '').lower()
                validation['expected_pattern_match'] = 'breakout' in regime or 'trending' in regime
        
        # Identify issues
        if not validation['overall_success']:
            validation['issues'].append('Analysis failed to complete successfully')
        if not validation['chart_generated']:
            validation['issues'].append('Chart generation failed')
        if not validation['llm_analysis_present']:
            validation['issues'].append('LLM analysis missing or empty')
        if not validation['structured_data_present']:
            validation['issues'].append('Structured data missing from LLM response')
        if not validation['expected_pattern_match']:
            validation['issues'].append(f'Analysis did not identify expected {expected_pattern} pattern')
        if validation['quality_score'] < 50:
            validation['issues'].append('Overall quality score below acceptable threshold')
        
        return validation
    
    def _compile_test_results(self, test_scenarios: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
        """Compile comprehensive test results."""
        successful_tests = sum(1 for test in test_scenarios if test.get('success', False))
        total_tests = len(test_scenarios)
        
        # Calculate average metrics
        execution_times = [test.get('execution_time', 0) for test in test_scenarios if test.get('execution_time', 0) > 0]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        quality_scores = [test.get('overall_quality_score', 0) for test in test_scenarios if test.get('overall_quality_score', 0) > 0]
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Chart generation success rate
        chart_generated_count = sum(1 for test in test_scenarios if test.get('chart_generated', False))
        chart_success_rate = chart_generated_count / total_tests if total_tests > 0 else 0
        
        # LLM response quality distribution
        quality_distribution = {}
        for test in test_scenarios:
            quality = test.get('llm_response_quality', 'unknown')
            quality_distribution[quality] = quality_distribution.get(quality, 0) + 1
        
        return {
            'test_summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
                'total_execution_time': total_time,
                'average_test_execution_time': avg_execution_time,
                'average_quality_score': avg_quality_score
            },
            'detailed_results': test_scenarios,
            'performance_metrics': {
                'chart_generation_success_rate': chart_success_rate,
                'llm_response_quality_distribution': quality_distribution,
                'execution_time_statistics': {
                    'min': min(execution_times) if execution_times else 0,
                    'max': max(execution_times) if execution_times else 0,
                    'avg': avg_execution_time
                }
            },
            'test_passed': successful_tests >= (total_tests * 0.8),  # 80% success rate required
            'timestamp': datetime.now().isoformat()
        }
    
    async def _save_test_report(self, results: Dict[str, Any]) -> str:
        """Save comprehensive test report."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"integrated_agent_test_report_{timestamp}.json"
            report_path = Path("test_reports") / report_filename
            
            # Create directory if it doesn't exist
            report_path.parent.mkdir(exist_ok=True)
            
            # Save report
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Also create a summary report
            summary_filename = f"test_summary_{timestamp}.txt"
            summary_path = report_path.parent / summary_filename
            
            with open(summary_path, 'w') as f:
                f.write("# Integrated Market Structure Agent Test Report\n\n")
                f.write(f"Test Date: {results['timestamp']}\n")
                f.write(f"Total Tests: {results['test_summary']['total_tests']}\n")
                f.write(f"Successful Tests: {results['test_summary']['successful_tests']}\n")
                f.write(f"Success Rate: {results['test_summary']['success_rate']:.1%}\n")
                f.write(f"Average Quality Score: {results['test_summary']['average_quality_score']:.1f}/100\n")
                f.write(f"Chart Generation Success: {results['performance_metrics']['chart_generation_success_rate']:.1%}\n")
                f.write(f"Overall Test Result: {'PASSED' if results['test_passed'] else 'FAILED'}\n\n")
                
                f.write("## Individual Test Results:\n")
                for test in results['detailed_results']:
                    status = "✅ PASSED" if test.get('success', False) else "❌ FAILED"
                    f.write(f"- {test.get('test_name', 'Unknown')}: {status}\n")
                    if 'error' in test:
                        f.write(f"  Error: {test['error']}\n")
            
            logger.info(f"📋 Test report saved: {report_path}")
            logger.info(f"📋 Test summary saved: {summary_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save test report: {e}")
            return ""

# Main test execution
async def main():
    """Run the comprehensive test suite."""
    logger.info("🚀 Starting Integrated Market Structure Agent Test Suite")
    
    # Initialize test suite
    test_suite = IntegratedAgentTestSuite()
    
    try:
        # Run comprehensive tests
        results = await test_suite.run_comprehensive_tests()
        
        # Display results summary
        print("\n" + "="*60)
        print("INTEGRATED MARKET STRUCTURE AGENT TEST RESULTS")
        print("="*60)
        print(f"📊 Total Tests: {results['test_summary']['total_tests']}")
        print(f"✅ Successful: {results['test_summary']['successful_tests']}")
        print(f"📈 Success Rate: {results['test_summary']['success_rate']:.1%}")
        print(f"⭐ Average Quality: {results['test_summary']['average_quality_score']:.1f}/100")
        print(f"📊 Chart Success: {results['performance_metrics']['chart_generation_success_rate']:.1%}")
        print(f"⏱️  Average Time: {results['test_summary']['average_test_execution_time']:.2f}s")
        print(f"🎯 Overall Result: {'PASSED' if results['test_passed'] else 'FAILED'}")
        print(f"📋 Report: {results.get('report_path', 'Not saved')}")
        print("="*60)
        
        # Individual test results
        print("\n📋 Individual Test Results:")
        for test in results['detailed_results']:
            status = "✅" if test.get('success', False) else "❌"
            name = test.get('test_name', 'Unknown').replace('_', ' ').title()
            time_taken = test.get('execution_time', 0)
            quality = test.get('overall_quality_score', 0)
            
            print(f"{status} {name}")
            if test.get('success', False):
                print(f"   ⏱️  {time_taken:.2f}s | ⭐ {quality}/100")
            else:
                error = test.get('error', 'Unknown error')
                print(f"   ❌ {error}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        logger.error(traceback.format_exc())
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    # Run the test suite
    test_results = asyncio.run(main())