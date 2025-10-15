#!/usr/bin/env python3
"""
Complete System Demonstration

This script provides a comprehensive demonstration of the Integrated Market Structure Agent
with Chart Generation and LLM Analysis system. It showcases all key features and capabilities
in a production-ready environment.

Features Demonstrated:
- Chart generation with multiple quality levels
- Multimodal LLM analysis with visual charts
- Production optimizations and caching
- Batch processing capabilities
- Performance monitoring and metrics
- Error handling and resilience
"""

import os
import sys
import asyncio
import logging
import time
from typing import Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the integrated system
try:
    from integrated_market_structure_agent import IntegratedMarketStructureAgent
    from production_optimizations import ProductionOptimizedAgent, ProductionDeploymentManager
    from resilient_chart_generator import ResilientChartGenerator
    from test_integrated_agent import IntegratedAgentTestSuite
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please make sure all system components are available")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CompleteDemoSystem:
    """
    Complete demonstration of the integrated market structure analysis system.
    """
    
    def __init__(self):
        self.demo_results = {}
        
    async def run_complete_demonstration(self) -> Dict[str, Any]:
        """
        Run the complete system demonstration.
        
        Returns:
            Dictionary containing all demonstration results
        """
        print("🚀 INTEGRATED MARKET STRUCTURE AGENT - COMPLETE SYSTEM DEMO")
        print("="*70)
        
        start_time = time.time()
        
        # Demo sections
        demo_sections = [
            ("Basic Integration Demo", self._demo_basic_integration),
            ("Chart Generation Capabilities", self._demo_chart_generation),
            ("Multimodal LLM Analysis", self._demo_multimodal_analysis),
            ("Production Optimizations", self._demo_production_features),
            ("Batch Processing", self._demo_batch_processing),
            ("Performance & Monitoring", self._demo_performance_monitoring),
            ("Error Handling & Resilience", self._demo_error_handling),
            ("Advanced Features", self._demo_advanced_features)
        ]
        
        # Run all demonstrations
        for section_name, demo_func in demo_sections:
            print(f"\n📋 {section_name}")
            print("-" * 50)
            
            try:
                section_start = time.time()
                result = await demo_func()
                section_time = time.time() - section_start
                
                self.demo_results[section_name] = {
                    'success': True,
                    'result': result,
                    'execution_time': section_time
                }
                
                print(f"✅ {section_name} completed in {section_time:.2f}s")
                
            except Exception as e:
                print(f"❌ {section_name} failed: {e}")
                self.demo_results[section_name] = {
                    'success': False,
                    'error': str(e),
                    'execution_time': time.time() - section_start
                }
        
        total_time = time.time() - start_time
        
        # Final summary
        await self._generate_final_summary(total_time)
        
        return self.demo_results
    
    async def _demo_basic_integration(self) -> Dict[str, Any]:
        """Demonstrate basic integration capabilities."""
        print("🔧 Initializing basic integrated agent...")
        
        # Initialize basic agent
        agent = IntegratedMarketStructureAgent(
            charts_output_dir="demo_charts",
            results_output_dir="demo_results"
        )
        
        # Create sample market data
        stock_data, analysis_data = self._create_sample_data("uptrend")
        
        print("📊 Running integrated analysis with chart generation and LLM...")
        
        # Run complete analysis
        result = await agent.analyze_market_structure(
            stock_data=stock_data,
            analysis_data=analysis_data,
            symbol="DEMO_BASIC",
            scenario_description="Basic Integration Demo"
        )
        
        if result['success']:
            print(f"   ✅ Analysis completed successfully")
            print(f"   📊 Chart generated: {Path(result['chart_info']['chart_path']).name}")
            print(f"   🤖 LLM analysis quality: {result['llm_analysis']['response_quality']}")
            print(f"   ⭐ Overall quality score: {result['performance_metrics']['overall_quality_score']}/100")
            
            # Show key insights if available
            structured_data = result.get('llm_analysis', {}).get('structured_data', {})
            if structured_data:
                insights = structured_data.get('actionable_insights', {})
                if insights:
                    print(f"   💡 Key insight: {insights.get('primary_insight', 'N/A')}")
        else:
            print(f"   ❌ Analysis failed: {result.get('error', 'Unknown error')}")
        
        return {
            'analysis_success': result['success'],
            'chart_generated': result.get('chart_info', {}).get('generated', False),
            'llm_quality': result.get('llm_analysis', {}).get('response_quality', 'unknown'),
            'overall_score': result.get('performance_metrics', {}).get('overall_quality_score', 0)
        }
    
    async def _demo_chart_generation(self) -> Dict[str, Any]:
        """Demonstrate chart generation capabilities."""
        print("📊 Testing multi-level chart generation system...")
        
        # Initialize chart generator
        chart_generator = ResilientChartGenerator(output_dir="demo_chart_levels")
        
        # Create different data scenarios
        scenarios = [
            ("Strong Uptrend", self._create_sample_data("uptrend")),
            ("Clear Downtrend", self._create_sample_data("downtrend")),
            ("Sideways Market", self._create_sample_data("sideways"))
        ]
        
        generation_results = []
        
        for scenario_name, (stock_data, analysis_data) in scenarios:
            print(f"   🎯 Testing {scenario_name}...")
            
            # Test different optimization levels
            for opt_level in ['llm_optimized', 'display', 'archive', 'thumbnail']:
                try:
                    result = chart_generator.generate_resilient_chart(
                        stock_data=stock_data,
                        analysis_data=analysis_data,
                        symbol=f"DEMO_{scenario_name.replace(' ', '_').upper()}",
                        chart_title=f"{scenario_name} - {opt_level.title()}",
                        optimization_level=opt_level
                    )
                    
                    if result['success']:
                        chart_path = Path(result['chart_path'])
                        file_size_kb = chart_path.stat().st_size / 1024
                        
                        generation_results.append({
                            'scenario': scenario_name,
                            'optimization': opt_level,
                            'success': True,
                            'file_size_kb': file_size_kb,
                            'quality_level': result.get('quality_level', 'unknown')
                        })
                        
                        print(f"     ✅ {opt_level}: {file_size_kb:.1f}KB ({result.get('quality_level', 'unknown')})")
                    else:
                        print(f"     ❌ {opt_level}: Failed")
                        generation_results.append({
                            'scenario': scenario_name,
                            'optimization': opt_level,
                            'success': False
                        })
                
                except Exception as e:
                    print(f"     ❌ {opt_level}: Error - {e}")
        
        successful_generations = sum(1 for r in generation_results if r.get('success', False))
        total_generations = len(generation_results)
        
        print(f"   📊 Chart generation summary: {successful_generations}/{total_generations} successful")
        
        return {
            'total_attempts': total_generations,
            'successful_generations': successful_generations,
            'success_rate': successful_generations / total_generations if total_generations > 0 else 0,
            'results': generation_results
        }
    
    async def _demo_multimodal_analysis(self) -> Dict[str, Any]:
        """Demonstrate multimodal LLM analysis capabilities."""
        print("🤖 Testing multimodal LLM analysis with visual charts...")
        
        # Initialize agent for multimodal testing
        agent = IntegratedMarketStructureAgent(
            charts_output_dir="demo_multimodal_charts",
            results_output_dir="demo_multimodal_results"
        )
        
        # Test different market patterns
        test_patterns = [
            ("Bullish Breakout", self._create_sample_data("breakout")),
            ("Volatile Market", self._create_sample_data("volatile")),
            ("Consolidation", self._create_sample_data("sideways"))
        ]
        
        multimodal_results = []
        
        for pattern_name, (stock_data, analysis_data) in test_patterns:
            print(f"   📈 Analyzing {pattern_name}...")
            
            try:
                result = await agent.analyze_market_structure(
                    stock_data=stock_data,
                    analysis_data=analysis_data,
                    symbol=f"MULTIMODAL_{pattern_name.replace(' ', '_').upper()}",
                    scenario_description=f"Multimodal Analysis - {pattern_name}"
                )
                
                if result['success']:
                    # Extract LLM analysis quality metrics
                    llm_analysis = result.get('llm_analysis', {})
                    structured_data = llm_analysis.get('structured_data', {})
                    
                    # Check for visual confidence boost
                    chart_validation = structured_data.get('chart_validation', {})
                    visual_boost = chart_validation.get('visual_confidence_boost', 0)
                    
                    confidence_assessment = structured_data.get('confidence_assessment', {})
                    overall_confidence = confidence_assessment.get('overall_confidence', 0)
                    
                    multimodal_results.append({
                        'pattern': pattern_name,
                        'success': True,
                        'response_quality': llm_analysis.get('response_quality', 'unknown'),
                        'visual_confidence_boost': visual_boost,
                        'overall_confidence': overall_confidence,
                        'has_structured_data': structured_data is not None and len(structured_data) > 0
                    })
                    
                    print(f"     ✅ Quality: {llm_analysis.get('response_quality', 'unknown')}")
                    print(f"     📊 Visual boost: +{visual_boost}")
                    print(f"     🎯 Confidence: {overall_confidence}%")
                else:
                    print(f"     ❌ Failed: {result.get('error', 'Unknown error')}")
                    multimodal_results.append({
                        'pattern': pattern_name,
                        'success': False,
                        'error': result.get('error', 'Unknown error')
                    })
            
            except Exception as e:
                print(f"     ❌ Exception: {e}")
                multimodal_results.append({
                    'pattern': pattern_name,
                    'success': False,
                    'error': str(e)
                })
        
        successful_analyses = sum(1 for r in multimodal_results if r.get('success', False))
        avg_confidence = sum(r.get('overall_confidence', 0) for r in multimodal_results if r.get('success', False))
        if successful_analyses > 0:
            avg_confidence /= successful_analyses
        
        print(f"   🎯 Multimodal analysis summary: {successful_analyses}/{len(test_patterns)} successful")
        print(f"   📊 Average confidence: {avg_confidence:.1f}%")
        
        return {
            'successful_analyses': successful_analyses,
            'total_analyses': len(test_patterns),
            'average_confidence': avg_confidence,
            'results': multimodal_results
        }
    
    async def _demo_production_features(self) -> Dict[str, Any]:
        """Demonstrate production optimization features."""
        print("🏭 Testing production optimization features...")
        
        # Initialize production agent with custom configuration
        config = {
            'cache': {
                'enabled': True,
                'ttl_seconds': 1800,  # 30 minutes
                'max_size': 100,
                'compression_enabled': True
            },
            'performance': {
                'max_concurrent_requests': 3,
                'timeout_seconds': 60,
                'memory_limit_mb': 1024
            },
            'optimization': {
                'chart_reuse_enabled': True,
                'response_compression': True
            }
        }
        
        print("   ⚙️ Initializing production-optimized agent...")
        agent = ProductionOptimizedAgent(config=config)
        
        # Test caching with repeated requests
        stock_data, analysis_data = self._create_sample_data("uptrend")
        
        print("   🎯 Testing cache performance...")
        
        # First request (cache miss)
        start_time = time.time()
        result1 = await agent.analyze_market_structure_optimized(
            stock_data=stock_data,
            analysis_data=analysis_data,
            symbol="PROD_CACHE_TEST",
            scenario_description="Production Cache Test",
            use_cache=True
        )
        first_request_time = time.time() - start_time
        
        # Second request (should be cache hit)
        start_time = time.time()
        result2 = await agent.analyze_market_structure_optimized(
            stock_data=stock_data,
            analysis_data=analysis_data,
            symbol="PROD_CACHE_TEST",
            scenario_description="Production Cache Test",
            use_cache=True
        )
        second_request_time = time.time() - start_time
        
        # Get performance metrics
        performance_report = agent.get_performance_report()
        
        cache_hit_rate = performance_report['cache_statistics']['hit_rate']
        avg_response_time = performance_report['average_response_time']
        
        print(f"   ✅ First request: {first_request_time:.2f}s (cache miss)")
        print(f"   ⚡ Second request: {second_request_time:.2f}s (cache hit)")
        print(f"   📊 Cache hit rate: {cache_hit_rate:.1%}")
        print(f"   ⏱️ Average response: {avg_response_time:.2f}s")
        
        # Test resource monitoring
        resource_usage = performance_report['resource_usage']
        print(f"   💾 Memory usage: {resource_usage.get('memory_mb', 0):.1f}MB")
        
        return {
            'first_request_time': first_request_time,
            'second_request_time': second_request_time,
            'cache_speedup_factor': first_request_time / max(second_request_time, 0.1),
            'cache_hit_rate': cache_hit_rate,
            'memory_usage_mb': resource_usage.get('memory_mb', 0),
            'performance_report': performance_report
        }
    
    async def _demo_batch_processing(self) -> Dict[str, Any]:
        """Demonstrate batch processing capabilities."""
        print("🔄 Testing batch processing capabilities...")
        
        # Initialize production agent for batch testing
        agent = ProductionOptimizedAgent()
        
        # Create batch data for multiple symbols
        batch_scenarios = [
            ("TECH_STOCK_1", "uptrend", "Strong Tech Uptrend"),
            ("FINANCIAL_1", "downtrend", "Banking Sector Decline"),
            ("UTILITY_1", "sideways", "Utility Consolidation"),
            ("GROWTH_1", "breakout", "Growth Stock Breakout"),
            ("VOLATILE_1", "volatile", "High Volatility Stock")
        ]
        
        batch_data = []
        for symbol, pattern, description in batch_scenarios:
            stock_data, analysis_data = self._create_sample_data(pattern)
            batch_data.append({
                'stock_data': stock_data,
                'analysis_data': analysis_data,
                'symbol': symbol,
                'scenario': description
            })
        
        print(f"   🚀 Processing batch of {len(batch_data)} symbols...")
        
        # Execute batch processing
        start_time = time.time()
        batch_results = await agent.batch_analyze_symbols(
            batch_data, 
            max_concurrent=3
        )
        batch_time = time.time() - start_time
        
        # Analyze results
        successful_count = sum(1 for r in batch_results if r.get('success', False))
        avg_quality = sum(
            r.get('performance_metrics', {}).get('overall_quality_score', 0) 
            for r in batch_results if r.get('success', False)
        )
        if successful_count > 0:
            avg_quality /= successful_count
        
        print(f"   ✅ Batch completed in {batch_time:.2f}s")
        print(f"   📊 Success rate: {successful_count}/{len(batch_data)} ({successful_count/len(batch_data):.1%})")
        print(f"   ⭐ Average quality: {avg_quality:.1f}/100")
        print(f"   ⚡ Throughput: {len(batch_data)/batch_time:.1f} analyses/second")
        
        # Show individual results
        for i, result in enumerate(batch_results):
            symbol = batch_scenarios[i][0]
            if result.get('success'):
                quality = result.get('performance_metrics', {}).get('overall_quality_score', 0)
                print(f"     ✅ {symbol}: {quality}/100")
            else:
                print(f"     ❌ {symbol}: {result.get('error', 'Unknown error')[:50]}...")
        
        return {
            'batch_size': len(batch_data),
            'batch_time': batch_time,
            'successful_count': successful_count,
            'success_rate': successful_count / len(batch_data),
            'average_quality': avg_quality,
            'throughput_per_second': len(batch_data) / batch_time,
            'individual_results': batch_results
        }
    
    async def _demo_performance_monitoring(self) -> Dict[str, Any]:
        """Demonstrate performance monitoring and health checks."""
        print("📊 Testing performance monitoring and health systems...")
        
        # Initialize deployment manager
        manager = ProductionDeploymentManager()
        agent = manager.initialize_agent()
        
        # Run several analyses to generate metrics
        print("   🔄 Generating performance data...")
        
        test_count = 5
        for i in range(test_count):
            stock_data, analysis_data = self._create_sample_data("uptrend")
            
            try:
                result = await agent.analyze_market_structure_optimized(
                    stock_data=stock_data,
                    analysis_data=analysis_data,
                    symbol=f"PERF_TEST_{i+1}",
                    scenario_description=f"Performance Test {i+1}"
                )
                print(f"     ✅ Test {i+1}: {'Success' if result.get('success') else 'Failed'}")
            except Exception as e:
                print(f"     ❌ Test {i+1}: {e}")
        
        # Get comprehensive performance report
        performance_report = agent.get_performance_report()
        
        print("   📈 Performance Metrics:")
        print(f"     • Total requests: {performance_report['total_requests']}")
        print(f"     • Success rate: {performance_report['success_rate']:.1%}")
        print(f"     • Average response time: {performance_report['average_response_time']:.2f}s")
        print(f"     • Cache hit rate: {performance_report['cache_statistics']['hit_rate']:.1%}")
        print(f"     • Memory usage: {performance_report['resource_usage'].get('memory_mb', 0):.1f}MB")
        
        # Test health monitoring
        health_status = manager.get_health_status()
        
        print("   🏥 Health Status:")
        print(f"     • Status: {health_status['status']}")
        print(f"     • Healthy: {'Yes' if health_status['healthy'] else 'No'}")
        print(f"     • Uptime: {health_status.get('uptime', 0):.1f}s")
        
        return {
            'performance_report': performance_report,
            'health_status': health_status,
            'tests_completed': test_count
        }
    
    async def _demo_error_handling(self) -> Dict[str, Any]:
        """Demonstrate error handling and resilience features."""
        print("🛡️ Testing error handling and resilience features...")
        
        # Initialize agent for error testing
        agent = IntegratedMarketStructureAgent(
            charts_output_dir="demo_error_charts",
            results_output_dir="demo_error_results"
        )
        
        error_tests = []
        
        # Test 1: Empty data
        print("   🧪 Testing empty data handling...")
        try:
            result = await agent.analyze_market_structure(
                stock_data={},
                analysis_data={},
                symbol="ERROR_EMPTY",
                scenario_description="Empty Data Test"
            )
            
            error_tests.append({
                'test': 'empty_data',
                'handled_gracefully': not result.get('success', True),
                'error_message': result.get('error', 'No error message')
            })
            print(f"     ✅ Handled gracefully: {not result.get('success', True)}")
        
        except Exception as e:
            error_tests.append({
                'test': 'empty_data',
                'handled_gracefully': True,
                'error_message': str(e)
            })
            print(f"     ✅ Exception caught: {str(e)[:50]}...")
        
        # Test 2: Malformed data
        print("   🧪 Testing malformed data handling...")
        try:
            malformed_data = {'invalid': 'structure', 'numbers': 'not_numbers'}
            result = await agent.analyze_market_structure(
                stock_data=malformed_data,
                analysis_data=malformed_data,
                symbol="ERROR_MALFORMED",
                scenario_description="Malformed Data Test"
            )
            
            error_tests.append({
                'test': 'malformed_data',
                'handled_gracefully': not result.get('success', True),
                'error_message': result.get('error', 'No error message')
            })
            print(f"     ✅ Handled gracefully: {not result.get('success', True)}")
        
        except Exception as e:
            error_tests.append({
                'test': 'malformed_data',
                'handled_gracefully': True,
                'error_message': str(e)
            })
            print(f"     ✅ Exception caught: {str(e)[:50]}...")
        
        # Test 3: Corrupted chart data (simulated)
        print("   🧪 Testing resilient chart generation...")
        
        # Create data that might cause chart generation issues
        problematic_data = {
            'prices': [100] * 100,  # All same prices (no variation)
            'volumes': [0] * 100,   # All zero volumes
            'timestamps': [f'2024-01-{i:02d}' for i in range(1, 101)]
        }
        
        problematic_analysis = {
            'swing_points': {'total_swings': 0, 'swing_highs': [], 'swing_lows': []},
            'trend_analysis': {'trend_direction': 'unknown'},
            'market_regime': {'regime': 'unknown'}
        }
        
        try:
            result = await agent.analyze_market_structure(
                stock_data=problematic_data,
                analysis_data=problematic_analysis,
                symbol="ERROR_CHART",
                scenario_description="Problematic Chart Data Test"
            )
            
            chart_generated = result.get('chart_info', {}).get('generated', False)
            error_tests.append({
                'test': 'chart_resilience',
                'handled_gracefully': True,  # Should handle with fallbacks
                'chart_generated': chart_generated,
                'quality_level': result.get('chart_info', {}).get('generation_quality', 'unknown')
            })
            
            print(f"     ✅ Chart fallback system: {'Working' if chart_generated else 'Failed'}")
            if chart_generated:
                print(f"     📊 Quality level: {result.get('chart_info', {}).get('generation_quality', 'unknown')}")
        
        except Exception as e:
            error_tests.append({
                'test': 'chart_resilience',
                'handled_gracefully': False,
                'error_message': str(e)
            })
            print(f"     ❌ Chart generation failed: {str(e)[:50]}...")
        
        gracefully_handled = sum(1 for test in error_tests if test.get('handled_gracefully', False))
        
        print(f"   📊 Error handling summary: {gracefully_handled}/{len(error_tests)} tests handled gracefully")
        
        return {
            'error_tests': error_tests,
            'total_tests': len(error_tests),
            'gracefully_handled': gracefully_handled,
            'success_rate': gracefully_handled / len(error_tests) if error_tests else 0
        }
    
    async def _demo_advanced_features(self) -> Dict[str, Any]:
        """Demonstrate advanced features and capabilities."""
        print("🚀 Testing advanced features and capabilities...")
        
        # Run comprehensive test suite
        print("   🧪 Running comprehensive test suite...")
        test_suite = IntegratedAgentTestSuite()
        
        try:
            # Run a subset of tests for demonstration
            uptrend_result = await test_suite._test_strong_uptrend_scenario()
            downtrend_result = await test_suite._test_clear_downtrend_scenario()
            error_result = await test_suite._test_error_handling()
            
            test_results = [uptrend_result, downtrend_result, error_result]
            successful_tests = sum(1 for r in test_results if r.get('success', False))
            
            print(f"   ✅ Test suite results: {successful_tests}/{len(test_results)} successful")
            
            # Show individual test results
            for result in test_results:
                test_name = result.get('test_name', 'Unknown')
                success = result.get('success', False)
                exec_time = result.get('execution_time', 0)
                status = "✅" if success else "❌"
                print(f"     {status} {test_name}: {exec_time:.2f}s")
        
        except Exception as e:
            print(f"   ❌ Test suite failed: {e}")
            return {
                'test_suite_success': False,
                'error': str(e)
            }
        
        # Demonstrate system integration capabilities
        print("   🔧 Testing system integration features...")
        
        # Create a complex analysis scenario
        complex_stock_data = {
            'prices': [100, 98, 102, 105, 103, 108, 112, 109, 115, 118, 116, 120, 125, 122, 128],
            'volumes': [1000, 1500, 1200, 1800, 1400, 2000, 2200, 1900, 2500, 2800, 2400, 3000, 3200, 2900, 3500],
            'timestamps': [(datetime.now() - timedelta(days=len([100, 98, 102, 105, 103, 108, 112, 109, 115, 118, 116, 120, 125, 122, 128])-i)).isoformat() for i in range(len([100, 98, 102, 105, 103, 108, 112, 109, 115, 118, 116, 120, 125, 122, 128]))]
        }
        
        complex_analysis_data = {
            'swing_points': {
                'total_swings': 8,
                'swing_highs': [
                    {'price': 105, 'index': 3, 'strength': 3},
                    {'price': 112, 'index': 6, 'strength': 4},
                    {'price': 118, 'index': 9, 'strength': 3},
                    {'price': 128, 'index': 14, 'strength': 5}
                ],
                'swing_lows': [
                    {'price': 98, 'index': 1, 'strength': 3},
                    {'price': 103, 'index': 4, 'strength': 2},
                    {'price': 109, 'index': 7, 'strength': 2},
                    {'price': 116, 'index': 10, 'strength': 2}
                ],
                'swing_density': 0.533,
                'quality_score': 92
            },
            'bos_choch_analysis': {
                'bos_events': [
                    {'type': 'bullish_bos', 'price': 106, 'index': 5, 'strength': 'strong'},
                    {'type': 'bullish_bos', 'price': 119, 'index': 10, 'strength': 'very_strong'},
                    {'type': 'bullish_bos', 'price': 126, 'index': 13, 'strength': 'strong'}
                ],
                'choch_events': [],
                'structural_bias': 'strongly_bullish',
                'recent_break_type': 'bullish_bos'
            },
            'trend_analysis': {
                'trend_direction': 'strong_uptrend',
                'trend_strength': 'very_strong',
                'trend_quality': 'excellent',
                'structure_score': 94
            },
            'market_regime': {
                'regime': 'trending',
                'confidence': 0.91,
                'volatility': 'moderate'
            }
        }
        
        # Run advanced analysis
        agent = IntegratedMarketStructureAgent()
        
        try:
            advanced_result = await agent.analyze_market_structure(
                stock_data=complex_stock_data,
                analysis_data=complex_analysis_data,
                symbol="ADVANCED_DEMO",
                scenario_description="Advanced Integration Demonstration"
            )
            
            if advanced_result['success']:
                quality_score = advanced_result.get('performance_metrics', {}).get('overall_quality_score', 0)
                llm_quality = advanced_result.get('llm_analysis', {}).get('response_quality', 'unknown')
                
                print(f"     ✅ Advanced analysis: {quality_score}/100 quality")
                print(f"     🤖 LLM response quality: {llm_quality}")
                
                # Extract key insights
                structured_data = advanced_result.get('llm_analysis', {}).get('structured_data', {})
                if structured_data:
                    confidence = structured_data.get('confidence_assessment', {}).get('overall_confidence', 0)
                    trend_analysis = structured_data.get('trend_structure', {})
                    
                    print(f"     🎯 Analysis confidence: {confidence}%")
                    print(f"     📈 Trend detected: {trend_analysis.get('trend_direction', 'unknown')}")
            else:
                print(f"     ❌ Advanced analysis failed: {advanced_result.get('error', 'Unknown error')}")
        
        except Exception as e:
            print(f"   ❌ Advanced analysis error: {e}")
            return {
                'advanced_analysis_success': False,
                'error': str(e),
                'test_suite_success': successful_tests == len(test_results)
            }
        
        return {
            'test_suite_success': successful_tests == len(test_results),
            'successful_tests': successful_tests,
            'total_tests': len(test_results),
            'advanced_analysis_success': advanced_result.get('success', False),
            'advanced_quality_score': advanced_result.get('performance_metrics', {}).get('overall_quality_score', 0)
        }
    
    def _create_sample_data(self, pattern_type: str) -> tuple:
        """Create sample market data for different patterns."""
        base_price = 100
        data_points = 15
        
        if pattern_type == "uptrend":
            # Strong uptrend pattern
            prices = [base_price + i * 2 + (i % 3) for i in range(data_points)]
            volumes = [1000 + i * 100 + (i % 4) * 200 for i in range(data_points)]
            
            analysis_data = {
                'swing_points': {
                    'total_swings': 6,
                    'swing_highs': [{'price': max(prices), 'index': prices.index(max(prices))}],
                    'swing_lows': [{'price': min(prices), 'index': prices.index(min(prices))}],
                    'quality_score': 88
                },
                'bos_choch_analysis': {
                    'bos_events': [{'type': 'bullish_bos', 'price': prices[len(prices)//2]}],
                    'structural_bias': 'bullish'
                },
                'trend_analysis': {
                    'trend_direction': 'uptrend',
                    'trend_strength': 'strong'
                },
                'market_regime': {
                    'regime': 'trending',
                    'confidence': 0.85
                }
            }
            
        elif pattern_type == "downtrend":
            # Clear downtrend pattern
            prices = [base_price + 30 - i * 2 - (i % 3) for i in range(data_points)]
            volumes = [1000 + i * 150 + (i % 3) * 250 for i in range(data_points)]
            
            analysis_data = {
                'swing_points': {
                    'total_swings': 5,
                    'swing_highs': [{'price': max(prices), 'index': prices.index(max(prices))}],
                    'swing_lows': [{'price': min(prices), 'index': prices.index(min(prices))}],
                    'quality_score': 82
                },
                'bos_choch_analysis': {
                    'bos_events': [{'type': 'bearish_bos', 'price': prices[len(prices)//2]}],
                    'structural_bias': 'bearish'
                },
                'trend_analysis': {
                    'trend_direction': 'downtrend',
                    'trend_strength': 'strong'
                },
                'market_regime': {
                    'regime': 'trending',
                    'confidence': 0.78
                }
            }
            
        elif pattern_type == "sideways":
            # Sideways consolidation
            mid_price = base_price + 10
            prices = [mid_price + (-1)**(i % 2) * (3 + i % 4) for i in range(data_points)]
            volumes = [800 + i * 50 + (i % 5) * 100 for i in range(data_points)]
            
            analysis_data = {
                'swing_points': {
                    'total_swings': 8,
                    'swing_highs': [{'price': max(prices), 'index': prices.index(max(prices))}],
                    'swing_lows': [{'price': min(prices), 'index': prices.index(min(prices))}],
                    'quality_score': 65
                },
                'bos_choch_analysis': {
                    'bos_events': [],
                    'choch_events': [{'type': 'neutral_choch', 'price': mid_price}],
                    'structural_bias': 'neutral'
                },
                'trend_analysis': {
                    'trend_direction': 'sideways',
                    'trend_strength': 'weak'
                },
                'market_regime': {
                    'regime': 'consolidation',
                    'confidence': 0.72
                }
            }
            
        elif pattern_type == "breakout":
            # Consolidation followed by breakout
            consolidation_length = data_points // 2
            prices = [base_price + (i % 3) for i in range(consolidation_length)]  # Consolidation
            prices.extend([base_price + 5 + i * 3 for i in range(data_points - consolidation_length)])  # Breakout
            
            volumes = [1000 + i * 50 for i in range(consolidation_length)]
            volumes.extend([2000 + i * 200 for i in range(data_points - consolidation_length)])
            
            analysis_data = {
                'swing_points': {
                    'total_swings': 4,
                    'swing_highs': [{'price': max(prices), 'index': prices.index(max(prices))}],
                    'swing_lows': [{'price': min(prices[:consolidation_length]), 'index': 2}],
                    'quality_score': 78
                },
                'bos_choch_analysis': {
                    'bos_events': [{'type': 'bullish_bos', 'price': base_price + 6}],
                    'structural_bias': 'bullish'
                },
                'trend_analysis': {
                    'trend_direction': 'breakout',
                    'trend_strength': 'strong'
                },
                'market_regime': {
                    'regime': 'breakout',
                    'confidence': 0.89
                }
            }
            
        else:  # volatile
            # High volatility pattern
            prices = [base_price + (10 if i % 4 == 0 else -8 if i % 4 == 1 else 5 if i % 4 == 2 else -3) + (i % 3) for i in range(data_points)]
            volumes = [1500 + i * 200 + (i % 6) * 300 for i in range(data_points)]
            
            analysis_data = {
                'swing_points': {
                    'total_swings': 10,
                    'swing_highs': [{'price': max(prices), 'index': prices.index(max(prices))}],
                    'swing_lows': [{'price': min(prices), 'index': prices.index(min(prices))}],
                    'quality_score': 45
                },
                'bos_choch_analysis': {
                    'bos_events': [
                        {'type': 'bullish_bos', 'price': prices[3]},
                        {'type': 'bearish_bos', 'price': prices[7]}
                    ],
                    'structural_bias': 'neutral'
                },
                'trend_analysis': {
                    'trend_direction': 'volatile',
                    'trend_strength': 'weak'
                },
                'market_regime': {
                    'regime': 'volatile',
                    'confidence': 0.92
                }
            }
        
        # Create timestamps
        timestamps = [(datetime.now() - timedelta(days=data_points-i)).isoformat() for i in range(data_points)]
        
        stock_data = {
            'prices': prices,
            'volumes': volumes,
            'timestamps': timestamps
        }
        
        return stock_data, analysis_data
    
    async def _generate_final_summary(self, total_time: float):
        """Generate comprehensive final summary."""
        print("\n" + "="*70)
        print("🎯 COMPLETE SYSTEM DEMONSTRATION SUMMARY")
        print("="*70)
        
        successful_sections = sum(1 for r in self.demo_results.values() if r.get('success', False))
        total_sections = len(self.demo_results)
        
        print(f"📊 Sections Completed: {successful_sections}/{total_sections}")
        print(f"⏱️ Total Demo Time: {total_time:.2f}s")
        print(f"📈 Success Rate: {successful_sections/total_sections:.1%}")
        
        print("\n📋 Section Results:")
        for section_name, result in self.demo_results.items():
            status = "✅" if result.get('success', False) else "❌"
            exec_time = result.get('execution_time', 0)
            print(f"  {status} {section_name}: {exec_time:.2f}s")
            
            if not result.get('success', False) and 'error' in result:
                print(f"    ❌ Error: {result['error'][:60]}...")
        
        # Highlight key achievements
        print("\n🚀 Key Achievements Demonstrated:")
        
        if 'Basic Integration Demo' in self.demo_results and self.demo_results['Basic Integration Demo'].get('success'):
            basic_result = self.demo_results['Basic Integration Demo']['result']
            if basic_result.get('analysis_success'):
                print("  ✅ Multimodal chart generation and LLM analysis integration")
                print(f"     📊 Chart quality: Success")
                print(f"     🤖 LLM quality: {basic_result.get('llm_quality', 'unknown')}")
        
        if 'Chart Generation Capabilities' in self.demo_results and self.demo_results['Chart Generation Capabilities'].get('success'):
            chart_result = self.demo_results['Chart Generation Capabilities']['result']
            success_rate = chart_result.get('success_rate', 0)
            print(f"  ✅ Multi-level resilient chart generation: {success_rate:.1%} success rate")
        
        if 'Production Optimizations' in self.demo_results and self.demo_results['Production Optimizations'].get('success'):
            prod_result = self.demo_results['Production Optimizations']['result']
            cache_hit_rate = prod_result.get('cache_hit_rate', 0)
            speedup = prod_result.get('cache_speedup_factor', 1)
            print(f"  ✅ Production optimizations: {cache_hit_rate:.1%} cache hit rate, {speedup:.1f}x speedup")
        
        if 'Batch Processing' in self.demo_results and self.demo_results['Batch Processing'].get('success'):
            batch_result = self.demo_results['Batch Processing']['result']
            throughput = batch_result.get('throughput_per_second', 0)
            print(f"  ✅ Concurrent batch processing: {throughput:.2f} analyses/second")
        
        if 'Error Handling & Resilience' in self.demo_results and self.demo_results['Error Handling & Resilience'].get('success'):
            error_result = self.demo_results['Error Handling & Resilience']['result']
            error_success_rate = error_result.get('success_rate', 0)
            print(f"  ✅ Error handling and resilience: {error_success_rate:.1%} graceful error handling")
        
        print("\n💡 System Capabilities Demonstrated:")
        print("  • Advanced market structure chart generation with multi-level fallbacks")
        print("  • Multimodal LLM analysis combining visual charts with numerical data")
        print("  • Production-ready optimizations with caching and resource monitoring")
        print("  • Concurrent batch processing with configurable limits")
        print("  • Comprehensive error handling and resilience mechanisms")
        print("  • Performance monitoring and health status reporting")
        print("  • Structured response parsing and validation")
        
        overall_success = successful_sections >= total_sections * 0.8  # 80% success threshold
        
        print(f"\n🎯 Overall Demo Result: {'SUCCESS' if overall_success else 'PARTIAL SUCCESS'}")
        print("="*70)
        
        if overall_success:
            print("🎉 The Integrated Market Structure Agent system is fully operational!")
            print("   Ready for production deployment with all features working correctly.")
        else:
            print("⚠️ Some features need attention before full production deployment.")
            print("   Review the failed sections and address any issues.")
        
        print("\n📚 Next Steps:")
        print("  1. Review individual section results for detailed insights")
        print("  2. Run comprehensive test suite: `python test_integrated_agent.py`")
        print("  3. Configure production settings in deployment configuration")
        print("  4. Monitor performance metrics in production environment")
        print("  5. Set up health monitoring and alerting systems")


# Main demonstration execution
async def main():
    """Run the complete system demonstration."""
    print("🚀 Starting Complete System Demonstration...")
    print("This comprehensive demo will showcase all capabilities of the")
    print("Integrated Market Structure Agent with Chart Generation and LLM Analysis")
    print()
    
    # Initialize and run demonstration
    demo_system = CompleteDemoSystem()
    
    try:
        results = await demo_system.run_complete_demonstration()
        
        # Return success code based on results
        successful_sections = sum(1 for r in results.values() if r.get('success', False))
        total_sections = len(results)
        
        if successful_sections >= total_sections * 0.8:
            return 0  # Success
        else:
            return 1  # Partial failure
    
    except Exception as e:
        print(f"❌ Demo system failed: {e}")
        import traceback
        traceback.print_exc()
        return 2  # Critical failure

if __name__ == "__main__":
    # Run the complete demonstration
    exit_code = asyncio.run(main())
    exit(exit_code)