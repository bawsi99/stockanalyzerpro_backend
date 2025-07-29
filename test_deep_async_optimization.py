#!/usr/bin/env python3
"""
Test script to demonstrate the deep async optimization of the Gemini client.
This script shows the performance improvements from starting all independent LLM calls immediately.
"""

import asyncio
import time
import sys
import os
from unittest.mock import Mock, patch

# Add the gemini directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'gemini'))

class MockGeminiClient:
    """Mock client to simulate async chart analysis without actual API calls"""
    
    def __init__(self):
        self.analysis_times = {
            'indicator_summary': 3.0,
            'comprehensive_overview': 2.0,
            'volume_analysis': 3.0,
            'reversal_patterns': 2.5,
            'continuation_levels': 2.0,
            'final_decision': 1.5
        }
    
    async def build_indicators_summary(self, symbol, indicators, period, interval, knowledge_context=None):
        """Simulate indicator summary analysis"""
        await asyncio.sleep(self.analysis_times['indicator_summary'])
        return "Indicator summary completed", {"trend": "bullish", "confidence": 75}
    
    async def analyze_comprehensive_overview(self, image_data):
        """Simulate comprehensive overview analysis"""
        await asyncio.sleep(self.analysis_times['comprehensive_overview'])
        return "Comprehensive technical analysis completed successfully."
    
    async def analyze_volume_comprehensive(self, images):
        """Simulate volume analysis"""
        await asyncio.sleep(self.analysis_times['volume_analysis'])
        return "Volume analysis completed successfully."
    
    async def analyze_reversal_patterns(self, images):
        """Simulate reversal pattern analysis"""
        await asyncio.sleep(self.analysis_times['reversal_patterns'])
        return "Reversal pattern analysis completed successfully."
    
    async def analyze_continuation_levels(self, images):
        """Simulate continuation and level analysis"""
        await asyncio.sleep(self.analysis_times['continuation_levels'])
        return "Continuation and level analysis completed successfully."
    
    def call_llm_with_code_execution(self, prompt):
        """Simulate final decision analysis"""
        time.sleep(self.analysis_times['final_decision'])
        return '{"signal": "buy", "confidence": 80}', [], []

async def test_sequential_vs_optimized():
    """Compare sequential vs optimized execution"""
    
    print("=== Testing Sequential vs Optimized Execution ===\n")
    
    # Mock chart paths
    chart_paths = {
        'comparison_chart': 'mock_chart1.png',
        'volume_anomalies': 'mock_chart2.png',
        'price_volume_correlation': 'mock_chart3.png',
        'candlestick_volume': 'mock_chart4.png',
        'divergence': 'mock_chart5.png',
        'double_tops_bottoms': 'mock_chart6.png',
        'triangles_flags': 'mock_chart7.png',
        'support_resistance': 'mock_chart8.png'
    }
    
    # Test 1: Sequential execution (old way)
    print("Test 1: Sequential Execution (Old Method)")
    mock_client = MockGeminiClient()
    
    start_time = time.time()
    
    # Simulate sequential execution
    ind_summary_md, ind_json = await mock_client.build_indicators_summary("RELIANCE", {}, 30, "1D")
    
    # Sequential chart analysis
    comprehensive_result = await mock_client.analyze_comprehensive_overview(b"mock_data")
    volume_result = await mock_client.analyze_volume_comprehensive([b"mock_data", b"mock_data", b"mock_data"])
    reversal_result = await mock_client.analyze_reversal_patterns([b"mock_data", b"mock_data"])
    continuation_result = await mock_client.analyze_continuation_levels([b"mock_data", b"mock_data"])
    
    # Final decision
    final_result = mock_client.call_llm_with_code_execution("mock_prompt")
    
    sequential_time = time.time() - start_time
    print(f"✅ Sequential execution completed in {sequential_time:.2f} seconds")
    print(f"   Results: {len([comprehensive_result, volume_result, reversal_result, continuation_result])} chart analyses")
    print()
    
    # Test 2: Optimized execution (new way)
    print("Test 2: Optimized Execution (New Method)")
    
    start_time = time.time()
    
    # Simulate optimized execution - all independent tasks start immediately
    indicator_task = mock_client.build_indicators_summary("RELIANCE", {}, 30, "1D")
    comprehensive_task = mock_client.analyze_comprehensive_overview(b"mock_data")
    volume_task = mock_client.analyze_volume_comprehensive([b"mock_data", b"mock_data", b"mock_data"])
    reversal_task = mock_client.analyze_reversal_patterns([b"mock_data", b"mock_data"])
    continuation_task = mock_client.analyze_continuation_levels([b"mock_data", b"mock_data"])
    
    # Execute all independent tasks in parallel
    all_results = await asyncio.gather(
        indicator_task, 
        comprehensive_task, 
        volume_task, 
        reversal_task, 
        continuation_task
    )
    
    # Final decision (depends on all previous results)
    final_result = mock_client.call_llm_with_code_execution("mock_prompt")
    
    optimized_time = time.time() - start_time
    print(f"✅ Optimized execution completed in {optimized_time:.2f} seconds")
    print(f"   Results: {len(all_results)} analyses completed in parallel")
    print()
    
    # Performance comparison
    print("=== Performance Comparison ===")
    print(f"Sequential time: {sequential_time:.2f} seconds")
    print(f"Optimized time: {optimized_time:.2f} seconds")
    print(f"Speedup: {sequential_time / optimized_time:.2f}x")
    print(f"Time saved: {sequential_time - optimized_time:.2f} seconds ({((sequential_time - optimized_time) / sequential_time * 100):.1f}%)")
    print()

async def test_batch_analysis():
    """Test batch analysis of multiple stocks"""
    
    print("=== Testing Batch Analysis ===\n")
    
    # Mock data for multiple stocks
    stock_analyses = [
        ("RELIANCE", {}, {"comparison_chart": "mock1.png"}, 30, "1D", ""),
        ("TCS", {}, {"comparison_chart": "mock2.png"}, 30, "1D", ""),
        ("INFY", {}, {"comparison_chart": "mock3.png"}, 30, "1D", ""),
        ("HDFC", {}, {"comparison_chart": "mock4.png"}, 30, "1D", "")
    ]
    
    mock_client = MockGeminiClient()
    
    # Test batch analysis
    print(f"Starting batch analysis of {len(stock_analyses)} stocks...")
    start_time = time.time()
    
    # Simulate batch analysis with optimized parallel execution
    batch_tasks = []
    for symbol, indicators, chart_paths, period, interval, knowledge_context in stock_analyses:
        # Simulate individual stock analysis with all independent tasks in parallel
        task = asyncio.create_task(simulate_optimized_stock_analysis(mock_client, symbol))
        batch_tasks.append(task)
    
    batch_results = await asyncio.gather(*batch_tasks)
    
    batch_time = time.time() - start_time
    print(f"✅ Batch analysis completed in {batch_time:.2f} seconds")
    print(f"   Analyzed {len(batch_results)} stocks")
    print(f"   Average time per stock: {batch_time / len(stock_analyses):.2f} seconds")
    print()

async def simulate_optimized_stock_analysis(client, symbol):
    """Simulate a complete stock analysis with optimized parallel execution"""
    # Start all independent tasks immediately
    indicator_task = client.build_indicators_summary(symbol, {}, 30, "1D")
    comprehensive_task = client.analyze_comprehensive_overview(b"mock_data")
    volume_task = client.analyze_volume_comprehensive([b"mock_data"])
    reversal_task = client.analyze_reversal_patterns([b"mock_data"])
    continuation_task = client.analyze_continuation_levels([b"mock_data"])
    
    # Execute all independent tasks in parallel
    all_results = await asyncio.gather(
        indicator_task, 
        comprehensive_task, 
        volume_task, 
        reversal_task, 
        continuation_task
    )
    
    # Final decision
    final_result = client.call_llm_with_code_execution("mock_prompt")
    
    return {
        'symbol': symbol,
        'status': 'success',
        'chart_analyses': len(all_results)
    }

async def test_error_handling():
    """Test error handling in optimized parallel execution"""
    
    print("=== Testing Error Handling ===\n")
    
    async def failing_analysis():
        await asyncio.sleep(1.0)
        raise Exception("Simulated API error")
    
    async def successful_analysis():
        await asyncio.sleep(2.0)
        return "Analysis completed successfully"
    
    # Test optimized execution with some failures
    print("Testing optimized execution with mixed success/failure...")
    start_time = time.time()
    
    tasks = [
        successful_analysis(),
        failing_analysis(),
        successful_analysis(),
        failing_analysis()
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    execution_time = time.time() - start_time
    print(f"✅ Mixed execution completed in {execution_time:.2f} seconds")
    
    # Count successes and failures
    successes = sum(1 for r in results if not isinstance(r, Exception))
    failures = sum(1 for r in results if isinstance(r, Exception))
    
    print(f"   Successful analyses: {successes}")
    print(f"   Failed analyses: {failures}")
    print(f"   Success rate: {successes / len(results) * 100:.1f}%")
    print()

async def main():
    """Run all tests"""
    print("🚀 Deep Async Optimization Test")
    print("=" * 50)
    
    # Test sequential vs optimized execution
    await test_sequential_vs_optimized()
    
    # Test batch analysis
    await test_batch_analysis()
    
    # Test error handling
    await test_error_handling()
    
    print("✅ All tests completed successfully!")
    print("\nKey Benefits of Deep Async Optimization:")
    print("1. All independent LLM calls start immediately")
    print("2. Indicator summary and chart analyses run in parallel")
    print("3. Maximum utilization of API resources")
    print("4. Significant reduction in total analysis time")
    print("5. Better user experience with faster response times")

if __name__ == "__main__":
    asyncio.run(main()) 