#!/usr/bin/env python3
"""
Test script to demonstrate the async optimization of the Gemini client.
This script shows the performance improvements from parallel execution of independent chart analyses.
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
            'comprehensive_overview': 2.0,
            'volume_analysis': 3.0,
            'reversal_patterns': 2.5,
            'continuation_levels': 2.0,
            'final_decision': 1.5
        }
    
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
    
    async def build_indicators_summary(self, symbol, indicators, period, interval, knowledge_context=None):
        """Simulate indicator summary analysis"""
        await asyncio.sleep(1.0)  # Simulate indicator analysis time
        return "Indicator summary", {"trend": "bullish", "confidence": 75}
    
    def call_llm_with_code_execution(self, prompt):
        """Simulate final decision analysis"""
        time.sleep(self.analysis_times['final_decision'])
        return '{"signal": "buy", "confidence": 80}', [], []

async def test_sequential_vs_parallel():
    """Compare sequential vs parallel execution"""
    
    print("=== Testing Sequential vs Parallel Execution ===\n")
    
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
    
    # Test 2: Parallel execution (new way)
    print("Test 2: Parallel Execution (New Method)")
    
    start_time = time.time()
    
    # Simulate parallel execution
    ind_summary_md, ind_json = await mock_client.build_indicators_summary("RELIANCE", {}, 30, "1D")
    
    # Parallel chart analysis
    chart_tasks = [
        mock_client.analyze_comprehensive_overview(b"mock_data"),
        mock_client.analyze_volume_comprehensive([b"mock_data", b"mock_data", b"mock_data"]),
        mock_client.analyze_reversal_patterns([b"mock_data", b"mock_data"]),
        mock_client.analyze_continuation_levels([b"mock_data", b"mock_data"])
    ]
    
    chart_results = await asyncio.gather(*chart_tasks)
    
    # Final decision
    final_result = mock_client.call_llm_with_code_execution("mock_prompt")
    
    parallel_time = time.time() - start_time
    print(f"✅ Parallel execution completed in {parallel_time:.2f} seconds")
    print(f"   Results: {len(chart_results)} chart analyses")
    print()
    
    # Performance comparison
    print("=== Performance Comparison ===")
    print(f"Sequential time: {sequential_time:.2f} seconds")
    print(f"Parallel time: {parallel_time:.2f} seconds")
    print(f"Speedup: {sequential_time / parallel_time:.2f}x")
    print(f"Time saved: {sequential_time - parallel_time:.2f} seconds ({((sequential_time - parallel_time) / sequential_time * 100):.1f}%)")
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
    
    # Simulate batch analysis
    batch_tasks = []
    for symbol, indicators, chart_paths, period, interval, knowledge_context in stock_analyses:
        # Simulate individual stock analysis
        task = asyncio.create_task(simulate_stock_analysis(mock_client, symbol))
        batch_tasks.append(task)
    
    batch_results = await asyncio.gather(*batch_tasks)
    
    batch_time = time.time() - start_time
    print(f"✅ Batch analysis completed in {batch_time:.2f} seconds")
    print(f"   Analyzed {len(batch_results)} stocks")
    print(f"   Average time per stock: {batch_time / len(stock_analyses):.2f} seconds")
    print()

async def simulate_stock_analysis(client, symbol):
    """Simulate a complete stock analysis"""
    # Indicator summary
    ind_summary_md, ind_json = await client.build_indicators_summary(symbol, {}, 30, "1D")
    
    # Parallel chart analysis
    chart_tasks = [
        client.analyze_comprehensive_overview(b"mock_data"),
        client.analyze_volume_comprehensive([b"mock_data"]),
        client.analyze_reversal_patterns([b"mock_data"]),
        client.analyze_continuation_levels([b"mock_data"])
    ]
    
    chart_results = await asyncio.gather(*chart_tasks)
    
    # Final decision
    final_result = client.call_llm_with_code_execution("mock_prompt")
    
    return {
        'symbol': symbol,
        'status': 'success',
        'chart_analyses': len(chart_results)
    }

async def test_error_handling():
    """Test error handling in parallel execution"""
    
    print("=== Testing Error Handling ===\n")
    
    async def failing_analysis():
        await asyncio.sleep(1.0)
        raise Exception("Simulated API error")
    
    async def successful_analysis():
        await asyncio.sleep(2.0)
        return "Analysis completed successfully"
    
    # Test parallel execution with some failures
    print("Testing parallel execution with mixed success/failure...")
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
    print("🚀 Gemini Client Async Optimization Test")
    print("=" * 50)
    
    # Test sequential vs parallel execution
    await test_sequential_vs_parallel()
    
    # Test batch analysis
    await test_batch_analysis()
    
    # Test error handling
    await test_error_handling()
    
    print("✅ All tests completed successfully!")
    print("\nKey Benefits of Async Optimization:")
    print("1. Parallel chart analysis reduces total analysis time")
    print("2. Batch processing allows multiple stocks to be analyzed simultaneously")
    print("3. Error handling ensures one failure doesn't stop other analyses")
    print("4. Better resource utilization and improved user experience")

if __name__ == "__main__":
    asyncio.run(main()) 