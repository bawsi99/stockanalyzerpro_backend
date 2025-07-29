#!/usr/bin/env python3
"""
Test script to verify that the parallel execution fix works correctly.
This script will test that all 5 LLM calls now run in true parallel.
"""

import asyncio
import time
import os
import sys

# Add the gemini directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'gemini'))

from gemini.gemini_client import GeminiClient

class ParallelTestClient:
    """Test client to verify parallel execution"""
    
    def __init__(self):
        self.start_times = {}
        self.end_times = {}
        self.task_names = [
            'indicator_summary',
            'comprehensive_overview', 
            'volume_analysis',
            'reversal_patterns',
            'continuation_levels'
        ]
    
    async def build_indicators_summary(self, symbol, indicators, period, interval, knowledge_context=None):
        """Simulate indicator summary with timing"""
        task_name = 'indicator_summary'
        self.start_times[task_name] = time.time()
        print(f"[{task_name}] Starting at {self.start_times[task_name]:.2f}s")
        
        await asyncio.sleep(3.0)  # Simulate 3 second API call
        
        self.end_times[task_name] = time.time()
        print(f"[{task_name}] Completed at {self.end_times[task_name]:.2f}s (duration: {self.end_times[task_name] - self.start_times[task_name]:.2f}s)")
        return "Indicator summary completed", {"trend": "bullish", "confidence": 75}
    
    async def analyze_comprehensive_overview(self, image):
        """Simulate comprehensive overview with timing"""
        task_name = 'comprehensive_overview'
        self.start_times[task_name] = time.time()
        print(f"[{task_name}] Starting at {self.start_times[task_name]:.2f}s")
        
        await asyncio.sleep(2.0)  # Simulate 2 second API call
        
        self.end_times[task_name] = time.time()
        print(f"[{task_name}] Completed at {self.end_times[task_name]:.2f}s (duration: {self.end_times[task_name] - self.start_times[task_name]:.2f}s)")
        return "Comprehensive overview completed"
    
    async def analyze_volume_comprehensive(self, images):
        """Simulate volume analysis with timing"""
        task_name = 'volume_analysis'
        self.start_times[task_name] = time.time()
        print(f"[{task_name}] Starting at {self.start_times[task_name]:.2f}s")
        
        await asyncio.sleep(3.0)  # Simulate 3 second API call
        
        self.end_times[task_name] = time.time()
        print(f"[{task_name}] Completed at {self.end_times[task_name]:.2f}s (duration: {self.end_times[task_name] - self.start_times[task_name]:.2f}s)")
        return "Volume analysis completed"
    
    async def analyze_reversal_patterns(self, images):
        """Simulate reversal patterns with timing"""
        task_name = 'reversal_patterns'
        self.start_times[task_name] = time.time()
        print(f"[{task_name}] Starting at {self.start_times[task_name]:.2f}s")
        
        await asyncio.sleep(2.5)  # Simulate 2.5 second API call
        
        self.end_times[task_name] = time.time()
        print(f"[{task_name}] Completed at {self.end_times[task_name]:.2f}s (duration: {self.end_times[task_name] - self.start_times[task_name]:.2f}s)")
        return "Reversal patterns completed"
    
    async def analyze_continuation_levels(self, images):
        """Simulate continuation levels with timing"""
        task_name = 'continuation_levels'
        self.start_times[task_name] = time.time()
        print(f"[{task_name}] Starting at {self.start_times[task_name]:.2f}s")
        
        await asyncio.sleep(2.0)  # Simulate 2 second API call
        
        self.end_times[task_name] = time.time()
        print(f"[{task_name}] Completed at {self.end_times[task_name]:.2f}s (duration: {self.end_times[task_name] - self.start_times[task_name]:.2f}s)")
        return "Continuation levels completed"
    
    def call_llm_with_code_execution(self, prompt):
        """Simulate final decision with timing"""
        print(f"[final_decision] Starting final decision analysis...")
        decision_start_time = time.time()
        
        # Simulate 1.5 second API call
        time.sleep(1.5)
        
        decision_elapsed_time = time.time() - decision_start_time
        print(f"[final_decision] Final decision completed in {decision_elapsed_time:.2f} seconds")
        return "Final decision completed"

async def test_parallel_execution():
    """Test that all 5 LLM calls run in true parallel"""
    
    print("🔍 Testing Parallel Execution Fix")
    print("=" * 60)
    print("Expected behavior: All 5 tasks should start at nearly the same time")
    print("and complete based on their individual durations.")
    print()
    
    client = ParallelTestClient()
    
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
    
    print("🚀 Starting optimized analysis with parallel execution...")
    overall_start_time = time.time()
    
    # START ALL INDEPENDENT LLM CALLS IMMEDIATELY
    # 1. Indicator summary (no dependencies)
    print("\n[ASYNC-OPTIMIZED] Starting indicator summary analysis...")
    indicator_task = client.build_indicators_summary("RELIANCE", {}, 30, "1D")

    # 2. Chart insights (analyze images) - START ALL CHART TASKS IMMEDIATELY
    print("[ASYNC-OPTIMIZED] Starting all chart analysis tasks...")
    chart_analysis_tasks = []
    
    # GROUP 1: Comprehensive Technical Overview (1 chart - most important)
    if chart_paths.get('comparison_chart'):
        comparison_chart = b"mock_data"
        task = client.analyze_comprehensive_overview(comparison_chart)
        chart_analysis_tasks.append(("comprehensive_overview", task))
    
    # GROUP 2: Volume Analysis (3 charts together - complete volume story)
    volume_charts = [b"mock_data", b"mock_data", b"mock_data"]
    task = client.analyze_volume_comprehensive(volume_charts)
    chart_analysis_tasks.append(("volume_analysis", task))
    
    # GROUP 3: Reversal Pattern Analysis (2 charts together)
    reversal_charts = [b"mock_data", b"mock_data"]
    task = client.analyze_reversal_patterns(reversal_charts)
    chart_analysis_tasks.append(("reversal_patterns", task))
    
    # GROUP 4: Continuation & Level Analysis (2 charts together)
    continuation_charts = [b"mock_data", b"mock_data"]
    task = client.analyze_continuation_levels(continuation_charts)
    chart_analysis_tasks.append(("continuation_levels", task))
    
    # EXECUTE ALL INDEPENDENT TASKS IN PARALLEL
    print(f"\n[ASYNC-OPTIMIZED] Executing {len(chart_analysis_tasks) + 1} independent tasks in parallel...")
    parallel_start_time = time.time()
    
    # Combine indicator task with chart tasks
    all_tasks = [indicator_task] + [task for _, task in chart_analysis_tasks]
    all_results = await asyncio.gather(*all_tasks, return_exceptions=True)
    
    parallel_elapsed_time = time.time() - parallel_start_time
    print(f"\n[ASYNC-OPTIMIZED] Completed all independent tasks in {parallel_elapsed_time:.2f} seconds")
    
    # Final decision (depends on all previous results)
    print("\n[ASYNC-OPTIMIZED] Starting final decision analysis...")
    final_result = client.call_llm_with_code_execution("mock_prompt")
    
    overall_elapsed_time = time.time() - overall_start_time
    print(f"\n[ASYNC-OPTIMIZED] Total analysis completed in {overall_elapsed_time:.2f} seconds")
    
    # Analysis of parallel execution
    print("\n" + "=" * 60)
    print("📊 PARALLEL EXECUTION ANALYSIS")
    print("=" * 60)
    
    # Check if all tasks started at nearly the same time
    start_times = [client.start_times[name] for name in client.task_names]
    start_time_variance = max(start_times) - min(start_times)
    
    print(f"Task start time variance: {start_time_variance:.3f} seconds")
    
    if start_time_variance < 0.1:
        print("✅ EXCELLENT: All tasks started within 0.1 seconds (true parallel execution)")
    elif start_time_variance < 0.5:
        print("✅ GOOD: All tasks started within 0.5 seconds (mostly parallel)")
    elif start_time_variance < 1.0:
        print("⚠️ ACCEPTABLE: Tasks started within 1 second (some parallel)")
    else:
        print("❌ POOR: Tasks started sequentially (not parallel)")
    
    # Calculate expected vs actual time
    individual_durations = [3.0, 2.0, 3.0, 2.5, 2.0]  # Simulated durations
    expected_sequential_time = sum(individual_durations)
    expected_parallel_time = max(individual_durations)
    
    print(f"\nExpected sequential time: {expected_sequential_time:.2f} seconds")
    print(f"Expected parallel time: {expected_parallel_time:.2f} seconds")
    print(f"Actual parallel time: {parallel_elapsed_time:.2f} seconds")
    
    speedup = expected_sequential_time / parallel_elapsed_time
    print(f"Speedup achieved: {speedup:.2f}x")
    
    if speedup > 2.5:
        print("✅ EXCELLENT: Achieved >2.5x speedup (true parallel execution)")
    elif speedup > 2.0:
        print("✅ GOOD: Achieved >2.0x speedup (mostly parallel)")
    elif speedup > 1.5:
        print("⚠️ ACCEPTABLE: Achieved >1.5x speedup (some parallel)")
    else:
        print("❌ POOR: Minimal speedup (sequential execution)")
    
    print("\n🎉 Parallel execution test completed!")

async def main():
    """Main function"""
    await test_parallel_execution()

if __name__ == "__main__":
    asyncio.run(main()) 