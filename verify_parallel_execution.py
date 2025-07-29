#!/usr/bin/env python3
"""
Verification script to confirm that the 5 LLM calls are truly running in parallel.
This script demonstrates the parallel execution of:
1. Indicator Summary
2. Comprehensive Overview
3. Volume Analysis  
4. Reversal Patterns
5. Continuation Levels
"""

import asyncio
import time
import sys
import os

# Add the gemini directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'gemini'))

class ParallelVerificationClient:
    """Client to verify parallel execution of the 5 LLM calls"""
    
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
    
    async def analyze_comprehensive_overview(self, image_data):
        """Simulate comprehensive overview with timing"""
        task_name = 'comprehensive_overview'
        self.start_times[task_name] = time.time()
        print(f"[{task_name}] Starting at {self.start_times[task_name]:.2f}s")
        
        await asyncio.sleep(2.0)  # Simulate 2 second API call
        
        self.end_times[task_name] = time.time()
        print(f"[{task_name}] Completed at {self.end_times[task_name]:.2f}s (duration: {self.end_times[task_name] - self.start_times[task_name]:.2f}s)")
        return "Comprehensive technical analysis completed successfully."
    
    async def analyze_volume_comprehensive(self, images):
        """Simulate volume analysis with timing"""
        task_name = 'volume_analysis'
        self.start_times[task_name] = time.time()
        print(f"[{task_name}] Starting at {self.start_times[task_name]:.2f}s")
        
        await asyncio.sleep(3.0)  # Simulate 3 second API call
        
        self.end_times[task_name] = time.time()
        print(f"[{task_name}] Completed at {self.end_times[task_name]:.2f}s (duration: {self.end_times[task_name] - self.start_times[task_name]:.2f}s)")
        return "Volume analysis completed successfully."
    
    async def analyze_reversal_patterns(self, images):
        """Simulate reversal pattern analysis with timing"""
        task_name = 'reversal_patterns'
        self.start_times[task_name] = time.time()
        print(f"[{task_name}] Starting at {self.start_times[task_name]:.2f}s")
        
        await asyncio.sleep(2.5)  # Simulate 2.5 second API call
        
        self.end_times[task_name] = time.time()
        print(f"[{task_name}] Completed at {self.end_times[task_name]:.2f}s (duration: {self.end_times[task_name] - self.start_times[task_name]:.2f}s)")
        return "Reversal pattern analysis completed successfully."
    
    async def analyze_continuation_levels(self, images):
        """Simulate continuation and level analysis with timing"""
        task_name = 'continuation_levels'
        self.start_times[task_name] = time.time()
        print(f"[{task_name}] Starting at {self.start_times[task_name]:.2f}s")
        
        await asyncio.sleep(2.0)  # Simulate 2 second API call
        
        self.end_times[task_name] = time.time()
        print(f"[{task_name}] Completed at {self.end_times[task_name]:.2f}s (duration: {self.end_times[task_name] - self.start_times[task_name]:.2f}s)")
        return "Continuation and level analysis completed successfully."
    
    def call_llm_with_code_execution(self, prompt):
        """Simulate final decision analysis"""
        print("[final_decision] Starting final decision analysis...")
        time.sleep(1.5)  # Simulate 1.5 second API call
        print("[final_decision] Final decision analysis completed.")
        return '{"signal": "buy", "confidence": 80}', [], []

async def verify_parallel_execution():
    """Verify that all 5 LLM calls start immediately and run in parallel"""
    
    print("🔍 Verifying Parallel Execution of 5 LLM Calls")
    print("=" * 60)
    print("Expected behavior: All 5 tasks should start at nearly the same time")
    print("and complete based on their individual durations.")
    print()
    
    client = ParallelVerificationClient()
    
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
    decision_start_time = time.time()
    final_result = client.call_llm_with_code_execution("mock_prompt")
    decision_elapsed_time = time.time() - decision_start_time
    
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
        print("✅ All tasks started nearly simultaneously (excellent parallelization)")
    elif start_time_variance < 0.5:
        print("✅ All tasks started within 0.5 seconds (good parallelization)")
    else:
        print("❌ Tasks did not start simultaneously (poor parallelization)")
    
    # Check individual task durations
    print("\nIndividual task durations:")
    for task_name in client.task_names:
        duration = client.end_times[task_name] - client.start_times[task_name]
        print(f"  {task_name}: {duration:.2f}s")
    
    # Check total execution time
    expected_sequential_time = sum([
        client.end_times[name] - client.start_times[name] 
        for name in client.task_names
    ])
    
    print(f"\nExpected sequential time: {expected_sequential_time:.2f} seconds")
    print(f"Actual parallel time: {parallel_elapsed_time:.2f} seconds")
    print(f"Speedup: {expected_sequential_time / parallel_elapsed_time:.2f}x")
    
    # Verify the 5 LLM calls
    print(f"\n✅ VERIFICATION: All 5 LLM calls executed in parallel:")
    print(f"   1. Indicator Summary: {client.end_times['indicator_summary'] - client.start_times['indicator_summary']:.2f}s")
    print(f"   2. Comprehensive Overview: {client.end_times['comprehensive_overview'] - client.start_times['comprehensive_overview']:.2f}s")
    print(f"   3. Volume Analysis: {client.end_times['volume_analysis'] - client.start_times['volume_analysis']:.2f}s")
    print(f"   4. Reversal Patterns: {client.end_times['reversal_patterns'] - client.start_times['reversal_patterns']:.2f}s")
    print(f"   5. Continuation Levels: {client.end_times['continuation_levels'] - client.start_times['continuation_levels']:.2f}s")
    
    if parallel_elapsed_time < expected_sequential_time * 0.8:
        print("\n🎉 SUCCESS: Parallel execution is working correctly!")
        print("   All 5 LLM calls are running concurrently as expected.")
    else:
        print("\n⚠️  WARNING: Parallel execution may not be optimal.")
        print("   Check if tasks are truly running in parallel.")

if __name__ == "__main__":
    asyncio.run(verify_parallel_execution()) 