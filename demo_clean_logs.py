#!/usr/bin/env python3
"""
Clean production-ready log format demonstration
Shows exactly what you'll see in the analysis service logs
"""

import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm.token_counter import (
    track_llm_usage, get_agent_model_combinations, 
    get_agent_timing_breakdown, reset_token_counter,
    get_token_usage_summary
)


def demo_production_logs():
    """Demo exactly what the production logs will look like."""
    # Reset for clean demo
    reset_token_counter()
    
    # Simulate a real analysis with various agents
    calls = [
        ("indicator_agent", "gemini-2.5-flash", 180, 90, 1200),
        ("volume_agent", "gemini-2.5-flash", 200, 100, 1350),
        ("mtf_agent", "gemini-2.5-flash", 220, 110, 1500),
        ("sector_agent", "gemini-2.5-pro", 350, 175, 2100),
        ("final_decision_agent", "gemini-2.5-pro", 500, 250, 3200),
        ("risk_agent", "gemini-2.5-pro", 300, 150, 2800),
    ]
    
    # Track all calls
    for agent, model, inp, out, dur in calls:
        track_llm_usage(
            response={'usageMetadata': {'promptTokenCount': inp, 'candidatesTokenCount': out, 'totalTokenCount': inp+out}},
            agent_name=agent, provider="gemini", model=model, duration_ms=dur, success=True
        )
    
    # Generate the exact logs as they'll appear in production
    token_summary = get_token_usage_summary()
    agent_model_combos = get_agent_model_combinations()
    agent_timings = get_agent_timing_breakdown()
    
    print("================================================================================")
    print("📊 TOKEN USAGE SUMMARY for AAPL")
    print("================================================================================")
    print(f"Total Analysis Time: 12.15s")
    print(f"Total LLM Calls: {token_summary['total_usage']['total_calls']}")
    print(f"Total Input Tokens: {token_summary['total_usage']['total_input_tokens']:,}")
    print(f"Total Output Tokens: {token_summary['total_usage']['total_output_tokens']:,}")
    print(f"Total Tokens: {token_summary['total_usage']['total_tokens']:,}")
    
    print(f"\n🤖 AGENT DETAILS:")
    
    # This is the exact format you requested
    for agent, models in agent_model_combos.items():
        for model, usage in models.items():
            total_time_s = agent_timings.get(agent, 0.0)
            print(f"  {agent:20} : {model:17} : {usage['input_tokens']:>4d} input : {usage['output_tokens']:>4d} output : {total_time_s:>6.2f}s")
    
    print("================================================================================")


if __name__ == "__main__":
    demo_production_logs()