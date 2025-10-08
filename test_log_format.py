#!/usr/bin/env python3
"""
Test script to demonstrate the exact log format requested:
agent : model : input tokens : output tokens : total time
"""

import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm.token_counter import (
    track_llm_usage, get_agent_model_combinations, 
    get_agent_timing_breakdown, reset_token_counter
)


def simulate_analysis_with_exact_format():
    """Simulate a stock analysis and show the exact log format requested."""
    print("🧪 Testing Exact Log Format: agent : model : input tokens : output tokens : total time")
    print("=" * 80)
    
    # Reset for clean test
    reset_token_counter()
    
    # Simulate realistic agent calls with different models and timings
    agent_calls = [
        # (agent_name, model, input_tokens, output_tokens, duration_ms)
        ("indicator_agent", "gemini-2.5-flash", 180, 90, 1200),
        ("volume_agent", "gemini-2.5-flash", 200, 100, 1350),
        ("mtf_agent", "gemini-2.5-flash", 220, 110, 1500),
        ("sector_agent", "gemini-2.5-pro", 350, 175, 2100),
        ("final_decision_agent", "gemini-2.5-pro", 500, 250, 3200),
        ("risk_agent", "gemini-2.5-pro", 300, 150, 2800),
        # Some agents make multiple calls
        ("indicator_agent", "gemini-2.5-pro", 250, 125, 2000),  # Same agent, different model
        ("volume_agent", "gemini-2.5-flash", 150, 75, 900),     # Same agent, additional call
    ]
    
    print(f"🚀 Simulating {len(agent_calls)} LLM calls...")
    
    # Track each call
    for agent, model, input_tokens, output_tokens, duration in agent_calls:
        mock_response = {
            'usageMetadata': {
                'promptTokenCount': input_tokens,
                'candidatesTokenCount': output_tokens,
                'totalTokenCount': input_tokens + output_tokens
            }
        }
        
        track_llm_usage(
            response=mock_response,
            agent_name=agent,
            provider="gemini",
            model=model,
            duration_ms=duration,
            success=True
        )
    
    print("✅ Simulation complete! Generating logs in requested format...\n")
    
    # Get the data
    agent_model_combos = get_agent_model_combinations()
    agent_timings = get_agent_timing_breakdown()
    
    # Show totals first
    total_calls = len(agent_calls)
    total_input = sum(call[2] for call in agent_calls)
    total_output = sum(call[3] for call in agent_calls)
    total_time = sum(call[4] for call in agent_calls) / 1000.0
    
    print(f"📊 TOKEN USAGE SUMMARY for AAPL")
    print(f"=" * 80)
    print(f"Total Analysis Time: {total_time:.2f}s")
    print(f"Total LLM Calls: {total_calls}")
    print(f"Total Input Tokens: {total_input:,}")
    print(f"Total Output Tokens: {total_output:,}")
    print(f"Total Tokens: {total_input + total_output:,}")
    
    # Show the exact format you requested
    print(f"\n🤖 AGENT DETAILS:")
    
    # Print in requested format: agent : model : input tokens : output tokens : total time
    for agent, models in agent_model_combos.items():
        for model, usage in models.items():
            total_time_s = agent_timings.get(agent, 0.0)
            
            print(f"  {agent:20} : {model:17} : {usage['input_tokens']:>4d} input : {usage['output_tokens']:>4d} output : {total_time_s:>6.2f}s")
    
    print(f"=" * 80)
    
    # Show what this looks like in practice
    print(f"\n📋 EXPLANATION:")
    print(f"This shows exactly what you requested:")
    print(f"• Each agent that made LLM calls")
    print(f"• Which model each agent used (flash vs pro)")  
    print(f"• Input and output tokens for each agent-model combination")
    print(f"• Total time spent by each agent (sum of all their LLM calls)")
    print(f"")
    print(f"Notice how:")
    print(f"• indicator_agent used both flash AND pro models (2 separate lines)")
    print(f"• volume_agent made multiple calls but all with flash (tokens are summed)")
    print(f"• Each agent's total time includes ALL their LLM call durations")


if __name__ == "__main__":
    simulate_analysis_with_exact_format()