#!/usr/bin/env python3
"""
Demo of the new table format for token usage output
Shows exactly what the improved logs will look like
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


def demo_table_format():
    """Demo the new table format using your actual data."""
    # Reset for clean demo
    reset_token_counter()
    
    # Simulate your actual BAJFINANCE analysis data
    bajfinance_calls = [
        ("institutional_activity_agent", "gemini-2.5-flash", 14320, 320, 41080),
        ("mtf_agent", "gemini-2.5-flash", 1460, 2586, 32990),
        ("risk_agent", "gemini-2.5-flash", 1524, 1031, 24280),
        ("sector_agent", "gemini-2.5-flash", 656, 192, 11540),
        ("support_resistance_agent", "gemini-2.5-flash", 27441, 2862, 23190),
        ("volume_anomaly_agent", "gemini-2.5-flash", 2076, 261, 16720),
        ("volume_confirmation_agent", "gemini-2.5-flash", 509, 104, 11010),
        ("volume_momentum_agent", "gemini-2.5-flash", 31155, 97, 8870),
        ("final_decision_agent", "gemini-2.5-pro", 1687, 900, 29160),
        ("indicator_agent", "gemini-2.5-pro", 1632, 496, 23780),
    ]
    
    # Track all calls
    for agent, model, inp, out, dur in bajfinance_calls:
        track_llm_usage(
            response={'usageMetadata': {'promptTokenCount': inp, 'candidatesTokenCount': out, 'totalTokenCount': inp+out}},
            agent_name=agent, provider="gemini", model=model, duration_ms=dur, success=True
        )
    
    # Generate the table format exactly as it will appear in production
    token_summary = get_token_usage_summary()
    agent_model_combos = get_agent_model_combinations()
    agent_timings = get_agent_timing_breakdown()
    
    # Simulate the exact output from analysis_service.py
    print(f"\n{'='*100}")
    print(f"📊 TOKEN USAGE SUMMARY for BAJFINANCE")
    print(f"{'='*100}")
    print(f"Total Analysis Time: 79.07s")
    print(f"Total LLM Calls: {token_summary['total_usage']['total_calls']}")
    print(f"Total Input Tokens: {token_summary['total_usage']['total_input_tokens']:,}")
    print(f"Total Output Tokens: {token_summary['total_usage']['total_output_tokens']:,}")
    print(f"Total Tokens: {token_summary['total_usage']['total_tokens']:,}")
    
    # Show per-agent breakdown in proper table format
    if agent_model_combos:
        print(f"\n🤖 AGENT DETAILS:")
        print(f"{'='*100}")
        
        # Table header
        print(f"{'Agent':25} | {'Model':17} | {'Input':>8} | {'Output':>8} | {'Total':>8} | {'Time':>8}")
        print(f"{'-'*25} | {'-'*17} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")
        
        # Sort by model (flash first, then pro) for cleaner display
        all_entries = []
        for agent, models in agent_model_combos.items():
            for model, usage in models.items():
                total_time_s = agent_timings.get(agent, 0.0)
                all_entries.append((agent, model, usage, total_time_s))
        
        # Sort: flash models first, then pro models, then by agent name
        all_entries.sort(key=lambda x: ("pro" in x[1], x[0]))
        
        # Table rows
        total_input = 0
        total_output = 0
        total_tokens_sum = 0
        total_time_sum = 0.0
        
        for agent, model, usage, total_time_s in all_entries:
            total_input += usage['input_tokens']
            total_output += usage['output_tokens']
            total_tokens_sum += usage['total_tokens']
            total_time_sum += total_time_s
            
            # Truncate long agent names
            agent_display = agent[:24] if len(agent) > 24 else agent
            model_display = model.replace('gemini-2.5-', '').upper()  # Show FLASH/PRO for brevity
            
            print(f"{agent_display:25} | {model_display:17} | {usage['input_tokens']:>8,} | {usage['output_tokens']:>8,} | {usage['total_tokens']:>8,} | {total_time_s:>7.2f}s")
        
        # Table footer with totals
        print(f"{'-'*25} | {'-'*17} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")
        print(f"{'TOTAL':25} | {'':17} | {total_input:>8,} | {total_output:>8,} | {total_tokens_sum:>8,} | {total_time_sum:>7.2f}s")
        
        # Add per-model breakdown
        print(f"\n📱 MODEL BREAKDOWN:")
        print(f"{'='*70}")
        print(f"{'Model':20} | {'Input':>12} | {'Output':>12} | {'Total':>12} | {'Calls':>6}")
        print(f"{'-'*20} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*6}")
        
        # Calculate per-model totals
        model_stats = {}
        for agent, model, usage, total_time_s in all_entries:
            if model not in model_stats:
                model_stats[model] = {
                    'input_tokens': 0,
                    'output_tokens': 0, 
                    'total_tokens': 0,
                    'calls': 0
                }
            model_stats[model]['input_tokens'] += usage['input_tokens']
            model_stats[model]['output_tokens'] += usage['output_tokens']
            model_stats[model]['total_tokens'] += usage['total_tokens']
            model_stats[model]['calls'] += usage['calls']
        
        # Sort models (flash first, then pro)
        sorted_models = sorted(model_stats.items(), key=lambda x: "pro" in x[0])
        
        model_total_input = 0
        model_total_output = 0
        model_total_tokens = 0
        model_total_calls = 0
        
        for model, stats in sorted_models:
            model_display = model.replace('gemini-2.5-', '').upper()
            model_total_input += stats['input_tokens']
            model_total_output += stats['output_tokens']
            model_total_tokens += stats['total_tokens']
            model_total_calls += stats['calls']
            
            print(f"{model_display:20} | {stats['input_tokens']:>12,} | {stats['output_tokens']:>12,} | {stats['total_tokens']:>12,} | {stats['calls']:>6}")
        
        # Model breakdown footer
        print(f"{'-'*20} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*6}")
        print(f"{'TOTAL':20} | {model_total_input:>12,} | {model_total_output:>12,} | {model_total_tokens:>12,} | {model_total_calls:>6}")
    
    print(f"{'='*100}")
    
    print(f"\n✨ Key Improvements in Table Format:")
    print(f"• Clean column alignment with proper spacing")
    print(f"• Header and footer separators for clarity") 
    print(f"• Thousand separators (commas) for large numbers")
    print(f"• Condensed model names (FLASH/PRO instead of full names)")
    print(f"• Total row at bottom for quick summary")
    print(f"• Per-model breakdown showing FLASH vs PRO usage")
    print(f"• Easy cost analysis with model-specific token counts")
    print(f"• Consistent width (100 chars) for professional appearance")
    print(f"• Long agent names truncated to fit table width")


if __name__ == "__main__":
    demo_table_format()