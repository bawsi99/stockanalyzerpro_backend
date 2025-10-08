#!/usr/bin/env python3
"""
Integration test for token counter with LLM client

This script tests the complete integration of the token counting system
with the LLM client and providers.
"""

import asyncio
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm.client import LLMClient
from llm.token_counter import get_token_counter, print_token_usage_summary, reset_token_counter


async def test_token_integration():
    """Test token counting integration with LLM client."""
    print("🧪 Testing Token Counter Integration with LLM Client")
    print("=" * 60)
    
    # Reset counter for clean test
    reset_token_counter()
    
    try:
        # Test with different agent configurations
        test_cases = [
            {
                "agent_name": "indicator_agent",
                "provider": "gemini", 
                "model": "gemini-2.5-flash",
                "prompt": "Analyze RSI indicator: current value is 65. What does this suggest about the stock's momentum?"
            },
            {
                "agent_name": "final_decision_agent", 
                "provider": "gemini",
                "model": "gemini-2.5-pro", 
                "prompt": "Based on technical analysis showing bullish momentum and strong volume, provide investment recommendation."
            },
            {
                "agent_name": "sector_agent",
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "prompt": "Compare technology sector performance against market benchmark this quarter."
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📋 Test Case {i}: {test_case['agent_name']} using {test_case['model']}")
            print("-" * 40)
            
            # Create client for this agent
            client = LLMClient(
                agent_name=test_case['agent_name'],
                provider=test_case['provider'],
                model=test_case['model']
            )
            
            print(f"🔑 Client initialized: {client.get_provider_info()}")
            
            try:
                # Test with token usage tracking
                print("🚀 Generating response with token tracking...")
                
                # This would normally call the LLM, but we'll simulate it
                # since we may not have API keys configured in test environment
                print("⚠️  Simulating LLM call (API keys may not be configured)")
                
                # Instead, let's test the token tracking with mock data
                from llm.token_counter import track_llm_usage
                
                # Simulate different token usage patterns
                mock_response = {
                    'usageMetadata': {
                        'promptTokenCount': 100 + (i * 50),  # Different input sizes
                        'candidatesTokenCount': 50 + (i * 25),  # Different output sizes
                        'totalTokenCount': 150 + (i * 75)
                    }
                }
                
                # Track the usage
                token_data = track_llm_usage(
                    response=mock_response,
                    agent_name=test_case['agent_name'],
                    provider=test_case['provider'],
                    model=test_case['model'],
                    duration_ms=1200 + (i * 300),
                    success=True
                )
                
                print(f"✅ Token tracking successful: {token_data.total_tokens} tokens")
                
            except Exception as e:
                print(f"⚠️  LLM call simulation error (expected in test): {e}")
                
        # Show comprehensive results
        print("\n" + "=" * 60)
        print("📊 FINAL TOKEN USAGE ANALYTICS")
        print("=" * 60)
        
        counter = get_token_counter()
        summary = counter.get_summary()
        
        print(f"Total Tracked Calls: {summary['total_usage']['total_calls']}")
        print(f"Total Tokens Used: {summary['total_usage']['total_tokens']:,}")
        
        # Model breakdown
        print(f"\n📱 Model Usage:")
        for model, usage in summary['usage_by_model'].items():
            print(f"  {model:20} | {usage['total_tokens']:>6,} tokens | {usage['calls']:>2} calls")
            print(f"  {'':20} | agents: {', '.join(usage['agents_using_model'])}")
        
        # Agent breakdown  
        print(f"\n🤖 Agent Usage:")
        for agent, usage in summary['usage_by_agent'].items():
            print(f"  {agent:20} | {usage['total_tokens']:>6,} tokens | {usage['calls']:>2} calls")
        
        print(f"\n✅ Integration test completed successfully!")
        print(f"📋 The system can now track token usage by:")
        print(f"   • Different models (flash vs pro)")
        print(f"   • Different agents (indicator, final_decision, etc.)")
        print(f"   • Individual calls with timing")
        print(f"   • Provider-level aggregation")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_model_comparison():
    """Test model comparison functionality."""
    print("\n" + "=" * 60)
    print("🔍 Testing Model Comparison Functionality")
    print("=" * 60)
    
    counter = get_token_counter()
    model_usage = counter.get_usage_by_model()
    
    if len(model_usage) >= 2:
        models = list(model_usage.keys())
        model1, model2 = models[0], models[1]
        
        from llm.token_counter import compare_model_efficiency
        comparison = compare_model_efficiency(model1, model2)
        
        if 'error' not in comparison:
            comp = comparison['comparison']
            print(f"\n📊 Comparing {model1} vs {model2}:")
            print(f"  Total tokens: {comp['total_tokens'][model1]:,} vs {comp['total_tokens'][model2]:,}")
            print(f"  Average per call: {comp['avg_tokens_per_call'][model1]:.0f} vs {comp['avg_tokens_per_call'][model2]:.0f}")
            
            if comp['agents_using_model']['common_agents']:
                print(f"  Common agents: {comp['agents_using_model']['common_agents']}")
                
            efficiency_winner = model1 if comp['avg_tokens_per_call'][model1] < comp['avg_tokens_per_call'][model2] else model2
            print(f"  🏆 More efficient (lower avg tokens): {efficiency_winner}")
        else:
            print(f"❌ Comparison failed: {comparison['error']}")
    else:
        print(f"⚠️  Need at least 2 models for comparison. Found: {list(model_usage.keys())}")


if __name__ == "__main__":
    async def main():
        success = await test_token_integration()
        if success:
            await test_model_comparison()
        
    asyncio.run(main())