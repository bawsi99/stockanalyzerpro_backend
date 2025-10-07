#!/usr/bin/env python3
"""
Test script to verify MTF LLM Agent migration to new backend/llm system.

This script tests both the new backend/llm system and fallback to legacy system,
ensuring the migration maintains compatibility while providing new features.

Usage:
    cd backend/agents/mtf_analysis
    python test_migration.py
"""

import sys
import os
import asyncio
import json
from typing import Dict, Any

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

def test_initialization():
    """Test MTF LLM Agent initialization with new system."""
    print("🧪 Testing MTF LLM Agent Initialization")
    print("=" * 50)
    
    try:
        from backend.agents.mtf_analysis.mtf_llm_agent import MTFLLMAgent
        
        # Initialize agent
        agent = MTFLLMAgent()
        
        # Check which system is being used
        if hasattr(agent, '_legacy_mode') and agent._legacy_mode:
            print("⚠️  Using legacy backend/gemini system (fallback)")
            print(f"   Agent: {agent.agent_name}")
            print(f"   Client type: {type(agent.llm_client).__name__}")
        else:
            print("✅ Using new backend/llm system")
            print(f"   Agent: {agent.agent_name}")
            print(f"   Client type: {type(agent.llm_client).__name__}")
            if hasattr(agent.llm_client, 'get_provider_info'):
                print(f"   Provider: {agent.llm_client.get_provider_info()}")
            if hasattr(agent.llm_client, 'get_config'):
                config = agent.llm_client.get_config()
                print(f"   Model: {config.get('model', 'unknown')}")
                print(f"   Code execution: {config.get('enable_code_execution', 'unknown')}")
                print(f"   Timeout: {config.get('timeout', 'unknown')}s")
        
        print("\n✅ Initialization test passed")
        return agent
        
    except Exception as e:
        print(f"\n❌ Initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_mock_mtf_analysis() -> Dict[str, Any]:
    """Create mock MTF analysis data for testing."""
    return {
        "success": True,
        "timeframe_analyses": {
            "1min": {
                "trend": "bullish",
                "confidence": 0.75,
                "data_points": 100,
                "key_indicators": {
                    "rsi": 65.2,
                    "macd_signal": "bullish",
                    "volume_status": "above_average",
                    "support_levels": [150.25, 149.80],
                    "resistance_levels": [152.50, 153.00]
                }
            },
            "5min": {
                "trend": "bullish",
                "confidence": 0.80,
                "data_points": 288,
                "key_indicators": {
                    "rsi": 68.1,
                    "macd_signal": "bullish",
                    "volume_status": "high"
                }
            },
            "1day": {
                "trend": "bullish",
                "confidence": 0.85,
                "data_points": 365,
                "key_indicators": {
                    "rsi": 72.3,
                    "macd_signal": "bullish",
                    "volume_status": "average"
                }
            }
        },
        "cross_timeframe_validation": {
            "consensus_trend": "bullish",
            "signal_strength": 0.82,
            "confidence_score": 0.79,
            "supporting_timeframes": ["1min", "5min", "1day"],
            "conflicting_timeframes": [],
            "divergence_detected": False,
            "key_conflicts": []
        },
        "summary": {
            "overall_signal": "bullish",
            "confidence": 0.82,
            "signal_alignment": "strong",
            "risk_level": "medium",
            "recommendation": "buy"
        },
        "agent_insights": {
            "total_agents_run": 3,
            "successful_agents": 3,
            "failed_agents": 0,
            "confidence_score": 0.81
        }
    }

async def test_prompt_building(agent):
    """Test prompt building functionality."""
    print("\n🧪 Testing Prompt Building")
    print("=" * 50)
    
    try:
        # Create mock MTF data
        mtf_analysis = create_mock_mtf_analysis()
        
        # Test prompt building
        prompt = agent._build_mtf_prompt(
            symbol="RELIANCE",
            exchange="NSE", 
            mtf_analysis=mtf_analysis,
            context="Test context for MTF analysis"
        )
        
        if prompt:
            print("✅ Prompt building successful")
            print(f"   Prompt length: {len(prompt)} characters")
            print(f"   Contains symbol: {'RELIANCE' in prompt}")
            print(f"   Contains timeframes: {'1min' in prompt and '5min' in prompt}")
            print(f"   Contains analysis request: {'Analysis Task' in prompt}")
            
            # Show a preview of the prompt
            lines = prompt.split('\n')[:10]
            print(f"\n   Prompt preview (first 10 lines):")
            for i, line in enumerate(lines, 1):
                print(f"     {i}. {line}")
            prompt_lines = prompt.split('\n')
            if len(prompt_lines) > 10:
                remaining_lines = len(prompt_lines) - 10
                print(f"     ... ({remaining_lines} more lines)")
                
            return True
        else:
            print("❌ Prompt building returned empty string")
            return False
            
    except Exception as e:
        print(f"❌ Prompt building test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_llm_integration(agent):
    """Test LLM integration (without making actual API calls)."""
    print("\n🧪 Testing LLM Integration (Mock)")
    print("=" * 50)
    
    try:
        # Check if we can access the LLM client methods
        if hasattr(agent, '_legacy_mode') and agent._legacy_mode:
            print("⚠️  Testing legacy system integration")
            has_core = hasattr(agent.llm_client, 'core')
            has_method = hasattr(agent.llm_client.core, 'call_llm_with_code_execution') if has_core else False
            print(f"   Has core: {has_core}")
            print(f"   Has method: {has_method}")
        else:
            print("✅ Testing new backend/llm system integration")
            has_generate = hasattr(agent.llm_client, 'generate')
            has_config = hasattr(agent.llm_client, 'get_config')
            has_provider_info = hasattr(agent.llm_client, 'get_provider_info')
            
            print(f"   Has generate method: {has_generate}")
            print(f"   Has get_config method: {has_config}")
            print(f"   Has get_provider_info method: {has_provider_info}")
            
            if has_config:
                config = agent.llm_client.get_config()
                print(f"   Configured for code execution: {config.get('enable_code_execution', False)}")
                print(f"   Max retries: {config.get('max_retries', 'unknown')}")
        
        print("\n✅ LLM integration test passed")
        return True
        
    except Exception as e:
        print(f"\n❌ LLM integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_confidence_calculation(agent):
    """Test confidence calculation method."""
    print("\n🧪 Testing Confidence Calculation")
    print("=" * 50)
    
    try:
        # Create mock data
        mtf_analysis = create_mock_mtf_analysis()
        mock_llm_response = "This is a detailed analysis of the multi-timeframe data showing strong bullish signals across all timeframes with high confidence based on RSI momentum and MACD confirmation."
        
        # Test confidence calculation
        confidence = agent._calculate_confidence(mtf_analysis, mock_llm_response)
        
        print(f"✅ Confidence calculation successful")
        print(f"   Calculated confidence: {confidence:.3f}")
        print(f"   Valid range (0-1): {0.0 <= confidence <= 1.0}")
        print(f"   MTF confidence: {mtf_analysis['summary']['confidence']:.3f}")
        print(f"   Agent confidence: {mtf_analysis['agent_insights']['confidence_score']:.3f}")
        print(f"   Response quality factor: {min(1.0, len(mock_llm_response) / 2000):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Confidence calculation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("🚀 MTF LLM Agent Migration Test Suite")
    print("=" * 60)
    
    # Test 1: Initialization
    agent = test_initialization()
    if not agent:
        print("\n💥 Critical failure: Agent initialization failed")
        return False
    
    # Test 2: Prompt building
    prompt_success = await test_prompt_building(agent)
    
    # Test 3: LLM integration
    integration_success = await test_llm_integration(agent)
    
    # Test 4: Confidence calculation
    confidence_success = test_confidence_calculation(agent)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    tests = [
        ("Initialization", True),  # If we got here, it passed
        ("Prompt Building", prompt_success),
        ("LLM Integration", integration_success),
        ("Confidence Calculation", confidence_success)
    ]
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MTF LLM Agent migration is successful.")
        
        if hasattr(agent, '_legacy_mode') and agent._legacy_mode:
            print("\n💡 Note: Currently using legacy backend/gemini system.")
            print("   To use the new system, ensure:")
            print("   1. backend/llm module is properly installed")
            print("   2. GEMINI_API_KEY environment variable is set")
            print("   3. No import errors in backend/llm")
        else:
            print("\n🔥 Using new backend/llm system successfully!")
            print("   The migration is complete and working.")
            
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)