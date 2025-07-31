#!/usr/bin/env python3
"""
Test script to verify that the context fixes work properly.
This script tests the prompt formatting with and without context parameters.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gemini.prompt_manager import PromptManager
from gemini.context_engineer import ContextEngineer, AnalysisType, ContextConfig

def test_prompt_formatting():
    """Test that prompt formatting works with and without context."""
    print("Testing prompt formatting fixes...")
    
    # Initialize prompt manager
    pm = PromptManager()
    
    # Test 1: Format prompt without context (should use default)
    print("\n1. Testing format_prompt without context parameter:")
    try:
        result = pm.format_prompt("optimized_technical_overview")
        print("✅ SUCCESS: Prompt formatted without context parameter")
        print(f"   Result length: {len(result)} characters")
        print(f"   Contains default context: {'No additional context provided' in result}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
    
    # Test 2: Format prompt with context parameter
    print("\n2. Testing format_prompt with context parameter:")
    try:
        test_context = "## Analysis Context:\nTest context for analysis."
        result = pm.format_prompt("optimized_technical_overview", context=test_context)
        print("✅ SUCCESS: Prompt formatted with context parameter")
        print(f"   Result length: {len(result)} characters")
        print(f"   Contains test context: {'Test context for analysis' in result}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
    
    # Test 3: Test all optimized prompts
    print("\n3. Testing all optimized prompts:")
    optimized_prompts = [
        "optimized_technical_overview",
        "optimized_pattern_analysis", 
        "optimized_volume_analysis",
        "optimized_mtf_comparison",
        "optimized_reversal_patterns",
        "optimized_continuation_levels",
        "optimized_indicators_summary",
        "optimized_final_decision"
    ]
    
    for prompt_name in optimized_prompts:
        try:
            result = pm.format_prompt(prompt_name)
            print(f"✅ {prompt_name}: SUCCESS")
        except Exception as e:
            print(f"❌ {prompt_name}: FAILED - {e}")

def test_context_engineer():
    """Test that context engineer works properly."""
    print("\n\nTesting context engineer...")
    
    # Initialize context engineer
    config = ContextConfig()
    ce = ContextEngineer(config)
    
    # Test 1: Structure context for different analysis types
    print("\n1. Testing context structuring:")
    test_data = {
        "analysis_focus": "test_analysis",
        "key_indicators": {"test": "value"},
        "critical_levels": {"support": [100, 200], "resistance": [300, 400]}
    }
    
    analysis_types = [
        AnalysisType.INDICATOR_SUMMARY,
        AnalysisType.VOLUME_ANALYSIS,
        AnalysisType.REVERSAL_PATTERNS,
        AnalysisType.CONTINUATION_LEVELS,
        AnalysisType.FINAL_DECISION
    ]
    
    for analysis_type in analysis_types:
        try:
            context = ce.structure_context(test_data, analysis_type, "TEST", "5 days, 1hour", "Test knowledge context")
            print(f"✅ {analysis_type.value}: SUCCESS")
            print(f"   Context length: {len(context)} characters")
        except Exception as e:
            print(f"❌ {analysis_type.value}: FAILED - {e}")

def test_integration():
    """Test integration between prompt manager and context engineer."""
    print("\n\nTesting integration...")
    
    pm = PromptManager()
    config = ContextConfig()
    ce = ContextEngineer(config)
    
    # Test: Create context and use it in prompt
    try:
        test_data = {
            "analysis_focus": "integration_test",
            "key_indicators": {"rsi": 65, "macd": "bullish"},
            "critical_levels": {"support": [100], "resistance": [200]}
        }
        
        context = ce.structure_context(test_data, AnalysisType.INDICATOR_SUMMARY, "TEST", "5 days, 1hour", "")
        result = pm.format_prompt("optimized_technical_overview", context=context)
        
        print("✅ Integration test: SUCCESS")
        print(f"   Final prompt length: {len(result)} characters")
        print(f"   Contains structured context: {'Analysis Context' in result}")
        
    except Exception as e:
        print(f"❌ Integration test: FAILED - {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("CONTEXT FIXES VERIFICATION TEST")
    print("=" * 60)
    
    test_prompt_formatting()
    test_context_engineer()
    test_integration()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60) 