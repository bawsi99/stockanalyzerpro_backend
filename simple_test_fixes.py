#!/usr/bin/env python3
"""
Simple Test for Context Engineering Fixes

This script provides a simple way to test that the context engineering fixes work.
It focuses on the core functionality and the original errors.

Usage:
    python simple_test_fixes.py
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic context engineering functionality."""
    print("🔍 Testing Basic Context Engineering Functionality...")
    
    try:
        from gemini.context_engineer import ContextEngineer, AnalysisType
        print("✓ Imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Test data
    test_data = {
        'close': ['100.0', '101.0', '102.0', '103.0', '104.0'],
        'volume': ['1000', '1100', '1200', '1300', '1400'],
        'rsi': ['50.0', '55.0', '60.0', '65.0', '70.0'],
        'macd': {
            'histogram': ['0.1', '0.2', '0.3', '0.4', '0.5']
        }
    }
    
    ce = ContextEngineer()
    
    # Test each analysis type
    analysis_types = [
        ("Volume Analysis", AnalysisType.VOLUME_ANALYSIS),
        ("Pattern Analysis", AnalysisType.REVERSAL_PATTERNS),
        ("MTF Comparison", AnalysisType.FINAL_DECISION),
        ("Indicator Summary", AnalysisType.INDICATOR_SUMMARY)
    ]
    
    passed = 0
    for name, analysis_type in analysis_types:
        try:
            # Test curation
            curated = ce.curate_indicators(test_data, analysis_type)
            if isinstance(curated, dict):
                print(f"✓ {name} curation passed")
                
                # Test context structuring
                context = ce.structure_context(curated, analysis_type, "TEST", "1D", "")
                if isinstance(context, str) and len(context) > 0:
                    print(f"✓ {name} context structuring passed")
                    passed += 1
                else:
                    print(f"✗ {name} context structuring failed")
            else:
                print(f"✗ {name} curation failed")
        except Exception as e:
            print(f"✗ {name} failed with exception: {e}")
    
    print(f"\nBasic functionality: {passed}/{len(analysis_types)} passed")
    return passed == len(analysis_types)

def test_original_errors():
    """Test that the original errors are fixed."""
    print("\n🔍 Testing Original Error Fixes...")
    
    from gemini.context_engineer import ContextEngineer, AnalysisType
    
    # Data that would have caused the original errors
    problematic_data = {
        'close': ['100.0', '101.0', '102.0'],
        'volume': ['1000', '1100', '1200'],
        'rsi': ['50.0', '55.0', '60.0'],
        'macd': {
            'histogram': ['0.1', '0.2', '0.3']
        }
    }
    
    ce = ContextEngineer()
    
    original_errors = [
        {
            "name": "Pattern Analysis Type Error",
            "description": "Was: '<' not supported between instances of 'str' and 'int'",
            "test": lambda: ce.curate_indicators(problematic_data, AnalysisType.REVERSAL_PATTERNS)
        },
        {
            "name": "Volume Analysis JSON Error",
            "description": "Was: '\\n  \"volume_anomalies\"'",
            "test": lambda: ce.structure_context(
                ce.curate_indicators(problematic_data, AnalysisType.VOLUME_ANALYSIS),
                AnalysisType.VOLUME_ANALYSIS, "TEST", "1D", ""
            )
        },
        {
            "name": "MTF Comparison JSON Error",
            "description": "Was: '\\n  \"timeframe_analysis\"'",
            "test": lambda: ce.structure_context(
                ce.curate_indicators(problematic_data, AnalysisType.FINAL_DECISION),
                AnalysisType.FINAL_DECISION, "TEST", "1D", ""
            )
        }
    ]
    
    passed = 0
    for error_test in original_errors:
        try:
            result = error_test["test"]()
            print(f"✓ {error_test['name']}")
            print(f"  {error_test['description']}")
            passed += 1
        except Exception as e:
            print(f"✗ {error_test['name']}")
            print(f"  {error_test['description']}")
            print(f"  Still failing: {e}")
    
    print(f"\nOriginal error fixes: {passed}/{len(original_errors)} passed")
    return passed == len(original_errors)

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n🔍 Testing Edge Cases...")
    
    from gemini.context_engineer import ContextEngineer, AnalysisType
    
    ce = ContextEngineer()
    
    edge_cases = [
        {
            "name": "Empty data",
            "data": {},
            "analysis_type": AnalysisType.VOLUME_ANALYSIS
        },
        {
            "name": "Mixed valid/invalid data",
            "data": {
                'close': ['100.0', 'invalid', '102.0'],
                'volume': ['1000', 'not_number', '1200']
            },
            "analysis_type": AnalysisType.REVERSAL_PATTERNS
        },
        {
            "name": "Missing fields",
            "data": {
                'close': ['100.0', '101.0', '102.0']
            },
            "analysis_type": AnalysisType.FINAL_DECISION
        }
    ]
    
    passed = 0
    for case in edge_cases:
        try:
            curated = ce.curate_indicators(case["data"], case["analysis_type"])
            context = ce.structure_context(curated, case["analysis_type"], "TEST", "1D", "")
            
            if isinstance(context, str):
                print(f"✓ {case['name']} handled gracefully")
                passed += 1
            else:
                print(f"✗ {case['name']} failed")
        except Exception as e:
            print(f"✗ {case['name']} failed with exception: {e}")
    
    print(f"\nEdge cases: {passed}/{len(edge_cases)} passed")
    return passed == len(edge_cases)

def test_prompt_formatting():
    """Test that prompt formatting works with problematic context."""
    print("\n🔍 Testing Prompt Formatting...")
    
    from gemini.prompt_manager import PromptManager
    
    pm = PromptManager()
    
    # Test context with JSON that contains curly braces
    problematic_context = '{"test": "value with {curly} braces", "json": {"nested": "data"}}'
    
    try:
        result = pm.format_prompt('optimized_pattern_analysis', context=problematic_context)
        if isinstance(result, str) and len(result) > 0:
            print("✓ Prompt formatting with problematic context passed")
            return True
        else:
            print("✗ Prompt formatting failed - invalid result")
            return False
    except Exception as e:
        print(f"✗ Prompt formatting failed with exception: {e}")
        return False

def main():
    """Run all tests and provide a summary."""
    print("🚀 Simple Context Engineering Fix Test")
    print("=" * 50)
    
    # Run tests
    basic_ok = test_basic_functionality()
    errors_ok = test_original_errors()
    edge_ok = test_edge_cases()
    prompt_ok = test_prompt_formatting()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", basic_ok),
        ("Original Error Fixes", errors_ok),
        ("Edge Cases", edge_ok),
        ("Prompt Formatting", prompt_ok)
    ]
    
    passed = 0
    for name, result in tests:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:<20} {status}")
        if result:
            passed += 1
    
    print("-" * 50)
    print(f"Overall: {passed}/{len(tests)} test categories passed")
    
    if passed == len(tests):
        print("\n🎉 ALL TESTS PASSED!")
        print("The context engineering fixes are working correctly.")
        return True
    else:
        print("\n⚠️  Some tests failed.")
        print("Please review the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 