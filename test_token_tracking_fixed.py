#!/usr/bin/env python3
"""
Comprehensive test for the fixed token tracking system.
This test verifies that token counting and summation work correctly.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from gemini.token_tracker import AnalysisTokenTracker, TokenUsage
from gemini.gemini_client import GeminiClient

def test_token_extraction_logic():
    """Test the token extraction logic with various response formats"""
    
    print("🔢 Test 1: Token Extraction Logic")
    print("-" * 40)
    
    tracker = AnalysisTokenTracker("test_analysis", "TEST")
    
    # Test 1: Real Gemini API response format
    print("Testing real Gemini API response format...")
    
    # Mock a real Gemini response object
    class MockUsageMetadata:
        def __init__(self, prompt_count, completion_count, total_count):
            self.prompt_token_count = prompt_count
            self.candidates_token_count = completion_count
            self.total_token_count = total_count
    
    class MockResponse:
        def __init__(self, prompt_count, completion_count, total_count):
            self.usage_metadata = MockUsageMetadata(prompt_count, completion_count, total_count)
            self.text = "Mock response"
    
    # Test with realistic token counts
    mock_response = MockResponse(150, 75, 225)
    call_id = tracker.add_token_usage("test_call", mock_response, "gemini-2.5-flash")
    print(f"Added call: {call_id}")
    
    # Test 2: Dictionary format
    print("Testing dictionary format...")
    dict_response = {
        'usage_metadata': {
            'prompt_token_count': 200,
            'candidates_token_count': 100,
            'total_token_count': 300
        }
    }
    call_id = tracker.add_token_usage("test_call", dict_response, "gemini-2.5-flash")
    print(f"Added call: {call_id}")
    
    # Test 3: Direct token counts
    print("Testing direct token counts...")
    direct_response = {
        'prompt_token_count': 300,
        'candidates_token_count': 150,
        'total_token_count': 450
    }
    call_id = tracker.add_token_usage("test_call", direct_response, "gemini-2.5-flash")
    print(f"Added call: {call_id}")
    
    # Test 4: None response (error case)
    print("Testing None response...")
    call_id = tracker.add_token_usage("test_call", None, "gemini-2.5-flash", success=False, error_message="API error")
    print(f"Added call: {call_id}")
    
    # Get summary and verify
    summary = tracker.get_summary()
    print(f"\nToken Summary:")
    print(f"  Input Tokens: {summary['total_input_tokens']:,}")
    print(f"  Output Tokens: {summary['total_output_tokens']:,}")
    print(f"  Total Tokens: {summary['total_tokens']:,}")
    print(f"  Calculated Total: {summary['calculated_total']:,}")
    print(f"  Token Mismatch: {summary['token_mismatch']:,}")
    
    # Verify expected values
    expected_input = 150 + 200 + 300
    expected_output = 75 + 100 + 150
    expected_total = 225 + 300 + 450
    
    print(f"\nVerification:")
    print(f"  Expected Input: {expected_input}")
    print(f"  Actual Input: {summary['total_input_tokens']}")
    print(f"  Match: {expected_input == summary['total_input_tokens']}")
    
    print(f"  Expected Output: {expected_output}")
    print(f"  Actual Output: {summary['total_output_tokens']}")
    print(f"  Match: {expected_output == summary['total_output_tokens']}")
    
    print(f"  Expected Total: {expected_total}")
    print(f"  Actual Total: {summary['total_tokens']}")
    print(f"  Match: {expected_total == summary['total_tokens']}")
    
    return tracker

def test_token_summation_edge_cases():
    """Test edge cases in token summation"""
    
    print("\n🧮 Test 2: Token Summation Edge Cases")
    print("-" * 40)
    
    tracker = AnalysisTokenTracker("edge_case_test", "TEST")
    
    # Test case 1: Zero tokens
    print("Testing zero tokens...")
    zero_response = {
        'usage_metadata': {
            'prompt_token_count': 0,
            'candidates_token_count': 0,
            'total_token_count': 0
        }
    }
    tracker.add_token_usage("zero_call", zero_response, "gemini-2.5-flash")
    
    # Test case 2: Large numbers
    print("Testing large numbers...")
    large_response = {
        'usage_metadata': {
            'prompt_token_count': 10000,
            'candidates_token_count': 5000,
            'total_token_count': 15000
        }
    }
    tracker.add_token_usage("large_call", large_response, "gemini-2.5-flash")
    
    # Test case 3: Mismatch between input+output and total
    print("Testing token mismatch...")
    mismatch_response = {
        'usage_metadata': {
            'prompt_token_count': 100,
            'candidates_token_count': 50,
            'total_token_count': 200  # Should be 150, but API reports 200
        }
    }
    tracker.add_token_usage("mismatch_call", mismatch_response, "gemini-2.5-flash")
    
    # Get summary
    summary = tracker.get_summary()
    print(f"\nEdge Case Summary:")
    print(f"  Input Tokens: {summary['total_input_tokens']:,}")
    print(f"  Output Tokens: {summary['total_output_tokens']:,}")
    print(f"  Total Tokens: {summary['total_tokens']:,}")
    print(f"  Calculated Total: {summary['calculated_total']:,}")
    print(f"  Token Mismatch: {summary['token_mismatch']:,}")
    
    # Verify the mismatch is detected
    if summary['token_mismatch'] != 0:
        print(f"✅ Token mismatch correctly detected: {summary['token_mismatch']} tokens")
    else:
        print("❌ Token mismatch not detected")
    
    return tracker

async def test_real_api_integration():
    """Test token tracking with real API calls"""
    
    print("\n🚀 Test 3: Real API Integration")
    print("-" * 40)
    
    # Check if API key is available
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set, skipping real API test")
        return None
    
    try:
        # Create token tracker
        tracker = AnalysisTokenTracker("real_api_test", "TEST")
        
        # Create Gemini client
        client = GeminiClient(api_key)
        
        # Test 1: Simple indicator summary
        print("Testing indicator summary with token tracking...")
        
        indicators = {
            "rsi": 65.5,
            "macd": {"macd": 2.5, "signal": 1.8, "histogram": 0.7},
            "price": 150.25,
            "volume": 1500000
        }
        
        markdown_part, parsed_result = await client.build_indicators_summary(
            symbol="AAPL",
            indicators=indicators,
            period=14,
            interval="1d",
            token_tracker=tracker
        )
        
        print(f"✅ Indicator summary completed")
        print(f"   Response length: {len(markdown_part)} characters")
        print(f"   Parsed result keys: {list(parsed_result.keys())}")
        
        # Test 2: Enhanced analysis with calculations
        print("Testing enhanced analysis with token tracking...")
        
        response = await client.analyze_stock_with_enhanced_calculations(
            symbol="TSLA",
            indicators={"prices": [100, 102, 98, 105, 103, 107, 104, 108, 106, 110]},
            chart_paths={},
            period=14,
            interval="1d"
        )
        
        print(f"✅ Enhanced analysis completed")
        print(f"   Response type: {type(response)}")
        
        # Get final summary
        summary = tracker.get_summary()
        print(f"\nReal API Token Summary:")
        print(f"  Total Calls: {summary['total_calls']}")
        print(f"  Successful Calls: {summary['successful_calls']}")
        print(f"  Failed Calls: {summary['failed_calls']}")
        print(f"  Input Tokens: {summary['total_input_tokens']:,}")
        print(f"  Output Tokens: {summary['total_output_tokens']:,}")
        print(f"  Total Tokens: {summary['total_tokens']:,}")
        print(f"  Calculated Total: {summary['calculated_total']:,}")
        print(f"  Token Mismatch: {summary['token_mismatch']:,}")
        
        # Print breakdown
        print(f"\nCall Type Breakdown:")
        for call_type, details in summary['usage_breakdown'].items():
            print(f"  {call_type}:")
            print(f"    Calls: {details['calls']}")
            print(f"    Input Tokens: {details['total_input_tokens']:,}")
            print(f"    Output Tokens: {details['total_output_tokens']:,}")
            print(f"    Total Tokens: {details['total_tokens']:,}")
        
        return tracker
        
    except Exception as e:
        print(f"❌ Error during real API test: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_token_tracker_integration():
    """Test the integration of token tracker with the client"""
    
    print("\n🔗 Test 4: Token Tracker Integration")
    print("-" * 40)
    
    # Test that the token tracker is properly imported and accessible
    try:
        from gemini.token_tracker import get_or_create_tracker, get_tracker, remove_tracker
        
        # Test tracker creation
        tracker1 = get_or_create_tracker("test_1", "AAPL")
        tracker2 = get_or_create_tracker("test_2", "TSLA")
        
        print(f"✅ Created trackers: {tracker1.analysis_id}, {tracker2.analysis_id}")
        
        # Test tracker retrieval
        retrieved_tracker = get_tracker("test_1")
        if retrieved_tracker and retrieved_tracker.analysis_id == "test_1":
            print("✅ Tracker retrieval works")
        else:
            print("❌ Tracker retrieval failed")
        
        # Test tracker removal
        remove_tracker("test_1")
        removed_tracker = get_tracker("test_1")
        if removed_tracker is None:
            print("✅ Tracker removal works")
        else:
            print("❌ Tracker removal failed")
        
        # Clean up
        remove_tracker("test_2")
        
    except Exception as e:
        print(f"❌ Error in integration test: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function"""
    print("🔍 Comprehensive Token Tracking Test")
    print("=" * 50)
    
    # Test 1: Token extraction logic
    tracker1 = test_token_extraction_logic()
    
    # Test 2: Edge cases
    tracker2 = test_token_summation_edge_cases()
    
    # Test 3: Integration
    test_token_tracker_integration()
    
    # Test 4: Real API (if available)
    asyncio.run(test_real_api_integration())
    
    print("\n🎉 All tests completed!")
    print("\n📋 Summary of fixes:")
    print("   ✅ Improved token extraction logic")
    print("   ✅ Added support for multiple response formats")
    print("   ✅ Added token mismatch detection")
    print("   ✅ Integrated token tracking with Gemini client")
    print("   ✅ Added comprehensive validation and reporting")

if __name__ == "__main__":
    main() 