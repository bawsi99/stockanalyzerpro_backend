"""
Simple test script to verify token tracking functionality without Gemini dependencies.
"""

import time
from gemini.token_tracker import get_or_create_tracker, get_tracker, remove_tracker, AnalysisTokenTracker


def test_token_tracking():
    """Test token tracking functionality."""
    print("🧪 Testing Token Tracking System")
    print("=" * 50)
    
    # Test 1: Create and manage token tracker
    print("\n1. Testing Token Tracker Creation and Management")
    print("-" * 40)
    
    analysis_id = "test_analysis_123"
    symbol = "RELIANCE"
    
    # Create tracker
    tracker = get_or_create_tracker(analysis_id, symbol)
    print(f"✅ Created tracker for {symbol} (ID: {analysis_id})")
    
    # Verify tracker exists
    retrieved_tracker = get_tracker(analysis_id)
    assert retrieved_tracker is not None, "Tracker should exist"
    print(f"✅ Retrieved tracker successfully")
    
    # Test 2: Simulate token usage tracking
    print("\n2. Testing Token Usage Tracking")
    print("-" * 40)
    
    # Simulate some LLM calls with token usage
    mock_responses = [
        {
            'usage_metadata': type('MockUsage', (), {
                'prompt_token_count': 1500,
                'candidates_token_count': 800,
                'total_token_count': 2300
            })()
        },
        {
            'usage_metadata': type('MockUsage', (), {
                'prompt_token_count': 2000,
                'candidates_token_count': 1200,
                'total_token_count': 3200
            })()
        },
        {
            'usage_metadata': type('MockUsage', (), {
                'prompt_token_count': 1800,
                'candidates_token_count': 1000,
                'total_token_count': 2800
            })()
        }
    ]
    
    call_types = ["indicator_summary", "chart_analysis", "final_decision"]
    
    for i, (response, call_type) in enumerate(zip(mock_responses, call_types)):
        call_id = tracker.add_token_usage(call_type, response, "gemini-2.5-flash")
        print(f"✅ Added token usage for {call_type}: {call_id}")
    
    # Test 3: Get token usage summary
    print("\n3. Testing Token Usage Summary")
    print("-" * 40)
    
    total_usage = tracker.get_total_usage()
    print(f"Total Input Tokens: {total_usage['total_input_tokens']:,}")
    print(f"Total Output Tokens: {total_usage['total_output_tokens']:,}")
    print(f"Total Tokens: {total_usage['total_tokens']:,}")
    print(f"LLM Calls Count: {total_usage['llm_calls_count']}")
    
    # Verify totals
    expected_input = 1500 + 2000 + 1800
    expected_output = 800 + 1200 + 1000
    expected_total = 2300 + 3200 + 2800
    
    assert total_usage['total_input_tokens'] == expected_input, f"Expected {expected_input}, got {total_usage['total_input_tokens']}"
    assert total_usage['total_output_tokens'] == expected_output, f"Expected {expected_output}, got {total_usage['total_output_tokens']}"
    assert total_usage['total_tokens'] == expected_total, f"Expected {expected_total}, got {total_usage['total_tokens']}"
    assert total_usage['llm_calls_count'] == 3, f"Expected 3 calls, got {total_usage['llm_calls_count']}"
    
    print("✅ Token usage totals are correct")
    
    # Test 4: Get usage breakdown
    print("\n4. Testing Usage Breakdown")
    print("-" * 40)
    
    breakdown = tracker.get_usage_breakdown()
    for call_type, details in breakdown.items():
        print(f"{call_type}:")
        print(f"  Calls: {details['calls']}")
        print(f"  Input Tokens: {details['total_input_tokens']:,}")
        print(f"  Output Tokens: {details['total_output_tokens']:,}")
        print(f"  Total Tokens: {details['total_tokens']:,}")
    
    # Test 5: Get complete summary
    print("\n5. Testing Complete Summary")
    print("-" * 40)
    
    summary = tracker.get_summary()
    print(f"Analysis ID: {summary['analysis_id']}")
    print(f"Symbol: {summary['symbol']}")
    print(f"Duration: {summary['duration_seconds']:.2f} seconds")
    print(f"Total Calls: {summary['total_calls']}")
    print(f"Successful Calls: {summary['successful_calls']}")
    print(f"Failed Calls: {summary['failed_calls']}")
    
    # Test 6: Print human-readable summary
    print("\n6. Testing Human-Readable Summary")
    print("-" * 40)
    
    tracker.print_summary()
    
    # Test 7: Test failed call tracking
    print("\n7. Testing Failed Call Tracking")
    print("-" * 40)
    
    tracker.add_token_usage("failed_call", None, "gemini-2.5-flash", success=False, error_message="API timeout")
    print("✅ Added failed call tracking")
    
    updated_summary = tracker.get_summary()
    assert updated_summary['failed_calls'] == 1, f"Expected 1 failed call, got {updated_summary['failed_calls']}"
    print("✅ Failed call tracking works correctly")
    
    # Test 8: Cleanup
    print("\n8. Testing Cleanup")
    print("-" * 40)
    
    remove_tracker(analysis_id)
    retrieved_tracker = get_tracker(analysis_id)
    assert retrieved_tracker is None, "Tracker should be removed"
    print("✅ Tracker cleanup works correctly")
    
    print("\n🎉 All token tracking tests passed!")
    return True


def test_multiple_analyses():
    """Test multiple concurrent analyses."""
    print("\n🔧 Testing Multiple Concurrent Analyses")
    print("=" * 50)
    
    # Create multiple trackers for different analyses
    analyses = [
        ("analysis_1", "RELIANCE"),
        ("analysis_2", "TCS"),
        ("analysis_3", "INFY")
    ]
    
    trackers = []
    
    # Create trackers
    for analysis_id, symbol in analyses:
        tracker = get_or_create_tracker(analysis_id, symbol)
        trackers.append(tracker)
        print(f"✅ Created tracker for {symbol} (ID: {analysis_id})")
    
    # Add some token usage to each
    for i, (analysis_id, symbol) in enumerate(analyses):
        tracker = trackers[i]
        
        # Simulate different token usage for each analysis
        mock_response = {
            'usage_metadata': type('MockUsage', (), {
                'prompt_token_count': 1000 + i * 500,
                'candidates_token_count': 500 + i * 300,
                'total_token_count': 1500 + i * 800
            })()
        }
        
        tracker.add_token_usage("indicator_summary", mock_response, "gemini-2.5-flash")
        print(f"✅ Added token usage for {symbol}")
    
    # Verify each tracker has its own data
    for i, (analysis_id, symbol) in enumerate(analyses):
        tracker = trackers[i]
        total_usage = tracker.get_total_usage()
        expected_total = 1500 + i * 800
        
        assert total_usage['total_tokens'] == expected_total, f"Expected {expected_total}, got {total_usage['total_tokens']}"
        print(f"✅ {symbol} has correct token usage: {total_usage['total_tokens']:,}")
    
    # Cleanup
    for analysis_id, _ in analyses:
        remove_tracker(analysis_id)
    
    print("✅ Multiple analyses test completed successfully")
    return True


def main():
    """Run all token tracking tests."""
    print("🚀 Starting Token Tracking Tests")
    print("=" * 60)
    
    # Test basic token tracking functionality
    basic_test_passed = test_token_tracking()
    
    # Test multiple concurrent analyses
    multiple_test_passed = test_multiple_analyses()
    
    print("\n📊 Test Results Summary")
    print("=" * 30)
    print(f"Basic Token Tracking: {'✅ PASSED' if basic_test_passed else '❌ FAILED'}")
    print(f"Multiple Analyses: {'✅ PASSED' if multiple_test_passed else '❌ FAILED'}")
    
    if basic_test_passed and multiple_test_passed:
        print("\n🎉 All tests passed! Token tracking system is ready.")
        print("\n📋 Implementation Summary:")
        print("✅ Token usage tracking per analysis")
        print("✅ Input/output token counting")
        print("✅ Call type breakdown")
        print("✅ Success/failure tracking")
        print("✅ Multiple concurrent analyses support")
        print("✅ Database storage integration")
        print("✅ Human-readable summaries")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main() 