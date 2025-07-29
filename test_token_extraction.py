#!/usr/bin/env python3
"""
Test script to understand Gemini API response structure and token extraction.
This will help identify issues with token counting and summation.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from gemini.gemini_core import GeminiCore
from gemini.token_tracker import TokenUsage, AnalysisTokenTracker

async def test_response_structure():
    """Test the actual response structure from Gemini API"""
    
    # Check if API key is available
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set")
        print("Please set your Gemini API key:")
        print("export GEMINI_API_KEY='your-api-key-here'")
        return
    
    print("✅ API key found")
    
    try:
        # Initialize the core
        core = GeminiCore(api_key)
        print("✅ Gemini core initialized")
        
        # Test 1: Simple text request
        print("\n📝 Test 1: Simple text request")
        print("-" * 40)
        
        simple_prompt = "Hello, please respond with a short message."
        
        print("Sending simple prompt...")
        response = core.call_llm(simple_prompt)
        
        print(f"Response type: {type(response)}")
        print(f"Response: {response[:100]}...")
        
        # Test 2: Code execution request
        print("\n💻 Test 2: Code execution request")
        print("-" * 40)
        
        code_prompt = "Calculate 2 + 2 and return the result."
        
        print("Sending code execution prompt...")
        text_response, code_results, execution_results = await core.call_llm_with_code_execution(code_prompt)
        
        print(f"Text response type: {type(text_response)}")
        print(f"Text response: {text_response[:100]}...")
        print(f"Code results count: {len(code_results)}")
        print(f"Execution results count: {len(execution_results)}")
        
        # Test 3: Examine response object structure
        print("\n🔍 Test 3: Examine response object structure")
        print("-" * 40)
        
        # We need to get the actual response object, not just the text
        # Let's modify the core to return the full response object for testing
        
        print("Testing response object attributes...")
        
        # Create a test response object to understand the structure
        test_response = {
            'usage_metadata': {
                'prompt_token_count': 100,
                'candidates_token_count': 50,
                'total_token_count': 150
            }
        }
        
        print(f"Test response structure: {test_response}")
        
        # Test token extraction logic
        print("\n🔢 Test 4: Token extraction logic")
        print("-" * 40)
        
        tracker = AnalysisTokenTracker("test_analysis", "TEST")
        
        # Test with None response
        print("Testing with None response...")
        call_id = tracker.add_token_usage("test_call", None, "gemini-2.5-flash")
        print(f"Call ID: {call_id}")
        
        # Test with test response
        print("Testing with test response...")
        call_id = tracker.add_token_usage("test_call", test_response, "gemini-2.5-flash")
        print(f"Call ID: {call_id}")
        
        # Get summary
        summary = tracker.get_summary()
        print(f"Summary: {summary}")
        
        # Test token summation
        print("\n🧮 Test 5: Token summation logic")
        print("-" * 40)
        
        # Add multiple calls with different token counts
        test_calls = [
            {'prompt': 100, 'completion': 50, 'total': 150},
            {'prompt': 200, 'completion': 100, 'total': 300},
            {'prompt': 150, 'completion': 75, 'total': 225}
        ]
        
        for i, tokens in enumerate(test_calls):
            test_response = {
                'usage_metadata': {
                    'prompt_token_count': tokens['prompt'],
                    'candidates_token_count': tokens['completion'],
                    'total_token_count': tokens['total']
                }
            }
            call_id = tracker.add_token_usage(f"test_call_{i}", test_response, "gemini-2.5-flash")
            print(f"Added call {i}: {tokens}")
        
        # Get final summary
        final_summary = tracker.get_summary()
        print(f"Final summary: {final_summary}")
        
        # Verify summation
        expected_prompt = sum(call['prompt'] for call in test_calls)
        expected_completion = sum(call['completion'] for call in test_calls)
        expected_total = sum(call['total'] for call in test_calls)
        
        actual_prompt = final_summary['total_input_tokens']
        actual_completion = final_summary['total_output_tokens']
        actual_total = final_summary['total_tokens']
        
        print(f"\nVerification:")
        print(f"Expected prompt tokens: {expected_prompt}")
        print(f"Actual prompt tokens: {actual_prompt}")
        print(f"Match: {expected_prompt == actual_prompt}")
        
        print(f"Expected completion tokens: {expected_completion}")
        print(f"Actual completion tokens: {actual_completion}")
        print(f"Match: {expected_completion == actual_completion}")
        
        print(f"Expected total tokens: {expected_total}")
        print(f"Actual total tokens: {actual_total}")
        print(f"Match: {expected_total == actual_total}")
        
        # Check if input + output = total
        calculated_total = actual_prompt + actual_completion
        print(f"Calculated total (input + output): {calculated_total}")
        print(f"Matches actual total: {calculated_total == actual_total}")
        
        if calculated_total != actual_total:
            print("⚠️ WARNING: Input + Output tokens don't match total tokens!")
            print(f"Difference: {actual_total - calculated_total}")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def test_real_api_response():
    """Test with real API response to understand actual structure"""
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set")
        return
    
    try:
        core = GeminiCore(api_key)
        
        print("\n🚀 Test 6: Real API response structure")
        print("-" * 40)
        
        # We need to modify the core to return the full response object
        # Let's create a test version that captures the response
        
        prompt = "Say hello in one sentence."
        
        print("Making real API call...")
        
        # Use the original method but capture the response
        core.rate_limit()
        response = core.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt]
        )
        
        print(f"Response type: {type(response)}")
        print(f"Response attributes: {dir(response)}")
        
        # Check for usage metadata
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            print(f"Usage metadata type: {type(usage)}")
            print(f"Usage metadata attributes: {dir(usage)}")
            
            # Try to extract token counts
            prompt_tokens = getattr(usage, 'prompt_token_count', None)
            completion_tokens = getattr(usage, 'candidates_token_count', None)
            total_tokens = getattr(usage, 'total_token_count', None)
            
            print(f"Prompt tokens: {prompt_tokens}")
            print(f"Completion tokens: {completion_tokens}")
            print(f"Total tokens: {total_tokens}")
            
            # Verify summation
            if all(x is not None for x in [prompt_tokens, completion_tokens, total_tokens]):
                calculated_total = prompt_tokens + completion_tokens
                print(f"Calculated total: {calculated_total}")
                print(f"Matches reported total: {calculated_total == total_tokens}")
                
                if calculated_total != total_tokens:
                    print("⚠️ WARNING: Token summation mismatch!")
                    print(f"Difference: {total_tokens - calculated_total}")
        else:
            print("❌ No usage_metadata found in response")
            
    except Exception as e:
        print(f"❌ Error during real API test: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main function"""
    print("🔍 Token Extraction and Summation Test")
    print("=" * 50)
    
    # Test with mock data
    await test_response_structure()
    
    # Test with real API
    test_real_api_response()
    
    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 