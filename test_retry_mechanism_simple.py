#!/usr/bin/env python3
"""
Simplified test script for the retry mechanism.
This script tests the retry logic without importing the actual Gemini client.
"""

import asyncio
import time
import sys
import os

# Add the gemini directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'gemini'))

from error_utils import RetryMechanism, RetryConfig, GoogleAPIErrorHandler

class MockGeminiClient:
    """Mock client to simulate different error scenarios"""
    
    def __init__(self):
        self.call_count = 0
        self.error_scenario = "success"  # success, quota_error, server_error, network_error
    
    def set_error_scenario(self, scenario):
        self.error_scenario = scenario
        self.call_count = 0
    
    def simulate_api_call(self):
        """Simulate an API call that may fail based on the error scenario"""
        self.call_count += 1
        print(f"[MOCK] API call attempt {self.call_count}")
        
        if self.error_scenario == "success":
            return "Success response"
        elif self.error_scenario == "quota_error":
            raise Exception("Quota exceeded for this API key")
        elif self.error_scenario == "server_error":
            raise Exception("Internal server error occurred")
        elif self.error_scenario == "network_error":
            raise Exception("Connection timeout")
        elif self.error_scenario == "permanent_error":
            raise Exception("Invalid API key provided")
        else:
            return "Success response"

async def test_retry_mechanism():
    """Test the retry mechanism with different scenarios"""
    
    print("=== Testing Retry Mechanism ===\n")
    
    # Test 1: Success scenario (no retries needed)
    print("Test 1: Success scenario")
    mock_client = MockGeminiClient()
    mock_client.set_error_scenario("success")
    
    retry_config = RetryConfig(max_retries=3, base_delay=0.1, max_delay=1.0)
    retry_func = RetryMechanism.retry_with_backoff(
        mock_client.simulate_api_call, 
        retry_config, 
        "Success test"
    )
    
    try:
        result = retry_func()
        print(f"✅ Success: {result}")
        print(f"   Calls made: {mock_client.call_count}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    print()
    
    # Test 2: Quota error (should retry with longer delays)
    print("Test 2: Quota error scenario")
    mock_client.set_error_scenario("quota_error")
    
    start_time = time.time()
    try:
        result = retry_func()
        print(f"✅ Success after retries: {result}")
    except Exception as e:
        print(f"❌ Failed after retries: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"   Calls made: {mock_client.call_count}")
    print(f"   Time elapsed: {elapsed_time:.2f} seconds")
    
    print()
    
    # Test 3: Server error (should retry with moderate delays)
    print("Test 3: Server error scenario")
    mock_client.set_error_scenario("server_error")
    
    start_time = time.time()
    try:
        result = retry_func()
        print(f"✅ Success after retries: {result}")
    except Exception as e:
        print(f"❌ Failed after retries: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"   Calls made: {mock_client.call_count}")
    print(f"   Time elapsed: {elapsed_time:.2f} seconds")
    
    print()
    
    # Test 4: Network error (should retry aggressively)
    print("Test 4: Network error scenario")
    mock_client.set_error_scenario("network_error")
    
    start_time = time.time()
    try:
        result = retry_func()
        print(f"✅ Success after retries: {result}")
    except Exception as e:
        print(f"❌ Failed after retries: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"   Calls made: {mock_client.call_count}")
    print(f"   Time elapsed: {elapsed_time:.2f} seconds")
    
    print()
    
    # Test 5: Permanent error (should not retry)
    print("Test 5: Permanent error scenario")
    mock_client.set_error_scenario("permanent_error")
    
    start_time = time.time()
    try:
        result = retry_func()
        print(f"✅ Success after retries: {result}")
    except Exception as e:
        print(f"❌ Failed (expected): {e}")
    
    elapsed_time = time.time() - start_time
    print(f"   Calls made: {mock_client.call_count}")
    print(f"   Time elapsed: {elapsed_time:.2f} seconds")
    
    print()
    
    # Test 6: Error classification
    print("Test 6: Error classification")
    error_handler = GoogleAPIErrorHandler()
    
    test_errors = [
        Exception("Quota exceeded for this API key"),
        Exception("Internal server error occurred"),
        Exception("Connection timeout"),
        Exception("Invalid API key provided"),
        Exception("Some unknown error")
    ]
    
    for error in test_errors:
        error_type = error_handler.classify_error(error)
        print(f"   Error: '{str(error)}' -> Classified as: {error_type}")
    
    print()
    print("=== Retry Mechanism Test Complete ===")

async def test_async_retry():
    """Test async retry mechanism"""
    print("\n=== Testing Async Retry Mechanism ===\n")
    
    async def async_api_call():
        await asyncio.sleep(0.1)  # Simulate async work
        raise Exception("Async network timeout")
    
    retry_config = RetryConfig(max_retries=2, base_delay=0.1, max_delay=1.0)
    retry_func = RetryMechanism.async_retry_with_backoff(
        async_api_call, 
        retry_config, 
        "Async test"
    )
    
    start_time = time.time()
    try:
        result = await retry_func()
        print(f"✅ Async success: {result}")
    except Exception as e:
        print(f"❌ Async failed (expected): {e}")
    
    elapsed_time = time.time() - start_time
    print(f"   Time elapsed: {elapsed_time:.2f} seconds")

async def main():
    # Run synchronous tests
    await test_retry_mechanism()
    
    # Run async tests
    await test_async_retry()

if __name__ == "__main__":
    asyncio.run(main()) 