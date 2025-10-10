#!/usr/bin/env python3
"""
Test Pattern Analysis Integration

This script tests the Pattern LLM Agent integration with the analysis service.
It verifies:
1. Pattern LLM Agent can be imported and instantiated
2. Pattern analyze-all endpoint can be called
3. End-to-end integration works
"""

import asyncio
import sys
import os
import httpx
import json

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_pattern_llm_agent_import():
    """Test 1: Can we import and instantiate the Pattern LLM Agent?"""
    print("🧪 Test 1: Pattern LLM Agent Import")
    try:
        from agents.patterns.pattern_llm_agent import PatternLLMAgent
        agent = PatternLLMAgent(gemini_client=None)
        print(f"✅ Successfully imported and created PatternLLMAgent")
        print(f"   Agent name: {agent.name}")
        print(f"   Version: {agent.version}")
        return True
    except Exception as e:
        print(f"❌ Failed to import PatternLLMAgent: {e}")
        return False

async def test_pattern_context_builder():
    """Test 2: Can we import and use the Pattern Context Builder?"""
    print("\n🧪 Test 2: Pattern Context Builder")
    try:
        from agents.patterns.pattern_context_builder import PatternContextBuilder
        builder = PatternContextBuilder()
        
        # Test with mock data
        mock_pattern_data = {
            "pattern_results": {
                "bullish_flag": {"confidence": 0.8, "entry_level": 100},
                "ascending_triangle": {"confidence": 0.7, "breakout_target": 110}
            },
            "overall_confidence": 0.75
        }
        
        context = builder.build_comprehensive_pattern_context(
            mock_pattern_data, "TESTSTOCK", 105
        )
        
        print(f"✅ Successfully created pattern context")
        print(f"   Context length: {len(context)} characters")
        print(f"   Contains pattern results: {'pattern_results' in context}")
        return True
    except Exception as e:
        print(f"❌ Failed to test PatternContextBuilder: {e}")
        return False

async def test_pattern_endpoint_availability():
    """Test 3: Is the pattern analyze-all endpoint available?"""
    print("\n🧪 Test 3: Pattern Endpoint Availability")
    try:
        # Try to connect to the analysis service
        async with httpx.AsyncClient(timeout=5.0) as client:
            # First check if the service is running
            try:
                health_resp = await client.get("http://localhost:8002/health")
                if health_resp.status_code == 200:
                    print("✅ Analysis service is running")
                else:
                    print("⚠️ Analysis service responded but not healthy")
                    return False
            except httpx.ConnectError:
                print("❌ Analysis service is not running")
                print("   Please start the analysis service with: cd backend && python services/analysis_service.py")
                return False
            
            # Check if patterns endpoint responds (we expect it to fail with validation error for empty request)
            try:
                patterns_resp = await client.post(
                    "http://localhost:8002/agents/patterns/analyze-all",
                    json={}  # Empty request should return validation error
                )
                # We expect a 422 validation error for missing required fields
                if patterns_resp.status_code == 422:
                    print("✅ Pattern analyze-all endpoint exists and validates requests")
                    return True
                else:
                    print(f"⚠️ Pattern endpoint responded unexpectedly: {patterns_resp.status_code}")
                    return False
            except Exception as e:
                print(f"❌ Pattern endpoint test failed: {e}")
                return False
    except Exception as e:
        print(f"❌ Failed to test pattern endpoint: {e}")
        return False

async def test_pattern_endpoint_with_valid_request():
    """Test 4: Can we make a valid request to the patterns endpoint?"""
    print("\n🧪 Test 4: Pattern Endpoint with Valid Request")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Make a valid request to the patterns endpoint
            request_data = {
                "symbol": "RELIANCE",
                "exchange": "NSE",
                "interval": "day",
                "period": 365,
                "context": "Test pattern analysis request"
            }
            
            print(f"   Making request to patterns endpoint...")
            print(f"   Symbol: {request_data['symbol']}")
            
            patterns_resp = await client.post(
                "http://localhost:8002/agents/patterns/analyze-all",
                json=request_data
            )
            
            print(f"   Response status: {patterns_resp.status_code}")
            
            if patterns_resp.status_code == 200:
                result = patterns_resp.json()
                success = result.get("success", False)
                print(f"✅ Pattern analysis completed successfully: {success}")
                if success:
                    print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
                    pattern_summary = result.get('pattern_summary', {})
                    print(f"   Confidence: {pattern_summary.get('overall_confidence', 0):.2%}")
                    print(f"   Patterns detected: {pattern_summary.get('patterns_detected', 0)}")
                return success
            else:
                print(f"❌ Pattern endpoint failed: {patterns_resp.status_code}")
                try:
                    error_detail = patterns_resp.json()
                    print(f"   Error: {error_detail}")
                except:
                    print(f"   Error text: {patterns_resp.text[:200]}")
                return False
    except Exception as e:
        print(f"❌ Failed to test pattern endpoint with valid request: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Testing Pattern Analysis Integration")
    print("=" * 60)
    
    tests = [
        test_pattern_llm_agent_import,
        test_pattern_context_builder,
        test_pattern_endpoint_availability,
        test_pattern_endpoint_with_valid_request
    ]
    
    results = []
    for i, test in enumerate(tests, 1):
        result = await test()
        results.append(result)
        if i < len(tests):  # Don't add separator after last test
            print("\n" + "-" * 40)
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, result in enumerate(results, 1):
        status = "✅ PASS" if result else "❌ FAIL"
        test_name = tests[i-1].__name__.replace("test_", "").replace("_", " ").title()
        print(f"Test {i}: {test_name:<30} {status}")
    
    print("-" * 60)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pattern analysis integration is ready.")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)