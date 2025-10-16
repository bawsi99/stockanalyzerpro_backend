#!/usr/bin/env python3
"""
Test script for the new Market Structure Agent endpoint.

This script tests the new `/agents/market-structure/analyze` endpoint to ensure it works correctly
and integrates properly with the analysis pipeline.
"""

import asyncio
import sys
import os
import json
import httpx
from datetime import datetime

# Add the backend directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_market_structure_endpoint():
    """Test the new market structure endpoint"""
    
    print("🧪 Testing Market Structure Agent Endpoint")
    print("=" * 60)
    
    # Test configuration
    base_url = "http://localhost:8002"  # Analysis service URL
    endpoint = "/agents/market-structure/analyze"
    test_symbol = "RELIANCE"
    
    # Test request payload
    test_payload = {
        "symbol": test_symbol,
        "exchange": "NSE",
        "interval": "day",
        "period": 365,
        "context": "Testing the new standalone market structure endpoint",
        "include_charts": True,
        "include_llm_analysis": True,
        "return_prompt": False
    }
    
    try:
        print(f"📡 Making request to: {base_url}{endpoint}")
        print(f"📦 Payload: {json.dumps(test_payload, indent=2)}")
        print()
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            start_time = datetime.now()
            
            response = await client.post(
                f"{base_url}{endpoint}",
                json=test_payload
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"⏱️  Request completed in {duration:.2f} seconds")
            print(f"📊 Response Status: {response.status_code}")
            print()
            
            if response.status_code == 200:
                result = response.json()
                
                print("✅ SUCCESS - Market Structure Analysis Completed")
                print("=" * 60)
                print(f"🎯 Symbol: {result.get('symbol', 'N/A')}")
                print(f"⚡ Success: {result.get('success', False)}")
                print(f"🕒 Processing Time: {result.get('processing_time', 0):.2f}s")
                print(f"📈 Confidence Score: {result.get('confidence_score', 0)}")
                print(f"🤖 Agent: {result.get('agent', 'N/A')}")
                
                # Check LLM analysis
                llm_analysis = result.get('llm_analysis', {})
                print(f"🧠 LLM Analysis Success: {llm_analysis.get('success', False)}")
                print(f"🔍 Enhanced Insights: {llm_analysis.get('enhanced_insights', False)}")
                
                # Check chart info
                chart_info = result.get('chart_info', {})
                print(f"📊 Chart Generated: {chart_info.get('chart_generated', False)}")
                print(f"📏 Chart Size: {chart_info.get('chart_size_bytes', 0)} bytes")
                
                # Check key insights
                key_insights = result.get('key_insights', [])
                print(f"💡 Key Insights Count: {len(key_insights)}")
                if key_insights:
                    print("💡 Key Insights:")
                    for i, insight in enumerate(key_insights[:3], 1):  # Show first 3
                        print(f"   {i}. {insight}")
                    if len(key_insights) > 3:
                        print(f"   ... and {len(key_insights) - 3} more")
                
                # Check technical analysis
                technical_analysis = result.get('technical_analysis', {})
                if technical_analysis and not technical_analysis.get('error'):
                    print(f"🔧 Technical Analysis Available: Yes")
                    
                    # Check trend analysis
                    trend_analysis = technical_analysis.get('trend_analysis', {})
                    if trend_analysis:
                        trend_direction = trend_analysis.get('trend_direction', 'unknown')
                        trend_strength = trend_analysis.get('trend_strength', 'unknown')
                        print(f"📈 Trend: {trend_direction} ({trend_strength})")
                    
                    # Check BOS/CHOCH
                    bos_choch = technical_analysis.get('bos_choch_analysis', {})
                    if bos_choch:
                        structural_bias = bos_choch.get('structural_bias', 'unknown')
                        total_bos = bos_choch.get('total_bos_events', 0)
                        total_choch = bos_choch.get('total_choch_events', 0)
                        print(f"🏗️  Structure Bias: {structural_bias} ({total_bos} BOS, {total_choch} CHoCH)")
                    
                else:
                    print(f"🔧 Technical Analysis Available: No")
                
                # Check for decision integration fields
                decision_insights = result.get('market_structure_insights_for_decision', {})
                if decision_insights:
                    print(f"🎯 Decision Integration: Available")
                    print(f"   - Structural Bias: {decision_insights.get('structural_bias', 'unknown')}")
                    print(f"   - Trend Direction: {decision_insights.get('trend_direction', 'unknown')}")
                    print(f"   - Key Levels Count: {decision_insights.get('key_levels_count', 0)}")
                
                # Check metadata
                metadata = result.get('request_metadata', {})
                if metadata:
                    print(f"📋 Metadata:")
                    print(f"   - Used Prefetched Data: {metadata.get('used_prefetched_data', False)}")
                    print(f"   - Correlation ID: {metadata.get('correlation_id', 'None')}")
                    print(f"   - Include Charts: {metadata.get('include_charts', 'N/A')}")
                    print(f"   - Include LLM: {metadata.get('include_llm_analysis', 'N/A')}")
                
                print()
                print("✅ ENDPOINT TEST PASSED")
                return True
                
            else:
                print(f"❌ FAILURE - HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"Error: {error_data.get('error', 'Unknown error')}")
                    print(f"Detail: {json.dumps(error_data, indent=2)}")
                except:
                    print(f"Response Text: {response.text}")
                return False
                
    except httpx.ConnectError:
        print("❌ CONNECTION ERROR")
        print("Make sure the analysis service is running on http://localhost:8002")
        print("You can start it with: python start_analysis_service.py")
        return False
        
    except httpx.TimeoutException:
        print("❌ TIMEOUT ERROR")
        print("The request took too long to complete (>120s)")
        return False
        
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_endpoint_with_correlation_id():
    """Test the endpoint with correlation_id for cache integration"""
    
    print("\n🧪 Testing Market Structure Endpoint with Correlation ID")
    print("=" * 60)
    
    base_url = "http://localhost:8002"
    endpoint = "/agents/market-structure/analyze"
    test_symbol = "TCS"
    correlation_id = "test_correlation_12345"
    
    test_payload = {
        "symbol": test_symbol,
        "exchange": "NSE",
        "interval": "day",
        "period": 180,
        "context": "Testing correlation ID integration",
        "correlation_id": correlation_id,
        "include_charts": False,  # Faster test
        "include_llm_analysis": False,  # Faster test
        "return_prompt": False
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{base_url}{endpoint}",
                json=test_payload
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Correlation ID Test: {result.get('success', False)}")
                print(f"🕒 Processing Time: {result.get('processing_time', 0):.2f}s")
                
                metadata = result.get('request_metadata', {})
                print(f"🔗 Correlation ID: {metadata.get('correlation_id', 'None')}")
                print(f"📦 Used Prefetched Data: {metadata.get('used_prefetched_data', False)}")
                
                return result.get('success', False)
            else:
                print(f"❌ Correlation ID Test Failed: HTTP {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Correlation ID Test Error: {e}")
        return False

def run_tests():
    """Run all tests"""
    print("🚀 Market Structure Agent Endpoint Test Suite")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run async tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Test 1: Basic endpoint functionality
        test1_passed = loop.run_until_complete(test_market_structure_endpoint())
        
        # Test 2: Correlation ID integration
        test2_passed = loop.run_until_complete(test_endpoint_with_correlation_id())
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Basic Endpoint Test: {'PASSED' if test1_passed else 'FAILED'}")
        print(f"✅ Correlation ID Test: {'PASSED' if test2_passed else 'FAILED'}")
        
        overall_success = test1_passed and test2_passed
        print(f"\n🎯 Overall Result: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'}")
        
        if overall_success:
            print("\n🎉 The Market Structure Agent endpoint is working correctly!")
            print("   You can now use it in production at: /agents/market-structure/analyze")
        else:
            print("\n⚠️  Some tests failed. Please check the implementation.")
            
        return overall_success
        
    finally:
        loop.close()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)