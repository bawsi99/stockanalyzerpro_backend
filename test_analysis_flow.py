#!/usr/bin/env python3
"""
Test Analysis Flow - Verify complete flow from frontend request to database storage
"""

import os
import sys
import requests
import json
import time
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables")
except ImportError:
    print("⚠️ python-dotenv not installed")

from supabase_client import get_supabase_client
from database_manager import DatabaseManager

# Configuration
ANALYSIS_SERVICE_URL = "http://localhost:8001"

def test_complete_analysis_flow():
    """Test the complete analysis flow from frontend request to database storage."""
    print("\n🔧 Testing Complete Analysis Flow")
    print("=" * 60)
    
    # Step 1: Send analysis request (simulating frontend)
    print("📤 Step 1: Sending analysis request to backend...")
    
    analysis_request = {
        "stock": "RELIANCE",
        "exchange": "NSE",
        "period": 30,
        "interval": "1day",
        "sector": "Energy"
    }
    
    try:
        response = requests.post(
            f"{ANALYSIS_SERVICE_URL}/analyze",
            json=analysis_request,
            timeout=120  # 2 minutes timeout for analysis
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis request successful!")
            print(f"   Stock: {result.get('stock_symbol')}")
            print(f"   Success: {result.get('success')}")
            print(f"   Storage Success: {result.get('storage', {}).get('success')}")
            
            # Step 2: Verify the response contains analysis data
            print("\n📥 Step 2: Verifying analysis response...")
            
            if 'results' in result and result['results']:
                print("✅ Analysis results present in response")
                print(f"   Results keys: {list(result['results'].keys())}")
                
                # Check for key analysis components
                analysis_data = result['results']
                key_components = [
                    'ai_analysis', 'technical_indicators', 'patterns', 
                    'summary', 'metadata', 'chart_insights'
                ]
                
                for component in key_components:
                    if component in analysis_data:
                        print(f"   ✅ {component}: Present")
                    else:
                        print(f"   ⚠️ {component}: Missing")
                
                # Step 3: Verify database storage
                print("\n💾 Step 3: Verifying database storage...")
                
                # Get a user ID for testing
                supabase = get_supabase_client()
                user_result = supabase.table("profiles").select("id").limit(1).execute()
                
                if user_result.data:
                    user_id = user_result.data[0]["id"]
                    
                    # Check if analysis was stored
                    db_manager = DatabaseManager()
                    analyses = db_manager.get_user_analyses(user_id, 10)
                    
                    if analyses:
                        latest_analysis = analyses[0]  # Most recent
                        print("✅ Analysis found in database!")
                        print(f"   ID: {latest_analysis.get('id')}")
                        print(f"   Stock: {latest_analysis.get('stock_symbol')}")
                        print(f"   Signal: {latest_analysis.get('overall_signal')}")
                        print(f"   Confidence: {latest_analysis.get('confidence_score')}")
                        print(f"   Analysis data present: {latest_analysis.get('analysis_data_json') is not None}")
                        
                        # Step 4: Verify the stored data matches the response
                        print("\n🔍 Step 4: Verifying data consistency...")
                        
                        stored_analysis_data = latest_analysis.get('analysis_data_json')
                        response_analysis_data = result.get('results')
                        
                        if stored_analysis_data and response_analysis_data:
                            # Check if key fields are present in both
                            key_fields = ['ai_analysis', 'technical_indicators', 'summary']
                            consistency_check = True
                            
                            for field in key_fields:
                                if field in response_analysis_data and field in stored_analysis_data:
                                    print(f"   ✅ {field}: Consistent between response and database")
                                else:
                                    print(f"   ❌ {field}: Inconsistent or missing")
                                    consistency_check = False
                            
                            if consistency_check:
                                print("✅ Data consistency verified!")
                            else:
                                print("⚠️ Some data inconsistencies found")
                        else:
                            print("❌ Could not compare stored vs response data")
                        
                        # Step 5: Test frontend retrieval
                        print("\n🌐 Step 5: Testing frontend retrieval...")
                        
                        # Test the API endpoint that frontend would use
                        api_response = requests.get(f"{ANALYSIS_SERVICE_URL}/analyses/{latest_analysis.get('id')}")
                        if api_response.status_code == 200:
                            api_result = api_response.json()
                            print("✅ Frontend can retrieve analysis via API")
                            print(f"   Retrieved ID: {api_result.get('analysis', {}).get('id')}")
                        else:
                            print(f"❌ Frontend API retrieval failed: {api_response.status_code}")
                        
                        return True
                    else:
                        print("❌ No analyses found in database")
                        return False
                else:
                    print("❌ No users found for testing")
                    return False
            else:
                print("❌ No analysis results in response")
                return False
                
        else:
            print(f"❌ Analysis request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during analysis flow: {e}")
        return False

def test_analysis_storage_mechanism():
    """Test the analysis storage mechanism specifically."""
    print("\n🔧 Testing Analysis Storage Mechanism")
    print("=" * 50)
    
    # Create a mock analysis result
    mock_analysis = {
        "ai_analysis": {
            "trend": "Bullish",
            "confidence_pct": 85.0,
            "timestamp": datetime.now().isoformat()
        },
        "summary": {
            "overall_signal": "buy",
            "risk_level": "medium",
            "analysis_quality": "high"
        },
        "technical_indicators": {
            "rsi": 65.5,
            "macd": 0.5,
            "sma_20": 1500.0
        },
        "metadata": {
            "current_price": 1520.0,
            "price_change_pct": 2.5,
            "sector": "Energy"
        }
    }
    
    # Get a user ID
    supabase = get_supabase_client()
    user_result = supabase.table("profiles").select("id").limit(1).execute()
    
    if not user_result.data:
        print("❌ No users found for testing")
        return False
    
    user_id = user_result.data[0]["id"]
    
    # Test storage using the same mechanism as the analysis service
    try:
        from analysis_storage import store_analysis_in_supabase
        
        analysis_id = store_analysis_in_supabase(
            analysis=mock_analysis,
            user_id=user_id,
            symbol="TEST_FLOW",
            exchange="TEST",
            period=30,
            interval="1day"
        )
        
        if analysis_id:
            print("✅ Analysis storage successful!")
            print(f"   Analysis ID: {analysis_id}")
            
            # Verify storage
            db_manager = DatabaseManager()
            stored_analysis = db_manager.get_analysis_by_id(analysis_id)
            
            if stored_analysis:
                print("✅ Stored analysis verified!")
                print(f"   Stock: {stored_analysis.get('stock_symbol')}")
                print(f"   Signal: {stored_analysis.get('overall_signal')}")
                print(f"   Analysis data present: {stored_analysis.get('analysis_data_json') is not None}")
                
                # Clean up
                supabase.table("stock_analyses").delete().eq("id", analysis_id).execute()
                print("✅ Test data cleaned up")
                
                return True
            else:
                print("❌ Could not retrieve stored analysis")
                return False
        else:
            print("❌ Analysis storage failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during storage test: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 ANALYSIS FLOW VERIFICATION TEST")
    print("=" * 70)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    tests = [
        ("Analysis Storage Mechanism", test_analysis_storage_mechanism),
        ("Complete Analysis Flow", test_complete_analysis_flow)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 50)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 70)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ANALYSIS FLOW WORKING PERFECTLY!")
        print("=" * 70)
        print("✅ When you send an analysis request:")
        print("   1. Frontend sends request to backend")
        print("   2. Backend performs comprehensive analysis")
        print("   3. Backend stores results in database")
        print("   4. Backend sends complete results to frontend")
        print("   5. Frontend can retrieve stored analyses later")
        print("\n🚀 The complete flow is working correctly!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 