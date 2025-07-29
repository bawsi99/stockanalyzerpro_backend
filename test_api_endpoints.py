#!/usr/bin/env python3
"""
Test API Endpoints - Verify that the new backend API endpoints work correctly
"""

import os
import sys
import uuid
import requests
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

def test_api_endpoint(endpoint: str, method: str = "GET", data: dict = None):
    """Test a specific API endpoint."""
    try:
        url = f"{ANALYSIS_SERVICE_URL}{endpoint}"
        print(f"🔗 Testing {method} {url}")
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        else:
            print(f"❌ Unsupported method: {method}")
            return False
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result.get('success', False)}")
            if 'count' in result:
                print(f"   Count: {result['count']}")
            return True
        else:
            print(f"❌ Failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_user_analyses_endpoint():
    """Test the user analyses endpoint."""
    print("\n🔧 Testing User Analyses Endpoint")
    print("=" * 50)
    
    # Get an existing user
    supabase = get_supabase_client()
    user_result = supabase.table("profiles").select("id").limit(1).execute()
    if not user_result.data:
        print("❌ No users found for testing")
        return False
    
    user_id = user_result.data[0]["id"]
    print(f"✅ Using user: {user_id}")
    
    # Test the endpoint
    endpoint = f"/analyses/user/{user_id}?limit=10"
    return test_api_endpoint(endpoint)

def test_analysis_by_id_endpoint():
    """Test the analysis by ID endpoint."""
    print("\n🔧 Testing Analysis by ID Endpoint")
    print("=" * 50)
    
    # First, create a test analysis
    db_manager = DatabaseManager()
    supabase = get_supabase_client()
    
    user_result = supabase.table("profiles").select("id").limit(1).execute()
    if not user_result.data:
        print("❌ No users found for testing")
        return False
    
    user_id = user_result.data[0]["id"]
    
    # Create test analysis
    test_analysis = {
        "ai_analysis": {
            "trend": "Bullish",
            "confidence_pct": 75.0
        },
        "summary": {
            "overall_signal": "buy",
            "risk_level": "medium"
        },
        "metadata": {
            "current_price": 100.0,
            "price_change_pct": 2.5,
            "sector": "Technology"
        }
    }
    
    analysis_id = db_manager.store_analysis(
        analysis=test_analysis,
        user_id=user_id,
        symbol="TEST_API",
        exchange="TEST",
        period=30,
        interval="day"
    )
    
    if not analysis_id:
        print("❌ Failed to create test analysis")
        return False
    
    print(f"✅ Created test analysis: {analysis_id}")
    
    # Test the endpoint
    endpoint = f"/analyses/{analysis_id}"
    success = test_api_endpoint(endpoint)
    
    # Clean up
    try:
        supabase.table("stock_analyses").delete().eq("id", analysis_id).execute()
        print("✅ Cleaned up test analysis")
    except Exception as e:
        print(f"⚠️ Warning: Failed to clean up: {e}")
    
    return success

def test_analyses_by_signal_endpoint():
    """Test the analyses by signal endpoint."""
    print("\n🔧 Testing Analyses by Signal Endpoint")
    print("=" * 50)
    
    # Get an existing user
    supabase = get_supabase_client()
    user_result = supabase.table("profiles").select("id").limit(1).execute()
    if not user_result.data:
        print("❌ No users found for testing")
        return False
    
    user_id = user_result.data[0]["id"]
    
    # Test the endpoint
    endpoint = f"/analyses/signal/bullish?user_id={user_id}&limit=10"
    return test_api_endpoint(endpoint)

def test_analyses_by_sector_endpoint():
    """Test the analyses by sector endpoint."""
    print("\n🔧 Testing Analyses by Sector Endpoint")
    print("=" * 50)
    
    # Get an existing user
    supabase = get_supabase_client()
    user_result = supabase.table("profiles").select("id").limit(1).execute()
    if not user_result.data:
        print("❌ No users found for testing")
        return False
    
    user_id = user_result.data[0]["id"]
    
    # Test the endpoint
    endpoint = f"/analyses/sector/technology?user_id={user_id}&limit=10"
    return test_api_endpoint(endpoint)

def test_high_confidence_analyses_endpoint():
    """Test the high confidence analyses endpoint."""
    print("\n🔧 Testing High Confidence Analyses Endpoint")
    print("=" * 50)
    
    # Get an existing user
    supabase = get_supabase_client()
    user_result = supabase.table("profiles").select("id").limit(1).execute()
    if not user_result.data:
        print("❌ No users found for testing")
        return False
    
    user_id = user_result.data[0]["id"]
    
    # Test the endpoint
    endpoint = f"/analyses/confidence/70?user_id={user_id}&limit=10"
    return test_api_endpoint(endpoint)

def test_user_analysis_summary_endpoint():
    """Test the user analysis summary endpoint."""
    print("\n🔧 Testing User Analysis Summary Endpoint")
    print("=" * 50)
    
    # Get an existing user
    supabase = get_supabase_client()
    user_result = supabase.table("profiles").select("id").limit(1).execute()
    if not user_result.data:
        print("❌ No users found for testing")
        return False
    
    user_id = user_result.data[0]["id"]
    
    # Test the endpoint
    endpoint = f"/analyses/summary/user/{user_id}"
    return test_api_endpoint(endpoint)

def main():
    """Main test function."""
    print("🧪 API ENDPOINTS VERIFICATION TEST")
    print("=" * 60)
    
    tests = [
        ("User Analyses", test_user_analyses_endpoint),
        ("Analysis by ID", test_analysis_by_id_endpoint),
        ("Analyses by Signal", test_analyses_by_signal_endpoint),
        ("Analyses by Sector", test_analyses_by_sector_endpoint),
        ("High Confidence Analyses", test_high_confidence_analyses_endpoint),
        ("User Analysis Summary", test_user_analysis_summary_endpoint)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL API ENDPOINTS WORKING! Frontend-backend communication is ready.")
        print("✅ The frontend can now fetch stored analyses from the backend.")
    else:
        print("⚠️ Some API endpoints failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 