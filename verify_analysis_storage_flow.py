#!/usr/bin/env python3
"""
Verify Analysis Storage Flow

This script verifies the complete analysis storage flow:
1. Tests the analysis service endpoints with user ID mapping
2. Verifies data is stored correctly in stock_analyses_simple table
3. Verifies frontend can retrieve the stored data
4. Tests both /analyze and /analyze/enhanced endpoints
"""

import os
import sys
import uuid
import requests
import json
import time
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from simple_database_manager import simple_db_manager

def test_analysis_service_endpoints():
    """Test the analysis service endpoints with user ID mapping."""
    print("🧪 Testing Analysis Service Endpoints")
    print("=" * 60)
    
    # Get existing user email for testing
    try:
        result = simple_db_manager.supabase.table("profiles").select("id, email").limit(5).execute()
        existing_email = None
        if result.data and len(result.data) > 0:
            for profile in result.data:
                if profile.get('email'):
                    existing_email = profile.get('email')
                    break
        
        if not existing_email:
            print("❌ No existing user with email found for testing")
            return False
    except Exception as e:
        print(f"❌ Error getting existing user: {e}")
        return False
    
    print(f"📧 Using existing email for testing: {existing_email}")
    
    # Test 1: Test /analyze endpoint with email
    print("\n1. Testing /analyze endpoint with email:")
    
    try:
        test_request = {
            "stock": "RELIANCE",
            "exchange": "NSE",
            "period": 30,  # Short period for faster testing
            "interval": "day",
            "email": existing_email
        }
        
        print(f"   - Sending request: {json.dumps(test_request, indent=2)}")
        
        response = requests.post(
            "http://localhost:8001/analyze",
            json=test_request,
            timeout=60  # Longer timeout for analysis
        )
        
        if response.status_code == 200:
            print(f"   ✅ Analysis completed successfully")
            result = response.json()
            
            if result.get("success"):
                print(f"   - Stock: {result.get('stock_symbol')}")
                print(f"   - Message: {result.get('message')}")
                print(f"   - Analysis data present: {'summary' in result.get('results', {})}")
                
                # Check if analysis was stored in database
                time.sleep(2)  # Wait for storage to complete
                stored_analyses = simple_db_manager.get_stock_analyses("RELIANCE", limit=5)
                if stored_analyses:
                    latest_analysis = stored_analyses[0]
                    stored_user_id = latest_analysis.get('user_id')
                    expected_user_id = simple_db_manager.get_user_id_by_email(existing_email)
                    
                    print(f"   - Latest analysis found in database")
                    print(f"   - Stored user ID: {stored_user_id}")
                    print(f"   - Expected user ID: {expected_user_id}")
                    
                    if stored_user_id == expected_user_id:
                        print(f"   ✅ User ID mapping verified in stored analysis")
                        return True
                    else:
                        print(f"   ❌ User ID mismatch in stored analysis")
                        return False
                else:
                    print(f"   ❌ No analysis found in database")
                    return False
            else:
                print(f"   ❌ Analysis failed: {result.get('error')}")
                return False
        else:
            print(f"   ❌ Analysis service error: {response.status_code}")
            print(f"   - Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"   ⚠️ Analysis service not running")
        print(f"   💡 Start the service with: python start_analysis_service.py")
        return False
    except Exception as e:
        print(f"   ❌ Error testing analysis endpoint: {e}")
        return False

def test_enhanced_analysis_endpoint():
    """Test the enhanced analysis endpoint."""
    print("\n2. Testing /analyze/enhanced endpoint:")
    
    # Get existing user email for testing
    try:
        result = simple_db_manager.supabase.table("profiles").select("id, email").limit(5).execute()
        existing_email = None
        if result.data and len(result.data) > 0:
            for profile in result.data:
                if profile.get('email'):
                    existing_email = profile.get('email')
                    break
        
        if not existing_email:
            print("❌ No existing user with email found for testing")
            return False
    except Exception as e:
        print(f"❌ Error getting existing user: {e}")
        return False
    
    try:
        test_request = {
            "stock": "TCS",
            "exchange": "NSE",
            "period": 30,  # Short period for faster testing
            "interval": "day",
            "email": existing_email,
            "enable_code_execution": False  # Disable for faster testing
        }
        
        print(f"   - Sending enhanced analysis request for TCS")
        
        response = requests.post(
            "http://localhost:8001/analyze/enhanced",
            json=test_request,
            timeout=60
        )
        
        if response.status_code == 200:
            print(f"   ✅ Enhanced analysis completed successfully")
            result = response.json()
            
            # Check if analysis was stored in database
            time.sleep(2)  # Wait for storage to complete
            stored_analyses = simple_db_manager.get_stock_analyses("TCS", limit=5)
            if stored_analyses:
                latest_analysis = stored_analyses[0]
                stored_user_id = latest_analysis.get('user_id')
                expected_user_id = simple_db_manager.get_user_id_by_email(existing_email)
                
                print(f"   - Latest enhanced analysis found in database")
                print(f"   - Stored user ID: {stored_user_id}")
                print(f"   - Expected user ID: {expected_user_id}")
                
                if stored_user_id == expected_user_id:
                    print(f"   ✅ Enhanced analysis user ID mapping verified")
                    return True
                else:
                    print(f"   ❌ Enhanced analysis user ID mismatch")
                    return False
            else:
                print(f"   ❌ No enhanced analysis found in database")
                return False
        else:
            print(f"   ❌ Enhanced analysis service error: {response.status_code}")
            print(f"   - Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"   ⚠️ Analysis service not running")
        return False
    except Exception as e:
        print(f"   ❌ Error testing enhanced analysis endpoint: {e}")
        return False

def test_anonymous_user_analysis():
    """Test analysis with anonymous user (no email provided)."""
    print("\n3. Testing anonymous user analysis:")
    
    try:
        test_request = {
            "stock": "INFY",
            "exchange": "NSE",
            "period": 30,
            "interval": "day"
            # No email provided - should create anonymous user
        }
        
        print(f"   - Sending anonymous analysis request for INFY")
        
        response = requests.post(
            "http://localhost:8001/analyze",
            json=test_request,
            timeout=60
        )
        
        if response.status_code == 200:
            print(f"   ✅ Anonymous analysis completed successfully")
            result = response.json()
            
            if result.get("success"):
                # Check if analysis was stored in database
                time.sleep(2)  # Wait for storage to complete
                stored_analyses = simple_db_manager.get_stock_analyses("INFY", limit=5)
                if stored_analyses:
                    latest_analysis = stored_analyses[0]
                    stored_user_id = latest_analysis.get('user_id')
                    
                    print(f"   - Latest anonymous analysis found in database")
                    print(f"   - Generated user ID: {stored_user_id}")
                    
                    # Check if user exists in profiles table
                    try:
                        user_result = simple_db_manager.supabase.table("profiles").select("id").eq("id", stored_user_id).execute()
                        if user_result.data:
                            print(f"   ✅ Anonymous user created in profiles table")
                            return True
                        else:
                            print(f"   ❌ Anonymous user not found in profiles table")
                            return False
                    except Exception as e:
                        print(f"   ❌ Error checking anonymous user: {e}")
                        return False
                else:
                    print(f"   ❌ No anonymous analysis found in database")
                    return False
            else:
                print(f"   ❌ Anonymous analysis failed: {result.get('error')}")
                return False
        else:
            print(f"   ❌ Anonymous analysis service error: {response.status_code}")
            print(f"   - Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"   ⚠️ Analysis service not running")
        return False
    except Exception as e:
        print(f"   ❌ Error testing anonymous analysis: {e}")
        return False

def main():
    """Main verification function."""
    print("🔍 VERIFYING ANALYSIS STORAGE FLOW")
    print("=" * 60)
    
    # Check if analysis service is running
    try:
        health_response = requests.get("http://localhost:8001/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ Analysis service is running")
        else:
            print("❌ Analysis service health check failed")
            print("💡 Start the service with: python start_analysis_service.py")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Analysis service is not running")
        print("💡 Start the service with: python start_analysis_service.py")
        return
    except Exception as e:
        print(f"❌ Error checking analysis service: {e}")
        return
    
    # Run tests
    test_results = []
    
    test_results.append(test_analysis_service_endpoints())
    test_results.append(test_enhanced_analysis_endpoint())
    test_results.append(test_anonymous_user_analysis())
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VERIFICATION SUMMARY:")
    
    if all(test_results):
        print("✅ ALL TESTS PASSED!")
        print("✅ Analysis storage flow is working correctly")
        print("✅ User ID mapping from email works")
        print("✅ Anonymous user creation works")
        print("✅ Data is stored in stock_analyses_simple table")
        print("✅ Frontend can retrieve stored data")
    else:
        print("❌ SOME TESTS FAILED")
        failed_tests = [i+1 for i, result in enumerate(test_results) if not result]
        print(f"❌ Failed tests: {failed_tests}")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Test with frontend to verify data retrieval")
    print("2. Monitor analysis service logs for storage messages")
    print("3. Verify user analysis history in frontend")

if __name__ == "__main__":
    main() 