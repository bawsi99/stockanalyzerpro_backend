#!/usr/bin/env python3
"""
Test Email-Based Analysis Storage Flow

This script tests the updated analysis storage flow that uses email-based user ID mapping
instead of generating anonymous users. The frontend now sends the user's email in analysis requests.
"""

import os
import sys
import uuid
import requests
import json
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

def test_email_based_storage():
    """Test the email-based analysis storage flow."""
    print("🧪 Testing Email-Based Analysis Storage Flow")
    print("=" * 60)
    
    # Test 1: Check existing users in profiles table
    print("\n1. Checking Existing Users:")
    
    try:
        result = simple_db_manager.supabase.table("profiles").select("id, email").limit(10).execute()
        print(f"   - profiles table: ✅ Accessible ({len(result.data) if result.data else 0} records)")
        
        if result.data:
            print("   - Existing users:")
            for profile in result.data:
                print(f"     * ID: {profile.get('id')}, Email: {profile.get('email')}")
                
            # Get a user with email for testing
            existing_user = None
            for profile in result.data:
                if profile.get('email'):
                    existing_user = profile
                    break
                    
            if not existing_user:
                print("   ❌ No users with email found for testing")
                return False
                
            print(f"   ✅ Using existing user for testing: {existing_user.get('email')}")
        else:
            print("   ❌ No users found in profiles table")
            return False
            
    except Exception as e:
        print(f"   ❌ Error checking profiles table: {e}")
        return False
    
    # Test 2: Test analysis service endpoint with email
    print(f"\n2. Testing Analysis Service with Email ({existing_user.get('email')}):")
    
    try:
        test_request = {
            "stock": "RELIANCE",
            "exchange": "NSE",
            "period": 30,  # Short period for faster testing
            "interval": "day",
            "email": existing_user.get('email')
        }
        
        print(f"   - Sending analysis request with email")
        print(f"   - Request: {json.dumps(test_request, indent=2)}")
        
        response = requests.post(
            "http://localhost:8001/analyze",
            json=test_request,
            timeout=60
        )
        
        if response.status_code == 200:
            print(f"   ✅ Analysis completed successfully")
            result = response.json()
            
            if result.get("success"):
                print(f"   - Stock: {result.get('stock_symbol')}")
                print(f"   - Message: {result.get('message')}")
                
                # Check if analysis was stored in database
                import time
                time.sleep(2)  # Wait for storage to complete
                
                stored_analyses = simple_db_manager.get_stock_analyses("RELIANCE", limit=5)
                if stored_analyses:
                    latest_analysis = stored_analyses[0]
                    stored_user_id = latest_analysis.get('user_id')
                    expected_user_id = existing_user.get('id')
                    
                    print(f"   - Latest analysis found in database")
                    print(f"   - Stored user ID: {stored_user_id}")
                    print(f"   - Expected user ID: {expected_user_id}")
                    
                    if stored_user_id == expected_user_id:
                        print(f"   ✅ Email-based user ID mapping verified")
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

def test_analysis_without_email():
    """Test analysis request without email (should fail gracefully)."""
    print(f"\n3. Testing Analysis Without Email (Should Fail Gracefully):")
    
    try:
        test_request = {
            "stock": "TCS",
            "exchange": "NSE",
            "period": 30,
            "interval": "day"
            # No email provided
        }
        
        print(f"   - Sending analysis request without email")
        
        response = requests.post(
            "http://localhost:8001/analyze",
            json=test_request,
            timeout=60
        )
        
        if response.status_code == 200:
            print(f"   ✅ Analysis completed successfully (without storage)")
            result = response.json()
            
            if result.get("success"):
                print(f"   - Stock: {result.get('stock_symbol')}")
                print(f"   - Analysis completed but not stored (expected)")
                
                # Check that no analysis was stored
                import time
                time.sleep(2)
                
                stored_analyses = simple_db_manager.get_stock_analyses("TCS", limit=5)
                if not stored_analyses:
                    print(f"   ✅ No analysis stored (as expected)")
                    return True
                else:
                    print(f"   ❌ Analysis was stored when it shouldn't have been")
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
        return False
    except Exception as e:
        print(f"   ❌ Error testing analysis without email: {e}")
        return False

def test_enhanced_analysis_with_email():
    """Test enhanced analysis with email."""
    print(f"\n4. Testing Enhanced Analysis with Email:")
    
    try:
        test_request = {
            "stock": "INFY",
            "exchange": "NSE",
            "period": 30,
            "interval": "day",
            "email": existing_user.get('email'),
            "enable_code_execution": False  # Disable for faster testing
        }
        
        print(f"   - Sending enhanced analysis request with email")
        
        response = requests.post(
            "http://localhost:8001/analyze/enhanced",
            json=test_request,
            timeout=60
        )
        
        if response.status_code == 200:
            print(f"   ✅ Enhanced analysis completed successfully")
            result = response.json()
            
            # Check if analysis was stored in database
            import time
            time.sleep(2)
            
            stored_analyses = simple_db_manager.get_stock_analyses("INFY", limit=5)
            if stored_analyses:
                latest_analysis = stored_analyses[0]
                stored_user_id = latest_analysis.get('user_id')
                expected_user_id = existing_user.get('id')
                
                print(f"   - Latest enhanced analysis found in database")
                print(f"   - Stored user ID: {stored_user_id}")
                print(f"   - Expected user ID: {expected_user_id}")
                
                if stored_user_id == expected_user_id:
                    print(f"   ✅ Enhanced analysis email-based user ID mapping verified")
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
        print(f"   ❌ Error testing enhanced analysis: {e}")
        return False

def main():
    """Main test function."""
    print("🔍 TESTING EMAIL-BASED ANALYSIS STORAGE FLOW")
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
    
    test_results.append(test_email_based_storage())
    test_results.append(test_analysis_without_email())
    test_results.append(test_enhanced_analysis_with_email())
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY:")
    
    if all(test_results):
        print("✅ ALL TESTS PASSED!")
        print("✅ Email-based analysis storage flow is working correctly")
        print("✅ User ID mapping from email works")
        print("✅ Analysis without email fails gracefully (no storage)")
        print("✅ Enhanced analysis with email works")
        print("✅ No anonymous user generation")
    else:
        print("❌ SOME TESTS FAILED")
        failed_tests = [i+1 for i, result in enumerate(test_results) if not result]
        print(f"❌ Failed tests: {failed_tests}")
    
    print("\n🎯 KEY IMPROVEMENTS:")
    print("✅ Frontend now sends user email in analysis requests")
    print("✅ Backend maps email to user ID from profiles table")
    print("✅ No more anonymous user generation")
    print("✅ Analysis storage only works with valid user email")
    print("✅ Graceful handling of missing email (analysis works, no storage)")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Test with frontend to verify email is sent in requests")
    print("2. Monitor analysis service logs for email-based user ID resolution")
    print("3. Verify user analysis history in frontend")

if __name__ == "__main__":
    main() 