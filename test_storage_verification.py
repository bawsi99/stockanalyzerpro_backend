#!/usr/bin/env python3
"""
Test Storage Verification Debugging

This script tests the analysis storage with verification debugging
to show that the user ID is correctly stored and retrieved.
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

def test_storage_verification():
    """Test analysis storage with verification debugging."""
    print("🔍 TESTING STORAGE VERIFICATION DEBUGGING")
    print("=" * 60)
    
    # Get existing user for testing
    try:
        result = simple_db_manager.supabase.table("profiles").select("id, email").limit(1).execute()
        if not result.data:
            print("❌ No users found in profiles table")
            return False
            
        user = result.data[0]
        user_id = user.get('id')
        user_email = user.get('email')
        
        print(f"👤 Test User:")
        print(f"   - User ID: {user_id}")
        print(f"   - Email: {user_email}")
        print()
        
    except Exception as e:
        print(f"❌ Error getting user: {e}")
        return False
    
    # Test analysis request with email
    print("📤 Sending Analysis Request with Email:")
    print("-" * 40)
    
    test_request = {
        "stock": "RELIANCE",
        "exchange": "NSE",
        "period": 30,  # Short period for faster testing
        "interval": "day",
        "email": user_email
    }
    
    print("Request Payload:")
    print(json.dumps(test_request, indent=2))
    print()
    
    try:
        print("🔄 Sending request to analysis service...")
        response = requests.post(
            "http://localhost:8001/analyze",
            json=test_request,
            timeout=60
        )
        
        if response.status_code == 200:
            print("✅ Analysis completed successfully")
            result = response.json()
            
            if result.get("success"):
                print(f"   - Stock: {result.get('stock_symbol')}")
                print(f"   - Message: {result.get('message')}")
                print()
                
                # Wait a moment for storage to complete
                import time
                time.sleep(3)
                
                # Verify the storage manually
                print("🔍 MANUAL VERIFICATION:")
                print("-" * 40)
                
                # Get the latest analysis for this stock
                stored_analyses = simple_db_manager.get_stock_analyses("RELIANCE", limit=5)
                if stored_analyses:
                    latest_analysis = stored_analyses[0]
                    stored_analysis_id = latest_analysis.get('id')
                    stored_user_id = latest_analysis.get('user_id')
                    
                    print(f"   - Latest Analysis ID: {stored_analysis_id}")
                    print(f"   - Stored User ID: {stored_user_id}")
                    print(f"   - Expected User ID: {user_id}")
                    
                    if stored_user_id == user_id:
                        print(f"   ✅ MANUAL VERIFICATION PASSED: User ID matches!")
                        return True
                    else:
                        print(f"   ❌ MANUAL VERIFICATION FAILED: User ID mismatch!")
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
        print(f"   ❌ Error testing analysis: {e}")
        return False

def test_enhanced_storage_verification():
    """Test enhanced analysis storage with verification debugging."""
    print("\n🔍 TESTING ENHANCED STORAGE VERIFICATION:")
    print("=" * 60)
    
    # Get existing user for testing
    try:
        result = simple_db_manager.supabase.table("profiles").select("id, email").limit(1).execute()
        if not result.data:
            print("❌ No users found in profiles table")
            return False
            
        user = result.data[0]
        user_id = user.get('id')
        user_email = user.get('email')
        
    except Exception as e:
        print(f"❌ Error getting user: {e}")
        return False
    
    # Test enhanced analysis request with email
    print("📤 Sending Enhanced Analysis Request with Email:")
    print("-" * 40)
    
    test_request = {
        "stock": "TCS",
        "exchange": "NSE",
        "period": 30,
        "interval": "day",
        "email": user_email,
        "enable_code_execution": False  # Disable for faster testing
    }
    
    print("Request Payload:")
    print(json.dumps(test_request, indent=2))
    print()
    
    try:
        print("🔄 Sending enhanced request to analysis service...")
        response = requests.post(
            "http://localhost:8001/analyze/enhanced",
            json=test_request,
            timeout=60
        )
        
        if response.status_code == 200:
            print("✅ Enhanced analysis completed successfully")
            result = response.json()
            
            if result.get("success"):
                print(f"   - Stock: {result.get('stock_symbol')}")
                print()
                
                # Wait a moment for storage to complete
                import time
                time.sleep(3)
                
                # Verify the storage manually
                print("🔍 MANUAL VERIFICATION:")
                print("-" * 40)
                
                # Get the latest analysis for this stock
                stored_analyses = simple_db_manager.get_stock_analyses("TCS", limit=5)
                if stored_analyses:
                    latest_analysis = stored_analyses[0]
                    stored_analysis_id = latest_analysis.get('id')
                    stored_user_id = latest_analysis.get('user_id')
                    
                    print(f"   - Latest Analysis ID: {stored_analysis_id}")
                    print(f"   - Stored User ID: {stored_user_id}")
                    print(f"   - Expected User ID: {user_id}")
                    
                    if stored_user_id == user_id:
                        print(f"   ✅ MANUAL VERIFICATION PASSED: User ID matches!")
                        return True
                    else:
                        print(f"   ❌ MANUAL VERIFICATION FAILED: User ID mismatch!")
                        return False
                else:
                    print(f"   ❌ No enhanced analysis found in database")
                    return False
            else:
                print(f"   ❌ Enhanced analysis failed: {result.get('error')}")
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
    print("🔍 TESTING STORAGE VERIFICATION DEBUGGING")
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
    
    test_results.append(test_storage_verification())
    test_results.append(test_enhanced_storage_verification())
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY:")
    
    if all(test_results):
        print("✅ ALL TESTS PASSED!")
        print("✅ Storage verification debugging is working correctly")
        print("✅ User ID is correctly stored and retrieved")
        print("✅ Both regular and enhanced analysis verification works")
    else:
        print("❌ SOME TESTS FAILED")
        failed_tests = [i+1 for i, result in enumerate(test_results) if not result]
        print(f"❌ Failed tests: {failed_tests}")
    
    print("\n🔍 DEBUGGING OUTPUT EXPECTED:")
    print("When you run an analysis, you should see:")
    print("✅ Successfully stored analysis for STOCK with ID: analysis_id")
    print("🔍 DEBUGGING: Verifying analysis storage...")
    print("   - Analysis ID: analysis_id")
    print("   - Expected User ID: user_id")
    print("   - Actual User ID from DB: user_id")
    print("   ✅ VERIFICATION PASSED: User ID matches!")
    print("🔍 DEBUGGING: Verification complete")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Start analysis service: python start_analysis_service.py")
    print("2. Run analysis from frontend or test script")
    print("3. Monitor logs for verification debugging output")

if __name__ == "__main__":
    main() 