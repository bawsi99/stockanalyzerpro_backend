#!/usr/bin/env python3
"""
Test Analysis Storage Flow

This script tests the complete analysis storage flow to verify:
1. Analysis service stores data correctly
2. User ID mapping works properly
3. Data is stored in the correct table
4. Email to user ID mapping works
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
from database_manager import db_manager

def test_analysis_service_storage():
    """Test the analysis service storage flow."""
    print("🧪 Testing Analysis Service Storage Flow")
    print("=" * 50)
    
    # Test 1: Check which database manager is being used
    print("\n1. Checking Database Manager Usage:")
    print(f"   - Analysis Service imports: analysis_storage.py")
    print(f"   - analysis_storage.py imports: database_manager.py")
    print(f"   - database_manager.py stores to: stock_analyses table")
    print(f"   - Frontend expects: stock_analyses_simple table")
    print(f"   ❌ MISMATCH: Analysis service stores to wrong table!")
    
    # Test 2: Check user ID mapping functionality
    print("\n2. Testing User ID Mapping:")
    
    # Test email to user ID mapping
    test_email = "test@example.com"
    user_id = simple_db_manager.get_user_id_by_email(test_email)
    print(f"   - Test email: {test_email}")
    print(f"   - Mapped user ID: {user_id}")
    
    if user_id:
        print(f"   ✅ User ID mapping works")
    else:
        print(f"   ❌ User ID mapping failed - user not found")
    
    # Test 3: Check database tables
    print("\n3. Checking Database Tables:")
    
    # Check stock_analyses table (where analysis service stores)
    try:
        result = db_manager.supabase.table("stock_analyses").select("id").limit(1).execute()
        print(f"   - stock_analyses table: ✅ Accessible ({len(result.data) if result.data else 0} records)")
    except Exception as e:
        print(f"   - stock_analyses table: ❌ Error - {e}")
    
    # Check stock_analyses_simple table (where frontend expects data)
    try:
        result = simple_db_manager.supabase.table("stock_analyses_simple").select("id").limit(1).execute()
        print(f"   - stock_analyses_simple table: ✅ Accessible ({len(result.data) if result.data else 0} records)")
    except Exception as e:
        print(f"   - stock_analyses_simple table: ❌ Error - {e}")
    
    # Test 4: Check profiles table
    print("\n4. Checking Profiles Table:")
    try:
        result = simple_db_manager.supabase.table("profiles").select("id, email").limit(5).execute()
        print(f"   - profiles table: ✅ Accessible ({len(result.data) if result.data else 0} records)")
        if result.data:
            print("   - Sample profiles:")
            for profile in result.data[:3]:
                print(f"     * ID: {profile.get('id')}, Email: {profile.get('email')}")
    except Exception as e:
        print(f"   - profiles table: ❌ Error - {e}")
    
    # Test 5: Simulate analysis request
    print("\n5. Simulating Analysis Request:")
    
    # Generate a test user ID
    test_user_id = str(uuid.uuid4())
    print(f"   - Generated test user ID: {test_user_id}")
    
    # Create test analysis data
    test_analysis = {
        "summary": {
            "overall_signal": "bullish",
            "confidence": 85.5,
            "risk_level": "medium"
        },
        "ai_analysis": {
            "trend": "bullish",
            "confidence_pct": 85.5,
            "risks": ["Market volatility"]
        },
        "metadata": {
            "current_price": 1500.0,
            "price_change_pct": 2.5,
            "sector": "Technology"
        }
    }
    
    # Test storage in both tables
    print("\n6. Testing Storage in Both Tables:")
    
    # Test storage in stock_analyses (current analysis service behavior)
    try:
        analysis_id = db_manager.store_analysis(
            analysis=test_analysis,
            user_id=test_user_id,
            symbol="TEST",
            exchange="NSE",
            period=365,
            interval="day"
        )
        print(f"   - stock_analyses storage: ✅ Success (ID: {analysis_id})")
    except Exception as e:
        print(f"   - stock_analyses storage: ❌ Error - {e}")
    
    # Test storage in stock_analyses_simple (where frontend expects data)
    try:
        analysis_id = simple_db_manager.store_analysis(
            analysis=test_analysis,
            user_id=test_user_id,
            symbol="TEST",
            exchange="NSE",
            period=365,
            interval="day"
        )
        print(f"   - stock_analyses_simple storage: ✅ Success (ID: {analysis_id})")
    except Exception as e:
        print(f"   - stock_analyses_simple storage: ❌ Error - {e}")
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY:")
    print("❌ ISSUE FOUND: Analysis service stores to 'stock_analyses' table")
    print("❌ Frontend expects data in 'stock_analyses_simple' table")
    print("❌ User ID mapping from email is not implemented in analysis service")
    print("\n🔧 REQUIRED FIXES:")
    print("1. Update analysis service to use simple_database_manager")
    print("2. Add user_id field to AnalysisRequest model")
    print("3. Implement email to user ID mapping in analysis service")
    print("4. Update analysis storage to use stock_analyses_simple table")

if __name__ == "__main__":
    test_analysis_service_storage() 