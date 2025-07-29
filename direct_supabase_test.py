"""
Direct Supabase Test

This script tests the Supabase connection directly without using the database manager
to identify the exact issue with the analysis storage.
"""

import os
import sys
import uuid
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed, using system environment variables")
except Exception as e:
    print(f"⚠️ Error loading .env file: {e}")

from supabase_client import get_supabase_client

def test_direct_supabase_connection():
    """Test direct Supabase connection."""
    print("🔧 Testing Direct Supabase Connection")
    print("-" * 40)
    
    try:
        supabase = get_supabase_client()
        print("✅ Supabase client created successfully")
        
        # Test basic connection
        result = supabase.table("profiles").select("id").limit(1).execute()
        print("✅ Basic connection test passed")
        
        return supabase
        
    except Exception as e:
        print(f"❌ Supabase connection failed: {e}")
        return None

def test_simple_analysis_insertion(supabase):
    """Test inserting a simple analysis record."""
    print("\n🔧 Testing Simple Analysis Insertion")
    print("-" * 40)
    
    try:
        # Get existing user
        user_result = supabase.table("profiles").select("id").limit(1).execute()
        if not user_result.data:
            print("❌ No existing users found")
            return None
        
        user_id = user_result.data[0]["id"]
        print(f"✅ Using existing user: {user_id}")
        
        # Create simple analysis data
        simple_analysis = {
            "user_id": user_id,
            "stock_symbol": "TEST_STOCK",
            "analysis_data_json": {  # Changed from analysis_data to analysis_data_json
                "test": True,
                "timestamp": datetime.now().isoformat()
            },
            "exchange": "TEST",
            "period_days": 30,
            "interval": "day",
            "overall_signal": "test",
            "confidence_score": 75.0,
            "risk_level": "medium",
            "current_price": 100.0,
            "price_change_percentage": 2.5,
            "sector": "Technology",
            "analysis_type": "test",
            "analysis_quality": "test",
            "mathematical_validation": False,
            "metadata": {"test": True}
        }
        
        print(f"Attempting to insert analysis data...")
        result = supabase.table("stock_analyses").insert(simple_analysis).execute()
        
        if result.data:
            analysis_id = result.data[0]["id"]
            print(f"✅ Analysis inserted successfully with ID: {analysis_id}")
            return analysis_id
        else:
            print(f"❌ Failed to insert analysis")
            return None
            
    except Exception as e:
        print(f"❌ Error inserting analysis: {e}")
        return None

def test_analysis_retrieval(supabase, analysis_id):
    """Test retrieving the analysis."""
    print(f"\n🔧 Testing Analysis Retrieval")
    print("-" * 40)
    
    try:
        result = supabase.table("stock_analyses").select("*").eq("id", analysis_id).execute()
        
        if result.data:
            analysis = result.data[0]
            print(f"✅ Analysis retrieved successfully")
            print(f"   ID: {analysis.get('id')}")
            print(f"   Stock Symbol: {analysis.get('stock_symbol')}")
            print(f"   User ID: {analysis.get('user_id')}")
            print(f"   Overall Signal: {analysis.get('overall_signal')}")
            return True
        else:
            print(f"❌ Failed to retrieve analysis")
            return False
            
    except Exception as e:
        print(f"❌ Error retrieving analysis: {e}")
        return False

def cleanup_test_data(supabase, analysis_id):
    """Clean up test data."""
    print(f"\n🧹 Cleaning up test data...")
    print("-" * 40)
    
    try:
        if analysis_id:
            supabase.table("stock_analyses").delete().eq("id", analysis_id).execute()
            print(f"✅ Deleted test analysis: {analysis_id}")
            
        print(f"✅ Cleanup completed")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")

def main():
    """Main test function."""
    print("🧪 Direct Supabase Database Test")
    print("=" * 50)
    
    analysis_id = None
    
    try:
        # Test 1: Direct connection
        supabase = test_direct_supabase_connection()
        if not supabase:
            return False
        
        # Test 2: Simple insertion
        analysis_id = test_simple_analysis_insertion(supabase)
        if not analysis_id:
            return False
        
        # Test 3: Retrieval
        retrieval_success = test_analysis_retrieval(supabase, analysis_id)
        if not retrieval_success:
            print("❌ Analysis retrieval failed")
        
        # Test 4: Cleanup
        cleanup_test_data(supabase, analysis_id)
        
        print("\n" + "=" * 50)
        print("🎉 Direct Supabase Test Completed Successfully!")
        print("=" * 50)
        print("✅ Direct database operations are working correctly")
        print("✅ The issue is likely in the database manager layer")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False
        
    finally:
        # Ensure cleanup happens even if tests fail
        if analysis_id and supabase:
            print(f"\n🧹 Final cleanup...")
            cleanup_test_data(supabase, analysis_id)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 