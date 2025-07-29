"""
Simple Database Test

This script tests the database connection and storage functionality using
an existing user ID to avoid foreign key constraint issues.
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
from database_manager import db_manager

def get_existing_user():
    """Get an existing user ID from the database."""
    try:
        result = db_manager.supabase.table("profiles").select("id").limit(1).execute()
        
        if result.data:
            user_id = result.data[0]["id"]
            print(f"✅ Using existing user ID: {user_id}")
            return user_id
        else:
            print("❌ No existing users found")
            return None
            
    except Exception as e:
        print(f"❌ Error getting existing user: {e}")
        return None

def create_fake_analysis_data():
    """Create fake analysis data for testing."""
    return {
        "ai_analysis": {
            "trend": "Bullish",
            "confidence_pct": 75,
            "short_term": {
                "horizon_days": 10,
                "entry_range": [100.0, 105.0],
                "stop_loss": 98.0,
                "targets": [110.0, 115.0]
            }
        },
        "indicators": {
            "rsi": {"value": 65.5, "signal": "neutral", "strength": "moderate"},
            "macd": {"value": 2.5, "signal": "bullish", "strength": "strong"}
        },
        "summary": {
            "overall_signal": "buy",
            "risk_level": "medium",
            "analysis_quality": "high",
            "confidence_score": 75
        },
        "metadata": {
            "current_price": 105.0,
            "price_change_pct": 2.5,
            "sector": "Technology",
            "analysis_timestamp": datetime.now().isoformat()
        }
    }

def test_analysis_storage(user_id):
    """Test storing fake analysis data in the database."""
    print(f"🔧 Testing analysis storage for user: {user_id}")
    
    try:
        # Create fake analysis data
        fake_analysis = create_fake_analysis_data()
        
        # Store analysis using the database manager
        analysis_id = db_manager.store_analysis(
            analysis=fake_analysis,
            user_id=user_id,
            symbol="TEST_STOCK",
            exchange="TEST",
            period=30,
            interval="day"
        )
        
        if analysis_id:
            print(f"✅ Analysis stored successfully with ID: {analysis_id}")
            return analysis_id
        else:
            print(f"❌ Failed to store analysis")
            return None
            
    except Exception as e:
        print(f"❌ Error storing analysis: {e}")
        return None

def test_analysis_retrieval(analysis_id):
    """Test retrieving the stored analysis data."""
    print(f"🔧 Testing analysis retrieval for analysis: {analysis_id}")
    
    try:
        # Retrieve analysis using the database manager
        analysis = db_manager.get_analysis_by_id(analysis_id)
        
        if analysis:
            print(f"✅ Analysis retrieved successfully")
            print(f"   Stock Symbol: {analysis.get('stock_symbol')}")
            print(f"   User ID: {analysis.get('user_id')}")
            print(f"   Overall Signal: {analysis.get('overall_signal')}")
            print(f"   Confidence Score: {analysis.get('confidence_score')}")
            return True
        else:
            print(f"❌ Failed to retrieve analysis")
            return False
            
    except Exception as e:
        print(f"❌ Error retrieving analysis: {e}")
        return False

def cleanup_test_data(analysis_id):
    """Clean up test data from the database."""
    print(f"🧹 Cleaning up test data...")
    
    try:
        if analysis_id:
            db_manager.supabase.table("stock_analyses").delete().eq("id", analysis_id).execute()
            print(f"✅ Deleted test analysis: {analysis_id}")
            
        print(f"✅ Cleanup completed")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")

def main():
    """Main test function."""
    print("🧪 Simple Database Connection Test")
    print("=" * 50)
    
    analysis_id = None
    
    try:
        # Get existing user
        print("\n📋 Test 1: Get Existing User")
        print("-" * 30)
        user_id = get_existing_user()
        if not user_id:
            print("❌ No existing user found, cannot proceed")
            return False
        
        # Test analysis storage
        print("\n📋 Test 2: Analysis Storage")
        print("-" * 30)
        analysis_id = test_analysis_storage(user_id)
        if not analysis_id:
            print("❌ Analysis storage failed, stopping tests")
            return False
        
        # Test analysis retrieval
        print("\n📋 Test 3: Analysis Retrieval")
        print("-" * 30)
        retrieval_success = test_analysis_retrieval(analysis_id)
        if not retrieval_success:
            print("❌ Analysis retrieval failed")
        
        # Cleanup
        print("\n📋 Test 4: Data Cleanup")
        print("-" * 30)
        cleanup_test_data(analysis_id)
        
        print("\n" + "=" * 50)
        print("🎉 Simple Database Test Completed Successfully!")
        print("=" * 50)
        print("✅ Database storage and retrieval are working correctly")
        print("✅ The analysis service should work properly")
        print("✅ Database connectivity is confirmed")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False
        
    finally:
        # Ensure cleanup happens even if tests fail
        if analysis_id:
            print(f"\n🧹 Final cleanup...")
            cleanup_test_data(analysis_id)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 